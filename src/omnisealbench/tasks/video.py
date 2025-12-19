# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

import torch  # type: ignore

from omnisealbench.attacks.helper import AttackConfig, load_default_attacks
from omnisealbench.data.base import ORIGINAL_KEY
from omnisealbench.data.video import read_local_video_dataset
from omnisealbench.interface import ScoresT
from omnisealbench.tasks._detection import (
    ATTACK_RESULT_FILENAME,
    WatermarkAttacksAndDetection,
    WatermarkAttacksConfig,
)
from omnisealbench.tasks._e2e import End2EndWatermarkingConfig, WatermarkEnd2End
from omnisealbench.tasks._generation import WatermarkGeneration, WatermarkGenerationConfig
import pandas as pd
from copy import deepcopy
from omnisealbench.tasks.base import DataReaderT
from omnisealbench.utils.common import resolve
from omnisealbench.utils.video import save_vid

logger = logging.getLogger("omnisealbench.tasks.audio")

# Suppress warnings from multiple modules
warnings.filterwarnings("ignore", category=UserWarning, module="torch|torchvision|numpy|pandas")


@dataclass
class VideoConfig:
    quality_metrics_chunk_size: int = 8
    """quality_metrics_chunk_size, to control the computation due to memory constraints"""

    fps: int = 24
    """Target frames per second for the generated video."""

    crf: int = 0
    """CRF (Constant Rate Factor) for video encoding. Lower values mean better quality but larger file size."""

    video_codec: Literal["libx264", "rawvideo", "ffv1"] = "libx264"
    """Codec for output video generation"""


@dataclass
class VideoWatermarkGenerationConfig(WatermarkGenerationConfig, VideoConfig): ...


@dataclass
class VideoAttacksConfig(WatermarkAttacksConfig, VideoConfig): ...


@dataclass
class VideoWatermarkEnd2EndConfig(End2EndWatermarkingConfig, VideoConfig): ...


def compute_video_quality_metrics(
    wm_video: Union[torch.Tensor, List[torch.Tensor]],
    video: Union[torch.Tensor, List[torch.Tensor]],
    metrics: Sequence[str],
    chunk_size: int = 1,
    metrics_device: Optional[str] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    scores = {}

    if "psnr" in metrics:
        from omnisealbench.metrics.video import compute_psnr_chunked

        scores["psnr"] = (
            compute_psnr_chunked(
                wm_video,
                video,
                is_video=True,
                chunk_size=chunk_size,
            )
            .cpu()
            .numpy()
            .tolist()
        )

    if "ssim" in metrics:
        from omnisealbench.metrics.ssim import compute_ssim_chunked

        scores["ssim"] = (
            compute_ssim_chunked(
                wm_video,
                video,
                chunk_size=chunk_size,
            )
            .cpu()
            .numpy()
            .tolist()
        )

    if "msssim" in metrics:
        from omnisealbench.metrics.msssim import compute_msssim_chunked

        scores["msssim"] = (
            compute_msssim_chunked(
                wm_video,
                video,
                chunk_size=chunk_size,
            )
            .cpu()
            .numpy()
            .tolist()
        )

    if "vmaf" in metrics:
        from omnisealbench.metrics.video import vmaf_on_tensor

        vmaf_score = vmaf_on_tensor(
            wm_video,
            video,
            fps=24,
            codec="libx264",
            crf=23,
            n_threads=8,
            **kwargs,  # pass additional VMAF parameters if needed
        )

        scores["vmaf"] = vmaf_score

    if "lpips" in metrics:
        from omnisealbench.metrics.lpips import compute_lpips_chunked

        lpips = compute_lpips_chunked(
            wm_video, video, net="alex", chunk_size=chunk_size, device=metrics_device
        )  # b x F
        scores["lpips"] = lpips.cpu().numpy().tolist()

    if "lpips" in metrics:
        from omnisealbench.metrics.lpips import compute_lpips_chunked

        lpips = compute_lpips_chunked(
            wm_video, video, net="alex", chunk_size=chunk_size, device=metrics_device
        )  # b x F
        scores["lpips"] = lpips.cpu().numpy().tolist()

    return scores


def _save_video(
    videos: torch.Tensor,
    indices: Sequence[int],
    output_dir: Path,
    prefix: str = "",
    **kwargs,
):
    """
    Save video tensors to files.
    Args:
        videos (torch.Tensor): Video tensors to save.
        indices (Sequence[int]): Indices of the videos to save.
        output_dir (Path): Directory to save the videos.
        prefix (str): Prefix for the saved video filenames.
    """
    fps = kwargs["fps"]
    crf = kwargs["crf"]
    video_codec = kwargs["video_codec"]
    for idx, video in zip(indices, videos):
        video_path = output_dir / f"{prefix}_{idx:05d}.mp4"
        # Assuming video is a 4D tensor with shape (C, T, H, W)
        # Convert to a suitable format and save
        # Placeholder for actual video saving logic
        save_vid(
            video, video_path, fps=fps, crf=crf, video_codec=video_codec
        )  # Replace with actual video saving function


class VideoWatermarkGeneration(WatermarkGeneration):
    """
    A task for generating audio watermarks.
    """

    config: VideoWatermarkGenerationConfig

    def compute_metrics(
        self,
        orig: torch.Tensor,
        watermarked: torch.Tensor,
        message: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute the quality metrics for the watermarked audio results."""
        video = orig
        wm_video = watermarked

        chunk_size = self.config.quality_metrics_chunk_size

        return compute_video_quality_metrics(
            wm_video,
            video,
            metrics=self.config.metrics,
            chunk_size=chunk_size,
            metrics_device=self.config.metrics_device,
        )

    def save_data(  # type: ignore
        self,
        orig: Union[torch.Tensor, List[torch.Tensor]],
        watermarked: Union[torch.Tensor, List[torch.Tensor]],
        indices: List[int],
        **kwargs,  # type: ignore
    ) -> None:
        """
        Save the results of the watermark generation to the specified result directory.
        Args:
            watermarked_results (ResultT): The results of the watermark generation.
        """
        watermark_dir = Path(self.config.result_dir) / self.config.watermark_subdir
        Path(watermark_dir).mkdir(exist_ok=True, parents=True)

        kwargs = {
            "fps": self.config.fps,
            "crf": self.config.crf,
            "video_codec": self.config.video_codec,
        }

        _save_video(
            watermarked,
            indices,
            watermark_dir,
            prefix="wmd_video",
            **kwargs,
        )

        # Write original video files if data_subdir is provided
        if self.config.data_subdir:
            data_dir = resolve(self.config.result_dir) / self.config.data_subdir
            data_dir.mkdir(exist_ok=True, parents=True)

            _save_video(
                orig,
                indices,
                data_dir,
                prefix="video",
                **kwargs,
            )


class VideoWatermarkAttacksAndDetection(WatermarkAttacksAndDetection):
    """A task for detecting and attacking video watermarks."""

    config: VideoAttacksConfig

    def compute_quality_metrics(
        self,
        wm_input: torch.Tensor,
        orig_input: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        chunk_size = self.config.quality_metrics_chunk_size
        return compute_video_quality_metrics(
            wm_input,
            orig_input,
            metrics=self.config.metrics,
            chunk_size=chunk_size,
            metrics_device=self.config.metrics_device,
        )

    def skip_quality_metrics(self, config: VideoAttacksConfig) -> bool:  # type: ignore
        for skip_by_name in ["H264_Crop_Brightness", "Resize", "Crop"]:
            if skip_by_name in config.attacks.func:
                return True

        return super().skip_quality_metrics(config)

    def save_data(  # type: ignore
        self,
        orig: torch.Tensor,
        watermarked: torch.Tensor,
        indices: Sequence[int],
        config: Optional[VideoAttacksConfig] = None,
        **kwargs,  # type: ignore
    ) -> None:
        config = config or self.config
        kwargs = {"fps": config.fps, "crf": config.crf, "video_codec": config.video_codec}

        # Save watermarked videos
        _save_video(
            watermarked,
            indices,
            Path(config.result_dir) / config.data_subdir,
            prefix="wmd_video",
            **kwargs,
        )

        # Save original videos
        _save_video(
            orig,
            indices,
            Path(config.result_dir) / config.data_subdir,
            prefix="video",
            **kwargs,
        )


class VideoWatermarkEnd2End(WatermarkEnd2End):
    config: VideoWatermarkEnd2EndConfig

    def get_generation_task(  # type: ignore
        self,
        config: VideoWatermarkEnd2EndConfig,
        data_reader: Optional[DataReaderT] = None,
        num_workers: int = 0,
        **data_args,
    ) -> WatermarkGeneration:
        wm_config = VideoWatermarkGenerationConfig(
            metrics=[],  # No metrics is calculated, this is done in identity task in detection step
            seed=config.seed,
            result_dir=config.cache_dir,
            overwrite=config.overwrite,
            result_filename="watermark_results.jsonl",
            ids_to_save="all",  # by default, save all watermarked data for next steps
            data_subdir="data",  # subdirectory for storing data
            message_size=config.message_size,
            how_to_generate_message=config.how_to_generate_message,
            watermark_subdir=config.watermark_subdir,
            keep_data_in_memory=config.keep_data_in_memory,
            quality_metrics_chunk_size=config.quality_metrics_chunk_size,
            fps=config.fps,
            crf=config.crf,
            video_codec=config.video_codec,
        )
        return VideoWatermarkGeneration(
            config=wm_config,
            data_reader=data_reader,
            num_workers=num_workers,
            **data_args,
        )

    def get_detection_task(  # type: ignore
        self,
        config: VideoWatermarkEnd2EndConfig,
        num_workers: int = 0,
        **data_args,
    ) -> WatermarkAttacksAndDetection:
        data_args["video_pattern"] = "data/*.mp4"

        det_config = VideoAttacksConfig(
            metrics=config.metrics,
            metrics_device=config.metrics_device,
            seed=config.seed,
            result_dir=config.result_dir,
            overwrite=config.overwrite,
            result_filename=ATTACK_RESULT_FILENAME,
            final_results_filename=config.final_results_filename,
            ids_to_save=config.ids_to_save,
            detection_threshold=config.detection_threshold,
            message_threshold=config.message_threshold,
            detection_bits=config.detection_bits,
            data_subdir=config.data_subdir,
            detection_result_filename=config.detection_result_filename,
            attacks=config.attacks,
            skip_quality_metrics_on_attacks=config.skip_quality_metrics_on_attacks,
            quality_metrics_chunk_size=config.quality_metrics_chunk_size,
            fps=config.fps,
            crf=config.crf,
            video_codec=config.video_codec,
        )

        return VideoWatermarkAttacksAndDetection(
            config=det_config,
            data_reader=read_local_video_dataset,
            num_workers=num_workers,
            watermarked_video_pattern="watermarks/*.mp4",
            message_pattern="watermarks/*.txt",
            **data_args,
        )

    def print_scores(self, scores: Union[ScoresT, List[ScoresT]], **kwargs) -> pd.DataFrame:
        _data_args = deepcopy(self.data_args)
        _data_args["dataset_dir"] = self.config.cache_dir

        det_config = VideoAttacksConfig(
            metrics=self.config.metrics,
            metrics_device=self.config.metrics_device,
            seed=self.config.seed,
            result_dir=self.config.result_dir,
            overwrite=self.config.overwrite,
            result_filename=ATTACK_RESULT_FILENAME,
            final_results_filename=self.config.final_results_filename,
            ids_to_save=self.config.ids_to_save,
            detection_threshold=self.config.detection_threshold,
            message_threshold=self.config.message_threshold,
            detection_bits=self.config.detection_bits,
            data_subdir=self.config.data_subdir,
            detection_result_filename=self.config.detection_result_filename,
            attacks=self.config.attacks,
            skip_quality_metrics_on_attacks=self.config.skip_quality_metrics_on_attacks,
            quality_metrics_chunk_size=self.config.quality_metrics_chunk_size,
            fps=self.config.fps,
            crf=self.config.crf,
            video_codec=self.config.video_codec,
        )
        det_task = self.get_detection_task(
            config=det_config,  # type: ignore
            num_workers=self.num_workers,  # type: ignore[arg-type]
            **_data_args,
        )

        return det_task.print_scores(scores, **kwargs)


########################################################################
# A factory of audio taks with ready-to-use configurations.
########################################################################


def evaluate_video_watermark_generation_on_local_data(
    dataset_dir: str,
    video_pattern: Optional[str] = None,
    output_resolution: Optional[int] = None,
    max_frames: Optional[int] = None,
    fps: int = 24,
    crf: int = 0,
    video_codec: Literal["libx264", "rawvideo", "ffv1"] = "libx264",
    metrics: Optional[Sequence[str]] = None,
    metrics_device: Optional[str] = None,
    quality_metrics_chunk_size: int = 1,
    seed: Optional[int] = None,
    result_dir: Union[str, Path] = "",
    data_subdir: str = "",
    ids_to_save: Optional[str] = None,
    overwrite: bool = False,
    message_size: Union[int, Sequence[int], None] = None,
    how_to_generate_message: Literal["per_batch", "per_dataset"] = "per_batch",
    batch_size: int = 1,
    num_workers: int = 0,
    num_samples: Optional[int] = None,
):
    config = VideoWatermarkGenerationConfig(
        metrics=metrics or [],
        quality_metrics_chunk_size=quality_metrics_chunk_size,
        message_size=message_size,
        how_to_generate_message=how_to_generate_message,
        seed=seed or 42,
        result_dir=result_dir,
        data_subdir=data_subdir,
        ids_to_save=ids_to_save,
        metrics_device=metrics_device,
        overwrite=overwrite,
        fps=fps,
        crf=crf,
        video_codec=video_codec,
    )

    return VideoWatermarkGeneration(
        config=config,
        data_reader=read_local_video_dataset,
        dataset_dir=dataset_dir,
        video_pattern=video_pattern,
        output_resolution=output_resolution,
        max_frames=max_frames,
        batch_size=batch_size,
        num_workers=num_workers,
        num_samples=num_samples,
    )


def evaluate_video_watermark_generation_on_custom_data(
    data_reader,
    fps: int = 24,
    crf: int = 0,
    video_codec: Literal["libx264", "rawvideo", "ffv1"] = "libx264",
    metrics: Optional[Sequence[str]] = None,
    metrics_device: Optional[str] = None,
    quality_metrics_chunk_size: int = 1,
    seed: Optional[int] = None,
    result_dir: Union[str, Path] = "",
    data_subdir: str = "",
    overwrite: bool = False,
    message_size: Union[int, Sequence[int], None] = None,
    how_to_generate_message: Literal["per_batch", "per_dataset"] = "per_batch",
    num_workers: int = 0,
):
    config = VideoWatermarkGenerationConfig(
        metrics=metrics or [],
        quality_metrics_chunk_size=quality_metrics_chunk_size,
        message_size=message_size,
        how_to_generate_message=how_to_generate_message,
        seed=seed or 42,
        result_dir=result_dir,
        data_subdir=data_subdir,
        metrics_device=metrics_device,
        overwrite=overwrite,
        fps=fps,
        crf=crf,
        video_codec=video_codec,
    )

    return VideoWatermarkGeneration(
        config=config,
        data_reader=data_reader,
        num_workers=num_workers,
    )


def evaluate_video_watermark_generation(
    dataset_dir: str = "",
    video_pattern: Optional[str] = None,
    output_resolution: Optional[int] = None,
    max_frames: Optional[int] = None,
    data_reader: Optional[DataReaderT] = None,
    fps: int = 24,
    crf: int = 0,
    video_codec: Literal["libx264", "rawvideo", "ffv1"] = "libx264",
    metrics: Optional[Sequence[str]] = None,
    metrics_device: Optional[str] = None,
    quality_metrics_chunk_size: int = 1,
    seed: Optional[int] = None,
    result_dir: Union[str, Path] = "",
    data_subdir: str = "",
    ids_to_save: Optional[str] = None,
    overwrite: bool = False,
    message_size: Union[int, Sequence[int], None] = None,
    how_to_generate_message: Literal["per_batch", "per_dataset"] = "per_batch",
    batch_size: int = 1,
    num_workers: int = 0,
    num_samples: Optional[int] = None,
):
    """
    Evaluates video watermark generation on a given dataset or custom data reader.
    This function supports evaluation on local datasets or custom data readers, computes specified metrics,
    and saves the results to a directory. It can handle different message generation strategies and supports
    batch processing.

    When "dataset_dir" is specified, the images from the dataset directory are used for evaluation. By default, all
    kinds of images present in the directory and subdirectories are included. This can be changed by specifying
    an image pattern in `image_pattern`.

    User can also write their own function of reading image data. The function should return an iterator of
    data batches of type `omnisealbench.data.base.InputBatch`. This option is when `data_reader` is specified.

    Args:
        dataset_dir (str, optional): Path to the dataset directory. If provided, evaluation is performed on local data.
        video_pattern (Optional[str], optional): Glob pattern to match video files in the dataset directory.
        even_shapes (bool, optional): Whether to enforce even shapes for videos.
        output_resolution (Optional[int], optional): Output resolution to process the input videos.
        max_frames (Optional[int], optional): Maximum number of frames to process per video.
        fps (int, optional): Frames per second for output videos.
        crf (int, optional): Constant Rate Factor for video encoding quality.
        video_codec (Literal["libx264", "rawvideo", "ffv1"], optional): Codec to use for video encoding.
        quality_metrics_chunk_size (int, optional): Chunk size for quality metric computation.
        data_reader (Optional[DataReaderT], optional): Custom data reader for evaluation. Used if `dataset_dir` is not provided.
        metrics (Optional[Sequence[str]], optional): List of metric names to compute during evaluation.
        metrics_device (Optional[str], optional): Device to use for metric computation (e.g., "cpu", "cuda").
        seed (Optional[int], optional): Random seed for reproducibility.
        result_dir (Union[str, Path], optional): Directory to save evaluation results.
        data_subdir (str, optional): Subdirectory within the dataset directory to use for evaluation.
        ids_to_save (Optional[str], optional): Comma-separated list of sample IDs to save.
        overwrite (bool, optional): Whether to overwrite existing results in the result directory.
        message_size (Union[int, Sequence[int], None], optional): Size(s) of the watermark message(s) to generate.
        how_to_generate_message (Literal["per_batch", "per_dataset"], optional): Strategy for generating messages.
        batch_size (int, optional): Number of samples per batch.
        num_workers (int, optional): Number of worker processes for data loading.
        num_samples (Optional[int], optional): Number of samples to evaluate.
    Returns:
        dict: A dictionary containing evaluation results and computed metrics.
    Raises:
        AssertionError: If `dataset_dir` is provided but does not exist.
    """

    config = VideoWatermarkGenerationConfig(
        metrics=metrics or [],
        quality_metrics_chunk_size=quality_metrics_chunk_size,
        message_size=message_size,
        how_to_generate_message=how_to_generate_message,
        seed=seed or 42,
        result_dir=result_dir,
        data_subdir=data_subdir,
        ids_to_save=ids_to_save,
        metrics_device=metrics_device,
        overwrite=overwrite,
        fps=fps,
        crf=crf,
        video_codec=video_codec,
    )

    if dataset_dir:
        return VideoWatermarkGeneration(
            config=config,
            data_reader=read_local_video_dataset,
            dataset_dir=dataset_dir,
            video_pattern=video_pattern,
            output_resolution=output_resolution,
            max_frames=max_frames,
            batch_size=batch_size,
            num_workers=num_workers,
            num_samples=num_samples,
        )

    else:
        return VideoWatermarkGeneration(
            config=config,
            data_reader=data_reader,
            num_workers=num_workers,
        )


def evaluate_video_watermark_attacks_and_detection(
    dataset_dir: str,
    original_video_pattern: Optional[str] = "data/*.mp4",
    watermarked_video_pattern: Optional[str] = "watermarks/*.mp4",
    message_pattern: Optional[str] = "watermarks/*.txt",
    output_resolution: Optional[int] = None,
    max_frames: Optional[int] = None,
    fps: int = 24,
    crf: int = 0,
    video_codec: Literal["libx264", "rawvideo", "ffv1"] = "libx264",
    even_shapes: bool = False,
    metrics: Optional[Sequence[str]] = None,
    skip_quality_metrics_on_attacks: bool = False,
    metrics_device: Optional[str] = None,
    quality_metrics_chunk_size: int = 1,
    detection_bits: Optional[int] = None,
    detection_threshold: Optional[float] = None,
    message_threshold: Optional[float] = None,
    seed: Optional[int] = None,
    result_dir: Union[str, Path] = "",
    ids_to_save: Optional[str] = None,
    overwrite: bool = False,
    attacks: Union[Sequence[AttackConfig], Sequence[str], None] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    num_samples: Optional[int] = None,
):
    """
    Read watermarked video files from a local directory, apply attacks to each video then run the
    detector to detect the watermarks and calculate different  scores (bit accuracy,
    capacity, log10_p_value, p_value, TPR, FPR) on the detection results.

    By default, all video files in the directory will be evaluated. If `original_video_pattern` and `watermarked_video_pattern`
    are specified, only the watermarked files that match `watermarked_video_pattern` will be evaluated and
    the metrics will be computed against the original video files that match `original_video_pattern`.

    If `message_pattern` is specified, it will use the watermark messages that match `message_pattern` to compute
    message-related metrics (bit accuracy, capacity, etc.). Otherwise only TPR and FPR are returned.

    Additionally, other quality metrics (e.g., ssim, lpips) are also computed when specified by `metrics`.

    Args:
        dataset_dir (str): Path to the dataset directory.
        original_video_pattern (Optional[str]): Pattern to match original video files.
        watermarked_video_pattern (Optional[str]): Pattern to match watermarked video files.
        output_resolution (Optional[int], optional): Output resolution to process the input videos.
        max_frames (Optional[int], optional): Maximum number of frames to process per video.
        fps (int, optional): Frames per second for output videos.
        crf (int, optional): Constant Rate Factor for video encoding quality.
        video_codec (Literal["libx264", "rawvideo", "ffv1"], optional): Codec to use for video encoding.
        quality_metrics_chunk_size (int, optional): Chunk size for quality metric computation.
        message_pattern (Optional[str]): Pattern to match watermark messages.
        even_shapes (bool): Whether to use even shapes for video.
        metrics (Optional[Sequence[str]]): List of extra quality metrics to compute in addition to detection scores
        skip_quality_metrics_on_attacks (bool): Whether to skip quality metrics when evaluating attacks.
        metrics_device (Optional[str]): Device to use for metrics computation.
        detection_bits (int): Number of bits to use for detection.
        detection_threshold (float): Threshold for detection.
        message_threshold (float): Threshold for message detection.
        seed (Optional[int]): Random seed for reproducibility.
        result_dir (Union[str, Path]): Directory to save results.
        ids_to_save (Optional[str]): IDs of videos to save.
        overwrite (bool): Whether to overwrite existing results.
        attacks (Union[Sequence[AttackConfig], Sequence[str], None]): List of attacks to apply.
            This can be a list of AttackConfig, or a list of attack names (strings) that map to default
            configurations, which are defined in omnisealbench/attacks/video_attacks.yaml, or a path to
            a YAML file defining the attack configurations.
        batch_size (int): Batch size for processing videos.
        num_workers (int): Number of workers for data loading.
        num_samples (Optional[int]): Number of samples to process.

    """

    if attacks == "all":
        attacks = load_default_attacks(None, attacks_source="video")  # type: ignore
    if isinstance(attacks, str) or (
        isinstance(attacks, (tuple, list)) and len(attacks) > 0 and isinstance(attacks[0], str)
    ):
        attacks = load_default_attacks(attacks, attacks_source="video")  # type: ignore

    config = VideoAttacksConfig(
        metrics=metrics or [],
        skip_quality_metrics_on_attacks=skip_quality_metrics_on_attacks,
        quality_metrics_chunk_size=quality_metrics_chunk_size,
        detection_bits=detection_bits,
        detection_threshold=detection_threshold,
        message_threshold=message_threshold,
        fps=fps,
        crf=crf,
        video_codec=video_codec,
        seed=seed or 42,
        result_dir=result_dir,
        ids_to_save=ids_to_save,
        overwrite=overwrite,
        attacks=attacks,  # type: ignore
        metrics_device=metrics_device,
    )

    return VideoWatermarkAttacksAndDetection(
        config=config,
        data_reader=read_local_video_dataset,
        dataset_dir=dataset_dir,
        video_pattern=original_video_pattern,
        watermarked_video_pattern=watermarked_video_pattern,
        message_pattern=message_pattern,
        output_resolution=output_resolution,
        max_frames=max_frames,
        even_shapes=even_shapes,
        batch_size=batch_size,
        num_workers=num_workers,
        num_samples=num_samples,
    )


def evaluate_video_watermark_end2end(
    dataset_dir: str,
    video_pattern: Optional[str] = "*.mp4",
    output_resolution: Optional[int] = None,
    max_frames: Optional[int] = None,
    fps: int = 24,
    crf: int = 0,
    video_codec: Literal["libx264", "rawvideo", "ffv1"] = "libx264",
    message_size: Union[int, Sequence[int], None] = None,
    how_to_generate_message: Literal["per_batch", "per_dataset"] = "per_batch",
    metrics: Optional[Sequence[str]] = None,
    skip_quality_metrics_on_attacks: bool = False,
    quality_metrics_chunk_size: int = 1,
    metrics_device: Optional[str] = None,
    detection_bits: Optional[int] = None,
    detection_threshold: Optional[float] = None,
    message_threshold: Optional[float] = None,
    seed: Optional[int] = None,
    cache_dir: str = "",
    result_dir: Union[str, Path] = "",
    ids_to_save: Optional[str] = None,
    overwrite: bool = False,
    attacks: Union[Sequence[AttackConfig], Sequence[str], None] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    num_samples: Optional[int] = None,
):
    if attacks == "all":
        attacks = load_default_attacks(None, attacks_source="video")  # type: ignore
    if isinstance(attacks, str) or (
        isinstance(attacks, (tuple, list)) and len(attacks) > 0 and isinstance(attacks[0], str)
    ):
        attacks = load_default_attacks(attacks, attacks_source="video")  # type: ignore

    config = VideoWatermarkEnd2EndConfig(
        metrics=metrics or [],
        cache_dir=cache_dir,
        result_dir=result_dir,
        ids_to_save=ids_to_save,
        attacks=attacks,  # type: ignore
        skip_quality_metrics_on_attacks=skip_quality_metrics_on_attacks,
        quality_metrics_chunk_size=quality_metrics_chunk_size,
        fps=fps,
        crf=crf,
        video_codec=video_codec,
        metrics_device=metrics_device,
        detection_threshold=detection_threshold,
        message_threshold=message_threshold,
        detection_bits=detection_bits,
        seed=seed or 42,
        overwrite=overwrite,
        message_size=message_size,
        how_to_generate_message=how_to_generate_message,
    )
    return VideoWatermarkEnd2End(
        config=config,
        data_reader=read_local_video_dataset,
        dataset_dir=dataset_dir,
        video_pattern=video_pattern,
        output_resolution=output_resolution,
        max_frames=max_frames,
        batch_size=batch_size,
        num_workers=num_workers,
        num_samples=num_samples,
    )
