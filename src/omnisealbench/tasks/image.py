# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

import torch
from torchvision import transforms

from omnisealbench.attacks.helper import AttackConfig, load_default_attacks
from omnisealbench.data.image import read_local_image_dataset
from omnisealbench.tasks._detection import (WatermarkAttacksAndDetection,
                                            WatermarkAttacksConfig)
from omnisealbench.tasks._e2e import (End2EndWatermarkingConfig,
                                      WatermarkEnd2End, detection_config,
                                      generation_config)
from omnisealbench.tasks._generation import (WatermarkGeneration,
                                             WatermarkGenerationConfig)
from omnisealbench.tasks.base import DataReaderT
from omnisealbench.utils.common import resolve


def _save_images(
    images: Union[torch.Tensor, List[torch.Tensor]],
    indices: Sequence[int],
    output_dir: Path,
    prefix: str = "",
):
    for idx, image in zip(indices, images):
        if image.ndim == 4:
            assert image.shape[0] == 1, "Batch size should be 1 when saving images."
            image = image[0]  # BxCxHxW -> CxHxW
        img = transforms.ToPILImage()(image.detach().cpu())
        img.save(output_dir / f"{prefix}_{idx:05d}.png")


def compute_quality_metrics(
    wm_input: Union[torch.Tensor, List[torch.Tensor]],
    orig_input: Union[torch.Tensor, List[torch.Tensor]],
    metrics: Sequence[str],
) -> Dict[str, torch.Tensor]:
    scores = {}
    if "ssim" in metrics:
        from omnisealbench.metrics.image import compute_ssim_scores

        scores["ssim"] = compute_ssim_scores(
            wm_input, orig_input, num_processes=int(os.environ.get("NUM_SSIM_WORKERS", 4))  # type: ignore
        )

    if "lpips" in metrics:
        from omnisealbench.metrics.lpips import compute_lpips
        scores["lpips"] = compute_lpips(wm_input, orig_input, net="vgg").squeeze().cpu().tolist()

    if "psnr" in metrics:
        from omnisealbench.metrics.image import psnr    
        scores["psnr"] = psnr(wm_input, orig_input).squeeze().cpu().tolist()
        
    return scores


class ImageWatermarkGeneration(WatermarkGeneration):
    """
    A task for generating image watermarks.
    """

    def compute_metrics(
        self,
        orig: torch.Tensor,
        watermarked: torch.Tensor,
        message: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        img = orig
        wm_img = watermarked

        return compute_quality_metrics(
            wm_img, img, metrics=self.config.metrics
        )

    def save_data(  # type: ignore
        self,
        orig: Union[torch.Tensor, List[torch.Tensor]],
        watermarked: Union[torch.Tensor, List[torch.Tensor]],
        indices: List[int],
        **kwargs,  # type: ignore
    ) -> None:
        watermark_dir = Path(self.config.result_dir) / self.config.watermark_subdir
        Path(watermark_dir).mkdir(exist_ok=True, parents=True)
        _save_images(
            watermarked,
            indices,
            watermark_dir,
            prefix="wmd_img",
        )

        # Write original image files if data_subdir is provided
        if self.config.data_subdir:
            data_dir = resolve(self.config.result_dir) / self.config.data_subdir
            data_dir.mkdir(exist_ok=True, parents=True)

            _save_images(
                orig,
                indices,
                data_dir,
                prefix="img",
            )


class ImageWatermarkAttacksAndDetection(WatermarkAttacksAndDetection):
    """
    A task for evaluating image watermark attacks and detection.
    This class is a placeholder for future implementation.
    """

    def skip_quality_metrics(self, config: WatermarkAttacksConfig) -> bool:
        for skip_by_name in ["median_filter", "comb"]:
            if skip_by_name in config.attacks.func:
                return True

        return super().skip_quality_metrics(config)

    def compute_quality_metrics(  # type: ignore
        self,
        wm_input: Union[torch.Tensor, List[torch.Tensor]],
        orig_input: Union[torch.Tensor, List[torch.Tensor]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        return compute_quality_metrics(
            wm_input, orig_input, metrics=self.config.metrics
        )

    def save_data(  # type: ignore
        self,
        orig: torch.Tensor,
        watermarked: torch.Tensor,
        indices: Sequence[int],
        config: Optional[WatermarkGenerationConfig] = None,
        **kwargs,  # type: ignore
    ) -> None:
        config = config or self.config

        # Save watermarked images
        _save_images(
            watermarked,
            indices,
            Path(config.result_dir) / config.data_subdir,  # type: ignore
            prefix="wmd_img",
        )

        # Save additional original images
        _save_images(
            orig,
            indices,
            Path(config.result_dir) / config.data_subdir,  # type: ignore
            prefix="img",
        )


class ImageWatermarkEnd2End(WatermarkEnd2End):

    def get_generation_task(
        self,
        config: End2EndWatermarkingConfig,
        data_reader: Optional[DataReaderT] = None,
        num_workers: int = 0,
        **data_args,
    ) -> WatermarkGeneration:
        wm_config = generation_config(config)
        return ImageWatermarkGeneration(
            config=wm_config,
            data_reader=data_reader,
            num_workers=num_workers,
            **data_args,
        )

    def get_detection_task(
        self,
        config: WatermarkAttacksConfig,
        num_workers: int = 0,
        **data_args,
    ) -> WatermarkAttacksAndDetection:
        det_config = detection_config(config)
        data_args["image_pattern"] = "data/*.png"
        return ImageWatermarkAttacksAndDetection(
            config=det_config,
            data_reader=read_local_image_dataset,
            num_workers=num_workers,
            watermarked_image_pattern="watermarks/*.png",
            message_pattern="watermarks/*.txt",
            **data_args,
        )


########################################################################
# A factory of audio taks with ready-to-use configurations.
########################################################################

def evaluate_image_watermark_generation(
    dataset_dir: str = "",
    image_pattern: Optional[str] = None,
    even_shapes: bool = False,
    output_size: Optional[int] = None,
    data_reader: Optional[DataReaderT] = None,
    metrics: Optional[Sequence[str]] = None,
    metrics_device: Optional[str] = None,
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
    Evaluates image watermark generation on a given dataset or custom data reader.
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
        image_pattern (Optional[str], optional): Glob pattern to match image files in the dataset directory.
        even_shapes (bool, optional): Whether to enforce even shapes for images.
        output_size (Optional[int], optional): If provided, resizes images to have this size.
        data_reader (Optional[DataReaderT], optional): Custom data reader for evaluation. Used if `dataset_dir` is not provided.
        metrics (Optional[Sequence[str]], optional): List of metric names to compute during evaluation.
        metrics_device (Optional[str], optional): Device to use for metric computation (e.g., "cpu", "cuda").
        seed (Optional[int], optional): Random seed for reproducibility.
        result_dir (Union[str, Path], optional): Directory to save evaluation results.
        data_subdir (str, optional): This specifies the subdirectory within the dataset directory to use. Defaults to "".
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
    config = WatermarkGenerationConfig(
        metrics=metrics or [],
        metrics_device=metrics_device,
        seed=seed or 42,
        overwrite=overwrite,
        result_dir=result_dir,
        data_subdir=data_subdir,
        ids_to_save=ids_to_save,
        message_size=message_size,
        how_to_generate_message=how_to_generate_message,
    )

    if dataset_dir:
        assert Path(dataset_dir).exists(), f"Dataset directory {dataset_dir} does not exist."

        return ImageWatermarkGeneration(
            config=config,
            data_reader=read_local_image_dataset,
            dataset_dir=dataset_dir,
            image_pattern=image_pattern,
            even_shapes=even_shapes,
            output_size=output_size,
            num_samples=num_samples,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        return ImageWatermarkGeneration(
            config=config,
            data_reader=data_reader,
            num_workers=num_workers,
        )


def evaluate_image_watermark_attacks_and_detection(
    dataset_dir: str,
    original_image_pattern: Optional[str] = "data/*.png",
    watermarked_image_pattern: Optional[str] = "watermarks/*.png",
    message_pattern: Optional[str] = "watermarks/*.txt",
    even_shapes: bool = False,
    output_size: Optional[int] = None,
    metrics: Optional[Sequence[str]] = None,
    skip_quality_metrics_on_attacks: bool = False,
    metrics_device: Optional[str] = None,
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
    Read watermarked images from a local directory, apply attacks to each image then run the 
    detector to detect the watermarks and calculate different  scores (bit accuracy, 
    capdacity, log10_p_value, p_value, TPR, FPR) on the detection results.

    By default, all image files in the directory will be evaluated. If `original_image_pattern` 
    and `watermarked_image_pattern` are specified, only the watermarked files that match 
    `watermarked_image_pattern` will be evaluated and the metrics will be computed against
    the original image files that match `original_image_pattern`.

    If `message_pattern` is specified, it will use the watermark messages that match `message_pattern` to compute
    message-related metrics (bit accuracy, capacity, etc.). Otherwise only TPR and FPR are returned.

    Additionally, other quality metrics (e.g., ssim, lpips) are also computed when specified by `metrics`.
    
    Args:
        dataset_dir (str): Path to the dataset directory.
        original_image_pattern (Optional[str]): Pattern to match original images.
        watermarked_image_pattern (Optional[str]): Pattern to match watermarked images.
        message_pattern (Optional[str]): Pattern to match watermark messages.
        even_shapes (bool): Whether to use even shapes for images.
        output_size (Optional[int]): If provided, resizes images to have this size.
        metrics (Optional[Sequence[str]]): List of extra quality metrics to compute in addition to detection scores
        skip_quality_metrics_on_attacks (bool): Whether to skip quality metrics when evaluating attacks.
        metrics_device (Optional[str]): Device to use for metrics computation.
        detection_bits (int): Number of bits to use for detection.
        detection_threshold (float): Threshold for detection.
        message_threshold (float): Threshold for message detection.
        seed (Optional[int]): Random seed for reproducibility.
        result_dir (Union[str, Path]): Directory to save results.
        ids_to_save (Optional[str]): IDs of images to save.
        overwrite (bool): Whether to overwrite existing results.
        attacks (Union[Sequence[AttackConfig], Sequence[str], None]): List of attacks to apply.
            This can be a list of AttackConfig, or a list of attack names (strings) that map to default
            configurations, which are defined in omnisealbench/attacks/image_attacks.yaml, or a path to
            a YAML file defining the attack configurations.
        batch_size (int): Batch size for processing images.
        num_workers (int): Number of workers for data loading.
        num_samples (Optional[int]): Number of samples to process.

    """

    if attacks == "all":
        attacks = load_default_attacks(None, attacks_source="image")  # type: ignore
    if isinstance(attacks, str) or (isinstance(attacks, (tuple, list)) and len(attacks) > 0 and isinstance(attacks[0], str)):
        attacks = load_default_attacks(attacks, attacks_source="image")  # type: ignore

    config = WatermarkAttacksConfig(
        metrics=metrics or [],
        skip_quality_metrics_on_attacks=skip_quality_metrics_on_attacks,
        metrics_device=metrics_device,
        seed=seed or 42,
        result_dir=result_dir,
        ids_to_save=ids_to_save,
        overwrite=overwrite,
        attacks=attacks,  # type: ignore
        detection_threshold=detection_threshold,
        message_threshold=message_threshold,
        detection_bits=detection_bits,
    )
    return ImageWatermarkAttacksAndDetection(
        config=config,
        data_reader=read_local_image_dataset,
        dataset_dir=dataset_dir,
        image_pattern=original_image_pattern,
        watermarked_image_pattern=watermarked_image_pattern,
        message_pattern=message_pattern,
        even_shapes=even_shapes,
        output_size=output_size,
        batch_size=batch_size,
        num_workers=num_workers,
        num_samples=num_samples,
    )


def evaluate_image_watermark_end2end(
    dataset_dir: str,
    original_image_pattern: Optional[str] = None,
    even_shapes: bool = False,
    output_size: Optional[int] = None,
    message_size: Union[int, Sequence[int], None] = None,
    how_to_generate_message: Literal["per_batch", "per_dataset"] = "per_batch",
    metrics: Optional[Sequence[str]] = None,
    skip_quality_metrics_on_attacks: bool = False,
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
    """
    End-to-end watermarking-attack-detection task for a given dataset. This task is a sequence of two individual tasks,
    "evaluate_image_watermark_generation" and "evaluate_image_watermark_attacks_and_detection", and it re-uses most of the
    arguments from those tasks.

    Similar to generation task, 4 types of datasets are supported (HuggingFace dataset, local, user-defined, and on-the-fly).

    Args:
        dataset_dir (str, optional): Path to the dataset directory. If provided, evaluation is performed on local data.
        image_pattern (Optional[str], optional): Glob pattern to match image files in the dataset directory.
        even_shapes (bool, optional): Whether to enforce even shapes for images.
        output_size (Optional[int], optional): If provided, resizes images to have this size.
        metrics (Optional[Sequence[str]], optional): List of metric names to compute during evaluation.
        metrics_device (Optional[str], optional): Device to use for metric computation (e.g., "cpu", "cuda").
        seed (Optional[int], optional): Random seed for reproducibility.
        cache_dir (Optional, str): If specified, the intermediate data (generated watermarked contents and secret message) will be saved here.
        result_dir (Union[str, Path], optional): Directory to save evaluation results.
        keep_data_in_memory (Opational, bool): Whether to keep all intermediate and result in the memory instead of saving and writing them
            to hard disk. Suitable for small datasets for quick and clean experiments. This parameter cannot be co-specified with `result_dir` and `cache_dir`
        data_subdir (str, optional): This specifies the subdirectory within the dataset directory to use. Defaults to "".
        ids_to_save (Optional[str], optional): Comma-separated list of sample IDs to save.
        overwrite (bool, optional): Whether to overwrite existing results in the result directory.
        message_size (Union[int, Sequence[int], None], optional): Size(s) of the watermark message(s) to generate.
        how_to_generate_message (Literal["per_batch", "per_dataset"], optional): Strategy for generating messages.
        attacks (Union[Sequence[AttackConfig], Sequence[str], None]): List of attacks to apply.
            This can be a list of AttackConfig, or a list of attack names (strings) that map to default
            configurations, which are defined in omnisealbench/attacks/audio_attacks.yaml, or a path to
            a YAML file defining the attack configurations.
        detection_bits (int): Number of bits to use for detection.
        detection_threshold (float): Threshold for detection.
        message_threshold (float): Threshold for message detection.
        skip_quality_metrics_on_attacks (bool): Whether to skip quality metrics when evaluating attacks.
        batch_size (int, optional): Number of samples per batch.
        num_workers (int, optional): Number of worker processes for data loading.
        num_samples (Optional[int], optional): Number of samples to evaluate.
    """

    if attacks == "all":
        attacks = load_default_attacks(None, attacks_source="image")  # type: ignore
    if isinstance(attacks, str) or (isinstance(attacks, (tuple, list)) and len(attacks) > 0 and isinstance(attacks[0], str)):
        attacks = load_default_attacks(attacks, attacks_source="image")  # type: ignore

    config = End2EndWatermarkingConfig(
        metrics=metrics or [],
        cache_dir=cache_dir,
        result_dir=result_dir,
        ids_to_save=ids_to_save,
        attacks=attacks,  # type: ignore
        skip_quality_metrics_on_attacks=skip_quality_metrics_on_attacks,
        metrics_device=metrics_device,
        detection_threshold=detection_threshold,
        message_threshold=message_threshold,
        detection_bits=detection_bits,
        seed=seed or 42,
        overwrite=overwrite,
        message_size=message_size,
        how_to_generate_message=how_to_generate_message,
    )
    return ImageWatermarkEnd2End(
        config=config,  # type: ignore
        data_reader=read_local_image_dataset,
        dataset_dir=dataset_dir,
        image_pattern=original_image_pattern,
        output_size=output_size,
        even_shapes=even_shapes,
        batch_size=batch_size,
        num_workers=num_workers,
        num_samples=num_samples,
    )
