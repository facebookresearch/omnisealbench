# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# mypy: ignore-errors

from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

import matplotlib.pyplot as plt
import torch
import torchaudio

from omnisealbench.attacks.helper import AttackConfig, load_default_attacks
from omnisealbench.data.audio import read_local_audio_dataset
from omnisealbench.metrics.audio import (compute_snr, si_snr, stoi,
                                         tensor_pesq_batch)
from omnisealbench.tasks._detection import (WatermarkAttacksAndDetection,
                                            WatermarkAttacksConfig)
from omnisealbench.tasks._e2e import (End2EndWatermarkingConfig,
                                      WatermarkEnd2End, detection_config,
                                      generation_config)
from omnisealbench.tasks._generation import (WatermarkGeneration,
                                             WatermarkGenerationConfig)
from omnisealbench.tasks.base import WATERMARKED_KEY, DataReaderT, Task
from omnisealbench.utils.common import resolve

# Filename mapping


def _save_spectrogram(waveform, sample_rate, title, fig_path):
    """
    Save a spectrogram image of the given audio waveform.

    Args:
        waveform (torch.Tensor): The audio waveform to visualize.
        sample_rate (int): The sample rate of the audio.
        title (str): The title to display on the spectrogram.
        fig_path (str or Path): The file path to save the spectrogram image.
    """
    waveform = waveform.squeeze().detach().cpu().numpy()

    plt.clf()  # Clear the current figure
    plt.specgram(waveform, Fs=sample_rate)

    plt.title(f"{title} - Spectrogram")
    plt.savefig(fig_path)
    plt.close()


def _save_audios(
    audios: torch.Tensor,
    indices: List[int],
    output_dir: Path,
    sample_rate: int = 16000,
    prefix_file: str = "",
    prefix_title: str = "",
) -> None:
    """
    Save the audio tensor to a file (.wav and .png for the spectrogram).
    Args:
        audio (torch.Tensor): The audio tensor to save.
        filename (str): The name of the file to save the audio to.
        sample_rate (int): The sample rate of the audio.
    """

    if audios.dim() == 2:
        # Add batch dimension if missing
        audios = audios.unsqueeze(0)

    for idx, audio in zip(indices, audios):
        filename = f"{prefix_file}_{idx:05d}.wav"

        torchaudio.save(
            output_dir / filename,
            audio.detach().cpu(),
            sample_rate=sample_rate,
        )

        _save_spectrogram(
            audio,
            sample_rate,
            title=f"{prefix_title} audio",
            fig_path=output_dir / f"{prefix_file}_{idx:05d}.png"
        )

class AudioWatermarkGeneration(WatermarkGeneration):
    """
    A task for generating audio watermarks.
    """

    def compute_metrics(  # type: ignore
        self,
        orig: torch.Tensor,
        watermarked: torch.Tensor,
        message: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute the quality metrics for the watermarked audio results."""

        audio = orig
        wm_audio = watermarked
        sr = kwargs.get("sample_rate", 16000)

        scores = {}

        if "snr" in self.config.metrics:
            scores["snr"] = compute_snr(audio, wm_audio)

        if "pesq" in self.config.metrics:
            scores["pesq"] = tensor_pesq_batch(wm_audio, audio, sr=sr, n_processor=self.num_workers)

        if "stoi" in self.config.metrics:
            scores["stoi"] = stoi(wm_audio, audio, sr=sr)

        if "sisnr" in self.config.metrics:
            scores["si_snr"] = si_snr(wm_audio, audio)

        return scores

    def save_data(
        self,
        orig: Union[torch.Tensor, List[torch.Tensor]],
        watermarked: Union[torch.Tensor, List[torch.Tensor]],
        indices: List[int],
        config: Optional[WatermarkGenerationConfig] = None,
        **kwargs,  # type: ignore
    ) -> None:
        """
        Save the results of the watermark generation to the specified result directory.
        Args:
            watermarked_results (ResultT): The results of the watermark generation.
        """
        sample_rate = kwargs.get("sample_rate", 16000)
        config = config or self.config
        watermark_dir = resolve(config.result_dir) / config.watermark_subdir
        watermark_dir.mkdir(exist_ok=True, parents=True)
        _save_audios(
            watermarked,
            indices,
            watermark_dir,
            sample_rate=sample_rate,
            prefix_file="wmd_",
            prefix_title="Watermarked",
        )

        # If data_subdir is provided: Write original audio files
        if config.data_subdir:
            data_dir = resolve(config.result_dir) / config.data_subdir
            data_dir.mkdir(exist_ok=True, parents=True)

            _save_audios(
                orig,
                indices,
                data_dir,
                sample_rate=sample_rate,
                prefix_file="",
                prefix_title="Original",
            )


class AudioWatermarkAttacksAndDetection(WatermarkAttacksAndDetection):
    """
    A task for audio watermark attacks and detection.
    This class extends the AudioWatermarkGeneration class to include watermark attacks and detection.
    """

    def skip_quality_metrics(self, config: WatermarkAttacksConfig) -> bool:
        for skip_by_name in ["speed", "speed_julius"]:
            if skip_by_name in config.attacks.func:
                return True

        return False

    def compute_quality_metrics(  # type: ignore
        self,
        wm_input: torch.Tensor,
        orig_input: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the quality metrics for the watermarked audio results.
        Args:
            wm_input (torch.Tensor): The watermarked audio input.
            orig_input (torch.Tensor): The original audio input.
        Returns:
            Dict[str, torch.Tensor]: A dictionary of computed quality metrics.
        """
        sr = kwargs.get("sample_rate", 16000) or 16000

        scores = {}

        if "snr" in self.config.metrics:
            scores["snr"] = compute_snr(orig_input, wm_input)

        if "pesq" in self.config.metrics:
            scores["pesq"] = tensor_pesq_batch(wm_input, orig_input, sr=sr, n_processor=self.num_workers)

        if "stoi" in self.config.metrics:
            scores["stoi"] = stoi(wm_input, orig_input, sr=sr)

        if "sisnr" in self.config.metrics:
            scores["si_snr"] = si_snr(wm_input, orig_input)

        return scores

    def save_data(
        self,
        orig: torch.Tensor,
        watermarked: torch.Tensor,
        indices: Sequence[int],
        config: Optional[WatermarkGenerationConfig] = None,
        **kwargs,  # type: ignore
    ) -> None:
        sample_rate = kwargs.get("sample_rate", 16000)
        config = config or self.config

        _save_audios(
            orig,
            indices,  # type: ignore
            Path(config.result_dir) / config.data_subdir,
            sample_rate=sample_rate,
            prefix_file="wmd_audio",
            prefix_title="Watermarked",
        )

        _save_audios(
            watermarked,
            indices,
            Path(config.result_dir) / config.data_subdir,
            sample_rate=sample_rate,
            prefix_file="audio",
            prefix_title="Original",
        )


class AudioWatermarkEnd2End(WatermarkEnd2End):

    def get_generation_task(
        self,
        config: End2EndWatermarkingConfig,
        data_reader: Optional[DataReaderT] = None,
        num_workers: int = 0,
        **data_args,
    ) -> WatermarkGeneration:
        wm_config = generation_config(config)
        return AudioWatermarkGeneration(
            config=wm_config,
            data_reader=data_reader,
            num_workers=num_workers,
            **data_args,
        )

    def get_detection_task(
        self,
        config: End2EndWatermarkingConfig,
        num_workers: int = 0,
        **data_args,
    ) -> WatermarkAttacksAndDetection:
        det_config = detection_config(config)
        data_args["audio_pattern"] = "data/*.wav"
        return AudioWatermarkAttacksAndDetection(
            config=det_config,
            data_reader=read_local_audio_dataset,
            num_workers=num_workers,
            watermarked_audio_pattern="watermarks/*.wav",
            message_pattern="watermarks/*.txt",
            **data_args,
        )


########################################################################
# A factory of audio taks with ready-to-use configurations.
########################################################################


def evaluate_audio_watermark_generation(
    dataset_type: str = "",
    dataset_name: str = "",
    dataset_hf_subset: Optional[str] = None,
    dataset_split: str = "",
    dataset_dir: str = "",
    audio_pattern: str = "*.wav",
    data_reader: Optional[DataReaderT] = None,
    metrics: Optional[Sequence[str]] = None,
    metrics_device: Optional[str] = None,
    seed: Optional[int] = None,
    result_dir: Union[str, Path] = "",
    data_subdir: str = "",
    ids_to_save: Optional[str] = None,
    overwrite: bool = False,
    padding_strategy: Literal["longest", "fixed", "even", "uneven"] = "fixed",
    max_length: Optional[int] = None,
    sample_rate: Optional[int] = None,
    message_size: Union[int, Sequence[int], None] = None,
    how_to_generate_message: Literal["per_batch", "per_dataset"] = "per_batch",
    batch_size: int = 1,
    num_workers: int = 0,
    num_samples: Optional[int] = None,
) -> Task:
    """This task evaluates audio watermark generation on a given dataset and list of metrics.
    There are four types of datasets that are supported:
    
    1. HuggingFace dataset:
       This type of dataset is loaded from the HuggingFace Hub. The dataset is specified by the parameter `dataset_name`.
       For this to work, the `datasets` library must be installed.

    2. Local audio dataset:
       This loads audio files of a specific patterns from a local directory specified by the parameter `dataset_dir`.

    3. Custom data reader function:
       This allows users to provide their own function for reading audio data. The function should return an iterator of
       data batches of type `omnisealbench.data.base.InputBatch`. This option is when `data_reader` is specified.

    4. On-the-fly data:
        This skips the dataset at pipeline definition time and allows to pass an iterator of InputBatch's at runtime. This option
        corresponds to case where none of the above inputs (dataset_name, dataset_dir, data_reader) is specified.
        # TODO: Implement on-the-fly data loading
        Expected dimension of the input: (batch_size, channels, time)

    The precedence order is 1 > 2 > 3 > 4, i.e. if one option is specified, it will take priority over the others and other
    arguments will be ignored.

    Args:
        dataset_name (str, optional): When specified, this will load a HuggingFace dataset.
        dataset_hf_subset (Optional[str], optional): This corresponds to the parameter `name` in `datasets.load_dataset`. Defaults to None.
        dataset_split (str, optional): This corresponds to the parameter `split` in `datasets.load_dataset`.
        dataset_dir (str, optional): When specified, this will load local audio files  from the given directory.
        audio_pattern (str, optional): This specifies the pattern for matching local audio files. Defaults to "data/*.wav".
        data_reader (Optional[DataReaderT], optional): This allows users to provide their own function for reading audio data. Defaults to None.
        metrics (Optional[Sequence[str]], optional): This specifies the metrics to evaluate. Defaults to None.
        metrics_device (Optional[str], optional): This specifies the device to use for metrics computation. Defaults to None.
        seed (Optional[int], optional): This specifies the random seed for reproducibility. Defaults to None.
        result_dir (Union[str, Path], optional): This specifies the directory where the results will be saved. Defaults to None, i.e. 
            the data will not be saved, but only metrics are returned from the function call.
        data_subdir (str, optional): This specifies the subdirectory within the dataset directory to use. Defaults to "".
        ids_to_save (Optional[str], optional): This specifies the IDs of the samples to save. When not specified, but the `result_dir` is defined,
            all samples will be saved to `result_dir`. Likewise, if `result_dir` is not specified, this parameter will be ignored.
        overwrite (bool, optional): This specifies whether to overwrite existing files in result_dir. Defaults to False.
        padding_strategy (Literal["longest", "fixed", "even", "uneven"], optional): This specifies the padding strategy to use. Defaults to "fixed".
        max_length (Optional[int], optional): This specifies the maximum length of the audio samples. Defaults to None.
        sample_rate (Optional[int], optional): The sample_rate to read the input data. Defaults to None.
        message_size (Union[int, Sequence[int], None], optional): This specifies the size of the message to embed in the audio. Defaults to None.
        how_to_generate_message (Literal["per_batch", "per_dataset"], optional): This specifies how to generate the message.
           Permissible values are "per_dataset" (same messages for the whole dataset) and "per_batch" (each batch receives a different message). 
           Defaults to "per_batch".
        batch_size (int, optional): This specifies the batch size for processing audio samples. Defaults to 1.
        num_workers (int, optional): This specifies the number of worker processes for data loading. Defaults to 0, i.e. no parallelization.
        num_samples (Optional[int], optional): This specifies the number of samples to process. Defaults to None.

    Raises:
        ValueError: If there are invalid inputs 

    Returns:
        Task: An instance of the Task class.
    """

    config = WatermarkGenerationConfig(
        metrics=metrics or [],
        metrics_device=metrics_device,
        seed=seed or 42,
        overwrite=overwrite,
        message_size=message_size,
        result_dir=result_dir,
        ids_to_save=ids_to_save,
        how_to_generate_message=how_to_generate_message,
        data_subdir=data_subdir,
    )

    if dataset_type == "hf":
        assert dataset_name, "When loading HuggingFace dataset, `dataset_name` must be specified"
        assert not dataset_dir, "When loading HuggingFace dataset (`dataset_name` is specified), `dataset_dir` must not be provided."
        assert not data_reader, "When loading HuggingFace dataset (`dataset_name` is specified), `data_reader` must not be provided."
    elif dataset_type == "local":
        assert dataset_dir, "When loading local dataset, 'dataset_dir' must be specified."
        assert not dataset_name and not dataset_hf_subset and not dataset_split, "When loading local dataset, `dataset_name`, `dataset_hf_subset`, and `dataset_split` must not be specified."
        assert not data_reader, "When loading local dataset, `data_reader` must not be provided."

    if dataset_name:
        from omnisealbench.data.audio import read_hf_audio_dataset

        return AudioWatermarkGeneration(
            config=config,
            data_reader=read_hf_audio_dataset,
            dataset_name=dataset_name,
            name=dataset_hf_subset,
            split=dataset_split,
            padding_strategy=padding_strategy,
            max_length=max_length,
            sample_rate=sample_rate,
            batch_size=batch_size,
            num_samples=num_samples,
            num_workers=num_workers,
        )

    elif dataset_dir:
        if dataset_hf_subset or dataset_split or dataset_name:
            raise ValueError(
                "`dataset_hf_subset` and `split` are not supported when dataset_dir is used."
            )
        assert not data_reader, "When loading local audio dataset (`dataset_dir` is specified), `data_reader` should not be provided."

        return AudioWatermarkGeneration(
            config=config,
            data_reader=read_local_audio_dataset,
            dataset_dir=dataset_dir,
            audio_pattern=audio_pattern,
            batch_size=batch_size,
            num_workers=num_workers,
            padding_strategy=padding_strategy,
            max_length=max_length,
            sample_rate=sample_rate,
            num_samples=num_samples,
        )

    # We run the evaluation on the custom data, the data_reader is given by user 
    # (or empty if user wants to pass the reader's output at runtime)
    else:
        return AudioWatermarkGeneration(
            config=config,
            data_reader=data_reader,
            num_workers=num_workers,
        )


def evaluate_audio_watermark_attacks_and_detection(
    dataset_dir: str = "",
    audio_pattern: str = "data/*.wav",
    watermarked_audio_pattern: Optional[str] = "watermarks/*.wav",
    message_pattern: Optional[str] = "watermarks/*.txt",
    metrics: Optional[Sequence[str]] = None,
    skip_quality_metrics_on_attacks: bool = False,
    metrics_device: Optional[str] = None,
    detection_threshold: Optional[float] = None,
    message_threshold: Optional[float] = None,
    detection_bits: Optional[int] = None,
    seed: Optional[int] = None,
    result_dir: Union[str, Path] = "",
    ids_to_save: Optional[str] = None,
    overwrite: bool = False,
    attacks: Union[Sequence[AttackConfig], Sequence[str], None] = None,
    padding_strategy: Literal["longest", "fixed", "even", "uneven"] = "fixed",
    max_length: Optional[int] = None,
    sample_rate: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 0,
) -> Task:
    """
    Read watermarked audio files from a local directory, apply attacks to each audio then run the
    detector to detect the watermarks and calculate different  scores (bit accuracy,
    capacity, log10_p_value, p_value, TPR, FPR) on the detection results.

    By default, all audio files in the directory will be evaluated. If `audio_pattern` and `watermarked_audio_pattern`
    are specified, only the watermarked files that match `watermarked_audio_pattern` will be evaluated and
    the metrics will be computed against the original audio files that match `audio_pattern`.

    If `message_pattern` is specified, it will use the watermark messages that match `message_pattern` to compute
    message-related metrics (bit accuracy, capacity, etc.). Otherwise only TPR and FPR are returned.

    Additionally, other quality metrics (e.g., ssim, lpips) are also computed when specified by `metrics`.

    Args:
        dataset_dir (str): Path to the dataset directory.
        audio_pattern (Optional[str]): Pattern to match original audio files.
        watermarked_audio_pattern (Optional[str]): Pattern to match watermarked audio files.
        message_pattern (Optional[str]): Pattern to match watermark messages.
        even_shapes (bool): Whether to use even shapes for audio.
        metrics (Optional[Sequence[str]]): List of extra quality metrics to compute in addition to detection scores
        skip_quality_metrics_on_attacks (bool): Whether to skip quality metrics when evaluating attacks.
        metrics_device (Optional[str]): Device to use for metrics computation.
        detection_bits (int): Number of bits to use for detection.
        detection_threshold (float): Threshold for detection.
        message_threshold (float): Threshold for message detection.
        seed (Optional[int]): Random seed for reproducibility.
        result_dir (Union[str, Path]): Directory to save results.
        ids_to_save (Optional[str]): IDs of audios to save.
        overwrite (bool): Whether to overwrite existing results.
        attacks (Union[Sequence[AttackConfig], Sequence[str], None]): List of attacks to apply.
            This can be a list of AttackConfig, or a list of attack names (strings) that map to default
            configurations, which are defined in omnisealbench/attacks/audio_attacks.yaml, or a path to
            a YAML file defining the attack configurations.
        batch_size (int): Batch size for processing audio.
        num_workers (int): Number of workers for data loading.
        num_samples (Optional[int]): Number of samples to process.

    """

    if attacks == "all":
        attacks = load_default_attacks(None, attacks_source="audio")  # type: ignore
    if isinstance(attacks, str) or (isinstance(attacks, (tuple, list)) and len(attacks) > 0 and isinstance(attacks[0], str)):
        attacks = load_default_attacks(attacks, attacks_source="audio")  # type: ignore

    config = WatermarkAttacksConfig(
        metrics=metrics or [],
        skip_quality_metrics_on_attacks=skip_quality_metrics_on_attacks,
        metrics_device=metrics_device,
        seed=seed or 42,
        result_dir=result_dir,
        overwrite=overwrite,
        attacks=attacks,  # type: ignore
        ids_to_save=ids_to_save,
        detection_threshold=detection_threshold,
        message_threshold=message_threshold,
        detection_bits=detection_bits,
    )

    return AudioWatermarkAttacksAndDetection(
        config=config,
        data_reader=read_local_audio_dataset,
        dataset_dir=dataset_dir,
        audio_pattern=audio_pattern,
        watermarked_audio_pattern=watermarked_audio_pattern,
        message_pattern=message_pattern,
        processed_data_name=WATERMARKED_KEY,
        padding_strategy=padding_strategy,
        max_length=max_length,
        sample_rate=sample_rate,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def evaluate_audio_watermark_end2end(
    dataset_type: str = "",
    dataset_name: str = "",
    dataset_hf_subset: Optional[str] = None,
    dataset_split: str = "",
    dataset_dir: str = "",
    audio_pattern: str = "*.wav",
    data_reader: Optional[DataReaderT] = None,
    metrics: Optional[Sequence[str]] = None,
    metrics_device: Optional[str] = None,
    attacks: Union[Sequence[AttackConfig], Sequence[str], None] = None,
    skip_quality_metrics_on_attacks: bool = False,
    detection_threshold: Optional[float] = None,
    message_threshold: Optional[float] = None,
    detection_bits: Optional[int] = None,
    seed: Optional[int] = None,
    cache_dir: str = "",
    keep_data_in_memory: bool = False,
    result_dir: Union[str, Path] = "",
    ids_to_save: Optional[str] = None,
    overwrite: bool = False,
    padding_strategy: Literal["longest", "fixed", "even", "uneven"] = "fixed",
    max_length: Optional[int] = None,
    sample_rate: Optional[int] = None,
    message_size: Union[int, Sequence[int], None] = None,
    how_to_generate_message: Literal["per_batch", "per_dataset"] = "per_batch",
    batch_size: int = 1,
    num_workers: int = 0,
    num_samples: Optional[int] = None,
) -> AudioWatermarkEnd2End:
    """
    End-to-end watermarking-attack-detection task for a given dataset. This task is a sequence of two individual tasks,
    "evaluate_audio_watermark_generation" and "evaluate_audio_watermark_attacks_and_detection", and it re-uses most of the
    arguments from those tasks.

    Similar to generation task, 4 types of datasets are supported:
    
    1. HuggingFace dataset:
       This type of dataset is loaded from the HuggingFace Hub. The dataset is specified by the parameter `dataset_name`.
       For this to work, the `datasets` library must be installed.

    2. Local audio dataset:
       This loads audio files of a specific patterns from a local directory specified by the parameter `dataset_dir`.

    3. Custom data reader function:
       This allows users to provide their own function for reading audio data. The function should return an iterator of
       data batches of type `omnisealbench.data.base.InputBatch`. This option is when `data_reader` is specified.

    4. On-the-fly data:
        This skips the dataset at pipeline definition time and allows to pass an iterator of InputBatch's at runtime. This option
        corresponds to case where none of the above inputs (dataset_name, dataset_dir, data_reader) is specified.
        # TODO: Implement on-the-fly data loading
        Expected dimension of the input: (batch_size, channels, time)

    The precedence order is 1 > 2 > 3 > 4, i.e. if one option is specified, it will take priority over the others and other
    arguments will be ignored.

    Args:
        dataset_name (str, optional): When specified, this will load a HuggingFace dataset.
        dataset_hf_subset (Optional[str], optional): This corresponds to the parameter `name` in `datasets.load_dataset`. Defaults to None.
        dataset_split (str, optional): This corresponds to the parameter `split` in `datasets.load_dataset`.
        dataset_dir (str, optional): When specified, this will load local audio files  from the given directory.
        audio_pattern (str, optional): This specifies the pattern for matching local audio files. Defaults to "data/*.wav".
        data_reader (Optional[DataReaderT], optional): This allows users to provide their own function for reading audio data. Defaults to None.
        metrics (Optional[Sequence[str]], optional): This specifies the metrics to evaluate. Defaults to None.
        metrics_device (Optional[str], optional): This specifies the device to use for metrics computation. Defaults to None.
        seed (Optional[int], optional): This specifies the random seed for reproducibility. Defaults to None.
        cache_dir (Optional, str): If specified, the intermediate data (generated watermarked contents and secret message) will be saved here.
        result_dir (Union[str, Path], optional): This specifies the directory where the final results will be saved. Defaults to None, i.e. 
            the data will not be saved, but only metrics are returned from the function call.
        keep_data_in_memory (Opational, bool): Whether to keep all intermediate and result in the memory instead of saving and writing them
            to hard disk. Suitable for small datasets for quick and clean experiments. This parameter cannot be co-specified with `result_dir` and `cache_dir`
        data_subdir (str, optional): This specifies the subdirectory within the dataset directory to use. Defaults to "".
        ids_to_save (Optional[str], optional): This specifies the IDs of the samples to save. When not specified, but the `result_dir` is defined,
            all samples will be saved to `result_dir`. Likewise, if `result_dir` is not specified, this parameter will be ignored.
        overwrite (bool, optional): This specifies whether to overwrite existing files in result_dir. Defaults to False.
        padding_strategy (Literal["longest", "fixed", "even", "uneven"], optional): This specifies the padding strategy to use. Defaults to "fixed".
        max_length (Optional[int], optional): This specifies the maximum length of the audio samples. Defaults to None.
        sample_rate (Optional[int], optional): The sample_rate to read the input data. Defaults to None.
        message_size (Union[int, Sequence[int], None], optional): This specifies the size of the message to embed in the audio. Defaults to None.
        how_to_generate_message (Literal["per_batch", "per_dataset"], optional): This specifies how to generate the message.
           Permissible values are "per_dataset" (same messages for the whole dataset) and "per_batch" (each batch receives a different message). 
           Defaults to "per_batch".
        attacks (Union[Sequence[AttackConfig], Sequence[str], None]): List of attacks to apply.
            This can be a list of AttackConfig, or a list of attack names (strings) that map to default
            configurations, which are defined in omnisealbench/attacks/audio_attacks.yaml, or a path to
            a YAML file defining the attack configurations.
        detection_bits (int): Number of bits to use for detection.
        detection_threshold (float): Threshold for detection.
        message_threshold (float): Threshold for message detection.
        skip_quality_metrics_on_attacks (bool): Whether to skip quality metrics when evaluating attacks.
        batch_size (int, optional): This specifies the batch size for processing audio samples. Defaults to 1.
        num_workers (int, optional): This specifies the number of worker processes for data loading. Defaults to 0, i.e. no parallelization.
        num_samples (Optional[int], optional): This specifies the number of samples to process. Defaults to None.
    """
    if attacks == "all":
        attacks = load_default_attacks(None, attacks_source="audio")  # type: ignore
    if isinstance(attacks, str) or (isinstance(attacks, (tuple, list)) and len(attacks) > 0 and isinstance(attacks[0], str)):
        attacks = load_default_attacks(attacks, attacks_source="audio")  # type: ignore

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
        keep_data_in_memory=keep_data_in_memory,
    )

    if dataset_type == "hf":
        assert dataset_name, "When loading HuggingFace dataset, `dataset_name` must be specified"
        assert not dataset_dir, "When loading HuggingFace dataset (`dataset_name` is specified), `dataset_dir` must not be provided."
        assert not data_reader, "When loading HuggingFace dataset (`dataset_name` is specified), `data_reader` must not be provided."
    elif dataset_type == "local":
        assert dataset_dir, "When loading local dataset, 'dataset_dir' must be specified."
        assert not dataset_name and not dataset_hf_subset and not dataset_split, "When loading local dataset, `dataset_name`, `dataset_hf_subset`, and `dataset_split` must not be specified."
        assert not data_reader, "When loading local dataset, `data_reader` must not be provided."

    if dataset_name:
        from omnisealbench.data.audio import read_hf_audio_dataset

        return AudioWatermarkEnd2End(
            config=config,  # type: ignore
            data_reader=read_hf_audio_dataset,
            num_workers=num_workers,
            dataset_name=dataset_name,
            name=dataset_hf_subset,
            split=dataset_split,
            padding_strategy=padding_strategy,
            max_length=max_length,
            sample_rate=sample_rate,
            batch_size=batch_size,
            num_samples=num_samples,
        )

    if dataset_dir:
        if dataset_hf_subset or dataset_split or dataset_name:
            raise ValueError(
                "`dataset_hf_subset` and `split` are not supported when dataset_dir is used."
            )
        assert not data_reader, "When loading local audio dataset (`dataset_dir` is specified), `data_reader` should not be provided."

        return AudioWatermarkEnd2End(
            config=config,  # type: ignore
            data_reader=read_local_audio_dataset,
            dataset_dir=dataset_dir,
            audio_pattern=audio_pattern,
            batch_size=batch_size,
            num_workers=num_workers,
            padding_strategy=padding_strategy,
            max_length=max_length,
            sample_rate=sample_rate,
            num_samples=num_samples,
        )

    return AudioWatermarkEnd2End(
        config=config,  # type: ignore
        data_reader=data_reader,
        num_workers=num_workers,
    )
