# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# Handle audio input data for tasks.

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Union

import torch
import torchaudio  # type: ignore[import-untyped]

from omnisealbench.data.base import (MESSAGE_KEY, ORIGINAL_KEY, BaseDataset,
                                     build_data_loader, collate_by_length,
                                     collate_even, collate_to_the_longest,
                                     collate_uneven)
from omnisealbench.utils.common import message_to_tensor, resolve

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    DatasetT = Union[Dataset, torch.utils.data.Dataset, IterableDataset]


class HFAudioDataset(BaseDataset):
    """
    A dataset class for handling audio data that has the Hugging Face Dataset format.
    Inherits from BaseDataset to provide basic dataset functionality.
    """

    def __init__(
        self,
        dataset: "DatasetT",
        column_name: str = "audio",
        num_samples: Optional[int] = None,
        sample_rate: Optional[int] = None,
):
        super().__init__(dataset, column_name, num_samples)  # type: ignore

        inferred_sr = self._inspect_sample_rate()

        # If sample_rate is not provided, use the inferred sampling rate
        self.sample_rate = sample_rate or inferred_sr

        # resample if the inferred sampling rate is different from the desired sampling rate
        if inferred_sr is not None and sample_rate is not None and inferred_sr != sample_rate:
            self.resampler = torchaudio.transforms.Resample(orig_freq=inferred_sr, new_freq=sample_rate, dtype=torch.float32)
        else:
            self.resampler = None

    def _inspect_sample_rate(self) -> Optional[int]:
        # Get the original sampling rate from the first sample

        sample_rate = None
        if hasattr(self.dataset, "features") and self.dataset.features is not None and "audio" in self.dataset.features:
            sample_rate = self.dataset.features["audio"].sampling_rate
        else:
            # Fallback: try to get sampling rate from the first sample. This only works for Map datasets
            try:
                first_sample = self.dataset[0]
                if "audio" in first_sample and "sampling_rate" in first_sample["audio"]:
                    sample_rate = first_sample["audio"]["sampling_rate"]
                else:
                    raise ValueError("No sampling rate found in dataset features or first sample.")
            except KeyError as _:  # type: ignore[catch]
                pass
        return sample_rate

    def to_tensor(self, sample):
        if isinstance(sample, dict):
            audio = torch.tensor(sample["array"])
        elif isinstance(sample, (list, tuple)):
            audio = torch.tensor(sample)
        else:
            raise ValueError(f"Unsupported audio format: {type(sample)}")

        # Add channel dimension if missing
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        return audio

    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.
        Args:
            idx (int): Index of the item to retrieve.
        Returns:
            dict: A dictionary containing the audio data and its metadata.
        """
        idx, sample = super().__getitem__(idx)
        audio = self.to_tensor(sample)
        # Convert to mono-channel if necessary
        if audio.size(0) > 1:
            # Convert multi-channel audio to mono
            audio = audio.mean(dim=0, keepdim=True)

        # resample if necessary
        if self.resampler is not None:
            audio = self.resampler(audio.to(torch.float32))

        return idx, audio


class LocaldAudioDataset(BaseDataset):
    """
    A dataset class that loads audio files from different directories: original and processed (e.g. watermarked).
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        audio_pattern: str = "*.wav",
        watermarked_audio_pattern: Optional[str] = None,
        message_pattern: Optional[str] = None,
        processed_data_name: str = "",
        padding_strategy: Literal["longest", "fixed", "even", "uneven"] = "even",
        num_samples: Optional[int] = None,
        sample_rate: Optional[int] = None
    ):
        self.sample_rate = sample_rate
        self.padding_strategy = padding_strategy
        self.data_dir = resolve(data_dir)
        self.processed_data_name = processed_data_name
        self.message_pattern = message_pattern

        self.data_paths = self.get_paths(audio_pattern)
        if watermarked_audio_pattern is not None:
            self.processed_data_paths = self.get_paths(watermarked_audio_pattern)
            assert len(self.data_paths) == len(self.processed_data_paths), \
                (
                    "The number of original and processed audio files must match. "
                    f"Found {len(self.data_paths)} original files in {self.data_paths}"
                    f"and {len(self.processed_data_paths)} processed files in {self.processed_data_paths}."
                )

        if message_pattern is not None:
            self.message_paths = self.get_paths(message_pattern)
            assert len(self.data_paths) == len(self.message_paths) or len(self.message_paths) == 0, \
                (
                    "The number of original audio files and messages must match, or messages can be empty."
                    f"Found {len(self.data_paths)} original files in {self.data_paths} "
                    f"and {len(self.message_paths)} messages in {self.message_paths}."
                )

        if num_samples is not None:
            assert num_samples <= len(self.data_paths), f"There are {len(self.data_paths)} auios in total, but configured num_samples = {num_samples}"
            self.num_samples = num_samples
        else:
            self.num_samples = len(self.data_paths)

    def get_paths(self, pattern: str) -> List[Path]:
        def filename_to_index(filename: str) -> int:
            """
            Extract the index from the filename.
            Assumes filenames are formatted as 'audio_<index>.wav'.
            """
            return int(filename.split('_')[-1].split('.')[0])
        files = sorted(
            Path(self.data_dir).rglob(pattern),
            key=lambda x: filename_to_index(x.name),
        )
        return files

    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.
        Args:
            idx (int): Index of the item to retrieve.
        Returns:
            tuple: A tuple containing the index and the audio tensor.
        """
        data_path = self.data_paths[idx]
        data, sr = torchaudio.load(data_path)

        if self.sample_rate is not None and sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            data = resampler(data)

        if  hasattr(self, 'processed_data_paths'):
            if len(self.processed_data_paths) <= idx:
                processed_path = None
            else:
                processed_path = self.processed_data_paths[idx]
        else:
            processed_path = None

        if hasattr(self, 'message_paths'):
            if len(self.message_paths) <= idx:
                message_path = None
            else:
                message_path = self.message_paths[idx]
        else:
            message_path = None

        if message_path is None and processed_path is None:
            return idx, data

        data_dict = {ORIGINAL_KEY: data}
        if processed_path is not None:
            processed_data, sr = torchaudio.load(processed_path)
            assert data.shape == processed_data.shape, \
                f"Original and processed audio shapes do not match: {data.shape} vs {processed_data.shape}"

            if self.sample_rate is not None and sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                processed_data = resampler(processed_data)

            data_dict[self.processed_data_name] = processed_data

        if message_path is not None:
            with open(message_path, 'r', encoding="utf-8") as f:
                msg_tensor = message_to_tensor(f.read().strip())
            data_dict[MESSAGE_KEY] = msg_tensor

        return idx, data_dict


def read_hf_audio_dataset(
    dataset_name: str,
    name: Optional[str] = None,
    split: str = "test",
    padding_strategy: Literal["longest", "fixed", "even", "uneven"] = "even",
    max_length: Optional[int] = None,
    sample_rate: Optional[int] = None,
    batch_size: int = 1,
    num_samples: Optional[int] = None,
):
    """
    Load an audio dataset from Hugging Face and return a list of audio samples.

    Args:
        dataset_name (str): The name of the dataset to load.
        name (str, optional): The specific configuration of the dataset to load.
        split (str): The split of the dataset to load (default is "test").
        padding_strategy (Literal["longest", "fixed"]): The padding strategy to use.
            - "longest": Pads all samples to the length of the longest sample in the batch
            - "fixed": Pads all samples to a fixed length specified by `max_length`.
        max_length (int, optional): The fixed length to pad samples to when using "fixed" padding strategy.
        sample_rate (int, optional): The sampling rate to use for the audio samples. If not provided,
            it will be inferred from the dataset features or the first sample. If provided, and the 
            inferred sampling rate is different, the audio will be resampled to this rate.
        batch_size (int): The number of samples per batch (default is 1).
        num_samples (int, optional): The max number of samples to load from the dataset. 
            If None, all samples will be loaded.

    Returns:
        data loader: A PyTorch DataLoader containing the audio samples.
        dict: A dictionary containing the sampling rate used for the audio samples
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    _dataset = load_dataset(dataset_name, name=name, split=split, trust_remote_code=True)
    dataset = HFAudioDataset(_dataset, column_name="audio", num_samples=num_samples, sample_rate=sample_rate)

    # Pad data based on the specified padding strategy
    if padding_strategy == "longest":
        collate_fn = collate_to_the_longest
    elif padding_strategy == "fixed":
        assert max_length is not None, "max_length must be specified for fixed padding strategy."
        collate_fn = partial(collate_by_length, max_length=max_length)
    elif padding_strategy == "uneven":
        collate_fn = collate_uneven
    else:
        collate_fn = collate_even

    # Build the data loader in local or distributed mode
    data_loader = build_data_loader(
        dataset,  # type: ignore
        batch_size=batch_size,
        collator=collate_fn,
    )
    return data_loader, {"sample_rate": dataset.sample_rate}


def read_local_audio_dataset(
    dataset_dir: Union[str, Path],
    audio_pattern: str = "*.wav",
    watermarked_audio_pattern: Optional[str] = None,
    message_pattern: Optional[str] = None,
    padding_strategy: Literal["longest", "fixed", "even", "uneven"] = "even",
    max_length: Optional[int] = None,
    sample_rate: Optional[int] = None,
    batch_size: int = 1,
    num_samples: Optional[int] = None,
):
    """
    Load a local audio dataset and return a data loader.

    Args:
        dataset_dir (str or Path): Directory containing the original audio files.
        processed_data_dir (str or Path, optional): Directory containing the processed audio files.
        audio_pattern (str): Pattern to match original audio files.
        watermarked_audio_pattern (str, optional): Pattern to match watermarked audio files.
        padding_strategy (Literal["longest", "fixed", "even", "uneven"]): Padding strategy to use.
        max_length (int, optional): Maximum length for fixed padding strategy.
        sample_rate (int, optional): Sampling rate for the audio samples.
        batch_size (int): Number of samples per batch.

    Returns:
        data loader: A PyTorch DataLoader containing the audio samples.
        dict: A dictionary containing the sampling rate used for the audio samples and input directory
    """
    dataset = LocaldAudioDataset(
        data_dir=dataset_dir,
        audio_pattern=audio_pattern,
        watermarked_audio_pattern=watermarked_audio_pattern,
        processed_data_name="watermarked",
        message_pattern=message_pattern,
        padding_strategy=padding_strategy,
        num_samples=num_samples,  # Load all samples by default
        sample_rate=sample_rate
    )

    # Pad data based on the specified padding strategy
    if padding_strategy == "longest":
        collate_fn = collate_to_the_longest
    elif padding_strategy == "fixed":
        assert max_length is not None, "max_length must be specified for fixed padding strategy."
        collate_fn = partial(collate_by_length, max_length=max_length)
    elif padding_strategy == "uneven":
        collate_fn = collate_uneven
    else:
        collate_fn = collate_even

    # Build the data loader in local or distributed mode
    data_loader = build_data_loader(
        dataset,  # type: ignore
        batch_size=batch_size,
        collator=collate_fn,
    )

    return (
        data_loader,
        {"sample_rate": dataset.sample_rate, "input_dir": str(dataset.data_dir), "data_len": len(dataset)},
    )
