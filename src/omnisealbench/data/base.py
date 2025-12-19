# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Sequence,
                    Union)

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from omnisealbench.interface import ContentT
from omnisealbench.utils.distributed import get_world_size

ORIGINAL_KEY = "original"
MESSAGE_KEY = "message"


class InputBatch(NamedTuple):
    """
    A simple named tuple to represent a batch of samples in the dataset.
    It contains an index and the corresponding data.
    """
    indices: torch.Tensor
    data: ContentT


# FIXME: To be deprecated in V2
class GenerationDataset(Dataset):
    def __init__(self, elements: Union[List, Sequence[Any]], num_samples: int = None):
        if num_samples is not None:
            assert num_samples <= len(elements), f"There are {len(elements)} in total. num_samples required is : {num_samples}"
            self.num_samples = num_samples
        else:
            self.num_samples = len(elements)

        self.elements = elements

    def __len__(self):
        return self.num_samples

    def __getitem__(
        self,
        idx,
    ) -> Any:
        return idx, self.elements[idx]


class BaseDataset(Dataset):
    """Base class for Omniseal Bench datasets."""
    def __init__(self, dataset: Union[Dataset, str], column_name: str = "data", num_samples: Optional[int] = None):
        self.dataset = dataset
        self.num_samples = num_samples if num_samples is not None else len(dataset)  # type: ignore
        self.column_name = column_name

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Any:
        return idx, self.dataset[idx][self.column_name]  # type: ignore


def get_dataset_implementation(
    db_config: Dict[str, Any], num_samples: int = None, **additional_kwargs
) -> GenerationDataset:
    builder_func = db_config["builder_func"]
    additional_arguments = db_config["additional_arguments"]
    kwargs = {**additional_arguments, **additional_kwargs}
    db = builder_func(**kwargs)

    return GenerationDataset(db, num_samples=num_samples)


def collate_tensors_to_the_longest(batch) -> InputBatch:
    """
    Collate function to pad audio samples to the longest length in the batch.
    Returna tutple of indices and padded audio tensors.
    """
    # pad_sequence pads along the first dimension, so we need to transpose the audio tensors
    indices = [idx for idx, _ in batch]
    data = [item.t() for _, item in batch]  # (1, samples) -> (samples, 1)
    padded_audio = pad_sequence(data, batch_first=True)
    indices_tensor = torch.tensor(indices, dtype=torch.int32)
    return InputBatch(indices_tensor, padded_audio.transpose(1, 2))


def collate_dicts_to_the_longest(batch) -> InputBatch:
    """
    Collate function to pad audio samples to the longest length in the batch.
    Returns a tuple of indices and a dictionary of padded audio tensors.
    """
    indices = []
    batch_dict = defaultdict(list)
    for i, d in batch:
        indices.append(i)
        for k, v in d.items():
            if k == MESSAGE_KEY and (isinstance(v, torch.Tensor) and v.ndim == 1):
                # Do not pad the message, keep it as is
                batch_dict[k].append(v)
            else:
                batch_dict[k].append(v.t())  # (1, samples) -> (samples, 1)

    for k, v in batch_dict.items():
        if k == MESSAGE_KEY and (isinstance(v[0], torch.Tensor) and v[0].ndim == 1):
            # Do not pad the message, keep it as is
            batch_dict[k] = torch.stack(v)  # type: ignore
        else:
            # Pad the audio tensors to the longest length in the batch
            # and transpose them back to (samples, 1)
            batch_dict[k] = pad_sequence(v, batch_first=True).transpose(1, 2)  # type: ignore

    return InputBatch(torch.tensor(indices, dtype=torch.int32), batch_dict)


def collate_to_the_longest(batch) -> InputBatch:
    """
    Collate function to pad audio samples to the longest length in the batch.
    Returns a tuple of indices and padded audio tensors.
    """
    if isinstance(batch[0][1], dict):
        return collate_dicts_to_the_longest(batch)
    else:
        return collate_tensors_to_the_longest(batch)


def collate_tensors_by_length(batch, max_length: int) -> InputBatch:
    """
    Collate function to pad audio samples to a fixed length.
    Returns a tuple of indices and tensor of padded audio samples.
    """
    indices = []
    audios = []
    for i, audio in batch:
        length = audio.size(-1)
        if length > max_length:
            audio = audio[:, :max_length]
        else:
            padding = max_length - length
            audio = torch.nn.functional.pad(audio, (0, padding))
        audios.append(audio)
        indices.append(i)

    return InputBatch(torch.tensor(indices, dtype=torch.int32), torch.stack(audios))


def collate_dicts_by_length(batch, max_length: int) -> InputBatch:
    """
    Collate function to pad audio samples to a fixed length.
    Returns a tuple of indices and tensor of padded audio samples.
    """
    indices = []
    batch_dict = defaultdict(list)
    for i, d in batch:
        _item = next(iter(d.values()))  # Get the first item in the dict
        length = _item.size(-1)
        if length > max_length:
            for k, v in d.items():
                batch_dict[k].append(v[:, :max_length])
        else:
            padding = max_length - length
            for k, v in d.items():

                # We do not pad the message
                if k == MESSAGE_KEY:
                    batch_dict[k].append(v)
                else:
                    batch_dict[k].append(torch.nn.functional.pad(v, (0, padding)))
        indices.append(i)

    for k, v in batch_dict.items():
        batch_dict[k] = torch.stack(v)  # type: ignore

    return InputBatch(torch.tensor(indices, dtype=torch.int32), batch_dict)


def collate_by_length(batch, max_length: int) -> InputBatch:
    """
    Collate function to pad audio samples to a fixed length.
    Returns a tuple of indices and tensor of padded audio samples.
    """
    if isinstance(batch[0][1], dict):
        return collate_dicts_by_length(batch, max_length)
    else:
        return collate_tensors_by_length(batch, max_length)


def collate_tensors_even(batch) -> InputBatch:
    """
    Collate function that does not perform any padding, since the data is already evenly sized.
    Returns a tuple of indices and the data as a tensor.
    """
    indices = [idx for idx, _ in batch]
    data = [item for _, item in batch]
    return InputBatch(torch.tensor(indices, dtype=torch.int32), torch.stack(data))


def collate_dicts_even(batch) -> InputBatch:
    """
    Collate function that does not perform any padding, since the data is already evenly sized.
    Returns a tuple of indices and the data as a dictionary of tensors.
    """
    indices = []
    batch_dict = defaultdict(list)
    for i, d in batch:
        indices.append(i)
        for k, v in d.items():
            batch_dict[k].append(v)

    for k, v in batch_dict.items():
        batch_dict[k] = torch.stack(v)  # type: ignore

    return InputBatch(torch.tensor(indices, dtype=torch.int32), batch_dict)


def collate_even(batch) -> InputBatch:
    """
    Collate function that does not perform any padding, since the data is already evenly sized.
    Returns a tuple of indices and the data as a tensor or dictionary.
    """
    if isinstance(batch[0][1], dict):
        return collate_dicts_even(batch)
    else:
        return collate_tensors_even(batch)


def collate_tensors_uneven(batch) -> InputBatch:
    """
    Collate function that does not perform any padding, instead returns data as a list
    of (uneven) examples"""
    indices = [idx for idx, _ in batch]
    data = [item for _, item in batch]
    return InputBatch(torch.tensor(indices, dtype=torch.int32), data)


def collate_dicts_uneven(batch) -> InputBatch:
    """
    Collate function that does not perform any padding, instead returns data as a dictionary
    of (uneven) examples"""
    indices = []
    batch_dict = defaultdict(list)
    for i, d in batch:
        indices.append(i)
        for k, v in d.items():
            batch_dict[k].append(v)

    return InputBatch(torch.tensor(indices, dtype=torch.int32), batch_dict)


def collate_uneven(batch) -> InputBatch:
    """
    Collate function that does not perform any padding, instead returns data as a list
    of (uneven) examples or a dictionary of (uneven) examples.
    """
    if isinstance(batch[0][1], dict):
        return collate_dicts_uneven(batch)
    else:
        return collate_tensors_uneven(batch)


def build_data_loader(
    dataset: Dataset,
    batch_size: int = 1,
    num_samples: Optional[int] = None,
    collator: Optional[Callable] = None,
) -> DataLoader:
    """
    Build a DataLoader for the given dataset.

    Args:
        dataset (Dataset): The dataset to load.
        batch_size (int): Number of samples per batch.
        num_samples (int, optional): Number of samples to use from the dataset. If None, uses the full dataset.
        collator (callable, optional): Function to merge a list of samples to form a mini-batch.

    Returns:
        DataLoader: Configured DataLoader instance.
    """

    if get_world_size() > 1 and dist.is_initialized():
        # If in distributed mode, use DistributedSampler
        from torch.utils.data.distributed import DistributedSampler
        if collator is None:
            raise ValueError("Collator must be provided when using DistributedSampler.")
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)  # type: ignore
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collator,
            drop_last=False,
        )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        drop_last=False,
    )
