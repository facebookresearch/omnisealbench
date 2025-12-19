# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import random
from functools import partial

import torch
from torch.utils.data import Dataset

from omnisealbench.data.base import (build_data_loader, collate_by_length,
                                     collate_to_the_longest)


class MockDataset(Dataset):
    def __init__(self, length=10):
        self.length = length
        self.data = [torch.randn((1, random.randint(1, 7))) for _ in range(length)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return a dummy tensor or any mock data
        return idx, self.data[idx]


def test_load_and_pad_data():
    dataset = MockDataset(length=10)
    lens = [item.shape[1] for item in dataset.data]
    batch_size = 5

    max_len_in_batch = [max(lens[i:i + batch_size]) for i in range(0, len(lens), batch_size)]

    assert len(dataset) == 10, "Dataset length should be 10"

    collate_fn = collate_to_the_longest
    
    data_loader = build_data_loader(
        dataset, batch_size=batch_size, collator=collate_fn
    )

    data_iter = iter(data_loader)
    
    for (_, batch), max_len in zip(data_iter, max_len_in_batch):
        
        assert batch.shape[0] == batch_size, f"Batch size should be {batch_size}"
        assert batch.shape[-1] == max_len, "Input values should be padded to the longest length in the batch"
    
    collate_fn = partial(collate_by_length, max_length=5)
    data_loader = build_data_loader(
        dataset, batch_size=5, collator=collate_fn
    )
    data_iter = iter(data_loader)
    for (_, batch) in data_iter:
        assert batch.shape[0] == batch_size, f"Batch size should be {batch_size}"
        assert batch.shape[-1] == 5, "Input values should be padded to 5"
