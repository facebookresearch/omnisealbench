# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch

from omnisealbench.utils.common import batch_safe


@batch_safe
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


@batch_safe
def fake_metrics(
    x: torch.Tensor,
    y: torch.Tensor,
    sr: int,
    max_length: int = 1000,
) -> torch.Tensor:
    """
    Fake audio metrics for testing purposes.
    """
    if sr != 16000:
        import julius
        x = julius.resample_frac(x, sr, 16000)
        y = julius.resample_frac(y, sr, 16000)
    
    x = x[:, :max_length]
    y = y[:, :max_length]
    
    return torch.norm(x - y, p=2)  # L2 norm as a fake metric


def test_batch_safe():
    # Test with a single input
    result1 = add(torch.Tensor([1]), torch.Tensor([2]))
    assert result1.item() == 3

    result2 = add(
        list(torch.Tensor([1, 2])), 
        list(torch.Tensor([3, 4])),
    )

    assert result2.tolist() == [4, 6]


@pytest.mark.skip(reason="debug")
def test_batch_safe_parallel():
    # Test with multiple inputs in parallel
    os.environ["BATCH_SAFE_PARALLEL"] = "2"
    result = add(
        list(torch.Tensor([1, 2, 3, 4])),
        list(torch.Tensor([4, 5, 6, 7])),
    )
    assert result.tolist() == [5, 7, 9, 11]


def test_batch_safe_with_args():
    x = [
        torch.rand(1, 4096),
        torch.rand(1, 3775),
        torch.rand(1, 2953),
        torch.rand(1, 1734),
    ]
    y = [
        torch.rand(1, 2345),
        torch.rand(1, 4567),
        torch.rand(1, 1894),
        torch.rand(1, 1234),
    ]
    sr = 16000
    result = fake_metrics(x, y, sr, max_length=1000)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (4,)
