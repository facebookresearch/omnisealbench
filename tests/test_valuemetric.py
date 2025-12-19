# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pytest

from omnisealbench.attacks import valuemetric as vm


@pytest.fixture
def sample_batch():
    torch.manual_seed(0)
    return torch.rand(2, 3, 32, 32)


@pytest.mark.parametrize(
    "transform_cls,strengths",
    [
        (vm.Brightness, [0.5, 1.5]),
        (vm.Contrast, [0.5, 1.5]),
        (vm.Saturation, [0.5, 1.5]),
        (vm.Hue, [-0.5, -0.25, 0.25, 0.5]),
        (vm.JPEG, [40, 60, 80]),
        (vm.GaussianBlur, [3, 5, 9]),
        (vm.MedianFilter, [3, 5, 9]),
    ],
)
@torch.no_grad()
def test_transforms_execute(transform_cls, strengths, sample_batch):
    transform = transform_cls()

    for strength in strengths:
        image_out, mask_out = transform(sample_batch.clone(), None, strength)

        assert isinstance(image_out, torch.Tensor)
        assert image_out.shape == sample_batch.shape
        assert mask_out is None
        assert torch.isfinite(image_out).all()


@torch.no_grad()
def test_compression_helpers(sample_batch):
    single_image = sample_batch[0]

    jpeg_result = vm.jpeg_compress(single_image, quality=70)
    webp_result = vm.webp_compress(single_image, quality=70)

    assert jpeg_result.shape == single_image.shape
    assert webp_result.shape == single_image.shape
    assert torch.isfinite(jpeg_result).all()
    assert torch.isfinite(webp_result).all()
    assert (0.0 <= jpeg_result).all() and (jpeg_result <= 1.0).all()
    assert (0.0 <= webp_result).all() and (webp_result <= 1.0).all()


@torch.no_grad()
def test_median_filter_variants(sample_batch):
    kernel = 3
    vanilla = vm.median_filter(sample_batch, kernel)
    chunked = vm.median_filter_in_chunks(sample_batch, kernel, chunk_size=1)
    quantile = vm.median_filter_quantile(sample_batch, kernel, chunk_size=1)

    assert vanilla.shape == sample_batch.shape
    assert chunked.shape == sample_batch.shape
    assert quantile.shape == sample_batch.shape
    assert torch.isfinite(vanilla).all()
    assert torch.isfinite(chunked).all()
    assert torch.isfinite(quantile).all()
