# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pytest
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from omnisealbench.attacks import geometric as geo


@pytest.fixture
def sample_tensors():
    torch.manual_seed(0)
    images = torch.rand(2, 3, 32, 32)
    mask = torch.rand_like(images)
    return images, mask


@torch.no_grad()
def test_identity_passthrough(sample_tensors):
    images, mask = sample_tensors
    transform = geo.Identity()

    img_out, mask_out = transform(images.clone(), mask.clone())

    assert torch.equal(img_out, images)
    assert torch.equal(mask_out, mask)


@pytest.mark.parametrize("angle", [15, -45])
@torch.no_grad()
def test_rotate_matches_functional(sample_tensors, angle):
    images, mask = sample_tensors
    transform = geo.Rotate()

    img_out, mask_out = transform(images.clone(), mask.clone(), angle=angle)

    expected_img = F.rotate(images, angle)
    expected_mask = F.rotate(mask, angle)

    assert torch.allclose(img_out, expected_img)
    assert torch.allclose(mask_out, expected_mask)


@pytest.mark.parametrize("scale", [0.5, 1.0])
@torch.no_grad()
def test_resize_matches_functional(sample_tensors, scale):
    images, mask = sample_tensors
    transform = geo.Resize()

    img_out, mask_out = transform(images.clone(), mask.clone(), size=scale)

    expected_size = (
        int(scale * images.shape[-2]),
        int(scale * images.shape[-1]),
    )
    expected_img = F.resize(images, expected_size, antialias=True)
    expected_mask = F.resize(mask, expected_size, antialias=True)

    assert img_out.shape[-2:] == expected_size
    assert mask_out.shape[-2:] == expected_size
    assert torch.allclose(img_out, expected_img)
    assert torch.allclose(mask_out, expected_mask)


@pytest.mark.parametrize("scale", [0.75])
@torch.no_grad()
def test_crop_matches_random_params(sample_tensors, scale):
    images, mask = sample_tensors
    transform = geo.Crop()

    h, w = images.shape[-2:]
    output_size = (int(scale * h), int(scale * w))

    torch.manual_seed(123)
    i, j, th, tw = transforms.RandomCrop.get_params(images, output_size)
    torch.manual_seed(123)

    img_out, mask_out = transform(images.clone(), mask.clone(), size=scale)

    expected_img = F.crop(images, i, j, th, tw)
    expected_mask = F.crop(mask, i, j, th, tw)

    assert img_out.shape[-2:] == output_size
    assert mask_out.shape[-2:] == output_size
    assert torch.allclose(img_out, expected_img)
    assert torch.allclose(mask_out, expected_mask)


@pytest.mark.parametrize("distortion", [0.2])
@torch.no_grad()
def test_perspective_matches_params(sample_tensors, distortion):
    images, mask = sample_tensors
    transform = geo.Perspective()

    h, w = images.shape[-2:]
    torch.manual_seed(99)
    startpoints, endpoints = transform.get_perspective_params(w, h, distortion)
    torch.manual_seed(99)

    img_out, mask_out = transform(images.clone(), mask.clone(), distortion_scale=distortion)

    expected_img = F.perspective(images, startpoints, endpoints)
    expected_mask = F.perspective(mask, startpoints, endpoints)

    assert img_out.shape == images.shape
    assert mask_out.shape == mask.shape
    assert torch.allclose(img_out, expected_img)
    assert torch.allclose(mask_out, expected_mask)


@torch.no_grad()
def test_horizontal_flip_matches_functional(sample_tensors):
    images, mask = sample_tensors
    transform = geo.HorizontalFlip()

    img_out, mask_out = transform(images.clone(), mask.clone())

    expected_img = torch.flip(images, dims=[-1])
    expected_mask = torch.flip(mask, dims=[-1])

    assert torch.allclose(img_out, expected_img)
    assert torch.allclose(mask_out, expected_mask)


# Ensure the transforms operate when mask is None
@pytest.mark.parametrize(
    "transform, kwargs",
    [
        (geo.Rotate(), {"angle": 10}),
        (geo.Resize(), {"size": 0.75}),
        (geo.Crop(), {"size": 0.8}),
        (geo.Perspective(), {"distortion_scale": 0.3}),
        (geo.HorizontalFlip(), {}),
    ],
)
@torch.no_grad()
def test_transforms_without_mask(transform, kwargs, sample_tensors):
    images, _ = sample_tensors

    # Synchronize RNG for stochastic ops
    torch.manual_seed(7)
    img_out, mask_out = transform(images.clone(), None, **kwargs)

    assert mask_out is None
    assert img_out.ndim == images.ndim
    assert torch.isfinite(img_out).all()
