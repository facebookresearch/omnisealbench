# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchvision import transforms

# pixel_mean: List[float] = [123.675, 116.28, 103.53],
# pixel_std: List[float] = [58.395, 57.12, 57.375],

image_mean = torch.tensor([0.485, 0.456, 0.406])
image_std = torch.tensor([0.229, 0.224, 0.225])

normalize_img = transforms.Normalize(image_mean, image_std)
unnormalize_img = transforms.Normalize(-image_mean / image_std, 1 / image_std)
unstd_img = transforms.Normalize(0, 1 / image_std)
std_img = transforms.Normalize(0, image_std)
imnet_to_lpips = transforms.Compose(
    [
        unnormalize_img,
        # transforms.Lambda(lambda x: x * 2 - 1),
    ]
)

default_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize_img,
    ]
)


def rgb_to_yuv(img):
    M = torch.tensor(
        [
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001],
        ],
        dtype=torch.float32,
    ).to(img.device)
    img = img.permute(0, 2, 3, 1)  # b h w c
    yuv = torch.matmul(img, M)
    yuv = yuv.permute(0, 3, 1, 2)
    return yuv


def yuv_to_rgb(img):
    M = torch.tensor(
        [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]],
        dtype=torch.float32,
    ).to(img.device)
    img = img.permute(0, 2, 3, 1)  # b h w c
    rgb = torch.matmul(img, M)
    rgb = rgb.permute(0, 3, 1, 2)
    return rgb
