# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from .builders import *


def collate_fn(batch):
    """Collate function for data loader. Allows to have img of different size"""
    return batch


class WatermarkedImageDataset(Dataset):
    def __init__(
        self,
        watermarks_dir: str,
        total: int,
    ):
        self.watermarks_dir = Path(watermarks_dir)
        self.total = total

        self.transform: transforms.Compose = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.untransform: transforms.Compose = transforms.Compose(
            [
                transforms.ToPILImage(),
            ]
        )

    def __len__(self):
        return self.total

    def __getitem__(
        self,
        idx,
    ) -> tuple[int, torch.tensor, torch.tensor, torch.tensor]:
        watermark_path = self.watermarks_dir / f"watermark_{idx}.png"
        image_path = self.watermarks_dir / f"image_{idx}.png"
        message_path = self.watermarks_dir / f"message_{idx}.npy"

        pil_image = default_loader(image_path)
        watermarked_image = default_loader(watermark_path)
        secret_message = np.load(message_path)

        img = self.transform(pil_image)
        watermarked_img = self.transform(watermarked_image)

        return (
            idx,
            watermarked_img,
            img,
            secret_message,
        )


def apply_attacks(
    img: torch.Tensor,
    watermarked_img: torch.Tensor,
    attack_implementation: Callable[..., torch.Tensor],
    attack_runtime_config: dict,
    postprocess_fn: Callable[
        [torch.Tensor, torch.Tensor, Callable[..., torch.Tensor], Dict], torch.Tensor
    ],
    device: str,
    attack_orig_img: bool = True,
) -> tuple[torch.tensor, torch.tensor]:

    img = img[None, ...]
    watermarked_img = watermarked_img[None, ...]

    img = img.to(device)
    watermarked_img = watermarked_img.to(device)

    attacked_watermarked_image = postprocess_fn(
        watermarked_img,
        img,
        attack_implementation,
        attack_runtime_config,
    )

    if attack_orig_img:
        attacked_image = postprocess_fn(
            img,
            img,
            attack_implementation,
            attack_runtime_config,
        )
    else:
        attacked_image = img

    attacked_wmd_img = attacked_watermarked_image[0]
    attacked_img = attacked_image[0]

    return attacked_wmd_img, attacked_img


class WatermarkedImageAttacksDataset(Dataset):
    def __init__(
        self,
        watermarks_dir: str,
        total: int,
        attack_implementation: Callable[..., torch.Tensor],
        attack_runtime_config: dict,
        postprocess_fn: Callable[
            [torch.Tensor, torch.Tensor, Callable[..., torch.Tensor], Dict],
            torch.Tensor,
        ],
        device: str,
    ):
        self.device = device
        self.watermarks_dir = Path(watermarks_dir)
        self.total = total

        self.attack_implementation = attack_implementation
        self.attack_runtime_config = attack_runtime_config
        self.postprocess_fn = postprocess_fn
        self.device = device

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.untransform = transforms.Compose(
            [
                transforms.ToPILImage(),
            ]
        )

    def __len__(self):
        return self.total

    def __getitem__(
        self,
        idx,
    ) -> tuple[int, torch.tensor, torch.tensor, torch.tensor]:
        watermark_path = self.watermarks_dir / f"watermark_{idx}.png"
        image_path = self.watermarks_dir / f"image_{idx}.png"
        message_path = self.watermarks_dir / f"message_{idx}.npy"

        pil_image = default_loader(image_path)
        watermarked_image = default_loader(watermark_path)
        secret_message = np.load(message_path)

        img = self.transform(pil_image)
        watermarked_img = self.transform(watermarked_image)

        attacked_wmd_img, attacked_img = apply_attacks(
            img,
            watermarked_img,
            self.attack_implementation,
            self.attack_runtime_config,
            self.postprocess_fn,
            self.device,
        )

        return (
            idx,
            attacked_wmd_img,
            attacked_img,
            secret_message,
        )
