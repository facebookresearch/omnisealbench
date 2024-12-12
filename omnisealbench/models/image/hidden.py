# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .base import Watermark


class Hidden(Watermark):
    def __init__(
        self,
        device: str,
        encoder_path: str,
        decoder_path: str,
        lambda_w=1,
        **kwargs,
    ):
        super().__init__(device)

        self.encoder = torch.jit.load(encoder_path).eval().to(device)
        self.decoder = torch.jit.load(decoder_path).eval().to(device)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.unnormalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        self.lambda_w = lambda_w

    @torch.no_grad()
    def encode(self, img: Image, msg: np.ndarray) -> Image:
        msg = torch.tensor(msg, dtype=torch.float, device=self.device).unsqueeze(
            0
        )  # 1 k
        msg = 2 * msg - 1  # k, hidden uses -1 and 1
        img = self.normalize(
            transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        )  # 1 3 h w
        img_w_aux = self.encoder(img, msg)  # 1 3 h w
        delta_w = img_w_aux - img
        img_w = img + self.lambda_w * delta_w
        clip_img = torch.clamp(self.unnormalize(img_w), 0, 1)
        clip_img = torch.round(255 * clip_img) / 255
        return transforms.ToPILImage()(clip_img.squeeze(0).cpu())

    @torch.no_grad()
    def decode(self, img: Image) -> np.ndarray:
        img = self.normalize(
            transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        )  # 1 3 h w
        msg = self.decoder(img)  # 1 k
        aux = (msg > 0).squeeze().cpu().numpy()
        return aux[16:], aux[:16]
