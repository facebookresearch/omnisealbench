# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .base import Watermark
from .jnd import JND
from .utils_img import psnr


class FNNS(Watermark):
    def __init__(
        self,
        nbits,
        lambda_w,
        device: str,
        decoder_path: str,
        nsteps=10,
        data_aug="none",
        log_iterative_optimisation: bool = False,
        **kwargs,
    ):
        super().__init__(device)

        self.model = torch.jit.load(decoder_path).to(self.device)
        print("Model loaded")
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.unnormalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        self.attenuation = JND(preprocess=self.unnormalize).to(device)
        self.data_aug = nn.Identity().to(device)
        self.nsteps = nsteps
        if data_aug != "none":
            self.nsteps *= 10
        self.alpha = float(lambda_w)

        self.log_iterative_optimisation = log_iterative_optimisation

    def loss_w(self, ft, msg, T=1e1):
        return F.binary_cross_entropy_with_logits(ft / T, msg)

    def encode(self, img: Image, msg: np.ndarray) -> Image:
        scaling = self.alpha
        msg = torch.tensor(msg, dtype=torch.float, device=self.device)  # k
        img = self.normalize(
            transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        )  # 1 3 h w
        heatmap = scaling * self.attenuation.heatmaps(img)  # 1 1 h w
        # distortion_scaler = torch.tensor([0.072*(1/0.299), 0.072*(1/0.587), 0.072*(1/0.114)])[:,None,None].to(device)
        distortion_scaler = 1.0
        distortion = 1e-3 * torch.randn_like(img, device=self.device)  # 1 c h w
        distortion.requires_grad = True
        optimizer = torch.optim.Adam([distortion], lr=0.5e-0)
        log_stats = []
        iter_time = time.time()
        for gd_it in range(self.nsteps):
            gd_it_time = time.time()
            clip_img = img + heatmap * torch.tanh(distortion) * distortion_scaler
            ft = self.model(self.data_aug(clip_img))  # 1 d
            loss_w = self.loss_w(ft[0], msg)
            optimizer.zero_grad()
            loss_w.backward()
            optimizer.step()

            if self.log_iterative_optimisation:
                with torch.no_grad():
                    psnrs = psnr(clip_img, img, img_space="img")
                    log_stats.append(
                        {
                            "gd_it": gd_it,
                            "loss_w": loss_w.item(),
                            "psnr": torch.nanmean(psnrs).item(),
                            "gd_it_time": time.time() - gd_it_time,
                            "iter_time": time.time() - iter_time,
                            "max_mem": torch.cuda.max_memory_allocated()
                            / (1024 * 1024),
                            "kw": "optim",
                        }
                    )
                    if (gd_it + 1) % 5 == 0:
                        print(json.dumps(log_stats[-1]))

        clip_img = img + heatmap * torch.tanh(distortion) * distortion_scaler
        clip_img = torch.clamp(self.unnormalize(clip_img), 0, 1)
        clip_img = torch.round(255 * clip_img) / 255
        return transforms.ToPILImage()(clip_img.squeeze(0).cpu())

    @torch.no_grad()
    def decode(self, img: Image) -> np.ndarray:
        img = self.normalize(
            transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        )  # 1 3 h w
        ft = self.model(img)  # 1 d
        return (ft > 0).squeeze().cpu().numpy()[16:], (ft > 0).squeeze().cpu().numpy()[
            :16
        ]
