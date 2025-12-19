# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


#
class NSM(nn.Module):
    def __init__(self, opt):
        super(NSM, self).__init__()
        # 
        self.opt = opt
        #
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)
        self.dct = DctBasic()

    def forward(self, noised_img):
        #
        with torch.no_grad():
            dct_blocks = self.dct(noised_img)
            out = self.resnet(dct_blocks)

        return out


class DctBasic(nn.Module):
    def __init__(self):
        super(DctBasic, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Step 1: Clamp and convert from [-1,1] to [0,255]
        image = (image.clamp(-1, 1) + 1) * 255 / 2

        # Step 2: Pad the image so that we can do DCT on 8x8 blocks
        pad_height = (8 - image.shape[2] % 8) % 8
        pad_width = (8 - image.shape[3] % 8) % 8
        image = nn.functional.pad(image, (0, pad_width, 0, pad_height), mode='constant', value=0.0)

        # Step 3: Convert RGB to YUV
        image_yuv = torch.empty_like(image)
        image_yuv[:, 0:1, :, :] = 0.299 * image[:, 0:1, :, :] + 0.587 * image[:, 1:2, :, :] + 0.114 * image[:, 2:3, :, :]
        image_yuv[:, 1:2, :, :] = -0.1687 * image[:, 0:1, :, :] - 0.3313 * image[:, 1:2, :, :] + 0.5 * image[:, 2:3, :, :]
        image_yuv[:, 2:3, :, :] = 0.5 * image[:, 0:1, :, :] - 0.4187 * image[:, 1:2, :, :] - 0.0813 * image[:, 2:3, :, :]

        # Step 4: Apply DCT
        coff = torch.zeros((8, 8), dtype=torch.float).to(image.device)
        coff[0, :] = 1 * math.sqrt(1 / 8)
        for i in range(1, 8):
            for j in range(8):
                coff[i, j] = math.cos(math.pi * i * (2 * j + 1) / (2 * 8)) * math.sqrt(2 / 8)

        split_num = image_yuv.shape[2] // 8
        image_dct = torch.cat(torch.cat(image_yuv.split(8, 2), 0).split(8, 3), 0)
        image_dct = torch.matmul(coff, image_dct)
        image_dct = torch.matmul(image_dct, coff.permute(1, 0))
        image_dct = torch.cat(torch.cat(image_dct.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

        # Step 5: Normalize and clamp the output
        out = nn.functional.normalize(image_dct)
        return out.clamp(-1, 1)
