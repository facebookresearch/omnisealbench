# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from .DEM import DEM
from .FSM import FSM
from .IM import IM
from .modules import InvDownscaling
from .NIAM import NIAM
from .NSM_module import NSM


# ---------------------------------------------------------------------------------
class CINEncoder(nn.Module):
    def __init__(self, opt):
        super(CINEncoder, self).__init__()
        # 
        self.opt = opt
        self.h, self.w = opt['network']['H'], opt['network']['W']
        self.msg_length = opt['network']['message_length']
        #
        self.invertible_model = IM(opt)
        self.cs_model = FSM(opt)
        if opt['network']['InvBlock']['downscaling']['use_down']:
            self.invDown = InvDownscaling(opt)
        self.fusion_model= DEM(opt, self.invDown)

    def forward(self, 
        image: torch.Tensor, 
        msg: torch.Tensor
    ) -> torch.Tensor:
        cover_down = self.invDown(image)                #[64]
        image, msg_down = self.fusion_model(cover_down, msg)          #[64]
        inv_encoded = self.invertible_model(torch.cat([image, msg_down], dim=1))    #[64]
        cs = self.cs_model(inv_encoded, cover_down)
        watermarking_img = self.invDown(cs, rev=True).clamp(-1, 1)   #[128]

        return watermarking_img


# ---------------------------------------------------------------------------------
class CINDecoder(nn.Module):
    def __init__(self, opt):
        super(CINDecoder, self).__init__()
        # 
        self.opt = opt
        self.h, self.w = opt['network']['H'], opt['network']['W']
        self.msg_length = opt['network']['message_length']
        #
        self.invertible_model = IM(opt)
        self.cs_model = FSM(opt)
        if opt['network']['InvBlock']['downscaling']['use_down']:
            self.invDown = InvDownscaling(opt)
        self.fusion_model= DEM(opt, self.invDown)

    def forward(self, imgs_w: torch.Tensor) -> torch.Tensor:
        # decoder1                                           
        down = self.invDown(imgs_w)            #[64]
        # Create empty tensor on the same device as input
        empty_tensor = torch.tensor([]).to(imgs_w.device)
        cs_rev = self.cs_model(down, empty_tensor, rev=True)
        inv_back = self.invertible_model(cs_rev, rev=True)   #[64]
        img_fake, msg_fake_1 = self.fusion_model(inv_back, empty_tensor, rev=True)   #[64]
        msg_nsm = msg_fake_1
        return msg_nsm


# ---------------------------------------------------------------------------------
class CINBasic(nn.Module):
    def __init__(self, opt):
        super(CINBasic, self).__init__()
        # 
        self.opt = opt
        self.h, self.w = opt['network']['H'], opt['network']['W']
        self.msg_length = opt['network']['message_length']
        #
        self.invertible_model = IM(opt)
        self.cs_model = FSM(opt)
        if opt['network']['InvBlock']['downscaling']['use_down']:
            self.invDown = InvDownscaling(opt)
        self.fusion_model= DEM(opt, self.invDown)
        self.decoder2 = NIAM(self.h, self.w, self.msg_length)
        self.nsm_model = NSM(opt)

    def encoder(self, image, msg):
        # down                                             #[128]
        cover_down = self.invDown(image)                #[64]
        # fusion
        # fusion = self.fusion_model(cover_down, msg)          #[64]
        image, msg_down = self.fusion_model(cover_down, msg)          #[64]
        # inv_forward
        fusion = torch.cat([image, msg_down], dim=1)
        inv_encoded = self.invertible_model(fusion)    #[64]
        # cs
        cs = self.cs_model(inv_encoded, cover_down)
        # up to out
        watermarking_img = self.invDown(cs, rev=True).clamp(-1, 1)   #[128]

        return watermarking_img

    def test_decoder(self, noised_img):
        # decoder1                                           
        down = self.invDown(noised_img)            #[64]
        cover_down = None
        cs_rev = self.cs_model(down, cover_down, rev=True)
        inv_back = self.invertible_model(cs_rev, rev=True)   #[64]
        img_fake, msg_fake_1 = self.fusion_model(inv_back, None, rev=True)   #[64]        
    #
        msg_nsm = msg_fake_1

        return msg_nsm        


class CIN(CINBasic):
    def __init__(self, opt):
        super(CIN, self).__init__(opt)

    def forward(self, image, msg, noise_choice):
        watermarking_img = self.encoder(image, msg)
        noised_img = self.noise_pool(watermarking_img.clone(), image, noise_choice)
        pre_noise = self.nsm(noised_img)
        img_fake, msg_fake_1, msg_fake_2, msg_nsm = self.test_decoder(noised_img, pre_noise)

        return watermarking_img, noised_img, img_fake, msg_fake_1, msg_fake_2, msg_nsm 
