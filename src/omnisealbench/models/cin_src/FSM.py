# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


#
class FSM(nn.Module):
    def __init__(self, opt):
        super(FSM, self).__init__()
        #
        self.split1_img = opt['network']['InvBlock']['split1_img']
        self.strength_factor = opt['noise']['StrengthFactor']['S']

    def forward(self, 
            encoded_img: torch.Tensor, 
            cover_down: torch.Tensor,
            rev:bool=False
        ):
        #
        if not rev:
            # 
            msg = encoded_img[:, self.split1_img:, :, :]
            out = cover_down + self.strength_factor * msg
            return out
        else:            
            # msg copy
            out = torch.cat((encoded_img, encoded_img), dim=1)
            return out
