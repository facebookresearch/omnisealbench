# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from .modules.Conv1x1 import InvertibleConv1x1
from .modules.InvArch import InvArch


class IM(nn.Module):
    def __init__(self, opt):
        super(IM, self).__init__()
        #
        self.operations = nn.ModuleList()
        #
        for _ in range(opt['network']['InvBlock']['block_num']):
            #
            if opt['network']['InvBlock']['downscaling']['use_conv1x1']:
                a = InvertibleConv1x1(opt['network']['InvBlock']['split1_img'] + \
                    opt['network']['InvBlock']['split2_repeat'])
                self.operations.append(a)
            #
            b = InvArch(opt['network']['InvBlock']['split1_img'], opt['network']['InvBlock']['split2_repeat'])
            self.operations.append(b)

    def forward(
        self, 
        x :torch.Tensor, 
        rev:bool=False
    ) -> torch.Tensor:
        if not rev:
            #
            for op in self.operations:
                x = op.forward(x, rev)
            #
        else:
            for op in self.operations[::-1]:
                x = op.forward(x, rev)
        return x