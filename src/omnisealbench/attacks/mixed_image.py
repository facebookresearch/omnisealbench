# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, List, Optional

import torch
import torch.nn as nn

from .geometric import Crop
from .valuemetric import Brightness, JPEG


class Sequential(nn.Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.transforms = args

    def forward(self, image, mask, args):
        """
        Apply a sequence of transformations to the image and mask.
        Args:
            image: The input image
            mask: The input mask
            *args: The arguments for each transformation, in order
        Returns:
            The transformed image and the transformed mask
        """
        # fill *args with None if not enough arguments are provided
        args = args + (None,) * (len(self.transforms) - len(args))
        # apply each transform
        for transform, aug_arg in zip(self.transforms, args):
            image, mask = transform(image, mask, aug_arg)
        return image, mask

    def __repr__(self):
        return f"{self.transforms}".replace(", ", "_")


class JPEG_Crop_Brightness(nn.Module):
    def __init__(self):
        super(JPEG_Crop_Brightness, self).__init__()

        self.op = Sequential(Crop(), Brightness(), JPEG())

    def forward(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None, xargs: List[Any] = None) -> torch.Tensor:
        return self.op.forward(image, mask, tuple(xargs))

    def __repr__(self):
        return "JPEG_Crop_Brightness"
