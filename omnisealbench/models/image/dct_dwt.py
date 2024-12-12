# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import numpy as np
import PIL
from PIL.Image import Image

from .base import Watermark


class DctDwt(Watermark):
    def __init__(self, nbits, **kwargs):
        super().__init__(device="cpu")

        import imwatermark

        self.encoder = imwatermark.WatermarkEncoder()
        self.decoder = imwatermark.WatermarkDecoder(wm_type="bits", length=nbits)

    def encode(self, img: Image, msg: np.ndarray) -> Image:
        img = np.array(img)[..., ::-1]  # converts to cv2 for imwatermark
        self.encoder.set_watermark(wmType="bits", content=msg)
        encoded = self.encoder.encode(img, "dwtDct")
        return PIL.Image.fromarray(encoded[..., ::-1])  # converts back to PIL

    def decode(self, img: Image) -> np.ndarray:
        img = np.array(img)[..., ::-1]  # converts to cv2 for imwatermark
        msg = self.decoder.decode(img)
        return msg[16:], msg[:16]

    def filter_batch(self, imgs: List[Image]) -> Tuple[List[int], List[Image]]:
        filtered_indices = []

        for i, img in enumerate(imgs):
            r, c = img.size
            if r * c < 256 * 256:
                continue  # 'image too small, should be larger than 256x256'

            filtered_indices.append(i)
        return filtered_indices
