# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from trustmark import TrustMark

from .base import Watermark


# Mapping function
def get_encoding_object(encoding_str):
    encoding_map = {
        "BCH_SUPER": TrustMark.Encoding.BCH_SUPER,
        "BCH_3": TrustMark.Encoding.BCH_3,
        "BCH_4": TrustMark.Encoding.BCH_4,
        "BCH_5": TrustMark.Encoding.BCH_5,
    }
    return encoding_map.get(encoding_str)


def custom_decode(trustmark_model, in_stego_image, MODE="text"):
    # Change the model.decode implementation and return secret_pred
    # Inputs
    # stego_image: PIL image
    # Outputs: secret numpy array (1, secret_len)
    stego_image = trustmark_model.get_the_image_for_processing(in_stego_image)
    if min(stego_image.size) > 256:
        stego_image = stego_image.resize((256, 256), Image.BILINEAR)

    stego = (
        transforms.ToTensor()(stego_image)
        .unsqueeze(0)
        .to(trustmark_model.decoder.device)
        * 2.0
        - 1.0
    )  # (1,3,256,256) in range [-1, 1]

    with torch.no_grad():
        secret_binaryarray = (
            (trustmark_model.decoder.decoder(stego) > 0).cpu().numpy()
        )  # (1, secret_len)
    if trustmark_model.use_ECC:
        secret_pred, detected, version = trustmark_model.ecc.decode_bitstream(
            secret_binaryarray, MODE
        )[0]
        if not detected:
            # last ditch attempt to recover a possible corruption of the version bits by trying all other schema types
            modeset = [x for x in range(0, 3) if x not in [version]]  # not bch_3
            for m in modeset:
                if m == 0:
                    secret_binaryarray[0][-2] = False
                    secret_binaryarray[0][-1] = False
                if m == 1:
                    secret_binaryarray[0][-2] = False
                    secret_binaryarray[0][-1] = True
                if m == 2:
                    secret_binaryarray[0][-2] = True
                    secret_binaryarray[0][-1] = False
                if m == 3:
                    secret_binaryarray[0][-2] = True
                    secret_binaryarray[0][-1] = True
                secret_pred, detected, version = trustmark_model.ecc.decode_bitstream(
                    secret_binaryarray, MODE
                )[0]
                if detected:
                    return secret_pred, detected, version
                else:
                    return secret_pred, False, -1
                    # return "", False, -1
        else:
            return secret_pred, detected, version
    else:
        assert len(secret_binaryarray.shape) == 2
        secret_pred = "".join(str(int(x)) for x in secret_binaryarray[0])
        return secret_pred, True, -1


class TrustMarkWM(Watermark):
    def __init__(
        self, device: str, model_type="Q", encoding_type="BCH_SUPER", **kwargs
    ):
        encoding_type = get_encoding_object(encoding_type)

        super().__init__(device)
        self.model = TrustMark(
            verbose=True,
            model_type=model_type,
            encoding_type=encoding_type,
        )

    @torch.no_grad()
    def encode(self, img: Image, msg: np.ndarray) -> Image:
        msg_str = ""
        for m in msg:
            msg_str += str(m)
        # msg = torch.tensor(msg, dtype=torch.float, device=device).unsqueeze(0) # 1 k
        return self.model.encode(img, msg_str, MODE="binary")

    @torch.no_grad()
    def decode(self, img: Image) -> np.ndarray:
        # wm_secret, wm_present, wm_schema = self.model.decode(img, MODE="binary")
        wm_secret, wm_present, wm_schema = custom_decode(self.model, img, MODE="binary")
        wm_secret = np.array(list(wm_secret)).astype(int)
        wm_secret = torch.tensor(
            wm_secret, dtype=torch.float, device=self.device
        ).unsqueeze(
            0
        )  # 1 k
        return (wm_secret > 0).squeeze().cpu().numpy()[:32], wm_present

    def get_detection_stats(
        self, detected_batch: torch.Tensor, detection_keys: torch.Tensor
    ) -> List[Dict[str, Any]]:
        all_stats = []
        for i in range(len(detected_batch)):
            all_stats.append({"det": detected_batch[i]})

        return all_stats
