# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import numpy as np
import math
import torch
from PIL.Image import Image
from scipy.stats import binomtest


class Watermark:
    def __init__(self, device: str, **kwargs):
        self.device = device

    def encode(self, img: Image, msg: np.ndarray) -> Image:
        """
        Encode a message into an image
        Args:
            img: PIL image
            msg: message to encode, np array of 0 or 1 (ex: 110000100)
        Returns:
            img_w: PIL image watermarked
        """
        pass

    def decode(self, img: Image) -> np.ndarray:
        """
        Decode the message from an image
        Args:
            img: PIL image
        Returns:
            msg: message to encode, np array of 0 or 1 (ex: 110000100)
        """
        pass

    def filter_batch(self, imgs: List[Image]) -> List[int]:
        return list(range(len(imgs)))

    def encode_batch(self, imgs: List[Image], msgs: List[np.ndarray]) -> List[Image]:
        """
        Encode messages into their respective images
        Args:
            imgs: a list of PIL images
            msgs: a list of messages to encode, np array of 0 or 1 (ex: 110000100)
        Returns:
            imgs_w: a list of PIL image watermarked
        """
        imgs_w = []
        for img, msg in zip(imgs, msgs):
            imgs_w.append(self.encode(img, msg))

        return imgs_w

    def decode_batch(
        self, imgs: List[Image]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Decode the messages from each image
        Args:
            imgs: a list of PIL images
        Returns:
            msgs: a list of messages encoded in the images, np array of 0 or 1 (ex: 110000100)
        """
        all_decoded = []
        all_detected = []

        for img in imgs:
            decoded, detected = self.decode(img)
            all_decoded.append(decoded)
            all_detected.append(detected)

        return all_decoded, all_detected

    def get_detection_stats(
        self, detected_batch: torch.Tensor, detection_keys: torch.Tensor
    ) -> List[Dict[str, Any]]:
        detected = (
            2
            * torch.tensor(
                np.array(detected_batch),
                dtype=torch.float,
                device=detection_keys.device,
            )
            - 1
        )  # b k

        diff = ~torch.logical_xor(detected > 0, detection_keys > 0)  # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # b k -> b
        detect = bit_accs > 0.95  # b

        all_stats = []
        for i in range(len(bit_accs)):
            all_stats.append(
                {
                    "det_score": bit_accs[i].item(),
                    "det": detect[i].item(),
                }
            )

        return all_stats

    def get_decoding_stats(
        self, decoded_batch: torch.Tensor, keys: torch.Tensor
    ) -> List[Dict[str, Any]]:
        decoded = (
            2 * torch.tensor(decoded_batch, dtype=torch.float, device=keys.device) - 1
        )  # b k

        # Bit accuracy: number of matching bits between the key and the message, divided by the total number of bits.
        # p-value: probability of observing a bit accuracy as high as the one observed, assuming the null hypothesis that the image is genuine.

        diff = ~torch.logical_xor(decoded > 0, keys > 0)  # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # b k -> b
        word_accs = bit_accs == 1  # b

        successes_k = diff.sum(axis=-1)

        all_stats = []
        for i in range(len(word_accs)):
            # k = len(diff[i]) - sum(diff[i])
            pval = binomtest(successes_k[i], diff.shape[1], 0.5, alternative="greater")

            all_stats.append(
                {
                    "bit_acc": bit_accs[i].item(),
                    "word_acc": word_accs[i].item(),
                    "p_value": pval.pvalue,
                    "log10_p_value": math.log10(pval.pvalue),
                }
            )

        return all_stats
