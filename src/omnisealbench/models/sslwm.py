# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from scipy.stats import ortho_group  # type: ignore[import-untyped]

from omnisealbench.interface import Device
from omnisealbench.models.fnns import FNNS
from omnisealbench.utils.common import get_device, set_seed
from omnisealbench.utils.detection import get_detection_and_decoded_keys


class SSLWaterMark(FNNS):
    """
    Wrapper for SSL Watermarking ("Watermarking images in
    self-supervised latent spaces", https://arxiv.org/pdf/2411.07231v1)
    """

    model: torch.nn.Module

    def __init__(
        self,
        model: torch.nn.Module,
        nbits: int,
        lambda_w: float,
        nsteps: int = 10,
        data_aug: str = "none",
        detection_bits: int = 16,
        seed: Optional[int] = None,
    ):
        super().__init__(
            model=model,
            lambda_w=lambda_w,
            nsteps=nsteps,
            data_aug=data_aug,
            detection_bits=detection_bits,
        )
        self.seed = seed
        if seed is not None:
            set_seed(seed)
        device = get_device(self)
        D = self.model(torch.randn(1, 3, 224, 224, device=device)).shape[-1]
        self.carriers = torch.tensor(ortho_group.rvs(D)[:nbits, :], device=device, dtype=torch.float)

    def loss_w(self, ft, msg, T=1e1):
        dot_products = ft @ self.carriers.T  # d, d k -> 1 k
        return F.binary_cross_entropy_with_logits(dot_products / T, msg)

    def detect_watermark(
        self,
        contents: List[torch.Tensor],
        detection_threshold: float = 0.95,  # not used
        message_threshold: float = 0.0,
        detection_bits: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        output: List[torch.Tensor] = []
        for img in contents:
            if img.ndim == 3:
                img = img.unsqueeze(0)
            img = self.normalize(img)
            extracted = self.model(img)
            extracted = extracted @ self.carriers.T  # d, d k -> k
            output.append(extracted.squeeze())  # b k

        msgs = torch.stack(output, dim=0)  # b k

        return get_detection_and_decoded_keys(
            msgs,
            detection_bits=detection_bits if detection_bits is not None else self.detection_bits,
            message_threshold=message_threshold,
        )


def build_model(
    model_card_or_path: str,
    nbits: int = 48,
    lambda_w: float = 1.0,
    nsteps: int = 10,
    data_aug: str = "none",
    detection_bits: int = 16,
    seed: Optional[int] = None,
    device: Device = "cpu",
) -> SSLWaterMark:
    """
    Build a SSL Watermark model.
    ("Watermarking images in self-supervised latent spaces", https://arxiv.org/pdf/2411.07231v1)
    Args:
        model_card_or_path: The path to the model card or the model itself.
        nbits: Number of bits for the watermark.
        lambda_w: Weighting factor for the watermark loss.
        nsteps: Number of optimization steps.
        data_aug: Data augmentation strategy.
        detection_bits: Number of bits used for detection.
        seed: Random seed for reproducibility.
        device: Device to run the model on.
    Returns:
        SSLWaterMark: An instance of the SSLWaterMark class.
    """
    model = torch.jit.load(model_card_or_path, map_location="cpu")
    model = model.to(device)
    model.eval()
    return SSLWaterMark(
        model=model,
        nbits=nbits,
        lambda_w=lambda_w,
        nsteps=nsteps,
        data_aug=data_aug,
        detection_bits=detection_bits,
        seed=seed,
    )
