# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# mypy: disable-error-code="import-untyped, assignment"

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from omnisealbench.utils.common import set_seed
from torchvision import transforms  # type: ignore[import-untyped]

from omnisealbench.interface import Device
from omnisealbench.utils.detection import get_detection_and_decoded_keys
from omnisealbench.utils.jnd import JND


class FNNS:
    """
    Wrapper for FNNS - "Fixed neural network steganography: Train the images, not the network"
    https://openreview.net/pdf?id=hcMvApxGSzZ
    """

    model: torch.nn.Module

    def __init__(
        self,
        model: torch.nn.Module,
        lambda_w: float,
        nbits: int = 48,
        nsteps: int = 10,
        data_aug: str = "none",
        detection_bits: int = 16,
        seed: Optional[int] = None,
    ):
        """
        Args:
            model (torch.nn.Module): underlying model for watermark generation
            lambda_w (float): weight for distorion of the image
            nsteps (int, optional): in-inference training of the distortion. Defaults to 10.
            data_aug (str, optional): extra data augmentation. Defaults to "none".
            detection_bits (int, optional): leading bits in the message used to compute the
                detection score. Defaults to 16.
            seed: Optional[int]: random seed for reproducibility in the generation process
        """
        self.model = model
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.unnormalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        self.attenuation = JND(preprocess=self.unnormalize)
        self.alpha = lambda_w
        self.nsteps = nsteps
        if data_aug != "none":
            self.nsteps *= 10
        self.detection_bits = detection_bits
        self.nbits = nbits
        self.seed = seed

    def loss_w(self, ft, msg, T=1e1):
        return F.binary_cross_entropy_with_logits(ft / T, msg)

    def generate_watermark(
        self, contents: List[torch.Tensor], message: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Generates a watermark for the given audio contents.
        Args:
            contents (torch.Tensor): The input audio data to be watermarked.
            message (Optional[torch.Tensor]): Optional tensor containing the message to embed in the watermark.
        Returns:
            torch.Tensor: The watermarked audio tensor.
        """
        device = contents[0].device
        imgs_w: List[torch.Tensor] = []

        if self.seed is not None:
            set_seed(self.seed)

        for img in contents:
            if img.ndim == 3:
                img = img.unsqueeze(0)
            img = self.normalize(img)  # b c h w
            attenuation = self.attenuation.to(device)
            heatmap = self.alpha * attenuation.heatmaps(img)  # b 1 h w
            distortion_scaler = 1.0
            distortion = 1e-3 * torch.randn_like(img, device=device)  # b c h w
            distortion.requires_grad = True
            optimizer = torch.optim.Adam([distortion], lr=0.5e-0)
            for _ in range(self.nsteps):
                clip_img = img + heatmap * torch.tanh(distortion) * distortion_scaler
                ft = self.model(clip_img)  # b d
                loss_w = self.loss_w(ft[0], message)
                optimizer.zero_grad()
                loss_w.backward()
                optimizer.step()

            img_w = img + heatmap * torch.tanh(distortion) * distortion_scaler
            img_w = torch.clamp(self.unnormalize(img_w), 0, 1)
            img_w = torch.round(255 * img_w) / 255
            imgs_w.append(img_w.squeeze(0))  # c h w

        return imgs_w

    @torch.no_grad()
    def detect_watermark(
        self,
        contents: List[torch.Tensor],
        detection_threshold: float = 0.95,  # not used
        message_threshold: float = 0.0,
        detection_bits: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        extracted: List[torch.Tensor] = []
        for img in contents:
            if img.ndim == 3:
                img = img.unsqueeze(0)
            img = self.normalize(img)
            extracted.append(self.model(img).squeeze())  # b d

        imgs_w = torch.stack(extracted, dim=0)  # b d

        return get_detection_and_decoded_keys(
            imgs_w,
            detection_bits=detection_bits if detection_bits is not None else self.detection_bits,
            message_threshold=message_threshold,
        )


def build_model(
    model_card_or_path: str,
    nbits: int = 48,
    lambda_w: float = 1.3,
    nsteps: int = 10,
    data_aug: str = "none",
    detection_bits: int = 16,
    seed: Optional[int] = None,
    device: Device = "cpu",
) -> FNNS:
    """
    Build a FNNS model - "Fixed neural network steganography: Train the images, not the network"
    https://openreview.net/pdf?id=hcMvApxGSzZ
    Args:
        model_card_or_path (str): The path to the model card or the model itself.
        lambda_w (float): The scaling factor for the watermark.
        nsteps (int): Number of steps for in-inference training of the distortion.
        data_aug (str): Extra data augmentation.
        detection_bits (int): Leading bits in the message used to compute the detection score.
        device (str): Device to load the model on. Default is "cpu".
    Returns:
        FNNS: An instance of the FNNS model.
    """
    # Load the model (this is a placeholder, actual loading logic will depend on the model format)
    model = torch.jit.load(model_card_or_path, map_location=torch.device(device))
    model.eval()
    return FNNS(
        model=model,
        nbits=nbits,
        lambda_w=lambda_w,
        nsteps=nsteps,
        data_aug=data_aug,
        detection_bits=detection_bits,
        seed=42,
    )
