# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
from torchvision import transforms  # type: ignore[import-untyped]

from omnisealbench.interface import Device
from omnisealbench.utils.detection import get_detection_and_decoded_keys


class HiDDeNWatermarkGenerator:
    """
    Implemetentation of the HiDDeN generator ("Hidden: Hiding data with deep networks")
    https://arxiv.org/abs/1807.09937
    """

    model: torch.nn.Module
    lambda_w: float = 1.0

    def __init__(self, model, lambda_w: float = 1.0, nbits: int = 48):
        """
        Initializes the HiDDeN watermark generator with a given model.
        Args:
            model: The HiDDeN model to use for watermark generation.
        """
        self.model = model
        self.lambda_w = lambda_w
        self.nbits = nbits

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.unnormalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )

    @torch.no_grad()
    def _encode(
        self,
        contents: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        img = self.normalize(contents)  # 1 3 h w
        img_w_aux = self.model(img, message)  # 1 3 h w
        delta_w = img_w_aux - img
        img_w = img + self.lambda_w * delta_w
        clip_img = torch.clamp(self.unnormalize(img_w), 0, 1)
        clip_img = torch.round(255 * clip_img) / 255
        return clip_img

    def generate_watermark(
        self,
        contents: List[torch.Tensor],
        message: torch.Tensor,
    ) -> List[torch.Tensor]:
        if message.ndim == 1:
            message = message.unsqueeze(0)
        message = 2 * message - 1

        return [
            self._encode(img.unsqueeze(0), message).squeeze(0)
            for img in contents
        ]


class HiDDeNWatermarkDetector:
    """
    HiDDeN Watermark Detector is responsible for detecting watermarks in images.
    It uses a HiDDeN model to extract the watermark from images.
    """

    model: torch.nn.Module

    def __init__(self, model: torch.nn.Module, nbits: int = 48, detection_bits: int = 16):
        self.model = model
        self.nbits = nbits

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Since HiDDeN does not predict the watermarking separately, but only extracts
        # the secret message, we reserve n leading bits in the results to indicate the
        # presence of a watermark. The rest of the bits are used for the secret message.
        self.detection_bits = detection_bits

    @torch.no_grad()
    def detect_watermark(
        self,
        contents: List[torch.Tensor],
        detection_threshold: float = 0.95,  # not used
        message_threshold: float = 0.0,
        detection_bits: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        extracted = []
        for img in contents:
            if img.ndim == 3:
                img = img.unsqueeze(0)
            img = self.normalize(img)  # 1 3 h w
            extracted_ = self.model(img).squeeze()  # 1 k -> k
            extracted.append(extracted_)

        extracted = torch.stack(extracted, dim=0)  # b k
        return get_detection_and_decoded_keys(
            extracted,
            detection_bits=detection_bits if detection_bits is not None else self.detection_bits,
            message_threshold=message_threshold,
        )


def build_generator(
    model_card_or_path: str,
    lambda_w: float = 1.0,
    nbits: int = 48,
    device: Device = "cpu",
) -> HiDDeNWatermarkGenerator:
    """
    Build a HiDDeN watermark generator from a model card or path.
    ("Hidden: Hiding data with deep networks")
    https://arxiv.org/abs/1807.09937
    Args:
        model_card_or_path: The path to the model card or the model itself.
        lambda_w: The scaling factor for the watermark.
        nbits: The number of bits in the watermark message. Default is 48.
        device: Device to load the model on. Default is "cpu".
    Returns:
        HiDDeNWatermarkGenerator: An instance of the HiDDeN watermark generator.
    """
    # Load the model (this is a placeholder, actual loading logic will depend on the model format)
    model = torch.jit.load(model_card_or_path, map_location=torch.device(device))
    model.eval()
    return HiDDeNWatermarkGenerator(model, nbits=nbits, lambda_w=lambda_w)


def build_detector(
    model_card_or_path: str,
    nbits: int = 48,
    detection_bits: int = 16,
    device: Device = "cpu",
) -> HiDDeNWatermarkDetector:
    """
    Build a HiDDeN watermark detector from a model card or path.
    ("Hidden: Hiding data with deep networks")
    https://arxiv.org/abs/1807.09937
    Args:
        model_card_or_path: The path to the model card or the model itself.
        detection_bits: The number of bits used for watermark detection.
    Returns:
        HiDDeNWatermarkDetector: An instance of the HiDDeN watermark detector.
    """
    # Load the model (this is a placeholder, actual loading logic will depend on the model format)
    model = torch.jit.load(model_card_or_path, map_location=torch.device(device))
    model.eval()
    return HiDDeNWatermarkDetector(model, nbits=nbits, detection_bits=detection_bits)
