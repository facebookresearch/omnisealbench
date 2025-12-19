# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
from torchvision import transforms  # type: ignore[import-untyped]

from omnisealbench.interface import Device
from omnisealbench.utils.detection import get_detection_and_decoded_keys


class CINWatermarkGenerator:
    """
    Wrapper for the CIN watermark generator model
    ("Towards Blind Watermarking: Combining Invertible and Non-invertible Mechanisms").
    https://arxiv.org/abs/2212.12678
    """

    def __init__(self, model: torch.nn.Module, scaler: float, nbits: int = 30):
        self.model = model
        self.scaler = scaler
        self.nbits = nbits

        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.resize = transforms.Resize((128, 128))
        self.postprocess = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])

    @torch.inference_mode()
    def generate_watermark(
        self, contents: List[torch.Tensor], message: torch.Tensor
    ) -> List[torch.Tensor]:
        if message.ndim < 2:  # Add batch dimension if missing
            message = message.repeat(len(contents), 1)
        message = message[:, : self.nbits].to(torch.float32)

        sizes: List[Tuple[int, int]] = []
        imgs = []
        normalized_imgs = []

        for img in contents:
            img = self.preprocess(img)
            imgs.append(img)  # Collect original images

            _, H, W = img.shape
            sizes.append((H, W))
            img_resize = self.resize(img)  # 3 x 128 x 128
            normalized_imgs.append(img_resize)

        normalized_contents = torch.stack(normalized_imgs, dim=0)  # N x 3 x 128 x 128
        delta_w = self.model(normalized_contents, message) - normalized_contents

        del normalized_contents  # Free memory

        imgs_w: List[torch.Tensor] = []
        for i, ((H, W), img) in enumerate(zip(sizes, imgs)):
            delta_w_resize = transforms.Resize((H, W))(delta_w[i])

            img_w = self.postprocess(img + self.scaler * delta_w_resize)
            imgs_w.append(img_w.clamp(0, 1))  # Ensure pixel values are in [0, 1]

        return imgs_w  # List of watermarked images with original sizes


class CINWatermarkDetector:
    """
    CINWatermarkDetector is responsible for detecting watermarks in audio contents
    and extracting secret messages.
    """

    def __init__(self, model: torch.nn.Module, detection_bits: int = 16):
        self.model = model
        self.preprocess = transforms.Compose(
            [
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.Resize((128, 128)),
            ]
        )
        self.detection_bits = detection_bits

    @torch.inference_mode()
    def detect_watermark(
        self,
        contents: List[torch.Tensor],
        detection_threshold: float = 0.95,  # not used
        message_threshold: float = 0.5,
        detection_bits: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        imgs_resize: List[torch.Tensor] = []
        for img in contents:
            img_resize = self.preprocess(img)  # 3 x 128 x 128
            imgs_resize.append(img_resize)

        imgs_w = torch.stack(imgs_resize, dim=0)  # N x 3 x 128 x 128
        extracted = self.model(imgs_w)
        if detection_bits is None:
            detection_bits = self.detection_bits
        return get_detection_and_decoded_keys(
            extracted,
            detection_bits=detection_bits,
            message_threshold=message_threshold,
        )


def build_generator(
    model_card_or_path: str,
    scaler: float = 1.0,
    nbits: int = 30,
    device: Device = "cpu",
) -> CINWatermarkGenerator:
    """
    Builds a CIN watermark generator.
    ("Towards Blind Watermarking: Combining Invertible and Non-invertible Mechanisms").
    https://arxiv.org/abs/2212.12678

    Args:
        model_card_or_path (str): Path to the model card or model file.
        nbits (int): Number of bits for the watermark message.
        scaler (float): Scaling factor for the watermark.
        device (str): Device to load the model on. Default is "cpu".

    Returns:
        CINWatermarkGenerator: An instance of the CIN watermark generator.
    """
    model = torch.jit.load(model_card_or_path, map_location=torch.device(device)).eval()
    return CINWatermarkGenerator(model, scaler, nbits=nbits)


def build_detector(
    model_card_or_path: str,
    detection_bits: int = 16,
    device: str = "cpu",
) -> CINWatermarkDetector:
    """
    Builds a CIN watermark detector.
    ("Towards Blind Watermarking: Combining Invertible and Non-invertible Mechanisms").
    https://arxiv.org/abs/2212.12678

    Args:
        model_card_or_path (str): Path to the model card or model file.
        detection_bits (int): Number of bits for the watermark detection.
        device (str): Device to load the model on. Default is "cpu".

    Returns:
        CINWatermarkDetector: An instance of the CIN watermark detector.
    """
    model = torch.jit.load(model_card_or_path, map_location=torch.device(device)).eval()
    return CINWatermarkDetector(model, detection_bits)
