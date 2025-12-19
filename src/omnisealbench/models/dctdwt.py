# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# mypy: disable-error-code="import-untyped, assignment"

from typing import Dict, List, Optional

import imwatermark  # type: ignore[import]
import torch

from omnisealbench.interface import Device
from omnisealbench.utils.detection import get_detection_and_decoded_keys


class DCTDWTGenerator:
    """
    Wrapper of DCTDWT ("Combined dwt-dct digital image watermarking")
    https://github.com/ShieldMnt/invisible-watermark

    This class is responsible both for generating and detecting watermarks.
    """

    model: torch.nn.Module
    nbits: int

    def __init__(self, model: torch.nn.Module, nbits: int = 48, device: Device = "cpu"):
        self.model = model
        self.nbits = nbits
        self.device = device

    def _encode(self, img: torch.Tensor) -> torch.Tensor:
        img = img.permute(1, 2, 0)  # (C,H,W) to (H,W,C)
        img = (img * 255).to(torch.uint8)  # Convert to uint8 for CV2-style image
        img = img.cpu().numpy()[..., ::-1]  # type: ignore
        encoded = self.model.encode(img, "dwtDct")  # type: ignore
        encoded = encoded[..., ::-1]
        img_w = (
            torch.from_numpy(encoded.copy()).float() / 255.0
        )  # Convert back to float tensor
        img_w = img_w.permute(2, 0, 1)
        return img_w.to(self.device)

    @torch.inference_mode()
    def generate_watermark(
        self, contents: List[torch.Tensor], message: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Generates a watermark for the given image contents.
        Args:
            contents (torch.Tensor): The input image data to be watermarked.
            message (torch.Tensor): Tensor containing the message to embed in the watermark.
        Returns:
            torch.Tensor: The watermarked image tensor.
        """
        assert isinstance(contents, list), "DCT-DWT expects a list of tensors."
        message = message.detach().cpu().numpy()  # type: ignore
        self.model.set_watermark(wmType="bits", content=message)  # type: ignore

        imgs_w = [self._encode(img) for img in contents]
        return imgs_w  # type: ignore[return-value]


class DCTDWTDetector:
    """
    DCTDWTDetector is responsible for detecting watermarks in image contents and extracting secret messages.
    """

    model: torch.nn.Module
    nbits: int

    def __init__(self, model: torch.nn.Module, nbits: int = 48, detection_bits: int = 16, device: str = "cpu"):
        self.model = model
        self.nbits = nbits
        self.detection_bits = detection_bits
        self.device = device

    def _decode(self, img: torch.Tensor) -> torch.Tensor:
        img = img.permute(1, 2, 0)  # (C,H,W) to (H,W,C)
        img = (img * 255).to(torch.uint8)  # Convert to uint8 for CV2-style image
        img = img.cpu().numpy()[..., ::-1]  # type: ignore
        res = self.model.decode(img, method="dwtDct")  # type: ignore
        return torch.from_numpy(res).to(self.device)

    @torch.inference_mode()
    def detect_watermark(
        self,
        contents: List[torch.Tensor],
        detection_threshold: float = 0.95,  # not used
        message_threshold: float = 0.0,  # not used, DctDwt uses raw message values
        detection_bits: Optional[int] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Detects the watermark in the given image contents.
        Args:
            contents (torch.Tensor): The input image data to be checked for watermark.
        Returns:
            torch.Tensor: The detected watermark message.
        """
        extracted = [self._decode(img) for img in contents]
        extracted = torch.stack(extracted, dim=0).float()
        if detection_bits is None:
            detection_bits = self.detection_bits

        return get_detection_and_decoded_keys(
            extracted, detection_bits=detection_bits, message_threshold=None)

def build_generator(device: Device = "cpu", nbits: int = 48) -> DCTDWTGenerator:
    """Load the pretrained DCTDWT model for watermark generation.
    See "Combined dwt-dct digital image watermarking"
    https://github.com/ShieldMnt/invisible-watermark

    Args:
        device (Device): The device to run the model on (e.g., "cpu", "cuda").
        nbits (int): Number of bits in the watermark message. Default is 48.
    """
    model = imwatermark.WatermarkEncoder()  # type: ignore

    # DCTDWT wraps over CV2 and does not work in CUDA
    # model = model.to(device)
    return DCTDWTGenerator(model=model, nbits=nbits, device=device)


def build_detector(
    wm_type: str = "bits", nbits: int = 48, detection_bits: int = 16, device: str = "cpu"
) -> DCTDWTDetector:
    """Load the pretrained DCTDWT model for watermark detection.
    See "Combined dwt-dct digital image watermarking"
    https://github.com/ShieldMnt/invisible-watermark.
    Args:
        wm_type (str): The type of watermark to detect (e.g., "bits").
        nbits (int): Number of bits in the watermark message. Default is 48.
        detection_bits (int): Number of bits used for detection. Default is 16.
        device (str): Device to load the model on (e.g., 'cpu', 'cuda').
    """
    model = imwatermark.WatermarkDecoder(wm_type=wm_type, length=nbits)  # type: ignore
    # DCTDWT wraps over CV2 and does not work in CUDA
    # model = model.to(device)
    return DCTDWTDetector(model=model, nbits=nbits, detection_bits=detection_bits, device=device)
