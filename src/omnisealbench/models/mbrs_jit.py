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


class MBRSWatermarkGenerator:
    """
    Wrapper for the MBRS watermark generator model
    ("Mbrs: Enhancing robustness of dnn-based watermarking by
    mini-batch of real and simulated jpeg compression",
    https://arxiv.org/abs/2108.08211)
    """

    model: torch.nn.Module

    def __init__(
        self,
        model: torch.nn.Module,
        nbits: int = 256,
        scaler: float = 1.0,
    ):
        """
        Initializes the MBRS watermark generator with a given model and parameters.

        Args:
            model (torch.nn.Module): The MBRS model to use for watermark generation.
            scaler (float): Scaling factor for the watermark. Default is 1.0.
        """
        self.model = model
        self.scaler = scaler
        self.nbits = nbits

        # Normalization used in the MBRS model
        self.preprocess = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
        self.postprocess = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
        self.resize = transforms.Resize((256, 256))

    @torch.inference_mode()
    def generate_watermark(
        self,
        contents: List[torch.Tensor],
        message: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Generates a watermark for the given images.

        Args:
            contents (torch.Tensor): The input image data to be watermarked.
            message (torch.Tensor): Tensor containing the message to embed in the watermark.

        Returns:
            torch.Tensor: The watermarked image tensor.
        """
        D = message.shape[-1]

        imgs_norm: List[torch.Tensor] = []
        imgs_size: List[Tuple[int, int]] = []
        imgs_resize: List[torch.Tensor] = []

        for img in contents:
            img_norm = self.preprocess(img)  # C H W 
            imgs_norm.append(img_norm)

            imgs_size.append(tuple(img.shape[-2:]))  # H W
            img_resize = self.resize(img_norm)  # C 256 256

            imgs_resize.append(img_resize)

        input_ = torch.stack(imgs_resize, dim=0)  # N C

        # encode a 256 message. Repeat the message to match the number of bits, and add 0s to complete to 256
        if message.ndim > 1:
            message = message[0]
        msgs_to_encode = torch.cat(
            [message.repeat(256 // D), torch.zeros(256 % D, device=message.device)]
        )
        msgs_to_encode = msgs_to_encode.repeat(len(contents), 1)
        delta_w = self.model(input_, msgs_to_encode) - input_

        imgs_w: List[torch.Tensor] = []
        for i, (size, img) in enumerate(zip(imgs_size, imgs_norm)):
            # Resize delta_w to the original image size
            delta_w_resize = transforms.Resize(size)(delta_w[i])

            img_w = self.postprocess(img + self.scaler * delta_w_resize)
            imgs_w.append(img_w.clamp(0, 1))  # Collect watermarked images

        return imgs_w


class MBRSWatermarkDetector:
    """
    Wrapper for the MBRS watermark detector model
    """

    model: torch.nn.Module

    def __init__(
        self,
        model: List[torch.nn.Module],
        detection_bits: int = 16,
        nbits: int = 16,
    ):
        """
        Initializes the MBRS watermark detector with a given model and parameters.

        Args:
            model (torch.nn.Module): The MBRS model to use for watermark detection.
            nbits (int): Number of bits in the watermark message. Default is 16.
            detection_bits (int): Number of bits used for detection. Default is 16.
        """
        self.model = model
        self.nbits = nbits
        self.transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.Resize((256, 256)),
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
        """
        Detects the watermark in the given images.

        Args:
            contents (torch.Tensor): The input image data to be checked for watermarks.

        Returns:
            torch.Tensor: The detection scores for the watermarks.
        """
        imgs = [self.transform(img) for img in contents]
        input_ = torch.stack(imgs, dim=0)  # N C H W

        extracted = self.model(input_) # N 256
        _, D = extracted.shape
        extracted = extracted[:, : D - D % self.nbits]

        # average every self.nbit output
        extracted = extracted.view(len(contents), -1, self.nbits).mean(1)

        return get_detection_and_decoded_keys(
            extracted,
            detection_bits=detection_bits if detection_bits is not None else self.detection_bits,
            message_threshold=message_threshold
        )


def build_generator(
    model_card_or_path: str,
    nbits: int = 256,
    scaler: float = 1.0,
    device: Device = "cpu",
) -> MBRSWatermarkGenerator:
    """
    Load the pretrained MBRS model for watermark generation.
    ("Mbrs: Enhancing robustness of dnn-based watermarking by
    mini-batch of real and simulated jpeg compression",
    https://arxiv.org/abs/2108.08211)

    Args:
        model_card_or_path (str): Path to the model card or model file.
        nbits (int): Number of bits in the watermark message. Default is 256.
        scaler (float): Scaling factor for the watermark. Default is 1.0.
        device (str): Device to load the model on. Default is "cpu".

    Returns:
        MBRSWatermarkGenerator: An instance of the MBRS watermark generator.
    """
    model = torch.jit.load(model_card_or_path, map_location=torch.device(device)).eval()
    return MBRSWatermarkGenerator(model=model, scaler=scaler, nbits=nbits)


def build_detector(
    model_card_or_path: str,
    detection_bits: int = 16,
    nbits: int = 256,
    device: Device = "cpu",
) -> MBRSWatermarkDetector:
    """
    Load the pretrained MBRS model for watermark detection.
    ("Mbrs: Enhancing robustness of dnn-based watermarking by
    mini-batch of real and simulated jpeg compression",
    https://arxiv.org/abs/2108.08211)
    
    Args:
        model_card_or_path (str): Path to the model card or model file.
        detection_bits (int): Number of bits used for detection. Default is 16.
        nbits (int): Number of bits in the watermark message. Default is 16.
        device (str): Device to load the model on. Default is "cpu".

    Returns:
        MBRSWatermarkDetector: An instance of the MBRS watermark detector.
    """
    model = torch.jit.load(model_card_or_path, map_location=torch.device(device)).eval()
    return MBRSWatermarkDetector(
        model=model, detection_bits=detection_bits, nbits=nbits
    )
