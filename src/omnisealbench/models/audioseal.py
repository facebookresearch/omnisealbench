# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
from audioseal import AudioSeal, AudioSealDetector, AudioSealWM

from omnisealbench.interface import Device


class AudioSealWatermarkGenerator:
    model: AudioSealWM
    alpha: float = 1.0
    nbits: int = 16

    def __init__(self, model: AudioSealWM, alpha: float = 1.0, nbits: int = 16):
        """
        Wrapper for the AudioSeal watermark generator
        ("Proactive Detection of Voice Cloning with Localized Watermarking",
        https://arxiv.org/abs/2401.17264), https://github.com/facebookresearch/audioseal
        Args:
            model (AudioSealWM): The watermarking model to use for generating watermarks.
            alpha (float): A scaling factor for watermark generation, default is 1.0.
        """
        self.model = model
        self.alpha = alpha
        self.nbits = nbits

    @torch.inference_mode()
    def generate_watermark(
        self,
        contents: torch.Tensor,
        message: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generates a watermark for the given audio contents.
        Optionally, a secret n-bits message tensor can be provided to enhance authorship.
        Args:
            contents (torch.Tensor): The input audio data to be watermarked.
            message (Optional[torch.Tensor]): Optional tensor containing the message to embed in the watermark.
        Returns:
            torch.Tensor: The watermarked audio tensor.
        """
        return self.model(contents, message=message, alpha=self.alpha, sample_rate=16_000)


class AudioSealWatermarkDetector:
    model: AudioSealDetector

    def __init__(self, model: AudioSealDetector):
        """
        AudioSealDetector is responsible for detecting watermarks in audio and extracting secret messages.
            Args:
                model (AudioSealDetector): The detection model to use for identifying watermarks.
                message_threshold (float): The threshold to convert tensor message to binary message, default is 0.5.
        """
        self.model = model

    @torch.inference_mode()
    def detect_watermark(
        self,
        contents: torch.Tensor,
        detection_threshold: float = 0.5,
        message_threshold: float = 0.5,
        detection_bits: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Detects the watermark in the given audio contents.
        Args:
            contents (torch.Tensor): The input audio data to check for watermarks.
            detection_threshold (float): Threshold for detecting the presence of a watermark.
            message_threshold (float): Threshold to convert tensor message to binary message.
        Returns:
            torch.Tensor: A tensor indicating the presence of a watermark.
            Optional[torch.Tensor]: A tensor containing the extracted secret message, if any.
        """
        pred, message = self.model.detect_watermark(
            contents,
            detection_threshold=detection_threshold,
            message_threshold=message_threshold,
            sample_rate=16_000,
        )
        # pred: b x 1
        # message: b x k
        if isinstance(pred, float):
            pred = torch.tensor([pred])
        return {"prediction": pred, "message": message}


def build_generator(
    model_card_or_path: str,
    nbits: int = 16,
    alpha: float = 1.0,
    device: Device = "cpu",
) -> AudioSealWatermarkGenerator:
    """
    Build an AudioSeal watermark generator.
    See "Proactive Detection of Voice Cloning with Localized Watermarking",
        (https://arxiv.org/abs/2401.17264), https://github.com/facebookresearch/audioseal
    Args:
        model_card_or_path (str): Path to the model card or model file for the generator.
        nbits (int): Number of bits in the secret message to be embedded in the audio.
        alpha (float): Scaling factor for watermark generation.
        device (str): Device to run the model on, e.g., 'cpu' or 'cuda'.
    Returns:
        AudioSealWatermarkGenerator: An instance of the watermark generator.
    """
    _device = torch.device(device) if device != "cpu" else None
    model = AudioSeal.load_generator(model_card_or_path, nbits=nbits, device=_device)
    model.eval()
    generator = AudioSealWatermarkGenerator(model, alpha=alpha)
    return generator


def build_detector(
    model_card_or_path: str,
    nbits: int = 16,
    device: Device = "cpu",
) -> AudioSealWatermarkDetector:
    """
    Build an AudioSeal watermark detector.
    See "Proactive Detection of Voice Cloning with Localized Watermarking",
        (https://arxiv.org/abs/2401.17264), https://github.com/facebookresearch/audioseal
    Args:
        model_card_or_path (str): Path to the model card or model file for the detector.
        nbits (int): Number of bits in the secret message to be detected.
        message_threshold (float): Threshold to convert tensor message to binary message.
        device (str): Device to run the model on, e.g., 'cpu' or 'cuda'.
    Returns:
        AudioSealWatermarkDetector: An instance of the watermark detector.
    """
    _device = torch.device(device) if device != "cpu" else None
    model = AudioSeal.load_detector(model_card_or_path, nbits=nbits, device=_device)
    model.eval()
    detector = AudioSealWatermarkDetector(model)
    return detector
