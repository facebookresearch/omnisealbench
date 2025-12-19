# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, Optional

import torch
import yaml

from omnisealbench.interface import Device
from omnisealbench.models.timbre_src import Decoder, Encoder
from omnisealbench.utils.detection import get_detection_and_decoded_keys


class TimbreWatermarkGenerator:
    model: Encoder
    nbits: int = 30

    def __init__(self, model: Encoder, nbits: int = 30):
        """
        Wrapper for the Timbre watermark generator
        ("Detecting Voice Cloning Attacks via Timbre Watermarking",
        https://arxiv.org/abs/2312.03410), https://github.com/TimbreWatermarking/TimbreWatermarking
        Args:
            model (Encoder): The watermarking model to use for generating watermarks.
        """
        self.model = model
        self.nbits = nbits

    @torch.inference_mode()
    def generate_watermark(
        self,
        contents: torch.Tensor,  # Audio tensor of shape (batch_size, channels, time)
        message: Optional[torch.Tensor] = None,  # Optional secret message of shape (nbits, )
    ) -> torch.Tensor:
        """
        Generates a watermarked audio signal by embedding a secret message into the input audio tensor.
        Args:
            contents (torch.Tensor): Audio tensor of shape (batch_size, channels, time).
                - batch_size: Number of audio samples in the batch.
                - channels: Number of audio channels (e.g., 1 for mono, 2 for stereo).
                - time: Number of time steps in the audio signal.
            message (Optional[torch.Tensor]): Optional secret message tensor of shape (nbits,) or (batch_size, nbits).
                - nbits: Number of bits in the secret message.
                - If shape is (nbits,), the message is broadcasted to all items in the batch.
        Returns:
            torch.Tensor: Watermarked audio tensor of shape (batch_size, channels, time), same as input contents.
        Notes:
            - The secret message is rescaled from {0, 1} to {-1, 1} before embedding.
            - Gradients are disabled during watermark generation.
        """

        B, C, T = contents.shape
        
        if len(message.shape) == 1:
            message = message.repeat(B, 1)  # (B, nbits)

        message = message.unsqueeze(1)  # (B, 1, nbits)
        secret_message_rescaled = message * 2 - 1  # (B, 1, nbits), values in {-1, 1}

        with torch.no_grad():  # Disable gradient calculation
            watermarked_signal, carrier_watermarked = self.model.test_forward(
            contents, secret_message_rescaled
            )  # watermarked_signal: (B, C, T)

        return watermarked_signal


class TimbreWatermarkDetector:
    model: Decoder

    def __init__(self, model: Decoder, nbits: int = 30, detection_bits: int = 16):
        """
        TimbreWatermarkDetector is responsible for detecting watermarks in audio and extracting secret messages.
            Args:
                model (TimbreWatermarkDetector): The detection model to use for identifying watermarks.
                message_threshold (float): The threshold to convert tensor message to binary message, default is 0.5.
        """
        self.model = model
        self.nbits = nbits
        self.detection_bits = detection_bits

    @torch.inference_mode()
    def detect_watermark(
        self,
        contents: torch.Tensor,
        detection_threshold: float = 0.95,  # not used
        message_threshold: float = 0.0,
        detection_bits: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Detects the watermark in the given audio contents.
        Args:
            contents (torch.Tensor): The input audio data to check for watermarks.
            detection_threshold (float): Threshold for detecting the presence of a watermark.
                For timbre the value is 0.0
            message_threshold (float): Threshold to convert tensor message to binary message.
                For timbre the value is 0.0
        Returns:
            torch.Tensor: A tensor indicating the presence of a watermark.
            Optional[torch.Tensor]: A tensor containing the extracted secret message, if any.
        """
        b, _, _ = contents.shape

        with torch.no_grad():  # Disable gradient calculation
            msg_decoded = self.model.test_forward(contents).squeeze(1)  # b x nbits
            
        if detection_bits is None:
            detection_bits = self.detection_bits

        return get_detection_and_decoded_keys(msg_decoded, detection_bits=detection_bits, message_threshold=message_threshold)


def build_generator(
    model_card_or_path: str,
    nbits: int = 30,
    device: Device = "cpu",
) -> TimbreWatermarkGenerator:
    """
    Build an Timbre watermark generator.
    ("Detecting Voice Cloning Attacks via Timbre Watermarking",
    https://arxiv.org/abs/2312.03410), https://github.com/TimbreWatermarking/TimbreWatermarking
    Args:
        model_card_or_path (str): Path to the model card or model file.
        nbits (int): Number of bits in the secret message to be embedded in the audio.
        device (str): Device to run the model on, e.g., 'cpu' or 'cuda'.
    Returns:
        TimbreWatermarkGenerator: An instance of the watermark generator.
    """
    assert nbits == 30, "nbits mismatch: {} != {}".format(nbits, 30)

    _device = torch.device(device) if device != "cpu" else None
    dir_path = os.path.dirname(os.path.realpath(__file__))
    timbre_pth = os.path.join(dir_path, "timbre_src")

    process_config = yaml.load(
        open(f"{timbre_pth}/config/process.yaml", "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(
        open(f"{timbre_pth}/config/model.yaml", "r"), Loader=yaml.FullLoader
    )

    win_dim = process_config["audio"]["win_len"]
    # embedding_dim = model_config["dim"]["embedding"]
    # nlayers_encoder = model_config["layer"]["nlayers_encoder"]
    # attention_heads_encoder = model_config["layer"]["attention_heads_encoder"]

    model = Encoder(
        process_config,
        model_config,
        nbits,
        win_dim,
    ).to(_device)

    checkpoint = torch.load(model_card_or_path)
    model.load_state_dict(checkpoint["encoder"])
    model.eval()

    generator = TimbreWatermarkGenerator(model)
    return generator


def build_detector(
    model_card_or_path: str,
    nbits: int = 30,
    detection_bits: int = 14,
    device: Device = "cpu",
) -> TimbreWatermarkDetector:
    """
    Build an Timbre watermark detector.
    ("Detecting Voice Cloning Attacks via Timbre Watermarking",
    https://arxiv.org/abs/2312.03410), https://github.com/TimbreWatermarking/TimbreWatermarking
    Args:
        model_card_or_path (str): Path to the model card or model file.
        nbits (int): Number of bits in the secret message to be detected.
        detection_bits (int): Number of bits used for detection.
        device (str): Device to run the model on, e.g., 'cpu' or 'cuda'.
    Returns:
        TimbreWatermarkDetector: An instance of the watermark detector.
    """
    assert nbits == 30, "nbits mismatch: {} != {}".format(nbits, 30)

    _device = torch.device(device) if device != "cpu" else "cpu"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    timbre_pth = os.path.join(dir_path, "timbre_src")

    process_config = yaml.load(
        open(f"{timbre_pth}/config/process.yaml", "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(
        open(f"{timbre_pth}/config/model.yaml", "r"), Loader=yaml.FullLoader
    )

    win_dim = process_config["audio"]["win_len"]
    # embedding_dim = model_config["dim"]["embedding"]
    # nlayers_decoder = model_config["layer"]["nlayers_decoder"]
    # attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]

    model = Decoder(
        process_config,
        model_config,
        nbits,
        win_dim,
        device=_device,
    ).to(_device)

    checkpoint = torch.load(model_card_or_path)
    model.load_state_dict(checkpoint["decoder"], strict=False)
    model.eval()

    detector = TimbreWatermarkDetector(model, nbits=nbits, detection_bits=detection_bits)
    return detector
