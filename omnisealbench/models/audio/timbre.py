# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import yaml
import numpy as np

from .generator import AudioWatermarkGenerator
from .detector import AudioWatermarkDetector
from .timbre_src import Encoder, Decoder


class TimbreWatermarkGenerator(AudioWatermarkGenerator):
    def __init__(self, name: str, checkpoint_pth: str, device: str):
        super().__init__(name, device)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        timbre_pth = os.path.join(dir_path, "timbre_src")

        process_config = yaml.load(
            open(f"{timbre_pth}/config/process.yaml", "r"), Loader=yaml.FullLoader
        )
        model_config = yaml.load(
            open(f"{timbre_pth}/config/model.yaml", "r"), Loader=yaml.FullLoader
        )

        win_dim = process_config["audio"]["win_len"]
        embedding_dim = model_config["dim"]["embedding"]
        nlayers_encoder = model_config["layer"]["nlayers_encoder"]
        attention_heads_encoder = model_config["layer"]["attention_heads_encoder"]

        msg_length = 30

        generator = Encoder(
            process_config,
            model_config,
            msg_length,
            win_dim,
            embedding_dim,
            nlayers_encoder=nlayers_encoder,
            attention_heads=attention_heads_encoder,
        ).to(device)

        checkpoint = torch.load(checkpoint_pth)
        generator.load_state_dict(checkpoint["encoder"])
        generator.eval()

        self.generator = generator

    @torch.inference_mode()
    def generate_watermark_audio(
        self,
        tensor: torch.Tensor,  # shape: (batch, channels, samples)
        sample_rate: int,
        secret_message: torch.tensor,  # shape: (batch, msg_length)
    ) -> torch.Tensor:  # shape: (batch, channels, samples)
        audios = tensor.to(self.device)
        secret_message = secret_message.to(device=self.device, dtype=torch.float32)

        """rescale the message to -1 and 1 according to the watermark function"""
        secret_message = secret_message.unsqueeze(0)
        secret_message_rescaled = secret_message * 2 - 1

        """watermark function call"""
        with torch.no_grad():  # Disable gradient calculation
            watermarked_signal = self.generator.test_forward(
                audios, secret_message_rescaled
            )[0]

        watermarked_audio = watermarked_signal.cpu()

        return watermarked_audio


class TimbreWatermarkDetector(AudioWatermarkDetector):
    def __init__(self, checkpoint_pth: str, device: str):
        super().__init__(device)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        timbre_pth = os.path.join(dir_path, "timbre_src")

        process_config = yaml.load(
            open(f"{timbre_pth}/config/process.yaml", "r"), Loader=yaml.FullLoader
        )
        model_config = yaml.load(
            open(f"{timbre_pth}/config/model.yaml", "r"), Loader=yaml.FullLoader
        )

        win_dim = process_config["audio"]["win_len"]
        embedding_dim = model_config["dim"]["embedding"]
        nlayers_decoder = model_config["layer"]["nlayers_decoder"]
        attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]

        detector = Decoder(
            process_config,
            model_config,
            30,
            win_dim,
            embedding_dim,
            nlayers_decoder=nlayers_decoder,
            attention_heads=attention_heads_decoder,
            hifigan_dir=f"{timbre_pth}/hifigan",
        ).to(device)

        checkpoint = torch.load(checkpoint_pth)
        detector.load_state_dict(checkpoint["decoder"], strict=False)
        detector.eval()

        self.detector = detector

    @torch.inference_mode()
    def detect_watermark_audio(
        self,
        tensor: torch.Tensor,  # b c=1 t
        sample_rate: int,
        detection_threshold: float = 0.5,
        message_threshold: float = 0.5,
    ) -> tuple[np.ndarray, torch.tensor]:  # b x 1, b x nbits
        tensor = tensor.to(self.device)

        b, _, _ = tensor.shape

        with torch.no_grad():  # Disable gradient calculation
            msg_decoded = self.detector.test_forward(tensor).squeeze()  # b x nbits

        detect_prob = np.ones((b,), dtype=np.float32)

        return detect_prob, msg_decoded
