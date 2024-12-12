# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from audioseal import AudioSeal

from .generator import AudioWatermarkGenerator
from .detector import AudioWatermarkDetector


class AudioSealWatermarkGenerator(AudioWatermarkGenerator):
    def __init__(self, name: str, model_card_or_path: str, device: str):
        super().__init__(name, device)

        model = AudioSeal.load_generator(model_card_or_path)
        self.model = model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def generate_watermark_audio(
        self,
        tensor: torch.Tensor,  # shape: (batch, channels, samples)
        sample_rate: int,
        secret_message: torch.tensor,
    ) -> torch.Tensor:  # shape: (batch, channels, samples)
        audios = tensor.to(self.device)
        secret_message = secret_message.to(self.device)

        with torch.no_grad():  # Disable gradient calculation
            watermarked_audio = self.model(
                audios,
                sample_rate=sample_rate,
                message=secret_message.to(self.device),
                alpha=1,
            )

        watermarked_audio = watermarked_audio.cpu()

        return watermarked_audio


class AudioSealWatermarkDetector(AudioWatermarkDetector):
    def __init__(self, model_card_or_path: str, device: str):
        super().__init__(device)

        detector = AudioSeal.load_detector(model_card_or_path)
        self.detector = detector.to(device)
        self.detector.eval()

    @torch.inference_mode()
    def detect_watermark_audio(
        self,
        tensor: torch.Tensor,
        sample_rate: int,
        detection_threshold: float = 0.5,
        message_threshold: float = 0.5,
    ) -> tuple[np.ndarray, torch.tensor]:  # b x 1, b x nbits
        tensor = tensor.to(self.device)

        with torch.no_grad():  # Disable gradient calculation
            result, message = self.detector.forward(
                tensor, sample_rate=sample_rate
            )  # b x 2+nbits

        detected = (
            torch.count_nonzero(torch.gt(result[:, 1, :], detection_threshold), dim=1)
            / result.shape[-1]
        )
        detect_prob = detected.cpu()  # type: ignore
        detect_prob = detect_prob.numpy()
        message = torch.gt(message, message_threshold).int()

        return detect_prob, message
