# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

try:
    import wavmark  # type: ignore
except ImportError:
    raise RuntimeError(
        "could not import wavmark, make sure it is installed, "
        "please run `pip install wavmark`"
    )

from .generator import AudioWatermarkGenerator
from .detector import AudioWatermarkDetector


class WavmarkWatermarkGenerator(AudioWatermarkGenerator):
    def __init__(self, name: str, device: str):
        super().__init__(name, device)

        model = wavmark.load_model()
        self.model = model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def generate_watermark_audio(
        self,
        tensor: torch.Tensor,  # shape: (batch, channels, samples)
        sample_rate: int,
        secret_message: torch.tensor,
    ) -> torch.Tensor:  # shape: (batch, channels, samples)
        audios = tensor[0, 0].numpy()  # single-channel 16kHz format
        secret_message = secret_message[0].numpy()

        with torch.no_grad():  # Disable gradient calculation
            watermarked_audio, info = wavmark.encode_watermark(
                self.model, audios, secret_message, show_progress=False
            )
            # snr = info['snr']

        watermarked_audio = torch.tensor(watermarked_audio)
        watermarked_audio = watermarked_audio[None, None, :]

        return watermarked_audio


class WavmarkWatermarkDetector(AudioWatermarkDetector):
    def __init__(self, device: str):
        super().__init__(device)

        detector = wavmark.load_model()
        self.detector = detector.to(device)
        self.detector.eval()

    @torch.inference_mode()
    def detect_watermark_audio(
        self,
        tensor: torch.Tensor,  # b c=1 t
        sample_rate: int,
        detection_threshold: float = 0.5,
        message_threshold: float = 0.5,
    ) -> tuple[np.ndarray, torch.tensor]:  # b x 1, b x nbits
        tensor = tensor.cpu()

        detect_probs = []
        messages = []
        with torch.no_grad():  # Disable gradient calculation
            b, c, t = tensor.shape
            for i in range(b):
                payload_decoded, info = wavmark.decode_watermark(
                    self.detector, tensor[i, 0], show_progress=False
                )
                if len(info["results"]) == 0:
                    detect_prob = 0.0
                    message = [0] * 16
                else:
                    detect_prob = info["results"][0]["sim"]
                    message = payload_decoded

                detect_probs.append(detect_prob)
                messages.append(message)

        messages = np.array(messages)
        message = torch.tensor(messages, device=self.device).int()
        detect_prob = np.array(detect_probs)

        return detect_prob, message
