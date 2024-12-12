# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import torch


class AudioWatermarkGenerator(ABC):
    def __init__(self, name: str, device: str):
        self.name = name
        self.device = device

    @abstractmethod
    def generate_watermark_audio(
        self,
        tensor: torch.Tensor,  # shape: (batch, channels, samples)
        sample_rate: int,
        secret_message: torch.tensor,
    ) -> torch.Tensor:  # shape: (batch, channels, samples)
        """
        Function to generate a watermark for the audio and embed it into a new audio tensor
        Args:
            tensor: Audio tensor with shape (batch, channels, samples)
            sample_rate: Sample rate of the audio
            secret_message: Secret message to be embedded in the audio
        Returns:
            Watermarked audio tensor with shape (batch, channels, samples)
        """

        raise NotImplementedError
