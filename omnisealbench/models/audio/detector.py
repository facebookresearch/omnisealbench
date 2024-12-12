# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import numpy as np
import torch


class AudioWatermarkDetector(ABC):
    def __init__(self, device: str):
        self.device = device

    @abstractmethod
    def detect_watermark_audio(
        self,
        tensor: torch.Tensor,  # b c=1 t
        sample_rate: int,
        detection_threshold: float = 0.5,
        message_threshold: float = 0.50,
    ) -> tuple[np.ndarray, torch.tensor]:  # b x 1, b x nbits
        """
        Function to get the confidence score that an audio tensor was watermarked by a watermark generator
        """

        raise NotImplementedError
