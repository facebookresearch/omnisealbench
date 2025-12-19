# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Optional

import torch
import torch.nn as nn


class ToyWatermarkGenerator:
    """
    A simple watermark generator for demonstration purposes.
    The model simply loads the cover content (audio, image, etc.)
    and generates a watermark by adding a constant secrete message of value M
    into it.
    """

    def __init__(self, model: torch.nn.Module, alpha: float = 0.0001, M: float = 1.0, nbits: int = 16):
        self.alpha = alpha
        self.M = M
        self.model = model
        self.dtype = torch.float32  # Assuming the model works with float32 tensors
        self.nbits = nbits

    def generate_watermark(
        self, contents: torch.Tensor, message: torch.Tensor
    ) -> torch.Tensor:

        # We ignore the given secret message and use a constant value M
        message = torch.full_like(
            contents, self.M, dtype=contents.dtype, device=contents.device
        )
        return contents + self.alpha * message


class ToyWatermark:

    model: nn.Module

    def __init__(
        self,
        model: torch.nn.Module,
        alpha: float = 0.0001,
        M: float = 1.0,
        nbits: int = 16,
    ):
        self.model = model

        # A wrapper can have any additional arguments. These arguments should be set in the builder func
        self.alpha = alpha
        self.M = M

        self.nbits = nbits

    @torch.inference_mode()
    def generate_watermark(
        self,
        contents: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:

        hidden = torch.full_like(
            contents, self.M, dtype=contents.dtype, device=contents.device
        )
        return contents + self.alpha * hidden

    @torch.inference_mode()
    def detect_watermark(
        self,
        contents: torch.Tensor,
        detection_threshold: float = 0.0,
        message_threshold: float = 0.0,
        detection_bits: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:

        B = len(contents)

        # Dummy implementation
        if self.alpha == 0:
            return {
                "prediction": torch.zeros(
                    (B,), device=contents.device
                ),  # No watermark detected
                "message": torch.zeros(
                    (B, self.nbits), device=contents.device
                ),  # Dummy message for demonstration
            }
        return {
            "prediction": torch.ones(
                (B,), device=contents.device
            ),  # Watermark detected
            "message": torch.ones(
                (B, self.nbits), dtype=contents.dtype, device=contents.device
            ),
        }


class ToyWatermarkDetector:
    """
    A simple watermark detector for demonstration purposes.
    It detects the watermark by checking if the content is modified by the watermark generator.
    """

    def __init__(self, model: torch.nn.Module, alpha: float = 0.0001, M: float = 1.0, nbits: int = 16):
        self.model = model
        self.alpha = alpha
        self.M = M
        self.nbits = nbits

    def detect_watermark(
        self, contents: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # The detect_watermark() should always return a dictionary with at least key "prediction" and
        # "message"

        B, *_ = contents.shape
        # Because of the simple nature of this toy model, we assume the contents are already watermarked
        # if alpha is non-zero.
        if self.alpha == 0:
            return {
                "detection_bits": torch.zeros(
                    (B,), device=contents.device
                ),  # No watermark detected
                "message": torch.zeros_like(
                    contents, device=contents.device
                ),  # Dummy message for demonstration
            }
        return {
            "detection_bits": torch.ones(
                (B,), device=contents.device
            ),  # Watermark detected
            "message": torch.full_like(
                contents, self.M, dtype=contents.dtype, device=contents.device
            ),
        }


def build_generator(alpha: float = 0.0001, M: float = 1.0, nbits: int = 16) -> ToyWatermarkGenerator:
    # Generate a dummy model
    model = torch.nn.Identity()  # No actual processing, just a placeholder
    return ToyWatermarkGenerator(model=model, alpha=alpha, M=M, nbits=nbits)


def build_detector(alpha: float = 0.0001, M: float = 1.0, nbits: int = 16) -> ToyWatermarkDetector:
    # Generate a dummy model
    model = torch.nn.Identity()  # No actual processing, just a placeholder
    return ToyWatermarkDetector(model=model, alpha=alpha, M=M, nbits=nbits)


def build_model(
    alpha: float = 0.0001, M: float = 1.0, nbits: int = 16, device: str = "cpu"
) -> ToyWatermark:

    model = torch.nn.Linear(2, 24)  # no actual model, just a placeholder
    model.eval().to(device)
    return ToyWatermark(model=model, alpha=alpha, M=M, nbits=nbits)
