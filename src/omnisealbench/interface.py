# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from abc import abstractmethod
from dataclasses import dataclass
from typing import (Callable, Dict, List, Optional, Protocol, Tuple, Type,
                    Union, get_origin, runtime_checkable)

import numpy as np
import torch

# A content to be watermarked or detected.
# It can be:
# - a tensor that corresponds to padded batch
# - a list of tensors in case of unpadded batch
# - a dictionary of tensors, e.g. {"content": tensor, "watermark_content": tensor}
# - a dictionary of lists of tensors, e.g. {"content": [tensor1, tensor2], "watermark_content": [tensor3, tensor4]}
ContentT = Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]


MessageT = Union[torch.Tensor, np.ndarray, None]

# A watermark result, can be a content or a dictionary.
# In case of single content, it indicates the localized detection probability
# per each content item (pixel, audio frame, etc.). No message is extracted.
# In case of dictionary, possible keys are:
# - "prediction": Nx1 tensor of detection probabilities
# - "message": Nxk tensor of secret message
# - "detection_bits": Nxd tensor indicating presence of watermark (d = detection bits)
ResultT = Union[
    ContentT,
    Dict[str, torch.Tensor],
    Dict[str, List[torch.Tensor]],
]

ScoresT = Dict[str, Union[List[float], torch.Tensor, np.ndarray]]

Device = Union[str, torch.device]


@runtime_checkable
class Generator(Protocol):
    """
    A post-hoc watermark model that applies watermark to a batch
    of contents (texts, audios, videos, images), and returns the watermarked content
    """
    
    @abstractmethod
    def generate_watermark(
        self,
        contents: ContentT,
        message: MessageT = None,
    ) -> ResultT:
        """
        Generate the watermarked conntents from the original contents
        Args:
            contents (ContentT): The input contents to be watermarked.
            message (MessageT): Optional tensor containing the message to embed in the watermark.
        Returns:
            ResultT: The watermarked contents in the same format as the input, optionally containing
            the extra info (embedded message, etc.).
        """
        ...


@runtime_checkable
class Detector(Protocol):
    """
    A watermark detection model that checks for the presence of a watermark in a batch
    of contents (texts, audios, videos, images), and optionally extracts the embedded message.
    """

    @abstractmethod
    def detect_watermark(
        self,
        contents: ContentT,
        detection_threshold: float = 0.95,
        message_threshold: float = 0.0,
        detection_bits: Optional[int] = None,
    ) -> ResultT:
        """
        Detect and extract watermark from the contents.
        Args:
            contents (ContentT): The input contents to check for watermarks.
            detection_threshold (float): Threshold for detecting the presence of a watermark.
            message_threshold (float): Threshold to convert tensor message to binary message.
            detection_bits (Optional[int]): Number of bits to use for detection. If None, use all available bits.
        Returns:
            ResultT: A result containing the detection probability and optionally the extracted secret message.
            The result can be a single detection probability, a tuple of detection probability and message,
            or a dictionary with keys "pred" for detection probabilities and "msg" for messages.
        """
        ...


ModelTypeT = Union[str, Callable, Type[Generator], Type[Detector]]


def cast_input(func: Callable, contents: ContentT) -> ContentT:
    """
    Adapt the input contents to the model's expected input format.
    This function inspects the model's input signature and adapts the contents accordingly.
    Args:
        model: The model to adapt the input for.
        func: A callable function that takes the contents as input.
        contents: The input contents to adapt.
    Returns:
        The adapted contents in the format expected by the model.
    """
    sig = inspect.signature(func)
    if "contents" not in sig.parameters:
        return contents
    content_T = sig.parameters["contents"].annotation

    if isinstance(contents, torch.Tensor) and (get_origin(content_T) is list or get_origin(content_T) is List):
        return list(contents)
    if isinstance(contents, list) and (get_origin(content_T) is torch.Tensor):
        try:
            return torch.stack(contents, dim=0)
        except RuntimeError as e:
            raise ValueError(
                f"The model expects a padded batch, but got contents with different shapes: {contents}. "
                "Usually, this happens when you choose the wrong collate function. "
                "Try using other collate functions like `collate_tensors_by_length` or `collate_tensors_to_the_longest`."
            ) from e

    return contents


@dataclass
class AverageMetric:
    """
    Average metric with confidence interval.

    avg is the mean of a list of values
    count is the length of this list
    square is the mean of the squares of the values
    avg_ci_fn is a function applied to the bounds of the confidence interval
    """

    avg: float
    count: int
    square: float
    avg_ci_fn: Optional[Callable] = None

    @property
    def value(self):
        return self.avg_ci_fn(self.avg) if self.avg_ci_fn else self.avg

    def update(self, value: float, count: int, square: Optional[float] = None) -> None:
        self.avg = (self.avg * self.count + value * count) / (self.count + count)
        if square is None:
            assert count == 1
            square = value**2
        self.square = (self.square * self.count + square * count) / (self.count + count)
        self.count += count

    def compute_ci(
        self, confidence_level: float = 0.95
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        returns bounds of confidence interval: ci_lb ('lower_bound') and ci_ub ('upper_bound').
        Confidence interval is computed with error margins:
        z * s / sqrt(n), where:
        - P(-z <= X <= z) = confidence_level and X follows a t student low with self.count - 1 parameters.
        - s is the unbiased std estimate: (1/(n-1) sum((xi - mean(xi) ** 2))) ** 0.5

        example: first 100 integers as metric_values and confidence_level = 0.95:
        >>> avg_m = AverageMetric(0, 0, 0)
        >>> for i in range(100):
        >>>     avg_m.update(value=i, count=1)
        >>> avg_m.compute_ci() # mean is 49.5, std is 29.0115, self.count = 100, z = 1.98
        >>> (43.743, 55.257)
        """
        from scipy.stats import t as student_t  # type: ignore

        if self.count < 2:
            return None, None

        std = (self.count / (self.count - 1) * (self.square - (self.avg) ** 2)) ** 0.5
        scale = std / (self.count**0.5)
        lb, ub = student_t.interval(confidence_level, self.count - 1, loc=self.avg, scale=scale)  # fmt: skip
        if self.avg_ci_fn:
            lb, ub = self.avg_ci_fn(lb), self.avg_ci_fn(ub)
        return (lb, ub)

AverageScoresT = Dict[str, AverageMetric]
