# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List, Optional, Union, overload

import torch


def get_detection_and_decoded_keys_tensor(
    extracted: torch.Tensor,
    detection_bits: int = 16,
    message_threshold: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    if message_threshold is not None:
        extracted = torch.gt(extracted, message_threshold).int()  # b k
    message = extracted[:, detection_bits:]
    detected = extracted[:, :detection_bits]
    return {
        "detection_bits": detected,
        "message": message,
    }


def get_detection_and_decoded_keys_list(
    extracted: List[torch.Tensor],
    detection_bits: int = 16,
    message_threshold: Optional[float] = None,
) -> Dict[str, List[torch.Tensor]]:
    results = defaultdict(list)

    for example in extracted:
        if message_threshold is not None:
            example = torch.gt(example, message_threshold).int()  # b k
        message = example[:, detection_bits:]
        detected = example[:, :detection_bits]

        results["detection_bits"].append(detected)
        results["message"].append(message)

    return results


@overload
def get_detection_and_decoded_keys(
    extracted: torch.Tensor,
    detection_bits: int = 16,
    message_threshold: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    ...


@overload
def get_detection_and_decoded_keys(
    extracted: List[torch.Tensor],
    detection_bits: int = 16,
    message_threshold: Optional[float] = None,
) -> Dict[str, List[torch.Tensor]]:
    ...


def get_detection_and_decoded_keys(
    extracted: Union[torch.Tensor, List[torch.Tensor]],
    detection_bits: int = 16,
    message_threshold: Optional[float] = None,
) -> Union[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:
    """
    Extract detection and decoding keys from the extracted message.
    Args:
        keys (torch.Tensor): A tensor containing the keys.
    Returns:
        Dict[str, torch.Tensor]: A dictionary with keys "detection" and "decoding" containing the respective keys.
    """
    if isinstance(extracted, torch.Tensor):
        return get_detection_and_decoded_keys_tensor(
            extracted,
            detection_bits=detection_bits,
            message_threshold=message_threshold,
        )
    elif isinstance(extracted, list):
        return get_detection_and_decoded_keys_list(
            extracted,
            detection_bits=detection_bits,
            message_threshold=message_threshold,
        )
    else:
        raise TypeError("extracted must be a torch.Tensor or a list of torch.Tensor")
