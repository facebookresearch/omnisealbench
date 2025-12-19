# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from scipy.stats import binomtest


def detection_bits_accuracy(
    detection_bits: torch.Tensor,
    gt_detection_bits: torch.Tensor,
    detection_threshold: float = 0.95,
) -> Dict[str, torch.Tensor]:
    """Calculate the detection bits accuracy between the detection bits and the ground truth bits."""

    detected = 2 * detection_bits - 1
    detected = detected.to(gt_detection_bits.device)  # b k

    gt_detection = 2 * gt_detection_bits - 1
    if gt_detection.ndim == 1:
        gt_detection = gt_detection.repeat(detected.shape[0], 1)  # b k

    diff = ~torch.logical_xor(detected > 0, gt_detection > 0)  # b k -> b k
    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # b k -> b
    detect = bit_accs > detection_threshold  # b

    return {
        "det_score": bit_accs,
        "det": detect,
    }


def plogp(p: torch.Tensor) -> torch.Tensor:
    """
    Return p log p
    Args:
        p (torch.Tensor): Probability tensor with shape BxK
    """
    plogp = p * torch.log2(p)
    plogp[p == 0] = 0
    return plogp


def message_accuracy(message: torch.Tensor, gt_message: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Calculate the message accuracy between the detected message and the ground truth message."""

    decoded = 2 * message - 1
    decoded = decoded.to(gt_message.device)  # b k

    gt_message = 2 * gt_message - 1
    if gt_message.ndim == 1 or (gt_message.ndim == 2 and gt_message.shape[0] == 1 and decoded.shape[0] > 1):
        gt_message = gt_message.repeat(decoded.shape[0], 1)  # b k

    # Bit accuracy: number of matching bits between the key and the message, divided by the total number of bits.
    # p-value: probability of observing a bit accuracy as high as the one observed, assuming the null hypothesis that the image is genuine.
    diff = ~torch.logical_xor(decoded > 0, gt_message > 0)  # b k -> b k

    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # b k -> b
    word_accs = bit_accs == 1  # b

    # Compute capacity based on bit accuracy
    nbits = decoded.shape[-1]
    entropy = - plogp(bit_accs) - plogp(1-bit_accs)
    capacities = 1 - entropy
    capacities = nbits * capacities

    successes_k = diff.sum(axis=-1)

    all_stats = defaultdict(list)
    for success_k, bit_acc, word_acc, capacity in zip(successes_k, bit_accs, word_accs, capacities):
        # k = len(diff[i]) - sum(diff[i])
        pval = binomtest(success_k, diff.shape[1], 0.5, alternative="greater")
        all_stats["bit_acc"].append(bit_acc.item())
        all_stats["word_acc"].append(word_acc.item())
        all_stats["p_value"].append(pval.pvalue)
        all_stats["capacity"].append(capacity.item())
        all_stats["log10_p_value"].append(np.log10(pval.pvalue) if pval.pvalue > 0 else -100)

    return {
        "bit_acc": torch.tensor(all_stats["bit_acc"]),
        "word_acc": torch.tensor(all_stats["word_acc"],),
        "p_value": torch.tensor(all_stats["p_value"]),
        "capacity": torch.tensor(all_stats["capacity"]),
        "log10_p_value": torch.tensor(all_stats["log10_p_value"]),
    }
