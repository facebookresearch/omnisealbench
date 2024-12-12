# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def get_bitacc(msg_decoded: torch.Tensor, message: torch.Tensor) -> float:
    msg_decoded = msg_decoded * 2 - 1
    message = message * 2 - 1

    return (msg_decoded >= 0).eq(message >= 0).sum(dim=-1).item() / message.shape[-1]


def get_bitacc_v2(msg_decoded: torch.Tensor, message: torch.Tensor) -> float:
    msg_decoded = msg_decoded * 2 - 1
    message = message * 2 - 1

    diff = ~torch.logical_xor(msg_decoded > 0, message > 0)  # b k -> b k
    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # b k -> b
    return bit_accs


def get_bitacc_v3(msg_decoded: torch.Tensor, message: torch.Tensor) -> float:
    msg_decoded = msg_decoded * 2 - 1  # preds
    message = message * 2 - 1  # targets

    correct = (msg_decoded == message).float()  # b k
    bit_acc = torch.mean(correct, dim=-1)  # b
    return bit_acc
