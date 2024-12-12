# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from omnisealbench.metrics.message import get_bitacc, get_bitacc_v2, get_bitacc_v3


def str2msg(str):
    return [True if el == "1" else False for el in str]


@pytest.mark.parametrize(
    "bool_msg, key",
    [
        (
            "000100010010001011011000001110000010011001100111",
            "111010110101000001010111010011010100010000100111",
        ),
        ("1111101010010100", "1111101010010100"),
    ],
)
def test_bit_accuracies(bool_msg, key):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bool_msg = str2msg(bool_msg)
    bool_key = str2msg(key)

    keys = torch.tensor(bool_key, dtype=torch.float32).to(device)
    decoded = torch.tensor(bool_msg, dtype=torch.float32).to(device)

    bit_accs_1 = get_bitacc(decoded, keys)
    bit_accs_2 = get_bitacc_v2(decoded, keys)
    bit_accs_3 = get_bitacc_v3(decoded, keys)

    bit_accs_2 = bit_accs_2.item()
    bit_accs_3 = bit_accs_3.item()

    assert bit_accs_1 == bit_accs_2
    assert bit_accs_1 == bit_accs_3
