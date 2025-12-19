# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Microsoft, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in same directory of this source file.


import struct
import uuid

import numpy as np
import torch


def uuid_to_bits(batch_size):
    uid = [uuid.uuid4() for _ in range(batch_size)]
    seq = np.array([[n for n in u.bytes] for u in uid], dtype=np.uint8)
    bits = torch.Tensor(np.unpackbits(seq, axis=1)).to(torch.float32)
    strs = [str(u) for u in uid]
    return bits, strs


def uuid_to_bytes(batch_size):
    return [uuid.uuid4().bytes for _ in range(batch_size)]


def bits_to_uuid(bits, threshold=0.5):
    bits = np.array(bits) >= threshold
    nums = np.packbits(bits.astype(np.int64), axis=-1)
    res = []
    for j in range(nums.shape[0]):
        bstr = b""
        for i in range(nums.shape[1]):
            bstr += struct.pack(">B", nums[j][i])
        res.append(str(uuid.UUID(bytes=bstr)))
    return res
