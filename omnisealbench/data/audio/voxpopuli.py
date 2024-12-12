# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from datasets import load_dataset

from .base import AudioSample


def get_voxpopuli_dataset() -> List[AudioSample]:
    voxpopuli_en_test = load_dataset(
        "facebook/voxpopuli", "en", split="test", trust_remote_code=True
    )

    dataset = []

    for sample in voxpopuli_en_test:
        audio = sample["audio"]

        audio_tensor = torch.tensor(audio["array"]).float()

        dataset.append(
            AudioSample(
                audio=audio_tensor[None, :],
                sample_rate=audio["sampling_rate"],
            )
        )

    return dataset
