# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections.abc
import os
from pathlib import Path

import torchaudio

from .base import AudioSample


def load_audio_file(audio_file_path: str) -> AudioSample:
    waveform, sample_rate = torchaudio.load(audio_file_path)
    # waveform is of shape (channels, samples)
    return AudioSample(audio=waveform, sample_rate=sample_rate)


class LocalDataset(collections.abc.Sequence):
    def __init__(self, dataset_path: str, file_extension: str = ".mp3"):
        dataset_path = Path(dataset_path)
        audio_file_paths = [
            dataset_path / file
            for file in os.listdir(dataset_path)
            if file.endswith(file_extension)
        ]

        self.audio_file_paths = sorted(audio_file_paths)

    def __getitem__(self, index) -> AudioSample:
        audio_file_path = self.audio_file_paths[index]

        return load_audio_file(audio_file_path)

    def __len__(self) -> int:
        return len(self.audio_file_paths)


def get_local_db(dataset_dir: str, file_extension: str):
    assert dataset_dir is not None
    assert os.path.isdir(
        dataset_dir
    ), f"the local db implementation expects for an existing dataset_dir argument: {dataset_dir} doesnt exist ..."

    return LocalDataset(dataset_dir, file_extension)
