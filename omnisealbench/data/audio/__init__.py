# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Callable, Dict

import torch
from torch.utils.data import Dataset

from .base import *
from .builders import *


class WatermarkedAudioDataset(Dataset):
    def __init__(
        self,
        watermarks_dir: str,
        total: int,
    ):
        self.watermarks_dir = Path(watermarks_dir)
        self.total = total

    def __len__(self):
        return self.total

    def __getitem__(
        self,
        idx,
    ) -> tuple[int, torch.tensor, torch.tensor, torch.tensor]:
        sample_path = self.watermarks_dir / f"sample_{idx}.pt"
        waveform_path = self.watermarks_dir / f"audio_{idx}.pt"
        message_path = self.watermarks_dir / f"message_{idx}.pt"

        waveform_audio = torch.load(waveform_path, weights_only=False)  # b=1 c=1 t
        watermarked_audio = torch.load(sample_path, weights_only=False)  # b=1 c=1 t
        secret_message = torch.load(message_path, weights_only=False)  # b=1 16

        audio = waveform_audio[0]  # c=1 t
        wmd_audio = watermarked_audio[0]  # c=1 t

        return idx, audio, wmd_audio, secret_message


def apply_attacks(
    audio: torch.Tensor,  # b=1 c=1 t
    watermarked_audio: torch.Tensor,  # b=1 c=1 t
    sample_rate: int,
    attack_implementation: Callable[..., torch.Tensor],
    attack_runtime_config: dict,
    postprocess_fn: Callable[
        [int, torch.Tensor, Callable[..., torch.Tensor], Dict], torch.Tensor
    ],
    device: str,
    attack_orig_audio: bool = True,
) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    audio = audio.to(device)
    watermarked_audio = watermarked_audio.to(device)

    b, _, t = audio.shape

    if attack_orig_audio:
        batch_audio = torch.cat([audio, watermarked_audio])

        attacked_batch_audio = postprocess_fn(
            sample_rate,
            batch_audio,
            attack_implementation,
            attack_runtime_config,
        )

        attacked_audio = attacked_batch_audio[:b]  # b c=1 t
        attacked_watermarked_audio = attacked_batch_audio[b:]  # b c=1 t

        # some attacks/effects (ex: speed) can change the tensor size, so
        # we truncate the attacked audio, wmd_audio and orig_audio to the same size
        min_t = min(t, attacked_audio.shape[2])

        audio = audio[:, :, :min_t]
        watermarked_audio = watermarked_audio[:, :, :min_t]
        attacked_audio = attacked_audio[:, :, :min_t]
        attacked_watermarked_audio = attacked_watermarked_audio[:, :, :min_t]
    else:
        attacked_watermarked_audio = postprocess_fn(
            sample_rate,
            watermarked_audio,
            attack_implementation,
            attack_runtime_config,
        )

        attacked_audio = None

        min_t = min(t, attacked_watermarked_audio.shape[2])
        audio = audio[:, :, :min_t]
        watermarked_audio = watermarked_audio[:, :, :min_t]
        attacked_watermarked_audio = attacked_watermarked_audio[:, :, :min_t]

    return audio, watermarked_audio, attacked_audio, attacked_watermarked_audio


class WatermarkedAudioAttacksDataset(Dataset):
    def __init__(
        self,
        sample_rate: int,
        watermarks_dir: str,
        total: int,
        attack_implementation: Callable[..., torch.Tensor],
        attack_runtime_config: dict,
        postprocess_fn: Callable[
            [int, torch.Tensor, Callable[..., torch.Tensor], Dict], torch.Tensor
        ],
        attack_orig_audio: bool = False,
    ):
        self.sample_rate = sample_rate
        self.watermarks_dir = Path(watermarks_dir)
        self.total = total
        self.attack_implementation = attack_implementation
        self.attack_runtime_config = attack_runtime_config
        self.postprocess_fn = postprocess_fn
        self.attack_orig_audio = attack_orig_audio

    def __len__(self):
        return self.total

    def __getitem__(
        self,
        idx,
    ) -> tuple[int, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        sample_path = self.watermarks_dir / f"sample_{idx}.pt"
        waveform_path = self.watermarks_dir / f"audio_{idx}.pt"
        message_path = self.watermarks_dir / f"message_{idx}.pt"

        waveform_audio = torch.load(waveform_path, weights_only=False)  # b=1 c=1 t
        watermarked_audio = torch.load(sample_path, weights_only=False)  # b=1 c=1 t
        secret_message = torch.load(message_path, weights_only=False)  # b=1 16

        audio, attacked_audio, attacked_watermarked_audio = apply_attacks(
            waveform_audio,
            watermarked_audio,
            sample_rate=self.sample_rate,
            attack_implementation=self.attack_implementation,
            attack_runtime_config=self.attack_runtime_config,
            attack_orig_audio=self.attack_orig_audio,
        )

        attacked_watermarked_audio = attacked_watermarked_audio[0]  # c=1 t
        attacked_audio = attacked_audio[0]  # c=1 t
        audio = audio[0]  # c=1 t

        return idx, attacked_watermarked_audio, attacked_audio, audio, secret_message


def postprocess_batch(
    batch: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[
    list[int],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    indices = []
    watermarked_audios = []
    attacked_audios = []
    audios = []
    secret_messages = []

    max_size = 0
    has_changes = False

    for idx, watermarked_audio, attacked_audio, audio, message in batch:
        indices.append(idx)

        watermarked_audios.append(watermarked_audio)
        attacked_audios.append(attacked_audio)
        audios.append(audio)
        secret_messages.append(message)

        current_size = watermarked_audio.shape[1]

        if current_size > max_size:
            has_changes = max_size != 0

            max_size = current_size
        elif current_size != max_size:
            has_changes = True

    # Pad each tensor in the list
    padded_wmd_audios = []
    padded_attacked_audios = []
    padded_audios = []
    for watermarked_audio, attacked_audio, audio in zip(
        watermarked_audios, attacked_audios, audios
    ):
        if has_changes:
            # Define the padding size
            padding_size = (0, max_size - watermarked_audio.shape[1])

            padded_wmd_audio = torch.nn.functional.pad(watermarked_audio, padding_size)
            padded_attacked_audio = torch.nn.functional.pad(
                attacked_audio, padding_size
            )
            padded_audio = torch.nn.functional.pad(audio, padding_size)
        else:
            padded_wmd_audio = watermarked_audio
            padded_attacked_audio = attacked_audio
            padded_audio = audio

        padded_wmd_audios.append(padded_wmd_audio)
        padded_attacked_audios.append(padded_attacked_audio)
        padded_audios.append(padded_audio)

    padded_wmd_audios = torch.stack(padded_wmd_audios)  # b c=1 t
    padded_attacked_audios = torch.stack(padded_attacked_audios)  # b c=1 t
    padded_audios = torch.stack(padded_audios)  # b c=1 t
    secret_messages = torch.cat(secret_messages, dim=0)  # b 16

    return (
        indices,
        padded_wmd_audios,
        padded_attacked_audios,
        padded_audios,
        secret_messages,
    )
