# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torchaudio
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .configs import config
from .data.audio import AudioSample, get_dataset_implementation
from .models import AudioWatermarkGenerator


def check_dataset(
    cfg_path: Path, target_sr: int = 16000, max_length_in_seconds: int = 5
) -> Dict[str, Any]:
    if cfg_path.exists():
        with open(cfg_path, "r") as infile:
            cfg = json.load(infile)

        has_changes = False

        if cfg["max_length_in_seconds"] != max_length_in_seconds:
            has_changes = True

        if cfg["target_sr"] != target_sr:
            has_changes = True

        if not has_changes:
            return cfg

    return None


def run_watermarkgen(
    generator: AudioWatermarkGenerator,
    db_config: Dict[str, Any],
    message_sz: int = 16,
    target_sr: int = 16000,
    max_length_in_seconds: int = 5,
    override: bool = False,
) -> Dict[str, Any]:
    dataset_name = db_config["name"]
    db = get_dataset_implementation(db_config)

    watermarks_dir = os.path.join(
        config.OMNISEAL_WATERMARKING_CACHE, f"{dataset_name}_{generator.name}"
    )

    watermarks_dir = Path(watermarks_dir)
    watermarks_dir.mkdir(exist_ok=True)

    cfg_path = watermarks_dir / "cfg.json"
    if not override:
        cfg = check_dataset(cfg_path, target_sr, max_length_in_seconds)
        if cfg is None:
            override = True

    has_changes = False

    audio_files = glob.glob(f"{watermarks_dir}/audio_*.pt")
    sample_files = glob.glob(f"{watermarks_dir}/sample_*.pt")
    message_files = glob.glob(f"{watermarks_dir}/message_*.pt")

    audio_files_dict = {Path(k).name: k for k in audio_files}
    sample_files_dict = {Path(k).name: k for k in sample_files}
    message_files_dict = {Path(k).name: k for k in message_files}

    with Progress(
        TextColumn(
            f"{dataset_name} - {generator.name} - watermarking •"
            + "[progress.percentage]{task.percentage:>3.0f}%"
        ),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as p:
        indices = list(range(len(db)))
        for idx in p.track(indices):
            sample_fn = f"sample_{idx}.pt"
            audio_fn = f"audio_{idx}.pt"
            message_fn = f"message_{idx}.pt"

            sample_path = watermarks_dir / sample_fn
            waveform_path = watermarks_dir / audio_fn
            message_path = watermarks_dir / message_fn

            if (
                override
                or not sample_fn in sample_files_dict
                or not audio_fn in audio_files_dict
                or not message_fn in message_files_dict
            ):
                has_changes = True

                sample: AudioSample = db[idx]
                waveform: torch.tensor = sample["audio"]  # shape: (channels, samples)
                assert (
                    len(waveform.shape) == 2
                ), "waveform should be of shape: (channels, samples)"

                sr: int = sample["sample_rate"]

                if waveform.size(0) > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                max_length = max_length_in_seconds * sr

                # Truncate or pad the waveform to max_length
                if waveform.shape[0] > max_length:
                    waveform = waveform[:max_length]
                elif waveform.shape[0] < max_length:
                    padding = max_length - waveform.shape[0]
                    waveform = torch.nn.functional.pad(waveform, (0, padding))

                waveform = waveform.unsqueeze(0)  # b=1 c=1 t

                # resample to 16k
                if target_sr != sr:
                    waveform = torchaudio.transforms.Resample(
                        orig_freq=sr,
                        new_freq=target_sr,
                    )(waveform)

                # generate secret message
                secret_message = torch.randint(
                    0, 2, (waveform.shape[0], message_sz), dtype=torch.int32
                )  # b=1 16
                watermarked_audio = generator.generate_watermark_audio(
                    tensor=waveform,
                    sample_rate=target_sr,
                    secret_message=secret_message,
                )

                torch.save(watermarked_audio, sample_path)
                torch.save(waveform, waveform_path)
                torch.save(secret_message, message_path)

    if has_changes:
        cfg = {
            "num_samples": len(db),
            "max_length_in_seconds": max_length_in_seconds,
            "target_sr": target_sr,
        }

        with open(cfg_path, "w") as outfile:
            json.dump(cfg, outfile, indent=4)

    return {
        "num_samples": len(db),
        "dataset_path": watermarks_dir,
    }
