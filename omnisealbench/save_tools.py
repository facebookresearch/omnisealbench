# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchaudio
from scipy.io import wavfile

from torchvision import transforms
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 3)


def save_image_examples(
    examples_dir: Path,
    idx: int,
    img: torch.Tensor,
    wmd_img: torch.Tensor,
    attacked_img: torch.Tensor,
    attacked_wmd_img: torch.Tensor,
    untransform: transforms.Compose,
) -> None:

    pil_img = untransform(img)
    pil_img.save(examples_dir / f"img_{idx:05d}.png")

    pil_wmd_img = untransform(wmd_img)
    pil_wmd_img.save(examples_dir / f"wmd_img_{idx:05d}.png")

    pil_attacked_img = untransform(attacked_img)
    pil_attacked_img.save(examples_dir / f"attacked_img_{idx:05d}.png")

    pil_attacked_wmd_img = untransform(attacked_wmd_img)
    pil_attacked_wmd_img.save(examples_dir / f"attacked_wmd_img_{idx:05d}.png")


def plot_waveform_and_specgram(waveform, sample_rate, title, fig_path):
    waveform = waveform.squeeze().detach().cpu().numpy()

    num_frames = waveform.shape[-1]
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(time_axis, waveform, linewidth=1)
    ax1.grid(True)
    ax2.specgram(waveform, Fs=sample_rate)

    figure.suptitle(f"{title} - Waveform and specgram")
    plt.savefig(fig_path)


def save_audio_examples(
    examples_dir: Path,
    idx: int,
    sample_rate: int,
    audio: torch.Tensor,
    wmd_audio: torch.Tensor,
    attacked_audio: torch.Tensor,
    attacked_wmd_audio: torch.Tensor,
    use_torch_audio: bool = False,
) -> None:
    plot_waveform_and_specgram(
        audio, sample_rate, "Original audio", examples_dir / f"audio_{idx:05d}.png"
    )

    plot_waveform_and_specgram(
        wmd_audio,
        sample_rate,
        "Watermarked audio",
        examples_dir / f"wmd_audio_{idx:05d}.png",
    )
    plot_waveform_and_specgram(
        attacked_audio,
        sample_rate,
        "Attacked original audio",
        examples_dir / f"attacked_audio_{idx:05d}.png",
    )
    plot_waveform_and_specgram(
        attacked_wmd_audio,
        sample_rate,
        "Attacked watermarked audio",
        examples_dir / f"attacked_wmd_audio_{idx:05d}.png",
    )

    if use_torch_audio:
        torchaudio.save(
            examples_dir / f"audio_{idx:05d}.wav",
            audio.squeeze().detach().cpu(),
            sample_rate=sample_rate,
        )
        torchaudio.save(
            examples_dir / f"wmd_audio_{idx:05d}.wav",
            wmd_audio.squeeze().detach().cpu(),
            sample_rate=sample_rate,
        )
        torchaudio.save(
            examples_dir / f"attacked_audio_{idx:05d}.wav",
            attacked_audio.squeeze().detach().cpu(),
            sample_rate=sample_rate,
        )
        torchaudio.save(
            examples_dir / f"attacked_wmd_audio_{idx:05d}.wav",
            attacked_wmd_audio.squeeze().detach().cpu(),
            sample_rate=sample_rate,
        )
    else:
        wavfile.write(
            examples_dir / f"audio_{idx:05d}.wav",
            sample_rate,
            audio.squeeze().detach().cpu().numpy(),
        )
        wavfile.write(
            examples_dir / f"wmd_audio_{idx:05d}.wav",
            sample_rate,
            wmd_audio.squeeze().detach().cpu().numpy(),
        )
        wavfile.write(
            examples_dir / f"attacked_audio_{idx:05d}.wav",
            sample_rate,
            attacked_audio.squeeze().detach().cpu().numpy(),
        )
        wavfile.write(
            examples_dir / f"attacked_wmd_audio_{idx:05d}.wav",
            sample_rate,
            attacked_wmd_audio.squeeze().detach().cpu().numpy(),
        )
