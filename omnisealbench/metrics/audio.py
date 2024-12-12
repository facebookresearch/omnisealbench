# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import julius
import pesq  # type: ignore
import torch
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


def compute_snr(signal: torch.Tensor, watermarked_signal: torch.Tensor) -> float:
    # Compute SNR
    signal_power = torch.mean(signal**2, dim=(1, 2))
    noise_power = torch.mean((watermarked_signal - signal) ** 2, dim=(1, 2))
    snr = 10 * torch.log10(signal_power / noise_power)

    return snr


def tensor_pesq(y_pred: torch.Tensor, y: torch.Tensor, sr: int):
    # pesq returns error if no speech is detected, so we catch it
    if sr != 16000:
        y_pred = julius.resample_frac(y_pred, sr, 16000)
        y = julius.resample_frac(y, sr, 16000)
    P, n = 0, 0
    for ii in range(y_pred.size(0)):
        try:  # torchmetrics crashes when there is one error in the batch so doing it manually..
            P += pesq.pesq(16000, y[ii, 0].cpu().numpy(), y_pred[ii, 0].cpu().numpy())
            n += 1
        except (
            pesq.NoUtterancesError
        ):  # this error can append when the sample don't contain speech
            pass
    p = P / n if n != 0 else 0.0
    return p
