# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from omnisealbench.metrics.pesq import tensor_pesq_batch
from omnisealbench.utils.common import batch_safe


@batch_safe
def compute_snr(signal: torch.Tensor, watermarked_signal: torch.Tensor) -> torch.Tensor:
    # batch mode
    if signal.ndim == 3:
        signal_power = torch.mean(signal**2, dim=(1, 2))
        noise_power = torch.mean((watermarked_signal - signal) ** 2, dim=(1, 2))
    # single sample mode
    else:
        signal_power = torch.mean(signal**2, dim=(1))
        noise_power = torch.mean((watermarked_signal - signal) ** 2, dim=(1))

    snr = 10 * torch.log10(signal_power / noise_power)

    return snr


@batch_safe
def stoi(y_pred: torch.Tensor, y: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """
    Compute the STOI (Short-Time Objective Intelligibility) score between two audio signals.
    """
    stoi_func = ShortTimeObjectiveIntelligibility(fs=sr).to(y_pred.device, dtype=torch.float32)

    # single sample mode
    if y_pred.ndim == 2:
        return stoi_func(y_pred, y)
    else:
        stoi_scores = [stoi_func(y_pred[i], y[i]) for i in range(y_pred.size(0))]
        return torch.stack(stoi_scores)


@batch_safe
def si_snr(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the Scale-Invariant Signal-to-Noise Ratio (SI-SNR) between two audio signals.
    """
    sisnr_func = ScaleInvariantSignalNoiseRatio().to(y_pred.device)

    if y_pred.ndim == 2:
        # single sample mode
        return sisnr_func(y_pred, y)

    # batch mode
    else:
        sisnr = [sisnr_func(y_pred[i], y[i]) for i in range(y_pred.size(0))]
        return torch.stack(sisnr)
