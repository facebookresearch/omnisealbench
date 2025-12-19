# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, overload

import julius
import pesq  # type: ignore
import torch

from omnisealbench.interface import ContentT


def tensor_pesq(y_pred: torch.Tensor, y: torch.Tensor, sr: int) -> float:
    assert y_pred.ndim <= 2 and y.ndim <= 2, "y_pred and y must be of shape 1 x T"

    if sr != 16000:
        y_pred = julius.resample_frac(y_pred, sr, 16000)
        y = julius.resample_frac(y, sr, 16000)
    P, n = 0, 0

    y = y.squeeze(0).detach().cpu()
    y_pred = y_pred.squeeze(0).detach().cpu()

    return pesq.pesq(16000, y.numpy(), y_pred.numpy())


def tensor_pesq_batch_tensor(y_pred: torch.Tensor, y: torch.Tensor, sr: int, n_processor: int = 8) -> List[float]:
    # Compute PESQ (Perceptual Evaluation of Speech Quality) score for a batch of audio signals (shape B x 1 x T)
    if sr != 16000:
        y_pred = julius.resample_frac(y_pred, sr, 16000)
        y = julius.resample_frac(y, sr, 16000)

    y = y[:, 0].detach().cpu()
    y_pred = y_pred[:, 0].detach().cpu()

    pesq_values = pesq.pesq_batch(16000, y.numpy(), y_pred.numpy(), mode='wb', n_processor=n_processor)
    pesq_values = [v if isinstance(v, float) else 0.0 for v in pesq_values]

    return pesq_values


def tensor_pesq_batch_list(y_pred: List[torch.Tensor], y: List[torch.Tensor], sr: int, n_processor: int = 8) -> List[float]:
    if n_processor <= 1:
        return [tensor_pesq(y_pred, y, sr) for y_pred, y in zip(y_pred, y)]

    arg_lst = [(y_pred, y, sr) for y_pred, y in zip(y_pred, y)]
    with torch.multiprocessing.Pool(processes=n_processor) as pool:
        pesq_scores = pool.starmap(tensor_pesq, arg_lst)

    return pesq_scores


@overload
def tensor_pesq_batch(y_pred: torch.Tensor, y: torch.Tensor, sr: int, n_processor: int = 8) -> List[float]:
    ...

@overload
def tensor_pesq_batch(y_pred: List[torch.Tensor], y: List[torch.Tensor], sr: int, n_processor: int = 8) -> List[float]:
    ...
    
def tensor_pesq_batch(y_pred: ContentT, y: ContentT, sr: int, n_processor: int = 8) -> List[float]:
    if isinstance(y_pred, torch.Tensor) and isinstance(y, torch.Tensor):
        return tensor_pesq_batch_tensor(y_pred, y, sr, n_processor)
    elif isinstance(y_pred, list) and isinstance(y, list):
        return tensor_pesq_batch_list(y_pred, y, sr, n_processor)
    else:
        raise TypeError("y_pred and y must be either both torch.Tensor or both List[torch.Tensor]")
