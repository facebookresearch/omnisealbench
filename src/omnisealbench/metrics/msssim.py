# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytorch_msssim
import torch

from omnisealbench.utils.common import batch_safe


def msssim(x, y, data_range=1.0):
    """
    Return MSSSIM
    Args:
        x: Image tensor with normalized values (≈ [0,1])
        y: Image tensor with normalized values (≈ [0,1]), ex: original image
    """
    return pytorch_msssim.ms_ssim(x, y, data_range=data_range, size_average=False)


@batch_safe
def compute_msssim_chunked(
    x: torch.Tensor,
    y: torch.Tensor,
    chunk_size: int = 8,
) -> torch.Tensor:
    """
    Compute MSSSIM scores in chunks to handle large tensors.

    Args:
        x (torch.Tensor): First set of images. Shape: (F, C, H, W).
        y (torch.Tensor): Second set of images. Shape: (F, C, H, W).
        chunk_size (int): Size of each chunk for processing.

    Returns:
        torch.Tensor: MSSSIM scores.
    """
    msssim_scores = []
    for i in range(0, x.size(0), chunk_size):
        x_chunk = x[i : i + chunk_size]  # Scale in [0, 1]
        y_chunk = y[i : i + chunk_size]  # Scale in [0, 1]

        msssim_score = msssim(x_chunk, y_chunk)
        msssim_scores.append(msssim_score.detach().reshape(-1))

    if not msssim_scores:
        return torch.tensor([], device=x.device)

    result = torch.cat(msssim_scores)
    return result.mean()
