# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
from functools import partial
from typing import List, Union

import numpy as np
import torch
from skimage.metrics import structural_similarity as compare_ssim

from omnisealbench.utils.common import batch_safe


@batch_safe
def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Return PSNR
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """

    batch_mode = True
    if x.ndim < 4:
        batch_mode = False
        x = x.unsqueeze(0)

    if y.ndim < 4:
        y = y.unsqueeze(0)

    delta = x - y
    delta = 255 * delta
    delta = delta.permute(0, 2, 3, 1)
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])  # BxCxHxW
    psnr = 20 * np.log10(255) - 10 * torch.log10(
        torch.mean(delta**2, dim=(1, 2, 3))
    )  # B

    if not batch_mode:
        psnr = psnr.squeeze(0)  # scalar
    return psnr


def compute_ssim_scores(
    x_img: Union[torch.Tensor, List[torch.Tensor]],
    y_img: Union[torch.Tensor, List[torch.Tensor]],
    num_processes: int = 4,
) -> Union[float, List[float]]:

    # Single example
    if isinstance(x_img, torch.Tensor) and x_img.ndim < 4:
        x_val = x_img.detach().cpu().numpy()
        y_val = y_img.detach().cpu().numpy()
        return compare_ssim(x_val, y_val, channel_axis=0, data_range=2)

    args_list = list(zip(x_img, y_img))
    args_list = [
        (x.squeeze().cpu().detach().numpy(), y.squeeze().cpu().detach().numpy())
        for x, y in args_list
    ]
    func = partial(compare_ssim, channel_axis=0, data_range=2)

    # Create a pool of processes
    if num_processes <= 1:
        return [func(*args) for args in args_list]

    with mp.Pool(processes=num_processes) as pool:
        # map() will dispatch each (x_sample, y_sample) pair to a worker
        # replace with map_async() to get a progress bar
        ssim_scores = pool.starmap(func, args_list)

    return ssim_scores
