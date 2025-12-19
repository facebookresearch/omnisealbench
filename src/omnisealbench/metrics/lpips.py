# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import lpips as lpips_lib
import torch
from torchvision import transforms

from omnisealbench.utils.cache import global_lru_cache
from omnisealbench.utils.common import batch_safe, get_tensor_device


@global_lru_cache(maxsize=64)
def lpips(network: str, device: str = "cpu", verbose: bool = False) -> lpips_lib.LPIPS:
    lpips_ = lpips_lib.LPIPS(net=network, verbose=verbose)
    lpips_.eval()

    lpips_.to(device)

    return lpips_


normalize_vqgan = transforms.Normalize(
    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
)  # Normalize (x - 0.5) / 0.5
unnormalize_vqgan = transforms.Normalize(
    mean=[-1, -1, -1], std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
)  # Unnormalize (x * 0.5) + 0.5
normalize_img = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)  # Normalize (x - mean) / std
unnormalize_img = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)  # Unnormalize (x * std) + mean


@batch_safe
def compute_lpips(
    x: torch.Tensor,
    y: torch.Tensor,
    net: str = "vgg",
    device: Optional[str] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Compute LPIPS scores between two sets of images.

    Args:
        images1 (torch.Tensor): First set of images.
        images2 (torch.Tensor): Second set of images.
        network (str): The LPIPS network to use (e.g., 'vgg').
        device (str): Device to run the computation on ('cpu' or 'cuda').
            If None, uses the default device.

    Returns:
        torch.Tensor: LPIPS scores.
    """

    if device is None:
        device = get_tensor_device(x)
    else:
        x = x.to(device)  # CxHxW or BxCxHxW
        y = y.to(device)

    lpips_loss = lpips(net, device, verbose=verbose)

    return lpips_loss(x, y).squeeze() # BxS or (B,) or scalar if B=1


@batch_safe
def compute_lpips_chunked(
    x: torch.Tensor,
    y: torch.Tensor,
    net: str = "vgg",
    device: Optional[str] = None,
    chunk_size: int = 8,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Compute LPIPS scores in chunks to handle large tensors.

    Args:
        x (torch.Tensor): First set of images.
        y (torch.Tensor): Second set of images.
        net (str): The LPIPS network to use (e.g., 'vgg').
        device (str): Device to run the computation on ('cpu' or 'cuda').
        chunk_size (int): Size of each chunk for processing.

    Returns:
        torch.Tensor: LPIPS scores.
    """

    lpips_scores = []
    for i in range(0, x.size(0), chunk_size):
        x_chunk = x[i : i + chunk_size]
        y_chunk = y[i : i + chunk_size]

        x_chunk = 2 * x_chunk - 1  # Scale to [-1, 1]
        y_chunk = 2 * y_chunk - 1  # Scale to [-1, 1]

        lpips_ = compute_lpips(x_chunk, y_chunk, net=net, device=device, verbose=verbose)
        lpips_scores.append(lpips_.detach().reshape(-1))

    if not lpips_scores:
        return torch.tensor([], device=x.device)

    result = torch.cat(lpips_scores)
    return result.mean()
