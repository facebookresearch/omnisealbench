# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
from skimage.metrics import structural_similarity as compare_ssim

from .image_effects import unnormalize_img, unnormalize_vqgan


def psnr(x, y, img_space="vqgan"):
    """
    Return PSNR
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    if img_space == "vqgan":
        delta = torch.clamp(unnormalize_vqgan(x), 0, 1) - torch.clamp(
            unnormalize_vqgan(y), 0, 1
        )
    elif img_space == "img":
        delta = torch.clamp(unnormalize_img(x), 0, 1) - torch.clamp(
            unnormalize_img(y), 0, 1
        )
    else:
        delta = x - y
    delta = 255 * delta
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])  # BxCxHxW
    psnr = 20 * np.log10(255) - 10 * torch.log10(
        torch.mean(delta**2, dim=(1, 2, 3))
    )  # B
    return psnr


def ssim(x, y, img_space="vqgan"):
    """
    Return SSIM
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    if img_space == "cpu":
        # needed when spawning processes
        x = x.transpose(1, 2, 0)
        y = y.transpose(1, 2, 0)
    else:
        if img_space == "vqgan":
            x = unnormalize_vqgan(x)
            y = unnormalize_vqgan(y)
        elif img_space == "img":
            x = unnormalize_img(x)
            y = unnormalize_img(y)
        else:
            x = x
            y = y

        x = x.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
        y = y.squeeze().cpu().detach().numpy().transpose(1, 2, 0)

    return compare_ssim(x, y, channel_axis=2, data_range=2)


def lpips(loss_fn_vgg, x, y, img_space="vqgan"):
    """
    Return LPIPS
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    if img_space == "vqgan":
        x = unnormalize_vqgan(x)
        y = unnormalize_vqgan(y)
    elif img_space == "img":
        x = unnormalize_img(x)
        y = unnormalize_img(y)

    # Ensure tensors are on the same device as the model
    device = next(loss_fn_vgg.parameters()).device
    x = x.to(device)
    y = y.to(device)

    # Compute LPIPS loss
    loss = loss_fn_vgg(x, y)

    # Return the loss as a scalar value
    return loss.item()
