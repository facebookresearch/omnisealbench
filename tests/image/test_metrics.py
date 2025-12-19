# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def test_psnr():
    from omnisealbench.metrics.image import psnr

    # Create dummy tensors for x and y
    x_img = torch.rand(4, 3, 64, 64)  # Simulating a single image
    y_img = torch.rand(4, 3, 64, 64)

    # Compute PSNR scores
    psnr_score1 = psnr(x_img, y_img)

    # Compute PSNR wih batch_safe activated
    psnr_score2 = psnr(list(x_img), list(y_img))

    assert psnr_score1.ndim == 1
    torch.testing.assert_close(psnr_score1, psnr_score2)
    
    # Test a single image computation
    x_img_single = torch.rand(3, 64, 64)  # Simulating a single image
    y_img_single = torch.rand(3, 64, 64)
    psnr_score_single = psnr(x_img_single, y_img_single)
    assert isinstance(psnr_score_single, torch.Tensor)
    assert psnr_score_single.ndim == 0  # Should return a single value for a single image


def test_psnr_batch():
    from omnisealbench.metrics.image import psnr

    # Create dummy tensors for a batch of images
    sizes = [(3, 64, 64), (3, 128, 128), (3, 256, 256), (3, 512, 512)]
    x_img_batch = [torch.rand(size) for size in sizes]  #
    y_img_batch = [torch.rand(size) for size in sizes]

    # Compute PSNR scores
    psnr_scores = psnr(x_img_batch, y_img_batch)

    # Check if the output is a tensor
    assert isinstance(psnr_scores, torch.Tensor)
    assert psnr_scores.shape == (4,)  # Should be a single value for each image in the batch


def test_ssim():
    from omnisealbench.metrics.image import compute_ssim_scores
    
    x_img_batch = torch.rand(4, 3, 64, 64)  # Simulating a batch of 4 images
    y_img_batch = torch.rand(4, 3, 64, 64)
    
    ssim_scores1 = compute_ssim_scores(x_img_batch, y_img_batch, num_processes=2)
    ssim_scores2 = compute_ssim_scores(list(x_img_batch), list(y_img_batch), num_processes=2)

    assert isinstance(ssim_scores1, list)
    assert len(ssim_scores1) == 4  # Should match the number of images in the batch
    assert ssim_scores1 == ssim_scores2