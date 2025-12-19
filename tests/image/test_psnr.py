# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
import unittest

import torch

from omnisealbench.metrics.video import (compute_psnr_chunked, original_psnr,
                                         psnr)


# Define the test cases.
class TestPSNRImplementations(unittest.TestCase):
    def setUp(self):
        # Use a fixed seed for reproducibility.
        torch.manual_seed(42)
        # Create a batch of 16 video frames (or images) with shape [16, 3, 64, 64].
        self.batch = torch.rand(16, 3, 64, 64)
        # Create a small noise tensor to simulate slight differences.
        self.noise = torch.rand(16, 3, 64, 64) * 0.1
        self.tol = 1e-4

    def test_psnr_video_mode(self):
        # Test overall video PSNR.
        x = self.batch
        y = self.batch + self.noise
        psnr_orig = psnr(x, y, is_video=True)
        psnr_orig_v2 = original_psnr(x, y, is_video=True).mean().item()
        psnr_chunk = compute_psnr_chunked(x, y, is_video=True, chunk_size=4)
        self.assertAlmostEqual(psnr_orig, psnr_chunk, delta=self.tol)
        self.assertAlmostEqual(psnr_orig, psnr_orig_v2, delta=self.tol)

    def test_psnr_per_image_mode(self):
        # Test per-image PSNR.
        x = self.batch
        y = self.batch + self.noise
        psnr_orig_tensor = psnr(x, y, is_video=False)
        psnr_chunk_tensor = compute_psnr_chunked(x, y, is_video=False, chunk_size=4)
        self.assertEqual(psnr_orig_tensor.shape, psnr_chunk_tensor.shape)
        # Compare each per-image PSNR value.
        for orig_val, chunk_val in zip(psnr_orig_tensor, psnr_chunk_tensor):
            self.assertAlmostEqual(orig_val.item(), chunk_val.item(), delta=self.tol)

    def test_psnr_zero_error(self):
        # When there is no error (identical inputs), PSNR should be infinite.
        x = self.batch
        y = self.batch
        psnr_orig = psnr(x, y, is_video=True)
        psnr_chunk = compute_psnr_chunked(x, y, is_video=True, chunk_size=4)
        self.assertTrue(math.isinf(psnr_orig))
        self.assertTrue(math.isinf(psnr_chunk))
        # Also test per-image mode.
        psnr_orig_tensor = psnr(x, y, is_video=False)
        psnr_chunk_tensor = compute_psnr_chunked(x, y, is_video=False, chunk_size=4)
        self.assertTrue(torch.all(torch.isinf(psnr_orig_tensor)))
        self.assertTrue(torch.all(torch.isinf(psnr_chunk_tensor)))