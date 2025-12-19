# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch


def test_vmaf():
    from omnisealbench.metrics.video import vmaf_on_tensor

    # Create dummy tensors for a batch of videos
    sizes = [(10, 3, 64, 64), (15, 3, 128, 128), (4, 3, 256, 256), (20, 3, 512, 512)]
    x_vid_batch = [torch.rand(size) for size in sizes]  #
    y_vid_batch = [torch.rand(size) for size in sizes]

    kwargs = {"return_aux": False, "n_threads": 8}

    vmaf_score = vmaf_on_tensor(
        x_vid_batch, y_vid_batch,
        fps=24, codec='libx264', crf=23,
        **kwargs,
    )

    assert isinstance(vmaf_score, list)

