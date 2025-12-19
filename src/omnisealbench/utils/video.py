# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torchvision
from torch import Tensor
from torchvision.utils import save_image


def save_video_frames(video_tensor: torch.Tensor, output_dir: str, prefix: str = "frame", fmt: str = "png") -> None:
    """
    Saves a video tensor (T, C, H, W) as individual image files.

    Args:
        video_tensor (Tensor): The video tensor with shape (T, C, H, W). Values should be in [0, 1].
        output_dir (str): Directory where the frames will be saved.
        prefix (str): Prefix for the saved frame filenames.
        fmt (str): Image format to save (e.g., "png", "jpg").
    """
    os.makedirs(output_dir, exist_ok=True)
    for t in range(video_tensor.shape[0]):
        # Each frame is already (C, H, W) and values should be in [0, 1]
        frame = video_tensor[t].clamp(0, 1)
        filename = os.path.join(output_dir, f"{prefix}_{t:04d}.{fmt}")
        save_image(frame, filename)


def save_vid(vid: Tensor, out_path: str, fps: int, crf: int = 23, video_codec: str = "libx264") -> None:
    """
    Saves a video tensor to a file.

    Args:
    vid (Tensor): The video tensor with shape (T, C, H, W) where
                  T is the number of frames,
                  C is the number of channels (should be 3),
                  H is the height,
                  W is the width.
    out_path (str): The output path for the saved video file.
    fps (int): Frames per second of the output video.
    crf (int): Constant Rate Factor for the output video (default is 23).

    Raises:
    AssertionError: If the input tensor does not have the correct dimensions or channel size.
    """
    # Assert the video tensor has the correct dimensions
    assert vid.dim() == 4, "Input video tensor must have 4 dimensions (T, C, H, W)"
    assert vid.size(1) == 3, "Video tensor's channel size must be 3"

    if video_codec == "pt":
        vid = vid * 255.0
        vid = vid.to(torch.uint8).cpu()
        torch.save(vid, out_path)
    else:
        vid = vid.clamp(0, 1)
        vid_uint8 = (vid.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().contiguous()

        fps_int = int(fps)
        crf_int = int(crf)

        if video_codec == "libx264":
            # Write the video file
            options = {"crf": str(crf_int)}
            torchvision.io.write_video(out_path, vid_uint8, fps=fps_int, video_codec="libx264", options=options)
        elif video_codec == "rawvideo":
            torchvision.io.write_video(out_path, vid_uint8, fps=fps_int, video_codec="rawvideo")
        elif video_codec == "ffv1":
            torchvision.io.write_video(out_path, vid_uint8, fps=fps_int, video_codec="ffv1")
        elif video_codec == "avc1":
            torchvision.io.write_video(out_path, vid_uint8, fps=fps_int, video_codec="avc1")
        elif video_codec == "VP90":
            torchvision.io.write_video(out_path, vid_uint8, fps=fps_int, video_codec="VP90")
        else:
            raise NotImplementedError(video_codec)
