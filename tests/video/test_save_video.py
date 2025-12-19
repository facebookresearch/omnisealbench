# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile

import pytest
import torch
import torchvision

from omnisealbench.utils.video import save_vid, save_video_frames

torch.manual_seed(2004)

# create dummy video data: 10 frames, 3 channels, 240x320 resolution
video_tensor = torch.rand(10, 3, 240, 320)


def load_and_look_for_differences(video_tensor: torch.Tensor, data_path: str):
    # Load the video back. read_video returns (video_tensor, audio_tensor, info)
    loaded_video, audio, info = torchvision.io.read_video(data_path, pts_unit="sec")
    # loaded_video shape will be (T, H, W, C)

    # Convert the original padded_data to match the loaded format (T, H, W, C) and uint8 type
    original_video = video_tensor.clamp(0, 1).permute(0, 2, 3, 1) * 255
    original_video = original_video.to(torch.uint8)

    # Compare the two tensors
    equal = torch.equal(original_video, loaded_video)
    print("Are the original and loaded videos equal?", equal)

    # Optionally, check the maximum absolute difference per element
    difference = (original_video.float() - loaded_video.float()).abs().max()
    print("Max difference between original and loaded video:", difference.item())


@pytest.mark.skip(reason="FFV1 codec is not available in some environments")
def test_mkv():
    with tempfile.NamedTemporaryFile(suffix=".mkv", delete=True) as f:
        data_path = f.name

        # Call the function with fps=24 and crf=0
        save_vid(video_tensor, data_path, fps=24, video_codec="ffv1", crf=0)

        assert os.path.exists(data_path)
        load_and_look_for_differences(video_tensor, data_path)


def test_mp4():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
        data_path = f.name

        # Call the function with fps=24 and crf=0
        save_vid(video_tensor, data_path, fps=24, video_codec="libx264", crf=0)

        assert os.path.exists(data_path)
        load_and_look_for_differences(video_tensor, data_path)


@pytest.mark.skip(reason="FFV1 codec is not available in some environments")
def test_rawvideo():
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=True) as f:
        data_path = f.name

        # Call the function with fps=24 and crf=0
        save_vid(video_tensor, data_path, fps=24, video_codec="rawvideo")

        assert os.path.exists(data_path)
        load_and_look_for_differences(video_tensor, data_path)


def test_frames():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert os.path.exists(tmpdir)
        save_video_frames(video_tensor, output_dir=tmpdir)

    assert not os.path.exists(tmpdir)
