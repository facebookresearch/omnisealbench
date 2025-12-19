# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def test_local_video_dataset(tmp_path):
    from omnisealbench.data.video import read_local_video_dataset
    from omnisealbench.utils.video import save_vid

    # Create a fake video dataset
    (tmp_path / "videos").mkdir(exist_ok=True, parents=True)

    video_tensor = [
        torch.rand(10, 3, 240, 320),
        torch.rand(15, 3, 128, 254),
    ] * 4
    
    for i, fake_video in enumerate(video_tensor):
        # Save the video tensor to a file
        # This is a placeholder for actual video saving logic
        # You can use save_vid or any other method to save the video tensor
        video_path = tmp_path / "videos" / f"video_{i:02d}.mp4"
        # Use libx264 (even spatial sizes above) for compatibility with torchvision write_video.
        save_vid(fake_video, video_path, fps=24, crf=0, video_codec="libx264")

    vid_dataset = read_local_video_dataset(
        dataset_dir=tmp_path / "videos",
        video_pattern="video_*.mp4",
        output_resolution=256,
        max_frames=10,
        batch_size=2,
    )
    
    for _, batch in vid_dataset:
        assert isinstance(batch, list), "Batch should be a dictionary of tensors"
        assert len(batch) == 2, "Batch size should be 2"
        assert isinstance(batch[0], torch.Tensor), "Each video sample should be a tensor"
        assert batch[0].shape[0] == 10, "Video frames should have 10 frames per sample"
        assert batch[0].shape[1] == 3, "Video frames should have 3 channels (RGB)"
        assert batch[0].shape[2] == 256, "Video frames should have a height of 256 pixels"
