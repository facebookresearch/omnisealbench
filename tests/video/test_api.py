# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch

from omnisealbench import get_model, task
from omnisealbench.attacks.helper import AttackConfig
from omnisealbench.utils.video import save_vid

video_models = {
    # "rivagan": {
    #     "nbits": 32,
    #     "seq_len": 3,
    #     "detection_bits": 16,
    #     "keep_last_frames": True,
    # },
    "videoseal_0.0": {
        "nbits": 96,
        "detection_bits": 16,
    },
    "videoseal_1.0": {
        "nbits": 256,
        "detection_bits": 16,
    }
}



@pytest.mark.parametrize("model_name", video_models.keys())
def test_video_end_to_end(tmp_path, model_name, device):
    num_samples = 1  # Limit to 1 video for testing

    # Create a fake video dataset
    (tmp_path / "videos").mkdir(exist_ok=True, parents=True)

    video_tensor = [
        torch.rand(5, 3, 240, 320),
        torch.rand(5, 3, 128, 254),
    ] * 2

    for i, fake_video in enumerate(video_tensor):
        # Save the video tensor to a file
        # This is a placeholder for actual video saving logic
        # You can use save_vid or any other method to save the video tensor
        video_path = tmp_path / "videos" / f"video_{i:02d}.mp4"
        save_vid(fake_video, video_path, fps=24, crf=0, video_codec="libx264")

    model_cfg = video_models[model_name]
    dataset_dir = tmp_path / "videos"
    detection_bits = model_cfg["detection_bits"]

    e2e = task(
        "default",
        modality="video",
        dataset_dir=str(dataset_dir),
        seed=42,
        result_dir=tmp_path / "wm_detect",
        cache_dir=tmp_path / "wm_gen",
        overwrite=True,
        output_resolution=256,
        max_frames=5,
        fps=24,
        crf=0,
        batch_size=1,
        num_workers=2,
        num_samples=num_samples,
        metrics=["psnr", "lpips", "msssim", "ssim"],
        attacks=[AttackConfig(name="H264", data_type="video", args={"crf": 23})],
        detection_bits=detection_bits,
        metrics_device="cpu",
        quality_metrics_chunk_size=1,  # Use chunking for large videos
    )
    model = get_model(model_name, device=device, **model_cfg)

    metrics, _ = e2e(model)  # type: ignore
    attack_dirs = [d for d in os.listdir(tmp_path / "wm_detect") if (tmp_path / "wm_detect" / d).is_dir()]

    assert len(attack_dirs) == len(e2e.make_sub_tasks()), "The number of attack directories should match the number of sub-tasks in the pipeline"

    assert "psnr" in metrics, "SSIM metric should be computed"
    assert "lpips" in metrics, "LPIPS metric should be computed"
    assert "msssim" in metrics, "msssim metric should be computed"
    assert "ssim" in metrics, "ssim metric should be computed"
