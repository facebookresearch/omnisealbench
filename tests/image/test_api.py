# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import shutil
from pathlib import Path

from omnisealbench.tasks._detection import ATTACK_RESULT_FILENAME
import pytest
import torch

from omnisealbench import AttackConfig, get_model, task
from omnisealbench.data.base import InputBatch

image_models = {
    # "dct_dwt": {
    #     "nbits": 48,
    #     "detection_bits": 16,
    # },
    "invismark": {
        "nbits": 100,
        "detection_bits": 16,
    },
    "mbrs": {
        "nbits": 256,
        "detection_bits": 16,
    },
    "trustmark": {
        "nbits": 40,
        "detection_bits": 0,  
    },
    # "hidden": {
    #     "nbits": 48,
    #     "detection_bits": 16,
    # },
    "fnns": {
        "nbits": 48,
        "detection_bits": 16,
    },
    "ssl": {
        "nbits": 48,
        "detection_bits": 16,
    },
    "wam": {
        "nbits": 32,
        "detection_bits": 16,
    }
}


def build_test_image_dataset(tmp_path: Path):
    # Make up a fake local dataset from test image
    tahiti = Path(__file__).parent.parent.parent.joinpath("examples/img/img_1.png")

    tmp_path.mkdir(exist_ok=True, parents=True)

    shutil.copy(tahiti, tmp_path / "img_1_1.png")
    shutil.copy(tahiti, tmp_path / "img_1_2.png")
    shutil.copy(tahiti, tmp_path / "img_1_3.png")
    shutil.copy(tahiti, tmp_path / "img_1_4.png")
    shutil.copy(tahiti, tmp_path / "other_random_image_1.png")
    shutil.copy(tahiti, tmp_path / "other_random_image_2.png")
    shutil.copy(tahiti, tmp_path / "other_random_image_3.png")
    shutil.copy(tahiti, tmp_path / "other_random_image_4.png")


@pytest.mark.parametrize("model_name", image_models.keys())
@pytest.mark.parametrize("result_dir", [None, "results_image"])
@pytest.mark.parametrize("overwrite", [True, False])
def test_image_tasks_end_to_end(tmp_path, model_name, result_dir, overwrite, device):
    num_samples = 4

    dataset_dir = tmp_path / "input"
    wm_dir = tmp_path / "wm_gen"
    
    if result_dir is not None:
        result_dir = str(tmp_path / result_dir)

    nbits = image_models[model_name]["nbits"]
    detection_bits = image_models[model_name]["detection_bits"]

    build_test_image_dataset(dataset_dir)
    attacks = [
        AttackConfig(name="jpeg", data_type="image", args={"quality": 50}),
        AttackConfig(name="gaussian_blur", data_type="image", args={"kernel_size": 13}),
    ]

    # First we generate the watermarked
    e2e = task(
        "default",
        modality="image",
        dataset_dir=str(dataset_dir),
        seed=42,
        cache_dir=str(wm_dir),
        result_dir=result_dir,
        overwrite=overwrite,
        batch_size=2,
        num_workers=2,
        num_samples=num_samples,
        attacks=attacks,
        detection_bits=detection_bits,
        metrics=["ssim", "lpips"],
        how_to_generate_message="per_dataset",
        ids_to_save="all",
    )

    model = get_model(model_name, device=device, nbits=nbits)
    metrics, _ = e2e(model)  # type: ignore

    assert "ssim" in metrics, "SSIM metric should be computed"
    assert "lpips" in metrics, "LPIPS metric should be computed"
    saved_messages = glob.glob(os.path.join(wm_dir, "watermarks", "*.txt"))
    assert len(saved_messages) == num_samples, f"There should be {num_samples} messages saved, get {len(saved_messages)}"
    
    if result_dir:
        attack_dirs = [d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))]
        assert len(attack_dirs) == len(e2e.make_sub_tasks()), "The number of attack directories should match the number of sub-tasks in the pipeline"

        assert "metrics.json" in os.listdir(result_dir), "There should be a metrics file in the detection directory"
        for attack_dir in attack_dirs:
            assert "metrics.json" in os.listdir(Path(result_dir) / attack_dir), f"There should be a metrics file in the attack subdirectory {attack_dir}"
            assert ATTACK_RESULT_FILENAME in os.listdir(Path(result_dir) / attack_dir), f"There should be an attack result file in attack subdirectory {attack_dir}"
            assert "data" in os.listdir(Path(result_dir) / attack_dir), f"There should be a data sub-directory in attack subdirectory {attack_dir}"
        assert "report.csv" in os.listdir(result_dir), "There should be a report file in the detection directory"


@pytest.mark.parametrize("model_name", image_models.keys())
def test_image_wm_pipeline_custom(model_name, device):
    nbits = image_models[model_name]["nbits"]

    fake_images = [
        InputBatch(torch.tensor(range(i, i + 4)), torch.randn(4, 3, 512, 512))
        for i in range(0, 32, 4)
    ]

    on_the_fly_wm = task(
        "generation",
        modality="image",
        seed=42,
        batch_size=2,
        num_workers=2,
        metrics=["ssim", "lpips"],
    )

    model = get_model(model_name, as_type="generator", device=device, nbits=nbits)

    metrics, _ = on_the_fly_wm(
        model,
        batches=fake_images,
    )  # type: ignore

    assert "ssim" in metrics, "SSIM metric should be computed"
    assert "lpips" in metrics, "LPIPS metric should be computed"
