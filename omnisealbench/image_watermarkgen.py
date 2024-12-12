# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import PIL
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .configs import config
from .data.image import get_dataset_implementation
from .models import Watermark


def check_dataset(
    cfg_path: Path,
) -> Dict[str, Any]:
    if cfg_path.exists():
        with open(cfg_path, "r") as infile:
            cfg = json.load(infile)

        has_changes = False

        if not has_changes:
            return cfg

    return None


def run_image_watermarkgen(
    generator: Watermark,
    model_key: str,
    db_config: Dict[str, Any],
    message_sz: int = 32,
    override: bool = False,
    num_samples: int = None,
) -> Dict[str, Any]:

    dataset_name = db_config["name"]
    db = get_dataset_implementation(db_config)

    watermarks_dir = os.path.join(
        config.OMNISEAL_WATERMARKING_CACHE, f"{dataset_name}_{model_key}"
    )

    watermarks_dir = Path(watermarks_dir)
    watermarks_dir.mkdir(exist_ok=True)

    cfg_path = watermarks_dir / "cfg.json"
    if not override:
        cfg = check_dataset(cfg_path)
        if cfg is None:
            override = True

    has_changes = False

    image_files = glob.glob(f"{watermarks_dir}/image_*.png")
    watermark_files = glob.glob(f"{watermarks_dir}/watermark_*.png")
    message_files = glob.glob(f"{watermarks_dir}/message_*.npy")

    image_files_dict = {Path(k).name: k for k in image_files}
    watermark_files_dict = {Path(k).name: k for k in watermark_files}
    message_files_dict = {Path(k).name: k for k in message_files}

    indices = list(range(len(db)))
    if num_samples:
        assert num_samples <= len(db)
        indices = list(range(num_samples))

    with Progress(
        TextColumn(
            f"{dataset_name} - {model_key} - watermarking •"
            + "[progress.percentage]{task.percentage:>3.0f}%"
        ),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as p:
        for idx in p.track(indices):
            watermark_fn = f"watermark_{idx}.png"
            image_fn = f"image_{idx}.png"
            message_fn = f"message_{idx}.npy"

            watermark_path = watermarks_dir / watermark_fn
            image_path = watermarks_dir / image_fn
            message_path = watermarks_dir / message_fn

            if (
                override
                or not watermark_fn in watermark_files_dict
                or not image_fn in image_files_dict
                or not message_fn in message_files_dict
            ):
                has_changes = True

                pil_img: PIL.Image.Image = db[idx]

                # generate secret message
                key = np.random.randint(0, 2, message_sz)

                wm_image = generator.encode(pil_img, key)  # encode

                wm_image.save(watermark_path)
                pil_img.save(image_path)
                np.save(message_path, key)

    if has_changes:
        cfg = {
            "num_samples": len(indices),
        }

        with open(cfg_path, "w") as outfile:
            json.dump(cfg, outfile, indent=4)

    return {
        "num_samples": len(indices),
        "dataset_path": watermarks_dir,
    }
