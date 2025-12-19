# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import shutil
from pathlib import Path

import torch

from omnisealbench.utils.common import tensor_to_message


def test_local_image_dataset(tmp_path):
    from omnisealbench.data.image import read_local_image_dataset

    # Make up a fake local dataset from test image
    tahiti = Path(__file__).parent.parent.parent.joinpath("examples/img/img_1.png")
    
    shutil.copy(tahiti, tmp_path / "tahiti_512.png")
    shutil.copy(tahiti, tmp_path / "tahiti_512_2.png") 
    shutil.copy(tahiti, tmp_path / "tahiti_512_3.png")
    shutil.copy(tahiti, tmp_path / "tahiti_512_4.png")

    image_dataset = read_local_image_dataset(
        dataset_dir=tmp_path,
        batch_size=2,
    )

    for _, batch in image_dataset:
        assert len(batch) == 2, "Batch size should be 2"
        
        for img in batch:
            assert img.shape == (3, 512, 512), "Each image should be of shape (3, 512, 512)"


def test_local_image_dataset_with_message(tmp_path):
    from omnisealbench.data.image import read_local_image_dataset

    # Make up a fake local dataset from test image
    tahiti = Path(__file__).parent.parent.parent.joinpath("examples/img/img_1.png")
    
    shutil.copy(tahiti, tmp_path / "tahiti_512_1.png")
    shutil.copy(tahiti, tmp_path / "tahiti_512_2.png") 
    shutil.copy(tahiti, tmp_path / "tahiti_512_3.png")
    shutil.copy(tahiti, tmp_path / "tahiti_512_4.png")
    
    tmp_path.joinpath("watermarks").mkdir(exist_ok=True, parents=True)
    shutil.copy(tahiti, tmp_path / "watermarks/wm_512_1.png")
    shutil.copy(tahiti, tmp_path / "watermarks/wm_512_2.png") 
    shutil.copy(tahiti, tmp_path / "watermarks/wm_512_3.png")
    shutil.copy(tahiti, tmp_path / "watermarks/wm_512_4.png")

    msg_tensor = torch.randint(0, 2, (16,), dtype=torch.float32)  # Empty message tensor for watermarking
    msg = tensor_to_message(msg_tensor)  # Convert message to tensor format
    for i in range(1, 5):
        with open(tmp_path / f"watermark_{i}.txt", "w", encoding="utf-8") as f:
            f.write(msg)

    image_dataset = read_local_image_dataset(
        dataset_dir=tmp_path,
        image_pattern="*.*",
        watermarked_image_pattern="watermarks/wm_*",
        message_pattern="watermark_*.txt",
        batch_size=2,
    )

    for _, batch in image_dataset:
        assert len(batch["original"]) == 2, "Batch size should be 2"
        img = batch["original"]
        assert img[0].shape == (3, 512, 512), "Each image should be of shape (3, 512, 512)"
        wm_img = batch["watermarked"]
        assert wm_img[0].shape == (3, 512, 512), "Each image should be of shape (3, 512, 512)"
        m = batch["message"][0]
        torch.testing.assert_close(msg_tensor, m)


