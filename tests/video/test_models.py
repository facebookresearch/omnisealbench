# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import Literal

import pytest
import torch

from omnisealbench.utils.distributed_logger import log_mem
from omnisealbench.utils.model_card import get_model


def test_load_rivagan(device):
    from omnisealbench.models.rivagan import RivaGANWM

    model = get_model("rivagan", device=device)
    assert isinstance(model, RivaGANWM)

    if os.getenv("GITHUB_ACTIONS") == "true":
        return
    
    # Only run the following inference locally
    msg = torch.randint(0, 2, (32,), dtype=torch.float32)
    videos = torch.randn(4, 10, 3, 128, 128)
    
    log_mem("Start RivaGAN watermarking inference")

    wm_videos = model.generate_watermark(videos, message=msg)
    extracted = model.detect_watermark(wm_videos)
    assert isinstance(extracted, dict)
    assert "detection_bits" in extracted
    assert "message" in extracted
    assert isinstance(extracted["detection_bits"], torch.Tensor)
    assert isinstance(extracted["message"], torch.Tensor)
    assert extracted["detection_bits"].shape == (4, 16)
    assert extracted["message"].shape == (4, 32 - 16)
    log_mem("End RivaGAN watermarking inference")
    # torch.testing.assert_close(extracted["message"].float(), msg[:,16:].repeat(4, 1), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(("model_card_or_path", "nbits", "scaling_w"), [("videoseal_0.0", 96, 1.0), ("videoseal_1.0", 256, 0.2)])
def test_load_inline_videoseal(model_card_or_path: str, nbits: int, scaling_w: float, device: str):
    from omnisealbench.models.videoseal import VideosealWM

    detection_bits = 16
    inline_model = get_model(        
        model_class_or_function="omnisealbench.models.videoseal.VideosealWM",
        model_card_or_path=model_card_or_path,
        nbits=nbits,
        detection_bits=detection_bits,
        scaling_w=scaling_w,
        videoseal_chunk_size=32,
        videoseal_step_size=4,
        attenuation="jnd_1_1",
        attenuation_config="src/omnisealbench/models/videoseal_src/configs/attenuation.yaml",
        video_aggregation="avg",
        videoseal_mode="repeat",
        lowres_attenuation=False,
        img_size_proc=256,
        device=device,
    )
    assert isinstance(inline_model, VideosealWM)

    # Check if the model support batch processing
    msg = torch.randint(0, 2, (nbits,), dtype=torch.float32)
    videos = torch.randn(4, 10, 3, 128, 128)

    wm_videos = inline_model.generate_watermark(videos, message=msg)
    extracted = inline_model.detect_watermark(wm_videos)
    assert isinstance(extracted, dict)
    assert "detection_bits" in extracted
    assert "message" in extracted
    assert isinstance(extracted["detection_bits"], torch.Tensor)
    assert isinstance(extracted["message"], torch.Tensor)
    assert extracted["detection_bits"].shape == (4, detection_bits)
    assert extracted["message"].shape == (4, nbits - detection_bits)
    # torch.testing.assert_close(extracted["message"].float(), msg[:,16:].repeat(4, 1), rtol=1e-3, atol=1e-3)    


@pytest.mark.parametrize("model_class_or_function", ["videoseal_0.0", "videoseal_1.0"])
def test_load_videoseal(model_class_or_function: Literal['videoseal_0.0'] | Literal['videoseal_1.0'], device: str):
    from omnisealbench.models.videoseal import VideosealWM

    model = get_model(model_class_or_function, device=device)
    assert isinstance(model, VideosealWM)

    nbits = model.nbits
    detection_bits = model.detection_bits

    # Check if the model support batch processing
    msg = torch.randint(0, 2, (nbits,), dtype=torch.float32)
    videos = torch.randn(4, 10, 3, 128, 128)

    wm_videos = model.generate_watermark(videos, message=msg)
    extracted = model.detect_watermark(wm_videos)
    assert isinstance(extracted, dict)
    assert "detection_bits" in extracted
    assert "message" in extracted
    assert isinstance(extracted["detection_bits"], torch.Tensor)
    assert isinstance(extracted["message"], torch.Tensor)
    assert extracted["detection_bits"].shape == (4, detection_bits)
    assert extracted["message"].shape == (4, nbits - detection_bits)
    # torch.testing.assert_close(extracted["message"].float(), msg[:,16:].repeat(4, 1), rtol=1e-3, atol=1e-3) 
