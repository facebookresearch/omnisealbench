# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Literal

import pytest
import torch

from omnisealbench.config import OMNISEAL_CHECKPOINTS_DIR
from omnisealbench.utils.cluster import ClusterType, guess_cluster
from omnisealbench.utils.model_card import get_model


def test_load_trustmark(device):
    from omnisealbench.models.trustmark import TrustMarkWM

    model = get_model("omnisealbench.models.trustmark.TrustMarkWM", model_type="Q", device=device)
    assert isinstance(model, TrustMarkWM)


def test_load_dct_dwt(device):
    from omnisealbench.models.dctdwt import DCTDWTDetector, DCTDWTGenerator

    generator = get_model("dct_dwt", as_type="generator", device=device)
    assert isinstance(generator, DCTDWTGenerator)
    msg = torch.randint(0, 2, (30,), dtype=torch.float32)
    imgs = [torch.randn(3, 256, 256) for _ in range(4)]
    imgs_w = generator.generate_watermark(imgs, message=msg)
    assert isinstance(imgs_w, list)
    assert len(imgs_w) == 4
    assert all(isinstance(img, torch.Tensor) for img in imgs_w)

    detector = get_model("dct_dwt", as_type="detector", wm_type="bits", device=device)
    assert isinstance(detector, DCTDWTDetector)

    extracted = detector.detect_watermark(imgs_w)
    assert isinstance(extracted, dict)
    assert "detection_bits" in extracted
    assert "message" in extracted
    assert isinstance(extracted["detection_bits"], torch.Tensor)
    assert isinstance(extracted["message"], torch.Tensor)
    assert extracted["detection_bits"].shape == (4, 16)
    assert extracted["message"].shape == (4, 32)


@pytest.mark.skipif(
    guess_cluster() != ClusterType.FAIR,
    reason="This test is only for FAIR cluster",
)
def test_load_cin(device):
    from omnisealbench.models.cin import CINWm

    model = get_model("cin", device=device)
    assert isinstance(model, CINWm)

    msg = torch.randint(0, 2, (30,), dtype=torch.float32, device=device)
    imgs = [torch.randn(3, 128, 128, device=device) for _ in range(4)]
    imgs_w = model.generate_watermark(imgs, message=msg)
    assert isinstance(imgs_w, list)
    assert len(imgs_w) == 4
    assert all(isinstance(img, torch.Tensor) for img in imgs_w)

    extracted = model.detect_watermark(imgs_w)
    assert isinstance(extracted, dict)
    assert "detection_bits" in extracted
    assert "message" in extracted
    assert isinstance(extracted["detection_bits"], torch.Tensor)
    assert isinstance(extracted["message"], torch.Tensor)
    assert extracted["detection_bits"].shape == (4, 16)
    assert extracted["message"].shape == (4, 14)


@pytest.mark.skipif(
    guess_cluster() != ClusterType.FAIR,
    reason="This test is only for FAIR cluster",
)
def test_load_invismark(device):
    from omnisealbench.models.invismark import InvisWatermark
    
    omniseal_ckpt = os.getenv("OMNISEAL_CHECKPOINT", OMNISEAL_CHECKPOINTS_DIR)
    model = get_model("invismark", model_card_or_path=f"{omniseal_ckpt}/invismark/paper.ckpt", device=device)
    assert isinstance(model, InvisWatermark)

    # Check if the model support batch processing
    msg = torch.randint(0, 2, (100,), dtype=torch.float32)
    imgs = [torch.randn(3, 128, 128) for _ in range(4)]  # 4 images of size (3, 128, 128)

    imgs_w = model.generate_watermark(imgs, message=msg)
    assert isinstance(imgs_w, list)
    assert len(imgs_w) == 4
    assert all(isinstance(img, torch.Tensor) for img in imgs_w)
    
    extracted = model.detect_watermark(imgs_w)
    assert isinstance(extracted, dict)
    assert "detection_bits" in extracted
    assert "message" in extracted
    assert isinstance(extracted["detection_bits"], torch.Tensor)
    assert isinstance(extracted["message"], torch.Tensor)
    assert extracted["detection_bits"].shape == (4, 16)
    assert extracted["message"].shape == (4, 100 - 16)


def test_load_mbrs(device):
    from omnisealbench.models.mbrs import MBRSWM

    model = get_model("mbrs", nbits=256, device=device)
    assert isinstance(model, MBRSWM)

    msg = torch.randint(0, 2, (256,), dtype=torch.float32, device=device)
    imgs = [torch.randn(3, 128, 128, device=device) for _ in range(4)]
    imgs_w = model.generate_watermark(imgs, message=msg)
    assert isinstance(imgs_w, list)
    assert len(imgs_w) == 4
    assert all(isinstance(img, torch.Tensor) for img in imgs_w)


    extracted = model.detect_watermark(imgs_w)
    assert isinstance(extracted, dict)
    assert "detection_bits" in extracted
    assert "message" in extracted
    assert isinstance(extracted["detection_bits"], torch.Tensor)
    assert isinstance(extracted["message"], torch.Tensor)
    assert extracted["detection_bits"].shape == (4, 16)
    assert extracted["message"].shape == (4, 256 - 16)


def test_load_hidden(device):
    from omnisealbench.models.hidden import (HiDDeNWatermarkDetector,
                                             HiDDeNWatermarkGenerator)

    generator = get_model("hidden", as_type="generator", lambda_w=0.45, device=device)
    assert isinstance(generator, HiDDeNWatermarkGenerator)

    msg = torch.randint(0, 2, (48,), dtype=torch.float32, device=device)
    imgs = [torch.randn(3, 128, 128, device=device) for _ in range(4)]
    imgs_w = generator.generate_watermark(imgs, message=msg)
    assert isinstance(imgs_w, list)
    assert len(imgs_w) == 4
    assert all(isinstance(img, torch.Tensor) for img in imgs_w)

    detector = get_model("hidden", as_type="detector", detection_bits=16, device=device)
    assert isinstance(detector, HiDDeNWatermarkDetector)

    extracted = detector.detect_watermark(imgs_w)
    assert isinstance(extracted, dict)
    assert "detection_bits" in extracted
    assert "message" in extracted
    assert isinstance(extracted["detection_bits"], torch.Tensor)
    assert isinstance(extracted["message"], torch.Tensor)
    assert extracted["detection_bits"].shape == (4, 16)
    assert extracted["message"].shape == (4, 48 - 16)


def test_load_fnns(device):
    from omnisealbench.models.fnns import FNNS

    model = get_model("fnns", lambda_w=1.3, device=device)
    assert isinstance(model, FNNS)

    msg = torch.randint(0, 2, (48,), dtype=torch.float32, device=device)
    imgs = [torch.randn(3, 128, 128, device=device) for _ in range(4)]
    imgs_w = model.generate_watermark(imgs, message=msg)
    assert isinstance(imgs_w, list)
    assert len(imgs_w) == 4
    assert all(isinstance(img, torch.Tensor) for img in imgs_w)
    extracted = model.detect_watermark(imgs_w)
    assert isinstance(extracted, dict)
    assert "detection_bits" in extracted
    assert "message" in extracted
    assert isinstance(extracted["detection_bits"], torch.Tensor)
    assert isinstance(extracted["message"], torch.Tensor)
    assert extracted["detection_bits"].shape == (4, 16)
    assert extracted["message"].shape == (4, 48 - 16)


def test_load_ssl(device):
    from omnisealbench.models.sslwm import SSLWaterMark

    model = get_model("ssl", lambda_w=1.3, device=device)
    assert isinstance(model, SSLWaterMark)

    # Check if the model support batch processing
    msg = torch.randint(0, 2, (48,), dtype=torch.float32, device=device)
    imgs = [torch.randn(3, 128, 128, device=device) for _ in range(4)]

    imgs_w = model.generate_watermark(imgs, message=msg)
    assert isinstance(imgs_w, list)
    assert len(imgs_w) == 4
    assert all(isinstance(img, torch.Tensor) for img in imgs_w)

    extracted = model.detect_watermark(imgs_w)
    assert isinstance(extracted, dict)
    assert "detection_bits" in extracted
    assert "message" in extracted
    assert isinstance(extracted["detection_bits"], torch.Tensor)
    assert isinstance(extracted["message"], torch.Tensor)
    assert extracted["detection_bits"].shape == (4, 16)
    assert extracted["message"].shape == (4, 48 - 16)


def test_load_wam(device):
    from omnisealbench.models.wam import WamWatermark

    model = get_model("wam", device=device)
    assert isinstance(model, WamWatermark)

    msg = torch.randint(0, 2, (32,), dtype=torch.float32, device=device)
    imgs = [torch.randn(3, 128, 128, device=device) for _ in range(4)]
    imgs_w = model.generate_watermark(imgs, message=msg)
    assert isinstance(imgs_w, list)
    assert len(imgs_w) == 4
    assert all(isinstance(img, torch.Tensor) for img in imgs_w)
    extracted = model.detect_watermark(imgs_w)
    assert isinstance(extracted, dict)
    assert "det" in extracted and "det_score" in extracted
    assert "message" in extracted
    assert isinstance(extracted["message"], torch.Tensor)
    assert extracted["message"].shape == (4, 32)


@pytest.mark.parametrize("model_class_or_function", ["videoseal_0.0", "videoseal_1.0"])
def test_load_videoseal(model_class_or_function: Literal['videoseal_0.0'] | Literal['videoseal_1.0'], device: str):
    from omnisealbench.models.videoseal import VideosealWM

    model = get_model(model_class_or_function, video_input=False, device=device)
    assert isinstance(model, VideosealWM)

    nbits = model.nbits
    detection_bits = model.detection_bits

    msg = torch.randint(0, 2, (nbits,), dtype=torch.float32, device=device)
    imgs = [torch.randn(3, 128, 128, device=device) for _ in range(4)]

    imgs_w = model.generate_watermark(imgs, message=msg)
    assert isinstance(imgs_w, list)
    assert len(imgs_w) == 4
    assert all(isinstance(img, torch.Tensor) for img in imgs_w)

    extracted = model.detect_watermark(imgs_w)
    assert isinstance(extracted, dict)
    assert "detection_bits" in extracted
    assert "message" in extracted
    assert isinstance(extracted["detection_bits"], torch.Tensor)
    assert extracted["detection_bits"].shape == (4, detection_bits)
    assert isinstance(extracted["message"], torch.Tensor)
    assert extracted["message"].shape == (4, nbits - detection_bits)


@pytest.mark.skipif(
    guess_cluster() not in [ClusterType.FAIR, ClusterType.AWS],
    reason="This test is only for FAIR and AWS clusters",
)
def test_latent_videoseal(device):

    if guess_cluster() == ClusterType.FAIR:
        ckpt_path = "/checkpoint/sylvestre/2025_logs/0630_vseal_maskgit_quantize/_scaling_w=1.4_scaling_w_schedule=1_embedder_model=0/checkpoint600.pth"
    else:
        ckpt_path = "/checkpoint/avseal/tuantran/experiments/latent/0630_vseal_maskgit_quantize/_scaling_w=1.4_scaling_w_schedule=1_embedder_model=0/checkpoint600.pth"

    model = get_model("latentseal", as_type="detector", ckpt_path=ckpt_path, device=device)

    nbits = model.nbits
    detection_bits = model.detection_bits

    # Latent vseal does not work on 3-channel images
    imgs_w = [torch.randn(3, 128, 128) for _ in range(4)]

    assert isinstance(imgs_w, list)
    assert len(imgs_w) == 4
    assert all(isinstance(img, torch.Tensor) for img in imgs_w)
    extracted = model.detect_watermark(imgs_w)
    assert isinstance(extracted, dict)
    assert "detection_bits" in extracted
    assert "message" in extracted
    assert isinstance(extracted["detection_bits"], torch.Tensor)
    assert extracted["detection_bits"].shape == (4, detection_bits)
    assert isinstance(extracted["message"], torch.Tensor)
    assert extracted["message"].shape == (4, nbits - detection_bits)
