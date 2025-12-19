# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from omnisealbench.utils.model_card import get_model


def test_load_audioseal(device):

    from omnisealbench.models.audioseal import (AudioSealWatermarkDetector,
                                                AudioSealWatermarkGenerator)

    generator = get_model(
        "omnisealbench.models.audioseal.AudioSealWatermarkGenerator",
        model_card_or_path="audioseal_wm_16bits",
        device=device,
    )
    assert isinstance(generator, AudioSealWatermarkGenerator)

    msg = torch.randint(0, 2, (16,), device=device, dtype=torch.float32)
    audios = torch.randn(4, 1, 16000 * 2, device=device)
    audios_wm = generator.generate_watermark(audios, msg)
    assert audios_wm.shape == audios.shape

    detector = get_model(
        "omnisealbench.models.audioseal.AudioSealWatermarkDetector",
        model_card_or_path="audioseal_detector_16bits",
        device=device,
    )
    extracted = detector.detect_watermark(audios_wm)
    assert isinstance(extracted, dict)
    assert "prediction" in extracted
    assert isinstance(extracted["prediction"], torch.Tensor)
    assert extracted["prediction"].shape == (4,)
    assert "message" in extracted
    assert isinstance(extracted["message"], torch.Tensor)
    assert extracted["message"].shape == (4, 16)

    assert isinstance(detector, AudioSealWatermarkDetector)


def test_load_timbre(device):

    from omnisealbench.models.timbre import (TimbreWatermarkDetector,
                                             TimbreWatermarkGenerator)

    generator = get_model("timbre", as_type="generator", device=device)
    assert isinstance(generator, TimbreWatermarkGenerator)

    msg = torch.randint(0, 2, (30,), device=device, dtype=torch.float32)
    audios = torch.randn(4, 1, 16000 * 2).to(device)
    audios_wm = generator.generate_watermark(audios, msg)
    assert audios_wm.shape == audios.shape

    detector = get_model("timbre", as_type="detector", device=device)
    extracted = detector.detect_watermark(audios_wm)
    assert isinstance(extracted, dict)
    assert "detection_bits" in extracted
    assert isinstance(extracted["detection_bits"], torch.Tensor)
    assert extracted["detection_bits"].shape == (4, 14)
    assert "message" in extracted
    assert isinstance(extracted["message"], torch.Tensor)
    assert extracted["message"].shape == (4, 16)
    assert isinstance(detector, TimbreWatermarkDetector)    


def test_load_wavmark_fast(device):
    from omnisealbench.models.wavmark import WavMark

    model = get_model("wavmark_fast", device=device)
    assert isinstance(model, WavMark)

    msg = torch.randint(0, 2, (16,), dtype=torch.float32, device=device)
    audios = torch.randn(4, 1, 16000 * 2, device=device)
    audios_wm = model.generate_watermark(audios, msg)
    assert audios_wm.shape == audios.shape

    extracted = model.detect_watermark(audios_wm)
    assert isinstance(extracted, dict)
    assert "det_score" in extracted
    assert isinstance(extracted["det_score"], torch.Tensor)
    assert "message" in extracted
    assert isinstance(extracted["message"], torch.Tensor)
    assert extracted["message"].shape == (4, 16)
