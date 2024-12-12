# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .detector import *
from .generator import *


def build_audioseal_generator(
    model: str,
    device: str,
    model_card_or_path: str,
    **kwargs,
) -> AudioWatermarkGenerator:
    from .audioseal import AudioSealWatermarkGenerator

    return AudioSealWatermarkGenerator(
        name=model,
        device=device,
        model_card_or_path=model_card_or_path,
        **kwargs,
    )


def build_timbre_generator(
    model: str,
    device: str,
    checkpoint_pth: str,
    **kwargs,
) -> AudioWatermarkGenerator:
    from .timbre import TimbreWatermarkGenerator

    return TimbreWatermarkGenerator(
        name=model,
        device=device,
        checkpoint_pth=checkpoint_pth,
        **kwargs,
    )


def build_wavmark_generator(
    model: str,
    device: str,
    **kwargs,
) -> AudioWatermarkGenerator:
    from .wavmark import WavmarkWatermarkGenerator

    return WavmarkWatermarkGenerator(name=model, device=device, **kwargs)


def build_audioseal_detector(
    device: str,
    model_card_or_path: str,
    **kwargs,
):
    from .audioseal import AudioSealWatermarkDetector

    return AudioSealWatermarkDetector(
        model_card_or_path=model_card_or_path, device=device, **kwargs
    )


def build_wavmark_detector(
    device: str,
    **kwargs,
):
    from .wavmark import WavmarkWatermarkDetector

    return WavmarkWatermarkDetector(device=device, **kwargs)


def build_timbre_detector(
    device: str,
    **kwargs,
):
    from .timbre import TimbreWatermarkDetector

    return TimbreWatermarkDetector(device=device, **kwargs)
