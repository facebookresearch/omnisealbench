# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from .base import Watermark


def build_wam_watermark(
    device: str,
    nbits: int,
    lambda_w: float,
    checkpoint_dir: str,
    checkpoint_path: str,
    embedder_cfg_yaml: str,
    augmenter_cfg_yaml: str,
    extractor_cfg_yaml: str,
    do_attenuation: bool = False,
    attenuation_cfg_yaml: str = None,
    color: str = "blue",
    interpolation: str = "bilinear",
    **kwargs,
) -> Watermark:
    from .wam import WAM

    return WAM(
        nbits=nbits,
        device=device,
        checkpoint_path=os.path.join(checkpoint_dir, checkpoint_path),
        embedder_cfg_yaml=os.path.join(checkpoint_dir, embedder_cfg_yaml),
        augmenter_cfg_yaml=os.path.join(checkpoint_dir, augmenter_cfg_yaml),
        extractor_cfg_yaml=os.path.join(checkpoint_dir, extractor_cfg_yaml),
        do_attenuation=do_attenuation,
        attenuation_cfg_yaml=attenuation_cfg_yaml,
        scaling_w=lambda_w,
        color=color,
        interpolation=interpolation,
    )


def build_hidden_watermark(
    device: str,
    encoder_path: str,
    decoder_path: str,
    lambda_w: float,
    **kwargs,
) -> Watermark:
    from .hidden import Hidden

    return Hidden(
        device=device,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        lambda_w=lambda_w,
    )


def build_ssl_watermark(
    device: str,
    nbits: int,
    decoder_path: str,
    lambda_w: float,
    **kwargs,
) -> Watermark:
    from .ssl_watermark import SSLWatermark

    return SSLWatermark(
        nbits=nbits,
        lambda_w=lambda_w,
        device=device,
        decoder_path=decoder_path,
    )


def build_fnns_watermark(
    device: str,
    nbits: int,
    decoder_path: str,
    lambda_w: float,
    **kwargs,
) -> Watermark:

    from .fnns import FNNS

    return FNNS(
        nbits=nbits,
        lambda_w=lambda_w,
        device=device,
        decoder_path=decoder_path,
    )


def build_trustmark_watermark(
    device: str,
    model_type: str,
    encoding_type: str,
    **kwargs,
) -> Watermark:
    from .trustmark import TrustMarkWM

    return TrustMarkWM(
        device=device, model_type=model_type, encoding_type=encoding_type
    )


def build_dct_dwt_watermark(
    device: str,
    nbits: int,
    **kwargs,
) -> Watermark:
    from .dct_dwt import DctDwt

    return DctDwt(nbits=nbits)
