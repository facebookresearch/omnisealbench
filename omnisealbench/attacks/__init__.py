# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import typing as tp

import torch

from .audio_effects import *
from .image_effects import *
from .image_quality import *


def get_attack_run_configs(
    attack_implementation: tp.Callable[..., torch.Tensor], attack_config: tp.Dict
) -> tp.List[tp.Dict]:
    full_args_spec: inspect.FullArgSpec = inspect.getfullargspec(attack_implementation)

    run_cfgs = []

    if "func" in attack_config:
        pass  # image attacks
    else:
        for k, v in attack_config.items():
            if k in full_args_spec.args:
                if isinstance(v, list):
                    for value in v:
                        run_cfg = attack_config.copy()
                        run_cfg[k] = value
                        run_cfg["attrs"] = (k, value)
                        run_cfgs.append(run_cfg)

    if len(run_cfgs) == 0:
        run_cfgs.append(attack_config)

    return run_cfgs


def run_attack(
    sample_rate: int,
    watermarked_audio: torch.tensor,
    attack_implementation: tp.Callable[..., torch.tensor],
    attack_runtime_config: tp.Dict[str, tp.Any],
):
    full_args_spec: inspect.FullArgSpec = inspect.getfullargspec(attack_implementation)

    kwargs = {}

    if "sample_rate" in full_args_spec.args:
        kwargs["sample_rate"] = sample_rate

    if "tensor" in full_args_spec.args:
        kwargs["tensor"] = watermarked_audio
    elif "waveform" in full_args_spec.args:
        kwargs["waveform"] = watermarked_audio
    else:
        raise NotImplementedError("cannot run attack")

    for k, v in attack_runtime_config.items():
        if k == "attrs":  # reserved key
            continue

        if k in full_args_spec.args:
            kwargs[k] = v

    attack_audio = attack_implementation(**kwargs)
    return attack_audio


def run_image_attack(
    watermarked_img: torch.tensor,
    original_img: torch.tensor,
    attack_implementation: tp.Callable[..., torch.tensor],
    attack_runtime_config: tp.Dict[str, tp.Any],
):
    full_args_spec: inspect.FullArgSpec = inspect.getfullargspec(attack_implementation)
    kwargs = {"x": watermarked_img}
    if "y" in full_args_spec.args:
        kwargs["y"] = original_img

    if "args" in attack_runtime_config:
        args_dict = attack_runtime_config["args"]
        for k, v in args_dict.items():
            assert k in full_args_spec.args
            kwargs[k] = v

    attack_audio = attack_implementation(**kwargs)
    return attack_audio
