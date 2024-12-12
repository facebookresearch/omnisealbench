# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Callable, Tuple


def get_builder_func(generator_builder_func_name: str) -> Callable:
    splits = generator_builder_func_name.split(".")

    module = None
    for s in splits:
        if module is None:
            module = locals().get(s) or globals().get(s) or __import__(s)
        else:
            module = getattr(module, s)

    return module


def build_card_yaml_path(card: str, card_type: str, cards_dir: str) -> Tuple[str, str]:
    if ".yaml" in card:
        config_path = card
    else:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.dirname(dir_path)
        config_path = os.path.join(dir_path, cards_dir, card_type, f"{card}.yaml")

    assert os.path.exists(
        config_path
    ), f"specified <<{card}>> card doesnt have a registered card yaml configuration: {config_path}"

    return config_path, Path(card).stem
