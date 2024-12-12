# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from typing import Any, Dict, List, Tuple, Union

import yaml

from ..base import build_card_yaml_path, get_builder_func


def load_model_config(config_path: str, model_name: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    assert "nbits" in model_config
    assert "generator" in model_config
    assert "builder_func" in model_config["generator"]
    assert "detector" in model_config
    assert "builder_func" in model_config["detector"]

    # Get the generator builder function from the configuration
    generator_builder_func = get_builder_func(model_config["generator"]["builder_func"])

    # Get the detector builder function from the configuration
    detector_builder_func = get_builder_func(model_config["detector"]["builder_func"])

    generator_additional_arguments = copy.deepcopy(model_config["generator"])
    del generator_additional_arguments["builder_func"]

    detector_additional_arguments = copy.deepcopy(model_config["detector"])
    del detector_additional_arguments["builder_func"]

    if "name" in model_config:
        model_name = model_config["name"]

    registered_model = {
        "name": model_name,
        "generator": {
            "builder_func": generator_builder_func,
            "additional_arguments": generator_additional_arguments,
        },
        "detector": {
            "builder_func": detector_builder_func,
            "additional_arguments": detector_additional_arguments,
        },
        "nbits": model_config["nbits"],
    }

    return registered_model
