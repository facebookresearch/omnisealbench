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


def load_image_model_config(config_path: str, model_name: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    assert "nbits" in model_config
    assert "builder_func" in model_config

    # Get the builder function from the configuration
    builder_func = get_builder_func(model_config["builder_func"])

    additional_arguments = copy.deepcopy(model_config)
    del additional_arguments["nbits"]
    del additional_arguments["builder_func"]

    if "name" in model_config:
        model_name = model_config["name"]
        del additional_arguments["name"]

    registered_model = {
        "name": model_name,
        "builder_func": builder_func,
        "nbits": model_config["nbits"],
        "additional_arguments": additional_arguments,
    }

    return registered_model
