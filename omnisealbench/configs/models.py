# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple, Union

from ..models import AudioWatermarkDetector, AudioWatermarkGenerator, Watermark
from .audio import load_model_config
from .base import build_card_yaml_path, get_builder_func
from .image import load_image_model_config


def build_registered_models_configs(
    card_type: str,
    config_models: List[str],
    model: Union[str, Tuple[AudioWatermarkGenerator, AudioWatermarkDetector]] = None,
) -> Dict[str, Any]:
    if (
        model is None
    ):  # no specific model is specified so we load all the registered model cardnames and yaml files
        registered_models = {}
        for registered_model in config_models[card_type]:
            config_path, model_name = build_card_yaml_path(
                registered_model, card_type, cards_dir="cards"
            )

            if card_type == "audio":
                cfg = load_model_config(config_path, model_name)
            else:
                cfg = load_image_model_config(config_path, model_name)

            model_name = cfg["name"]

            if model_name in registered_models:
                raise Exception(f"Model {model_name} has already been registered ...")
            registered_models[model_name] = cfg
    else:
        assert isinstance(
            model, (str, tuple)
        ), f"Model '{model}' has to be a valid registered model key or a tuple of AudioWatermark implementations."

        if isinstance(model, str):
            config_path, model_name = build_card_yaml_path(
                model, card_type=card_type, cards_dir="cards"
            )
            if card_type == "audio":
                cfg = load_model_config(config_path, model_name)
            else:
                cfg = load_image_model_config(config_path, model_name)

            model_name = cfg["name"]

            registered_models = {model_name: cfg}
        else:
            if len(model) == 2:
                assert isinstance(model[0], AudioWatermarkGenerator) and isinstance(
                    model[1], AudioWatermarkDetector
                )

                detector: AudioWatermarkGenerator = model[0]
                generator: AudioWatermarkDetector = model[1]

                registered_models = {
                    "custom_audio": {
                        "generator": generator,
                        "detector": detector,
                    }
                }
            elif len(model) == 1:
                assert isinstance(model[0], Watermark)
                raise NotImplementedError
            else:
                raise NotImplementedError

    return registered_models
