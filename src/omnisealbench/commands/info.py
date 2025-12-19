# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import inspect
import os
from pathlib import Path

import yaml

from omnisealbench.commands.utils import GracefulExit, set_verbosity
from omnisealbench.config import OMNISEAL_CHECKPOINTS_DIR
from omnisealbench.utils.model_card import (get_builder_function,
                                            parse_yaml_model_card)


class InfoCommand:
    """commands for information about the OmnisealBench components"""

    def models(self, modality: str):
        """Show the list of default models for a specific modality."""
        assert modality in ["audio", "image", "video"], "Only accept modality='image', 'audio', or 'video'"
        set_verbosity(False)

        config = str(Path(__file__).parent.joinpath("configs", f"{modality}.yaml"))

        print(f"\nPath to default models for {modality}: {config}")

        with open(config, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        models = config_dict.get("models", [])
        print(f"\nDefault models for {modality}:")
        if isinstance(models, list):
            for model in models:
                print(f" - {model}")
        else:
            print(f" {models}")

        print("\nTo see detailed information about a specific model, use the following command:")
        print(" omniseal info model <model_name>\n")


    def model(self, model_name: str):
        """Show information about a specific model."""
        assert isinstance(model_name, str), f"Model name must be a string, got {type(model_name)}"
        set_verbosity(False)

        model_card_dir = os.getenv("OMNISEAL_MODEL_DIR")
        if model_card_dir is None:
            model_card_dir = Path(__file__).parent.parent.joinpath("cards")
        else:
            model_card_dir = Path(model_card_dir)

        model_card_path = model_card_dir.joinpath(f"{model_name}.yaml")

        if not model_card_path.exists():
            print(f"\nModel card {model_name} not found in: {model_card_dir}\n")
            print("If you have a custom model card, you can specify its directory using the OMNISEAL_MODEL_DIR environment variable.")

            return

        print(f"\nModel card found in: {model_card_path}\n")

        with model_card_path.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)

        if not data or not isinstance(data, dict):
            raise GracefulExit(f"Mal-formed content for {model_card_path}.\nSee example model card in {model_card_dir}")

        def print_parameters(model_card_path, section):
            try:
                model_card_info = parse_yaml_model_card(
                    model_card_path,
                    section=section,
                    substitutions={"OMNISEAL_CHECKPOINTS": OMNISEAL_CHECKPOINTS_DIR}
                )
                model_class_or_function, cfg, deps = model_card_info
                builder_func = get_builder_function(model_class_or_function, deps=deps)

            except (ImportError, KeyError, AttributeError, FileNotFoundError) as e:
                raise GracefulExit(f"Failed to parse model card for {model_name}:\n {e}. ")

            sig = inspect.signature(builder_func)
            parameters = sig.parameters
            doc = f"  Builder function: {model_class_or_function}"
            doc += builder_func.__doc__ or ""
            info = ""
            for name, param in parameters.items():
                if name in cfg:
                    value = cfg[name]
                elif param.default is not param.empty:
                    value = param.default
                else:
                    value = None
                if value:
                    info += f"\n    {name}: {value}"
                else:
                    info += f"\n    {name}: (missing, must be specified in CLI)"
            return doc, info

        # Model card has one config for both generator and detector
        if "builder_func" in data:
            doc, info = print_parameters(model_card_path, None)
            print(f"Definition of {model_name}:\n{doc}")
            print(f"Model {model_name} is registered with the following information:\n{info}")

        else:
            assert "generator" in data or "detector" in data, (
                f"Model card {model_name} is missing both 'generator' and 'detector' keys."
            )
            doc, info = print_parameters(model_card_path, "generator")
            print("Generator:")
            print(f"Definition of the generator for {model_name}:\n{doc}")
            print(f"\nThe generator for {model_name} is registered with the following information: {info}\n")
            doc, info = print_parameters(model_card_path, "detector")
            print("Detector:")
            print(f"Definition of the detector for {model_name}:\n{doc}")
            print(f"\nThe detector for {model_name} is registered with the following information: {info}")

    def attacks(self, modality: str):
        """Show the list of default attacks for a specific modality."""
        assert modality in ["audio", "image", "video"], "Only accept modality='image', 'audio', or 'video'"
        set_verbosity(False)

        config = str(Path(__file__).parent.parent.joinpath("attacks", f"{modality}_attacks.yaml"))

        print(f"\nPath to default attacks for {modality}: {config}")

        with open(config, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        print(yaml.dump(config_dict, sort_keys=False, default_flow_style=False, indent=2))
