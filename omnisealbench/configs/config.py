# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# Cache location
DEFAULT_XDG_CACHE_HOME = "~/.cache"
XDG_CACHE_HOME = os.getenv("XDG_CACHE_HOME", DEFAULT_XDG_CACHE_HOME)
DEFAULT_CACHE_HOME = os.path.join(XDG_CACHE_HOME, "omnisealbench")
OMNISEALBENCH_CACHE_HOME = os.path.expanduser(
    os.getenv("OMNISEALBENCH_HOME", DEFAULT_CACHE_HOME)
)


DEFAULT_OMNISEAL_WATERMARKING_CACHE = os.path.join(
    OMNISEALBENCH_CACHE_HOME, "watermarking"
)
OMNISEAL_WATERMARKING_CACHE = Path(
    os.getenv("OMNISEAL_WATERMARKS_CACHE", DEFAULT_OMNISEAL_WATERMARKING_CACHE)
)

os.makedirs(OMNISEAL_WATERMARKING_CACHE, exist_ok=True)


def load_config(file_path):
    # Load the YAML configuration file
    try:
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
    except Exception as e:
        print(f"An error occurred: {e}")


def load_configs(
    config_yaml: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], List[str], List[str]]:
    if config_yaml is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir_path, "config.yaml")
        assert os.path.exists(config_path)

        # Load the configuration
        cfg = load_config(config_path)
    else:
        assert os.path.exists(config_yaml)

        cfg = load_config(config_yaml)

    watermarking_config = cfg["watermarking_config"]

    attacks = cfg["attacks"]
    models = cfg["models"]
    datasets = cfg["datasets"]

    dir_path = os.path.dirname(dir_path)

    audio_attacks_config_path = os.path.join(dir_path, "attacks", "audio_attacks.yaml")
    assert os.path.exists(audio_attacks_config_path), audio_attacks_config_path

    # Load the configuration
    audio_attacks_config = load_config(audio_attacks_config_path)

    image_attacks_config_path = os.path.join(dir_path, "attacks", "image_attacks.yaml")
    assert os.path.exists(image_attacks_config_path), image_attacks_config_path

    # Load the configuration
    image_attacks_config = load_config(image_attacks_config_path)

    return (
        watermarking_config,
        audio_attacks_config,
        image_attacks_config,
        attacks,
        models,
        datasets,
    )
