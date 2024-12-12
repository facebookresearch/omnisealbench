# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .base import build_card_yaml_path, get_builder_func


def load_dataset_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        db_config = yaml.safe_load(f)

    assert "builder_func" in db_config
    assert "name" in db_config

    builder_func = get_builder_func(db_config["builder_func"])
    assert callable(
        builder_func
    ), f"builder_func {db_config['builder_func']} must be callable"

    additional_arguments = copy.deepcopy(db_config)
    del additional_arguments["name"]
    del additional_arguments["builder_func"]

    return {
        "name": db_config["name"],
        "builder_func": builder_func,
        "additional_arguments": additional_arguments,
    }


def build_registered_datasets_configs(
    card_type: str,
    config_datasets: List[str],
    dataset: str,
    dataset_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if dataset is None:
        registered_datasets = {}
        for dataset in config_datasets:
            config_path, _ = build_card_yaml_path(dataset, cards_dir="datasets")
            cfg = load_dataset_config(config_path)

            if "dataset_dir" in cfg["additional_arguments"]:
                cfg["additional_arguments"]["dataset_dir"] = dataset_dir
                cfg["name"] = Path(dataset_dir).stem

            if cfg["name"] in registered_datasets:
                raise Exception(
                    f"dataset {cfg['name']} has already been registered ..."
                )
            registered_datasets[cfg["name"]] = cfg
    else:
        config_path, _ = build_card_yaml_path(
            dataset, card_type=card_type, cards_dir="datasets"
        )
        cfg = load_dataset_config(config_path)

        if "dataset_dir" in cfg["additional_arguments"]:
            cfg["additional_arguments"]["dataset_dir"] = dataset_dir
            cfg["name"] = Path(dataset_dir).stem

        registered_datasets = {cfg["name"]: cfg}

    return registered_datasets
