# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# mypy: disable-error-code="import-untyped, assignment"

import importlib
import inspect
import os
import sys
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import yaml

from omnisealbench.config import OMNISEAL_CHECKPOINTS_DIR
from omnisealbench.interface import Detector, Device, Generator, ModelTypeT
from omnisealbench.utils.common import filter_args, resolve
from omnisealbench.utils.thirdparty import install_lib


def _load_module_from_file(file_path):
    module_name = f"dynamic_module_{hash(file_path)}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[module_name]
        raise
    return module


def get_builder_function(
    model_class_or_function: ModelTypeT,
    deps: Optional[List[str]] = None,
    as_type: Optional[str] = None,
) -> Callable:
    """
    Get the function object that build the generator or detector. The main logic is as follows:
    1. If the model_class_or_function is a function, return it directly.
    2. If the model_class_or_function is a generator or detector class, return the builder function
        that creates an instance of the class, i.e. build_generator(), build_detector() or build_model().
        Here the precedence is strictly followed in the order below:
        - If as_type is "generator" and build_generator() exists, return it.
        - If as_type is "detector" and build_detector() exists, return it.
        - If build_model() exists, return it.
        - If build_generator() exists, return it.
        - If build_detector() exists, return it.
    3. If the model_class_or_function is a string, import the module and get the class or function, then
        return the builder function using rules 1.-2.
    """

    if deps:
        print(f"{model_class_or_function} has the following dependencies: {deps}")
        for dep in deps:
            try:
                install_lib(dep)
            except Exception as ex:
                raise ImportError(
                    f"Failed to install dependency {dep} which is required for {model_class_or_function}: {ex}"
                ) from ex

    if isinstance(model_class_or_function, str):
        if resolve(model_class_or_function).exists():
            module_ = _load_module_from_file(model_class_or_function)
            model_class_or_function = None  # type: ignore
        else:
            qualified_names = model_class_or_function.rsplit(".", 1)
            if len(qualified_names) == 1:
                return globals()[model_class_or_function]
            else:
                module_name, class_or_func_name = qualified_names
                module_ = importlib.import_module(module_name)
                model_class_or_function = getattr(module_, class_or_func_name)

    if inspect.isfunction(model_class_or_function):
        return model_class_or_function
    else:
        assert module_, "Module must be defined for class or function"  # type: ignore[assert]
        if (
            as_type == "generator"
            or (
                model_class_or_function is not None
                and issubclass(model_class_or_function, Generator)
            )
        ) and hasattr(module_, "build_generator"):
            return module_.build_generator
        if (
            as_type == "detector"
            or (
                model_class_or_function is not None
                and issubclass(model_class_or_function, Detector)
            )
        ) and hasattr(module_, "build_detector"):
            return module_.build_detector

        if hasattr(module_, "build_model"):
            return module_.build_model
        elif hasattr(module_, "build_generator"):
            return module_.build_generator
        elif hasattr(module_, "build_detector"):
            return module_.build_detector

        raise TypeError(
            f"Invalid model class or function: {model_class_or_function}: "
            "Missing build_generator(), build_detector(), or build_model() method."
        )


def parse_yaml_model_card(
    model_card_path: Union[str, Path],
    *,
    section: Optional[str] = None,
    substitutions: Optional[Mapping[str, str]] = None,
) -> Tuple[str, Dict[str, Any], Optional[List[str]]]:
    """
    Parse a YAML model card and return (builder_func, config_dict).

    Two supported layouts:

    1) Top-level (single card):
        builder_func: "pkg.mod:build_fn"
        name: "my_model"            # optional, ignored
        model_card_or_path: "${OMNISEAL_CHECKPOINTS}/foo/bar.pt"  # optional

    2) Sectioned (multiple cards, e.g. {"generator": {...}, "detector": {...}}):
        generator:
        builder_func: "pkg.gen:build"
        name: "gen"               # optional, ignored
        model_card_or_path: "..."

    Args:
        model_card_path: Path to the YAML file.
        section: If provided, select this sub-dict (e.g. "generator" / "detector")
                when the top-level doesn’t contain `builder_func`.
        substitutions: Values for Template-style placeholders like ${FOO}.
                    Environment variables are also expanded.

    Returns:
        (model_class_or_function, config_dict)

    Raises:
        FileNotFoundError, yaml.YAMLError, ValueError, KeyError
    """
    path = Path(model_card_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model card not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML at {path}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a mapping at the top level of {path}, got {type(data).__name__}"
        )

    # Drop optional fields we don't use
    data.pop("name", None)

    # If there is a __deps__, take this
    deps = data.pop("__deps__", None)

    # Select the relevant node
    if "builder_func" in data:
        node = dict(data)  # shallow copy; we'll pop from it
    else:
        if not section:
            raise ValueError(
                f"{path} does not contain a top-level 'builder_func'. "
                f"Provide `section=` (e.g., 'generator' or 'detector')."
            )
        if section not in data or not isinstance(data[section], dict):
            raise KeyError(f"Section '{section}' not found or not a mapping in {path}.")
        node = dict(data[section])  # shallow copy

    # Extract required key
    builder = node.pop("builder_func", None)
    if not builder:
        raise KeyError(
            f"'builder_func' missing in the selected node of {path} (section={section!r})."
        )

    # Resolve templated path if present
    if "model_card_or_path" in node and isinstance(node["model_card_or_path"], str):
        value = node["model_card_or_path"]

        has_substitutions = "OMNISEAL_CHECKPOINTS" in value

        # 1) user-provided substitutions via string.Template
        if substitutions:
            value = Template(value).safe_substitute(**substitutions)

        # 2) expand environment variables like $VAR or ${VAR} and ~
        value = os.path.expandvars(value)

        # 3) expand ~
        value = os.path.expanduser(value)

        if has_substitutions and Path(value).suffix and not Path(value).exists():
            raise FileNotFoundError(
                f"Model card or path does not exist: {value}.\n"
                " This can happen when you forget to download models using scripts/download_[modality]_models.sh,"
                " or forget to set the proper OMNISEAL_CHECKPOINTS environment variable."
                f"\nYaml file: {path}."
            )

        node["model_card_or_path"] = value

    return builder, node, deps


def read_model_card(
    model_card: str, model_type: Optional[str]
) -> Optional[Tuple[str, Dict, Optional[List[str]]]]:
    if Path(model_card).exists():
        return parse_yaml_model_card(
            model_card,
            section=model_type,
            substitutions={"OMNISEAL_CHECKPOINTS": OMNISEAL_CHECKPOINTS_DIR},
        )

    else:
        model_card_dir = os.getenv("OMNISEAL_MODEL_DIR")
        if model_card_dir is None:
            model_card_dir = Path(__file__).parent.parent.joinpath("cards")
        else:
            model_card_dir = Path(model_card_dir)

        model_card_path = model_card_dir.joinpath(f"{model_card}.yaml")

        if model_card_path.exists():
            return parse_yaml_model_card(
                model_card_path,
                section=model_type,
                substitutions={"OMNISEAL_CHECKPOINTS": OMNISEAL_CHECKPOINTS_DIR},
            )
        else:
            return None


def get_model(
    model_class_or_function: ModelTypeT,
    as_type: Optional[str] = None,
    device: Device = "cpu",
    **model_args,
) -> Union[Generator, Detector]:
    """
    Build a model instance from the given model type and class/function.

    Args:
        model_class_or_function (ModelTypeT): Type of the model to build. Can be:
            - A string representing the model card in the model regsitry.
            - A path to the model card YAML
            - The fully-qualified name of the model class or function.
            - A class or function that builds the model.
        as_type (Optional[str]): Type of the model to build. Can be "generator" or "detector".
        **model_args: Additional keyword arguments for the model instantiation.

    Returns:
        Union[Generator, Detector]: An instance of the specified model type.
    """

    cfg = {}
    if isinstance(model_class_or_function, str):
        # Load direct model from the name, and if not successful, try to load from the model card
        # this allows us to use the model card only as an optional fallabck
        try:
            builder_func = get_builder_function(
                model_class_or_function, as_type=as_type
            )
        except (ImportError, KeyError, AttributeError, FileNotFoundError) as e:
            model_card = read_model_card(model_class_or_function, model_type=as_type)
            if model_card:
                # read the model_class_or_function from the model card
                model_card_class_or_function, cfg, deps = model_card
                builder_func = get_builder_function(
                    model_card_class_or_function, deps, as_type=as_type
                )
            else:
                raise ValueError(
                    f"Model card {model_class_or_function} not found in the model registry, and "
                    + "neither it is a class/function nor as a YAML file. See nested error for details."
                ) from e
    else:
        builder_func = model_class_or_function

    model_args["device"] = device

    for k, v in cfg.items():
        if k not in model_args:
            model_args[k] = v

    kwargs = filter_args(inspect.signature(builder_func).parameters, **model_args)
    model_instance = builder_func(**kwargs)

    if isinstance(model_class_or_function, str):
        model_name = model_class_or_function
    else:
        model_name = model_instance.__class__.__name__

    setattr(model_instance, "name", model_name)

    return model_instance


# fmt: on
