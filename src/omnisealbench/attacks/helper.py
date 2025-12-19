
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import importlib
import inspect
import itertools
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
import yaml
from omegaconf import OmegaConf

from omnisealbench.utils.cache import global_lru_cache
from omnisealbench.utils.common import filter_args, resolve, to_device

NO_ATTACK = "identity"

@dataclass(frozen=True)
class AttackConfig:
    """
    Configuration for a dynamic function or class which can be used to create a callable
    that can be executed as an attack at runtime.
    Permissible format:
    - A single function or class with no arguments:
        AttackConfig(name='torch.nn.functional.relu')
    - A single function class with keyword arguments:
        AttackConfig(name='torch.nn.functional.conv2d', args={'in_channels': 3})
    - A function with list of arguments:
        AttackConfig(
            name='torch.nn.functional.conv2d', 
            args={'in_channels': [3,4], 'kernel_size': [3,4]}
        )
    - A class with initialization arguments and empty forward arguments:
        AttackConfig(
            func='torch.nn.Conv2d',
            init_args={
                'in_channels': 3,
                'out_channels': 16,
                'kernel_size': 3,
                'stride': 1,
            }
            args={}
        )
    - A registered attack for a modality
        AttackConfig(
            name="H264",
            data_type="video",
            args={"crf": 23},
        )
    """

    func: str = ""
    """Name of the function. This can be a registered method name from a function registry (e.g. list of
    audio attacks), or a fully-qualified name of the function e.g. 'torch.nn.functional.relu'"""

    data_type: str = ""
    """The expected data type for the attack"""

    category: str = "none"
    """Category of the attack"""

    init_args: Optional[Dict[str, Any]] = None
    """Initialization arguments for a class."""

    args: Optional[Dict[str, Any]] = None
    """Function arguments. Note that a dynamic function must not have positional arguments"""

    name: Optional[str] = None
    """Name of the attack in case passed explicitly by the user or from the YAML-based attack list"""

    device: Optional[str] = None
    """Device to run the attack on, e.g. 'cpu' or 'cuda:0'. Only applicable for class."""

    def __post_init__(self):
        assert self.func or self.name, "Either `func` or `name` must be specified in AttackConfig"

    def __hash__(self):
        def make_hashable(obj):
            if isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            if isinstance(obj, list):
                return tuple(make_hashable(x) for x in obj)
            if isinstance(obj, set):
                return tuple(sorted(make_hashable(x) for x in obj))
            return obj
        return hash((
            self.func,
            make_hashable(self.init_args),
            make_hashable(self.args),
            self.device,
        ))

    def __repr__(self):
        res = self.name if self.name else self.func
        # simplify the function name if it is a fully-qualified name
        if "." in res:
            res = res.split(".")[-1]
        if self.init_args:
            res = res + "(" + "__".join(f"{k}_{v}" for k, v in self.init_args.items()) + ")"
        if self.args:
            res = res + "__" + "__".join(f"{k}_{v}" for k, v in self.args.items())
        return res

    def get_category(self):
        if self.func == NO_ATTACK:
            return "none"
        if self.category and self.category != "none":
            return self.category
        assert self.name and self.data_type, (
            "AttackConfig must have a non-empty `name` and `data_type` if `category` is not specified, "
            "in order to infer the category from the attack YAML registry"
        )
        return get_attack_category(self.name, attacks_source=self.data_type)


no_attack_config = AttackConfig(
    func=NO_ATTACK,
    name="identity",
    category="none",
    init_args=None,
    args=None,
    device=None,
)


def identity_attack(t: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Identity attack function that returns the input tensor as is.

    Args:
        t (torch.Tensor): Input tensor to be returned.
        **kwargs: Additional keyword arguments (not used).

    Returns:
        torch.Tensor: The input tensor.
    """
    return t


def unroll_attack_config(config: AttackConfig) -> List[AttackConfig]:
    """
    Get a list of attack functions from the provided configuration.

    Args:
        config (AttackConfig): Configuration containing attack functions.

    Returns:
        List[AttackFuncion]: List of attack functions with their names and parameters.
    """
    func = to_func(config)
    if config.args is None or len(config.args) == 0:
        return [config]
    attack_cfgs = get_attack_run_configs(func, config.args)

    cfgs = []
    for cfg in attack_cfgs:
        # "attrs " is for V1, remove them
        cfg.pop("attrs", None)

        cfgs.append(
            AttackConfig(
                func=config.func,
                name=config.name,
                data_type=config.data_type,
                category=config.category,
                init_args=config.init_args,
                args=cfg,
                device=config.device if hasattr(config, 'device') else None,
            )
        )

    return cfgs


def get_attack_run_configs(
    attack_implementation: Callable[..., torch.Tensor], attack_config: Dict
) -> List[Dict]:

    if inspect.isfunction(attack_implementation):
        full_args_spec: inspect.FullArgSpec = inspect.getfullargspec(attack_implementation)
    else:
        full_args_spec: inspect.FullArgSpec = inspect.getfullargspec(attack_implementation.forward)

    run_cfgs = []

    # image attacks
    if "args" in attack_config:
        for k, values in attack_config["args"].items():
            if k in full_args_spec.args:
                if isinstance(values, list):
                    for value in values:
                        run_cfg = copy.deepcopy(attack_config)
                        run_cfg["args"][k] = value
                        run_cfg["attrs"] = (k, value)
                        run_cfgs.append(run_cfg)
    else:
        param_names = []
        param_values = []

        for k, v in attack_config.items():
            if isinstance(v, list):
                param_names.append(k)
                param_values.append(v)

        if len(param_names) > 0:
            for value_combination in itertools.product(*param_values):
                run_cfg = copy.deepcopy(attack_config)
                for i, value in enumerate(value_combination):
                    run_cfg[param_names[i]] = value

                run_cfg["attrs"] = ("_".join(param_names), "_".join(map(str, value_combination)))
                run_cfgs.append(run_cfg)

    if len(run_cfgs) == 0:
        run_cfgs.append(attack_config)

    return run_cfgs


def to_func(cfg: AttackConfig, resolve: bool = True) -> Callable:
    func_name = cfg.func
    if func_name == NO_ATTACK:
        return identity_attack

    try:
        splits = func_name.split(".")
        module = None

        # probing the package until we get ModuleNotFoundError or ImportError
        for i in range(len(splits)):
            s = ".".join(splits[:(i + 1)])
            try:
                module = locals().get(s) or globals().get(s) or importlib.import_module(s)
            except (ModuleNotFoundError, ImportError) as err:
                if s not in str(err):
                    raise err
                break

        if module is None:
            raise ImportError(f"Module {func_name} not found. Ensure it is a valid module.")

        for j in range(i, len(splits)):
            module = getattr(module, splits[j])

        return module

    except (ImportError, ModuleNotFoundError, AttributeError, ValueError) as ex:
        # Cannot load the function from cfg.func, so we try to read the "attack registry" from
        # the YAML files and load it accordingly

        if resolve:
            file_ = Path(__file__).parent / f"{cfg.data_type}_attacks.yaml"
            with open(file_, "r", encoding="utf-8") as f:
                default_attacks = yaml.safe_load(f)
            name = cfg.name or cfg.func
            if name in default_attacks:
                func_name = OmegaConf.create(default_attacks[name])
                return to_func(func_name, resolve=False)
            else:
                raise ValueError(f"Unknown attack function: {name}") from ex
        else:
            raise ValueError(f"Failed to resolve attack function: {cfg.func}") from ex


def _get_device(input_) -> Optional[str]:
    device_ = None
    if isinstance(input_, torch.Tensor):
        device_ = input_.device
    elif isinstance(input_, (tuple, list)) and isinstance(input_[0], torch.Tensor):
        device_ = input_[0].device
    return device_


@global_lru_cache(maxsize=64)
def init_attack(cfg: AttackConfig, device: Optional[str] = None) -> torch.nn.Module:
    attack_implementation = to_func(cfg)

    assert issubclass(attack_implementation, torch.nn.Module), \
        f"Attack {cfg.func}: Expected a torch.nn.Module, get {type(attack_implementation)}"

    if cfg.init_args is not None:
        # If the attack is a class, we need to instantiate it first
        attacker = attack_implementation(**cfg.init_args)
    else:
        # If the attack is a function, we can just return it
        attacker = attack_implementation()

    if device is not None:
        attacker.to(device)

    return attacker


def run_attack(attack_cfg: AttackConfig, seed: Optional[int], *input, **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Run the attack on the input tensor using the provided configuration.

    Args:
        input (torch.Tensor): The input tensor to be attacked.
        attack_cfg (AttackConfig): Configuration for the attack function.
        **kwargs: Additional keyword arguments for the attack function.

    Returns:
        torch.Tensor: The attacked tensor.
    """
    if attack_cfg.func == "identity":
        # Identity function just returns the input as is
        return input[0] if input else None

    if seed is not None:
        # Set the seed to ensure the attack is deterministic
        from omnisealbench.utils.common import set_seed
        set_seed(seed)

    attack_implementation = to_func(attack_cfg)

    # Build the kwargs for the attack function
    if not attack_cfg.args:
        kwargs_ = kwargs
    else:
        kwargs_ = {**attack_cfg.args, **kwargs}

    if "args" in kwargs_:
        args_dict = kwargs_.pop("args")
        for k, v in args_dict.items():
            kwargs_[k] = v

    # Check how many positional arguments the function has
    if inspect.isfunction(attack_implementation):
        sig = inspect.signature(attack_implementation)
    else:
        sig = inspect.signature(attack_implementation.forward)

    positional_args = [
        p for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.name != "self"
        and p.name != "cls"  # Exclude class methods
        and p.name not in kwargs_
        and (p.annotation is torch.Tensor or p.annotation is inspect.Parameter.empty)
    ]

    # Trim the input to match the number of positional arguments
    # This happens when the attack function needs only a watermarked audio / image or video, 
    # and the task provides the original content as well.
    input = input[:len(positional_args)]

    kwargs_ = filter_args(sig.parameters, **kwargs_)

    if inspect.isfunction(attack_implementation):
        attack_func = attack_implementation
    else:
        # Resolve device between config and the input
        device_ = attack_cfg.device if attack_cfg.device else _get_device(input[0])
        attack_func = init_attack(attack_cfg, device=device_)
        if device_ is not None:
            input = to_device(input, device=device_) 

    # Since all attack functions are applied to individual tensors, we need to run them over loop
    # if the input is the list of tensors (e.g. in imae or video attacks, where inputs are unpadded).
    if isinstance(input[0], list) and isinstance(input[0][0], torch.Tensor):
        attacked_inputs = []
        for input_ in zip(*input):
            attacked_input = attack_func(*input_, **kwargs_)
            if isinstance(attacked_input, torch.Tensor):
                attacked_inputs.append(attacked_input)
            elif isinstance(attacked_input, tuple):
                # If the attack function returns a tuple, the second element is usually a mask or some other output
                # We only need the first element, which is the attacked tensor
                attacked_inputs.append(attacked_input[0])
        return attacked_inputs
    return attack_func(*input, **kwargs_)


def load_default_attacks(
    attacks: Union[str, Sequence[str], None],
    attacks_source: str = "audio",
) -> List[AttackConfig]:
    """
    Load default attacks based on the specified type.

    Args:
        attacks_source (str): Source of attack YAML registry to load. Options are 'audio', 'image', or 'video'.

    Returns:
        Dict[str, AttackConfig]: A dictionary of default attack configurations.
    """

    files = {
        "audio": Path(__file__).parent / "audio_attacks.yaml",
        "image": Path(__file__).parent / "image_attacks.yaml",
        "video": Path(__file__).parent / "video_attacks.yaml",
    }

    file_ = files[attacks_source]
    if isinstance(attacks, str):
        if Path(attacks).is_file():
            file_ = str(resolve(Path(attacks)))
            attacks = None
        else:
            attacks = [attacks]
    
    attack_configs = []
    with open(file_, "r", encoding="utf-8") as f:
        default_attacks = yaml.safe_load(f)
    for attack, cfg in default_attacks.items():
        if attacks and attack not in attacks:
            continue
        func = cfg.pop("func")
        category = cfg.pop("category", None)
        if not category:
            category = get_attack_category(func, attacks_source=attacks_source)
        attack_config = AttackConfig(func=func, data_type=attacks_source, name=attack, category=category, **cfg)
        attack_configs.append(attack_config)

    return attack_configs


@lru_cache(maxsize=8)
def _attack_category(attacks_source: str) -> Dict[str, str]:
    with open(Path(__file__).parent / "categories.yaml", "r", encoding="utf-8") as f:
        attack_category = yaml.safe_load(f)

    return attack_category.get(attacks_source, {})


def get_attack_category(name: str, attacks_source: str) -> str:
    category = _attack_category(attacks_source)
    attack_name = name.split(".")[-1] if "." in name else name
    attack_name = attack_name.lower()
    return category.get(attack_name, "none")
