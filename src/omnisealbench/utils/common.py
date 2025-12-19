# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import contextlib
import inspect
import json
import os
from pathlib import Path
from typing import (Any, Dict, List, Optional, Tuple, TypeVar,
                    Union, get_args, get_origin)

import numpy as np
import torch

from omnisealbench.interface import ScoresT, Generator, Detector

T = TypeVar("T")


def is_optional(annotation):
    return (
        get_origin(annotation) is Union and
        len(get_args(annotation)) == 2 and
        type(None) in get_args(annotation)
    )


def resolve(p: Union[str, Path]) -> Path:
    p = os.path.expandvars(str(p))
    return Path(p).expanduser().resolve()


def batch_safe(func):
    """
    Make a function safe to apply to both single inputs and batches of inputs.
    If the arguments are list of T, while the function expects T, it will apply
    the function to each element of the list. Otherwise, it will apply the function directly.
    """
    def wrapper(*args, **kwargs):
        # Use inspect to check if the func's argument types and the args are compatible
        sig = inspect.signature(func)
        stop_early = False
        annots = []
        for i, (name, param) in enumerate(sig.parameters.items()):
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                # If we don't have enough annotations, we stop early
                if param.annotation is inspect.Parameter.empty:
                    stop_early = True
                annots.append((i, name, param.annotation))

        if stop_early:
            return func(*args)

        # This happens when the function call's signature has a positional argument in the form of a keyword argument
        if len(annots) > len(args):
            missing_args = []
            for (i, name, annot) in annots:
                if name in kwargs:
                    val = kwargs.pop(name)
                    missing_args.append((i, val))

            missing_args.sort(key=lambda x: x[0])  # Sort by the original index

            args = list(args)  + [val for _, val in missing_args]  # Append the missing args to the end of the args list

        batch_args_cnt: int = 0  # the number of arguments that are lists of tensors while the function expects tensors
        single_args = []  # the positions of arguments that have a matching type with the function's annotation
        batch_size = 1

        # We iterate over the positional arguments only
        for i, (arg, annot) in enumerate(zip(args, annots)):
            _type = annot[2]
            if args is None and is_optional(_type):
                single_args.append(i)  # Allow None for Optional types
            elif isinstance(arg, _type) or (get_origin(_type) and isinstance(arg, get_origin(_type))):
                single_args.append(i)
            elif (
                (issubclass(_type, torch.Tensor) or get_origin(_type) is torch.Tensor) and
                (isinstance(arg, list) and isinstance(arg[0], torch.Tensor))
            ):
                batch_args_cnt += 1
                batch_size = len(arg)
            else:
                raise TypeError(
                    f"Argument {i} of function {func.__name__} is not compatible with its annotation: "
                    f"expected {_type}, got {type(arg)}"
                )

        if batch_args_cnt == 0:
            assert len(single_args) == len(args), \
                f"Function {func.__name__} expects all arguments to be tensors, but got {len(args)} arguments"
            return func(*args, **kwargs)

        if batch_args_cnt == len(args):
            batch_args = args

        else:
            # If we have a mix of single tensors and lists of tensors, we need to handle them separately
            batch_args = []
            for i, arg in enumerate(args):
                if i in single_args:
                    # If the argument is a single tensor, we just append it
                    batch_args.append([arg] * batch_size)
                elif isinstance(arg, list) and all(isinstance(x, torch.Tensor) for x in arg):
                    # If the argument is a list of tensors, we append each tensor as a separate element
                    batch_args.append(arg)
                else:
                    raise TypeError(
                        f"Argument {i} of function {func.__name__} is not compatible with its annotation: "
                        f"expected {annots[i]}, got {type(arg)}"
                    )

        # If all arguments are lists of tensors, we apply the function to each element of the list
        results = [func(*batch, **kwargs) for batch in zip(*batch_args)]

        return_type = sig.return_annotation
        if issubclass(return_type, torch.Tensor) or (get_origin(return_type) is torch.Tensor):
            # If the return type is a tensor, we stack the results
            return torch.stack(results)
        else:
            return results

    return wrapper


def set_seed(seed: int):
    """
    Set the random seed for reproducibility.
    Args:
        seed (int): The random seed to set.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def filter_args(parameters, **kwargs):
    has_kwargs = (
        len({k: v for k, v in parameters.items() if v.kind == v.VAR_KEYWORD}) > 0
    )
    if has_kwargs:
        return kwargs
    else:
        return {k: v for k, v in kwargs.items() if k in parameters}


@contextlib.contextmanager
def set_path(path: Path):
    """
    A context manager to set the PYTHONPATH to the current directory.
    This is useful for importing modules from the current directory.
    """
    import sys

    path = str(resolve(path))  # type: ignore
    original_path = sys.path.copy()
    sys.path.append(path)

    yield
    sys.path = original_path


def tensor_to_message(tensor: torch.Tensor) -> str:
    """
    Convert a tensor to a string message.
    This is useful for logging or displaying the tensor content.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(tensor)}")

    if tensor.dim() > 1:
        tensor = tensor[0]

    return " ".join(map(str, tensor.tolist()))


def message_to_tensor(message: str) -> str:
    """Convert a string message to a tensor.
    This is useful for converting messages back to tensors for processing
    """
    if not isinstance(message, str):
        raise TypeError(f"Expected a string, got {type(message)}")

    # Split the message by whitespace and convert to float
    return torch.tensor([float(x) for x in message.split()], dtype=torch.float32)


def to_python_type(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, (np.generic, np.ndarray)):
        return obj.item() if hasattr(obj, "item") else obj.tolist()
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_type(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple([to_python_type(x) for x in obj])
    return obj

def restore_types(obj):
    if isinstance(obj, list):
        # Convert lists of numbers to torch.Tensor
        if all(isinstance(x, (int, float)) for x in obj):
            return torch.tensor(obj)
        else:
            return [restore_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: restore_types(v) for k, v in obj.items()}
    elif isinstance(obj, float):
        return np.float32(obj)
    else:
        return obj


def save_raw_results(scores: Union[ScoresT, List[ScoresT]], filename: Path) -> None:
    if isinstance(scores, dict):
        scores = [scores]  # Convert single dict to list for consistency

    filename.parent.mkdir(exist_ok=True, parents=True)
    with open(filename, "w", encoding="utf-8") as fp:
        for item in scores:
            if not isinstance(item, dict):
                raise TypeError(f"Expected a dictionary, got {type(item)}")
            # Convert item to a JSON string and write it to the file
            fp.write(json.dumps(to_python_type(item), separators=(',', ':'), indent=None, sort_keys=True) + "\n")


def load_raw_results(filename: Path) -> Union[ScoresT, List[ScoresT]]:
    raw_results = []
    with open(filename, "r", encoding="utf-8") as fp:
        for line in fp:
            line = json.loads(line.rstrip())
            raw_result = restore_types(line)
            raw_results.append(raw_result)

    return raw_results[0] if len(raw_results) == 1 else raw_results


def save_results(scores: Dict[str, Any], filename: Path) -> None:
    """
    Save the scores to a file.
    Args:
        scores (ScoresT): The scores to save.
        output_file (Path): The path to the output file.
    """
    resolve(filename).parent.mkdir(exist_ok=True, parents=True)

    if filename.suffix == ".json":
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(to_python_type(scores), f, indent=4, sort_keys=True)
    elif filename.suffix == ".csv":
        import pandas as pd  # type: ignore
        df = pd.DataFrame(scores)
        df.to_csv(filename, index=False)
    else:
        raise ValueError(
            f"Unsupported file format {filename.suffix}. "
            "Supported formats are .json and .csv."
        )


def load_results(filename: Path) -> Dict[str, Any]:
    """
    Load the scores from a file.
    Args:
        filename (Path): The path to the input file.
    Returns:
        ScoresT: The loaded scores.
    """

    if filename.suffix == ".json":
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return restore_types(data)  # type: ignore
    elif filename.suffix == ".csv":
        import pandas as pd  # type: ignore
        df = pd.read_csv(filename)
        return df.to_dict(orient="list")
    else:
        raise ValueError(
            f"Unsupported file format {filename.suffix}. "
            "Supported formats are .json and .csv."
        )


def get_file_paths(root_path: str, pattern: str, num_samples: Optional[int] = None) -> List[Path]:
    def filename_to_index(filename: str) -> int:
        """
        Extract the index from the filename.
        Assumes filenames are formatted as 'audio_<index>.wav'.
        """
        return int(filename.split('_')[-1].split('.')[0])
    files = sorted(
        Path(root_path).rglob(pattern),
        key=lambda x: filename_to_index(x.name),
    )
    if num_samples:
        files = files[:num_samples]
    return files


def get_tensor_device(x):
    if isinstance(x, torch.Tensor):
        return x.device
    elif isinstance(x, list) and isinstance(x[0], torch.Tensor):
        return get_tensor_device(x[0])
    elif isinstance(x, dict):
        return get_tensor_device(next(iter(x.values())))
    else:
        return "cpu"


def get_device(model: Union["Generator", "Detector", "torch.nn.Module"]) -> str:
    """
    Get the device of the model.
    Args:
        model (Generator or Detector): The model to get the device from.
    Returns:
        Device: The device of the model, e.g., 'cpu' or 'cuda'.
    """
    if hasattr(model, "device"):
        device = model.device
    
    elif hasattr(model, "model"):
        if hasattr(model.model, "device"):
            device = model.model.device
        else:
            try:
                device = next(model.model.parameters()).device
            except Exception:
                try:
                    # If no parameters, try buffers
                    return next(model.model.buffers()).device
                except Exception:
                    # If neither parameters nor buffers exist, return CPU as default
                    return "cpu"
    else:
        return "cpu"

    if isinstance(device, torch.device):
        return device.type
    elif isinstance(device, str):
        return device
    else:
        return str(device) if hasattr(device, "__str__") else str(type(device))


def get_dtype(model: Union["Generator", "Detector"]) -> Optional[torch.dtype]:
    """
    Get the dtype of the model.
    Args:
        model (Generator or Detector): The model to get the dtype from.
    Returns:
        Optional[torch.dtype]: The dtype of the model, e.g., torch.float32 or None if not available.
    """
    if hasattr(model, "dtype"):
        return model.dtype
    elif hasattr(model, "model") and hasattr(model.model, "dtype"):
        return model.model.dtype
    elif hasattr(model, "model") and hasattr(model.model, "parameters"):
        return next(model.model.parameters()).dtype
    else:
        return None


def to_device(obj: T, device: str, dtype: Optional[torch.dtype] = None) -> T:
    """
    Recursively convert all tensors in the object to the device of the model.
    Args:
        obj (T): The object to convert.
    Returns:
        T: The object with all tensors converted to the device of the model.
    """
    
    if isinstance(obj, torch.Tensor):
        dtype = dtype or obj.dtype
        return obj.to(device, dtype=dtype)  # type: ignore
    elif isinstance(obj, list) and isinstance(obj[0], torch.Tensor):
        dtype = dtype or obj[0].dtype
        return [x.to(device, dtype=dtype) for x in obj]  # type: ignore
    elif isinstance(obj, dict):
        dtype = dtype or next(iter(obj.values())).dtype
        return {k: v.to(device, dtype=dtype) for k, v in obj.items()}  # type: ignore
    elif isinstance(obj, (Generator, Detector)):
        if hasattr(obj, "to"):
            obj.to(device)
        elif hasattr(obj, "model") and hasattr(obj.model, "to"):
            obj.model.to(device)
        return obj
    else:
        return obj


def dist_is_supported() -> bool:
    """
    Check if the current environment supports distributed setting.
    Returns:
        bool: True if distributed training is supported, False otherwise.
    """
    return torch.cuda.is_available() or torch.backends.mps.is_available()


def auto_device(device: Optional[str] = None) -> str:
    """
    Returns the device string, defaulting to 'cuda' if available.
    If a specific device is provided, it returns that device.
    """
    if device is not None:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def filter_by_index(data: Union[torch.Tensor, List[torch.Tensor]], indices: List[int]) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Filter the data by the given indices.
    Args:
        data (Union[torch.Tensor, List[torch.Tensor]]): The data to filter.
        indices (List[int]): The indices to filter by.
    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: The filtered data.
    """
    if isinstance(data, torch.Tensor):
        return data[indices]  # type: ignore
    elif isinstance(data, list):
        return [data[i] for i in indices]
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def parse_page_options(save_ids: str) -> List[Tuple[int, int]]:
    """
    Parse the page options string and return start and end indices.

    Args:
        save_ids (str): The input string containing page options, e.g., "0,3,4,5,10-20"

    Returns:
        list: A list of tuples containing start and end indices for each range
    """
    # Split the input string into individual ranges
    if "," not in save_ids and "-" not in save_ids:
        # If there are no commas or hyphens, treat the entire string as a single range
        return [(int(save_ids), int(save_ids))]
    ranges = save_ids.split(",")
    # Initialize an empty list to store the parsed ranges
    parsed_ranges = []
    # Iterate over each range
    for r in ranges:
        # Check if the range contains a hyphen (i.e., it's a range)
        if "-" in r:
            # Split the range into start and end indices
            start, end = map(int, r.split("-"))
            # Append the parsed range to the list
            parsed_ranges.append((start, end))
        else:
            # If it's not a range, append a tuple with the single index as both start and end
            parsed_ranges.append((int(r), int(r)))
    return parsed_ranges


def extract_numbers(intervals):
    """
    Given a list of closed intervals (tuples of two integers),
    return a sorted list of natural numbers (including 0) that lie in any of the intervals.
    """
    numbers = set()  # Use a set to avoid duplicates
    for start, end in intervals:
        for n in range(start, end + 1):  # Closed interval: include end
            if n >= 0:
                numbers.add(n)
    return sorted(numbers)


def is_index_in_range(index: int, ranges: List[Tuple[int, int]]) -> bool:
    """
    Check if an index belongs to any of the given ranges.

    Args:
        index (int): The page index to check
        ranges (list): A list of tuples representing the parsed ranges

    Returns:
        bool: True if the index belongs to any range, False otherwise
    """
    # Iterate over each range
    for start, end in ranges:
        # Check if the index falls within the current range
        if start <= index <= end:
            # If it does, return True immediately
            return True

    # If the index doesn't belong to any range, return False
    return False
