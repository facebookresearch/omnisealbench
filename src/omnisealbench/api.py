# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# A factory of different task pipelines
#
# mypy: disable-error-code="import-untyped, assignment"

import importlib
import inspect
from typing import TYPE_CHECKING, Callable, Literal, NamedTuple, Tuple

if TYPE_CHECKING:
    from omnisealbench.tasks.base import Task


class Entry(NamedTuple):
    description: str
    class_or_func_name: str


def list_tasks():
    """Print the available tasks and their descriptions."""
    for name, entry in _TASK_REGISTRY.items():
        print(f"- \"{name}\": {entry.description} (class/function: {entry.class_or_func_name})")


_TASK_REGISTRY = {
    ("generation", "audio"): Entry(
        "Running audio watermarking on a HuggingFace dataset and evaluating the quality of the generated watermarks.",
        "omnisealbench.tasks.audio.evaluate_audio_watermark_generation"
    ),
    ("detection", "audio"): Entry(
        "Running audio watermarking on a custom dataset and evaluating the quality and robustness of the generated watermarks.",
        "omnisealbench.tasks.audio.evaluate_audio_watermark_attacks_and_detection",
    ),
    ("default", "audio"): Entry(
        "Running audio watermarking end-to-end",
        "omnisealbench.tasks.audio.evaluate_audio_watermark_end2end",
    ),
    ("default", "image"): Entry(
        "Running image watermarking end-to-end",
        "omnisealbench.tasks.image.evaluate_image_watermark_end2end",
    ),
    ("generation", "image"): Entry(
        "Running image watermarking on a local dataset and evaluating the quality of the generated watermarks.",
        "omnisealbench.tasks.image.evaluate_image_watermark_generation",
    ),
    ("detection", "image"): Entry(
        "Running image watermarking on a local dataset and evaluating the quality of the generated watermarks.",
        "omnisealbench.tasks.image.evaluate_image_watermark_attacks_and_detection",
    ),
    ("generation", "video"): Entry(
        "Running video watermarking on a local dataset and evaluating the quality of the generated watermarks.",
        "omnisealbench.tasks.video.evaluate_video_watermark_generation",
    ),
    ("detection", "video"): Entry(
        "Running video watermarking on a local dataset and evaluating the quality of the generated watermarks.",
        "omnisealbench.tasks.video.evaluate_video_watermark_attacks_and_detection",
    ),
    ("default", "video"): Entry(
        "Running video watermarking end-to-end",
        "omnisealbench.tasks.video.evaluate_video_watermark_end2end",
    )
}


def _get_func(key: Tuple[str, str]) -> Callable:
    """Get the function for a given task name."""
    if key not in _TASK_REGISTRY:
        raise ValueError(f"Unknown task name: {key}")

    entry = _TASK_REGISTRY[key]
    module_name, func_name = entry.class_or_func_name.rsplit(".", 1)
    module_ = importlib.import_module(module_name)
    func = getattr(module_, func_name)
    assert inspect.isfunction(func), f"{func_name} is not a function"

    return func


def explain(
    name: Literal["generation", "detection", "default"],
    /,
    modality: str,
) -> str:
    func = _get_func((name, modality))
    return func.__doc__ or ""


def task(
    name: Literal["generation", "detection", "default"],
    /,
    modality: str,
    **kwargs,
) -> "Task":
    """Get the task pipeline for a given name. Depending on the name, the keyword arguments are
    the function arguments for the task class or function."""
    
    # TODO: Give a better name of the tasks

    assert name in ["generation", "detection", "default"], f"Unknown pipeline name: {name}"
    func = _get_func((name, modality))

    return func(**kwargs)  # type: ignore[call-arg]
