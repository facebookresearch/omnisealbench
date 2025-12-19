# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from time import time
from typing import (TYPE_CHECKING, Iterable, List, Literal, Optional, Sequence,
                    Union)

import torch

from omnisealbench.data.base import InputBatch
from omnisealbench.interface import Generator, ScoresT, cast_input
from omnisealbench.tasks.base import (ID_COLUMN, Task, TaskConfig,
                                      generate_message)
from omnisealbench.utils.common import (filter_by_index, get_device, get_dtype,
                                        is_index_in_range, resolve, set_seed,
                                        tensor_to_message, to_device,
                                        to_python_type)

if TYPE_CHECKING:
    import pandas as pd
    from rich.progress import Progress

@dataclass
class WatermarkGenerationConfig(TaskConfig):
    """
    Configuration class for watermark generator models.
    This class holds the preprocessing, postprocessing and metrics configuration.
    """

    result_filename: str = "watermark_results.jsonl"
    """Store per-sample results of the watermark generation task in this file."""

    message_size: Union[int, Sequence[int], None] = None
    """Many watermark generator evaluation runs with a randomly generated message.
    If set, the message will be generated with this size.
    If multiple sizes are provided, the watermark generation will be run for each size in
    a sub-task."""

    how_to_generate_message: Literal["per_batch", "per_dataset"] = "per_batch"
    """
    How to generate the message for the evaluation. 
    If "per_batch", a new random message will be generated for each sample in the dataset.
    If "per_dataset", a single random message will be generated for the whole dataset.
    """

    data_subdir: Union[str, Path] = ""
    """Sub directory to store the original data. Most of the time, this is not needed,
    especially when the original data is already downloaded locally. So you should
    consider set the data_subdir value only if necessary."""

    watermark_subdir: Union[str, Path] = "watermarks"
    """Directory where the watermarks will be saved, relative to the result_dir.
    This is only used if result_dir is not empty."""

    keep_data_in_memory: bool = False
    """If result_dir is not specified, we can set this flag to pass the contents (original
    and watermarked data) to the result"""

    def __post_init__(self):
        # If the user specifies the result_dir and not ids_to_save, by default we should save all
        # the result. This logic however is only applied to generation tasks.
        if self.keep_data_in_memory:
            assert not self.result_dir, f"result_dir for watermark generation must not be specified if keep_data_in_memory is True, got {self.result_dir}"
            if self.ids_to_save or self.ids_to_save != "all":
                raise ValueError(f"ids_to_save must not be specified if keep_data_in_memory is True, got {self.ids_to_save}")
            self.ids_to_save = None  # keep all in memory, save none to hard disk

        if self.result_dir:
            assert not self.keep_data_in_memory, f"result_dir for watermark generation ({self.result_dir}) and 'keep_in_data_memory' cannot co-exist"
            if not self.ids_to_save:
                self.ids_to_save = "all"  # Save all by default
        super().__post_init__()


class WatermarkGeneration(Task):
    """
    Base class for watermark generation, i.e. getting cover contents (images, audio, etc.)
    and generated the watermarked contents with a secret message. Depending on whether the
    watermark generation is post-hoc or not, the results can be saved directly to the
    result directory or returned as a list of dictionaries containing the watermarked contents.
    """

    config: WatermarkGenerationConfig

    def make_sub_tasks(self, config: Optional[WatermarkGenerationConfig] = None) -> Optional[Sequence[WatermarkGenerationConfig]]:  # type: ignore
        """
        Create sub-tasks for watermark generation based on different message sizes.
        If the message_size is a list, each size will be used to create a sub-task.
        If it is a single integer, only one sub-task will be created.
        Returns:
            Optional[List[WatermarkGenerationConfig]]: A list of sub-tasks or None if no sub-tasks are created.
        """
        config = config or self.config
        if isinstance(config.message_size, list):
            sub_tasks = []

            for msg_size in config.message_size:
                if not isinstance(msg_size, int):
                    raise ValueError(
                        f"Invalid message size {msg_size} in {config.message_size}. "
                        "Expected an integer or a list of integers."
                    )
                config_ = deepcopy(config)
                config_.message_size = msg_size
                config_.watermark_subdir = f"watermark_{msg_size}"
                config_.result_filename = f"watermark_{msg_size}.jsonl"
                config_.final_results_filename = f"{config_.final_results_filename[:-4]}_{msg_size}.json"

                sub_tasks.append(config_)
            return sub_tasks

        return None

    def run(  # type: ignore
        self,
        model: Generator,  # type: ignore
        batches: Iterable[InputBatch],
        config: TaskConfig,
        show_progress: bool = False,
        task_progress: Optional["Progress"] = None,
        **kwargs,
    ) -> ScoresT:
        """
        Run the watermark generation task on the given model and batches of contents.
        Args:
            model (Generator): The watermark generator model to use.
            batches (Iterable[ContentT]): An iterable of contents to process.
            show_progress (bool): Whether to show progress during execution.
        Returns:
            TaskResult: The result of the watermark generation, which can be a dictionary or a list of dictionaries.
        """
        results = defaultdict(list)  # type: ignore

        # Inline override of config to allow for dynamic configuration changes
        _old_config = self.config
        self.config = config  # type: ignore

        if self.config.seed is not None:
            set_seed(self.config.seed)

        device = get_device(model)
        dtype = get_dtype(model)

        message: Optional[torch.Tensor] = None
        if not isinstance(self.config.message_size, int):
            assert hasattr(model, "nbits"), "If the message size is not set, the model must have a nbits attribute."
            self.config.message_size = model.nbits

        if self.config.how_to_generate_message == "per_dataset":
            # message = torch.zeros((self.config.message_size, ), dtype=torch.float32)
            
            message = generate_message(self.config.message_size)  # type: ignore

        # We save the results if the "result_dir" is specified, and "ids_to_save" is not None
        save_results: bool = (
            bool(self.config.result_dir and self.config.watermark_subdir)
            and (self.config.save_all or self.config.ids_to_save is not None)
        )
        for batch in batches:
            if self.config.how_to_generate_message == "per_batch":
                assert isinstance(self.config.message_size, int), "message size must be set"
                message = generate_message(self.config.message_size)

            # Some models expect a specific type of input, e.g. a list of tensors or numpy arrays.
            # So we inspect the model annotation and convert the input accordingly.
            orig = cast_input(model.generate_watermark, batch.data)

            # Move to model's device just before the wm generation
            orig = to_device(orig, device, dtype=dtype)  # type: ignore
            message = to_device(message, device, dtype=dtype)  # type: ignore

            indices = batch.indices.cpu().tolist()  # type: ignore

            # Run the watermark generation
            watermarked = model.generate_watermark(orig, message=message)

            assert isinstance(watermarked, (torch.Tensor, list)), "watermark generation model must return a tensor or a dictionary."  # fmt: off

            # Compute the scores
            if self.config.metrics_device:
                orig = to_device(orig, self.config.metrics_device)
                watermarked = to_device(watermarked, self.config.metrics_device)
            
            quality_btime = time()
            scores = self.compute_metrics(orig, watermarked, message, **kwargs)
            batch_quality_time = time() - quality_btime

            for k, v in scores.items():  # type: ignore
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().tolist()  # type: ignore
                elif isinstance(v, float):
                    v = [v] * len(batch.indices)  # type: ignore
                results[k].extend(v)

            # Add runtime results
            quality_time_per_item = round(batch_quality_time / len(batch.indices), 4)
            results["qual_time"].extend([quality_time_per_item] * len(batch.indices))

            # Add indices to results for easier debugging and reports
            results[ID_COLUMN].extend(indices)  # type: ignore

            # Save the watermarked contents. message can now be placed on CPU
            message = message.cpu() if message is not None else None

            if self.config.keep_data_in_memory:
                results["input"].append(to_device(orig, "cpu"))  # type: ignore
                results["watermarked"].append(to_device(watermarked, "cpu"))  # type: ignore

            if save_results:
                if not self.config.save_all:
                    # In this case, 'ids_to_save' must not be None and we have already parsed it to self.config.ids_range_to_save
                    local_indices = [i for i, idx in enumerate(indices) if is_index_in_range(idx, self.config.ids_range_to_save)]  # type: ignore
                    indices = [idx for idx in indices if is_index_in_range(idx, self.config.ids_range_to_save)]  # type: ignore
                else:
                    local_indices = list(range(len(indices)))  # type: ignore

                if indices:
                    orig_to_save = filter_by_index(orig, local_indices)  # type: ignore
                    watermarked_to_save = filter_by_index(watermarked, local_indices)  # type: ignore

                    self.save_data(orig_to_save, watermarked_to_save, indices, **kwargs)
                    self._save_messages(message, indices)  # type: ignore

        # Change back to the old config if it was set
        self.config = _old_config

        return results

    def print_scores(
        self,
        scores: Union[ScoresT, List[ScoresT]],
        config: Optional[WatermarkGenerationConfig] = None,
        message_sizes: Optional[List[str]] = None,
        **kwargs,
    ) -> "pd.DataFrame":  # type: ignore
        # Get the subtasks from self.config and sync with the results
        config = config or self.config

        import pandas as pd

        if message_sizes is None:
            sub_configs = self.make_sub_tasks(config)
            if sub_configs is None:
                sub_configs = [config]
                assert isinstance(scores, dict), "If no sub-tasks are created, scores must be a dictionary of scores"
                scores = [scores]
            message_sizes = self.__get_message_sizes_as_string(sub_configs)

        assert len(scores) == len(message_sizes), (
            f"Number of scores {len(scores)} does not match number of sub-tasks {len(message_sizes)}."
        )

        dfs = []
        for message_size, scores_ in zip(message_sizes, scores):
            scores_ = to_python_type(scores_)
            df_ = pd.DataFrame(scores_)
            df_["message_size"] = message_size
            dfs.append(df_)

        return pd.concat(dfs, ignore_index=True)

    def __get_message_sizes_as_string(self, sub_configs: Sequence[WatermarkGenerationConfig]) -> List[str]:
        """
        Get the message sizes as a list of strings from the sub-configs.
        Args:
            sub_configs (Sequence[WatermarkGenerationConfig]): The sub-configs to get the message sizes from.
        Returns:
            List[str]: A list of message sizes as strings.
        """
        return [str(sub_config.message_size) for sub_config in sub_configs]

    def compute_metrics(
        self,
        orig: torch.Tensor,
        watermarked: torch.Tensor,
        message: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ScoresT:
        """
        Compute the metrics for the given results of watermark generation.
    Args:
            results (ResultT): The results of the watermark generation.
            kwargs (Any): Additional metadata from the data reader, such as sampling rate,
                original contents, etc. This can be used to compute metrics that require
        Returns:
            Dict[str, Any]: A dictionary containing the computed metrics.
        """
        raise NotImplementedError(
            "compute_metrics method must be implemented in the subclass of WatermarkGeneration."
        )

    def _save_messages(
        self,
        messages: torch.Tensor,
        indices: Sequence[int],
        prefix: str = "message",
    ) -> None:
        """
        Save the messages to the specified directory.
        Args:
            messages (torch.Tensor): The messages to save.
            indices (Sequence[int]): The indices of the messages.
            message_dir (Path): The directory to save the messages to.
            prefix (str): The prefix for the message files.
        """
        message_dir = resolve(self.config.result_dir) / self.config.watermark_subdir
        message_dir.mkdir(exist_ok=True, parents=True)
        if messages.ndim == 1:
            messages = messages.repeat(len(indices), 1)  # type: ignore
        for idx, message in zip(indices, messages):
            file_path = message_dir / f"{prefix}_{idx}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(tensor_to_message(message))  # type: ignore

    def save_data(
        self,
        orig: Union[torch.Tensor, List[torch.Tensor]],
        watermarked: Union[torch.Tensor, List[torch.Tensor]],
        indices: List[int],
        **kwargs,  # type: ignore
    ) -> None:
        """
        Save the results of the watermark generation to the specified result directory.
        Args:
            watermarked_results (ResultT): The results of the watermark generation.
        """
        ...
