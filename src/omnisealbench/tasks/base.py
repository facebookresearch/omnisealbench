# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import inspect
import os
import shutil
from abc import ABC, abstractmethod
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Optional, Sequence, Tuple, Union)

import numpy as np
import pandas as pd
import torch

from omnisealbench.attacks.helper import no_attack_config
from omnisealbench.data.base import InputBatch
from omnisealbench.interface import (AverageMetric, Detector, Generator,
                                     ModelTypeT, ScoresT)
from omnisealbench.utils.cache import clear_global_cache
from omnisealbench.utils.common import (auto_device, dist_is_supported,
                                        filter_args, get_device,
                                        load_raw_results, load_results,
                                        parse_page_options,
                                        save_raw_results, save_results, to_device)
from omnisealbench.utils.distributed import (average_metrics, clear_cuda,
                                             gather_metrics, get_global_rank,
                                             is_local_execution,
                                             safe_init_distributed)
from omnisealbench.utils.distributed_logger import (
    rank_zero_print, setup_distributed_logging_and_exceptions)
from omnisealbench.utils.model_card import get_model

setup_distributed_logging_and_exceptions()


# A data reader is a function or Callable class that
# returns an iterable of contents, and optionally a dictionary of metadata.
DataReaderT = Callable[
    ...,
    Union[
        Iterable[InputBatch],  # only emit iterable of contents
        Tuple[Iterable[InputBatch], Dict[str, Any]]  # return both content iterator and metadata
    ]   
]

METRICS_FILE = "metrics.json"
ID_COLUMN = "idx"

if TYPE_CHECKING:
    from rich.progress import Progress

    from omnisealbench.utils.slurm import JobConfig, Launcher

WATERMARKED_KEY = "watermarked"

def random_message(size: int) -> torch.Tensor:
    return torch.randint(0, 2, (size,), dtype=torch.float32)


def test_message(size: int) -> torch.Tensor:
    key = np.array([i % 2 for i in range(size)], dtype=np.float32)
    return torch.from_numpy(key)


def generate_message(size: int) -> torch.Tensor:
    method = os.getenv("OMNISEAL_MESSAGE", "random")
    if method == "random":
        return random_message(size)
    elif method == "test":
        return test_message(size)
    else:
        raise ValueError(f"Unknown message generation method: {method}")


def get_task_progress(enable_rich_progress: bool, tasks_cnt: int):
    if enable_rich_progress:
        try:
            from omnisealbench.utils.progress_tools import EvaluationProgress
            progress = EvaluationProgress(runtime_tasks=tasks_cnt)
            task_progress = progress.attack_variant_progress
        except (ImportError, ModuleNotFoundError) as exc:
            raise ValueError(
                "It seems you enable rich progress with 'show_progress=True'."
                " Please install rich package to use progress bar: pip install rich"
            ) from exc
    else:
        progress = nullcontext()
        task_progress = None

    return progress, task_progress


@dataclass
class TaskConfig:
    metrics: Sequence[str] = field(default_factory=list)
    """List of metrics to be computed for the task."""

    metrics_device: Optional[str] = None
    """Device to compute the metrics on. If None, use data's device."""

    seed: Optional[int] = None
    """Random seed for the task. If not set, a random seed will be used."""

    result_dir: Union[str, Path] = ""
    """Directory where the results of the watermark generation will be saved.
    If empty (default), results will not be saved."""
    
    data_subdir: Union[str, Path] = "data"
    """Subdirectory to store intermediate data files (attacked contents)."""

    overwrite: bool = False
    """Whether to overwrite the existing results in the result_dir."""

    result_filename: str = ""
    """Filename for the full raw results of the task.
    This config is only used if result_dir is not empty."""

    final_results_filename: str = METRICS_FILE
    """Filename for the final results of the task."""

    ids_to_save: Optional[str] = "all"
    """Ids to save the watermarked contents and original contents. By default, all ids are saved.
    Other options are 'id1,id2,...' or 'id1-id2,id3,...' to specify a range of ids.
    When set to None, no contents will be saved."""

    def __post_init__(self):
        self.save_all = (self.ids_to_save == "all")
        if self.ids_to_save == 'all' or self.ids_to_save is None:
            self.ids_range_to_save = None
        else:
            self.ids_range_to_save = parse_page_options(str(self.ids_to_save))


class Task(ABC):
    """
    A unit of eval task (generation or detection) that can be executed on a worker node.
    It contains the model, runtime requirements (optional), and the dataset
    to be evaluated against. Each task is instantiated with a task configuration
    """

    def name(self) -> str:
        """Name of the task unit, as can be seen in SLURM monitoring tools"""
        return self.__class__.__name__

    def __init__(
        self,
        config: TaskConfig,
        data_reader: Optional[DataReaderT] = None,
        num_workers: int = 0,
        **data_args,
    ):
        """
        Build a task from the given configuration and optional data reader.
        Args:
            config (TaskConfig): Configuration for the task, consisting of how to handle .
            data_reader (Optional[DataReaderT]): Optional data reader to process the dataset.
                This can be any function that returns an iterable of contents. If not provided,
                a default data handler will be used, and each task implementation should
                ensure that the data handler provides compatible contents for the model
            num_workers (int): Number of workers to use for data processing. If not set, it will
            data_args (Any): Additional arguments for the data handler, such as dataset paths,
                preprocessing functions, etc. These arguments will be passed to the data handler
                when it is called to process the dataset.
        Returns:
            Task: An instance of the task.
        """
        self.config = config
        self.data_reader = data_reader
        self.data_args = data_args
        self.num_workers = max(num_workers or os.cpu_count() or 1, 1)

        # Hacky way to turn off the pre_eval function
        self.has_pre_eval = False

    @abstractmethod
    def make_sub_tasks(self, config: Optional[TaskConfig] = None) -> Optional[Sequence["TaskConfig"]]:
        """
        Create sub-tasks from the current task configuration.
        This is useful for tasks that can be split into smaller units, such as
        running multiple models or datasets in parallel.

        Returns:
            Optional[List[Task]]: A list of sub-tasks or None if we only have one task.
        """
        return None

    async def schedule(
        self,
        model_class_or_function: ModelTypeT,
        task_run_name: Optional[str] = None,
        launcher: Optional["Launcher"] = None,
        runtime: Optional["JobConfig"] = None,
        show_progress: bool = False,
        auto_distributed: bool = True,
        **kwargs,
    ) -> Any:
        """
        Schedule a task for execution in an asynchronous (and distributed) manner using an optional launcher.
        Args:
            model_card_or_path (string or file): name of a model card or path to a model checkpoint to
                instantiate the model. This can be a string name of a registered model, or a path to a
                model checkpoint file. The model will be instantiated using the model_class_or_function.

            model_class_or_function (ModelTypeT): The model class or function to instantiate the model.
                This can be a class that implements the Generator or Detector protocol, or a function that
                takes the model_card_or_path and model_args as input and returns an instance of the model.
                The model should be compatible with the task, i.e. it should implement the generate_watermark or
                detect_watermark methods as required by the task.

            task_run_name (Optional[str]): Optional name for the task run, used for logging and tracking.

            launcher (Optional[Launcher]): Optional launcher to use for scheduling the task. If this is
                provided, the task will be scheduled as a job in the SLURM cluster.

            runtime (Optional[JobConfig]): Optional runtime requirements for the task.
                Expected behaviour:
                - If this is not provided, the behaviour depends on the launcher: If a
                launcher is provided, the task will be scheduled using a default runtime
                config, otherwise, it will be run locally.
                - If this is provided and launcher is None, raises an exception. Otherwise
                the task will be scheduled with the given runtime config.

            show_progress (bool): Whether to show progress during execution.

            model_args (Any): Additional arguments to pass to the model instantiation.

        Returns:
            A job object  that is registered in the launcher's registry
        """
        if task_run_name is None:
            task_run_name = self.name()

        # If launcher is None, run the task locally
        if launcher is None:
            _auto_distributed = auto_distributed and dist_is_supported()
            return self.eval(
                model_class_or_function=model_class_or_function,
                config=self.config,
                show_progress=show_progress,
                auto_distributed=_auto_distributed,
                **kwargs,
            )

        # Run in SLURM cluster

        sub_tasks = self.make_sub_tasks()

        if sub_tasks is None:
            res = self._load_results(self.config)
            if res is not None:
                return res

            return await launcher.run(
                task_name=task_run_name,
                task_func=self.eval,
                runtime=runtime,
                model_class_or_function=model_class_or_function,
                config=None,
                show_progress=show_progress,
                auto_distributed=auto_distributed,
                **kwargs,
            )

        # Run each sub-task in parallel in the SLURM cluster
        rank_zero_print(f"Number of sub-tasks for {task_run_name}: {len(sub_tasks)}")
        results: List[Dict[str, AverageMetric]] = [None] * len(sub_tasks)  # type: ignore
        raw_results: List[ScoresT] = [None] * len(sub_tasks)  # type: ignore
        task_ids_to_run = []
        for i, sub_task in enumerate(sub_tasks):
            _res = self._load_results(sub_task)
            if _res:
                results[i], raw_results[i] = _res
            else:
                task_ids_to_run.append(i)

        # Run the task for each sub-config if results are not already computed

        if task_ids_to_run:
            rank_zero_print(f"{len(task_ids_to_run)} sub detection tasks are not finished and will be scheduled for {task_run_name}...")
            if self.has_pre_eval:
                rank_zero_print(f"Running first generation task for {task_run_name} as the preparation step...")
                await launcher.run(
                    task_name=f"{task_run_name}_pre",
                    task_func=self.pre_eval,
                    model_class_or_function=model_class_or_function,
                    config=None,
                    runtime=runtime,
                    show_progress=show_progress,
                    auto_distributed=auto_distributed,
                    **kwargs,
                )

            # Launch tasks in batches to avoid overloading the cluster
            max_jobs = getattr(launcher, "max_jobarray_jobs", None)
            if not max_jobs or max_jobs <= 0:
                # fallback: run all at once
                max_jobs = len(task_ids_to_run)

            # run in chunks of at most max_jobs
            for start in range(0, len(task_ids_to_run), max_jobs):
                batch_ids = task_ids_to_run[start : start + max_jobs]
                batch_tasks = [
                    launcher.run(
                        task_name=f"{task_run_name}_{task_id}",
                        task_func=self.eval,
                        runtime=runtime,
                        config=sub_tasks[task_id],
                        model_class_or_function=model_class_or_function,
                        show_progress=show_progress,
                        auto_distributed=auto_distributed,
                        **kwargs,
                    )
                    for task_id in batch_ids
                ]
                if not batch_tasks:
                    continue
                async_results = await asyncio.gather(*batch_tasks)
                for i, res in zip(batch_ids, async_results):
                    results[i], raw_results[i] = res

        avg_results = self.__aggregate_run_metrics__(results)

        if self.config.result_dir is not None:
            formatted_scores = self.print_scores(raw_results)
            formatted_scores.to_csv(f"{self.config.result_dir}/report.csv", header=True, index=False)

        return avg_results, raw_results

    def pre_eval(
        self,
        model_class_or_function: ModelTypeT,
        config: Optional[TaskConfig] = None,
        show_progress: bool = False,
        auto_distributed: bool = False,
        **kwargs,
    ):
        """
        [Optional] Run of pre-evaluation task. Currently noops, this method
        is reserved for future extension
        """
        pass

    def check_subtasks(self):
        """Get the number of sub-tasks that have been finished and sub-tasks that need to be run"""
        sub_tasks = self.make_sub_tasks()
        if sub_tasks is None:
            total = 1
            finished = 1 if self._load_results(self.config) is not None else 0
        else:
            total = len(sub_tasks)
            finished = sum(1 for sub_task in sub_tasks if self._load_results(sub_task) is not None)
        return total, total - finished

    def eval(
        self,
        model_class_or_function: ModelTypeT,
        config: Optional[TaskConfig] = None,
        show_progress: bool = False,
        auto_distributed: bool = False,
        **kwargs,
    ):

        # TODO: This is a temporary fix to find out the model type
        if "as_type" in kwargs:
            model_type = kwargs.pop("as_type")
        else:
            _name = self.__class__.__name__
            if "Generation" in _name:
                model_type = "generator"
            elif "Detection" in _name:
                model_type = "detector"
            else:
                model_type = None

        # When calling eval(), model will be placed on the appropriate device automatically, unless when auto_distributed is True
        device = kwargs.pop("device", None)
        if auto_distributed:
            assert device is None or device != "cpu", f"Invalid device argument when auto_distributed=True ({kwargs['device']})"
            dist_ctx = safe_init_distributed
            kwargs.pop("device", None)  # remove device argument if any
            model = get_model(model_class_or_function, as_type=model_type, **kwargs)
        else:
            device = auto_device(device)
            model = get_model(model_class_or_function, as_type=model_type, device=device, **kwargs)
            dist_ctx = nullcontext

        if config is None:
            return self(
                model,
                show_progress=show_progress,
                auto_distributed=auto_distributed,
                **kwargs,
            )

        res = self._load_results(config)
        if res:
            return res

        elif config.overwrite and config.result_dir and Path(config.result_dir).exists():
            rank_zero_print(
                f"Result {config.result_dir} exists and will be overriden as --overwrite is set"
            )
            if get_global_rank() == 0:
                shutil.rmtree(config.result_dir, ignore_errors=True)

        with dist_ctx():
            if auto_distributed:
                to_device(model, auto_device())
            return self.__call_one_run__(model, config, show_progress=show_progress, **kwargs)

    def __call__(
        self,
        model: Union[Generator, Detector],
        config: Optional[TaskConfig] = None,
        batches: Optional[Iterable[InputBatch]] = None,
        show_progress: bool = False,
        auto_distributed: bool = False,
        **kwargs,
    ):
        config = config or self.config

        # Only run if results do not exist or overwrite is set to True
        res = self._load_results(config)
        if res:
            return res

        elif config.overwrite and config.result_dir and Path(config.result_dir).exists():
            rank_zero_print(
                f"Result {config.result_dir} exists and will be overriden as --overwrite is set"
            )
            shutil.rmtree(config.result_dir, ignore_errors=True)

        sub_configs = self.make_sub_tasks(config)

        if auto_distributed:
            assert dist_is_supported(), "Distributed is not supported"
            dist_ctx = safe_init_distributed
        else:
            dist_ctx = nullcontext

        with dist_ctx():
            if auto_distributed:
                to_device(model, auto_device())
                
            # There is only one task to run
            if sub_configs is None:
                
                return self.__call_one_run__(model, config, batches, show_progress, **kwargs)

            # There are multiple sub-tasks to run            
            return self.__call_multiple_runs__(model, sub_configs, batches, show_progress, **kwargs)  # type: ignore


    def __call_multiple_runs__(
        self,
        model: Union[Generator, Detector],
        sub_configs: Sequence[TaskConfig],
        batches: Optional[Iterable[InputBatch]] = None,
        show_progress: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, AverageMetric], List[ScoresT]]:

        results: List[Dict[str, AverageMetric]] = [None] * len(sub_configs)  # type: ignore
        raw_results: List[ScoresT] = [None] * len(sub_configs)  # type: ignore
        for i, sub_config in enumerate(sub_configs):
            _res = self._load_results(sub_config)
            if _res:
                results[i], raw_results[i] = _res

            elif sub_config.overwrite and sub_config.result_dir and Path(sub_config.result_dir).exists():
                rank_zero_print(
                    f"Result {sub_config.result_dir} exists and will be overriden as --overwrite is set"
                )
                shutil.rmtree(sub_config.result_dir)

        enable_rich_progress = show_progress and is_local_execution() and (get_global_rank() == 0)
        # Run the task for each sub-config if results are not already computed

        progress, task_progress = get_task_progress(enable_rich_progress, tasks_cnt=len(sub_configs))
        with progress:  # type: ignore
            for i, sub_config in enumerate(sub_configs):
                if results[i] is None:

                    # Run the task for each sub-config
                    results[i], raw_results[i] = self.__call_one_run__(
                        model,
                        config=sub_config,
                        batches=batches,
                        show_progress=show_progress,
                        task_progress=task_progress,
                        **kwargs,
                    )
                if enable_rich_progress:
                    progress.update(
                        det_time=sum(raw_results[i]["det_time"]),
                        decoder_time=sum(raw_results[i]["decoder_time"]),
                        attack_time=sum(raw_results[i]["attack_time"]),
                        quality_time=sum(raw_results[i]["qual_time"]),
                    )

        # Aggregate results from all sub-configs
        avg_results = self.__aggregate_run_metrics__(results)

        if get_global_rank() == 0 and self.config.result_dir is not None:
            self._save_metrics(avg_results, self.config)

        return avg_results, raw_results

    def __call_one_run__(
        self,
        model: Union[Generator, Detector],
        config: Optional[TaskConfig] = None,
        batches: Optional[Iterable[InputBatch]] = None,
        show_progress: bool = False,
        task_progress: Optional["Progress"] = None,
        **kwargs,
    ) -> Tuple[Dict[str, AverageMetric], ScoresT]:

        config = config or self.config

        if batches is None:
            assert self.data_reader is not None, "Data reader must be provided if not using batches directly"
            if inspect.isfunction(self.data_reader):
                params = inspect.signature(self.data_reader).parameters
                _kwargs = filter_args(params, **self.data_args)
                data_result = self.data_reader(**_kwargs)
            else:
                assert isinstance(self.data_reader, Iterable), "Data reader must be an iterable or a callable"
                data_result = self.data_reader

            if isinstance(data_result, tuple):
                batches, metadata = data_result
            else:
                batches, metadata = data_result, {}
        else:
            metadata = {}
        
        # Allow inline change of parameters
        metadata = {**metadata, **kwargs}

        scores = self.run(
            model=model,
            batches=batches,  # type: ignore
            config=config,  # type: ignore—
            show_progress=show_progress,
            task_progress=task_progress,
            **metadata,
        )

        # Remove the original and watermarked tensors from the results before gathering the metric scores from
        # all devices
        scores.pop("input", None)
        scores.pop("watermarked", None)

        # cleanup memory
        clear_global_cache()
        if get_device(model) == "cuda":
            clear_cuda()

        # Gather raw results from all nodes
        raw_results = gather_metrics(scores)

        # average results across all nodes
        avg_results = average_metrics(scores)


        if get_global_rank() == 0 and config.result_dir and len(raw_results) > 0:  # type: ignore
            filename = Path(config.result_dir) / config.result_filename  # type: ignore
            save_raw_results(raw_results, filename)

            self._save_metrics(avg_results, config)

        return avg_results, raw_results

    def __aggregate_run_metrics__(self, results: List[Dict[str, AverageMetric]]) -> Dict[str, AverageMetric]:
        """
        Aggregate the results from all sub-tasks.
        This is a helper method to be used in the __call__ method.
        """

        assert len(results) > 0, "Results must not be empty"
        avg_results: Dict[str, AverageMetric] = deepcopy(results[0])

        if len(results) > 1:
            for result in results[1:]:
                for k, v in result.items():
                    if k not in avg_results:
                        avg_results[k] = AverageMetric(0, 0, 0)
                    if v.count > 0:  # some metric point is just noisy, skip them
                        avg_results[k].update(v.avg, v.count, v.square)

        return avg_results

    def _save_metrics(self, metrics: Dict[str, AverageMetric], config: TaskConfig) -> None:
        # This method can be overridden in subclasses to save metrics in a specific format
        filename = Path(config.result_dir) / config.final_results_filename

        formatted_metrics = {k: (v.value, v.count, v.square) for k, v in metrics.items()}
        save_results(formatted_metrics, filename)

    def _load_results(self, config: TaskConfig) -> Optional[Tuple[Dict[str, AverageMetric], ScoresT]]:
        if (
            config.result_dir
            and Path(config.result_dir).joinpath(config.final_results_filename).exists()
            and not config.overwrite
        ):
            filename = Path(config.result_dir) / config.final_results_filename

            metrics = load_results(filename)

            # Convert to AverageMetric
            avg_metrics = {k: AverageMetric(avg=v, count=c, square=s) for k, (v, c, s) in metrics.items()}  # type: ignore

            raw_filename = Path(config.result_dir) / config.result_filename
            if raw_filename.exists():
                raw_results = load_raw_results(raw_filename)
            else:
                raw_results = {}

            return avg_metrics, raw_results  # type: ignore
        else:
            return None

    @abstractmethod
    def run(
        self,
        model: Union[Generator, Detector],
        batches: Iterable[InputBatch],
        config: TaskConfig,
        show_progress: bool = False,
        task_progress: Optional["Progress"] = None,
        **kwargs: Any,
    ) -> ScoresT:
        """
        Execute the task on the given model and batches of contents.
        Args:
            model (Union[Generator, Detector]): The model to use for the task.
            batches (Iterable[ContentT]): An iterable of contents to process.
            config (Optional[TaskConfig]): By default, the task configuration is used, but user
                can override it with a different configuration.
            show_progress (bool): Whether to show progress during execution.
            task_progress (Optional[Progress]): Optional progress bar to use for tracking progress.
        Returns:
            ScoresT: The average result of the task execution
        """
        ...

    @abstractmethod
    def print_scores(self, scores: Union[ScoresT, List[ScoresT]], **kwargs) -> pd.DataFrame:
        pass
