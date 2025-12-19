# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from abc import abstractmethod
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import os
from typing import Iterable, List, Optional, Sequence, Union

import pandas as pd

from omnisealbench.config import default_omniseal_cache
from omnisealbench.data.base import InputBatch
from omnisealbench.interface import Detector, Generator, ModelTypeT, ScoresT
from omnisealbench.tasks._detection import (
    ATTACK_RESULT_FILENAME,
    WatermarkAttacksAndDetection,
    WatermarkAttacksConfig,
    unroll_attack_configs,
)
from omnisealbench.tasks._generation import (
    WatermarkGeneration,
    WatermarkGenerationConfig,
)
from omnisealbench.tasks.base import DataReaderT, Task, TaskConfig
from omnisealbench.utils.common import auto_device, dist_is_supported, parse_page_options, resolve, to_device
from omnisealbench.utils.distributed import safe_init_distributed
from omnisealbench.utils.distributed_logger import rank_zero_print
from omnisealbench.utils.model_card import get_model


@dataclass
class End2EndWatermarkingConfig(WatermarkGenerationConfig, WatermarkAttacksConfig):
    """Configuration for the end-to-end watermark generation and detection"""

    cache_dir: str = ""
    """
    This property determines where intermediate watermarked data will be saved.
    If 'keep_data_in_memory' is set, no intermediate data is saved.
    If 'keep_data_in_memory' is set but 'cache_dir' is not, the value will
    be set internally to default_omniseal_cache/tmp/,  so that
    the result is accessible from all detection tasks over different SLURM nodes.
    """

    def __post_init__(self):
        if self.keep_data_in_memory:
            assert (
                not self.cache_dir
            ), f"keep_data_in_memory is set, expect empty cache_dir but get {self.cache_dir}"
            if self.overwrite:
                rank_zero_print(
                    "Warning: 'overwrite' is set to True, but 'keep_data_in_memory' is also True. "
                    "This means that intermediate watermarked data will not be saved, "
                    "and the overwrite option will have no effect."
                )
            self.ids_to_save = None

        if not self.keep_data_in_memory:
            # Default temporary directory
            if not self.cache_dir:
                dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
                self.cache_dir = str(resolve(f"{default_omniseal_cache}/tmp/{dt_str}"))
            # By default all results will be saved

        self.save_all = self.ids_to_save == "all"
        if self.ids_to_save == "all" or self.ids_to_save is None:
            self.ids_range_to_save = None
        else:
            self.ids_range_to_save = parse_page_options(str(self.ids_to_save))


def generation_config(config: End2EndWatermarkingConfig) -> WatermarkGenerationConfig:
    return WatermarkGenerationConfig(
        metrics=[],  # No metrics is calculated, this is done in identity task in detection step
        seed=config.seed,
        result_dir=config.cache_dir,
        overwrite=config.overwrite,
        result_filename="watermark_results.jsonl",
        ids_to_save="all",  # by default, save all watermarked data for next steps
        data_subdir="data",  # subdirectory for storing data
        message_size=config.message_size,
        how_to_generate_message=config.how_to_generate_message,
        watermark_subdir=config.watermark_subdir,
        keep_data_in_memory=config.keep_data_in_memory,
    )


def detection_config(config: End2EndWatermarkingConfig) -> WatermarkAttacksConfig:
    return WatermarkAttacksConfig(
        metrics=config.metrics,
        metrics_device=config.metrics_device,
        seed=config.seed,
        result_dir=config.result_dir,
        overwrite=config.overwrite,
        result_filename=ATTACK_RESULT_FILENAME,
        final_results_filename=config.final_results_filename,
        ids_to_save=config.ids_to_save,
        detection_threshold=config.detection_threshold,
        message_threshold=config.message_threshold,
        detection_bits=config.detection_bits,
        data_subdir=config.data_subdir,
        detection_result_filename=config.detection_result_filename,
        attacks=config.attacks,
        skip_quality_metrics_on_attacks=config.skip_quality_metrics_on_attacks,
    )


class WatermarkEnd2End(Task):
    """
    A composed task that has 2 steps: watermark generation, followed by watermark attacks and detection.
    The sub_tasks was derived directly from WatermarkAttacksAndDetection task.
    """

    config: End2EndWatermarkingConfig

    def __init__(
        self,
        config: End2EndWatermarkingConfig,
        data_reader: Optional[DataReaderT] = None,
        num_workers: int = 0,
        **data_args,
    ):
        self.config = config
        self.orig_data_reader = data_reader
        self.num_workers = num_workers
        self.data_args = data_args
        self.has_pre_eval = True

    def make_sub_tasks(self, config: Optional[End2EndWatermarkingConfig] = None) -> Optional[Sequence["TaskConfig"]]:  # type: ignore
        config = config or self.config
        return unroll_attack_configs(detection_config(config))

    def pre_eval(
        self,
        model_class_or_function: ModelTypeT,
        config: Optional[End2EndWatermarkingConfig] = None,
        show_progress: bool = False,
        auto_distributed: bool = False,
        **kwargs,
    ):
        config = config or self.config

        # We do not assign the device yet, but after the dist is initialized
        if auto_distributed:
            assert "device" not in kwargs or kwargs["device"] != "cpu", f"Invalid device argument when auto_distributed=True ({kwargs['device']})"
            model = get_model(model_class_or_function, **kwargs)
        else:
            device = auto_device(kwargs.pop("device", None))
            model = get_model(model_class_or_function, device=device, **kwargs)

        assert isinstance(
            model, Generator
        ), f"Invalid model: {model_class_or_function}, expect a Generator"

        return self.__call_wm__(
            model,
            config=config,  # type: ignore
            show_progress=show_progress,
            auto_distributed=auto_distributed,
        )

    def eval(
        self,
        model_class_or_function: ModelTypeT,
        config: Optional[TaskConfig] = None,
        show_progress: bool = False,
        auto_distributed: bool = False,
        **kwargs,
    ):
        """
        The eval() will be run in one SLURM node. For the end-2-end task, it has two possible flows:

        1) Run watermark generation, then all attacks run ON THE SAME worker node
        2) Run watermark generation in one worker (in pre_eval), then each of the attacks in a separate worker node in a parallel job

        We do it in a bit hacky way: 1) will happen if config is of type End2EndWatermarkingConfig, while 2) is when config is an
        WatermarkAttacksConfig

        """
        
        device = kwargs.pop("device", None)
        if auto_distributed:
            assert device is None or device != "cpu", "Invalid device when using auto_distributed=True"
            device = "cpu"
        else:
            device = auto_device(device)

        model = get_model(model_class_or_function, device=device, **kwargs)

        if isinstance(config, End2EndWatermarkingConfig):
            return self(
                model,
                config=config,
                show_progress=show_progress,
                auto_distributed=auto_distributed,
            )

        elif isinstance(config, WatermarkAttacksConfig):
            return self.__call_detection__(
                model,
                config=config,
                show_progress=show_progress,
                auto_distributed=auto_distributed,
            )

    def __call__(
        self,
        model: Union[Generator, Detector],
        config: Optional[End2EndWatermarkingConfig] = None,  # type: ignore
        batches: Optional[Iterable[InputBatch]] = None,
        show_progress: bool = False,
        auto_distributed: bool = False,
        **kwargs,
    ):
        assert isinstance(model, Generator) and isinstance(
            model, Detector
        ), f"Expect model to be both Generator and Detector for the end-to-end task, but get {type(model)}"


        config = config or self.config

        if auto_distributed:
            assert dist_is_supported(), "Distributed is not supported"
            dist_ctx = safe_init_distributed
        else:
            dist_ctx = nullcontext

        with dist_ctx():

            # move model to GPU after the process group is initialized
            if auto_distributed:
                model = to_device(model, auto_device())

            wmark_res = self.__call_wm__(model, config=config, batches=batches, show_progress=show_progress, **kwargs)
            final_res = self.__call_detection__(model, config=config, show_progress=show_progress, **kwargs)

            return final_res

    def __call_wm__(
        self,
        model: Generator,
        config: End2EndWatermarkingConfig,
        show_progress: bool = False,
        auto_distributed: bool = False,
        **kwargs,
    ):
        gen_task = self.get_generation_task(
            config=config,
            data_reader=self.orig_data_reader,
            num_workers=self.num_workers,  # type: ignore[arg-type]
            **self.data_args,
        )
        return gen_task(
            model,
            show_progress=show_progress,
            auto_distributed=auto_distributed,
            **kwargs,
        )

    def __call_detection__(
        self,
        model: Detector,
        config: End2EndWatermarkingConfig,
        show_progress: bool = False,
        auto_distributed: bool = False,
        **kwargs,
    ):
        _data_args = deepcopy(self.data_args)

        # INCOMPLETE: The current implementation sets 'dataset_dir' to the cache directory,
        # but proper detection requires access to the local original data.
        # To fully implement this, ensure that the original data is loaded or referenced here.
        # TODO: Update this logic to handle local original data for detection.
        _data_args["dataset_dir"] = self.config.cache_dir

        det_task = self.get_detection_task(
            config=config,
            num_workers=self.num_workers,  # type: ignore[arg-type]
            **_data_args,
        )
        return det_task(
            model,
            show_progress=show_progress,
            auto_distributed=auto_distributed,
            **kwargs,
        )

    @abstractmethod
    def get_generation_task(
        self,
        config: End2EndWatermarkingConfig,
        data_reader: Optional[DataReaderT] = None,
        num_workers: int = 0,
        **data_args,
    ) -> WatermarkGeneration:
        pass

    @abstractmethod
    def get_detection_task(
        self,
        config: End2EndWatermarkingConfig,
        num_workers: int = 0,
        **data_args,
    ) -> WatermarkAttacksAndDetection:
        pass

    def run(
        self,
        model: Union[Generator, Detector],
        batches: Iterable[InputBatch],
        config: TaskConfig,
        show_progress: bool = False,
        **kwargs,
    ) -> ScoresT:

        raise NotImplementedError(
            "Use pre_eval() and eval() instead of run() for the end-to-end task"
        )

    def print_scores(
        self, scores: Union[ScoresT, List[ScoresT]], **kwargs
    ) -> pd.DataFrame:
        _data_args = deepcopy(self.data_args)
        _data_args["dataset_dir"] = self.config.cache_dir

        det_config = detection_config(self.config)
        det_task = self.get_detection_task(
            config=det_config,
            num_workers=self.num_workers,  # type: ignore[arg-type]
            **_data_args,
        )

        return det_task.print_scores(scores, **kwargs)
