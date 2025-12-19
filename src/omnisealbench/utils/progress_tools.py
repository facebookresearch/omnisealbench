# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from collections import deque
from typing import Union

from rich.console import Group, NewLine
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Live,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class LimitedProgress(Progress):
    def __init__(self, *columns: Union[str, ProgressColumn], max_tasks: int = 10, **kwargs):
        if columns is None:
            columns = []
        super().__init__(*columns, **kwargs)
        self.max_tasks = max_tasks
        self.tasks_queue = deque()

    def add_task(self, description, **kwargs):
        task_id = super().add_task(description, **kwargs)
        self.tasks_queue.append(task_id)

        if len(self.tasks_queue) > self.max_tasks:
            oldest_task_id = self.tasks_queue.popleft()
            # if task is still present
            if oldest_task_id in self.task_ids:
                self.stop_task(oldest_task_id)
                self.update(oldest_task_id, visible=False)

        return task_id


class EvaluationProgress(ABC):
    def __init__(self, runtime_tasks: int) -> None:
        super().__init__()

        runtime_progress = Progress(
            TextColumn("{task.fields[label]:<10}"),  # fixed width 10, left aligned
            TextColumn("{task.fields[elapsed]:>6.2f}s"),  # right align, width 6
        )

        # overall progress bar
        overall_progress = Progress(
            TextColumn("[bold green]{task.description}: [progress.percentage]{task.percentage:>3.0f}%"),            
            BarColumn(), 
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        attack_variant_progress = LimitedProgress(
            TextColumn("{task.description} • [progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            max_tasks=8,
        )

        progress_group = Group(
            Panel(Group(
                attack_variant_progress, 
                NewLine(),
                runtime_progress,
                NewLine(),
                overall_progress,
            ), title="Evaluation Progress"),
        )

        # create overall progress bar
        self.overall_task_id = overall_progress.add_task("Total", total=runtime_tasks)
        self.detection_progress_id = runtime_progress.add_task("", label="Detection", elapsed=0.0)
        self.decoding_progress_id = runtime_progress.add_task("", label="Decoding", elapsed=0.0)
        self.quality_progress_id = runtime_progress.add_task("", label="Quality", elapsed=0.0)
        self.attack_progress_id = runtime_progress.add_task("", label="Attacks", elapsed=0.0)

        self.runtime_progress = runtime_progress
        self.overall_progress = overall_progress
        self.attack_variant_progress = attack_variant_progress

        self.live = Live(progress_group)

        self.detection_time = 0
        self.decoder_time = 0
        self.attack_time = 0
        self.quality_time = 0

    def __enter__(self):
        self.live.start()

        return self

    def update(self, det_time: float, decoder_time: float, attack_time: float, quality_time: float) -> None:

        self.detection_time += det_time
        self.decoder_time += decoder_time
        self.attack_time += attack_time
        self.quality_time += quality_time

        self.runtime_progress.update(self.detection_progress_id, elapsed=self.detection_time)
        self.runtime_progress.update(self.decoding_progress_id, elapsed=self.decoder_time)
        self.runtime_progress.update(self.attack_progress_id, elapsed=self.attack_time)
        self.runtime_progress.update(self.quality_progress_id, elapsed=self.quality_time)


        # increase overall progress now this task is done
        self.overall_progress.update(self.overall_task_id, advance=1)

    def __exit__(self, exc_type, exc_value, traceback):
        self.live.stop()