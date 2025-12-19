# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from pathlib import Path
import time

import pytest

from omnisealbench.utils.cluster import ClusterType, guess_cluster
from omnisealbench.utils.distributed import init_torch_distributed
from omnisealbench.utils.slurm import (JobConfig, Launcher,
                                       ManuallyKilledByLauncherStatuses,
                                       RegistryStatuses)


def test_simple_launcher(tmp_path):
    launcher = Launcher(
        cluster="local",
        log_dir=tmp_path / "logs",
        max_jobarray_jobs=3,
    )
    
    def double(x):
        return x * 2

    result = asyncio.run(
        launcher.run(
            task_name="test_task",
            task_func=double,
            x=5,
        )
    )
    assert result == 10, "The task function should return the doubled value"


def test_launcher_with_failure(tmp_path):
    launcher = Launcher(
        cluster="local",
        log_dir=tmp_path / "logs",
        max_jobarray_jobs=3,
    )
    
    from submitit.core.utils import UncompletedJobError
    
    def failing_task(x):
        raise ValueError("This is a test failure")

    with pytest.raises(
        UncompletedJobError,
        check=lambda ex: "This is a test failure" in str(ex)
    ):
        asyncio.run(
            launcher.run(
                task_name="failing_task",
                task_func=failing_task,
                retries=1,
                x=5,
            )
        )


def test_kill_job(tmp_path):
    launcher = Launcher(
        cluster="local",
        log_dir=tmp_path / "logs",
        max_jobarray_jobs=3,
    )
    
    def long_running_task(x, timeout):
        time.sleep(timeout)  # Simulate a long-running task
        return x * 2

    # Simulate killing the job while it is running
    job_id = launcher.schedule(
        task_name="long_running_task",
        task_func=long_running_task,
        timeout=100,
        x=5,
    )
    time.sleep(10)  # Give it some time to start
    asyncio.run(launcher.kill(job_id))
    job_status, system_status = launcher.get_status(job_id)
    assert job_status == RegistryStatuses.KILLED_BY_LAUNCHER.value, f"Unexpected job status: {job_status}"
    assert system_status == ManuallyKilledByLauncherStatuses.SUCCESSFULLY_KILLED.value, f"Unexpected system status: {system_status}"


    # Simulate the killing of a finished job
    job_id = launcher.schedule(
        task_name="short_running_task",
        task_func=long_running_task,
        timeout=1,
        x=5,
    )
    time.sleep(10)
    launcher.clear()
    job_status, system_status = launcher.get_status(job_id)
    assert job_status == RegistryStatuses.COMPLETED.value, f"Unexpected job status: {job_status}"
    assert system_status in [
        ManuallyKilledByLauncherStatuses.JOB_FINISHED_BEFORE_ATTEMPTED_KILL.value,
        ManuallyKilledByLauncherStatuses.JOB_FINISHED_DURING_ATTEMPTED_KILL.value,
        ManuallyKilledByLauncherStatuses.NEVER_ATTEMPTED_TO_KILL_JOB.value,
    ], f"Unexpected system status: {system_status}"


@pytest.mark.skipif(guess_cluster() == ClusterType.UNKNOWN, reason="This test requires a SLURM environment with GPUs to run.")
def test_launcher_with_multiple_gpus(tmp_path):
    launcher = Launcher(
        cluster="slurm",
        log_dir=Path(__file__).parent / "logs",
        max_jobarray_jobs=3,
    )

    def gpu_task(x):
        # Simulate a GPU computation

        init_torch_distributed()
        return x + 42

    runtime = JobConfig(
        timeout_min=5,  # Increased timeout to allow for SLURM job allocation
        nodes=1,
        gpus_per_node=2,
        cpus_per_task=2,
        mem_gb=4,
    )

    result = asyncio.run(
        launcher.run(
            task_name="gpu_task",
            task_func=gpu_task,
            x=8,
            runtime=runtime,
        )
    )
    assert result == 50, "The GPU task should return the correct result"
