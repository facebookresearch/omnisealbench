# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Helper functions for SLURM job submission and management.
#
# Much of this code is adapted from the Stopes project
# (https://github.com/facebookresearch/stopes), which
# which is licensed under the same MIT License (Meta).

import asyncio
import importlib
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import random
from typing import (Any, Callable, Dict, List, Literal, Optional, Tuple,
                    TypeVar, Union)

if importlib.util.find_spec("submitit") is None:
    import subprocess
    import sys

    # Install submitit on-the-fly
    print("First-time run on SLURM: Installing submitit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "submitit"], stdout=subprocess.DEVNULL)
    print("submitit installed")

from submitit import AutoExecutor, Executor
from submitit import Job as SubmititJob
from submitit.core.utils import UncompletedJobError

# Type variable for the return type of the task function.
R = TypeVar("R", covariant=True)
logger = logging.getLogger("omnisealbench.slurm.launcher")

# Maximum number of retries for job inquiry operations.
MAX_RETRIES = 3


class RegistryStatuses(Enum):
    """
    The job registry displays 6 essential high-level status of a job, shown below.
    """

    COMPLETED = "Completed"
    FAILED = "Failed"
    PENDING = "Pending"
    RUNNING = "Running"
    UNKNOWN = "Unknown"
    KILLED_BY_LAUNCHER = "Killed by launcher"


class ManuallyKilledByLauncherStatuses(Enum):
    """If a job is killed by the launcher, this enum helps to track the status of this kill.
    Sometimes the kill may succeed, fail, be in progress, etc. Below is a description of each status:

    NEVER_ATTEMPTED_TO_KILL_JOB: Launcher never attempted to kill this job
    JOB_FINISHED_BEFORE_ATTEMPTED_KILL: Launcher attempted to kill this job but the job had finished already
    JOB_FINISHED_DURING_ATTEMPTED_KILL: This has been added to deal with a flaky situation with the launcher,
        where: the Launcher attempted to kill this job (while the job was still unfinished), but in the time
        of trying to kill the job, the job had finished on its own
    ATTEMPT_IN_PROGRESS: Launcher currently attempting to kill this unfinished job
    SUCCESSFULLY_KILLED: Launcher successfully killed this unfinished job
    """

    NEVER_ATTEMPTED_TO_KILL_JOB = "Never attempted to kill job"
    JOB_FINISHED_BEFORE_ATTEMPTED_KILL = "Job finished before attempted kill"
    JOB_FINISHED_DURING_ATTEMPTED_KILL = "Job finished on its own during attempted kill by launcher"  # added to deal with flakiness described above
    ATTEMPT_IN_PROGRESS = "Attempt in progress"
    SUCCESSFULLY_KILLED = "Successfully killed by launcher"


################################################################################
#  Submitit Launcher - SLURM Job Statuses Dictionary
################################################################################
# Submitit/Slurm jobs statuses can take a total of ~28 different values
# These are documented here: https://slurm.schedmd.com/squeue.html under the "Job State Codes" Heading
# The job registry however displays only 6 high-level essential statuses. This can be seen in the RegistryStatuses Enum in stopes/core/jobs_registry/stopes_job.py
# Below is a dictionary to map each of the submitit job statuses 24 SLURM job + a few local Job statuses to the 6 accepted registry statuses

submitit_state_to_registry_state_dict = {
    # ############## RegistryStatuses.COMPLETED below ###########################
    "COMPLETED": RegistryStatuses.COMPLETED,
    # ############## RegistryStatuses.FAILED below ##############################
    "FAILED": RegistryStatuses.FAILED,
    "STOPPED": RegistryStatuses.FAILED,
    "SUSPENDED": RegistryStatuses.FAILED,
    "TIMEOUT": RegistryStatuses.FAILED,
    "NODE_FAIL": RegistryStatuses.FAILED,
    "OUT_OF_MEMORY": RegistryStatuses.FAILED,
    "DEADLINE": RegistryStatuses.FAILED,
    "BOOT_FAIL": RegistryStatuses.FAILED,
    "RESV_DEL_HOLD": RegistryStatuses.FAILED,
    "REVOKED": RegistryStatuses.FAILED,
    "SIGNALING": RegistryStatuses.FAILED,  # I'm quite sure signalling means being cancelled
    # ############## RegistryStatuses.PENDING below ##############################
    "QUEUED": RegistryStatuses.PENDING,
    "PENDING": RegistryStatuses.PENDING,
    "REQUEUE_FED": RegistryStatuses.PENDING,
    "REQUEUE_HOLD": RegistryStatuses.PENDING,
    "REQUEUED": RegistryStatuses.PENDING,
    "RESIZING": RegistryStatuses.PENDING,
    "READY": RegistryStatuses.PENDING,  # from local submitit job statuses
    # ############## RegistryStatuses.RUNNING below ##############################
    "RUNNING": RegistryStatuses.RUNNING,
    "COMPLETING": RegistryStatuses.RUNNING,
    "CONFIGURING": RegistryStatuses.RUNNING,
    # ############## RegistryStatuses.UNKNOWN below ##############################
    "UNKNOWN": RegistryStatuses.UNKNOWN,
    "FINISHED": RegistryStatuses.UNKNOWN,
    # "FINISHED" is set to UNKNOWN because both successful and failed jobs are BOTH labelled as 'finished'
    # Must perform an additional check to see if the job finished due to success or failure. This check is done in convert_slurm_status_into_registry_job_status function
    "INTERRUPTED": RegistryStatuses.UNKNOWN,  # from local submitit job statuses
    # "INTERRUPTED" Status is a bit tricky because if the job was manually interrupted as part of dying strategies, then we set the status to be KILLED_BY_LAUNCHER
    # If it was interrupted otherwise, we leave it as FAILED
    "CANCELLED": RegistryStatuses.FAILED,
    # Same logic applies as for INTERRUPTED status. If the job was cancelled by the launcher, then we set the status to be KILLED_BY_LAUNCHER
    # Else, we leave it as FAILED
    # (INTERRUPTED [local cluster] and CANCELLED [slurm cluster] seem to be equivalents of each other)
    "STAGE_OUT": RegistryStatuses.UNKNOWN,
    # STAGE_OUT status: Occurs once the job has completed or been cancelled, but Slurm has not released resources for the job yet. Source: https://slurm.schedmd.com/burst_buffer.html
    # Hence, setting it as unknown as this state doesn't specifify completion or cancellation.
    "SPECIAL_EXIT": RegistryStatuses.UNKNOWN,
    # SPECIAL_EXIT status: This occurs when a job exits with a specific pre-defined reason (e.g a specific error case).
    # This is useful when users want to automatically requeue and flag a job that exits with a specific error case. Source: https://slurm.schedmd.com/faq.html
    # Hence, setting as unknown as the status may change depending on automatic re-queue; without this requeue, job should be FAILED.
    "PREEMPTED": RegistryStatuses.UNKNOWN,
    # PREEMPTED Status: Different jobs react to preemption differently - some may automatically requeue and other's may not
    # (E.g if checkpoint method is implemented or not). Hence, setting as Unknown for now; without automatic re-queue, job should be FAILED
}


@dataclass
class JobConfig:
    """
    configuration to set up a SLURM job for running an eval task
    """

    nodes: int = 1
    mem_gb: Optional[int] = None
    mem_per_cpu_gb: Optional[int] = None
    tasks_per_node: int = 1
    gpus_per_node: int = 0
    cpus_per_task: int = 5
    timeout_min: int = 720
    constraint: Optional[str] = None


class Job:
    """A wrapper around submitit's Job class with extra status tracking for jobs managed by the launcher,
    A job has some internal attributes to track its state and results:

    killed_by_launcher_status:
      - status of the kill by launcher (ie kill in progress, kill succeeded, etc)
      - can take any value from within ManuallyKilledByLauncherStatuses enum

    status_of_job_before_launcher_kill:
      - status of job right before launcher attempted to kill it (ie Running, Pending, etc)
      - if killed_by_launcher_status is NEVER_ATTEMPTED_TO_KILL_JOB, this should be None.

    _done: Set to true if the job has finished and the results are available.
    _result: The result of the job, if available. This is set after the job has completed successfully.
    """

    def __init__(self, job: SubmititJob, task_name: str):
        """
        Initialize the Job wrapper.

        Args:
            job (SubmititJob): The submitit job object.
            task_name (str): The name of the task associated with this job. A task can spawn one or multiple jobs,
                and this name helps to identify the task in the job registry.
        """
        self.job = job
        self.task_name = task_name
        self._done = False
        self._result: Any = None
        self.killed_by_launcher_status = ManuallyKilledByLauncherStatuses.NEVER_ATTEMPTED_TO_KILL_JOB
        self.status_of_job_before_launcher_kill: Optional[RegistryStatuses] = None

    @property
    def job_id(self) -> str:
        return self.job.job_id

    def cancel(self, check: bool = True) -> None:
        return self.job.cancel(check=check)

    def wait(self) -> None:
        """
        Wait for the job to finish. This will block until the job is completed, failed, or cancelled.
        """
        try:
            self.job.wait()
        except KeyboardInterrupt:
            logger.info(f"KeyboardInterrupt received for job {self.job_id}. Cancelling job...")
            try:
                self.job.cancel(check=False)
                logger.info(f"Job {self.job_id} cancelled successfully.")
            except Exception as e:
                logger.warning(f"Failed to cancel job {self.job_id}: {e}")
            raise

    async def result(self, retries: int = MAX_RETRIES) -> Any:
        if self._done:
            return self._result
        for attempt in range(retries + 1):
            try:
                results = await self.job.awaitable().results()
                self._done = True
                self._result = results[0]
                return self._result
            except Exception as e:  # type: ignore
                # by default, retry on missing output errors
                # they usually mean that there was a problem in slurm. 
                # Otherwise, the job is failed because of an internal error, in this case we don't retry
                # and raise the error right away.
                if not isinstance(e, UncompletedJobError) or ("has not produced any output" not in str(e)):
                    raise e                
                if attempt == retries:
                    logger.error(
                        f"Job {self.job_id} failed after {retries} retries. "
                        "Please check the job logs for more details."
                    )
                    raise e
                await asyncio.sleep(5)

        self._done = True  # mark failed job as done too
        return self._result

    def status(self) -> RegistryStatuses:
        """
        This function maps slurm's/submitit's ~28 different values to return 6 essential statuses understood by the
        registry, and logs a warning for unrecognized statuses. For most of the statuses this is as simple as a
        dictionary lookup. For a couple of them (FINISHED, CANCELLED, INTERRUPTED) we have to deal with them
        in a bit of a special way.
        """
        job_id, slurm_status = self.job_id, self.job.state
        # First, handle "FINISHED" status:
        if slurm_status in ("FINISHED", "DONE"):

            # If launcher never attempted kill, keep killed_by_launcher_status as it is.
            # Else, set it to JOB_FINISHED_DURING_ATTEMPTED_KILL
            self.killed_by_launcher_status = (
                self.killed_by_launcher_status
                if (
                    self.killed_by_launcher_status
                    == ManuallyKilledByLauncherStatuses.NEVER_ATTEMPTED_TO_KILL_JOB
                    or self.killed_by_launcher_status
                    == ManuallyKilledByLauncherStatuses.JOB_FINISHED_BEFORE_ATTEMPTED_KILL
                )
                else ManuallyKilledByLauncherStatuses.JOB_FINISHED_DURING_ATTEMPTED_KILL
            )
            
            # Now determine if (local) job finished due to success or failure
            # If the job doesn't raise an exception, it succeeded.
            # Else, it failed (failed on its own; wasn't killed by launcher)
            job_failed = self.job.exception()

            if job_failed:
                return RegistryStatuses.FAILED
            else:
                return RegistryStatuses.COMPLETED

        # Second, handle statuses: INTERRUPTED (local)/CANCELLED (slurm)
        if (
            slurm_status == "INTERRUPTED"
            or slurm_status == "CANCELLED"
            or slurm_status.find("CANCELLED") != -1
            # The last one is included because sometimes, if killed by the launcher, the status may come out to be
            # "CANCELLED by <number>" (e.g. for number: 1713700156)
        ):
            # Check if the launcher attempted to kill this job
            has_launcher_attempted_kill = (
                self.killed_by_launcher_status
                == ManuallyKilledByLauncherStatuses.ATTEMPT_IN_PROGRESS
            ) or (
                self.killed_by_launcher_status
                == ManuallyKilledByLauncherStatuses.SUCCESSFULLY_KILLED
            )

            # If has_launcher_attempted_kill, then the interruption is due to the launcher. Hence return status 
            # KILLED_BY_LAUNCHER. Else, return FAILED
            if has_launcher_attempted_kill:
                self.killed_by_launcher_status = (
                    ManuallyKilledByLauncherStatuses.SUCCESSFULLY_KILLED
                )
                return RegistryStatuses.KILLED_BY_LAUNCHER
            else:
                return RegistryStatuses.FAILED

        # Now, we handle the general cases - simply do dictionary lookup:
        try:
            job_status = submitit_state_to_registry_state_dict[slurm_status]
            if (
                job_status == RegistryStatuses.COMPLETED
                or job_status == RegistryStatuses.FAILED
            ):
                if (
                    self.killed_by_launcher_status
                    == ManuallyKilledByLauncherStatuses.ATTEMPT_IN_PROGRESS
                ):
                    self.killed_by_launcher_status = (
                        ManuallyKilledByLauncherStatuses.JOB_FINISHED_DURING_ATTEMPTED_KILL
                    )

            return job_status

        # Entering this except block means slurm_status doesn't exist in submitit_state_to_registry_state_dict
        except KeyError:  
            logger.warning(
                f"Job with id: {job_id} has unrecognized slurm status: {slurm_status}. "
                "Please inspect and if suitable, add this status to the slurm_state_to_registry_state_map converter."
            )
            return RegistryStatuses.UNKNOWN

    def job_finished_after_attempted_kill(self):
        """
        This private helper:
            Returns True if job has finished (either Completed, Failed, or Killed by Launcher)
            Returns False if job is unfinished (either pending, running, or unknown)

        Also, this helper:
            Is to be called only after a job has been attempted to be killed AND waited (job.kill() AND job.wait())
            Will set the job's killed_by_launcher_status
        """
        job_status_after_attempted_kill = self.status()
        if job_status_after_attempted_kill == RegistryStatuses.KILLED_BY_LAUNCHER:
            # Job has been successfully killed by launcher
            self.killed_by_launcher_status = (
                ManuallyKilledByLauncherStatuses.SUCCESSFULLY_KILLED
            )
            return True
        elif job_status_after_attempted_kill in [
            RegistryStatuses.COMPLETED,
            RegistryStatuses.FAILED,
        ]:
            # Even though the launcher attempted to kill an unfinished job, the job finished on its own before
            # it was killed by the launcher
            self.killed_by_launcher_status = (
                ManuallyKilledByLauncherStatuses.JOB_FINISHED_DURING_ATTEMPTED_KILL
            )
            return True
        else:
            # Job must be pending, unknown or still running, meaning it wasn't killed yet
            return False


async def run_tasks(tasks: List[Any]):
    await asyncio.gather(*tasks)


class Launcher:
    """Manage SLURM job submission and execution within a cluster.
    A launcher is responsible for setting up the SLURM job parameters,"""

    def __init__(
        self,
        cluster: Literal["local", "slurm", "debug"] = "slurm",

        # some slurm clusters do not support mem_gb, if you set this to False,
        # the Requirements.mem_gb coming from the module will be ignored
        supports_mem_spec: bool = True,

        partition: Optional[str] = None,
        qos: Optional[str] = None,
        account: Optional[str] = None,
        log_dir: Union[str, Path] = Path("submitit_logs"),
        max_jobarray_jobs: int = 20,
    ):
        """
        Initialize the SLURM job launcher with the specified cluster and runtime settings.

        Args:
            cluster (Literal["local", "slurm", "debug"]): The type of cluster to use.
            supports_mem_spec (bool): Whether the cluster supports memory specification.
            partition (Optional[str]): SLURM partition to use.
            qos (Optional[str]): Quality of Service for SLURM jobs. Support multiple QOS separated by comma.
            account (Optional[str]): SLURM account to charge for the job.
            log_dir (Union[str, Path]): Directory to store SLURM job logs.
            max_jobarray_jobs (int): Maximum number of parallel jobs in a job array.
        """
        self.cluster = cluster
        self.supports_mem_spec = supports_mem_spec
        self.partition = partition
        
        if qos is not None and "," in qos:
            _qos = qos.split(",")
        else:
            _qos = [qos]
        self.qos = _qos
        self.account = account
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_jobarray_jobs = max_jobarray_jobs

        self.registry: Dict[str, Tuple[Job, str]] = {}
        
    def schedule(
        self,
        task_name: str,
        task_func: Callable[..., R],
        runtime: Optional[JobConfig] = None,
        **kwargs,
    ) -> str:
        """
        Schedule a task to run later on the SLURM cluster and return the job id.

        Args:
            task_name (str): The name of the task to run.
            task_func (Callable[..., R]): The function to execute as the task.
            runtime (Optional[JobConfig]): Configuration for the job runtime.
            **kwargs: Additional keyword arguments to pass to the task function.

        Returns:
            Any: The result of the executed task.
        """
        if runtime is None:
            runtime = JobConfig()
        executor = self._get_executor(task_name, runtime)
        submitit_job = executor.submit(task_func, **kwargs).cancel_at_deletion()
        job = Job(submitit_job, task_name)
        self._register_job(task_name, job)
        logger.debug(f"Task '{task_name}' submitted with job ID {job.job_id}.")
        return job.job_id

    async def run(
        self,
        task_name: str,
        task_func: Callable[..., R],
        runtime: Optional[JobConfig] = None,
        retries: int = MAX_RETRIES,
        **kwargs,
    ) -> Any:
        """run a single task on a SLURM node that is allocated to the launcher, await for the result.
        This method submits a task to the SLURM cluster and registers it in the launcher.
        

        Args:
            task_name (str): _description_
            runtime (JobConfig): _description_
            task_func (Callable[..., R]): _description_
            show_progress (bool, optional): _description_. Defaults to False.

        Returns:
            Any: _description_
        """
        job_id = self.schedule(
            task_name=task_name,
            task_func=task_func,
            runtime=runtime,
            **kwargs
        )
        job, _task_name = self.registry[job_id]
        assert _task_name == task_name, (
            f"Found job with different task_name in registry: expected {_task_name}, got {task_name}"
        )
        return await job.result(retries=retries)
    
    def get_status(self, job_id: str) -> Tuple[str, str]:
        """
        Get the simplified SLURM status of a job. In case 

        Args:
            job_id (str): The ID of the job to check.

        Returns:
            RegistryStatuses: The status of the job as provided by SLURM
            ManuallyKilledByLauncherStatuses: In case the job is killed by the launcher, 
                this will return the status of the kill.
        """
        if job_id not in self.registry:
            raise ValueError(f"Job ID {job_id} not found in registry.")

        job, _ = self.registry[job_id]
        return job.status().value, job.killed_by_launcher_status.value

    async def kill(self, job_id: str, silent: bool = False):
        """
        Kills a job and waits till the job is killed
        Should only be called on unfinished jobs

        Args:
            job_id (str): The ID of the job to kill.
        """
        if job_id not in self.registry:
            logger.warning(f"Job ID {job_id} not found in registry. Cannot kill.")
            return

        job, task_name = self.registry[job_id]
        if not silent:
            logger.info(f"Killing job '{task_name}' with ID {job_id}.")

        job_status = job.status()
        if job_status in [
            RegistryStatuses.COMPLETED,
            RegistryStatuses.FAILED,
        ]:
            job.killed_by_launcher_status = (
                ManuallyKilledByLauncherStatuses.JOB_FINISHED_BEFORE_ATTEMPTED_KILL
            )
            logger.debug(
                f"Tried to kill a job with id: {job_id}, but this job has already {job_status}"
            )
            return

        if job_status == RegistryStatuses.KILLED_BY_LAUNCHER:
            logger.debug(
                f"Tried to kill a job with id: {job_id}, but this has already been killed by launcher."
            )
            return

        if job_status not in [  # Job is unfinished
            RegistryStatuses.RUNNING,
            RegistryStatuses.PENDING,
            RegistryStatuses.UNKNOWN,
        ]:
            raise NotImplementedError(
                f"Job with id: {job_id} and status: {job_status} can't be killed due to unidentifiable status."
            )

        job.killed_by_launcher_status = ManuallyKilledByLauncherStatuses.ATTEMPT_IN_PROGRESS
        job.cancel()
        try:
            job.wait()  # Wait for the job to be fully cancelled
        except KeyboardInterrupt:
            logger.info(f"KeyboardInterrupt received while waiting for job {job_id} to be killed.")
            # Job cancellation was already initiated, so we continue with cleanup
            raise

        is_job_finished = job.job_finished_after_attempted_kill()
        if not is_job_finished:
            # Job must be pending, unknown or still running, meaning it wasn't killed yet
            # It's highly possible that the job was indeed killed, but the the status from
            # the cluster isn't up to date
            # This is because the status that fetched from cluster is only updated ~ once
            # per minute (to avoid overloading cluster)
            # Hence, put job to sleep for ~10 seconds to let the status update
            await asyncio.sleep(10)

            is_job_finished = job.job_finished_after_attempted_kill()
            if not is_job_finished:
                logger.warning(
                   f"Somehow Job: {job_id} is unfinished despite attempt by launcher to kill and a wait of 10 seconds"
                )

    def clear(self):
        """Clear all pending jobs"""
        # Attempt to kill all jobs, even if interrupted

        jobs = [
            job_id
            for job_id, (job, _) in self.registry.items()
            if job is not None and job.status() == RegistryStatuses.RUNNING
        ]

        if jobs:
            for job_id in jobs:
                asyncio.run(self.kill(job_id=job_id, silent=True))

    def _register_job(self, task_name: str, job: Job):
        """
        Register a job in the launcher registry.

        Args:
            task_name (str): The name of the task.
            job (Job): The job object to register.
        """
        job_id = job.job_id
        if not job_id:
            raise ValueError("Job ID is not set. Ensure the job is submitted before registration.")
        if job_id in self.registry:
            logger.debug(f"Task '{task_name}' is already registered under job ID {job_id}.")
        else:
            self.registry[job_id] = (job, task_name)

    def _get_executor(self, task_name: str, runtime: JobConfig) -> Executor:
        """
        Get a SLURM executor with the specified runtime configuration.

        Args:
            runtime (RuntimeConfig): The runtime configuration for the SLURM job.

        Returns:
            submitit.Executor: An executor configured with the given runtime settings.
        """

        task_log_dir = Path(self.log_dir) / task_name
        task_log_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)

        # Submitit expects each parameter to be prefixed by "executor named", so we fake
        # a "slurm_" prefix here to match the expected format.
        task_name = self.cluster + "_" + task_name

        executor = AutoExecutor(folder=task_log_dir, cluster=self.cluster)
        executor.update_parameters(name=task_name)

        if runtime.nodes:
            executor.update_parameters(nodes=runtime.nodes)

        if runtime.mem_gb is not None and self.supports_mem_spec:
            executor.update_parameters(mem_gb=runtime.mem_gb)

        if runtime.mem_per_cpu_gb is not None:
            executor.update_parameters(mem_per_cpu=f"{runtime.mem_per_cpu_gb}G")

        if runtime.tasks_per_node:
            executor.update_parameters(tasks_per_node=runtime.tasks_per_node)

        # set cpus_per_task if provided
        if getattr(runtime, "cpus_per_task", None) is not None:
            executor.update_parameters(cpus_per_task=runtime.cpus_per_task)

        # Prefer explicit slurm_gres when available (injects --gres into sbatch)
        if runtime.gpus_per_node > 0:
            try:
                executor.update_parameters(slurm_gres=f"gpu:{runtime.gpus_per_node}")
            except Exception:
                logger.debug("slurm_gres not supported by executor; falling back to slurm_extra with --gres flag")

                # preserve existing slurm_extra by retrieving it if set
                existing = getattr(executor, 'params', {})
                slurm_extra = existing.get('slurm_extra', []) or []
                slurm_extra = list(slurm_extra) + [f"--gres=gpu:{runtime.gpus_per_node}", "--cpu-bind=none"]
                executor.update_parameters(slurm_extra=slurm_extra)

        if runtime.timeout_min > 0:
            executor.update_parameters(timeout_min=runtime.timeout_min)

        if self.cluster == "slurm":

            partition = self.partition or os.environ.get("SLURM_JOB_PARTITION")
            if partition:
                executor.update_parameters(slurm_partition=partition)

            qos = self.qos or os.environ.get("SLURM_QOS")
            if qos:
                _qos = random.choice(self.qos)
                executor.update_parameters(slurm_qos=_qos)

            account = self.account or os.environ.get("SLURM_JOB_ACCOUNT")
            if account:
                executor.update_parameters(slurm_account=account)

            if runtime.constraint:
                executor.update_parameters(slurm_constraint=runtime.constraint)

        return executor
