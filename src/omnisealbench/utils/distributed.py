# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import atexit
from contextlib import contextmanager
import datetime
import os
import random
import shutil
import socket
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import multiprocessing as mp
import torch
import torch.distributed as dist


from omnisealbench.interface import AverageMetric, Device, ScoresT
from omnisealbench.utils.common import to_python_type, dist_is_supported

if TYPE_CHECKING:
    from omnisealbench.interface import Detector, Generator

SPAWN_METHOD = "spawn"  # "forkserver" or "spawn" are safer than "fork"

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def all_reduce(
    tensor: torch.Tensor, op: str = "sum", group: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    """All-reduces single scalar value if torch distributed is in use."""
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    dop = None
    if op == "sum" or op == "mean":
        dop = dist.ReduceOp.SUM
    elif op == "min":
        dop = dist.ReduceOp.MIN
    elif op == "max":
        dop = dist.ReduceOp.MAX
    elif op == "product":
        dop = dist.ReduceOp.PRODUCT
    else:
        raise ValueError(f"Unsupported reduction operation: {op}")
    
    assert dop is not None
    dist.all_reduce(tensor, op=dop, group=group)
    if op == "mean":
        tensor /= dist.get_world_size(group)
    return tensor


def barrier(group: Optional[dist.ProcessGroup] = None) -> None:
    """Synchronize all processes in the group."""
    if is_dist_avail_and_initialized():
        dist.barrier(group=group)


def get_dp_group() -> Optional[dist.ProcessGroup]:
    if dist.is_initialized():
        return dist.group.WORLD
    return None


def is_submitit_job() -> bool:
    """Check if we're running inside a submitit-managed job.
    
    Note: This will only return True inside the actual submitted job,
    not in the process that calls Executor.submit().
    """
    return "SUBMITIT_JOB_ID" in os.environ


def is_torch_run() -> bool:
    return os.environ.get("TORCHELASTIC_RUN_ID") is not None


def is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ


def setup_distributed_from_submitit() -> None:
    """Initialize torch.distributed within a submitit job."""
    try:
        from submitit.helpers import TorchDistributedEnvironment
        TorchDistributedEnvironment().export()
    except Exception as e:
        print(f"Error setting up distributed environment from submitit: {e}")
        raise e


def setup_nccl_interface(backend: str) -> None:
    """Set up NCCL network interface for SLURM environments."""
    if backend != "nccl" or "NCCL_SOCKET_IFNAME" in os.environ:
        return
    
    # Try common network interfaces in order of preference
    for ifname in ["ib0", "eth0", "front0", "enp0s8", "ens3"]:
        try:
            # Check if interface exists and is up
            result = subprocess.run(
                ['ip', 'link', 'show', ifname],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and "state UP" in result.stdout:
                os.environ["NCCL_SOCKET_IFNAME"] = ifname
                print(f"Set NCCL_SOCKET_IFNAME to {ifname}")
                return
        except Exception:
            continue
    
    # If no interface found with 'state UP', try to auto-detect the default route interface
    try:
        result = subprocess.run(
            ['ip', 'route', 'get', '8.8.8.8'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Extract interface from output like "dev eth0"
            for line in result.stdout.split('\n'):
                if 'dev ' in line:
                    dev_part = line.split('dev ')[1].split()[0]
                    os.environ["NCCL_SOCKET_IFNAME"] = dev_part
                    print(f"Auto-detected NCCL_SOCKET_IFNAME as {dev_part}")
                    return
    except Exception:
        pass
    
    print("Warning: Could not auto-detect network interface for NCCL")


def debug_slurm_env() -> None:
    """Debug function to print all SLURM environment variables"""
    if not is_slurm_job():
        print("Not running in SLURM environment")
        return

    print("SLURM Environment Variables:")
    slurm_vars = {k: v for k, v in os.environ.items() if k.startswith('SLURM')}
    for key, value in sorted(slurm_vars.items()):
        print(f"  {key}={value}")

    print("Computed values:")
    print(f"  World size (GPU-based): {get_world_size()}")
    print(f"  Global rank: {get_global_rank()}")
    print(f"  Local rank: {get_local_rank()}")
    print(f"  Master addr: {get_master_addr()}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")


def get_global_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    if is_torch_run():
        return int(os.environ["RANK"])
    if is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    return 0


def is_local_execution() -> bool:
    if "SUBMITIT_JOB_ID" in os.environ:
        # "Running as a Submitit job"
        return False
    elif is_slurm_job() and get_world_size() > 1:
        # "Running as a native SLURM job"
        return False
    elif is_torch_run():
        assert "LOCAL_RANK" in os.environ
        # "Running locally with torchrun"
        return True
    else:
        # "Running locally in a non-distributed or non-SLURM environment"
        return True


def get_local_rank() -> int:
    if is_torch_run():
        return int(os.environ["LOCAL_RANK"])
    if is_slurm_job():
        return int(os.environ["SLURM_LOCALID"])
    return 0


def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    if dist.is_initialized():
        return dist.get_world_size(group)
    if is_torch_run() or "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    if is_slurm_job():
        # For submitit jobs, always use SLURM_NTASKS as it represents the actual number of processes
        if "SUBMITIT_EXECUTOR" in os.environ or is_submitit_job():
            return int(os.environ.get("SLURM_NTASKS", "1"))

        # For GPU-based distributed training (non-submitit), use GPU count instead of task count
        # Try different SLURM GPU environment variables in order of preference
        if "SLURM_GPUS_ON_NODE" in os.environ:
            # Number of GPUs on current node
            gpus_on_node = int(os.environ["SLURM_GPUS_ON_NODE"])
            nodes = int(os.environ.get("SLURM_NNODES", "1"))
            return gpus_on_node * nodes
        elif "SLURM_GPUS_PER_NODE" in os.environ:
            # GPUs per node format (might be like "1" or "4(x2)" for 2 nodes with 4 GPUs each)
            gpus_per_node_str = os.environ["SLURM_GPUS_PER_NODE"]
            # Handle format like "4(x2)" meaning 4 GPUs per node across multiple nodes
            if "(x" in gpus_per_node_str:
                gpus_per_node = int(gpus_per_node_str.split("(")[0])
                nodes = int(gpus_per_node_str.split("x")[1].rstrip(")"))
            else:
                gpus_per_node = int(gpus_per_node_str)
                nodes = int(os.environ.get("SLURM_NNODES", "1"))
            return gpus_per_node * nodes
        elif "SLURM_STEP_GPUS" in os.environ:
            # Total GPUs in the job step
            step_gpus = os.environ["SLURM_STEP_GPUS"]
            # Count comma-separated GPU IDs
            return len(step_gpus.split(","))
        elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # Fallback: use detected CUDA devices
            cuda_devices = torch.cuda.device_count()
            nodes = int(os.environ.get("SLURM_NNODES", "1"))
            return cuda_devices * nodes
        else:
            # Final fallback: use SLURM_NTASKS (original behavior)
            print(f"Falling back to SLURM_NTASKS={os.environ.get('SLURM_NTASKS', '1')}")
            return int(os.environ.get("SLURM_NTASKS", "1"))
    return 1


def get_master_addr() -> str:
    if is_torch_run():
        return os.environ["MASTER_ADDR"]
    if is_slurm_job():
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        )
        return hostnames.split()[0].decode("utf-8")
    return "127.0.0.1"


def get_master_port(job_id: int) -> int:
    if is_torch_run():
        return int(os.environ["MASTER_PORT"])
    else:
        MIN_MASTER_PORT, MAX_MASTER_PORT = (20000, 60000)
        rng = random.Random(job_id)
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)


def setup_env() -> None:
    triton_cache_dir = tempfile.mkdtemp(prefix="/dev/shm/")
    tiktoken_cache_dir = tempfile.mkdtemp(prefix="/dev/shm/")
    atexit.register(shutil.rmtree, triton_cache_dir, ignore_errors=True)
    atexit.register(shutil.rmtree, tiktoken_cache_dir, ignore_errors=True)
    env_vars = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_IB_TIMEOUT": "22",
        "TRITON_CACHE_DIR": triton_cache_dir,
        "TIKTOKEN_CACHE_DIR": tiktoken_cache_dir,
        "AWS_MAX_ATTEMPTS": "10",
        "AWS_RETRY_MODE": "standard",
    }
    for name, value in env_vars.items():
        if os.environ.get(name) != value:
            os.environ[name] = value


def get_open_port() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return str(port)


def set_torch_variables() -> None:
    from torch._utils_internal import TEST_MASTER_ADDR

    os.environ["MASTER_ADDR"] = str(TEST_MASTER_ADDR)
    os.environ["MASTER_PORT"] = str(get_open_port())
    os.environ["TORCHELASTIC_RUN_ID"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


def init_torch_distributed(
    backend: str = "cpu:gloo,cuda:nccl",
    port: Optional[str] = None,
    timeout_minutes: int = 10,
) -> None:
    # If already initialized (e.g., by torchrun), just set CUDA device and return
    if dist.is_initialized():
        local_rank = get_local_rank()
        print(f"[Rank {get_global_rank()}] torch.distributed already initialized. Set device to cuda:{local_rank}")
        if torch.cuda.is_available() and local_rank >= 0:
            torch.cuda.set_device(local_rank)
        return

    setup_env()

    # Ensure TORCH_DISTRIBUTED_DEBUG is set to a valid value
    debug_val = os.environ.get("TORCH_DISTRIBUTED_DEBUG", "").upper()
    if debug_val not in ["OFF", "INFO", "DETAIL"]:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

    local_rank = get_local_rank()
    global_rank = get_global_rank()
    world_size = get_world_size()

    if world_size == 1:
        return

    # Determine the actual backend to use
    actual_backend = backend
    if torch.cuda.is_available() and "nccl" in backend:
        actual_backend = "nccl"
        # Set CUDA device BEFORE initializing process group
        torch.cuda.set_device(local_rank)
    elif "gloo" in backend:
        actual_backend = "gloo"

    # Set up environment variables based on the execution context
    if is_submitit_job():
        try:
            setup_distributed_from_submitit()
            return  # submitit setup handles everything
        except Exception as e:
            print(f"[Rank {global_rank}] Submitit initialization failed: {e}")
            print("Falling back to SLURM initialization...")

    if is_slurm_job():
        # Set up SLURM-based distributed environment
        job_id = int(os.environ.get("SLURM_JOB_ID", -1))
        master_addr = get_master_addr()
        master_port = port or get_master_port(job_id=job_id)

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(global_rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Set up NCCL interface only once per node to avoid race conditions
        if actual_backend == "nccl" and local_rank == 0:
            setup_nccl_interface(actual_backend)

    # Add small delay for non-master ranks to avoid initialization race conditions
    # Also ensure all processes start at roughly the same time
    if global_rank != 0:
        import time
        time.sleep(1.0 + global_rank * 0.2)
    else:
        # Master waits a bit to ensure all workers are ready
        import time
        time.sleep(2.0)

    if actual_backend == "nccl":
        # See https://github.com/pytorch/pytorch/issues/46874.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    # Always print initialization info for all ranks to debug process sync issues
    print(f"[Rank {global_rank}] Initializing process group with:")
    print(f"  backend={actual_backend}")
    print(f"  master_addr={os.environ.get('MASTER_ADDR', 'not set')}")
    print(f"  master_port={os.environ.get('MASTER_PORT', 'not set')}")
    print(f"  world_size={world_size}")
    print(f"  rank={global_rank}")
    print(f"  local_rank={local_rank}")
    print(f"  timeout={timeout_minutes} minutes")

    if is_slurm_job():
        print(f"  SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID')}")
        print(f"  SLURM_NODEID={os.environ.get('SLURM_NODEID')}")
        print(f"  SLURM_PROCID={os.environ.get('SLURM_PROCID')}")
        print(f"  SLURM_LOCALID={os.environ.get('SLURM_LOCALID')}")
        print(f"  SLURM_NTASKS={os.environ.get('SLURM_NTASKS')}")
        print(f"  SLURM_GPUS_PER_NODE={os.environ.get('SLURM_GPUS_PER_NODE')}")
        print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        if actual_backend == "nccl":
            print(f"  NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', 'not set')}")

    try:
        pg_options = None
        kwargs = {}
        if actual_backend == "nccl":
            pg_options = dist.ProcessGroupNCCL.Options()

            # To efficiently utilize a GPU, the compute kernels should provide enough
            # work to completely fill the GPU. The comms kernels, on the other hand, are
            # bottlenecked on some external resource (e.g., network bandwidth) thus they
            # don't benefit from more GPU resources and only require a fraction of it.
            # If compute goes first, comms can only start once the compute is finished,
            # and at that point it will leave most of the GPU idle. However, if comms
            # goes first, compute can overlap with it.
            # To ensure this happens we must tell CUDA to prioritize comms kernels.
            pg_options.is_high_priority_stream = True

            # NCCL can segfault in some specific circumstances, such as when using bound
            # device ids, non-blocking init, communicator splits and multiple threads (such
            # as PyTorch's autograd thread). A workaround is to disable non-blocking init.
            # See https://github.com/NVIDIA/nccl/issues/1605
            pg_options.config.blocking = 1

            # use of DDP atm demands CUDA, we set the device appropriately according to rank
            assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
            # set GPU device
            assert 0 <= local_rank < torch.cuda.device_count()
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(local_rank)

            kwargs = {"device_id": torch.device("cuda", local_rank)}

        with mp.get_context(SPAWN_METHOD).Manager():
            pass

        dist.init_process_group(
            backend=actual_backend,
            timeout=datetime.timedelta(minutes=timeout_minutes),
            pg_options=pg_options,
            **kwargs,  # type: ignore
        )

        torch._C._set_print_stack_traces_on_fatal_signal(True)

        # Synchronize all processes
        dist.barrier()

        if global_rank == 0:
            print(f"[Rank {global_rank}] Distributed initialization successful!")

    except Exception as e:
        print(f"[Rank {global_rank}] Failed to initialize distributed: {e}")
        print(f"[Rank {global_rank}] Environment details:")
        print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
        print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")
        print(f"  RANK: {os.environ.get('RANK', 'not set')}")
        print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'not set')}")
        print(f"  LOCAL_RANK: {local_rank}")

        if is_slurm_job():
            print(f"  SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'not set')}")
            print(f"  SLURM_PROCID: {os.environ.get('SLURM_PROCID', 'not set')}")
            print(f"  SLURM_LOCALID: {os.environ.get('SLURM_LOCALID', 'not set')}")

        if actual_backend == "nccl":
            print(f"  NCCL_SOCKET_IFNAME: {os.environ.get('NCCL_SOCKET_IFNAME', 'not set')}")
            print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

        import traceback
        traceback.print_exc()
        raise


def cleanup_distributed() -> None:
    if get_world_size() <= 1:
        return
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


@contextmanager
def safe_init_distributed(
    backend: str = "cpu:gloo,cuda:nccl",
    port: Optional[str] = None,
    timeout_minutes: int = 10,
) -> Any:
    """
    Context manager for safely initializing distributed training.
    """
    # fall back to no-ops if the system does not support distributed settings
    if not dist_is_supported():
        yield
        return

    try:
        init_torch_distributed(backend=backend, port=port, timeout_minutes=timeout_minutes)
        yield
    finally:
        cleanup_distributed()


def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def average_metrics(
    data: ScoresT,
    device: Device = "cuda",
) -> Dict[str, AverageMetric]:
    """Get average metrics from a dictionary of scores, potentially from a different GPUs"""
    avg_results: Dict[str, AverageMetric] = {}

    group: Optional[dist.ProcessGroup] = dist.group.WORLD if dist.is_initialized() else None
    world_size = get_world_size(group)

    assert isinstance(data, dict), "data must be a dictionary of scores"
    for k, v in data.items():

        # Do not average the ID column
        if k == "idx":
            continue

        v = [x for x in v if x == x and x is not None]  # ignore nan
        try:
            stats = [sum(v), len(v), sum(x * x for x in v)]
        except Exception as e:
            print(f"Error computing stats for key {k} with values {v}: {e}")
            raise e
        if not dist.is_initialized() or world_size == 1:
            sum_v, len_v, sum_v2 = stats
        else:
            tensor = torch.tensor(stats, device=device, dtype=torch.float32)
            sum_v, len_v, sum_v2 = all_reduce(
                tensor, op="sum", group=group
            ).tolist()
        avg_results[k] = AverageMetric(
            avg=sum_v / max(len_v, 1),
            count=int(len_v),
            square=sum_v2 / max(len_v, 1),
        )
    return avg_results


def gather_metrics(results: ScoresT) -> ScoresT:
    """
    Gather metrics from all processes and return a dictionary of results.
    """

    group: Optional[dist.ProcessGroup] = dist.group.WORLD if dist.is_initialized() else None
    if not dist.is_initialized() or get_world_size(group) == 1:
        return results

    if get_global_rank() == 0:
        gathered_results = [None] * get_world_size(group)
    else:
        gathered_results = None

    barrier(group)

    dist.gather_object(results, object_gather_list=gathered_results, dst=0, group=group)
    if get_global_rank() == 0:
        merged_results = {}
        for proc_result in gathered_results:  # type: ignore
            if proc_result is None:
                continue
            for k, v in proc_result.items():  # type: ignore
                # v = [x for x in v if x == x]  # ignore nan
                if k not in merged_results:
                    merged_results[k] = v
                elif isinstance(v, torch.Tensor):
                    merged_results[k] = torch.concat([merged_results[k], v])
                elif isinstance(v, list):
                    merged_results[k].extend(v)
                elif isinstance(v, np.ndarray):
                    merged_results[k] = np.concatenate([(merged_results[k], v)])
        merged_results = to_python_type(merged_results)
        return merged_results  # type: ignore

    return {}


def gather_lists(list: List[Any]) -> List[Any]:
    """
    Gather lists from all processes and return a list of results.
    """
    group: Optional[dist.ProcessGroup] = dist.group.WORLD if dist.is_initialized() else None
    if not dist.is_initialized() or get_world_size(group) == 1:
        return list

    if get_global_rank() == 0:
        gathered_lists = [None] * get_world_size(group)
    else:
        gathered_lists = None

    barrier(group)

    dist.gather_object(list, object_gather_list=gathered_lists, dst=0, group=group)
    if get_global_rank() == 0:
        merged_list = []
        for proc_list in gathered_lists:  # type: ignore
            if proc_list is None:
                continue
            merged_list.extend(proc_list)
        return merged_list

    return []
