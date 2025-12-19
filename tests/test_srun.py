# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
"""
Test the fixed distributed initialization with multi-node SLURM setup.
"""

import os
import sys
from pathlib import Path

import pytest

from omnisealbench.utils.cluster import ClusterType, guess_cluster


@pytest.mark.skipif(guess_cluster() == ClusterType.UNKNOWN, reason="This test requires a SLURM environment to run.")
def test_distributed():
    """Test function to be run in SLURM with multiple processes"""
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    from omnisealbench.utils.distributed import (
        init_torch_distributed, 
        is_slurm_job, 
        get_global_rank, 
        get_world_size,
        get_local_rank
    )
    import torch.distributed as dist
    
    print(f"=== Process {os.environ.get('SLURM_PROCID', '0')} starting ===")
    print("Environment check:")
    print(f"  is_slurm_job(): {is_slurm_job()}")
    print(f"  SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'NOT SET')}")
    print(f"  SLURM_PROCID: {os.environ.get('SLURM_PROCID', 'NOT SET')}")
    print(f"  SLURM_LOCALID: {os.environ.get('SLURM_LOCALID', 'NOT SET')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    
    global_rank = get_global_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    
    print(f"Process {global_rank}: Computed ranks:")
    print(f"  global_rank: {global_rank}")
    print(f"  world_size: {world_size}")
    print(f"  local_rank: {local_rank}")
    
    try:
        print(f"Process {global_rank}: Initializing distributed...")
        init_torch_distributed()
        
        if dist.is_initialized():
            actual_rank = dist.get_rank()
            actual_world_size = dist.get_world_size()
            print(f"Process {global_rank}: SUCCESS! Distributed initialized")
            print(f"  Actual rank: {actual_rank}")
            print(f"  Actual world size: {actual_world_size}")
            
            if world_size > 1:
                print(f"Process {global_rank}: Testing all-reduce...")
                import torch
                if torch.cuda.is_available():
                    device = f"cuda:{local_rank}"
                    tensor = torch.tensor([float(actual_rank)], device=device)
                else:
                    tensor = torch.tensor([float(actual_rank)])
                
                dist.all_reduce(tensor)
                print(f"Process {global_rank}: All-reduce result: {tensor.item()}")
            
            print(f"Process {global_rank}: Test completed successfully!")
            return "SUCCESS"
        else:
            print(f"Process {global_rank}: Single process mode")
            return "SUCCESS_SINGLE"
            
    except Exception as e:
        print(f"Process {global_rank}: ERROR: {e}")
        import traceback
        traceback.print_exc()
        return f"ERROR: {e}"


# Example SLURM command to run this test:
# 1. without torchrun:
# srun --nodes=1 --ntasks-per-node=2 --gres=gpu:2 --cpus-per-task=4 \
#   --job-name=omniseal_test_distributed \
#   --output=omniseal_test_distributed.out \
#   --error=omniseal_test_distributed.err \
#   conda run -n <env_name> python tests/test_srun.py
#
# 2. with torchrun:
#
# torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=123 --rdzv_backend=c10d \
#   --rdzv_endpoint=localhost:29500 python tests/test_srun.py
if __name__ == "__main__":
    result = test_distributed()
    print(f"Final result: {result}")
