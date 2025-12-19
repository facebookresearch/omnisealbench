# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch


def str2msg(s):
    return [True if el == "1" else False for el in s]


@pytest.mark.parametrize(
    "bool_msg, key",
    [
        (
            "000100010010001011011000001110000010011001100111",
            "111010110101000001010111010011010100010000100111",
        ),
        ("1111101010010100", "1111101010010100"),
    ],
)
def test_bit_accuracies(bool_msg, key):
    from omnisealbench.metrics.message import (get_bitacc, get_bitacc_v2,
                                               get_bitacc_v3)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bool_msg = str2msg(bool_msg)
    bool_key = str2msg(key)

    keys = torch.tensor(bool_key, dtype=torch.float32).to(device)
    decoded = torch.tensor(bool_msg, dtype=torch.float32).to(device)

    bit_accs_1 = get_bitacc(decoded, keys)
    bit_accs_2 = get_bitacc_v2(decoded, keys)
    bit_accs_3 = get_bitacc_v3(decoded, keys)

    bit_accs_1 = bit_accs_1.item()
    bit_accs_2 = bit_accs_2.item()
    bit_accs_3 = bit_accs_3.item()

    assert bit_accs_1 == bit_accs_2
    assert bit_accs_1 == bit_accs_3


def test_average_metrics():
    from omnisealbench.utils.distributed import average_metrics

    data = {
        "metric1": [1.0, 2.0, 3.0],
        "metric2": [4.0, 5.0, 6.0],
    }

    avg_results = average_metrics(data)

    assert np.isclose(avg_results["metric1"].avg, 2.0)
    assert np.isclose(avg_results["metric2"].avg, 5.0)
    assert np.isclose(avg_results["metric1"].square, 4.666666666666667)
    assert np.isclose(avg_results["metric2"].square, 25.666666666668)

    assert avg_results["metric1"].count == 3
    assert avg_results["metric2"].count == 3

        

@pytest.mark.skipif(not torch.cuda.is_available() or  not torch.cuda.device_count(), reason="only run in multi-GPU")
def test_average_metrics_distributed():
    from omnisealbench.utils.distributed import (average_metrics,
                                                 get_world_size,
                                                 init_torch_distributed)

    data = {
        "metric1": [1.0, 2.0, 3.0],
        "metric2": [4.0, 5.0, 6.0],
    }

    init_torch_distributed()

    world_size = get_world_size()

    torch.distributed.barrier()
    avg_results = average_metrics(data)

    assert np.isclose(avg_results["metric1"].avg, 2.0)
    assert np.isclose(avg_results["metric2"].avg, 5.0)
    assert np.isclose(avg_results["metric1"].square, 4.666666666666667)
    assert np.isclose(avg_results["metric2"].square, 25.666666666668)

    assert avg_results["metric1"].count == world_size * 3
    assert avg_results["metric2"].count == world_size * 3

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


# To test gather_metrics in distributed setting:
# srun --nodes=1 --ntasks=2 --gpus-per-node=2 --time=120 --job-name=test_osb_init   conda run -n omnisealbench python -m pytest tests/test_metrics.py
# (replace `omnisealbench` with your conda env name)
@pytest.mark.skipif(not torch.cuda.is_available() or  not torch.cuda.device_count(), reason="only run in multi-GPU")
def test_gather_metrics():
    from omnisealbench.utils.distributed import (gather_metrics,
                                                 get_global_rank,
                                                 get_world_size,
                                                 init_torch_distributed)

    init_torch_distributed()

    results = {
        "ids": [1, 2, 3],
        "scores": [0.9, 0.8, 0.7],
        "other_metric": [10, float('nan'), 30],
    }
    global_rank = get_global_rank()
    world_size = get_world_size()

    gathered_results = gather_metrics(results)
    assert isinstance(gathered_results, dict)

    if global_rank == 0:
        assert "ids" in gathered_results
        assert "scores" in gathered_results
        assert len(gathered_results["ids"]) == world_size * 3
        assert len(gathered_results["scores"]) == world_size * 3
        assert len(gathered_results["other_metric"]) == world_size * 3

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
