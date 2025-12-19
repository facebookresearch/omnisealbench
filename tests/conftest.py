# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from omnisealbench.utils.distributed_logger import \
    setup_distributed_logging_and_exceptions


@pytest.fixture(scope="session", autouse=True)
def setup_distributed_logging():
    """Setup distributed logging for all tests."""
    setup_distributed_logging_and_exceptions()
    yield


@pytest.fixture
def device():
    """
    Returns the device string, defaulting to 'cuda' if available.
    If a specific device is provided, it returns that device.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
