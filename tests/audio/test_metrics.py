# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch


def test_stoi():
    from omnisealbench.metrics.audio import stoi

    # Create dummy tensors for y_pred and y
    y_pred = torch.randn(2, 1, 16000)  # Simulating two mono-audio samples of 1 second at 16kHz
    y = torch.randn(2, 1, 16000)

    # Compute STOI scores
    scores = stoi(y_pred, y, sr=16000)

    # Check if the output is a list of floats
    assert isinstance(scores, torch.Tensor)
    assert len(scores) == 2  # Should match the number of samples


def test_pesq():
    from omnisealbench.metrics.audio import tensor_pesq_batch

    x = [torch.rand(1, 64000) for _ in range(4)]  # Simulating two mono-audio samples of 1 second at 16kHz
    y = [torch.rand(1, 64000) for _ in range(4)]

    pesq1 = tensor_pesq_batch(x, y, sr=16000, n_processor=1)
    pesq2 = tensor_pesq_batch(torch.stack(x), torch.stack(y), sr=16000, n_processor=1)
    # Check if the output is a list of floats
    assert isinstance(pesq1, list)
    assert len(pesq1) == 4  # Should match the number of samples
    assert np.allclose(pesq1, pesq2, rtol=0.01, atol=1e-2)


def test_stoi_list():
    from omnisealbench.metrics.audio import stoi

    # Create dummy tensors for y_pred and y
    lens = [32000, 48000]
    x = [torch.randn(1, lens[i]) for i in range(2)]  # Simulating two mono-audio samples of 1 second at 16kHz
    y = [torch.randn(1, lens[i]) for i in range(2)]

    # Compute STOI scores
    scores = stoi(x, y, sr=16000)

    # Check if the output is a list of floats
    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (2,)  # Should match the number of samples


def test_si_snr():
    from omnisealbench.metrics.audio import si_snr

    # Create dummy tensors for y_pred and y
    y_pred = torch.randn(2, 1, 16000)  # Simulating two mono-audio samples of 1 second at 16kHz
    y = torch.randn(2, 1, 16000)

    # Compute SI-SNR scores
    scores = si_snr(y_pred, y)

    # Check if the output is a tensor
    assert isinstance(scores, torch.Tensor)
    assert len(scores) == 2  # Should match the number of samples


def test_si_snr_list():
    from omnisealbench.metrics.audio import si_snr

    # Create dummy tensors for y_pred and y
    y_pred = torch.randn(2, 1, 16000)  # Simulating two mono-audio samples of 1 second at 16kHz
    y = torch.randn(2, 1, 16000)

    # Compute SI-SNR scores
    scores = si_snr(y_pred, y)

    # Check if the output is a tensor
    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (2,)  # Should match the number of samples