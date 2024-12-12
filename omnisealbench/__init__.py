# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
AudioCraft is a general framework for evaluating the performance of neural watermarking techniques.
At the moment we provide the evaluation tools for the audio watermarking models.
"""

# flake8: noqa
from . import attacks, data, models

__version__ = "1.0.0"
