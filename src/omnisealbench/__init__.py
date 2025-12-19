# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
OmniSeal Bench is a general framework for evaluating the performance of neural watermarking techniques.
At the moment we provide the evaluation tools for the audio watermarking models.
"""

from omnisealbench.api import explain, list_tasks, task
from omnisealbench.attacks.helper import AttackConfig
from omnisealbench.utils.model_card import get_model

__version__ = "1.0.0"
