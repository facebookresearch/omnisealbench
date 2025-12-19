# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

default_omniseal_cache = os.path.expanduser("~/.omnisealbench")
OMNISEAL_HOME = os.path.expanduser(os.getenv("OMNISEAL_HOME", default_omniseal_cache))
default_omniseal_checkpoint = os.path.join(default_omniseal_cache, "checkpoints")
OMNISEAL_CHECKPOINTS_DIR = os.path.expanduser(os.getenv("OMNISEAL_CHECKPOINTS", default_omniseal_checkpoint))

# os.makedirs(OMNISEAL_CHECKPOINTS_DIR, exist_ok=True)
