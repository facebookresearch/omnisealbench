# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Sequence

from .image_folder import *


def get_dataset_implementation(
    db_config: Dict[str, Any],
) -> Callable[[], Sequence[str]]:

    builder_func = db_config["builder_func"]
    additional_arguments = db_config["additional_arguments"]
    db = builder_func(**additional_arguments)

    return db
