# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def test_attack_category():
    from omnisealbench.attacks.helper import load_default_attacks

    attacks = load_default_attacks(None, "video")

    for attack in attacks:
        if attack.name == "Perspective":
            assert attack.category == "Geometric"
        if attack.name == "Brightness":
            assert attack.category == "Visual"
