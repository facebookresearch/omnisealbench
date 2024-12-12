# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import numpy as np


def get_attack_cat(attack_name, attacks_cat):
    for a, v in attacks_cat.items():
        if attack_name.startswith(a):
            return v
    return None


def get_attack_model_variations(
    image_data, metrics, model, dataset, attacks_with_variations
):
    all_variations = {}

    for attack in attacks_with_variations:
        variations = []
        for k, attack_data in image_data["eval"][dataset][model].items():
            if len(attack_data) == 0:
                continue

            if attack in k:
                variation_key = k.replace(attack + "_", "")
                variation_val = variation_key.split("_")[-1]

                variation_stats = {}
                for metric in metrics:
                    # metric is absent
                    if metric not in attack_data[0]:
                        metric_avg = np.nan
                    else:
                        metric_avg = np.mean(
                            [
                                float(a[metric])
                                for a in attack_data
                                if not np.isnan(float(a[metric]))
                            ]
                        )

                    variation_stats[metric] = metric_avg

                variations.append((variation_val, variation_stats))
        all_variations[attack] = variations

    return all_variations


def create_attack_strength_variation_stats(
    modality_data,
    models,
    attacks_with_variations,
    metrics,
    dataset: str,
):
    all_dfs = []
    for attack in attacks_with_variations:
        all_variations = []

        for model in models:
            model_variations = get_attack_model_variations(
                modality_data, metrics, model, dataset, attacks_with_variations
            )[attack]
            all_variations.append(model_variations)

        interleaved_variations = [x for pair in zip(*all_variations) for x in pair]

        attack_stats = {
            "strength": [s[0] for s in interleaved_variations],
            "model": models * len(all_variations[0]),
            "attack": [attack] * len(interleaved_variations),
        }

        for _, v_stats in interleaved_variations:
            for m in metrics:
                m_value = v_stats[m]
                if m in attack_stats:
                    attack_stats[m].append(m_value)
                else:
                    attack_stats[m] = [m_value]

        all_dfs.append(pd.DataFrame(attack_stats))

    return pd.concat(all_dfs)
