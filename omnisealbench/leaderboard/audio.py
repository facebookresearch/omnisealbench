# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd

from .base import get_attack_cat

audio_attacks_cat = {
    "speed": "TimeDomain",
    "updownresample": "TimeDomain",
    "echo": "TimeDomain",
    "random_noise": "AmplitudeDomain",
    "lowpass_filter": "AmplitudeDomain",
    "highpass_filter": "AmplitudeDomain",
    "bandpass_filter": "AmplitudeDomain",
    "smooth": "AmplitudeDomain",
    "boost_audio": "AmplitudeDomain",
    "duck_audio": "AmplitudeDomain",
    "shush": "AmplitudeDomain",
}

attacks_with_variations = [
    "random_noise",
    "lowpass_filter",
    "highpass_filter",
    "boost_audio",
    "duck_audio",
    "shush_fraction",
]
models = ["wavmark", "timbre", "audioseal"]
metrics = ["snr", "sisnr", "stoi", "ba", "pesq", "detect_prob"]


def create_audio_dataframe(audio_data, dataset):
    rows = []

    for model_key, model_data in audio_data["eval"][dataset].items():
        attack_rows = model_data["identity"]

        res = {
            "model": model_key,
        }

        for metric in ["snr", "sisnr", "stoi", "pesq"]:
            metric_avg = np.mean(
                [
                    float(r[metric])
                    for r in attack_rows
                    if not np.isnan(float(r[metric]))
                ]
            )
            res["metric"] = metric_avg

        for attack_name, attack_rows in model_data.items():
            detect_prob_avg = np.mean([float(r["detect_prob"]) for r in attack_rows])
            tn_detect_prob_avg = np.mean(
                [float(r["tn_detect_prob"]) for r in attack_rows]
            )
            ba_avg = np.mean([r["ba"] for r in attack_rows])
            tn_ba_avg = np.mean([r["tn_ba"] for r in attack_rows])

            res[f"{attack_name}_detect_prob"] = detect_prob_avg
            res[f"{attack_name}_tn_detect_prob"] = tn_detect_prob_avg
            res[f"{attack_name}_bit_acc"] = ba_avg
            res[f"{attack_name}_tn_bit_acc"] = tn_ba_avg

            for metric in ["snr", "sisnr", "stoi", "pesq"]:
                metric_avg = np.mean(
                    [
                        float(r[metric])
                        for r in attack_rows
                        if not np.isnan(float(r[metric]))
                    ]
                )
                res[f"{attack_name}_{metric}"] = metric_avg

        extra = {}
        for score in ["detect_prob", "tn_detect_prob", "bit_acc", "tn_bit_acc"]:
            for k, v in res.items():
                if score in k:
                    cat = get_attack_cat(k, audio_attacks_cat)

                    if cat is None:
                        continue

                    cat = f"{cat}_{score}"
                    if cat in extra:
                        extra[cat].append(v)
                    else:
                        extra[cat] = [v]

                    cat_avg = f"avg_{score}"
                    if cat_avg in extra:
                        extra[cat_avg].append(v)
                    else:
                        extra[cat_avg] = [v]

        cols_to_del = []
        for score in ["fake_det_score", "fake_det_value", "watermark_det_value"]:
            for k, v in res.items():
                if score in k:
                    cols_to_del.append(k)

        for k in cols_to_del:
            del res[k]

        for k, v in extra.items():
            res[k] = np.mean(v)

        rows.append(res)

    return pd.DataFrame(rows)
