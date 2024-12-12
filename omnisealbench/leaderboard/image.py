# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd

from .base import get_attack_cat

image_attacks_cat = {
    "proportion": "Geometric",
    "collage": "Inpainting",
    "crop": "Geometric",
    "rot": "Geometric",
    "jpeg": "Compression",
    "brightness": "Visual",
    "contrast": "Visual",
    "saturation": "Visual",
    "sharpness": "Visual",
    "resize": "Geometric",
    "overlay_text": "Inpainting",
    "hflip": "Geometric",
    "perspective": "Geometric",
    "median_filter": "Visual",
    "hue": "Visual",
    "gaussian_blur": "Visual",
    "comb": "Mixed",
}

attacks_with_variations = [
    "crop",
    "jpeg",
    "brightness",
    "contrast",
    "saturation",
    "sharpness",
    "resize",
    "perspective",
    "median_filter",
    "hue",
    "gaussian_blur",
]
models = ["dctdwt", "fnns", "hidden", "ssl", "trustmark", "wam"]
metrics = [
    "psnr",
    "ssim",
    "lpips",
    "bit_acc",
    "p_value",
    "word_acc",
    "watermark_det_score",
]


def create_image_dataframe(image_data, dataset):
    rows = []

    for model_key, model_data in image_data["eval"][dataset].items():
        attack_rows = model_data["none"]

        psnr_avg = np.mean([r["psnr"] for r in attack_rows])
        ssim_avg = np.mean([float(r["ssim"]) for r in attack_rows])
        lpips_avg = np.mean([r["lpips"] for r in attack_rows])
        decoder_time_avg = np.mean([r["decoder_time"] for r in attack_rows])

        res = {
            "model": model_key,
            "psnr": psnr_avg,
            "ssim": ssim_avg,
            "lpips": lpips_avg,
            "decoder_time": decoder_time_avg,
        }

        for attack_name, attack_rows in model_data.items():
            if len(attack_rows) > 0 and "fake_det_score" in attack_rows[0]:
                fake_det_score_avg = np.mean([r["fake_det_score"] for r in attack_rows])
            else:
                fake_det_score_avg = np.nan
            fake_det_avg = np.mean([float(r["fake_det"]) for r in attack_rows])

            if len(attack_rows) > 0 and "watermark_det_score" in attack_rows[0]:
                watermark_det_score_avg = np.mean(
                    [r["watermark_det_score"] for r in attack_rows]
                )
            else:
                watermark_det_score_avg = np.nan
            watermark_det_avg = np.mean(
                [float(r["watermark_det"]) for r in attack_rows]
            )

            bit_acc_avg = np.mean([r["bit_acc"] for r in attack_rows])
            word_acc_avg = np.mean([float(r["word_acc"]) for r in attack_rows])
            p_value_avg = np.mean([r["p_value"] for r in attack_rows])

            res[f"{attack_name}_p_value"] = p_value_avg
            res[f"{attack_name}_bit_acc"] = bit_acc_avg
            res[f"{attack_name}_word_acc"] = word_acc_avg

            res[f"{attack_name}_fake_det_score"] = fake_det_score_avg
            res[f"{attack_name}_fake_det_value"] = fake_det_avg
            res[f"{attack_name}_watermark_det_score"] = watermark_det_score_avg
            res[f"{attack_name}_watermark_det_value"] = watermark_det_avg

        extra = {}
        for score in [
            "fake_det_score",
            "fake_det_value",
            "watermark_det_score",
            "watermark_det_value",
            "bit_acc",
            "word_acc",
            "p_value",
        ]:
            for k, v in res.items():
                if score in k:
                    cat = get_attack_cat(k, image_attacks_cat)
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
