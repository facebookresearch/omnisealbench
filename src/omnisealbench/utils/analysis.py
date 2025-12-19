# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pandas as pd


def aggregate_by_attack_variants(
        scores_df: pd.DataFrame, 
        group_cols: list[str] = ["attack", "attack_variant", "cat"],
        drop_cols: list[str] = ['idx', 'p_value', 'word_acc'],
) -> pd.DataFrame:
    benchmark_cols = ["decoder_time", "qual_time", "det_time", "attack_time"]
    agg_df_ = scores_df.drop(columns=drop_cols + benchmark_cols)

    columns_to_agg = set(agg_df_.columns.tolist()) - set(group_cols) - set(benchmark_cols)
    columns_to_agg = list(columns_to_agg)

    agg_df_ = agg_df_.groupby(group_cols)[columns_to_agg].agg(lambda x: x.mean(skipna=False)).reset_index()

    column_new_names = {
        "watermark_det": "TPR",
        "fake_det": "FPR",
    }
    agg_df_ = agg_df_.rename(columns=column_new_names)

    detection_metrics_cols = ["bit_acc", "log10_p_value", "TPR", "FPR", "watermark_det_score", "fake_det_score", "capacity"]
    detection_metrics_cols = [col for col in detection_metrics_cols if col in agg_df_.columns]
    # Get columns not in detection_metrics_cols
    other_cols = [col for col in agg_df_.columns if col not in detection_metrics_cols]
    # Reorder DataFrame
    agg_df_ = agg_df_[other_cols + detection_metrics_cols]

    return agg_df_


def aggregate_by_attacks(scores_df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_by_attack_variants(scores_df, group_cols=["attack", "cat"], drop_cols=['idx', 'p_value', 'word_acc', 'attack_variant'])

