# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from datetime import datetime
from pathlib import Path
import shutil
from typing import List
from tqdm import tqdm

import pandas as pd

from omnisealbench.commands.utils import extract_eval_metrics, quality_metrics_map
from omnisealbench.utils.common import (extract_numbers, parse_page_options,
                                        resolve)
from omnisealbench.utils.distributed_logger import rank_zero_print as print

benchmark_metrics = ["decoder_time"]
detection_metrics = ["bit_acc", "log10_p_value"]
attack_variations_metrics = [
    "bit_acc",
    "log10_p_value",
    "watermark_det",
    "fake_det",
    "watermark_det_score",
]

profiling_columns = ["decoder_time", "qual_time", "det_time", "attack_time"]

leaderboard_v1 = {
    "image": [
        "center_crop",
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
    ],
    "audio": [
        "speed",
        "random_noise",
        "lowpass_filter",
        "highpass_filter",
        "boost_audio",
        "duck_audio",
        "shush",
        "smooth",
        "aac_compression",
        "mp3_compression",
    ],
    "video": [
        "Rotate",
        "Resize",
        "Crop",
        "Brightness",
        "Contrast",
        "Saturation",
        "H264",
        "H264rgb",
        "H265",
        "VP9",
        "H264_Crop_Brightness0",
        "H264_Crop_Brightness1",
        "H264_Crop_Brightness2",
        "H264_Crop_Brightness3",
    ],
}


def save_benchmark_csv(df: pd.DataFrame, modality: str, output_file: str) -> None:
    """Save the benchmark CSV file."""
    benchmark_df = df.rename(columns={"attack": "attack_name"})
    columns_to_keep = [
        "model",
        "attack_name",
        "attack_variant",
        "cat",
    ] + quality_metrics_map[modality] + detection_metrics + ["decoder_time"]
    
    if "TPR" in benchmark_df.columns:
        columns_to_keep += ["TPR", "FPR"]
    
    benchmark_df = benchmark_df[columns_to_keep]
    benchmark_df.to_csv(output_file, header=True, index=False)


def save_examples_stats(final_json_output: Path, results: dict) -> None:
    with open(final_json_output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def save_attack_variations(df: pd.DataFrame, versrion: int, modality: str, output_file: str) -> None:
    """Save the attack variations CSV file."""
    columns_to_keep = [
        "strength",
        "model",
        "attack",
    ] + detection_metrics + quality_metrics_map[modality][:1]
    
    if "TPR" in df.columns:
        columns_to_keep += ["TPR", "FPR", "watermark_det_score"]

    columns_to_keep = list(set(columns_to_keep) & set(df.columns))
    attack_df = df[columns_to_keep]

    # filter to only attacks in the leaderboard_v1 if version==1
    if versrion == 1:
        attack_df = attack_df[attack_df["attack"].isin(leaderboard_v1[modality])]

    attack_df.to_csv(output_file, header=True, index=False)


def save_examples(image_dir: Path, output_dir_path: Path, modality: str, samples: str, ids: List[int]) -> None:
    total_input = list(image_dir.iterdir())

    # Hack to work with Omniseal Leaderboard (version up to 2025-09-04): 
    # For image modality, rename 'identity' attack to 'none'
    def _patch_image_identity_attack(attack):
        if attack == "identity":
            return "none"
        return attack

    def copy_and_rename(src_dir, dest_dir, prefix=""):
        dest_dir.mkdir(parents=True, exist_ok=True)
        for item in src_dir.iterdir():
            if item.is_file():
                idx = int(item.stem.split("_")[-1])
                if samples == "all" or (ids is not None and idx in ids):
                    new_name = f"{prefix}{item.name}"
                    shutil.copy2(item, dest_dir / new_name)

    for subdir in tqdm(total_input, total=len(total_input), desc="Export evaluation results models"):
        if not subdir.is_dir():
            continue
        model = subdir.name

        for attack_dir in subdir.iterdir():

            # Skip the directories that do not correspond to an attack, or to a skipped attack (no 'data' is saved)
            if not attack_dir.is_dir() or not (attack_dir / "data").exists():
                continue

            attack = attack_dir.name
            if modality == "image":
                attack = _patch_image_identity_attack(attack)

            examples_dir = output_dir_path / "examples" / modality / model / attack

            copy_and_rename(attack_dir / "data", examples_dir, prefix="attacked_")
            copy_and_rename(subdir / "identity" / "data", examples_dir, prefix="")


def example_ids_to_export(strategy: str):
    """A function that parse the sampling strategy and returns the list of ids to export, or None to sample all examples"""
    if not strategy:
        print("Export all samples.")
        strategy = "all"
    if strategy and strategy != "all":
        ids_range = parse_page_options(str(strategy))
        return extract_numbers(ids_range)
    return None


def parse_model_label_mapping(model_label: str) -> dict:
    """Parse the model label mapping from string to dict."""
    mapping = {}
    if model_label:
        pairs = model_label.split(",")
        for pair in pairs:
            key, value = pair.split(":")
            mapping[key.strip()] = value.strip()
    return mapping

class LeaderboardCommand:
    """Commands to connect / prepare data for the OmnisealBench leaderboard."""

    def export(
        self,
        source: str,
        target: str = "",
        version: int = 1,
        dataset_label: str = "default",
        model_label: str = "",
        modality: str = "",
        samples: str = "",
    ) -> None:
        """Export results to leaderboard format.
        
        Args:
            source: Directory containing the results.
            target: File to save the formatted leaderboard results.
            version: Version of the leaderboard (1 or 2). V1 exports to the existing
                leaderboard in HuggingFace:
                https://huggingface.co/spaces/facebook/omnisealbench
            dataset_label: Label of the dataset.
            model_label: Mapping from model to the labels for pretty printing of the model in the leaderboard. 
                Format: "model_name1:Pretty Name 1,model_name2:Pretty Name 2"
            modality: Modality type ('image', 'audio', or 'video').
            samples: Sampling strategy for examples to export.
        """
        assert modality in ["audio", "image", "video"], "Only accept modality='image', 'audio', or 'video'"

        results = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "eval": {},
        }

        model_mapping = parse_model_label_mapping(model_label)

        # Hack to work with Omniseal Leaderboard (version up to 2025-09-04): 
        # For image modality, rename 'identity' attack to 'none'
        def _patch_image_identity_attack(data: dict) -> dict:
            identity_val = None
            for attack, val in data.items():
                if attack == "identity":
                    identity_val = val
            data["none"] = identity_val
            del data["identity"]
            return data

        print(f"Export to {target}")
        ids = example_ids_to_export(samples)

        if not target:
            target = str((Path(__file__).parent.parent.parent / "leaderboard_space" / "backend" / "data").resolve())

        eval_results = {}
        final_pd = []
        result_dir_path = resolve(source)

        for subdir in result_dir_path.iterdir():
            if not subdir.is_dir():
                continue
            model = subdir.name
            if model in model_mapping:
                model = model_mapping[model]
            
            report_file = subdir / "report.csv"
            if not report_file.exists():
                continue
            json_res, pd_res = extract_eval_metrics(str(report_file), ids=ids)
            if modality == "image":
                json_res = _patch_image_identity_attack(json_res)

            json_res[model] = json_res
            pd_res["model"] = model
            pd_res["strength"] = pd_res["attack_variant"].apply(lambda x: x.split("_")[-1])
            pd_res["attack_variant"] = pd_res["attack_variant"].replace("__", "_")
            final_pd.append(pd_res)

        results["eval"][dataset_label] = eval_results
        final_df = pd.concat(final_pd, ignore_index=True)

        output_dir_path = resolve(target).joinpath(dataset_label)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # save examples_eval_results.json
        final_json_output = output_dir_path / "examples_eval_results.json"
        if not final_json_output.exists():
            save_examples_stats(final_json_output, results)

        # save modality_benchmark.csv
        benchmark_csv_output = output_dir_path / f"{modality}_benchmark.csv"
        if not benchmark_csv_output.exists():
            save_benchmark_csv(final_df, modality, str(benchmark_csv_output))

        # save modality_attacks_variations.csv
        final_attack_output = output_dir_path / f"{modality}_attacks_variations.csv"
        if not final_attack_output.exists():
            save_attack_variations(final_df, version, modality, str(final_attack_output))

        # save "examples" folder that contains images for each attack
        save_examples(result_dir_path, output_dir_path, modality, samples, ids)

        print("Export done. Please reload your leaderboard !")
