# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from omnisealbench.attacks.helper import NO_ATTACK
from omnisealbench.tasks._detection import (ATTACK_RESULT_FILENAME,
                                            parse_attack_dirs)
from omnisealbench.utils.common import resolve, to_python_type


# Quality metrics per modality. The first item is the main quality metric.
quality_metrics_map = {
    "image": ["psnr", "ssim", "lpips"],
    "audio": ["pesq", "stoi", "snr", "sisnr"],
    "video": ["vmaf", "psnr", "ssim", "msssim", "lpips"]
}


@dataclass
class SlurmCommandOptions:
    """
    Options for Slurm job submission.
    """
    partition: str = ""
    qos: Optional[str] = None
    account: Optional[str] = None
    log_dir: str = "submitit_logs"
    nodes: int = 1
    mem_gb: Optional[int] = None
    tasks_per_node: int = 1
    gpus_per_node: int = 0
    cpus_per_task: int = 5
    timeout_min: int = 720
    constraint: Optional[str] = None
    job_name: str = "omniseal_eval"
    max_jobarray_jobs: int = 16

    @classmethod
    def from_args(cls, args: Optional[Dict[str, str]] = None) -> 'SlurmCommandOptions':

        default_cfg_file = Path(__file__).parent.joinpath("configs/submitit_config.yaml")
        with open(default_cfg_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}
        if args:
            config_dict.update(args)
        return cls(**config_dict)

    def get_launcher(self):
        from omnisealbench.utils.slurm import Launcher

        return Launcher(
            cluster="slurm",
            partition=self.partition,
            qos=self.qos,
            account=self.account,
            log_dir=self.log_dir,
            max_jobarray_jobs=self.max_jobarray_jobs,
        )

    def get_runtime(self):
        from omnisealbench.utils.slurm import JobConfig

        return JobConfig(
            nodes=self.nodes,
            tasks_per_node=self.tasks_per_node,
            gpus_per_node=self.gpus_per_node,
            cpus_per_task=self.cpus_per_task,
            mem_gb=self.mem_gb,
            timeout_min=self.timeout_min,
            constraint=self.constraint,
        )

    def get_job_name(self) -> str:
        return self.job_name

    def __repr__(self) -> str:
        args = [
            f"--partition={self.partition}",
        ]
        if self.qos:
            args.append(f"--qos={self.qos}")
        if self.account:
            args.append(f"--account={self.account}")
        args.append(f"--log-dir={self.log_dir}")
        args.append(f"--nodes={self.nodes}")
        args.append(f"--tasks-per-node={self.tasks_per_node}")
        args.append(f"--gpus-per-node={self.gpus_per_node}")
        args.append(f"--cpus-per-task={self.cpus_per_task}")
        if self.mem_gb:
            args.append(f"--mem={self.mem_gb}G")
        args.append(f"--time={self.timeout_min}")
        if self.constraint:
            args.append(f"--constraint={self.constraint}")
        args.append(f"--job-name={self.job_name}")
        return " ".join(args)


class GracefulExit(Exception):
    """Custom exception for CLI errors. When this error is caught,
    CLI only prints the message without the stack trace, so the user is not overwhelmed."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def set_verbosity(verbose: bool = False):
    if not verbose:
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="torch")
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)


def report(input_dir: Path, modality: str):
    if Path(input_dir).joinpath("report.csv").exists():
        print(f"Report already exists at {input_dir}/report.csv")
        return

    # Get the detailed results
    raw_scores = []
    for attack_dir in input_dir.iterdir():
        if not attack_dir.is_dir() and not Path(attack_dir).joinpath(ATTACK_RESULT_FILENAME).exists():
            continue
        with open(attack_dir / ATTACK_RESULT_FILENAME, "r", encoding="utf-8") as f:
            scores = [json.loads(line) for line in f]
            raw_scores.append(scores[0])

    attacks = parse_attack_dirs(input_dir, modality)
    dfs = []
    for (func_name, attack_variant, attack_category), scores_ in zip(attacks, raw_scores):

        # Should we print the identity results?
        if func_name == NO_ATTACK:
            continue

        scores_ = to_python_type(scores_)
        df_ = pd.DataFrame(scores_)
        df_["attack"] = func_name
        df_["attack_variant"] = attack_variant if func_name != NO_ATTACK else "default"
        df_["cat"] = attack_category if func_name != NO_ATTACK else "default"
        df_.rename(columns={"det_score": "TPR", "det_score_negative": "FPR"}, inplace=True)
        dfs.append(df_)

    formatted_scores = pd.concat(dfs, ignore_index=True)
    formatted_scores.to_csv(f"{input_dir}/report.csv", header=True, index=False)
    print(f"Report saved to {input_dir}/report.csv")


def parse_args(args: Dict) -> Dict:
    """
    Parse the free arguments into generator, detector, and model arguments.
    Args:
    """
    res = {}
    for k, v in args.items():
        if k.startswith("generator."):
            res.setdefault("generator", {})[k[len("generator."):]] = v
        elif k.startswith("detector."):
            res.setdefault("detector", {})[k[len("detector."):]] = v
        elif k.startswith("slurm."):
            res.setdefault("slurm", {})[k[len("slurm."):]] = v
        else:
            res.setdefault("model", {})[k] = v

    if "generator" in res:
        for mk, mv in res.get("model", {}).items():
            res["generator"][mk] = mv

    if "detector" in res:
        for mk, mv in res.get("model", {}).items():
            res["detector"][mk] = mv

    return res


def extract_eval_metrics(report_file: str, ids: Optional[List[int]] = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Read the raw report and group metrics by attack variants. 
    Return the grouped scores in 2 format: JSON dict and Pandas, where the JSON dicts are just group of scores
    and the Pandas dataframe contains average scores"""
    df = pd.read_csv(report_file)

    if df.empty:
        raise ValueError(
            f"No data found in {report_file}. "
            "Most of the time, this happens when you specify a wrong input data (e.g. `dataset_dir`)."
            "Make sure your arguments are correct"
        )

    if ids is not None:
        df = df[df["idx"].isin(ids)]

    grouped_as_dict = {}
    for attack_name, group1 in df.groupby('attack'):
        grouped_as_dict[attack_name] = {}
        for attack_variant, group2 in group1.groupby('attack_variant'):
            grouped_as_dict[attack_name][attack_variant] = group2.to_dict(orient='records')

    if "watermark_det" in df.columns:
        df["watermark_det"] = df["watermark_det"].apply(lambda x: float(x))
        df = df.rename(columns={"watermark_det": "TPR"})
    
    if "fake_det" in df.columns:
        df["fake_det"] = df["fake_det"].apply(lambda x: float(x))
        df = df.rename(columns={"fake_det": "FPR"})

    df.drop(columns=["idx", "word_acc"], inplace=True, errors="ignore")
    grouped_as_df = df.groupby(["attack", "attack_variant", "cat"]).mean().reset_index()

    return grouped_as_dict, grouped_as_df


def merge_eval_results(result_dir: str, modality: str) -> pd.DataFrame:
    eval_results = {}
    eval_dfs = []
    result_dir_path = Path(result_dir)
    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "eval": {},
    }
    for subdir in result_dir_path.iterdir():
        if subdir.is_dir():
            model = subdir.name
            report_file = subdir / "report.csv"
            if report_file.exists():
                json_res, pd_res = extract_eval_metrics(str(report_file))
                eval_results[model] = json_res
                pd_res.insert(0, "model", model)
                eval_dfs.append(pd_res)
    results["eval"][f"{modality} data"] = eval_results

    # Save to output_dir
    output_dir_path = resolve(result_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    final_json_output = output_dir_path / f"{modality}_eval_results.json"
    if not final_json_output.exists():
        with open(final_json_output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    # Reformat the Pandas output to print to console in the same schema as V1
    eval_df = pd.concat(eval_dfs, ignore_index=True)
    columns = {
        "watermark_det_score": "wm_det_score",
        "fake_det_score": "no_wm_det_score",
    }
    return eval_df.rename(columns=columns)
