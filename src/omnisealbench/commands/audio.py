# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import shutil
from pathlib import Path
from tempfile import mkdtemp
from time import time
from typing import List, Optional

import pandas as pd
import yaml

from omnisealbench.commands.utils import (SlurmCommandOptions, quality_metrics_map,
                                          merge_eval_results, parse_args,
                                          report, set_verbosity)
from omnisealbench.tasks._detection import parse_attack_dirs
from omnisealbench.tasks.audio import (
    evaluate_audio_watermark_attacks_and_detection,
    evaluate_audio_watermark_generation)
from omnisealbench.utils.common import auto_device
from omnisealbench.utils.distributed_logger import get_global_rank, rank_zero_print as print


def validate(**kargs):
    result_dir: str = kargs.get("result_dir", "")
    metrics: Optional[List[str]] = kargs.get("metrics", None)
    how_to_generate_message: str = kargs.get("how_to_generate_message", "per_dataset")
    padding: str = kargs.get("padding", "longest")
    max_length: Optional[int] = kargs.get("max_length", None)

    if not result_dir:
        raise ValueError(
            "'result_dir' must be specified and must not be empty. Run `omniseal eval audio --help` for more info."
        )

    if metrics == "all":
        metrics = quality_metrics_map["audio"]

    if how_to_generate_message not in ["per_batch", "per_dataset"]:
        raise ValueError(
            f"Invalid value for `how_to_generate_message`: {how_to_generate_message}. "
            "Permissible values are 'per_batch' or 'per_dataset'."
        )

    if padding not in ["longest", "fixed", "even", "uneven"]:
        raise ValueError(
            f"Invalid value for `padding`: {padding}. "
            "Permissible values are 'longest', 'fixed', 'even', or 'uneven'."
        )

    if padding == "fixed" and max_length is None:
        raise ValueError("If `padding` is 'fixed', `max_length` must be specified.")

    return result_dir, metrics


class AudioCommand:
    """Running tasks for audio datasets. Run: omniseal audio eval [task] [options]."""

    def generate_watermark(
        self,
        dataset_type: str = "",
        dataset_name: str = "",
        dataset_hf_subset: str = "",
        dataset_split: str = "test",
        dataset_dir: str = "",
        audio_pattern: str = "*.wav",
        metrics: Optional[List[str]] = None,
        seed: Optional[int] = None,
        result_dir: str = "results",
        overwrite: bool = False,
        message_size: Optional[List[int]] = None,
        how_to_generate_message: str = "per_dataset",
        padding: str = "longest",
        max_length: Optional[int] = None,
        sample_rate: Optional[int] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        num_samples: Optional[int] = None,
        device: Optional[str] = None,
        show_progress: bool = True,
        slurm: bool = False,
        model: str = "",
        **kwargs,
    ):
        """Generate audio watermarks for a given dataset."""
        result_dir, metrics = validate(
            result_dir=result_dir,
            metrics=metrics,
            how_to_generate_message=how_to_generate_message,
            padding=padding,
            max_length=max_length,
        )

        pipeline = evaluate_audio_watermark_generation(
            dataset_type=dataset_type,
            dataset_name=dataset_name,
            dataset_hf_subset=dataset_hf_subset,
            dataset_split=dataset_split,
            dataset_dir=dataset_dir,
            audio_pattern=audio_pattern,
            metrics=metrics,
            seed=seed,
            result_dir=result_dir,
            data_subdir="data",
            overwrite=overwrite,
            message_size=message_size,
            how_to_generate_message=how_to_generate_message,  # type: ignore
            padding_strategy=padding,  # type: ignore
            max_length=max_length,
            sample_rate=sample_rate,
            batch_size=batch_size,
            num_workers=num_workers,
            num_samples=num_samples,
        )

        _args = parse_args(kwargs)
        launcher_args = _args.get("slurm", None)
        model_kwargs = _args.get("model", {})

        device = auto_device(device)
        auto_distributed = (device != "cpu")
        btime = time()
        if slurm:
            slurm_opts = SlurmCommandOptions.from_args(launcher_args)
            print(f"Running SLURM job with options: {slurm_opts}")

            metrics, raw_scores = asyncio.run(
                pipeline.schedule(
                    model,
                    task_run_name=slurm_opts.job_name,
                    launcher=slurm_opts.get_launcher(),
                    runtime=slurm_opts.get_runtime(),
                    show_progress=show_progress,
                    device=device,
                    **model_kwargs,
                )
            )
        else:
            metrics, raw_scores = pipeline.eval(model, show_progress=show_progress, auto_distributed=auto_distributed, device=device, **model_kwargs)

        print(f"Metrics: {metrics}")
        if overwrite or not Path(result_dir).joinpath("report.csv").exists() and get_global_rank() == 0:
            # For backward compatibility
            attack_attrs = parse_attack_dirs(Path(result_dir), "audio")
            formatted_scores: pd.DataFrame = pipeline.print_scores(raw_scores, attacks=attack_attrs)
            formatted_scores.to_csv(f"{result_dir}/report.csv", header=True, index=False)

        print(f"Finished in: {(time() - btime):.2f} seconds")

    def detect_watermark(
        self,
        input_dir: str = "",
        audio_pattern: str = "data/*.wav",
        watermarked_audio_pattern: Optional[str] = "watermarks/*.wav",
        metrics: Optional[List[str]] = None,
        skip_quality_metrics_on_attacks: bool = False,
        detection_threshold: Optional[float] = None,
        message_threshold: Optional[float] = None,
        detection_bits: Optional[int] = None,
        seed: Optional[int] = None,
        result_dir: str = "results",
        ids_to_save: Optional[str] = None,
        overwrite: bool = False,
        attacks: Optional[str] = None,
        padding: str = "longest",
        max_length: Optional[int] = None,
        sample_rate: Optional[int] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        device: Optional[str] = None,
        show_progress: bool = True,
        slurm: bool = False,
        model: str = "",
        **kwargs,
    ):
        """Detect watermarks in audio files."""

        result_dir, metrics = validate(
            result_dir=result_dir,
            metrics=metrics,
            padding=padding,
            max_length=max_length,
        )

        pipeline = evaluate_audio_watermark_attacks_and_detection(
            dataset_dir=input_dir,
            audio_pattern=audio_pattern,
            watermarked_audio_pattern=watermarked_audio_pattern,
            metrics=metrics,
            skip_quality_metrics_on_attacks=skip_quality_metrics_on_attacks,
            detection_threshold=detection_threshold,
            message_threshold=message_threshold,
            detection_bits=detection_bits,
            seed=seed,
            result_dir=result_dir,
            ids_to_save=ids_to_save,
            overwrite=overwrite,
            attacks=attacks,
            padding_strategy=padding,  # type: ignore
            max_length=max_length,
            sample_rate=sample_rate,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Some models get detection_bits and detection_threshold, message_threshold
        if detection_bits is not None:
            kwargs["detection_bits"] = detection_bits
        if detection_threshold is not None:
            kwargs["detection_threshold"] = detection_threshold
        if message_threshold is not None:
            kwargs["message_threshold"] = message_threshold

        _args = parse_args(kwargs)
        launcher_args = _args.get("slurm", None)
        model_kwargs = _args.get("model", {})

        device = auto_device(device)
        auto_distributed = (device != "cpu")
        btime = time()
        if slurm:
            slurm_opts = SlurmCommandOptions.from_args(launcher_args)
            print(f"Running SLURM job with options: {slurm_opts}")

            metrics, raw_scores = asyncio.run(
                pipeline.schedule(
                    model,
                    task_run_name=slurm_opts.job_name,
                    launcher=slurm_opts.get_launcher(),
                    runtime=slurm_opts.get_runtime(),
                    show_progress=show_progress,
                    device=device,
                    **model_kwargs,
                )
            )
        else:
            metrics, raw_scores = pipeline.eval(model, show_progress=show_progress, auto_distributed=auto_distributed, device=device, **model_kwargs)

        print(f"Metrics: {metrics}")
        if overwrite or not Path(result_dir).joinpath("report.csv").exists() and get_global_rank() == 0:
            # For backward compatibility
            attack_attrs = parse_attack_dirs(Path(result_dir), "audio")
            formatted_scores: pd.DataFrame = pipeline.print_scores(raw_scores, attacks=attack_attrs)
            formatted_scores.to_csv(f"{result_dir}/report.csv", header=True, index=False)

        print(f"Finished in: {(time() - btime):.2f} seconds")

    def report(self, input_dir: str = ""):
        return report(input_dir=Path(input_dir), modality="audio")

    def __call__(
        self,
        config: Optional[str] = None,
        dataset_type: str = "",
        dataset_name: str = "",
        dataset_hf_subset: str = "",
        dataset_split: str = "",
        dataset_dir: str = "",
        audio_pattern: str = "*.wav",
        model: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        seed: Optional[int] = None,
        result_dir: str = "",
        ids_to_save: Optional[str] = None,
        overwrite: bool = False,
        message_size: Optional[int] = None,
        how_to_generate_message: str = "per_dataset",
        cache_dir: Optional[str] = None,
        detection_threshold: Optional[float] = None,
        message_threshold: Optional[float] = None,
        detection_bits: Optional[int] = None,
        attacks: Optional[str] = None,
        skip_quality_metrics_on_attacks: bool = False,
        padding: str = "longest",
        max_length: Optional[int] = None,
        sample_rate: Optional[int] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        num_samples: Optional[int] = None,
        device: Optional[str] = None,
        show_progress: bool = True,
        slurm: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        set_verbosity(verbose)

        # Load config from YAML file if provided, or from the default config file
        config_dict = {}
        config = config or str(Path(__file__).parent.joinpath("configs", "audio.yaml"))
        with open(config, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # Override config values with function arguments if provided
        def override(key, value, prefix=""):
            if value not in [None, "", []]:
                return value
            if prefix and prefix in config_dict and key in config_dict[prefix]:
                return config_dict[prefix][key]
            if key in config_dict:
                return config_dict[key]
            else:
                return None

        # Override dataset and data loading options
        dataset_type = override("dataset_type", dataset_type, "dataset")
        dataset_name = override("dataset_name", dataset_name, "dataset")
        dataset_hf_subset = override("dataset_hf_subset", dataset_hf_subset, "dataset")
        dataset_split = override("dataset_split", dataset_split, "dataset")
        dataset_dir = override("dataset_dir", dataset_dir, "dataset")
        audio_pattern = override("audio_pattern", audio_pattern, "dataset")

        padding = override("padding", padding, "data_loading")
        max_length = override("max_length", max_length, "data_loading")
        sample_rate = override("sample_rate", sample_rate, "data_loading")
        batch_size = override("batch_size", batch_size, "data_loading")
        num_samples = override("num_samples", num_samples, "data_loading")
        slurm = override("slurm", slurm)

        result_dir, metrics = validate(
            result_dir=result_dir,
            metrics=metrics,
            how_to_generate_message=how_to_generate_message,
            padding=padding,
            max_length=max_length,
        )

        # Override common evaluation options
        seed = override("seed", seed, "eval")
        result_dir = override("result_dir", result_dir, "eval")
        overwrite = override("overwrite", overwrite, "eval")
        device = override("device", device, "eval")
        device = auto_device(device)
        ids_to_save = override("ids_to_save", ids_to_save, "eval")
        show_progress = override("show_progress", show_progress, "eval")
        num_workers = override("num_workers", num_workers, "eval")
        metrics = override("quality_metrics", metrics, "eval")
        cache_dir = override("cache_dir", cache_dir, "eval")

        # Override watermark generation options
        message_size = override("message_size", message_size, "watermarking")
        how_to_generate_message = override("how_to_generate_message", how_to_generate_message, "watermarking")

        # Override detection options
        detection_threshold = override("detection_threshold", detection_threshold, "detection")
        message_threshold = override("message_threshold", message_threshold, "detection")
        detection_bits = override("detection_bits", detection_bits, "detection")
        attacks = override("attacks", attacks, "detection")
        skip_quality_metrics_on_attacks = override("skip_quality_metrics_on_attacks", skip_quality_metrics_on_attacks, "detection")

        # Input validation
        assert result_dir, "'result_dir' must be specified and must not be empty. Run `omniseal eval audio --help` for more info."

        # Some models get detection_bits and detection_threshold, message_threshold
        if detection_bits is not None:
            kwargs["detection_bits"] = detection_bits
        if detection_threshold is not None:
            kwargs["detection_threshold"] = detection_threshold
        if message_threshold is not None:
            kwargs["message_threshold"] = message_threshold

        _args = parse_args(kwargs)
        model_kwargs = _args.get("model", {})

        if "generator" in _args and "detector" in _args:
            generator_args, detector_args = _args["generator"], _args["detector"]
        else:
            generator_args = detector_args = _args

        if model is not None:
            if isinstance(model, (list, tuple)):
                models = list(model)
            else:
                assert isinstance(model, str), "Model should be a string representing the model name."
                models = model.split(",")
        else:
            models = config_dict["models"]

            # We turn off the inline arguments in multi-model evaluation
            if len(models) > 1 and len(model_kwargs) > 0:
                print(f"Warning: No `model` is specified, and we have {len(models)} models in the config.")
                model_kwargs_str = str(model_kwargs) if len(model_kwargs) > 0 else "empty"
                print(f"This means that the inline arguments ({model_kwargs_str}) will be ignored, and the model cards will be used instead.")
                model_kwargs = {}
        
        auto_distributed = (device != "cpu")
        if slurm:
            from omnisealbench.utils.slurm import run_tasks

            launcher_args = _args.get("slurm", None)

            slurm_opts = SlurmCommandOptions.from_args(launcher_args)
            job_name = slurm_opts.get_job_name()
            launcher = slurm_opts.get_launcher()
            runtime = slurm_opts.get_runtime()
        else:
            launcher = None
            job_name = ""
            runtime = None


        # Run the e2e pipeline in 2 steps: watermark generation and detection, each with all models
        wm_tasks, detect_tasks = [], []
        wm_dirs = []
        try:
            for model in models:
                if cache_dir is not None:
                    wm_dir = str(Path(cache_dir).joinpath(model))
                else:
                    wm_dir = mkdtemp(suffix=model)
                wm_dirs.append(wm_dir)

                wm_pipeline = evaluate_audio_watermark_generation(
                    dataset_name=dataset_name,
                    dataset_hf_subset=dataset_hf_subset,
                    dataset_split=dataset_split,
                    dataset_dir=dataset_dir,
                    audio_pattern=audio_pattern,
                    num_samples=num_samples,
                    metrics=metrics,
                    seed=seed,
                    result_dir=wm_dir,
                    data_subdir="data",
                    overwrite=overwrite,
                    message_size=message_size,
                    how_to_generate_message=how_to_generate_message,  # type: ignore
                    padding_strategy=padding,  # type: ignore
                    max_length=max_length,
                    sample_rate=sample_rate,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )

                # Schedule watermark generation for each model
                if slurm:
                    wm_tasks.append(
                        wm_pipeline.schedule(
                            model,
                            task_run_name=job_name + "_" + model,
                            launcher=launcher,
                            runtime=runtime,
                            show_progress=show_progress,
                            device=device,
                            **generator_args
                        )
                    )
                else:
                    print(f"Run watermak generation for {model}...")
                    btime = time()
                    wm_pipeline.eval(model, show_progress=show_progress, auto_distributed=auto_distributed, device=device, **generator_args)
                    print(f"Finished {model} in {(time() - btime):.2f} seconds.")

            # Execute all scheduled tasks concurrently
            if wm_tasks:
                print(f"Submit {len(wm_tasks)} tasks for watermark generation ({models})")
                btime = time()
                asyncio.run(run_tasks(wm_tasks))  # type: ignore
                print(f"Finished SLURM jobs for watermarking in {(time() - btime):.2f} seconds.")
            del wm_tasks[:]

            # Run detection tasks
            total_task_cnt = 0
            total_unfinished_task_cnt = 0
            for i, model in enumerate(models):
                model_result_dir = Path(result_dir).joinpath(model)
                detect_pipeline = evaluate_audio_watermark_attacks_and_detection(
                    dataset_dir=wm_dirs[i],
                    metrics=metrics,
                    detection_threshold=detection_threshold,
                    message_threshold=message_threshold,
                    detection_bits=detection_bits,
                    seed=seed,
                    result_dir=model_result_dir,
                    ids_to_save=ids_to_save,
                    overwrite=overwrite,
                    attacks=attacks,
                    skip_quality_metrics_on_attacks=skip_quality_metrics_on_attacks,
                    padding_strategy=padding,  # type: ignore
                    max_length=max_length,
                    sample_rate=sample_rate,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
                # log how many subtasks to be submitted
                total_subtasks_no, unfinished_subtasks_no = detect_pipeline.check_subtasks()

                if slurm:
                    total_task_cnt += total_subtasks_no
                    total_unfinished_task_cnt += unfinished_subtasks_no
                    detect_tasks.append(
                        detect_pipeline.schedule(
                            model,
                            task_run_name=job_name + "_detect_" + model,
                            launcher=launcher,
                            runtime=runtime,
                            show_progress=show_progress,
                            device=device,
                            **detector_args,
                        )
                    )
                else:
                    print(f"Run {unfinished_subtasks_no} detection tasks for {model}...")
                    btime = time()
                    detect_pipeline.eval(model, show_progress=show_progress, device=device, auto_distributed=auto_distributed, **detector_args)
            if detect_tasks:
                print(f"Number of attack-and-detection tasks in total: {total_task_cnt}.")
                print(f"Number of unfinished tasks that will be submitted: {total_unfinished_task_cnt}.")
                btime = time()
                asyncio.run(run_tasks(detect_tasks))  # type: ignore
                print(f"Finished SLURM jobs for detection in {(time() - btime):.2f} seconds.")
            del detect_tasks[:]

            # Merge all eval results and print the summary
            if get_global_rank() == 0:
                results_summary = merge_eval_results(result_dir, "audio")
                print(results_summary.head().to_string())

        finally:
            # If no cache_dir is specified, this means we are saving the 
            # watermarked result in tempdir, and we should remove them afterwards
            if not cache_dir:
                for tmp_dir in wm_dirs:
                    try:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        print(f"Cleaned up temporary directory: {tmp_dir}")
                    except Exception as e:
                        print(f"Warning: Failed to clean up temporary directory {tmp_dir}: {e}")

            # kill pending jobs in case of exception
            if launcher:
                print("Cleaning SLURM resource")
                launcher.clear()
            print("Finished.")
