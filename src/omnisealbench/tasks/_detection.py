# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import time

from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional,
                    Sequence, Tuple, Union)

import torch

from omnisealbench.attacks.helper import (NO_ATTACK, AttackConfig,
                                          get_attack_category,
                                          no_attack_config, run_attack,
                                          unroll_attack_config)
from omnisealbench.data.base import MESSAGE_KEY, ORIGINAL_KEY, InputBatch
from omnisealbench.interface import (ContentT, Detector, ResultT, ScoresT,
                                     cast_input)
from omnisealbench.metrics.detection_scores import (detection_bits_accuracy,
                                                    message_accuracy)
from omnisealbench.tasks.base import (ID_COLUMN, WATERMARKED_KEY, Task,
                                      TaskConfig)
from omnisealbench.utils.common import (filter_by_index, get_device, get_dtype,
                                        is_index_in_range, resolve, set_seed,
                                        to_device, to_python_type)
from omnisealbench.utils.detection import get_detection_and_decoded_keys
from omnisealbench.utils.distributed import get_global_rank
from omnisealbench.utils.distributed_logger import rank_zero_print

DETECTION_RESULT_FILENAME = "detection_results"
ATTACK_RESULT_FILENAME = "attack_results.jsonl"

if TYPE_CHECKING:
    from pandas import DataFrame
    from rich.progress import Progress


@dataclass
class WatermarkAttacksConfig(TaskConfig):
    """Configuration for watermark attacks and detection tasks."""

    result_filename: str = ATTACK_RESULT_FILENAME

    detection_result_filename: str = DETECTION_RESULT_FILENAME

    attacks: Union[AttackConfig, List[AttackConfig], None] = None

    skip_quality_metrics_on_attacks: bool = False

    detection_threshold: Optional[float] = None

    message_threshold: Optional[float] = None

    detection_bits: Optional[int] = None

    data_subdir: Union[str, Path] = "data"
    """Subdirectory to store intermediate data files (attacked contents)."""

    ids_to_save: Optional[str] = None
    """If specified, only analyze these IDs by saving the attacked contents and detection results for these IDs."""

    def __post_init__(self):
        if self.ids_to_save is not None:
            self.data_subdir = "data"
        super().__post_init__()


def get_attacks_as_string(config: List[AttackConfig]) -> List[Tuple[str, str, str]]:
    """
    Get a list of attack names, variants, and categories as strings.
    """
    attacks = []
    for attack_cfg in config:
        assert isinstance(attack_cfg.attacks, AttackConfig)
        _name = attack_cfg.attacks.name or attack_cfg.attacks.func
        if "." in _name:
            _name = _name.split(".")[-1]

        attack_variant = (
            "__".join(f"{k}_{v}" for k, v in attack_cfg.attacks.args.items())
            if attack_cfg.attacks.args
            else "default"
        )
        attack_category = attack_cfg.attacks.get_category()
        attacks.append((_name, attack_variant, attack_category))
    return attacks


def parse_attack_dirs(result_dir: Path, modality: str) -> List[Tuple[str, str, str]]:
    """
    Parses the result directory to extract attack directories.
    Assumes the directory structure is like `results/attack_name/variant`.
    """
    attack_dirs = []
    for path in result_dir.iterdir():
        if (
            not path.is_dir()
            and not Path(path).joinpath(ATTACK_RESULT_FILENAME).exists()
        ):
            continue
        attack_dir = path.stem
        if attack_dir == "no_attack":
            attack_dirs.append(("no_attack", "default", "none"))
            continue

        strs = attack_dir.split("__")
        if len(strs) == 1:
            attack_name, attack_variant = strs[0], "default"
        else:
            attack_name, attack_variant = strs[0], strs[1]

        attack_cat = get_attack_category(attack_name, attacks_source=modality)  # type: ignore

        attack_dirs.append((attack_name, attack_variant, attack_cat))

    return attack_dirs


def skip_quality_metrics(config: WatermarkAttacksConfig) -> bool:
    if config.skip_quality_metrics_on_attacks:
        return True

    if not isinstance(config.attacks, AttackConfig):
        return True
    # if self.config.attacks.func == "identity":
    #     return True
    if config.attacks.get_category().lower() == "geometric":
        return True

    return False


def unroll_attack_configs(
    config: WatermarkAttacksConfig,
) -> Optional[Sequence[WatermarkAttacksConfig]]:
    """
    Create sub-tasks for watermark attacks and detection based on the configuration.
    Each sub-task will apply a specific attack
    """

    if config.attacks is None:
        config.attacks = no_attack_config

        # A special case: If no attack is specified, but we want to save the results,
        # we set the result_dir to a sub-directory "identity" to keep the structure
        # of the result directory consistent (i.e. results/attack).
        if config.result_dir:
            config.result_dir = f"{str(resolve(config.result_dir))}/identity"
        return None

    elif isinstance(config.attacks, AttackConfig):
        if config.attacks.func == NO_ATTACK:
            return None

        # If a single FunctionConfig is provided, we convert it to a list
        config.attacks = [no_attack_config, config.attacks]

    # If the attacks are a list, we ensure that no_attack is the first one
    else:
        has_no_attack_config = any(
            isinstance(attack, AttackConfig) and attack.func == NO_ATTACK
            for attack in config.attacks
        )
        if not has_no_attack_config:
            # Insert the identity config at the beginning of the list
            config.attacks.insert(0, no_attack_config)

    task_variant_cfgs = []
    for attack in config.attacks:
        cfgs = unroll_attack_config(attack)
        task_variant_cfgs.extend(cfgs)

    sub_configs: List[WatermarkAttacksConfig] = []

    for task_variant_cfg in task_variant_cfgs:
        sub_config = deepcopy(config)
        sub_config.attacks = task_variant_cfg
        if config.result_dir:
            sub_config.result_dir = (
                f"{str(resolve(config.result_dir))}/{task_variant_cfg}"
            )
        else:
            sub_config.result_dir = ""
        sub_configs.append(sub_config)

    return sub_configs


class WatermarkAttacksAndDetection(Task):
    """
    Base class for watermark attacks, i.e. applying a number of attacks and checking if the
    watermarks are still extracted / detected from tampered contents.

    This task is designed to work with two cases: Where the original
    content is not available, and the case where the original content is available.
    """

    config: WatermarkAttacksConfig

    def make_sub_tasks(self, config: Optional[WatermarkAttacksConfig] = None) -> Optional[Sequence[WatermarkAttacksConfig]]:  # type: ignore
        """
        Create sub-tasks for watermark attacks and detection based on the configuration.
        Each sub-task will apply a specific attack
        """
        return unroll_attack_configs(config or self.config)

    def skip_quality_metrics(self, config: WatermarkAttacksConfig) -> bool:
        return skip_quality_metrics(config)

    def skip_attack(self, model: Detector, attack: AttackConfig) -> bool:
        """
        Check if the attack should be skipped based on the configuration and the model
        """
        assert isinstance(attack, AttackConfig), f"Expected AttackConfig, got {type(attack)}"

        func_name = attack.func.split(".")[-1] if "." in attack.func else attack.func
        model_cls = model.__class__.__name__

        # TODO: Hard-code the model-specific incompatible attack list for now
        if func_name in ["center_crop", "resize", "comb"] and "DCTDWT" in model_cls:
            return True

        return False

    def __call__(  # type: ignore
        self,
        model: Detector,
        config: Optional[TaskConfig] = None,
        batches: Optional[Iterable[InputBatch]] = None,
        show_progress: bool = False,
        **kwargs,
    ):
        config = config or self.config
        metrics, raw_scores = super(WatermarkAttacksAndDetection, self).__call__(  # type: ignore
            model=model,
            config=config,
            batches=batches,
            show_progress=show_progress,
            **kwargs,
        )

        # If result_dir is specified, we save the final report
        if get_global_rank() == 0 and config.result_dir is not None:
            rank_zero_print(f"Saving final report to {config.result_dir}/report.csv")
            formatted_scores = self.print_scores(raw_scores, config=config)  # type: ignore
            formatted_scores.to_csv(
                f"{config.result_dir}/report.csv", header=True, index=False
            )

        return metrics, raw_scores

    def run(  # type: ignore
        self,
        model: Detector,  # type: ignore
        batches: Iterable[InputBatch],
        config: WatermarkAttacksConfig,  # type: ignore
        show_progress: bool = False,
        task_progress: Optional["Progress"] = None,
        **kwargs,
    ) -> ScoresT:
        results = defaultdict(list)  # type: ignore

        if config.seed is not None:
            set_seed(config.seed)
        if self.skip_attack(model, config.attacks):
            rank_zero_print(
                f"Skipping attack {config.attacks.func} for {self.__class__.__name__} on {model.__class__.__name__}"
            )
            return results  # type: ignore

        device = get_device(model)
        dtype = get_dtype(model)

        detect_func = model.detect_watermark
        if config.detection_threshold is None and hasattr(model, "detection_threshold"):
            detection_threshold = model.detection_threshold   # type: ignore
        else:
            detection_threshold = config.detection_threshold

        if config.message_threshold is None and hasattr(model, "message_threshold"):
            message_threshold = model.message_threshold  # type: ignore
        else:
            message_threshold = config.message_threshold

        # If detection_bits is not configured, try to get it from the model
        if config.detection_bits is None and hasattr(model, "detection_bits"):
            detection_bits = model.detection_bits  # type: ignore
        else:
            detection_bits = config.detection_bits

        detect_scoring_func = self.compute_detection_scores
        if detection_threshold is not None:
            detect_func = partial(detect_func, detection_threshold=detection_threshold)  # type: ignore
            detect_scoring_func = partial(detect_scoring_func, detection_threshold=detection_threshold)  # type: ignore

        if message_threshold is not None:
            detect_func = partial(detect_func, message_threshold=message_threshold)  # type: ignore

        if detection_bits is not None:
            detect_scoring_func = partial(detect_scoring_func, detection_bits=detection_bits)  # type: ignore
            detect_func = partial(detect_func, detection_bits=detection_bits)  # type: ignore

        # We save the results if the "result_dir" is specified, and "ids_to_save" is not None
        save_results: bool = bool(config.result_dir and config.data_subdir) and (
            config.save_all or config.ids_to_save is not None
        )

        # Print the run configuration
        if task_progress is None:
            rank_zero_print(f"Running {self.__class__.__name__} with attack: {config.attacks}")
        else:
            assert show_progress, "task_progress is only used when show_progress is True"

            # create current attack_variant task progress bar
            task_id = task_progress.add_task(
                description=f"{getattr(model, 'name', model.__class__.__name__)} - ({config.attacks})",
                total=len(batches),
            )
        global_message = kwargs.pop("wm_message", None)

        for batch in batches:
            # Ensure the batch is compatible with the model
            if not isinstance(batch, InputBatch):
                raise TypeError(f"Expected InputBatch, got {type(batch)}")
            assert isinstance(batch.data, (torch.Tensor, dict)) and (
                WATERMARKED_KEY in batch.data or ORIGINAL_KEY in batch.data
            ), "Detection needs both watermarked and original content"
            wm_input = cast_input(model.detect_watermark, batch.data[WATERMARKED_KEY])  # type: ignore
            orig_input = cast_input(model.detect_watermark, batch.data[ORIGINAL_KEY])  # type: ignore

            wm_input = to_device(wm_input, device, dtype=dtype)  # type: ignore
            orig_input = to_device(orig_input, device, dtype=dtype)

            attack_btime = time()
            attacked_wm_input = run_attack(config.attacks, config.seed, wm_input, orig_input, **kwargs)  # type: ignore
            batch_attack_time = time() - attack_btime

            del wm_input
            # Some attacks return the output and mask
            if isinstance(attacked_wm_input, tuple):
                attacked_wm_input = attacked_wm_input[0]

            attacked_orig_input = run_attack(config.attacks, config.seed, orig_input, orig_input, **kwargs)  # type: ignore
            if isinstance(attacked_orig_input, tuple):
                attacked_orig_input = attacked_orig_input[0]

            # Run detection on positive (attacked watermarked content) and negative (attacked original content) samples.
            # We also delete the attacked_wm_input (unless when we need to save the results for debugging, in which case we move
            # it to CPU to save memory)

            # Some attacks place the tensors in CPU, we need to ensure they are on the correct device
            attacked_wm_input = to_device(attacked_wm_input, device)
            attacked_orig_input = to_device(attacked_orig_input, device)

            decoder_btime = time()
            positive_detection = detect_func(attacked_wm_input)
            batch_decoder_time = time() - decoder_btime

            negative_detection = detect_func(attacked_orig_input)
            if MESSAGE_KEY in batch.data or global_message is not None:
                if MESSAGE_KEY in batch.data:
                    message = batch.data[MESSAGE_KEY]  # type: ignore
                else:
                    message = global_message
                if isinstance(message, list) and len(message) > 0 and isinstance(message[0], torch.Tensor):
                    message = torch.stack(message, dim=0)
                elif isinstance(message, torch.Tensor):
                    if message.ndim == 0:
                        message = message.unsqueeze(0).unsqueeze(0)
                    elif message.ndim == 1:
                        message = message.unsqueeze(0)
                message = to_device(message, device, dtype=dtype)  # type: ignore

                detection_btime = time()
                detection_scores = detect_scoring_func(
                    positive_detection, negative_detection, message=message, **kwargs  # type: ignore
                )
                batch_detection_time = time() - detection_btime
            else:
                message = None
                detection_scores = {}
                batch_detection_time = 0.0

            for k, v in detection_scores.items():
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().tolist()  # type: ignore
                results[k].extend(v)

            if not self.skip_quality_metrics(config):
                if config.metrics_device:
                    attacked_wm_input = to_device(
                        attacked_wm_input, config.metrics_device
                    )
                    orig_input = to_device(
                        orig_input, config.metrics_device
                    )
                # TODO: For now, only image requires passing both wm_input and orig_input to compute_quality_metrics.
                # For audios and videos, keeping orig_input is not necessary, and for videos this increases the memory
                # usage significantly.
                # We should find a way to pass only wm_input to compute_quality_metrics depending on the modality.
                quality_btime = time()
                quality_scores = self.compute_quality_metrics(attacked_wm_input, orig_input, **kwargs)  # type: ignore
                batch_quality_time = time() - quality_btime

                for k, v in quality_scores.items():
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().tolist()  # type: ignore
                    elif isinstance(v, float):
                        v = [v] * len(batch.indices)  # type: ignore
                    results[k].extend(v)
            else:
                batch_quality_time = 0.0

            # Add runtime numbers
            self.add_runtime_numbers(
                results,
                batch,
                batch_decoder_time,
                batch_quality_time,
                batch_detection_time,
                batch_attack_time,
            )

            # Add indices to results for easier debugging and reports
            results[ID_COLUMN].extend(batch.indices.cpu().tolist())  # type: ignore

            if save_results:
                self.save_evaluation_results(
                    config,
                    indices=batch.indices,
                    attacked_orig_input=attacked_orig_input,
                    attacked_wm_input=attacked_wm_input,
                    positive_detection=positive_detection,
                    negative_detection=negative_detection,
                    **kwargs
                )  # type: ignore

            if task_progress is not None:
                task_progress.update(task_id, advance=1)  # type: ignore

        if task_progress is not None:
            task_progress.stop_task(task_id)  # type: ignore

        return results  # type: ignore

    def add_runtime_numbers(
        self,
        results: defaultdict[Any, list],
        batch: InputBatch,
        batch_decoder_time: float,
        batch_quality_time: float,
        batch_detection_time: float,
        batch_attack_time: float,
    ) -> None:
        decoder_time_per_item = round(batch_decoder_time / len(batch.indices), 4)
        results["decoder_time"].extend([decoder_time_per_item] * len(batch.indices))  # type: ignore

        quality_time_per_item = round(batch_quality_time / len(batch.indices), 4)
        results["qual_time"].extend([quality_time_per_item] * len(batch.indices))  # type: ignore

        detection_time_per_item = round(
            batch_detection_time / len(batch.indices), 4
        )
        results["det_time"].extend([detection_time_per_item] * len(batch.indices))

        attack_time_per_item = round(batch_attack_time / len(batch.indices), 4)
        results["attack_time"].extend([attack_time_per_item] * len(batch.indices))

    def save_evaluation_results(
        self,
        config: WatermarkAttacksConfig,  # type: ignore
        indices: torch.Tensor,
        attacked_orig_input: ContentT,
        attacked_wm_input: ContentT,
        positive_detection: ResultT,
        negative_detection: ResultT,
        **kwargs,
    ) -> None:        
        output_dir = Path(config.result_dir) / config.data_subdir
        output_dir = resolve(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Resolve the indices to save
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().tolist()  # type: ignore

        if not config.save_all:
            # In this case, 'ids_to_save' must not be None and we have already parsed it to config.ids_range_to_save
            local_indices = [i for i, idx in enumerate(indices) if is_index_in_range(idx, config.ids_range_to_save)]  # type: ignore
            indices = [idx for idx in indices if is_index_in_range(idx, config.ids_range_to_save)]  # type: ignore
        else:
            local_indices = list(range(len(indices)))  # type: ignore

        if indices:
            orig_input_to_save = filter_by_index(attacked_orig_input, local_indices)  # type: ignore
            wm_input_to_save = filter_by_index(attacked_wm_input, local_indices)  # type: ignore
            self.save_data(orig_input_to_save, wm_input_to_save, indices, config=config, **kwargs)  # type: ignore

            positive_detection = {k: filter_by_index(v, local_indices) for k, v in positive_detection.items()}  # type: ignore
            negative_detection = {k: filter_by_index(v, local_indices) for k, v in negative_detection.items()}  # type: ignore
            result_dir = config.result_dir
            det_filename = config.detection_result_filename
            self._save_detection_results(positive_detection, indices, result_dir=result_dir, det_filename=det_filename, type="positive")  # type: ignore
            self._save_detection_results(negative_detection, indices, result_dir=result_dir, det_filename=det_filename, type="negative")  # type: ignore

    def compute_detection_scores(
        self,
        positive_detection: Dict[str, torch.Tensor],
        negative_detection: Dict[str, torch.Tensor],
        message: Optional[torch.Tensor] = None,
        detection_bits: Optional[int] = None,
        detection_threshold: float = 0.95,  # default value as taken from V1
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        assert (
            message is not None
        ), "watermark detection requires a message to be provided."

        # We only use the trick of extracting detection bits from the message for the models that do not generate the prediction
        # directly
        if (
            "prediction" not in positive_detection
            and "det" not in positive_detection
            and "det_score" not in positive_detection
        ):
            assert (
                detection_bits is not None
            ), "detection_bits must be set for the detector that do not generate prediction"
            gt_d = get_detection_and_decoded_keys(message, detection_bits=detection_bits)
            gt_detected = gt_d["detection_bits"]
            gt_decoded = gt_d["message"]
        else:
            gt_decoded = message
        scores = {}

        if "prediction" in positive_detection:
            # If the detection results are already in the form of predictions
            scores["watermark_det_score"] = positive_detection["prediction"]
            scores["watermark_det"] = (
                positive_detection["prediction"] > detection_threshold
            )  # Convert to binary detection
        elif "det" not in positive_detection and "det_score" not in positive_detection and positive_detection["detection_bits"].numel() > 0:
            det_acc = detection_bits_accuracy(positive_detection["detection_bits"], gt_detected, detection_threshold=detection_threshold)  # type: ignore
            scores["watermark_det_score"] = det_acc["det_score"]
            scores["watermark_det"] = det_acc["det"]
        else:
            scores.update(
                {
                    "watermark_" + k: v
                    for k, v in positive_detection.items()
                    if k in ["det", "det_score"]
                }
            )
        if "prediction" in negative_detection:
            # If the negative detection results are already in the form of predictions
            scores["fake_det_score"] = negative_detection["prediction"]  # type: ignore
            scores["fake_det"] = negative_detection["prediction"] > detection_threshold  # type: ignore
        elif "det" not in negative_detection and "det_score" not in negative_detection and negative_detection["detection_bits"].numel() > 0:
            neg_det_acc = detection_bits_accuracy(negative_detection["detection_bits"], gt_detected, detection_threshold=detection_threshold)  # type: ignore
            scores["fake_det_score"] = neg_det_acc["det_score"]
            scores["fake_det"] = neg_det_acc["det"]
        else:
            scores.update(
                {
                    "fake_" + k: v
                    for k, v in negative_detection.items()
                    if k in ["det", "det_score"]
                }
            )

        if "message" in positive_detection:
            true_msg_acc = message_accuracy(positive_detection["message"], gt_decoded)  # type: ignore
            for k, v in true_msg_acc.items():
                scores[f"{k}"] = v

        return scores

    def compute_quality_metrics(
        self,
        wm_input: ContentT,
        orig_input: ContentT,
        **kwargs,
    ) -> ScoresT:
        """
        Compute metrics based on the detection results.
        """
        raise NotImplementedError(
            "compute_metrics method must be implemented in the subclass of WatermarkAttacksAndDetection."
        )

    def print_scores(
        self,
        scores: Union[ScoresT, List[ScoresT]],
        config: Optional[WatermarkAttacksConfig] = None,
        attacks: Optional[List[Tuple[str, str, str]]] = None,
        **kwargs,
    ) -> "DataFrame":  # type: ignore
        """
        Format the results into a DataFrame.
        """
        import pandas as pd

        config = config or self.config

        if (
            not config.overwrite
            and config.result_dir
            and Path(config.result_dir).joinpath("report.csv").exists()
        ):
            return pd.read_csv(
                Path(config.result_dir) / "report.csv", header=0, index_col=None
            )

        if attacks is None:
            attack_cfgs = self.make_sub_tasks(config)
            if attack_cfgs is None:
                attack_cfgs = [config]
            attacks = get_attacks_as_string(attack_cfgs)
        dfs = []
        if isinstance(scores, dict):
            scores = [scores]  # type: ignore

        for (_name, attack_variant, attack_category), scores_ in zip(attacks, scores):
            scores_ = to_python_type(scores_)

            # Handle case where scores_ contains scalar values by wrapping them in lists
            processed_scores = {
                k: v if isinstance(v, list) else [v] for k, v in scores_.items()
            }

            df_ = pd.DataFrame(processed_scores)
            df_["attack"] = _name
            df_["attack_variant"] = attack_variant if _name != "none" else "default"
            df_["cat"] = attack_category if _name != "none" else "default"
            dfs.append(df_)
        
        # In other ranks, we do not gather results, so dfs can be empty
        if len(dfs) == 0:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def _save_detection_results(
        self,
        detection_results: ResultT,
        indices: Sequence[int],
        result_dir: Union[str, Path],
        det_filename: str,
        type: Literal["positive", "negative"] = "positive",
    ) -> None:
        """
        Save the detection results to a file.
        Args:
            detection_results (ResultT): The detection results to save.
            indices (Sequence[int]): The indices of the samples in the batch.
            filename (str): result file name.
        """
        assert isinstance(
            detection_results, dict
        ), "Detection results must be a dictionary"
        flatten_results = [{"id": idx} for idx in indices]
        for key, value in detection_results.items():
            if isinstance(value, torch.Tensor):
                value = list(value.detach().cpu())
            for i, val in enumerate(value):
                flatten_results[i][key] = val  # type: ignore

        outdir = resolve(result_dir).joinpath("detection")
        outdir.mkdir(parents=True, exist_ok=True)

        for d_ in flatten_results:
            idx = d_.pop("id", None)
            result_file = outdir.joinpath(f"{det_filename}_{type}_{idx}.pt")
            torch.save(d_, result_file)

    def save_data(
        self,
        orig: torch.Tensor,
        watermarked: torch.Tensor,
        indices: List[int],
        config: Optional[WatermarkAttacksConfig] = None,
        **kwargs
    ) -> None:
        """Save the detection results to the specified result directory."""
        ...
