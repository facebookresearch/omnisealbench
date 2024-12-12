# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import multiprocessing
import random
from datetime import datetime
from functools import singledispatch
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch

from .attacks import AudioEffects as af
from .attacks import get_attack_run_configs
from .configs import *
from .detection import run_audio_detection, run_image_detection
from .image_watermarkgen import run_image_watermarkgen
from .models import AudioWatermarkDetector, AudioWatermarkGenerator, Watermark
from .watermarkgen import run_watermarkgen


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


def build_generator(
    model_key: str, model_config: Dict[str, Any], device: str
) -> Union[AudioWatermarkGenerator, Watermark]:
    if model_key == "custom_audio":
        generator: AudioWatermarkGenerator = model_config[model_key]["generator"]
    elif model_key == "custom_image":
        # generator: Watermark = model_config[model_key]["generator"]
        raise NotImplementedError
    else:
        if "builder_func" in model_config:
            builder_func = model_config["builder_func"]
            generator: Watermark = builder_func(
                device=device,
                nbits=model_config["nbits"],
                **model_config["additional_arguments"],
            )
        else:
            generator_config = model_config["generator"]
            build_generator = generator_config["builder_func"]
            additional_arguments = generator_config["additional_arguments"]

            generator: AudioWatermarkGenerator = build_generator(
                model=model_key, device=device, **additional_arguments
            )

    return generator


def evaluate(
    eval_type: str,
    dataset: str,
    config_yaml: Optional[str] = None,
    watermark_override: bool = False,
    model: Union[
        str, Tuple[AudioWatermarkGenerator, AudioWatermarkDetector], Watermark
    ] = None,
    dataset_dir: Optional[str] = None,
    device: str = "cuda",
    num_samples: int = None,
    batch_size: int = 8,
    num_workers: int = 0,
    results_dir: str = ".",
    return_json: bool = False,
    seed: int = 42,
    save_ids: Optional[str] = None,
    append_results: bool = False,
) -> Dict[str, Any] | None:
    assert eval_type in ["audio", "image"]

    # Set seeds for reproductibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    results_dir = Path(results_dir)
    result_path = results_dir / f"{eval_type}_eval_results.json"

    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "eval": {},
    }
    if os.path.exists(result_path):
        if append_results:
            results = json.loads(open(result_path).read())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    (
        watermarking_config,
        audio_attacks_config,
        image_attacks_config,
        registered_attacks,
        models,
        datasets,
    ) = load_configs(config_yaml=config_yaml)

    if eval_type == "audio":
        attacks_config = audio_attacks_config
        wmk_config = watermarking_config["audio"]
    elif eval_type == "image":
        attacks_config = image_attacks_config
        wmk_config = watermarking_config["image"]
    else:
        raise

    registered_attacks = registered_attacks[eval_type]

    registered_models_configs = build_registered_models_configs(
        eval_type, models, model
    )
    registered_dataset_configs = build_registered_datasets_configs(
        eval_type, datasets, dataset, dataset_dir=dataset_dir
    )

    for db_key, db_config in registered_dataset_configs.items():
        db_results = {}

        if db_key in results["eval"]:
            db_results = results["eval"][db_key]
        else:
            results["eval"][db_key] = db_results

        for model_key, model_config in registered_models_configs.items():
            model_results = {}
            if model_key in db_results:
                # Could later try to resume evaluation from the last attack.
                # Cor the moment we consider that all the attacks have been evaluated
                # for this model (the partial saving happens at the end of the model evaluation block)
                # model_results = db_results[model_key]
                continue

            generator = build_generator(model_key, model_config, device)

            # watermark generation ..
            if eval_type == "audio":
                watermark_gen = run_watermarkgen(
                    generator=generator,
                    db_config=db_config,
                    override=watermark_override,
                    message_sz=model_config["nbits"],
                    target_sr=wmk_config["sample_rate"],
                    max_length_in_seconds=wmk_config["max_length_in_seconds"],
                )
            else:
                watermark_gen = run_image_watermarkgen(
                    generator=generator,
                    model_key=model_key,
                    db_config=db_config,
                    override=watermark_override,
                    message_sz=model_config["nbits"],
                    num_samples=num_samples,
                )

            assert watermark_gen["num_samples"] > 0, "No watermarked audio"

            samples_count = watermark_gen["num_samples"]

            if num_samples and num_samples < samples_count:
                samples_count = num_samples

            if model_key == "custom_audio":
                detector: AudioWatermarkDetector = model_config["detector"]
            elif model_key == "custom_image":
                raise NotImplementedError
            else:
                if "builder_func" in model_config:
                    detector: Watermark = generator
                else:
                    detector_config = model_config["detector"]
                    build_detector = detector_config["builder_func"]
                    additional_arguments = detector_config["additional_arguments"]
                    detector: AudioWatermarkDetector = build_detector(
                        device=device, **additional_arguments
                    )

            # evaluate watermarking detection ...
            for attack in registered_attacks:
                if eval_type == "audio":
                    if not hasattr(af, attack):
                        raise NotImplementedError(f"attack {attack} is not implemented")

                if attack in attacks_config:
                    attack_config = attacks_config[attack]
                    if "num_workers" in attack_config:
                        num_workers = attack_config["num_workers"]
                else:
                    if eval_type == "image":
                        raise NotImplementedError(
                            f"attack {attack} has no configuration"
                        )

                    attack_config = {}

                if eval_type == "audio":
                    attack_implementation: Callable[..., torch.Tensor] = getattr(
                        af, attack
                    )
                else:
                    attack_implementation = get_builder_func(attack_config["func"])

                for run_cfg in get_attack_run_configs(
                    attack_implementation, attack_config
                ):
                    attack_name = attack

                    if eval_type == "audio":
                        if "attrs" in run_cfg:
                            k, v = run_cfg["attrs"]
                            attack_name = f"{attack}_{k}_{v}"

                    if eval_type == "audio":
                        attack_results = run_audio_detection(
                            watermarks_dir=watermark_gen["dataset_path"],
                            watermarks_sample_rate=wmk_config["sample_rate"],
                            samples_count=samples_count,
                            detector=detector,
                            watermark_name=model_key,
                            attack_implementation=attack_implementation,
                            attack_runtime_config=run_cfg,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            description=f"{db_key} - {model_key} - {attack}",
                            results_dir=results_dir,
                            save_ids=save_ids,
                            attack=attack_name,
                        )
                    elif eval_type == "image":
                        attack_results = run_image_detection(
                            watermarks_dir=watermark_gen["dataset_path"],
                            samples_count=samples_count,
                            detector=detector,
                            device=device,
                            watermark_name=model_key,
                            attack_implementation=attack_implementation,
                            attack_runtime_config=run_cfg,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            description=f"{db_key} - {model_key} - {attack}",
                            results_dir=results_dir,
                            save_ids=save_ids,
                            attack=attack_name,
                        )

                    model_results[attack_name] = attack_results
            db_results[model_key] = model_results

            # save after each model evaluation (of all the attacks)
            with open(result_path, "w") as f:
                json.dump(results, f, default=to_serializable)

    if return_json:
        return results


def main():
    from fire import Fire

    # multiprocessing.set_start_method("spawn")

    Fire(evaluate)


if __name__ == "__main__":
    main()
