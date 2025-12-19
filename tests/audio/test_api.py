# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from glob import glob
from pathlib import Path

import pytest
import torch
import torchaudio

from omnisealbench import get_model, task
from omnisealbench.data.base import InputBatch

audio_models = {
    "audioseal": {"nbits": 16},
    # "timbre": {"nbits": 30, "detection_bits": 14},
}


@pytest.mark.parametrize("model_name", audio_models.keys())
def test_audio_wm_pipeline_hf(tmp_path, model_name, device):

    wm_gen = task(
        "generation",
        modality="audio",
        dataset_name="hf-internal-testing/librispeech_asr_dummy",
        dataset_hf_subset="clean",
        dataset_split="validation[:4]",
        seed=42,
        result_dir=tmp_path,
        data_subdir=tmp_path / "data",
        padding_strategy="fixed",
        max_length=24000 * 2,  # 2 seconds of audio at 16kHz
        sample_rate=24000,
        batch_size=2,
        num_workers=2,
        metrics=[],  # There is no metrics, the pipeline just save the watermarked audios
    )
    model_args = audio_models[model_name]
    model = get_model(
        model_class_or_function=model_name,
        as_type="generator",
        device=device,
        **model_args,
    )
    _ = wm_gen(model)  # type: ignore

    saved_audios = glob(os.path.join(tmp_path, "data", "*.wav"))
    assert len(saved_audios) == 4, f"There should be 4 audio files saved, get {len(saved_audios)}"


@pytest.mark.parametrize("task_name", ["generation", "default"])
def test_toy_model(tmp_path, device, task_name):

    if task_name == "default":
        kwargs = {"attacks": [], "detection_bits": 4}
    else:
        kwargs = {}

    custom_task = task(
        task_name,
        modality="audio",
        dataset_name="hf-internal-testing/librispeech_asr_dummy",
        dataset_hf_subset="clean",
        dataset_split="validation[:4]",
        sample_rate=24_000,
        padding_strategy="longest",
        batch_size=2,
        num_workers=2,
        result_dir=tmp_path / "results",
        metrics=["snr", "pesq"],
        **kwargs,
    )
    toy_model_file = str(Path(__file__).parent / "toy.py")

    model_args = {"as_type": "generator"} if task_name == "generation" else {}
    model = get_model(toy_model_file, device=device, **model_args)

    metrics, scores = custom_task(model)  # type: ignore
    assert "snr" in metrics, "SNR metric should be computed"
    assert "pesq" in metrics, "PESQ metric should be computed"

    # For detection, raw scores is a list of metrics dictionary per attack
    if isinstance(scores, list):
        scores = scores[0]
    assert "idx" in scores and len(scores["idx"]) == 4, "There should be 4 IDs in the scores"


@pytest.mark.parametrize("model_name", audio_models.keys())
def test_audio_attacks(tmp_path, model_name, device):

    samples_cnt = 2
    fake_audios = [
        torch.randn(1, 24000 * 4) for _ in range(samples_cnt)
    ]  # 2 audio samples, each with 1 channel and 4 seconds of audio at 16kHz

    (tmp_path / "data").mkdir(exist_ok=True)
    for i, audio in enumerate(fake_audios):
        torchaudio.save(
            tmp_path / "data" / f"audio_{i}.wav",
            audio,
            sample_rate=24000,
        )

    result_dir = tmp_path / "results"
    eval_task = task(
        "detection",
        modality="audio",
        dataset_dir=tmp_path / "data",
        audio_pattern="audio_*.wav",
        watermarked_audio_pattern="audio_*.wav",  # We just fake the watermarked audios
        metrics=["pesq", "snr"],
        result_dir=tmp_path / "results",
        max_length=24000 * 2,  # 2 seconds of audio at 24kHz
        sample_rate=24000,
        overwrite=True,
        attacks=["bandpass_filter"],
        batch_size=2,
        ids_to_save="all"
    )

    # Get the detector
    detector = get_model(model_name, as_type="detector", device=device)
    _, scores = eval_task(detector, sample_rate=24000)
    for scores_ in scores:
        assert "idx" in scores_ and len(scores_["idx"]) == samples_cnt, f"There should be {samples_cnt} IDs in the scores"

    attack_dirs = [d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))]
    for attack_dir in attack_dirs:
        assert "metrics.json" in os.listdir(result_dir / attack_dir), f"There should be a metrics file in the attack subdirectory {attack_dir}"
        assert "data" in os.listdir(result_dir / attack_dir), f"There should be a data sub-directory in attack subdirectory {attack_dir}"
        assert len(list((result_dir / attack_dir / "data").glob("*.wav"))) == 2 * samples_cnt, f"There should be {2 * samples_cnt} audio files in the data sub-directory {attack_dir}/data for original and watermarked audios"
        assert len(list((result_dir / attack_dir / "data").glob("*.png"))) == 2 * samples_cnt, f"There should be {2 * samples_cnt} spectrogram files in the data sub-directory {attack_dir}/data for original and watermarked audios"

@pytest.mark.parametrize("model_name", audio_models.keys())
def test_audio_wm_pipeline_custom(tmp_path, model_name, device):
    on_the_fly_wm = task(
        "generation",
        modality="audio",
        seed=42,
        result_dir=tmp_path,
        data_subdir=tmp_path / "data",
        batch_size=2,
        num_workers=2,
        metrics=[],  # There is no metrics, the pipeline just save the watermarked audios
    )

    # a fake evaluation data loader that has 8 batches of 4 audio samples, each
    # with 1 channel and 4 seconds of audio at 16kHz
    # In OmniSealBench, each data loader must return a list of `InputBatch` objects,
    fake_audios = [
        InputBatch(torch.tensor(range(i, i + 4)), torch.randn(4, 1, 24000 * 4))
        for i in range(0, 32, 4)
    ]

    model_args = audio_models[model_name]
    # get_model() can get model-specific arguments in kwargs
    model = get_model(
        model_class_or_function=model_name,
        as_type="generator",
        device=device,
        **model_args,
    )

    _ = on_the_fly_wm(  # type: ignore
        model,
        batches=fake_audios
    )

    saved_messages = glob(os.path.join(tmp_path, "watermarks", "*.txt"))
    assert len(saved_messages) == 32, f"There should be 32 messages saved, get {len(saved_messages)}"

    saved_audios = glob(os.path.join(tmp_path, "data", "*.wav"))
    assert len(saved_audios) == 32, f"There should be 32 audio files saved, get {len(saved_audios)}"

    saved_wm_audios = glob(os.path.join(tmp_path, "watermarks", "*.wav"))
    assert len(saved_wm_audios) == 32, f"There should be 32 audio files saved, get {len(saved_wm_audios)}"


def test_audio_e2e(tmp_path, device):
    e2e_task = task(
        "default",
        modality="audio",
        dataset_name="hf-internal-testing/librispeech_asr_dummy",
        dataset_hf_subset="clean",
        dataset_split="validation[:4]",
        attacks=["updownresample"],
        metrics=["pesq", "snr"],
        seed=42,
        cache_dir=tmp_path / "intermediate",
        result_dir=tmp_path / "results",
        padding_strategy="fixed",
        max_length=24000,  # 2 seconds of audio at 24kHz
        sample_rate=24000,
        batch_size=2,
        num_workers=2,
    )

    model = get_model("wavmark_fast", device=device)
    _, scores = e2e_task(model)  # type: ignore
    for score_ in scores:
        assert "idx" in score_ and len(score_["idx"]) == 4, "There should be 4 IDs in the scores"
        assert "snr" in score_ and len(score_["snr"]) == 4, "There should be 4 SNR scores"
        assert "pesq" in score_ and len(score_["pesq"]) == 4, "There should be 4 PESQ scores"
