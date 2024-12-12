# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import lpips as lpips_lib
import numpy as np
import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from .attacks import run_attack as run_audio_attack
from .attacks import run_image_attack
from .attacks.image_quality import lpips, psnr, ssim
from .data.audio import WatermarkedAudioDataset
from .data.audio import apply_attacks as apply_audio_attacks
from .data.image import WatermarkedImageDataset, apply_attacks, collate_fn
from .metrics import (
    ScaleInvariantSignalNoiseRatio,
    ShortTimeObjectiveIntelligibility,
    compute_snr,
    get_bitacc,
    tensor_pesq,
)
from .models import AudioWatermarkDetector, Watermark
from .tools import is_index_in_range, parse_page_options
from .save_tools import save_audio_examples, save_image_examples

loss_fn_vgg: lpips_lib.LPIPS = None


def run_audio_detection(
    watermarks_dir: str,
    watermarks_sample_rate: int,
    samples_count: int,
    detector: AudioWatermarkDetector,
    watermark_name: str,
    attack_implementation: Callable[..., torch.tensor],
    attack_runtime_config: Dict[str, Any] = {},
    batch_size: int = 8,
    num_workers: int = 0,
    description: str = "",
    detection_threshold: float = 0.5,
    message_threshold: float = 0.5,
    results_dir: Optional[Path] = None,
    save_ids: Optional[str] = None,
    attack: Optional[str] = None,
):
    save_ids_ranges: Optional[List[Tuple[int, int]]] = None
    examples_dir: Optional[Path] = None
    if save_ids:
        save_ids_ranges = parse_page_options(save_ids)
        examples_dir = results_dir / "examples" / "audio" / watermark_name / attack
        examples_dir.mkdir(exist_ok=True, parents=True)

    if "attrs" in attack_runtime_config:
        k, v = attack_runtime_config["attrs"]

        description = f"{description}_{k}_{v}"
    attack_results = []

    with Progress(
        TextColumn(
            f"{description} •" + "[progress.percentage]{task.percentage:>3.0f}%"
        ),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as p:

        task_description = f"{description} • Processing..."
        task1 = p.add_task(task_description, start=True, total=samples_count)

        watermarked_dataset = WatermarkedAudioDataset(
            watermarks_dir=watermarks_dir,
            total=samples_count,
        )

        dataloader = DataLoader(
            watermarked_dataset,
            batch_size=batch_size,
            persistent_workers=num_workers > 0,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        cnt = 0
        for batch in dataloader:
            max_size = 0
            for _, audio, _, _ in batch:

                if audio.shape[1] > max_size:
                    max_size = audio.shape[1]

            padded_audio_batch = []
            padded_wmd_audio_batch = []

            for _, audio, wmd_audio, _ in batch:
                padding_size = (0, max_size - audio.shape[1])

                padded_wmd_audio = torch.nn.functional.pad(wmd_audio, padding_size)
                padded_audio = torch.nn.functional.pad(audio, padding_size)

                padded_audio_batch.append(padded_audio)
                padded_wmd_audio_batch.append(padded_wmd_audio)

            padded_audio_batch = torch.stack(padded_audio_batch)
            padded_wmd_audio_batch = torch.stack(padded_wmd_audio_batch)

            # apply effects on a batch of audio samples
            (
                padded_audio_batch,
                padded_wmd_audio_batch,
                attacked_padded_audio_batch,
                attacked_padded_wmd_audio_batch,
            ) = apply_audio_attacks(
                padded_audio_batch,
                padded_wmd_audio_batch,
                watermarks_sample_rate,
                attack_implementation,
                attack_runtime_config,
                run_audio_attack,
                detector.device,
                attack_orig_audio=True,
            )

            indices = []
            secret_messages = []

            for idx, _, _, secret_message in batch:
                indices.append(idx)
                secret_messages.append(secret_message)

            secret_messages = torch.cat(secret_messages, dim=0)  # b nbits
            secret_messages = secret_messages.to(detector.device)

            # Compute SNR
            snr: torch.Tensor = compute_snr(
                padded_audio_batch, attacked_padded_wmd_audio_batch
            )
            snr = snr.cpu().numpy()

            # Compute FPR
            watermark_results, messages = detector.detect_watermark_audio(
                attacked_padded_wmd_audio_batch,
                sample_rate=watermarks_sample_rate,
                detection_threshold=detection_threshold,
                message_threshold=message_threshold,
            )

            # test watermark detection on TRUE NEGATIVES
            tn_results, tn_messages = detector.detect_watermark_audio(
                attacked_padded_audio_batch,
                sample_rate=watermarks_sample_rate,
                detection_threshold=detection_threshold,
                message_threshold=message_threshold,
            )

            si_snr = ScaleInvariantSignalNoiseRatio().to(detector.device)
            stoi_snr = ShortTimeObjectiveIntelligibility(fs=watermarks_sample_rate)

            # bw_accs = get_bitacc(messages, secret_messages)
            # tn_bw_accs = get_bitacc(tn_messages, secret_messages)

            batch_idx = 0
            for (
                idx,
                detect_prob,
                tn_detect_prob,
            ) in zip(indices, watermark_results, tn_results):
                secret_message = secret_messages[batch_idx]
                message = messages[batch_idx]
                tn_message = tn_messages[batch_idx]

                # Bit accuracy: number of matching bits between the key and the message, divided by the total number of bits.
                bw_acc = get_bitacc(message[None, :], secret_message)
                tn_bw_acc = get_bitacc(tn_message[None, :], secret_message)

                sisnr = si_snr(
                    padded_audio_batch[batch_idx],
                    attacked_padded_wmd_audio_batch[batch_idx],
                )
                stoi = stoi_snr(
                    padded_audio_batch[batch_idx],
                    attacked_padded_wmd_audio_batch[batch_idx],
                )
                pesq = tensor_pesq(
                    padded_audio_batch[batch_idx][None, :],
                    attacked_padded_wmd_audio_batch[batch_idx][None, :],
                    sr=watermarks_sample_rate,
                )

                if save_ids:
                    if is_index_in_range(idx, save_ids_ranges):
                        save_audio_examples(
                            examples_dir,
                            idx,
                            watermarks_sample_rate,
                            padded_audio_batch[batch_idx],
                            padded_wmd_audio_batch[batch_idx],
                            attacked_padded_audio_batch[batch_idx],
                            attacked_padded_wmd_audio_batch[batch_idx],
                        )

                attack_results.append(
                    {
                        "idx": idx,
                        "detect_prob": detect_prob,
                        "tn_detect_prob": tn_detect_prob,
                        "ba": bw_acc,
                        "tn_ba": tn_bw_acc,
                        "snr": snr[batch_idx],
                        "sisnr": sisnr.item(),
                        "stoi": stoi.item(),
                        "pesq": pesq,
                    }
                )

                batch_idx += 1
            cnt += len(indices)
            p.update(task1, completed=cnt)

    return attack_results


def run_image_detection(
    watermarks_dir: str,
    samples_count: int,
    detector: Watermark,
    device: str,
    watermark_name: str,
    attack_implementation: Callable[..., torch.tensor],
    attack_runtime_config: Dict[str, Any] = {},
    batch_size: int = 1,
    num_workers: int = 0,
    description: str = "",
    detection_threshold: float = 0.05,
    results_dir: Optional[Path] = None,
    save_ids: Optional[str] = None,
    attack: Optional[str] = None,
):
    save_ids_ranges: Optional[List[Tuple[int, int]]] = None
    examples_dir: Optional[Path] = None
    if save_ids:
        save_ids_ranges = parse_page_options(save_ids)
        examples_dir = results_dir / "examples" / "image" / watermark_name / attack
        examples_dir.mkdir(exist_ok=True, parents=True)

    global loss_fn_vgg

    if loss_fn_vgg is None:
        # Initialize the LPIPS model once
        loss_fn_vgg = lpips_lib.LPIPS(net="vgg").to(device)

    if "attrs" in attack_runtime_config:
        k, v = attack_runtime_config["attrs"]

        description = f"{description}_{k}_{v}"
    attack_results = []

    if (
        attack in ["crop_01", "crop_02", "crop_05", "resize_01", "resize_07", "comb"]
        and "dctdwt" in watermark_name
    ):
        # skip this attack
        return attack_results

    with Progress(
        TextColumn(
            f"{description} •" + "[progress.percentage]{task.percentage:>3.0f}%"
        ),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as p:

        task_description = f"{description} • Processing..."
        task1 = p.add_task(task_description, start=True, total=samples_count)

        watermarked_dataset = WatermarkedImageDataset(
            watermarks_dir=watermarks_dir,
            total=samples_count,
        )

        dataloader = DataLoader(
            watermarked_dataset,
            batch_size=batch_size,
            persistent_workers=num_workers > 0,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
        )

        for batch in dataloader:
            attacked_wmd_img_batch = []
            attacked_img_batch = []
            message_batch = []

            # apply attacks on both original and watermarked images
            for _, watermarked_img, img, message in batch:
                attacked_wmd_img, attacked_img = apply_attacks(
                    img,
                    watermarked_img,
                    attack_implementation,
                    attack_runtime_config,
                    run_image_attack,
                    device,
                    attack_orig_img=True,
                )

                attacked_wmd_img_batch.append(attacked_wmd_img)
                attacked_img_batch.append(attacked_img)
                message_batch.append(message[None, :])

            # filtered_indices, filtered_imgs = detector.filter_batch()

            message_batch = np.concatenate(message_batch, axis=0)
            keys = (2 * torch.tensor(message_batch, dtype=torch.float32) - 1).to(device)
            if "trustmark" not in watermark_name and "wam" not in watermark_name:
                detection_keys = keys[:, :16]
                keys = keys[:, 16:]
            else:
                detection_keys = None

            # test watermark detection on attacked original image
            t = time.time()
            _, fake_detected = detector.decode_batch(
                [watermarked_dataset.untransform(a) for a in attacked_img_batch]
            )  # b c h w -> b k

            decoder_time = time.time() - t

            # test watermark detection on attacked watermarked image
            decoded, detected = detector.decode_batch(
                [watermarked_dataset.untransform(a) for a in attacked_wmd_img_batch]
            )  # b c h w -> b k

            fake_stats_batch = detector.get_detection_stats(
                fake_detected, detection_keys
            )
            wmd_stats_batch = detector.get_detection_stats(detected, detection_keys)
            dec_stats_batch = detector.get_decoding_stats(decoded, keys)

            for i in range(len(batch)):
                attacked_wmd_img = attacked_wmd_img_batch[i]
                attacked_img = attacked_img_batch[i]
                idx, wmd_img, img, _ = batch[i]
                fake_stats = fake_stats_batch[i]
                wmd_stats = wmd_stats_batch[i]
                dec_stats = dec_stats_batch[i]

                # Compute psnr, ssim, lpips
                psnr_score = psnr(attacked_wmd_img, attacked_img, img_space=None)
                lpips_score = lpips(
                    loss_fn_vgg, attacked_wmd_img, attacked_img, img_space=None
                )
                ssim_score = ssim(attacked_wmd_img, attacked_img, img_space=None)

                stats = {
                    "idx": idx,
                    "psnr": psnr_score.item(),
                    "ssim": ssim_score,
                    "lpips": lpips_score,
                    "decoder_time": decoder_time / len(batch),
                }

                for k, v in fake_stats.items():
                    stats[f"fake_{k}"] = v

                for k, v in wmd_stats.items():
                    stats[f"watermark_{k}"] = v

                stats.update(dec_stats)

                if save_ids:
                    if is_index_in_range(idx, save_ids_ranges):
                        save_image_examples(
                            examples_dir,
                            idx,
                            img,
                            wmd_img,
                            attacked_img,
                            attacked_wmd_img,
                            watermarked_dataset.untransform,
                        )

                attack_results.append(stats)
            p.update(task1, completed=len(attack_results))

    return attack_results
