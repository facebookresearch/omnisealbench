# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# type: ignore

from typing import Dict, Optional

import torch
import torch.nn as nn
import wavmark

from omnisealbench.interface import Device

# TODO: Check detection_bits = 0 with wavmark
class WavMark:
    """
    Wrapper over WavMark ("WavMark: Watermarking for Audio Generation")
    https://github.com/wavmark/wavmark

    Attributes:
        model (WavMark): The watermarking model used to generate watermarks.
        alpha (float): A scaling factor for watermark generation, default is 1.0.
    """

    model: nn.Module  # Replace with actual WavMark type when available
    nbits: int
    one_shot: bool
    sample_rate: int

    def __init__(
        self,
        model: nn.Module,
        nbits: int = 16,
        one_shot: bool = False,
        sample_rate: int = 16000,
    ):
        self.model = model
        self.nbits = nbits
        self.pattern_nbits = 32 - self.nbits
        if self.pattern_nbits < 0:
            raise ValueError("nbits must be less than or equal to 32")
        pattern_npy = wavmark.wm_add_util.fix_pattern[0 : self.pattern_nbits]

        self.pattern = torch.FloatTensor(pattern_npy).unsqueeze(0)  # 1 16

        # shift in seconds at detection time
        self.shift = 0.05

        # zero shot mode only look at first 1s, is usefull to reduce the computation time
        self.one_shot = one_shot
        self.gather_detections_bits = False
        self.sample_rate = sample_rate
        self.nbits = nbits

    @torch.inference_mode()
    @torch.inference_mode()
    def generate_watermark(
        self,
        contents: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        B, C, T = contents.shape
        assert C == 1, "WavMark only supports mono-channel audios"

        if len(message.shape) == 1:
            message = message.repeat(B, 1)

        assert message.shape[0] == B, "message must have shape (B, nbits)"
        assert message.shape[1] == self.nbits, "message must have shape (B, nbits)"

        # pattern + message
        pattern = self.pattern.repeat(B, 1).to(contents.device)  # b nbits-pattern
        if message.ndim == 1:
            message = message.repeat(B, 1)
        message = torch.cat([pattern, message], dim=-1)  # b 32

        # watermark in chunks of 16000 time steps
        watermark = torch.zeros_like(contents)  # b 1 t
        n_chunks = T // self.sample_rate
        for ii in range(n_chunks):
            t1 = ii * 16000
            t2 = (ii + 1) * 16000
            signal = contents[:, 0, t1:t2]  # b 16000  # b nbits
            audio_wm = self.model.encode(signal, message.float())  # b 16000
            watermark[:, 0, t1:t2] = audio_wm
        return watermark

    @torch.inference_mode()
    @torch.inference_mode()
    def detect_watermark(
        self,
        contents: torch.Tensor,
        detection_threshold: float = 0.5,
        message_threshold: float = 0.0,
        detection_bits: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        B, C, T = contents.shape
        assert C == 1, "WavMark only supports mono-channel audios"

        pattern = self.pattern.repeat(B, 1).to(contents.device)  # b nbits-pattern

        # initialize shift and results
        shift = int(self.shift * self.sample_rate)  # shift unit in timesteps
        shift_t = 0  # pointer to current timestep
        results = torch.zeros(
            B, 2 + self.nbits, T, device=contents.device
        )  # b 2+nbits t

        if self.gather_detections_bits:
            detection_results = torch.zeros(
                B, self.pattern_nbits, T, device=contents.device
            )  # b pattern_nbits t
        else:
            detection_results = None

        counts = torch.zeros(T, device=contents.device)  # t

        while shift_t + self.sample_rate <= T:
            # decode 16000 frames of signal
            t1 = shift_t
            t2 = shift_t + self.sample_rate
            signal = contents[:, 0, t1:t2]  # b 16000
            decoded = self.model.decode(signal)  # b 32

            # update detect results using bit acc on pattern bits
            decoded_hard = (decoded >= 0.5).int()  # b 32
            detected_pattern = decoded_hard[:, : self.pattern_nbits]  # b p
            bit_acc = detected_pattern == pattern  # b p -> b 1
            bit_acc = bit_acc.float().mean(dim=-1, keepdim=True)  # b 1
            bit_acc = bit_acc.repeat(1, self.sample_rate)  # b 1 -> b t

            # update detect results (2 first units) depending on bit acc and threshold
            results[:, 0, t1:t2] += (
                detection_threshold - bit_acc
            ) + 0.5  # activate (>0.5) when not wm
            results[:, 1, t1:t2] += (
                bit_acc - detection_threshold
            ) + 0.5  # activate (>0.5) when wm

            if self.gather_detections_bits:
                detected = (
                    decoded[:, : self.pattern_nbits].unsqueeze(-1).repeat(1, 1, t2 - t1)  # type: ignore
                )  # b pattern_nbits -> b pattern_nbits t
                detection_results[:, :, t1:t2] += detected - 0.5  # b pattern_nbits t

            # updates bit accs
            if self.nbits != 0:
                decoded = (
                    decoded[:, self.pattern_nbits :].unsqueeze(-1).repeat(1, 1, t2 - t1)
                )  # b 32 -> b nbits t
                results[:, 2:, t1:t2] += decoded - 0.5  # b nbits t -> b nbits t

            # update counts
            counts[t1:t2] += 1

            # if bit_acc > threshold (detected wm) shift by 1s, else shift by 50ms
            # TODO: shift is done batch-wise, could be done element-wise
            if bit_acc.mean() > detection_threshold:
                shift_t += self.sample_rate
            else:
                shift_t += shift

            # zero shot mode, only look at first 1s
            if self.one_shot:
                if self.gather_detections_bits:
                    detection_results = detection_results[:, :, : self.sample_rate]
                else:
                    detection_results = None

                results = results[:, :, : self.sample_rate]
                return {
                    "detection_bits": detection_results,
                    "message": results,
                }

        counts[counts == 0] = 1  # avoid division by 0

        # Detection + decoding results of shape (B, 2+nbits, T).
        results = results / counts.unsqueeze(0).unsqueeze(0)  # b 2+nbits t

        if self.gather_detections_bits:
            # Detection results of shape (B, pattern_nbits, T).
            detection_results = detection_results / counts.unsqueeze(0).unsqueeze(
                0
            )  # b pattern_nbits t
        else:
            detection_results = None

        detect_prob = results[:, 1, :].mean(dim=-1)  # b x 1

        decoded = results[:, 2:, :]  # b 2+nbits t -> b nbits t

        # average decision over time, then threshold
        decoded = decoded.mean(dim=-1) > message_threshold  # b nbits

        results = {
            "det_score": detect_prob,  # b x 1
            "det": detect_prob > detection_threshold,  # b x 1
            "message": decoded.int(),  # b x nbits
        }

        if self.gather_detections_bits:
            detected = detection_results  # b pattern_nbits t

            # average decision over time, then threshold
            detected = detected.mean(dim=-1) > message_threshold  # b pattern_nbits TODO: Check
            detected = detected.int()
            results["detection_bits"] = detected  # b x pattern_nbits

        return results


def build_model(
    model_name_or_path: str = "default",
    nbits: int = 16,
    one_shot: bool = False,
    sample_rate: int = 16000,
    device: Device = "cpu",
) -> WavMark:
    """
    Build a WavMark model ("WavMark: Watermarking for Audio Generation")
    https://github.com/wavmark/wavmark.
    Args:
        model_name_or_path (str): Path to the WavMark model or model card.
        nbits (int): Number of bits in the secret message.
        one_shot (bool): Whether to use zero-shot mode.
        sample_rate (int): Sample rate for audio processing.
        device (Device): The device to run the model on.
    Returns:
        WavMark: An instance of the WavMark generator.
    """
    model = wavmark.load_model(path=model_name_or_path)
    model = model.to(device)
    model.eval()
    return WavMark(model, nbits=nbits, one_shot=one_shot, sample_rate=sample_rate)
