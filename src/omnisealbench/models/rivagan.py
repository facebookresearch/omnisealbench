# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Data To AI Lab. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in same directory of this source file.

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from omnisealbench.utils.common import set_path
from omnisealbench.utils.detection import get_detection_and_decoded_keys


class RivaGANWM:
    """
    Implementation of the RivaGAN watermark model.
    ("Robust Invisible Video Watermarking with Attention",
    https://arxiv.org/abs/1909.01285)
    """

    encoder: torch.nn.Module
    decoder: torch.nn.Module

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        seq_len: int = 3,
        nbits: int = 32,
        detection_bits: int = 16,
        keep_last_frames: bool = False,
        batch_size: Optional[int] = None,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.seq_len = seq_len
        self.seq_len_middle = seq_len // 2
        self.detection_bits = detection_bits
        self.nbits = nbits
        self.keep_last_frames = keep_last_frames
        self.batch_size = batch_size
        self.device = next(encoder.parameters()).device

    def to(self, device: Union[str, torch.device]) -> "RivaGANWM":
        """
        Move the model to the specified device.
        """
        self.encoder.to(device)
        self.decoder.to(device)
        self.device = device
        return self

    def _encode(self, imgs: torch.Tensor, msgs: torch.Tensor) -> torch.Tensor:
        """watermark a single video (sequence of frames) using sliding window"""

        f, c, height, width = imgs.shape
        num_windows = f - self.seq_len + 1
        
        if num_windows <= 0:
            if self.keep_last_frames:
                return torch.zeros((f, c, height, width), dtype=torch.float32)
            else:
                return torch.zeros((0, c, height, width), dtype=torch.float32)
        
        # Expand messages if needed: (1, nbits) -> (num_windows, nbits)
        if msgs.shape[0] == 1 and num_windows > 1:
            batch_msgs = msgs.expand(num_windows, -1)
        else:
            batch_msgs = msgs
        
        # Determine batch size
        batch_size = self.batch_size if self.batch_size is not None else num_windows
        
        encoded_frames = []
        for start_idx in range(0, num_windows, batch_size):
            end_idx = min(start_idx + batch_size, num_windows)
            
            # Collect only the windows needed for this batch (chunked collection)
            batch_windows = []
            for window_idx in range(start_idx, end_idx):
                frame_idx = window_idx + self.seq_len - 1
                window = imgs[frame_idx - self.seq_len + 1 : frame_idx + 1]  # (seq_len, C, H, W)
                batch_windows.append(window)
            
            batch_windows = torch.stack(batch_windows, dim=0)  # (B, seq_len, C, H, W)
            batch_msgs_chunk = batch_msgs[start_idx:end_idx]
            
            # Scale to range [-1.0, 1.0] and permute: (B, seq_len, C, H, W) -> (B, C, seq_len, H, W)
            x = batch_windows.float() * 2.0 - 1.0
            x = x.permute(0, 2, 1, 3, 4).contiguous().to(self.device)
            
            # Encode the batch
            y = self.encoder(x, batch_msgs_chunk.to(self.device))  # (B, C, seq_len, H, W)
            
            # Extract middle frame from each window
            images = torch.clamp(y[:, :, self.seq_len_middle, :, :], min=-1.0, max=1.0)  # (B, C, H, W)
            images = ((images + 1.0) * 127.5).int().float() / 127.5 - 1.0  # quantize
            images = ((images + 1.0) / 2).detach().cpu()
            
            encoded_frames.append(images)
        
        output = torch.cat(encoded_frames, dim=0)  # (num_windows, C, H, W)
        
        if self.keep_last_frames:
            # Pad with zeros at the beginning for frames that couldn't be encoded
            padding = torch.zeros((f - num_windows, c, height, width), dtype=torch.float32)
            return torch.cat([padding, output], dim=0)
        else:
            return output

    def _decode(self, imgs: torch.Tensor) -> torch.Tensor:
        """decode a single video (sequence of frames) using sliding window"""
        f, c, height, width = imgs.shape
        num_windows = f - self.seq_len + 1
        
        if num_windows <= 0:
            return torch.zeros((1, self.nbits), dtype=torch.float32, device=self.device)
        
        # Determine batch size
        batch_size = self.batch_size if self.batch_size is not None else num_windows
        
        messages = []
        for start_idx in range(0, num_windows, batch_size):
            end_idx = min(start_idx + batch_size, num_windows)
            
            # Collect only the windows needed for this batch (chunked collection)
            batch_windows = []
            for window_idx in range(start_idx, end_idx):
                frame_idx = window_idx + self.seq_len - 1
                window = imgs[frame_idx - self.seq_len + 1 : frame_idx + 1]  # (seq_len, C, H, W)
                batch_windows.append(window)
            
            batch_windows = torch.stack(batch_windows, dim=0)  # (B, seq_len, C, H, W)
            
            # Scale to range [-1.0, 1.0] and permute: (B, seq_len, C, H, W) -> (B, C, seq_len, H, W)
            x = batch_windows.float() * 2.0 - 1.0
            x = x.permute(0, 2, 1, 3, 4).contiguous().to(self.device)
            
            # Decode the batch
            y_out: torch.Tensor = self.decoder(x)  # (B, nbits)
            messages.append(y_out.detach())
        
        # Concatenate and average all decoded messages
        all_messages = torch.cat(messages, dim=0)  # (num_windows, nbits)
        averaged = torch.mean(all_messages, dim=0, keepdim=True)  # (1, nbits)
        return averaged

    @torch.inference_mode()
    def generate_watermark(
        self, contents: List[torch.Tensor], message: torch.Tensor
    ) -> List[torch.Tensor]:
        if message.ndim == 1:
            message = message.unsqueeze(0)
        return [self._encode(c, message) for c in contents]

    @torch.inference_mode()
    def detect_watermark(
        self,
        contents: Union[torch.Tensor, List[torch.Tensor]],
        detection_threshold: float = 0.95,  # not used
        message_threshold: float = 0.0,
        detection_bits: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = torch.concatenate([self._decode(c) for c in contents])  # b x nbits
        
        return get_detection_and_decoded_keys(
            outputs,
            detection_bits=detection_bits if detection_bits is not None else self.detection_bits,
            message_threshold=message_threshold,
        )


def build_model(
    model_card_or_path: str,
    seq_len: int = 3,
    nbits: int = 32,
    keep_last_frames: bool = False,
    detection_bits: int = 16,
    device: str = "cpu",
    batch_size: Optional[int] = None,
) -> RivaGANWM:
    """
    Build the RivaGAN watermark model.
    ("Robust Invisible Video Watermarking with Attention",
    https://arxiv.org/abs/1909.01285)

    Args:
        model_card_or_path (str): Path to the model card or pre-trained model.
        seq_len (int): Sequence length for video frames.
        nbits (int): Number of bits for the watermark message.
        keep_last_frames (bool): Whether to keep the last frames in the output.
        detection_bits (int): Number of bits for detection.
        device (str): Device to run the model on.
        batch_size (int, optional): Batch size for processing frames/windows.
            If None, process all frames at once.

    Returns:
        RivaGANWM: An instance of the RivaGAN watermark model.
    """

    with set_path(Path(__file__).parent.joinpath("rivagan_src")):
        encoder, decoder, _, _ = torch.load(
            model_card_or_path, weights_only=False, map_location=device
        )
        encoder.eval()
        decoder.eval()
    return RivaGANWM(
        encoder=encoder,
        decoder=decoder,
        seq_len=seq_len,
        keep_last_frames=keep_last_frames,
        nbits=nbits,
        detection_bits=detection_bits,
        batch_size=batch_size,
    )
