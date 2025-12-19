# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Test with:
    python -m videoseal.augmentation.video
"""

import io
from typing import Optional, Tuple

import av
import numpy as np
import torch
import torch.nn as nn


class VideoCompression(nn.Module):
    """
    Simulate video artifacts by compressing and decompressing a video using PyAV.
    Attributes:
        codec (str): Codec to use for compression.
        crf (int): Constant Rate Factor for compression quality.
        fps (int): Frames per second of the video.
    """

    def __init__(self, codec: str = 'libx264', crf: int = 28, fps: int = 24):
        super(VideoCompression, self).__init__()
        self.codec = codec  # values [28, 34, 40, 46]
        self.pix_fmt = 'yuv420p' if codec != 'libx264rgb' else 'rgb24'
        self.threads = 1  # limit the number of threads to avoid memory issues
        self.crf = crf
        self.fps = fps

    def _preprocess_frames(self, frames) -> torch.Tensor:
        frames = frames.clamp(0, 1).permute(0, 2, 3, 1)
        frames = (frames * 255).to(torch.uint8).detach().cpu().numpy()
        return frames

    def _postprocess_frames(self, frames) -> torch.Tensor:
        frames = np.stack(frames) / 255
        frames = torch.tensor(frames, dtype=torch.float32)
        frames = frames.permute(0, 3, 1, 2)
        return frames

    def _compress_frames(self, buffer, frames) -> io.BytesIO:
        """
        Compress the input video frames.
        Uses the PyAV library to compress the frames, then writes them to the buffer.
        Finally, returns the buffer with the compressed video.
        """
        with av.open(buffer, mode='w', format='mp4') as container:
            stream = container.add_stream(self.codec, rate=self.fps)
            stream.width, stream.height = frames.shape[2], frames.shape[1]
            stream.pix_fmt = self.pix_fmt
            stream.options = {
                'crf': str(self.crf), 
                'threads': str(self.threads), 
                'x265-params': 'log_level=none'  # Disable x265 logging
            }
            for frame_arr in frames:
                frame = av.VideoFrame.from_ndarray(frame_arr, format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        buffer.seek(0)
        # file_size = buffer.getbuffer().nbytes
        # print(f"Compressed video size - crf:{self.crf} - {file_size / 1e6:.2f} MB")
        return buffer

    def _decompress_frames(self, buffer) -> list:
        """
        Decompress the input video frames.
        Uses the PyAV library to decompress the frames, then returns them as a list of frames.
        """
        with av.open(buffer, mode='r') as container:
            output_frames = []
            frame = ""
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format='rgb24')
                output_frames.append(img)
        return output_frames

    def forward(self, frames: torch.Tensor, mask: Optional[torch.Tensor], crf: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress and decompress the input video frames.
        Parameters:
            frames (torch.Tensor): Video frames as a tensor with shape (T, C, H, W).
            mask (torch.Tensor): Optional mask for the video frames.
            crf (int): Constant Rate Factor for compression quality, if not provided, uses the self.crf value.
        Returns:
            torch.Tensor: Decompressed video frames as a tensor with shape (T, C, H, W).
        """
        self.crf = crf or self.crf

        input_frames = self._preprocess_frames(frames)  # convert to np.uint8
        with io.BytesIO() as buffer:
            buffer = self._compress_frames(buffer, input_frames)
            output_frames = self._decompress_frames(buffer)
        output_frames = self._postprocess_frames(output_frames) 
        output_frames = output_frames.to(frames.device)

        compressed_frames = frames + (output_frames - frames).detach()
        del frames  # Free memory

        return compressed_frames, mask

    def __repr__(self) -> str:
        return f"Compressor(codec={self.codec}, crf={self.crf}, fps={self.fps})"


class BasicVideoCompression(VideoCompression):
    """
    A compressor augmenter that randomly selects a CRF value from a list of values.

    Attributes:
        codec (str): Codec to use for compression.
        fps (int): Frames per second of the video.
        crf_values (list): List of CRF values to select from.
    """
    def __init__(self, codec: str, crf_min: Optional[int] = None, crf_max: Optional[int] = None, fps: int = 24):
        super().__init__(codec=codec, fps=fps)
        self.crf_min = crf_min
        self.crf_max = crf_max

    def get_random_crf(self):
        if self.crf_min is None or self.crf_max is None:
            raise ValueError("min_crf and max_crf must be provided")
        return torch.randint(self.crf_min, self.crf_max + 1, size=(1,)).item()

    def forward(self, frames: torch.Tensor, mask: Optional[torch.Tensor] = None, crf: Optional[int] = None) -> torch.Tensor:
        crf = crf or self.get_random_crf()
        return super().forward(frames, mask, crf)


class H264(BasicVideoCompression):
    def __init__(self, crf_min: Optional[int] = None, crf_max: Optional[int] = None, fps: int = 24):
        super().__init__(codec='libx264', crf_min=crf_min, crf_max=crf_max, fps=fps)

    def __repr__(self) -> str:
        return "H264"


class H264rgb(BasicVideoCompression):
    def __init__(self, crf_min: Optional[int] = None, crf_max: Optional[int] = None, fps: int = 24):
        super().__init__(codec='libx264rgb', crf_min=crf_min, crf_max=crf_max, fps=fps)

    def __repr__(self) -> str:
        return "H264rgb"


class H265(BasicVideoCompression):
    def __init__(self, crf_min: Optional[int] = None, crf_max: Optional[int] = None, fps: int = 24):
        super().__init__(codec='libx265', crf_min=crf_min, crf_max=crf_max, fps=fps)

    def __repr__(self) -> str:
        return "H265"


class VP9(BasicVideoCompression):
    def __init__(self, crf_min: Optional[int] = None, crf_max: Optional[int] = None, fps: int = 24):
        super().__init__(codec='libvpx-vp9', crf_min=crf_min, crf_max=crf_max, fps=fps)

    def __repr__(self) -> str:
        return "VP9"


class AV1(BasicVideoCompression):
    def __init__(self, crf_min: Optional[int] = None, crf_max: Optional[int] = None, fps: int = 24):
        super().__init__(codec='libaom-av1', crf_min=crf_min, crf_max=crf_max, fps=fps)

    def __repr__(self) -> str:
        return "AV1"
