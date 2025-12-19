# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import functools
import logging
import os
from pathlib import Path
from time import time
from typing import List, Optional, Sequence, Union
import numpy as np
import torch

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

from omnisealbench.data.base import (MESSAGE_KEY, ORIGINAL_KEY, BaseDataset,
                                     build_data_loader, collate_even,
                                     collate_uneven)
from omnisealbench.utils.common import get_file_paths, message_to_tensor

logger = logging.getLogger(__name__)


def get_video_size(video_path):
    if ffmpeg is None:
        raise ImportError("ffmpeg-python is required for video processing. Install it with: pip install ffmpeg-python")
    try:
        probe = ffmpeg.probe(video_path)
    except AttributeError:
        # Fallback: ffmpeg-python might be installed as a different API
        import subprocess
        result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                                '-show_entries', 'stream=width,height', '-of', 'json', video_path],
                               capture_output=True, text=True)
        import json
        data = json.loads(result.stdout)
        width = data['streams'][0]['width']
        height = data['streams'][0]['height']
        return (width, height)
    info = [s for s in probe["streams"] if s["codec_type"] == "video"][0]
    return (info["width"], info["height"])


def get_video_frames(video_path, size=None, max_frames: Optional[int] = None):
    if ffmpeg is None:
        raise ImportError("ffmpeg-python is required for video processing. Install it with: pip install ffmpeg-python")
    try:
        cmd = ffmpeg.input(video_path)
    except AttributeError:
        # Fallback: use ffmpeg command line
        import subprocess
        cmd_args = ['ffmpeg', '-i', video_path]
        if isinstance(size, int):
            size = (size, size)
        if size:
            cmd_args.extend(['-vf', f'scale={size[0]}:{size[1]}'])
        cmd_args.extend(['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
        if max_frames:
            cmd_args.extend(['-vframes', str(max_frames)])
        
        result = subprocess.run(cmd_args, capture_output=True)
        out = result.stdout
        if size is None:
            size = get_video_size(video_path)
        video = np.frombuffer(out, np.uint8).reshape([-1, size[1], size[0], 3])
        return video

    if isinstance(size, int):
        size = (size, size)
    cmd = cmd.filter("scale", size[0], size[1])

    kwargs = {
        "format": "rawvideo",
        "pix_fmt": "rgb24",
    }
    if max_frames is not None:
        kwargs["vframes"] = max_frames

    out, _ = cmd.output("pipe:", **kwargs).run(capture_stdout=True, quiet=True)

    video = np.frombuffer(out, np.uint8).reshape([-1, size[1], size[0], 3])
    return video


@functools.lru_cache()
def get_video_paths(
    root_path: str,
    pattern: Optional[str] = None,
    sorted_by_index: bool = False,
    num_samples: Optional[int] = None,
) -> List[str]:
    if pattern is None:
        paths = []
        for path, _, files in os.walk(root_path):
            for filename in files:
                if filename.endswith(('.mp4')):
                    paths.append(os.path.join(path, filename))
    else:
        paths = list(map(str, Path(root_path).glob(pattern)))

    if sorted_by_index:
        sort_func = lambda x: int(Path(x).stem.split('_')[-1].split('.')[0])
    else:
        sort_func = lambda x: x
    res = sorted(paths, key=sort_func)
    if num_samples:
        res = res[:num_samples]
    return res


class VideoDataset(BaseDataset):
    
    def __init__(
        self,
        paths: Union[str, Sequence[str]],
        sorted_by_index: bool = False,
        file_pattern: Optional[str] = None,
        processed_file_pattern: Optional[str] = None,
        processed_data_name: str = "",
        message_pattern: Optional[str] = None,
        output_resolution: Optional[int] = None,
        max_frames: Optional[int] = None,
        num_samples: Optional[int] = None,
        logging: bool = False,
    ):
        # make sure paths and file_pattern are mutually exclusive
        if isinstance(paths, list) and file_pattern is not None:
            raise ValueError("Either 'paths' or 'file_pattern' should be provided, not both.")

        self.output_resolution = output_resolution
        self.max_frames = max_frames
        self.enable_logging = logging
        self.processed_data_name = processed_data_name

        self.samples = get_video_paths(
            root_path=paths,
            pattern=file_pattern,
            sorted_by_index=sorted_by_index,
            num_samples=num_samples,
        )

        if processed_file_pattern:
            self.processed_samples = get_video_paths(
                root_path=paths,
                pattern=processed_file_pattern,
                sorted_by_index=sorted_by_index,
            )
        else:
            self.processed_samples = None

        if message_pattern:
            self.message_samples = get_file_paths(
                root_path=str(paths),
                pattern=message_pattern,
            )
        else:
            self.message_samples = None
        if num_samples is not None:
            assert num_samples <= len(self.samples), f"There are {len(self.samples)} videos in total, but configured num_samples = {num_samples}"
            self.num_samples = num_samples
        else:
            self.num_samples = len(self.samples)

    def __getitem__(self, idx):
        assert 0 <= idx < len(self.samples)
        fn = self.samples[idx]

        t1 = time()
        video_frames = self.get_frames(fn)
        if self.enable_logging and idx < 2:
            logger.info(f"extract_frames: {fn} took {time() - t1:.2f} seconds")

        if self.processed_samples is None and self.message_samples is None:
            return idx, video_frames

        data_dict = {ORIGINAL_KEY: video_frames}

        if self.processed_samples is not None:
            processed_fn = self.processed_samples[idx]
            processed_video_frames = self.get_frames(processed_fn)
            if video_frames.size() != processed_video_frames.size():
                raise ValueError(f"Video frames size mismatch: {video_frames.size()} vs {processed_video_frames.size()}")
            data_dict[self.processed_data_name] = processed_video_frames

        if self.message_samples is not None:
            with open(self.message_samples[idx], 'r', encoding="utf-8") as f:
                msg_tensor = message_to_tensor(f.read().strip())
            data_dict[MESSAGE_KEY] = msg_tensor

        return idx, data_dict

    def get_frames(self, fn: str) -> torch.Tensor:
        size = get_video_size(fn)

        if self.output_resolution is not None:
            size_tmp_ = size
            # gcd_ = math.gcd(size[0], size[1])
            # gcd_size_mult_ = round(self.output_resolution / (min(size) / gcd_))
            # size = (size[0] // gcd_ * gcd_size_mult_, size[1] // gcd_ * gcd_size_mult_)
            mult_ = min(size) / self.output_resolution
            size = (int(size[0] / mult_), int(size[1] / mult_))
            size = (
                round(size[0] / 2) * 2,
                round(size[1] / 2) * 2,
            )  # prevent odd size -- results in ffmpeg error
            if self.enable_logging:
                logger.info(f"[INFO]: video {fn} resized from {size_tmp_} to {size}.", flush=True)

        frames = get_video_frames(fn, size, max_frames=self.max_frames)
        frames_torch = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float().div_(255.0)
        return frames_torch


def read_local_video_dataset(
    dataset_dir: str,
    video_pattern: Optional[str] = None,
    watermarked_video_pattern: Optional[str] = None,
    message_pattern: Optional[str] = None,
    even_shapes: bool = False,
    output_resolution: Optional[int] = None,
    max_frames: Optional[int] = None,
    batch_size: int = 1,
    num_samples: Optional[int] = None
):
    """
    Reads a local video dataset from the specified directory.

    Args:
        dataset_dir (str): The directory containing the video files.
        video_pattern (Optional[str]): The pattern to match video files.
        watermarked_video_pattern (Optional[str]): The pattern to match watermarked video files.
        message_pattern (Optional[str]): The pattern to match message files.
        output_resolution (Optional[int]): The resolution to resize videos to.
        max_frames (Optional[int]): The maximum number of frames to extract from each video.
        num_samples (Optional[int]): The number of samples to return. If None, returns all samples.

    Returns:
        VideoDataset: An instance of VideoDataset containing the loaded data.
    """

    sorted_by_index = False
    if video_pattern is not None or watermarked_video_pattern is not None:
        # If both patterns are provided, assume they are sorted by index
        sorted_by_index = True

    video_dataset = VideoDataset(
        paths=dataset_dir,
        file_pattern=video_pattern,
        processed_file_pattern=watermarked_video_pattern,
        processed_data_name="watermarked",
        message_pattern=message_pattern,
        output_resolution=output_resolution,
        max_frames=max_frames,
        sorted_by_index=sorted_by_index,
        num_samples=num_samples
    )

    # For now, ony support even shapes
    collate_fn = collate_even if even_shapes else collate_uneven

    return build_data_loader(
        video_dataset,
        batch_size=batch_size,
        collator=collate_fn,
    )
