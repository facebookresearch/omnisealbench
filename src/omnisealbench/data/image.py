# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import os
from pathlib import Path
from typing import Callable, List, Optional

from torchvision import transforms
from torchvision.datasets.folder import default_loader, is_image_file

from omnisealbench.data.base import (MESSAGE_KEY, ORIGINAL_KEY, BaseDataset,
                                     build_data_loader, collate_even,
                                     collate_uneven)
from omnisealbench.utils.common import get_file_paths, message_to_tensor


@functools.lru_cache()
def get_image_paths(root_path: str, pattern: Optional[str] = None, sorted_by_index: bool = False, num_samples: Optional[int] = None) -> List[str]:
    if pattern is None:
        paths = []
        for path, _, files in os.walk(root_path):
            for filename in files:
                paths.append(os.path.join(path, filename))
    else:
        paths = list(map(str, Path(root_path).glob(pattern)))
    image_fns = [fn for fn in paths if is_image_file(fn)]
    if sorted_by_index:
        sort_func = lambda x: int(Path(x).stem.split('_')[-1].split('.')[0])
    else:
        sort_func = lambda x: x
    res = sorted(image_fns, key=sort_func)
    if num_samples:
        res = res[:num_samples]
    return res


class ImageDataset(BaseDataset):

    def __init__(
        self, 
        path: str,
        loader: Callable = default_loader,
        sorted_by_index: bool = False,
        file_pattern: Optional[str] = None,
        processed_file_pattern: Optional[str] = None,
        processed_data_name: str = "",
        message_pattern: Optional[str] = None,
        img_transform: Optional[Callable] = None,
        num_samples: Optional[int] = None
    ):
        # Main / original images
        self.samples = get_image_paths(path, pattern=file_pattern, sorted_by_index=sorted_by_index, num_samples=num_samples)
        if processed_file_pattern:
            self.processed_samples = get_image_paths(path, pattern=processed_file_pattern, sorted_by_index=sorted_by_index)
        else:
            self.processed_samples = None

        if message_pattern:
            self.message_samples = get_file_paths(path, pattern=message_pattern)
        else:
            self.message_samples = None

        self.loader = loader
        
        self.img_transform = img_transform
        self.transform = transforms.ToTensor()
        
        if num_samples is not None:
            assert num_samples <= len(self.samples), f"There are {len(self.samples)} images in {path}, but configured num_samples = {num_samples}"
            self.num_samples = num_samples
        else:
            self.num_samples = len(self.samples)
        
        if self.num_samples == 0:
            if file_pattern is None:
                file_pattern = "*"
            raise ValueError(f"No images found in {path} with pattern {file_pattern}")

        self.processed_data_name = processed_data_name

    def __getitem__(self, idx):
        img = self.loader(self.samples[idx])
        img = self.transform(img)
        if self.img_transform:
            img = self.img_transform(img)

        if self.processed_samples is None and self.message_samples is None:
            return idx, img

        data_dict = {ORIGINAL_KEY: img}

        if self.processed_samples is not None:
            processed_img = self.loader(self.processed_samples[idx])
            processed_img = self.transform(processed_img)

            if img.size() != processed_img.size():
                raise ValueError(f"Image size mismatch: {img.size()} vs {processed_img.size()}")

            data_dict[self.processed_data_name] = processed_img

        if self.message_samples is not None:
            with open(self.message_samples[idx], 'r', encoding="utf-8") as f:
                msg_tensor = message_to_tensor(f.read().strip())
            data_dict[MESSAGE_KEY] = msg_tensor

        return idx, data_dict


def read_local_image_dataset(
    dataset_dir: str,
    image_pattern: Optional[str] = None,
    watermarked_image_pattern: Optional[str] = None,
    message_pattern: Optional[str] = None,
    even_shapes: bool = False,
    batch_size: int = 1,
    num_samples: Optional[int] = None,
    output_size: Optional[int] = None,
):
    """
    Reads a local image dataset from a specified directory and returns a data loader.
    
    Args:
        dataset_dir (str): Path to the directory containing the image dataset.
        original_image_pattern (Optional[str], optional): Glob pattern to match original images. If provided with watermarked_image_pattern, enables index-based sorting. Defaults to None.
        watermarked_image_pattern (Optional[str], optional): Glob pattern to match watermarked images. If provided with original_image_pattern, enables index-based sorting. Defaults to None.
        message_pattern (Optional[str], optional): Glob pattern to match message files associated with images. Defaults to None.
        even_shapes (bool, optional): If True, collates images to have even shapes; otherwise, allows uneven shapes. Defaults to False.
        batch_size (int, optional): Number of samples per batch in the data loader. Defaults to 1.
        num_samples (Optional[int], optional): Number of samples to read from the dataset. If None, reads all samples. Defaults to None.
        output_size (Optional[int], optional): If provided, resizes images to have this short edge size. Defaults to None.
    Returns:
        BaseDataset: A dataset object containing the images.
    """

    sorted_by_index = False
    if image_pattern is not None and watermarked_image_pattern is not None:
        sorted_by_index = True
    
    resize_fn = transforms.Resize((output_size, output_size)) if output_size is not None else None

    image_dataset = ImageDataset(
        dataset_dir,
        file_pattern=image_pattern,
        processed_file_pattern=watermarked_image_pattern,
        processed_data_name="watermarked",
        message_pattern=message_pattern,
        img_transform=resize_fn,
        sorted_by_index=sorted_by_index,
        num_samples=num_samples,
    )
    collate_fn = collate_even if even_shapes else collate_uneven
    return build_data_loader(
        image_dataset,
        batch_size=batch_size,
        collator=collate_fn,
    )
