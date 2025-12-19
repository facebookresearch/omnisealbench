"""
Utilities for value-based augmentations. Runtime checks now live in
`omni/tests/test_valuemetric.py`.
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io

import torch
import torch.nn as nn
from omnisealbench.models.wam_src.modules.dist import is_main_process
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image


def jpeg_compress(image: torch.Tensor, quality: int) -> torch.Tensor:
    """
    Compress a PyTorch image using JPEG compression and return as a PyTorch tensor.

    Parameters:
        image (torch.Tensor): The input image tensor of shape 3xhxw.
        quality (int): The JPEG quality factor.

    Returns:
        torch.Tensor: The compressed image as a PyTorch tensor.
    """
    assert image.min() >= 0 and image.max(
    ) <= 1, f'Image pixel values must be in the range [0, 1], got [{image.min()}, {image.max()}]'
    pil_image = transforms.ToPILImage()(image)  # convert to PIL image
    # Create a BytesIO object and save the PIL image as JPEG to this object
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    # Load the JPEG image from the BytesIO object and convert back to a PyTorch tensor
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    tensor_image = transforms.ToTensor()(compressed_image)
    return tensor_image


def webp_compress(image: torch.Tensor, quality: int) -> torch.Tensor:
    """
    Compress a PyTorch image using WebP compression and return as a PyTorch tensor.

    Parameters:
        image (torch.Tensor): The input image tensor of shape 3xhxw.
        quality (int): The WebP quality factor.

    Returns:
        torch.Tensor: The compressed image as a PyTorch tensor.
    """
    image = torch.clamp(image, 0, 1)  # clamp the pixel values to [0, 1]
    pil_image = transforms.ToPILImage()(image)  # convert to PIL image
    # Create a BytesIO object and save the PIL image as WebP to this object
    buffer = io.BytesIO()
    pil_image.save(buffer, format='WebP', quality=quality)
    # Load the WebP image from the BytesIO object and convert back to a PyTorch tensor
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    tensor_image = transforms.ToTensor()(compressed_image)
    return tensor_image


def median_filter(images: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Apply a median filter to tensors shaped (..., C, H, W)."""
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    if images.ndim < 3:
        raise ValueError("Expected at least 3D tensor with shape (..., C, H, W)")

    padding = kernel_size // 2
    original_shape = images.shape
    c, h, w = original_shape[-3:]
    flat = images.reshape(-1, c, h, w)

    flat = torch.nn.functional.pad(flat, (padding, padding, padding, padding))
    blocks = flat.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    medians = blocks.median(dim=-1).values.median(dim=-1).values
    return medians.reshape(*original_shape[:-3], c, h, w)


def median_filter_in_chunks(images: torch.Tensor, kernel_size: int, chunk_size=10) -> torch.Tensor:
    """Median filter helper that tolerates both BCHW and CHW tensors."""
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if images.ndim < 3:
        raise ValueError("Expected at least 3D tensor with shape (..., C, H, W)")

    padding = kernel_size // 2
    original_shape = images.shape
    c, h, w = original_shape[-3:]

    # to be compatible with both video/batch images and single image inputs (when used in the image modality)
    flat = images.reshape(-1, c, h, w)

    num_chunks = (flat.shape[0] + chunk_size - 1) // chunk_size
    processed_chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, flat.shape[0])
        chunk = flat[start_idx:end_idx]
        chunk_padded = torch.nn.functional.pad(chunk, (padding, padding, padding, padding))
        blocks = chunk_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        medians = blocks.median(dim=-1).values.median(dim=-1).values
        processed_chunks.append(medians)

    output = torch.cat(processed_chunks, dim=0)

    # restore the exact shape the caller passed in (argument images)
    return output.reshape(*original_shape[:-3], c, h, w)
    

def median_filter_quantile(images: torch.Tensor, kernel_size: int, chunk_size=10) -> torch.Tensor:
    """
    Apply a median filter to a batch of images using torch.quantile.
    Parameters:
        images (torch.Tensor): The input images tensor of shape BxCxHxW.
        kernel_size (int): The size of the median filter kernel.
        chunk_size (int, optional): The size of each chunk. Defaults to 10.
    Returns:
        torch.Tensor: The filtered images.
    """
    # Ensure the kernel size is odd
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    # Compute the padding size
    padding = kernel_size // 2
    # Calculate the number of chunks
    num_chunks = (images.shape[0] + chunk_size - 1) // chunk_size
    # Initialize an empty list to store the processed chunks
    processed_chunks = []
    # Process each chunk separately
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, images.shape[0])
        chunk = images[start_idx:end_idx]
        # Pad the chunk
        chunk_padded = torch.nn.functional.pad(
            chunk, (padding, padding, padding, padding))
        # Extract local blocks from the chunk
        blocks = chunk_padded.unfold(2, kernel_size, 1).unfold(
            3, kernel_size, 1)  # BxCxHxWxKxK
        # Flatten kernel dims so torch.quantile can operate on a single axis
        blocks = blocks.reshape(*blocks.shape[:4], -1)
        q = blocks.new_tensor(0.5)
        medians = torch.quantile(blocks, q=q, dim=-1)  # BxCxHxW
        processed_chunks.append(medians)
    # Concatenate the processed chunks into a single tensor
    output_images = torch.cat(processed_chunks, dim=0)
    return output_images    


class JPEG(nn.Module):
    def __init__(self, min_quality=None, max_quality=None, passthrough=True):
        super(JPEG, self).__init__()
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.passthrough = passthrough

    def get_random_quality(self):
        if self.min_quality is None or self.max_quality is None:
            raise ValueError("Quality range must be specified")
        return torch.randint(self.min_quality, self.max_quality + 1, size=(1,)).item()

    def jpeg_single(self, image, quality):
        if is_main_process():
            print(f"Applying JPEG compression with quality={quality}")
        if self.passthrough:
            return (jpeg_compress(image, quality).to(image.device) - image).detach() + image
        else:
            return jpeg_compress(image, quality).to(image.device)

    def forward(self, image: torch.tensor, mask=None, quality=None):
        quality = quality or self.get_random_quality()
        image = torch.clamp(image, 0, 1)
        if len(image.shape) == 4:  # b c h w
            for ii in range(image.shape[0]):
                image[ii] = self.jpeg_single(image[ii], quality)
        else:
            image = self.jpeg_single(image, quality)
        return image, mask
    
    def __repr__(self):
        return f"JPEG"


class GaussianBlur(nn.Module):
    def __init__(self, min_kernel_size=None, max_kernel_size=None):
        super(GaussianBlur, self).__init__()
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size

    def get_random_kernel_size(self):
        if self.min_kernel_size is None or self.max_kernel_size is None:
            raise ValueError("Kernel size range must be specified")
        kernel_size = torch.randint(self.min_kernel_size, self.max_kernel_size + 1, size=(1,)).item()
        return kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

    def forward(self, image, mask=None, kernel_size=None):
        kernel_size = kernel_size or self.get_random_kernel_size()
        image = F.gaussian_blur(image, kernel_size)
        return image, mask

    def __repr__(self):
        return f"GaussianBlur"


class MedianFilter(nn.Module):
    def __init__(self, min_kernel_size=None, max_kernel_size=None, passthrough=True):
        super(MedianFilter, self).__init__()
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.passthrough = passthrough

    def get_random_kernel_size(self):
        if self.min_kernel_size is None or self.max_kernel_size is None:
            raise ValueError("Kernel size range must be specified")
        kernel_size = torch.randint(self.min_kernel_size, self.max_kernel_size + 1, size=(1,)).item()
        return kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

    def forward(self, image, mask=None, kernel_size=None):
        kernel_size = kernel_size or self.get_random_kernel_size()
        
        image_filtered = median_filter_in_chunks(image, kernel_size, chunk_size=2)
        # image_filtered = median_filter(image, kernel_size)

        if self.passthrough:
            image = (image_filtered - image).detach() + image
        else:
            image = image_filtered
        return image, mask
    
    def __repr__(self):
        return f"MedianFilter"


class Brightness(nn.Module):
    def __init__(self, min_factor=None, max_factor=None):
        super(Brightness, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def get_random_factor(self):
        if self.min_factor is None or self.max_factor is None:
            raise ValueError("min_factor and max_factor must be provided")
        return torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor

    def forward(self, image, mask=None, factor=None):
        factor = self.get_random_factor() if factor is None else factor
        image = F.adjust_brightness(image, factor)
        return image, mask

    def __repr__(self):
        return f"Brightness"


class Contrast(nn.Module):
    def __init__(self, min_factor=None, max_factor=None):
        super(Contrast, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def get_random_factor(self):
        if self.min_factor is None or self.max_factor is None:
            raise ValueError("min_factor and max_factor must be provided")
        return torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor

    def forward(self, image, mask=None, factor=None):
        factor = self.get_random_factor() if factor is None else factor
        image = F.adjust_contrast(image, factor)
        return image, mask

    def __repr__(self):
        return f"Contrast"

class Saturation(nn.Module):
    def __init__(self, min_factor=None, max_factor=None):
        super(Saturation, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def get_random_factor(self):
        if self.min_factor is None or self.max_factor is None:
            raise ValueError("Factor range must be specified")
        return torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor

    def forward(self, image, mask=None, factor=None):
        factor = self.get_random_factor() if factor is None else factor
        image = F.adjust_saturation(image, factor)
        return image, mask

    def __repr__(self):
        return f"Saturation"


def adjust_hue_in_chunks(data: torch.Tensor, hue_factor: float, chunk_size: int = 10) -> torch.Tensor:
    """Adjust hue while limiting peak memory use.

    Supports tensors shaped (..., C, H, W) as returned by typical vision
    pipelines. When a single data tensor (C, H, W) is provided we temporarily
    add a batch dimension so the per-chunk iteration still operates over a
    batch axis.
    Args:
        data (torch.Tensor): The input video/image tensor with shape (batch_size, 3, height, width).
        hue_factor (float): The hue factor to apply.
        chunk_size (int, optional): The size of each chunk. Defaults to 10.
    Returns:
        torch.Tensor: The output video tensor with adjusted hue.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    if data.ndim < 3:
        raise ValueError("Expected at least 3D tensor with shape (..., C, H, W)")

    original_shape = data.shape
    c, h, w = original_shape[-3:]
    # Reshape so we always iterate chunks across the batch dimension.
    flat = data.reshape(-1, c, h, w)

    # Calculate the number of chunks
    num_chunks = (flat.shape[0] + chunk_size - 1) // chunk_size
    # Initialize an empty list to store the processed chunks
    processed_chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, flat.shape[0])
        chunk = flat[start_idx:end_idx]
        processed_chunks.append(F.adjust_hue(chunk, hue_factor=hue_factor))

    output = torch.cat(processed_chunks, dim=0)

    del processed_chunks
    torch.cuda.empty_cache()

    return output.reshape(*original_shape[:-3], c, h, w)


class Hue(nn.Module):
    def __init__(self, min_factor=None, max_factor=None):
        super(Hue, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def get_random_factor(self):
        if self.min_factor is None or self.max_factor is None:
            raise ValueError("Factor range must be specified")
        return torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor

    def forward(self, image, mask=None, factor=None):
        factor = self.get_random_factor() if factor is None else factor
        # image = F.adjust_hue(image, factor)
        image = adjust_hue_in_chunks(image, factor, chunk_size=2)

        return image, mask

    def __repr__(self):
        return f"Hue"


class Grayscale(nn.Module):
    def __init__(self):
        super(Grayscale, self).__init__()
        
    def forward(self, image, mask=None, *args, **kwargs):
        """
        Convert image to grayscale. The strength parameter is ignored.
        """
        # Convert to grayscale using the ITU-R BT.601 standard (luma component)
        # Y = 0.299 R + 0.587 G + 0.114 B
        grayscale = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        grayscale = grayscale.expand_as(image)
        return grayscale, mask

    def __repr__(self):
        return f"Grayscale"
