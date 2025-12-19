# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from pathlib import Path
from functools import lru_cache
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as F_vision
from augly.image import functional as aug_functional
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms import functional

default_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

normalize_vqgan = transforms.Normalize(
    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
)  # Normalize (x - 0.5) / 0.5
unnormalize_vqgan = transforms.Normalize(
    mean=[-1, -1, -1], std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
)  # Unnormalize (x * 0.5) + 0.5



def create_gradient_background(width=512, height=512, start_color=(255, 0, 0), end_color=(0, 0, 255)):
    """Create a horizontal gradient background"""
    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)
    
    for x in range(width):
        # Calculate interpolation factor
        factor = x / width
        
        # Interpolate between start and end colors
        r = int(start_color[0] * (1 - factor) + end_color[0] * factor)
        g = int(start_color[1] * (1 - factor) + end_color[1] * factor)
        b = int(start_color[2] * (1 - factor) + end_color[2] * factor)
        
        # Draw vertical line with interpolated color
        draw.line([(x, 0), (x, height)], fill=(r, g, b))
    
    return image


def create_noise_background(width=512, height=512, noise_type='uniform'):
    """Create a noise background"""
    if noise_type == 'uniform':
        # Uniform random noise using torch RNG
        t = torch.randint(0, 256, (height, width, 3), dtype=torch.uint8)
        noise_array = t.cpu().numpy()
    elif noise_type == 'gaussian':
        # Gaussian noise using torch RNG
        t = torch.normal(mean=128.0, std=50.0, size=(height, width, 3))
        t = torch.clamp(t, 0.0, 255.0).to(torch.uint8)
        noise_array = t.cpu().numpy()
    elif noise_type == 'salt_pepper':
        # Salt and pepper noise using torch RNG
        t = torch.full((height, width, 3), 128, dtype=torch.uint8)
        mask_salt = torch.rand((height, width)) < 0.1
        mask_pepper = torch.rand((height, width)) < 0.1
        # expand masks to 3 channels
        mask_salt = mask_salt.unsqueeze(-1).expand(-1, -1, 3)
        mask_pepper = mask_pepper.unsqueeze(-1).expand(-1, -1, 3)
        t[mask_salt] = 255
        t[mask_pepper] = 0
        noise_array = t.cpu().numpy()
    
    return Image.fromarray(noise_array)


def create_checkerboard_background(width=512, height=512, square_size=32, color1=(255, 255, 255), color2=(0, 0, 0)):
    """Create a checkerboard pattern background"""
    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)

    for x in range(0, width, square_size):
        for y in range(0, height, square_size):
            # Determine if this square should be color1 or color2
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                color = color1
            else:
                color = color2

            draw.rectangle([x, y, x + square_size, y + square_size], fill=color)

    return image

def create_stripe_background(width=512, height=512, stripe_width=20, color1=(255, 255, 255), color2=(0, 0, 0)):
    """Create a striped background"""
    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)
    
    for x in range(0, width, stripe_width * 2):
        draw.rectangle([x, 0, x + stripe_width, height], fill=color1)
        draw.rectangle([x + stripe_width, 0, x + stripe_width * 2, height], fill=color2)
    
    return image


def create_circular_background(width=512, height=512, center_color=(255, 255, 255), edge_color=(0, 0, 0)):
    """Create a circular gradient background"""
    image = Image.new('RGB', (width, height))
    pixels = image.load()
    
    center_x, center_y = width // 2, height // 2
    max_distance = math.sqrt(center_x**2 + center_y**2)
    
    for x in range(width):
        for y in range(height):
            # Calculate distance from center
            distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            factor = min(distance / max_distance, 1.0)
            
            # Interpolate color
            r = int(center_color[0] * (1 - factor) + edge_color[0] * factor)
            g = int(center_color[1] * (1 - factor) + edge_color[1] * factor)
            b = int(center_color[2] * (1 - factor) + edge_color[2] * factor)
            
            pixels[x, y] = (r, g, b)
    
    return image


def create_spiral_background(width=512, height=512):
    """Create a spiral pattern background"""
    image = Image.new('RGB', (width, height))
    pixels = image.load()
    
    center_x, center_y = width // 2, height // 2
    
    for x in range(width):
        for y in range(height):
            dx, dy = x - center_x, y - center_y
            angle = math.atan2(dy, dx)
            distance = math.sqrt(dx**2 + dy**2)
            
            # Create spiral pattern
            spiral_value = (angle + distance * 0.02) % (2 * math.pi)
            intensity = int(255 * (spiral_value / (2 * math.pi)))
            
            pixels[x, y] = (intensity, intensity // 2, 255 - intensity)
    
    return image


def create_user_background():
    background_img_path = Path(__file__).parent / "assets/tahiti_512.png"
    return Image.open(background_img_path)


@lru_cache(maxsize=10)
def get_background_img(width=512, height=512, bg_type='gradient'):
    """Get background image for collage"""
    if bg_type == 'gradient':
        # Smooth gradient - good for testing watermark visibility
        img = create_gradient_background(width, height, (240, 240, 240), (60, 60, 60))

    elif bg_type == 'texture':
        # Textured background - tests robustness
        base = create_noise_background(width, height, 'gaussian')
        # Overlay with some structure
        overlay = create_checkerboard_background(width, height, 64, (255, 255, 255, 50), (0, 0, 0, 50))
        img = Image.alpha_composite(base.convert('RGBA'), overlay).convert('RGB')

    elif bg_type == 'natural':
        # Simulate natural image characteristics
        # Create multiple frequency components
        low_freq = create_gradient_background(width, height, (200, 150, 100), (100, 150, 200))
        high_freq = create_noise_background(width, height, 'gaussian')

        # Blend them
        blended = Image.blend(low_freq, high_freq, 0.3)
        img = blended

    elif bg_type == 'uniform':
        # Simple uniform background
        img = Image.new('RGB', (width, height), (128, 128, 128))

    elif bg_type == 'user':
        # User provided background
        img = create_user_background()

    else:
        raise ValueError(f"Unknown background type: {bg_type}")

    return default_transform(img)


to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()


def center_crop(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: Tensor image
        scale: target area scale
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s * scale) for s in x.shape[-2:]][::-1]
    return functional.center_crop(x, new_edges_size)


def resize(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: Tensor image
        scale: target area scale
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s * scale) for s in x.shape[-2:]][::-1]
    return functional.resize(x, new_edges_size)


def rotate(x: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate image by angle
    Args:
        x: image (PIl or tensor)
        angle: angle in degrees
    """
    return functional.rotate(x, angle)


def adjust_brightness(x: torch.Tensor, brightness_factor: float) -> torch.Tensor:
    """Adjust brightness of an image
    Args:
        x: Tensor image
        brightness_factor: brightness factor
    """
    return functional.adjust_brightness(x, brightness_factor)


def adjust_contrast(x: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    """Adjust contrast of an image
    Args:
        x: Tensor image
        contrast_factor: contrast factor
    """
    return functional.adjust_contrast(x, contrast_factor)


def adjust_saturation(x: torch.Tensor, saturation_factor: float) -> torch.Tensor:
    """Adjust saturation of an image
    Args:
        x: Tensor image
        saturation_factor: saturation factor
    """
    return functional.adjust_saturation(x, saturation_factor)


def adjust_hue(x: torch.Tensor, hue_factor: float) -> torch.Tensor:
    """Adjust hue of an image
    Args:
        x: Tensor image
        hue_factor: hue factor
    """
    return functional.adjust_hue(x, hue_factor)


def adjust_gamma(x: torch.Tensor, gamma: float, gain=1) -> torch.Tensor:
    """Adjust gamma of an image
    Args:
        x: Tensor image
        gamma: gamma factor
        gain: gain factor
    """
    return functional.adjust_gamma(x, gamma, gain)


def proportion(x: torch.Tensor, y: torch.Tensor, prop=0.5) -> torch.Tensor:
    """take a proportion of the watermarked image and the original image
    Args:
        x: Tensor (watermarked) image
        y: Tensor (original) image
        prop: proportion of the watermarked image to keep
    """

    prop = np.sqrt(prop)
    H, W = x.shape[-2:]
    rect_H = int(prop * H)
    rect_W = int(prop * W)

    # Calculate the starting and ending indices to center the rectangle
    start_H = (H - rect_H) // 2
    end_H = start_H + rect_H
    start_W = (W - rect_W) // 2
    end_W = start_W + rect_W

    # Create the mask
    mask = torch.zeros_like(x)
    mask[:, start_H:end_H, start_W:end_W] = 1
    mask = mask.to(x.device)

    return x * mask + y.to(x.device) * (1 - mask)


def select_lama(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Select the lama part of the image
    Args:
        x: Tensor image
        mask: mask of the lama part
    """

    return (transforms.ToTensor()(mask)).unsqueeze(0).to(
        x.device
    ) * transforms.ToTensor()(x).to(x.device)


def adjust_sharpness(x: torch.Tensor, sharpness_factor: float) -> torch.Tensor:
    """Adjust sharpness of an image
    Args:
        x: Tensor image
        sharpness_factor: sharpness factor
    """
    return functional.adjust_sharpness(x, sharpness_factor)


def get_perspective_params(width, height, distortion_scale):
    half_height = height // 2
    half_width = width // 2
    topleft = [
        int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
        int(
            torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()
        ),
    ]
    topright = [
        int(
            torch.randint(
                width - int(distortion_scale * half_width) - 1, width, size=(1,)
            ).item()
        ),
        int(
            torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()
        ),
    ]
    botright = [
        int(
            torch.randint(
                width - int(distortion_scale * half_width) - 1, width, size=(1,)
            ).item()
        ),
        int(
            torch.randint(
                height - int(distortion_scale * half_height) - 1, height, size=(1,)
            ).item()
        ),
    ]
    botleft = [
        int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
        int(
            torch.randint(
                height - int(distortion_scale * half_height) - 1, height, size=(1,)
            ).item()
        ),
    ]
    startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    endpoints = [topleft, topright, botright, botleft]
    return startpoints, endpoints


def hflip(x):
    """Horizontally flip image
    Args:
        x: Tensor image
    """
    return functional.hflip(x)


def perspective(x: torch.Tensor, distortion_scale: float) -> torch.Tensor:
    """Apply perspective transformation to image
    Args:
        x: Tensor image
        distortion_scale: distortion scale
    """
    width, height = x.shape[-1], x.shape[-2]
    startpoints, endpoints = get_perspective_params(width, height, distortion_scale)
    return F_vision.perspective(x, startpoints, endpoints)


overlay_text_default = [76, 111, 114, 101, 109, 32, 73, 112, 115, 117, 109]


def overlay_text(
    x: torch.Tensor,
    text: List[Union[int, List[int]]] = overlay_text_default,
    font_size: float = 0.15,
    opacity: float = 1.0,
    color: Tuple[int, int, int] = aug_functional.utils.RED_RGB_COLOR,
    position: Tuple[float, float] = (0.0, 0.5),
) -> torch.Tensor:
    """Overlay text on image
    Args:
        x: Tensor image
        text: text to overlay
        font_path: path to font
        font_size: font size
        color: text color
        position: text position
    """
    x_pos, y_pos = position

    if x.ndim == 3:
        xs = [x]
    else:
        xs = x

    imgs_aug = [None] * len(xs)
    for i in range(len(xs)):
        pil_img = to_pil(xs[i])
        imgs_aug[i] = to_tensor(
            aug_functional.overlay_text(
                pil_img,
                text=text,
                font_size=font_size,
                opacity=opacity,
                color=color,
                x_pos=x_pos,
                y_pos=y_pos,
            )
        )

    if len(imgs_aug) == 1:
        return imgs_aug[0]
    else:
        return torch.stack(imgs_aug, dim=0)



def meme(x: torch.Tensor) -> torch.Tensor:
    """Apply meme text to image
    Args:
        x: Tensor image
    """
    pil_img = to_pil(x)
    img_aug = to_tensor(aug_functional.meme_format(pil_img))
    return img_aug


def screen_shot(x: torch.Tensor) -> torch.Tensor:
    """Apply screen shot to image
    Args:
        x: Tensor image
    """
    img_aug = torch.zeros_like(x, device=x.device)
    pil_img = to_pil(x)
    img_aug = to_tensor(aug_functional.overlay_onto_screenshot(pil_img))
    return img_aug


def jpeg_compress(x: torch.Tensor, quality_factor: float) -> torch.Tensor:
    """Apply jpeg compression to image
    Args:
        x: Tensor image
        quality_factor: quality factor
    """
    img_aug = torch.zeros_like(x, device=x.device)
    pil_img = to_pil(x)
    img_aug = to_tensor(
        aug_functional.encoding_quality(pil_img, quality=quality_factor)
    )
    return img_aug


def random_crop(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: Tensor image
        scale: target area scale
    """
    scale = np.sqrt(scale)
    h, w = x.shape[-2:]
    th, tw = int(h * scale), int(w * scale)
    # Guard against degenerate sizes
    max_i = max(0, h - th)
    max_j = max(0, w - tw)
    # Use torch RNG so seeding via torch.manual_seed affects both implementations
    i = int(torch.randint(0, max_i + 1, (1,)).item()) if max_i > 0 else 0
    j = int(torch.randint(0, max_j + 1, (1,)).item()) if max_j > 0 else 0
    return functional.crop(x, i, j, th, tw)


def collage(x: torch.Tensor, scale: float, bg_type: str = "user") -> torch.Tensor:
    """Perform collage of image
    Args:
        x: Tensor image
        scale: target area scale
    """
    h, w = x.shape[-2:]
    desired_area = h * w * scale
    mask = torch.zeros_like(x)
    aspect_ratio = (
        torch.rand(1) * 1.5 + 0.25
    )  # Random aspect ratio between 0.5 and 2.5
    width = min(
        int(torch.sqrt(torch.Tensor([desired_area * aspect_ratio])).item()), w - 1
    )
    # Ensure width is less than w
    if width >= w:
        width = w - 1

    height = int(desired_area / width)

    # Ensure height is less than h
    if height >= h:
        height = h - 1
    left = torch.randint(0, w - width, (1,)).item()
    top = torch.randint(0, h - height, (1,)).item()
    right = left + width
    bottom = top + height
    mask[..., top:bottom, left:right] = 1

    background_img = get_background_img(bg_type=bg_type)
    return x * mask + (1 - mask) * transforms.Resize((h, w))(
        background_img.to(x.device)
    )


def gaussian_noise(x: torch.Tensor, mean: float = 0, std: float = 1) -> torch.Tensor:
    """Add gaussian noise to image
    Args:
        x: Tensor image
        mean: mean of gaussian noise
        std: std of gaussian noise
    """
    x = x + torch.randn_like(x) * std / 255.0 + mean / 255.0
    return x


def gaussian_blur(x: torch.Tensor, kernel_size: int=1) -> torch.Tensor:
    """Add gaussian blur to image
    Args:
        x: Tensor image
        mean: mean of gaussian noise
        std: std of gaussian noise
    """
    x = x.clamp(0, 1)
    x = F_vision.gaussian_blur(x, kernel_size)
    return x


def median_filter(x: torch.Tensor, kernel_size: int=5) -> torch.Tensor:
    """Add median filter blur to image of shape (B, C, H, W) **or** (C, H, W).
    Args:
        x: Tensor image
       kernel_size: kernel size of median filter
    """
    if x.dim() == 3:             # (C, H, W)  →  fake batch dim
        x = x.unsqueeze(0)
        squeeze_back = True
    elif x.dim() == 4:           # (B, C, H, W)
        squeeze_back = False
    else:
        raise ValueError("Expected tensor of shape (C, H, W) or (B, C, H, W).")
    
    kk = kernel_size // 2
    # use constant padding to match valuemetric implementation
    x = F.pad(x, (kk, kk, kk, kk)) # , mode="reflect"

    # Unfold height and width with the full kernel size
    # NOTE: v1 use padded kernel-size kk to unfold the images, while Videoseal 3.0
    # uses kernel_size instead of kk to maintain original dimensions
    # x = x.unfold(2, kk, 1).unfold(3, kk, 1)
    x = x.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    # compute median over the KxK patch in two steps to match valuemetric
    # blocks shape: B x C x H x W x K x K
    # x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
    x = x.median(dim=-1)[0].median(dim=-1)[0]

    if squeeze_back:                # restore original (C, H, W) shape
        x = x.squeeze(0)
    return x


def comb(x: torch.Tensor, compress: int = 80, brightness: float = 1.5, center_crop_scale: float = 0.5) -> torch.Tensor:
    return jpeg_compress(
        adjust_brightness(center_crop(x, center_crop_scale), brightness), compress
    )
