# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyright: reportMissingModuleSource=false

import os

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F_vision
from augly.image import functional as aug_functional
from PIL import Image, ImageDraw, ImageFilter
from skimage.metrics import structural_similarity as compare_ssim
from torch.autograd.variable import Variable
from torchvision import transforms
from torchvision.transforms import functional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
normalize_img = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)  # Normalize (x - mean) / std
unnormalize_img = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)  # Unnormalize (x * std) + mean


dir_path = os.path.dirname(os.path.realpath(__file__))
png_path = os.path.join(dir_path, "../../assets/tahiti_512.png")

background_img = default_transform(Image.open(png_path)).unsqueeze(0).to(device)
to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()


def psnr(x, y, img_space="vqgan"):
    """
    Return PSNR
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    if img_space == "vqgan":
        delta = torch.clamp(unnormalize_vqgan(x), 0, 1) - torch.clamp(
            unnormalize_vqgan(y), 0, 1
        )
    elif img_space == "img":
        delta = torch.clamp(unnormalize_img(x), 0, 1) - torch.clamp(
            unnormalize_img(y), 0, 1
        )
    else:
        delta = x - y
    delta = 255 * delta
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])  # BxCxHxW
    psnr = 20 * np.log10(255) - 10 * torch.log10(
        torch.mean(delta**2, dim=(1, 2, 3))
    )  # B
    return psnr


def ssim(x, y, img_space="vqgan"):
    """
    Return SSIM
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    if img_space == "vqgan":
        x = unnormalize_vqgan(x)
        y = unnormalize_vqgan(y)
    elif img_space == "img":
        x = unnormalize_img(x)
        y = unnormalize_img(y)
    else:
        x = x
        y = y

    x = x.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    y = y.squeeze().cpu().detach().numpy().transpose(1, 2, 0)

    return compare_ssim(x, y, channel_axis=2, data_range=2)


import lpips

# Initialize the LPIPS model once
loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)


def lpips(x, y, img_space="vqgan"):
    """
    Return LPIPS
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    if img_space == "vqgan":
        x = unnormalize_vqgan(x)
        y = unnormalize_vqgan(y)
    elif img_space == "img":
        x = unnormalize_img(x)
        y = unnormalize_img(y)

    # Ensure tensors are on the same device as the model
    x = x.to(device)
    y = y.to(device)

    # Compute LPIPS loss
    loss = loss_fn_vgg(x, y)

    # Return the loss as a scalar value
    return loss.item()


def center_crop(x, scale):
    """Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: Tensor image
        scale: target area scale
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s * scale) for s in x.shape[-2:]][::-1]
    return functional.center_crop(x, new_edges_size)


def resize(x, scale):
    """Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: Tensor image
        scale: target area scale
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s * scale) for s in x.shape[-2:]][::-1]
    return functional.resize(x, new_edges_size)


def rotate(x, angle):
    """Rotate image by angle
    Args:
        x: image (PIl or tensor)
        angle: angle in degrees
    """
    x = unnormalize_img(x)
    x = functional.rotate(x, angle)
    x = normalize_img(x)
    return x


def adjust_brightness(x, brightness_factor):
    """Adjust brightness of an image
    Args:
        x: Tensor image
        brightness_factor: brightness factor
    """
    return normalize_img(
        functional.adjust_brightness(unnormalize_img(x), brightness_factor)
    )


def adjust_contrast(x, contrast_factor):
    """Adjust contrast of an image
    Args:
        x: Tensor image
        contrast_factor: contrast factor
    """
    return normalize_img(
        functional.adjust_contrast(unnormalize_img(x), contrast_factor)
    )


def adjust_saturation(x, saturation_factor):
    """Adjust saturation of an image
    Args:
        x: Tensor image
        saturation_factor: saturation factor
    """
    return normalize_img(
        functional.adjust_saturation(unnormalize_img(x), saturation_factor)
    )


def adjust_hue(x, hue_factor):
    """Adjust hue of an image
    Args:
        x: Tensor image
        hue_factor: hue factor
    """
    return normalize_img(functional.adjust_hue(unnormalize_img(x), hue_factor))


def adjust_gamma(x, gamma, gain=1):
    """Adjust gamma of an image
    Args:
        x: Tensor image
        gamma: gamma factor
        gain: gain factor
    """
    return normalize_img(functional.adjust_gamma(unnormalize_img(x), gamma, gain))


def proportion(x, y, prop=0.5):
    """take a proportion of the watermarked image and the original image
    Args:
        x: Tensor (watermarked) image
        y: Tensor (original) image
        prop: proportion of the watermarked image to keep
    """

    prop = np.sqrt(prop)
    _, _, H, W = x.shape
    rect_H = int(prop * H)
    rect_W = int(prop * W)

    # Calculate the starting and ending indices to center the rectangle
    start_H = (H - rect_H) // 2
    end_H = start_H + rect_H
    start_W = (W - rect_W) // 2
    end_W = start_W + rect_W

    # Create the mask
    mask = torch.zeros_like(x)
    mask[:, :, start_H:end_H, start_W:end_W] = 1
    mask.cuda()

    return x * mask + y.cuda() * (1 - mask)


def select_lama(x, mask):
    """Select the lama part of the image
    Args:
        x: Tensor image
        mask: mask of the lama part
    """

    return (transforms.ToTensor()(mask)).unsqueeze(0).cuda() * transforms.ToTensor()(
        x
    ).cuda()


def inpainting(x, mask):
    """Adjust gamma of an image
    Args:
        x: Tensor (watermarked) image
        y: Tensor (original) image
        prop: proportion of the watermarked image to keep
    """

    prop = np.sqrt(prop)
    _, _, H, W = x.shape
    rect_H = int(prop * H)
    rect_W = int(prop * W)

    # Calculate the starting and ending indices to center the rectangle
    start_H = (H - rect_H) // 2
    end_H = start_H + rect_H
    start_W = (W - rect_W) // 2
    end_W = start_W + rect_W

    # Create the mask
    mask = torch.zeros_like(x)
    mask[:, :, start_H:end_H, start_W:end_W] = 1
    mask.cuda()

    return x * mask + y.cuda() * (1 - mask)


def adjust_sharpness(x, sharpness_factor):
    """Adjust sharpness of an image
    Args:
        x: Tensor image
        sharpness_factor: sharpness factor
    """
    return normalize_img(
        functional.adjust_sharpness(unnormalize_img(x), sharpness_factor)
    )


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


def perspective(x, distortion_scale):
    """Apply perspective transformation to image
    Args:
        x: Tensor image
        distortion_scale: distortion scale
    """
    width, height = x.shape[-1], x.shape[-2]
    startpoints, endpoints = get_perspective_params(width, height, distortion_scale)
    return F_vision.perspective(x, startpoints, endpoints)


def overlay_text(x, text="Lorem Ipsum"):
    """Overlay text on image
    Args:
        x: Tensor image
        text: text to overlay
        font_path: path to font
        font_size: font size
        color: text color
        position: text position
    """
    img_aug = torch.zeros_like(x, device=x.device)
    for ii, img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(aug_functional.overlay_text(pil_img, text=text))
    return normalize_img(img_aug)


def meme(x):
    """Apply meme text to image
    Args:
        x: Tensor image
    """
    for ii, img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug = to_tensor(aug_functional.meme_format(pil_img))
    return normalize_img(img_aug.unsqueeze(0))


def screen_shot(x):
    """Apply screen shot to image
    Args:
        x: Tensor image
    """
    img_aug = torch.zeros_like(x, device=x.device)
    for ii, img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(aug_functional.overlay_onto_screenshot(pil_img))
    return normalize_img(img_aug)


def jpeg_compress(x, quality_factor):
    """Apply jpeg compression to image
    Args:
        x: Tensor image
        quality_factor: quality factor
    """
    img_aug = torch.zeros_like(x, device=x.device)
    for ii, img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(
            aug_functional.encoding_quality(pil_img, quality=quality_factor)
        )
    return normalize_img(img_aug)


def random_crop(x, scale):
    """Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: Tensor image
        scale: target area scale
    """
    scale = np.sqrt(scale)
    h, w = x.shape[-2:]
    th, tw = int(h * scale), int(w * scale)
    i = np.random.randint(0, int(h - th + 1))
    j = np.random.randint(0, int(h - th + 1))
    return functional.crop(x, i, j, th, tw)


def collage(x, scale):
    """Perform collage of image
    Args:
        x: Tensor image
        scale: target area scale
    """
    h, w = x.shape[-2:]
    desired_area = h * w * scale
    mask = torch.zeros_like(x)
    for ii in range(x.shape[0]):
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
        mask[ii, ..., top:bottom, left:right] = 1
    return x * mask + (1 - mask) * transforms.Resize((h, w))(background_img)


def gaussian_noise(x, mean=0, std=1):
    """Add gaussian noise to image
    Args:
        x: Tensor image
        mean: mean of gaussian noise
        std: std of gaussian noise
    """
    x = unnormalize_img(x)
    x = x + torch.randn_like(x) * std / 255.0 + mean / 255.0
    x = normalize_img(x)
    return x


def gaussian_blur(x, kernel_size=1):
    """Add gaussian blur to image
    Args:
        x: Tensor image
        mean: mean of gaussian noise
        std: std of gaussian noise
    """
    x = unnormalize_img(x).clamp(0, 1)
    x = F_vision.gaussian_blur(x, kernel_size)
    x = normalize_img(x)
    return x


def median_filter(x, kernel_size=5):
    """Add median filter blur to image
    Args:
        x: Tensor image
       kernel_size: kernel size of median filter
    """
    kk = kernel_size // 2
    x = unnormalize_img(x)
    x = F.pad(x, (kk, kk, kk, kk), mode="reflect")
    x = x.unfold(2, kk, 1).unfold(3, kk, 1)
    x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
    x = normalize_img(x)
    return x


# python run_tree_ring_watermark.py --run_name no_attack --w_channel 3 --w_pattern ring --start 0 --end 1000 --with_tracking --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k
# python run_tree_ring_watermark.py --run_name rotation --w_channel 3 --w_pattern ring --r_degree 75 --start 0 --end 1000 --with_tracking
# python run_tree_ring_watermark.py --run_name jpeg --w_channel 3 --w_pattern ring --jpeg_ratio 25 --start 0 --end 1000 --with_tracking
# python run_tree_ring_watermark.py --run_name cropping --w_channel 3 --w_pattern ring --crop_scale 0.75 --crop_ratio 0.75 --start 0 --end 1000 --with_tracking
# python run_tree_ring_watermark.py --run_name blurring --w_channel 3 --w_pattern ring --gaussian_blur_r 4 --start 0 --end 1000 --with_tracking
# python run_tree_ring_watermark.py --run_name noise --w_channel 3 --w_pattern ring --gaussian_std 0.1 --start 0 --end 1000 --with_tracking
# python run_tree_ring_watermark.py --run_name color_jitter --w_channel 3 --w_pattern ring --brightness_factor 6 --start 0 --end 1000 --with_tracking


def threering_distortion(
    x,
    r_degree=None,
    jpeg_ratio=None,
    crop_scale=None,
    crop_ratio=None,
    gaussian_blur_r=None,
    gaussian_std=None,
    brightness_factor=None,
):

    img_aug = torch.zeros_like(x, device=x.device)
    for ii, img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))

        run_name = "output"
        if r_degree is not None:
            pil_img = transforms.RandomRotation((r_degree, r_degree))(pil_img)

        if jpeg_ratio is not None:
            pil_img.save(f"tmp_{jpeg_ratio}_{run_name}.jpg", quality=jpeg_ratio)
            pil_img = Image.open(f"tmp_{jpeg_ratio}_{run_name}.jpg")

        if crop_scale is not None and crop_ratio is not None:
            pil_img = transforms.RandomResizedCrop(
                pil_img.size,
                scale=(crop_scale, crop_scale),
                ratio=(crop_ratio, crop_ratio),
            )(pil_img)

        if gaussian_blur_r is not None:
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=gaussian_blur_r))

        if gaussian_std is not None:
            img_shape = np.array(pil_img).shape
            g_noise = np.random.normal(0, gaussian_std, img_shape) * 255
            g_noise = g_noise.astype(np.uint8)
            pil_img = Image.fromarray(np.clip(np.array(pil_img) + g_noise, 0, 255))

        if brightness_factor is not None:
            pil_img = transforms.ColorJitter(brightness=brightness_factor)(pil_img)

        img_aug[ii] = to_tensor(pil_img)
    return normalize_img(img_aug)
