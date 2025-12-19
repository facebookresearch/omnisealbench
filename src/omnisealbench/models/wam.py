# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf
from torchvision import transforms  # type: ignore[import-untyped]

from omnisealbench.models.wam_src.augmentation.augmenter import Augmenter
from omnisealbench.models.wam_src.data.metrics import msg_predict_inference
from omnisealbench.models.wam_src.models import (Embedder, Extractor, Wam,
                                                 build_embedder,
                                                 build_extractor)
from omnisealbench.models.wam_src.modules import JND as WAM_JND
from omnisealbench.utils.common import set_path, set_seed


class WamWatermark:
    """
    Implementation of the WamMark model
    ("WAM: Watermark Anything with Localized Messages",
    https://arxiv.org/abs/2411.07231)
    This class provides methods to both generate watermarks and detect them.
    """

    encoder: Embedder
    decoder: Extractor

    def __init__(
        self,
        encoder: Embedder,
        decoder: Extractor,
        normalize: transforms.Normalize,
        unnormalize: transforms.Normalize,
        scaling_i: float,
        scaling_w: float,
        attenuation: WAM_JND | None = None,
        target_image_size: tuple[int, int] = (256, 256),
        detection_bits: int = 16,
        nbits: int = 32,
        interpolation: str = "bilinear",
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.attenuation = attenuation

        self.normalize = normalize
        self.unnormalize = unnormalize
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w

        self.detection_bits = detection_bits
        self.nbits = nbits

        self.interpolation = (
            transforms.InterpolationMode.BILINEAR
            if interpolation == "bilinear"
            else transforms.InterpolationMode.NEAREST
        )

        self.resize = transforms.Resize(target_image_size, interpolation=self.interpolation)
        self.device = next(encoder.parameters()).device


    def generate_watermark(
        self, contents: List[torch.Tensor], message: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Generates a watermark for the given image contents.
        Args:
            contents (torch.Tensor): The input image data to be watermarked.
            message (torch.Tensor): Tensor containing the message to embed in the watermark.
        Returns:
            torch.Tensor: The watermarked image tensor.
        """
        if message.ndim == 1:
            message = message.repeat(len(contents), 1)  # Add batch dimension if missing

        imgs_size: List[Tuple[int, int]] = []
        imgs_norm: List[torch.Tensor] = []
        imgs_resize: List[torch.Tensor] = []

        imgs_w = []  # List to collect watermarked images

        for img in contents:
            img = self.normalize(img).unsqueeze(0)  # 1 x 3 x H x W
            imgs_norm.append(img)  # Collect normalised images
            imgs_size.append(img.shape[-2:])  # (H, W)
            img_resize = self.resize(img)  # 1 x 3 x 256 x 256
            imgs_resize.append(img_resize)

        imgs_resize = torch.cat(imgs_resize, dim=0)  # N x 3 x 256 x 256

        # Compute delta on the resized version
        deltas_w = self.encoder(imgs_resize, message)  # N x 3 x 256 x 256
        for i in range(len(contents)):
            img = imgs_norm[i]  # 1 3 h w
            img_w = self.scaling_i * img + self.scaling_w * transforms.Resize(
                imgs_size[i], interpolation=self.interpolation
            )(
                deltas_w[i: i + 1]
            )  # 1 3 h w

            if self.attenuation is not None:                
                img_w = self.attenuation(img, img_w)  # 1 3 h w
            clip_img = torch.clamp(self.unnormalize(img_w), 0, 1)
            clip_img = torch.round(255 * clip_img) / 255
            wm_img = clip_img.squeeze(0).detach()  # 3 h w
            imgs_w.append(wm_img)

        return imgs_w  # List of watermarked images with original sizes

    def detect_watermark(
        self,
        contents: List[torch.Tensor],
        detection_threshold: float = 0.07,
        message_threshold: float = 0.5,
        detection_bits: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        imgs_resize: List[torch.Tensor] = []
        for img in contents:
            img = self.normalize(img).unsqueeze(0)  # 1 x 3 x H x W
            img_resize = self.resize(img)  # Resize to target size 1 x 3 x target_h x target_w
            imgs_resize.append(img_resize)

        imgs_w = torch.cat(imgs_resize, dim=0)  # N x 3 x target_h x target_w
        preds = self.decoder(imgs_w)  # N x (1 + k) x target_h x target_w
        bit_preds = msg_predict_inference(
            preds[:, 1:, :, :],
            torch.sigmoid(preds[:, 0:1, :, :]),
            method="semihard",
            msg_threshold=message_threshold,
        )  # b k
        detection = (
            (torch.sigmoid(preds[:, 0:1, :, :])).float().mean(dim=(1, 2, 3))
        )  # b 1 h w -> b

        #hard_detection = (
        #    (torch.sigmoid(preds[:, 0:1, :, :]) > 0.5).float().mean(dim=(1, 2, 3))
        #)  # b 1 h w -> b

        decoded = bit_preds.squeeze()  # b k -> b k or k if b = 1
        detection_conf_score = detection.detach()

        return {
            "det_score": detection_conf_score,  # b x 1
            "det": (detection_conf_score > detection_threshold),  # b x 1
            "message": decoded,  # b k
        }


def build_model(
    model_card_or_path: str,
    do_attenuation: bool,
    embedder_model: str = "vae_small",
    extractor_model: str = "sam_base",
    interpolation: str = "bilinear",
    scaling_w: float = 2.0,
    color: str = "blue",
    detection_bits: int = 16,
    nbits: int = 32,
    device: str = "cpu",
):
    """
    Build an WAM model.
    ("WAM: Watermark Anything with Localized Messages",
    https://arxiv.org/abs/2411.07231)
    Args:
        model_card_or_path: The path to the model card or the model itself.
        do_attenuation: Whether to apply attenuation.
        embedder_model: The model to use for embedding.
        extractor_model: The model to use for extraction.
        interpolation: The interpolation method to use.
        scaling_w: The scaling factor for the watermark.
        color: Which configuration of JND map to use. Default is "blue".
        detection_bits: Number of bits used for detection.
        nbits: Number of bits for the watermark message.
        device: The device to run the model on.
    Returns:
        An instance of WamWatermark.
    """

    with set_path(Path(__file__).parent):  # omnisealbench/models

        # If there is a config directory in the checkpoint directory, use it.
        # Otherwise, use the default config files.
        user_config_path = Path(model_card_or_path).parent / "configs"
        default_config_path = Path(__file__).parent / "wam_src" / "configs"

        if (user_config_path / "embedder.yaml").exists():
            embedder_cfg_yaml = os.path.join(user_config_path, "embedder.yaml")
        else:
            embedder_cfg_yaml = os.path.join(default_config_path, "embedder.yaml")

        if (user_config_path / "augs.yaml").exists():
            augmenter_cfg_yaml = os.path.join(user_config_path, "augs.yaml")
        else:
            augmenter_cfg_yaml = os.path.join(default_config_path, "augs.yaml")

        if (user_config_path / "extractor.yaml").exists():
            extractor_cfg_yaml = os.path.join(user_config_path, "extractor.yaml")
        else:
            extractor_cfg_yaml = os.path.join(default_config_path, "extractor.yaml")

        embedder_cfg = OmegaConf.load(embedder_cfg_yaml)
        embedder_params = embedder_cfg[embedder_model]
        embedder = build_embedder(embedder_model, embedder_params, nbits)

        augmenter_cfg = OmegaConf.load(augmenter_cfg_yaml)
        augmenter = Augmenter(**augmenter_cfg)

        extractor_cfg = OmegaConf.load(extractor_cfg_yaml)

        extractor_params = extractor_cfg[extractor_model]
        extractor = build_extractor(extractor_model, extractor_params, 256, nbits)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        unnormalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )

        if do_attenuation:
            if (user_config_path / "attenuation.yaml").exists():
                attenuation_cfg_yaml = os.path.join(user_config_path, "attenuation.yaml")
            else:
                attenuation_cfg_yaml = os.path.join(default_config_path, "attenuation.yaml")
            attenuation_cfg = OmegaConf.load(attenuation_cfg_yaml)
            attenuation = WAM_JND(
                **attenuation_cfg[f"jnd_1_3_{color}"],
                preprocess=unnormalize,
                postprocess=normalize,
            )
            attenuation.to(device)
        else:
            attenuation = None

        wam = Wam(
            embedder,
            extractor,
            augmenter,
            attenuation,
            scaling_w,
            scaling_i=1,
            roll_probability=0,
        )
        wam.to(device)
        wam.eval()

        checkpoint = torch.load(model_card_or_path, map_location=device)
        wam.load_state_dict(checkpoint)

        encoder = wam.embedder
        decoder = wam.detector

    return WamWatermark(
        encoder,
        decoder,
        normalize=normalize,
        unnormalize=unnormalize,
        attenuation=attenuation,
        scaling_i=wam.scaling_i,
        scaling_w=wam.scaling_w,
        nbits=nbits,
        detection_bits=detection_bits,
        interpolation=interpolation,
    )
