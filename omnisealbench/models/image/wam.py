# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List

import numpy as np
import omegaconf
import torch
from PIL.Image import Image
from torchvision import transforms

from .base import Watermark
from .wam_src.augmentation.augmenter import Augmenter
from .wam_src.data.metrics import msg_predict_inference
from .wam_src.models import Wam, build_embedder, build_extractor
from .wam_src.modules import JND as WAM_JND
from .wam_src.utils import optim as uoptim


class WAM(Watermark):
    def __init__(
        self,
        nbits,
        device,
        checkpoint_path,
        embedder_cfg_yaml,
        augmenter_cfg_yaml,
        extractor_cfg_yaml,
        do_attenuation=False,
        attenuation_cfg_yaml=None,
        scaling_w=2.0,
        color="blue",
        interpolation="bilinear",
        extractor_model="sam_base",  # sam_small
    ):
        super().__init__(device)

        embedder_cfg = omegaconf.OmegaConf.load(embedder_cfg_yaml)
        embedder_model = "vae_small"
        embedder_params = embedder_cfg[embedder_model]
        embedder = build_embedder(embedder_model, embedder_params, nbits)

        augmenter_cfg = omegaconf.OmegaConf.load(augmenter_cfg_yaml)
        augmenter = Augmenter(**augmenter_cfg)

        extractor_cfg = omegaconf.OmegaConf.load(extractor_cfg_yaml)

        extractor_params = extractor_cfg[extractor_model]
        extractor = build_extractor(extractor_model, extractor_params, 256, nbits)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.unnormalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )

        if do_attenuation:
            attenuation_cfg = omegaconf.OmegaConf.load(attenuation_cfg_yaml)
            attenuation = WAM_JND(
                **attenuation_cfg[f"jnd_1_3_{color}"],
                preprocess=self.unnormalize,
                postprocess=self.normalize,
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
            1,
            roll_probability=0,
        )
        wam.to(device)
        wam.eval()
        uoptim.restart_from_checkpoint(
            checkpoint_path,
            model=wam,
        )
        self.encoder = wam.embedder
        self.decoder = wam.detector
        self.attenuation = attenuation
        self.wam = wam

        self.interpolation = (
            transforms.InterpolationMode.BILINEAR
            if interpolation == "bilinear"
            else transforms.InterpolationMode.NEAREST
        )
        print("using interpolation", self.interpolation)

    @torch.no_grad()
    def encode(self, img: Image, msg: np.ndarray) -> Image:
        msg = torch.tensor(msg, dtype=torch.float, device=self.device).unsqueeze(
            0
        )  # 1 k
        img = self.normalize(
            transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        )  # 1 3 h w
        # Compute delta on the resized version
        deltas_w = self.encoder(transforms.Resize((256, 256))(img), msg)  # 1 3 h w
        img_w = self.wam.scaling_i * img + self.wam.scaling_w * transforms.Resize(
            img.shape[-2:], interpolation=self.interpolation
        )(
            deltas_w
        )  # 1 3 h w

        if self.attenuation is not None:
            img_w = self.attenuation(img, img_w)  # 1 3 h w
        clip_img = torch.clamp(self.unnormalize(img_w), 0, 1)
        clip_img = torch.round(255 * clip_img) / 255
        return transforms.ToPILImage()(clip_img.squeeze(0).cpu())  # 3 h w

    def decode_batch(self, imgs: List[Image]) -> List[np.ndarray]:
        image_tensors = []
        for img in imgs:
            img = self.normalize(
                transforms.ToTensor()(img).unsqueeze(0).to(self.device)
            )  # 1 3 h w

            img = transforms.Resize((256, 256), self.interpolation)(img)
            image_tensors.append(img)

        image_tensors = torch.cat(image_tensors, dim=0)

        preds = self.decoder(image_tensors)

        bit_preds = msg_predict_inference(
            preds[:, 1:, :, :], torch.sigmoid(preds[:, 0:1, :, :]), method="semihard"
        )  # b k
        detection = (torch.sigmoid(preds[:, 0:1, :, :])).float().mean(dim=(1, 2, 3))
        hard_detection = (
            (torch.sigmoid(preds[:, 0:1, :, :]) > 0.5).float().mean(dim=(1, 2, 3))
        )
        return bit_preds.squeeze().cpu().numpy(), (detection, hard_detection)

    @torch.no_grad()
    def decode(self, img: Image) -> np.ndarray:
        img = self.normalize(
            transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        )  # 1 3 h w
        preds = self.decoder(transforms.Resize((256, 256), self.interpolation)(img))
        bit_preds = msg_predict_inference(
            preds[:, 1:, :, :], torch.sigmoid(preds[:, 0:1, :, :]), method="semihard"
        )  # b k
        detection = (torch.sigmoid(preds[:, 0:1, :, :])).float().mean().item()
        hard_detection = (
            (torch.sigmoid(preds[:, 0:1, :, :]) > 0.5).float().mean().item()
        )
        return (bit_preds).squeeze().cpu().numpy(), (detection, hard_detection)

    @torch.no_grad()
    def detect_map(self, img: Image) -> np.ndarray:
        img = self.normalize(
            transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        )  # 1 3 h w
        preds = self.decoder(transforms.Resize((256, 256), self.interpolation)(img))
        detection_map = (torch.sigmoid(preds[:, 0:1, :, :])).float()
        return detection_map

    def get_detection_stats(
        self, detected_batch: torch.Tensor, detection_keys: torch.Tensor
    ) -> List[Dict[str, Any]]:
        all_stats = []

        dsb, dhb = detected_batch

        for i in range(len(dsb)):
            detected_soft = dsb[i]
            detected_hard = dhb[i]

            detect = detected_soft > 0.05

            stats = {
                "det_score": detected_soft.item(),
                "det_score_hard": detected_hard.item(),
                "det": detect.item(),
            }

            all_stats.append(stats)

        return all_stats
