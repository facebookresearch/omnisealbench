# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from torchvision import transforms  # type: ignore[import-untyped]

from omnisealbench.interface import Device
from omnisealbench.models.cin_src import CINModel
from omnisealbench.utils.detection import get_detection_and_decoded_keys


class CINWm:
    """
    Wrapper for the CIN watermark generator model
    ("Towards Blind Watermarking: Combining Invertible and Non-invertible Mechanisms").
    https://arxiv.org/abs/2212.12678
    """

    def __init__(self, model: CINModel, alpha: float, nbits: int = 30, detection_bits: int = 16):
        self.model = model
        self.alpha = alpha
        self.nbits = nbits
        self.detection_bits = detection_bits

        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.resize = transforms.Resize((128, 128))
        self.postprocess = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])

        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def generate_watermark(
        self, contents: List[torch.Tensor], message: torch.Tensor
    ) -> List[torch.Tensor]:
        if message.ndim < 2:  # Add batch dimension if missing
            message = message.repeat(len(contents), 1)
        message = message[:, : self.nbits].to(torch.float32)

        sizes: List[Tuple[int, int]] = []
        imgs = []
        normalized_imgs = []

        for img in contents:
            img = self.preprocess(img)
            imgs.append(img)  # Collect original images

            _, H, W = img.shape
            sizes.append((H, W))
            img_resize = self.resize(img)  # 3 x 128 x 128
            normalized_imgs.append(img_resize)

        normalized_contents = torch.stack(normalized_imgs, dim=0)  # N x 3 x 128 x 128
        with torch.no_grad():
            delta_w = self.model.encoder(normalized_contents, message) - normalized_contents

        del normalized_contents  # Free memory

        imgs_w: List[torch.Tensor] = []
        for i, ((H, W), img) in enumerate(zip(sizes, imgs)):
            delta_w_resize = transforms.Resize((H, W))(delta_w[i])

            img_w = self.postprocess(img + self.alpha * delta_w_resize)
            imgs_w.append(img_w.clamp(0, 1))  # Ensure pixel values are in [0, 1]

        return imgs_w  # List of watermarked images with original sizes    

    @torch.inference_mode()
    def detect_watermark(
        self,
        contents: List[torch.Tensor],
        detection_threshold: float = 0.95,  # not used
        message_threshold: float = 0.5,
        detection_bits: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        imgs_resize: List[torch.Tensor] = []
        for img in contents:
            img_resize = self.resize(self.preprocess(img))  # 3 x 128 x 128
            imgs_resize.append(img_resize)

        imgs_w = torch.stack(imgs_resize, dim=0)  # N x 3 x 128 x 128
        extracted = self.model.test_decoder(imgs_w)

        if detection_bits is None:
            detection_bits = self.detection_bits

        return get_detection_and_decoded_keys(
            extracted,
            detection_bits=detection_bits,
            message_threshold=message_threshold,
        )


def build_model(
    model_card_or_path: str,
    alpha: float = 1.0,
    nbits: int = 30,
    detection_bits: int = 16,
    device: Device = "cpu",
) -> CINWm:
    """
    Builds the CIN watermark model from a given model card or path.
    ("Towards Blind Watermarking: Combining Invertible and Non-invertible Mechanisms").
    https://arxiv.org/abs/2212.12678

    Args:
        model_card_or_path (str): Path to the model card or pre-trained model.
        alpha (float): Scaling factor for watermark generation.
        nbits (int): Number of bits in the watermark message. Default is 16.
        detection_bits (int): Number of bits used for detection. Default is 16.
        device (str): Device to load the model on (e.g., 'cpu', 'cuda').

    Returns:
        CINWm: An instance of the CIN watermark model.
    """

    state_dict = torch.load(model_card_or_path, map_location=device)['cinNet']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    yaml_path = Path(model_card_or_path).parent/'opt.yml'
    opt = yaml.load(open(yaml_path, 'r'), Loader=yaml.FullLoader)

    # Create device-agnostic model, then move to device
    model = CINModel(opt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    return CINWm(model=model, alpha=alpha, nbits=nbits, detection_bits=detection_bits)
