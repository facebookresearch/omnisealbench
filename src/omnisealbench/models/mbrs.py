# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
from torchvision import transforms  # type: ignore[import-untyped]

from omnisealbench.interface import Device
from omnisealbench.models.mbrs_src import Network
from omnisealbench.utils.detection import get_detection_and_decoded_keys


class MBRSWM:
    """
    Wrapper for the MBRS watermark generator model
    ("Mbrs: Enhancing robustness of dnn-based watermarking by
    mini-batch of real and simulated jpeg compression",
    https://arxiv.org/abs/2108.08211)
    """

    model: Network

    def __init__(self, model: Network, nbits: int = 256, detection_bits: int = 16, alpha: float = 1.0):
        self.model = model
        self.alpha = alpha
        self.nbits = nbits
        self.detection_bits = detection_bits

        # Normalization used in the MBRS model
        self.preprocess = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
        self.postprocess = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
        self.resize = transforms.Resize((256, 256))

        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def generate_watermark(
        self, contents: List[torch.Tensor], message: torch.Tensor
    ) -> torch.Tensor:
        assert message.ndim == 1, "Message should be a 1D tensor."

        imgs_norm: List[torch.Tensor] = []
        imgs_size: List[Tuple[int, int]] = []
        imgs_resize: List[torch.Tensor] = []

        for img in contents:
            img_norm = self.preprocess(img)  # C H W 
            imgs_norm.append(img_norm)

            imgs_size.append(tuple(img.shape[-2:]))  # H W
            img_resize = self.resize(img_norm)  # C 256 256

            imgs_resize.append(img_resize)

        img_in = torch.stack(imgs_resize, dim=0)  # N C
        msg_in = message.repeat(img_in.shape[0], 1)  # N D

        with torch.no_grad():
            img_wm = self.model.encoder(img_in, msg_in)
        delta = img_wm - img_in

        imgs_w: List[torch.Tensor] = []
        for i, (size, img) in enumerate(zip(imgs_size, imgs_norm)):
            # Resize delta_w to the original image size
            delta_w_resize = transforms.Resize(size)(delta[i])

            img_w = self.postprocess(img + self.alpha * delta_w_resize)
            imgs_w.append(img_w.clamp(0, 1))  # Collect watermarked images

        return imgs_w  # type: ignore

    @torch.inference_mode()
    def detect_watermark(
        self,
        contents: List[torch.Tensor],
        detection_threshold: float = 0.95,  # not used
        message_threshold: float = 0.5,
        detection_bits: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        imgs_w = [self.resize(self.preprocess(img_w)) for img_w in contents]
        input_ = torch.stack(imgs_w, dim=0)  # N C H W

        extracted = self.model.decoder(input_) # N 256

        return get_detection_and_decoded_keys(
            extracted, 
            detection_bits=detection_bits if detection_bits is not None else self.detection_bits, 
            message_threshold=message_threshold
        )


def build_model(
    model_card_or_path: str,
    nbits: int = 256,
    detection_bits: int = 16,
    device: Device = "cpu",
) -> MBRSWM:
    """
    Builds the MBRS watermark model from a given model card or path.
    ("Mbrs: Enhancing robustness of dnn-based watermarking by
    mini-batch of real and simulated jpeg compression",
    https://arxiv.org/abs/2108.08211)

    Args:
        model_card_or_path (str): Path to the model card or pre-trained model.
        nbits (int): Number of bits in the watermark message. Default is 16.
        detection_bits (int): Number of bits used for detection. Default is 16.
        device (str): Device to load the model on (e.g., 'cpu', 'cuda').

    Returns:        
        MBRSWM: An instance of the MBRS watermark detector.
    """

    # Create device-agnostic model, then move to device
    model = Network(256, 256, nbits, with_diffusion=False)
    model.load_model_ed(model_card_or_path)
    model.encoder_decoder.to(device).eval()

    return MBRSWM(model=model.encoder_decoder, nbits=nbits, detection_bits=detection_bits)
