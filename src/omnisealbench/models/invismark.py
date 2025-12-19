# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Part of this source code is derived from the original work at:
# https://github.com/microsoft/InvisMark/blob/main/train.py


from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torchvision import transforms  # type: ignore[import-untyped]

from omnisealbench.models.invismark_src import \
    configs as configs_  # noqa: F401
from omnisealbench.models.invismark_src.modules import Encoder, Extractor
from omnisealbench.utils.common import set_path, set_seed
from omnisealbench.utils.detection import get_detection_and_decoded_keys


class InvisWatermark:
    """
    Implementation of the InvisMark model
    ("InvisMark: Invisible and Robust Watermarking for AI-generated Image",
    https://arxiv.org/abs/2411.07795)
    This class provides methods to both generate watermarks and detect them.
    """

    encoder: Encoder
    decoder: Extractor

    def __init__(
        self,
        encoder: Encoder,
        decoder: Extractor,
        target_image_size: tuple[int, int] = (256, 256),
        detection_bits: int = 16,
        nbits: int = 100,  # Number of bits in the watermark message
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.resize = transforms.Resize(target_image_size)
        self.detection_bits = detection_bits
        self.nbits = nbits
        self.device = next(encoder.parameters()).device

    @torch.inference_mode()
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
        for img in contents:
            img = self.normalize(img)  # 3 x H x W
            imgs_norm.append(img)  # Collect original images
            
            imgs_size.append(img.shape[-2:])  # (H, W)
            
            img_resize = self.resize(img)
            imgs_resize.append(img_resize)  # 3 x 256 x 256
        
        imgs_resize = torch.stack(imgs_resize, dim=0)  # N x 3 x 256 x 256
        encoded = self.encoder(imgs_resize, message)  # N x 3 x 256 x 256
        delta = encoded - imgs_resize  # N x 3 x 256 x 256
        
        imgs_w: List[torch.Tensor] = []
        for i, (img, _size) in enumerate(zip(imgs_norm, imgs_size)):
            
            orig_diff = transforms.Resize(_size)(delta[i])  # Resize delta to original image size
            output = torch.clamp(img + orig_diff, min=-1.0, max=1.0)
            img_w = output * 0.5 + 0.5  # pixel values now roughly in [0,1]
            imgs_w.append(img_w)
        
        return imgs_w  # List of watermarked images with original sizes

    @torch.inference_mode()
    def detect_watermark(
        self,
        contents: List[torch.Tensor],
        detection_threshold: float = 0.95,  # not used
        message_threshold: float = 0.0,
        detection_bits: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        set_seed(42)
        imgs_resize: List[torch.Tensor] = []
        for img in contents:
            img = self.normalize(img)
            img_resize = self.resize(img)  # Resize to target size
            imgs_resize.append(img_resize)

        imgs_w = torch.stack(imgs_resize, dim=0)  # N x 3 x target_h x target_w
        extracted = self.decoder(imgs_w)
        
        return get_detection_and_decoded_keys(  # type: ignore
            extracted,
            detection_bits=detection_bits if detection_bits is not None else self.detection_bits,
            message_threshold=message_threshold,
        )


def build_model(
    model_card_or_path: str,
    detection_bits: int = 16,
    nbits: int = 100,  # Number of bits in the watermark message
    device: str = "cpu",
):
    """
    Build an InvisMark model
    ("InvisMark: Invisible and Robust Watermarking for AI-generated Image",
    https://arxiv.org/abs/2411.07795).
    Args:
        model_card_or_path: The path to the model card or the model itself.
        detection_bits: The number of bits used for watermark detection.
        nbits: The number of bits in the watermark message.
        device: The device to run the model on.
    Returns:
        An instance of InvisWatermark.
    """

    with set_path(Path(configs_.__file__).parent):
        state_dict = torch.load(
            model_card_or_path, weights_only=False, map_location=torch.device(device)
        )
        config = state_dict["config"]

        encoder = Encoder(config).to(device)
        encoder.load_state_dict(state_dict["encoder_state_dict"])
        encoder.eval()

        decoder = Extractor(config).to(device)
        decoder.load_state_dict(state_dict["decoder_state_dict"])
        decoder.eval()
        target_image_size = getattr(config, "image_shape", (256, 256))

    return InvisWatermark(
        encoder,
        decoder,
        target_image_size=target_image_size,
        detection_bits=detection_bits,
        nbits=nbits,
    )
