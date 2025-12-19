# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# This is a quick adapted version of VideoSeal to detect images in batches
# TODO: We should merge this to VideoSeal

from typing import List, Optional, Dict, Union

import torch
import torch.nn as nn
import torchvision.transforms as T

from videoseal.evals.full import setup_model_from_checkpoint

from omnisealbench.utils.common import get_device
from omnisealbench.utils.detection import get_detection_and_decoded_keys



class LatentWatermark:

    model: nn.Module

    def __init__(self, model: torch.nn.Module, img_size: float = 256, nbits: int = 64, detection_bits: int = 16):
        self.model = model

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
        ])        

        # Each model should have an attribute 'nbits'. If the model does not have this attribute,
        # we must set the value `message_size` in the task. If Omniseal could not find information 
        # from either model or the task, it will raise the ValueError
        self.nbits = nbits
        self.detection_bits = detection_bits

    @torch.inference_mode()
    def detect_watermark(
        self,
        contents: Union[torch.Tensor, List[torch.Tensor]],
        detection_threshold: float = 0.0,
        message_threshold: float = 0.0,
        detection_bits: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        # A detect_watermark() must have a specific signature:
        # Args:
        #  - contents: a torch.Tensor (with batch dimension at dim=0) or a list of torch.Tensor (each without batch dimension), shape [B, C, H, W] or List of [C, H, W]
        #  - message_threshold: threshold used to convert the watermark output (probability
        #    of each bits being 0 or 1) into the binary n-bit message.
        #  - detection_threshold: threshold to convert the softmax output to binary indicating
        #    the probability of the content being watermarked
        # Returns:
        #  - a dictionary of with some keys such as:
        #    - "prediction": The prediction probability of the content being watermarked or not. The dimension should be 1 for batch size of `B`.
        #    - "message": The secret message of dimension `B x nbits`
        #    - "detection_bits": The list of bits reserved to calculating the detection accuracy.
        #   
        #    One of "prediction" and "detection_bits" must be provided. "message" is optional
        #    If "message" is returned, Omniseal Bench will compute message accuracy scores: "bit_acc", "word_acc", "p_value", "capacity", and "log10_p_value"
        #    Otherwise, these metrics will be skipped
        device = get_device(self)
        if isinstance(contents, List):
            image_tensors = []
            for img in contents:
                img_tensor = self.transform(img).unsqueeze(0).to(device)
                image_tensors.append(img_tensor)
            image_tensors = torch.cat(image_tensors, dim=0)
        else:
            image_tensors = self.transform(contents).to(device)
        
        extracted_bits = self.model.detector(image_tensors)  # type: ignore
        extracted_bits = extracted_bits[:, 1:]

        return get_detection_and_decoded_keys(
            extracted_bits,
            detection_bits=detection_bits if detection_bits is not None else self.detection_bits,
            message_threshold=message_threshold,
        )


def build_latent_watermark_model(ckpt_path: str, img_size: int = 512, nbits: int = 64, detection_bits: int = 0, device: str = "cpu") -> LatentWatermark:
    watermarker = setup_model_from_checkpoint(ckpt_path)
    watermarker = watermarker.eval()
    watermarker = watermarker.to(device)

    return LatentWatermark(model=watermarker, img_size=img_size, nbits=nbits, detection_bits=detection_bits)
