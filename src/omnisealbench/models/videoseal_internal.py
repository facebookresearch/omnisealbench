# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F

try:
    # try loading videoseal-internal first
    from videoseal.models import Videoseal

except ImportError as ex:
    if "videoseal" not in str(ex):
        raise ex
    # Use fairinternal videoseal
    from videoseal.models import VideoWam as Videoseal

from omnisealbench.utils.detection import get_detection_and_decoded_keys

from .videoseal_src import setup_model_from_checkpoint


class VideosealWM:
    """
    Implementation of the Videoseal watermark model.
    ("Video Seal: Open and Efficient Video Watermarking",
    https://arxiv.org/abs/2412.09492)
    """

    model: torch.nn.Module

    def __init__(
        self,
        model: torch.nn.Module,
        method: str,
        video_aggregation: str,
        lowres_attenuation: bool,
        interpolation: dict,
        nbits: int = 96,
        detection_bits: int = 0,
        video_input: bool = True,
    ):
        """
        Args:
            model (torch.nn.Module): The Videoseal model to use.
            video_aggregation (str): Method for aggregating video frames during detection.
            lowres_attenuation (bool): Whether to apply low-resolution attenuation.
            interpolation (dict): Interpolation parameters for resizing frames.
            nbits (int): Number of bits for the watermark message.
            detection_bits (int): Number of bits used for detection.
            video_input: Videoseal is developed for both image and video watermarking. Turn off
                `video_input` flag will fall back the watermarking to WAM
        """

        self.model = model
        self.method = method
        self.video_aggregation = video_aggregation
        self.lowres_attenuation = lowres_attenuation
        self.interpolation = interpolation

        self.nbits = nbits
        self.detection_bits = detection_bits
        self.device = next(model.parameters()).device
        self.video_input = video_input

    def to(self, device: Union[str, torch.device]) -> "VideosealWM":
        """
        Move the model to the specified device.
        """
        self.model.to(device)
        self.device = device
        return self

    def _encode(self, imgs: torch.Tensor, msgs: torch.Tensor) -> torch.Tensor:
        """watermark a single video (sequence of frames)"""

        # add a batch dimension to the imgs if missing
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        
        # VideoWam handles messages dimension internall for video inputs, but let it
        # to users to handle the messages dimension themselves
        if not self.video_input:
            msgs = msgs.repeat(imgs.shape[0], 1)  # b k

        # forward embedder, at any resolution
        # does cpu -> gpu -> cpu when gpu is available
        # TODO: check if we need/it helps to have imgs on device already
        outputs = self.model.embed(
            imgs, 
            msgs=msgs, 
            is_video=self.video_input, 
            interpolation=self.interpolation, 
            lowres_attenuation=self.lowres_attenuation
        )

        msgs: torch.Tensor = outputs["msgs"]  # f k
        imgs_w: torch.Tensor = outputs["imgs_w"]  # f c h w

        assert msgs.shape[0] == imgs_w.shape[0]
        return imgs_w  # f c h w

    def _decode(self, imgs: torch.Tensor) -> torch.Tensor:
        """decode a single video (sequence of frames)"""

        # does imgs.device -> model.device -> imgs.device

        # add a batch dimension to the imgs if missing
        if imgs.ndim == 3:
            assert not self.video_input, "Video input expected to have 4 dimensions"
            imgs = imgs.unsqueeze(0)
            
            if self.method.lower().startswith("cin"):
                img_sizes = (128, 128)
            else:
                img_sizes = (self.model.img_size, self.model.img_size)
            
            if imgs.shape[-2:] != img_sizes:
                imgs = F.interpolate(imgs, size=img_sizes, **self.interpolation)
                imgs = torch.nn.functional.interpolate(
                    imgs,
                    size=img_sizes,
                    **self.interpolation,
                )

            outputs = self.model.detect(imgs, is_video=False)
            preds = outputs["preds"]
            preds = preds[:, 1:]  # Remove first bit for all WAM-compatible models
        else:
            preds = self.model.detect_and_aggregate(
                imgs, self.video_aggregation, self.interpolation
            )  # 1 nbits
        return preds
    
    @torch.inference_mode()
    def generate_watermark(
        self, 
        contents: Union[torch.Tensor, List[torch.Tensor]], 
        message: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        
        if message.ndim == 1:
            message = message.unsqueeze(0)
        
        if isinstance(contents, torch.Tensor):
            return self._encode(contents, message)
        
        results = [self._encode(c, message) for c in contents]
        if not self.video_input:
            results = [r.squeeze(0) for r in results]  # remove batch dim for images
        return results

    @torch.inference_mode()
    def detect_watermark(
        self,
        contents: Union[torch.Tensor, List[torch.Tensor]],
        detection_threshold: float = 0.95,  # not used
        message_threshold: float = 0.0,
        detection_bits: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = torch.cat([self._decode(c) for c in contents])  # b x nbits
        return get_detection_and_decoded_keys(
            outputs,
            detection_bits=detection_bits if detection_bits is not None else self.detection_bits,
            message_threshold=message_threshold,
        )


def build_model(
    model_card_or_path: str,
    nbits: Optional[int] = None,
    detection_bits: int = 16,
    scaling_w: Optional[float] = None,
    video_input: bool = True,
    videoseal_chunk_size: Optional[int] = None,  # default 32
    videoseal_step_size: Optional[int] = None,  # default 4
    attenuation: Optional[str] = None,
    attenuation_config: str = "attenuation.yaml",
    video_aggregation: str = "avg",
    videoseal_mode: Optional[str] = None,
    lowres_attenuation: bool = False,
    time_pooling_depth: Optional[int] = None,
    img_size_proc: Optional[List[int]] = None,
    interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
    device: str = "cpu",
) -> VideosealWM:
    """
    Build the Videoseal watermark model.
    ("Video Seal: Open and Efficient Video Watermarking",
    https://arxiv.org/abs/2412.09492)

    Args:
        model_card_or_path (str): Path to the model card or checkpoint.
        nbits (int): Number of bits for the watermark message.
        detection_bits (int): Number of bits for detection.
        scaling_w (float): Scaling factor for the watermark.
        video_input: A flag to turn off video watermarking and use WAM watermarking code instead
        videoseal_chunk_size (int): Chunk size for video watermarking.
        videoseal_step_size (int): Step size for video watermarking.
        attenuation (str): Type of attenuation to apply (e.g., "jnd").
        attenuation_config (str): Configuration file for attenuation.
        videoseal_mode (str): Mode for video watermarking.
        img_size_proc (int): Image size for processing.
        video_aggregation (str): Aggregation method for detection of video frames.
        lowres_attenuation (bool): Whether to do attenuation at low resolution.
        time_pooling_depth (int): Depth for temporal pooling.
        interpolation (dict): Interpolation parameters.
        device (str): Device to run the model on.

    Returns:
        VideosealWM: An instance of the Videoseal watermark model.
    """
    
    if "baseline" in model_card_or_path:
        method = model_card_or_path.split('/')[-1]
    else:
        method = "videoseal"
    
    # Setup the VideWam model
    model: Videoseal = setup_model_from_checkpoint(
        model_card_or_path, attenuation=attenuation, attenuation_config=attenuation_config 
    )
    model.eval()
    # model.compile()

    if nbits is not None:
        assert nbits == model.embedder.msg_processor.nbits, (
            f"nbits mismatch: {nbits} != {model.embedder.msg_processor.nbits}"
        )
    elif method == "videoseal":
        nbits = model.embedder.msg_processor.nbits
    elif hasattr(model.embedder, "nbits"):
        nbits = model.embedder.nbits
    else:
        raise ValueError("nbits must be specified for baseline methods")

    # Override model parameters in args
    if method == "videoseal":
        model.blender.scaling_w = scaling_w or model.blender.scaling_w
        model.chunk_size = videoseal_chunk_size or model.chunk_size
        model.step_size = videoseal_step_size or model.step_size
        model.video_mode = videoseal_mode or getattr(model, "mode", "repeat")
        model.img_size = img_size_proc or model.img_size

    # Setup the device
    model.to(device)

    # Override the temporal pooling
    if hasattr(model.embedder, "unet") and hasattr(model.embedder.unet, "time_pooling"):
        if time_pooling_depth is not None:
            model.embedder.unet.time_pooling = True
            model.embedder.unet.time_pooling_depth = time_pooling_depth
            model.embedder.unet.temporal_pool.kernel_size = model.step_size
            # When doing time average pooling, the step size should be set to 1.
            model.step_size = 1

    return VideosealWM(
        model=model,
        method=method,
        video_aggregation=video_aggregation,
        lowres_attenuation=lowres_attenuation,
        interpolation=interpolation,
        nbits=nbits,
        detection_bits=detection_bits,
        video_input=video_input,
    )
