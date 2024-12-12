# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Test with:
    python -m src.models.wam
"""

import random

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from ..augmentation.augmenter import Augmenter
from ..modules.jnd import JND
from .embedder import Embedder
from .extractor import Extractor


class Wam(nn.Module):
    wm_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        embedder: Embedder,
        detector: Extractor,
        augmenter: Augmenter,
        attenuation: JND = None,
        scaling_w: float = 1.0,
        scaling_i: float = 1.0,
        roll_probability: float = 0,
    ) -> None:
        """
        WAM (watermark-anything models) model that combines an embedder, a detector, and an augmenter.
        Embeds a message into an image and detects it as a mask.

        Arguments:
            embedder: The watermark embedder
            detector: The watermark detector
            augmenter: The image augmenter
            attenuation: The JND model to attenuate the watermark distortion
            scaling_w: The scaling factor for the watermark
            scaling_i: The scaling factor for the image
        """
        super().__init__()
        # modules
        self.embedder = embedder
        self.detector = detector
        self.augmenter = augmenter
        self.attenuation = attenuation
        # scalings
        self.scaling_w = scaling_w
        self.scaling_i = scaling_i
        # rolling
        self.roll_probability = roll_probability

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        return self.embedder.get_random_msg(bsz, nb_repetitions)  # b x k

    def get_secong_msg(
        self, msgs, prop_swith_min=0.5, prop_switch_max=0.75
    ) -> torch.Tensor:
        msgs2 = msgs.clone()

        # Determine the number of columns to flip
        num_columns = msgs.shape[1]
        num_columns_to_flip = random.randint(
            int(prop_swith_min * num_columns), int(prop_switch_max * num_columns)
        )

        # Randomly choose columns to flip
        columns_to_flip = random.sample(range(num_columns), num_columns_to_flip)

        # Flip the bits in the chosen columns
        for col in columns_to_flip:
            msgs2[:, col] = 1 - msgs2[:, col]

        msgs2 = msgs2.to(msgs.device)
        return msgs2

    def forward(
        self,
        imgs: torch.Tensor,
        masks: torch.Tensor,
        msgs: torch.Tensor = None,
        no_overlap: bool = False,
        nb_times: int = None,
        params=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate watermarked images from the input images.
        """
        if random.random() < self.roll_probability:
            roll = True
        else:
            roll = False
        h, w = params.img_size_extractor, params.img_size_extractor
        # If the image does not have the size expected by the encoder, resize it
        resize = transforms.Resize(
            (h, w), interpolation=transforms.InterpolationMode.BILINEAR
        )
        inverse_resize = transforms.Resize(
            (imgs.shape[-2:]),
            interpolation=transforms.InterpolationMode.NEAREST,
            antialias=True,
        )
        # Get the masks with shape 256 x 256
        aux = self.augmenter.mask_embedder(
            resize(imgs), masks=masks, no_overlap=no_overlap, nb_times=nb_times
        ).to(imgs.device)
        if len(aux.shape) == 4:
            # add channel dimension, corresponding to the number of masks per
            aux = aux.unsqueeze(2)
        B, C, K, H, W = aux.shape
        aux = aux.view(B * K, C, H, W)
        aux = F.interpolate(
            aux,
            size=(imgs.shape[-2], imgs.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        mask_targets = aux.view(B, C, K, imgs.shape[-2], imgs.shape[-1])
        mask_targets = mask_targets.float()
        # combined_mask = torch.zeros_like(mask_targets[:, 0].float())
        msgs_l = []
        combined_imgs = imgs.clone()
        for nb_wm in range(mask_targets.shape[1]):
            mask = mask_targets[:, nb_wm, :, :, :].float()
            # combined_mask += mask
            msgs = self.get_random_msg(imgs.shape[0])  # b x k
            msgs = msgs.to(imgs.device)
            msgs_l.append(msgs)
            # deltas_w = self.embedder(imgs, msgs)
            # imgs_w = self.scaling_i * imgs + self.scaling_w * deltas_w
            deltas_w = self.embedder(resize(imgs), msgs)
            imgs_w = self.scaling_i * imgs + self.scaling_w * inverse_resize(deltas_w)
            if self.attenuation is not None:
                imgs_w = self.attenuation(imgs, imgs_w)
            if not roll:
                combined_imgs = combined_imgs * (1 - mask) + imgs_w * mask
            else:
                combined_imgs = combined_imgs * torch.roll(
                    1 - mask, shifts=-1, dims=0
                ) + torch.roll(imgs_w, shifts=-1, dims=0) * torch.roll(
                    mask, shifts=-1, dims=0
                )
        if not roll:
            imgs_aug, mask_targets, selected_aug = self.augmenter.post_augment(
                combined_imgs, mask_targets.squeeze(2)
            )
        else:
            imgs_aug, mask_targets, selected_aug = self.augmenter.post_augment(
                combined_imgs, torch.roll(mask_targets.squeeze(2), shifts=-1, dims=0)
            )
        preds = self.detector(resize(imgs_aug))  # b (1+nbits) h w
        msgs_l = torch.stack(msgs_l)
        msgs_l = msgs_l.transpose(0, 1)
        if roll:
            msgs_l = torch.roll(msgs_l, shifts=-1, dims=0)
        outputs = {
            "msgs": msgs_l,  # original messages: b k
            "masks": resize(mask_targets).bool(),  # augmented masks: b z h w
            "imgs_w": (
                imgs_w if not roll else torch.roll(imgs_w, shifts=-1, dims=0)
            ),  # watermarked images: b c H W (original shape)
            "imgs_aug": resize(imgs_aug),  # augmented images: b c h w
            "preds": preds,  # predicted masks and/or messages: b (1+nbits) h w
            "selected_aug": selected_aug,  # selected augmentation
        }
        return outputs

    def embed(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate watermarked images from the input images.
        """

        # optionally create message
        if msgs is None:
            msgs = self.get_random_msg(imgs.shape[0])  # b x k
            msgs = msgs.to(imgs.device)

        # generate watermarked images
        deltas_w = self.embedder(imgs, msgs)
        imgs_w = self.scaling_i * imgs + self.scaling_w * deltas_w

        outputs = {
            "msgs": msgs,  # original messages: b k
            "deltas_w": deltas_w,  # predicted watermarks: b c h w
            "imgs_w": imgs_w,  # watermarked images: b c h w
        }
        return outputs

    def detect(
        self,
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Detect watermarks in the input images.
        """

        # detect watermark
        preds = self.detector(imgs)

        outputs = {
            "preds": preds,  # predicted masks and/or messages: b (1+nbits) h w
        }
        return outputs

    def postprocess_masks(
        self,
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.
        """
        return ...

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


if __name__ == "__main__":

    from functools import partial

    import torch
    from src.models.embedder import Embedder
    from src.models.extractor import Extractor
    from src.modules.msg_processor import MsgProcessor
    from src.modules.pixel_decoder import PixelDecoder
    from src.modules.vae import VAEDecoder, VAEEncoder
    from src.modules.vit import ImageEncoderViT

    nbits = 0
    ch = 64
    ch_mult = (1, 2, 2)
    msg_emb_ch = 4 + 2 * nbits

    print("\ntest for the embedder model\n")

    # test the embedder model
    encoder = VAEEncoder(
        ch=ch,
        out_ch=3,
        ch_mult=ch_mult,
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        in_channels=3,
        resolution=64,
        z_channels=4,
        double_z=False,
    )
    msg_processor = MsgProcessor(
        nbits=nbits,
        hidden_size=msg_emb_ch,
        msg_processor_type="concat",
    )
    decoder = VAEDecoder(
        ch=ch,
        out_ch=3,
        ch_mult=ch_mult,
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        in_channels=3,
        resolution=64,
        z_channels=4 + msg_emb_ch,
    )

    # build the model
    model = Embedder(encoder, decoder, msg_processor)
    print(model)
    print(
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M parameters"
    )
    print(
        f"encoder: {sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1e6:.1f}M parameters"
    )
    print(
        f"decoder: {sum(p.numel() for p in decoder.parameters() if p.requires_grad) / 1e6:.1f}M parameters"
    )
    print(
        f"msg_processor: {sum(p.numel() for p in msg_processor.parameters() if p.requires_grad) / 1e6:.1f}M parameters"
    )

    # test the model
    imgs = torch.randn(2, 3, 256, 256)
    msgs = torch.randint(0, 2, (2, nbits))
    out = model(imgs, msgs)
    print(out.shape)

    print("\ntest for the detector model\n")

    image_size = 256
    vit_patch_size = 16

    model = "tiny"
    if model == "base":
        encoder_embed_dim = 768
        encoder_depth = 12
        encoder_num_heads = 12
    elif model == "small":
        encoder_embed_dim = 384
        encoder_depth = 12
        encoder_num_heads = 6
    elif model == "tiny":
        encoder_embed_dim = 192
        encoder_depth = 12
        encoder_num_heads = 3

    encoder_global_attn_indexes = [2, 5, 8, 11]
    out_chans = 512

    image_embedding_size = image_size // vit_patch_size
    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        # window_size = 14,
        out_chans=out_chans,
    )
    pixel_decoder = PixelDecoder(embed_dim=out_chans)

    detector = Extractor(image_encoder=image_encoder, pixel_decoder=pixel_decoder)

    print(detector)
    print(
        f"{sum(p.numel() for p in detector.parameters() if p.requires_grad) / 1e6:.1f}M parameters"
    )
    print(
        f"image_encoder: {sum(p.numel() for p in image_encoder.parameters() if p.requires_grad) / 1e6:.1f}M parameters"
    )
    print(
        f"pixel_decoder: {sum(p.numel() for p in pixel_decoder.parameters() if p.requires_grad) / 1e6:.1f}M parameters"
    )
