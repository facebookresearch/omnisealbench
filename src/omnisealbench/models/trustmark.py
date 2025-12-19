# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
from torchvision import transforms  # type: ignore[import-untyped]

try:
    from PIL import Image

    BILINEAR = Image.BILINEAR  # type: ignore[attr-defined]
except ImportError:
    from PIL import Image

    BILINEAR = Image.Resampling.BILINEAR

from trustmark import TrustMark  # type: ignore

from omnisealbench.interface import Device


class TrustMarkWM:
    """
    Implementation of the TrustMark watermark model.
    ("Trustmark: Universal watermarking for arbitrary resolution images",
    https://arxiv.org/abs/2311.18297)
    """

    model: TrustMark

    def __init__(self, model: torch.nn.Module, nbits: int = 32):
        self.model = model
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.nbits = nbits

    def _encode(self, content: torch.Tensor, msg: str) -> torch.Tensor:
        img = self.to_pil(content)
        img_w = self.model.encode(img, msg, MODE="binary")
        return self.to_tensor(img_w).to(content.device)

    def _decode(  # type: ignore
        self,
        content: torch.Tensor,
        mode="text",
        message_threshold: float = 0.0,
    ) -> Tuple[str, bool, int]:
        """
        Change the model.decode implementation and return secret_pred
        Inputs
            content: tensor of a single image
            mode: decoding mode, can be "binary" or "text"
            detection_threshold: float, threshold for detecting the watermark
        Outputs:
            secret_pred: str, the decoded secret message
            detected: bool, whether the watermark was detected
            version: int, the version of the watermark schema used
        """
        img = self.to_pil(content)
        stego_image = self.model.get_the_image_for_processing(img)
        if min(stego_image.size) > 256:
            stego_image = stego_image.resize((256, 256), BILINEAR)

        stego = self.to_tensor(stego_image).unsqueeze(0).to(self.model.decoder.device)
        # Normalize to range [-1, 1]
        stego = stego * 2.0 - 1.0

        with torch.no_grad():
            secret_binaryarray = (
                (self.model.decoder.decoder(stego) > message_threshold).detach().cpu().numpy()
            )  # (1, secret_len)
        if self.model.use_ECC:
            secret_pred, detected, version = self.model.ecc.decode_bitstream(
                secret_binaryarray, mode
            )[0]
            if not detected:
                # last ditch attempt to recover a possible corruption of
                # the version bits by trying all other schema types
                modeset = [x for x in range(0, 3) if x not in [version]]  # not bch_3
                for m in modeset:
                    if m == 0:
                        secret_binaryarray[0][-2] = False
                        secret_binaryarray[0][-1] = False
                    if m == 1:
                        secret_binaryarray[0][-2] = False
                        secret_binaryarray[0][-1] = True
                    if m == 2:
                        secret_binaryarray[0][-2] = True
                        secret_binaryarray[0][-1] = False
                    if m == 3:
                        secret_binaryarray[0][-2] = True
                        secret_binaryarray[0][-1] = True
                    secret_pred, detected, version = self.model.ecc.decode_bitstream(
                        secret_binaryarray, mode
                    )[0]
                    if detected:
                        return secret_pred, detected, version
                    else:
                        return secret_pred, False, -1
                        # return "", False, -1
            else:
                return secret_pred, detected, version
        else:
            assert len(secret_binaryarray.shape) == 2
            secret_pred = "".join(str(int(x)) for x in secret_binaryarray[0])
            return secret_pred, True, -1

    @torch.inference_mode()
    def generate_watermark(
        self, contents: List[torch.Tensor], message: torch.Tensor
    ) -> torch.Tensor:
        assert message.ndim == 1, "Message should be a 1D tensor."
        msg = "".join(str(int(m)) for m in message.tolist())
        imgs_w = [self._encode(content, msg) for content in contents]
        return imgs_w  # type: ignore

    @torch.inference_mode()
    def detect_watermark(
        self,
        contents: List[torch.Tensor],
        detection_threshold: float = 0.0,  # not used
        message_threshold: float = 0.0,
        detection_bits: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = [
            self._decode(
                content, mode="binary", message_threshold=message_threshold
            )
            for content in contents
        ]

        preds, msgs = [], []
        for msg, detected, _ in outputs:
            msg_tensor = torch.tensor([int(bit) for bit in msg], dtype=torch.int32)

            # Remove ECC bits and others
            msg_tensor = msg_tensor[:self.nbits]
            msgs.append(msg_tensor)
            preds.append(detected)

        preds = torch.tensor(preds, dtype=torch.int32)  # type: ignore
        msgs = torch.stack(msgs, dim=0)  # type: ignore

        message = torch.gt(msgs, message_threshold).int()  # b k

        return {
            "prediction": preds,  # type: ignore
            "message": message,  # type: ignore
        }


def build_model(
    model_type: str = "Q",
    encoding_type: str = "BCH_SUPER",
    nbits: Optional[int] = None,
    verbose: bool = True,
    device: Device = "cpu",
) -> TrustMarkWM:
    """
    Builds the TrustMark watermark model.
    ("Trustmark: Universal watermarking for arbitrary resolution images",
    https://arxiv.org/abs/2311.18297)
    Args:
        model_card_or_path (str): Path to the model card or pre-trained model. See
            https://github.com/adobe/trustmark/blob/main/python/CONFIG.md#encoding-modes
        nbits (int): Number of bits in the secret message to be embedded in the image.
        verbose (bool): Whether to print verbose output.
        device (str): Device to load the model on (e.g., 'cpu', 'cuda').

    Returns:
        TrustMarkWM: An instance of the TrustMark watermark model.
    """

    # https://github.com/adobe/trustmark/blob/main/python/CONFIG.md#encoding-modes
    encoding_map = {
        "BCH_SUPER": (TrustMark.Encoding.BCH_SUPER, 40),
        "BCH_3": (TrustMark.Encoding.BCH_3, 75),
        "BCH_4": (TrustMark.Encoding.BCH_4, 68),
        "BCH_5": (TrustMark.Encoding.BCH_5, 61),
    }
    encoding, nbits_ = encoding_map.get(encoding_type)

    if nbits is None:
        nbits = nbits_
    else:
        assert nbits <= nbits_, f"model {encoding_type} requires nbits<={nbits_}, but user passes {nbits}"

    model = TrustMark(
        verbose=verbose,
        model_type=model_type,
        encoding_type=encoding,
        device=device,
    )
    return TrustMarkWM(model=model, nbits=nbits)
