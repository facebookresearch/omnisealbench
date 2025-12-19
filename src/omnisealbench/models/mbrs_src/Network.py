# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .Encoder_MP_Decoder import EncoderDecoder, EncoderDecoder_Diffusion


class Network:

	def __init__(self, H, W, message_length, with_diffusion=False):
		# network
		if not with_diffusion:
			self.encoder_decoder = EncoderDecoder(H, W, message_length)
		else:
			self.encoder_decoder = EncoderDecoder_Diffusion(H, W, message_length)


	def load_model_ed(self, path_encoder_decoder: str):
		self.encoder_decoder.load_state_dict(torch.load(path_encoder_decoder, map_location="cpu"))
	