# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from .Decoder import Decoder, Decoder_Diffusion
from .Encoder_MP import Encoder_MP, Encoder_MP_Diffusion


class EncoderDecoder(nn.Module):
	'''
	A Sequential of Encoder_MP-Noise-Decoder
	'''

	def __init__(self, H, W, message_length):
		super(EncoderDecoder, self).__init__()
		self.encoder = Encoder_MP(H, W, message_length)
		self.decoder = Decoder(H, W, message_length)

	def forward(self, image, message):
		encoded_image = self.encoder(image, message)
		noised_image = encoded_image
		decoded_message = self.decoder(encoded_image)
		return encoded_image, noised_image, decoded_message


class EncoderDecoder_Diffusion(nn.Module):
	'''
	A Sequential of Encoder_MP-Noise-Decoder
	'''

	def __init__(self, H, W, message_length):
		super(EncoderDecoder_Diffusion, self).__init__()
		self.encoder = Encoder_MP_Diffusion(H, W, message_length)
		self.decoder = Decoder_Diffusion(H, W, message_length)

	def forward(self, image, message):
		encoded_image = self.encoder(image, message)
		noised_image = encoded_image
		decoded_message = self.decoder(encoded_image)

		return encoded_image, noised_image, decoded_message