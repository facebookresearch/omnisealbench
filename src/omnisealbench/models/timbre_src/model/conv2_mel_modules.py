# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json

import torch
import torch.nn as nn
from torch.nn import LeakyReLU

from ..distortions.frequency import TacotronSTFT, fixed_STFT
from ..hifigan import AttrDict as hifigan_AttrDict
from ..hifigan import Generator as hifigan_Generator
from .blocks import (Conv2Encoder, FCBlock, WatermarkEmbedder,
                     WatermarkExtracter)


def get_vocoder(device, hifigan_dir):
    with open(f"{hifigan_dir}/config.json", "r") as f:        
        config = json.load(f)

    config = hifigan_AttrDict(config)
    vocoder = hifigan_Generator(config)
    
    ckpt = torch.load(f"{hifigan_dir}/generator_v1", map_location=device)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)    
    return vocoder


class Encoder(nn.Module):
    def __init__(
        self,
        process_config,
        model_config,
        msg_length,
        win_dim,
    ):
        super(Encoder, self).__init__()
        self.name = "conv2"
        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.add_carrier_noise = False
        self.block = model_config["conv2"]["block"]
        self.layers_CE = model_config["conv2"]["layers_CE"]
        self.EM_input_dim = model_config["conv2"]["hidden_dim"] + 2
        self.layers_EM = model_config["conv2"]["layers_EM"]

        self.vocoder_step = model_config["structure"]["vocoder_step"]
        # MLP for the input wm
        self.msg_linear_in = FCBlock(
            msg_length, win_dim, activation=LeakyReLU(inplace=True)
        )

        # stft transform
        self.stft = fixed_STFT(
            process_config["mel"]["n_fft"],
            process_config["mel"]["hop_length"],
            process_config["mel"]["win_length"],
        )

        self.ENc = Conv2Encoder(
            input_channel=1,
            hidden_dim=model_config["conv2"]["hidden_dim"],
            block=self.block,
            n_layers=self.layers_CE,
        )

        self.EM = WatermarkEmbedder(
            input_channel=self.EM_input_dim,
            hidden_dim=model_config["conv2"]["hidden_dim"],
            block=self.block,
            n_layers=self.layers_EM,
        )

    def forward(self, x, msg, global_step):
        num_samples = x.shape[2]
        spect, phase = self.stft.transform(x)

        carrier_encoded = self.ENc(spect.unsqueeze(1))
        watermark_encoded = (
            self.msg_linear_in(msg)
            .transpose(1, 2)
            .unsqueeze(1)
            .repeat(1, 1, 1, carrier_encoded.shape[3])
        )
        concatenated_feature = torch.cat(
            (carrier_encoded, spect.unsqueeze(1), watermark_encoded), dim=1
        )
        carrier_wateramrked = self.EM(concatenated_feature)

        self.stft.num_samples = num_samples
        y = self.stft.inverse(carrier_wateramrked.squeeze(1), phase.squeeze(1))
        return y, carrier_wateramrked

    def test_forward(self, x, msg):
        num_samples = x.shape[2]
        spect, phase = self.stft.transform(x)

        carrier_encoded = self.ENc(spect.unsqueeze(1))
        watermark_encoded = (
            self.msg_linear_in(msg)
            .transpose(1, 2)
            .unsqueeze(1)
            .repeat(1, 1, 1, carrier_encoded.shape[3])
        )
        concatenated_feature = torch.cat(
            (carrier_encoded, spect.unsqueeze(1), watermark_encoded), dim=1
        )
        carrier_wateramrked = self.EM(concatenated_feature)

        self.stft.num_samples = num_samples
        y = self.stft.inverse(carrier_wateramrked.squeeze(1), phase.squeeze(1))
        return y, carrier_wateramrked


class Decoder(nn.Module):
    def __init__(
        self,
        process_config,
        model_config,
        msg_length,
        win_dim,
        device: str = "cpu",
    ):
        super(Decoder, self).__init__()
        
        self.mel_transform = TacotronSTFT(
            filter_length=process_config["mel"]["n_fft"],
            hop_length=process_config["mel"]["hop_length"],
            win_length=process_config["mel"]["win_length"],
        )
        device = torch.device(device)
        self.vocoder_step = model_config["structure"]["vocoder_step"]

        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.block = model_config["conv2"]["block"]
        self.EX = WatermarkExtracter(
            input_channel=1,
            hidden_dim=model_config["conv2"]["hidden_dim"],
            block=self.block,
        )
        self.stft = fixed_STFT(
            process_config["mel"]["n_fft"],
            process_config["mel"]["hop_length"],
            process_config["mel"]["win_length"],
        )
        self.msg_linear_out = FCBlock(win_dim, msg_length)

    def forward(self, y, global_step):
        y_identity = y.clone()
        # pdb.set_trace()
        if global_step > self.vocoder_step:
            y_mel = self.mel_transform.mel_spectrogram(y.squeeze(1))
            # y = self.vocoder(y_mel)
            y_d = (self.mel_transform.griffin_lim(magnitudes=y_mel)).unsqueeze(1)
        else:
            y_d = y
            
        y_d_d = y_d
        spect, phase = self.stft.transform(y_d_d)

        extracted_wm = self.EX(spect.unsqueeze(1)).squeeze(1)
        msg = torch.mean(extracted_wm, dim=2, keepdim=True).transpose(1, 2)
        msg = self.msg_linear_out(msg)

        spect_identity, phase_identity = self.stft.transform(y_identity)
        extracted_wm_identity = self.EX(spect_identity.unsqueeze(1)).squeeze(1)
        msg_identity = torch.mean(extracted_wm_identity, dim=2, keepdim=True).transpose(
            1, 2
        )
        msg_identity = self.msg_linear_out(msg_identity)
        return msg, msg_identity

    def test_forward(self, y):
        spect, phase = self.stft.transform(y)
        extracted_wm = self.EX(spect.unsqueeze(1)).squeeze(1)
        msg = torch.mean(extracted_wm, dim=2, keepdim=True).transpose(1, 2)
        msg = self.msg_linear_out(msg)
        return msg

    def mel_test_forward(self, spect):
        extracted_wm = self.EX(spect.unsqueeze(1)).squeeze(1)
        msg = torch.mean(extracted_wm, dim=2, keepdim=True).transpose(1, 2)
        msg = self.msg_linear_out(msg)
        return msg
