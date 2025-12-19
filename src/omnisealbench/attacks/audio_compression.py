# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import io
import logging
import re
import subprocess
import tempfile
from typing import Optional, Tuple

import torch
import torchaudio

logger = logging.getLogger(__name__)


def convert(
    wav: torch.Tensor, sr: int, format: str = "mp3", bitrate: str = "128k"
) -> Tuple[torch.Tensor, int]:
    """
    Convert WAV file to a specified lossy format.

    :param wav: Input WAV tensor. (torch.Tensor): Audio data to save. must be 2D tensor.
    :param sr: Sampling rate.
    :param format: Compression format (e.g., 'mp3').
    :param bitrate: Bitrate for compression.
    :return: Tuple of compressed WAV tensor and sampling rate.
    """
    # Check if the specified format is supported
    supported_formats = ["mp3", "ogg", "flac"]
    assert format in supported_formats, f"Unsupported format: {format}"

    # Extract the bit rate from string (e.g., '128k')
    match = re.search(r"\d+(\.\d+)?", str(bitrate))
    parsed_bitrate = float(match.group()) if match else None
    assert parsed_bitrate, "Invalid bitrate specified"
    try:
        # Create a virtual file instead of saving to disk
        buffer = io.BytesIO()

        # ffmpeg backend requires codecInfo
        codecInfo = torchaudio.io.CodecConfig(bit_rate=int(parsed_bitrate))
        # SoX backend does not support writing to file-like objects so we choose ffmpeg
        torchaudio.save(buffer, wav, sr, format=format, backend="ffmpeg", compression=codecInfo)

        # Move to the beginning of the file
        buffer.seek(0)
        compressed_wav, sr = torchaudio.load(buffer)
        return compressed_wav, sr
    except RuntimeError:
        logger.info(
            f"compression failed skipping compression: {format} {parsed_bitrate}"
        )
        return wav, sr


def get_mp3(wav_tensor: torch.Tensor, sr: int, bitrate: str = "128k") -> torch.Tensor:
    """
    Convert a batch of audio files to MP3 format, maintaining the original shape.

    This function takes a batch of audio files represented as a PyTorch tensor, converts
    them to MP3 format using the specified bitrate, and returns the batch in the same
    shape as the input.

    Args:
        wav_tensor (torch.Tensor): Batch of audio files represented as a tensor.
            Shape should be (batch_size, channels, length).
        sr (int): Sampling rate of the audio.
        bitrate (str): Bitrate for MP3 conversion, default is '128k'.

    Returns:
        torch.Tensor: Batch of audio files converted to MP3 format, with the same
            shape as the input tensor.
    """
    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape

    # Flatten tensor for conversion and move to CPU
    wav_tensor_flat = wav_tensor.view(1, -1).cpu()

    # Convert to MP3 format with specified bitrate
    wav_tensor_flat, _ = convert(wav_tensor_flat, sr, format="mp3", bitrate=bitrate)

    # Reshape back to original batch format and trim or pad if necessary
    wav_tensor = wav_tensor_flat.view(batch_size, channels, -1)
    compressed_length = wav_tensor.shape[-1]
    if compressed_length > original_length:
        wav_tensor = wav_tensor[:, :, :original_length]  # Trim excess frames
    elif compressed_length < original_length:
        padding = torch.zeros(
            batch_size, channels, original_length - compressed_length, device=device
        )
        wav_tensor = torch.cat((wav_tensor, padding), dim=-1)  # Pad with zeros

    # Move tensor back to the original device
    return wav_tensor.to(device)


def get_aac(
    wav_tensor: torch.Tensor,
    sr: int,
    bitrate: str = "128k",
    lowpass_freq: Optional[int] = None,
) -> torch.Tensor:
    """
    Converts a batch of audio tensors to AAC format and then back to tensors.

    This function first saves the input tensor batch as WAV files, then uses FFmpeg to convert
    these WAV files to AAC format. Finally, it loads the AAC files back into tensors.

    Args:
        wav_tensor (torch.Tensor): A batch of audio files represented as a tensor.
                                   Shape should be (batch_size, channels, length).
        sr (int): Sampling rate of the audio.
        bitrate (str): Bitrate for AAC conversion, default is '128k'.
        lowpass_freq (Optional[int]): Frequency for a low-pass filter. If None, no filter is applied.

    Returns:
        torch.Tensor: Batch of audio files converted to AAC and back, with the same
                      shape as the input tensor.
    """
    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape

    # Parse the bitrate value from the string
    match = re.search(r"\d+(\.\d+)?", bitrate)
    parsed_bitrate = (
        match.group() if match else "128"
    )  # Default to 128 if parsing fails

    # Flatten tensor for conversion and move to CPU
    wav_tensor_flat = wav_tensor.view(1, -1).cpu()

    with tempfile.NamedTemporaryFile(
        suffix=".wav"
    ) as f_in, tempfile.NamedTemporaryFile(suffix=".aac") as f_out:
        input_path, output_path = f_in.name, f_out.name

        # Save the tensor as a WAV file
        torchaudio.save(input_path, wav_tensor_flat, sr)

        # Prepare FFmpeg command for AAC conversion
        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ar",
            str(sr),
            "-b:a",
            f"{parsed_bitrate}k",
            "-c:a",
            "aac",
        ]
        if lowpass_freq is not None:
            command += ["-cutoff", str(lowpass_freq)]
        command.append(output_path)

        # Run FFmpeg and suppress output
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Load the AAC audio back into a tensor
        aac_tensor, _ = torchaudio.load(output_path)

    # Reshape and adjust length to match original tensor
    q = batch_size * channels
    if aac_tensor.shape[1] % q != 0:
        new_size = aac_tensor.shape[1] - aac_tensor.shape[1] % q
        aac_tensor = aac_tensor[:, :new_size]
    wav_tensor = aac_tensor.view(batch_size, channels, -1)
    compressed_length = wav_tensor.shape[-1]
    if compressed_length > original_length:
        wav_tensor = wav_tensor[:, :, :original_length]  # Trim excess frames
    elif compressed_length < original_length:
        padding = torch.zeros(
            batch_size, channels, original_length - compressed_length, device=device
        )
        wav_tensor = torch.cat((wav_tensor, padding), dim=-1)  # Pad with zeros

    return wav_tensor.to(device)