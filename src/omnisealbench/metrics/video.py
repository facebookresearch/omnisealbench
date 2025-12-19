# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import re
import subprocess
import tempfile

import torch

from omnisealbench.utils.common import batch_safe


def tensor_to_video(
        tensor, filename, fps, 
        codec=None, crf=23,
    ):
    """ Saves a video tensor into a video file."""
    T, C, H, W = tensor.shape
    assert C == 3, "Video must have 3 channels (RGB)."
    video_data = (tensor * 255).to(torch.uint8).numpy()
    
    ffmpeg_bin = os.environ.get('FFMPEG_BINARY', 'ffmpeg')
    
    # write video using ffmpeg
    if codec is None:
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_file:
            video_data.tofile(temp_file.name)
            temp_filename = temp_file.name
        command = [
            f'{ffmpeg_bin}', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
            '-r', f'{fps}', '-i', temp_filename, filename
        ]
    elif codec == 'libx264':
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_file:
            video_data.tofile(temp_file.name)
            temp_filename = temp_file.name
        command = [
            f'{ffmpeg_bin}', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
            '-r', f'{fps}', '-i', temp_filename,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', f'{crf}',
            filename
        ]
    else:
        raise ValueError(f"Unsupported codec: {codec}")
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.remove(temp_filename)

def vmaf_on_file(
    vid_o: str,
    vid_w: str,
    n_threads: int = 8,
    ffmpeg_bin: str = '/private/home/pfz/09-videoseal/vmaf-dev/ffmpeg-git-20240629-amd64-static/ffmpeg',
) -> float:
    """
    Runs `ffmpeg -i vid_o.mp4 -i vid_w.mp4 -filter_complex libvmaf` and returns the score.
    """
    # Execute the command and capture the output to get the VMAF score
    if n_threads == 0:
        command = [
                ffmpeg_bin,
                '-i', vid_o,
                '-i', vid_w,
                '-filter_complex', 'libvmaf',
                '-f', 'null', '-'
            ]
    else:
        command = [
            ffmpeg_bin,
            '-i', vid_o,
            '-i', vid_w,
            '-lavfi', f'libvmaf=\'n_threads={n_threads}\'',
            '-f', 'null', '-'
        ]
    
    result = subprocess.run(command, text=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    vmaf_score = None
    for line in result.stderr.split('\n'):
        if "VMAF score:" in line:
            # numerical part of the VMAF score with regex
            match = re.search(r"VMAF score: ([0-9.]+)", line)
            if match:
                vmaf_score = float(match.group(1))
                break
    return vmaf_score

@batch_safe
def vmaf_on_tensor(
    tensor1: torch.Tensor, 
    tensor2: torch.Tensor, 
    fps: int = 24, codec: str = 'libx264', crf: int = 23, 
    n_threads: int = 8,
) -> float:
    """
    Compute VMAF between original and compressed/watermarked video tensors.
    Args:
        tensor1: Original video tensor
        tensor2: Compressed/Watermarked video tensor. \
            If None, the original tensor is used as the second tensor, but with the codec and crf specified.
        fps: Frames per second
        codec: Codec to use when saving the video to file
        crf: Constant Rate Factor for libx264 codec
    """

    tensor1 = tensor1.to("cpu")
    tensor2 = tensor2.to("cpu")

    ffmpeg_bin = os.environ.get('FFMPEG_BINARY', 'ffmpeg')
    with tempfile.NamedTemporaryFile(suffix='.mp4') as file1, \
         tempfile.NamedTemporaryFile(suffix='.mp4') as file2:
        
        # save tensors to video files
        if tensor2 is None:
            tensor_to_video(tensor1, file1.name, fps=fps, codec=None)
            tensor2 = tensor1
        else:
            tensor_to_video(tensor1, file1.name, fps=fps, codec=codec, crf=crf)
        tensor_to_video(tensor2, file2.name, fps=fps, codec=codec, crf=crf)
        
        # compute VMAF
        vmaf_score = vmaf_on_file(file1.name, file2.name, n_threads=n_threads, ffmpeg_bin=ffmpeg_bin)

        return vmaf_score


# FIXME: Remove the ffmpeg_bin from the function signature and use system env instead
def vmaf_on_tensor_chunked(
    tensor1, 
    tensor2=None,
    fps=24, 
    codec='libx264', 
    crf=23, 
    chunk_size=30,
    return_aux=False,
    n_threads=8,
):
    """
    Compute VMAF between original and compressed/watermarked video tensors by chunking over the temporal dimension.
    
    Args:
        tensor1: Original video tensor of shape [T, C, H, W]
        tensor2: Compressed/Watermarked video tensor of the same shape. 
                 If None, tensor2 is set to tensor1 (and tensor1 will be re-encoded with codec and crf).
        fps: Frames per second.
        codec: Codec to use when saving the video files.
        crf: Constant Rate Factor for the codec.
        chunk_size: Number of frames per chunk.
        return_aux: If True, also return auxiliary info (a list of dictionaries, one per chunk).
        
    Returns:
        overall_vmaf: Weighted average VMAF score over all chunks.
        (Optional) aux_info: List of auxiliary info for each chunk.

    Note:
        This implementation assumes that the functions `tensor_to_video` and `vmaf_on_file`
        are defined elsewhere.
    """
    if tensor2 is None:
        tensor2 = tensor1

    total_frames = tensor1.shape[0]
    num_chunks = (total_frames + chunk_size - 1) // chunk_size

    vmaf_scores = []  # list of (chunk_vmaf, num_frames)
    aux_info_list = []  # auxiliary info for each chunk

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_frames)
        chunk1 = tensor1[start:end]
        chunk2 = tensor2[start:end]

        with tempfile.NamedTemporaryFile(suffix='.mp4') as file1, \
             tempfile.NamedTemporaryFile(suffix='.mp4') as file2:

            # Save the chunk as video files.
            tensor_to_video(chunk1, file1.name, fps=fps, codec=codec, crf=crf)
            tensor_to_video(chunk2, file2.name, fps=fps, codec=codec, crf=crf)

            # Compute VMAF for this chunk.
            chunk_vmaf = vmaf_on_file(file1.name, file2.name, n_threads=n_threads)

            # Collect auxiliary info.
            MB = 1024 * 1024
            filesize1 = os.path.getsize(file1.name) / MB
            filesize2 = os.path.getsize(file2.name) / MB
            duration = (end - start) / fps  # in seconds
            bps1 = filesize1 / duration if duration > 0 else 0
            bps2 = filesize2 / duration if duration > 0 else 0

            aux = {
                'filesize1_MB': filesize1,
                'filesize2_MB': filesize2,
                'duration_sec': duration,
                'bps1_MBps': bps1,
                'bps2_MBps': bps2,
                'num_frames': end - start,
                'chunk_vmaf': chunk_vmaf
            }

            vmaf_scores.append((chunk_vmaf, end - start))
            aux_info_list.append(aux)

    # Compute the weighted average VMAF: weight each chunk's VMAF by its number of frames.
    total_frames_used = sum(num_frames for _, num_frames in vmaf_scores)
    overall_vmaf = sum(score * num_frames for score, num_frames in vmaf_scores) / total_frames_used

    if return_aux:
        return overall_vmaf, aux_info_list
    return overall_vmaf


@batch_safe
def psnr(x: torch.Tensor, y: torch.Tensor, is_video: bool = False):
    """
    Compute PSNR over a batch of images (or video frames).

    Args:
        x: Tensor with normalized values in [0,1] (shape: [B, C, H, W] or [T, C, H, W])
        y: Tensor with normalized values in [0,1], same shape as x
        is_video: If True, compute PSNR over all pixels (one scalar);
                  if False, compute PSNR per image.
    Returns:
        A scalar (float) if is_video is True, or a tensor of shape [B] otherwise.
    """
    peak = 20 * math.log10(255.0)
    delta = 255 * (x - y)
    
    if is_video:
        # Combine all frames/images into one tensor.
        delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        noise = torch.mean(delta ** 2)
        # If noise is exactly zero, return infinity.
        if noise.item() == 0:
            return float('inf')
        return peak - 10 * math.log10(noise.item())
    else:
        # Compute per-image MSE.
        mse = torch.mean(delta ** 2, dim=(1, 2, 3))
        psnr_vals = []
        for m in mse:
            if m.item() == 0:
                psnr_vals.append(float('inf'))
            else:
                psnr_vals.append(peak - 10 * math.log10(m.item()))
        return torch.tensor(psnr_vals)


@batch_safe
def original_psnr(x: torch.Tensor, y: torch.Tensor, is_video: bool = False):
    """ 
    Return PSNR 
    Args:
        x: Image tensor with normalized values (≈ [0,1])
        y: Image tensor with normalized values (≈ [0,1]), ex: original image
        is_video: If True, the PSNR is computed over the entire batch, not on each image separately
    """
    delta = 255 * (x - y)
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])  # BxCxHxW
    peak = 20 * math.log10(255.0)
    avg_on_dims = (0, 1, 2, 3) if is_video else (1, 2, 3)
    noise = torch.mean(delta**2, dim=avg_on_dims)
    psnr = peak - 10 * torch.log10(noise)
    return psnr


@batch_safe
def compute_psnr_chunked(x: torch.Tensor, y: torch.Tensor, is_video: bool = False, chunk_size: int = 8) -> torch.Tensor:
    """
    Compute PSNR by processing in chunks.
    
    This implementation explicitly distinguishes between:
      - Video mode (is_video=True): A single PSNR computed from the total squared error over all frames.
      - Per-image mode (is_video=False): PSNR computed for each image individually.
    
    Args:
        x: Tensor with normalized values in [0,1], shape: [B, C, H, W] or [T, C, H, W]
        y: Tensor with normalized values in [0,1], same shape as x
        is_video: If True, compute one overall PSNR value; if False, compute per-image PSNR.
        chunk_size: Number of images/frames to process at once.
    
    Returns:
        A scalar (float) if is_video is True, or a tensor of PSNR values (one per image) if False.
    """
    peak = 20 * math.log10(255.0)
    
    if is_video:
        total_squared_error = 0.0
        total_pixels = 0
        for i in range(0, x.shape[0], chunk_size):
            chunk_x = x[i:i+chunk_size]
            chunk_y = y[i:i+chunk_size]
            # Scale differences and flatten all frames in the chunk.
            delta = 255 * (chunk_x - chunk_y)
            delta = delta.reshape(-1, chunk_x.shape[-3], chunk_x.shape[-2], chunk_x.shape[-1])
            total_squared_error += (delta ** 2).sum().item()
            total_pixels += delta.numel()
        mse_total = total_squared_error / total_pixels
        if mse_total == 0:
            return torch.tensor(float('inf'))
        return torch.tensor(peak - 10 * math.log10(mse_total))
    
    else:
        psnr_vals = []
        for i in range(0, x.shape[0], chunk_size):
            chunk_x = x[i:i+chunk_size]
            chunk_y = y[i:i+chunk_size]
            delta = 255 * (chunk_x - chunk_y)
            mse = torch.mean(delta ** 2, dim=(1, 2, 3))
            for m in mse:
                if m.item() == 0:
                    psnr_vals.append(float('inf'))
                else:
                    psnr_vals.append(peak - 10 * math.log10(m.item()))
        return torch.tensor(psnr_vals)
