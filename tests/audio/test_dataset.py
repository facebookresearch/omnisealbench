# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torchaudio

from omnisealbench.data.audio import read_local_audio_dataset
from omnisealbench.utils.common import tensor_to_message


def test_audio_dataset():
    from omnisealbench.data.audio import read_hf_audio_dataset


    data, metadata = read_hf_audio_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean",
        split="validation[:10]",
        padding_strategy="fixed",
        max_length=64000,  # 4 seconds at 16kHz
        sample_rate=24000,
        batch_size=5,
    )

    assert "sample_rate" in metadata and metadata["sample_rate"] == 24000, "Sampling rate should be 16kHz"

    for i, (_, batch) in enumerate(data):
        assert batch.shape[2] == 64000, "Audio samples should be padded to 64000 samples (4 seconds at 16kHz)"
        assert batch.shape[1] == 1, "Audio samples should have a single channel (mono)"
        assert batch.shape[0] == 5, "Audio samples should have batch 5"
        assert i < 2, "Should stop the first two batches"


def test_local_audio_dataset(tmp_path):

    # Make up a fake local dataset from test audio
    fake_audios = [torch.randn(1, 24000 * 4) for _ in range(8)]  # 8 audio samples, each 4 seconds at 16kHz
    for i, audio in enumerate(fake_audios):
        torchaudio.save(tmp_path / f"test_audio_{i}.wav", audio, sample_rate=24000)
    
    msg_tensor = torch.randint(0, 2, (16,), dtype=torch.float32)  # Empty message tensor for watermarking
    msg = tensor_to_message(msg_tensor)  # Convert message to tensor format
    
    fake_processed_audios = [torch.randn(1, 24000 * 4) for _ in range(8)]  # 8 audio samples, each 4 seconds at 16kHz
    for i, audio in enumerate(fake_processed_audios):
        torchaudio.save(tmp_path / f"processed_audio_{i}.wav", audio, sample_rate=24000)
        with open(tmp_path / f"watermark_{i}.txt", "w", encoding="utf-8") as f:
            f.write(msg)

    audio_dataset, metadata = read_local_audio_dataset(
        dataset_dir=tmp_path,
        audio_pattern="test_audio_*.wav",
        watermarked_audio_pattern="processed_audio_*.wav",
        message_pattern="watermark_*.txt",
        padding_strategy="fixed",
        max_length=24000 * 4,  # 4 seconds at 16kHz
        batch_size=2,
    )

    for _, batch in audio_dataset:
        assert isinstance(batch, dict), "Batch should be a dictionary of tensors"
        assert "original" in batch, "Batch should contain 'original' key for original audio"
        assert "watermarked" in batch, "Batch should contain 'watermark' key for processed audio"
        assert "message" in batch, "Batch should contain 'message' key for watermark messages"
        assert batch["original"].shape[1] == 1, "Audio samples should have a single channel (mono)"
        assert batch["original"].shape[2] == 24000 * 4, "Audio samples should be padded to 24000 * 4 samples (4 seconds at 16kHz)"
        assert batch["original"].shape[0] == 2, "Audio samples should have batch size of 2"
        torch.testing.assert_close(msg_tensor, batch["message"][0])
