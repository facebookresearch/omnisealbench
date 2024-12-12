# Audio Watermarking Models 🎵

## Proactive Detection of Voice Cloning with Localized Watermarking (AudioSeal)

[[`Arxiv`](https://arxiv.org/abs/2401.17264)] | [[`Github`](https://github.com/facebookresearch/audioseal)]

### Description 📜

AudioSeal is the first audio watermarking technique designed specifically for localized detection of AI-generated speech. Its main characteristics are:

- Employs a generator/detector architecture.
- Has detection separate from decoding, enabling localized watermark detection up to the sample level.
- Features a fast detection process.

See the [Card](../../cards/audio/audioseal.yaml).

## WavMark: Watermarking for Audio Generation

[[`Arxiv`](https://arxiv.org/abs/2308.12770)] | [[`Github`](https://github.com/swesterfeld/audiowmark)]

### Description 📜

WavMark uses invertible networks to hide 32 bits in 1-second audio segments. Detection is performed by sliding along the audio in 0.05-second steps and decoding the message for each window. If the first 10 decoded bits match a synchronization pattern, the rest of the payload is saved (22 bits), and the window can directly slide 1 second (instead of 0.05 seconds). Main characteristics:

- 32 bits in 1-second audio segments, but the synchronization bits reduce the capacity for the encoded message, accounting for 31% of the total capacity.

See the [Card](../../cards/audio/wavmark.yaml).

## Detecting Voice Cloning Attacks via Timbre Watermarking

[[`Arxiv`](https://arxiv.org/abs/2312.03410)] | [[`Github`](https://github.com/TimbreWatermarking/TimbreWatermarking)]

### Description 📜

Timbre embeds the watermark into the frequency domain, which is inherently robust against common data processing methods. Main characteristics:

- Encoder/decoder method.
- Default 10-bit capacity.

See the [Card](../../cards/audio/timbre.yaml).