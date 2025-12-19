# Video Watermarking Models 🎵

## VideoSeal: Watermarking for Video Generation

[[`Arxiv`](https://arxiv.org/abs/2412.09492)] | [[`Github`](https://github.com/facebookresearch/videoseal)]

### Description 📜

VideoSeal introduces a robust watermarking technique for video generation models. Its main features include:

- Frame-level watermark embedding and detection.
- High robustness against common video transformations such as compression and resizing.
- Minimal impact on visual quality.

Model cards: [Videoseal_0.0](../../src/omnisealbench/cards/videoseal_0.0.yaml) , [Videoseal_1.0](../../src/omnisealbench/cards/videoseal_1.0.yaml)

## RivaGAN: Blind Watermarking for Deep Neural Networks

[[`Arxiv`](https://arxiv.org/abs/1909.01285)] | [[`Github`](https://github.com/DAI-Lab/RivaGAN)]

### Description 📜

RivaGAN introduces a deep learning-based blind watermarking framework tailored for video content. Built on an adversarial training scheme, RivaGAN leverages an encoder-decoder architecture paired with a discriminator to ensure both imperceptibility of the watermark and robustness against common video distortions and attacks. Notably, it operates in a blind setting—meaning the original (unwatermarked) video is not required for watermark extraction.

- End-to-end trainable encoder and decoder.
- Blind detection (no need for original video).
- Robust to various video processing attacks.
- Low Perceptual Impact.
- Message Agnostic: Supports embedding arbitrary bit strings into video content.

See the [Card](../../src/omnisealbench/cards/rivagan.yaml).