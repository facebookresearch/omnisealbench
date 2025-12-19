# Image Watermarking Models 🖼️

**Note**: The cards use default parameters that achieve a PSNR of approximately 38, which are the same parameters used in the WAM paper. The cards can flexibly be changed.

## Watermark Anything Model (WAM) 

[[`Arxiv`](https://arxiv.org/abs/2411.07231)] | [[`Github`](https://github.com/facebookresearch/watermark-anything)]

### Description 📜

The Watermark Anything Model (WAM) is designed for localized image watermarking. Its main characteristics are:
- 32-bit capacity.
- Locates where the watermark is present in the image.
- Separates detection from decoding: detection can be done by thresholding the number of pixels detected as watermarked.
- Can extract multiple distinct messages from small watermarked areas.
- Designed to be robust against social media/internet sharing transformations such as heavy cropping, padding, inpainting, compression, etc.
- Can work on arbitrary resolution images.

See the [Card](../../src/omnisealbench/cards/wam.yaml).

The checkpoint can be downloaded with:

```bash
!wget https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth -P ../../checkpoints/wam/
```

## TrustMark: Universal Watermarking for Arbitrary Resolution Images 

[[`Arxiv`](https://arxiv.org/abs/2311.18297)] | [[`Github`](https://github.com/adobe/trustmark)]

### Description 📜

TrustMark is a GAN-based watermarking method. Some main characteristics are:
- Uses a 100-bit total payload, TrustMark is accessible with different effective capacities (61 bits (+ 35 ECC bits) - allows for 5 bit flips; 75 bits (+ 21 ECC bits) - allows for 3 bit flips; 40 bits (+ 56 ECC bits)).
- Different qualities (i.e., different PSNR) are available.
- Adds a distortion, with a scaling that can trade perceptibility and robustness.
- Can work on arbitrary resolution images.

See the [Card](../../src/omnisealbench/cards/trustmark.yaml).

## Watermarking Images in Self-Supervised Latent Spaces (SSL) 

[[`Arxiv`](https://arxiv.org/abs/2112.09581)] | [[`Github`](https://github.com/facebookresearch/ssl_watermarking)]

### Description 📜

This approach revisits watermarking techniques using pre-trained deep networks and self-supervised methods to embed marks and binary messages into latent spaces. The main characteristics are:
- Can hide an arbitrary number of bits (with a tradeoff with quality/robustness).
- Does not use an encoder/decoder training pipeline.
- Optimization needs to be done for each image.
- Adds a distortion, with a scaling that can trade perceptibility and robustness.

See the [Card](../../src/omnisealbench/cards/ssl.yaml).

## Fixed Neural Network Steganography: Train the Images, Not the Network (FNNS) 🧠

[[`Paper`](https://openreview.net/pdf?id=hcMvApxGSzZ)] | [[`Github`](https://github.com/varshakishore/FNNS)]

### Description 📜

This approach revisits steganography through adversarial perturbation: it modifies the image such that a fixed decoder correctly outputs the desired message (similar to SSL but with a different network). The main characteristics are:
- Can hide an arbitrary number of bits (with a tradeoff with quality/robustness).
- Does not use an encoder/decoder training pipeline.
- Optimization needs to be done for each image.
- Adds a distortion, with a scaling that can trade perceptibility and robustness.

See the [Card](../../src/omnisealbench/cards/fnns.yaml).

A checkpoint from the dino model can be downloaded from:

```bash
!wget https://dl.fbaipublicfiles.com/ssl_watermarking/dino_r50_90_plus_w.torchscript.pt -P ../../checkpoints/fnns/
```

## HiDDeN: Hiding Data With Deep Networks 🕵️

[[`Arxiv`](https://arxiv.org/abs/1807.09937)]

### Description 📜

First deep watermarking approach from 2018. We use the model trained and open-sourced [here](https://github.com/facebookresearch/stable_signature), which uses the same architecture and a similar training procedure. Note that this implementation uses a Just Noticeable Difference heatmap to modulate the watermark distortion for less visibility instead of using a perceptual loss during training like in the original paper. Main characteristics:
- Adds a distortion, with a scaling that can trade perceptibility and robustness.
- 48-bit capacity.

See the [Card](../../src/omnisealbench/cards/hidden.yaml).

The checkpoint trained for the stable signature paper can be downloaded with:

```bash
!wget https://dl.fbaipublicfiles.com/ssl_watermarking/encoder_with_jnd_48b.torchscript.pt -P ../../checkpoints/hidden
!wget https://dl.fbaipublicfiles.com/ssl_watermarking/decoder_48b.torchscript.pt -P ../../checkpoints/hidden
```

## Combined DWT-DCT Digital Image Watermarking 

[[`Paper`](https://pdfs.semanticscholar.org/1c47/f281c00cffad4e30deff48a922553cb04d17.pdf)]

### Description 📜

The algorithm watermarks a given image using a combination of the Discrete Wavelet Transform (DWT) and the Discrete Cosine Transform (DCT). Performance evaluation results show that combining the two transforms improved the performance of the watermarking algorithms that are based solely on the DWT transform.


## InvisMark: InvisMark: Invisible and Robust Watermarking for AI-generated Image Provenance

[[`Arxiv`](https://arxiv.org/pdf/2411.07795)] | [[`Github`](https://github.com/microsoft/InvisMark)]

### Description 📜

InvisMark (from Xu et al., arXiv:2411.07795):

- Supports 256-bit payload, allowing embedding of UUIDs with ECC.
- Achieves high imperceptibility (PSNR ≈ 51 dB, SSIM ≈ 0.998).
- Maintains >97 % bit-accuracy under manipulations.
- Tailored for high-resolution AI-generated imagery.
- Includes strategies to mitigate attacks targeting watermark robustness and security.

See the [Card](../../src/omnisealbench/cards/invismark.yaml).

Download the pretrained model weights (with 100 encoded bits and no ECC included) from: https://1drv.ms/f/c/7882afab383c8474/Ei_Lasu5CrpHsrNIkYRLenYBmx662VSAovq5hD8r-NsB5A?e=gbHNVX.