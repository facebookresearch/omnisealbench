# Metrics 📊

This document provides an overview of various metrics used for evaluating the imperceptibility and robustness of watermarking models.

## Imperceptibility 👁️

These metrics measure some difference between the original and watermarked image.

### SSIM (Structural Similarity Index) 🔍

**Description**: SSIM is a perceptual metric that was designed to quantify image quality degradation caused by processing such as data compression or transmission losses. It considers changes in structural information, luminance, and contrast to assess the similarity between two images. Lower is better.

### PSNR (Peak Signal-to-Noise Ratio) 📈

**Description**: PSNR is a L2-based metric used to measure the quality of a reconstructed image compared to its original version. It is expressed in decibels and is commonly used in image compression. Higher PSNR values indicate better image quality.

### LPIPS (Learned Perceptual Image Patch Similarity) 🤖

**Description**: LPIPS is a metric that measures perceptual similarity between images using deep learning models. It evaluates the difference between image patches in a way that is supposed to align more closely with human perception compared to traditional metrics like SSIM and PSNR.

## Capacity and Robustness 🛡️

We assess the robustness of watermarking methods by evaluating whether a model can accurately decode the hidden message in an image after it has undergone various degradations.

### Bit Accuracy ✔️

**Description**: Bit accuracy is the proportion of decoded bits that match the ground truth. It provides a straightforward measure of how accurately the watermarking method can recover the original message.

### p-value 📉

Different watermarking methods have varying capacities, meaning they can hide different numbers of bits. Bit accuracy alone does not account for these differences, making it challenging to compare methods directly. Instead, we can use the p-value for a more nuanced comparison.

**Description**: In statistical hypothesis testing, the p-value helps determine the significance of results. In this context, it represents the probability of the decoded message matching as many bits as it does by chance, assuming the null hypothesis (H₀) is true. This is typically assessed using a binomial test.

Let's define:

- **M_orig**: the original message embedded in the watermark.
- **M_ext**: the message extracted from the watermarked content.
- **H₀**: the null hypothesis, which states that M_ext is random noise and not a meaningful watermark.

The binomial test is used to calculate the p-value by considering the number of matching bits between M_ext and M_orig. Under H₀, each bit in M_ext has a 50% chance of matching the corresponding bit in M_orig, assuming random noise.

The p-value is calculated as the probability of observing the number of matching bits (or more) by chance, given the total number of bits and the probability of a match under H₀. Mathematically, this can be expressed as:

$$
p\text{-value} = P(X \geq k \mid n, p)
$$

where:
- **X** is the random variable representing the number of matching bits.
- **k** is the observed number of matching bits.
- **n** is the total number of bits.
- **p = 0.5** is the probability of a match under H₀.

A lower p-value suggests stronger evidence against the null hypothesis, indicating that the extracted message is likely not random noise. This provides a statistical measure to assess the robustness and reliability of the watermarking method in accurately recovering the original message.