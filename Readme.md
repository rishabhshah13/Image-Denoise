
---

# Image Denoising using GAN

## Project Overview

This project focuses on removing unwanted noise from real scene images using Generative Adversarial Networks (GANs). Image denoising is crucial for various computer vision applications, where clean images are essential for accurate analysis and model training.

## Implementation Details

Pix2Pix GAN architecture was employed for image denoising, with implementation streamlined using PyTorch Lightning. The Smartphone Image Denoising Dataset (SIDD) was utilized for training and evaluation.

## Results

### Evaluation Metrics

The performance of the denoising model was assessed using two common metrics:

1. **Peak Signal-to-Noise Ratio (PSNR):** Measures the quality of the denoised image compared to the original, with higher values indicating better denoising.
   
2. **Structural Similarity Index (SSIM):** Evaluates the structural similarity between the denoised and original images, where 1 indicates perfect similarity.

### Results Tables

#### Denoising Performance

| Image Set | PSNR (dB) | SSIM |
|-----------|------------|------|
| Small Dataset     | 35.1063785 | 0.93385 |
| Medium Dataset     | 38.851     | 0.9488 |


## Smartphone Image Denoising Dataset (SIDD)

The Smartphone Image Denoising Dataset (SIDD) is a widely-used benchmark dataset for image denoising tasks. It comprises approximately 30,000 noisy images captured under different scenes and lighting conditions using various smartphones. The dataset provides a diverse range of real-world scenarios, making it valuable for training and evaluating denoising models.

## Conclusion

In conclusion, the utilization of Pix2Pix GAN architecture combined with PyTorch Lightning facilitated effective denoising of real scene images, significantly enhancing image quality. Leveraging the Smartphone Image Denoising Dataset (SIDD) for training and evaluation contributed to the robustness and generalization of the denoising model.

Moving forward, future image denoising projects could explore advancements in architecture design to further improve denoising performance. Additionally, incorporating techniques for handling different types of noise and enhancing computational efficiency could lead to more efficient and accurate denoising solutions. Overall, this project underscores the potential of GANs in addressing real-world challenges in image processing and computer vision, paving the way for continued advancements in image denoising technology.

---
