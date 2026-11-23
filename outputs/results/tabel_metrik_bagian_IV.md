| Arsitektur Model | Dim Laten | Training Time / Epoch (s) | Latency (ms / batch 64) | Test MSE | Test PSNR (dB) | Test SSIM | FID Score | Stabilitas (5 epoch terakhir) |
|---|---|---|---|---|---|---|---|---|
| Vanilla Autoencoder | 64 | 1.4300 | 0.2300 | 0.0071 | 22.5500 | 0.8544 | N/A (Latent Voids) | CV loss=0.0247 |
| Sparse Autoencoder | 64 (Bottleneck) / 256 (Sparse) | 1.4900 | 0.2000 | 0.0115 | 20.3300 | 0.7811 | N/A (Latent Voids) | CV loss=0.0183 |
| Variational Autoencoder (VAE) | 64 | 1.7300 | 0.1200 | 0.0134 | 19.5600 | 0.7635 | 97.1800 | CV loss=0.0010 |
| Vanilla GAN (MLP) | 100 (Prior Noise) | 2.9300 | 0.2400 | N/A | N/A | N/A | 128.5900 | CV loss(G)=0.003; CV loss(D)=0.001 |
| DCGAN | 100 (Prior Noise) | 17.1000 | 8.2100 | N/A | N/A | N/A | 30.7800 | CV loss(G)=0.007; CV loss(D)=0.009 |
| DDPM | 28x28 (Dimensi Citra) | 46.7800 | 30328.3900 | N/A | N/A | N/A | 19.6800 | CV loss=0.0103 |
