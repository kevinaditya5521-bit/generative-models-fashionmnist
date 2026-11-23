| Metode | Test MSE | Test PSNR (dB) | Test SSIM | FID |
|---|---|---|---|---|
| Vanilla AE (z_AE, 64-d) [pembanding] | 0.0071 | 22.5500 | 0.8544 | N/A |
| Frozen Latent Decoder (z_MLP, 64-d) | 0.0215 | 17.4700 | 0.6756 | N/A |
| Latent Transpose Projection W^T (z_MLP) | 0.2429 | 6.2400 | 0.0869 | N/A |
| Prior sampling z~N(mu_z, diag(sigma_z^2)) -> Decoder | N/A | N/A | N/A | 152.5900 |
