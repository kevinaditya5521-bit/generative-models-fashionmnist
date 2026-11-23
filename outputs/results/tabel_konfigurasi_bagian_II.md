| Model | Arsitektur | Dim Laten | Aktivasi | Loss | Optimizer | Learning Rate | Epoch | Batch | Jumlah Parameter (GAN: Generator) |
|---|---|---|---|---|---|---|---|---|---|
| MLP Classifier | 784-256-64-10 (Dropout 0.2) | 64 | ReLU / Softmax | Cross-Entropy | Adam | 0.0010 | 20 | 128 | 218,058 |
| Frozen Latent Decoder | 64-256-784 (input z_MLP beku) | 64 | ReLU / Sigmoid | MSE | Adam | 0.0010 | 15 | 128 | 218,128 |
| Vanilla AE | Enc 784-256-64, Dec 64-256-784 | 64 | ReLU / Sigmoid | MSE | Adam | 0.0010 | 20 | 128 | 435,536 |
| Sparse AE | Enc 784-256-64, Dec 64-256-784 | 64 | ReLU / Sigmoid | MSE + 0.0001*||h||_1 | Adam | 0.0010 | 20 | 128 | 435,536 |
| VAE | Enc 784-256-(mu,logvar 64), Dec 64-256-784 | 64 | ReLU / Sigmoid | BCE + KL(q(z|x)||N(0,I)) | Adam | 0.0010 | 20 | 128 | 451,984 |
| Vanilla GAN | G: 100-256-512-1024-784; D: 784-512-256-1 (LeakyReLU 0.2) | 100 | LeakyReLU 0.2 / Tanh | BCE minimax (non-saturating G) | Adam b=(0.5, 0.999) | 0.0002 | 50 | 128 | 1,486,352 |
| DCGAN | G: ConvT 100-256(7x7)-128(14x14)-1(28x28) + BN; D: Conv(1-64-128, stride 2) + BN, LeakyReLU 0.2, Conv 7x7 -> 1 | 100 | ReLU / Tanh | BCE minimax (non-saturating G) | Adam b=(0.5, 0.999) | 0.0002 | 20 | 128 | 1,781,505 |
| DDPM | U-Net (ch 64-128, 28-14-7), T=1000, beta linear 0.0001-0.02, EMA 0.999 | 28x28 | SiLU, GroupNorm | MSE noise prediction (L_simple) | Adam | 0.0002 | 20 | 128 | 3,098,561 |
