# Studi Komparatif Model Generatif: AE, Sparse AE, VAE, GAN, DCGAN, DDPM

Tugas Take-Home Deep Learning Sesi 07 (BINUS). Dataset: Fashion-MNIST (split 80/10/10, seed 42).

## Struktur
```
src/
  config.py        hyperparameter
  data.py          dataloader + split 80/10/10
  models/          mlp.py, autoencoders.py, gans.py, ddpm.py
  train.py         training semua model (checkpoint per epoch, bisa resume)
  experiments.py   eksperimen batasan MLP Classifier (Tahap 2-5) + interpolasi laten
  evaluate.py      MSE, PSNR, SSIM, FID, latency, training time
  figures.py       Figur 1-5
  run.py           menjalankan seluruh pipeline
notebook/DL_Tugas_Sesi07.ipynb
```

## Menjalankan
```bash
pip install -r requirements.txt
python -m src.run --part A   # MLP + frozen decoder, Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN
python -m src.run --part B   # DDPM (training + sampling)
python -m src.run --part C   # evaluasi + 5 figur
```
Uji cepat pipeline: `QUICK_TEST=1 python -m src.run --part all` (hasil di `outputs_test/`).

Hasil: `outputs/results/` (tabel CSV/Markdown), `outputs/figures/` (Figur 1-5), `outputs/checkpoints/`.
Semua eksperimen dijalankan di Google Colab (GPU Tesla T4).
