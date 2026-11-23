"""Menjalankan seluruh pipeline dari terminal.

    python -m src.run --part A   # MLP + eksperimen batasan, Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN
    python -m src.run --part B   # DDPM (training + sampling)
    python -m src.run --part C   # evaluasi (tabel metrik) + 5 figur
    python -m src.run --part all
"""
import argparse

from . import data, evaluate, figures, train
from .utils import ensure_dirs, get_device


def part_a(loaders, device):
    clf, _ = train.train_mlp_classifier(loaders, device)
    train.train_frozen_decoder(clf, loaders, device)
    train.train_autoencoder(loaders, device, sparse=False)
    train.train_autoencoder(loaders, device, sparse=True)
    train.train_vae(loaders, device)
    train.train_gan(loaders, device, kind="vanilla_gan")
    train.train_gan(loaders, device, kind="dcgan")


def part_b(loaders, device):
    train.train_ddpm(loaders, device)
    train.generate_ddpm_samples(device)


def part_c(loaders, device):
    ctx = evaluate.run_evaluation(loaders, device)
    figures.make_all_figures(ctx)
    return ctx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["A", "B", "C", "all"], default="all")
    args = p.parse_args()
    ensure_dirs()
    device = get_device()
    print("Device:", device)
    loaders = data.get_loaders()
    if args.part in ("A", "all"):
        part_a(loaders, device)
    if args.part in ("B", "all"):
        part_b(loaders, device)
    if args.part in ("C", "all"):
        part_c(loaders, device)


if __name__ == "__main__":
    main()
