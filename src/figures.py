"""Lima figur wajib laporan (Bagian V). Disimpan PNG 300 dpi di outputs/figures."""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from . import config as C
from . import experiments as X
from .data import pick_one_per_class

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                     "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight"})


def _np(imgs, nrow):
    grid = make_grid(imgs.detach().cpu().clamp(0, 1), nrow=nrow, padding=2, pad_value=1.0)
    return grid.permute(1, 2, 0).numpy()


def _save(fig, name):
    path = os.path.join(C.FIG_DIR, name)
    fig.savefig(path)
    print(f"Tersimpan: {path}")
    plt.show()
    plt.close(fig)


def _row_figure(rows, title, ncols, fname, col_labels=None):
    """Figur bertingkat: setiap baris = satu strip citra dengan label di sebelah kiri."""
    fig, axes = plt.subplots(len(rows), 1, figsize=(1.0 * ncols + 2.6, 1.05 * len(rows) + 0.8))
    axes = np.atleast_1d(axes)
    for ax, (label, imgs) in zip(axes, rows):
        ax.imshow(_np(imgs, ncols), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    if col_labels is not None:
        w = 28 + 2
        axes[0].set_xticks([2 + w * i + 14 for i in range(ncols)])
        axes[0].set_xticklabels(col_labels, fontsize=7.5)
        axes[0].xaxis.set_ticks_position("top")
        axes[0].tick_params(length=0)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, fname)


# ------------------------------------------------------------------ Figur 1
@torch.no_grad()
def figure1_reconstruction(ctx):
    m, dev = ctx["models"], ctx["device"]
    x, _ = pick_one_per_class(8)
    x = x.to(dev)
    vae = m["vae"]
    rows = [("Ground Truth", x),
            ("Vanilla AE", m["vanilla_ae"](x)[0]),
            ("Sparse AE", m["sparse_ae"](x)[0]),
            ("VAE", vae.decode(vae.encode(x)[0]))]
    _row_figure(rows, "Figur 1. Rekonstruksi Side-by-Side pada Test Set", 8, "figur1_rekonstruksi.png",
                col_labels=C.CLASS_NAMES[:8])


# ------------------------------------------------------------------ Figur 2
def figure2_sampling(ctx):
    s = ctx["samples"]
    panels = [("A. VAE  z~N(0,I)", s["vae"]), ("B. Vanilla GAN  z~N(0,I)", s["vanilla_gan"]),
              ("C. DCGAN  z~N(0,I)", s["dcgan"]), ("D. DDPM  x_T~N(0,I)", s["ddpm"])]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (t, imgs) in zip(axes, panels):
        ax.imshow(_np(imgs[:16], 4), cmap="gray", vmin=0, vmax=1)
        ax.set_title(t); ax.axis("off")
    fig.suptitle("Figur 2. Random Generative Sampling (kisi 4x4 per model)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "figur2_random_sampling.png")


# ------------------------------------------------------------------ Figur 3
@torch.no_grad()
def figure3_mlp_limitations(ctx):
    m, dev = ctx["models"], ctx["device"]
    clf, dec, ae = m["mlp"], m["frozen_decoder"], m["vanilla_ae"]
    x, _ = pick_one_per_class(10)
    x = x.to(dev)
    z = clf.features(x)
    top = X.topdown_from_class(clf)
    lo = top.flatten(1).min(1)[0].view(-1, 1, 1, 1)
    hi = top.flatten(1).max(1)[0].view(-1, 1, 1, 1)
    top_norm = (top - lo) / (hi - lo + 1e-8)
    rnd = X.mlp_prior_sampling(dec, ctx["mu_z"], ctx["sd_z"], 10, dev, seed=C.SEED + 7)
    rows = [("(1) Ground Truth", x),
            ("(2a) W^T dari one-hot y\n(skala asli)", top),
            ("(2b) W^T dari one-hot y\n(min-max, visibilitas)", top_norm),
            ("(2c) W^T dari z_MLP\n(latent transpose)", X.latent_transpose(clf, z)),
            ("(3a) Frozen Decoder\ndari z_MLP", dec(z)),
            ("(3b) Vanilla AE\ndari z_AE (pembanding)", ae(x)[0]),
            ("(4) Prior sampling\nz~N(mu_z, diag(sigma_z^2))", rnd)]
    _row_figure(rows, "Figur 3. Bukti Empiris Batasan Bobot & Latent Features MLP Classifier", 10,
                "figur3_batasan_mlp.png", col_labels=C.CLASS_NAMES)


# ------------------------------------------------------------------ Figur 4
@torch.no_grad()
def figure4_interpolation(ctx):
    m, dev = ctx["models"], ctx["device"]
    x, _ = pick_one_per_class(10)
    xa = x[C.INTERP_CLASS_A:C.INTERP_CLASS_A + 1].to(dev)
    xb = x[C.INTERP_CLASS_B:C.INTERP_CLASS_B + 1].to(dev)
    rows = [("Vanilla AE", X.interpolate_autoencoder(m["vanilla_ae"], xa, xb)),
            ("Sparse AE", X.interpolate_autoencoder(m["sparse_ae"], xa, xb)),
            ("VAE", X.interpolate_autoencoder(m["vae"], xa, xb))]
    for key, label in [("vanilla_gan", "Vanilla GAN"), ("dcgan", "DCGAN")]:
        za, zb, pa, pb = X.find_gan_endpoints(m[key], m["mlp"], dev)
        print(f"{label}: endpoint z_A P({C.CLASS_NAMES[C.INTERP_CLASS_A]})={pa:.3f}, "
              f"z_B P({C.CLASS_NAMES[C.INTERP_CLASS_B]})={pb:.3f}")
        rows.append((label, X.interpolate_gan(m[key], za, zb)))
    _row_figure(rows, f"Figur 4. Linear Latent Interpolation z(a)=(1-a)z_A + a z_B "
                      f"({C.CLASS_NAMES[C.INTERP_CLASS_A]} -> {C.CLASS_NAMES[C.INTERP_CLASS_B]})",
                len(C.INTERP_ALPHAS), "figur4_interpolasi.png",
                col_labels=[f"a={a:.1f}" for a in C.INTERP_ALPHAS])


# ------------------------------------------------------------------ Figur 5
def _smooth(v, k=50):
    v = np.asarray(v, dtype=float)
    if len(v) < k:
        return v
    return np.convolve(v, np.ones(k) / k, mode="valid")


def figure5_losses_and_trajectory(ctx):
    L = ctx["logs"]
    fig = plt.figure(figsize=(15, 11.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.0], hspace=0.45, wspace=0.28)

    def ep(v):
        return np.arange(1, len(v) + 1)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ep(L["vanilla_ae"]["train_mse"]), L["vanilla_ae"]["train_mse"], marker="o", ms=3, label="train MSE")
    ax.plot(ep(L["vanilla_ae"]["val_mse"]), L["vanilla_ae"]["val_mse"], "--", marker="o", ms=3, label="val MSE")
    ax.set_title("(a) Vanilla AE"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ep(L["sparse_ae"]["train_loss"]), L["sparse_ae"]["train_loss"], marker="o", ms=3, label="total (MSE + lambda L1)")
    ax.plot(ep(L["sparse_ae"]["train_mse"]), L["sparse_ae"]["train_mse"], marker="o", ms=3, label="train MSE")
    ax.plot(ep(L["sparse_ae"]["val_mse"]), L["sparse_ae"]["val_mse"], "--", marker="o", ms=3, label="val MSE")
    ax.set_title("(b) Sparse AE"); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(ep(L["vae"]["train_loss"]), L["vae"]["train_loss"], marker="o", ms=3, label="-ELBO train")
    ax.plot(ep(L["vae"]["val_loss"]), L["vae"]["val_loss"], "--", marker="o", ms=3, label="-ELBO val")
    ax.plot(ep(L["vae"]["train_bce"]), L["vae"]["train_bce"], marker="o", ms=3, label="BCE")
    ax2 = ax.twinx()
    ax2.plot(ep(L["vae"]["train_kld"]), L["vae"]["train_kld"], color="tab:red", marker="o", ms=3, label="KL (sumbu kanan)")
    ax2.set_ylabel("KL")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    ax.set_title("(c) VAE"); ax.set_xlabel("epoch"); ax.set_ylabel("loss / citra")

    for col, (key, label) in enumerate([("vanilla_gan", "(d) Vanilla GAN"), ("dcgan", "(e) DCGAN")]):
        ax = fig.add_subplot(gs[1, col])
        it_per_ep = len(L[key]["iter_lossD"]) / max(len(L[key]["lossD"]), 1)
        for k_it, name, col in [("iter_lossD", "D", "tab:orange"), ("iter_lossG", "G", "tab:blue")]:
            v = L[key][k_it]
            ax.plot(np.arange(len(v)) / it_per_ep, v, color=col, alpha=0.15, lw=0.6)
            sm = _smooth(v)
            ax.plot(np.arange(len(sm)) / it_per_ep, sm, color=col, label=f"loss {name} (moving avg)")
        ax.set_title(label); ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss"); ax.legend()

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(ep(L["ddpm"]["train_loss"]), L["ddpm"]["train_loss"], marker="o", ms=3, label="train")
    ax.plot(ep(L["ddpm"]["val_loss"]), L["ddpm"]["val_loss"], "--", marker="o", ms=3, label="val")
    ax.set_title("(f) DDPM noise-prediction loss"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE(eps)")
    ax.legend()

    traj = torch.load(os.path.join(C.SAMPLE_DIR, "ddpm_trajectory.pt"))
    steps = [s for s in C.DDPM_TRAJ_STEPS if s in traj]
    n = traj[steps[0]].size(0)
    n = min(n, 4)
    imgs = torch.stack([((traj[s][:n] + 1) / 2).clamp(0, 1) for s in steps], dim=1).flatten(0, 1)  # sampel x step
    ax = fig.add_subplot(gs[2, :])
    ax.imshow(_np(imgs, len(steps)), cmap="gray", vmin=0, vmax=1)
    w = 30
    ax.set_xticks([2 + w * i + 14 for i in range(len(steps))])
    ax.set_xticklabels([f"t={s}" for s in steps], fontsize=9)
    ax.set_yticks([]); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"(g) DDPM reverse diffusion trajectory x_T -> x_0 (T={C.DDPM_T}, {n} sampel)")
    fig.suptitle("Figur 5. Training Loss Curves dan DDPM Denoising Trajectory", fontsize=13, fontweight="bold")
    _save(fig, "figur5_loss_dan_trajektori_ddpm.png")


def make_all_figures(ctx):
    os.makedirs(C.FIG_DIR, exist_ok=True)
    figure1_reconstruction(ctx)
    figure2_sampling(ctx)
    figure3_mlp_limitations(ctx)
    figure4_interpolation(ctx)
    figure5_losses_and_trajectory(ctx)
