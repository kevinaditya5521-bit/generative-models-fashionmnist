"""Evaluasi terstandarisasi pada Test Set: MSE, PSNR, SSIM, FID, latency, training time, stabilitas."""
import os

import numpy as np
import pandas as pd
import torch
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
from torchmetrics.image.fid import FrechetInceptionDistance

from . import config as C
from . import experiments as X
from .data import get_test_tensors
from .models import Diffusion
from .train import load_model, accuracy, generate_ddpm_samples
from .utils import Timer, save_json, to_01, count_params, set_seed


# ------------------------------------------------------------------ metrik rekonstruksi
@torch.no_grad()
def recon_metrics(fn, loader, device):
    """MSE & PSNR dihitung per citra lalu dirata-rata; SSIM (Gaussian 11x11) dirata-rata. Rentang [0,1]."""
    s_mse = s_psnr = s_ssim = n = 0.0
    for x, _ in loader:
        x = x.to(device)
        xh = fn(x).clamp(0, 1)
        mse = ((xh - x) ** 2).flatten(1).mean(1)
        s_mse += mse.sum().item()
        s_psnr += (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).sum().item()
        s_ssim += ssim_fn(xh, x, data_range=1.0).item() * x.size(0)
        n += x.size(0)
    return {"mse": s_mse / n, "psnr": s_psnr / n, "ssim": s_ssim / n}


@torch.no_grad()
def sparsity_stats(model, loader, device):
    """Statistik aktivasi hidden h (256 unit): proporsi unit aktif dan rata-rata ||h||_1."""
    act = l1 = n = 0.0
    for x, _ in loader:
        _, h = model.encode(x.to(device))
        act += (h > 1e-6).float().mean(1).sum().item()
        l1 += h.abs().sum(1).sum().item()
        n += x.size(0)
    return {"active_ratio": act / n, "mean_l1": l1 / n}


# ------------------------------------------------------------------ FID
class FIDEvaluator:
    """FID (Inception-v3, pool 2048). Citra grayscale 28x28 diulang 3 kanal; resize 299 dilakukan internal."""

    def __init__(self, real, device):
        self.device = device
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device)
        for chunk in real.split(C.FID_BATCH):
            self.fid.update(chunk.repeat(1, 3, 1, 1).to(device), real=True)

    @torch.no_grad()
    def score(self, fake):
        self.fid.reset()                              # hanya statistik fake yang di-reset
        for chunk in fake.split(C.FID_BATCH):
            self.fid.update(chunk.clamp(0, 1).repeat(1, 3, 1, 1).to(self.device), real=False)
        return float(self.fid.compute())


@torch.no_grad()
def generate(kind, model, n, device, seed=C.SEED):
    """Sampling acak dari prior z ~ N(0, I). Output di [0,1], disimpan di CPU."""
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for i in range(0, n, 500):
        b = min(500, n - i)
        if kind == "vae":
            out.append(model.decode(torch.randn(b, C.LATENT_DIM, device=device, generator=g)).cpu())
        else:
            out.append(to_01(model(torch.randn(b, C.NOISE_DIM, device=device, generator=g))).cpu())
    return torch.cat(out)


# ------------------------------------------------------------------ latency
@torch.no_grad()
def measure_latency(fn, device, runs, warmup_fn=None, warmup=3):
    """Latency (ms) per batch 64 sampel, dengan warm-up dan cuda.synchronize."""
    for _ in range(warmup):
        (warmup_fn or fn)()
    times = []
    for _ in range(runs):
        with Timer(device) as t:
            fn()
        times.append(t.elapsed * 1000)
    return float(np.mean(times)), float(np.std(times))


def _stability(values, k=5):
    """Koefisien variasi (std/mean) loss pada k epoch terakhir: makin kecil makin stabil."""
    v = np.array(values[-k:], dtype=float)
    return float(np.std(v) / (abs(np.mean(v)) + 1e-12))


def _fmt(v, nd=4):
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}f}"
    return str(v)


def df_to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _save_table(df, stem):
    df.to_csv(os.path.join(C.RESULT_DIR, f"{stem}.csv"), index=False)
    with open(os.path.join(C.RESULT_DIR, f"{stem}.md"), "w") as f:
        f.write(df_to_markdown(df) + "\n")


# ------------------------------------------------------------------ evaluasi utama
def run_evaluation(loaders, device):
    set_seed()
    os.makedirs(C.RESULT_DIR, exist_ok=True)
    names = ["mlp", "frozen_decoder", "vanilla_ae", "sparse_ae", "vae", "vanilla_gan", "dcgan", "ddpm"]
    models, logs = {}, {}
    for n in names:
        models[n], logs[n] = load_model(n, device)
    clf, dec = models["mlp"], models["frozen_decoder"]
    ctx = {"models": models, "logs": logs, "device": device}

    # ---------------- (a) rekonstruksi pada Test Set
    print("Menghitung MSE / PSNR / SSIM pada Test Set ...")
    rec = {
        "vanilla_ae": recon_metrics(lambda x: models["vanilla_ae"](x)[0], loaders["test"], device),
        "sparse_ae": recon_metrics(lambda x: models["sparse_ae"](x)[0], loaders["test"], device),
        "vae": recon_metrics(lambda x: models["vae"].decode(models["vae"].encode(x)[0]), loaders["test"], device),
        "frozen_decoder": recon_metrics(lambda x: dec(clf.features(x)), loaders["test"], device),
        "latent_transpose": recon_metrics(lambda x: X.latent_transpose(clf, clf.features(x)), loaders["test"], device),
    }
    sparsity = {k: sparsity_stats(models[k], loaders["test"], device) for k in ["vanilla_ae", "sparse_ae"]}

    # ---------------- (b) sampel generatif untuk FID
    print(f"Membuat {C.N_FID} sampel per model generatif ...")
    x_test, _ = get_test_tensors()
    g = torch.Generator().manual_seed(C.SEED)
    real = x_test[torch.randperm(len(x_test), generator=g)[:C.N_FID]]
    samples = {
        "vae": generate("vae", models["vae"], C.N_FID, device),
        "vanilla_gan": generate("gan", models["vanilla_gan"], C.N_FID, device),
        "dcgan": generate("gan", models["dcgan"], C.N_FID, device),
        "ddpm": generate_ddpm_samples(device),
    }
    mu_z, sd_z, _ = X.latent_stats(clf, loaders["test"], device)
    samples["mlp_prior"] = X.mlp_prior_sampling(dec, mu_z, sd_z, C.N_FID, device).cpu()
    ctx.update(samples=samples, mu_z=mu_z, sd_z=sd_z)

    print("Menghitung FID (Inception-v3) ...")
    fid_eval = FIDEvaluator(real, device)
    fid = {k: fid_eval.score(v) for k, v in samples.items()}
    for k, v in fid.items():
        print(f"  FID {k:<12}: {v:.2f}")

    # ---------------- (c) latency per batch 64 (fp32, GPU yang sama untuk semua model)
    print("Mengukur inference latency (batch 64) ...")
    B = C.LATENCY_BATCH
    x64 = x_test[:B].to(device)
    zl = torch.randn(B, C.LATENT_DIM, device=device)
    zn = torch.randn(B, C.NOISE_DIM, device=device)
    R = C.LATENCY_RUNS
    lat = {
        "vanilla_ae": measure_latency(lambda: models["vanilla_ae"](x64), device, R),
        "sparse_ae": measure_latency(lambda: models["sparse_ae"](x64), device, R),
        "vae": measure_latency(lambda: models["vae"].decode(zl), device, R),
        "vanilla_gan": measure_latency(lambda: models["vanilla_gan"](zn), device, R),
        "dcgan": measure_latency(lambda: models["dcgan"](zn), device, R),
    }
    diff = Diffusion(device=device)
    t0 = torch.zeros(B, dtype=torch.long, device=device)
    xT = torch.randn(B, *C.IMG_SHAPE, device=device)
    lat["ddpm"] = measure_latency(lambda: diff.sample(models["ddpm"], B, use_amp=False), device,
                                  C.LATENCY_RUNS_DDPM, warmup_fn=lambda: models["ddpm"](xT, t0))
    for k, (m, s) in lat.items():
        print(f"  latency {k:<12}: {m:10.2f} +- {s:.2f} ms / batch {B}")

    # ---------------- (d) training time & stabilitas
    def t_epoch(k):
        return float(np.mean(logs[k]["epoch_time"]))

    main_loss = {"vanilla_ae": "train_mse", "sparse_ae": "train_loss", "vae": "train_loss",
                 "vanilla_gan": "lossG", "dcgan": "lossG", "ddpm": "train_loss"}
    stab = {k: _stability(logs[k][v]) for k, v in main_loss.items()}
    stab_d = {k: _stability(logs[k]["lossD"]) for k in ["vanilla_gan", "dcgan"]}

    # ---------------- Tabel Bagian IV
    rows = [
        ("Vanilla Autoencoder", "vanilla_ae", f"{C.LATENT_DIM}"),
        ("Sparse Autoencoder", "sparse_ae", f"{C.LATENT_DIM} (Bottleneck) / {C.HIDDEN_DIM} (Sparse)"),
        ("Variational Autoencoder (VAE)", "vae", f"{C.LATENT_DIM}"),
        ("Vanilla GAN (MLP)", "vanilla_gan", f"{C.NOISE_DIM} (Prior Noise)"),
        ("DCGAN", "dcgan", f"{C.NOISE_DIM} (Prior Noise)"),
        ("DDPM", "ddpm", "28x28 (Dimensi Citra)"),
    ]
    table = []
    for label, k, zdim in rows:
        r = rec.get(k)
        stab_txt = f"CV loss(G)={stab[k]:.3f}; CV loss(D)={stab_d[k]:.3f}" if k in stab_d else f"CV loss={stab[k]:.4f}"
        table.append({
            "Arsitektur Model": label,
            "Dim Laten": zdim,
            "Training Time / Epoch (s)": round(t_epoch(k), 2),
            "Latency (ms / batch 64)": round(lat[k][0], 2),
            "Test MSE": round(r["mse"], 5) if r else "N/A",
            "Test PSNR (dB)": round(r["psnr"], 2) if r else "N/A",
            "Test SSIM": round(r["ssim"], 4) if r else "N/A",
            "FID Score": round(fid[k], 2) if k in fid else "N/A (Latent Voids)",
            "Stabilitas (5 epoch terakhir)": stab_txt,
        })
    df_main = pd.DataFrame(table)
    _save_table(df_main, "tabel_metrik_bagian_IV")

    # ---------------- Tabel Bagian III (MLP Classifier)
    test_acc = accuracy(clf, loaders["test"], device)
    ortho = X.orthogonality_check(clf)
    df_mlp = pd.DataFrame([
        {"Metode": "Vanilla AE (z_AE, 64-d) [pembanding]", **_rec_cols(rec["vanilla_ae"]), "FID": "N/A"},
        {"Metode": "Frozen Latent Decoder (z_MLP, 64-d)", **_rec_cols(rec["frozen_decoder"]), "FID": "N/A"},
        {"Metode": "Latent Transpose Projection W^T (z_MLP)", **_rec_cols(rec["latent_transpose"]), "FID": "N/A"},
        {"Metode": "Prior sampling z~N(mu_z, diag(sigma_z^2)) -> Decoder", "Test MSE": "N/A",
         "Test PSNR (dB)": "N/A", "Test SSIM": "N/A", "FID": round(fid["mlp_prior"], 2)},
    ])
    _save_table(df_mlp, "tabel_mlp_bagian_III")
    df_ortho = pd.DataFrame([{"Matriks": k, "Shape": "x".join(map(str, v["shape"])), "Rank": v["rank"],
                              "||W^T W - I||_F": round(v["fro_norm_WtW_minus_I"], 3),
                              "Relatif thd ||I||_F": round(v["rel_fro_norm"], 3)} for k, v in ortho.items()])
    _save_table(df_ortho, "tabel_ortogonalitas_bobot")
    df_sp = pd.DataFrame([{"Model": k, "Proporsi unit hidden aktif": round(v["active_ratio"], 4),
                           "Rata-rata ||h||_1": round(v["mean_l1"], 2)} for k, v in sparsity.items()])
    _save_table(df_sp, "tabel_sparsity")

    # ---------------- Tabel konfigurasi (Bagian II)
    df_cfg = hyperparam_table(models)
    _save_table(df_cfg, "tabel_konfigurasi_bagian_II")

    results = {"mlp_test_acc": test_acc, "recon": rec, "fid": fid, "latency_ms": lat,
               "train_time_per_epoch_s": {k: t_epoch(k) for k in names}, "stability_cv": stab,
               "stability_cv_D": stab_d, "sparsity": sparsity, "orthogonality": ortho,
               "n_fid": C.N_FID, "epochs": C.EPOCHS, "device": torch.cuda.get_device_name(0)
               if device.type == "cuda" else "cpu"}
    save_json(results, os.path.join(C.RESULT_DIR, "all_results.json"))
    print(f"\nMLP test accuracy: {test_acc:.4f} (syarat >= {C.MLP_MIN_ACC:.0%})")
    print(f"Semua tabel tersimpan di: {C.RESULT_DIR}")
    ctx.update(results=results, tables={"main": df_main, "mlp": df_mlp, "ortho": df_ortho,
                                        "sparsity": df_sp, "config": df_cfg})
    return ctx


def _rec_cols(r):
    return {"Test MSE": round(r["mse"], 5), "Test PSNR (dB)": round(r["psnr"], 2), "Test SSIM": round(r["ssim"], 4)}


def hyperparam_table(models):
    gan_d = {"vanilla_gan": "D: 784-512-256-1 (LeakyReLU 0.2)",
             "dcgan": "D: Conv(1-64-128, stride 2) + BN, LeakyReLU 0.2, Conv 7x7 -> 1"}
    rows = [
        ("MLP Classifier", "mlp", "784-256-64-10 (Dropout 0.2)", C.LATENT_DIM, "ReLU / Softmax", "Cross-Entropy",
         "Adam", C.LR["mlp"]),
        ("Frozen Latent Decoder", "frozen_decoder", "64-256-784 (input z_MLP beku)", C.LATENT_DIM, "ReLU / Sigmoid",
         "MSE", "Adam", C.LR["frozen_decoder"]),
        ("Vanilla AE", "vanilla_ae", "Enc 784-256-64, Dec 64-256-784", C.LATENT_DIM, "ReLU / Sigmoid", "MSE",
         "Adam", C.LR["vanilla_ae"]),
        ("Sparse AE", "sparse_ae", "Enc 784-256-64, Dec 64-256-784", C.LATENT_DIM, "ReLU / Sigmoid",
         f"MSE + {C.SPARSE_LAMBDA:g}*||h||_1", "Adam", C.LR["sparse_ae"]),
        ("VAE", "vae", "Enc 784-256-(mu,logvar 64), Dec 64-256-784", C.LATENT_DIM, "ReLU / Sigmoid",
         "BCE + KL(q(z|x)||N(0,I))", "Adam", C.LR["vae"]),
        ("Vanilla GAN", "vanilla_gan", "G: 100-256-512-1024-784; " + gan_d["vanilla_gan"], C.NOISE_DIM,
         "LeakyReLU 0.2 / Tanh", "BCE minimax (non-saturating G)", f"Adam b={C.GAN_BETAS}", C.LR["vanilla_gan"]),
        ("DCGAN", "dcgan", "G: ConvT 100-256(7x7)-128(14x14)-1(28x28) + BN; " + gan_d["dcgan"], C.NOISE_DIM,
         "ReLU / Tanh", "BCE minimax (non-saturating G)", f"Adam b={C.GAN_BETAS}", C.LR["dcgan"]),
        ("DDPM", "ddpm", f"U-Net (ch {C.DDPM_BASE_CH}-{2 * C.DDPM_BASE_CH}, 28-14-7), T={C.DDPM_T}, "
         f"beta linear {C.DDPM_BETA_START}-{C.DDPM_BETA_END}, EMA {C.EMA_DECAY}", "28x28", "SiLU, GroupNorm",
         "MSE noise prediction (L_simple)", "Adam", C.LR["ddpm"]),
    ]
    return pd.DataFrame([{"Model": a, "Arsitektur": b, "Dim Laten": d, "Aktivasi": e, "Loss": f,
                          "Optimizer": g, "Learning Rate": lr, "Epoch": C.EPOCHS[k], "Batch": C.BATCH_SIZE,
                          "Jumlah Parameter (GAN: Generator)": f"{count_params(models[k]):,}"}
                         for a, k, b, d, e, f, g, lr in rows])
