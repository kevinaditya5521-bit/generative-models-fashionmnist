"""Training loop untuk 7 model. Setiap epoch disimpan ke checkpoint sehingga bisa di-resume.

Bila checkpoint sudah mencapai jumlah epoch target, training dilewati (tidak mengulang dari nol).
"""
import os

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from . import config as C
from .models import (MLPClassifier, MLPDecoder, Autoencoder, VAE, vae_loss, MLPGenerator,
                     MLPDiscriminator, DCGANGenerator, DCGANDiscriminator, dcgan_weights_init,
                     UNet, Diffusion, EMA)
from .utils import Timer, set_seed, save_ckpt, load_ckpt, save_json, to_pm1, to_01, count_params


# ------------------------------------------------------------------ helper resume / log
def _resume(name, modules, optims, device):
    ck = load_ckpt(name, device)
    if ck is None:
        return 0, {}, {}
    for k, m in modules.items():
        m.load_state_dict(ck["models"][k])
    for k, o in optims.items():
        o.load_state_dict(ck["optims"][k])
    print(f"[{name}] checkpoint ditemukan -> lanjut dari epoch {ck['epoch']}")
    return ck["epoch"], ck["logs"], ck.get("extra", {})


def _save(name, epoch, modules, optims, logs, extra=None):
    save_ckpt(name, {
        "epoch": epoch,
        "models": {k: m.state_dict() for k, m in modules.items()},
        "optims": {k: o.state_dict() for k, o in optims.items()},
        "logs": logs,
        "extra": extra or {},
    })
    save_json(logs, os.path.join(C.LOG_DIR, f"{name}.json"))


def _log(logs, **kv):
    for k, v in kv.items():
        logs.setdefault(k, []).append(float(v))


def _header(name, model, epochs, start):
    print(f"\n===== {name} | params: {count_params(model):,} | epoch target: {epochs} | mulai: {start} =====")
    if start >= epochs:
        print(f"[{name}] sudah selesai dilatih -> dilewati.")


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


# ------------------------------------------------------------------ 1. MLP Classifier
def train_mlp_classifier(loaders, device):
    name = "mlp"
    set_seed()
    model = MLPClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR[name])
    start, logs, extra = _resume(name, {"model": model}, {"opt": opt}, device)
    best_state, best_val = extra.get("best_state"), extra.get("best_val", -1.0)
    E = C.EPOCHS[name]
    _header(name, model, E, start)
    for ep in range(start, E):
        model.train()
        tot = n = 0
        with Timer(device) as tm:
            for x, y in loaders["train"]:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                loss = F.cross_entropy(model(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item() * x.size(0)
                n += x.size(0)
        val_acc = accuracy(model, loaders["val"], device)
        _log(logs, epoch_time=tm.elapsed, train_loss=tot / n, val_acc=val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        _save(name, ep + 1, {"model": model}, {"opt": opt}, logs, {"best_state": best_state, "best_val": best_val})
        print(f"[{name}] epoch {ep + 1:>3}/{E} | CE loss {tot / n:.4f} | val acc {val_acc:.4f} | {tm.elapsed:.1f}s")
    model.load_state_dict(best_state)
    model.eval()
    test_acc = accuracy(model, loaders["test"], device)
    status = "MEMENUHI" if test_acc >= C.MLP_MIN_ACC else "BELUM MEMENUHI (tambah epoch!)"
    print(f"[{name}] Test accuracy (bobot val terbaik): {test_acc:.4f} -> syarat >= {C.MLP_MIN_ACC:.0%}: {status}")
    logs["test_acc"] = test_acc
    save_json(logs, os.path.join(C.LOG_DIR, f"{name}.json"))
    return model, logs


# ------------------------------------------------------------------ 2 & 3. Vanilla AE / Sparse AE
def train_autoencoder(loaders, device, sparse=False):
    name = "sparse_ae" if sparse else "vanilla_ae"
    lam = C.SPARSE_LAMBDA if sparse else 0.0
    set_seed()
    model = Autoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR[name])
    start, logs, _ = _resume(name, {"model": model}, {"opt": opt}, device)
    E = C.EPOCHS[name]
    _header(name, model, E, start)
    for ep in range(start, E):
        model.train()
        s_loss = s_mse = s_l1 = n = 0
        with Timer(device) as tm:
            for x, _ in loaders["train"]:
                x = x.to(device, non_blocking=True)
                x_hat, z, h = model(x)
                mse = F.mse_loss(x_hat, x)
                l1 = h.abs().sum(1).mean()            # ||h||_1 rata-rata per citra
                loss = mse + lam * l1
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                b = x.size(0)
                s_loss += loss.item() * b; s_mse += mse.item() * b; s_l1 += l1.item() * b; n += b
        val_mse = _val_recon(lambda v: model(v)[0], loaders["val"], device, model)
        _log(logs, epoch_time=tm.elapsed, train_loss=s_loss / n, train_mse=s_mse / n,
             train_l1=s_l1 / n, val_mse=val_mse)
        _save(name, ep + 1, {"model": model}, {"opt": opt}, logs)
        print(f"[{name}] epoch {ep + 1:>3}/{E} | loss {s_loss / n:.5f} | MSE {s_mse / n:.5f} | "
              f"L1(h) {s_l1 / n:.2f} | val MSE {val_mse:.5f} | {tm.elapsed:.1f}s")
    model.eval()
    return model, logs


@torch.no_grad()
def _val_recon(fn, loader, device, model):
    model.eval()
    tot = n = 0
    for x, _ in loader:
        x = x.to(device)
        tot += F.mse_loss(fn(x), x, reduction="sum").item() / x[0].numel()
        n += x.size(0)
    return tot / n


# ------------------------------------------------------------------ 4. VAE
def train_vae(loaders, device):
    name = "vae"
    set_seed()
    model = VAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR[name])
    start, logs, _ = _resume(name, {"model": model}, {"opt": opt}, device)
    E = C.EPOCHS[name]
    _header(name, model, E, start)
    for ep in range(start, E):
        model.train()
        s_loss = s_bce = s_kld = n = 0
        with Timer(device) as tm:
            for x, _ in loaders["train"]:
                x = x.to(device, non_blocking=True)
                x_hat, mu, logvar = model(x)
                loss, bce, kld = vae_loss(x_hat, x, mu, logvar)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                b = x.size(0)
                s_loss += loss.item() * b; s_bce += bce.item() * b; s_kld += kld.item() * b; n += b
        # validasi: loss ELBO + MSE rekonstruksi dari mu (deterministik)
        model.eval()
        v_loss = v_n = 0
        with torch.no_grad():
            for x, _ in loaders["val"]:
                x = x.to(device)
                xh, mu, lv = model(x)
                v_loss += vae_loss(xh, x, mu, lv)[0].item() * x.size(0)
                v_n += x.size(0)
        val_mse = _val_recon(lambda v: model.decode(model.encode(v)[0]), loaders["val"], device, model)
        _log(logs, epoch_time=tm.elapsed, train_loss=s_loss / n, train_bce=s_bce / n, train_kld=s_kld / n,
             val_loss=v_loss / v_n, val_mse=val_mse)
        _save(name, ep + 1, {"model": model}, {"opt": opt}, logs)
        print(f"[{name}] epoch {ep + 1:>3}/{E} | -ELBO {s_loss / n:.2f} (BCE {s_bce / n:.2f} + KL {s_kld / n:.2f}) "
              f"| val -ELBO {v_loss / v_n:.2f} | val MSE {val_mse:.5f} | {tm.elapsed:.1f}s")
    model.eval()
    return model, logs


# ------------------------------------------------------------------ 5. Frozen Latent Decoder (Tahap 4)
def train_frozen_decoder(clf, loaders, device):
    name = "frozen_decoder"
    set_seed()
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False                     # seluruh bobot MLP Classifier dibekukan
    dec = MLPDecoder().to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=C.LR[name])
    start, logs, _ = _resume(name, {"model": dec}, {"opt": opt}, device)
    E = C.EPOCHS[name]
    _header(name, dec, E, start)
    for ep in range(start, E):
        dec.train()
        tot = n = 0
        with Timer(device) as tm:
            for x, _ in loaders["train"]:
                x = x.to(device, non_blocking=True)
                with torch.no_grad():
                    z = clf.features(x)             # z_MLP beku
                loss = F.mse_loss(dec(z), x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item() * x.size(0); n += x.size(0)
        val_mse = _val_recon(lambda v: dec(clf.features(v)), loaders["val"], device, dec)
        _log(logs, epoch_time=tm.elapsed, train_loss=tot / n, val_mse=val_mse)
        _save(name, ep + 1, {"model": dec}, {"opt": opt}, logs)
        print(f"[{name}] epoch {ep + 1:>3}/{E} | MSE {tot / n:.5f} | val MSE {val_mse:.5f} | {tm.elapsed:.1f}s")
    dec.eval()
    return dec, logs


# ------------------------------------------------------------------ 6 & 7. Vanilla GAN / DCGAN
def train_gan(loaders, device, kind="vanilla_gan", snapshot_every=5):
    assert kind in ("vanilla_gan", "dcgan")
    name = kind
    set_seed()
    if kind == "vanilla_gan":
        G, D = MLPGenerator().to(device), MLPDiscriminator().to(device)
    else:
        G, D = DCGANGenerator().to(device), DCGANDiscriminator().to(device)
        G.apply(dcgan_weights_init)
        D.apply(dcgan_weights_init)
    optG = torch.optim.Adam(G.parameters(), lr=C.LR[name], betas=C.GAN_BETAS)
    optD = torch.optim.Adam(D.parameters(), lr=C.LR[name], betas=C.GAN_BETAS)
    start, logs, _ = _resume(name, {"G": G, "D": D}, {"optG": optG, "optD": optD}, device)
    E = C.EPOCHS[name]
    print(f"\n[{name}] params D: {count_params(D):,}")
    _header(name, G, E, start)
    g_fixed = torch.Generator(device=device).manual_seed(C.SEED)
    fixed_z = torch.randn(64, C.NOISE_DIM, device=device, generator=g_fixed)
    snap_dir = os.path.join(C.FIG_DIR, "training_progress")
    os.makedirs(snap_dir, exist_ok=True)
    bce = torch.nn.BCEWithLogitsLoss()
    for ep in range(start, E):
        G.train(); D.train()
        sD = sG = sDx = sDgz = n = 0
        itD, itG = [], []
        with Timer(device) as tm:
            for x, _ in loaders["train"]:
                real = to_pm1(x.to(device, non_blocking=True))
                b = real.size(0)
                ones, zeros = torch.ones(b, device=device), torch.zeros(b, device=device)
                fake = G(torch.randn(b, C.NOISE_DIM, device=device))
                # --- Discriminator: max log D(x) + log(1 - D(G(z)))
                d_real, d_fake = D(real), D(fake.detach())
                lossD = bce(d_real, ones) + bce(d_fake, zeros)
                optD.zero_grad(set_to_none=True)
                lossD.backward()
                optD.step()
                # --- Generator (non-saturating): max log D(G(z))
                lossG = bce(D(fake), ones)
                optG.zero_grad(set_to_none=True)
                lossG.backward()
                optG.step()
                itD.append(lossD.item()); itG.append(lossG.item())
                sD += lossD.item() * b; sG += lossG.item() * b
                sDx += torch.sigmoid(d_real).mean().item() * b
                sDgz += torch.sigmoid(d_fake).mean().item() * b
                n += b
        logs.setdefault("iter_lossD", []).extend(itD)
        logs.setdefault("iter_lossG", []).extend(itG)
        _log(logs, epoch_time=tm.elapsed, lossD=sD / n, lossG=sG / n, D_x=sDx / n, D_G_z=sDgz / n)
        _save(name, ep + 1, {"G": G, "D": D}, {"optG": optG, "optD": optD}, logs)
        if (ep + 1) % snapshot_every == 0 or ep + 1 == E:
            G.eval()
            with torch.no_grad():
                save_image(to_01(G(fixed_z)), os.path.join(snap_dir, f"{name}_epoch{ep + 1:03d}.png"), nrow=8)
        print(f"[{name}] epoch {ep + 1:>3}/{E} | loss D {sD / n:.4f} | loss G {sG / n:.4f} | "
              f"D(x) {sDx / n:.3f} | D(G(z)) {sDgz / n:.3f} | {tm.elapsed:.1f}s")
    G.eval()
    return G, logs


# ------------------------------------------------------------------ 8. DDPM
def train_ddpm(loaders, device):
    name = "ddpm"
    set_seed()
    model = UNet().to(device)
    ema = EMA(model)
    diff = Diffusion(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR[name])
    use_amp = C.USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    start, logs, extra = _resume(name, {"model": model, "ema": ema.shadow}, {"opt": opt}, device)
    if scaler is not None and extra.get("scaler"):
        scaler.load_state_dict(extra["scaler"])
    E = C.EPOCHS[name]
    _header(name, model, E, start)
    print(f"[{name}] T = {diff.T} | mixed precision (AMP): {use_amp}")
    for ep in range(start, E):
        model.train()
        tot = n = 0
        with Timer(device) as tm:
            for x, _ in loaders["train"]:
                x0 = to_pm1(x.to(device, non_blocking=True))
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    loss = diff.loss(model, x0)
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                ema.update(model)
                if not torch.isfinite(loss):
                    raise RuntimeError("Loss DDPM NaN/Inf - hentikan dan periksa learning rate.")
                tot += loss.item() * x0.size(0); n += x0.size(0)
        model.eval()
        v_tot = v_n = 0
        with torch.no_grad():
            for x, _ in loaders["val"]:
                x0 = to_pm1(x.to(device))
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    v_tot += diff.loss(model, x0).item() * x0.size(0)
                v_n += x0.size(0)
        _log(logs, epoch_time=tm.elapsed, train_loss=tot / n, val_loss=v_tot / v_n)
        _save(name, ep + 1, {"model": model, "ema": ema.shadow}, {"opt": opt}, logs,
              {"scaler": scaler.state_dict() if scaler is not None else None})
        sisa = (E - ep - 1) * tm.elapsed / 60
        print(f"[{name}] epoch {ep + 1:>3}/{E} | noise MSE {tot / n:.5f} | val {v_tot / v_n:.5f} | "
              f"{tm.elapsed:.1f}s | perkiraan sisa {sisa:.1f} menit")
    return ema.shadow, logs


@torch.no_grad()
def generate_ddpm_samples(device, n=None, chunk=500):
    """Sampling DDPM (model EMA) untuk FID + trajektori denoising. Disimpan per-chunk (bisa resume)."""
    n = n or C.N_FID
    out_path = os.path.join(C.SAMPLE_DIR, "ddpm_fid_samples.pt")
    traj_path = os.path.join(C.SAMPLE_DIR, "ddpm_trajectory.pt")
    os.makedirs(C.SAMPLE_DIR, exist_ok=True)
    model, _ = load_model("ddpm", device)
    diff = Diffusion(device=device)
    use_amp = C.USE_AMP and device.type == "cuda"
    if not os.path.exists(traj_path):
        g = torch.Generator(device=device).manual_seed(C.SEED)
        _, traj = diff.sample(model, 8, use_amp=use_amp, traj_steps=C.DDPM_TRAJ_STEPS, generator=g)
        torch.save(traj, traj_path)
        print("[ddpm] trajektori denoising tersimpan.")
    if os.path.exists(out_path):
        print(f"[ddpm] sampel FID sudah ada: {out_path}")
        return torch.load(out_path).float() / 255.0
    parts = []
    n_chunks = (n + chunk - 1) // chunk
    for i in range(n_chunks):
        p = os.path.join(C.SAMPLE_DIR, f"ddpm_part{i:02d}.pt")
        if not os.path.exists(p):
            b = min(chunk, n - i * chunk)
            g = torch.Generator(device=device).manual_seed(C.SEED + 1000 + i)
            with Timer(device) as tm:
                x = diff.sample(model, b, use_amp=use_amp, generator=g)
            torch.save((to_01(x) * 255).round().to(torch.uint8).cpu(), p)
            print(f"[ddpm] chunk {i + 1}/{n_chunks} ({b} sampel) selesai dalam {tm.elapsed:.1f}s")
        parts.append(torch.load(p))
    samples = torch.cat(parts)[:n]
    torch.save(samples, out_path)
    for i in range(n_chunks):
        os.remove(os.path.join(C.SAMPLE_DIR, f"ddpm_part{i:02d}.pt"))
    print(f"[ddpm] {len(samples)} sampel FID tersimpan -> {out_path}")
    return samples.float() / 255.0


# ------------------------------------------------------------------ memuat model terlatih
def load_model(name, device):
    ck = load_ckpt(name, device)
    if ck is None:
        raise FileNotFoundError(f"Checkpoint '{name}' belum ada. Jalankan training '{name}' dulu.")
    if ck["epoch"] < C.EPOCHS[name]:
        print(f"PERINGATAN: '{name}' baru {ck['epoch']}/{C.EPOCHS[name]} epoch.")
    if name == "mlp":
        model = MLPClassifier()
        state = ck["extra"]["best_state"]
    elif name == "frozen_decoder":
        model, state = MLPDecoder(), ck["models"]["model"]
    elif name in ("vanilla_ae", "sparse_ae"):
        model, state = Autoencoder(), ck["models"]["model"]
    elif name == "vae":
        model, state = VAE(), ck["models"]["model"]
    elif name == "vanilla_gan":
        model, state = MLPGenerator(), ck["models"]["G"]
    elif name == "dcgan":
        model, state = DCGANGenerator(), ck["models"]["G"]
    elif name == "ddpm":
        model, state = UNet(), ck["models"]["ema"]
    else:
        raise ValueError(name)
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck["logs"]
