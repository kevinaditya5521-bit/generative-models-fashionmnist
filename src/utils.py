"""Utilitas umum: seed, device, checkpoint, timer, konversi rentang piksel."""
import json
import os
import random
import time

import numpy as np
import torch

from . import config as C


def set_seed(seed=None):
    """Menetapkan random seed agar eksperimen reprodusibel."""
    seed = C.SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs():
    for d in [C.OUT_DIR, C.CKPT_DIR, C.LOG_DIR, C.FIG_DIR, C.RESULT_DIR, C.SAMPLE_DIR, C.DATA_DIR]:
        os.makedirs(d, exist_ok=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def ckpt_path(name):
    return os.path.join(C.CKPT_DIR, f"{name}.pt")


def save_ckpt(name, state):
    """Simpan checkpoint secara atomik (aman bila Colab terputus saat menyimpan)."""
    path = ckpt_path(name)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_ckpt(name, device):
    path = ckpt_path(name)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=device, weights_only=False)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


class Timer:
    """Timer akurat untuk GPU (memanggil cuda.synchronize sebelum/sesudah)."""

    def __init__(self, device):
        self.device = device

    def __enter__(self):
        sync(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        sync(self.device)
        self.elapsed = time.perf_counter() - self.t0


def to_pm1(x):
    """[0, 1] -> [-1, 1] (input GAN dan DDPM)."""
    return x * 2.0 - 1.0


def to_01(x):
    """[-1, 1] -> [0, 1] (untuk metrik & visualisasi)."""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
