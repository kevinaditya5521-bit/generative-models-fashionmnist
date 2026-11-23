"""Pipeline data Fashion-MNIST: gabung 70.000 citra lalu split 80/10/10 dengan seed tetap."""
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from . import config as C

_CACHE = {}


def load_fashion_mnist():
    """Unduh (legal, via torchvision) dan kembalikan tensor x in [0,1] (N,1,28,28) dan y (N,)."""
    train = datasets.FashionMNIST(C.DATA_DIR, train=True, download=True)
    test = datasets.FashionMNIST(C.DATA_DIR, train=False, download=True)
    x = torch.cat([train.data, test.data]).float().div(255.0).unsqueeze(1)
    y = torch.cat([train.targets, test.targets]).long()
    return x, y


def get_splits():
    """Split 80% train, 10% val, 10% test (56.000 / 7.000 / 7.000)."""
    if "splits" in _CACHE:
        return _CACHE["splits"]
    x, y = load_fashion_mnist()
    n = len(x)
    n_train = int(C.SPLIT[0] * n)
    n_val = int(C.SPLIT[1] * n)
    g = torch.Generator().manual_seed(C.SEED)
    perm = torch.randperm(n, generator=g)
    idx = {"train": perm[:n_train], "val": perm[n_train:n_train + n_val], "test": perm[n_train + n_val:]}
    os.makedirs(C.OUT_DIR, exist_ok=True)
    torch.save(idx, os.path.join(C.OUT_DIR, "split_indices.pt"))  # bukti split reprodusibel
    splits = {k: TensorDataset(x[v], y[v]) for k, v in idx.items()}
    _CACHE["splits"] = splits
    return splits


def get_loaders(batch_size=None):
    """shuffle=True hanya untuk train; val/test shuffle=False (sesuai aturan tugas)."""
    bs = batch_size or C.BATCH_SIZE
    s = get_splits()
    g = torch.Generator().manual_seed(C.SEED)
    pin = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(s["train"], batch_size=bs, shuffle=True, generator=g, pin_memory=pin),
        "val": DataLoader(s["val"], batch_size=bs, shuffle=False, pin_memory=pin),
        "test": DataLoader(s["test"], batch_size=bs, shuffle=False, pin_memory=pin),
    }
    print(f"Split -> train: {len(s['train'])} | val: {len(s['val'])} | test: {len(s['test'])} "
          f"| batch_size: {bs}")
    return loaders


def get_test_tensors():
    s = get_splits()["test"]
    return s.tensors[0], s.tensors[1]


def pick_one_per_class(n_classes=10):
    """Ambil citra uji pertama untuk setiap kelas (untuk figur yang konsisten)."""
    x, y = get_test_tensors()
    idx = [int((y == c).nonzero()[0]) for c in range(n_classes)]
    return x[idx], y[idx]
