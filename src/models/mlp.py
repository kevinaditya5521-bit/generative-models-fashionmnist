"""MLP Classifier (784 -> 256 -> 64 -> 10) dan Decoder MLP terpisah (64 -> 256 -> 784)."""
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C


class MLPClassifier(nn.Module):
    """Classifier diskriminatif P(y|x). fc1=W1, fc2=W2, fc3=W3."""

    def __init__(self, in_dim=C.IMG_DIM, hidden=C.HIDDEN_DIM, latent=C.LATENT_DIM, n_classes=10, p_drop=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)      # W1: 256 x 784
        self.fc2 = nn.Linear(hidden, latent)      # W2: 64 x 256
        self.fc3 = nn.Linear(latent, n_classes)   # W3: 10 x 64
        self.drop = nn.Dropout(p_drop)

    def features(self, x):
        """z_MLP in R^64: representasi laten sebelum lapisan klasifikasi akhir."""
        h = self.drop(F.relu(self.fc1(x.flatten(1))))
        return F.relu(self.fc2(h))

    def forward(self, x):
        return self.fc3(self.drop(self.features(x)))


class MLPDecoder(nn.Module):
    """Decoder 64 -> 256 -> 784 (dipakai untuk Frozen Latent Decoding, Tahap 4)."""

    def __init__(self, latent=C.LATENT_DIM, hidden=C.HIDDEN_DIM, out_dim=C.IMG_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, out_dim), nn.Sigmoid())

    def forward(self, z):
        return self.net(z).view(-1, *C.IMG_SHAPE)
