"""Vanilla / Sparse Autoencoder (arsitektur sama, beda loss) dan Variational Autoencoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C


class Autoencoder(nn.Module):
    """Encoder 784 -> 256 (ReLU) -> 64 (linear), Decoder 64 -> 256 (ReLU) -> 784 (Sigmoid).

    Vanilla AE : L = MSE(x, x_hat)
    Sparse AE  : L = MSE(x, x_hat) + lambda * ||h||_1   (h = aktivasi hidden 256 dimensi)
    """

    def __init__(self, in_dim=C.IMG_DIM, hidden=C.HIDDEN_DIM, latent=C.LATENT_DIM):
        super().__init__()
        self.enc1 = nn.Linear(in_dim, hidden)
        self.enc2 = nn.Linear(hidden, latent)
        self.dec1 = nn.Linear(latent, hidden)
        self.dec2 = nn.Linear(hidden, in_dim)

    def encode(self, x):
        h = F.relu(self.enc1(x.flatten(1)))
        z = self.enc2(h)
        return z, h

    def decode(self, z):
        return torch.sigmoid(self.dec2(F.relu(self.dec1(z)))).view(-1, *C.IMG_SHAPE)

    def forward(self, x):
        z, h = self.encode(x)
        return self.decode(z), z, h


class VAE(nn.Module):
    """q(z|x) = N(mu, diag(sigma^2)), p(z) = N(0, I). Arsitektur sepadan dengan AE."""

    def __init__(self, in_dim=C.IMG_DIM, hidden=C.HIDDEN_DIM, latent=C.LATENT_DIM):
        super().__init__()
        self.latent = latent
        self.enc1 = nn.Linear(in_dim, hidden)
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.dec1 = nn.Linear(latent, hidden)
        self.dec2 = nn.Linear(hidden, in_dim)

    def encode(self, x):
        h = F.relu(self.enc1(x.flatten(1)))
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return torch.sigmoid(self.dec2(F.relu(self.dec1(z)))).view(-1, *C.IMG_SHAPE)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(x_hat, x, mu, logvar):
    """Negative ELBO = BCE rekonstruksi + KL(q(z|x) || N(0,I)), dirata-rata per citra."""
    b = x.size(0)
    bce = F.binary_cross_entropy(x_hat, x, reduction="sum") / b
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
    return bce + kld, bce, kld
