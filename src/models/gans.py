"""Vanilla GAN (fully connected) dan DCGAN (konvolusional penuh) untuk citra 28x28."""
import torch.nn as nn

from .. import config as C


class MLPGenerator(nn.Module):
    """G: z (100) -> 256 -> 512 -> 1024 -> 784 (Tanh)."""

    def __init__(self, z_dim=C.NOISE_DIM, out_dim=C.IMG_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, out_dim), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(-1, *C.IMG_SHAPE)


class MLPDiscriminator(nn.Module):
    """D: 784 -> 512 -> 256 -> 1 (logit)."""

    def __init__(self, in_dim=C.IMG_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x.flatten(1)).view(-1)


class DCGANGenerator(nn.Module):
    """Prinsip DCGAN: ConvTranspose (tanpa pooling), BatchNorm, ReLU, output Tanh.
    z (100x1x1) -> 256x7x7 -> 128x14x14 -> 1x28x28
    """

    def __init__(self, z_dim=C.NOISE_DIM, ngf=128):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 2, 7, 1, 0, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(-1, self.z_dim, 1, 1))


class DCGANDiscriminator(nn.Module):
    """Strided conv (tanpa pooling), BatchNorm, LeakyReLU(0.2). 1x28x28 -> 64x14x14 -> 128x7x7 -> 1."""

    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, 7, 1, 0),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def dcgan_weights_init(m):
    """Inisialisasi bobot N(0, 0.02) sesuai paper DCGAN (Radford et al., 2016)."""
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
