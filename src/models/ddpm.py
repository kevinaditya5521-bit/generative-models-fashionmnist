"""DDPM (Ho et al., 2020): U-Net prediktor derau + jadwal difusi linear + EMA."""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C


def timestep_embedding(t, dim):
    """Sinusoidal positional embedding untuk timestep t."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, groups=8, p_drop=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.drop = nn.Dropout(p_drop)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class UNet(nn.Module):
    """U-Net kecil untuk 28x28: 28 -> 14 -> 7 -> 14 -> 28 dengan skip connection."""

    def __init__(self, ch=C.DDPM_BASE_CH, tdim=256):
        super().__init__()
        self.ch = ch
        self.time_mlp = nn.Sequential(nn.Linear(ch, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.conv_in = nn.Conv2d(1, ch, 3, padding=1)
        self.d1a, self.d1b = ResBlock(ch, ch, tdim), ResBlock(ch, ch, tdim)
        self.down1 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)                  # 28 -> 14
        self.d2a, self.d2b = ResBlock(ch, 2 * ch, tdim), ResBlock(2 * ch, 2 * ch, tdim)
        self.down2 = nn.Conv2d(2 * ch, 2 * ch, 3, stride=2, padding=1)          # 14 -> 7
        self.m1, self.m2 = ResBlock(2 * ch, 2 * ch, tdim), ResBlock(2 * ch, 2 * ch, tdim)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(2 * ch, 2 * ch, 3, padding=1))
        self.u2a, self.u2b = ResBlock(4 * ch, 2 * ch, tdim), ResBlock(2 * ch, 2 * ch, tdim)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(2 * ch, 2 * ch, 3, padding=1))
        self.u1a, self.u1b = ResBlock(3 * ch, ch, tdim), ResBlock(ch, ch, tdim)
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, x, t):
        temb = self.time_mlp(timestep_embedding(t, self.ch))
        x = self.conv_in(x)
        s1 = self.d1b(self.d1a(x, temb), temb)
        x = self.down1(s1)
        s2 = self.d2b(self.d2a(x, temb), temb)
        x = self.down2(s2)
        x = self.m2(self.m1(x, temb), temb)
        x = self.up2(x)
        x = self.u2b(self.u2a(torch.cat([x, s2], 1), temb), temb)
        x = self.up1(x)
        x = self.u1b(self.u1a(torch.cat([x, s1], 1), temb), temb)
        return self.conv_out(F.silu(self.norm_out(x)))


class Diffusion:
    """Forward q(x_t|x_0) dan reverse p(x_{t-1}|x_t) dengan jadwal beta linear."""

    def __init__(self, T=C.DDPM_T, beta_start=C.DDPM_BETA_START, beta_end=C.DDPM_BETA_END, device="cpu"):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_ab = self.alpha_bar.sqrt()
        self.sqrt_1m_ab = (1.0 - self.alpha_bar).sqrt()
        self.sqrt_recip_a = 1.0 / self.alphas.sqrt()
        self.eps_coef = self.betas / self.sqrt_1m_ab
        self.sigma = self.betas.sqrt()

    def q_sample(self, x0, t, noise):
        """x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps."""
        return self.sqrt_ab[t].view(-1, 1, 1, 1) * x0 + self.sqrt_1m_ab[t].view(-1, 1, 1, 1) * noise

    def loss(self, model, x0):
        """L_simple = E || eps - eps_theta(x_t, t) ||^2 (supervised noise prediction)."""
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        return F.mse_loss(model(self.q_sample(x0, t, noise), t), noise)

    @torch.no_grad()
    def sample(self, model, n, use_amp=False, traj_steps=None, generator=None):
        """Reverse denoising x_T ~ N(0,I) -> x_0 (T langkah). Output di rentang [-1, 1]."""
        model.eval()
        amp = use_amp and str(self.device).startswith("cuda")
        x = torch.randn(n, *C.IMG_SHAPE, device=self.device, generator=generator)
        traj = {}
        if traj_steps is not None and self.T in traj_steps:
            traj[self.T] = x.clamp(-1, 1).cpu()
        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                eps = model(x, t)
            eps = eps.float()
            mean = self.sqrt_recip_a[i] * (x - self.eps_coef[i] * eps)
            if i > 0:
                z = torch.randn(x.shape, device=self.device, generator=generator)
                x = mean + self.sigma[i] * z
            else:
                x = mean
            if traj_steps is not None and i in traj_steps:
                traj[i] = x.clamp(-1, 1).cpu()
        x = x.clamp(-1, 1)
        return (x, traj) if traj_steps is not None else x


class EMA:
    """Exponential Moving Average bobot model (meningkatkan kualitas sampel DDPM)."""

    def __init__(self, model, decay=C.EMA_DECAY):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ps, p in zip(self.shadow.parameters(), model.parameters()):
            ps.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for bs, b in zip(self.shadow.buffers(), model.buffers()):
            bs.copy_(b)
