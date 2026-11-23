"""Eksperimen Bagian 2 (batasan MLP Classifier) dan eksperimen interpolasi laten (Q4 / Figur 4)."""
import torch
import torch.nn.functional as F

from . import config as C
from .utils import to_01


# ------------------------------------------------------------------ Tahap 2: proyeksi balik dari one-hot y
@torch.no_grad()
def topdown_from_class(clf):
    """z_hat = ReLU(W3^T y), h_hat = ReLU(W2^T z_hat), x_hat = Sigmoid(W1^T h_hat), tanpa pelatihan tambahan."""
    W1, W2, W3 = clf.fc1.weight, clf.fc2.weight, clf.fc3.weight     # (256,784), (64,256), (10,64)
    y = torch.eye(10, device=W1.device)                               # 10 vektor one-hot (baris)
    z_hat = F.relu(y @ W3)                                            # = (W3^T y)^T  -> (10, 64)
    h_hat = F.relu(z_hat @ W2)                                        # = (W2^T z)^T  -> (10, 256)
    x_hat = torch.sigmoid(h_hat @ W1)                                 # = (W1^T h)^T  -> (10, 784)
    return x_hat.view(-1, *C.IMG_SHAPE)


# ------------------------------------------------------------------ Tahap 3: proyeksi balik dari z_MLP
@torch.no_grad()
def latent_transpose(clf, z):
    """x_hat = Sigmoid(W1^T ReLU(W2^T z_MLP))."""
    W1, W2 = clf.fc1.weight, clf.fc2.weight
    return torch.sigmoid(F.relu(z @ W2) @ W1).view(-1, *C.IMG_SHAPE)


@torch.no_grad()
def orthogonality_check(clf):
    """Bukti numerik W^T W != I: ||W^T W - I||_F dan error proyeksi balik W^T W x vs x."""
    out = {}
    for name, layer in [("W1", clf.fc1), ("W2", clf.fc2), ("W3", clf.fc3)]:
        W = layer.weight
        WtW = W.t() @ W
        I = torch.eye(WtW.size(0), device=W.device)
        out[name] = {
            "shape": list(W.shape),
            "rank": int(torch.linalg.matrix_rank(W).item()),
            "fro_norm_WtW_minus_I": float(torch.linalg.norm(WtW - I).item()),
            "rel_fro_norm": float((torch.linalg.norm(WtW - I) / torch.linalg.norm(I)).item()),
        }
    return out


# ------------------------------------------------------------------ Tahap 5: statistik ruang laten MLP
@torch.no_grad()
def latent_stats(clf, loader, device):
    zs = []
    for x, _ in loader:
        zs.append(clf.features(x.to(device)))
    z = torch.cat(zs)
    return z.mean(0), z.std(0), z


@torch.no_grad()
def mlp_prior_sampling(decoder, mu, sigma, n, device, seed=C.SEED):
    """z_rand ~ N(mu_z, diag(sigma_z^2)) -> Decoder Tahap 4."""
    g = torch.Generator(device=device).manual_seed(seed)
    z = mu + sigma * torch.randn(n, mu.numel(), device=device, generator=g)
    return decoder(z)


# ------------------------------------------------------------------ Interpolasi laten (Figur 4)
def _lerp(za, zb, alphas):
    return torch.cat([(1 - a) * za + a * zb for a in alphas])


@torch.no_grad()
def interpolate_autoencoder(model, xa, xb, alphas=C.INTERP_ALPHAS):
    """Vanilla AE / Sparse AE: z = encoder(x). VAE: z = mu(x)."""
    za, zb = model.encode(xa)[0], model.encode(xb)[0]
    return model.decode(_lerp(za, zb, alphas))


@torch.no_grad()
def find_gan_endpoints(G, clf, device, class_a=C.INTERP_CLASS_A, class_b=C.INTERP_CLASS_B, n=2048, seed=C.SEED):
    """GAN tidak punya encoder -> pilih z_A, z_B (dari prior) yang hasilnya paling yakin
    diklasifikasikan sebagai kelas A dan kelas B oleh MLP Classifier."""
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(n, C.NOISE_DIM, device=device, generator=g)
    probs = torch.cat([F.softmax(clf(to_01(G(zc))), 1) for zc in z.split(512)])
    ia, ib = probs[:, class_a].argmax(), probs[:, class_b].argmax()
    return z[ia:ia + 1], z[ib:ib + 1], float(probs[ia, class_a]), float(probs[ib, class_b])


@torch.no_grad()
def interpolate_gan(G, za, zb, alphas=C.INTERP_ALPHAS):
    return to_01(G(_lerp(za, zb, alphas)))
