"""Konfigurasi global eksperimen."""
import os

PROJECT_DIR = os.environ.get("PROJECT_DIR", os.path.abspath("."))
QUICK_TEST = os.environ.get("QUICK_TEST", "0") == "1"   # 1 = uji cepat pipeline
DATA_DIR = os.environ.get(
    "DATA_DIR", "/content/data" if os.path.isdir("/content") else os.path.join(PROJECT_DIR, "data")
)

# ---------------------------------------------------------------- Umum
SEED = 42
BATCH_SIZE = 128            # sesuai aturan: 128 atau 256
SPLIT = (0.8, 0.1, 0.1)     # train / val / test
IMG_SHAPE = (1, 28, 28)
IMG_DIM = 28 * 28
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# ---------------------------------------------------------------- Arsitektur
HIDDEN_DIM = 256            # 784 -> 256 -> 64 (sama untuk MLP Classifier dan semua AE)
LATENT_DIM = 64             # dimensi laten AE / Sparse AE / VAE / z_MLP
NOISE_DIM = 100             # dimensi prior noise Vanilla GAN dan DCGAN
SPARSE_LAMBDA = 1e-4        # lambda pada penalti L1 ||h||_1 Sparse AE

# DDPM
DDPM_T = 1000
DDPM_BETA_START = 1e-4
DDPM_BETA_END = 0.02
DDPM_BASE_CH = 64
EMA_DECAY = 0.999
USE_AMP = True              # mixed precision (hanya aktif bila GPU tersedia)

# ---------------------------------------------------------------- Training
EPOCHS = dict(mlp=20, frozen_decoder=15, vanilla_ae=20, sparse_ae=20, vae=20,
              vanilla_gan=50, dcgan=20, ddpm=20)
LR = dict(mlp=1e-3, frozen_decoder=1e-3, vanilla_ae=1e-3, sparse_ae=1e-3, vae=1e-3,
          vanilla_gan=2e-4, dcgan=2e-4, ddpm=2e-4)
GAN_BETAS = (0.5, 0.999)
MLP_MIN_ACC = 0.88          # syarat tugas: akurasi test MLP >= 88%

# ---------------------------------------------------------------- Evaluasi
N_FID = 2000                # jumlah sampel real & sintetis untuk FID (sama untuk semua model)
FID_BATCH = 250
LATENCY_BATCH = 64          # latency diukur per batch 64 sampel
LATENCY_RUNS = 10
LATENCY_RUNS_DDPM = 3       # DDPM 1000 langkah -> cukup 3 kali ulang
INTERP_ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
INTERP_CLASS_A = 0          # T-shirt/top
INTERP_CLASS_B = 9          # Ankle boot (kelas yang kontras)
DDPM_TRAJ_STEPS = [1000, 800, 600, 400, 300, 200, 100, 50, 0]

# ---------------------------------------------------------------- Mode uji cepat
if QUICK_TEST:
    EPOCHS = {k: 1 for k in EPOCHS}
    DDPM_T = 20
    DDPM_TRAJ_STEPS = [20, 15, 10, 5, 0]
    N_FID = 300
    LATENCY_RUNS = 2
    LATENCY_RUNS_DDPM = 1

# ---------------------------------------------------------------- Folder output
OUT_DIR = os.path.join(PROJECT_DIR, "outputs_test" if QUICK_TEST else "outputs")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUT_DIR, "logs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
RESULT_DIR = os.path.join(OUT_DIR, "results")
SAMPLE_DIR = os.path.join(OUT_DIR, "samples")
