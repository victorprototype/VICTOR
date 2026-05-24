# ============================================================
# VICTOR v8.0 — config.py
# Central configuration for VICTOR v8 (poloidal-aware)
# ============================================================
# v8 changes vs v7
# ----------------
#  * N_HARMONICS     : number of poloidal Fourier modes (default 2)
#  * N_CHANNELS_OUT  : 1 + 2*N_HARMONICS  (5 for default)
#  * N_EQ_CHANNELS   : equilibrium input channels (psi + rho = 2)
#                      increase to 4 when adding Te, ne later
#  * W_POLAR         : poloidal regularisation weight (0.001)
#  * HARMONIC_INIT_SCALE : scale applied to harmonic head at init (0.1)
#  * LOG_EVERY set to 200 for expanded metric display
# All v7 constants preserved unchanged.
# ============================================================

import os

# ── Geometry ──────────────────────────────────────────────────────────
N_GRID       = 128          # pixels per side (128×128 reconstruction grid)
EXT          = 0.25         # half-extent [m] of the square domain
R0           = 2.65         # major radius of grid centre [m]  (WEST)
AP           = 0.50         # ellipse semi-axis in R  [m]
BP           = 0.75         # ellipse semi-axis in Z  [m]
RHO_MAX      = 1.2          # rho axis extends to 1.2 to cover SOL
N_RADIAL     = 128          # radial grid points (model output length L)

# ── Poloidal harmonics (NEW v8) ───────────────────────────────────────
N_HARMONICS       = 2       # m = 1, 2  →  a1,b1,a2,b2
N_CHANNELS_OUT    = 1 + 2 * N_HARMONICS   # 5 channels total
N_EQ_CHANNELS     = 2       # psi_n + rho_n  (extend to 4 for Te, ne)
HARMONIC_INIT_SCALE = 0.1   # multiplier applied to harmonic channels at init
                             # keeps harmonics small early so a0 drives training

# ── Ray march ─────────────────────────────────────────────────────────
DS_MAX_FACTOR = 2.0         # ds_max = (2*ext/n_grid) * DS_MAX_FACTOR
DS_MIN_FACTOR = 0.2         # ds_min = ds_max * DS_MIN_FACTOR

# ── rho-proximity graph ───────────────────────────────────────────────
RHO_GRAPH_N_NB  = 8         # nearest neighbours per node
RHO_GRAPH_SIGMA = 0.04      # Laplacian kernel bandwidth
RHO_GRAPH_STRIDE= 4         # sub-sample stride for graph construction

# ── WEST camera configuration ─────────────────────────────────────────
# Each entry: (px, py, a_min_deg, a_max_deg, n_chords)
CAMERAS = [
    ( 0.10, -0.70,  25.0,  85.0, 83),
    (-0.10,  0.70, 205.0, 265.0, 45),
]

# ── Training ──────────────────────────────────────────────────────────
N_PROFILES   = 50           # profiles to load
N_EPOCHS     = 10_000       # steps per profile (single-profile: total steps)
LR           = 3e-4         # peak learning rate
LOG_EVERY    = 200          # log + host-sync interval (steps)
SAVE_EVERY   = 2_000        # checkpoint interval (steps)
BATCH_SIZE   = 16           # samples per JIT step

# Noise curriculum stages: list of (n_steps, sigma) tuples
STAGES = [
    (3_334, 0.010),
    (3_333, 0.003),
    (3_333, 0.001),
]

# ── Loss weights (v8) ─────────────────────────────────────────────────
W_PROJ     = 1.0            # sinogram data-fidelity (dominant)
W_BOUNDARY = 5.0            # zero outside LCFS
W_SMOOTH   = 0.02           # total-variation on a0 profile
W_POLAR    = 0.001          # poloidal harmonic regularisation (NEW v8)
# PDE and positivity disabled in v8 (handled by softplus + clip)

# ── Dataset ───────────────────────────────────────────────────────────
DATASET_DIR = os.environ.get(
    "VICTOR_DATASET",
    os.path.join(ROOT_DIR, "victor_dataset")
)
CKPT_DIR = os.environ.get(
    "VICTOR_CKPT",
    os.path.join(ROOT_DIR, "checkpoints")
)
RESULTS_DIR = os.environ.get(
    "VICTOR_RESULTS",
    os.path.join(ROOT_DIR, "results")
)

def summary():
    """Print a summary of all VICTOR v8 config constants."""
    import victor.config as _c
    print("── VICTOR v8 config ─────────────────────────────────────")
    for k, v in vars(_c).items():
        if k.startswith("_") or callable(v) or isinstance(v, type):
            continue
        print(f"  {k:<22}: {v}")
    print("─────────────────────────────────────────────────────────")