# ============================================================
# VICTOR v7.0 — config.py
# All project constants: grid, FourierDeepONet, training, paths
# ============================================================
# Usage:
#   from victor import config as cfg
#   cfg.N_GRID, cfg.R0, cfg.N_RADIAL, ...
#
# v7 changes vs v6
# ----------------
#  • Added N_RADIAL  : radial output length of FourierDeepONet (128)
#  • Added RHO_MAX   : radial grid upper bound (1.2, slightly beyond LCFS)
#  • Removed T_COORD, T_FIELD, L_HASH, F_HASH  (hash grid — v6 only)
#  • Removed N_ENS                              (ensemble — v6 only)
#  • Removed PIGNO_LAYERS, PIGNO_HIDDEN         (PIGNO — v6 only)
#  • summary() updated to reflect v7 fields
# ============================================================

import os

# ── Grid / geometry ──────────────────────────────────────────
N_GRID  = 128       # pixels per side  (128×128 reconstruction grid)
EXT     = 0.64      # half-extent [m] of the square pixel domain
R0      = 2.5       # major radius of grid centre [m]
AP      = 0.5       # semi-axis in R (ellipse normalisation)
BP      = 0.65      # semi-axis in Z (ellipse normalisation)

# ── Radial output grid (FourierDeepONet) ─────────────────────
N_RADIAL = 128      # radial output points of FourierDeepONet
RHO_MAX  = 1.2      # upper bound of radial axis (beyond LCFS at ρ=1)

# ── Poloidal harmonics (v8) ───────────────────────────────────
N_HARMONICS   = 2   # number of Fourier harmonics H
                    # output channels = 1 + 2*H  (a0, a1, b1, a2, b2)

# ── Equilibrium encoder input channels (v8) ──────────────────
N_EQ_CHANNELS = 2   # currently psi_2d + rho_2d
                    # increment here to add Te, ne, q later — no other file changes needed

# ── Training ─────────────────────────────────────────────────
N_EPOCHS   = 10_000
LR         = 3e-4
SAVE_EVERY = 50
LOG_EVERY  = 200

# Noise stages: list of (n_steps, sigma_fraction) tuples
STAGES = [
    (N_EPOCHS // 3,              0.001),
    (N_EPOCHS // 3,              0.003),
    (N_EPOCHS - 2*(N_EPOCHS//3), 0.008),
]

# ── Profiles ─────────────────────────────────────────────────
N_PROFILES = 10     # number of TORAX equilibrium profiles to load

# ── rho-graph (geometry adjacency) ──────────────────────────
RHO_GRAPH_N_NB   = 6      # k-nearest neighbours in rho-space
RHO_GRAPH_SIGMA  = 0.04   # RBF bandwidth
RHO_GRAPH_STRIDE = 4      # sub-sample stride for graph building

# ── Ray tracing (WEST-tuned adaptive march) ──────────────────
# Camera specs: (px, py, angle_min_deg, angle_max_deg, n_rays)
CAMERAS = [
    (1.5,  0.0, 155., 205., 64),
    (0.0, -1.5,  65., 115., 64),
]
DS_MAX_FACTOR = 0.5        # ds_max = (2*EXT/N_GRID) * DS_MAX_FACTOR
DS_MIN_FACTOR = 0.5        # ds_min = ds_max * DS_MIN_FACTOR

# ── Paths ─────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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


def summary() -> None:
    """Print a compact summary of all active constants."""
    print("── config.py ───────────────────────────────────────────")
    print(f"  Grid     : {N_GRID}×{N_GRID}  EXT={EXT}  R0={R0}  AP={AP}  BP={BP}")
    print(f"  Radial   : N_RADIAL={N_RADIAL}  RHO_MAX={RHO_MAX}")
    print(f"  Harmonics: N_HARMONICS={N_HARMONICS}  "
          f"output_channels={1 + 2*N_HARMONICS}  (a0 + {N_HARMONICS} cos/sin pairs)")
    print(f"  Eq enc.  : N_EQ_CHANNELS={N_EQ_CHANNELS}  (psi+rho now; extend for Te/ne/q)")
    print(f"  Training : epochs={N_EPOCHS}  LR={LR}  save={SAVE_EVERY}  log={LOG_EVERY}")
    print(f"  Stages   : {STAGES}")
    print(f"  Profiles : {N_PROFILES}")
    print(f"  Paths    : dataset={DATASET_DIR}")
    print(f"             ckpt={CKPT_DIR}")
    print("────────────────────────────────────────────────────────")


if __name__ == "__main__":
    summary()
