# ============================================================
# VICTOR v6.0 — config.py
# All project constants: grid, hash, ensemble, training, paths
# ============================================================
# Usage:
#   from victor import config as cfg
#   cfg.N_GRID, cfg.R0, cfg.T_COORD, ...
#
# Centralized project configuration shared across all VICTOR modules.
# ============================================================

import os

# ── Grid / geometry ──────────────────────────────────────────
N_GRID  = 128       # pixels per side  (128×128 reconstruction grid)
EXT     = 0.64      # half-extent [m] of the square pixel domain
R0      = 2.5       # major radius of grid centre [m]
AP      = 0.5       # semi-axis in R (ellipse normalisation)
BP      = 0.65      # semi-axis in Z (ellipse normalisation)

# ── Hash grid ────────────────────────────────────────────────
T_COORD = 8192      # hash table size  — coordinate hash
T_FIELD = 4096      # hash table size  — field quantity hash
L_HASH  = 16        # number of resolution levels
F_HASH  = 2         # features per level

# ── Ensemble ─────────────────────────────────────────────────
N_ENS   = 5         # number of ensemble members

# ── PIGNO ────────────────────────────────────────────────────
PIGNO_LAYERS = 2
PIGNO_HIDDEN = 96

# ── Training ─────────────────────────────────────────────────
N_EPOCHS   = 10_000
LR         = 3e-4
SAVE_EVERY = 50
LOG_EVERY  = 200

# Noise stages: list of (n_steps, sigma_fraction) tuples
# Three progressive noise levels over the full training run
STAGES = [
    (N_EPOCHS // 3,              0.001),
    (N_EPOCHS // 3,              0.003),
    (N_EPOCHS - 2*(N_EPOCHS//3), 0.008),
]

# ── Profiles ─────────────────────────────────────────────────
N_PROFILES = 10     # number of TORAX equilibrium profiles to load

# ── rho-graph (PIGNO adjacency) ──────────────────────────────
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
# Repo-relative default paths (overrideable via environment variables).
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
    print(f"  HashGrid : T_coord={T_COORD}  T_field={T_FIELD}  L={L_HASH}  F={F_HASH}")
    print(f"  Ensemble : N={N_ENS}  Profiles={N_PROFILES}")
    print(f"  Training : epochs={N_EPOCHS}  LR={LR}  save={SAVE_EVERY}  log={LOG_EVERY}")
    print(f"  Stages   : {STAGES}")
    print(f"  Paths    : dataset={DATASET_DIR}")
    print(f"             ckpt={CKPT_DIR}")
    print("────────────────────────────────────────────────────────")


if __name__ == "__main__":
    summary()
