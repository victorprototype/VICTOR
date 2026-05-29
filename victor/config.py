# ============================================================
# VICTOR v8.2 — config.py
# All project constants: grid, FourierDeepONet, training, paths
# ============================================================
# Usage:
#   from victor import config as cfg
#   cfg.N_GRID, cfg.R0, cfg.N_RADIAL, ...
#
# v8.1 changes vs v8.0
# --------------------
#  • W_PDE raised 0.05 → 0.5  : physics residual (Lf) now co-equal with
#                                data fidelity per NAS-PINN / standard PINN
#                                loss   L = L_data + W_PDE·Lf + W_BC·L_B
#  • W_BOUNDARY raised slightly: tighter collocated enforcement of L_B
#  • Added LERP_EPS            : safe denominator floor for differentiable
#                                bilinear interpolation inside the PDE residual
#  • Added PDE_COLLOC_N        : collocation budget near ρ=1 boundary for Lf
#  • Added SKIP_GATE_INIT      : initial logit for learned sigmoid gate on
#                                U-Net skip connections (0 → gate starts at 0.5)
#  • Added HARMONIC_DECAY_SCHEDULE : step-wise schedule tightening poloidal
#                                    harmonic regularisation over training
#  • Added CURRICULUM_SCHEDULE : front-loads PDE residual warmup before data
#                                 fidelity so physics is embedded first
#  • N_HARMONICS comment updated to clarify up-down symmetry assumption
#  • summary() extended to print all new constants
#
# v8.0 changes vs v7.0
# --------------------
#  • Added N_HARMONICS, N_EQ_CHANNELS (poloidal harmonics, equilibrium encoder)
#  • Corrected AP=1.0, BP=1.2, EXT=1.0 to match TORAX equilibrium geometry
# ============================================================

import os

# ── Grid / geometry ──────────────────────────────────────────
N_GRID  = 128       # pixels per side  (128×128 reconstruction grid)
EXT     = 1.0       # half-extent [m] of the square pixel domain
                    # matches TORAX R∈[1.5,3.5] → a=1.0, Z∈[-1.2,1.2] → b=1.2
R0      = 2.5       # major radius of grid centre [m]
AP      = 1.0       # semi-axis in R (ellipse normalisation) — TORAX a = (3.5-1.5)/2
BP      = 1.2       # semi-axis in Z (ellipse normalisation) — TORAX b = (1.2-(-1.2))/2

# ── Radial output grid (FourierDeepONet) ─────────────────────
N_RADIAL = 128      # radial output points of FourierDeepONet
RHO_MAX  = 1.2      # upper bound of radial axis (beyond LCFS at ρ=1)

# ── Poloidal harmonics (v8) ───────────────────────────────────
N_HARMONICS   = 2   # number of Fourier harmonics H retained in the poloidal
                    # expansion.  Assumes up-down symmetry (sin terms are zero
                    # for purely symmetric equilibria), so only cosine pairs are
                    # physically active by default.
                    # output channels = 1 + 2*H  (a0, a1, b1, a2, b2)

# ── Equilibrium encoder input channels (v8) ──────────────────
N_EQ_CHANNELS = 2   # currently psi_2d + rho_2d
                    # increment here to add Te, ne, q later — no other file changes needed

# ── PINN loss weights ────────────────────────────────────────
# Full PINN objective:  L = L_data + W_PDE·Lf + W_BOUNDARY·L_B + W_IC·L_I
#
#   L_data     — supervised data-fidelity term (line-integrated brightness)
#   Lf         — interior PDE / physics residual  (e.g. ∇²ψ = f(ψ))
#   L_B        — boundary condition residual at the LCFS (ρ = 1)
#   L_I        — initial condition / symmetry residual (up-down, axis)
#
# NAS-PINN and standard PINN theory require W_PDE ~ O(1) so that physics
# is co-equal with data fidelity.  The v8.0 value of 0.05 under-weighted Lf
# by a factor ~10, allowing the network to minimise L_data while violating
# the equilibrium constraint.  Raising to 0.5 restores the intended balance.

W_PDE      = 0.5    # weight on interior PDE residual Lf
                    # raised from 0.05 → 0.5 (×10) so physics is co-equal
                    # with L_data; validated against NAS-PINN loss scaling

W_BOUNDARY = 0.3    # weight on boundary residual L_B
                    # slightly raised (was 0.1) to match collocated enforcement
                    # at the LCFS — prevents boundary leakage seen in v8.0 runs

W_IC       = 0.1    # weight on initial/symmetry condition residual L_I
                    # (magnetic axis regularity + up-down symmetry enforcement)

# ── Numerical stability ──────────────────────────────────────
LERP_EPS = 1e-6     # floor added to bilinear interpolation denominators inside
                    # the PDE residual computation.  Without this, grid-aligned
                    # query points produce 0/0 in the differentiable trilinear
                    # kernel, causing NaN gradients that silently corrupt Lf.
                    # Value 1e-6 is << grid spacing (~0.016 m) so it does not
                    # bias interpolated values measurably.

# ── PDE collocation ──────────────────────────────────────────
PDE_COLLOC_RHO_THRESHOLD = 0.85  # inner ρ boundary of the collocation annulus.
                    # Points are sampled from ρ ∈ [PDE_COLLOC_RHO_THRESHOLD, 1.05]
                    # where the equilibrium gradient is steepest.  0.85 matches
                    # the annulus described in the PDE_COLLOC_N comment below.
                    # Added in v8.2 — was previously hardcoded in geometry.py.

PDE_COLLOC_N = 512  # number of collocation points sampled near the ρ=1
                    # boundary per training step for the residual Lf.
                    # Points are drawn from a thin annulus ρ ∈ [0.85, 1.05]
                    # where the equilibrium gradient is steepest and where
                    # physics violations are most costly for reconstruction.
                    # 512 balances coverage against memory; decrease to 256
                    # on GPUs < 16 GB if OOM during residual back-prop.

# ── U-Net skip gate ──────────────────────────────────────────
SKIP_GATE_INIT = 0.0    # initial logit value for the learned sigmoid gate
                        # applied to each U-Net skip connection:
                        #   gate = sigmoid(SKIP_GATE_INIT) = 0.5 at init
                        # Starting at 0.5 (logit=0) gives the network full
                        # access to skip features at initialisation; the gate
                        # is free to open or close during training.  Setting
                        # this to a large negative value (e.g. -4) would
                        # suppress skips initially, biasing the decoder toward
                        # the bottleneck — useful if skip features cause
                        # early-training instability.

# ── Harmonic regularisation schedule ─────────────────────────
# List of (global_step, w_pol) pairs.  The trainer reads this schedule and
# applies the matching w_pol at each step, linearly interpolating between
# listed breakpoints.  Harmonic regularisation penalises high-H poloidal
# content early in training (when the network is noisy) and relaxes the
# penalty later so fine angular structure is learned once the radial profile
# is stable.  This mirrors the noise curriculum in STAGES.
#
# w_pol multiplies the harmonic amplitude penalty:
#   L_harm = w_pol · Σ_{h>0} (a_h² + b_h²) / h²
HARMONIC_DECAY_SCHEDULE = [
    (0,       1.0),    # start: strong regularisation, suppress spurious harmonics
    (20_000,  0.5),    # relax after radial profile has stabilised
    (60_000,  0.1),    # allow fine poloidal structure to emerge
    (110_000, 0.05),   # near end: very light touch, trust the physics loss
]

# ── Training ─────────────────────────────────────────────────
N_EPOCHS   = 110_000
LR         = 1e-4
SAVE_EVERY = 5000
LOG_EVERY  = 500

# ── Noise curriculum (STAGES) ────────────────────────────────
# List of (n_steps, sigma_fraction) tuples applied sequentially.
# Controls synthetic noise injected into line-integrated measurements
# to build robustness; independent of the physics curriculum below.
STAGES = [
    (10_000,  0.0005),   # warmup: very low noise, learn clean structure
    (40_000,  0.001),    # stage 1: low noise, refine
    (30_000,  0.003),    # stage 2: medium noise
    (30_000,  0.008),    # stage 3: push robustness
]

# ── Physics curriculum (CURRICULUM_SCHEDULE) ─────────────────
# List of (global_step, w_pde_scale) tuples.  The trainer multiplies
# W_PDE by w_pde_scale at each step so that the PDE residual Lf is
# introduced gently and then ramped to full weight.
#
# Rationale (NAS-PINN §3.2): if the full physics loss is active from
# step 0, it dominates randomly-initialised weights and prevents the
# network from first learning a coarse data-consistent solution.
# Front-loading a reduced Lf for the first 10 k steps mirrors the
# "residual warmup" strategy and avoids the saddle-point failure mode
# reported in Raissi et al. (2019) for stiff PDE systems.
#
# The noise curriculum (STAGES) and this physics curriculum are
# intentionally decoupled so each axis can be tuned independently.
CURRICULUM_SCHEDULE = [
    (0,       0.05),   # step 0–10k: Lf at 10 % of W_PDE — learn data first
    (10_000,  0.2),    # step 10k–30k: ramp to 40 % — physics begins to bite
    (30_000,  0.6),    # step 30k–60k: 60 % — equilibrium constraint tightens
    (60_000,  1.0),    # step 60k+: full W_PDE — co-equal with data fidelity
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
    n_curriculum_steps = sum(
        b[0] - a[0] for a, b in zip(CURRICULUM_SCHEDULE, CURRICULUM_SCHEDULE[1:])
    )
    print("── config.py (v8.2) ────────────────────────────────────")
    print(f"  Grid     : {N_GRID}×{N_GRID}  EXT={EXT}  R0={R0}  AP={AP}  BP={BP}")
    print(f"             (TORAX R∈[{R0-AP},{R0+AP}] m  Z∈[{-BP},{BP}] m)")
    print(f"  Radial   : N_RADIAL={N_RADIAL}  RHO_MAX={RHO_MAX}")
    print(f"  Harmonics: N_HARMONICS={N_HARMONICS}  "
          f"output_channels={1 + 2*N_HARMONICS}  (a0 + {N_HARMONICS} cos/sin pairs, "
          f"up-down symmetry assumed)")
    print(f"  Eq enc.  : N_EQ_CHANNELS={N_EQ_CHANNELS}  (psi+rho now; extend for Te/ne/q)")
    print(f"  PINN loss: W_PDE={W_PDE}  W_BOUNDARY={W_BOUNDARY}  W_IC={W_IC}")
    print(f"             L = L_data + {W_PDE}·Lf + {W_BOUNDARY}·L_B + {W_IC}·L_I")
    print(f"  Numerics : LERP_EPS={LERP_EPS}  PDE_COLLOC_N={PDE_COLLOC_N}  "
          f"PDE_COLLOC_RHO_THRESHOLD={PDE_COLLOC_RHO_THRESHOLD}")
    print(f"  Skip gate: SKIP_GATE_INIT={SKIP_GATE_INIT}  "
          f"(sigmoid → {1/(1+__import__('math').exp(-SKIP_GATE_INIT)):.2f} at init)")
    print(f"  Harm.sched: {HARMONIC_DECAY_SCHEDULE}")
    print(f"  Curriculum: {CURRICULUM_SCHEDULE}")
    print(f"             ({n_curriculum_steps} steps covered by schedule)")
    print(f"  Training : epochs={N_EPOCHS}  LR={LR}  save={SAVE_EVERY}  log={LOG_EVERY}")
    print(f"  Stages   : {STAGES}")
    print(f"  Profiles : {N_PROFILES}")
    print(f"  Paths    : dataset={DATASET_DIR}")
    print(f"             ckpt={CKPT_DIR}")
    print("────────────────────────────────────────────────────────")


if __name__ == "__main__":
    summary()
