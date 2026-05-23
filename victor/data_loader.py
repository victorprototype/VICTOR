# ============================================================
# VICTOR v7.0 — data_loader.py
# W matrix loading, TORAX profile loading, noise injection,
# and field interpolation onto the reconstruction pixel grid.
# ============================================================
# Public API
# ----------
#   load_W_matrix(path, device)              -> WBundle
#   load_profiles(dataset_dir, ..., device)  -> list[dict]
#   inject_noise(g, sigma, key)              -> jnp.ndarray
#   interp_field(field, R_from, Z_from, R_to, Z_to) -> np.ndarray
#   load_cell2(dataset_dir, device)          -> tuple
#
# v7.2 changes vs v7.1
# --------------------
#  * Removed all explicit device_put / _resolve_device tracking to
#    match geometry.py v7.2.  All `device` keyword arguments are now
#    accepted for backward compatibility but silently ignored.  JAX
#    dispatches to the default device automatically.
#
#  * XI_DEFAULT is a plain module-level constant; jnp.array() is
#    called at use-time rather than device_put at load-time.
#
# Design principles (unchanged)
# -----------------
#  * Profile dicts contain plain jnp arrays — never nested dicts
#    passed into @jax.jit.
#  * W operators come from geometry.make_W_operators().
#  * Field interpolation is NumPy / SciPy (runs once at load time).
# ============================================================

from __future__ import annotations

import os
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import scipy.sparse as sp_sci
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags as spd

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from victor import config as cfg
from victor import geometry as geom


# ── Default WEST GEM hardware vector (9-D, normalised) ───────────────
XI_DEFAULT = jnp.array([
    83.0  / 128,    # [0] cam_a_chord_frac
    107.0 / 128,    # [1] cam_b_chord_frac
    2.0   / 15,     # [2] e_low_norm
    15.0  / 15,     # [3] e_high_norm
    50.0  / 100,    # [4] be_window_norm
    473.0 / 1000,   # [5] detector_depth_norm
    0.8   / 2,      # [6] strip_pitch_norm
    3.0   / 4,      # [7] gas_gain_log_norm
    80.0  / 128,    # [8] sampling_rate_norm
], dtype=jnp.float32)


# ── Named return type ─────────────────────────────────────────────────

class WBundle(NamedTuple):
    """Everything derived from the W matrix file."""
    W_norm        : Any             # scipy.sparse.csr_matrix (row-normalised)
    W_BCOO        : jnp.ndarray     # JAX BCOO sparse tensor
    ACTIVE_MASK   : jnp.ndarray     # (128,) bool — rows with non-zero sum
    ACTIVE_MASK_NP: np.ndarray      # same, as numpy
    N_ACTIVE      : int
    w_ops         : geom.WOperators


# ── 1. Load & normalise W matrix ──────────────────────────────────────

def load_W_matrix(
    path   : Optional[str] = None,
    device = None,   # accepted for API compatibility; ignored
) -> WBundle:
    """
    Load the W projection matrix, row-normalise, build BCOO and
    matrix-free operators.

    Parameters
    ----------
    path   : str | None   Path to W_matrix.npz (default cfg.DATASET_DIR).
    device : ignored — kept for backward compatibility.

    Returns
    -------
    WBundle
    """
    if path is None:
        path = os.path.join(cfg.DATASET_DIR, "W_matrix.npz")

    print(f"Loading W matrix from {path} ...")
    W_sp = sp_sci.load_npz(path).tocsr()

    rs             = np.array(W_sp.sum(axis=1)).flatten()
    ACTIVE_MASK_NP = (rs > 1e-8)
    rs_safe        = np.where(ACTIVE_MASK_NP, rs, 1.0)

    W_norm      = spd(1.0 / rs_safe) @ W_sp
    ACTIVE_MASK = jnp.array(ACTIVE_MASK_NP)
    N_ACTIVE    = int(ACTIVE_MASK_NP.sum())

    Wcoo   = W_norm.tocoo()
    W_BCOO = BCOO(
        (
            jnp.array(Wcoo.data,  dtype=jnp.float32),
            jnp.array(np.stack([Wcoo.row, Wcoo.col], axis=1), dtype=jnp.int32),
        ),
        shape=W_norm.shape,
    )

    w_ops = geom.make_W_operators(W_norm.tocsr())

    print(
        f"  W: {W_norm.shape}  NNZ={W_norm.nnz}  "
        f"active={N_ACTIVE}/{W_sp.shape[0]}"
    )

    return WBundle(
        W_norm         = W_norm,
        W_BCOO         = W_BCOO,
        ACTIVE_MASK    = ACTIVE_MASK,
        ACTIVE_MASK_NP = ACTIVE_MASK_NP,
        N_ACTIVE       = N_ACTIVE,
        w_ops          = w_ops,
    )


# ── 2. Field interpolation helper ─────────────────────────────────────

def interp_field(
    field  : np.ndarray,
    R_from : np.ndarray,
    Z_from : np.ndarray,
    R_to   : np.ndarray,
    Z_to   : np.ndarray,
) -> np.ndarray:
    """
    Bilinear interpolation of a 2-D field from one (R,Z) grid to another.
    Runs on CPU (NumPy/SciPy) — called once at load time only.

    Returns np.ndarray of shape R_to.shape, dtype float32.
    Out-of-bounds points are filled with 0.
    """
    fn  = RegularGridInterpolator(
        (R_from, Z_from), field,
        method      = "linear",
        bounds_error= False,
        fill_value  = 0.0,
    )
    pts = np.stack([R_to.flatten(), Z_to.flatten()], axis=1)
    return fn(pts).reshape(R_to.shape).astype(np.float32)


# ── 3. Noise injection ────────────────────────────────────────────────

def inject_noise(
    g     : jnp.ndarray,
    sigma : float,
    key   : jnp.ndarray,
    ps    : float = 1e5,
) -> jnp.ndarray:
    """
    Add Poisson + Gaussian noise to a sinogram.

    NOTE: `key` must already be on the same device as `g`.
    Use jax.device_put(jax.random.PRNGKey(seed), device) at the call
    site (trainer.py) — a CPU key forces this entire function to CPU.

    Parameters
    ----------
    g     : (128,)  clean sinogram [a.u.]
    sigma : float   Gaussian noise fraction of peak signal
    key   : JAX PRNG key  — must be on GPU
    ps    : float   photon scale (higher -> less Poisson noise)

    Returns
    -------
    jnp.ndarray (128,) noisy sinogram, clipped >= 0
    """
    k_poisson, k_gaussian = jax.random.split(key)
    lam = jnp.maximum(g * ps, 1.0)
    g_poisson = jnp.maximum(
        (lam + jnp.sqrt(lam) * jax.random.normal(k_poisson, lam.shape)) / ps,
        0.0,
    )
    return jnp.maximum(
        g_poisson
        + sigma * jnp.max(jnp.abs(g)) * jax.random.normal(k_gaussian, g.shape),
        0.0,
    )


# ── 4. Field normalisation helper ─────────────────────────────────────

def _safe_norm_11(x: np.ndarray) -> np.ndarray:
    """Linearly rescale numpy array to [-1, 1] (safe against zero range)."""
    mn, mx = x.min(), x.max()
    return (2.0 * (x - mn) / (mx - mn + 1e-8) - 1.0).astype(np.float32)


# ── 5. Load TORAX equilibrium profiles ───────────────────────────────

def load_profiles(
    dataset_dir  : Optional[str]            = None,
    n_profiles   : int                      = cfg.N_PROFILES,
    w_bundle     : Optional[WBundle]        = None,
    grids        : Optional[geom.PixelGrids]= None,
    xi           : Optional[jnp.ndarray]    = None,
    device       = None,   # accepted for API compatibility; ignored
) -> List[Dict[str, Any]]:
    """
    Load TORAX equilibrium profiles and pre-process for training.

    Parameters
    ----------
    dataset_dir : str | None   Directory with profile_XXXX.npz files.
    n_profiles  : int          Number of profiles to load.
    w_bundle    : WBundle | None
    grids       : PixelGrids | None
    xi          : jnp.ndarray | None   9-D hardware vector (default XI_DEFAULT).
    device      : ignored — kept for backward compatibility.

    Returns
    -------
    list of dicts.  JAX arrays:
        psi_n, bpol_n, rho_1d, Te_1d, ne_1d  — field / profile arrays
        g_ideal  : (128,)  clean sinogram
        xi       : (9,)    hardware vector
    NumPy arrays (monitoring only, never in JIT):
        eps_n, psi_2d_np, R_grid_np, Z_grid_np
    Scalars:
        g_scale, idx
    """
    if dataset_dir is None:
        dataset_dir = cfg.DATASET_DIR
    if xi is None:
        xi = XI_DEFAULT
    if grids is None:
        grids = geom.build_pixel_grids()

    R_pix_np = np.array(grids.R_PIX).reshape(cfg.N_GRID, cfg.N_GRID)
    Z_pix_np = np.array(grids.Z_PIX).reshape(cfg.N_GRID, cfg.N_GRID)

    if w_bundle is not None:
        _W_matvec = w_bundle.w_ops.matvec
    else:
        def _W_matvec(ef):
            return jnp.zeros(128, dtype=jnp.float32)

    def _put(arr):
        return jnp.array(arr)

    profiles: List[Dict[str, Any]] = []
    print(f"\nLoading {n_profiles} profiles from {dataset_dir}/")

    for profile_idx in range(n_profiles):
        fp = os.path.join(dataset_dir, f"profile_{profile_idx:04d}.npz")

        if not os.path.exists(fp):
            print(f"  MISSING {fp}")
            continue

        d = np.load(fp, allow_pickle=True)

        # ── Emissivity (target) ──────────────────────────────────────
        eps_raw = d["epsilon_2d"].astype(np.float32)
        eps_max = float(eps_raw.max()) + 1e-10
        eps_n   = eps_raw / eps_max                          # (N_GRID, N_GRID)

        # Ideal sinogram via w_ops.matvec
        g_raw   = _W_matvec(_put(eps_n.flatten()))
        g_max   = float(g_raw.max()) + 1e-10
        g_ideal = g_raw / g_max                              # (128,)

        # ── Equilibrium fields (eq-grid -> recon-grid, CPU) ──────────
        R_g      = d["R_grid"].astype(np.float32)
        Z_g      = d["Z_grid"].astype(np.float32)
        psi_raw  = d["psi_2d"].astype(np.float32)
        Bpol_raw = d["B_pol"].astype(np.float32)

        psi_grid  = interp_field(psi_raw,  R_g, Z_g, R_pix_np, Z_pix_np)
        bpol_grid = interp_field(Bpol_raw, R_g, Z_g, R_pix_np, Z_pix_np)

        # ── 1-D TORAX profiles ────────────────────────────────────────
        rho_1d = d["rho"].astype(np.float32)
        Te_1d  = d["Te"].astype(np.float32)
        ne_1d  = d["ne"].astype(np.float32)

        profiles.append(dict(
            # ── JAX arrays on GPU ─────────────────────────────────────
            psi_n   = _put(_safe_norm_11(psi_grid).flatten()),   # (N²,)
            bpol_n  = _put(_safe_norm_11(bpol_grid).flatten()),  # (N²,)
            rho_1d  = _put(rho_1d),
            Te_1d   = _put(Te_1d),
            ne_1d   = _put(ne_1d),
            g_ideal = g_ideal,                                   # (128,)
            xi      = jnp.array(xi),                             # (9,)

            # ── Scalars ───────────────────────────────────────────────
            g_scale = g_max,
            idx     = profile_idx,

            # ── NumPy (monitoring only, never in JIT) ─────────────────
            eps_n      = eps_n,
            psi_2d_np  = psi_raw,
            R_grid_np  = R_g,
            Z_grid_np  = Z_g,
        ))

        print(
            f"  profile_{profile_idx:04d}: "
            f"Te_core={float(Te_1d[0]):.2f} keV  "
            f"g_max={g_max:.4f}"
        )

    print(f"\nLoaded {len(profiles)}/{n_profiles} profiles OK")
    return profiles


# ── 6. Convenience: load everything for Cell 2 ───────────────────────

def load_cell2(
    dataset_dir : Optional[str] = None,
    device      = None,   # accepted for API compatibility; ignored
) -> tuple:
    """
    Full data pipeline in one call.

    Returns
    -------
    w_bundle  : WBundle
    grids     : PixelGrids
    rays      : RayArrays
    rho_graph : RhoGraph
    profiles  : list[dict]
    """
    if dataset_dir is None:
        dataset_dir = cfg.DATASET_DIR

    w_bundle = load_W_matrix(os.path.join(dataset_dir, "W_matrix.npz"))
    grids, rays, rho_graph, _ = geom.build_all_geometry()
    profiles = load_profiles(
        dataset_dir = dataset_dir,
        n_profiles  = cfg.N_PROFILES,
        w_bundle    = w_bundle,
        grids       = grids,
    )

    return w_bundle, grids, rays, rho_graph, profiles
