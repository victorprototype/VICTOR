# ============================================================
# VICTOR v6.0 — data_loader.py
# W matrix loading, TORAX profile loading, noise injection,
# and field interpolation onto the reconstruction pixel grid.
# ============================================================
# Public API
# ----------
#   load_W_matrix(path)              → WBundle (namedtuple)
#   load_profiles(dataset_dir, ...)  → list[dict]
#   inject_noise(g, sigma, key)      → jnp.ndarray
#   interp_field(field, R_from, Z_from, R_to, Z_to) → np.ndarray
#
# Design principles
# -----------------
#  • Profile dicts contain plain jnp arrays — never passed as
#    dict items into @jax.jit (FIX from v5).
#  • W operators come from geometry.make_W_operators(); this
#    module only handles I/O (load, normalise, report).
#  • Field interpolation is NumPy / SciPy (runs once at load time).
# ============================================================

from __future__ import annotations

import os
import numpy as np
import scipy.sparse as sp_sci
from scipy.sparse import diags as spd
from scipy.interpolate import RegularGridInterpolator
from typing import NamedTuple, List, Dict, Any, Optional

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from victor import config as cfg
from victor import geometry as geom


# ── Named return type for the W bundle ───────────────────────────────

class WBundle(NamedTuple):
    """Everything derived from the W matrix file."""
    W_norm      : Any           # scipy.sparse.csr_matrix  (row-normalised)
    W_BCOO      : jnp.ndarray   # JAX BCOO sparse tensor
    ACTIVE_MASK : jnp.ndarray   # (128,) bool  — rows with non-zero sum
    ACTIVE_MASK_NP: np.ndarray  # same, as numpy (for Python-side indexing)
    N_ACTIVE    : int
    w_ops       : geom.WOperators


# ── 1. Load & normalise W matrix ──────────────────────────────────────

def load_W_matrix(path: Optional[str] = None) -> WBundle:
    """
    Load the W projection matrix, row-normalise, build BCOO and
    matrix-free operators.

    Parameters
    ----------
    path : str, optional
        Full path to W_matrix.npz.
        Defaults to ``cfg.DATASET_DIR/W_matrix.npz``.

    Returns
    -------
    WBundle
    """
    if path is None:
        path = os.path.join(cfg.DATASET_DIR, "W_matrix.npz")

    print(f"Loading W matrix from {path} ...")
    W_sp  = sp_sci.load_npz(path).tocsr()

    # Row sums → active-chord mask
    rs              = np.array(W_sp.sum(axis=1)).flatten()
    ACTIVE_MASK_NP  = (rs > 1e-8)
    rs_safe         = np.where(ACTIVE_MASK_NP, rs, 1.0)

    # Row-normalise (each chord sums to 1)
    W_norm          = spd(1.0 / rs_safe) @ W_sp
    ACTIVE_MASK     = jnp.array(ACTIVE_MASK_NP)
    N_ACTIVE        = int(ACTIVE_MASK_NP.sum())

    # BCOO sparse tensor (for reference / loss debugging)
    Wcoo   = W_norm.tocoo()
    W_BCOO = BCOO(
        (
            jnp.array(Wcoo.data,  dtype=jnp.float32),
            jnp.array(np.stack([Wcoo.row, Wcoo.col], axis=1), dtype=jnp.int32),
        ),
        shape=W_norm.shape,
    )

    # Matrix-free operators (JIT compiled)
    w_ops = geom.make_W_operators(W_norm.tocsr())

    print(
        f"  W: {W_norm.shape}  NNZ={W_norm.nnz}  "
        f"active={N_ACTIVE}/{W_sp.shape[0]}"
    )

    return WBundle(
        W_norm        = W_norm,
        W_BCOO        = W_BCOO,
        ACTIVE_MASK   = ACTIVE_MASK,
        ACTIVE_MASK_NP= ACTIVE_MASK_NP,
        N_ACTIVE      = N_ACTIVE,
        w_ops         = w_ops,
    )


# ── 2. Field interpolation helper ────────────────────────────────────

def interp_field(
    field  : np.ndarray,
    R_from : np.ndarray,
    Z_from : np.ndarray,
    R_to   : np.ndarray,
    Z_to   : np.ndarray,
) -> np.ndarray:
    """
    Bilinear interpolation of a 2-D field from one (R,Z) grid to another.

    Parameters
    ----------
    field  : (nR, nZ)  source field values
    R_from : (nR,)     source R axis
    Z_from : (nZ,)     source Z axis
    R_to   : (nR2, nZ2) target R coordinates
    Z_to   : (nR2, nZ2) target Z coordinates

    Returns
    -------
    np.ndarray of shape R_to.shape, dtype float32
        Out-of-bounds points are filled with 0.
    """
    fn  = RegularGridInterpolator(
        (R_from, Z_from), field,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
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
    Add Poisson + Gaussian noise to a sinogram measurement.

    The Poisson term models photon statistics; the Gaussian term
    models electronic / systematic noise scaled to the peak signal.

    Parameters
    ----------
    g     : (128,)  clean sinogram  [a.u.]
    sigma : float   Gaussian noise fraction of peak signal
    key   : JAX PRNG key
    ps    : float   photon scale factor (higher → less Poisson noise)

    Returns
    -------
    jnp.ndarray  (128,)  noisy sinogram, clipped to ≥ 0
    """
    k_poisson, k_gaussian = jax.random.split(key)
    lam = jnp.maximum(g * ps, 1.0)
    g_poisson = jnp.maximum(
        (lam + jnp.sqrt(lam) * jax.random.normal(k_poisson, lam.shape)) / ps,
        0.0,
    )
    return jnp.maximum(
        g_poisson + sigma * jnp.max(jnp.abs(g)) * jax.random.normal(k_gaussian, g.shape),
        0.0,
    )


# ── 4. Field normalisation helper ────────────────────────────────────

def _safe_norm_11(x: np.ndarray) -> np.ndarray:
    """Linearly rescale array to [-1, 1] (safe against zero-range)."""
    mn, mx = x.min(), x.max()
    return (2.0 * (x - mn) / (mx - mn + 1e-8) - 1.0).astype(np.float32)


# ── 5. Load TORAX equilibrium profiles ───────────────────────────────

def load_profiles(
    dataset_dir  : Optional[str]       = None,
    n_profiles   : int                  = cfg.N_PROFILES,
    w_bundle     : Optional[WBundle]    = None,
    grids        : Optional[geom.PixelGrids] = None,
) -> List[Dict[str, Any]]:
    """
    Load TORAX equilibrium profiles and pre-process for training.

    Each profile dict contains **plain jnp arrays** (never nested dicts
    passed into @jax.jit), plus numpy arrays for Python-side diagnostics.

    Parameters
    ----------
    dataset_dir : str, optional
        Directory containing profile_XXXX.npz files.
        Defaults to ``cfg.DATASET_DIR``.
    n_profiles  : int
        Number of profiles to load.
    w_bundle    : WBundle
        Used to compute ideal sinogram g_ideal = W · ε.
        If None, g_ideal is set to zeros.
    grids       : PixelGrids
        Pixel grid arrays for field interpolation.
        If None, built internally via geometry.build_pixel_grids().

    Returns
    -------
    list of dicts, each containing:
        # ── JAX arrays (safe to index in JIT via positional args) ──
        psi_n    : (N_GRID²,) float32   normalised ψ  field  ∈ [-1,1]
        bpol_n   : (N_GRID²,) float32   normalised Bpol field ∈ [-1,1]
        rho_1d   : (n_rho,)   float32   TORAX 1-D rho axis
        Te_1d    : (n_rho,)   float32   electron temperature [keV]
        ne_1d    : (n_rho,)   float32   electron density
        g_ideal  : (128,)     float32   clean projected sinogram

        # ── Scalars ────────────────────────────────────────────────
        g_scale  : float   max of raw sinogram (for de-normalisation)
        idx      : int     profile index

        # ── NumPy (monitoring / GS loss, never in JIT) ─────────────
        eps_n       : (N_GRID, N_GRID) float32  normalised emissivity
        psi_2d_np   : (nR, nZ)         float32  raw ψ on eq-grid
        R_grid_np   : (nR,)            float32  R axis of eq-grid
        Z_grid_np   : (nZ,)            float32  Z axis of eq-grid
    """
    if dataset_dir is None:
        dataset_dir = cfg.DATASET_DIR

    if grids is None:
        grids = geom.build_pixel_grids()

    R_pix_np = np.array(grids.R_PIX).reshape(cfg.N_GRID, cfg.N_GRID)
    Z_pix_np = np.array(grids.Z_PIX).reshape(cfg.N_GRID, cfg.N_GRID)

    # W matvec function (or a no-op if w_bundle not supplied)
    if w_bundle is not None:
        _W_matvec = w_bundle.w_ops.matvec
    else:
        def _W_matvec(ef):
            return jnp.zeros(128, dtype=jnp.float32)

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
        eps_n   = eps_raw / eps_max                              # (N_GRID, N_GRID)

        # Ideal sinogram
        g_raw   = _W_matvec(jnp.array(eps_n.flatten()))
        g_max   = float(g_raw.max()) + 1e-10
        g_ideal = g_raw / g_max

        # ── Equilibrium fields (eq-grid → recon-grid) ────────────────
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
            # ── JIT-safe JAX arrays (positional args to forward pass) ─
            psi_n   = jnp.array(_safe_norm_11(psi_grid).flatten()),    # (N²,)
            bpol_n  = jnp.array(_safe_norm_11(bpol_grid).flatten()),   # (N²,)
            rho_1d  = jnp.array(rho_1d),
            Te_1d   = jnp.array(Te_1d),
            ne_1d   = jnp.array(ne_1d),
            g_ideal = g_ideal,

            # ── Scalars ───────────────────────────────────────────────
            g_scale = g_max,
            idx     = profile_idx,

            # ── NumPy arrays (monitoring / GS loss, not in JIT) ──────
            eps_n       = eps_n,
            psi_2d_np   = psi_raw,
            R_grid_np   = R_g,
            Z_grid_np   = Z_g,
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
    dataset_dir: Optional[str] = None,
) -> tuple:
    """
    Full Cell 2 data pipeline in one call.

    Returns
    -------
    w_bundle : WBundle
    grids    : PixelGrids
    rays     : RayArrays
    rho_graph: RhoGraph
    profiles : list[dict]
    """
    if dataset_dir is None:
        dataset_dir = cfg.DATASET_DIR

    # W matrix
    w_bundle = load_W_matrix(os.path.join(dataset_dir, "W_matrix.npz"))

    # Geometry
    grids, rays, rho_graph, _ = geom.build_all_geometry(
        W_csr=None  # W operators already built inside load_W_matrix
    )

    # Profiles
    profiles = load_profiles(
        dataset_dir = dataset_dir,
        n_profiles  = cfg.N_PROFILES,
        w_bundle    = w_bundle,
        grids       = grids,
    )

    return w_bundle, grids, rays, rho_graph, profiles
