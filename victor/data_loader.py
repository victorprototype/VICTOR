# ============================================================
# VICTOR v8.2 — data_loader.py
# W matrix loading, TORAX profile loading, noise injection,
# and field interpolation onto the reconstruction pixel grid.
# ============================================================
# Public API
# ----------
#   load_W_matrix(path, device)                        -> WBundle
#   load_profiles(dataset_dir, ..., device)            -> list[dict]
#   inject_noise(g, sigma, key, ps, mode)              -> jnp.ndarray
#   interp_field(field, R_from, Z_from, R_to, Z_to)   -> np.ndarray
#   load_cell2(dataset_dir, device)                    -> tuple
#
# v8.2 changes vs v8.1
# --------------------
#  * WBundle gains W_dense_np : np.ndarray (128, N_PIXELS) float32.
#    Dense W for notebook diagnostics — never used inside JIT.
#
#  * load_profiles — each profile dict now includes four PixelGrids
#    fields required by losses.build_eps2d() and losses.loss_boundary():
#      lerp_idx_lo      : grids.LERP_IDX_LO
#      lerp_idx_hi      : grids.LERP_IDX_HI
#      lerp_frac        : grids.LERP_FRAC
#      boundary_colloc  : grids.BOUNDARY_COLLOC_IDX
#    These are shared across all profiles (geometry is fixed) but
#    stored per-profile so that the trainer can batch profiles as plain
#    dicts without a separate geometry side-channel.
#
#  * load_profiles — new helper _build_eq_channels() stacks equilibrium
#    fields into a single (N², n_eq_channels) tensor stored as
#    `eq_channels` in the profile dict.  Individual fields psi_n /
#    bpol_n are preserved for backward compatibility.
#
#  * inject_noise — new `mode` argument:
#      'both'     (default) — Poisson + Gaussian (existing behaviour)
#      'poisson'  — Poisson only
#      'gaussian' — Gaussian only
#    Curriculum warmup in trainer.py can start with mode='gaussian'
#    before switching to mode='both' once the loss landscape stabilises.
#
#  * load_cell2 — grids is now explicitly passed into load_profiles so
#    the lerp/collocation arrays are populated without a second
#    build_pixel_grids() call inside load_profiles.
#
# v7.2 changes vs v7.1 (retained for history)
# --------------------------------------------
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
from typing import Any, Dict, List, Literal, NamedTuple, Optional

import numpy as np
import scipy.sparse as sp_sci
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags as spd

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from victor import config as cfg
from victor import geometry as geom
from victor import geometry


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
    """
    Everything derived from the W projection matrix file.

    Fields
    ------
    W_norm        : scipy.sparse.csr_matrix (row-normalised)
    W_BCOO        : JAX BCOO sparse tensor  — used inside JIT for
                    differentiable forward projection.
    ACTIVE_MASK   : (128,) bool jnp array  — rows with non-zero row sum.
    ACTIVE_MASK_NP: (128,) bool np array   — same, CPU copy for masking
                    outside JAX (e.g. numpy-side loss diagnostics).
    N_ACTIVE      : int — number of active detector rows.
    w_ops         : geom.WOperators — matrix-free matvec / rmatvec.
    W_dense_np    : (128, N_PIXELS) float32 np.ndarray — dense W for
                    notebook sinogram reconstruction plots and debugging.
                    NEVER pass into JIT; use W_BCOO or w_ops instead.
    """
    W_norm        : Any             # scipy.sparse.csr_matrix (row-normalised)
    W_BCOO        : jnp.ndarray     # JAX BCOO sparse tensor
    ACTIVE_MASK   : jnp.ndarray     # (128,) bool — rows with non-zero sum
    ACTIVE_MASK_NP: np.ndarray      # same, as numpy
    N_ACTIVE      : int
    w_ops         : geom.WOperators
    W_dense_np    : np.ndarray      # (128, N_PIXELS) float32 — diagnostics only


# ── 1. Load & normalise W matrix ──────────────────────────────────────

def load_W_matrix(
    path   : Optional[str] = None,
    device = None,   # accepted for API compatibility; ignored
) -> WBundle:
    """
    Load the W projection matrix, row-normalise, build BCOO and
    matrix-free operators, and materialise a dense copy for diagnostics.

    Parameters
    ----------
    path   : str | None
        Path to W_matrix.npz.  Defaults to cfg.DATASET_DIR/W_matrix.npz.
    device : ignored — kept for backward compatibility.

    Returns
    -------
    WBundle
        Named tuple with sparse (JAX + scipy) and dense representations
        of W, the active-row mask, and matrix-free operators.
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

    # Dense copy for diagnostic use only (notebooks, plots, non-JAX checks).
    # Materialise from the normalised sparse matrix so it stays consistent
    # with W_BCOO and w_ops.  Cast to float32 to match profile array dtypes.
    W_dense_np = W_norm.toarray().astype(np.float32)

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
        W_dense_np     = W_dense_np,
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

    Parameters
    ----------
    field  : (nR_from, nZ_from) float32 — source field values.
    R_from : (nR_from,) 1-D R coordinates of source grid.
    Z_from : (nZ_from,) 1-D Z coordinates of source grid.
    R_to   : (...) target R coordinates (arbitrary shape).
    Z_to   : (...) target Z coordinates (same shape as R_to).

    Returns
    -------
    np.ndarray of shape R_to.shape, dtype float32.
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
    mode  : Literal["both", "poisson", "gaussian"] = "both",
) -> jnp.ndarray:
    """
    Add noise to a sinogram for data-augmentation during training.

    Three modes are provided to support curriculum noise scheduling:

    ``'gaussian'``  (early training / warmup)
        Pure additive Gaussian noise.  Poisson statistics are skipped,
        so gradients are smoother and the loss landscape is easier to
        optimise when the network is still far from convergence.

    ``'poisson'``   (mid-curriculum)
        Poisson shot noise only, no additional Gaussian term.  Useful
        for isolating photon-counting effects without the extra
        Gaussian component masking convergence diagnostics.

    ``'both'``      (default, full physics)
        Poisson + Gaussian, matching the detector forward model used
        in production.  Use once the Gaussian-only warmup has
        stabilised the loss.

    NOTE: ``key`` must already be on the same device as ``g``.
    Use ``jax.device_put(jax.random.PRNGKey(seed), device)`` at the
    call site (trainer.py) — a CPU key forces this entire function to
    run on CPU.

    Parameters
    ----------
    g     : (128,) clean sinogram [a.u.]
    sigma : float  Gaussian noise fraction of peak signal.
                   Ignored when mode='poisson'.
    key   : JAX PRNG key — must be on the same device as g.
    ps    : float  Photon scale factor (higher → less Poisson noise).
                   Ignored when mode='gaussian'.
    mode  : {'both', 'poisson', 'gaussian'}  Noise model selector.

    Returns
    -------
    jnp.ndarray (128,) noisy sinogram, clipped >= 0.
    """
    k_poisson, k_gaussian = jax.random.split(key)

    if mode == "gaussian":
        # Gaussian-only: bypass Poisson branch entirely.
        # Used in early-curriculum warmup to keep gradients smooth.
        return jnp.maximum(
            g + sigma * jnp.max(jnp.abs(g)) * jax.random.normal(k_gaussian, g.shape),
            0.0,
        )

    # Poisson branch (shared by 'poisson' and 'both')
    lam = jnp.maximum(g * ps, 1.0)
    g_poisson = jnp.maximum(
        (lam + jnp.sqrt(lam) * jax.random.normal(k_poisson, lam.shape)) / ps,
        0.0,
    )

    if mode == "poisson":
        return g_poisson

    # mode == 'both': add Gaussian on top of Poisson output
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


# ── 5. Equilibrium channel stacking helper ───────────────────────────

def _build_eq_channels(
    d           : Any,
    R_pix_np    : np.ndarray,
    Z_pix_np    : np.ndarray,
    n_eq_channels: int = 2,
) -> np.ndarray:
    """
    Build a stacked equilibrium-field tensor on the pixel grid.

    Each channel is independently interpolated from the native
    equilibrium grid and rescaled to [-1, 1] before stacking.
    Extending to n_eq_channels=3 adds Te interpolated onto the pixel
    grid as a third input channel to the encoder (no code changes
    needed in load_profiles — only this function needs updating).

    Parameters
    ----------
    d             : npz archive dict (opened via np.load).
    R_pix_np      : (N_GRID, N_GRID) R coordinates of pixel grid.
    Z_pix_np      : (N_GRID, N_GRID) Z coordinates of pixel grid.
    n_eq_channels : int  Number of equilibrium channels to stack.
                    2 → [psi_n, bpol_n]  (default, backward-compatible)
                    3 → [psi_n, bpol_n, Te_grid_n]

    Returns
    -------
    np.ndarray  shape (N_GRID*N_GRID, n_eq_channels), dtype float32.
                Column order matches the channel index above.

    Raises
    ------
    ValueError  if n_eq_channels is not 2 or 3.
    """
    if n_eq_channels not in (2, 3):
        raise ValueError(
            f"n_eq_channels must be 2 or 3, got {n_eq_channels}"
        )

    R_g      = d["R_grid"].astype(np.float32)
    Z_g      = d["Z_grid"].astype(np.float32)
    psi_raw  = d["psi_2d"].astype(np.float32)
    Bpol_raw = d["B_pol"].astype(np.float32)

    psi_grid  = interp_field(psi_raw,  R_g, Z_g, R_pix_np, Z_pix_np)
    bpol_grid = interp_field(Bpol_raw, R_g, Z_g, R_pix_np, Z_pix_np)

    channels = [
        _safe_norm_11(psi_grid).flatten(),   # channel 0: ψ_n
        _safe_norm_11(bpol_grid).flatten(),  # channel 1: B_pol_n
    ]

    if n_eq_channels == 3:
        # Te is defined on rho; interpolate onto the pixel-grid rho map
        # via a 1-D rho→Te lookup projected through psi→rho mapping.
        # For now we interpolate the raw 2-D Te field if present,
        # falling back to a zero channel if the archive predates it.
        if "Te_2d" in d:
            Te_raw  = d["Te_2d"].astype(np.float32)
            Te_grid = interp_field(Te_raw, R_g, Z_g, R_pix_np, Z_pix_np)
        else:
            Te_grid = np.zeros_like(psi_grid)
        channels.append(_safe_norm_11(Te_grid).flatten())  # channel 2: Te_n

    # Stack → (N², n_eq_channels)
    return np.stack(channels, axis=-1).astype(np.float32)


# ── 6. Load TORAX equilibrium profiles ───────────────────────────────

def load_profiles(
    dataset_dir   : Optional[str]             = None,
    n_profiles    : int                       = cfg.N_PROFILES,
    w_bundle      : Optional[WBundle]         = None,
    grids         : Optional[geom.PixelGrids] = None,
    xi            : Optional[jnp.ndarray]     = None,
    n_eq_channels : int                       = 2,
    device        = None,   # accepted for API compatibility; ignored
) -> List[Dict[str, Any]]:
    """
    Load TORAX equilibrium profiles and pre-process for training.

    Each returned dict is safe to pass directly to the trainer as a
    flat collection of JAX arrays and Python scalars — no nested dicts,
    no scipy objects.

    Parameters
    ----------
    dataset_dir   : str | None
        Directory containing profile_XXXX.npz files.
        Defaults to cfg.DATASET_DIR.
    n_profiles    : int
        Number of profiles to load.  Stops early if files are missing.
    w_bundle      : WBundle | None
        Provides w_ops.matvec for ideal sinogram construction.
        If None, g_ideal is set to zeros (useful for geometry-only tests).
    grids         : PixelGrids | None
        Pixel grid geometry.  If None, built via geom.build_pixel_grids().
        Pass the same grids object used by the trainer to avoid redundant
        geometry recomputation.
    xi            : jnp.ndarray | None
        9-D normalised WEST GEM hardware vector.  Defaults to XI_DEFAULT.
    n_eq_channels : int
        Number of equilibrium input channels to stack in `eq_channels`.
        2 (default) → [psi_n, bpol_n].  3 → adds Te channel.
    device        : ignored — kept for backward compatibility.

    Returns
    -------
    list[dict]

    JAX arrays (safe inside @jax.jit):
        psi_n            : (N²,)               ψ_n on pixel grid [-1,1]
        bpol_n           : (N²,)               B_pol_n on pixel grid [-1,1]
        eq_channels      : (N², n_eq_channels) stacked equilibrium fields
        rho_1d           : (K,)                normalised flux coordinate
        Te_1d            : (K,)                electron temperature [keV]
        ne_1d            : (K,)                electron density [m⁻³]
        g_ideal          : (128,)              clean sinogram (W @ eps_n)
        xi               : (9,)                hardware descriptor vector
        lerp_idx_lo      : (N²,) int32  — lower rho-bin index per pixel;
                           consumed by losses.build_eps2d() to look up
                           Te/ne values during differentiable ε assembly.
        lerp_idx_hi      : (N²,) int32  — upper rho-bin index per pixel;
                           paired with lerp_idx_lo for linear interpolation.
        lerp_frac        : (N²,) float32 — fractional position between
                           lerp_idx_lo and lerp_idx_hi; avoids recomputing
                           the rho→profile mapping inside each JIT step.
        boundary_colloc  : (PDE_COLLOC_N,) int32 — flat pixel indices on
                           the plasma boundary; used by losses.loss_boundary()
                           to enforce the zero-emissivity Dirichlet condition.

    Python scalars:
        g_scale : float  — sinogram peak before normalisation (for de-norm).
        idx     : int    — profile index (0-based, for logging/debugging).

    NumPy arrays (monitoring only — never pass into @jax.jit):
        eps_n      : (N_GRID, N_GRID) float32 — min-max normalised emissivity.
        psi_2d_np  : (nR, nZ) float32         — raw ψ on equilibrium grid.
        R_grid_np  : (nR,)   float32          — R axis of equilibrium grid.
        Z_grid_np  : (nZ,)   float32          — Z axis of equilibrium grid.
    """
    if dataset_dir is None:
        dataset_dir = cfg.DATASET_DIR
    if xi is None:
        xi = XI_DEFAULT
    if grids is None:
        grids = geom.build_pixel_grids()

    R_pix_np = np.array(grids.R_PIX).reshape(cfg.N_GRID, cfg.N_GRID)
    Z_pix_np = np.array(grids.Z_PIX).reshape(cfg.N_GRID, cfg.N_GRID)

    # Pre-convert PixelGrids interpolation arrays to JAX once, outside
    # the profile loop — geometry is identical for every profile.
    lerp_idx_lo     = jnp.array(grids.LERP_IDX_LO,        dtype=jnp.int32)
    lerp_idx_hi     = jnp.array(grids.LERP_IDX_HI,        dtype=jnp.int32)
    lerp_frac       = jnp.array(grids.LERP_FRAC,          dtype=jnp.float32)
    boundary_colloc = jnp.array(grids.BOUNDARY_COLLOC_IDX, dtype=jnp.int32)

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
        # Min-max normalise to [0, 1] to preserve full spatial contrast.
        # A max-only normalisation (v7.x) compressed narrow-range fields
        # like ε ∈ [1.41, 1.61] to [0.875, 1.0], destroying gradients.
        eps_raw = d["epsilon_2d"].astype(np.float32)
        eps_min = float(eps_raw.min())
        eps_max = float(eps_raw.max()) + 1e-10
        eps_n   = (eps_raw - eps_min) / (eps_max - eps_min) # (N_GRID, N_GRID)

        # Ideal sinogram via w_ops.matvec — normalise to [0, 1] so
        # g_scale retains the absolute amplitude for de-normalisation.
        g_raw   = _W_matvec(_put(eps_n.flatten()))
        g_max   = float(g_raw.max()) + 1e-10
        g_ideal = g_raw / g_max                              # (128,)

        # ── Equilibrium fields (eq-grid → recon-grid, CPU) ───────────
        # Interpolated on CPU at load time; never recomputed in JIT.
        R_g      = d["R_grid"].astype(np.float32)
        Z_g      = d["Z_grid"].astype(np.float32)
        psi_raw  = d["psi_2d"].astype(np.float32)
        Bpol_raw = d["B_pol"].astype(np.float32)

        psi_grid  = interp_field(psi_raw,  R_g, Z_g, R_pix_np, Z_pix_np)
        bpol_grid = interp_field(Bpol_raw, R_g, Z_g, R_pix_np, Z_pix_np)

        # ── Flux-surface-following angle (Grad-Shafranov geometry) ───
        # Precomputed once per profile — replaces geometric theta in
        # build_eps2d for physically correct poloidal decomposition.
        theta_flux_np = geometry.compute_flux_angle(
            psi_2d  = psi_raw.astype(np.float64),
            R_grid  = R_g.astype(np.float64),
            Z_grid  = Z_g.astype(np.float64),
            R_flat  = R_pix_np.astype(np.float64),
            Z_flat  = Z_pix_np.astype(np.float64),
        )
        flux_bin_idx_np = geometry.compute_flux_surface_bins(
            psi_flat = np.array(_safe_norm_11(psi_grid).flatten()),
            rho_flat = np.array(grids.RHO_FLAT),
        )

        # ── Stacked equilibrium channels ─────────────────────────────
        # eq_channels is the primary equilibrium input to the encoder.
        # Individual psi_n / bpol_n are retained for backward compat
        # with any code that accesses them by name directly.
        eq_ch = _build_eq_channels(d, R_pix_np, Z_pix_np, n_eq_channels)

        # ── 1-D TORAX profiles ────────────────────────────────────────
        rho_1d = d["rho"].astype(np.float32)
        Te_1d  = d["Te"].astype(np.float32)
        ne_1d  = d["ne"].astype(np.float32)

        profiles.append(dict(
            # ── JAX arrays — safe inside @jax.jit ────────────────────

            # Individual equilibrium fields (backward compatibility)
            psi_n   = _put(_safe_norm_11(psi_grid).flatten()),   # (N²,)
            bpol_n  = _put(_safe_norm_11(bpol_grid).flatten()),  # (N²,)

            # Stacked equilibrium channels for encoder input
            eq_channels = _put(eq_ch),                           # (N², C)

            # 1-D radial profiles for ε assembly in losses.build_eps2d()
            rho_1d  = _put(rho_1d),
            Te_1d   = _put(Te_1d),
            ne_1d   = _put(ne_1d),

            # Projection target — used in losses.loss_projection()
            g_ideal = g_ideal,                                   # (128,)

            # Hardware descriptor — fed to the instrument encoder branch
            xi      = jnp.array(xi),                             # (9,)

            # Differentiable rho interpolation indices/weights.
            # Stored per-profile so the trainer dict-batch contains
            # everything needed for a JIT step with no geometry side-channel.
            # losses.build_eps2d() indexes Te_1d/ne_1d with these to
            # assemble ε on the pixel grid in a single gather + lerp.
            lerp_idx_lo     = lerp_idx_lo,      # (N²,) int32
            lerp_idx_hi     = lerp_idx_hi,      # (N²,) int32
            lerp_frac       = lerp_frac,         # (N²,) float32

            # Boundary collocation indices for the zero-ε Dirichlet BC.
            # losses.loss_boundary() gathers ε at these pixel positions
            # and penalises any non-zero predicted emissivity outside
            # the last closed flux surface.
            boundary_colloc = boundary_colloc,   # (PDE_COLLOC_N,) int32

            # Flux-surface geometry (Grad-Shafranov precomputed)
            theta_flux   = _put(theta_flux_np),          # (N²,) float32
            flux_bin_idx = jnp.array(flux_bin_idx_np),   # (N²,) int32

            # ── Python scalars ────────────────────────────────────────
            g_scale = g_max,        # sinogram peak before normalisation
            idx     = profile_idx,  # profile index for logging

            # ── NumPy (monitoring only — never pass into @jax.jit) ────
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


# ── 7. Convenience: load everything for Cell 2 ───────────────────────

def load_cell2(
    dataset_dir : Optional[str] = None,
    device      = None,   # accepted for API compatibility; ignored
) -> tuple:
    """
    Full data pipeline in one call.

    Builds geometry, loads W, and loads all profiles.  The same
    PixelGrids object is passed into load_profiles so that the
    lerp/collocation arrays are populated without a redundant
    build_pixel_grids() call inside load_profiles.

    Returns
    -------
    w_bundle  : WBundle      — W matrix, BCOO, mask, operators, dense W.
    grids     : PixelGrids   — pixel grid coords + lerp/colloc arrays.
    rays      : RayArrays    — ray geometry for forward model.
    rho_graph : RhoGraph     — flux-surface graph for PDE losses.
    profiles  : list[dict]   — pre-processed profile dicts (see load_profiles).
    """
    if dataset_dir is None:
        dataset_dir = cfg.DATASET_DIR

    w_bundle = load_W_matrix(os.path.join(dataset_dir, "W_matrix.npz"))
    grids, rays, rho_graph, _ = geom.build_all_geometry()

    # Pass grids explicitly so load_profiles does not call build_pixel_grids()
    # a second time and so lerp/boundary arrays are guaranteed consistent
    # with the grids object returned to the caller.
    profiles = load_profiles(
        dataset_dir = dataset_dir,
        n_profiles  = cfg.N_PROFILES,
        w_bundle    = w_bundle,
        grids       = grids,
    )

    return w_bundle, grids, rays, rho_graph, profiles
