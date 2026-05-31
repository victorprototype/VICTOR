# ============================================================
# VICTOR v8.2 — geometry.py
# Pixel grids, ray coordinates, rho-proximity graph,
# W matrix-free matvec / vecmat operators,
# lerp weights, and boundary collocation indices.
# ============================================================
# Public API
# ----------
#   build_pixel_grids()          -> PixelGrids (namedtuple)
#   build_ray_coords()           -> RayArrays  (namedtuple)
#   build_rho_graph()            -> RhoGraph   (namedtuple)
#   make_W_operators()           -> WOperators (namedtuple)
#   build_lerp_weights()         -> (idx_lo, idx_hi, frac)   [NEW v8.2]
#   build_collocation_points()   -> jnp.int32 array          [NEW v8.2]
#   build_all_geometry()         -> (grids, rays, rho_graph, w_ops)
#
# v8.2 additions vs v7.2
# ----------------------
#  * build_lerp_weights(rho_flat, rho_radial)
#      Returns (idx_lo, idx_hi, frac) for differentiable linear
#      interpolation of any 1-D radial profile onto the pixel grid.
#      Uses jnp.searchsorted — no argmin, no discrete gradient kill.
#
#  * build_collocation_points(rho_flat, theta_flat, n_points, rho_threshold)
#      Samples pixel indices near the LCFS boundary (rho >= rho_threshold)
#      for use in the collocated boundary / PDE loss.  Returns a
#      (n_points,) int32 JAX array.  Pure JAX, no Python pixel loop.
#
#  * PixelGrids namedtuple gains five new fields:
#      LERP_IDX_LO        (N_GRID²,) int32   — lower radial bin index
#      LERP_IDX_HI        (N_GRID²,) int32   — upper radial bin index
#      LERP_FRAC          (N_GRID²,) float32 — interpolation fraction in [0, 1]
#      BOUNDARY_COLLOC_IDX (PDE_COLLOC_N,) int32 — boundary collocation pixel indices
#
#  * build_all_geometry() prints shapes of all new PixelGrids fields.
#
#  * All existing public API is fully preserved — no removals.
#
# v7.2 changes (retained for history)
# ------------------------------------
#  * Removed explicit device_put / _resolve_device / _dev tracking.
#  * Fixed vecmat n_pixels capture outside @jax.jit.
#  * NODE_DEG comment clarified.
#
# All heavy arrays are returned as JAX arrays (jnp.float32 / jnp.int32).
# No JAX globals are mutated; callers own the returned objects.
# ============================================================

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from victor import config as cfg


# ── Named return types ────────────────────────────────────────────────

class PixelGrids(NamedTuple):
    """
    All per-pixel geometry arrays.

    Shape (N_GRID*N_GRID,) unless noted.  Five new fields were added in
    v8.2: LERP_IDX_LO, LERP_IDX_HI, LERP_FRAC (differentiable radial
    interpolation helpers) and BOUNDARY_COLLOC_IDX (LCFS collocation
    sample indices).
    """
    RHO_2D              : jnp.ndarray   # (N_GRID, N_GRID)  normalised elliptic radius
    RHO_FLAT            : jnp.ndarray   # (N_GRID²,)        rho at every pixel
    THETA_FLAT          : jnp.ndarray   # (N_GRID²,)        atan2(Y, X)
    R_PIX               : jnp.ndarray   # (N_GRID²,)        major radius [m]
    Z_PIX               : jnp.ndarray   # (N_GRID²,)        vertical position [m]
    RHO_RADIAL          : jnp.ndarray   # (N_RADIAL,)       1-D radial axis
    # ── v8.2 additions ──────────────────────────────────────────────
    LERP_IDX_LO         : jnp.ndarray   # (N_GRID²,) int32   lower radial bin
    LERP_IDX_HI         : jnp.ndarray   # (N_GRID²,) int32   upper radial bin
    LERP_FRAC           : jnp.ndarray   # (N_GRID²,) float32 interp fraction ∈ [0,1]
    BOUNDARY_COLLOC_IDX : jnp.ndarray   # (PDE_COLLOC_N,) int32  LCFS pixel indices


class RayArrays(NamedTuple):
    """Padded ray-march arrays, shape (n_rays, MAX_STEPS)."""
    RAY_R     : jnp.ndarray   # major radius  [m]
    RAY_Z     : jnp.ndarray   # vertical coord [m]
    RAY_DS    : jnp.ndarray   # step length   [m]
    RAY_V     : jnp.ndarray   # valid-step mask (0/1)
    MAX_STEPS : int


class RhoGraph(NamedTuple):
    """Edge lists and weights for the rho-proximity graph."""
    EDGES_SRC : jnp.ndarray   # (E,) int32
    EDGES_DST : jnp.ndarray   # (E,) int32
    EDGE_W    : jnp.ndarray   # (E,) float32
    NODE_DEG  : jnp.ndarray   # (N_GRID²,) float32  >= 1
                               # counts in-edges (dst occurrences); not used in training


class WOperators(NamedTuple):
    """Matrix-free W operators built from a padded CSR representation."""
    W_IDX : jnp.ndarray    # (128, MX) int32   — column indices
    W_LEN : jnp.ndarray    # (128, MX) float32 — row values
    W_MSK : jnp.ndarray    # (128, MX) float32 — valid-entry mask
    matvec: object          # jit-compiled (ef -> projections)
    vecmat: object          # jit-compiled (v  -> back-projection)


# ── 1. Pixel grids ────────────────────────────────────────────────────

def build_pixel_grids(
    n_grid      : int   = cfg.N_GRID,
    ext         : float = cfg.EXT,
    r0          : float = cfg.R0,
    ap          : float = cfg.AP,
    bp          : float = cfg.BP,
    rho_threshold: float = cfg.PDE_COLLOC_RHO_THRESHOLD,
    n_colloc    : int   = cfg.PDE_COLLOC_N,
    device              = None,   # accepted for API compatibility; ignored
) -> PixelGrids:
    """
    Build the 128×128 reconstruction pixel grid, including v8.2 lerp
    weights and boundary collocation indices.

    Parameters
    ----------
    n_grid        : int    Pixels per side (default cfg.N_GRID = 128).
    ext           : float  Half-extent [m] of the square domain.
    r0            : float  Major radius of grid centre [m].
    ap            : float  Semi-axis in R for elliptic normalisation.
    bp            : float  Semi-axis in Z for elliptic normalisation.
    rho_threshold : float  Minimum rho for boundary collocation sampling
                           (default cfg.PDE_COLLOC_RHO_THRESHOLD = 0.9).
    n_colloc      : int    Number of boundary collocation points to sample
                           (default cfg.PDE_COLLOC_N).
    device        : ignored — kept for backward compatibility.

    Returns
    -------
    PixelGrids namedtuple.  All arrays are jnp.float32 except
    LERP_IDX_LO, LERP_IDX_HI, and BOUNDARY_COLLOC_IDX (jnp.int32).
    """
    lin    = np.linspace(-ext, ext, n_grid)
    XX, YY = np.meshgrid(lin, lin)   # 'xy' indexing: XX varies along cols (R), YY along rows (Z)

    rho_2d_np = np.sqrt((XX / ap)**2 + (YY / bp)**2).astype(np.float32)
    theta_np  = np.arctan2(YY, XX).flatten().astype(np.float32)
    r_pix_np  = (r0 + XX).flatten().astype(np.float32)
    z_pix_np  = YY.flatten().astype(np.float32)

    rho_flat   = jnp.array(rho_2d_np.flatten())
    rho_radial = jnp.linspace(0.0, cfg.RHO_MAX, cfg.N_RADIAL, dtype=jnp.float32)

    # v8.2: differentiable lerp weights
    idx_lo, idx_hi, frac = build_lerp_weights(rho_flat, rho_radial)

    # v8.2: boundary collocation indices
    theta_flat = jnp.array(theta_np)
    colloc_idx = build_collocation_points(
        rho_flat,
        theta_flat,
        n_points=n_colloc,
        rho_threshold=rho_threshold,
    )

    return PixelGrids(
        RHO_2D              = jnp.array(rho_2d_np),
        RHO_FLAT            = rho_flat,
        THETA_FLAT          = theta_flat,
        R_PIX               = jnp.array(r_pix_np),
        Z_PIX               = jnp.array(z_pix_np),
        RHO_RADIAL          = rho_radial,
        LERP_IDX_LO         = idx_lo,
        LERP_IDX_HI         = idx_hi,
        LERP_FRAC           = frac,
        BOUNDARY_COLLOC_IDX = colloc_idx,
    )


# ── 2. Ray coordinates (WEST-tuned adaptive march) ────────────────────

def build_ray_coords(
    cameras            = cfg.CAMERAS,
    ext    : float     = cfg.EXT,
    n_grid : int       = cfg.N_GRID,
    ap     : float     = cfg.AP,
    bp     : float     = cfg.BP,
    r0     : float     = cfg.R0,
    ds_max_factor: float = cfg.DS_MAX_FACTOR,
    ds_min_factor: float = cfg.DS_MIN_FACTOR,
    device             = None,   # accepted for API compatibility; ignored
) -> RayArrays:
    """
    Compute adaptive-step ray march for all WEST LOS chords.

    Each camera entry in `cameras` is (px, py, a_min_deg, a_max_deg, nc).
    The total number of rays equals the sum of all nc values, which must
    match the number of rows in the W matrix (128 for WEST).

    Parameters
    ----------
    cameras       : list of (px, py, a_min_deg, a_max_deg, nc) tuples.
    ext           : float  Half-extent [m].
    n_grid        : int    Pixels per side.
    ap, bp        : float  Ellipse semi-axes.
    r0            : float  Major radius [m].
    ds_max_factor : float  ds_max = (2*ext/n_grid) * ds_max_factor.
    ds_min_factor : float  ds_min = ds_max * ds_min_factor.
    device        : ignored — kept for backward compatibility.

    Returns
    -------
    RayArrays namedtuple.  All arrays are jnp.float32.
    """
    ds_max = (2.0 * ext / n_grid) * ds_max_factor
    ds_min = ds_max * ds_min_factor

    all_R, all_Z, all_ds = [], [], []

    for px, py, a_min_deg, a_max_deg, nc in cameras:
        for ang in np.linspace(np.radians(a_min_deg), np.radians(a_max_deg), nc):
            dx, dy = np.cos(ang), np.sin(ang)

            A_ = (dx / ap)**2 + (dy / bp)**2
            B_ = 2.0 * (px * dx / ap**2 + py * dy / bp**2)
            C_ = (px / ap)**2 + (py / bp)**2 - 1.0
            disc = B_**2 - 4.0 * A_ * C_

            if disc < 0:
                all_R.append([])
                all_Z.append([])
                all_ds.append([])
                continue

            t_in  = (-B_ - np.sqrt(disc)) / (2.0 * A_)
            t_out = (-B_ + np.sqrt(disc)) / (2.0 * A_)

            Rs, Zs, dss = [], [], []
            t = t_in
            while t < t_out:
                xm = px + t * dx
                ym = py + t * dy
                rho_loc = np.sqrt((xm / ap)**2 + (ym / bp)**2)

                ds = ds_min + (ds_max - ds_min) * (1.0 - min(rho_loc, 1.0))
                ds = min(ds, t_out - t)   # clamp so t lands exactly on t_out

                xc = px + (t + ds / 2.0) * dx
                yc = py + (t + ds / 2.0) * dy

                Rs.append(r0 + xc)
                Zs.append(yc)
                dss.append(ds)
                t += ds

            all_R.append(Rs)
            all_Z.append(Zs)
            all_ds.append(dss)

    n_rays = len(all_R)
    mx     = max((len(r) for r in all_R), default=1)

    RR = np.zeros((n_rays, mx), dtype=np.float32)
    ZZ = np.zeros((n_rays, mx), dtype=np.float32)
    DS = np.zeros((n_rays, mx), dtype=np.float32)
    VV = np.zeros((n_rays, mx), dtype=np.float32)

    for i in range(n_rays):
        n = len(all_R[i])
        if n > 0:
            RR[i, :n] = all_R[i]
            ZZ[i, :n] = all_Z[i]
            DS[i, :n] = all_ds[i]
            VV[i, :n] = 1.0

    return RayArrays(
        RAY_R     = jnp.array(RR),
        RAY_Z     = jnp.array(ZZ),
        RAY_DS    = jnp.array(DS),
        RAY_V     = jnp.array(VV),
        MAX_STEPS = mx,
    )


# ── 3. rho-proximity graph ────────────────────────────────────────────

def build_rho_graph(
    rho_flat : np.ndarray,
    n_nb     : int   = cfg.RHO_GRAPH_N_NB,
    sigma    : float = cfg.RHO_GRAPH_SIGMA,
    stride   : int   = cfg.RHO_GRAPH_STRIDE,
    device           = None,   # accepted for API compatibility; ignored
) -> RhoGraph:
    """
    Build a k-nearest-neighbour graph in rho-space.

    For each strided pixel i, finds the n_nb closest pixels j (by |rho_i - rho_j|)
    with edge weight w = exp(-|rho_i - rho_j| / sigma).  Edges with w < 0.05
    are pruned.

    Parameters
    ----------
    rho_flat : (N_GRID²,) numpy array of normalised elliptic radii.
    n_nb     : int    Number of nearest neighbours per node.
    sigma    : float  Laplacian kernel bandwidth (cfg.RHO_GRAPH_SIGMA = 0.04).
    stride   : int    Sub-sample stride (builds graph on every stride-th pixel).
    device   : ignored — kept for backward compatibility.

    Returns
    -------
    RhoGraph namedtuple.
    """
    rho = np.array(rho_flat)
    N   = len(rho)
    idx = np.arange(0, N, stride)   # strided subset of pixel indices

    src_list, dst_list, w_list = [], [], []

    for i in idx:
        ri  = rho[i]
        dr  = np.abs(rho[idx] - ri)          # distance to every strided pixel
        w   = np.exp(-dr / sigma)             # Laplacian RBF weights
        top = np.argsort(w)[-(n_nb + 1):]    # n_nb+1 highest-weight neighbours
        for j2 in top:
            j = idx[j2]                       # j2 indexes into idx; j is absolute pixel index
            if j != i and w[j2] > 0.05:
                src_list.append(i)
                dst_list.append(j)
                w_list.append(float(w[j2]))

    # NODE_DEG: count of in-edges per node (how many edges point into each pixel).
    # Equals out-degree only for symmetric graphs; this graph is not guaranteed
    # symmetric due to strided sub-sampling.  Not used in the training loop.
    deg = np.zeros(N, dtype=np.float32)
    for d_ in dst_list:
        deg[d_] += 1.0

    return RhoGraph(
        EDGES_SRC = jnp.array(src_list, dtype=jnp.int32),
        EDGES_DST = jnp.array(dst_list, dtype=jnp.int32),
        EDGE_W    = jnp.array(w_list,   dtype=jnp.float32),
        NODE_DEG  = jnp.array(np.maximum(deg, 1.0)),
    )


# ── 4. W matrix-free operators ────────────────────────────────────────

def make_W_operators(W_csr, device=None) -> WOperators:
    """
    Build padded CSR arrays and JIT-compile W*eps and W^T*v operators.

    Parameters
    ----------
    W_csr  : scipy.sparse.csr_matrix   row-normalised W (128 x N_PIXELS)
    device : ignored — kept for backward compatibility.

    Returns
    -------
    WOperators namedtuple.
    """
    n_rows   = W_csr.shape[0]
    n_pixels = int(W_csr.shape[1])   # captured as a plain int OUTSIDE @jax.jit
                                      # to avoid holding a live scipy reference
                                      # inside the JIT closure
    MX = int(np.diff(W_csr.indptr).max())

    IDX_ = np.zeros((n_rows, MX), dtype=np.int32)
    LEN_ = np.zeros((n_rows, MX), dtype=np.float32)
    MSK_ = np.zeros((n_rows, MX), dtype=np.float32)

    for i in range(n_rows):
        start, end = W_csr.indptr[i], W_csr.indptr[i + 1]
        n = end - start
        IDX_[i, :n] = W_csr.indices[start:end]
        LEN_[i, :n] = W_csr.data[start:end]
        MSK_[i, :n] = 1.0

    W_IDX = jnp.array(IDX_)
    W_LEN = jnp.array(LEN_)
    W_MSK = jnp.array(MSK_)

    @jax.jit
    def matvec(ef):
        """W * ef  ->  (n_rows,)  sinogram projection."""
        return jnp.sum(ef[W_IDX] * W_LEN * W_MSK, axis=1)

    @jax.jit
    def vecmat(v):
        """W^T * v  ->  (N_PIXELS,)  adjoint back-projection."""
        return (
            jnp.zeros(n_pixels, dtype=jnp.float32)
            .at[W_IDX.ravel()]
            .add((v[:, None] * W_LEN * W_MSK).ravel())
        )

    return WOperators(
        W_IDX  = W_IDX,
        W_LEN  = W_LEN,
        W_MSK  = W_MSK,
        matvec = matvec,
        vecmat = vecmat,
    )


# ── 5. [NEW v8.2] Differentiable radial lerp weights ─────────────────

def build_lerp_weights(
    rho_flat   : jnp.ndarray,
    rho_radial : jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute differentiable linear-interpolation weights mapping each pixel
    to its bracketing pair of radial bins.

    This replaces the argmin nearest-neighbour lookup formerly used in
    ``losses.build_eps2d``, which broke gradients by introducing a discrete
    selection.  Here, every output is a smooth function of ``rho_flat`` and
    can be differentiated through by JAX's autodiff without any stop-gradient.

    Implementation notes
    --------------------
    ``jnp.searchsorted(rho_radial, rho_flat, side='right')`` returns the
    insertion index ``k`` such that ``rho_radial[k-1] <= rho_flat < rho_radial[k]``.
    We then compute:

    .. code-block:: python

        idx_hi = clip(k,     1, N_RADIAL - 1)
        idx_lo = clip(k - 1, 0, N_RADIAL - 2)
        frac   = (rho_flat - rho_radial[idx_lo])
                 / (rho_radial[idx_hi] - rho_radial[idx_lo] + eps)

    ``frac`` is clamped to [0, 1] to handle pixels at or beyond the axis
    endpoints.  Because ``rho_radial`` is fixed at construction time, the
    integer indices are trace-time constants and contribute no gradient;
    the only differentiable path is through ``frac``, which is a smooth
    affine function of ``rho_flat``.

    Parameters
    ----------
    rho_flat   : (N_GRID²,) float32  JAX array of normalised elliptic radii
                 for every pixel, as returned by ``build_pixel_grids``.
    rho_radial : (N_RADIAL,) float32 JAX array — the 1-D radial axis on which
                 radial profiles are tabulated (e.g. ``PixelGrids.RHO_RADIAL``).

    Returns
    -------
    idx_lo : (N_GRID²,) int32   Index of the lower bracketing radial bin.
    idx_hi : (N_GRID²,) int32   Index of the upper bracketing radial bin.
    frac   : (N_GRID²,) float32 Linear interpolation fraction in [0, 1]:
             ``profile[idx_lo] * (1 - frac) + profile[idx_hi] * frac``
             recovers the linearly interpolated value at each pixel.
    """
    n_radial = rho_radial.shape[0]

    # searchsorted gives k such that rho_radial[k-1] <= rho_flat < rho_radial[k]
    k = jnp.searchsorted(rho_radial, rho_flat, side="right")   # (N_GRID²,) int

    idx_hi = jnp.clip(k,     1,           n_radial - 1).astype(jnp.int32)
    idx_lo = jnp.clip(k - 1, 0,           n_radial - 2).astype(jnp.int32)

    rho_lo = rho_radial[idx_lo]   # (N_GRID²,) float32
    rho_hi = rho_radial[idx_hi]   # (N_GRID²,) float32

    # Avoid division by zero at degenerate bin boundaries (shouldn't occur
    # with a uniform rho_radial, but guard defensively).
    eps  = jnp.finfo(jnp.float32).eps
    frac = (rho_flat - rho_lo) / (rho_hi - rho_lo + eps)
    frac = jnp.clip(frac, 0.0, 1.0).astype(jnp.float32)

    return idx_lo, idx_hi, frac


# ── 6. [NEW v8.2] Boundary collocation point sampler ─────────────────

def build_collocation_points(
    rho_flat      : jnp.ndarray,
    theta_flat    : jnp.ndarray,
    n_points      : int   = cfg.PDE_COLLOC_N,
    rho_threshold : float = cfg.PDE_COLLOC_RHO_THRESHOLD,
) -> jnp.ndarray:
    """
    Sample pixel indices near the Last Closed Flux Surface (LCFS) boundary
    for use in the collocated boundary / PDE loss term.

    Pixels qualify when their normalised elliptic radius satisfies
    ``rho >= rho_threshold`` (default 0.9).  Among all qualifying pixels,
    ``n_points`` are selected by sorting on ``theta_flat`` (poloidal angle)
    and taking evenly spaced samples.  This gives a roughly uniform
    angular distribution around the boundary without any Python-level loop
    over pixels.

    Design rationale
    ----------------
    *   No Python loop over pixels — uses ``jnp.where`` / ``jnp.argsort`` /
        ``jnp.take``, all of which are JAX-traceable.
    *   The returned index array is an **integer** index into the flat pixel
        layout; it is not differentiable and is not intended to be.
        Gradients flow through the *values* gathered at these indices
        (e.g. ``eps_flat[colloc_idx]``), not through the indices themselves.
    *   If fewer than ``n_points`` pixels satisfy the threshold, the
        selection wraps (``mode='wrap'`` in ``jnp.take``) to fill the
        requested size — the loss will see repeated pixels rather than
        silently shrinking the batch.

    Parameters
    ----------
    rho_flat      : (N_GRID²,) float32  Normalised elliptic radii (same as
                    ``PixelGrids.RHO_FLAT``).
    theta_flat    : (N_GRID²,) float32  Poloidal angles in radians (same as
                    ``PixelGrids.THETA_FLAT``).
    n_points      : int    Number of collocation points to return
                    (default ``cfg.PDE_COLLOC_N``).
    rho_threshold : float  Minimum rho to be considered a boundary pixel
                    (default ``cfg.PDE_COLLOC_RHO_THRESHOLD = 0.9``).

    Returns
    -------
    colloc_idx : (n_points,) int32  Pixel indices (into the flat N_GRID²
                 layout) of the selected boundary collocation points.
    """
    n_pixels = rho_flat.shape[0]
    all_idx  = jnp.arange(n_pixels, dtype=jnp.int32)

    # Replace non-qualifying pixels with large theta so they sort to the end.
    _LARGE = jnp.finfo(jnp.float32).max
    masked_theta = jnp.where(rho_flat >= rho_threshold, theta_flat, _LARGE)

    # Sort qualifying pixels by poloidal angle for uniform angular coverage.
    sorted_positions = jnp.argsort(masked_theta)          # (N_GRID²,) — positions of pixels
    sorted_idx       = all_idx[sorted_positions]          # pixel indices in theta order

    # Count qualifying pixels (needed to choose evenly spaced offsets).
    n_valid = jnp.sum(rho_flat >= rho_threshold)          # scalar int

    # Evenly spaced sample positions within [0, n_valid).
    # jnp.linspace produces float; cast to int for indexing.
    sample_pos = jnp.linspace(0, n_valid - 1, n_points, dtype=jnp.float32)
    sample_pos = jnp.clip(
        jnp.round(sample_pos).astype(jnp.int32),
        0, n_pixels - 1,
    )

    # Gather pixel indices; wrap so output is always exactly (n_points,).
    colloc_idx = jnp.take(sorted_idx, sample_pos, mode="wrap").astype(jnp.int32)
    return colloc_idx


# ── 7. Convenience: build everything in one call ──────────────────────

def build_all_geometry(W_csr=None, device=None):
    """
    Build pixel grids, ray coords, rho-graph, and optionally W operators.

    Parameters
    ----------
    W_csr  : scipy.sparse.csr_matrix | None
    device : ignored — kept for backward compatibility.

    Returns
    -------
    grids     : PixelGrids
    rays      : RayArrays
    rho_graph : RhoGraph
    w_ops     : WOperators | None
    """
    print("Building geometry...")

    print("  Building pixel grids...")
    grids = build_pixel_grids()
    print(f"    RHO_2D={grids.RHO_2D.shape}  RHO_RADIAL={grids.RHO_RADIAL.shape}")
    # v8.2 fields
    print(f"    LERP_IDX_LO={grids.LERP_IDX_LO.shape}  "
          f"LERP_IDX_HI={grids.LERP_IDX_HI.shape}  "
          f"LERP_FRAC={grids.LERP_FRAC.shape}")
    print(f"    BOUNDARY_COLLOC_IDX={grids.BOUNDARY_COLLOC_IDX.shape}")

    print("  Building WEST ray coords...")
    rays   = build_ray_coords()
    n_rays = rays.RAY_R.shape[0]
    mem_kb = n_rays * rays.MAX_STEPS * 4 * 4 / 1024
    print(f"    Rays={n_rays}  MaxSteps={rays.MAX_STEPS}  Mem~{mem_kb:.0f} KB")

    print("  Building rho-proximity graph...")
    rho_graph = build_rho_graph(np.array(grids.RHO_FLAT))
    print(f"    Edges={len(rho_graph.EDGES_SRC)}")

    w_ops = None
    if W_csr is not None:
        print("  Building W matrix-free operators...")
        w_ops = make_W_operators(W_csr)
        print(f"    W_IDX={w_ops.W_IDX.shape}  MX={w_ops.W_IDX.shape[1]}")

    return grids, rays, rho_graph, w_ops

# =======================================================================
# Flux-surface-following poloidal angle  (Grad-Shafranov geometry)
# =======================================================================

def compute_flux_angle(
    psi_2d   : np.ndarray,    # (NR, NZ) poloidal flux on equilibrium grid
    R_grid   : np.ndarray,    # (NR,)    R coordinates
    Z_grid   : np.ndarray,    # (NZ,)    Z coordinates
    R_flat   : np.ndarray,    # (N_GRID²,) pixel R coordinates
    Z_flat   : np.ndarray,    # (N_GRID²,) pixel Z coordinates
    n_theta  : int   = 128,   # contour integration resolution
    n_psi    : int   = 48,    # number of flux surface levels
) -> np.ndarray:              # (N_GRID²,) float32 ∈ [0, 2π)
    """
    Compute the flux-surface-following poloidal angle θ* for each pixel.

    Uses the Grad-Shafranov flux function ψ(R,Z) to define flux surfaces
    (ψ=const contours) and integrates arc length weighted by 1/|∇ψ| along
    each contour to obtain the straight field line angle θ*.

    This replaces the geometric angle θ=arctan(Z/R) in build_eps2d,
    ensuring that harmonic decomposition follows actual flux surface
    geometry (elongation, triangularity, Shafranov shift).

    Parameters
    ----------
    psi_2d  : (NR, NZ)    poloidal flux ψ(R,Z) from TORAX
    R_grid  : (NR,)       R coordinates [m]
    Z_grid  : (NZ,)       Z coordinates [m]
    R_flat  : (N_GRID²,)  pixel R values
    Z_flat  : (N_GRID²,)  pixel Z values
    n_theta : int         integration points along each contour
    n_psi   : int         number of flux surface levels

    Returns
    -------
    theta_flux : (N_GRID²,) float32  straight field line angle ∈ [0, 2π)
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ── 1. Smooth ψ to reduce finite-difference noise ────────────────────
    psi_smooth = gaussian_filter(psi_2d.astype(np.float64), sigma=1.0)

    # ── 2. Compute ∇ψ on equilibrium grid ────────────────────────────────
    dR = float(R_grid[1] - R_grid[0])
    dZ = float(Z_grid[1] - Z_grid[0])
    dpsi_dR      = np.gradient(psi_smooth, dR, axis=0)
    dpsi_dZ      = np.gradient(psi_smooth, dZ, axis=1)
    grad_psi_mag = np.sqrt(dpsi_dR**2 + dpsi_dZ**2) + 1e-10

    # ── 3. Interpolators for ψ and |∇ψ| ──────────────────────────────────
    psi_interp  = RegularGridInterpolator(
        (R_grid, Z_grid), psi_smooth,
        method='linear', bounds_error=False, fill_value=None
    )
    grad_interp = RegularGridInterpolator(
        (R_grid, Z_grid), grad_psi_mag,
        method='linear', bounds_error=False, fill_value=1.0
    )

    # ── 4. Interpolate ψ onto pixel grid ─────────────────────────────────
    R_flat_1d  = np.array(R_flat).ravel()
    Z_flat_1d  = np.array(Z_flat).ravel()
    query      = np.stack([R_flat_1d, Z_flat_1d], axis=-1)   # (N², 2)
    psi_pixel  = psi_interp(query).astype(np.float32)         # (N²,)

    # ── 5. Find magnetic axis (minimum ψ) ────────────────────────────────
    axis_idx = np.unravel_index(np.argmin(psi_smooth), psi_smooth.shape)
    R_axis   = float(R_grid[axis_idx[0]])
    Z_axis   = float(Z_grid[axis_idx[1]])

    # ── 6. Define flux surface levels ────────────────────────────────────
    psi_min    = float(psi_smooth.min())
    psi_lcfs   = float(psi_smooth.max()) * 0.95
    psi_levels = np.linspace(psi_min + 1e-4, psi_lcfs, n_psi)

    # ── 7. Extract contours and compute θ* via arc-length integration ────
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    cs = ax.contour(R_grid, Z_grid, psi_smooth.T, levels=psi_levels)
    plt.close(fig)

    # Build lookup: psi_level → (contour_R, contour_Z, theta_star)
    contour_table = {}   # psi_level → (pts (M,2), theta_star (M,))

    for i, level in enumerate(psi_levels):
        try:
            segs = cs.allsegs[i]   # list of (M, 2) arrays for this level
        except (AttributeError, IndexError):
            continue
        if not segs:
            continue
        # Keep longest segment
        pts = np.array(max(segs, key=len))        # (M, 2): col0=R, col1=Z
        

        if len(pts) < 4:
            continue

        # Interpolate |∇ψ| at contour points — vectorized
        grad_c = grad_interp(pts)                  # (M,)

        # Arc length elements — vectorized
        dpts = np.diff(pts, axis=0)                # (M-1, 2)
        dl   = np.sqrt((dpts**2).sum(axis=1))      # (M-1,)

        # Straight field line weight: 1/|∇ψ| averaged between adjacent pts
        w_avg     = 0.5 * (grad_c[:-1] + grad_c[1:])
        integrand = dl / (w_avg + 1e-10)           # (M-1,)

        cumulative = np.concatenate([[0.0], np.cumsum(integrand)])  # (M,)
        total      = cumulative[-1]
        if total < 1e-10:
            continue

        # Normalize to [0, 2π]
        theta_c = 2.0 * np.pi * cumulative / total  # (M,)

        # Set θ*=0 at outboard midplane (max R, Z≈Z_axis)
        # Find point with max R that is closest to Z=Z_axis
        outboard_score = (pts[:, 0] - R_axis) - 10.0 * np.abs(pts[:, 1] - Z_axis)
        ref_idx        = np.argmax(outboard_score)
        theta_c        = (theta_c - theta_c[ref_idx]) % (2.0 * np.pi)

        contour_table[level] = (pts, theta_c)

    # ── 8. Assign θ* to each pixel — fully vectorized ────────────────────
    if not contour_table:
        # Fallback to geometric angle if no contours found
        print("WARNING: compute_flux_angle — no contours found, using geometric angle")
        return (np.arctan2(Z_flat - Z_axis, R_flat - R_axis) % (2*np.pi)).astype(np.float32)

    psi_keys   = np.array(sorted(contour_table.keys()))   # (n_psi,)
    theta_flux = np.zeros(len(R_flat_1d), dtype=np.float32)

    for j_start in range(0, len(R_flat_1d), 512):
        j_end    = min(j_start + 512, len(R_flat_1d))
        R_chunk  = R_flat_1d[j_start:j_end]
        Z_chunk  = Z_flat_1d[j_start:j_end]
        psi_chunk= psi_pixel[j_start:j_end]

        # Find nearest flux surface level for each pixel in chunk
        # Vectorized: (chunk, n_psi) distance matrix
        nearest_idx = np.argmin(
            np.abs(psi_chunk[:, None] - psi_keys[None, :]), axis=1
        )  # (chunk,)

        for k in np.unique(nearest_idx):
            mask_k       = nearest_idx == k
            pts_k, th_k  = contour_table[psi_keys[k]]
            R_k          = R_chunk[mask_k]
            Z_k          = Z_chunk[mask_k]

            # Vectorized nearest-point on contour: (pixels, contour_pts)
            dist2 = (
                (R_k[:, None] - pts_k[None, :, 0])**2 +
                (Z_k[:, None] - pts_k[None, :, 1])**2
            )  # (pixels_in_group, M)
            nn_idx = np.argmin(dist2, axis=1)   # (pixels_in_group,)
            theta_flux[j_start:j_end][mask_k] = th_k[nn_idx].astype(np.float32)

    return theta_flux


def compute_flux_surface_bins(
    psi_flat : np.ndarray,   # (N_GRID²,) normalised flux ψ ∈ [-1, 1]
    rho_flat : np.ndarray,   # (N_GRID²,) normalised radius
    n_bins   : int = 32,     # number of flux surface bins
) -> np.ndarray:             # (N_GRID²,) int32 bin index, -1 = outside
    """
    Precompute flux surface bin membership for each pixel.

    Assigns each interior pixel (rho < 1.0) to one of n_bins flux surface
    bins based on its normalised ψ value.  Pixels outside the LCFS are
    assigned -1.

    Storing this precomputed array in the profile dict and passing it to
    loss_flux_surface() avoids recomputing bin membership every training
    step, making the loss ~20x faster.

    Parameters
    ----------
    psi_flat : (N_GRID²,)  normalised poloidal flux ∈ [-1, 1]
    rho_flat : (N_GRID²,)  normalised radius (from geometry.PixelGrids)
    n_bins   : int         number of flux surface bins

    Returns
    -------
    bin_idx : (N_GRID²,) int32  bin index ∈ [0, n_bins-1], -1 = outside LCFS
    """
    psi_norm = ((psi_flat + 1.0) / 2.0).astype(np.float32)   # → [0, 1]
    interior = rho_flat < 1.0
    bin_idx  = np.full(len(psi_flat), -1, dtype=np.int32)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        mask = interior & (psi_norm >= edges[i]) & (psi_norm < edges[i+1])
        bin_idx[mask] = i

    return bin_idx
