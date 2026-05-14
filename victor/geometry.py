# ============================================================
# VICTOR v6.0 — geometry.py
# Pixel grids, ray coordinates, rho-proximity graph,
# and W matrix-free matvec / vecmat operators.
# ============================================================
# Public API
# ----------
#   build_pixel_grids()  → PixelGrids (namedtuple)
#   build_ray_coords()   → RayArrays  (namedtuple)
#   build_rho_graph()    → RhoGraph   (namedtuple)
#   make_W_operators()   → WOperators (namedtuple)   [called from data_loader]
#
# All heavy arrays are returned as JAX arrays (jnp.float32 / jnp.int32).
# No JAX globals are mutated; callers own the returned objects.
# ============================================================

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple

from victor import config as cfg


# ── Named return types ────────────────────────────────────────────────

class PixelGrids(NamedTuple):
    """All per-pixel geometry arrays, shape (N_GRID*N_GRID,) unless noted."""
    RHO_2D   : jnp.ndarray   # (N_GRID, N_GRID)  normalised elliptic radius
    RHO_FLAT : jnp.ndarray   # (N_GRID²,)
    THETA_FLAT: jnp.ndarray  # (N_GRID²,)  atan2(Y, X)
    R_PIX    : jnp.ndarray   # (N_GRID²,)  major radius of each pixel [m]
    Z_PIX    : jnp.ndarray   # (N_GRID²,)  vertical position of each pixel [m]


class RayArrays(NamedTuple):
    """Padded ray-march arrays, shape (n_rays, MAX_STEPS)."""
    RAY_R     : jnp.ndarray   # major radius  [m]
    RAY_Z     : jnp.ndarray   # vertical coord [m]
    RAY_DS    : jnp.ndarray   # step length   [m]
    RAY_V     : jnp.ndarray   # valid-step mask (0/1)
    MAX_STEPS : int


class RhoGraph(NamedTuple):
    """Edge lists and weights for the PIGNO rho-proximity graph."""
    EDGES_SRC : jnp.ndarray   # (E,) int32
    EDGES_DST : jnp.ndarray   # (E,) int32
    EDGE_W    : jnp.ndarray   # (E,) float32
    NODE_DEG  : jnp.ndarray   # (N_GRID²,) float32  ≥ 1


class WOperators(NamedTuple):
    """Matrix-free W operators built from a padded CSR representation."""
    W_IDX : jnp.ndarray    # (128, MX) int32   — column indices
    W_LEN : jnp.ndarray    # (128, MX) float32 — row values
    W_MSK : jnp.ndarray    # (128, MX) float32 — valid-entry mask
    matvec: object          # jit-compiled (ef → projections)
    vecmat: object          # jit-compiled (v  → back-projection)


# ── 1. Pixel grids ────────────────────────────────────────────────────

def build_pixel_grids(
    n_grid: int = cfg.N_GRID,
    ext: float  = cfg.EXT,
    r0: float   = cfg.R0,
    ap: float   = cfg.AP,
    bp: float   = cfg.BP,
) -> PixelGrids:
    """
    Build the 128×128 reconstruction pixel grid.

    Returns
    -------
    PixelGrids
        RHO_2D    : (n_grid, n_grid) normalised elliptic radius ρ = √[(x/ap)²+(y/bp)²]
        RHO_FLAT  : (n_grid²,)
        THETA_FLAT: (n_grid²,)  arctan2(y, x)
        R_PIX     : (n_grid²,)  R = r0 + x   [m]
        Z_PIX     : (n_grid²,)  Z = y          [m]
    """
    lin       = np.linspace(-ext, ext, n_grid)
    XX, YY    = np.meshgrid(lin, lin)

    rho_2d_np   = np.sqrt((XX / ap)**2 + (YY / bp)**2).astype(np.float32)
    theta_np    = np.arctan2(YY, XX).flatten().astype(np.float32)
    r_pix_np    = (r0 + XX).flatten().astype(np.float32)
    z_pix_np    = YY.flatten().astype(np.float32)

    return PixelGrids(
        RHO_2D    = jnp.array(rho_2d_np),
        RHO_FLAT  = jnp.array(rho_2d_np.flatten()),
        THETA_FLAT= jnp.array(theta_np),
        R_PIX     = jnp.array(r_pix_np),
        Z_PIX     = jnp.array(z_pix_np),
    )


# ── 2. Ray coordinates (WEST-tuned adaptive march) ────────────────────

def build_ray_coords(
    cameras          = cfg.CAMERAS,
    ext: float       = cfg.EXT,
    n_grid: int      = cfg.N_GRID,
    ap: float        = cfg.AP,
    bp: float        = cfg.BP,
    r0: float        = cfg.R0,
    ds_max_factor: float = cfg.DS_MAX_FACTOR,
    ds_min_factor: float = cfg.DS_MIN_FACTOR,
) -> RayArrays:
    """
    Compute adaptive-step ray march for all WEST LOS chords.

    The march uses a finer step near the axis (where gradients are
    steepest) and a coarser step at the plasma boundary.

    Parameters
    ----------
    cameras : list of (px, py, angle_min_deg, angle_max_deg, n_rays)

    Returns
    -------
    RayArrays
        RAY_R, RAY_Z, RAY_DS, RAY_V : (n_rays_total, MAX_STEPS)
        MAX_STEPS                    : int
    """
    ds_max = (2.0 * ext / n_grid) * ds_max_factor
    ds_min = ds_max * ds_min_factor

    all_R, all_Z, all_ds = [], [], []

    for px, py, a_min_deg, a_max_deg, nc in cameras:
        for ang in np.linspace(np.radians(a_min_deg), np.radians(a_max_deg), nc):
            dx, dy = np.cos(ang), np.sin(ang)

            # Ellipse entry/exit: A t² + B t + C = 0
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

                # Step size: smaller near axis, larger near edge
                ds = ds_min + (ds_max - ds_min) * (1.0 - min(rho_loc, 1.0))
                ds = min(ds, t_out - t)

                # Midpoint quadrature
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
    mx = max((len(r) for r in all_R), default=1)

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


# ── 3. rho-proximity graph for PIGNO ─────────────────────────────────

def build_rho_graph(
    rho_flat  : np.ndarray,
    n_nb      : int   = cfg.RHO_GRAPH_N_NB,
    sigma     : float = cfg.RHO_GRAPH_SIGMA,
    stride    : int   = cfg.RHO_GRAPH_STRIDE,
) -> RhoGraph:
    """
    Build a k-nearest-neighbour graph in rho-space for the PIGNO layers.

    Uses sub-sampling (stride) to keep edge count tractable, then
    weights edges by an RBF kernel exp(-|ρᵢ - ρⱼ| / σ).

    Parameters
    ----------
    rho_flat : (N,) numpy array of normalised radii
    n_nb     : number of neighbours per node
    sigma    : RBF bandwidth in ρ-space
    stride   : sub-sample nodes every `stride` pixels

    Returns
    -------
    RhoGraph
        EDGES_SRC, EDGES_DST : (E,) int32
        EDGE_W               : (E,) float32
        NODE_DEG             : (N,) float32  ≥ 1
    """
    rho = np.array(rho_flat)
    N   = len(rho)
    idx = np.arange(0, N, stride)

    src_list, dst_list, w_list = [], [], []

    for i in idx:
        ri  = rho[i]
        dr  = np.abs(rho[idx] - ri)
        w   = np.exp(-dr / sigma)
        top = np.argsort(w)[-(n_nb + 1):]   # highest weights (incl. self)
        for j2 in top:
            j = idx[j2]
            if j != i and w[j2] > 0.05:
                src_list.append(i)
                dst_list.append(j)
                w_list.append(float(w[j2]))

    # Node degree (for normalisation inside PIGNO)
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

def make_W_operators(W_csr) -> WOperators:
    """
    Build padded CSR arrays and JIT-compile W·ε and Wᵀ·v operators.

    The padded representation avoids Python loops at inference time:
    every row is padded to MX (the max nnz per row), controlled by a
    binary mask W_MSK.

    Parameters
    ----------
    W_csr : scipy.sparse.csr_matrix
        Row-normalised W matrix (128 × N_PIXELS).

    Returns
    -------
    WOperators
        W_IDX, W_LEN, W_MSK : JAX arrays
        matvec               : jit fn  ef → (128,)
        vecmat               : jit fn  v  → (N_PIXELS,)
    """
    n_rows = W_csr.shape[0]
    MX     = int(np.diff(W_csr.indptr).max())

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
        """W · ef   →   (n_rows,)  sinogram projection."""
        return jnp.sum(ef[W_IDX] * W_LEN * W_MSK, axis=1)

    @jax.jit
    def vecmat(v):
        """Wᵀ · v   →   (N_PIXELS,)  adjoint back-projection."""
        n_pixels = W_csr.shape[1]
        return (
            jnp.zeros(n_pixels)
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


# ── Convenience: build everything in one call ─────────────────────────

def build_all_geometry(W_csr=None):
    """
    Build pixel grids, ray coords, and rho-graph.
    Optionally also build W operators if W_csr is provided.

    Returns
    -------
    grids    : PixelGrids
    rays     : RayArrays
    rho_graph: RhoGraph
    w_ops    : WOperators | None
    """
    print("Building pixel grids...")
    grids = build_pixel_grids()
    print(f"  RHO_2D={grids.RHO_2D.shape}  R_PIX={grids.R_PIX.shape}")

    print("Building WEST ray coords...")
    rays = build_ray_coords()
    n_rays = rays.RAY_R.shape[0]
    mem_kb = n_rays * rays.MAX_STEPS * 4 * 4 / 1024
    print(f"  Rays={n_rays}  MaxSteps={rays.MAX_STEPS}  Mem≈{mem_kb:.0f} KB")

    print("Building rho-proximity graph...")
    rho_graph = build_rho_graph(np.array(grids.RHO_FLAT))
    print(f"  Edges={len(rho_graph.EDGES_SRC)}")

    w_ops = None
    if W_csr is not None:
        print("Building W matrix-free operators...")
        w_ops = make_W_operators(W_csr)
        print(f"  W_IDX={w_ops.W_IDX.shape}  MX={w_ops.W_IDX.shape[1]}")

    return grids, rays, rho_graph, w_ops
