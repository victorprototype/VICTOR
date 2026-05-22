# ============================================================
# VICTOR v7.0 — geometry.py
# Pixel grids, ray coordinates, rho-proximity graph,
# and W matrix-free matvec / vecmat operators.
# ============================================================
# Public API
# ----------
#   build_pixel_grids(device)  -> PixelGrids (namedtuple)
#   build_ray_coords(device)   -> RayArrays  (namedtuple)
#   build_rho_graph(device)    -> RhoGraph   (namedtuple)
#   make_W_operators(device)   -> WOperators (namedtuple)
#   build_all_geometry(device) -> (grids, rays, rho_graph, w_ops)
#
# v7.1 changes vs v7.0
# --------------------
#  * All builder functions now accept a `device` keyword argument.
#    When supplied, every JAX array is explicitly placed on that device
#    via jax.device_put().  When None, a helper _gpu() resolves to
#    GPU:0 if available, otherwise falls back to the default device.
#
#  * make_W_operators: W_IDX, W_LEN, W_MSK and the matvec/vecmat
#    closures are all placed on the resolved device.  This is the most
#    important fix: previously these arrays could land on CPU at import
#    time and the closed-over device would be used for every forward
#    pass during training.
#
#  * build_all_geometry: threads `device` through to all builders.
#
# All heavy arrays are returned as JAX arrays (jnp.float32 / jnp.int32).
# No JAX globals are mutated; callers own the returned objects.
# ============================================================

from __future__ import annotations

import functools

from typing import NamedTuple, Optional

import numpy as np
import jax
import jax.numpy as jnp

from victor import config as cfg


# ── Device helper ─────────────────────────────────────────────────────

def _resolve_device(device=None):
    """Return `device` if given, else GPU:0 if available, else CPU:0."""
    if device is not None:
        return device
    gpus = jax.devices("gpu")
    return gpus[0] if gpus else jax.devices()[0]


# ── Named return types ────────────────────────────────────────────────

class PixelGrids(NamedTuple):
    """All per-pixel geometry arrays, shape (N_GRID*N_GRID,) unless noted."""
    RHO_2D    : jnp.ndarray   # (N_GRID, N_GRID)  normalised elliptic radius
    RHO_FLAT  : jnp.ndarray   # (N_GRID²,)        rho at every pixel
    THETA_FLAT: jnp.ndarray   # (N_GRID²,)        atan2(Y, X)
    R_PIX     : jnp.ndarray   # (N_GRID²,)        major radius [m]
    Z_PIX     : jnp.ndarray   # (N_GRID²,)        vertical position [m]
    RHO_RADIAL: jnp.ndarray   # (N_RADIAL,)       1-D radial axis


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


class WOperators(NamedTuple):
    """Matrix-free W operators built from a padded CSR representation."""
    W_IDX : jnp.ndarray    # (128, MX) int32   — column indices
    W_LEN : jnp.ndarray    # (128, MX) float32 — row values
    W_MSK : jnp.ndarray    # (128, MX) float32 — valid-entry mask
    matvec: object          # jit-compiled (ef -> projections)
    vecmat: object          # jit-compiled (v  -> back-projection)


# ── 1. Pixel grids ────────────────────────────────────────────────────

def build_pixel_grids(
    n_grid : int   = cfg.N_GRID,
    ext    : float = cfg.EXT,
    r0     : float = cfg.R0,
    ap     : float = cfg.AP,
    bp     : float = cfg.BP,
    device         = None,
) -> PixelGrids:
    """
    Build the 128x128 reconstruction pixel grid.

    All returned JAX arrays are explicitly placed on `device`
    (GPU:0 by default).

    Parameters
    ----------
    device : JAX device | None
        Target device for all output arrays.  Defaults to GPU:0
        when available, otherwise the JAX default device.

    Returns
    -------
    PixelGrids
    """
    _dev = _resolve_device(device)

    lin       = np.linspace(-ext, ext, n_grid)
    XX, YY    = np.meshgrid(lin, lin)

    rho_2d_np  = np.sqrt((XX / ap)**2 + (YY / bp)**2).astype(np.float32)
    theta_np   = np.arctan2(YY, XX).flatten().astype(np.float32)
    r_pix_np   = (r0 + XX).flatten().astype(np.float32)
    z_pix_np   = YY.flatten().astype(np.float32)

    def _put(arr):
        return jax.device_put(jnp.array(arr), _dev)

    return PixelGrids(
        RHO_2D     = _put(rho_2d_np),
        RHO_FLAT   = _put(rho_2d_np.flatten()),
        THETA_FLAT = _put(theta_np),
        R_PIX      = _put(r_pix_np),
        Z_PIX      = _put(z_pix_np),
        RHO_RADIAL = jax.device_put(
            jnp.linspace(0.0, cfg.RHO_MAX, cfg.N_RADIAL, dtype=jnp.float32),
            _dev,
        ),
    )


# ── 2. Ray coordinates (WEST-tuned adaptive march) ────────────────────

def build_ray_coords(
    cameras          = cfg.CAMERAS,
    ext    : float   = cfg.EXT,
    n_grid : int     = cfg.N_GRID,
    ap     : float   = cfg.AP,
    bp     : float   = cfg.BP,
    r0     : float   = cfg.R0,
    ds_max_factor: float = cfg.DS_MAX_FACTOR,
    ds_min_factor: float = cfg.DS_MIN_FACTOR,
    device           = None,
) -> RayArrays:
    """
    Compute adaptive-step ray march for all WEST LOS chords.

    All returned JAX arrays are placed on `device` (GPU:0 by default).

    Parameters
    ----------
    device : JAX device | None
        Target device.  Defaults to GPU:0 when available.

    Returns
    -------
    RayArrays
    """
    _dev   = _resolve_device(device)
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
                ds = min(ds, t_out - t)

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

    def _put(arr):
        return jax.device_put(jnp.array(arr), _dev)

    return RayArrays(
        RAY_R     = _put(RR),
        RAY_Z     = _put(ZZ),
        RAY_DS    = _put(DS),
        RAY_V     = _put(VV),
        MAX_STEPS = mx,
    )


# ── 3. rho-proximity graph ────────────────────────────────────────────

def build_rho_graph(
    rho_flat : np.ndarray,
    n_nb     : int   = cfg.RHO_GRAPH_N_NB,
    sigma    : float = cfg.RHO_GRAPH_SIGMA,
    stride   : int   = cfg.RHO_GRAPH_STRIDE,
    device           = None,
) -> RhoGraph:
    """
    Build a k-nearest-neighbour graph in rho-space.

    All returned JAX arrays are placed on `device` (GPU:0 by default).

    Parameters
    ----------
    rho_flat : (N,) numpy array of normalised radii
    device   : JAX device | None
    """
    _dev = _resolve_device(device)
    rho  = np.array(rho_flat)
    N    = len(rho)
    idx  = np.arange(0, N, stride)

    src_list, dst_list, w_list = [], [], []

    for i in idx:
        ri  = rho[i]
        dr  = np.abs(rho[idx] - ri)
        w   = np.exp(-dr / sigma)
        top = np.argsort(w)[-(n_nb + 1):]
        for j2 in top:
            j = idx[j2]
            if j != i and w[j2] > 0.05:
                src_list.append(i)
                dst_list.append(j)
                w_list.append(float(w[j2]))

    deg = np.zeros(N, dtype=np.float32)
    for d_ in dst_list:
        deg[d_] += 1.0

    def _put(arr, dtype=None):
        a = jnp.array(arr, dtype=dtype) if dtype else jnp.array(arr)
        return jax.device_put(a, _dev)

    return RhoGraph(
        EDGES_SRC = _put(src_list, jnp.int32),
        EDGES_DST = _put(dst_list, jnp.int32),
        EDGE_W    = _put(w_list,   jnp.float32),
        NODE_DEG  = _put(np.maximum(deg, 1.0)),
    )


# ── 4. W matrix-free operators ────────────────────────────────────────

def make_W_operators(W_csr, device=None) -> WOperators:
    """
    Build padded CSR arrays and JIT-compile W*eps and W^T*v operators.

    All three weight arrays (W_IDX, W_LEN, W_MSK) and the matvec/vecmat
    closures are placed on `device` (GPU:0 by default).  This is critical:
    if these arrays are on CPU the entire forward projection runs on CPU
    during every training step, even when params are on GPU.

    Parameters
    ----------
    W_csr  : scipy.sparse.csr_matrix   row-normalised W (128 x N_PIXELS)
    device : JAX device | None         target device (default GPU:0)

    Returns
    -------
    WOperators
    """
    _dev   = _resolve_device(device)
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

    # Place ALL three arrays explicitly on the target device.
    # The closures capture these references — if they are on GPU,
    # every matvec call dispatches on GPU.
    W_IDX = jax.device_put(jnp.array(IDX_), _dev)
    W_LEN = jax.device_put(jnp.array(LEN_), _dev)
    W_MSK = jax.device_put(jnp.array(MSK_), _dev)

    # Build explicit single-device shardings for matvec/vecmat so that
    # a global multi-device mesh (set by trainer.py via jax.set_mesh)
    # does not bleed into these operators.  Without explicit in/out
    # shardings, jax.jit inherits the ambient mesh context and expects
    # inputs to be sharded across all devices, causing a
    # "incompatible devices for jitted computation" ValueError at
    # load time when the W operators are first called.
    _dev_sharding = jax.sharding.SingleDeviceSharding(_dev)
    _ef_sharding  = _dev_sharding   # (n_pixels,)   float32
    _g_sharding   = _dev_sharding   # (n_rows,)     float32
    _v_sharding   = _dev_sharding   # (n_rows,)     float32
    _bp_sharding  = _dev_sharding   # (n_pixels,)   float32

    @functools.partial(
        jax.jit,
        in_shardings  = (_ef_sharding,),
        out_shardings = _g_sharding,
    )
    def matvec(ef):
        """W * ef  ->  (n_rows,)  sinogram projection."""
        return jnp.sum(ef[W_IDX] * W_LEN * W_MSK, axis=1)

    @functools.partial(
        jax.jit,
        in_shardings  = (_v_sharding,),
        out_shardings = _bp_sharding,
    )
    def vecmat(v):
        """W^T * v  ->  (N_PIXELS,)  adjoint back-projection."""
        n_pixels = W_csr.shape[1]
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


# ── 5. Convenience: build everything in one call ──────────────────────

def build_all_geometry(W_csr=None, device=None):
    """
    Build pixel grids, ray coords, rho-graph, and optionally W operators.

    Parameters
    ----------
    W_csr  : scipy.sparse.csr_matrix | None
    device : JAX device | None
        Passed through to every sub-builder so all arrays land on the
        same device.  Defaults to GPU:0 when available.

    Returns
    -------
    grids     : PixelGrids
    rays      : RayArrays
    rho_graph : RhoGraph
    w_ops     : WOperators | None
    """
    _dev = _resolve_device(device)
    print(f"Building geometry on device: {_dev}")

    print("  Building pixel grids...")
    grids = build_pixel_grids(device=_dev)
    print(f"    RHO_2D={grids.RHO_2D.shape}  RHO_RADIAL={grids.RHO_RADIAL.shape}")

    print("  Building WEST ray coords...")
    rays   = build_ray_coords(device=_dev)
    n_rays = rays.RAY_R.shape[0]
    mem_kb = n_rays * rays.MAX_STEPS * 4 * 4 / 1024
    print(f"    Rays={n_rays}  MaxSteps={rays.MAX_STEPS}  Mem~{mem_kb:.0f} KB")

    print("  Building rho-proximity graph...")
    rho_graph = build_rho_graph(np.array(grids.RHO_FLAT), device=_dev)
    print(f"    Edges={len(rho_graph.EDGES_SRC)}")

    w_ops = None
    if W_csr is not None:
        print("  Building W matrix-free operators...")
        w_ops = make_W_operators(W_csr, device=_dev)
        print(f"    W_IDX={w_ops.W_IDX.shape}  MX={w_ops.W_IDX.shape[1]}")

    return grids, rays, rho_graph, w_ops
