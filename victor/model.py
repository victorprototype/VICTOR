# ============================================================
# VICTOR v6.0 — model.py
# Architecture: HashGrid, SIRENLayer, SO2Harmonics,
#               SharedTrunk, PIGNO, VICTOR_v6
# ============================================================
# Public API
# ----------
#   HashGrid       — multi-resolution hash grid (pure JAX)
#   SIRENLayer     — SIREN layer with learnable ω₀
#   SO2Harmonics   — SO(2) angular harmonics encoder
#   SharedTrunk    — shared feature extractor (hash + SIREN)
#   MemberAdapter  — per-ensemble-member adapter MLP
#   PIGNO          — physics-informed graph neural operator
#   VICTOR_v6      — full model: shared trunk + adapters + PIGNO
#
#   build_model()  — instantiate VICTOR_v6 and initialise params
#   count_params() — count total trainable parameters
#
# Design principles
# -----------------
#  • All modules are pure Flax nn.Module subclasses.
#  • No JAX globals are mutated; callers own the returned params.
#  • profile field arrays (psi_n, bpol_n) are plain jnp arrays —
#    never nested dicts passed into @jax.jit (v5 FIX).
#  • PIGNO boundary mask is a smooth sigmoid (hard boundary via ρ).
# ============================================================

from __future__ import annotations

from typing import List, Tuple, NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from victor import config as cfg


# ── Named return type ─────────────────────────────────────────────────

class ModelBundle(NamedTuple):
    """Model instance together with its initialised parameters."""
    model  : nn.Module
    params : dict


# ═══════════════════════════════════════════════════════════════════════
# 1.  HashGrid
# ═══════════════════════════════════════════════════════════════════════

class HashGrid(nn.Module):
    """
    Multi-resolution hash grid encoding (pure JAX — no CUDA extensions).

    Maps a batch of 2-D coordinates in [-1, 1]² to a concatenated
    feature vector of length L × F by bilinearly interpolating
    per-level hash tables.

    Parameters
    ----------
    L : int   Number of resolution levels.
    T : int   Hash-table size (number of entries per level).
    F : int   Feature dimension per entry.
    b : float Growth factor between consecutive level resolutions.

    Input
    -----
    xy : (N, 2)  coordinates in [-1, 1]²

    Output
    ------
    (N, L*F)  concatenated multi-scale features
    """

    L : int   = cfg.L_HASH
    T : int   = cfg.T_COORD
    F : int   = cfg.F_HASH
    b : float = 1.38

    # Fibonacci hash constants (int32 two's-complement safe)
    _PI1 : int = -1_640_531_535
    _PI2 : int =    805_459_861

    @nn.compact
    def __call__(self, xy: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters
        ----------
        xy : (N, 2)  float32, values in [-1, 1]

        Returns
        -------
        (N, L*F)  float32
        """
        pi1 = jnp.int32(self._PI1)
        pi2 = jnp.int32(self._PI2)

        feats: List[jnp.ndarray] = []

        for lev in range(self.L):
            res   = int(16 * (self.b ** lev))
            table = self.param(
                f"t{lev}",
                nn.initializers.uniform(1e-4),
                (self.T, self.F),
            )

            # Map to [0, res]
            sc = (xy * 0.5 + 0.5) * res
            fl = jnp.floor(sc).astype(jnp.int32)
            fr = sc - fl.astype(jnp.float32)           # fractional part

            # Four corners of the enclosing cell (N, 4, 2)
            corners = jnp.stack(
                [
                    fl,
                    fl + jnp.array([1, 0]),
                    fl + jnp.array([0, 1]),
                    fl + jnp.array([1, 1]),
                ],
                axis=1,
            )

            cx = corners[:, :, 0]   # (N, 4)
            cy = corners[:, :, 1]   # (N, 4)

            # Spatial hash: h = |(cx * π₁) XOR (cy * π₂)| mod T
            h = jnp.abs(
                jnp.mod(
                    jnp.bitwise_xor(
                        (cx * pi1).astype(jnp.int32),
                        (cy * pi2).astype(jnp.int32),
                    ),
                    self.T,
                )
            )

            cf = table[h]           # (N, 4, F)

            # Bilinear weights (N, 4, 1)
            fx = fr[:, 0:1]
            fy = fr[:, 1:2]
            w  = jnp.stack(
                [(1 - fx) * (1 - fy), fx * (1 - fy),
                 (1 - fx) * fy,       fx * fy],
                axis=1,
            )

            feats.append(jnp.sum(w * cf, axis=1))   # (N, F)

        return jnp.concatenate(feats, axis=-1)        # (N, L*F)


# ═══════════════════════════════════════════════════════════════════════
# 2.  SIRENLayer
# ═══════════════════════════════════════════════════════════════════════

class SIRENLayer(nn.Module):
    """
    Sinusoidal representation network (SIREN) layer with a
    learnable per-layer frequency parameter ω₀.

    The first layer uses a uniform weight initialisation over
    (-1/fan_in, 1/fan_in); subsequent layers use the SIREN-paper
    scheme √(6/fan_in) / ω₀.

    Parameters
    ----------
    features  : int    Output width.
    is_first  : bool   True for the first layer of a SIREN stack.
    omega_0   : float  Initial value of the learnable ω₀.
    """

    features : int
    is_first : bool  = False
    omega_0  : float = 30.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters
        ----------
        x : (N, D)  float32

        Returns
        -------
        (N, features)  float32   sin(ω₀ · W x)
        """
        om = self.param(
            "om",
            nn.initializers.constant(self.omega_0),
            (1,),
        )

        if self.is_first:
            scale = 1.0 / x.shape[-1]
        else:
            scale = jnp.sqrt(6.0 / x.shape[-1]) / self.omega_0

        w = nn.Dense(
            self.features,
            kernel_init=nn.initializers.uniform(scale),
        )(x)

        return jnp.sin(om * w)


# ═══════════════════════════════════════════════════════════════════════
# 3.  SO2Harmonics
# ═══════════════════════════════════════════════════════════════════════

class SO2Harmonics(nn.Module):
    """
    SO(2) angular harmonic encoder.

    Computes cos(m·θ) and sin(m·θ) for m = 0 … n_orders-1, then
    projects the raw harmonics through a learned Dense layer.

    Parameters
    ----------
    n_orders : int   Number of angular frequency orders.

    Input
    -----
    theta : (N,)  angles in radians

    Output
    ------
    (N, n_orders*2)  float32
    """

    n_orders : int = 4

    @nn.compact
    def __call__(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters
        ----------
        theta : (N,)  float32

        Returns
        -------
        (N, n_orders*2)  float32
        """
        parts = [jnp.ones_like(theta[:, None])]   # m = 0  (constant)

        for m in range(1, self.n_orders):
            parts.append(jnp.cos(m * theta[:, None]))
            parts.append(jnp.sin(m * theta[:, None]))

        h = jnp.concatenate(parts, axis=-1)        # (N, 2*n_orders - 1)

        return nn.Dense(
            self.n_orders * 2,
            kernel_init=nn.initializers.glorot_normal(),
        )(h)                                        # (N, n_orders*2)


# ═══════════════════════════════════════════════════════════════════════
# 4.  SharedTrunk
# ═══════════════════════════════════════════════════════════════════════

class SharedTrunk(nn.Module):
    """
    Shared feature extractor — parameters shared across all ensemble
    members.

    Concatenates spatial hash features, field hash features, and SO(2)
    angular features, then passes them through a two-layer SIREN MLP.

    Input dimensions (defaults)
    ---------------------------
    hash_coord : (N, L*F)     = (N, 32)   spatial hash
    hash_field : (N, L//2*F)  = (N, 16)   field hash
    so2_feat   : (N, 8)                   angular harmonics
    ───────────────────────────────────────
    concat       (N, 56)

    Output
    ------
    (N, hidden)  float32   shared trunk features
    """

    hidden : int = 256

    @nn.compact
    def __call__(
        self,
        hash_coord : jnp.ndarray,
        hash_field : jnp.ndarray,
        so2_feat   : jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Parameters
        ----------
        hash_coord : (N, 32)
        hash_field : (N, 16)
        so2_feat   : (N, 8)

        Returns
        -------
        (N, hidden)  float32
        """
        x = jnp.concatenate([hash_coord, hash_field, so2_feat], axis=-1)
        x = SIRENLayer(self.hidden, is_first=True)(x)
        x = SIRENLayer(self.hidden)(x)
        return x                                    # (N, hidden)


# ═══════════════════════════════════════════════════════════════════════
# 5.  MemberAdapter
# ═══════════════════════════════════════════════════════════════════════

class MemberAdapter(nn.Module):
    """
    Tiny per-ensemble-member adapter MLP.

    Each adapter has independent weights and maps the shared trunk
    output to a positive scalar emissivity at every pixel.

    Architecture:  trunk → Dense(64) → GELU → Dense(1) → softplus

    Input
    -----
    trunk_out : (N, 256)   shared trunk features

    Output
    ------
    (N,)  float32   positive emissivity prediction for one member
    """

    @nn.compact
    def __call__(self, trunk_out: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters
        ----------
        trunk_out : (N, hidden)

        Returns
        -------
        (N,)  float32  softplus-activated scalar output
        """
        h   = nn.Dense(64, kernel_init=nn.initializers.glorot_normal())(trunk_out)
        h   = nn.gelu(h)
        out = nn.Dense(1,  kernel_init=nn.initializers.normal(0.01))(h)
        return nn.softplus(out.squeeze(-1))          # (N,)


# ═══════════════════════════════════════════════════════════════════════
# 6.  PIGNO  —  Physics-Informed Graph Neural Operator
# ═══════════════════════════════════════════════════════════════════════

class PIGNO(nn.Module):
    """
    Physics-Informed Graph Neural Operator.

    Performs message passing on the rho-proximity graph to propagate
    information along iso-flux surfaces, enforcing the physical
    constraint that emissivity varies smoothly with normalised radius ρ.

    Each layer computes edge messages from (h_src, h_dst, weight),
    aggregates by degree-normalised scatter, and applies LayerNorm
    with a residual connection.

    Parameters
    ----------
    n_layers : int   Number of message-passing layers (default 2).
    hidden   : int   Hidden width in the edge MLP (default 96).
    N        : int   Grid size; output is (N, N) (default 128).

    Inputs
    ------
    eps_2d : (N, N)   initial emissivity estimate from the ensemble mean
    esrc   : (E,)     int32  edge source indices  (flat pixel index)
    edst   : (E,)     int32  edge destination indices
    ew     : (E,)     float32 edge weights (RBF in ρ-space)
    ndeg   : (N²,)   float32 per-node degree (≥ 1)

    Output
    ------
    (N, N)  float32   refined emissivity, positive via softplus
    """

    n_layers : int = cfg.PIGNO_LAYERS
    hidden   : int = cfg.PIGNO_HIDDEN
    N        : int = cfg.N_GRID

    @nn.compact
    def __call__(
        self,
        eps_2d : jnp.ndarray,
        esrc   : jnp.ndarray,
        edst   : jnp.ndarray,
        ew     : jnp.ndarray,
        ndeg   : jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Parameters
        ----------
        eps_2d : (N, N)
        esrc   : (E,) int32
        edst   : (E,) int32
        ew     : (E,) float32
        ndeg   : (N²,) float32

        Returns
        -------
        (N, N)  float32
        """
        n_nodes = self.N * self.N
        h       = eps_2d.flatten()[:, None]          # (N², 1)
        deg     = ndeg[:, None]                      # (N², 1)

        for _ in range(self.n_layers):
            h_src  = h[esrc]                         # (E, 1)
            h_dst  = h[edst]                         # (E, 1)
            w      = ew[:, None]                     # (E, 1)

            # Edge message: concat [h_src | h_dst | weight]
            msg = jnp.concatenate([h_src, h_dst, w], axis=-1)   # (E, 3)
            msg = nn.Dense(
                self.hidden,
                kernel_init=nn.initializers.glorot_normal(),
            )(msg)
            msg = nn.gelu(msg)
            msg = nn.Dense(
                1,
                kernel_init=nn.initializers.glorot_normal(),
            )(msg)                                               # (E, 1)

            # Degree-normalised scatter-add to destination nodes
            h_agg = (
                jnp.zeros((n_nodes, 1))
                .at[edst]
                .add(msg * w)
            )
            h_agg = h_agg / deg                      # normalise

            # Residual + LayerNorm
            h = nn.LayerNorm()(h + h_agg)

        return nn.softplus(h).squeeze(-1).reshape(self.N, self.N)


# ═══════════════════════════════════════════════════════════════════════
# 7.  VICTOR_v6  —  Full Model
# ═══════════════════════════════════════════════════════════════════════

class VICTOR_v6(nn.Module):
    """
    VICTOR v6.0 — full reconstruction model.

    Architecture summary
    --------------------
    ┌─────────────────────────────────────────────────────┐
    │  Dual hash grid                                      │
    │    coord hash  (T=8192, L=16, F=2) → (N, 32)        │
    │    field hash  (T=4096, L= 8, F=2) → (N, 16)        │
    │  SO2Harmonics  (n_orders=4)        → (N,  8)        │
    │                                    ─────────        │
    │  Concat                            → (N, 56)        │
    │  SharedTrunk  (hidden=256)         → (N,256)        │
    │                                                     │
    │  MemberAdapter × n_members         → (M, N)         │
    │    mean / std of ensemble predictions               │
    │                                                     │
    │  PIGNO  (n_layers=2, hidden=96)                     │
    │    applied to ensemble mean (128,128)               │
    │                                                     │
    │  Sigmoid boundary mask (ρ < 1)                      │
    └─────────────────────────────────────────────────────┘

    Total ~ 570 k parameters (target 500 k – 600 k).

    Forward signature
    -----------------
    (R_flat, Z_flat, psi_n, bpol_n, esrc, edst, ew, ndeg, rho_2d)

    All array inputs are plain jnp arrays — never nested dicts
    (FIX over v5: dicts must not be passed as jax.jit arguments).

    Returns
    -------
    eps_out  : (N, N)     final emissivity (masked, positive)
    mean     : (N²,)      ensemble mean  (pre-PIGNO)
    std      : (N²,)      ensemble uncertainty (pre-PIGNO)
    preds    : (M, N²)    per-member predictions
    """

    n_members : int = cfg.N_ENS

    def setup(self) -> None:
        # ── Shared encoders ───────────────────────────────────────────
        self.hash_coord = HashGrid(L=cfg.L_HASH,       T=cfg.T_COORD, F=cfg.F_HASH)
        self.hash_field = HashGrid(L=cfg.L_HASH // 2,  T=cfg.T_FIELD, F=cfg.F_HASH)
        self.so2        = SO2Harmonics(n_orders=4)
        self.trunk      = SharedTrunk(hidden=256)

        # ── Per-member adapters ───────────────────────────────────────
        # List comprehension; each adapter has independent weights.
        self.adapters   = [MemberAdapter() for _ in range(self.n_members)]

        # ── Learnable per-member noise floor ─────────────────────────
        self.log_noise  = self.param(
            "log_noise",
            nn.initializers.constant(-3.0),
            (self.n_members,),
        )

        # ── PIGNO ─────────────────────────────────────────────────────
        self.pigno = PIGNO(
            n_layers = cfg.PIGNO_LAYERS,
            hidden   = cfg.PIGNO_HIDDEN,
            N        = cfg.N_GRID,
        )

    def __call__(
        self,
        R_flat  : jnp.ndarray,   # (N²,) major radius [m]
        Z_flat  : jnp.ndarray,   # (N²,) vertical coord [m]
        psi_n   : jnp.ndarray,   # (N²,) normalised ψ  ∈ [-1,1]
        bpol_n  : jnp.ndarray,   # (N²,) normalised Bpol ∈ [-1,1]
        esrc    : jnp.ndarray,   # (E,)  int32  edge sources
        edst    : jnp.ndarray,   # (E,)  int32  edge destinations
        ew      : jnp.ndarray,   # (E,)  float32 edge weights
        ndeg    : jnp.ndarray,   # (N²,) float32 node degrees
        rho_2d  : jnp.ndarray,   # (N, N) normalised elliptic radius
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Forward pass.

        Returns
        -------
        eps_out : (N, N)   final masked emissivity
        mean    : (N²,)    ensemble mean (pre-PIGNO)
        std     : (N²,)    ensemble uncertainty estimate
        preds   : (M, N²)  per-member predictions
        """
        # ── Normalise spatial coordinates to [-1, 1] ─────────────────
        R_n = (R_flat - cfg.R0) / cfg.AP
        Z_n =  Z_flat           / cfg.BP
        xy  = jnp.stack([R_n, Z_n], axis=-1)        # (N², 2)

        # ── Field coordinates for field hash ─────────────────────────
        # psi_n and bpol_n are already in [-1, 1] (normalised in data_loader)
        field_xy = jnp.stack([psi_n, bpol_n], axis=-1)    # (N², 2)

        # ── Encode ───────────────────────────────────────────────────
        hc   = self.hash_coord(xy)        # (N², 32)
        hf   = self.hash_field(field_xy)  # (N², 16)
        theta = jnp.arctan2(Z_n, R_n)    # (N²,)
        so2  = self.so2(theta)            # (N², 8)

        # ── Shared trunk ─────────────────────────────────────────────
        trunk = self.trunk(hc, hf, so2)  # (N², 256)  — shared weights

        # ── Per-member predictions ────────────────────────────────────
        preds = jnp.stack(
            [adapter(trunk) for adapter in self.adapters],
            axis=0,
        )                                            # (M, N²)

        mean = jnp.mean(preds, axis=0)               # (N²,)
        std  = jnp.std( preds, axis=0) + jnp.exp(self.log_noise[0])

        # ── PIGNO refinement on ensemble mean ────────────────────────
        eps_raw   = mean.reshape(cfg.N_GRID, cfg.N_GRID)
        eps_pigno = self.pigno(eps_raw, esrc, edst, ew, ndeg)  # (N, N)

        # ── Hard boundary mask (smooth sigmoid at ρ = 1) ─────────────
        mask    = jax.nn.sigmoid(50.0 * (1.0 - rho_2d))       # (N, N)
        eps_out = eps_pigno * mask                              # (N, N)

        return eps_out, mean, std, preds


# ═══════════════════════════════════════════════════════════════════════
# 8.  Utilities
# ═══════════════════════════════════════════════════════════════════════

def count_params(params: dict) -> int:
    """Return total number of scalar parameters in a Flax param tree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def build_model(
    R_flat  : jnp.ndarray,
    Z_flat  : jnp.ndarray,
    psi_n   : jnp.ndarray,
    bpol_n  : jnp.ndarray,
    esrc    : jnp.ndarray,
    edst    : jnp.ndarray,
    ew      : jnp.ndarray,
    ndeg    : jnp.ndarray,
    rho_2d  : jnp.ndarray,
    seed    : int = 0,
    n_members : int = cfg.N_ENS,
) -> ModelBundle:
    """
    Instantiate VICTOR_v6 and initialise parameters with a random key.

    Parameters
    ----------
    R_flat … rho_2d : example arrays matching the shapes expected at
        inference time (used only for shape inference — values ignored).
    seed      : int   PRNGKey seed.
    n_members : int   Number of ensemble members (default cfg.N_ENS).

    Returns
    -------
    ModelBundle
        model  : VICTOR_v6 instance
        params : initialised parameter tree
    """
    model  = VICTOR_v6(n_members=n_members)
    key    = jax.random.PRNGKey(seed)
    params = model.init(
        key,
        R_flat, Z_flat, psi_n, bpol_n,
        esrc, edst, ew, ndeg, rho_2d,
    )
    return ModelBundle(model=model, params=params)


def verify_model(bundle: ModelBundle,
                 R_flat, Z_flat, psi_n, bpol_n,
                 esrc, edst, ew, ndeg, rho_2d) -> None:
    """
    Run a single forward pass and print a diagnostic summary.

    Checks parameter count, output shapes, and NaN counts.
    Raises AssertionError if any output contains NaNs.
    """
    model, params = bundle

    outputs = model.apply(
        params,
        R_flat, Z_flat, psi_n, bpol_n,
        esrc, edst, ew, ndeg, rho_2d,
    )

    n_params  = count_params(params)
    nan_count = sum(int(jnp.any(jnp.isnan(o))) for o in outputs)
    shapes    = [o.shape for o in outputs]

    print("── model.py  VICTOR_v6 ─────────────────────────────────")
    print(f"  Parameters     : {n_params:,}  (target 500k–600k)")
    print(f"  Output shapes  : {shapes}")
    print(f"  NaN in outputs : {nan_count}  (must be 0)")
    print("────────────────────────────────────────────────────────")

    assert nan_count == 0, (
        f"Forward pass produced NaNs in {nan_count} output(s). "
        "Check hash grid initialisers or SIREN ω₀."
    )
    print("OK  model.py verified")
