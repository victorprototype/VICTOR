# ============================================================
# VICTOR v7.0 — model.py
# Architecture: FourierLayer1D, UFourierLayer1D,
#               FourierFeatureEmbedding,
#               BranchNet, TrunkNet, FourierDeepONet
# ============================================================
# Public API
# ----------
#   FourierFeatureEmbedding — Random Fourier Feature lifting (NEW v7.1)
#   FourierLayer1D          — single 1-D Fourier layer (spectral + residual
#                             + LayerNorm)
#   UFourierLayer1D         — U-Fourier layer with U-Net skip + LayerNorm
#   BranchNet               — MLP encoder for chord measurements g_obs
#   TrunkNet                — MLP encoder for xi; prepends RFF embedding
#                             + LayerNorm (NEW v7.1)
#   FourierDeepONet         — full model: branch + trunk + Fourier decoder
#
#   build_model()           — instantiate FourierDeepONet and init params
#   count_params()          — count total trainable parameters
#
# v7.1 additions vs v7.0
# ----------------------
#  * FourierFeatureEmbedding: maps xi (9,) -> (2*rff_features,) using
#    random Fourier features (Tancik et al. 2020).  Lifts xi out of its
#    low-D space to overcome spectral bias in TrunkNet.
#  * TrunkNet: prepends FourierFeatureEmbedding; adds LayerNorm after
#    each hidden SiLU (matching BranchNet).
#  * FourierLayer1D: adds LayerNorm after the ReLU output.
#  * UFourierLayer1D: adds LayerNorm after the skip-connection ReLU
#    and after the final Dense+ReLU.
#  * build_model: exposes rff_features / rff_sigma kwargs.
#
# Design principles (unchanged)
# -----------------
#  * All modules are pure Flax nn.Module subclasses.
#  * No JAX globals are mutated; callers own the returned params.
#  * Branch net encodes g_obs -> (n_radial, C).
#  * Trunk net encodes xi (9-D) -> (C,) via RFF + MLP.
#  * Merger: pointwise multiply (per vanilla DeepONet).
#  * Decoder: 1 FourierLayer1D + 2 UFourierLayer1D -> radial output.
#  * Output: sigmoid -> emissivity profile in [0, 1].
# ============================================================

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, NamedTuple

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
# 1.  FourierLayer1D
# ═══════════════════════════════════════════════════════════════════════

class FourierLayer1D(nn.Module):
    """
    Single Fourier layer for 1-D sequences.

    Keeps sequence length unchanged via two parallel paths:
      Spectral path : IFFT(R · FFT(x))   truncated to n_modes
      Residual path : W · x              learned Dense projection

    Output: relu(spectral + residual)

    Parameters
    ----------
    n_channels : int   Number of feature channels C.
    n_modes    : int   Number of Fourier modes retained (k_max).

    Input / Output
    --------------
    x : (L, C)  ->  (L, C)
    """

    n_channels : int
    n_modes    : int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters
        ----------
        x : (L, C)  float32

        Returns
        -------
        (L, C)  float32
        """
        L, C = x.shape

        # ── Spectral path ─────────────────────────────────────────────
        xf     = jnp.fft.rfft(x, axis=0)              # (L//2+1, C)
        n_freq = xf.shape[0]
        k      = min(self.n_modes, n_freq)

        # Complex weight tensor R: (k, C, C)
        R_re = self.param('R_re', nn.initializers.glorot_normal(), (k, C, C))
        R_im = self.param('R_im', nn.initializers.glorot_normal(), (k, C, C))
        R    = R_re + 1j * R_im

        # Mix kept modes: (k, C) x (k, C, C) -> (k, C)
        xf_out  = jnp.einsum('ki,kij->kj', xf[:k], R)
        xf_full = jnp.zeros_like(xf).at[:k].set(xf_out)   # (n_freq, C)
        x_spec  = jnp.fft.irfft(xf_full, n=L, axis=0)     # (L, C)

        # ── Residual path ─────────────────────────────────────────────
        x_res = nn.Dense(self.n_channels)(x)               # (L, C)

        out = nn.relu(x_spec + x_res)
        return nn.LayerNorm()(out)                          # (L, C)


# ═══════════════════════════════════════════════════════════════════════
# 2.  UFourierLayer1D
# ═══════════════════════════════════════════════════════════════════════

class UFourierLayer1D(nn.Module):
    """
    U-Fourier layer: Fourier layer with an additional U-Net-style skip
    from the global average of the input.

    Optionally projects the sequence length L -> out_len via linear
    interpolation along the sequence axis.

    Parameters
    ----------
    n_channels : int        Number of feature channels C.
    n_modes    : int        Number of Fourier modes retained.
    out_len    : int | None Target output length. None keeps L unchanged.

    Input / Output
    --------------
    x : (L, C)  ->  (out_len or L, C)
    """

    n_channels : int
    n_modes    : int
    out_len    : Optional[int] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters
        ----------
        x : (L, C)  float32

        Returns
        -------
        (target, C)  float32  where target = out_len if set, else L
        """
        L, C   = x.shape
        target = self.out_len if self.out_len is not None else L

        # ── Fourier spectral path ─────────────────────────────────────
        xf     = jnp.fft.rfft(x, axis=0)
        n_freq = xf.shape[0]
        k      = min(self.n_modes, n_freq)

        R_re   = self.param('R_re', nn.initializers.glorot_normal(), (k, C, C))
        R_im   = self.param('R_im', nn.initializers.glorot_normal(), (k, C, C))
        R      = R_re + 1j * R_im
        xf_out = jnp.einsum('ki,kij->kj', xf[:k], R)
        xf_full= jnp.zeros_like(xf).at[:k].set(xf_out)
        x_spec = jnp.fft.irfft(xf_full, n=L, axis=0)     # (L, C)

        # ── Residual (W · x) path ─────────────────────────────────────
        x_res  = nn.Dense(self.n_channels)(x)              # (L, C)

        # ── U-Net skip: global average broadcast ──────────────────────
        x_avg  = jnp.mean(x, axis=0, keepdims=True)       # (1, C)
        x_skip = nn.Dense(self.n_channels)(x_avg)          # (1, C)
        x_skip = jnp.broadcast_to(x_skip, (L, C))         # (L, C)  explicit

        z = nn.relu(x_spec + x_res + x_skip)              # (L, C)
        z = nn.LayerNorm()(z)                              # normalise before resize

        # ── Optional length projection L -> target ────────────────────
        if target != L:
            z = jax.image.resize(z, shape=(target, C), method='linear')

        # ── Per-position linear projection (W' in paper) ──────────────
        z = nn.Dense(self.n_channels)(z)
        z = nn.relu(z)
        return nn.LayerNorm()(z)                           # (target, C)


# ═══════════════════════════════════════════════════════════════════════
# 3.  BranchNet
# ═══════════════════════════════════════════════════════════════════════

class BranchNet(nn.Module):
    """
    MLP branch net that encodes chord measurements g_obs.

    Projects g_obs (n_chords,) to a latent sequence (n_radial, C) that
    aligns with the output radial grid.

    Architecture:  g -> [Dense(h) -> SiLU -> LayerNorm] × n_layers
                      -> Dense(n_radial * C) -> reshape(n_radial, C)

    Parameters
    ----------
    hidden_dims : Sequence[int]   Hidden layer widths.
    n_radial    : int             Output sequence length L.
    n_channels  : int             Output channel width C.

    Input / Output
    --------------
    g : (n_chords,)  ->  (n_radial, n_channels)
    """

    hidden_dims : Sequence[int]
    n_radial    : int
    n_channels  : int

    @nn.compact
    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters
        ----------
        g : (n_chords,)  float32

        Returns
        -------
        (n_radial, n_channels)  float32
        """
        h = g
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = nn.silu(h)
            h = nn.LayerNorm()(h)
        h = nn.Dense(self.n_radial * self.n_channels)(h)  # (L*C,)
        return h.reshape(self.n_radial, self.n_channels)  # (L, C)


# ═══════════════════════════════════════════════════════════════════════
# 4.  FourierFeatureEmbedding  — NEW in v7.1
# ═══════════════════════════════════════════════════════════════════════

class FourierFeatureEmbedding(nn.Module):
    """
    Random Fourier Feature (RFF) embedding for low-dimensional inputs.

    Maps a d-dimensional input x to a 2*n_features vector:
        gamma(x) = [cos(2*pi*B*x), sin(2*pi*B*x)]

    where B is a (n_features, d) matrix of frequencies sampled from
    N(0, sigma^2) at init time (not trained).

    This lifts the xi hardware vector out of its low-dimensional 9-D
    space into a high-frequency feature space where the TrunkNet MLP
    can learn sharp, non-smooth response functions more easily.
    Without this embedding, deep MLPs on low-D inputs are biased toward
    low-frequency solutions (spectral bias / frequency principle).

    Reference: Tancik et al. 2020, "Fourier Features Let Networks Learn
    High Frequency Functions in Low Dimensional Domains".

    Parameters
    ----------
    n_features : int    Number of frequency samples (output dim = 2*n_features).
    sigma      : float  Standard deviation of the frequency distribution.
                        Higher sigma -> higher-frequency features.
                        Default 1.0 is calibrated for normalised xi in [0,1].

    Input / Output
    --------------
    x : (d,)  ->  (2 * n_features,)
    """

    n_features : int
    sigma      : float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters
        ----------
        x : (d,)  float32   low-dimensional input

        Returns
        -------
        (2 * n_features,)  float32
        """
        d = x.shape[0]
        # B is fixed at init — not a trainable param.
        # Use param() so Flax serialises it with the model state.
        B = self.param(
            'B',
            lambda key, shape: jax.random.normal(key, shape) * self.sigma,
            (self.n_features, d),
        )
        proj = 2.0 * jnp.pi * x @ B.T          # (n_features,)
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)  # (2*n_features,)


# ═══════════════════════════════════════════════════════════════════════
# 5.  TrunkNet
# ═══════════════════════════════════════════════════════════════════════

class TrunkNet(nn.Module):
    """
    MLP trunk net that encodes WEST GEM source/geometry parameters xi.

    v7.1: Prepends a FourierFeatureEmbedding to lift the 9-D xi vector
    into a 2*rff_features-dimensional frequency space before the MLP.
    This overcomes the MLP's spectral bias and lets the trunk learn
    sharp responses to hardware parameter variations.

    v7.1: Added LayerNorm after each hidden SiLU layer (matches BranchNet).

    Architecture:
        xi (9,)
          -> FourierFeatureEmbedding(rff_features)  -> (2*rff_features,)
          -> [Dense(h) -> SiLU -> LayerNorm] x n_layers
          -> Dense(C)
          -> (C,)

    Parameters
    ----------
    hidden_dims  : Sequence[int]   Hidden layer widths.
    n_channels   : int             Output width C.
    rff_features : int             Number of RFF frequency samples.
                                   Output of embedding = 2*rff_features.
                                   Default 64 gives 128-D embedding for xi.
    rff_sigma    : float           RFF frequency scale (default 1.0).

    Input / Output
    --------------
    xi : (9,)  ->  (n_channels,)
    """

    hidden_dims  : Sequence[int]
    n_channels   : int
    rff_features : int   = 64
    rff_sigma    : float = 1.0

    @nn.compact
    def __call__(self, xi: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters
        ----------
        xi : (9,)  float32

        Returns
        -------
        (n_channels,)  float32
        """
        # Fourier feature embedding: (9,) -> (2*rff_features,)
        h = FourierFeatureEmbedding(self.rff_features, self.rff_sigma)(xi)

        # MLP with LayerNorm (matches BranchNet style)
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = nn.silu(h)
            h = nn.LayerNorm()(h)

        return nn.Dense(self.n_channels)(h)               # (C,)


# ═══════════════════════════════════════════════════════════════════════
# 5.  FourierDeepONet  —  Full Model
# ═══════════════════════════════════════════════════════════════════════

class FourierDeepONet(nn.Module):
    """
    Fourier-enhanced Deep Operator Network for VICTOR Phase 2.

    Architecture (mirrors Zhu et al. 2023, adapted for 1-D radial output):
    ┌─────────────────────────────────────────────────────┐
    │  Branch net  MLP(g_obs)          -> (n_radial, C)   │
    │  Trunk net   MLP(xi)             -> (C,)            │
    │  Merger      pointwise multiply  -> (n_radial, C)   │
    │                                                     │
    │  Decoder:                                           │
    │    FourierLayer1D (C, n_modes)   -> (n_radial, C)   │
    │    UFourierLayer1D × 2           -> (n_radial, C)   │
    │    Dense(64) -> ReLU                                │
    │    Dense(1)  -> squeeze -> sigmoid                  │
    │                                  -> (n_radial,)     │
    └─────────────────────────────────────────────────────┘

    Parameters
    ----------
    branch_hidden : Sequence[int]   Branch MLP hidden widths.
    trunk_hidden  : Sequence[int]   Trunk MLP hidden widths.
    n_channels    : int             Latent channel width C.
    n_modes       : int             Truncated Fourier modes k_max.
    n_radial      : int             Output radial grid points L.

    Forward signature
    -----------------
    (g, xi)

    Inputs
    ------
    g  : (n_chords,)  normalised chord measurements
    xi : (9,)         WEST GEM geometry/source parameters (see TrunkNet)

    Returns
    -------
    eps1d : (n_radial,)  emissivity radial profile in [0, 1]
    """

    branch_hidden : Sequence[int]
    trunk_hidden  : Sequence[int]
    n_channels    : int
    n_modes       : int
    n_radial      : int
    rff_features  : int   = 64     # RFF samples in TrunkNet embedding
    rff_sigma     : float = 1.0    # RFF frequency scale

    @nn.compact
    def __call__(
        self,
        g  : jnp.ndarray,   # (n_chords,)
        xi : jnp.ndarray,   # (9,)
    ) -> jnp.ndarray:
        """
        Parameters
        ----------
        g  : (n_chords,)  float32
        xi : (9,)         float32

        Returns
        -------
        eps1d : (n_radial,)  float32  in [0, 1]
        """
        C = self.n_channels
        L = self.n_radial

        # ── Branch & trunk encoding ───────────────────────────────────
        b = BranchNet(self.branch_hidden, L, C)(g)                          # (L, C)
        t = TrunkNet(self.trunk_hidden, C,
                     self.rff_features, self.rff_sigma)(xi)                 # (C,)

        # ── Merger: pointwise multiply (broadcast t over L) ───────────
        z = b * t[None, :]                                 # (L, C)

        # ── Decoder: 1 Fourier + 2 U-Fourier layers ───────────────────
        z = FourierLayer1D(C, self.n_modes)(z)             # (L, C)
        z = UFourierLayer1D(C, self.n_modes)(z)            # (L, C)
        z = UFourierLayer1D(C, self.n_modes)(z)            # (L, C)

        # ── Projection to radial profile ──────────────────────────────
        z     = nn.Dense(64)(z)                            # (L, 64)
        z     = nn.relu(z)
        z     = nn.Dense(1)(z)                             # (L, 1)
        eps1d = nn.sigmoid(z.squeeze(-1))                  # (L,)
        return eps1d


# ═══════════════════════════════════════════════════════════════════════
# 6.  Utilities
# ═══════════════════════════════════════════════════════════════════════

def count_params(params: dict) -> int:
    """Return total number of scalar parameters in a Flax param tree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def build_model(
    g        : jnp.ndarray,
    xi       : jnp.ndarray,
    seed     : int = 0,
    branch_hidden : Sequence[int] = (256, 256),
    trunk_hidden  : Sequence[int] = (64, 128, 128),
    n_channels    : int = 64,
    n_modes       : int = 16,
    n_radial      : int = 128,
    rff_features  : int = 64,
    rff_sigma     : float = 1.0,
) -> ModelBundle:
    """
    Instantiate FourierDeepONet and initialise parameters.

    Parameters
    ----------
    g             : (n_chords,)  example chord array (shape init only).
    xi            : (9,)         example geometry vector (shape init only).
    seed          : int          PRNGKey seed.
    branch_hidden : Sequence[int]  Branch MLP hidden widths.
    trunk_hidden  : Sequence[int]  Trunk MLP hidden widths.
    n_channels    : int            Latent channel width C.
    n_modes       : int            Truncated Fourier modes k_max.
    n_radial      : int            Output radial grid length L.
    rff_features  : int            RFF samples in TrunkNet (default 64).
    rff_sigma     : float          RFF frequency scale (default 1.0).

    Returns
    -------
    ModelBundle  (model, params)
    """
    model  = FourierDeepONet(
        branch_hidden = branch_hidden,
        trunk_hidden  = trunk_hidden,
        n_channels    = n_channels,
        n_modes       = n_modes,
        n_radial      = n_radial,
        rff_features  = rff_features,
        rff_sigma     = rff_sigma,
    )
    key    = jax.random.PRNGKey(seed)
    params = model.init(key, g, xi)
    return ModelBundle(model=model, params=params)


def verify_model(
    bundle : ModelBundle,
    g      : jnp.ndarray,
    xi     : jnp.ndarray,
) -> None:
    """
    Run a single forward pass and print a diagnostic summary.

    Checks parameter count, output shape, and NaN count.
    Raises AssertionError if the output contains NaNs.

    Parameters
    ----------
    bundle : ModelBundle   (model, params) from build_model().
    g      : (n_chords,)   normalised chord measurements.
    xi     : (9,)          WEST GEM geometry/source parameters.
    """
    model, params = bundle

    eps1d = model.apply(params, g, xi)

    n_params  = count_params(params)
    nan_count = int(jnp.any(jnp.isnan(eps1d)))

    print("── model.py  FourierDeepONet ────────────────────────────")
    print(f"  Parameters     : {n_params:,}")
    print(f"  Output shape   : {eps1d.shape}  (expect ({model.n_radial},))")
    print(f"  Output range   : [{float(eps1d.min()):.4f}, {float(eps1d.max()):.4f}]")
    print(f"  NaN in output  : {nan_count}  (must be 0)")
    print("────────────────────────────────────────────────────────")

    assert nan_count == 0, (
        "Forward pass produced NaNs. "
        "Check Fourier layer initialisers or branch/trunk widths."
    )
    print("OK  model.py verified")
