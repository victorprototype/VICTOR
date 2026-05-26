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
        return out                                          # (L, C)                         


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

        # ── Optional length projection L -> target ────────────────────
        if target != L:
            z = jax.image.resize(z, shape=(target, C), method='linear')

        # ── Per-position linear projection (W' in paper) ──────────────
        z = nn.Dense(self.n_channels)(z)
        z = nn.relu(z)
        return z                                           # (target, C)

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
# 3b.  EquilibriumEncoder  — NEW in v8
# ═══════════════════════════════════════════════════════════════════════

class EquilibriumEncoder(nn.Module):
    """
    MLP encoder for equilibrium fields (psi_2d, rho_2d).

    Input is the concatenation of two flattened (N_GRID²,) fields,
    giving a 32768-D vector.  The first layer projects this down
    aggressively to avoid a parameter explosion.

    Architecture:
        concat(psi_flat, rho_flat)  — (2*N_GRID²,)
          -> Dense(512) -> GELU
          -> Dense(256) -> GELU
          -> Dense(n_channels)      — (C,)

    Future extension: concatenate Te_flat, ne_flat etc. to the input
    and increment N_EQ_CHANNELS in config — no decoder rewrite needed.
    """

    n_channels : int

    @nn.compact
    def __call__(self, eq_flat: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters
        ----------
        eq_flat : (N_EQ_CHANNELS * N_GRID²,)  concatenated equilibrium fields

        Returns
        -------
        (n_channels,)  float32
        """
        h = nn.Dense(512)(eq_flat)
        h = nn.gelu(h)
        h = nn.Dense(256)(h)
        h = nn.gelu(h)
        return nn.Dense(self.n_channels)(h)                # (C,)


# ═══════════════════════════════════════════════════════════════════════
# 4a.  FourierFeatureEmbedding  — NEW in v7.1
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
# 4b.  TrunkNet
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
# 5.  FourierDeepONetV8  —  Full Model
# ═══════════════════════════════════════════════════════════════════════

class FourierDeepONetV8(nn.Module):
    """
    VICTOR v8: Dual Encoder + Fourier Decoder + Poloidal Harmonic Output.

    Inputs
    ------
    g           : (n_chords,)   normalised chord measurements
    psi_flat    : (N_GRID²,)    normalised poloidal flux [-1, 1]
    rho_flat_eq : (N_GRID²,)    normalised elliptic radius [-1, 1]
    xi          : (9,)          WEST GEM hardware vector

    Returns
    -------
    coeffs : (n_radial, 1 + 2*n_harmonics)  harmonic coefficients
        channel 0      = a0   (softplus applied -> >= 0)
        channels 1,2   = a1, b1  (1st harmonic, unconstrained)
        channels 3,4   = a2, b2  (2nd harmonic, unconstrained)
    """

    branch_hidden  : Sequence[int]
    trunk_hidden   : Sequence[int]
    n_channels     : int
    n_modes        : int
    n_radial       : int
    n_harmonics    : int   = 2
    n_eq_channels  : int   = 2
    rff_features   : int   = 64
    rff_sigma      : float = 1.0

    @property
    def n_out_channels(self) -> int:
        return 1 + 2 * self.n_harmonics

    @nn.compact
    def __call__(
        self,
        g           : jnp.ndarray,   # (n_chords,)
        psi_flat    : jnp.ndarray,   # (N_GRID²,)
        rho_flat_eq : jnp.ndarray,   # (N_GRID²,)
        xi          : jnp.ndarray,   # (9,)
    ) -> jnp.ndarray:

        C  = self.n_channels
        L  = self.n_radial
        NC = self.n_out_channels

        # ── Branch: g -> (L, C) ───────────────────────────────────────
        latent_g = BranchNet(self.branch_hidden, L, C)(g)              # (L, C)

        # ── Equilibrium: concat(psi, rho) -> (C,) ────────────────────
        eq_flat   = jnp.concatenate([psi_flat, rho_flat_eq], axis=-1)  # (2*N²,)
        latent_eq = EquilibriumEncoder(C)(eq_flat)                      # (C,)

        # ── Hardware vector: xi -> (C,) ───────────────────────────────
        latent_xi = TrunkNet(
            self.trunk_hidden, C,
            self.rff_features, self.rff_sigma,
        )(xi)                                                           # (C,)

        # ── Fusion: concat(eq, xi) -> C, then multiply with branch ───
        fused = jnp.concatenate([latent_eq, latent_xi], axis=-1)       # (2C,)
        fused = nn.Dense(C)(fused)                                      # (C,)
        fused = nn.gelu(fused)

        z = latent_g * fused[None, :]                                   # (L, C)

        # ── Fourier decoder ───────────────────────────────────────────
        z = FourierLayer1D(C, self.n_modes)(z)                          # (L, C)
        z = UFourierLayer1D(C, self.n_modes)(z)                         # (L, C)
        z = UFourierLayer1D(C, self.n_modes)(z)                         # (L, C)

        # ── Output head -> harmonic coefficients ──────────────────────
        z      = nn.Dense(64)(z)                                        # (L, 64)
        z      = nn.relu(z)
        coeffs = nn.Dense(NC)(z)                                        # (L, NC)

        # ── softplus on a0 only; small init on harmonics ──────────────
        a0        = nn.softplus(coeffs[:, :1])                          # (L, 1)
        harmonics = coeffs[:, 1:] * 0.1                                 # (L, NC-1)
        return jnp.concatenate([a0, harmonics], axis=-1)                # (L, NC)


# ═══════════════════════════════════════════════════════════════════════
# 6.  Utilities
# ═══════════════════════════════════════════════════════════════════════

def count_params(params: dict) -> int:
    """Return total number of scalar parameters in a Flax param tree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def build_model(
    g            : jnp.ndarray,
    psi_flat     : jnp.ndarray,
    rho_flat_eq  : jnp.ndarray,
    xi           : jnp.ndarray,
    seed         : int = 0,
    branch_hidden  : Sequence[int] = (256, 256),
    trunk_hidden   : Sequence[int] = (64, 128, 128),
    n_channels     : int   = 64,
    n_modes        : int   = 16,
    n_radial       : int   = cfg.N_RADIAL,
    n_harmonics    : int   = cfg.N_HARMONICS,
    n_eq_channels  : int   = cfg.N_EQ_CHANNELS,
    rff_features   : int   = 64,
    rff_sigma      : float = 1.0,
) -> ModelBundle:
    """
    Instantiate FourierDeepONetV8 and initialise parameters.

    Parameters
    ----------
    g            : (n_chords,)  example chord array (shape init only).
    psi_flat     : (N_GRID²,)   example psi field (shape init only).
    rho_flat_eq  : (N_GRID²,)   example rho field (shape init only).
    xi           : (9,)         example hardware vector (shape init only).
    seed         : int
    """
    model = FourierDeepONetV8(
        branch_hidden  = branch_hidden,
        trunk_hidden   = trunk_hidden,
        n_channels     = n_channels,
        n_modes        = n_modes,
        n_radial       = n_radial,
        n_harmonics    = n_harmonics,
        n_eq_channels  = n_eq_channels,
        rff_features   = rff_features,
        rff_sigma      = rff_sigma,
    )
    key    = jax.random.PRNGKey(seed)
    params = model.init(key, g, psi_flat, rho_flat_eq, xi)
    return ModelBundle(model=model, params=params)


def verify_model(
    bundle       : ModelBundle,
    g            : jnp.ndarray,
    psi_flat     : jnp.ndarray,
    rho_flat_eq  : jnp.ndarray,
    xi           : jnp.ndarray,
) -> None:
    """
    Run a single forward pass and print a diagnostic summary.
    Raises AssertionError if the output contains NaNs.
    """
    model, params = bundle
    NC = model.n_out_channels

    coeffs = model.apply(params, g, psi_flat, rho_flat_eq, xi)

    n_params  = count_params(params)
    nan_count = int(jnp.any(jnp.isnan(coeffs)))

    print("── model.py  FourierDeepONetV8 ─────────────────────────")
    print(f"  Parameters      : {n_params:,}")
    print(f"  N_HARMONICS     : {model.n_harmonics}  (N_CHANNELS={NC})")
    print(f"  Output shape    : {coeffs.shape}  "
          f"(expect ({model.n_radial}, {NC}))")
    print(f"  a0 range        : [{float(coeffs[:,0].min()):.4f}, "
          f"{float(coeffs[:,0].max()):.4f}]  (>= 0 via softplus)")
    print(f"  harmonics range : [{float(coeffs[:,1:].min()):.4f}, "
          f"{float(coeffs[:,1:].max()):.4f}]  (unconstrained)")
    print(f"  NaN in output   : {nan_count}  (must be 0)")
    print("────────────────────────────────────────────────────────")

    assert nan_count == 0, "Forward pass produced NaNs."
    print("OK  model.py v8 verified")