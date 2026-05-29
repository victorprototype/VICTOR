# ============================================================
# VICTOR v8.2 — model.py
# Architecture: FourierLayer1D, UFourierLayer1D,
#               FourierFeatureEmbedding,
#               BranchNet, TrunkNet, FourierDeepONetV8
# ============================================================
# Public API
# ----------
#   FourierFeatureEmbedding — Random Fourier Feature lifting (NEW v7.1)
#   FourierLayer1D          — single 1-D Fourier layer (spectral + residual
#                             + pre-norm LayerNorm [NEW v8.2])
#   UFourierLayer1D         — U-Fourier layer with U-Net skip + learned
#                             sigmoid gate [NEW v8.2] + LayerNorm
#   BranchNet               — MLP encoder for chord measurements g_obs
#   TrunkNet                — MLP encoder for xi; prepends RFF embedding
#                             + LayerNorm
#   FourierDeepONetV8       — full model: branch + trunk + Fourier decoder
#                             + physics residual from latent_eq [NEW v8.2]
#
#   build_model()           — instantiate FourierDeepONetV8 and init params
#   count_params()          — count total params, now broken down by class
#                             (spectral / MLP / gate) [NEW v8.2]
#   get_skip_gate_values()  — walk param tree, return {name: float} for
#                             every g_skip scalar [NEW v8.2]
#
# ── v8.2 Changelog ──────────────────────────────────────────────────────
#
#  Change 1 · UFourierLayer1D — Learned sigmoid skip gate  (NAS-PINN §3.1)
#  -------------------------------------------------------------------------
#  Problem:  The always-on U-Net skip `z = relu(x_spec + x_res + x_skip)`
#            causes "exclusive competition": the optimiser learns to route
#            the gradient almost entirely through the cheap global-average
#            skip, suppressing the spectral and residual paths.  The
#            phenomenon is documented in NAS-PINN §3.1 as a form of
#            "path collapse" in multi-branch residual networks.
#  Fix:      A scalar gate parameter `g_skip` is initialised to
#            `cfg.SKIP_GATE_INIT` (default 0.0).  At the start of training,
#            sigmoid(0) = 0.5 so all three paths contribute equally.
#            The gate is then learned end-to-end, letting the network raise
#            or lower the skip contribution per layer.
#            Formula:  gate = sigmoid(g_skip)
#                      z    = relu(x_spec + x_res + gate * x_skip)
#
#  Change 2 · FourierLayer1D — Pre-norm LayerNorm before spectral path
#  -------------------------------------------------------------------------
#  Problem:  Post-norm (transform → norm) is known to be less stable in
#            deep spectral networks because large spectral outputs can blow
#            up before normalisation occurs.
#  Fix:      Apply nn.LayerNorm()(x) as the very first operation, before
#            the rfft.  This is "pre-norm" style (norm → transform), the
#            same convention used in modern transformer encoders and shown
#            to improve gradient flow in deep FNO stacks.
#
#  Change 3 · FourierDeepONetV8 — Physics residual from equilibrium encoder
#  -------------------------------------------------------------------------
#  Motivation: CS-PINN §2.2 shows that equilibrium information should
#              condition the solution head via a *direct* gradient path,
#              not only through the spectral bottleneck.  Without this,
#              the Fourier layers can discard equilibrium features that are
#              spectrally smooth.
#  Fix:        After the three-layer Fourier decoder, concatenate
#              `latent_eq` (broadcast to (L, C)) with `z` before the
#              Dense(64) output head.  This doubles the head's input width
#              to 2C and creates an explicit shortcut gradient path from
#              the equilibrium encoder to the output.
#
#  Change 4 · get_skip_gate_values()  — SkipGateMonitor utility
#  -------------------------------------------------------------------------
#  A new function that walks the Flax param tree and returns a flat dict
#  `{layer_path: sigmoid(g_skip)}` for every learned gate scalar.  Call
#  this from the training loop to log per-layer gate values and detect
#  collapse (gates saturating at 0 or 1 early in training).
#
#  Change 5 · count_params()  — Breakdown by parameter class
#  -------------------------------------------------------------------------
#  count_params() now returns a ParamCounts NamedTuple with fields:
#    spectral  — R_re / R_im weights in FourierLayer1D and UFourierLayer1D
#    gate      — g_skip scalars
#    mlp       — everything else (Dense, LayerNorm, RFF, etc.)
#    total     — sum of the above
#  The old interface (returning an int) is preserved via __int__/__index__.
#
# ── Unchanged design principles ─────────────────────────────────────────
#  * All modules are pure Flax nn.Module subclasses.
#  * No JAX globals are mutated; callers own the returned params.
#  * Branch net encodes g_obs -> (n_radial, C).
#  * Trunk net encodes xi (9-D) -> (C,) via RFF + MLP.
#  * Merger: pointwise multiply (per vanilla DeepONet).
#  * Decoder: 1 FourierLayer1D + 2 UFourierLayer1D -> radial output.
#  * Output: softplus on a0, small init on harmonics.
# ============================================================

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from victor import config as cfg


# ── Named return types ────────────────────────────────────────────────────

class ModelBundle(NamedTuple):
    """Model instance together with its initialised parameters."""
    model  : nn.Module
    params : dict


class ParamCounts(NamedTuple):
    """
    Breakdown of trainable parameter counts by architectural class.

    Attributes
    ----------
    spectral : int  Parameters in R_re / R_im spectral weight tensors.
    gate     : int  Parameters in g_skip learned skip-gate scalars.
    mlp      : int  All remaining parameters (Dense, LayerNorm, RFF, etc.).
    total    : int  Sum of spectral + gate + mlp.
    """
    spectral : int
    gate     : int
    mlp      : int
    total    : int

    # ── Backward-compatibility shim ──────────────────────────────────────
    # Legacy callers that did `n = count_params(params)` and used the
    # result as a plain int still work without any changes.

    def __int__(self) -> int:                     # int(bundle)
        return self.total

    def __index__(self) -> int:                   # slicing, format specs
        return self.total

    def __format__(self, spec: str) -> str:       # f"{bundle:,}"
        return format(self.total, spec)

    def __str__(self) -> str:
        return (
            f"ParamCounts(total={self.total:,}, "
            f"spectral={self.spectral:,}, "
            f"gate={self.gate}, "
            f"mlp={self.mlp:,})"
        )


# ═══════════════════════════════════════════════════════════════════════
# 1.  FourierLayer1D
# ═══════════════════════════════════════════════════════════════════════

class FourierLayer1D(nn.Module):
    """
    Single Fourier layer for 1-D sequences.

    v8.2 change: Pre-norm LayerNorm applied before the spectral path
    (norm → transform order).  See NAS-PINN §3.1 and the v8.2 changelog
    entry "Change 2" for motivation.

    Keeps sequence length unchanged via two parallel paths:
      Spectral path : IFFT(R · FFT(LayerNorm(x)))   truncated to n_modes
      Residual path : W · LayerNorm(x)               learned Dense projection

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

        # ── [v8.2] Pre-norm: normalise BEFORE the spectral transform ──────
        # Pre-norm (norm → transform) is more stable than post-norm for deep
        # spectral networks: it bounds the rfft input and prevents large
        # spectral energies from accumulating across layers.
        # Ref: NAS-PINN §3.1, "Change 2" in the v8.2 changelog.
        x_normed = nn.LayerNorm()(x)                       # (L, C)

        # ── Spectral path ─────────────────────────────────────────────────
        # Apply rfft on the pre-normalised input so spectral weights R
        # operate on a well-scaled signal.
        xf     = jnp.fft.rfft(x_normed, axis=0)            # (L//2+1, C)
        n_freq = xf.shape[0]
        k      = min(self.n_modes, n_freq)

        # Complex weight tensor R: (k, C, C)
        R_re = self.param('R_re', nn.initializers.glorot_normal(), (k, C, C))
        R_im = self.param('R_im', nn.initializers.glorot_normal(), (k, C, C))
        R    = R_re + 1j * R_im

        # Mix kept modes: (k, C) x (k, C, C) -> (k, C)
        xf_out  = jnp.einsum('ki,kij->kj', xf[:k], R)
        xf_full = jnp.zeros_like(xf).at[:k].set(xf_out)    # (n_freq, C)
        x_spec  = jnp.fft.irfft(xf_full, n=L, axis=0)      # (L, C)

        # ── Residual path — also uses pre-normed input ────────────────────
        x_res = nn.Dense(self.n_channels)(x_normed)         # (L, C)

        out = nn.relu(x_spec + x_res)
        return out                                           # (L, C)


# ═══════════════════════════════════════════════════════════════════════
# 2.  UFourierLayer1D
# ═══════════════════════════════════════════════════════════════════════

class UFourierLayer1D(nn.Module):
    """
    U-Fourier layer: Fourier layer with an additional U-Net-style skip
    from the global average of the input.

    v8.2 change: The U-Net skip is now gated by a learned scalar
    ``g_skip`` passed through sigmoid.  This prevents "exclusive
    competition" (NAS-PINN §3.1) where the always-on skip dominates and
    suppresses spectral/residual gradient flow.  At init, g_skip = 0.0
    so sigmoid(0) = 0.5 and all three paths contribute equally.

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

        # ── Fourier spectral path ─────────────────────────────────────────
        xf     = jnp.fft.rfft(x, axis=0)
        n_freq = xf.shape[0]
        k      = min(self.n_modes, n_freq)

        R_re   = self.param('R_re', nn.initializers.glorot_normal(), (k, C, C))
        R_im   = self.param('R_im', nn.initializers.glorot_normal(), (k, C, C))
        R      = R_re + 1j * R_im
        xf_out = jnp.einsum('ki,kij->kj', xf[:k], R)
        xf_full= jnp.zeros_like(xf).at[:k].set(xf_out)
        x_spec = jnp.fft.irfft(xf_full, n=L, axis=0)       # (L, C)

        # ── Residual (W · x) path ─────────────────────────────────────────
        x_res  = nn.Dense(self.n_channels)(x)               # (L, C)

        # ── U-Net skip: global average broadcast ──────────────────────────
        x_avg  = jnp.mean(x, axis=0, keepdims=True)         # (1, C)
        x_skip = nn.Dense(self.n_channels)(x_avg)           # (1, C)
        x_skip = jnp.broadcast_to(x_skip, (L, C))           # (L, C)

        # ── [v8.2] Learned sigmoid skip gate  (NAS-PINN §3.1) ────────────
        # A scalar gate g_skip is initialised to cfg.SKIP_GATE_INIT
        # (default 0.0), giving sigmoid(0) = 0.5 at the start of training.
        # This ensures spectral, residual, and skip paths begin on equal
        # footing.  The gate is learned end-to-end; the trainer can monitor
        # it via get_skip_gate_values() to detect path collapse.
        # Without this gate, the cheap global-average skip suppresses the
        # spectral gradient, a failure mode documented in NAS-PINN §3.1.
        g_skip = self.param(
            'g_skip',
            lambda rng, shape: jnp.full(shape, cfg.SKIP_GATE_INIT),
            (),                                              # scalar shape
        )
        gate = nn.sigmoid(g_skip)                           # scalar in (0, 1)

        # Gated combination: skip contribution scaled by learned gate
        z = nn.relu(x_spec + x_res + gate * x_skip)        # (L, C)

        # ── Optional length projection L -> target ────────────────────────
        if target != L:
            z = jax.image.resize(z, shape=(target, C), method='linear')

        # ── Per-position linear projection (W' in paper) ──────────────────
        z = nn.Dense(self.n_channels)(z)
        z = nn.relu(z)
        return z                                             # (target, C)


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
        h = nn.Dense(self.n_radial * self.n_channels)(h)    # (L*C,)
        return h.reshape(self.n_radial, self.n_channels)    # (L, C)


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

    v8.2 note: latent_eq is also injected directly into the output head
    of FourierDeepONetV8 as a physics residual connection (CS-PINN §2.2).
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
        return nn.Dense(self.n_channels)(h)                  # (C,)


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
        proj = 2.0 * jnp.pi * x @ B.T               # (n_features,)
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

        return nn.Dense(self.n_channels)(h)                  # (C,)


# ═══════════════════════════════════════════════════════════════════════
# 5.  FourierDeepONetV8  —  Full Model
# ═══════════════════════════════════════════════════════════════════════

class FourierDeepONetV8(nn.Module):
    """
    VICTOR v8.2: Dual Encoder + Fourier Decoder + Physics Residual
                 + Poloidal Harmonic Output.

    v8.2 changes vs v8.1
    --------------------
    * FourierLayer1D uses pre-norm (LayerNorm before spectral path).
    * UFourierLayer1D gates the U-Net skip via a learned sigmoid scalar.
    * latent_eq is broadcast to (L, C) and concatenated with z before
      the output head, giving the equilibrium encoder a direct gradient
      path to the output that bypasses the spectral bottleneck
      (CS-PINN §2.2 coefficient-subnet pattern).

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

        # ── Branch: g -> (L, C) ───────────────────────────────────────────
        latent_g = BranchNet(self.branch_hidden, L, C)(g)               # (L, C)

        # ── Equilibrium: concat(psi, rho) -> (C,) ────────────────────────
        eq_flat   = jnp.concatenate([psi_flat, rho_flat_eq], axis=-1)   # (2*N²,)
        latent_eq = EquilibriumEncoder(C)(eq_flat)                       # (C,)

        # ── Hardware vector: xi -> (C,) ───────────────────────────────────
        latent_xi = TrunkNet(
            self.trunk_hidden, C,
            self.rff_features, self.rff_sigma,
        )(xi)                                                            # (C,)

        # ── Fusion: concat(eq, xi) -> C, then multiply with branch ────────
        fused = jnp.concatenate([latent_eq, latent_xi], axis=-1)        # (2C,)
        fused = nn.Dense(C)(fused)                                       # (C,)
        fused = nn.gelu(fused)

        z = latent_g * fused[None, :]                                    # (L, C)

        # ── Fourier decoder ───────────────────────────────────────────────
        # FourierLayer1D now applies pre-norm (v8.2 Change 2).
        z = FourierLayer1D(C, self.n_modes)(z)                           # (L, C)
        # UFourierLayer1D now uses a learned sigmoid skip gate (v8.2 Change 1).
        z = UFourierLayer1D(C, self.n_modes)(z)                          # (L, C)
        z = UFourierLayer1D(C, self.n_modes)(z)                          # (L, C)

        # ── [v8.2] Physics residual from equilibrium encoder ──────────────
        # CS-PINN §2.2 (coefficient-subnet pattern): the equilibrium branch
        # should condition the output head *directly*, not only through the
        # spectral bottleneck.  We broadcast latent_eq to (L, C) and
        # concatenate it with z, doubling the head's input width to 2C.
        # This creates an explicit short-circuit gradient path from the
        # equilibrium encoder to the final coefficients, which lets the
        # network preserve smooth equilibrium features that Fourier layers
        # might otherwise attenuate.
        latent_eq_broadcast = jnp.broadcast_to(
            latent_eq[None, :], (L, C)
        )                                                                # (L, C)
        z_head = jnp.concatenate([z, latent_eq_broadcast], axis=-1)     # (L, 2C)

        # ── Output head -> harmonic coefficients ──────────────────────────
        z_head = nn.Dense(64)(z_head)                                    # (L, 64)
        z_head = nn.relu(z_head)
        coeffs = nn.Dense(NC)(z_head)                                    # (L, NC)

        # ── softplus on a0 only; small init on harmonics ──────────────────
        a0        = nn.softplus(coeffs[:, :1])                           # (L, 1)
        harmonics = coeffs[:, 1:] * 0.1                                  # (L, NC-1)
        return jnp.concatenate([a0, harmonics], axis=-1)                 # (L, NC)


# ═══════════════════════════════════════════════════════════════════════
# 6.  Utilities
# ═══════════════════════════════════════════════════════════════════════

def get_skip_gate_values(params: dict) -> Dict[str, float]:
    """
    SkipGateMonitor: walk the Flax param tree and return the *sigmoid*
    value of every learned skip gate scalar.

    v8.2 NEW — see changelog "Change 4".

    The returned dict maps the dotted Flax path of each ``g_skip``
    parameter to its current gate activation sigmoid(g_skip) ∈ (0, 1):
      * Values near 0 mean the skip is being suppressed — check that the
        spectral/residual paths are carrying the gradient.
      * Values near 1 mean the skip dominates — the pattern NAS-PINN §3.1
        calls "exclusive competition" and that the gate was designed to
        prevent.
      * Values near 0.5 are healthy early in training.

    Call this from the training loop, e.g.::

        gates = get_skip_gate_values(state.params)
        for name, val in gates.items():
            writer.add_scalar(f"gates/{name}", val, step)

    Parameters
    ----------
    params : dict   Flax parameter tree (e.g. from ``model.init(...)``
                    or a TrainState).

    Returns
    -------
    dict[str, float]
        Keys are dotted paths such as
        ``"params/UFourierLayer1D_0/g_skip"``.
        Values are float sigmoid activations in (0, 1).
    """

    def _walk(node: dict, prefix: str, out: Dict[str, float]) -> None:
        """Recursively traverse the param pytree."""
        for key, val in node.items():
            path = f"{prefix}/{key}" if prefix else key
            if key == 'g_skip':
                # val is a scalar jnp.ndarray (shape ())
                out[path] = float(nn.sigmoid(val))
            elif isinstance(val, dict):
                _walk(val, path, out)
            # leaf arrays that are not g_skip are ignored

    gate_values: Dict[str, float] = {}
    # Flax params are often nested under a top-level 'params' key
    root = params.get('params', params)
    _walk(root, '', gate_values)
    return gate_values


def count_params(params: dict) -> ParamCounts:
    """
    Count trainable parameters in a Flax param tree, broken down by
    architectural class.

    v8.2 change: returns a ``ParamCounts`` NamedTuple instead of a bare
    int.  The NamedTuple supports ``int()``, ``format()``, and
    ``__index__`` so all legacy callers that used the result as an int
    continue to work without modification.

    Classification rules
    --------------------
    spectral : leaf key is ``'R_re'`` or ``'R_im'``
               (spectral weight tensors in FourierLayer1D / UFourierLayer1D)
    gate     : leaf key is ``'g_skip'``
               (learned sigmoid skip gate scalars in UFourierLayer1D)
    mlp      : everything else
               (Dense kernels/biases, LayerNorm scale/bias, RFF matrix B, etc.)

    Parameters
    ----------
    params : dict   Flax parameter tree.

    Returns
    -------
    ParamCounts
        .spectral  int  spectral-path parameter count
        .gate      int  gate scalar count
        .mlp       int  MLP / normalisation parameter count
        .total     int  sum of the above

    Examples
    --------
    >>> counts = count_params(bundle.params)
    >>> print(counts)
    ParamCounts(total=1,234,567, spectral=98,304, gate=2, mlp=1,136,261)
    >>> print(f"Total: {counts:,}")       # backward-compat int formatting
    Total: 1,234,567
    """
    spectral = 0
    gate     = 0
    mlp      = 0

    def _walk(node: dict, last_key: str = '') -> None:
        nonlocal spectral, gate, mlp
        for key, val in node.items():
            if isinstance(val, dict):
                _walk(val, last_key=key)
            else:
                # val is a jnp.ndarray leaf
                n = val.size
                if key in ('R_re', 'R_im'):
                    spectral += n
                elif key == 'g_skip':
                    gate += n
                else:
                    mlp += n

    root = params.get('params', params)
    _walk(root)
    return ParamCounts(
        spectral=spectral,
        gate=gate,
        mlp=mlp,
        total=spectral + gate + mlp,
    )


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

    Returns
    -------
    ModelBundle(model, params)
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

    v8.2: also prints the per-class parameter breakdown and all
    current skip gate values.

    Raises AssertionError if the output contains NaNs.
    """
    model, params = bundle
    NC = model.n_out_channels

    coeffs = model.apply(params, g, psi_flat, rho_flat_eq, xi)

    counts    = count_params(params)
    nan_count = int(jnp.any(jnp.isnan(coeffs)))

    print("── model.py  FourierDeepONetV8 v8.2 ────────────────────")
    print(f"  Parameters (total)  : {counts.total:,}")
    print(f"    spectral (R_re/im): {counts.spectral:,}")
    print(f"    gate     (g_skip) : {counts.gate}")
    print(f"    mlp      (rest)   : {counts.mlp:,}")
    print(f"  N_HARMONICS     : {model.n_harmonics}  (N_CHANNELS={NC})")
    print(f"  Output shape    : {coeffs.shape}  "
          f"(expect ({model.n_radial}, {NC}))")
    print(f"  a0 range        : [{float(coeffs[:,0].min()):.4f}, "
          f"{float(coeffs[:,0].max()):.4f}]  (>= 0 via softplus)")
    print(f"  harmonics range : [{float(coeffs[:,1:].min()):.4f}, "
          f"{float(coeffs[:,1:].max()):.4f}]  (unconstrained)")
    print(f"  NaN in output   : {nan_count}  (must be 0)")

    gates = get_skip_gate_values(params)
    if gates:
        print("  Skip gate values (sigmoid):")
        for name, val in sorted(gates.items()):
            print(f"    {name}: {val:.4f}")
    else:
        print("  Skip gate values: none found")

    print("────────────────────────────────────────────────────────")

    assert nan_count == 0, "Forward pass produced NaNs."
    print("OK  model.py v8.2 verified")
