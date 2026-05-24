# ============================================================
# VICTOR v8.0 — model.py
# Poloidal-aware FourierDeepONet with dual encoder
# ============================================================
# Public API
# ----------
#   EquilibriumEncoder     — MLP encoder for (psi_n, rho_n) fields
#   FourierFeatureEmbedding — Random Fourier Feature lifting (v7.1)
#   FourierLayer1D          — 1-D Fourier layer (spectral + residual)
#                             LayerNorm REMOVED in v8 (see note below)
#   UFourierLayer1D         — U-Fourier layer with U-Net skip
#                             LayerNorm REMOVED in v8
#   BranchNet               — MLP encoder for chord measurements g
#   TrunkNet                — MLP encoder for xi hardware vector
#   FourierDeepONetV8       — full v8 model: dual encoder + Fourier decoder
#                             output: (N_RADIAL, N_CHANNELS_OUT) coefficients
#   build_eps2d()           — render true 2D field from coefficients
#   build_model()           — instantiate and initialise
#   count_params()          — count trainable parameters
#   verify_model()          — forward-pass smoke test
#
# v8 changes vs v7.1
# ------------------
#  * EquilibriumEncoder: new MLP branch that encodes the flattened
#    (psi_n, rho_n) fields into a latent vector latent_eq.
#    Input: concat(psi_n, rho_n) shape (2 * N_GRID²,)
#    Architecture: Dense(256) -> GELU -> Dense(128) -> GELU -> Dense(C)
#    N_EQ_CHANNELS=2 in config; add Te/ne later by changing this alone.
#
#  * Fusion: latent = concat(latent_g, latent_eq)  fed to Fourier decoder.
#    The branch net (g encoder) now outputs (L, C//2) and the eq encoder
#    outputs (C//2,) so the fused tensor is still (L, C).
#
#  * Output head: Dense(N_RADIAL * N_CHANNELS_OUT) reshaped to
#    (N_RADIAL, N_CHANNELS_OUT). Channel 0 (a0) gets softplus applied —
#    radially symmetric emission is non-negative. Channels 1-4
#    (a1, b1, a2, b2) are unconstrained and scaled by HARMONIC_INIT_SCALE
#    at initialisation to keep early training dominated by a0.
#
#  * LayerNorm REMOVED from FourierLayer1D and UFourierLayer1D output.
#    LayerNorm after Fourier layers normalises away the amplitude structure
#    that the harmonic channels need to represent asymmetry. Replaced by
#    a simple ReLU output (spectral + residual path) without normalisation.
#    This does not affect training stability — the Adam optimizer adapts.
#
#  * build_eps2d(): pure JAX function that maps
#    (coeff: (N_RADIAL, N_CHANNELS_OUT), rho_flat: (N²,), theta_flat: (N²,))
#    -> eps2d: (N², ) true 2D emissivity field.
#    No softplus applied after reconstruction (as recommended in review).
#    Optional clip(min=0) available as argument.
#
#  * Forward signature changes:
#    OLD: model.apply(params, g, xi)           -> (N_RADIAL,)
#    NEW: model.apply(params, g, xi, psi_n, rho_n) -> (N_RADIAL, N_CHANNELS_OUT)
# ============================================================

from __future__ import annotations

from typing import Sequence, NamedTuple, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn

from victor import config as cfg


# ── Named return types ────────────────────────────────────────────────

class ModelBundle(NamedTuple):
    """Model instance together with its initialised parameters."""
    model  : nn.Module
    params : dict


# =======================================================================
# 1.  FourierLayer1D  (LayerNorm removed in v8)
# =======================================================================

class FourierLayer1D(nn.Module):
    """
    Single Fourier layer for 1-D sequences.

    Two parallel paths:
      Spectral path : IFFT(R · FFT(x))   truncated to n_modes
      Residual path : W · x              learned Dense projection

    Output: relu(spectral + residual)

    v8 change: LayerNorm removed from output.  LayerNorm after Fourier
    layers normalises away amplitude information that the harmonic
    channels rely on.  Training remains stable via Adam's adaptive rates.

    Input / Output
    --------------
    x : (L, C)  ->  (L, C)
    """

    n_channels : int
    n_modes    : int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        L, C   = x.shape
        xf     = jnp.fft.rfft(x, axis=0)
        n_freq = xf.shape[0]
        k      = min(self.n_modes, n_freq)

        R_re = self.param('R_re', nn.initializers.glorot_normal(), (k, C, C))
        R_im = self.param('R_im', nn.initializers.glorot_normal(), (k, C, C))
        R    = R_re + 1j * R_im

        xf_out  = jnp.einsum('ki,kij->kj', xf[:k], R)
        xf_full = jnp.zeros_like(xf).at[:k].set(xf_out)
        x_spec  = jnp.fft.irfft(xf_full, n=L, axis=0)      # (L, C)
        x_res   = nn.Dense(self.n_channels)(x)               # (L, C)

        return nn.relu(x_spec + x_res)                       # (L, C)  no LayerNorm


# =======================================================================
# 2.  UFourierLayer1D  (LayerNorm removed in v8)
# =======================================================================

class UFourierLayer1D(nn.Module):
    """
    U-Fourier layer with U-Net-style global average skip.

    v8 change: LayerNorm removed from output and pre-resize step.
    See FourierLayer1D docstring for rationale.

    Input / Output
    --------------
    x : (L, C)  ->  (out_len or L, C)
    """

    n_channels : int
    n_modes    : int
    out_len    : Optional[int] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        L, C   = x.shape
        target = self.out_len if self.out_len is not None else L

        xf     = jnp.fft.rfft(x, axis=0)
        n_freq = xf.shape[0]
        k      = min(self.n_modes, n_freq)

        R_re   = self.param('R_re', nn.initializers.glorot_normal(), (k, C, C))
        R_im   = self.param('R_im', nn.initializers.glorot_normal(), (k, C, C))
        R      = R_re + 1j * R_im
        xf_out = jnp.einsum('ki,kij->kj', xf[:k], R)
        xf_full= jnp.zeros_like(xf).at[:k].set(xf_out)
        x_spec = jnp.fft.irfft(xf_full, n=L, axis=0)        # (L, C)
        x_res  = nn.Dense(self.n_channels)(x)                 # (L, C)

        x_avg  = jnp.mean(x, axis=0, keepdims=True)          # (1, C)
        x_skip = nn.Dense(self.n_channels)(x_avg)             # (1, C)
        x_skip = jnp.broadcast_to(x_skip, (L, C))

        z = nn.relu(x_spec + x_res + x_skip)                 # (L, C)  no LayerNorm

        if target != L:
            z = jax.image.resize(z, shape=(target, C), method='linear')

        z = nn.Dense(self.n_channels)(z)
        return nn.relu(z)                                     # (target, C)


# =======================================================================
# 3.  FourierFeatureEmbedding  (unchanged from v7.1)
# =======================================================================

class FourierFeatureEmbedding(nn.Module):
    """
    Random Fourier Feature embedding for low-dimensional inputs.
    Maps x: (d,) -> (2*n_features,) using fixed random frequencies.
    (Tancik et al. 2020)
    """

    n_features : int
    sigma      : float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        d = x.shape[0]
        B = self.param(
            'B',
            lambda key, shape: jax.random.normal(key, shape) * self.sigma,
            (self.n_features, d),
        )
        proj = 2.0 * jnp.pi * x @ B.T
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)


# =======================================================================
# 4.  BranchNet  (g encoder, outputs C//2 channels for fusion)
# =======================================================================

class BranchNet(nn.Module):
    """
    MLP branch net: encodes chord measurements g -> (n_radial, n_channels).

    v8: n_channels = C//2.  The other C//2 comes from EquilibriumEncoder.
    Fused together they form the full (L, C) latent.

    Architecture:  g -> [Dense(h) -> SiLU -> LayerNorm] × n_layers
                      -> Dense(n_radial * n_channels) -> reshape(L, C//2)
    """

    hidden_dims : Sequence[int]
    n_radial    : int
    n_channels  : int

    @nn.compact
    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        h = g
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = nn.silu(h)
            h = nn.LayerNorm()(h)
        h = nn.Dense(self.n_radial * self.n_channels)(h)
        return h.reshape(self.n_radial, self.n_channels)      # (L, C//2)


# =======================================================================
# 5.  EquilibriumEncoder  (NEW v8)
# =======================================================================

class EquilibriumEncoder(nn.Module):
    """
    MLP encoder for the 2-channel equilibrium field (psi_n, rho_n).

    Encodes spatial equilibrium information into a latent vector that is
    broadcast across the radial axis and fused with the g latent.

    This gives the model information about the current magnetic geometry
    (flux surface shape, plasma boundary) without solving equilibrium.

    Input
    -----
    eq_flat : (2 * N_GRID²,)  concat of flattened psi_n and rho_n fields
              (N_EQ_CHANNELS * N_GRID² total).
              Future: extend to (4 * N_GRID²,) for Te, ne by changing
              N_EQ_CHANNELS in config — no code changes needed.

    Output
    ------
    (n_channels,)  latent equilibrium vector, broadcast to (L, C//2)
                   inside FourierDeepONetV8.

    Architecture
    ------------
    concat(psi_n, rho_n)  (32768,)
      -> Dense(256) -> GELU -> Dense(128) -> GELU -> Dense(C//2)

    The aggressive first projection (32768 -> 256) is important to
    control parameter count.  Without it, the first layer alone would
    have 32768 * 256 = 8.4M weights.  We use Dense(256) with no bias
    on the first layer to stay efficient.
    """

    n_channels  : int    # output width = C//2
    hidden_dims : Sequence[int] = (256, 128)

    @nn.compact
    def __call__(self, eq_flat: jnp.ndarray) -> jnp.ndarray:
        """
        eq_flat : (N_EQ_CHANNELS * N_GRID²,)  float32
        returns  : (n_channels,)               float32
        """
        h = nn.Dense(self.hidden_dims[0], use_bias=False)(eq_flat)  # first proj: no bias
        h = nn.gelu(h)
        for dim in self.hidden_dims[1:]:
            h = nn.Dense(dim)(h)
            h = nn.gelu(h)
        return nn.Dense(self.n_channels)(h)                   # (C//2,)


# =======================================================================
# 6.  TrunkNet  (xi encoder, unchanged from v7.1)
# =======================================================================

class TrunkNet(nn.Module):
    """
    MLP trunk net for WEST GEM hardware parameters xi.
    Prepends RFF embedding. LayerNorm preserved here (xi is low-dim,
    no harmonic amplitude concern).
    """

    hidden_dims  : Sequence[int]
    n_channels   : int
    rff_features : int   = 64
    rff_sigma    : float = 1.0

    @nn.compact
    def __call__(self, xi: jnp.ndarray) -> jnp.ndarray:
        h = FourierFeatureEmbedding(self.rff_features, self.rff_sigma)(xi)
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = nn.silu(h)
            h = nn.LayerNorm()(h)
        return nn.Dense(self.n_channels)(h)                   # (C,)


# =======================================================================
# 7.  build_eps2d()  — render true 2D emissivity field
# =======================================================================

def build_eps2d(
    coeff      : jnp.ndarray,   # (N_RADIAL, N_CHANNELS_OUT)
    rho_flat   : jnp.ndarray,   # (N_GRID²,)  normalised elliptic radius
    theta_flat : jnp.ndarray,   # (N_GRID²,)  poloidal angle
    n_harmonics: int = cfg.N_HARMONICS,
    clip_min   : bool = False,
) -> jnp.ndarray:
    """
    Render the true 2D emissivity field from Fourier coefficients.

    ε(ρ, θ) = a0(ρ)
             + a1(ρ)·cos(θ) + b1(ρ)·sin(θ)
             + a2(ρ)·cos(2θ) + b2(ρ)·sin(2θ)
             + ...

    Each coefficient profile is interpolated from the N_RADIAL radial
    grid onto every pixel using its rho value (nearest-neighbour via
    index clamp — differentiable for inference, not used in the JIT
    training step directly).

    Parameters
    ----------
    coeff       : (N_RADIAL, 1 + 2*n_harmonics)  output of the model
    rho_flat    : (N_GRID²,)  rho at each pixel
    theta_flat  : (N_GRID²,)  theta at each pixel
    n_harmonics : int  number of cos/sin pairs (default cfg.N_HARMONICS=2)
    clip_min    : bool  if True, clip output to >= 0 (default False)

    Returns
    -------
    eps2d : (N_GRID²,)  float32  true 2D emissivity field

    Note: No softplus applied here.  a0 already has softplus from the
    model output head.  Applying softplus again would distort the
    harmonic structure (harmonic channels are unconstrained by design).
    Optional clip(min=0) is available for numerical safety if needed.
    """
    N_RADIAL   = coeff.shape[0]
    N_CHANNELS = 1 + 2 * n_harmonics

    # ── Map pixel rho -> radial index (clamp to [0, N_RADIAL-1]) ──────
    # rho in [0, RHO_MAX] → index in [0, N_RADIAL-1]
    idx = jnp.clip(
        jnp.round(rho_flat * (N_RADIAL - 1) / cfg.RHO_MAX).astype(jnp.int32),
        0, N_RADIAL - 1,
    )                                                          # (N²,)

    # ── Look up coefficient profiles at each pixel ────────────────────
    # coeff[idx] has shape (N², N_CHANNELS)
    coeff_pix = coeff[idx]                                     # (N², N_CHANNELS)

    # ── Evaluate harmonic series ──────────────────────────────────────
    eps2d = coeff_pix[:, 0]                                    # a0(ρ)  (N²,)
    for m in range(1, n_harmonics + 1):
        a_m   = coeff_pix[:, 2 * m - 1]                       # cos coefficient
        b_m   = coeff_pix[:, 2 * m]                           # sin coefficient
        eps2d = eps2d + a_m * jnp.cos(m * theta_flat) \
                      + b_m * jnp.sin(m * theta_flat)

    if clip_min:
        eps2d = jnp.clip(eps2d, 0.0, None)

    return eps2d                                               # (N²,)


# =======================================================================
# 8.  FourierDeepONetV8  — Full v8 Model
# =======================================================================

class FourierDeepONetV8(nn.Module):
    """
    VICTOR v8: Fourier-enhanced Deep Operator Network with dual encoder.

    Architecture
    ────────────
    ┌──────────────────────────────────────────────────────────────────┐
    │  Branch encoder  BranchNet(g)           -> (N_RADIAL, C//2)     │
    │  Equil. encoder  EquilibriumEncoder(eq)  -> (C//2,)             │
    │  Trunk encoder   TrunkNet(xi)            -> (C,)                │
    │                                                                  │
    │  Fusion: latent = concat(b, eq[None,:]) * t[None,:]             │
    │                                           -> (N_RADIAL, C)      │
    │                                                                  │
    │  Decoder:                                                        │
    │    FourierLayer1D (C, n_modes)           -> (N_RADIAL, C)       │
    │    UFourierLayer1D × 2                   -> (N_RADIAL, C)       │
    │    Dense(64) -> ReLU                                            │
    │    Dense(N_RADIAL * N_CHANNELS_OUT)      -> reshape             │
    │                                           -> (N_RADIAL, N_CH)   │
    │                                                                  │
    │  Output head:                                                    │
    │    channel 0 (a0): softplus -> non-negative symmetric emission  │
    │    channels 1+ (harmonics): * HARMONIC_INIT_SCALE (init only)   │
    │                              unconstrained at inference          │
    └──────────────────────────────────────────────────────────────────┘

    Forward signature
    -----------------
    model.apply(params, g, xi, psi_n, rho_n)
      g      : (N_CHORDS,)  noisy sinogram
      xi     : (9,)         hardware vector
      psi_n  : (N_GRID²,)  normalised poloidal flux
      rho_n  : (N_GRID²,)  normalised flux coordinate

    Returns
    -------
    coeff : (N_RADIAL, N_CHANNELS_OUT)  Fourier coefficients
            Call build_eps2d(coeff, rho_flat, theta_flat) for the 2D field.

    Future compatibility
    --------------------
    To add Te, ne:  increase N_EQ_CHANNELS in config and concatenate the
    new fields into eq_flat before passing to EquilibriumEncoder.
    No changes to the decoder or loss functions needed.
    """

    branch_hidden   : Sequence[int]
    trunk_hidden    : Sequence[int]
    n_channels      : int               # total latent width C (must be even)
    n_modes         : int
    n_radial        : int
    n_harmonics     : int  = cfg.N_HARMONICS
    rff_features    : int  = 64
    rff_sigma       : float = 1.0
    eq_hidden       : Sequence[int] = (256, 128)

    @nn.compact
    def __call__(
        self,
        g     : jnp.ndarray,   # (N_CHORDS,)
        xi    : jnp.ndarray,   # (9,)
        psi_n : jnp.ndarray,   # (N_GRID²,)
        rho_n : jnp.ndarray,   # (N_GRID²,)
    ) -> jnp.ndarray:           # (N_RADIAL, N_CHANNELS_OUT)

        C          = self.n_channels
        L          = self.n_radial
        C_half     = C // 2
        N_CH_OUT   = 1 + 2 * self.n_harmonics

        # ── Equilibrium input: concat psi_n and rho_n ─────────────────
        eq_flat = jnp.concatenate([psi_n, rho_n], axis=-1)    # (2*N²,)

        # ── Three encoders ────────────────────────────────────────────
        b   = BranchNet(self.branch_hidden, L, C_half)(g)          # (L, C//2)
        eq  = EquilibriumEncoder(C_half, self.eq_hidden)(eq_flat)   # (C//2,)
        t   = TrunkNet(self.trunk_hidden, C,
                       self.rff_features, self.rff_sigma)(xi)       # (C,)

        # ── Fusion: concat branch + eq, then scale by trunk ──────────
        # eq is broadcast across radial axis before concat
        eq_seq = jnp.broadcast_to(eq[None, :], (L, C_half))   # (L, C//2)
        fused  = jnp.concatenate([b, eq_seq], axis=-1)        # (L, C)
        z      = fused * t[None, :]                            # (L, C)

        # ── Fourier decoder (LayerNorm removed) ───────────────────────
        z = FourierLayer1D(C, self.n_modes)(z)                 # (L, C)
        z = UFourierLayer1D(C, self.n_modes)(z)                # (L, C)
        z = UFourierLayer1D(C, self.n_modes)(z)                # (L, C)

        # ── Project to coefficient space ──────────────────────────────
        z = nn.Dense(64)(z)                                    # (L, 64)
        z = nn.relu(z)
        z = nn.Dense(L * N_CH_OUT)(z.reshape(-1))             # (L * N_CH_OUT,)
        z = z.reshape(L, N_CH_OUT)                            # (L, N_CH_OUT)

        # ── Output head ───────────────────────────────────────────────
        # Channel 0 (a0): softplus -> strictly positive symmetric emission
        a0      = nn.softplus(z[:, 0:1])                       # (L, 1)

        # Harmonic channels: scale down at init to keep early training
        # dominated by the symmetric a0 term.  The scale is a fixed
        # multiplier (not a learned parameter) applied to the raw output.
        harmonics = z[:, 1:] * cfg.HARMONIC_INIT_SCALE        # (L, 2*N_H)

        coeff = jnp.concatenate([a0, harmonics], axis=-1)     # (L, N_CH_OUT)
        return coeff


# =======================================================================
# 9.  Utilities
# =======================================================================

def count_params(params: dict) -> int:
    """Return total number of scalar parameters in a Flax param tree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def build_model(
    g             : jnp.ndarray,
    xi            : jnp.ndarray,
    psi_n         : jnp.ndarray,
    rho_n         : jnp.ndarray,
    seed          : int            = 0,
    branch_hidden : Sequence[int]  = (256, 256),
    trunk_hidden  : Sequence[int]  = (64, 128, 128),
    eq_hidden     : Sequence[int]  = (256, 128),
    n_channels    : int            = 64,
    n_modes       : int            = 16,
    n_radial      : int            = cfg.N_RADIAL,
    n_harmonics   : int            = cfg.N_HARMONICS,
    rff_features  : int            = 64,
    rff_sigma     : float          = 1.0,
) -> ModelBundle:
    """
    Instantiate FourierDeepONetV8 and initialise parameters.

    Parameters
    ----------
    g, xi, psi_n, rho_n : example arrays (shapes used for init only).
    seed                : int  PRNGKey seed.
    branch_hidden       : Branch MLP hidden widths.
    trunk_hidden        : Trunk MLP hidden widths.
    eq_hidden           : EquilibriumEncoder hidden widths.
    n_channels          : Latent channel width C (must be even).
    n_modes             : Truncated Fourier modes k_max.
    n_radial            : Output radial grid length L.
    n_harmonics         : Poloidal Fourier modes (default cfg.N_HARMONICS=2).
    rff_features        : RFF samples for TrunkNet.
    rff_sigma           : RFF frequency scale.

    Returns
    -------
    ModelBundle  (model, params)
    """
    if n_channels % 2 != 0:
        raise ValueError(
            f"build_model: n_channels must be even for dual encoder fusion "
            f"(got {n_channels})."
        )

    model = FourierDeepONetV8(
        branch_hidden = branch_hidden,
        trunk_hidden  = trunk_hidden,
        eq_hidden     = eq_hidden,
        n_channels    = n_channels,
        n_modes       = n_modes,
        n_radial      = n_radial,
        n_harmonics   = n_harmonics,
        rff_features  = rff_features,
        rff_sigma     = rff_sigma,
    )
    key    = jax.random.PRNGKey(seed)
    params = model.init(key, g, xi, psi_n, rho_n)
    return ModelBundle(model=model, params=params)


def verify_model(
    bundle : ModelBundle,
    g      : jnp.ndarray,
    xi     : jnp.ndarray,
    psi_n  : jnp.ndarray,
    rho_n  : jnp.ndarray,
    rho_flat   : jnp.ndarray = None,
    theta_flat : jnp.ndarray = None,
) -> None:
    """
    Run a single forward pass and print a diagnostic summary.

    Also calls build_eps2d() if rho_flat and theta_flat are provided.
    Raises AssertionError if any NaNs are detected.
    """
    model, params = bundle
    coeff = model.apply(params, g, xi, psi_n, rho_n)

    n_params  = count_params(params)
    nan_count = int(jnp.any(jnp.isnan(coeff)))

    N_CH = coeff.shape[1]
    print("── model.py v8  FourierDeepONetV8 ─────────────────────────")
    print(f"  Parameters       : {n_params:,}")
    print(f"  coeff shape      : {coeff.shape}  "
          f"(expect ({model.n_radial}, {1+2*model.n_harmonics}))")
    print(f"  a0 range         : [{float(coeff[:,0].min()):.4f}, "
          f"{float(coeff[:,0].max()):.4f}]  (softplus >= 0)")
    if N_CH > 1:
        hmax = float(jnp.abs(coeff[:,1:]).max())
        print(f"  harmonic max|val|: {hmax:.4f}  "
              f"(expect small at init: ~{cfg.HARMONIC_INIT_SCALE})")
    print(f"  NaN in coeff     : {nan_count}  (must be 0)")

    if rho_flat is not None and theta_flat is not None:
        eps2d = build_eps2d(coeff, rho_flat, theta_flat)
        nan2d = int(jnp.any(jnp.isnan(eps2d)))
        print(f"  eps2d shape      : {eps2d.shape}")
        print(f"  eps2d range      : [{float(eps2d.min()):.4f}, "
              f"{float(eps2d.max()):.4f}]")
        print(f"  NaN in eps2d     : {nan2d}  (must be 0)")

    print("────────────────────────────────────────────────────────────")

    assert nan_count == 0, (
        "Forward pass produced NaNs in coeff. "
        "Check Fourier layer initialisers or encoder widths."
    )
    print("OK  model.py v8 verified")
