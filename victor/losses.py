# ============================================================
# VICTOR v8.2 — losses.py
# All individual loss functions + combined loss_fn
# ============================================================
# Public API
# ----------
#   build_eps2d(coeffs, rho_flat, theta_flat, rho_radial,
#               lerp_idx_lo, lerp_idx_hi, lerp_frac, ...)
#       -> (N_GRID²,)  flat 2D emissivity from harmonic coefficients
#
#   loss_projection(eps2d_flat, g_noisy, w_ops, active_mask)
#       -> scalar    Sinogram data-fidelity (MSE on active chords)
#                    Corresponds to L_f in PINN formulation (Lf)
#
#   loss_boundary(eps2d_flat, rho_flat, boundary_colloc_idx=None)
#       -> scalar    Hard boundary: emissivity must vanish for rho >= 1
#                    Now includes an optional collocated sub-term (L_B)
#
#   loss_smoothness_2d(eps2d_flat)
#       -> scalar    2-D total-variation regulariser
#
#   loss_positivity(eps2d_flat)
#       -> scalar    Soft penalty for negative values
#
#   loss_pde(coeffs_a0, rho_radial)
#       -> scalar    PDE-informed radial flux continuity residual
#                    Corresponds to L_I in PINN formulation
#
#   loss_monotonicity(a0_radial, rho_radial)
#       -> scalar    [NEW v8.2] Radial monotonicity soft penalty
#                    Penalises positive d(a0)/d(rho) via relu (L_I extension)
#
#   loss_poloidal_reg(coeffs)
#       -> scalar    L2 regularisation on harmonic amplitudes (channels 1..)
#
#   init_log_vars()
#       -> dict      Initial log-variance params for adaptive weighting
#                    All zeros → sigma=1.0, no initial bias.
#                    v8.2: includes "boundary_colloc" and "mono" keys.
#                    IMPORTANT: include these in value_and_grad so they
#                    are actually learned (see trainer.py §2 notes).
#
#   loss_fn(coeffs, g_noisy, w_ops, active_mask,
#           rho_flat, theta_flat, rho_radial,
#           lerp_idx_lo, lerp_idx_hi, lerp_frac,
#           boundary_colloc_idx,
#           weights, log_vars, n_harmonics)
#       -> (total_loss, loss_dict)
#
# v8.2 changes vs v8.1
# --------------------
#   [1] build_eps2d: replaced non-differentiable argmin nearest-neighbour
#       lookup with fully differentiable linear (lerp) interpolation.
#       New args: lerp_idx_lo, lerp_idx_hi, lerp_frac (from
#       geometry.build_lerp_weights(), precomputed in PixelGrids).
#       This unblocks gradient flow from L_f through the radial
#       interpolation into the Fourier decoder coefficients.
#       Old path:  coeffs_pix = coeffs[nn_idx]          ← ∂/∂coeffs = 0
#       New path:  coeffs_pix = (1-frac)*coeffs[lo] + frac*coeffs[hi]
#       When lerp_idx_lo / lerp_idx_hi / lerp_frac are None (default),
#       the old argmin fallback is used for backward compatibility.
#
#   [2] loss_boundary: added collocated boundary sub-term (L_B, §3.2).
#       When boundary_colloc_idx is provided, a second MSE term is
#       computed specifically at BOUNDARY_COLLOC_IDX pixels (rho≈1),
#       concentrating the penalty where the physics constraint matters
#       most rather than diffusely across the full grid.
#       Sub-term weighted 2.0× relative to the global mask term.
#       New optional arg: boundary_colloc_idx (default None).
#
#   [3] LossWeights.w_pde raised from 0.05 → 0.5 to bring the PDE
#       physics term (L_I) to coequal status with projection loss (L_f).
#
#   [4] loss_monotonicity: new function penalising positive radial
#       gradients in a0 (emissivity should generally decrease outward).
#       Implemented via jnp.diff + relu, fully differentiable.
#       Corresponds to the radial-monotonicity component of L_I.
#
#   [5] LossWeights: added w_mono=0.1 for the new monotonicity term.
#
#   [6] init_log_vars: added "boundary_colloc" and "mono" entries
#       so adaptive weighting covers all eight loss components.
#
#   [7] loss_fn signature extended with lerp_idx_lo, lerp_idx_hi,
#       lerp_frac, boundary_colloc_idx (all default None for backward
#       compatibility). Now assembles eight components:
#       proj, boundary, boundary_colloc, smooth, positivity,
#       pde, mono, pol.
#
#   [8] verify_losses: extended smoke-test to cover boundary_colloc
#       and mono with lerp weights and a synthetic boundary_colloc_idx.
#
# v8.1 changes vs v8.0
# --------------------
#   [1] log_vars are now JAX arrays, not Python floats.
#       init_log_vars() returns {key: jnp.zeros(())} so that
#       jax.value_and_grad can differentiate through them when they
#       are merged into the param tree.  Trainer must include them.
#
#   [2] Dynamic loss balancing (lbPINN / Kendall et al. 2018):
#       For each component i:
#           L_total += exp(-s_i) * L_i  +  beta * softplus(s_i)
#       where s_i is a learned log-variance scalar.
#       The softplus regulariser (replaces the raw 0.5*s) is strictly
#       positive and smooth → prevents s_i from collapsing to -inf
#       (which would cause exp(-s_i) → +inf = gradient explosion).
#       beta controls how aggressively variance is penalised; default 0.5.
#
#   [3] Hard prior weights (LossWeights) are reduced to gentle hints.
#       w_boundary was 5.0 → 1.0.  The adaptive sigma now controls the
#       true balance; the prior only breaks symmetry at step 0.
#       w_pde kept at 0.05 (was implicitly 0 in v8 — it was dropped from
#       loss_fn's components dict).
#
#   [4] Gradient-explosion guard: s_i clamped to [-4, +4].
#       exp(-s_min=-4) = e^4 ≈ 54.6  — highest any component can be boosted.
#       exp(-s_max=+4) = e^{-4} ≈ 0.018 — lowest suppression.
#       This is tighter than v7.1's [-5, 5].
#
#   [5] loss_pde now receives (coeffs[:, 0], rho_radial) — the radial
#       a0 profile on the clean N_RADIAL grid.  v7/v8 incorrectly passed
#       the pixel-grid rho_flat (N_GRID²), computing a meaningless
#       gradient on a 16384-point shuffled signal.
#
#   [6] loss_fn now includes ALL five components:
#       proj, boundary, smooth, positivity, pde, pol.
#       None are silently dropped.
#
#   [7] Per-component effective weight logged as "w_eff_{name}" in
#       loss_dict for diagnostic monitoring.
# ============================================================

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from victor import config as cfg


# =======================================================================
# 0a.  build_eps2d  — polar harmonic field reconstruction
# =======================================================================

def build_eps2d(
    coeffs       : jnp.ndarray,            # (N_RADIAL, 1 + 2*H)
    rho_flat     : jnp.ndarray,            # (N_GRID²,)  pixel ρ values
    theta_flat   : jnp.ndarray,            # (N_GRID²,)  pixel θ values
    rho_radial   : jnp.ndarray,            # (N_RADIAL,) model's radial axis
    # ── v8.2: lerp args from geometry.build_lerp_weights() ──────────────
    lerp_idx_lo  : Optional[jnp.ndarray] = None,  # (N_GRID²,) int
    lerp_idx_hi  : Optional[jnp.ndarray] = None,  # (N_GRID²,) int
    lerp_frac    : Optional[jnp.ndarray] = None,  # (N_GRID²,) float in [0,1]
    # ────────────────────────────────────────────────────────────────────
    n_harmonics  : int   = cfg.N_HARMONICS,
    clip_min     : float = None,
) -> jnp.ndarray:
    """
    Evaluate the polar-harmonic emissivity field on the pixel grid.

    v8.2: Replaces the non-differentiable argmin nearest-neighbour lookup
    with a fully-differentiable linear interpolation between adjacent
    radial bins.  This allows gradients of L_f (projection loss) to flow
    back through the radial interpolation into the Fourier decoder
    coefficients, which was previously blocked (∂argmin/∂coeffs = 0).

    Reference: PINN paper §3.1, Lf formulation — differentiability of
    the reconstruction operator is required for the combined gradient
    ∂(Lf + L_B + L_I)/∂θ to train the Fourier decoder end-to-end.

    Interpolation scheme
    --------------------
    For each pixel p with radius rho_p, we find the surrounding radial
    bin indices lo, hi (lo = floor index, hi = lo+1, both clamped to
    [0, N_RADIAL-1]) and the fractional position frac ∈ [0, 1]:

        coeffs_pix = (1 - frac) * coeffs[lo] + frac * coeffs[hi]

    This is exactly linear interpolation in the radial direction and is
    smooth everywhere (unlike argmin which has zero gradient almost
    everywhere and undefined gradient at bin boundaries).

    Backward compatibility
    ----------------------
    When lerp_idx_lo / lerp_idx_hi / lerp_frac are all None (e.g. old
    callers), the function falls back to the v8.1 argmin behaviour with
    a runtime warning.  New callers must pass precomputed lerp weights
    from geometry.PixelGrids.lerp_idx_lo / .lerp_idx_hi / .lerp_frac.

    Parameters
    ----------
    coeffs      : (N_RADIAL, 1+2*H)  harmonic coefficients from model
    rho_flat    : (N_GRID²,)  pixel-grid ρ  (grids.RHO_FLAT)
    theta_flat  : (N_GRID²,)  pixel-grid θ  (grids.THETA_FLAT)
    rho_radial  : (N_RADIAL,) model radial axis (grids.RHO_RADIAL)
    lerp_idx_lo : (N_GRID²,)  lower radial bin index per pixel  [v8.2]
    lerp_idx_hi : (N_GRID²,)  upper radial bin index per pixel  [v8.2]
    lerp_frac   : (N_GRID²,)  interpolation fraction ∈ [0,1]   [v8.2]
    n_harmonics : int
    clip_min    : float | None

    Returns
    -------
    eps2d_flat : (N_GRID²,)  float32
    """
    # ── v8.2: differentiable linear interpolation ────────────────────────
    if lerp_idx_lo is not None and lerp_idx_hi is not None and lerp_frac is not None:
        # Gather coefficients at the two bracketing radial bins.
        # Both index operations are standard integer gathers — JAX can
        # differentiate through the *values* at these indices via the
        # chain rule, unlike argmin which returns an integer with no grad.
        coeffs_lo  = coeffs[lerp_idx_lo]                          # (N², NC)
        coeffs_hi  = coeffs[lerp_idx_hi]                          # (N², NC)

        # Linear blend: fully differentiable w.r.t. coeffs (Lf gradient
        # flows back through here into the Fourier decoder).
        # lerp_frac[:, None] broadcasts over the NC coefficient channels.
        coeffs_pix = (                                             # (N², NC)
            (1.0 - lerp_frac[:, None]) * coeffs_lo
            +       lerp_frac[:, None]  * coeffs_hi
        )
    else:
        # ── v8.1 fallback: argmin nearest-neighbour (non-differentiable) ─
        # Retained for backward compatibility with callers that have not
        # yet been updated to pass lerp weights.  Gradients cannot flow
        # through this path.
        import warnings
        warnings.warn(
            "build_eps2d: lerp weights not supplied; falling back to "
            "non-differentiable argmin nearest-neighbour lookup (v8.1 "
            "behaviour).  Pass lerp_idx_lo / lerp_idx_hi / lerp_frac "
            "from geometry.PixelGrids to enable full gradient flow.",
            stacklevel=2,
        )
        diffs      = jnp.abs(rho_flat[:, None] - rho_radial[None, :])
        nn_idx     = jnp.argmin(diffs, axis=1)                    # (N²,)
        coeffs_pix = coeffs[nn_idx]                               # (N², NC)

    # ── Evaluate ε(ρ, θ) = a0 + Σ_n [an·cos(nθ) + bn·sin(nθ)] ─────────
    # Start with the isotropic (a0) component
    eps = coeffs_pix[:, 0]                                        # (N²,)

    # Accumulate angular harmonics  (corresponds to L_f §3.1)
    for h in range(1, n_harmonics + 1):
        a_col = 2 * h - 1
        b_col = 2 * h
        eps   = (eps
                 + coeffs_pix[:, a_col] * jnp.cos(h * theta_flat)
                 + coeffs_pix[:, b_col] * jnp.sin(h * theta_flat))

    # Zero pixels outside the model's radial domain
    eps = jnp.where(rho_flat <= cfg.RHO_MAX, eps, 0.0)

    if clip_min is not None:
        eps = jnp.clip(eps, a_min=clip_min)

    return eps                                                     # (N²,)


# =======================================================================
# 0b.  Loss weights  (prior scale hints — NOT the adaptive sigmas)
# =======================================================================

class LossWeights(NamedTuple):
    """
    Fixed prior scale multipliers for each loss component.

    These are *gentle hints* that break symmetry at initialisation.
    The adaptive log-variance parameters learned during training override
    the effective balance.  Keep all values O(1) — do NOT use large
    multipliers like 5.0 here; the sigma mechanism handles relative
    importance automatically.

    Attributes
    ----------
    w_proj            : data-fidelity (sinogram MSE) — anchor, keep 1.0
    w_boundary        : boundary zero enforcement     — was 5.0, now 1.0
    w_smooth          : 2-D total variation           — small regulariser
    w_positivity      : soft non-negativity penalty   — very small
    w_pde             : radial flux continuity (L_I)  — v8.2: raised 0.05→0.5
    w_pol             : poloidal harmonic L2 reg      — prevent asymmetry
    w_boundary_colloc : collocated boundary term      — v8.2 new; 2× global
    w_mono            : radial monotonicity penalty   — v8.2 new (L_I ext.)
    """
    w_proj            : float = 1.0
    w_boundary        : float = 0.05   # was 1.0 — trivially zero, reduce
    w_smooth          : float = 0.02
    w_positivity      : float = 0.01   # was 0.1 — trivially zero, reduce
    w_pde             : float = 0.5
    w_pol             : float = 0.001
    w_boundary_colloc : float = 0.1    # was 2.0 — was exploding to 109
    w_mono            : float = 0.01   # was 0.1 — trivially zero, reduce

# Singleton default weights
DEFAULT_WEIGHTS = LossWeights()


# =======================================================================
# 1.  Projection loss  (data fidelity)  — Lf in PINN §3.1
# =======================================================================

def loss_projection(
    eps2d_flat  : jnp.ndarray,
    g_noisy     : jnp.ndarray,
    w_ops,
    active_mask : jnp.ndarray,
) -> jnp.ndarray:
    """
    Sinogram data-fidelity loss (MSE restricted to active chords).

    Corresponds to L_f in the PINN formulation (§3.1):
        L_f = (1/N_active) Σ_i  mask_i * (W·ε - g)_i²

    Parameters
    ----------
    eps2d_flat  : (N_GRID²,) predicted 2D emissivity field (flat)
    g_noisy     : (128,)     noisy measured sinogram (normalised)
    w_ops       : WOperators (geometry.make_W_operators)
    active_mask : (128,)     bool/float — 1 for active chords

    Returns
    -------
    scalar float32
    """
    g_pred   = w_ops.matvec(eps2d_flat)
    residual = (g_pred - g_noisy) * active_mask
    n_active = jnp.maximum(active_mask.sum(), 1.0)
    return jnp.sum(residual ** 2) / n_active


# =======================================================================
# 2.  Boundary loss  — L_B in PINN §3.2
# =======================================================================

def loss_boundary(
    eps2d_flat        : jnp.ndarray,
    rho_flat          : jnp.ndarray,
    boundary_colloc_idx : Optional[jnp.ndarray] = None,  # v8.2 new arg
) -> jnp.ndarray:
    """
    Boundary-zero loss: penalise emissivity outside the LCFS (rho >= 1).

    v8.2: Added collocated boundary sub-term (L_B §3.2).
    When boundary_colloc_idx is provided, a second MSE term is evaluated
    specifically at the precomputed boundary collocation pixels (from
    geometry.PixelGrids.BOUNDARY_COLLOC_IDX), concentrating the penalty
    near rho=1 where it matters most rather than diffusely across the
    full pixel grid.  This tightens the effective boundary condition
    enforcement per the PINN L_B formulation without increasing the cost
    of the global sigmoid mask.

    Uses a smooth sigmoid ramp so gradients exist up to ~rho = 1.1.

    Parameters
    ----------
    eps2d_flat          : (N_GRID²,)  flat emissivity field
    rho_flat            : (N_GRID²,)  normalised elliptic radius at each pixel
    boundary_colloc_idx : (N_COLLOC,) int array of LCFS-adjacent pixel indices
                          (geometry.PixelGrids.BOUNDARY_COLLOC_IDX) [v8.2]

    Returns
    -------
    scalar float32  — global mask term only when boundary_colloc_idx=None;
                      global + 2× collocated sub-term otherwise (L_B §3.2)
    """
    # ── Global sigmoid-ramp mask term (unchanged from v8.1) ─────────────
    outside     = jax.nn.sigmoid(20.0 * (rho_flat - 1.0))
    l_global    = jnp.mean((eps2d_flat * outside) ** 2)

    # ── v8.2: Collocated boundary sub-term (L_B §3.2) ───────────────────
    # Evaluate emissivity specifically at the BOUNDARY_COLLOC_IDX pixels.
    # These are preselected to lie near rho=1 (e.g. |rho-1| < 0.02),
    # so this term concentrates the gradient signal exactly where the
    # physics constraint ε(rho≈1) = 0 needs to be enforced.
    if boundary_colloc_idx is not None:
        eps_colloc  = eps2d_flat[boundary_colloc_idx]              # (N_COLLOC,)
        l_colloc    = jnp.mean(eps_colloc ** 2)                    # MSE at LCFS
        # Weight 2.0 relative to the global mask term (as specified in
        # the PINN L_B formulation — collocated term gets higher weight
        # because it directly enforces the hard BC at the boundary).
        return l_global + 2.0 * l_colloc                          # L_B combined
    else:
        return l_global


# =======================================================================
# 3.  Smoothness loss  (2-D total variation)
# =======================================================================

def loss_smoothness_2d(
    eps2d_flat : jnp.ndarray,
    n_grid     : int = cfg.N_GRID,
) -> jnp.ndarray:
    """
    2-D total-variation regulariser on the emissivity field.

    Parameters
    ----------
    eps2d_flat : (N_GRID²,)  flat 2D emissivity field
    n_grid     : int         pixels per side

    Returns
    -------
    scalar float32
    """
    eps2d = eps2d_flat.reshape(n_grid, n_grid)
    tv_h  = jnp.mean(jnp.abs(jnp.diff(eps2d, axis=1)))   # horizontal
    tv_v  = jnp.mean(jnp.abs(jnp.diff(eps2d, axis=0)))   # vertical
    return 0.5 * (tv_h + tv_v)


# =======================================================================
# 4.  Positivity loss
# =======================================================================

def loss_positivity(eps2d_flat: jnp.ndarray) -> jnp.ndarray:
    """
    Soft positivity penalty — should be near zero for softplus outputs.

    Parameters
    ----------
    eps2d_flat : (N_GRID²,)  predicted emissivity field

    Returns
    -------
    scalar float32
    """
    return jnp.mean(jnp.maximum(-eps2d_flat, 0.0) ** 2)


# =======================================================================
# 5.  PDE loss  (radial flux continuity) — L_I in PINN §3.3
# =======================================================================

def loss_pde(
    a0_radial  : jnp.ndarray,
    rho_radial : jnp.ndarray,
) -> jnp.ndarray:
    """
    Physics-informed PDE residual: radial flux continuity.

    Enforces a 1-D analogue of Gauss's law for the soft-X-ray emissivity:

        d/drho [ rho * eps(rho) ] ≈ 0   (smooth radial flux)

    Corresponds to the flux-continuity component of L_I (PINN §3.3).

    v8.2: w_pde raised from 0.05 → 0.5 in LossWeights to bring this term
    to coequal status with projection loss (Lf), reflecting the physical
    importance of the radial flux constraint.

    v8.1 FIX: This function receives the *radial* a0 profile
    (N_RADIAL points on the clean radial axis), NOT the pixel-grid
    rho_flat (N_GRID² shuffled values).  The v7/v8 implementation
    inadvertently applied jnp.gradient to the pixel-grid order which
    is not sorted by rho, producing a meaningless PDE residual.

    Parameters
    ----------
    a0_radial  : (N_RADIAL,)  predicted isotropic radial emissivity (a0)
                              = coeffs[:, 0] from the model output
    rho_radial : (N_RADIAL,)  normalised radial axis (monotone ascending)

    Returns
    -------
    scalar float32
    """
    # Flux: rho * eps on the clean sorted radial grid
    flux   = rho_radial * a0_radial                               # (N_RADIAL,)

    # Centred finite-difference divergence d(flux)/d(rho)
    d_flux = jnp.gradient(flux)                                   # (N_RADIAL,)

    # PDE mask: smooth window over [0.15, 0.95] — valid physics interior
    w_inner  = jax.nn.sigmoid(30.0 * (rho_radial - 0.15))
    w_outer  = jax.nn.sigmoid(30.0 * (0.95 - rho_radial))
    pde_mask = w_inner * w_outer                                  # (N_RADIAL,)

    return jnp.mean(pde_mask * d_flux ** 2)


# =======================================================================
# 5b.  Monotonicity loss  [NEW v8.2] — radial L_I extension
# =======================================================================

def loss_monotonicity(
    a0_radial  : jnp.ndarray,
    rho_radial : jnp.ndarray,
) -> jnp.ndarray:
    """
    Radial monotonicity soft penalty  [NEW in v8.2].

    Penalises positive gradients in the symmetric emissivity component a0
    along the radial axis, enforcing the physical expectation that
    emissivity is broadly decreasing outward (core-peaked profile).

    Implements the soft constraint from PINN §3.3 (L_I extension):

        loss_mono = mean( relu( d(a0)/d(rho) ) )

    where relu(x) = max(0, x) is applied element-wise so that only
    outward-increasing segments are penalised; decreasing segments
    contribute zero loss (the constraint is one-sided).

    Implementation uses jnp.diff to compute first-order forward
    differences along the radial axis, which is fully differentiable
    w.r.t. a0_radial (unlike argmin or sign-based operations).

    Parameters
    ----------
    a0_radial  : (N_RADIAL,)  symmetric radial emissivity = coeffs[:, 0]
    rho_radial : (N_RADIAL,)  normalised radial axis (monotone ascending)
                              Only used for the interior mask.

    Returns
    -------
    scalar float32
    """
    # Forward-difference approximation of d(a0)/d(rho).
    # jnp.diff returns (N_RADIAL-1,) differences; drho is the spacing.
    drho      = jnp.diff(rho_radial)                              # (N-1,)  Δρ
    da0       = jnp.diff(a0_radial)                               # (N-1,)  Δa0
    grad_a0   = da0 / jnp.where(drho > 1e-8, drho, 1e-8)         # (N-1,)  ∂a0/∂ρ

    # Interior mask: only penalise in [0.15, 0.90] where the profile
    # is expected to be well-defined (edge bins may legitimately be flat).
    rho_mid   = 0.5 * (rho_radial[:-1] + rho_radial[1:])         # (N-1,)
    w_inner   = jax.nn.sigmoid(30.0 * (rho_mid - 0.15))
    w_outer   = jax.nn.sigmoid(30.0 * (0.90 - rho_mid))
    mono_mask = w_inner * w_outer                                  # (N-1,)

    # relu: penalise only positive radial gradients (outward-increasing).
    # Fully differentiable via sub-gradient at 0.
    return jnp.mean(mono_mask * jax.nn.relu(grad_a0))


# =======================================================================
# 6.  Poloidal regularisation
# =======================================================================

def loss_poloidal_reg(coeffs: jnp.ndarray) -> jnp.ndarray:
    """
    L2 regularisation on the poloidal harmonic amplitudes.

    Penalises large harmonic coefficients (channels 1..NC-1) to prevent
    spurious asymmetry not supported by the sinogram data.
    Channel 0 (a0) is excluded — it is already constrained by softplus.

    Parameters
    ----------
    coeffs : (N_RADIAL, 1+2*H)  harmonic coefficient array from model

    Returns
    -------
    scalar float32
    """
    harmonics = coeffs[:, 1:]        # (N_RADIAL, 2H)
    return jnp.mean(harmonics ** 2)


# =======================================================================
# 7.  Adaptive weight helpers  (v8.1 — now JAX arrays; v8.2 +2 keys)
# =======================================================================

def init_log_vars() -> Dict[str, jnp.ndarray]:
    """
    Return initial log-variance scalars for adaptive loss weighting.

    Each s_i = log(sigma_i^2) starts at 0.0 (sigma=1, no scaling).

    v8.2 CHANGE: Added "boundary_colloc" and "mono" entries to cover
    all eight loss components in the adaptive weighting scheme.

    v8.1 CHANGE: Values are jnp.zeros(()) (JAX scalar arrays), not
    Python floats.  Required for jax.value_and_grad to differentiate
    through them when merged into the param tree.

    TRAINER INTEGRATION (required for adaptive weights to work):
    ------------------------------------------------------------
    log_vars = losses.init_log_vars()
    all_params = {"model": model_params, "log_vars": log_vars}

    def _loss_for_grad(all_p):
        coeffs = model.apply(all_p["model"], ...)
        return loss_fn(..., log_vars=all_p["log_vars"])

    (total, ld), grads = jax.value_and_grad(
        _loss_for_grad, has_aux=True
    )(all_params)

    # Apply updates to BOTH model weights and log_vars:
    updates, new_opt_state = tx.update(grads, opt_state, all_params)
    new_params = optax.apply_updates(all_params, updates)

    Returns
    -------
    dict mapping loss-component name -> jnp.zeros(())
        Keys: "proj", "boundary", "boundary_colloc", "smooth",
              "positivity", "pde", "mono", "pol"
    """
    # v8.2: eight keys — six original + two new (boundary_colloc, mono)
    keys = (
        "proj",
        "boundary",
        "boundary_colloc",   # v8.2 new: collocated L_B sub-term
        "smooth",
        "positivity",
        "pde",
        "mono",              # v8.2 new: radial monotonicity L_I ext.
        "pol",
    )
    return {k: jnp.zeros(()) for k in keys}


def _adaptive_combine(
    components : Dict[str, jnp.ndarray],
    log_vars   : Dict[str, jnp.ndarray],
    weights    : LossWeights,
    beta       : float = 0.5,
    s_min      : float = -2.0,
    s_max      : float = 2.0,
    L_threshold: float = 1e-4,
    L_init     : float = 1e-2,
    norm_clip  : float = 5.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Combine loss components using blended lbPINN + GradNorm-style weighting.

    Blends two strategies:
      Option 1 — Threshold gating: terms below L_threshold are trivially
                 satisfied; their sigma is frozen at 0 (fixed weight) so
                 they cannot hijack the gradient budget.
      Option 2 — Loss-normalized weighting: active terms adapt relative
                 to their own initial scale (L_init / L_current), so
                 proj and pde balance properly even on different scales.
    """
    weight_map = {
        "proj"            : weights.w_proj,
        "boundary"        : weights.w_boundary,
        "boundary_colloc" : weights.w_boundary_colloc,
        "smooth"          : weights.w_smooth,
        "positivity"      : weights.w_positivity,
        "pde"             : weights.w_pde,
        "mono"            : weights.w_mono,
        "pol"             : weights.w_pol,
    }
    total = jnp.zeros(())
    w_eff = {}

    for name, L in components.items():
        w = weight_map.get(name, 1.0)

        s_raw = log_vars.get(name, jnp.zeros(()))
        s     = jnp.clip(jnp.asarray(s_raw, dtype=jnp.float32), s_min, s_max)
        s     = jnp.where(L > L_threshold, s, jnp.zeros_like(s))

        precision_base = jnp.exp(-s)
        norm_factor    = jnp.clip(
            L_init / jnp.maximum(L, 1e-8),
            1.0 / norm_clip,
            norm_clip,
        )
        norm_factor  = jnp.where(L > L_threshold, norm_factor, jnp.ones_like(norm_factor))
        precision    = precision_base * norm_factor

        regulariser  = beta * jax.nn.softplus(s)

        total        = total + precision * w * L + regulariser
        w_eff[f"w_eff_{name}"] = precision * w

    return total, w_eff


# =======================================================================
# 8.  Combined loss_fn  (v8.2 — eight components)
# =======================================================================

def loss_fn(
    coeffs              : jnp.ndarray,              # (N_RADIAL, NC)   model output
    g_noisy             : jnp.ndarray,              # (128,)
    w_ops,
    active_mask         : jnp.ndarray,              # (128,)
    rho_flat            : jnp.ndarray,              # (N_GRID²,)  pixel-grid rho
    theta_flat          : jnp.ndarray,              # (N_GRID²,)  pixel-grid theta
    rho_radial          : jnp.ndarray,              # (N_RADIAL,) model radial axis
    # ── v8.2: lerp interpolation weights (from geometry.PixelGrids) ──────
    lerp_idx_lo         : Optional[jnp.ndarray]            = None,  # (N_GRID²,)
    lerp_idx_hi         : Optional[jnp.ndarray]            = None,  # (N_GRID²,)
    lerp_frac           : Optional[jnp.ndarray]            = None,  # (N_GRID²,)
    # ── v8.2: collocated boundary enforcement ────────────────────────────
    boundary_colloc_idx : Optional[jnp.ndarray]            = None,  # (N_COLLOC,)
    # ─────────────────────────────────────────────────────────────────────
    weights             : LossWeights                      = DEFAULT_WEIGHTS,
    log_vars            : Optional[Dict[str, jnp.ndarray]] = None,
    n_harmonics         : int                              = cfg.N_HARMONICS,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Combined VICTOR v8.2 loss.

    Eight components total:
        proj            — sinogram data-fidelity (Lf, PINN §3.1)
        boundary        — global sigmoid mask (L_B global, PINN §3.2)
        boundary_colloc — collocated LCFS enforcement (L_B colloc §3.2)  [v8.2]
        smooth          — 2-D total variation regulariser
        positivity      — soft non-negativity penalty
        pde             — radial flux continuity (L_I, PINN §3.3)
        mono            — radial monotonicity penalty (L_I ext.)          [v8.2]
        pol             — poloidal harmonic L2 regularisation

    boundary_colloc is always computed when boundary_colloc_idx is given;
    it is zero (and logged as 0.0) otherwise so loss_dict always has eight
    entries regardless of which optional args are provided.

    When log_vars is None:  fixed weights from LossWeights are used.
    When log_vars is dict:  dynamic lbPINN weighting via _adaptive_combine.
                            log_vars must be JAX arrays (from init_log_vars)
                            and must be included in value_and_grad to be learned.

    Parameters
    ----------
    coeffs              : (N_RADIAL, 1+2*H)  harmonic coefficients from model
    g_noisy             : (128,)  noisy sinogram
    w_ops               : WOperators
    active_mask         : (128,)  float32
    rho_flat            : (N_GRID²,)  pixel-grid elliptic radius
    theta_flat          : (N_GRID²,)  pixel-grid poloidal angle
    rho_radial          : (N_RADIAL,) model radial axis — passed to loss_pde
    lerp_idx_lo         : (N_GRID²,)  lower lerp bin index  [v8.2, optional]
    lerp_idx_hi         : (N_GRID²,)  upper lerp bin index  [v8.2, optional]
    lerp_frac           : (N_GRID²,)  lerp fraction ∈ [0,1] [v8.2, optional]
    boundary_colloc_idx : (N_COLLOC,) LCFS-adjacent pixel indices [v8.2, optional]
    weights             : LossWeights (prior scale hints; effective when log_vars=None)
    log_vars            : dict of JAX scalars | None
    n_harmonics         : int

    Returns
    -------
    total_loss : scalar float32
    loss_dict  : dict with keys:
        "proj", "boundary", "boundary_colloc", "smooth", "positivity",
        "pde", "mono", "pol", "total"
        (adaptive mode also adds) "sigma_{name}", "w_eff_{name}"
    """
    # ── Build the full 2D emissivity field (v8.2: with lerp weights) ────
    eps2d_flat = build_eps2d(
        coeffs, rho_flat, theta_flat, rho_radial,
        lerp_idx_lo=lerp_idx_lo,   # v8.2: differentiable interpolation
        lerp_idx_hi=lerp_idx_hi,
        lerp_frac=lerp_frac,
        n_harmonics=n_harmonics,
    )

    # ── Individual loss components ────────────────────────────────────────

    # Lf — sinogram data-fidelity (PINN §3.1)
    l_proj     = loss_projection(eps2d_flat, g_noisy, w_ops, active_mask)

    # L_B — boundary enforcement (PINN §3.2); collocated sub-term v8.2
    l_boundary = loss_boundary(eps2d_flat, rho_flat)               # global term

    # v8.2: Collocated boundary sub-term; 0.0 when idx not provided so
    # loss_dict always has a consistent structure for logging.
    if boundary_colloc_idx is not None:
        eps_colloc      = eps2d_flat[boundary_colloc_idx]           # (N_COLLOC,)
        l_boundary_colloc = jnp.mean(eps_colloc ** 2)               # MSE at LCFS
    else:
        l_boundary_colloc = jnp.zeros(())                           # sentinel

    l_smooth   = loss_smoothness_2d(eps2d_flat)
    l_pos      = loss_positivity(eps2d_flat)

    # L_I — PDE flux continuity on clean radial axis (PINN §3.3)
    l_pde      = loss_pde(coeffs[:, 0], rho_radial)

    # v8.2: L_I extension — radial monotonicity penalty
    l_mono     = loss_monotonicity(coeffs[:, 0], rho_radial)

    l_pol      = loss_poloidal_reg(coeffs)

    components = {
        "proj"            : l_proj,
        "boundary"        : l_boundary,
        "boundary_colloc" : l_boundary_colloc,   # v8.2 new
        "smooth"          : l_smooth,
        "positivity"      : l_pos,
        "pde"             : l_pde,
        "mono"            : l_mono,              # v8.2 new
        "pol"             : l_pol,
    }

    loss_dict = dict(components)

    # ── Weighted combination ──────────────────────────────────────────────
    if log_vars is None:
        # Fixed-weight mode (no adaptive learning)
        total = (
            weights.w_proj            * l_proj
          + weights.w_boundary        * l_boundary
          + weights.w_boundary_colloc * l_boundary_colloc   # v8.2 new
          + weights.w_smooth          * l_smooth
          + weights.w_positivity      * l_pos
          + weights.w_pde             * l_pde
          + weights.w_mono            * l_mono               # v8.2 new
          + weights.w_pol             * l_pol
        )
    else:
        # Dynamic lbPINN mode — log_vars must be JAX arrays in grad tree
        total, w_eff = _adaptive_combine(components, log_vars, weights)

        # Expose learned sigmas and effective weights for monitoring
        for name in components:
            s_raw = log_vars.get(name, jnp.zeros(()))
            s     = jnp.clip(jnp.asarray(s_raw, dtype=jnp.float32), -4.0, 4.0)
            loss_dict[f"sigma_{name}"] = jnp.exp(0.5 * s)

        loss_dict.update(w_eff)   # w_eff_{name} entries

    loss_dict["total"] = total
    return total, loss_dict


# =======================================================================
# 9.  Smoke-test / quick verification
# =======================================================================

def verify_losses(
    model,
    params  : dict,
    profile : dict,
    w_ops,
    grids,
    rho_graph = None,
) -> None:
    """
    Run a single forward pass + loss_fn and print a diagnostic table.

    v8.2: Extended to exercise:
      • lerp-interpolation path in build_eps2d (uses synthetic lerp weights
        derived from the same grids that would be produced by
        geometry.build_lerp_weights).
      • boundary_colloc sub-term (uses a synthetic BOUNDARY_COLLOC_IDX
        that selects pixels with |rho - 1.0| < 0.05).
      • mono component in both fixed-weight and adaptive modes.

    Tests fixed-weight mode, adaptive-weight mode, and grad flow through
    log_vars.  All values must be finite.  Asserts on failure.
    """
    import jax

    coeffs = model.apply(
        params,
        profile["g_ideal"],
        profile["psi_n"],
        grids.RHO_FLAT,
        profile["xi"],
    )

    g_noisy     = profile["g_ideal"]
    active_mask = jnp.ones(128, dtype=jnp.float32)
    rho_flat    = grids.RHO_FLAT
    theta_flat  = grids.THETA_FLAT
    rho_radial  = grids.RHO_RADIAL

    # ── Build synthetic lerp weights (mirrors geometry.build_lerp_weights)
    # rho_flat[p] falls between rho_radial[lo] and rho_radial[hi].
    rho_r      = rho_radial
    n_rad      = len(rho_r)
    # Clamp rho_flat into [rho_r[0], rho_r[-1]] to avoid out-of-range indices
    rho_clamped  = jnp.clip(rho_flat, rho_r[0], rho_r[-1])
    # lo index = number of radial knots strictly less than rho_flat
    lo_raw       = jnp.searchsorted(rho_r, rho_clamped, side='right') - 1
    lo_idx       = jnp.clip(lo_raw, 0, n_rad - 2).astype(jnp.int32)
    hi_idx       = jnp.clip(lo_idx + 1, 0, n_rad - 1).astype(jnp.int32)
    denom        = rho_r[hi_idx] - rho_r[lo_idx]
    frac         = jnp.where(
        denom > 1e-8,
        (rho_clamped - rho_r[lo_idx]) / denom,
        0.0
    )
    # Synthetic boundary collocation: pixels near the LCFS (|rho-1|<0.05)
    boundary_colloc_idx = jnp.where(jnp.abs(rho_flat - 1.0) < 0.05)[0].astype(jnp.int32)

    print("── losses.py v8.2  verify_losses ──────────────────────────")
    print(f"  coeffs shape : {coeffs.shape}  "
          f"a0=[{float(coeffs[:,0].min()):.4f}, {float(coeffs[:,0].max()):.4f}]")
    print(f"  lerp weights : lo={lo_idx.shape}  hi={hi_idx.shape}  frac={frac.shape}")
    print(f"  boundary_colloc pixels : {len(boundary_colloc_idx)}")

    # ── Fixed-weight mode ────────────────────────────────────────────────
    total, ld = loss_fn(
        coeffs, g_noisy, w_ops, active_mask,
        rho_flat, theta_flat, rho_radial,
        lerp_idx_lo=lo_idx,
        lerp_idx_hi=hi_idx,
        lerp_frac=frac,
        boundary_colloc_idx=boundary_colloc_idx,
    )
    print("\n  Fixed-weight mode (8 components + lerp + boundary_colloc):")
    for k, v in ld.items():
        flag = "  OK" if jnp.isfinite(v) else "  *** NON-FINITE ***"
        print(f"    {k:<30}: {float(v):.6f}{flag}")

    # ── Adaptive-weight mode ─────────────────────────────────────────────
    lv = init_log_vars()
    total_a, ld_a = loss_fn(
        coeffs, g_noisy, w_ops, active_mask,
        rho_flat, theta_flat, rho_radial,
        lerp_idx_lo=lo_idx,
        lerp_idx_hi=hi_idx,
        lerp_frac=frac,
        boundary_colloc_idx=boundary_colloc_idx,
        log_vars=lv,
    )
    print("\n  Adaptive-weight mode (log_vars=0, all 8 components):")
    for k, v in ld_a.items():
        flag = "  OK" if jnp.isfinite(v) else "  *** NON-FINITE ***"
        print(f"    {k:<30}: {float(v):.6f}{flag}")

    # ── Gradient-flow check through log_vars ─────────────────────────────
    def _loss_wrt_logvars(lv_):
        t, _ = loss_fn(
            coeffs, g_noisy, w_ops, active_mask,
            rho_flat, theta_flat, rho_radial,
            lerp_idx_lo=lo_idx,
            lerp_idx_hi=hi_idx,
            lerp_frac=frac,
            boundary_colloc_idx=boundary_colloc_idx,
            log_vars=lv_,
        )
        return t

    lv_grads = jax.grad(_loss_wrt_logvars)(lv)
    print("\n  Grad of total w.r.t. log_vars (all 8 terms incl. v8.2 new):")
    for k, g in lv_grads.items():
        flag = "  OK" if jnp.isfinite(g) else "  *** NON-FINITE ***"
        print(f"    d_total/d_s_{k:<24}: {float(g):+.6f}{flag}")

    # ── New v8.2 terms isolated ───────────────────────────────────────────
    print("\n  v8.2 specific checks:")
    bc_val = float(ld.get("boundary_colloc", jnp.nan))
    mo_val = float(ld.get("mono", jnp.nan))
    print(f"    boundary_colloc (L_B colloc)      : {bc_val:.6f}  "
          f"{'OK' if jnp.isfinite(bc_val) else '*** NON-FINITE ***'}")
    print(f"    mono (L_I monotonicity ext.)      : {mo_val:.6f}  "
          f"{'OK' if jnp.isfinite(mo_val) else '*** NON-FINITE ***'}")

    print("────────────────────────────────────────────────────────────")

    bad  = [k for k, v in ld.items()     if not jnp.isfinite(v)]
    bad += [k for k, v in ld_a.items()   if not jnp.isfinite(v)]
    bad += [k for k, v in lv_grads.items() if not jnp.isfinite(v)]
    assert not bad, f"Non-finite values detected: {bad}"
    print("OK  losses.py v8.2 verified")


# ── Module self-test ─────────────────────────────────────────────────

if __name__ == "__main__":
    import jax

    key        = jax.random.PRNGKey(42)
    N          = cfg.N_GRID
    NC         = 1 + 2 * cfg.N_HARMONICS
    NR         = cfg.N_RADIAL

    coeffs     = jax.random.uniform(key, (NR, NC))
    coeffs     = coeffs.at[:, 0].set(jax.nn.softplus(coeffs[:, 0]))
    g          = jax.random.uniform(key, (128,))
    rho_flat   = jnp.linspace(0.0, 1.5, N * N)
    theta_flat = jnp.linspace(-jnp.pi, jnp.pi, N * N)
    rho_radial = jnp.linspace(0.0, cfg.RHO_MAX, NR)
    amask      = jnp.ones(128)

    # Build synthetic lerp weights (same logic as geometry.build_lerp_weights)
    rho_clamped  = jnp.clip(rho_flat, rho_radial[0], rho_radial[-1])
    lo_raw       = jnp.searchsorted(rho_radial, rho_clamped, side='right') - 1
    lo_idx       = jnp.clip(lo_raw, 0, NR - 2).astype(jnp.int32)
    hi_idx       = jnp.clip(lo_idx + 1, 0, NR - 1).astype(jnp.int32)
    denom        = rho_radial[hi_idx] - rho_radial[lo_idx]
    lerp_frac    = jnp.where(
        denom > 1e-8,
        (rho_clamped - rho_radial[lo_idx]) / denom,
        0.0
    )
    # Synthetic boundary colloc index: pixels near rho=1
    boundary_colloc_idx = jnp.where(jnp.abs(rho_flat - 1.0) < 0.05)[0].astype(jnp.int32)

    class _StubOps:
        @staticmethod
        def matvec(ef):
            return jnp.zeros(128)

    stub_ops = _StubOps()

    print("─── Fixed weights (v8.2, 8 components) ─────────────────────")
    total, ld = loss_fn(
        coeffs, g, stub_ops, amask,
        rho_flat, theta_flat, rho_radial,
        lerp_idx_lo=lo_idx,
        lerp_idx_hi=hi_idx,
        lerp_frac=lerp_frac,
        boundary_colloc_idx=boundary_colloc_idx,
    )
    for k, v in ld.items():
        print(f"  {k:<30}: {float(v):.6f}")

    print("\n─── Adaptive weights (v8.2) ─────────────────────────────────")
    lv = init_log_vars()
    total_a, ld_a = loss_fn(
        coeffs, g, stub_ops, amask,
        rho_flat, theta_flat, rho_radial,
        lerp_idx_lo=lo_idx,
        lerp_idx_hi=hi_idx,
        lerp_frac=lerp_frac,
        boundary_colloc_idx=boundary_colloc_idx,
        log_vars=lv,
    )
    for k, v in ld_a.items():
        print(f"  {k:<30}: {float(v):.6f}")

    print("\n─── Grad through log_vars (all 8 terms) ─────────────────────")
    def _t(lv_):
        tot, _ = loss_fn(
            coeffs, g, stub_ops, amask,
            rho_flat, theta_flat, rho_radial,
            lerp_idx_lo=lo_idx,
            lerp_idx_hi=hi_idx,
            lerp_frac=lerp_frac,
            boundary_colloc_idx=boundary_colloc_idx,
            log_vars=lv_,
        )
        return tot

    lv_g = jax.grad(_t)(lv)
    for k, v in lv_g.items():
        print(f"  d_total/d_s_{k:<24}: {float(v):+.6f}")

    ok = jnp.isfinite(total) and jnp.isfinite(total_a)
    print("\nSelf-test passed." if ok else "FAILED — non-finite total!")
