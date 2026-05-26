# ============================================================
# VICTOR v7.0 — losses.py
# All individual loss functions + combined loss_fn
# ============================================================
# Public API
# ----------
#   build_eps2d(coeffs, rho_flat, theta_flat, rho_radial)
#       -> (N_GRID²,)  flat 2D emissivity from harmonic coefficients
#
#   loss_projection(eps1d, g_noisy, w_ops, active_mask)
#       -> scalar    Sinogram data-fidelity (MSE on active chords)
#
#   loss_boundary(eps1d, rho_flat)
#       -> scalar    Hard boundary: emissivity must vanish for rho >= 1
#
#   loss_smoothness(eps1d)
#       -> scalar    Total-variation regulariser (finite differences)
#
#   loss_positivity(eps1d)
#       -> scalar    Soft penalty for negative values
#
#   loss_pde(eps1d, rho_flat)
#       -> scalar    PDE-informed radial flux continuity residual
#
#   init_log_vars()
#       -> dict      Initial log-variance params for adaptive weighting
#
#   loss_fn(eps1d, g_noisy, w_ops, active_mask,
#           rho_flat, weights, log_vars)
#       -> (total_loss, loss_dict)
#
# v7.1 additions vs v7.0
# ----------------------
#  * loss_pde: physics-informed PDE residual enforcing radial flux
#    continuity d/drho [rho * eps] ≈ 0 in the vacuum region and
#    smooth monotone decay inside the plasma.  Weighted by a PDE
#    importance mask focused on the interior (0.2 <= rho <= 0.9).
#
#  * Adaptive loss weights (Kendall et al. 2018 homoscedastic
#    uncertainty weighting):
#      weighted_i = exp(-s_i) * L_i  +  0.5 * s_i
#    where s_i = log(sigma_i^2) is a learnable log-variance scalar.
#    The network automatically balances components without manual
#    tuning.  log_vars is an optional dict {key: scalar}; when None,
#    fixed LossWeights are used — fully backward-compatible.
#
#  * init_log_vars() returns a {key: 0.0} dict matching LossWeights.
#    Merge with model params and include in value_and_grad.
#
# v7 design principles (unchanged)
# ---------------------------------
#  * All functions are pure JAX — no side effects, no globals mutated.
#  * loss_fn is the single entry point consumed by the training step.
#  * loss_dict exposes every individual component for logs.
#  * All losses return scalar float32.
#  * W operators are WOperators namedtuples (geometry.py).
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
    coeffs      : jnp.ndarray,   # (N_RADIAL, 1 + 2*H)
    rho_flat    : jnp.ndarray,   # (N_GRID²,)  pixel ρ values
    theta_flat  : jnp.ndarray,   # (N_GRID²,)  pixel θ values
    rho_radial  : jnp.ndarray,   # (N_RADIAL,) model's radial axis
    n_harmonics : int = cfg.N_HARMONICS,
    clip_min    : float = None,
) -> jnp.ndarray:
    """
    Evaluate the polar-harmonic emissivity field on the pixel grid.

    For each pixel, finds the nearest radial bin by rho, then evaluates:
        ε(ρ,θ) = a0 + Σ_{n=1}^{H} [an·cos(nθ) + bn·sin(nθ)]

    No softplus applied here. a0 is already >= 0 from the model's softplus.
    clip_min is optional — only use 0.0 if negative pixels cause instability.

    Parameters
    ----------
    coeffs      : (N_RADIAL, 1+2*H)  harmonic coefficients from model
    rho_flat    : (N_GRID²,)  pixel-grid ρ  (grids.RHO_FLAT)
    theta_flat  : (N_GRID²,)  pixel-grid θ  (grids.THETA_FLAT)
    rho_radial  : (N_RADIAL,) model radial axis (grids.RHO_RADIAL)
    n_harmonics : int
    clip_min    : float | None

    Returns
    -------
    eps2d_flat : (N_GRID²,)  float32
    """
    # Nearest-neighbour lookup: each pixel -> closest radial bin
    diffs  = jnp.abs(rho_flat[:, None] - rho_radial[None, :])  # (N², N_RADIAL)
    nn_idx = jnp.argmin(diffs, axis=1)                          # (N²,)

    # Gather coefficients at each pixel's nearest radial bin
    coeffs_pix = coeffs[nn_idx]                                 # (N², NC)

    # Start with symmetric component a0
    eps = coeffs_pix[:, 0]                                      # (N²,)

    # Add harmonics
    for h in range(1, n_harmonics + 1):
        a_col = 2 * h - 1
        b_col = 2 * h
        eps = (eps
               + coeffs_pix[:, a_col] * jnp.cos(h * theta_flat)
               + coeffs_pix[:, b_col] * jnp.sin(h * theta_flat))

    # Zero pixels outside the model's radial domain
    eps = jnp.where(rho_flat <= cfg.RHO_MAX, eps, 0.0)

    if clip_min is not None:
        eps = jnp.clip(eps, a_min=clip_min)

    return eps                                                   # (N²,)

# =======================================================================
# 0.  Loss weights
# =======================================================================

class LossWeights(NamedTuple):
    """
    Scalar multipliers for each loss component.

    Used when log_vars=None (fixed-weight mode).  When adaptive
    weighting is active (log_vars supplied to loss_fn), these serve
    as initial scale hints but are overridden by the learnable sigmas.

    Attributes
    ----------
    w_proj       : weight for projection / data-fidelity loss
    w_boundary   : weight for boundary-zero enforcement
    w_smooth     : weight for total-variation smoothness regulariser
    w_positivity : weight for soft positivity penalty
    w_pde        : weight for PDE flux-continuity residual
    """
    w_proj       : float = 1.0
    w_boundary   : float = 5.0
    w_smooth     : float = 0.02
    w_positivity : float = 0.5
    w_pde        : float = 0.1
    w_pol        : float = 0.001  # poloidal harmonic L2 regularisation (v8)


# Singleton default weights
DEFAULT_WEIGHTS = LossWeights()


# =======================================================================
# 1.  Projection loss  (data fidelity)
# =======================================================================

def loss_projection(
    eps1d       : jnp.ndarray,
    g_noisy     : jnp.ndarray,
    w_ops,
    active_mask : jnp.ndarray,
) -> jnp.ndarray:
    """
    Sinogram data-fidelity loss (MSE restricted to active chords).

    Parameters
    ----------
    eps1d       : (n_radial,)  predicted radial emissivity profile
    g_noisy     : (128,)       noisy measured sinogram (normalised)
    w_ops       : WOperators   (geometry.make_W_operators)
    active_mask : (128,)       bool/float — 1 for active chords

    Returns
    -------
    scalar float32
    """
    g_pred   = w_ops.matvec(eps1d)
    residual = (g_pred - g_noisy) * active_mask
    n_active = jnp.maximum(active_mask.sum(), 1.0)
    return jnp.sum(residual ** 2) / n_active


# =======================================================================
# 2.  Boundary loss
# =======================================================================

def loss_boundary(
    eps1d    : jnp.ndarray,
    rho_flat : jnp.ndarray,
) -> jnp.ndarray:
    """
    Boundary-zero loss: penalise emissivity outside the LCFS (rho >= 1).

    Uses a smooth sigmoid ramp so gradients exist up to ~rho = 1.1.

    Parameters
    ----------
    eps1d    : (n_radial,)  predicted radial emissivity profile
    rho_flat : (n_radial,)  normalised elliptic radius at each radial point

    Returns
    -------
    scalar float32
    """
    outside = jax.nn.sigmoid(20.0 * (rho_flat - 1.0))
    return jnp.mean((eps1d * outside) ** 2)


# =======================================================================
# 3a.  Smoothness loss  (1-D total variation)
# =======================================================================

def loss_smoothness(eps1d: jnp.ndarray) -> jnp.ndarray:
    """
    1-D total-variation (TV) regulariser along the radial profile.

    Parameters
    ----------
    eps1d : (n_radial,)  predicted radial emissivity profile

    Returns
    -------
    scalar float32
    """
    return jnp.mean(jnp.abs(jnp.diff(eps1d)))

# =======================================================================
# 3b.  Smoothness loss  (2-D total variation)  — v8
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

def loss_positivity(eps1d: jnp.ndarray) -> jnp.ndarray:
    """
    Soft positivity penalty.  Should stay near zero for sigmoid outputs.

    Parameters
    ----------
    eps1d : (n_radial,)  predicted radial emissivity profile

    Returns
    -------
    scalar float32
    """
    return jnp.mean(jnp.maximum(-eps1d, 0.0) ** 2)


# =======================================================================
# 5a.  PDE loss  (radial flux continuity)  — NEW in v7.1
# =======================================================================

def loss_pde(
    eps1d    : jnp.ndarray,
    rho_flat : jnp.ndarray,
) -> jnp.ndarray:
    """
    Physics-informed PDE residual: radial flux continuity.

    Enforces a 1-D analogue of Gauss's law for the soft-X-ray emissivity:

        d/drho [ rho * eps(rho) ] = 0   (no sources outside plasma core)

    Physically, this penalises spurious local emission peaks at mid-radius
    that are not supported by the sinogram data — a common artefact of
    under-constrained neural reconstructions.

    The residual is computed as the finite-difference approximation of the
    divergence and weighted by a PDE importance mask that focuses on the
    plasma interior (0.2 <= rho <= 0.9), where the constraint is valid.
    The mask tapers smoothly to zero near the axis (to avoid the 1/rho
    singularity) and near the boundary (where boundary loss already acts).

    PDE weight mask
    ---------------
    w(rho) = sigmoid(30*(rho - 0.15)) * sigmoid(30*(0.95 - rho))

    This is 1 inside [0.15, 0.95] and falls to 0 outside, giving a
    differentiable soft window.

    Parameters
    ----------
    eps1d    : (n_radial,)  predicted radial emissivity profile
    rho_flat : (n_radial,)  normalised radial axis (matches eps1d)

    Returns
    -------
    scalar float32
    """
    # Flux: rho * eps  (n_radial,)
    flux = rho_flat * eps1d

    # Finite-difference divergence d(flux)/d(rho), interior points only
    # Use centred differences on interior, forward/backward at edges
    d_flux = jnp.gradient(flux)                              # (n_radial,)

    # PDE importance mask: smooth window over [0.15, 0.95]
    w_inner = jax.nn.sigmoid(30.0 * (rho_flat - 0.15))
    w_outer = jax.nn.sigmoid(30.0 * (0.95 - rho_flat))
    pde_mask = w_inner * w_outer                             # (n_radial,)

    # Weighted mean-square divergence residual
    return jnp.mean(pde_mask * d_flux ** 2)

# =======================================================================
# 5b.  Poloidal regularisation  — NEW in v8
# =======================================================================

def loss_poloidal_reg(coeffs: jnp.ndarray) -> jnp.ndarray:
    """
    L2 regularisation on the poloidal harmonic amplitudes.

    Penalises large harmonic coefficients (channels 1..NC-1) to prevent
    spurious asymmetry not supported by the sinogram data.
    Channel 0 (a0) is excluded — it is already constrained by softplus.

    Weight 0.001 is conservative and will not suppress real asymmetry
    that is consistent with the projection data.

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
# 6.  Adaptive weight helpers  — NEW in v7.1
# =======================================================================

def init_log_vars() -> Dict[str, float]:
    """
    Return initial log-variance scalars for adaptive loss weighting.

    Each entry s_i = log(sigma_i^2) starts at 0.0 (sigma=1, no scaling).
    Add these to your Flax param tree and include them in value_and_grad
    so they are learned alongside the model weights.

    Returns
    -------
    dict mapping loss-component name -> 0.0 (Python float)
        Keys: "proj", "boundary", "smooth", "positivity", "pde"

    Usage in trainer (sketch)
    -------------------------
    log_vars = init_log_vars()
    all_params = {"model": model_params, "log_vars": log_vars}

    def _loss(all_p):
        eps1d = model.apply(all_p["model"], g, xi)
        return loss_fn(..., log_vars=all_p["log_vars"])

    (total, ld), grads = jax.value_and_grad(_loss, has_aux=True)(all_params)
    """
    return {k: 0.0 for k in ("proj", "boundary", "smooth", "pol")}


def _adaptive_combine(
    components : Dict[str, jnp.ndarray],
    log_vars   : Dict[str, jnp.ndarray],
    weights    : LossWeights,
) -> jnp.ndarray:
    """
    Combine loss components using Kendall et al. 2018 homoscedastic
    uncertainty weighting.

    For each component i:
        L_i_weighted = exp(-s_i) * w_i * L_i  +  0.5 * s_i

    where s_i is the learned log-variance and w_i is the fixed prior
    weight from LossWeights.  The exp(-s_i) term down-weights noisy
    losses; the 0.5*s_i regulariser prevents s_i from diverging to +inf.

    Parameters
    ----------
    components : dict  {name: scalar loss value}
    log_vars   : dict  {name: scalar log-variance (learnable)}
    weights    : LossWeights  fixed prior scale per component

    Returns
    -------
    scalar float32  weighted total loss
    """
    weight_map = {
        "proj"      : weights.w_proj,
        "boundary"  : weights.w_boundary,
        "smooth"    : weights.w_smooth,
        "positivity": weights.w_positivity,
        "pde"       : weights.w_pde,
        "pol"       : weights.w_pol,
    }
    total = jnp.zeros(())
    for name, L in components.items():
        w  = weight_map.get(name, 1.0)
        s  = log_vars.get(name, 0.0)
        # Clamp s to [-5, 5] to prevent numerical blow-up during early training
        s  = jnp.clip(jnp.asarray(s, dtype=jnp.float32), -5.0, 5.0)
        total = total + jnp.exp(-s) * w * L + 0.5 * s
    return total


# =======================================================================
# 7.  Combined loss_fn  (v8)
# =======================================================================

def loss_fn(
    coeffs      : jnp.ndarray,              # (N_RADIAL, NC)   model output
    g_noisy     : jnp.ndarray,              # (128,)
    w_ops,
    active_mask : jnp.ndarray,              # (128,)
    rho_flat    : jnp.ndarray,              # (N_GRID²,)  pixel-grid rho
    theta_flat  : jnp.ndarray,              # (N_GRID²,)  pixel-grid theta
    rho_radial  : jnp.ndarray,              # (N_RADIAL,) model radial axis
    weights     : LossWeights                          = DEFAULT_WEIGHTS,
    log_vars    : Optional[Dict[str, float]]           = None,
    n_harmonics : int                                  = cfg.N_HARMONICS,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Combined VICTOR v8 loss.

    Reconstructs the 2D emissivity field via build_eps2d(), then computes:
        projection + boundary + smoothness (2D TV) + poloidal_reg

    Parameters
    ----------
    coeffs      : (N_RADIAL, 1+2*H)  harmonic coefficients from model
    g_noisy     : (128,)  noisy sinogram
    w_ops       : WOperators
    active_mask : (128,)  float32
    rho_flat    : (N_GRID²,)  pixel-grid elliptic radius
    theta_flat  : (N_GRID²,)  pixel-grid poloidal angle
    rho_radial  : (N_RADIAL,) model radial axis
    weights     : LossWeights
    log_vars    : dict | None  keys: "proj", "boundary", "smooth", "pol"
    n_harmonics : int

    Returns
    -------
    total_loss : scalar float32
    loss_dict  : dict with keys "proj", "boundary", "smooth", "pol", "total"
    """
    # Build the true 2D emissivity field
    eps2d_flat = build_eps2d(
        coeffs, rho_flat, theta_flat, rho_radial,
        n_harmonics=n_harmonics,
    )

    # Individual components
    l_proj     = loss_projection(eps2d_flat, g_noisy, w_ops, active_mask)
    l_boundary = loss_boundary(eps2d_flat, rho_flat)
    l_smooth   = loss_smoothness_2d(eps2d_flat)
    l_pol      = loss_poloidal_reg(coeffs)

    components = {
        "proj"    : l_proj,
        "boundary": l_boundary,
        "smooth"  : l_smooth,
        "pol"     : l_pol,
    }

    # Weighted combination — fixed or adaptive
    if log_vars is None:
        total = (
            weights.w_proj     * l_proj
          + weights.w_boundary * l_boundary
          + weights.w_smooth   * l_smooth
          + weights.w_pol      * l_pol
        )
        loss_dict = dict(components)
    else:
        total     = _adaptive_combine(components, log_vars, weights)
        loss_dict = dict(components)
        for name in components:
            s = log_vars.get(name, 0.0)
            s = jnp.clip(jnp.asarray(s, dtype=jnp.float32), -5.0, 5.0)
            loss_dict[f"sigma_{name}"] = jnp.exp(0.5 * s)

    loss_dict["total"] = total
    return total, loss_dict

# =======================================================================
# 8.  Smoke-test / quick verification
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
    Tests both fixed-weight and adaptive-weight modes.
    All values must be finite. Asserts on failure.
    """
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

    print("── losses.py v8  verify_losses ─────────────────────────")
    print(f"  coeffs shape : {coeffs.shape}  "
          f"a0=[{float(coeffs[:,0].min()):.4f}, {float(coeffs[:,0].max()):.4f}]")

    # Fixed-weight mode
    total, ld = loss_fn(
        coeffs, g_noisy, w_ops, active_mask,
        rho_flat, theta_flat, rho_radial,
    )
    print("\n  Fixed-weight mode:")
    for k, v in ld.items():
        flag = "  OK" if jnp.isfinite(v) else "  NON-FINITE"
        print(f"    {k:<20}: {float(v):.6f}{flag}")

    # Adaptive-weight mode
    lv = init_log_vars()
    total_a, ld_a = loss_fn(
        coeffs, g_noisy, w_ops, active_mask,
        rho_flat, theta_flat, rho_radial,
        log_vars=lv,
    )
    print("\n  Adaptive-weight mode (log_vars=0):")
    for k, v in ld_a.items():
        flag = "  OK" if jnp.isfinite(v) else "  NON-FINITE"
        print(f"    {k:<20}: {float(v):.6f}{flag}")

    print("────────────────────────────────────────────────────────")
    bad = [k for k, v in ld.items() if not jnp.isfinite(v)]
    bad += [k for k, v in ld_a.items() if not jnp.isfinite(v)]
    assert not bad, f"Non-finite loss components: {bad}"
    print("OK  losses.py v8 verified")


# ── Module self-test ─────────────────────────────────────────────────

if __name__ == "__main__":
    import jax
    key        = jax.random.PRNGKey(42)
    N          = cfg.N_GRID
    NC         = 1 + 2 * cfg.N_HARMONICS

    coeffs     = jax.random.uniform(key, (cfg.N_RADIAL, NC))
    coeffs     = coeffs.at[:, 0].set(jax.nn.softplus(coeffs[:, 0]))
    g          = jax.random.uniform(key, (128,))
    rho_flat   = jnp.linspace(0.0, 1.5, N * N)
    theta_flat = jnp.linspace(-jnp.pi, jnp.pi, N * N)
    rho_radial = jnp.linspace(0.0, cfg.RHO_MAX, cfg.N_RADIAL)
    amask      = jnp.ones(128)

    class _StubOps:
        @staticmethod
        def matvec(ef):
            return jnp.zeros(128)

    stub_ops = _StubOps()

    print("--- Fixed weights ---")
    total, ld = loss_fn(coeffs, g, stub_ops, amask,
                        rho_flat, theta_flat, rho_radial)
    for k, v in ld.items():
        print(f"  {k:<20}: {float(v):.6f}")

    print("\n--- Adaptive weights ---")
    lv = init_log_vars()
    total_a, ld_a = loss_fn(coeffs, g, stub_ops, amask,
                             rho_flat, theta_flat, rho_radial,
                             log_vars=lv)
    for k, v in ld_a.items():
        print(f"  {k:<20}: {float(v):.6f}")

    ok = jnp.isfinite(total) and jnp.isfinite(total_a)
    print("\nSelf-test passed." if ok else "FAILED — non-finite total!")