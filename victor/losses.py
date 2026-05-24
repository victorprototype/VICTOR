# ============================================================
# VICTOR v7.0 — losses.py
# All individual loss functions + combined loss_fn
# ============================================================
# Public API
# ----------
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
# 3.  Smoothness loss  (1-D total variation)
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
# 5.  PDE loss  (radial flux continuity)  — NEW in v7.1
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
    return {k: 0.0 for k in ("proj", "boundary", "smooth", "positivity", "pde")}


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
# 7.  Combined loss_fn
# =======================================================================

def loss_fn(
    eps1d       : jnp.ndarray,
    g_noisy     : jnp.ndarray,
    w_ops,
    active_mask : jnp.ndarray,
    rho_flat    : jnp.ndarray,
    weights     : LossWeights                  = DEFAULT_WEIGHTS,
    log_vars    : Optional[Dict[str, float]]   = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Combined physics-informed loss for VICTOR v7.

    When log_vars is None (default), uses fixed LossWeights — identical
    to v7.0 behaviour (fully backward-compatible).

    When log_vars is provided (dict of learnable log-variance scalars),
    switches to Kendall et al. 2018 adaptive weighting.  Each component
    is scaled by exp(-s_i) and regularised by 0.5*s_i, allowing the
    network to automatically balance data fidelity vs physics priors.

    Parameters
    ----------
    eps1d       : (n_radial,)  radial emissivity profile from FourierDeepONet
    g_noisy     : (128,)       noisy sinogram for current training sample
    w_ops       : WOperators   (geometry.make_W_operators)
    active_mask : (128,)       float32, 1 for active chords
    rho_flat    : (n_radial,)  normalised elliptic radius at each radial point
    weights     : LossWeights  fixed prior weights (default: DEFAULT_WEIGHTS)
    log_vars    : dict | None  learnable log-variance scalars per component.
                               Keys: "proj", "boundary", "smooth",
                                     "positivity", "pde".
                               Initialise with init_log_vars().

    Returns
    -------
    total_loss : scalar float32
    loss_dict  : dict[str -> scalar float32]
        Keys: "proj", "boundary", "smooth", "positivity", "pde", "total"
        When adaptive: also includes "sigma_proj", "sigma_boundary", etc.
                       (effective noise std = exp(0.5 * s_i)) for monitoring.
    """
    # Individual components
    l_proj       = loss_projection(eps1d, g_noisy, w_ops, active_mask)
    l_boundary   = loss_boundary(eps1d, rho_flat)
    l_smooth     = loss_smoothness(eps1d)
    l_positivity = loss_positivity(eps1d)
    l_pde        = loss_pde(eps1d, rho_flat)

    components = {
        "proj"      : l_proj,
        "boundary"  : l_boundary,
        "smooth"    : l_smooth,
        "positivity": l_positivity,
        "pde"       : l_pde,
    }

    # Weighted sum — fixed or adaptive
    if log_vars is None:
        # Fixed weights (v7.0-compatible)
        total = (
            weights.w_proj       * l_proj
          + weights.w_boundary   * l_boundary
          + weights.w_smooth     * l_smooth
          + weights.w_positivity * l_positivity
          + weights.w_pde        * l_pde
        )
        loss_dict = dict(components)
    else:
        # Adaptive weights (Kendall et al. 2018)
        total = _adaptive_combine(components, log_vars, weights)
        loss_dict = dict(components)
        # Expose effective sigma per component for monitoring
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
    rho_graph,
) -> None:
    """
    Run a single forward pass + loss_fn and print a diagnostic table.

    Tests both fixed-weight and adaptive-weight modes.
    All values must be finite.  Asserts on failure.

    Parameters
    ----------
    model     : FourierDeepONet instance
    params    : initialised param tree
    profile   : one entry from data_loader.load_profiles()
    w_ops     : WOperators
    grids     : PixelGrids
    rho_graph : RhoGraph  (unused in v7; kept for API compatibility)
    """
    eps1d = model.apply(
        params,
        profile["g_ideal"],
        profile["xi"],
    )

    g_noisy     = profile["g_ideal"]
    active_mask = jnp.ones(128, dtype=jnp.float32)
    rho_flat    = grids.RHO_RADIAL

    print("── losses.py  verify_losses ────────────────────────────")
    print(f"  eps1d shape  : {eps1d.shape}  "
          f"range=[{float(eps1d.min()):.4f}, {float(eps1d.max()):.4f}]")

    # Fixed-weight mode
    total, ld = loss_fn(eps1d, g_noisy, w_ops, active_mask, rho_flat)
    print("\n  Fixed-weight mode:")
    for k, v in ld.items():
        flag = "  OK" if jnp.isfinite(v) else "  NON-FINITE"
        print(f"    {k:<18}: {float(v):.6f}{flag}")

    # Adaptive-weight mode
    lv = init_log_vars()
    total_a, ld_a = loss_fn(eps1d, g_noisy, w_ops, active_mask, rho_flat,
                            log_vars=lv)
    print("\n  Adaptive-weight mode (log_vars=0):")
    for k, v in ld_a.items():
        flag = "  OK" if jnp.isfinite(v) else "  NON-FINITE"
        print(f"    {k:<18}: {float(v):.6f}{flag}")

    print("────────────────────────────────────────────────────────")

    bad = [k for k, v in ld.items() if not jnp.isfinite(v)]
    bad += [k for k, v in ld_a.items() if not jnp.isfinite(v)]
    assert not bad, f"Non-finite loss components: {bad}"
    print("OK  losses.py verified")


# ── Module self-test ─────────────────────────────────────────────────

if __name__ == "__main__":
    import jax
    key      = jax.random.PRNGKey(42)
    n_radial = cfg.N_RADIAL if hasattr(cfg, "N_RADIAL") else 128

    eps1d    = jax.random.uniform(key, (n_radial,))
    g        = jax.random.uniform(key, (128,))
    rho_flat = jnp.linspace(0.0, 1.2, n_radial)
    amask    = jnp.ones(128)

    class _StubOps:
        @staticmethod
        def matvec(ef):
            return jnp.zeros(128)

    stub_ops = _StubOps()

    print("--- Fixed weights ---")
    total, ld = loss_fn(eps1d, g, stub_ops, amask, rho_flat)
    for k, v in ld.items():
        print(f"  {k:<18}: {float(v):.6f}")

    print("\n--- Adaptive weights ---")
    lv = init_log_vars()
    total_a, ld_a = loss_fn(eps1d, g, stub_ops, amask, rho_flat, log_vars=lv)
    for k, v in ld_a.items():
        print(f"  {k:<18}: {float(v):.6f}")

    ok = jnp.isfinite(total) and jnp.isfinite(total_a)
    print("\nSelf-test passed." if ok else "FAILED — non-finite total!")
