# ============================================================
# VICTOR v7.0 — losses.py
# All individual loss functions + combined loss_fn
# ============================================================
# Public API
# ----------
#   loss_projection(eps1d, g_noisy, w_ops, active_mask)
#       → scalar    Sinogram data-fidelity (MSE on active chords)
#
#   loss_boundary(eps1d, rho_flat)
#       → scalar    Hard boundary: emissivity must vanish for ρ ≥ 1
#
#   loss_smoothness(eps1d)
#       → scalar    Total-variation regulariser (finite differences)
#
#   loss_isotropy(eps1d, rho_flat)
#       → scalar    Physics prior: ε should vary mainly with ρ
#
#   loss_positivity(eps1d)
#       → scalar    Soft penalty for negative values
#
#   loss_fn(eps1d, g_noisy, w_ops, active_mask,
#           rho_flat, weights)
#       → (total_loss, loss_dict)
#
# v7 changes vs v6
# ----------------
#  • Model now outputs eps1d : (n_radial,) — a 1-D radial profile.
#    All 2-D (N, N) array assumptions are removed.
#  • Ensemble outputs (mean, std, preds) and log_noise are gone;
#    the ensemble NLL and diversity losses are removed accordingly.
#  • loss_boundary uses rho_flat (1-D) instead of rho_2d (2-D).
#  • loss_smoothness operates on the 1-D radial vector.
#  • loss_isotropy is removed (it was a 2-D binning prior; the 1-D
#    radial output is already ρ-indexed by construction).
#  • loss_fn signature is simplified to match.
#  • verify_losses forward call updated to model.apply(params, g, xi).
#
# Design principles (unchanged)
# -----------------
#  • All functions are pure JAX — no side effects, no globals mutated.
#  • loss_fn is the single entry point consumed by the training step.
#  • loss_dict exposes every individual component for logs.
#  • Weights are tunable via the LossWeights dataclass.
#  • All losses return scalar float32.
#  • W operators are passed as WOperators namedtuples (geometry.py).
# ============================================================

from __future__ import annotations

from typing import NamedTuple, Dict, Tuple

import jax
import jax.numpy as jnp

from victor import config as cfg


# ═══════════════════════════════════════════════════════════════════════
# 0.  Loss weights
# ═══════════════════════════════════════════════════════════════════════

class LossWeights(NamedTuple):
    """
    Scalar multipliers for each loss component.

    Tune these to balance physics priors against data fidelity.
    Defaults are calibrated for WEST soft-X-ray inversion at VICTOR v7.

    Attributes
    ----------
    w_proj       : weight for projection / data-fidelity loss
    w_boundary   : weight for boundary-zero enforcement
    w_smooth     : weight for total-variation smoothness regulariser
    w_positivity : weight for soft positivity penalty
    """
    w_proj       : float = 1.0
    w_boundary   : float = 5.0
    w_smooth     : float = 0.02
    w_positivity : float = 0.5


# Singleton default weights — importable as `losses.DEFAULT_WEIGHTS`
DEFAULT_WEIGHTS = LossWeights()


# ═══════════════════════════════════════════════════════════════════════
# 1.  Projection loss  (data fidelity)
# ═══════════════════════════════════════════════════════════════════════

def loss_projection(
    eps1d       : jnp.ndarray,
    g_noisy     : jnp.ndarray,
    w_ops,
    active_mask : jnp.ndarray,
) -> jnp.ndarray:
    """
    Sinogram data-fidelity loss (MSE restricted to active chords).

    Computes the forward projection g_pred = W · eps1d and
    penalises the mean-square residual on all active (non-zero-sum)
    rows of the W matrix.

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
    g_pred   = w_ops.matvec(eps1d)                     # (128,)
    residual = (g_pred - g_noisy) * active_mask        # zero-out inactive
    n_active = jnp.maximum(active_mask.sum(), 1.0)
    return jnp.sum(residual ** 2) / n_active


# ═══════════════════════════════════════════════════════════════════════
# 2.  Boundary loss
# ═══════════════════════════════════════════════════════════════════════

def loss_boundary(
    eps1d    : jnp.ndarray,
    rho_flat : jnp.ndarray,
) -> jnp.ndarray:
    """
    Boundary-zero loss: penalise emissivity outside the last closed
    flux surface (ρ ≥ 1) on the radial grid.

    Uses a smooth sigmoid ramp so gradients exist up to ~ρ = 1.1.

    Parameters
    ----------
    eps1d    : (n_radial,)  predicted radial emissivity profile
    rho_flat : (n_radial,)  normalised elliptic radius at each radial point

    Returns
    -------
    scalar float32
    """
    # Soft mask: 0 inside plasma, rises to 1 outside (ρ > 1)
    outside = jax.nn.sigmoid(20.0 * (rho_flat - 1.0))  # (n_radial,)
    return jnp.mean((eps1d * outside) ** 2)


# ═══════════════════════════════════════════════════════════════════════
# 3.  Smoothness loss  (1-D total variation)
# ═══════════════════════════════════════════════════════════════════════

def loss_smoothness(eps1d: jnp.ndarray) -> jnp.ndarray:
    """
    1-D total-variation (TV) regulariser along the radial profile.

    Penalises the mean absolute finite difference between adjacent
    radial points, encouraging piece-wise smooth reconstructions.

    Parameters
    ----------
    eps1d : (n_radial,)  predicted radial emissivity profile

    Returns
    -------
    scalar float32
    """
    return jnp.mean(jnp.abs(jnp.diff(eps1d)))


# ═══════════════════════════════════════════════════════════════════════
# 4.  Positivity loss  (soft belt-and-suspenders)
# ═══════════════════════════════════════════════════════════════════════

def loss_positivity(eps1d: jnp.ndarray) -> jnp.ndarray:
    """
    Soft positivity penalty: the model already outputs sigmoid-activated
    values so this should remain near zero during healthy training.

    Provided as a diagnostic and as a numerical safeguard against
    rare floating-point underflow below zero.

    Parameters
    ----------
    eps1d : (n_radial,)  predicted radial emissivity profile

    Returns
    -------
    scalar float32
    """
    return jnp.mean(jnp.maximum(-eps1d, 0.0) ** 2)


# ═══════════════════════════════════════════════════════════════════════
# 5.  Combined loss_fn
# ═══════════════════════════════════════════════════════════════════════

def loss_fn(
    eps1d       : jnp.ndarray,
    g_noisy     : jnp.ndarray,
    w_ops,
    active_mask : jnp.ndarray,
    rho_flat    : jnp.ndarray,
    weights     : LossWeights = DEFAULT_WEIGHTS,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Combined physics-informed loss for VICTOR v7.

    Aggregates all individual components using the given LossWeights.
    The returned loss_dict exposes every component for logging.

    Parameters
    ----------
    eps1d       : (n_radial,)  radial emissivity profile from FourierDeepONet
    g_noisy     : (128,)       noisy sinogram for current training sample
    w_ops       : WOperators   (geometry.make_W_operators)
    active_mask : (128,)       float32, 1 for active chords
    rho_flat    : (n_radial,)  normalised elliptic radius at each radial point
    weights     : LossWeights  (default: DEFAULT_WEIGHTS)

    Returns
    -------
    total_loss : scalar float32
    loss_dict  : dict[str → scalar float32]
        Keys: "proj", "boundary", "smooth", "positivity", "total"
    """
    # ── Individual components ────────────────────────────────────────
    l_proj      = loss_projection(eps1d, g_noisy, w_ops, active_mask)

    l_boundary  = loss_boundary(eps1d, rho_flat)

    l_smooth    = loss_smoothness(eps1d)

    l_positivity = loss_positivity(eps1d)

    # ── Weighted sum ─────────────────────────────────────────────────
    total = (
        weights.w_proj       * l_proj
      + weights.w_boundary   * l_boundary
      + weights.w_smooth     * l_smooth
      + weights.w_positivity * l_positivity
    )

    loss_dict = {
        "proj"       : l_proj,
        "boundary"   : l_boundary,
        "smooth"     : l_smooth,
        "positivity" : l_positivity,
        "total"      : total,
    }

    return total, loss_dict


# ═══════════════════════════════════════════════════════════════════════
# 6.  Smoke-test / quick verification
# ═══════════════════════════════════════════════════════════════════════

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

    All values must be finite (no NaN / Inf).  Asserts on failure.

    Parameters
    ----------
    model     : FourierDeepONet instance
    params    : initialised param tree
    profile   : one entry from data_loader.load_profiles()
                Must contain keys: "g_ideal", "xi", "rho_flat"
    w_ops     : WOperators
    grids     : PixelGrids  (provides RHO_RADIAL for boundary loss)
    rho_graph : RhoGraph    (unused in v7; kept for API compatibility)
    """
    # ── Forward pass — v7: model takes (g, xi) only ──────────────────
    eps1d = model.apply(
        params,
        profile["g_ideal"],   # (n_chords,)
        profile["xi"],        # (9,)
    )

    # ── Noisy sinogram (use sigma=0 for deterministic test) ──────────
    g_noisy     = profile["g_ideal"]
    active_mask = jnp.ones(128, dtype=jnp.float32)   # test: all active
    rho_flat    = grids.RHO_RADIAL                    # (N_RADIAL,)

    # ── Run combined loss ─────────────────────────────────────────────
    total, ld = loss_fn(
        eps1d,
        g_noisy,
        w_ops,
        active_mask,
        rho_flat,
    )

    # ── Report ────────────────────────────────────────────────────────
    print("── losses.py  verify_losses ────────────────────────────")
    print(f"  eps1d shape  : {eps1d.shape}  "
          f"range=[{float(eps1d.min()):.4f}, {float(eps1d.max()):.4f}]")
    for k, v in ld.items():
        flag = "  ✓" if jnp.isfinite(v) else "  ✗ NON-FINITE"
        print(f"  {k:<12}: {float(v):.6f}{flag}")
    print("────────────────────────────────────────────────────────")

    bad = [k for k, v in ld.items() if not jnp.isfinite(v)]
    assert not bad, f"Non-finite loss components: {bad}"
    print("OK  losses.py verified")


# ── Module self-test ─────────────────────────────────────────────────

if __name__ == "__main__":
    import jax
    key      = jax.random.PRNGKey(42)
    n_radial = cfg.N_RADIAL if hasattr(cfg, "N_RADIAL") else 128

    # Dummy arrays with correct shapes
    eps1d    = jax.random.uniform(key, (n_radial,))
    g        = jax.random.uniform(key, (128,))
    rho_flat = jnp.linspace(0.0, 1.2, n_radial)
    amask    = jnp.ones(128)

    # Minimal stub WOperators for self-test (no actual W matrix needed)
    class _StubOps:
        @staticmethod
        def matvec(ef):
            return jnp.zeros(128)

    stub_ops = _StubOps()

    total, ld = loss_fn(eps1d, g, stub_ops, amask, rho_flat)

    print("Self-test loss_fn output:")
    for k, v in ld.items():
        print(f"  {k:<12}: {float(v):.6f}")
    print("Self-test passed." if jnp.isfinite(total) else "FAILED — non-finite total!")
