# ============================================================
# VICTOR v8.0 — losses.py
# Physics-informed losses for poloidal-aware reconstruction
# ============================================================
# Public API
# ----------
#   LossWeights          — scalar multipliers NamedTuple
#   DEFAULT_WEIGHTS      — singleton with v8 defaults
#
#   loss_projection(eps2d, g_noisy, w_ops, active_mask)
#       -> scalar    Sinogram data-fidelity (MSE on active chords)
#
#   loss_boundary(coeff, rho_radial)
#       -> scalar    a0 must vanish for rho >= 1
#
#   loss_smoothness(coeff)
#       -> scalar    Total-variation on a0 radial profile
#
#   loss_polar(coeff)
#       -> scalar    Poloidal regularisation: penalise large harmonics
#
#   compute_metrics(eps2d_pred, eps2d_gt)
#       -> dict      MSE, PSNR, CC  (used at log time, not in grad)
#
#   loss_fn(coeff, g_noisy, w_ops, active_mask,
#           rho_radial, theta_flat, rho_flat, weights)
#       -> (total_loss, loss_dict)
#
# v8 changes vs v7.1
# ------------------
#  * All loss functions now operate on coeff: (N_RADIAL, N_CHANNELS_OUT)
#    or on eps2d: (N_GRID²,) — clearly documented per function.
#
#  * loss_projection: now calls build_eps2d() -> eps2d -> W(eps2d)
#    instead of W(eps1d).  True 2D field used in projection.
#
#  * loss_boundary: operates on a0 channel only (coeff[:, 0]).
#    Radial smoothness is meaningless for unconstrained harmonic channels.
#
#  * loss_smoothness: operates on a0 channel only (coeff[:, 0]).
#
#  * loss_polar (NEW): L_pol = mean(harmonics²) with weight W_POLAR=0.001.
#    Encourages near-symmetry while allowing asymmetry.
#
#  * loss_pde: DISABLED (v8 spec). Replaced by build_eps2d structure.
#  * loss_positivity: DISABLED (v8 spec). softplus on a0 handles this.
#
#  * compute_metrics (NEW): pure JAX function returning MSE, PSNR, CC.
#    Called at log intervals (outside JIT or via jax.device_get).
#    Requires eps2d_gt from the profile dict.
#
#  * LossWeights: removed w_positivity, w_pde. Added w_polar.
# ============================================================

from __future__ import annotations

from typing import Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from victor import config as cfg
from victor.model import build_eps2d


# =======================================================================
# 0.  Loss weights
# =======================================================================

class LossWeights(NamedTuple):
    """
    Scalar multipliers for each v8 loss component.

    Attributes
    ----------
    w_proj     : weight for sinogram data-fidelity loss (dominant, = 1.0)
    w_boundary : weight for boundary-zero enforcement on a0
    w_smooth   : weight for TV regulariser on a0 radial profile
    w_polar    : weight for poloidal harmonic regularisation (L_pol)
    """
    w_proj     : float = cfg.W_PROJ
    w_boundary : float = cfg.W_BOUNDARY
    w_smooth   : float = cfg.W_SMOOTH
    w_polar    : float = cfg.W_POLAR


DEFAULT_WEIGHTS = LossWeights()


# =======================================================================
# 1.  Projection loss  (data fidelity)
# =======================================================================

def loss_projection(
    eps2d       : jnp.ndarray,   # (N_GRID²,) true 2D field
    g_noisy     : jnp.ndarray,   # (N_CHORDS,) noisy sinogram
    w_ops,                        # WOperators from geometry.py
    active_mask : jnp.ndarray,   # (N_CHORDS,) float32
) -> jnp.ndarray:
    """
    Sinogram data-fidelity loss: MSE between predicted and measured
    sinogram, restricted to active chords.

    v8: uses eps2d (true 2D field) instead of eps1d.

    Parameters
    ----------
    eps2d       : (N_GRID²,)   true 2D emissivity, output of build_eps2d()
    g_noisy     : (N_CHORDS,)  noisy measured sinogram (normalised)
    w_ops       : WOperators   geometry.make_W_operators()
    active_mask : (N_CHORDS,)  1 for active chords, 0 for dead chords

    Returns
    -------
    scalar float32
    """
    g_pred   = w_ops.matvec(eps2d)
    residual = (g_pred - g_noisy) * active_mask
    n_active = jnp.maximum(active_mask.sum(), 1.0)
    return jnp.sum(residual ** 2) / n_active


# =======================================================================
# 2.  Boundary loss  (a0 channel only)
# =======================================================================

def loss_boundary(
    coeff      : jnp.ndarray,   # (N_RADIAL, N_CHANNELS_OUT)
    rho_radial : jnp.ndarray,   # (N_RADIAL,)
) -> jnp.ndarray:
    """
    Boundary-zero loss: penalise symmetric emission a0 outside LCFS.

    Operates on a0 = coeff[:, 0] only.  Harmonic channels are not
    penalised here — they are naturally small due to L_pol regularisation
    and the HARMONIC_INIT_SCALE initialisation.

    Parameters
    ----------
    coeff      : (N_RADIAL, N_CHANNELS_OUT)  coefficient tensor
    rho_radial : (N_RADIAL,)  normalised radial axis

    Returns
    -------
    scalar float32
    """
    a0      = coeff[:, 0]                                      # (N_RADIAL,)
    outside = jax.nn.sigmoid(20.0 * (rho_radial - 1.0))
    return jnp.mean((a0 * outside) ** 2)


# =======================================================================
# 3.  Smoothness loss  (a0 TV, radial direction)
# =======================================================================

def loss_smoothness(coeff: jnp.ndarray) -> jnp.ndarray:
    """
    1-D total-variation regulariser on the a0 symmetric emission profile.

    Operates on a0 = coeff[:, 0] only.  Harmonic profiles can be rough
    (they represent genuine poloidal asymmetry) and are not regularised here.

    Parameters
    ----------
    coeff : (N_RADIAL, N_CHANNELS_OUT)

    Returns
    -------
    scalar float32
    """
    a0 = coeff[:, 0]                                           # (N_RADIAL,)
    return jnp.mean(jnp.abs(jnp.diff(a0)))


# =======================================================================
# 4.  Polar regularisation loss  (NEW v8)
# =======================================================================

def loss_polar(coeff: jnp.ndarray) -> jnp.ndarray:
    """
    Poloidal harmonic regularisation.

    Penalises large harmonic coefficients (channels 1+) to encourage
    near flux-surface symmetry while allowing genuine asymmetry to emerge
    when supported by the sinogram data.

    L_pol = mean(harmonics²)

    With w_polar = 0.001, this is a soft prior: the network can deviate
    from symmetry if the projection loss benefit exceeds 0.001 per unit
    harmonic energy.

    Parameters
    ----------
    coeff : (N_RADIAL, N_CHANNELS_OUT)
            coeff[:, 0]  = a0 (not penalised here)
            coeff[:, 1:] = harmonics (penalised)

    Returns
    -------
    scalar float32
    """
    harmonics = coeff[:, 1:]                                   # (N_RADIAL, 2*N_H)
    return jnp.mean(harmonics ** 2)


# =======================================================================
# 5.  Quality metrics  (MSE, PSNR, CC)
# =======================================================================

def compute_metrics(
    eps2d_pred : jnp.ndarray,   # (N_GRID²,) or (N_GRID, N_GRID)
    eps2d_gt   : jnp.ndarray,   # (N_GRID²,) or (N_GRID, N_GRID) ground truth
) -> Dict[str, jnp.ndarray]:
    """
    Compute reconstruction quality metrics.

    Pure JAX — can be called inside or outside JIT.
    Called via jax.device_get() at log intervals in the trainer.

    Metrics
    -------
    MSE  : mean squared error between prediction and ground truth.
           Lower is better.

    PSNR : peak signal-to-noise ratio [dB].
           PSNR = 20 * log10(max(target) / sqrt(MSE))
           Higher is better.  Target > 25 dB.

    CC   : Pearson correlation coefficient.
           CC = cov(pred, gt) / (std(pred) * std(gt))
           Range [-1, 1].  Target > 0.95.

    Proj_MSE : MSE on the sinogram (computed separately in loss_fn,
               reported here for convenience if g_pred/g_gt are passed).
               NOT computed by this function — see loss_projection().

    Parameters
    ----------
    eps2d_pred : predicted emissivity field (flattened or 2D)
    eps2d_gt   : ground truth emissivity field (same shape)

    Returns
    -------
    dict with keys: "mse", "psnr", "cc"
    All values are scalar float32 JAX arrays.
    """
    pred = eps2d_pred.ravel().astype(jnp.float32)
    gt   = eps2d_gt.ravel().astype(jnp.float32)

    # MSE
    mse = jnp.mean((pred - gt) ** 2)

    # PSNR  (guard against log(0) with eps)
    peak = jnp.maximum(jnp.max(jnp.abs(gt)), 1e-8)
    psnr = 20.0 * jnp.log10(peak / jnp.sqrt(jnp.maximum(mse, 1e-12)))

    # Pearson CC
    pred_c = pred - jnp.mean(pred)
    gt_c   = gt   - jnp.mean(gt)
    num    = jnp.sum(pred_c * gt_c)
    denom  = jnp.sqrt(jnp.sum(pred_c**2) * jnp.sum(gt_c**2) + 1e-12)
    cc     = num / denom

    return {"mse": mse, "psnr": psnr, "cc": cc}


# =======================================================================
# 6.  Symmetry diagnostics  (for evaluation / reporting)
# =======================================================================

def compute_symmetry_diagnostics(
    coeff : jnp.ndarray,   # (N_RADIAL, N_CHANNELS_OUT)
) -> Dict[str, jnp.ndarray]:
    """
    Compute poloidal symmetry diagnostics for the evaluation report.

    Symmetry Ratio   = mean(a0²) / (mean(a0²) + mean(harmonics²))
                       Range [0, 1].  1.0 = perfectly symmetric.

    Poloidal Energy  = 1 - Symmetry Ratio
                       Fraction of total field energy in harmonics.

    Parameters
    ----------
    coeff : (N_RADIAL, N_CHANNELS_OUT)

    Returns
    -------
    dict with keys: "symmetry_ratio", "poloidal_energy"
    """
    a0_energy  = jnp.mean(coeff[:, 0] ** 2)
    harm_energy= jnp.mean(coeff[:, 1:] ** 2) if coeff.shape[1] > 1 \
                 else jnp.zeros(())
    total      = a0_energy + harm_energy + 1e-12
    sym_ratio  = a0_energy / total
    return {
        "symmetry_ratio" : sym_ratio,
        "poloidal_energy": 1.0 - sym_ratio,
    }


# =======================================================================
# 7.  Combined loss_fn
# =======================================================================

def loss_fn(
    coeff       : jnp.ndarray,         # (N_RADIAL, N_CHANNELS_OUT)
    g_noisy     : jnp.ndarray,         # (N_CHORDS,)
    w_ops,                              # WOperators
    active_mask : jnp.ndarray,         # (N_CHORDS,)
    rho_radial  : jnp.ndarray,         # (N_RADIAL,)
    theta_flat  : jnp.ndarray,         # (N_GRID²,)
    rho_flat    : jnp.ndarray,         # (N_GRID²,)
    weights     : LossWeights = DEFAULT_WEIGHTS,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Combined physics-informed loss for VICTOR v8.

    Builds the true 2D emissivity field from coeff internally, then
    computes all loss components.

    Parameters
    ----------
    coeff       : (N_RADIAL, N_CHANNELS_OUT)  model output
    g_noisy     : (N_CHORDS,)   noisy sinogram
    w_ops       : WOperators    from geometry.make_W_operators()
    active_mask : (N_CHORDS,)   float32, 1 for active chords
    rho_radial  : (N_RADIAL,)   radial axis for boundary / smoothness
    theta_flat  : (N_GRID²,)   poloidal angle per pixel
    rho_flat    : (N_GRID²,)   rho per pixel (for build_eps2d mapping)
    weights     : LossWeights   component multipliers

    Returns
    -------
    total_loss : scalar float32
    loss_dict  : dict[str -> scalar float32]
        Keys: "proj", "boundary", "smooth", "polar", "total"
    """
    # Build true 2D field (no clip — softplus on a0 keeps it well-behaved)
    eps2d = build_eps2d(coeff, rho_flat, theta_flat)           # (N_GRID²,)

    # Individual components
    l_proj     = loss_projection(eps2d, g_noisy, w_ops, active_mask)
    l_boundary = loss_boundary(coeff, rho_radial)
    l_smooth   = loss_smoothness(coeff)
    l_polar    = loss_polar(coeff)

    # Weighted sum
    total = (
        weights.w_proj     * l_proj
      + weights.w_boundary * l_boundary
      + weights.w_smooth   * l_smooth
      + weights.w_polar    * l_polar
    )

    loss_dict = {
        "proj"    : l_proj,
        "boundary": l_boundary,
        "smooth"  : l_smooth,
        "polar"   : l_polar,
        "total"   : total,
    }
    return total, loss_dict


# =======================================================================
# 8.  Smoke-test
# =======================================================================

def verify_losses(model, params, profile, w_ops, grids) -> None:
    """
    Run a single forward pass + loss_fn and print a diagnostic table.
    All values must be finite.  Asserts on failure.
    """
    from victor.model import build_eps2d

    coeff = model.apply(
        params,
        profile["g_ideal"],
        profile["xi"],
        profile["psi_n"].reshape(-1),
        profile["rho_n"].reshape(-1),
    )

    g_noisy     = profile["g_ideal"]
    active_mask = jnp.ones(128, dtype=jnp.float32)
    rho_radial  = grids.RHO_RADIAL
    theta_flat  = grids.THETA_FLAT
    rho_flat    = grids.RHO_FLAT

    print("── losses.py v8  verify_losses ─────────────────────────────")
    print(f"  coeff shape  : {coeff.shape}")

    total, ld = loss_fn(
        coeff, g_noisy, w_ops, active_mask,
        rho_radial, theta_flat, rho_flat,
    )

    for k, v in ld.items():
        flag = "  OK" if jnp.isfinite(v) else "  NON-FINITE !!!"
        print(f"    {k:<12}: {float(v):.6f}{flag}")

    # Metrics (requires eps2d_gt)
    eps2d_pred = build_eps2d(coeff, rho_flat, theta_flat)
    eps2d_gt   = jnp.array(profile["eps_n"].flatten())
    metrics    = compute_metrics(eps2d_pred, eps2d_gt)
    print(f"\n  MSE  = {float(metrics['mse']):.6f}")
    print(f"  PSNR = {float(metrics['psnr']):.2f} dB")
    print(f"  CC   = {float(metrics['cc']):.4f}")

    print("────────────────────────────────────────────────────────────")
    bad = [k for k, v in ld.items() if not jnp.isfinite(v)]
    assert not bad, f"Non-finite loss components: {bad}"
    print("OK  losses.py v8 verified")


if __name__ == "__main__":
    print("losses_v8.py — run verify_losses() from the notebook.")
