# ============================================================
# VICTOR v6.0 — losses.py
# All individual loss functions + combined loss_fn
# ============================================================
# Public API
# ----------
#   loss_projection(eps_out, g_noisy, w_ops, active_mask)
#       → scalar    Sinogram data-fidelity (MSE on active chords)
#
#   loss_boundary(eps_out, rho_2d)
#       → scalar    Hard boundary: emissivity must vanish for ρ ≥ 1
#
#   loss_smoothness(eps_out)
#       → scalar    Total-variation regulariser (finite differences)
#
#   loss_ensemble_nll(preds, g_noisy, w_ops, active_mask, log_noise)
#       → scalar    Negative log-likelihood of ensemble members
#
#   loss_ensemble_diversity(preds)
#       → scalar    Repulsion term: penalises collapsed ensemble
#
#   loss_isotropy(eps_out, rho_flat)
#       → scalar    Physics prior: ε should vary mainly with ρ, not θ
#
#   loss_positivity(eps_out)
#       → scalar    Soft penalty for negative pixels (belt-and-suspenders)
#
#   loss_fn(eps_out, mean, std, preds, g_noisy,
#           w_ops, active_mask, rho_2d, rho_flat, log_noise)
#       → (total_loss, loss_dict)
#
# Design principles
# -----------------
#  • All functions are pure JAX — no side effects, no globals mutated.
#  • loss_fn is the single entry point consumed by the training step.
#  • loss_dict exposes every individual component for TensorBoard / logs.
#  • Weights are tunable via the LossWeights dataclass (or cfg constants).
#  • All losses return scalar float32; shapes are asserted in docstrings.
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
    Defaults are calibrated for WEST soft-X-ray inversion at VICTOR v6.

    Attributes
    ----------
    w_proj       : weight for projection / data-fidelity loss
    w_boundary   : weight for boundary-zero enforcement
    w_smooth     : weight for total-variation smoothness regulariser
    w_nll        : weight for ensemble NLL (heteroscedastic)
    w_diversity  : weight for ensemble diversity repulsion
    w_isotropy   : weight for flux-surface isotropy prior
    w_positivity : weight for soft positivity penalty
    """
    w_proj       : float = 1.0
    w_boundary   : float = 5.0
    w_smooth     : float = 0.02
    w_nll        : float = 0.5
    w_diversity  : float = 0.1
    w_isotropy   : float = 0.05
    w_positivity : float = 0.5


# Singleton default weights — importable as `losses.DEFAULT_WEIGHTS`
DEFAULT_WEIGHTS = LossWeights()


# ═══════════════════════════════════════════════════════════════════════
# 1.  Projection loss  (data fidelity)
# ═══════════════════════════════════════════════════════════════════════

def loss_projection(
    eps_out     : jnp.ndarray,
    g_noisy     : jnp.ndarray,
    w_ops,
    active_mask : jnp.ndarray,
) -> jnp.ndarray:
    """
    Sinogram data-fidelity loss (MSE restricted to active chords).

    Computes the forward projection g_pred = W · ε_flat and
    penalises the mean-square residual on all active (non-zero-sum)
    rows of the W matrix.

    Parameters
    ----------
    eps_out     : (N, N)    predicted emissivity (masked, positive)
    g_noisy     : (128,)    noisy measured sinogram (normalised)
    w_ops       : WOperators  (geometry.make_W_operators)
    active_mask : (128,)    bool/float — 1 for active chords

    Returns
    -------
    scalar float32
    """
    eps_flat = eps_out.flatten()                       # (N²,)
    g_pred   = w_ops.matvec(eps_flat)                  # (128,)

    residual = (g_pred - g_noisy) * active_mask        # zero-out inactive
    n_active = jnp.maximum(active_mask.sum(), 1.0)
    return jnp.sum(residual ** 2) / n_active


# ═══════════════════════════════════════════════════════════════════════
# 2.  Boundary loss
# ═══════════════════════════════════════════════════════════════════════

def loss_boundary(
    eps_out : jnp.ndarray,
    rho_2d  : jnp.ndarray,
) -> jnp.ndarray:
    """
    Boundary-zero loss: penalise emissivity outside the last closed
    flux surface (ρ ≥ 1).

    Uses a smooth sigmoid ramp so gradients exist up to ~ρ = 1.1.

    Parameters
    ----------
    eps_out : (N, N)      predicted emissivity
    rho_2d  : (N, N)      normalised elliptic radius (PixelGrids.RHO_2D)

    Returns
    -------
    scalar float32
    """
    # Soft mask: 0 inside plasma, rises to 1 outside (ρ > 1)
    outside = jax.nn.sigmoid(20.0 * (rho_2d - 1.0))   # (N, N)
    return jnp.mean((eps_out * outside) ** 2)


# ═══════════════════════════════════════════════════════════════════════
# 3.  Smoothness loss  (anisotropic total variation)
# ═══════════════════════════════════════════════════════════════════════

def loss_smoothness(eps_out: jnp.ndarray) -> jnp.ndarray:
    """
    Anisotropic total-variation (TV) regulariser.

    Penalises the mean absolute finite difference in both the R and Z
    directions, encouraging piece-wise smooth reconstructions without
    over-blurring sharp radial gradients.

    Parameters
    ----------
    eps_out : (N, N)   predicted emissivity

    Returns
    -------
    scalar float32
    """
    # Finite differences along both axes
    dR = jnp.abs(jnp.diff(eps_out, axis=1))    # (N, N-1)
    dZ = jnp.abs(jnp.diff(eps_out, axis=0))    # (N-1, N)
    return 0.5 * (jnp.mean(dR) + jnp.mean(dZ))


# ═══════════════════════════════════════════════════════════════════════
# 4.  Ensemble NLL  (heteroscedastic negative log-likelihood)
# ═══════════════════════════════════════════════════════════════════════

def loss_ensemble_nll(
    preds       : jnp.ndarray,
    g_noisy     : jnp.ndarray,
    w_ops,
    active_mask : jnp.ndarray,
    log_noise   : jnp.ndarray,
) -> jnp.ndarray:
    """
    Heteroscedastic NLL: each ensemble member predicts a sinogram and
    the combined Gaussian likelihood is maximised.

    For member m with prediction ε_m and learnable noise log σ_m:
        NLL_m = 0.5 * [ (g_m - g_noisy)² / σ_m² + log σ_m² ]
    averaged over active chords and members.

    Parameters
    ----------
    preds       : (M, N²)   per-member emissivity predictions (pre-PIGNO)
    g_noisy     : (128,)    noisy sinogram
    w_ops       : WOperators
    active_mask : (128,)    bool/float
    log_noise   : (M,)      learnable log(σ_m)  — from model.log_noise

    Returns
    -------
    scalar float32
    """
    M        = preds.shape[0]
    n_active = jnp.maximum(active_mask.sum(), 1.0)

    def member_nll(m):
        g_m     = w_ops.matvec(preds[m])                       # (128,)
        sigma_m = jnp.exp(log_noise[m]) + 1e-6
        r       = (g_m - g_noisy) * active_mask
        return jnp.sum(0.5 * (r ** 2 / sigma_m ** 2 + 2.0 * log_noise[m])) / n_active

    # vmap over members (avoids Python loop in JIT)
    nll_per_member = jax.vmap(member_nll)(jnp.arange(M))
    return jnp.mean(nll_per_member)


# ═══════════════════════════════════════════════════════════════════════
# 5.  Ensemble diversity  (variance repulsion)
# ═══════════════════════════════════════════════════════════════════════

def loss_ensemble_diversity(preds: jnp.ndarray) -> jnp.ndarray:
    """
    Ensemble diversity loss: penalise collapsed ensembles by
    maximising the mean inter-member variance across pixels.

    A collapsed ensemble (all members identical) gives loss ≈ 0;
    a diverse ensemble gives a large negative contribution, so the
    sign is negated: we *minimise* the negative variance.

    Parameters
    ----------
    preds : (M, N²)   per-member emissivity predictions

    Returns
    -------
    scalar float32   (non-positive; 0 when fully collapsed)
    """
    # Variance across members at each pixel, then mean over pixels
    var_per_pixel = jnp.var(preds, axis=0)          # (N²,)
    return -jnp.mean(var_per_pixel)


# ═══════════════════════════════════════════════════════════════════════
# 6.  Isotropy loss  (flux-surface physics prior)
# ═══════════════════════════════════════════════════════════════════════

def loss_isotropy(
    eps_out  : jnp.ndarray,
    rho_flat : jnp.ndarray,
) -> jnp.ndarray:
    """
    Physics prior: emissivity should be approximately constant on
    iso-ρ surfaces (i.e., ε ≈ f(ρ) near the core).

    Bins pixels by ρ into N_BINS radial shells and penalises the
    intra-shell variance, weighted by the shell's mean emissivity
    (so dim shells don't dominate).

    Parameters
    ----------
    eps_out  : (N, N)    predicted emissivity
    rho_flat : (N²,)     normalised elliptic radius (PixelGrids.RHO_FLAT)

    Returns
    -------
    scalar float32
    """
    N_BINS   = 32
    eps_flat = eps_out.flatten()          # (N²,)

    # Discretise ρ ∈ [0, 1) into N_BINS uniform shells
    bin_idx  = jnp.clip(
        (rho_flat * N_BINS).astype(jnp.int32), 0, N_BINS - 1
    )                                      # (N²,)

    # For each shell: compute mean and variance via scatter
    # --- mean ---
    bin_sum  = jnp.zeros(N_BINS).at[bin_idx].add(eps_flat)
    bin_cnt  = jnp.zeros(N_BINS).at[bin_idx].add(1.0)
    bin_cnt  = jnp.maximum(bin_cnt, 1.0)
    bin_mean = bin_sum / bin_cnt           # (N_BINS,)

    # --- variance ---
    residuals   = eps_flat - bin_mean[bin_idx]      # (N²,)
    bin_var_sum = jnp.zeros(N_BINS).at[bin_idx].add(residuals ** 2)
    bin_var     = bin_var_sum / bin_cnt             # (N_BINS,)

    # Weight shells by mean emissivity (ignore empty outer shells)
    w = jnp.maximum(bin_mean, 1e-8)                 # (N_BINS,)
    return jnp.sum(bin_var * w) / jnp.sum(w)


# ═══════════════════════════════════════════════════════════════════════
# 7.  Positivity loss  (soft belt-and-suspenders)
# ═══════════════════════════════════════════════════════════════════════

def loss_positivity(eps_out: jnp.ndarray) -> jnp.ndarray:
    """
    Soft positivity penalty: the model already outputs softplus-activated
    values so this should remain near zero during healthy training.

    Provided as a diagnostic and as a numerical safeguard against
    rare floating-point underflow below zero.

    Parameters
    ----------
    eps_out : (N, N)   predicted emissivity

    Returns
    -------
    scalar float32
    """
    return jnp.mean(jnp.maximum(-eps_out, 0.0) ** 2)


# ═══════════════════════════════════════════════════════════════════════
# 8.  Combined loss_fn
# ═══════════════════════════════════════════════════════════════════════

def loss_fn(
    eps_out     : jnp.ndarray,
    mean        : jnp.ndarray,
    std         : jnp.ndarray,
    preds       : jnp.ndarray,
    g_noisy     : jnp.ndarray,
    w_ops,
    active_mask : jnp.ndarray,
    rho_2d      : jnp.ndarray,
    rho_flat    : jnp.ndarray,
    log_noise   : jnp.ndarray,
    weights     : LossWeights = DEFAULT_WEIGHTS,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Combined physics-informed loss for VICTOR v6.

    Aggregates all individual components using the given LossWeights.
    The returned loss_dict exposes every component for logging/TensorBoard.

    Parameters
    ----------
    eps_out     : (N, N)    final masked emissivity  [model output #0]
    mean        : (N²,)     ensemble mean  (pre-PIGNO)  [model output #1]
    std         : (N²,)     ensemble std   (pre-PIGNO)  [model output #2]
    preds       : (M, N²)   per-member predictions     [model output #3]
    g_noisy     : (128,)    noisy sinogram for current training sample
    w_ops       : WOperators  (geometry.make_W_operators)
    active_mask : (128,)    float32, 1 for active chords
    rho_2d      : (N, N)    normalised elliptic radius
    rho_flat    : (N²,)     same, flattened
    log_noise   : (M,)      learnable log(σ_m) from model.log_noise
    weights     : LossWeights  (default: DEFAULT_WEIGHTS)

    Returns
    -------
    total_loss : scalar float32
    loss_dict  : dict[str → scalar float32]
        Keys: "proj", "boundary", "smooth", "nll", "diversity",
              "isotropy", "positivity", "total"
    """
    # ── Individual components ────────────────────────────────────────
    l_proj      = loss_projection(
                      eps_out, g_noisy, w_ops, active_mask)

    l_boundary  = loss_boundary(eps_out, rho_2d)

    l_smooth    = loss_smoothness(eps_out)

    l_nll       = loss_ensemble_nll(
                      preds, g_noisy, w_ops, active_mask, log_noise)

    l_diversity = loss_ensemble_diversity(preds)

    l_isotropy  = loss_isotropy(eps_out, rho_flat)

    l_positivity = loss_positivity(eps_out)

    # ── Weighted sum ────────────────────────────────────────────────
    total = (
        weights.w_proj       * l_proj
      + weights.w_boundary   * l_boundary
      + weights.w_smooth     * l_smooth
      + weights.w_nll        * l_nll
      + weights.w_diversity  * l_diversity
      + weights.w_isotropy   * l_isotropy
      + weights.w_positivity * l_positivity
    )

    loss_dict = {
        "proj"       : l_proj,
        "boundary"   : l_boundary,
        "smooth"     : l_smooth,
        "nll"        : l_nll,
        "diversity"  : l_diversity,
        "isotropy"   : l_isotropy,
        "positivity" : l_positivity,
        "total"      : total,
    }

    return total, loss_dict


# ═══════════════════════════════════════════════════════════════════════
# 9.  Smoke-test / quick verification
# ═══════════════════════════════════════════════════════════════════════

def verify_losses(
    model,
    params    : dict,
    profile   : dict,
    w_ops,
    grids,
    rho_graph,
) -> None:
    """
    Run a single forward pass + loss_fn and print a diagnostic table.

    All values must be finite (no NaN / Inf).  Asserts on failure.

    Parameters
    ----------
    model     : VICTOR_v6 instance
    params    : initialised param tree
    profile   : one entry from data_loader.load_profiles()
    w_ops     : WOperators
    grids     : PixelGrids
    rho_graph : RhoGraph
    """
    import jax.numpy as jnp

    # ── Forward pass ────────────────────────────────────────────────
    eps_out, mean, std, preds = model.apply(
        params,
        grids.R_PIX,
        grids.Z_PIX,
        profile["psi_n"],
        profile["bpol_n"],
        rho_graph.EDGES_SRC,
        rho_graph.EDGES_DST,
        rho_graph.EDGE_W,
        rho_graph.NODE_DEG,
        grids.RHO_2D,
    )

    # ── Noisy sinogram (use sigma=0 for deterministic test) ─────────
    g_noisy     = profile["g_ideal"]
    active_mask = jnp.ones(128, dtype=jnp.float32)   # test: all active

    # ── Extract log_noise from params ───────────────────────────────
    log_noise   = params["params"]["log_noise"]       # (M,)

    # ── Run combined loss ────────────────────────────────────────────
    total, ld   = loss_fn(
        eps_out, mean, std, preds,
        g_noisy, w_ops, active_mask,
        grids.RHO_2D, grids.RHO_FLAT,
        log_noise,
    )

    # ── Report ───────────────────────────────────────────────────────
    print("── losses.py  verify_losses ────────────────────────────")
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
    key   = jax.random.PRNGKey(42)
    N     = cfg.N_GRID
    M     = cfg.N_ENS

    # Dummy arrays with correct shapes
    eps   = jax.random.uniform(key, (N, N))
    preds = jax.random.uniform(key, (M, N * N))
    g     = jax.random.uniform(key, (128,))
    rho2d = jnp.ones((N, N)) * 0.5
    rhof  = jnp.ones((N * N,)) * 0.5
    lnoise= jnp.full((M,), -3.0)
    amask = jnp.ones(128)
    mean  = jnp.mean(preds, axis=0)
    std   = jnp.std(preds, axis=0)

    # Minimal stub WOperators for self-test (no actual W matrix needed)
    class _StubOps:
        @staticmethod
        def matvec(ef):
            return jnp.zeros(128)

    stub_ops = _StubOps()

    total, ld = loss_fn(
        eps, mean, std, preds, g,
        stub_ops, amask, rho2d, rhof, lnoise,
    )

    print("Self-test loss_fn output:")
    for k, v in ld.items():
        print(f"  {k:<12}: {float(v):.6f}")
    print("Self-test passed." if jnp.isfinite(total) else "FAILED — non-finite total!")
