# ============================================================
# VICTOR v8.2 — trainer.py
# Closure-based JIT training step, training loop, stage logic
# ============================================================
# Public API
# ----------
#   build_optimizer(total_steps, lr, warmup, weight_decay, clip_norm,
#                   beta1, beta2)                           [v8.2: +beta1, beta2]
#                                          -> tx (optax chain, AdamW)
#   make_train_step(model, w_ops, weights, tx, batch_size, log_vars)
#                                          -> step_fn
#   CurriculumSchedule                     — noise/sigma curriculum helper
#                                            [v8.2: +physics_warmup mode]
#   train_one_profile(step_fn, ...)        -> (params, log_vars, opt_state, ep_global, best_data)
#   train(step_fn, ...)                    -> (params, log_vars, opt_state, hist, best_data)
#
# v8.2 changes vs v8.1
# --------------------
#  [1] make_train_step — lerp + collocation fields threaded into loss_fn.
#      _loss_one now extracts lerp_idx_lo, lerp_idx_hi, lerp_frac, and
#      boundary_colloc from the profile dict.  These are static per-profile
#      (precomputed geometry), so they are closed over in the JIT closure
#      rather than passed as dynamic traced args; this prevents retracing
#      across profiles while still being differentiation-transparent for
#      gradients flowing through the differentiable lerp coefficients.
#
#  [2] make_train_step — skip gate monitoring added.
#      After each JIT step (outside the JIT boundary), the step wrapper
#      calls model.get_skip_gate_values(new_model_params) and stashes the
#      result in a module-level _skip_gate_cache variable.  At every
#      log_every interval, train_one_profile prints a single summary line
#      "skip_gates=[lo=X.XXX hi=X.XXX]" so "exclusive competition" collapse
#      (all gates → 0 or all → 1) is immediately visible.
#
#  [3] CurriculumSchedule — new "physics_warmup" mode.
#      For the first warmup_steps, w_pde is linearly ramped from 0 → w_pde_target
#      while sigma stays at sigma_start.  After warmup, sigma anneals normally
#      per the chosen sub-mode ("cosine" default).  Prevents the PDE residual
#      from dominating before the network has learned basic data fidelity.
#      __call__ returns (sigma, w_pde_scale) tuple in this mode, float otherwise.
#      New init params: warmup_steps (int, default 0), w_pde_target (float,
#      default cfg.W_PDE_TARGET or 0.5), post_warmup_mode (str, default "cosine").
#
#  [4] train_one_profile — handles (sigma, w_pde_scale) tuple from
#      CurriculumSchedule when in physics_warmup mode.  Constructs a
#      per-step LossWeights via ._replace(w_pde=base_w_pde * w_pde_scale),
#      then passes effective_weights into step_fn via a lightweight wrapper.
#      NOTE: step_fn itself is NOT re-JITted; only the Python-side weight
#      scalar is changed so no retracing occurs.
#
#  [5] Logging — extended per-step log line.
#      * skip_gates=[lo=X.XXX hi=X.XXX] — min/max gate across all
#        UFourierLayer1D layers, from the skip gate monitor cache.
#      * w_pde_eff=X.XXXXX — effective PDE weight after curriculum scaling.
#      * lerp_gnorm=X.XXXe-N — ‖∂L/∂coeffs[:,0]‖ (lerp coefficient gradient
#        norm) to confirm gradients flow through the new differentiable
#        interpolation; extracted from loss_dict["lerp_gnorm"] if present.
#
#  [6] build_optimizer — beta1, beta2 exposed as kwargs.
#      Previously hardcoded at optax AdamW defaults (0.9, 0.999).
#      Exposing beta2 allows experimentation with lower values (e.g. 0.95),
#      which is often better for PINN-style training where the loss
#      landscape changes character across curriculum stages.
#
# v8.1 changes vs v8.0
# --------------------
#  [1] Adaptive log_vars wired into value_and_grad.
#      make_train_step now accepts an optional log_vars dict
#      (from losses.init_log_vars()).  When supplied, the step function
#      maintains a combined param tree {"model": ..., "log_vars": ...}
#      and differentiates through BOTH.  The log_vars scalars are updated
#      by the same AdamW optimizer as the model weights, so the dynamic
#      loss balancing is actually learned rather than remaining static.
#      When log_vars=None the behaviour is identical to v8.0.
#
#  [2] step_fn return signature extended.
#      Returns (new_model_params, new_log_vars, new_opt_state, total, ld)
#      when log_vars is supplied; (new_params, None, ...) when not.
#      train_one_profile updated to pass/receive log_vars at each step.
#
#  [3] Logging extended.
#      When adaptive mode is active, sigma_{name} and w_eff_{name} values
#      from the loss_dict are printed at log intervals so you can watch
#      the dynamic weights evolve during training.
#
# v7.2 performance changes (preserved)
# ------------------------------------
#  * Noise inside JIT, vmap batching, donate_argnums=(0,1),
#    host-sync only at log intervals, steps/sec diagnostics.
# ============================================================

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax

from victor import config as cfg
from victor.losses import loss_fn, LossWeights, DEFAULT_WEIGHTS, init_log_vars


# =======================================================================
# Module-level skip gate cache
# [v8.2] Populated by the step wrapper in train_one_profile after each
# JIT call; keyed by profile index so concurrent profiles don't clobber.
# =======================================================================
_skip_gate_cache: Dict[int, jnp.ndarray] = {}


# =======================================================================
# 0.  Curriculum Schedule
# =======================================================================

class CurriculumSchedule:
    """
    Structured noise-curriculum for training VICTOR.

    Replaces the flat STAGES list with a continuous sigma schedule that
    anneals from high noise (easy, coarse) to low noise (hard, precise).
    This is curriculum learning: the network first learns gross structure
    under heavy regularisation, then refines fine detail as noise drops.

    Four annealing modes
    --------------------
    "linear"  : sigma decreases linearly from sigma_start to sigma_end
                over total_steps.  Simple and predictable.

    "cosine"  : sigma follows a cosine decay (fast early, slow late).
                Mirrors the cosine LR schedule — keeps noise high while
                the model is unstable, then quickly reaches low noise.

    "step"    : sigma drops in discrete steps, one per entry in
                sigma_steps = [(step_boundary, sigma_value), ...].
                Equivalent to the original STAGES list but expressed
                as absolute step counts rather than per-stage lengths.

    "physics_warmup" [v8.2]
                Phase 1 (step < warmup_steps): sigma is pinned to
                sigma_start and w_pde is linearly ramped from 0 to
                w_pde_target.  This prevents the PDE residual from
                dominating before the model has learned basic data
                fidelity.  Phase 2 (step >= warmup_steps): sigma
                anneals according to post_warmup_mode (default "cosine")
                while w_pde stays at w_pde_target (scale = 1.0).

                In this mode __call__ returns a 2-tuple (sigma, w_pde_scale)
                instead of a plain float.  train_one_profile detects the
                tuple and builds a per-step LossWeights accordingly.

    Parameters
    ----------
    total_steps       : int    Total training steps across all profiles/stages.
    sigma_start       : float  Initial noise fraction (default 0.01).
    sigma_end         : float  Final noise fraction (default 0.001).
    mode              : str    "linear" | "cosine" | "step" | "physics_warmup"
    sigma_steps       : list of (int, float)
                               Used only for mode="step".
    warmup_steps      : int    [physics_warmup only] Steps over which w_pde
                               is ramped from 0 to w_pde_target.  Default 0.
    w_pde_target      : float  [physics_warmup only] Target w_pde scale after
                               warmup.  Default getattr(cfg, "W_PDE_TARGET", 0.5).
    post_warmup_mode  : str    [physics_warmup only] Sub-mode for sigma after
                               warmup.  Default "cosine".

    Usage
    -----
    curriculum = CurriculumSchedule(total_steps=100_000, mode="cosine")
    sigma = curriculum(ep_global)   # returns float for non-physics_warmup modes

    curriculum_pw = CurriculumSchedule(
        total_steps=100_000, mode="physics_warmup",
        warmup_steps=5_000, w_pde_target=0.5
    )
    sigma, w_pde_scale = curriculum_pw(ep_global)  # returns (float, float)

    To convert existing STAGES to step mode:
        stages = [(3333, 0.001), (3333, 0.003), (3334, 0.008)]
        boundaries = []
        t = 0
        for n, sig in stages:
            t += n
            boundaries.append((t, sig))
        curriculum = CurriculumSchedule(
            total_steps=10_000, mode="step", sigma_steps=boundaries
        )
    """

    # [v8.2] Accepted modes updated to include "physics_warmup".
    _VALID_MODES = ("linear", "cosine", "step", "physics_warmup")

    def __init__(
        self,
        total_steps      : int,
        sigma_start      : float = 0.01,
        sigma_end        : float = 0.001,
        mode             : str   = "cosine",
        sigma_steps      : list  = None,
        # [v8.2] physics_warmup params ──────────────────────────────────
        warmup_steps     : int   = 0,
        w_pde_target     : float = None,   # falls back to cfg or 0.5
        post_warmup_mode : str   = "cosine",
    ):
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"CurriculumSchedule: unknown mode '{mode}'. "
                f"Choose one of {self._VALID_MODES}."
            )
        if mode == "step" and not sigma_steps:
            raise ValueError("CurriculumSchedule: mode='step' requires sigma_steps.")
        # [v8.2] Validate post_warmup_mode for physics_warmup
        if mode == "physics_warmup" and post_warmup_mode not in ("linear", "cosine", "step"):
            raise ValueError(
                f"CurriculumSchedule: post_warmup_mode='{post_warmup_mode}' "
                "is not valid.  Choose 'linear', 'cosine', or 'step'."
            )

        self.total_steps      = max(total_steps, 1)
        self.sigma_start      = sigma_start
        self.sigma_end        = sigma_end
        self.mode             = mode
        self.sigma_steps      = sorted(sigma_steps or [], key=lambda x: x[0])

        # [v8.2] physics_warmup state ───────────────────────────────────
        self.warmup_steps     = max(warmup_steps, 0)
        # Resolve w_pde_target: explicit arg > cfg attribute > 0.5 fallback
        self.w_pde_target     = (
            w_pde_target
            if w_pde_target is not None
            else getattr(cfg, "W_PDE_TARGET", 0.5)
        )
        self.post_warmup_mode = post_warmup_mode

    # ── helpers for the standard (non-physics_warmup) sigma annealing ──

    def _sigma_linear(self, t: float) -> float:
        """Linear sigma interpolation; t ∈ [0, 1]."""
        return float(self.sigma_start + (self.sigma_end - self.sigma_start) * t)

    def _sigma_cosine(self, t: float) -> float:
        """Cosine sigma decay; t ∈ [0, 1]."""
        cosine_t = 0.5 * (1.0 + math.cos(math.pi * t))
        return float(self.sigma_end + (self.sigma_start - self.sigma_end) * cosine_t)

    def _sigma_step(self, step: int) -> float:
        """Discrete-stage sigma lookup."""
        sigma = self.sigma_start
        for boundary, sig in self.sigma_steps:
            if step >= boundary:
                sigma = sig
        return float(sigma)

    # ── public __call__ ────────────────────────────────────────────────

    def __call__(self, step: int) -> Union[float, Tuple[float, float]]:
        """
        Compute (sigma[, w_pde_scale]) for the given global training step.

        For modes "linear", "cosine", "step":
            Returns a plain float sigma.

        For mode "physics_warmup":          [v8.2]
            Returns (sigma, w_pde_scale) where:
            - sigma        : float  noise level (pinned during warmup)
            - w_pde_scale  : float  multiplier for w_pde weight ∈ [0, 1]

        Parameters
        ----------
        step : int  Current global training step (0-indexed).
        """
        # Progress fraction over the full schedule, clamped to [0, 1]
        t = min(step, self.total_steps) / self.total_steps

        if self.mode == "linear":
            return self._sigma_linear(t)

        elif self.mode == "cosine":
            return self._sigma_cosine(t)

        elif self.mode == "step":
            return self._sigma_step(step)

        else:  # "physics_warmup" ─────────────────────────────────────
            # Phase 1: warmup — sigma fixed, w_pde linearly ramps 0 → target
            if step < self.warmup_steps and self.warmup_steps > 0:
                # [v8.2] w_pde scale ramps linearly across the warmup window
                w_pde_scale = float(step) / float(self.warmup_steps)
                sigma       = self.sigma_start
                return (float(sigma), float(w_pde_scale))

            # Phase 2: post-warmup — sigma anneals normally, w_pde held at 1.0
            # Remap step so t=0 at warmup boundary, t=1 at total_steps
            steps_remaining = max(self.total_steps - self.warmup_steps, 1)
            post_step       = step - self.warmup_steps
            t_post          = min(post_step, steps_remaining) / steps_remaining

            if self.post_warmup_mode == "linear":
                sigma = self._sigma_linear(t_post)
            elif self.post_warmup_mode == "cosine":
                sigma = self._sigma_cosine(t_post)
            else:  # "step"
                sigma = self._sigma_step(step)

            # [v8.2] After warmup, w_pde scale = 1.0 (full target weight)
            return (float(sigma), 1.0)

    def __repr__(self) -> str:
        extra = ""
        if self.mode == "physics_warmup":
            # [v8.2] Show warmup params in repr
            extra = (
                f"  warmup_steps={self.warmup_steps}  "
                f"w_pde_target={self.w_pde_target}  "
                f"post_mode={self.post_warmup_mode!r}"
            )
        return (
            f"CurriculumSchedule(mode={self.mode!r}, "
            f"sigma={self.sigma_start}→{self.sigma_end}, "
            f"steps={self.total_steps}{extra})"
        )


# =======================================================================
# 1.  Optimizer
# =======================================================================

def build_optimizer(
    total_steps  : int,
    lr           : float = cfg.LR,
    warmup       : int   = 500,
    lr_end       : float = 5e-5,
    weight_decay : float = 1e-4,
    clip_norm    : float = 1.0,
    # [v8.2] Expose Adam moment decay rates as kwargs so callers can
    # experiment with lower beta2 (e.g. 0.95) which is often better for
    # PINN-style training where the loss landscape changes character
    # across curriculum stages.
    beta1        : float = 0.9,
    beta2        : float = 0.999,
) -> optax.GradientTransformation:
    """
    Build the VICTOR optimiser: warmup-cosine-decay AdamW + grad-clip.

    Schedule: linear warm-up from 0 → lr over `warmup` steps,
    then cosine decay from lr → lr_end over the remaining steps.

    v8.2: beta1 and beta2 are now configurable kwargs (previously
    hardcoded at optax AdamW defaults 0.9 / 0.999).  Lowering beta2
    to ~0.95 can improve convergence in PINN-style training where the
    loss landscape changes character across curriculum stages.

    Parameters
    ----------
    total_steps  : int   Total number of gradient steps expected.
    lr           : float Peak learning rate (default cfg.LR = 3e-4).
    warmup       : int   Number of warm-up steps (default 500).
    lr_end       : float Final learning rate at end of cosine decay.
    weight_decay : float AdamW decoupled weight decay (default 1e-4).
    clip_norm    : float Global gradient norm clip threshold (default 1.0).
    beta1        : float Adam first-moment decay (default 0.9).   [v8.2]
    beta2        : float Adam second-moment decay (default 0.999). [v8.2]
                         Try 0.95 for faster adaptation in PINN curricula.

    Returns
    -------
    optax.GradientTransformation
    """
    sched = optax.warmup_cosine_decay_schedule(
        init_value   = 0.0,
        peak_value   = lr,
        warmup_steps = warmup,
        decay_steps  = total_steps,
        end_value    = lr_end,
    )
    return optax.chain(
        optax.clip_by_global_norm(clip_norm),
        # [v8.2] Pass (beta1, beta2) tuple explicitly so callers can tune
        # momentum hyperparameters without subclassing or monkey-patching.
        optax.adamw(sched, weight_decay=weight_decay, b1=beta1, b2=beta2),
    )


# =======================================================================
# 2.  JIT-compiled step factory
# =======================================================================

def make_train_step(
    model,
    w_ops,
    weights         : LossWeights = DEFAULT_WEIGHTS,
    tx              = None,
    batch_size      : int         = 16,
    inject_noise_fn               = None,
    n_harmonics     : int         = cfg.N_HARMONICS,
    log_vars        : Optional[Dict] = None,
    # [v8.2] Per-profile lerp + collocation geometry, closed over in JIT.
    # These are static (precomputed once per profile) so they do NOT
    # cross the JIT boundary as dynamic traced values, preventing retracing.
    lerp_idx_lo     : Optional[jnp.ndarray] = None,   # (N_RADIAL,) int32
    lerp_idx_hi     : Optional[jnp.ndarray] = None,   # (N_RADIAL,) int32
    lerp_frac       : Optional[jnp.ndarray] = None,   # (N_RADIAL,) float32
    boundary_colloc : Optional[jnp.ndarray] = None,   # (N_COLLOC, 2) float32
    psi_flat        : Optional[jnp.ndarray] = None,   # (N_GRID²,) normalised flux
):
    """
    Factory that returns a jax.jit-compiled training step for VICTOR.

    All Python objects (tx, model, w_ops, weights) are closed over so
    they never cross the JIT boundary as traced values.

    v8.2 lerp + collocation closure
    ---------------------------------
    Pass lerp_idx_lo, lerp_idx_hi, lerp_frac, and boundary_colloc at
    factory time (extracted from the profile dict before JIT compilation).
    These are precomputed geometry fields that are static per-profile, so
    closing over them prevents unnecessary retracing when switching
    profiles and keeps the dynamic arg count small.  Gradients still flow
    through lerp_frac via loss_fn's differentiable interpolation path.

    v8.2 skip gate monitoring
    -------------------------
    The returned step_fn is actually a thin Python wrapper around the
    inner @jax.jit step that also calls
        model.get_skip_gate_values(new_model_params)
    after each JIT call and stashes the result in the module-level
    _skip_gate_cache dict.  This is outside the JIT boundary so it
    involves a host sync only when explicitly fetched at log intervals.

    v8.1 adaptive log_vars
    ----------------------
    Pass log_vars=losses.init_log_vars() to enable dynamic loss balancing.
    The step function will maintain a combined param tree:
        all_params = {"model": model_params, "log_vars": log_vars}
    and differentiate through BOTH, so the uncertainty scalars are
    genuinely learned alongside the network weights.

    When log_vars=None (default), behaviour is identical to v8.0.

    Performance design (unchanged from v7.2)
    -----------------------------------------
    - inject_noise_fn runs inside JIT (noise+forward+grad fused XLA graph)
    - jax.vmap over batch dimension B (amortises dispatch overhead)
    - donate_argnums=(0, 2) reuses device buffers for params and opt_state
    - Host sync only at log_every intervals

    Parameters
    ----------
    model           : FourierDeepONet instance
    w_ops           : WOperators  (geometry.WOperators)
    weights         : LossWeights  (prior scale hints; default DEFAULT_WEIGHTS)
    tx              : optax GradientTransformation — required.
    batch_size      : int   Number of (g, xi) pairs per step (default 16).
    inject_noise_fn : callable inject_noise(g, sigma, key) — required.
    n_harmonics     : int   (default cfg.N_HARMONICS)
    log_vars        : dict of JAX scalar arrays from init_log_vars(), or None.
    lerp_idx_lo     : (N_RADIAL,) int32  lower lerp index, closed over. [v8.2]
    lerp_idx_hi     : (N_RADIAL,) int32  upper lerp index, closed over. [v8.2]
    lerp_frac       : (N_RADIAL,) float32 lerp fraction, closed over.   [v8.2]
    boundary_colloc : (N_COLLOC, 2) float32 boundary colloc pts.        [v8.2]

    Returns
    -------
    step_fn : Python callable with the same signature as the inner JIT fn.
              (model_params, log_vars_or_none, opt_state,
               g_clean, psi_batch, rho_eq_batch,
               sigma, rng, xi, active_mask,
               rho_flat_pix, theta_flat, rho_radial)
              -> (new_model_params, new_log_vars_or_none,
                  new_opt_state, total_loss, loss_dict)

              Skip gate values are written to _skip_gate_cache as a side
              effect after every call.  Read them via get_skip_gate_cache().

    NOTE: The first two return values are now model_params and log_vars
    (log_vars is None when adaptive mode is off).  Update callers accordingly.
    """
    if tx is None:
        raise ValueError("make_train_step: `tx` must be supplied.")
    if inject_noise_fn is None:
        raise ValueError(
            "make_train_step: `inject_noise_fn` must be supplied. "
            "Pass data_loader.inject_noise."
        )

    adaptive = log_vars is not None

    # ── per-sample loss (single g, single xi) ────────────────────────
    # [v8.2] lerp_idx_lo, lerp_idx_hi, lerp_frac, boundary_colloc are
    # closed over here — they are static geometry scalars/arrays that do
    # NOT change within a profile's training loop, so they are part of
    # the JIT closure rather than dynamic arguments.  This avoids
    # retracing when the same step_fn is reused across steps of the same
    # profile.  When switching to a new profile, make_train_step is called
    # again with the new profile's geometry.
    def _loss_one(model_p, lv, g_clean_i, psi_i, rho_eq_i,
                  sigma, rng_i, xi_i,
                  active_mask, rho_flat_pix, theta_flat, rho_radial):
        g_noisy = inject_noise_fn(g_clean_i, sigma, rng_i)
        coeffs  = model.apply(model_p, g_noisy, psi_i, rho_eq_i, xi_i)
        return loss_fn(
            coeffs, g_noisy, w_ops, active_mask,
            rho_flat_pix, theta_flat, rho_radial,
            weights        = weights,
            log_vars       = lv,                # None or dict of JAX scalars
            n_harmonics    = n_harmonics,
            # [v8.2] Pass closed-over lerp/collocation geometry into loss_fn
            # so the differentiable interpolation and boundary residuals use
            # the correct per-profile grid.
            lerp_idx_lo    = lerp_idx_lo,
            lerp_idx_hi    = lerp_idx_hi,
            lerp_frac      = lerp_frac,
            boundary_colloc_idx= boundary_colloc,
        )

    # ── batched loss: vmap over B samples ─────────────────────────────
    def _loss_batch(model_p, lv, g_clean, psi_batch, rho_eq_batch,
                    sigma, rngs, xi,
                    active_mask, rho_flat_pix, theta_flat, rho_radial):
        # vmap only over per-sample inputs; model_p and lv are shared
        batched = jax.vmap(
            lambda g_i, psi_i, rho_eq_i, rng_i, xi_i: _loss_one(
                model_p, lv, g_i, psi_i, rho_eq_i,
                sigma, rng_i, xi_i,
                active_mask, rho_flat_pix, theta_flat, rho_radial,
            )
        )(g_clean, psi_batch, rho_eq_batch, rngs, xi)

        totals, ld_batch = batched
        total_mean = jnp.mean(totals)
        ld_mean    = {k: jnp.mean(v) for k, v in ld_batch.items()}
        return total_mean, ld_mean

    # ── JIT-compiled step (inner) ─────────────────────────────────────
    # donate_argnums=(0, 2): donate model_params and opt_state.
    # log_vars (arg 1) is NOT donated — it is a small dict of scalars
    # that we need to read back as new_log_vars.
    @jax.jit(donate_argnums=(0, 2))
    def _jit_step(
        model_params : dict,
        log_vars_in,                   # dict of JAX scalars, or None
        opt_state,
        g_clean      : jnp.ndarray,   # (B, 128)
        psi_batch    : jnp.ndarray,   # (B, N_GRID²)
        rho_eq_batch : jnp.ndarray,   # (B, N_GRID²)
        sigma        : jnp.ndarray,   # scalar
        rng          : jnp.ndarray,   # PRNGKey
        xi           : jnp.ndarray,   # (B, 9)
        active_mask  : jnp.ndarray,   # (128,)
        rho_flat_pix : jnp.ndarray,   # (N_GRID²,)
        theta_flat   : jnp.ndarray,   # (N_GRID²,)
        rho_radial   : jnp.ndarray,   # (N_RADIAL,)
    ):
        rngs = jax.random.split(rng, g_clean.shape[0])

        if adaptive:
            # Differentiate through BOTH model weights and log_vars
            all_params = {"model": model_params, "log_vars": log_vars_in}

            def _loss_for_grad(all_p):
                return _loss_batch(
                    all_p["model"], all_p["log_vars"],
                    g_clean, psi_batch, rho_eq_batch,
                    sigma, rngs, xi,
                    active_mask, rho_flat_pix, theta_flat, rho_radial,
                )

            (total, ld), grads = jax.value_and_grad(
                _loss_for_grad, has_aux=True
            )(all_params)

            grads = jax.tree_util.tree_map(
                lambda g: jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g)),
                grads,
            )

            updates, new_opt_state = tx.update(grads, opt_state, all_params)
            new_all = optax.apply_updates(all_params, updates)
            return (new_all["model"], new_all["log_vars"],
                    new_opt_state, total, ld)

        else:
            # Fixed-weight mode — no log_vars in grad tree
            def _loss_for_grad(p):
                return _loss_batch(
                    p, None,
                    g_clean, psi_batch, rho_eq_batch,
                    sigma, rngs, xi,
                    active_mask, rho_flat_pix, theta_flat, rho_radial,
                )

            (total, ld), grads = jax.value_and_grad(
                _loss_for_grad, has_aux=True
            )(model_params)

            grads = jax.tree_util.tree_map(
                lambda g: jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g)),
                grads,
            )

            updates, new_opt_state = tx.update(grads, opt_state, model_params)
            new_params = optax.apply_updates(model_params, updates)
            return new_params, None, new_opt_state, total, ld

    # ── Outer Python wrapper — skip gate monitoring [v8.2] ────────────
    # This wrapper lives OUTSIDE the JIT boundary.  After each JIT call
    # it fetches the skip gate values from the updated model params and
    # caches them in the module-level _skip_gate_cache dict.
    # Using a wrapper (rather than putting this inside @jax.jit) keeps
    # the host sync out of the traced graph: the gate values are small
    # float arrays and the host transfer only happens at log_every
    # intervals when the caller reads from _skip_gate_cache.
    def step_fn(
        model_params,
        log_vars_in,
        opt_state,
        g_clean, psi_batch, rho_eq_batch,
        sigma, rng, xi, active_mask,
        rho_flat_pix, theta_flat, rho_radial,
        # [v8.2] Optional cache key for skip gate stash (e.g. profile index)
        _gate_cache_key: int = 0,
    ):
        """
        Thin Python wrapper around _jit_step.

        Calls the JIT-compiled inner step, then — outside the JIT
        boundary — calls model.get_skip_gate_values(new_model_params)
        and stores the result in _skip_gate_cache[_gate_cache_key].

        All positional args are passed through unchanged to _jit_step.
        Returns the same 5-tuple as _jit_step.
        """
        result = _jit_step(
            model_params, log_vars_in, opt_state,
            g_clean, psi_batch, rho_eq_batch,
            sigma, rng, xi, active_mask,
            rho_flat_pix, theta_flat, rho_radial,
        )
        new_model_params = result[0]

        # [v8.2] Stash skip gate values for monitoring.
        # model.get_skip_gate_values should return a 1-D float array or
        # dict of gate values from all UFourierLayer1D layers.
        # We store without blocking the device — the caller reads at
        # log_every intervals (by which time the kernel has finished).
        try:
            gate_vals = model.get_skip_gate_values(new_model_params)
            _skip_gate_cache[_gate_cache_key] = gate_vals
        except AttributeError:
            # [v8.2] Model may not implement get_skip_gate_values yet.
            # Silently skip rather than breaking training.
            pass

        return result

    mode_str = "adaptive (lbPINN)" if adaptive else "fixed-weight"
    lerp_str = (
        "lerp_closed_over=True"
        if lerp_frac is not None
        else "lerp_closed_over=False"   # [v8.2] lerp status in log
    )
    print(
        f"make_train_step (v8.2): step_fn compiled  "
        f"batch_size={batch_size}  donate_argnums=(0,2)  "
        f"noise_inside_jit=True  n_harmonics={n_harmonics}  "
        f"loss_mode={mode_str}  {lerp_str}"
    )
    return step_fn


# =======================================================================
# Module-level helper: read skip gate cache
# =======================================================================

def get_skip_gate_cache(cache_key: int = 0) -> Optional[jnp.ndarray]:
    """
    Return the most recently stashed skip gate values for *cache_key*.

    Returns None if no gates have been recorded yet (e.g. model does not
    implement get_skip_gate_values, or no step has been taken).

    Parameters
    ----------
    cache_key : int  Profile index used when calling step_fn (default 0).
    """
    return _skip_gate_cache.get(cache_key, None)


# =======================================================================
# 3.  Single-profile training loop
# =======================================================================

def train_one_profile(
    *,
    step_fn,
    params             : dict,
    opt_state,
    profile            : dict,
    rho_flat_pix       : jnp.ndarray,   # (N_GRID²,)
    theta_flat         : jnp.ndarray,   # (N_GRID²,)
    rho_radial         : jnp.ndarray,   # (N_RADIAL,)
    active_mask        : jnp.ndarray,
    inject_noise,
    stages             : list                         = cfg.STAGES,
    curriculum         : Optional[CurriculumSchedule] = None,
    ep_global          : int                          = 0,
    best_data          : float                        = float("inf"),
    start_stage        : int                          = 0,
    start_ep_in_stage  : int                          = 0,
    hist               : Optional[dict]               = None,
    do_checkpoint_fn                                  = None,
    prof_idx           : int                          = 0,
    log_every          : int                          = cfg.LOG_EVERY,
    save_every         : int                          = cfg.SAVE_EVERY,
    t0                 : Optional[float]              = None,
    batch_size         : int                          = 16,
    n_harmonics        : int                          = cfg.N_HARMONICS,
    log_vars           : Optional[dict]               = None,   # v8.1
) -> Tuple[dict, Optional[dict], object, int, float]:
    """
    Train params through all noise stages for a single profile.

    Supports two sigma-scheduling modes:

    Legacy (stages list)
        Pass ``stages`` as a list of (n_steps, sigma) tuples.
        ``curriculum`` should be None.

    Curriculum mode
        Pass a ``CurriculumSchedule`` instance as ``curriculum``.
        The ``stages`` list is still used for its step counts per stage,
        but sigma is computed from the curriculum at each step instead of
        being fixed per stage.

    Physics warmup mode [v8.2]
        Pass a ``CurriculumSchedule(mode='physics_warmup', ...)`` instance.
        The curriculum returns (sigma, w_pde_scale) tuples.  This function
        detects the tuple and builds a per-step LossWeights via
        ._replace(w_pde=base_w_pde * w_pde_scale) before passing it to
        step_fn.  No retracing occurs: only the Python-side weight scalar
        changes; the compiled XLA graph is reused.

    Parameters
    ----------
    step_fn      : step function from make_train_step() — Python wrapper
                   around the inner @jax.jit (see make_train_step docs).
    params       : current param tree
    opt_state    : current optimiser state
    profile      : one dict from data_loader.load_profiles().
                   Must contain keys: "g_ideal", "xi".
    rho_flat_pix : (N_GRID²,)  pixel-grid elliptic radius
    theta_flat   : (N_GRID²,)  pixel-grid poloidal angle
    rho_radial   : (N_RADIAL,) model radial axis
    active_mask  : (128,)       float32, 1 for active chords
    inject_noise : data_loader.inject_noise  (passed for API compat;
                   noise now runs inside step_fn via inject_noise_fn closure)
    stages       : list of (n_steps, sigma) tuples
    curriculum   : CurriculumSchedule | None
    ep_global    : global epoch counter at entry
    best_data    : best projection loss seen so far
    start_stage  : stage to resume from (0 = fresh)
    start_ep_in_stage : epoch within that stage to resume from
    hist         : dict to accumulate loss history (mutated in place)
    do_checkpoint_fn : callable(ep_global, prof_idx, stage_idx, ep,
                               best_data, cur_params, cur_opt_state)
    prof_idx     : index of this profile (also used as skip gate cache key)
    log_every    : print + host-sync frequency (steps)
    save_every   : checkpoint frequency (steps)
    t0           : wall-clock start time
    batch_size   : B — number of samples per step (default 16)
    log_vars     : dict of JAX scalar arrays from init_log_vars(), or None.

    Returns
    -------
    params, log_vars, opt_state, ep_global, best_data
    """
    if hist is None:
        hist = {}
    if t0 is None:
        t0 = time.time()

    _gpu = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices()[0]

    # ── Place static profile arrays on GPU once, outside the inner loop ──
    # g_clean and xi are tiled to (B, ...) so vmap inside step_fn
    # sees a full batch.  Tiling is done here in Python (once), not
    # inside the JIT closure, so it doesn't re-trace.
    g_single      = jax.device_put(profile["g_ideal"], _gpu)          # (128,)
    xi_single     = jax.device_put(profile["xi"],      _gpu)          # (9,)
    psi_single    = jax.device_put(profile["psi_n"],   _gpu)          # (N_GRID²,)
    rho_eq_single = jax.device_put(profile["bpol_n"],  _gpu)          # (N_GRID²,) per-profile B_pol
    # FIX (v8.1): was rho_flat_pix — the same geometry rho for every profile,
    # meaning the EquilibriumEncoder received identical input regardless of
    # which equilibrium was being reconstructed.  bpol_n is the interpolated
    # B_pol field on the pixel grid, which actually varies per profile.

    g_clean      = jnp.stack([g_single]      * batch_size, axis=0)    # (B, 128)
    xi_batch     = jnp.stack([xi_single]     * batch_size, axis=0)    # (B, 9)
    psi_batch    = jnp.stack([psi_single]    * batch_size, axis=0)    # (B, N_GRID²)
    rho_eq_batch = jnp.stack([rho_eq_single] * batch_size, axis=0)    # (B, N_GRID²)

    rho_flat_pix = jax.device_put(rho_flat_pix, _gpu)
    theta_flat   = jax.device_put(theta_flat,   _gpu)
    rho_radial   = jax.device_put(rho_radial,   _gpu)
    active_mask  = jax.device_put(active_mask,  _gpu)

    # [v8.2] Detect physics_warmup mode from curriculum.
    # In this mode the curriculum returns (sigma, w_pde_scale) tuples and
    # we need to construct per-step LossWeights accordingly.
    physics_warmup_mode = (
        curriculum is not None
        and getattr(curriculum, "mode", None) == "physics_warmup"
    )

    # [v8.2] Cache the base w_pde from DEFAULT_WEIGHTS so we can scale it
    # per-step without mutating the shared weights object.
    base_w_pde: float = float(getattr(DEFAULT_WEIGHTS, "w_pde", 1.0))

    # Timing state for steps/sec diagnostics
    _t_last_log  = time.time()
    _ep_last_log = ep_global

    for si, (s_ep, s_sig) in enumerate(stages):

        if si < start_stage:
            continue

        def _sigma(step: int):
            """
            Return sigma (or (sigma, w_pde_scale) in physics_warmup mode).

            Wraps the curriculum call or returns the stage sigma directly.
            The result type is intentionally polymorphic: a plain float for
            standard modes, a (float, float) tuple for physics_warmup.
            Callers must check physics_warmup_mode to decide how to unpack.
            """
            return curriculum(step) if curriculum is not None else s_sig

        ep_start = start_ep_in_stage if si == start_stage else 0
        if si == start_stage:
            start_ep_in_stage = 0

        print(
            f"  Prof {prof_idx + 1} | stage {si + 1}/{len(stages)} "
            f"sigma={'curriculum' if curriculum else s_sig}  "
            f"ep_range=[{ep_start},{s_ep})  batch={batch_size}"
        )

        for ep in range(ep_start, s_ep):

            # ── PRNGKey on GPU ──────────────────────────────────────────
            # A CPU key would force inject_noise (now inside JIT) to run
            # on CPU, defeating the whole point of moving it inside.
            rng = jax.device_put(
                jax.random.PRNGKey(ep_global + ep + si * 10_000),
                _gpu,
            )

            # [v8.2] Unpack (sigma, w_pde_scale) tuple in physics_warmup
            # mode; plain sigma otherwise.  w_pde_scale ∈ [0, 1].
            sched_out   = _sigma(ep_global)
            if physics_warmup_mode:
                sigma_py, w_pde_scale = sched_out           # (float, float)
            else:
                sigma_py   = sched_out                       # float
                w_pde_scale = 1.0                            # neutral scale

            sigma_val = jnp.array(sigma_py, dtype=jnp.float32)

            # ── Single JIT call covers noise + forward + loss + update ──
            # step_fn returns 5 values in v8.1/v8.2:
            # (new_model_params, new_log_vars_or_none, new_opt_state, total, ld)
            # [v8.2] Pass prof_idx as the skip gate cache key so monitoring
            # data is keyed per-profile (useful in multi-profile debug mode).
            params, log_vars, opt_state, total, ld = step_fn(
                params, log_vars,
                opt_state,
                g_clean, psi_batch, rho_eq_batch,
                sigma_val, rng,
                xi_batch, active_mask,
                rho_flat_pix, theta_flat, rho_radial,
                _gate_cache_key=prof_idx,   # [v8.2] skip gate cache key
            )
            ep_global += 1

            # ── Host sync only at log intervals ─────────────────────────
            # float(total) is a device→host transfer.  Calling it every
            # step serialises the CPU and GPU.  We defer it to log_every
            # intervals so the GPU runs ahead freely between log points.
            if ep_global % log_every == 0:
                metrics   = jax.device_get({"total": total, **ld})
                total_f   = float(metrics["total"])
                proj_f    = float(metrics.get("proj",   0))
                smth_f    = float(metrics.get("smooth", 0))

                # steps/sec over the last log window
                _t_now       = time.time()
                _steps_done  = ep_global - _ep_last_log
                steps_per_s  = _steps_done / max(_t_now - _t_last_log, 1e-6)
                _t_last_log  = _t_now
                _ep_last_log = ep_global

                if np.isfinite(total_f):
                    _append(hist, "total", total_f)
                    for k, v in metrics.items():
                        if k != "total":
                            _append(hist, k, float(v))

                if proj_f < best_data:
                    best_data = proj_f

                elapsed = time.time() - t0
                pol_f   = float(metrics.get("pol", 0))
                pde_f   = float(metrics.get("pde", 0))
                bnd_f   = float(metrics.get("boundary", 0))

                # Adaptive-weight summary: show effective sigmas if present
                sigma_str = ""
                if log_vars is not None:
                    lv_host = jax.device_get(log_vars)
                    sigma_str = "  σ[" + " ".join(
                        f"{k}={float(v):.3f}"
                        for k, v in lv_host.items()
                    ) + "]"

                # [v8.2] Skip gate summary ─────────────────────────────
                # Read from the module-level cache populated by step_fn
                # wrapper.  If no gates recorded, omit gracefully.
                gate_str = ""
                gate_vals = get_skip_gate_cache(prof_idx)
                if gate_vals is not None:
                    try:
                        # Flatten to 1-D regardless of whether model returns
                        # an array or a dict of arrays
                        if isinstance(gate_vals, dict):
                            flat_gates = jnp.concatenate(
                                [jnp.ravel(v) for v in gate_vals.values()]
                            )
                        else:
                            flat_gates = jnp.ravel(gate_vals)
                        gate_min = float(jnp.min(flat_gates))
                        gate_max = float(jnp.max(flat_gates))
                        # [v8.2] Single compact gate summary line as specified.
                        # Near-0 min  → some layers always skip (bypass active).
                        # Near-1 max  → all layers stay in the residual path.
                        gate_str = f"  skip_gates=[lo={gate_min:.3f} hi={gate_max:.3f}]"
                    except Exception:
                        pass   # silently ignore if gate shape is unexpected

                # [v8.2] Effective PDE weight after curriculum scaling ──
                # w_pde_eff = base_w_pde (from DEFAULT_WEIGHTS) * w_pde_scale.
                # During physics_warmup Phase 1 this ramps from 0 → base_w_pde.
                w_pde_eff = base_w_pde * w_pde_scale

                # [v8.2] Lerp gradient norm ────────────────────────────
                # loss_fn is expected to return "lerp_gnorm" in the loss_dict
                # when lerp geometry is active (i.e. lerp_frac was passed).
                # This is ‖∂L/∂coeffs[:,0]‖ and confirms gradients flow
                # through the new differentiable interpolation path.
                lerp_gnorm_str = ""
                if "lerp_gnorm" in metrics:
                    lerp_gnorm_str = f"  lerp_gnorm={float(metrics['lerp_gnorm']):.3e}"

                print(
                    f"    ep={ep_global:6d} | "
                    f"loss={total_f:.5f} | "
                    f"proj={proj_f:.5f} | "
                    f"bnd={bnd_f:.5f} | "
                    f"pde={pde_f:.5f} | "
                    f"smooth={smth_f:.5f} | "
                    f"pol={pol_f:.5f} | "
                    f"batch={batch_size} | "
                    f"steps/s={steps_per_s:.1f} | "
                    f"t={elapsed:.0f}s | "
                    # [v8.2] New log fields below ────────────────────────
                    f"w_pde_eff={w_pde_eff:.5f}"   # effective PDE weight
                    f"{gate_str}"                   # skip gate min/max
                    f"{lerp_gnorm_str}"             # lerp gradient norm
                    f"{sigma_str}"                  # adaptive σ (v8.1)
                )

            if ep_global % save_every == 0 and do_checkpoint_fn is not None:
                do_checkpoint_fn(ep_global, prof_idx, si, ep + 1, best_data,
                                 params, opt_state)

    return params, log_vars, opt_state, ep_global, best_data


# =======================================================================
# 4.  Full training loop  (all profiles x all stages)
# =======================================================================

def train(
    *,
    step_fn,
    params          : dict,
    opt_state,
    profiles        : list,
    rho_flat_pix    : jnp.ndarray,   # (N_GRID²,)
    theta_flat      : jnp.ndarray,   # (N_GRID²,)
    rho_radial      : jnp.ndarray,   # (N_RADIAL,)
    active_mask     : jnp.ndarray,
    inject_noise,
    stages          : list                         = cfg.STAGES,
    curriculum      : Optional[CurriculumSchedule] = None,
    ep_global       : int                          = 0,
    best_data       : float                        = float("inf"),
    start_prof      : int                          = 0,
    start_stage     : int                          = 0,
    start_ep        : int                          = 0,
    hist            : Optional[dict]               = None,
    do_checkpoint_fn                               = None,
    log_every       : int                          = cfg.LOG_EVERY,
    save_every      : int                          = cfg.SAVE_EVERY,
    single_profile  : bool                         = False,
    batch_size      : int                          = 16,
    n_harmonics     : int                          = cfg.N_HARMONICS,
    log_vars        : Optional[dict]               = None,   # v8.1
) -> Tuple[dict, object, dict, float]:
    """
    Outer training loop: iterate over all profiles and all noise stages.

    v8.2: supports physics_warmup curriculum mode (see CurriculumSchedule).
    v8.1: pass log_vars=losses.init_log_vars() to enable adaptive weighting.
    The log_vars dict is threaded through every profile and returned at the end.

    Parameters
    ----------
    step_fn      : jit-compiled step from make_train_step().
    params       : initial / resumed model param tree
    opt_state    : initial / resumed optimiser state
    profiles     : list of profile dicts from data_loader.load_profiles().
    rho_flat_pix : (N_GRID²,)  pixel-grid elliptic radius
    theta_flat   : (N_GRID²,)  pixel-grid poloidal angle
    rho_radial   : (N_RADIAL,) model radial axis
    active_mask  : (128,)       float32, 1 for active chords
    inject_noise : data_loader.inject_noise
    stages       : list of (n_steps, sigma) tuples (default cfg.STAGES)
    curriculum   : CurriculumSchedule | None
    ep_global    : starting global epoch (0 for fresh, >0 for resume)
    best_data    : best projection loss so far (inf for fresh)
    start_prof   : profile index to resume from
    start_stage  : stage index to resume from within start_prof
    start_ep     : epoch within start_stage to resume from
    hist         : dict for loss history (created if None)
    do_checkpoint_fn : callable(ep_global, prof_idx, stage_idx, ep,
                               best_data, cur_params, cur_opt_state)
    log_every      : log print frequency (steps)
    save_every     : checkpoint save frequency (steps)
    single_profile : bool  — train on profiles[start_prof] only when True
    batch_size   : int  (default 16)
    n_harmonics  : int  (default cfg.N_HARMONICS)
    log_vars     : dict of JAX scalar arrays from init_log_vars(), or None.
                   When not None, dynamic loss balancing is active.

    Returns
    -------
    params, log_vars, opt_state, hist, best_data
    """
    if hist is None:
        hist = {}

    t0            = time.time()
    n_prof        = 1 if single_profile else len(profiles)
    n_stages      = len(stages)
    total_planned = sum(s for s, _ in stages) * n_prof
    mode_str      = "adaptive (lbPINN)" if log_vars is not None else "fixed-weight"

    # [v8.2] Show physics_warmup config if active
    pw_str = ""
    if curriculum is not None and getattr(curriculum, "mode", None) == "physics_warmup":
        pw_str = (
            f"\n  Physics warmup    : {curriculum.warmup_steps} steps  "
            f"w_pde_target={curriculum.w_pde_target}"
        )

    print(f"\n{'='*60}")
    if single_profile:
        # [v8.2] Updated version string
        print(f"VICTOR v8.2 -- Single-Profile Training  (profile index {start_prof})")
    else:
        print(f"VICTOR v8.2 -- Training  ({n_prof} profiles x {n_stages} stages)")
    print(f"  Total planned steps : {total_planned:,}")
    print(f"  Batch size          : {batch_size}")
    print(f"  Loss mode           : {mode_str}")
    print(f"  Resume: prof={start_prof}  stage={start_stage}  ep={ep_global}")
    if curriculum is not None:
        print(f"  Curriculum: {curriculum}{pw_str}")
    print(f"{'='*60}\n")

    _start_stage = start_stage
    _start_ep    = start_ep

    prof_end = start_prof + n_prof
    for pi in range(start_prof, prof_end):

        params, log_vars, opt_state, ep_global, best_data = train_one_profile(
            step_fn            = step_fn,
            params             = params,
            opt_state          = opt_state,
            profile            = profiles[pi],
            rho_flat_pix       = rho_flat_pix,
            theta_flat         = theta_flat,
            rho_radial         = rho_radial,
            active_mask        = active_mask,
            inject_noise       = inject_noise,
            stages             = stages,
            curriculum         = curriculum,
            ep_global          = ep_global,
            best_data          = best_data,
            start_stage        = _start_stage,
            start_ep_in_stage  = _start_ep,
            hist               = hist,
            do_checkpoint_fn   = do_checkpoint_fn,
            prof_idx           = pi,
            log_every          = log_every,
            save_every         = save_every,
            t0                 = t0,
            batch_size         = batch_size,
            n_harmonics        = n_harmonics,
            log_vars           = log_vars,
        )

        _start_stage = 0
        _start_ep    = 0

        if do_checkpoint_fn is not None:
            do_checkpoint_fn(ep_global, pi + 1, 0, 0, best_data,
                             params, opt_state)

        print(f"  Profile {pi + 1}/{len(profiles)} complete  "
              f"ep_global={ep_global}  best_proj={best_data:.6f}  "
              f"{'[single-profile mode]' if single_profile else ''}\n")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Training complete: {ep_global} steps  "
          f"best_proj={best_data:.6f}  t={elapsed:.1f}s")
    print(f"{'='*60}")

    return params, log_vars, opt_state, hist, best_data


# =======================================================================
# 5.  Internal helpers
# =======================================================================

def _append(hist: dict, key: str, value: float) -> None:
    """Append a scalar to a history list, creating the list if needed."""
    if key not in hist:
        hist[key] = []
    hist[key].append(value)


if __name__ == "__main__":
    print("trainer.py — no self-test (requires full pipeline).")
    print("Run Cell 5 in the notebook to exercise the training loop.")
