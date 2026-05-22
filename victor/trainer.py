# ============================================================
# VICTOR v7.0 — trainer.py
# Closure-based JIT/pmap training step, training loop, stage logic
# ============================================================
# Public API
# ----------
#   build_optimizer(total_steps, lr, warmup, weight_decay, clip_norm)
#                                          -> tx (optax chain, AdamW)
#   make_train_step(model, w_ops, weights, tx)
#                                          -> (step_fn, n_dev)
#   replicate_params(params, opt_state, n_dev) -> (params_rep, opt_rep)
#   unreplicate_params(params, opt_state, n_dev) -> (params, opt_state)
#   CurriculumSchedule                     — noise/sigma curriculum helper
#   train_one_profile(step_fn, ..., n_dev) -> (params, opt_state, ep, best)
#   train(step_fn, ..., n_dev)             -> (params, opt_state, hist, best)
#
# v7.2 GPU placement fixes
# ------------------------
#  * make_train_step now returns (step_fn, n_dev).
#    - n_dev == 1 : jax.jit path (unchanged behaviour).
#    - n_dev >= 2 : jax.pmap path — runs on all visible GPUs in parallel.
#      Gradients are averaged across devices via lax.pmean before the
#      optimiser update.  Both T4s on Kaggle will be fully utilised.
#
#  * replicate_params / unreplicate_params helpers:
#    - replicate_params: stacks params/opt_state along a leading device
#      axis for pmap.  No-op for n_dev == 1.
#    - unreplicate_params: strips that axis before checkpointing or eval.
#
#  * _shard(arr, n_dev): adds a leading device axis to per-step inputs
#    (g_stage, xi, active_mask, rho_flat) for pmap dispatch.
#    No-op for n_dev == 1.
#
#  * PRNGKey device placement: jax.random.PRNGKey() creates a CPU key
#    by default.  All inject_noise call sites now device_put the key
#    onto _gpu before passing it.  A CPU key forces inject_noise to
#    dispatch entirely on CPU even when g_clean is on GPU.
#
#  * train_one_profile and train both accept n_dev and thread it
#    through to _shard and scalar extraction (pmap outputs have a
#    leading device axis; device 0 is taken for logging).
#
# v7.1 additions (unchanged)
# --------------------------
#  * AdamW + configurable clip_norm + cosine-warmup LR schedule.
#  * CurriculumSchedule (linear / cosine / step sigma annealing).
#  * donate_argnums=(0,1) on jit/pmap for buffer reuse.
#  * float() calls only at log_every intervals (not every step).
#  * All per-step inputs device_put onto _gpu before the step call.
# ============================================================

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from victor import config as cfg
from victor.losses import loss_fn, LossWeights, DEFAULT_WEIGHTS


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

    Three annealing modes
    ---------------------
    "linear"  : sigma decreases linearly from sigma_start to sigma_end
                over total_steps.  Simple and predictable.

    "cosine"  : sigma follows a cosine decay (fast early, slow late).
                Mirrors the cosine LR schedule — keeps noise high while
                the model is unstable, then quickly reaches low noise.

    "step"    : sigma drops in discrete steps, one per entry in
                sigma_steps = [(step_boundary, sigma_value), ...].
                Equivalent to the original STAGES list but expressed
                as absolute step counts rather than per-stage lengths.

    Parameters
    ----------
    total_steps  : int    Total training steps across all profiles/stages.
    sigma_start  : float  Initial noise fraction (default 0.01).
    sigma_end    : float  Final noise fraction (default 0.001).
    mode         : str    "linear" | "cosine" | "step"
    sigma_steps  : list of (int, float)
                          Used only for mode="step".
                          List of (step_boundary, sigma) pairs, sorted
                          by step_boundary ascending.

    Usage
    -----
    curriculum = CurriculumSchedule(total_steps=100_000, mode="cosine")
    sigma = curriculum(ep_global)   # call with current step number

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

    def __init__(
        self,
        total_steps  : int,
        sigma_start  : float             = 0.01,
        sigma_end    : float             = 0.001,
        mode         : str               = "cosine",
        sigma_steps  : list              = None,
    ):
        if mode not in ("linear", "cosine", "step"):
            raise ValueError(f"CurriculumSchedule: unknown mode '{mode}'. "
                             "Choose 'linear', 'cosine', or 'step'.")
        if mode == "step" and not sigma_steps:
            raise ValueError("CurriculumSchedule: mode='step' requires sigma_steps.")

        self.total_steps = max(total_steps, 1)
        self.sigma_start = sigma_start
        self.sigma_end   = sigma_end
        self.mode        = mode
        # Sort step boundaries ascending
        self.sigma_steps = sorted(sigma_steps or [], key=lambda x: x[0])

    def __call__(self, step: int) -> float:
        """
        Return the noise sigma for the given global step.

        Parameters
        ----------
        step : int   Current global training step (0-indexed).

        Returns
        -------
        float   Noise sigma fraction.
        """
        t = min(step, self.total_steps) / self.total_steps   # progress ∈ [0,1]

        if self.mode == "linear":
            return float(self.sigma_start + (self.sigma_end - self.sigma_start) * t)

        elif self.mode == "cosine":
            import math
            cosine_t = 0.5 * (1.0 + math.cos(math.pi * t))   # 1→0
            return float(self.sigma_end + (self.sigma_start - self.sigma_end) * cosine_t)

        else:  # "step"
            sigma = self.sigma_start
            for boundary, sig in self.sigma_steps:
                if step >= boundary:
                    sigma = sig
            return float(sigma)

    def __repr__(self) -> str:
        return (f"CurriculumSchedule(mode={self.mode!r}, "
                f"sigma={self.sigma_start}→{self.sigma_end}, "
                f"steps={self.total_steps})")

def build_optimizer(
    total_steps  : int,
    lr           : float = cfg.LR,
    warmup       : int   = 500,
    lr_end       : float = 5e-5,
    weight_decay : float = 1e-4,
    clip_norm    : float = 1.0,
) -> optax.GradientTransformation:
    """
    Build the VICTOR v7 optimiser: warmup-cosine-decay AdamW + grad-clip.

    Replaces plain Adam with AdamW (Adam + decoupled weight decay).
    Weight decay acts as an L2 regulariser on parameters independently
    of the adaptive learning rate, which prevents large-norm solutions
    and improves generalisation on small datasets like TORAX profiles.

    Schedule
    --------
    0 -> warmup steps : linear warm-up from 0 to lr
    warmup -> total   : cosine decay from lr to lr_end

    Parameters
    ----------
    total_steps  : int   Total number of gradient steps expected.
    lr           : float Peak learning rate (default cfg.LR = 3e-4).
    warmup       : int   Number of warm-up steps (default 500).
    lr_end       : float Final learning rate at end of cosine decay.
    weight_decay : float AdamW decoupled weight decay (default 1e-4).
                         Set to 0.0 to recover plain Adam behaviour.
    clip_norm    : float Global gradient norm clip threshold (default 1.0).

    Returns
    -------
    optax.GradientTransformation   (chain of clip + adamw)
    """
    sched = optax.warmup_cosine_decay_schedule(
        init_value       = 0.0,
        peak_value       = lr,
        warmup_steps     = warmup,
        decay_steps      = total_steps,
        end_value        = lr_end,
    )
    return optax.chain(
        optax.clip_by_global_norm(clip_norm),   # already present; clip_norm now configurable
        optax.adamw(sched, weight_decay=weight_decay),
    )


# =======================================================================
# 2.  JIT-compiled step factory  (sole training-step entry point)
# =======================================================================

def make_train_step(
    model,
    w_ops,
    weights : LossWeights = DEFAULT_WEIGHTS,
    tx      = None,
):
    """
    Factory that returns the compiled training step for VICTOR.

    Single-GPU (1 T4): returns a jax.jit step — identical to v7.0.
    Multi-GPU (2 T4):  returns a jax.pmap step that runs a full
                       forward+backward pass on each device in parallel.
                       Params and opt_state must be replicated across
                       devices before the first call (use replicate_params
                       defined below).  Gradients are averaged across
                       devices via lax.pmean before the optimiser update.

    All Python objects (tx, model, w_ops, weights) are closed over here
    so they never cross the JIT/pmap boundary as traced values.

    Parameters
    ----------
    model   : FourierDeepONet instance
    w_ops   : WOperators  (geometry.WOperators)
    weights : LossWeights  (default DEFAULT_WEIGHTS)
    tx      : optax GradientTransformation — required.

    Returns
    -------
    step_fn  : compiled callable
    n_dev    : int  number of devices used (1 or 2)
    """
    if tx is None:
        raise ValueError(
            "make_train_step: `tx` must be supplied to the factory."
        )

    n_dev = jax.device_count()

    def _step_raw(
        params,
        opt_state,
        g_noisy     : jnp.ndarray,   # (n_chords,)      or (n_dev, n_chords)
        xi          : jnp.ndarray,   # (9,)             or (n_dev, 9)
        active_mask : jnp.ndarray,   # (128,)           or (n_dev, 128)
        rho_flat    : jnp.ndarray,   # (n_radial,)      or (n_dev, n_radial)
    ) -> Tuple[dict, object, jnp.ndarray, Dict[str, jnp.ndarray]]:

        def _loss(p):
            eps1d = model.apply(p, g_noisy, xi)
            return loss_fn(eps1d, g_noisy, w_ops, active_mask, rho_flat, weights)

        (total, ld), grads = jax.value_and_grad(_loss, has_aux=True)(params)

        # Zero NaN / Inf gradients
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g)),
            grads,
        )

        # Average gradients across devices when using pmap
        if n_dev > 1:
            grads = jax.lax.pmean(grads, axis_name="devices")
            total = jax.lax.pmean(total, axis_name="devices")
            ld    = jax.tree_util.tree_map(
                lambda v: jax.lax.pmean(v, axis_name="devices"), ld
            )

        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, total, ld

    if n_dev > 1:
        # pmap: each device runs a full step on its own data shard.
        # donate_argnums donates params (0) and opt_state (1).
        _step = jax.pmap(
            _step_raw,
            axis_name    = "devices",
            donate_argnums = (0, 1),
        )
        print(f"make_train_step: using pmap across {n_dev} GPUs.")
    else:
        # jit: single-GPU path, unchanged from v7.0.
        _step = jax.jit(_step_raw, donate_argnums=(0, 1))
        print("make_train_step: using jit on single GPU.")

    return _step, n_dev


# =======================================================================
# 3.  pmap helpers  (no-ops when n_dev == 1)
# =======================================================================

def replicate_params(params, opt_state, n_dev: int):
    """
    Replicate params and opt_state across `n_dev` devices for pmap.

    When n_dev == 1 (jit path) returns the inputs unchanged.

    Call this once after model.init() and before the training loop:
        step_fn, n_dev = make_train_step(...)
        params, opt_state = replicate_params(params, opt_state, n_dev)

    Parameters
    ----------
    params    : Flax param pytree
    opt_state : optax state pytree
    n_dev     : int  from make_train_step return value

    Returns
    -------
    params_rep, opt_state_rep  — stacked along a leading device axis
    """
    if n_dev == 1:
        return params, opt_state
    stack = lambda x: jnp.stack([x] * n_dev)
    return (
        jax.tree_util.tree_map(stack, params),
        jax.tree_util.tree_map(stack, opt_state),
    )


def unreplicate_params(params, opt_state, n_dev: int):
    """
    Remove the leading device axis added by replicate_params.

    Call this before saving a checkpoint or running evaluation:
        params_cpu, opt_state_cpu = unreplicate_params(params, opt_state, n_dev)

    When n_dev == 1 returns the inputs unchanged.
    """
    if n_dev == 1:
        return params, opt_state
    take0 = lambda x: x[0]
    return (
        jax.tree_util.tree_map(take0, params),
        jax.tree_util.tree_map(take0, opt_state),
    )


def _shard(arr: jnp.ndarray, n_dev: int) -> jnp.ndarray:
    """
    Add a leading device axis by stacking `arr` n_dev times.

    For pmap, every device receives the same sinogram / mask / rho
    (identical data, replicated weights — standard data-parallel setup
    for single-sample-per-step training).

    When n_dev == 1 returns arr unchanged.
    """
    if n_dev == 1:
        return arr
    return jnp.stack([arr] * n_dev)


# =======================================================================
# 4.  Single-profile training loop  (one profile x all stages)
# =======================================================================

def train_one_profile(
    *,
    step_fn,
    params            : dict,
    opt_state,
    profile           : dict,
    rho_flat          : jnp.ndarray,
    active_mask       : jnp.ndarray,
    inject_noise,
    stages            : list                         = cfg.STAGES,
    curriculum        : Optional[CurriculumSchedule] = None,
    n_dev             : int                          = 1,
    ep_global         : int                          = 0,
    best_data         : float                        = float("inf"),
    start_stage       : int                          = 0,
    start_ep_in_stage : int                          = 0,
    hist              : Optional[dict]               = None,
    do_checkpoint_fn                                 = None,
    prof_idx          : int                          = 0,
    log_every         : int                          = cfg.LOG_EVERY,
    save_every        : int                          = cfg.SAVE_EVERY,
    t0                : Optional[float]              = None,
) -> Tuple[dict, object, int, float]:
    """
    Train params through all noise stages for a single profile.

    Supports two sigma-scheduling modes:

    Legacy (stages list)
        Pass ``stages`` as a list of (n_steps, sigma) tuples exactly
        as before.  ``curriculum`` should be None.

    Curriculum mode
        Pass a ``CurriculumSchedule`` instance as ``curriculum``.
        The ``stages`` list is still used for its total step counts per
        stage, but sigma is computed from the curriculum at each step
        instead of being fixed per stage.  This gives a smooth noise
        anneal rather than discrete jumps.

    Parameters
    ----------
    step_fn      : jit-compiled step function from make_train_step().
    params       : current param tree
    opt_state    : current optimiser state
    profile      : one dict from data_loader.load_profiles().
                   Must contain keys: "g_ideal", "xi".
    rho_flat     : (n_radial,)  normalised elliptic radius per radial point
    active_mask  : (128,)       float32, 1 for active chords
    inject_noise : data_loader.inject_noise
    stages       : list of (n_steps, sigma) — step counts (sigma used only
                   when curriculum=None)
    curriculum   : CurriculumSchedule | None
                   When supplied, overrides per-stage sigma with a
                   continuous schedule keyed on ep_global.
    ep_global    : global epoch counter at entry
    best_data    : best projection loss seen so far
    start_stage  : stage to resume from (0 = fresh)
    start_ep_in_stage : epoch within that stage to resume from
    hist         : dict to accumulate loss history (mutated in place)
    do_checkpoint_fn : callable(ep_global, prof_idx, stage_idx, ep, best_data)
    prof_idx     : index of this profile
    log_every    : print frequency
    save_every   : checkpoint frequency
    t0           : wall-clock start time

    Returns
    -------
    params, opt_state, ep_global, best_data
    """
    if hist is None:
        hist = {}
    if t0 is None:
        t0 = time.time()

    # Place static profile arrays on GPU once, outside the inner loop
    _gpu = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices()[0]
    g_clean = jax.device_put(profile["g_ideal"], _gpu)   # (n_chords,)
    xi      = jax.device_put(profile["xi"],      _gpu)   # (9,)

    for si, (s_ep, s_sig) in enumerate(stages):

        # -- Resume: skip completed stages for this profile ---------------
        if si < start_stage:
            continue

        # Place per-stage arrays on GPU
        rho_flat    = jax.device_put(rho_flat,    _gpu)
        active_mask = jax.device_put(active_mask, _gpu)

        # Sigma: curriculum schedule overrides fixed per-stage sigma
        def _sigma(step: int) -> float:
            return curriculum(step) if curriculum is not None else s_sig

        # Noise seed: mix global epoch + stage index
        # PRNGKey must be on GPU — a CPU key forces inject_noise to run on CPU
        _key    = jax.device_put(jax.random.PRNGKey(ep_global + si), _gpu)
        g_stage = inject_noise(g_clean, _sigma(ep_global), _key)
        g_stage = jax.device_put(g_stage, _gpu)

        # Epoch range for this stage (support resume mid-stage)
        ep_start = start_ep_in_stage if si == start_stage else 0
        # After consuming the resume offset, reset to 0 for all later stages
        if si == start_stage:
            start_ep_in_stage = 0

        print(
            f"  Prof {prof_idx + 1} | stage {si + 1}/{len(stages)} "
            f"sigma={'curriculum' if curriculum else s_sig}  "
            f"ep_range=[{ep_start},{s_ep})"
        )

        for ep in range(ep_start, s_ep):

            # Re-draw noise periodically to avoid memorising a fixed sample
            if ep % log_every == 0 and ep > ep_start:
                _key    = jax.device_put(jax.random.PRNGKey(ep_global + ep), _gpu)
                g_stage = inject_noise(g_clean, _sigma(ep_global), _key)
                g_stage = jax.device_put(g_stage, _gpu)

            # -- Gradient step --------------------------------------------
            # For pmap: add leading device axis to all per-step inputs.
            # For jit (n_dev==1): _shard is a no-op.
            params, opt_state, total, ld = step_fn(
                params, opt_state,
                _shard(g_stage,     n_dev),
                _shard(xi,          n_dev),
                _shard(active_mask, n_dev),
                _shard(rho_flat,    n_dev),
            )
            ep_global += 1

            # -- Logging + history (materialise to CPU only here) ---------
            # For pmap outputs, take device 0's scalar (all devices agree
            # after pmean so any index is equivalent).
            if ep_global % log_every == 0:
                total_s = total[0] if n_dev > 1 else total
                ld_s    = {k: v[0] if n_dev > 1 else v for k, v in ld.items()}
                total_f = float(total_s)
                proj_f  = float(ld_s.get("proj",   0))
                smth_f  = float(ld_s.get("smooth", 0))

                if np.isfinite(total_f):
                    _append(hist, "total", total_f)
                    for k, v in ld_s.items():
                        _append(hist, k, float(v))
                    if proj_f < best_data:
                        best_data = proj_f

                elapsed = time.time() - t0
                print(
                    f"    ep={ep_global:6d} | L={total_f:.5f} | "
                    f"proj={proj_f:.5f} | smooth={smth_f:.5f} | "
                    f"t={elapsed:.0f}s"
                )

            # -- Checkpoint -----------------------------------------------
            if ep_global % save_every == 0 and do_checkpoint_fn is not None:
                do_checkpoint_fn(ep_global, prof_idx, si, ep + 1, best_data)

    return params, opt_state, ep_global, best_data


# =======================================================================
# 4.  Full training loop  (all profiles x all stages x N_EPOCHS)
# =======================================================================

def train(
    *,
    step_fn,
    params       : dict,
    opt_state,
    profiles     : list,
    rho_flat     : jnp.ndarray,
    active_mask  : jnp.ndarray,
    inject_noise,
    stages       : list                         = cfg.STAGES,
    curriculum   : Optional[CurriculumSchedule] = None,
    n_dev        : int                          = 1,
    ep_global    : int                          = 0,
    best_data    : float                        = float("inf"),
    start_prof   : int                          = 0,
    start_stage  : int                          = 0,
    start_ep     : int                          = 0,
    hist         : Optional[dict]               = None,
    do_checkpoint_fn                            = None,
    log_every    : int                          = cfg.LOG_EVERY,
    save_every   : int                          = cfg.SAVE_EVERY,
) -> Tuple[dict, object, dict, float]:
    """
    Outer training loop: iterate over all profiles and all noise stages.

    Parameters
    ----------
    step_fn      : jit-compiled step from make_train_step().
    params       : initial / resumed param tree
    opt_state    : initial / resumed optimiser state
    profiles     : list of profile dicts from data_loader.load_profiles().
    rho_flat     : (n_radial,)  normalised elliptic radius per radial point
    active_mask  : (128,)       float32, 1 for active chords
    inject_noise : data_loader.inject_noise
    stages       : list of (n_steps, sigma) tuples (default cfg.STAGES)
    curriculum   : CurriculumSchedule | None
                   When provided, sigma is taken from the schedule at each
                   step.  Overrides per-stage sigma values from ``stages``.
                   The step counts from ``stages`` are still respected.
    ep_global    : starting global epoch (0 for fresh, >0 for resume)
    best_data    : best projection loss so far (inf for fresh)
    start_prof   : profile index to resume from
    start_stage  : stage index to resume from within start_prof
    start_ep     : epoch within start_stage to resume from
    hist         : dict for loss history (created if None)
    do_checkpoint_fn : callable(ep_global, prof_idx, stage_idx, ep, best_data)
    log_every    : log print frequency (steps)
    save_every   : checkpoint save frequency (steps)

    Returns
    -------
    params, opt_state, hist, best_data
    """
    if hist is None:
        hist = {}

    t0       = time.time()
    n_prof   = len(profiles)
    n_stages = len(stages)
    total_planned = sum(s for s, _ in stages) * n_prof
    print(f"\n{'='*60}")
    print(f"VICTOR v7.0 -- Training  ({n_prof} profiles x {n_stages} stages)")
    print(f"  Total planned steps : {total_planned:,}")
    print(f"  Resume: prof={start_prof}  stage={start_stage}  ep={ep_global}")
    if curriculum is not None:
        print(f"  Curriculum: {curriculum}")
    print(f"{'='*60}\n")

    _start_stage = start_stage
    _start_ep    = start_ep

    for pi in range(start_prof, n_prof):

        params, opt_state, ep_global, best_data = train_one_profile(
            step_fn           = step_fn,
            params            = params,
            opt_state         = opt_state,
            profile           = profiles[pi],
            rho_flat          = rho_flat,
            active_mask       = active_mask,
            inject_noise      = inject_noise,
            stages            = stages,
            curriculum        = curriculum,
            n_dev             = n_dev,
            ep_global         = ep_global,
            best_data         = best_data,
            start_stage       = _start_stage,
            start_ep_in_stage = _start_ep,
            hist              = hist,
            do_checkpoint_fn  = do_checkpoint_fn,
            prof_idx          = pi,
            log_every         = log_every,
            save_every        = save_every,
            t0                = t0,
        )

        # After first profile, resume offsets are consumed — start fresh
        _start_stage = 0
        _start_ep    = 0

        # Final checkpoint at end of each profile
        if do_checkpoint_fn is not None:
            do_checkpoint_fn(ep_global, pi + 1, 0, 0, best_data)

        print(f"  Profile {pi + 1}/{n_prof} complete  "
              f"ep_global={ep_global}  best_proj={best_data:.6f}\n")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Training complete: {ep_global} steps  "
          f"best_proj={best_data:.6f}  t={elapsed:.1f}s")
    print(f"{'='*60}")

    return params, opt_state, hist, best_data


# =======================================================================
# 5.  Internal helpers
# =======================================================================

def _append(hist: dict, key: str, value: float) -> None:
    """Append a scalar to a history list, creating the list if needed."""
    if key not in hist:
        hist[key] = []
    hist[key].append(value)


# -- Module self-test ----------------------------------------------------

if __name__ == "__main__":
    print("trainer.py — no self-test (requires full pipeline).")
    print("Run Cell 5 in the notebook to exercise the training loop.")
