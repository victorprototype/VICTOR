# ============================================================
# VICTOR v7.0 — trainer.py
# Closure-based JIT training step, training loop, stage logic
# ============================================================
# Public API
# ----------
#   build_optimizer(total_steps, lr, warmup, weight_decay, clip_norm)
#                                          -> tx (optax chain, AdamW)
#   make_train_step(model, w_ops, weights, tx, batch_size)
#                                          -> step_fn
#   CurriculumSchedule                     — noise/sigma curriculum helper
#   train_one_profile(step_fn, ...)        -> (params, opt_state, ep, best)
#   train(step_fn, ...)                    -> (params, opt_state, hist, best)
#
# v7.2 performance changes vs v7.1
# ---------------------------------
#  [1] Noise moved INSIDE the JIT step.
#      inject_noise() is now called inside _loss(), so noise generation,
#      forward pass, and grad computation fuse into one compiled XLA graph.
#      The CPU no longer blocks the GPU to sample noise every step.
#      step_fn signature changes:
#        OLD: (params, opt_state, g_noisy, xi, active_mask, rho_flat)
#        NEW: (params, opt_state, g_clean, sigma, rng, xi, active_mask, rho_flat)
#
#  [2] Batch support via jax.vmap.
#      step_fn now accepts g_clean: (B, 128) and xi: (B, 9).
#      Loss is averaged across the batch dimension.
#      B=1 is valid and preserves old behaviour exactly.
#      Batching amortises JAX dispatch overhead across B steps per call,
#      reducing the CPU↔GPU round-trips that caused high CPU utilisation.
#
#  [3] donate_argnums=(0, 1) on jax.jit.
#      params and opt_state buffers are donated to XLA, which reuses the
#      same GPU memory rather than allocating new buffers each step.
#      This eliminates the most common source of extra GPU allocations.
#
#  [4] Host sync only at log intervals.
#      float(total) (a device→host transfer) is called only every
#      log_every steps.  Between log points, losses stay on-device.
#
#  [5] Steps/sec diagnostics printed every log_every steps.
#
# v7.1 additions (preserved)
# --------------------------
#  * AdamW + configurable clip_norm + cosine-warmup LR schedule.
#  * CurriculumSchedule (linear / cosine / step sigma annealing).
#  * All per-step inputs device_put onto GPU before the step call.
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
        sigma_start  : float = 0.01,
        sigma_end    : float = 0.001,
        mode         : str   = "cosine",
        sigma_steps  : list  = None,
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
        self.sigma_steps = sorted(sigma_steps or [], key=lambda x: x[0])

    def __call__(self, step: int) -> float:
        t = min(step, self.total_steps) / self.total_steps   # progress ∈ [0,1]

        if self.mode == "linear":
            return float(self.sigma_start + (self.sigma_end - self.sigma_start) * t)

        elif self.mode == "cosine":
            import math
            cosine_t = 0.5 * (1.0 + math.cos(math.pi * t))
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
) -> optax.GradientTransformation:
    """
    Build the VICTOR v7 optimiser: warmup-cosine-decay AdamW + grad-clip.

    Schedule: linear warm-up from 0 → lr over `warmup` steps,
    then cosine decay from lr → lr_end over the remaining steps.

    Parameters
    ----------
    total_steps  : int   Total number of gradient steps expected.
    lr           : float Peak learning rate (default cfg.LR = 3e-4).
    warmup       : int   Number of warm-up steps (default 500).
    lr_end       : float Final learning rate at end of cosine decay.
    weight_decay : float AdamW decoupled weight decay (default 1e-4).
    clip_norm    : float Global gradient norm clip threshold (default 1.0).

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
        optax.adamw(sched, weight_decay=weight_decay),
    )


# =======================================================================
# 2.  JIT-compiled step factory
# =======================================================================

def make_train_step(
    model,
    w_ops,
    weights    : LossWeights = DEFAULT_WEIGHTS,
    tx         = None,
    batch_size : int         = 16,
    inject_noise_fn          = None,
):
    """
    Factory that returns a jax.jit-compiled training step for VICTOR.

    All Python objects (tx, model, w_ops, weights) are closed over so
    they never cross the JIT boundary as traced values.

    Performance design
    ------------------
    [noise inside JIT]
        inject_noise_fn is called inside _loss_one() so noise sampling,
        the forward pass, and gradient computation all fuse into one XLA
        graph.  The CPU no longer has to call inject_noise and then
        dispatch step_fn separately every iteration.

    [vmap batching]
        The per-sample loss is defined for a single (g, xi) pair, then
        vmapped over a batch of B pairs.  This amortises JAX dispatch
        overhead: instead of B round-trips per stage-step, there is one.
        B=1 is valid and preserves old behaviour.

    [donate_argnums=(0, 1)]
        params and opt_state are donated to XLA on each call.  XLA is
        permitted to reuse those device buffers for the output params and
        opt_state instead of allocating new memory.  This removes the
        largest per-step allocation and reduces peak GPU memory.

    Parameters
    ----------
    model           : FourierDeepONet instance
    w_ops           : WOperators  (geometry.WOperators)
    weights         : LossWeights  (default DEFAULT_WEIGHTS)
    tx              : optax GradientTransformation — required.
    batch_size      : int   Number of (g, xi) pairs per step (default 16).
                            B=1 is valid.
    inject_noise_fn : callable with signature inject_noise(g, sigma, key)
                      Must be provided so noise runs inside JIT.

    Returns
    -------
    step_fn : jit-compiled callable with signature
                (params, opt_state, g_clean, sigma, rng, xi,
                 active_mask, rho_flat)
                -> (params, opt_state, total_loss, loss_dict)

              g_clean    : (B, 128)   clean sinograms for the batch
              sigma      : scalar     noise level for this step
              rng        : JAX PRNGKey
              xi         : (B, 9)     hardware vectors for the batch
              active_mask: (128,)
              rho_flat   : (n_radial,)
    """
    if tx is None:
        raise ValueError("make_train_step: `tx` must be supplied.")
    if inject_noise_fn is None:
        raise ValueError(
            "make_train_step: `inject_noise_fn` must be supplied. "
            "Pass data_loader.inject_noise."
        )

    # ── per-sample loss (single g, single xi) ────────────────────────
    def _loss_one(p, g_clean_i, sigma, rng_i, xi_i, active_mask, rho_flat):
        # [noise inside JIT] Noise is generated here, inside the compiled
        # graph, so this entire function — noise → forward → loss — is one
        # fused XLA computation with no CPU synchronisation in between.
        g_noisy = inject_noise_fn(g_clean_i, sigma, rng_i)
        eps1d   = model.apply(p, g_noisy, xi_i)
        return loss_fn(eps1d, g_noisy, w_ops, active_mask, rho_flat, weights)

    # ── batched loss: average total and each aux dict entry over B ────
    def _loss_batch(p, g_clean, sigma, rngs, xi, active_mask, rho_flat):
        # [vmap batching] vmap over the batch axis of g_clean, rngs, xi.
        # Each sample gets its own noise draw via a split RNG key (rngs).
        # Loss values are averaged across B to keep gradient magnitudes
        # independent of batch size — training objective is unchanged.
        batched = jax.vmap(
            lambda g_i, rng_i, xi_i: _loss_one(
                p, g_i, sigma, rng_i, xi_i, active_mask, rho_flat
            )
        )(g_clean, rngs, xi)

        # batched = ((B,) totals, {key: (B,) values})
        totals, ld_batch = batched
        total_mean = jnp.mean(totals)
        ld_mean    = {k: jnp.mean(v) for k, v in ld_batch.items()}
        return total_mean, ld_mean

    # ── JIT-compiled step ─────────────────────────────────────────────
    # donate_argnums=(0, 1): donate params and opt_state so XLA can
    # reuse their device buffers for the updated values.  This avoids
    # allocating a fresh copy of all parameters every single step.
    @jax.jit(donate_argnums=(0, 1))
    def step_fn(
        params,
        opt_state,
        g_clean     : jnp.ndarray,   # (B, 128)
        sigma       : jnp.ndarray,   # scalar
        rng         : jnp.ndarray,   # PRNGKey
        xi          : jnp.ndarray,   # (B, 9)
        active_mask : jnp.ndarray,   # (128,)
        rho_flat    : jnp.ndarray,   # (n_radial,)
    ) -> Tuple[dict, object, jnp.ndarray, Dict[str, jnp.ndarray]]:

        # Split rng into B independent keys — one noise draw per sample.
        rngs = jax.random.split(rng, g_clean.shape[0])

        def _loss_for_grad(p):
            return _loss_batch(p, g_clean, sigma, rngs, xi, active_mask, rho_flat)

        (total, ld), grads = jax.value_and_grad(
            _loss_for_grad, has_aux=True
        )(params)

        # Zero NaN / Inf gradients (numerical safety, unchanged)
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g)),
            grads,
        )

        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, total, ld

    print(
        f"make_train_step: step_fn compiled with jax.jit  "
        f"batch_size={batch_size}  donate_argnums=(0,1)  "
        f"noise_inside_jit=True"
    )
    return step_fn


# =======================================================================
# 3.  Single-profile training loop
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
    batch_size        : int                          = 16,
) -> Tuple[dict, object, int, float]:
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

    Parameters
    ----------
    step_fn      : jit-compiled step function from make_train_step().
    params       : current param tree
    opt_state    : current optimiser state
    profile      : one dict from data_loader.load_profiles().
                   Must contain keys: "g_ideal", "xi".
    rho_flat     : (n_radial,)  normalised elliptic radius per radial point
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
    prof_idx     : index of this profile
    log_every    : print + host-sync frequency (steps)
    save_every   : checkpoint frequency (steps)
    t0           : wall-clock start time
    batch_size   : B — number of samples per step (default 16)

    Returns
    -------
    params, opt_state, ep_global, best_data
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
    g_single    = jax.device_put(profile["g_ideal"], _gpu)            # (128,)
    xi_single   = jax.device_put(profile["xi"],      _gpu)            # (9,)
    g_clean     = jnp.stack([g_single] * batch_size, axis=0)          # (B, 128)
    xi_batch    = jnp.stack([xi_single] * batch_size, axis=0)         # (B, 9)
    rho_flat    = jax.device_put(rho_flat,   _gpu)
    active_mask = jax.device_put(active_mask, _gpu)

    # Timing state for steps/sec diagnostics
    _t_last_log  = time.time()
    _ep_last_log = ep_global

    for si, (s_ep, s_sig) in enumerate(stages):

        if si < start_stage:
            continue

        def _sigma(step: int) -> float:
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
            sigma_val = jnp.array(_sigma(ep_global), dtype=jnp.float32)

            # ── Single JIT call covers noise + forward + loss + update ──
            params, opt_state, total, ld = step_fn(
                params, opt_state,
                g_clean, sigma_val, rng,
                xi_batch, active_mask, rho_flat,
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
                print(
                    f"    ep={ep_global:6d} | "
                    f"loss={total_f:.5f} | "
                    f"proj={proj_f:.5f} | "
                    f"smooth={smth_f:.5f} | "
                    f"batch={batch_size} | "
                    f"steps/s={steps_per_s:.1f} | "
                    f"t={elapsed:.0f}s"
                )

            if ep_global % save_every == 0 and do_checkpoint_fn is not None:
                do_checkpoint_fn(ep_global, prof_idx, si, ep + 1, best_data,
                                 params, opt_state)

    return params, opt_state, ep_global, best_data


# =======================================================================
# 4.  Full training loop  (all profiles x all stages)
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
    ep_global    : int                          = 0,
    best_data    : float                        = float("inf"),
    start_prof   : int                          = 0,
    start_stage  : int                          = 0,
    start_ep     : int                          = 0,
    hist         : Optional[dict]               = None,
    do_checkpoint_fn                            = None,
    log_every    : int                          = cfg.LOG_EVERY,
    save_every   : int                          = cfg.SAVE_EVERY,
    single_profile : bool                       = False,
    batch_size   : int                          = 16,
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
    single_profile : bool
                     When True, train on profiles[start_prof] only and
                     return immediately after that profile completes.
                     Useful for smoke-testing or single-profile experiments
                     without touching config or the data pipeline.
    batch_size   : int  Passed through to train_one_profile (default 16).

    Returns
    -------
    params, opt_state, hist, best_data
    """
    if hist is None:
        hist = {}

    t0            = time.time()
    n_prof        = 1 if single_profile else len(profiles)
    n_stages      = len(stages)
    total_planned = sum(s for s, _ in stages) * n_prof

    print(f"\n{'='*60}")
    if single_profile:
        print(f"VICTOR v7.0 -- Single-Profile Training  (profile index {start_prof})")
    else:
        print(f"VICTOR v7.0 -- Training  ({n_prof} profiles x {n_stages} stages)")
    print(f"  Total planned steps : {total_planned:,}")
    print(f"  Batch size          : {batch_size}")
    print(f"  Resume: prof={start_prof}  stage={start_stage}  ep={ep_global}")
    if curriculum is not None:
        print(f"  Curriculum: {curriculum}")
    print(f"{'='*60}\n")

    _start_stage = start_stage
    _start_ep    = start_ep

    prof_end = start_prof + n_prof   # n_prof is already 1 when single_profile=True
    for pi in range(start_prof, prof_end):

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
            batch_size        = batch_size,
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

    return params, opt_state, hist, best_data


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
