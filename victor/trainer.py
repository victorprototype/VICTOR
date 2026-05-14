# ============================================================
# VICTOR v6.0 — trainer.py
# Closure-based JIT training step, training loop, stage logic
# ============================================================
# Public API
# ----------
#   build_optimizer(total_steps, lr, warmup)   -> tx (optax chain)
#   make_train_step(model, w_ops, weights, tx) -> step_fn (jit-compiled)
#   train_one_profile(step_fn, ...)            -> (params, opt_state, ep_global, best_data)
#   train(step_fn, ...)                        -> (params, opt_state, hist, best_data)
#
# Architecture
# ------------
#  * tx, model, w_ops, and weights are Python objects — they cannot
#    cross the JIT boundary as traced values.  They are closed over
#    inside make_train_step() and never passed at runtime.
#
#  * Correct usage:
#      tx      = build_optimizer(total_steps=N)
#      step_fn = make_train_step(model=m, w_ops=ops, weights=w, tx=tx)
#      train(step_fn=step_fn, ...)
#
#  * step_fn is compiled once and reused across all profiles / stages.
#  * Noisy sinograms are re-drawn every log_every steps to avoid
#    the network memorising a fixed noise realisation.
#  * STAGES is consumed from config.py so Cell 1 constants propagate.
#  * hist dict is updated in-place; callers receive the same object.
#  * Gradient NaN / Inf values are zeroed before the optimiser update
#    (the "safe grad" pattern).
#  * loss_fn is imported from losses.py; value_and_grad is called only
#    inside the closure returned by make_train_step().
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
# 1.  Optimiser factory
# =======================================================================

def build_optimizer(
    total_steps : int,
    lr          : float = cfg.LR,
    warmup      : int   = 500,
    lr_end      : float = 5e-5,
) -> optax.GradientTransformation:
    """
    Build the VICTOR v6 optimiser: warmup-cosine-decay Adam + grad-clip.

    Schedule
    --------
    0 -> warmup steps : linear warm-up from 0 to lr
    warmup -> total   : cosine decay from lr to lr_end

    Parameters
    ----------
    total_steps : int   Total number of gradient steps expected.
    lr          : float Peak learning rate (default cfg.LR = 3e-4).
    warmup      : int   Number of warm-up steps (default 500).
    lr_end      : float Final learning rate at end of cosine decay.

    Returns
    -------
    optax.GradientTransformation   (chain of clip + adam)
    """
    sched = optax.warmup_cosine_decay_schedule(
        init_value       = 0.0,
        peak_value       = lr,
        warmup_steps     = warmup,
        decay_steps      = total_steps,
        end_value        = lr_end,
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(sched),
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
    Factory that returns the single JIT-compiled training step for VICTOR.

    All Python objects (tx, model, w_ops, weights) are closed over here
    so they never cross the JIT boundary as traced values.  Call this
    function ONCE before the training loop and reuse the returned step_fn
    across all profiles and stages.

    Parameters
    ----------
    model   : VICTOR_v6 instance
    w_ops   : WOperators  (geometry.WOperators)
    weights : LossWeights  (default DEFAULT_WEIGHTS)
    tx      : optax GradientTransformation -- required, closed over.
              Do NOT pass tx to the returned step_fn at call time.

    Returns
    -------
    step_fn : jit-compiled callable with signature
        step_fn(params, opt_state,
                g_noisy, psi_n, bpol_n,
                R_flat, Z_flat, esrc, edst, ew, ndeg,
                rho_2d, rho_flat, active_mask)
        -> (new_params, new_opt_state, total, loss_dict)
    """
    if tx is None:
        raise ValueError(
            "make_train_step: `tx` (optax GradientTransformation) must be "
            "supplied to the factory so it can be closed over. "
            "Do not pass it as an argument to the returned step_fn."
        )

    @jax.jit
    def _step(
        params,
        opt_state,
        g_noisy     : jnp.ndarray,
        psi_n       : jnp.ndarray,
        bpol_n      : jnp.ndarray,
        R_flat      : jnp.ndarray,
        Z_flat      : jnp.ndarray,
        esrc        : jnp.ndarray,
        edst        : jnp.ndarray,
        ew          : jnp.ndarray,
        ndeg        : jnp.ndarray,
        rho_2d      : jnp.ndarray,
        rho_flat    : jnp.ndarray,
        active_mask : jnp.ndarray,
    ) -> Tuple[dict, object, jnp.ndarray, Dict[str, jnp.ndarray]]:

        def _loss(p):
            eps_out, mean, std, preds = model.apply(
                p,
                R_flat, Z_flat, psi_n, bpol_n,
                esrc, edst, ew, ndeg, rho_2d,
            )
            log_noise = p["params"]["log_noise"]
            return loss_fn(
                eps_out, mean, std, preds,
                g_noisy, w_ops,
                active_mask,
                rho_2d, rho_flat,
                log_noise,
                weights,
            )

        (total, ld), grads = jax.value_and_grad(_loss, has_aux=True)(params)

        # Zero out any NaN / Inf gradients (numerical safety)
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g)),
            grads,
        )

        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, total, ld

    return _step


# =======================================================================
# 3.  Single-profile training loop  (one profile x all stages)
# =======================================================================

def train_one_profile(
    *,
    step_fn,
    params            : dict,
    opt_state,
    profile           : dict,
    R_flat            : jnp.ndarray,
    Z_flat            : jnp.ndarray,
    esrc              : jnp.ndarray,
    edst              : jnp.ndarray,
    ew                : jnp.ndarray,
    ndeg              : jnp.ndarray,
    rho_2d            : jnp.ndarray,
    rho_flat          : jnp.ndarray,
    active_mask       : jnp.ndarray,
    inject_noise,
    stages            : list            = cfg.STAGES,
    ep_global         : int             = 0,
    best_data         : float           = float("inf"),
    start_stage       : int             = 0,
    start_ep_in_stage : int             = 0,
    hist              : Optional[dict]  = None,
    do_checkpoint_fn                    = None,
    prof_idx          : int             = 0,
    log_every         : int             = cfg.LOG_EVERY,
    save_every        : int             = cfg.SAVE_EVERY,
    t0                : Optional[float] = None,
) -> Tuple[dict, object, int, float]:
    """
    Train params through all noise stages for a single profile.

    Parameters
    ----------
    step_fn      : jit-compiled step function from make_train_step().
                   tx is already closed over inside step_fn.
    params       : current param tree (input/output)
    opt_state    : current optimiser state (input/output)
    profile      : one dict from data_loader.load_profiles()
    R_flat ... active_mask : geometry arrays from Cell 2 scope
    inject_noise : data_loader.inject_noise
    stages       : list of (n_steps, sigma) from config.STAGES
    ep_global    : global epoch counter at entry
    best_data    : best projection loss seen so far (for checkpoint metadata)
    start_stage  : which stage to resume from (0 = fresh)
    start_ep_in_stage : which epoch within that stage to resume from (0 = fresh)
    hist         : dict to accumulate loss history (mutated in place)
    do_checkpoint_fn : callable(ep_global, prof_idx, stage_idx, ep_in_stage, best_data)
    prof_idx     : index of this profile (for logging / checkpoint metadata)
    log_every    : print frequency
    save_every   : checkpoint frequency
    t0           : wall-clock start time (for elapsed display)

    Returns
    -------
    params, opt_state, ep_global, best_data
    """
    if hist is None:
        hist = {}
    if t0 is None:
        t0 = time.time()

    psi_n   = profile["psi_n"]
    bpol_n  = profile["bpol_n"]
    g_clean = profile["g_ideal"]

    for si, (s_ep, s_sig) in enumerate(stages):

        # -- Resume: skip completed stages for this profile ---------------
        if si < start_stage:
            continue

        # Noise seed: mix global epoch + stage index
        g_stage = inject_noise(g_clean, s_sig,
                               jax.random.PRNGKey(ep_global + si))

        # Epoch range for this stage (support resume mid-stage)
        ep_start = start_ep_in_stage if si == start_stage else 0
        # After consuming the resume offset, reset to 0 for all later stages
        if si == start_stage:
            start_ep_in_stage = 0

        print(
            f"  Prof {prof_idx + 1} | stage {si + 1}/{len(stages)} "
            f"sigma={s_sig}  ep_range=[{ep_start},{s_ep})  "
            f"Te_core={float(profile['Te_1d'][0]):.2f} keV"
        )

        for ep in range(ep_start, s_ep):

            # Re-draw noise periodically to avoid memorising a fixed sample
            if ep % log_every == 0 and ep > ep_start:
                g_stage = inject_noise(
                    g_clean, s_sig,
                    jax.random.PRNGKey(ep_global + ep),
                )

            # -- Gradient step --------------------------------------------
            params, opt_state, total, ld = step_fn(
                params, opt_state,
                g_stage, psi_n, bpol_n,
                R_flat, Z_flat,
                esrc, edst, ew, ndeg,
                rho_2d, rho_flat,
                active_mask,
            )
            ep_global += 1

            # -- History accumulation -------------------------------------
            total_f = float(total)
            if np.isfinite(total_f):
                _append(hist, "total", total_f)
                for k, v in ld.items():
                    _append(hist, k, float(v))
                if float(ld.get("proj", float("inf"))) < best_data:
                    best_data = float(ld["proj"])

            # -- Checkpoint -----------------------------------------------
            if ep_global % save_every == 0 and do_checkpoint_fn is not None:
                do_checkpoint_fn(ep_global, prof_idx, si, ep + 1, best_data)

            # -- Logging --------------------------------------------------
            if ep_global % log_every == 0:
                proj_f  = float(ld.get("proj",   0))
                nll_f   = float(ld.get("nll",    0))
                smth_f  = float(ld.get("smooth", 0))
                elapsed = time.time() - t0
                print(
                    f"    ep={ep_global:6d} | L={total_f:.5f} | "
                    f"proj={proj_f:.5f} | nll={nll_f:.5f} | "
                    f"smooth={smth_f:.5f} | t={elapsed:.0f}s"
                )

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
    R_flat       : jnp.ndarray,
    Z_flat       : jnp.ndarray,
    esrc         : jnp.ndarray,
    edst         : jnp.ndarray,
    ew           : jnp.ndarray,
    ndeg         : jnp.ndarray,
    rho_2d       : jnp.ndarray,
    rho_flat     : jnp.ndarray,
    active_mask  : jnp.ndarray,
    inject_noise,
    stages       : list                  = cfg.STAGES,
    ep_global    : int                   = 0,
    best_data    : float                 = float("inf"),
    start_prof   : int                   = 0,
    start_stage  : int                   = 0,
    start_ep     : int                   = 0,
    hist         : Optional[dict]        = None,
    do_checkpoint_fn                     = None,
    log_every    : int                   = cfg.LOG_EVERY,
    save_every   : int                   = cfg.SAVE_EVERY,
) -> Tuple[dict, object, dict, float]:
    """
    Outer training loop: iterate over all profiles and all noise stages.

    Thin wrapper that delegates to train_one_profile() for each
    (profile, stage) block, handles resume logic, and saves a final
    checkpoint after each profile completes.

    Parameters
    ----------
    step_fn      : jit-compiled step from make_train_step().
                   tx is already closed over inside step_fn.
    params       : initial / resumed param tree
    opt_state    : initial / resumed optimiser state
    profiles     : list of profile dicts from data_loader.load_profiles()
    R_flat ... active_mask : geometry arrays (from Cell 2)
    inject_noise : data_loader.inject_noise
    stages       : list of (n_steps, sigma) tuples (default cfg.STAGES)
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
    params     : final param tree
    opt_state  : final optimiser state
    hist       : accumulated loss history dict
    best_data  : best projection loss achieved
    """
    if hist is None:
        hist = {}

    t0       = time.time()
    n_prof   = len(profiles)
    n_stages = len(stages)
    total_planned = sum(s for s, _ in stages) * n_prof
    print(f"\n{'='*60}")
    print(f"VICTOR v6.0 -- Training  ({n_prof} profiles x {n_stages} stages)")
    print(f"  Total planned steps : {total_planned:,}")
    print(f"  Resume: prof={start_prof}  stage={start_stage}  ep={ep_global}")
    print(f"{'='*60}\n")

    _start_stage = start_stage
    _start_ep    = start_ep

    for pi in range(start_prof, n_prof):

        params, opt_state, ep_global, best_data = train_one_profile(
            step_fn           = step_fn,
            params            = params,
            opt_state         = opt_state,
            profile           = profiles[pi],
            R_flat            = R_flat,
            Z_flat            = Z_flat,
            esrc              = esrc,
            edst              = edst,
            ew                = ew,
            ndeg              = ndeg,
            rho_2d            = rho_2d,
            rho_flat          = rho_flat,
            active_mask       = active_mask,
            inject_noise      = inject_noise,
            stages            = stages,
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

        # After first profile, resume offsets are consumed -- start fresh
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
    print("trainer.py -- no self-test (requires full pipeline).")
    print("Run Cell 5 in the notebook to exercise the training loop.")
