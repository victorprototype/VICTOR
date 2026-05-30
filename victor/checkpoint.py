# ============================================================
# VICTOR v8.2 — checkpoint.py
# Orbax-backed checkpointing with JSON metadata sidecar
# ============================================================
# Public API
# ----------
#   build_ckpt_manager(ckpt_dir, max_to_keep)  → CheckpointManager
#   save_meta(ckpt_dir, ep_global, ...)        → None   (atomic JSON write)
#   load_meta(ckpt_dir)                        → dict | None
#   do_checkpoint(mgr, params, opt_state, ...) → None   (blocking save)
#   resume(mgr, ckpt_dir, params, opt_state,   → ResumeBundle
#          tx)
#
# Design principles
# -----------------
#  • Params and opt_state are saved as SEPARATE items under
#    fixed keys CKPT_PARAMS_KEY / CKPT_OPT_KEY.  Mixing them
#    in a single dict causes Orbax PyTree-shape mismatches on restore.
#  • Scalar training state (epoch, profile index, stage index, best loss,
#    curriculum/harmonic schedule positions) lives in a JSON sidecar next
#    to the Orbax directory, written with os.replace() (atomic rename) so
#    a Colab disconnect mid-write never corrupts the sidecar.
#  • wait_until_finished() is called after every save to guarantee the
#    async Orbax write is flushed to disk before we return.  This is the
#    key fix that prevents file corruption on GPU-timeout disconnects.
#  • Adam moments (opt_state) are always restored alongside params so
#    the LR schedule and momentum estimates survive a restart.
#  • All NumPy conversions (jax → np) happen here so the caller's
#    params/opt_state remain on device throughout training.
#  • cfg.PDE_COLLOC_RHO_THRESHOLD is written to the sidecar as a
#    read-only diagnostic field (for self-documentation) but is never
#    restored — it is always read live from config at runtime.
#
# v8.2 changes vs v6.0
# --------------------
#  • ResumeBundle: added four new fields —
#      curriculum_step  (int)   schedule index into CURRICULUM_SCHEDULE
#      harmonic_step    (int)   schedule index into HARMONIC_DECAY_SCHEDULE
#      w_pde_current    (float) actual W_PDE × w_pde_scale at save time
#      w_pol_current    (float) actual harmonic penalty weight at save time
#  • save_meta: accepts and writes all four new fields plus the
#    read-only diagnostic cfg.PDE_COLLOC_RHO_THRESHOLD
#  • do_checkpoint: forwards the four new keyword arguments to save_meta
#  • resume: reads all four new keys from the sidecar with safe .get()
#    defaults; both fresh-start and sidecar-missing branches set defaults
#  • Resume log line extended to include curriculum_step, harmonic_step,
#    w_pde_current, w_pol_current, and PDE_COLLOC_RHO_THRESHOLD
#  • Self-test (__main__) updated with new fields, assertions, and a
#    cfg.summary() call for context
# ============================================================

from __future__ import annotations

import json
import os
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

try:
    import orbax.checkpoint as ocp
except ImportError as e:
    raise ImportError(
        "orbax-checkpoint is required.  Install it with:\n"
        "  pip install orbax-checkpoint"
    ) from e

from victor import config as cfg


# ── Constants ────────────────────────────────────────────────────────

CKPT_PARAMS_KEY = "params"
CKPT_OPT_KEY    = "opt_state"
META_FILENAME   = "train_meta.json"


# ── Named return type for resume bundle ──────────────────────────────

class ResumeBundle(NamedTuple):
    """
    All state restored from a checkpoint, ready to hand to the training loop.

    Attributes
    ----------
    params           : restored Flax param tree (on-device jnp arrays)
    opt_state        : restored optax optimiser state
    ep_global        : global epoch counter at the point of the save
    start_prof       : profile index to resume from
    start_stage      : noise-stage index to resume from
    start_ep         : epoch-within-stage to resume from
    best_data        : best projection loss seen so far
    curriculum_step  : index into cfg.CURRICULUM_SCHEDULE at save time.
                       Used by the trainer to resume the physics curriculum
                       mid-schedule without restarting from step 0.
    harmonic_step    : index into cfg.HARMONIC_DECAY_SCHEDULE at save time.
                       Mirrors curriculum_step for the harmonic regularisation
                       ramp so the w_pol weight is correctly restored.
    w_pde_current    : actual W_PDE × w_pde_scale value at save time.
                       Stored so the trainer can sanity-check the resumed
                       value against the schedule rather than recomputing it
                       blindly from the schedule index alone.
    w_pol_current    : actual harmonic penalty weight at save time.
                       Same sanity-check purpose as w_pde_current for the
                       harmonic regularisation axis.
    resumed          : True if an actual checkpoint was found and loaded
    """
    params           : dict
    opt_state        : object
    ep_global        : int
    start_prof       : int
    start_stage      : int
    start_ep         : int
    best_data        : float
    curriculum_step  : int
    harmonic_step    : int
    w_pde_current    : float
    w_pol_current    : float
    resumed          : bool


# ═══════════════════════════════════════════════════════════════════════
# 1.  CheckpointManager factory
# ═══════════════════════════════════════════════════════════════════════

def build_ckpt_manager(
    ckpt_dir    : str,
    max_to_keep : int = 3,
) -> ocp.CheckpointManager:
    """
    Create an Orbax CheckpointManager pointing at ``ckpt_dir``.

    Parameters
    ----------
    ckpt_dir    : str   Directory where checkpoint subdirs are written.
                        Created if it does not exist.
    max_to_keep : int   Maximum number of checkpoint steps to retain.
                        Older ones are deleted automatically.

    Returns
    -------
    ocp.CheckpointManager
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    options = ocp.CheckpointManagerOptions(
        max_to_keep         = max_to_keep,
        save_interval_steps = 1,       # we control frequency ourselves
    )
    # orbax 0.11.x: declare item names upfront so the manager knows the keys.
    # Use args=ocp.args.Composite(...) at save/restore time (not items=).
    mgr = ocp.CheckpointManager(
        ckpt_dir,
        options=options,
        item_names=(CKPT_PARAMS_KEY, CKPT_OPT_KEY),
    )
    print(f"CheckpointManager ready: {ckpt_dir}  (max_to_keep={max_to_keep})")
    return mgr


# ═══════════════════════════════════════════════════════════════════════
# 2.  JSON metadata sidecar
# ═══════════════════════════════════════════════════════════════════════

def _meta_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, META_FILENAME)


def save_meta(
    ckpt_dir         : str,
    ep_global        : int,
    prof_idx         : int,
    stage_idx        : int,
    ep_in_stage      : int,
    best_data        : float,
    curriculum_step  : int   = 0,
    harmonic_step    : int   = 0,
    w_pde_current    : float = cfg.W_PDE,
    w_pol_current    : float = 1.0,
) -> None:
    """
    Atomically write scalar training state to a JSON sidecar file.

    Uses a write-then-rename pattern so that a process crash between
    writes never leaves the sidecar in a partially-written state.

    Parameters
    ----------
    ckpt_dir        : str   Same directory passed to build_ckpt_manager().
    ep_global       : int   Global step counter.
    prof_idx        : int   Profile index (loop variable ``pi``).
    stage_idx       : int   Noise-stage index (loop variable ``si``).
    ep_in_stage     : int   Local epoch within the current stage.
    best_data       : float Best projection loss seen so far.
    curriculum_step : int   Index into cfg.CURRICULUM_SCHEDULE at save time.
                            Default 0 (beginning of schedule).
    harmonic_step   : int   Index into cfg.HARMONIC_DECAY_SCHEDULE at save time.
                            Default 0 (beginning of schedule).
    w_pde_current   : float Actual W_PDE × w_pde_scale at save time.
                            Default cfg.W_PDE (full weight, no curriculum scaling).
    w_pol_current   : float Actual harmonic penalty weight at save time.
                            Default 1.0 (strongest regularisation, start of schedule).

    Notes
    -----
    cfg.PDE_COLLOC_RHO_THRESHOLD is written as a read-only diagnostic field
    so each sidecar is self-documenting (you can inspect it to know which
    config generated the checkpoint).  It is NEVER restored from the sidecar —
    the live value is always read from cfg at runtime, so changing the config
    between runs takes effect immediately without a checkpoint migration.
    """
    payload = dict(
        ep_global        = int(ep_global),
        prof_idx         = int(prof_idx),
        stage_idx        = int(stage_idx),
        ep_in_stage      = int(ep_in_stage),
        best_data        = float(best_data),
        curriculum_step  = int(curriculum_step),
        harmonic_step    = int(harmonic_step),
        w_pde_current    = float(w_pde_current),
        w_pol_current    = float(w_pol_current),
        # ── read-only diagnostic — never restored, always read from cfg ──
        _pde_colloc_rho_threshold = float(cfg.PDE_COLLOC_RHO_THRESHOLD),
    )
    tmp_path   = _meta_path(ckpt_dir) + ".tmp"
    final_path = _meta_path(ckpt_dir)

    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)

    os.replace(tmp_path, final_path)   # atomic rename — safe on crash


def load_meta(ckpt_dir: str) -> Optional[dict]:
    """
    Read the JSON sidecar if it exists.

    Returns
    -------
    dict with keys ep_global, prof_idx, stage_idx, ep_in_stage, best_data,
    curriculum_step, harmonic_step, w_pde_current, w_pol_current, and the
    read-only diagnostic _pde_colloc_rho_threshold;
    or None if the file does not exist or is unreadable.
    """
    path = _meta_path(ckpt_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [checkpoint] Meta read failed ({e}) — scalar state reset")
        return None


# ═══════════════════════════════════════════════════════════════════════
# 3.  Blocking checkpoint save
# ═══════════════════════════════════════════════════════════════════════

def do_checkpoint(
    mgr              : ocp.CheckpointManager,
    params           : dict,
    opt_state        : object,
    ep_global        : int,
    prof_idx         : int,
    stage_idx        : int,
    ep_in_stage      : int,
    best_data        : float,
    ckpt_dir         : Optional[str] = None,
    curriculum_step  : int   = 0,
    harmonic_step    : int   = 0,
    w_pde_current    : float = cfg.W_PDE,
    w_pol_current    : float = 1.0,
) -> None:
    """
    Save params + opt_state via Orbax, then flush the JSON sidecar.

    Calls wait_until_finished() after the Orbax save so that async
    writes are completed before this function returns.  This guarantees
    the checkpoint is safely on disk even if the Python process is
    killed immediately afterwards (Colab GPU-timeout pattern).

    Parameters
    ----------
    mgr             : ocp.CheckpointManager  built by build_ckpt_manager()
    params          : Flax param tree (on-device jnp arrays)
    opt_state       : optax optimiser state (on-device)
    ep_global       : step index passed to mgr.save() as the "step" key
    prof_idx        : current profile index (written to sidecar)
    stage_idx       : current stage index (written to sidecar)
    ep_in_stage     : local epoch within stage (written to sidecar)
    best_data       : best projection loss (written to sidecar)
    ckpt_dir        : directory for the JSON sidecar.
                      Required only if mgr.directory is not accessible as an
                      attribute (older Orbax versions); defaults to
                      inferred from mgr._directory.
    curriculum_step : index into cfg.CURRICULUM_SCHEDULE at call time.
                      Default 0; pass the trainer's current schedule index.
    harmonic_step   : index into cfg.HARMONIC_DECAY_SCHEDULE at call time.
                      Default 0; pass the trainer's current schedule index.
    w_pde_current   : actual W_PDE × w_pde_scale at call time.
                      Default cfg.W_PDE; pass the live scaled value from the
                      trainer so the sidecar can be used for sanity checks on
                      resume.
    w_pol_current   : actual harmonic penalty weight at call time.
                      Default 1.0; pass the live value from the trainer.

    Notes
    -----
    Params and opt_state are converted to NumPy arrays before passing to
    Orbax.  The on-device originals are not affected.
    """
    # Convert to NumPy for serialisation (non-destructive)
    params_np    = jax.tree_util.tree_map(np.array, params)
    opt_state_np = jax.tree_util.tree_map(np.array, opt_state)

    # orbax 0.11.x uses args= with ocp.args.Composite instead of items=
    mgr.save(
        ep_global,
        args=ocp.args.Composite(
            **{
                CKPT_PARAMS_KEY: ocp.args.StandardSave(params_np),
                CKPT_OPT_KEY   : ocp.args.StandardSave(opt_state_np),
            }
        ),
    )
    mgr.wait_until_finished()   # CRITICAL: blocks until disk write completes

    # Write the JSON sidecar
    resolved_dir = ckpt_dir or _resolve_dir(mgr)
    save_meta(
        resolved_dir,
        ep_global,
        prof_idx,
        stage_idx,
        ep_in_stage,
        best_data,
        curriculum_step = curriculum_step,
        harmonic_step   = harmonic_step,
        w_pde_current   = w_pde_current,
        w_pol_current   = w_pol_current,
    )


# ═══════════════════════════════════════════════════════════════════════
# 4.  Resume logic
# ═══════════════════════════════════════════════════════════════════════

def resume(
    mgr       : ocp.CheckpointManager,
    ckpt_dir  : str,
    params    : dict,
    opt_state : object,
    tx,
) -> ResumeBundle:
    """
    Attempt to restore params, opt_state, and scalar training state.

    If a valid checkpoint is found it is loaded and a ResumeBundle with
    ``resumed=True`` is returned.  If no checkpoint exists, or if the
    restore fails for any reason, the original (fresh) params and
    opt_state are returned inside a bundle with ``resumed=False``.

    Parameters
    ----------
    mgr       : ocp.CheckpointManager
    ckpt_dir  : str   Directory for the JSON sidecar.
    params    : dict  Freshly-initialised param tree (used as shape reference).
    opt_state : object  Freshly-initialised optimiser state.
    tx        : optax GradientTransformation  (kept for potential re-init).

    Returns
    -------
    ResumeBundle
    """
    latest = mgr.latest_step()

    if latest is None:
        print("No checkpoint found — starting fresh.")
        return ResumeBundle(
            params          = params,
            opt_state       = opt_state,
            ep_global       = 0,
            start_prof      = 0,
            start_stage     = 0,
            start_ep        = 0,
            best_data       = float("inf"),
            curriculum_step = 0,
            harmonic_step   = 0,
            w_pde_current   = cfg.W_PDE,
            w_pol_current   = 1.0,
            resumed         = False,
        )

    # ── Load sidecar first (non-destructive — just reads a file) ─────
    meta = load_meta(ckpt_dir)

    # ── Restore Orbax checkpoint ──────────────────────────────────────
    try:
        # orbax 0.11.x uses args= with ocp.args.Composite instead of items=
        restored = mgr.restore(
            latest,
            args=ocp.args.Composite(
                **{
                    CKPT_PARAMS_KEY: ocp.args.StandardRestore(params),
                    CKPT_OPT_KEY   : ocp.args.StandardRestore(opt_state),
                }
            ),
        )

        # Move arrays back onto the JAX device
        r_params    = jax.tree_util.tree_map(jnp.array, restored[CKPT_PARAMS_KEY])
        r_opt_state = jax.tree_util.tree_map(jnp.array, restored[CKPT_OPT_KEY])

    except Exception as e:
        print(f"  [checkpoint] Restore failed ({e}) — starting fresh.")
        return ResumeBundle(
            params          = params,
            opt_state       = opt_state,
            ep_global       = 0,
            start_prof      = 0,
            start_stage     = 0,
            start_ep        = 0,
            best_data       = float("inf"),
            curriculum_step = 0,
            harmonic_step   = 0,
            w_pde_current   = cfg.W_PDE,
            w_pol_current   = 1.0,
            resumed         = False,
        )

    # ── Parse scalar state from sidecar ──────────────────────────────
    if meta:
        ep_global       = int(meta.get("ep_global",        0))
        start_prof      = int(meta.get("prof_idx",          0))
        start_stage     = int(meta.get("stage_idx",         0))
        start_ep        = int(meta.get("ep_in_stage",       0))
        best_data       = float(meta.get("best_data",       float("inf")))
        curriculum_step = int(meta.get("curriculum_step",   0))
        harmonic_step   = int(meta.get("harmonic_step",     0))
        w_pde_current   = float(meta.get("w_pde_current",   cfg.W_PDE))
        w_pol_current   = float(meta.get("w_pol_current",   1.0))

        print(
            f"\n── Checkpoint resumed (step {latest}) ──────────────────────────\n"
            f"  ep_global     : {ep_global}\n"
            f"  prof/stage/ep : prof={start_prof}  stage={start_stage}"
            f"  ep_in_stage={start_ep}\n"
            f"  best_data     : {best_data:.6g}\n"
            f"  Curriculum    : schedule_idx={curriculum_step}"
            f"  →  w_pde_current={w_pde_current:.4f}"
            f"  (W_PDE={cfg.W_PDE})\n"
            f"  Harmonics     : schedule_idx={harmonic_step}"
            f"  →  w_pol_current={w_pol_current:.4f}\n"
            f"  PDE colloc    : rho_threshold={cfg.PDE_COLLOC_RHO_THRESHOLD}"
            f"  (live from cfg — not restored from sidecar)\n"
            f"────────────────────────────────────────────────────────────────"
        )
    else:
        # Params/opt_state loaded but sidecar missing — scalar state reset
        ep_global       = 0
        start_prof      = 0
        start_stage     = 0
        start_ep        = 0
        best_data       = float("inf")
        curriculum_step = 0
        harmonic_step   = 0
        w_pde_current   = cfg.W_PDE
        w_pol_current   = 1.0
        print(
            f"\n── Checkpoint resumed (step {latest}) — sidecar missing ────────\n"
            f"  params/opt_state loaded; all scalar state reset to defaults.\n"
            f"  curriculum_step={curriculum_step}  harmonic_step={harmonic_step}\n"
            f"  w_pde_current={w_pde_current}  w_pol_current={w_pol_current}\n"
            f"  PDE colloc rho_threshold={cfg.PDE_COLLOC_RHO_THRESHOLD} (from cfg)\n"
            f"────────────────────────────────────────────────────────────────"
        )

    return ResumeBundle(
        params          = r_params,
        opt_state       = r_opt_state,
        ep_global       = ep_global,
        start_prof      = start_prof,
        start_stage     = start_stage,
        start_ep        = start_ep,
        best_data       = best_data,
        curriculum_step = curriculum_step,
        harmonic_step   = harmonic_step,
        w_pde_current   = w_pde_current,
        w_pol_current   = w_pol_current,
        resumed         = True,
    )


# ═══════════════════════════════════════════════════════════════════════
# 5.  Internal helpers
# ═══════════════════════════════════════════════════════════════════════

def _resolve_dir(mgr: ocp.CheckpointManager) -> str:
    """
    Robustly retrieve the checkpoint directory from the manager object.

    Orbax has changed the attribute name across versions; we try several.
    """
    for attr in ("directory", "_directory", "_manager_dir"):
        val = getattr(mgr, attr, None)
        if val is not None:
            return str(val)
    raise AttributeError(
        "Cannot determine checkpoint directory from CheckpointManager. "
        "Pass ckpt_dir explicitly to do_checkpoint()."
    )


# ── Module self-test ─────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("checkpoint.py — self-test (v8.2)\n")
    cfg.summary()
    print()

    with tempfile.TemporaryDirectory() as tmp:
        mgr = build_ckpt_manager(tmp, max_to_keep=2)

        # Fake params / opt_state
        fake_params    = {"params": {"w": np.ones((4, 4), dtype=np.float32)}}
        fake_opt_state = {"mu": np.zeros((4, 4), dtype=np.float32)}

        # Save — include all v8.2 schedule fields
        do_checkpoint(
            mgr,
            fake_params,
            fake_opt_state,
            ep_global       = 10,
            prof_idx        = 0,
            stage_idx       = 1,
            ep_in_stage     = 5,
            best_data       = 0.42,
            ckpt_dir        = tmp,
            curriculum_step = 2,
            harmonic_step   = 1,
            w_pde_current   = 0.3,
            w_pol_current   = 0.5,
        )

        # ── Verify JSON sidecar ───────────────────────────────────────
        meta = load_meta(tmp)
        assert meta is not None, "Meta not found after save"
        assert meta["ep_global"]       == 10,       f"ep_global mismatch: {meta}"
        assert meta["best_data"]       == 0.42,     f"best_data mismatch: {meta}"
        assert meta["curriculum_step"] == 2,        f"curriculum_step mismatch: {meta}"
        assert meta["harmonic_step"]   == 1,        f"harmonic_step mismatch: {meta}"
        assert meta["w_pde_current"]   == 0.3,      f"w_pde_current mismatch: {meta}"
        assert meta["w_pol_current"]   == 0.5,      f"w_pol_current mismatch: {meta}"
        assert "_pde_colloc_rho_threshold" in meta, "Diagnostic field missing from sidecar"
        assert meta["_pde_colloc_rho_threshold"] == cfg.PDE_COLLOC_RHO_THRESHOLD, \
            "PDE_COLLOC_RHO_THRESHOLD diagnostic mismatch"
        print(f"  save_meta / load_meta OK: {meta}")

        # ── Verify ResumeBundle ───────────────────────────────────────
        bundle = resume(mgr, tmp, fake_params, fake_opt_state, tx=None)
        assert bundle.resumed,                          "Expected resumed=True"
        assert bundle.ep_global       == 10,            f"ep_global mismatch: {bundle}"
        assert bundle.curriculum_step == 2,             f"curriculum_step mismatch: {bundle}"
        assert bundle.harmonic_step   == 1,             f"harmonic_step mismatch: {bundle}"
        assert abs(bundle.w_pde_current - 0.3) < 1e-9, f"w_pde_current mismatch: {bundle}"
        assert abs(bundle.w_pol_current - 0.5) < 1e-9, f"w_pol_current mismatch: {bundle}"
        print(
            f"  resume OK: ep_global={bundle.ep_global}  "
            f"curriculum_step={bundle.curriculum_step}  "
            f"harmonic_step={bundle.harmonic_step}  "
            f"w_pde={bundle.w_pde_current}  "
            f"w_pol={bundle.w_pol_current}"
        )

    print("\ncheckpoint.py self-test PASSED (v8.2)")
