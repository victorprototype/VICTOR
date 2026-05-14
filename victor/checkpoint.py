# ============================================================
# VICTOR v6.0 — checkpoint.py
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
#  • Scalar training state (epoch, profile index, stage index, best loss)
#    lives in a JSON sidecar next to the Orbax directory, written with
#    os.replace() (atomic rename) so a Colab disconnect mid-write never
#    corrupts the sidecar.
#  • wait_until_finished() is called after every save to guarantee the
#    async Orbax write is flushed to disk before we return.  This is the
#    key fix that prevents file corruption on GPU-timeout disconnects.
#  • Adam moments (opt_state) are always restored alongside params so
#    the LR schedule and momentum estimates survive a restart.
#  • All NumPy conversions (jax → np) happen here so the caller's
#    params/opt_state remain on device throughout training.
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
    params        : restored Flax param tree (on-device jnp arrays)
    opt_state     : restored optax optimiser state
    ep_global     : global epoch counter at the point of the save
    start_prof    : profile index to resume from
    start_stage   : noise-stage index to resume from
    start_ep      : epoch-within-stage to resume from
    best_data     : best projection loss seen so far
    resumed       : True if an actual checkpoint was found and loaded
    """
    params      : dict
    opt_state   : object
    ep_global   : int
    start_prof  : int
    start_stage : int
    start_ep    : int
    best_data   : float
    resumed     : bool


# ═══════════════════════════════════════════════════════════════════════
# 1.  CheckpointManager factory
# ═══════════════════════════════════════════════════════════════════════

def build_ckpt_manager(
    ckpt_dir     : str,
    max_to_keep  : int = 3,
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
        max_to_keep        = max_to_keep,
        save_interval_steps= 1,       # we control frequency ourselves
    )
    mgr = ocp.CheckpointManager(ckpt_dir, options=options)
    print(f"CheckpointManager ready: {ckpt_dir}  (max_to_keep={max_to_keep})")
    return mgr


# ═══════════════════════════════════════════════════════════════════════
# 2.  JSON metadata sidecar
# ═══════════════════════════════════════════════════════════════════════

def _meta_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, META_FILENAME)


def save_meta(
    ckpt_dir    : str,
    ep_global   : int,
    prof_idx    : int,
    stage_idx   : int,
    ep_in_stage : int,
    best_data   : float,
) -> None:
    """
    Atomically write scalar training state to a JSON sidecar file.

    Uses a write-then-rename pattern so that a process crash between
    writes never leaves the sidecar in a partially-written state.

    Parameters
    ----------
    ckpt_dir    : str   Same directory passed to build_ckpt_manager().
    ep_global   : int   Global step counter.
    prof_idx    : int   Profile index (loop variable ``pi``).
    stage_idx   : int   Noise-stage index (loop variable ``si``).
    ep_in_stage : int   Local epoch within the current stage.
    best_data   : float Best projection loss seen so far.
    """
    payload = dict(
        ep_global   = int(ep_global),
        prof_idx    = int(prof_idx),
        stage_idx   = int(stage_idx),
        ep_in_stage = int(ep_in_stage),
        best_data   = float(best_data),
    )
    tmp_path  = _meta_path(ckpt_dir) + ".tmp"
    final_path = _meta_path(ckpt_dir)

    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)

    os.replace(tmp_path, final_path)   # atomic rename — safe on crash


def load_meta(ckpt_dir: str) -> Optional[dict]:
    """
    Read the JSON sidecar if it exists.

    Returns
    -------
    dict with keys ep_global, prof_idx, stage_idx, ep_in_stage, best_data
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
    mgr         : ocp.CheckpointManager,
    params      : dict,
    opt_state   : object,
    ep_global   : int,
    prof_idx    : int,
    stage_idx   : int,
    ep_in_stage : int,
    best_data   : float,
    ckpt_dir    : Optional[str] = None,
) -> None:
    """
    Save params + opt_state via Orbax, then flush the JSON sidecar.

    Calls wait_until_finished() after the Orbax save so that async
    writes are completed before this function returns.  This guarantees
    the checkpoint is safely on disk even if the Python process is
    killed immediately afterwards (Colab GPU-timeout pattern).

    Parameters
    ----------
    mgr         : ocp.CheckpointManager  built by build_ckpt_manager()
    params      : Flax param tree (on-device jnp arrays)
    opt_state   : optax optimiser state (on-device)
    ep_global   : step index passed to mgr.save() as the "step" key
    prof_idx    : current profile index (written to sidecar)
    stage_idx   : current stage index (written to sidecar)
    ep_in_stage : local epoch within stage (written to sidecar)
    best_data   : best projection loss (written to sidecar)
    ckpt_dir    : directory for the JSON sidecar.
                  Required only if mgr.directory is not accessible as an
                  attribute (older Orbax versions); defaults to
                  inferred from mgr._directory.

    Notes
    -----
    Params and opt_state are converted to NumPy arrays before passing to
    Orbax.  The on-device originals are not affected.
    """
    # Convert to NumPy for serialisation (non-destructive)
    params_np    = jax.tree_util.tree_map(np.array, params)
    opt_state_np = jax.tree_util.tree_map(np.array, opt_state)

    mgr.save(
        ep_global,
        items={
            CKPT_PARAMS_KEY: params_np,
            CKPT_OPT_KEY   : opt_state_np,
        },
    )
    mgr.wait_until_finished()   # CRITICAL: blocks until disk write completes

    # Write the JSON sidecar
    resolved_dir = ckpt_dir or _resolve_dir(mgr)
    save_meta(resolved_dir, ep_global, prof_idx, stage_idx, ep_in_stage, best_data)


# ═══════════════════════════════════════════════════════════════════════
# 4.  Resume logic
# ═══════════════════════════════════════════════════════════════════════

def resume(
    mgr         : ocp.CheckpointManager,
    ckpt_dir    : str,
    params      : dict,
    opt_state   : object,
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
            params      = params,
            opt_state   = opt_state,
            ep_global   = 0,
            start_prof  = 0,
            start_stage = 0,
            start_ep    = 0,
            best_data   = float("inf"),
            resumed     = False,
        )

    # ── Load sidecar first (non-destructive — just reads a file) ─────
    meta = load_meta(ckpt_dir)

    # ── Restore Orbax checkpoint ──────────────────────────────────────
    try:
        restored = mgr.restore(
            latest,
            items={
                CKPT_PARAMS_KEY: params,
                CKPT_OPT_KEY   : opt_state,
            },
        )

        # Move arrays back onto the JAX device
        r_params    = jax.tree_util.tree_map(jnp.array, restored[CKPT_PARAMS_KEY])
        r_opt_state = jax.tree_util.tree_map(jnp.array, restored[CKPT_OPT_KEY])

    except Exception as e:
        print(f"  [checkpoint] Restore failed ({e}) — starting fresh.")
        return ResumeBundle(
            params      = params,
            opt_state   = opt_state,
            ep_global   = 0,
            start_prof  = 0,
            start_stage = 0,
            start_ep    = 0,
            best_data   = float("inf"),
            resumed     = False,
        )

    # ── Parse scalar state from sidecar ──────────────────────────────
    if meta:
        ep_global   = int(meta.get("ep_global",   0))
        start_prof  = int(meta.get("prof_idx",    0))
        start_stage = int(meta.get("stage_idx",   0))
        start_ep    = int(meta.get("ep_in_stage", 0))
        best_data   = float(meta.get("best_data", float("inf")))
        print(
            f"Resumed step {latest}: "
            f"ep={ep_global}  prof={start_prof}  "
            f"stage={start_stage}  ep_in_stage={start_ep}"
        )
    else:
        # Params/opt_state loaded but sidecar missing — scalar state reset
        ep_global   = 0
        start_prof  = 0
        start_stage = 0
        start_ep    = 0
        best_data   = float("inf")
        print(
            f"Resumed params/opt_state from step {latest} "
            f"(sidecar missing — scalar state reset)"
        )

    return ResumeBundle(
        params      = r_params,
        opt_state   = r_opt_state,
        ep_global   = ep_global,
        start_prof  = start_prof,
        start_stage = start_stage,
        start_ep    = start_ep,
        best_data   = best_data,
        resumed     = True,
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

    print("checkpoint.py — self-test")

    with tempfile.TemporaryDirectory() as tmp:
        mgr = build_ckpt_manager(tmp, max_to_keep=2)

        # Fake params / opt_state
        fake_params    = {"params": {"w": np.ones((4, 4), dtype=np.float32)}}
        fake_opt_state = {"mu": np.zeros((4, 4), dtype=np.float32)}

        # Save
        do_checkpoint(mgr, fake_params, fake_opt_state,
                      ep_global=10, prof_idx=0, stage_idx=1,
                      ep_in_stage=5, best_data=0.42, ckpt_dir=tmp)

        # Load meta
        meta = load_meta(tmp)
        assert meta is not None, "Meta not found after save"
        assert meta["ep_global"] == 10
        assert meta["best_data"] == 0.42
        print(f"  save_meta / load_meta OK: {meta}")

        # Resume
        bundle = resume(mgr, tmp, fake_params, fake_opt_state, tx=None)
        assert bundle.resumed, "Expected resumed=True"
        assert bundle.ep_global == 10
        print(f"  resume OK: ep_global={bundle.ep_global}")

    print("checkpoint.py self-test PASSED")
