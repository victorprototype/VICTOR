# ============================================================
# VICTOR v6.0 — Cell 5: Training Loop
# ============================================================
# Depends on:
# Depends on:
#   Cell 1 — installs, package imports, CKPT_DIR, RESULTS_DIR
#   Cell 2 — W matrix, geometry, profiles in cell scope
#   Cell 3 — model, params built
#   Cell 4 — loss_fn, LossWeights, DEFAULT_WEIGHTS imported
#
# Modules consumed:
#   trainer.py    — build_optimizer, make_train_step, train
#   checkpoint.py — build_ckpt_manager, do_checkpoint, resume
# ============================================================

# ── Access imported package modules ─────────────────────────────────
trainer_mod = trainer
ckpt_mod    = checkpoint
cfg_mod     = cfg
losses_mod  = losses

# ── Unpack conveniences ─────────────────────────────────────────────
build_optimizer   = trainer_mod.build_optimizer
make_train_step   = trainer_mod.make_train_step
train             = trainer_mod.train

build_ckpt_manager = ckpt_mod.build_ckpt_manager
do_checkpoint      = ckpt_mod.do_checkpoint
resume             = ckpt_mod.resume

DEFAULT_WEIGHTS    = losses_mod.DEFAULT_WEIGHTS

# ── Optimiser ────────────────────────────────────────────────────────
# Total steps: N_EPOCHS steps per profile, one full pass through all profiles.
# The schedule is shared across profiles so the LR keeps decaying globally.
total_steps = cfg_mod.N_EPOCHS * len(profiles)

tx = build_optimizer(
    total_steps = total_steps,
    lr          = cfg_mod.LR,          # 3e-4
    warmup      = 500,
    lr_end      = 5e-5,
)

# ── JIT-compiled training step ───────────────────────────────────────
# make_train_step closes over model and w_ops so neither is passed as
# a JIT argument (JAX cannot trace nn.Module or WOperators objects).
step_fn = make_train_step(
    model   = model,
    w_ops   = w_bundle.w_ops,
    weights = DEFAULT_WEIGHTS,
)

# ── Initialise fresh state ──────────────────────────────────────────
params_init = params                      # from Cell 3 build_model()
opt_state   = tx.init(params_init)

# ── CheckpointManager ────────────────────────────────────────────────
ckpt_mgr = build_ckpt_manager(CKPT_DIR, max_to_keep=3)

# ── Checkpoint save callback (passed into the training loop) ─────────
def _do_checkpoint(ep_global, prof_idx, stage_idx, ep_in_stage, best_data):
    """Thin wrapper so the training loop stays decoupled from CKPT_DIR."""
    do_checkpoint(
        mgr         = ckpt_mgr,
        params      = params,      # captured from outer scope (updated in-place)
        opt_state   = opt_state,   # same
        ep_global   = ep_global,
        prof_idx    = prof_idx,
        stage_idx   = stage_idx,
        ep_in_stage = ep_in_stage,
        best_data   = best_data,
        ckpt_dir    = CKPT_DIR,
    )

# NOTE: params and opt_state are reassigned by train() — the callback
# must reference them through the module-level names below (see the
# "mutable closure" pattern: reassign at cell scope, callback reads them).

# ── Resume from checkpoint (or start fresh) ──────────────────────────
bundle = resume(
    mgr       = ckpt_mgr,
    ckpt_dir  = CKPT_DIR,
    params    = params_init,
    opt_state = opt_state,
    tx        = tx,
)

params    = bundle.params
opt_state = bundle.opt_state

# ── Report parameter count ───────────────────────────────────────────
import jax
n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Parameters: {n_params:,}")

# ── Training ─────────────────────────────────────────────────────────
# The loop mutates params and opt_state and returns them at the end.
# The do_checkpoint callback is a closure that reads the CURRENT values
# of these variables — we update the cell-scope names after each call
# by reassigning from the returned tuple.

# History dict persists across re-runs in the same session so loss
# curves accumulate without gaps.
try:
    hist
except NameError:
    hist = {}

params, opt_state, hist, best_data = train(
    step_fn          = step_fn,
    params           = params,
    opt_state        = opt_state,
    tx               = tx,
    profiles         = profiles,
    R_flat           = R_PIX,
    Z_flat           = Z_PIX,
    esrc             = EDGES_SRC,
    edst             = EDGES_DST,
    ew               = EDGE_W,
    ndeg             = NODE_DEG,
    rho_2d           = RHO_2D,
    rho_flat         = RHO_FLAT,
    active_mask      = ACTIVE_MASK.astype(float),
    inject_noise     = inject_noise,
    stages           = cfg_mod.STAGES,
    ep_global        = bundle.ep_global,
    best_data        = bundle.best_data,
    start_prof       = bundle.start_prof,
    start_stage      = bundle.start_stage,
    start_ep         = bundle.start_ep,
    hist             = hist,
    do_checkpoint_fn = _do_checkpoint,
    log_every        = LOG_EVERY,
    save_every       = SAVE_EVERY,
)

# ── Final checkpoint ─────────────────────────────────────────────────
ep_global = bundle.ep_global + sum(s for s, _ in cfg_mod.STAGES) * len(profiles)
do_checkpoint(
    mgr         = ckpt_mgr,
    params      = params,
    opt_state   = opt_state,
    ep_global   = ep_global,
    prof_idx    = len(profiles),
    stage_idx   = 0,
    ep_in_stage = 0,
    best_data   = best_data,
    ckpt_dir    = CKPT_DIR,
)

# ── Quick loss-curve plot ─────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("VICTOR v6.0 — Training curves", fontweight="bold")

# Total loss
if "total" in hist and hist["total"]:
    axes[0].semilogy(hist["total"], lw=0.7, alpha=0.8)
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss (log)")
    axes[0].grid(True, alpha=0.3)

# Projection loss
if "proj" in hist and hist["proj"]:
    axes[1].semilogy(hist["proj"], lw=0.7, alpha=0.8, color="tab:orange")
    axes[1].set_title("Projection (data-fidelity) loss")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("loss (log)")
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/training_curves.png", dpi=150)
plt.show()
print(f"Curves saved → {RESULTS_DIR}/training_curves.png")

print("\n" + "="*50)
print("Cell 5 complete — training finished")
print(f"  Best projection loss : {best_data:.6f}")
print(f"  Checkpoint dir       : {CKPT_DIR}")
print("="*50)
