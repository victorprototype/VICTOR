# ============================================================
# VICTOR v6.0 — Cell 6: Evaluation + Plots
# ============================================================
# Depends on:
#   Cell 1 — installs, package imports, RESULTS_DIR
#             load_modules() in scope
#   Cell 2 — W matrix, geometry, profiles in cell scope
#   Cell 3 — model, params in cell scope
#   Cell 5 — hist, best_data from training (optional)
#
# Modules consumed:
#   evaluate.py — all metrics and plot functions
#
# What this cell does
# -------------------
#  1. Runs a full forward pass on every loaded profile
#  2. Runs a full forward pass on every loaded profile
#  3. Computes PSNR, CC, RelErr, sinogram MSE for each profile
#  4. Saves per-profile figures to RESULTS_DIR:
#       reconstruction_profile_NNN.png  — 4-panel (GT|Pred|Err|Uncert)
#       sinogram_profile_NNN.png        — measured vs predicted + residual
#       radial_profile_NNN.png          — ε vs ρ with uncertainty band
#       uncertainty_profile_NNN.png     — ensemble σ map
#       ensemble_members_profile_NNN.png— all M member predictions
#  5. Saves aggregate figures:
#       metrics_summary.png             — PSNR / CC / RelErr bar charts
#       loss_curves.png                 — training history (if hist exists)
#       metrics_summary.csv             — machine-readable metrics table
#  6. Displays inline previews of selected plots
# ============================================================

# ── Access imported evaluation module ───────────────────────
eval_mod = evaluate

# ── Expose evaluation functions at cell scope ─────────────
evaluate_profile   = eval_mod.evaluate_profile
evaluate_all       = eval_mod.evaluate_all
plot_all_profiles  = eval_mod.plot_all_profiles
plot_reconstruction= eval_mod.plot_reconstruction
plot_sinogram_residual = eval_mod.plot_sinogram_residual
plot_radial_profile    = eval_mod.plot_radial_profile
plot_uncertainty       = eval_mod.plot_uncertainty
plot_ensemble_members  = eval_mod.plot_ensemble_members
plot_loss_curves       = eval_mod.plot_loss_curves
plot_metrics_summary   = eval_mod.plot_metrics_summary
save_summary_csv       = eval_mod.save_summary_csv
EvalBundle             = eval_mod.EvalBundle
MetricBundle           = eval_mod.MetricBundle

# ── Geometry convenience objects ─────────────────────────────
# Constructed in Cell 2; referenced here by name.
# We re-use the cell-scope variables directly — no re-build needed.
cfg_mod  = cfg
geom_mod = geometry

from types import SimpleNamespace

_grids = SimpleNamespace(
    RHO_2D    = RHO_2D,
    RHO_FLAT  = RHO_FLAT,
    THETA_FLAT= THETA_FLAT,
    R_PIX     = R_PIX,
    Z_PIX     = Z_PIX,
)

_rho_graph = SimpleNamespace(
    EDGES_SRC = EDGES_SRC,
    EDGES_DST = EDGES_DST,
    EDGE_W    = EDGE_W,
    NODE_DEG  = NODE_DEG,
)

# ── Training history (graceful fallback) ─────────────────────
try:
    _hist = hist          # from Cell 5 training
except NameError:
    _hist = None
    print("Note: 'hist' not found in cell scope — loss curves will be skipped.")

# ── Run full evaluation + save all figures ────────────────────
print(f"\nEvaluation output directory: {RESULTS_DIR}")
print("=" * 55)

eval_bundles = plot_all_profiles(
    model       = model,
    params      = params,
    profiles    = profiles,
    w_ops       = w_bundle.w_ops,
    grids       = _grids,
    rho_graph   = _rho_graph,
    results_dir = RESULTS_DIR,
    hist        = _hist,
)

# ── Inline preview: reconstruction for first 3 profiles ──────
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

N_PREVIEW = min(3, len(eval_bundles))
print(f"\nInline previews ({N_PREVIEW} profiles):")

for eb in eval_bundles[:N_PREVIEW]:
    pid = eb.profile_idx

    # Reconstruction 4-panel
    fig = plot_reconstruction(
        eps_pred    = eb.eps_pred,
        eps_gt      = eb.eps_gt,
        std_2d      = eb.std_2d,
        rho_2d      = RHO_2D,
        R_pix       = R_PIX,
        Z_pix       = Z_PIX,
        metrics     = eb.metrics,
        profile_idx = pid,
    )
    plt.show()

    # Radial profile
    fig = plot_radial_profile(
        eps_pred    = eb.eps_pred,
        eps_gt      = eb.eps_gt,
        std_2d      = eb.std_2d,
        rho_flat    = RHO_FLAT,
        profile_idx = pid,
    )
    plt.show()

    # Sinogram residual
    fig = plot_sinogram_residual(
        g_pred      = eb.g_pred,
        g_gt        = eb.g_gt,
        active_mask = ACTIVE_MASK.astype(float),
        profile_idx = pid,
    )
    plt.show()

# ── Inline preview: metrics summary ─────────────────────────
fig = plot_metrics_summary(eval_bundles)
plt.show()

# ── Inline preview: loss curves (if available) ───────────────
if _hist is not None and any(len(v) > 0 for v in _hist.values()):
    fig = plot_loss_curves(_hist)
    plt.show()
else:
    print("Loss curve plot skipped (no training history in scope).")

# ── Final summary table ──────────────────────────────────────
import numpy as np

print("\n" + "=" * 60)
print("VICTOR v6.0 — Evaluation Summary")
print("=" * 60)
print(f"  {'Profile':>8}  {'PSNR [dB]':>10}  {'CC':>8}  {'RelErr':>8}  {'ProjMSE':>10}")
print(f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}")

for eb in eval_bundles:
    m = eb.metrics
    print(
        f"  {m.profile_idx:>8d}  "
        f"{m.psnr:>10.2f}  "
        f"{m.cc:>8.4f}  "
        f"{m.rel_err:>8.4f}  "
        f"{m.proj_mse:>10.2e}"
    )

psnrs   = [b.metrics.psnr    for b in eval_bundles]
ccs     = [b.metrics.cc      for b in eval_bundles]
rerrs   = [b.metrics.rel_err for b in eval_bundles]

print(f"  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}")
print(
    f"  {'mean':>8}  "
    f"{np.mean(psnrs):>10.2f}  "
    f"{np.mean(ccs):>8.4f}  "
    f"{np.mean(rerrs):>8.4f}"
)
print(
    f"  {'std':>8}  "
    f"{np.std(psnrs):>10.2f}  "
    f"{np.std(ccs):>8.4f}  "
    f"{np.std(rerrs):>8.4f}"
)
print("=" * 60)
print(f"\nAll figures and CSV saved to: {RESULTS_DIR}")
print("=" * 60)
print("Cell 6 complete — evaluation finished")
print("=" * 60)
