# ============================================================
# VICTOR v6.0 — Cell 4: Build Loss Functions
# ============================================================
# Depends on Cell 3 (model, params, w_bundle, grids, rho_graph,
#                    profiles all in scope).
# Imports: losses.py  (loss_projection, loss_boundary,
#                      loss_smoothness, loss_ensemble_nll,
#                      loss_ensemble_diversity, loss_isotropy,
#                      loss_positivity, loss_fn, verify_losses)
# ============================================================

# ── Reload modules (picks up any Drive edits) ───────────────
losses_mod = losses

# ── Expose individual loss functions at cell scope ───────────
loss_projection      = losses_mod.loss_projection
loss_boundary        = losses_mod.loss_boundary
loss_smoothness      = losses_mod.loss_smoothness
loss_ensemble_nll    = losses_mod.loss_ensemble_nll
loss_ensemble_diversity = losses_mod.loss_ensemble_diversity
loss_isotropy        = losses_mod.loss_isotropy
loss_positivity      = losses_mod.loss_positivity
loss_fn              = losses_mod.loss_fn
LossWeights          = losses_mod.LossWeights
DEFAULT_WEIGHTS      = losses_mod.DEFAULT_WEIGHTS

# ── Verify all loss components on profile[0] ─────────────────
losses_mod.verify_losses(
    model     = model,
    params    = params,
    profile   = profiles[0],
    w_ops     = w_bundle.w_ops,
    grids     = grids,
    rho_graph = rho_graph,
)

# ── Print active loss weights ─────────────────────────────────
print("\nActive loss weights:")
for field, val in DEFAULT_WEIGHTS._asdict().items():
    print(f"  {field:<18}: {val}")

print("\n" + "="*50)
print("Cell 4 complete — loss functions ready")
print("="*50)
