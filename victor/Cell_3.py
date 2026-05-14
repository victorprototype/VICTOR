# ============================================================
# VICTOR v6.0 — Cell 3: Build Model
# ============================================================
# Depends on Cell 2 (W matrix, geometry, profiles all loaded).
# Imports: model.py  (HashGrid, SIRENLayer, SO2Harmonics,
#                     SharedTrunk, MemberAdapter, PIGNO, VICTOR_v6)
# ============================================================

# ── Reload modules (picks up any Drive edits) ───────────────
cfg_mod = cfg
model_mod = model

# ── Build model from first-profile shapes ───────────────────
# Use profile[0] field arrays for init (same shape for all profiles).
_p0 = profiles[0]["psi_n"]    # (N_GRID²,) float32
_b0 = profiles[0]["bpol_n"]   # (N_GRID²,) float32

bundle = model_mod.build_model(
    R_flat   = R_PIX,
    Z_flat   = Z_PIX,
    psi_n    = _p0,
    bpol_n   = _b0,
    esrc     = EDGES_SRC,
    edst     = EDGES_DST,
    ew       = EDGE_W,
    ndeg     = NODE_DEG,
    rho_2d   = RHO_2D,
    seed     = 0,
)

# ── Verify: shapes + NaN check ──────────────────────────────
model_mod.verify_model(
    bundle,
    R_PIX, Z_PIX, _p0, _b0,
    EDGES_SRC, EDGES_DST, EDGE_W, NODE_DEG, RHO_2D,
)

# ── Expose at cell scope for downstream cells ────────────────
model  = bundle.model
params = bundle.params

print("\n" + "="*50)
print("Cell 3 complete — model ready")
print("="*50)
