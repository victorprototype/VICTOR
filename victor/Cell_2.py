# ============================================================
# VICTOR v6.0 — Cell 2: Load W matrix + geometry + profiles
# ============================================================
# Depends on Cell 1 (installs, Drive mount, MODULES_DIR on sys.path).
# Calls: geometry.py, data_loader.py, config.py
# ============================================================

# ── Reload modules (picks up any Drive edits) ───────────────
modules = load_modules(force_reload=True)

cfg         = modules["config"]
geom        = modules["geometry"]
data_loader = modules["data_loader"]

# ── Propagate Drive path into config ────────────────────────
cfg.update_paths(DRIVE)

# ── Load W matrix ────────────────────────────────────────────
w_bundle = data_loader.load_W_matrix()

# Expose operators at cell scope for downstream cells
W_BCOO       = w_bundle.W_BCOO
ACTIVE_MASK  = w_bundle.ACTIVE_MASK
N_ACTIVE     = w_bundle.N_ACTIVE
W_matvec     = w_bundle.w_ops.matvec
W_vecmat     = w_bundle.w_ops.vecmat

# ── Build geometry ───────────────────────────────────────────
grids, rays, rho_graph, _ = geom.build_all_geometry()

# Expose geometry arrays at cell scope
RHO_2D     = grids.RHO_2D
RHO_FLAT   = grids.RHO_FLAT
THETA_FLAT = grids.THETA_FLAT
R_PIX      = grids.R_PIX
Z_PIX      = grids.Z_PIX

RAY_R      = rays.RAY_R
RAY_Z      = rays.RAY_Z
RAY_DS     = rays.RAY_DS
RAY_V      = rays.RAY_V
MAX_STEPS  = rays.MAX_STEPS

EDGES_SRC  = rho_graph.EDGES_SRC
EDGES_DST  = rho_graph.EDGES_DST
EDGE_W     = rho_graph.EDGE_W
NODE_DEG   = rho_graph.NODE_DEG

# ── Load profiles ─────────────────────────────────────────────
profiles = data_loader.load_profiles(
    w_bundle = w_bundle,
    grids    = grids,
)

# ── Noise helper (re-export for downstream cells) ─────────────
inject_noise = data_loader.inject_noise

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*50)
print(f"Cell 2 complete")
print(f"  W        : {w_bundle.W_norm.shape}  NNZ={w_bundle.W_norm.nnz}  active={N_ACTIVE}/128")
print(f"  Grid     : {RHO_2D.shape}  R∈[{float(R_PIX.min()):.2f},{float(R_PIX.max()):.2f}]m")
print(f"  Rays     : {RAY_R.shape[0]}×{MAX_STEPS}")
print(f"  Edges    : {len(EDGES_SRC)}")
print(f"  Profiles : {len(profiles)}/{cfg.N_PROFILES}")
print("="*50)
