# ============================================================
# VICTOR v7.0 — evaluate.py
# Evaluation metrics + diagnostic plots for FourierDeepONet
# ============================================================
# Public API
# ----------
#   compute_psnr(pred, gt, data_range)           → float
#   compute_cc(pred, gt)                         → float
#   compute_rel_err(pred, gt)                    → float
#   compute_all_metrics(...)                     → MetricBundle
#
#   evaluate_profile(model, params, profile,     → EvalBundle
#                    w_ops, grids, rho_graph)
#   evaluate_all(model, params, profiles,        → list[EvalBundle]
#                w_ops, grids, rho_graph)
#
#   plot_reconstruction_2d(eb, grids)            → Figure
#       3-panel per profile: GT | VICTOR Pred | Absolute Error
#       Full 2-D emissivity maps on the (R, Z) pixel grid with
#       flux-surface (ρ) contours overlaid.
#
#   plot_gt_gallery(eval_bundles, grids)         → Figure
#       All-profile ground-truth emissivity gallery in one figure.
#
#   plot_pred_gallery(eval_bundles, grids)       → Figure
#       All-profile VICTOR reconstruction gallery in one figure.
#
#   plot_comparison_grid(eval_bundles, grids)    → Figure
#       GT vs VICTOR side-by-side for every profile, plus error map.
#       The centrepiece visualisation.
#
#   plot_radial_profile(eb, grids)               → Figure
#       1-D radial profile comparison: raw model output vs GT binned
#       to the same radial axis.
#
#   plot_sinogram_residual(eb)                   → Figure
#       Measured vs predicted chord integrals + residual bar chart.
#
#   plot_metrics_summary(eval_bundles)           → Figure
#       PSNR / CC / RelErr bar charts across profiles.
#
#   plot_loss_curves(hist)                       → Figure
#       Training loss components on log scale.
#
#   plot_all_profiles(model, params, profiles,   → list[EvalBundle]
#                     w_ops, grids, rho_graph,
#                     results_dir, hist)
#       Full pipeline: evaluate + save all figures + CSV.
#
#   save_summary_csv(eval_bundles, path)         → None
#
# v7 changes vs v6
# ----------------
#  • Model now outputs eps1d : (N_RADIAL,) — a 1-D radial profile.
#    evaluate_profile() maps this back onto the 2-D pixel grid by
#    nearest-neighbour lookup in RHO_FLAT so all 2-D plots still work.
#  • No ensemble outputs (mean, std, preds) — EvalBundle simplified.
#  • N_ENS / cfg.N_ENS removed from all code paths.
#  • New centrepiece: plot_comparison_grid() — all profiles GT vs VICTOR.
#  • plot_gt_gallery() and plot_pred_gallery() for standalone galleries.
#  • plot_reconstruction_2d() replaces the old 4-panel (drops uncertainty).
#  • Radial plot uses grids.RHO_RADIAL axis (matches model output).
#  • All v6 ensemble / PIGNO / hash-grid references removed.
# ============================================================

from __future__ import annotations

import csv
import os
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import jax
import jax.numpy as jnp

from victor import config as cfg


# ══════════════════════════════════════════════════════════════════════
# 0.  Helpers
# ══════════════════════════════════════════════════════════════════════

def safe_numpy(x) -> np.ndarray:
    """Convert JAX array, jnp scalar, or plain numpy to a numpy array."""
    if hasattr(x, "__jax_array__") or hasattr(x, "device_buffer"):
        return np.asarray(x)
    if isinstance(x, (int, float)):
        return np.array(x)
    return np.asarray(x)


def _eps1d_to_2d(eps1d: np.ndarray, grids) -> np.ndarray:
    """
    Map a 1-D radial emissivity profile onto the 2-D pixel grid.

    Each pixel is assigned the emissivity of the radial bin whose ρ
    value is closest to that pixel's ρ.  Pixels outside the model's
    radial range (ρ > RHO_MAX) receive zero.

    Parameters
    ----------
    eps1d : (N_RADIAL,)  model output
    grids : PixelGrids   from geometry.build_pixel_grids()

    Returns
    -------
    np.ndarray  (N_GRID, N_GRID)  float32
    """
    N        = cfg.N_GRID
    rho_rad  = safe_numpy(grids.RHO_RADIAL)         # (N_RADIAL,)
    rho_flat = safe_numpy(grids.RHO_FLAT)           # (N_GRID²,)
    eps1d_np = safe_numpy(eps1d)                    # (N_RADIAL,)

    # Nearest-neighbour lookup: for each pixel find closest radial bin
    diffs   = np.abs(rho_flat[:, None] - rho_rad[None, :])  # (N², N_RADIAL)
    nn_idx  = np.argmin(diffs, axis=1)                       # (N²,)

    eps_2d_flat = eps1d_np[nn_idx]                           # (N²,)

    # Zero out pixels beyond the model's radial grid
    eps_2d_flat = np.where(rho_flat <= cfg.RHO_MAX, eps_2d_flat, 0.0)

    return eps_2d_flat.reshape(N, N).astype(np.float32)


def _rho_contour_overlay(ax, R_2d, Z_2d, rho_2d,
                          levels=(0.25, 0.5, 0.75, 1.0), lw=0.7):
    """Overlay ρ contours on an (R, Z) axis."""
    cs = ax.contour(
        R_2d, Z_2d, rho_2d,
        levels=list(levels),
        colors="white", linewidths=lw, alpha=0.6,
    )
    ax.clabel(cs, fmt={v: f"ρ={v:.2f}" for v in levels},
              fontsize=6, inline=True)


def _pcolormesh_with_cbar(ax, R_2d, Z_2d, data,
                           vmin, vmax, cmap, label=""):
    """pcolormesh + attached colorbar; returns (pcm, cbar)."""
    pcm  = ax.pcolormesh(R_2d, Z_2d, data,
                          vmin=vmin, vmax=vmax,
                          cmap=cmap, shading="auto")
    div  = make_axes_locatable(ax)
    cax  = div.append_axes("right", size="5%", pad=0.05)
    cbar = ax.get_figure().colorbar(pcm, cax=cax)
    if label:
        cbar.set_label(label, fontsize=8)
    return pcm, cbar


# ══════════════════════════════════════════════════════════════════════
# 1.  Named return types
# ══════════════════════════════════════════════════════════════════════

class MetricBundle(NamedTuple):
    """Per-profile scalar evaluation metrics."""
    profile_idx : int
    psnr        : float
    cc          : float
    rel_err     : float
    proj_mse    : float


class EvalBundle(NamedTuple):
    """Full per-profile evaluation outputs (v7 — no ensemble fields)."""
    profile_idx : int
    eps1d       : np.ndarray   # (N_RADIAL,)  raw model output
    eps_pred    : np.ndarray   # (N_GRID, N_GRID)  mapped to 2-D pixel grid
    eps_gt      : np.ndarray   # (N_GRID, N_GRID)  normalised ground truth
    g_pred      : np.ndarray   # (128,)  projected sinogram
    g_gt        : np.ndarray   # (128,)  ground-truth sinogram
    metrics     : MetricBundle


# ══════════════════════════════════════════════════════════════════════
# 2.  Scalar metrics
# ══════════════════════════════════════════════════════════════════════

def compute_psnr(
    pred       : np.ndarray,
    gt         : np.ndarray,
    data_range : float = 1.0,
) -> float:
    pred = safe_numpy(pred).astype(np.float64)
    gt   = safe_numpy(gt).astype(np.float64)
    mse  = np.mean((pred - gt) ** 2)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def compute_cc(pred: np.ndarray, gt: np.ndarray) -> float:
    pred   = safe_numpy(pred).flatten().astype(np.float64)
    gt     = safe_numpy(gt).flatten().astype(np.float64)
    pred_c = pred - pred.mean()
    gt_c   = gt   - gt.mean()
    denom  = np.sqrt(np.sum(pred_c ** 2) * np.sum(gt_c ** 2)) + 1e-12
    return float(np.sum(pred_c * gt_c) / denom)


def compute_rel_err(pred: np.ndarray, gt: np.ndarray) -> float:
    pred    = safe_numpy(pred).flatten().astype(np.float64)
    gt      = safe_numpy(gt).flatten().astype(np.float64)
    norm_gt = np.linalg.norm(gt)
    if norm_gt < 1e-12:
        return float(np.linalg.norm(pred - gt))
    return float(np.linalg.norm(pred - gt) / norm_gt)


def compute_all_metrics(
    eps_pred    : np.ndarray,
    eps_gt      : np.ndarray,
    g_pred      : np.ndarray,
    g_gt        : np.ndarray,
    active_mask : np.ndarray,
    profile_idx : int = 0,
) -> MetricBundle:
    eps_pred = safe_numpy(eps_pred)
    eps_gt   = safe_numpy(eps_gt)
    g_pred   = safe_numpy(g_pred)
    g_gt     = safe_numpy(g_gt)
    mask     = safe_numpy(active_mask).astype(bool)
    res      = g_pred[mask] - g_gt[mask]
    return MetricBundle(
        profile_idx = profile_idx,
        psnr        = compute_psnr(eps_pred, eps_gt),
        cc          = compute_cc(eps_pred, eps_gt),
        rel_err     = compute_rel_err(eps_pred, eps_gt),
        proj_mse    = float(np.mean(res ** 2)),
    )


# ══════════════════════════════════════════════════════════════════════
# 3.  Forward evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate_profile(
    model,
    params  : dict,
    profile : dict,
    w_ops,
    grids,
    rho_graph,          # kept for API compat — unused in v7
) -> EvalBundle:
    """
    Run a full v7 forward pass on one profile.

    v7 model: (g, xi) → eps1d (N_RADIAL,)
    The 1-D output is mapped back to 2-D for plotting via _eps1d_to_2d().

    Parameters
    ----------
    model     : FourierDeepONet instance
    params    : trained param tree
    profile   : dict from data_loader.load_profiles()
    w_ops     : WOperators  (for sinogram projection of 2-D output)
    grids     : PixelGrids
    rho_graph : RhoGraph    (unused in v7; kept for API compatibility)

    Returns
    -------
    EvalBundle
    """
    # ── Forward pass ─────────────────────────────────────────────────
    eps1d_jax = model.apply(
        params,
        profile["g_ideal"],   # (n_chords,)
        profile["xi"],        # (9,)
    )
    eps1d_np = safe_numpy(eps1d_jax)                    # (N_RADIAL,)

    # ── Map 1-D output → 2-D pixel grid ──────────────────────────────
    eps_pred_2d = _eps1d_to_2d(eps1d_np, grids)         # (N_GRID, N_GRID)

    # ── Ground truth ─────────────────────────────────────────────────
    eps_gt_2d   = safe_numpy(profile["eps_n"])           # (N_GRID, N_GRID)

    # ── Projected sinogram of 2-D reconstruction ─────────────────────
    g_pred_jax = w_ops.matvec(
        jnp.array(eps_pred_2d.flatten(), dtype=jnp.float32)
    )
    g_pred_np  = safe_numpy(g_pred_jax)                 # (128,)
    g_gt_np    = safe_numpy(profile["g_ideal"])         # (128,)

    active_mask = np.ones(128, dtype=np.float32)

    metrics = compute_all_metrics(
        eps_pred    = eps_pred_2d,
        eps_gt      = eps_gt_2d,
        g_pred      = g_pred_np,
        g_gt        = g_gt_np,
        active_mask = active_mask,
        profile_idx = int(profile.get("idx", 0)),
    )

    return EvalBundle(
        profile_idx = int(profile.get("idx", 0)),
        eps1d       = eps1d_np,
        eps_pred    = eps_pred_2d,
        eps_gt      = eps_gt_2d,
        g_pred      = g_pred_np,
        g_gt        = g_gt_np,
        metrics     = metrics,
    )


def evaluate_all(
    model,
    params   : dict,
    profiles : list,
    w_ops,
    grids,
    rho_graph,
) -> list:
    """Evaluate all profiles and return list[EvalBundle]."""
    bundles = []
    print(f"\nEvaluating {len(profiles)} profile(s)...")
    for prof in profiles:
        eb = evaluate_profile(model, params, prof, w_ops, grids, rho_graph)
        m  = eb.metrics
        print(
            f"  Profile {m.profile_idx:3d}: "
            f"PSNR={m.psnr:6.2f} dB  "
            f"CC={m.cc:.4f}  "
            f"RelErr={m.rel_err:.4f}  "
            f"ProjMSE={m.proj_mse:.2e}"
        )
        bundles.append(eb)

    psnrs  = [b.metrics.psnr    for b in bundles]
    ccs    = [b.metrics.cc      for b in bundles]
    rerrs  = [b.metrics.rel_err for b in bundles]
    print(f"\n  ── Aggregate (n={len(bundles)}) ────────────────────────")
    print(f"     PSNR    : {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")
    print(f"     CC      : {np.mean(ccs):.4f} ± {np.std(ccs):.4f}")
    print(f"     RelErr  : {np.mean(rerrs):.4f} ± {np.std(rerrs):.4f}")
    return bundles


# ══════════════════════════════════════════════════════════════════════
# 4.  Per-profile 2-D reconstruction plot  (GT | Pred | Error)
# ══════════════════════════════════════════════════════════════════════

def plot_reconstruction_2d(
    eb          : EvalBundle,
    grids,
    cmap        : str = "inferno",
    title_extra : str = "",
) -> plt.Figure:
    """
    Three-panel 2-D emissivity diagnostic for one profile:
      [Ground Truth ε  |  VICTOR Prediction ε  |  Absolute Error]

    ρ contours (0.25, 0.5, 0.75, 1.0) are overlaid on every panel.

    Parameters
    ----------
    eb    : EvalBundle  from evaluate_profile()
    grids : PixelGrids
    cmap  : colormap for emissivity panels (default "inferno")

    Returns
    -------
    matplotlib Figure
    """
    N      = cfg.N_GRID
    R_2d   = safe_numpy(grids.R_PIX).reshape(N, N)
    Z_2d   = safe_numpy(grids.Z_PIX).reshape(N, N)
    rho_2d = safe_numpy(grids.RHO_2D)

    gt   = eb.eps_gt
    pred = eb.eps_pred
    err  = np.abs(gt - pred)
    vmax = max(gt.max(), pred.max(), 1e-8)
    emax = err.max() or 1e-8

    m = eb.metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 0 — Ground truth
    _pcolormesh_with_cbar(axes[0], R_2d, Z_2d, gt,   0, vmax, cmap, "ε [a.u.]")
    _rho_contour_overlay(axes[0], R_2d, Z_2d, rho_2d)
    axes[0].set_title("Ground Truth ε", fontsize=12)
    axes[0].set_xlabel("R [m]"); axes[0].set_ylabel("Z [m]")
    axes[0].set_aspect("equal")

    # Panel 1 — VICTOR prediction
    _pcolormesh_with_cbar(axes[1], R_2d, Z_2d, pred, 0, vmax, cmap, "ε [a.u.]")
    _rho_contour_overlay(axes[1], R_2d, Z_2d, rho_2d)
    axes[1].set_title("VICTOR Reconstruction ε", fontsize=12)
    axes[1].set_xlabel("R [m]")
    axes[1].set_aspect("equal")

    # Panel 2 — Absolute error
    _pcolormesh_with_cbar(axes[2], R_2d, Z_2d, err,  0, emax, "hot", "|Δε| [a.u.]")
    _rho_contour_overlay(axes[2], R_2d, Z_2d, rho_2d, lw=0.5)
    axes[2].set_title("|GT − VICTOR|", fontsize=12)
    axes[2].set_xlabel("R [m]")
    axes[2].set_aspect("equal")

    suptitle = (
        f"VICTOR v7.0 — Profile {eb.profile_idx}"
        f"  PSNR={m.psnr:.1f} dB"
        f"  CC={m.cc:.3f}"
        f"  RelErr={m.rel_err:.3f}"
    )
    if title_extra:
        suptitle += f"  {title_extra}"
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 5.  Gallery: all-profile ground-truth 2-D emissivity
# ══════════════════════════════════════════════════════════════════════

def plot_gt_gallery(
    eval_bundles : list,
    grids,
    cmap         : str = "inferno",
    ncols        : int = 5,
) -> plt.Figure:
    """
    Single figure showing the ground-truth 2-D emissivity for every
    profile used in training.

    Each sub-panel has ρ contours and is labelled with the profile index.

    Parameters
    ----------
    eval_bundles : list[EvalBundle]
    grids        : PixelGrids
    cmap         : colormap (default "inferno")
    ncols        : columns in the grid (default 5)

    Returns
    -------
    matplotlib Figure
    """
    N      = cfg.N_GRID
    R_2d   = safe_numpy(grids.R_PIX).reshape(N, N)
    Z_2d   = safe_numpy(grids.Z_PIX).reshape(N, N)
    rho_2d = safe_numpy(grids.RHO_2D)

    n      = len(eval_bundles)
    ncols  = min(ncols, n)
    nrows  = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.2 * ncols, 4.5 * nrows),
                              squeeze=False)

    # Shared colour scale across all profiles
    vmax = max(eb.eps_gt.max() for eb in eval_bundles) or 1.0

    for i, eb in enumerate(eval_bundles):
        ax = axes[i // ncols][i % ncols]
        pcm = ax.pcolormesh(R_2d, Z_2d, eb.eps_gt,
                             vmin=0, vmax=vmax, cmap=cmap, shading="auto")
        _rho_contour_overlay(ax, R_2d, Z_2d, rho_2d, lw=0.6)
        ax.set_title(f"Profile {eb.profile_idx}", fontsize=10)
        ax.set_xlabel("R [m]", fontsize=8)
        ax.set_ylabel("Z [m]", fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    # Hide spare axes
    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    # Shared colorbar on the right
    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.35)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, vmax),
                              cmap=plt.get_cmap(cmap)),
        cax=cbar_ax, label="ε [a.u.]"
    )

    fig.suptitle("VICTOR v7.0 — Ground-Truth Emissivity (all profiles)",
                 fontsize=14, fontweight="bold", y=1.01)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 6.  Gallery: all-profile VICTOR reconstruction
# ══════════════════════════════════════════════════════════════════════

def plot_pred_gallery(
    eval_bundles : list,
    grids,
    cmap         : str = "inferno",
    ncols        : int = 5,
) -> plt.Figure:
    """
    Single figure showing the VICTOR 2-D reconstruction for every
    profile.  Shared colour scale with the GT gallery for direct
    visual comparison.

    Each sub-panel shows PSNR and CC below the title.

    Parameters
    ----------
    eval_bundles : list[EvalBundle]
    grids        : PixelGrids
    cmap         : colormap (default "inferno")
    ncols        : columns in the grid (default 5)

    Returns
    -------
    matplotlib Figure
    """
    N      = cfg.N_GRID
    R_2d   = safe_numpy(grids.R_PIX).reshape(N, N)
    Z_2d   = safe_numpy(grids.Z_PIX).reshape(N, N)
    rho_2d = safe_numpy(grids.RHO_2D)

    n      = len(eval_bundles)
    ncols  = min(ncols, n)
    nrows  = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.2 * ncols, 4.8 * nrows),
                              squeeze=False)

    vmax = max(eb.eps_gt.max() for eb in eval_bundles) or 1.0

    for i, eb in enumerate(eval_bundles):
        ax  = axes[i // ncols][i % ncols]
        m   = eb.metrics
        pcm = ax.pcolormesh(R_2d, Z_2d, eb.eps_pred,
                             vmin=0, vmax=vmax, cmap=cmap, shading="auto")
        _rho_contour_overlay(ax, R_2d, Z_2d, rho_2d, lw=0.6)
        ax.set_title(
            f"Profile {eb.profile_idx}\n"
            f"PSNR={m.psnr:.1f} dB  CC={m.cc:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("R [m]", fontsize=8)
        ax.set_ylabel("Z [m]", fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.subplots_adjust(right=0.88, hspace=0.45, wspace=0.35)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, vmax),
                              cmap=plt.get_cmap(cmap)),
        cax=cbar_ax, label="ε [a.u.]"
    )

    fig.suptitle("VICTOR v7.0 — VICTOR Reconstruction (all profiles)",
                 fontsize=14, fontweight="bold", y=1.01)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 7.  Centrepiece: GT vs VICTOR comparison grid (all profiles)
# ══════════════════════════════════════════════════════════════════════

def plot_comparison_grid(
    eval_bundles : list,
    grids,
    cmap         : str = "inferno",
) -> plt.Figure:
    """
    The centrepiece visualisation.

    For every profile, three columns are shown side-by-side:
      Col A — Ground Truth ε
      Col B — VICTOR Reconstruction ε
      Col C — Absolute Error |GT − VICTOR|

    Each row is one profile.  Shared colour scale per row for columns A/B;
    error column has its own per-row scale.  ρ contours on every panel.
    Metrics (PSNR, CC, RelErr) annotated on the right of each row.

    Parameters
    ----------
    eval_bundles : list[EvalBundle]
    grids        : PixelGrids
    cmap         : colormap for ε panels (default "inferno")

    Returns
    -------
    matplotlib Figure
    """
    N      = cfg.N_GRID
    R_2d   = safe_numpy(grids.R_PIX).reshape(N, N)
    Z_2d   = safe_numpy(grids.Z_PIX).reshape(N, N)
    rho_2d = safe_numpy(grids.RHO_2D)

    n_prof = len(eval_bundles)
    # 3 image cols + 1 text col, per row = 1 profile
    n_cols = 4
    fig    = plt.figure(figsize=(16, 4.8 * n_prof))
    gs     = gridspec.GridSpec(
        n_prof, n_cols,
        figure=fig,
        width_ratios=[1, 1, 1, 0.22],
        hspace=0.35,
        wspace=0.30,
    )

    col_headers = ["Ground Truth ε", "VICTOR Reconstruction ε", "|GT − VICTOR|"]

    for row, eb in enumerate(eval_bundles):
        gt   = eb.eps_gt
        pred = eb.eps_pred
        err  = np.abs(gt - pred)
        vmax = max(gt.max(), pred.max(), 1e-8)
        emax = err.max() or 1e-8
        m    = eb.metrics

        data_list  = [gt,   pred,  err]
        vmin_list  = [0,    0,     0  ]
        vmax_list  = [vmax, vmax,  emax]
        cmap_list  = [cmap, cmap,  "hot"]
        clabel_list= ["ε [a.u.]", "ε [a.u.]", "|Δε|"]

        for col in range(3):
            ax  = fig.add_subplot(gs[row, col])
            pcm = ax.pcolormesh(
                R_2d, Z_2d, data_list[col],
                vmin=vmin_list[col], vmax=vmax_list[col],
                cmap=cmap_list[col], shading="auto",
            )
            _rho_contour_overlay(ax, R_2d, Z_2d, rho_2d, lw=0.55)

            # Column headers on first row only
            if row == 0:
                ax.set_title(col_headers[col], fontsize=11, fontweight="bold")

            # Row label on leftmost column
            if col == 0:
                ax.set_ylabel(f"Profile {eb.profile_idx}\nZ [m]", fontsize=9)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

            if row == n_prof - 1:
                ax.set_xlabel("R [m]", fontsize=9)
            else:
                ax.tick_params(labelbottom=False)

            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)

            # Small inline colorbar
            div  = make_axes_locatable(ax)
            cax  = div.append_axes("right", size="5%", pad=0.04)
            cb   = fig.colorbar(pcm, cax=cax)
            cb.set_label(clabel_list[col], fontsize=7)
            cb.ax.tick_params(labelsize=6)

        # Metrics text panel (4th column)
        ax_txt = fig.add_subplot(gs[row, 3])
        ax_txt.axis("off")
        txt = (
            f"Profile {eb.profile_idx}\n\n"
            f"PSNR\n{m.psnr:.2f} dB\n\n"
            f"CC\n{m.cc:.4f}\n\n"
            f"RelErr\n{m.rel_err:.4f}\n\n"
            f"ProjMSE\n{m.proj_mse:.2e}"
        )
        ax_txt.text(
            0.1, 0.95, txt,
            transform=ax_txt.transAxes,
            fontsize=9, va="top", linespacing=1.6,
            family="monospace",
        )

    fig.suptitle(
        "VICTOR v7.0 — Ground Truth vs Reconstruction (all profiles)",
        fontsize=14, fontweight="bold", y=1.005,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
# 8.  Radial profile comparison
# ══════════════════════════════════════════════════════════════════════

def plot_radial_profile(
    eb          : EvalBundle,
    grids,
    n_bins      : int = 40,
) -> plt.Figure:
    """
    1-D radial emissivity comparison for one profile.

    Left panel  — raw model output eps1d on RHO_RADIAL axis vs
                  GT binned to the same axis.
    Right panel — pixel-binned comparison (eps_pred and eps_gt binned
                  by RHO_FLAT, inside plasma only).

    Parameters
    ----------
    eb     : EvalBundle
    grids  : PixelGrids
    n_bins : int   number of radial bins for the pixel-binned panel

    Returns
    -------
    matplotlib Figure
    """
    rho_rad  = safe_numpy(grids.RHO_RADIAL)   # (N_RADIAL,)
    rho_flat = safe_numpy(grids.RHO_FLAT)     # (N²,)
    eps1d    = eb.eps1d                        # (N_RADIAL,)
    eps_gt_f = eb.eps_gt.flatten()            # (N²,)
    eps_pr_f = eb.eps_pred.flatten()          # (N²,)

    # Bin GT and pred onto the same radial axis as the model
    def _bin_to_axis(values, rho_px, rho_axis):
        """Bin pixel values onto rho_axis bin edges."""
        edges   = np.concatenate([
            [rho_axis[0] - (rho_axis[1] - rho_axis[0]) / 2],
            (rho_axis[:-1] + rho_axis[1:]) / 2,
            [rho_axis[-1] + (rho_axis[-1] - rho_axis[-2]) / 2],
        ])
        means = np.zeros(len(rho_axis))
        for k in range(len(rho_axis)):
            sel = (rho_px >= edges[k]) & (rho_px < edges[k + 1])
            if sel.sum() > 0:
                means[k] = values[sel].mean()
        return means

    gt_binned = _bin_to_axis(eps_gt_f, rho_flat, rho_rad)

    # Right panel: pixel-binned (inside plasma only)
    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    inside      = rho_flat < 1.0

    def _simple_bin(vals):
        means = np.zeros(n_bins)
        for k in range(n_bins):
            sel = inside & (rho_flat >= bin_edges[k]) & (rho_flat < bin_edges[k + 1])
            if sel.sum() > 0:
                means[k] = vals[sel].mean()
        return means

    gt_pix  = _simple_bin(eps_gt_f)
    pr_pix  = _simple_bin(eps_pr_f)

    m   = eb.metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw model output vs GT binned to same axis
    ax1.plot(rho_rad, gt_binned, lw=2.0, color="steelblue",  label="GT (binned)")
    ax1.plot(rho_rad, eps1d,     lw=1.8, color="darkorange",
             linestyle="--", label="VICTOR eps1d (raw)")
    ax1.axvline(1.0, color="grey", lw=0.8, linestyle=":", label="ρ = 1 (LCFS)")
    ax1.set_xlabel("Normalised radius ρ", fontsize=11)
    ax1.set_ylabel("Emissivity [a.u.]",   fontsize=11)
    ax1.set_title("Model output vs GT on radial axis", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0.0, cfg.RHO_MAX)
    ax1.grid(True, alpha=0.3)

    # Right: pixel-binned inside plasma
    ax2.plot(bin_centres, gt_pix, lw=2.0, color="steelblue",  label="GT (pixel-binned)")
    ax2.plot(bin_centres, pr_pix, lw=1.8, color="darkorange",
             linestyle="--", label="VICTOR (pixel-binned)")
    ax2.set_xlabel("Normalised radius ρ", fontsize=11)
    ax2.set_ylabel("Emissivity [a.u.]",   fontsize=11)
    ax2.set_title("Pixel-binned comparison (ρ < 1)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.set_xlim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"VICTOR v7.0 — Radial profile  Profile {eb.profile_idx}"
        f"  PSNR={m.psnr:.1f} dB  CC={m.cc:.3f}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 9.  Sinogram residual
# ══════════════════════════════════════════════════════════════════════

def plot_sinogram_residual(
    eb          : EvalBundle,
    active_mask : Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Two-panel sinogram diagnostic:
      Top    — GT vs predicted chord integrals
      Bottom — residual (pred − GT) bar chart

    Parameters
    ----------
    eb          : EvalBundle
    active_mask : (128,) float, optional.  Defaults to all-ones.

    Returns
    -------
    matplotlib Figure
    """
    g_pred = eb.g_pred
    g_gt   = eb.g_gt
    if active_mask is None:
        active_mask = np.ones(128, dtype=np.float32)
    mask     = safe_numpy(active_mask).astype(bool)
    chords   = np.arange(len(g_pred))
    residual = g_pred - g_gt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(chords, g_gt,   lw=1.5, color="steelblue",  label="Ground truth g")
    ax1.plot(chords, g_pred, lw=1.2, color="darkorange",
             linestyle="--", label="VICTOR predicted g")
    for c in chords[~mask]:
        ax1.axvspan(c - 0.5, c + 0.5, color="grey", alpha=0.15)
    ax1.set_ylabel("Sinogram [a.u.]")
    ax1.set_title(f"Sinogram — Profile {eb.profile_idx}")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.bar(chords[mask],  residual[mask],  width=0.8,
            color="darkorange", alpha=0.75, label="active")
    ax2.bar(chords[~mask], residual[~mask], width=0.8,
            color="grey", alpha=0.4, label="inactive")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel("Chord index")
    ax2.set_ylabel("Residual (pred − GT)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 10.  Metrics summary bar chart
# ══════════════════════════════════════════════════════════════════════

def plot_metrics_summary(eval_bundles: list) -> plt.Figure:
    """Bar charts of PSNR, CC, and RelErr across all profiles."""
    idxs     = [b.metrics.profile_idx for b in eval_bundles]
    psnrs    = [b.metrics.psnr        for b in eval_bundles]
    ccs      = [b.metrics.cc          for b in eval_bundles]
    rel_errs = [b.metrics.rel_err     for b in eval_bundles]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(idxs))
    w = 0.65

    axes[0].bar(x, psnrs, width=w, color="steelblue", alpha=0.85)
    axes[0].set_title("PSNR [dB]  (↑ better)", fontsize=11)
    axes[0].set_xlabel("Profile"); axes[0].set_xticks(x)
    axes[0].set_xticklabels(idxs)
    axes[0].axhline(np.mean(psnrs), color="red", lw=1.2, linestyle="--",
                    label=f"mean={np.mean(psnrs):.1f} dB")
    axes[0].legend(fontsize=9); axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, ccs, width=w, color="darkorange", alpha=0.85)
    axes[1].set_title("Cross-Correlation  (↑ better)", fontsize=11)
    axes[1].set_xlabel("Profile"); axes[1].set_xticks(x)
    axes[1].set_xticklabels(idxs)
    axes[1].axhline(np.mean(ccs), color="red", lw=1.2, linestyle="--",
                    label=f"mean={np.mean(ccs):.3f}")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=9); axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(x, rel_errs, width=w, color="forestgreen", alpha=0.85)
    axes[2].set_title("Relative L2 Error  (↓ better)", fontsize=11)
    axes[2].set_xlabel("Profile"); axes[2].set_xticks(x)
    axes[2].set_xticklabels(idxs)
    axes[2].axhline(np.mean(rel_errs), color="red", lw=1.2, linestyle="--",
                    label=f"mean={np.mean(rel_errs):.3f}")
    axes[2].legend(fontsize=9); axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("VICTOR v7.0 — Evaluation Metrics Summary",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# 11.  Loss curves
# ══════════════════════════════════════════════════════════════════════

def plot_loss_curves(
    hist      : dict,
    save_path : Optional[str] = None,
) -> plt.Figure:
    """
    Multi-panel training loss curves from trainer.train() history dict.

    Panels shown (if key present): total, proj, boundary, smooth, positivity.
    """
    KEYS   = ["total", "proj", "boundary", "smooth", "positivity"]
    COLORS = ["black", "tab:orange", "tab:brown", "tab:green", "tab:red"]
    TITLES = ["Total loss", "Projection (data fidelity)",
              "Boundary enforcement", "TV smoothness", "Positivity penalty"]

    present = [(k, c, t) for k, c, t in zip(KEYS, COLORS, TITLES)
               if k in hist and len(hist[k]) > 0]

    if not present:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No loss history available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    ncols = min(len(present), 3)
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 3.8 * nrows),
                              squeeze=False)

    for idx, (key, color, title) in enumerate(present):
        ax  = axes[idx // ncols][idx % ncols]
        arr = np.array(hist[key], dtype=np.float64)
        xs  = np.arange(len(arr))
        ax.semilogy(xs, np.where(np.isfinite(arr), arr, np.nan),
                    lw=0.7, alpha=0.85, color=color)
        win = max(1, len(arr) // 20)
        finite = arr[np.isfinite(arr)]
        if len(finite) >= win:
            rm   = np.convolve(finite, np.ones(win) / win, mode="valid")
            xs_r = np.linspace(win // 2, len(arr) - win // 2, len(rm))
            ax.semilogy(xs_r, rm, lw=2.0, color=color, alpha=0.5,
                        label="rolling mean")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step"); ax.set_ylabel("loss (log)")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    for idx in range(len(present), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("VICTOR v7.0 — Training loss curves",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Loss curves saved → {save_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════
# 12.  Full pipeline
# ══════════════════════════════════════════════════════════════════════

def plot_all_profiles(
    model,
    params      : dict,
    profiles    : list,
    w_ops,
    grids,
    rho_graph,
    results_dir : str            = cfg.RESULTS_DIR,
    hist        : Optional[dict] = None,
) -> list:
    """
    Full evaluation pipeline:
      1. evaluate_all() — forward pass on every profile
      2. plot_comparison_grid() — centrepiece GT vs VICTOR for all profiles
      3. plot_gt_gallery()      — all GT maps in one figure
      4. plot_pred_gallery()    — all VICTOR maps in one figure
      5. Per-profile:
           plot_reconstruction_2d()  — 3-panel (GT|Pred|Error)
           plot_radial_profile()     — 1-D radial comparison
           plot_sinogram_residual()  — chord-by-chord residual
      6. plot_metrics_summary()     — PSNR/CC/RelErr bar charts
      7. plot_loss_curves()         — training history (if hist provided)
      8. save_summary_csv()         — metrics CSV

    All figures saved as PNG to results_dir/.

    Returns
    -------
    list[EvalBundle]
    """
    os.makedirs(results_dir, exist_ok=True)

    # ── Evaluate ──────────────────────────────────────────────────────
    eval_bundles = evaluate_all(model, params, profiles, w_ops, grids, rho_graph)

    # ── Centrepiece: comparison grid ──────────────────────────────────
    fig = plot_comparison_grid(eval_bundles, grids)
    path = os.path.join(results_dir, "comparison_grid_all_profiles.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── GT gallery ────────────────────────────────────────────────────
    fig = plot_gt_gallery(eval_bundles, grids)
    path = os.path.join(results_dir, "gallery_ground_truth.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Prediction gallery ────────────────────────────────────────────
    fig = plot_pred_gallery(eval_bundles, grids)
    path = os.path.join(results_dir, "gallery_victor_reconstruction.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Per-profile figures ───────────────────────────────────────────
    for eb in eval_bundles:
        pid = eb.profile_idx

        fig = plot_reconstruction_2d(eb, grids)
        path = os.path.join(results_dir, f"reconstruction_profile_{pid:03d}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        fig = plot_radial_profile(eb, grids)
        path = os.path.join(results_dir, f"radial_profile_{pid:03d}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        fig = plot_sinogram_residual(eb)
        path = os.path.join(results_dir, f"sinogram_profile_{pid:03d}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── Metrics summary ───────────────────────────────────────────────
    fig  = plot_metrics_summary(eval_bundles)
    path = os.path.join(results_dir, "metrics_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Loss curves ───────────────────────────────────────────────────
    if hist is not None:
        path = os.path.join(results_dir, "loss_curves.png")
        fig  = plot_loss_curves(hist, save_path=path)
        plt.close(fig)

    # ── CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "metrics_summary.csv")
    save_summary_csv(eval_bundles, csv_path)

    print(f"\nAll figures saved to: {results_dir}")
    return eval_bundles


# ══════════════════════════════════════════════════════════════════════
# 13.  CSV export
# ══════════════════════════════════════════════════════════════════════

def save_summary_csv(eval_bundles: list, path: str) -> None:
    """Write per-profile metrics to CSV."""
    rows = [("profile_idx", "psnr_dB", "cc", "rel_err", "proj_mse")]
    for eb in eval_bundles:
        m = eb.metrics
        rows.append((
            m.profile_idx,
            round(m.psnr,     4),
            round(m.cc,       6),
            round(m.rel_err,  6),
            round(m.proj_mse, 8),
        ))
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Metrics CSV saved → {path}")


# ══════════════════════════════════════════════════════════════════════
# 14.  Module self-test
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from types import SimpleNamespace

    N        = cfg.N_GRID
    N_RADIAL = cfg.N_RADIAL
    rng      = np.random.default_rng(0)

    lin    = np.linspace(-cfg.EXT, cfg.EXT, N)
    XX, YY = np.meshgrid(lin, lin)
    rho_2d = np.sqrt((XX / cfg.AP)**2 + (YY / cfg.BP)**2).astype(np.float32)

    grids = SimpleNamespace(
        R_PIX     = (cfg.R0 + XX).flatten().astype(np.float32),
        Z_PIX     = YY.flatten().astype(np.float32),
        RHO_2D    = rho_2d,
        RHO_FLAT  = rho_2d.flatten(),
        RHO_RADIAL= np.linspace(0.0, cfg.RHO_MAX, N_RADIAL).astype(np.float32),
    )

    eps1d  = rng.random(N_RADIAL).astype(np.float32)
    eps_gt = rng.random((N, N)).astype(np.float32)
    g      = rng.random(128).astype(np.float32)

    eb = EvalBundle(
        profile_idx = 0,
        eps1d       = eps1d,
        eps_pred    = _eps1d_to_2d(eps1d, grids),
        eps_gt      = eps_gt,
        g_pred      = g,
        g_gt        = g * 0.9,
        metrics     = MetricBundle(0, 25.0, 0.92, 0.08, 1e-4),
    )

    fig = plot_reconstruction_2d(eb, grids); plt.close(fig)
    print("plot_reconstruction_2d OK")

    fig = plot_gt_gallery([eb, eb], grids); plt.close(fig)
    print("plot_gt_gallery OK")

    fig = plot_pred_gallery([eb, eb], grids); plt.close(fig)
    print("plot_pred_gallery OK")

    fig = plot_comparison_grid([eb, eb], grids); plt.close(fig)
    print("plot_comparison_grid OK")

    fig = plot_radial_profile(eb, grids); plt.close(fig)
    print("plot_radial_profile OK")

    fig = plot_sinogram_residual(eb); plt.close(fig)
    print("plot_sinogram_residual OK")

    fig = plot_metrics_summary([eb, eb]); plt.close(fig)
    print("plot_metrics_summary OK")

    hist_dummy = {
        "total": list(np.exp(-np.linspace(0, 5, 300))),
        "proj":  list(np.exp(-np.linspace(0, 5, 300)) * 0.8),
    }
    fig = plot_loss_curves(hist_dummy); plt.close(fig)
    print("plot_loss_curves OK")

    print("\nevaluate.py v7 self-test PASSED")
