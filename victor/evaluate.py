# ============================================================
# VICTOR v6.0 — evaluate.py
# Evaluation metrics (PSNR, CC, RelErr) + all diagnostic plots
# ============================================================
# Public API
# ----------
#   compute_psnr(pred, gt, data_range)          → float
#   compute_cc(pred, gt)                        → float
#   compute_rel_err(pred, gt)                   → float
#   compute_all_metrics(pred, gt, label)        → MetricBundle
#
#   plot_reconstruction(pred, gt, profile, ...)  → Figure
#   plot_sinogram_residual(pred, profile, ...)   → Figure
#   plot_radial_profile(pred, profile, ...)      → Figure
#   plot_uncertainty(std_2d, rho_2d, ...)        → Figure
#   plot_ensemble_members(preds, ...)            → Figure
#   plot_loss_curves(hist, ...)                  → Figure
#   plot_all_profiles(model, params, profiles,   → list[MetricBundle]
#                     w_ops, grids, rho_graph)
#
#   evaluate_profile(model, params, profile,     → EvalBundle
#                    w_ops, grids, rho_graph)
#   evaluate_all(model, params, profiles,        → list[EvalBundle]
#                w_ops, grids, rho_graph)
#   save_summary_csv(eval_bundles, path)         → None
#
# Design principles
# -----------------
#  • All functions are pure (no JAX globals mutated).
#  • Figures are returned — callers decide whether to show/save.
#  • Physics-aware plots: reconstruction uses rho contours; sinogram
#    plots annotate active-chord boundaries.
#  • safe_numpy() converts JAX arrays / scalars to plain NumPy to avoid
#    matplotlib dtype warnings.
#  • All axes are labelled with units; colorbars are added where needed.
# ============================================================

from __future__ import annotations

import os
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import jax
import jax.numpy as jnp

from victor import config as cfg


# ── Helpers ───────────────────────────────────────────────────────────

def safe_numpy(x) -> np.ndarray:
    """Convert JAX array, jnp scalar, or plain numpy to a numpy array."""
    if hasattr(x, "__jax_array__") or hasattr(x, "device_buffer"):
        return np.asarray(x)
    if isinstance(x, (int, float)):
        return np.array(x)
    return np.asarray(x)


# ── Named return types ────────────────────────────────────────────────

class MetricBundle(NamedTuple):
    """Per-profile scalar evaluation metrics."""
    profile_idx : int
    psnr        : float
    cc          : float
    rel_err     : float
    proj_mse    : float   # sinogram MSE on active chords (data fidelity)


class EvalBundle(NamedTuple):
    """Full per-profile evaluation outputs."""
    profile_idx : int
    eps_pred    : np.ndarray   # (N, N)  final model output
    eps_gt      : np.ndarray   # (N, N)  normalised ground truth
    mean_2d     : np.ndarray   # (N, N)  ensemble mean (pre-PIGNO)
    std_2d      : np.ndarray   # (N, N)  ensemble std  (pre-PIGNO)
    preds_2d    : np.ndarray   # (M, N, N)  per-member predictions
    g_pred      : np.ndarray   # (128,)  projected sinogram
    g_gt        : np.ndarray   # (128,)  ground-truth sinogram
    metrics     : MetricBundle


# ═══════════════════════════════════════════════════════════════════════
# 1.  Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_psnr(
    pred       : np.ndarray,
    gt         : np.ndarray,
    data_range : float = 1.0,
) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).

    PSNR = 10 · log₁₀(data_range² / MSE)

    Parameters
    ----------
    pred       : (N, N)  predicted image
    gt         : (N, N)  ground-truth image
    data_range : float   value range of the images (default 1.0)

    Returns
    -------
    float  (positive; higher is better; ∞ if pred == gt)
    """
    pred = safe_numpy(pred).astype(np.float64)
    gt   = safe_numpy(gt).astype(np.float64)
    mse  = np.mean((pred - gt) ** 2)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def compute_cc(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Pearson cross-correlation coefficient.

    CC = Cov(pred, gt) / (σ_pred · σ_gt)

    Parameters
    ----------
    pred : (N, N)  predicted image
    gt   : (N, N)  ground-truth image

    Returns
    -------
    float in [-1, 1]  (1.0 is perfect)
    """
    pred = safe_numpy(pred).flatten().astype(np.float64)
    gt   = safe_numpy(gt).flatten().astype(np.float64)

    pred_c = pred - pred.mean()
    gt_c   = gt   - gt.mean()

    denom = np.sqrt(np.sum(pred_c ** 2) * np.sum(gt_c ** 2)) + 1e-12
    return float(np.sum(pred_c * gt_c) / denom)


def compute_rel_err(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Relative L2 error.

    RelErr = ‖pred - gt‖₂ / ‖gt‖₂

    Parameters
    ----------
    pred : (N, N)  predicted image
    gt   : (N, N)  ground-truth image

    Returns
    -------
    float ≥ 0  (0.0 is perfect)
    """
    pred = safe_numpy(pred).flatten().astype(np.float64)
    gt   = safe_numpy(gt).flatten().astype(np.float64)
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
    """
    Compute PSNR, CC, RelErr, and sinogram MSE for one profile.

    Parameters
    ----------
    eps_pred    : (N, N)   predicted emissivity
    eps_gt      : (N, N)   ground-truth emissivity (normalised)
    g_pred      : (128,)   projected sinogram from model
    g_gt        : (128,)   ground-truth sinogram
    active_mask : (128,)   1 for active chords
    profile_idx : int      index label

    Returns
    -------
    MetricBundle
    """
    eps_pred = safe_numpy(eps_pred)
    eps_gt   = safe_numpy(eps_gt)
    g_pred   = safe_numpy(g_pred)
    g_gt     = safe_numpy(g_gt)
    mask     = safe_numpy(active_mask).astype(bool)

    # Sinogram MSE restricted to active chords
    res      = (g_pred[mask] - g_gt[mask])
    proj_mse = float(np.mean(res ** 2))

    return MetricBundle(
        profile_idx = profile_idx,
        psnr        = compute_psnr(eps_pred, eps_gt, data_range=1.0),
        cc          = compute_cc(eps_pred, eps_gt),
        rel_err     = compute_rel_err(eps_pred, eps_gt),
        proj_mse    = proj_mse,
    )


# ═══════════════════════════════════════════════════════════════════════
# 2.  Reconstruction plot  (GT | Pred | Error | Uncertainty)
# ═══════════════════════════════════════════════════════════════════════

def plot_reconstruction(
    eps_pred   : np.ndarray,
    eps_gt     : np.ndarray,
    std_2d     : np.ndarray,
    rho_2d     : np.ndarray,
    R_pix      : np.ndarray,
    Z_pix      : np.ndarray,
    metrics    : Optional[MetricBundle] = None,
    profile_idx: int = 0,
    title_extra: str = "",
    cmap       : str = "inferno",
) -> plt.Figure:
    """
    Four-panel reconstruction diagnostic:
      [Ground Truth | Prediction | Absolute Error | Ensemble Uncertainty]

    Rho contours (ρ = 0.25, 0.5, 0.75, 1.0) are overlaid on each panel.

    Parameters
    ----------
    eps_pred    : (N, N)   predicted emissivity (masked)
    eps_gt      : (N, N)   ground-truth emissivity
    std_2d      : (N, N)   ensemble std map (pre-PIGNO, reshaped)
    rho_2d      : (N, N)   normalised elliptic radius
    R_pix       : (N²,)    major radius pixel centres [m]
    Z_pix       : (N²,)    vertical pixel centres [m]
    metrics     : MetricBundle, optional — used in suptitle
    profile_idx : int
    title_extra : str       appended to the figure title
    cmap        : str       matplotlib colormap for emissivity panels

    Returns
    -------
    matplotlib Figure
    """
    N        = cfg.N_GRID
    eps_pred = safe_numpy(eps_pred)
    eps_gt   = safe_numpy(eps_gt)
    std_2d   = safe_numpy(std_2d).reshape(N, N)
    rho_2d   = safe_numpy(rho_2d)
    R_2d     = safe_numpy(R_pix).reshape(N, N)
    Z_2d     = safe_numpy(Z_pix).reshape(N, N)

    abs_err  = np.abs(eps_pred - eps_gt)
    vmax     = max(eps_gt.max(), eps_pred.max(), 1e-8)
    emax     = abs_err.max() or 1e-8

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    rho_levels = [0.25, 0.50, 0.75, 1.00]

    def _add_rho_contours(ax, lw=0.6):
        cs = ax.contour(
            R_2d, Z_2d, rho_2d,
            levels=rho_levels,
            colors="white", linewidths=lw, alpha=0.55,
        )
        ax.clabel(cs, fmt={v: f"ρ={v:.2f}" for v in rho_levels},
                  fontsize=6, inline=True)

    def _pcolormesh(ax, data, vmin, vmax, cmap_):
        pcm = ax.pcolormesh(
            R_2d, Z_2d, data,
            vmin=vmin, vmax=vmax, cmap=cmap_, shading="auto",
        )
        return pcm

    # Panel 0: Ground truth
    pcm = _pcolormesh(axes[0], eps_gt, 0, vmax, cmap)
    _add_rho_contours(axes[0])
    plt.colorbar(pcm, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title("Ground Truth ε")
    axes[0].set_xlabel("R [m]"); axes[0].set_ylabel("Z [m]")
    axes[0].set_aspect("equal")

    # Panel 1: Prediction
    pcm = _pcolormesh(axes[1], eps_pred, 0, vmax, cmap)
    _add_rho_contours(axes[1])
    plt.colorbar(pcm, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("VICTOR Prediction ε")
    axes[1].set_xlabel("R [m]"); axes[1].set_aspect("equal")

    # Panel 2: Absolute error
    pcm = _pcolormesh(axes[2], abs_err, 0, emax, "hot")
    _add_rho_contours(axes[2], lw=0.5)
    plt.colorbar(pcm, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("|GT − Pred|")
    axes[2].set_xlabel("R [m]"); axes[2].set_aspect("equal")

    # Panel 3: Uncertainty (ensemble std)
    umax = max(std_2d.max(), 1e-8)
    pcm  = _pcolormesh(axes[3], std_2d, 0, umax, "viridis")
    _add_rho_contours(axes[3], lw=0.5)
    plt.colorbar(pcm, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].set_title("Ensemble σ (uncertainty)")
    axes[3].set_xlabel("R [m]"); axes[3].set_aspect("equal")

    # Suptitle
    metric_str = ""
    if metrics is not None:
        metric_str = (
            f"  PSNR={metrics.psnr:.1f} dB"
            f"  CC={metrics.cc:.3f}"
            f"  RelErr={metrics.rel_err:.3f}"
        )
    title = f"VICTOR v6.0 — Profile {profile_idx}{metric_str}"
    if title_extra:
        title += f"  {title_extra}"
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 3.  Sinogram residual plot
# ═══════════════════════════════════════════════════════════════════════

def plot_sinogram_residual(
    g_pred      : np.ndarray,
    g_gt        : np.ndarray,
    active_mask : np.ndarray,
    profile_idx : int = 0,
) -> plt.Figure:
    """
    Two-panel sinogram diagnostic:
      Top   — measured vs predicted sinogram per chord
      Bottom — residual  (pred − gt) per chord

    Active vs inactive chords are annotated.

    Parameters
    ----------
    g_pred      : (128,)  predicted sinogram
    g_gt        : (128,)  ground-truth sinogram
    active_mask : (128,)  float/bool — 1 for active chords
    profile_idx : int

    Returns
    -------
    matplotlib Figure
    """
    g_pred = safe_numpy(g_pred)
    g_gt   = safe_numpy(g_gt)
    mask   = safe_numpy(active_mask).astype(bool)

    chords   = np.arange(len(g_pred))
    residual = g_pred - g_gt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Top: signals
    ax1.plot(chords, g_gt,   lw=1.5, label="Ground truth g", color="steelblue")
    ax1.plot(chords, g_pred, lw=1.2, label="Predicted g", color="darkorange",
             linestyle="--")
    # Shade inactive chords
    for c in chords[~mask]:
        ax1.axvspan(c - 0.5, c + 0.5, color="grey", alpha=0.15)
    ax1.set_ylabel("Sinogram [a.u.]")
    ax1.set_title(f"Sinogram — Profile {profile_idx}")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bottom: residual
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


# ═══════════════════════════════════════════════════════════════════════
# 4.  Radial profile plot  (ε vs ρ)
# ═══════════════════════════════════════════════════════════════════════

def plot_radial_profile(
    eps_pred  : np.ndarray,
    eps_gt    : np.ndarray,
    std_2d    : np.ndarray,
    rho_flat  : np.ndarray,
    profile_idx: int = 0,
    n_bins    : int  = 40,
) -> plt.Figure:
    """
    Radial profile comparison: binned ε vs ρ for GT and prediction,
    with ensemble ±1σ uncertainty band.

    Parameters
    ----------
    eps_pred    : (N, N)   predicted emissivity
    eps_gt      : (N, N)   ground-truth emissivity
    std_2d      : (N, N)   ensemble std (pre-PIGNO)
    rho_flat    : (N²,)    normalised elliptic radius (flattened)
    profile_idx : int
    n_bins      : int      number of radial bins

    Returns
    -------
    matplotlib Figure
    """
    eps_pred = safe_numpy(eps_pred).flatten()
    eps_gt   = safe_numpy(eps_gt).flatten()
    std_flat = safe_numpy(std_2d).flatten()
    rho      = safe_numpy(rho_flat).flatten()

    # Keep only plasma interior (ρ < 1)
    inside   = rho < 1.0
    rho      = rho[inside]
    eps_pred = eps_pred[inside]
    eps_gt   = eps_gt[inside]
    std_flat = std_flat[inside]

    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def _bin_mean_std(vals):
        means = np.zeros(n_bins)
        stds  = np.zeros(n_bins)
        for k in range(n_bins):
            sel = (rho >= bin_edges[k]) & (rho < bin_edges[k + 1])
            if sel.sum() > 0:
                means[k] = vals[sel].mean()
                stds[k]  = vals[sel].std()
        return means, stds

    gt_mean,   gt_std   = _bin_mean_std(eps_gt)
    pred_mean, pred_std = _bin_mean_std(eps_pred)
    unc_mean,  _        = _bin_mean_std(std_flat)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(bin_centres, gt_mean,   lw=2.0, color="steelblue",  label="Ground Truth")
    ax.fill_between(bin_centres,
                    gt_mean - gt_std, gt_mean + gt_std,
                    alpha=0.2, color="steelblue")

    ax.plot(bin_centres, pred_mean, lw=1.8, color="darkorange",
            linestyle="--", label="VICTOR Pred.")
    ax.fill_between(bin_centres,
                    pred_mean - unc_mean, pred_mean + unc_mean,
                    alpha=0.25, color="darkorange", label="Ensemble ±σ")

    ax.set_xlabel("Normalised radius ρ", fontsize=11)
    ax.set_ylabel("Emissivity [a.u.]",  fontsize=11)
    ax.set_title(f"Radial emissivity profile — Profile {profile_idx}", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 5.  Uncertainty map
# ═══════════════════════════════════════════════════════════════════════

def plot_uncertainty(
    std_2d      : np.ndarray,
    rho_2d      : np.ndarray,
    R_pix       : np.ndarray,
    Z_pix       : np.ndarray,
    profile_idx : int = 0,
) -> plt.Figure:
    """
    Standalone uncertainty-map panel with rho contours.

    Parameters
    ----------
    std_2d      : (N, N)   ensemble std map
    rho_2d      : (N, N)   normalised elliptic radius
    R_pix       : (N²,)    major radius [m]
    Z_pix       : (N²,)    vertical coord [m]
    profile_idx : int

    Returns
    -------
    matplotlib Figure
    """
    N      = cfg.N_GRID
    std_2d = safe_numpy(std_2d).reshape(N, N)
    rho_2d = safe_numpy(rho_2d)
    R_2d   = safe_numpy(R_pix).reshape(N, N)
    Z_2d   = safe_numpy(Z_pix).reshape(N, N)

    fig, ax = plt.subplots(figsize=(6, 6))
    pcm = ax.pcolormesh(
        R_2d, Z_2d, std_2d,
        vmin=0, vmax=std_2d.max() or 1e-8, cmap="plasma", shading="auto",
    )
    plt.colorbar(pcm, ax=ax, label="Ensemble σ [a.u.]")
    cs = ax.contour(R_2d, Z_2d, rho_2d,
                    levels=[0.25, 0.5, 0.75, 1.0],
                    colors="white", linewidths=0.7, alpha=0.6)
    ax.clabel(cs, fmt="%.2f", fontsize=7, inline=True)
    ax.set_title(f"Ensemble Uncertainty — Profile {profile_idx}", fontsize=12)
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 6.  Ensemble members
# ═══════════════════════════════════════════════════════════════════════

def plot_ensemble_members(
    preds_2d    : np.ndarray,
    R_pix       : np.ndarray,
    Z_pix       : np.ndarray,
    profile_idx : int = 0,
    cmap        : str = "inferno",
) -> plt.Figure:
    """
    Grid plot of all M ensemble member predictions.

    Parameters
    ----------
    preds_2d    : (M, N, N)  per-member emissivity
    R_pix       : (N²,)
    Z_pix       : (N²,)
    profile_idx : int
    cmap        : str

    Returns
    -------
    matplotlib Figure
    """
    N        = cfg.N_GRID
    preds_2d = safe_numpy(preds_2d)
    M        = preds_2d.shape[0]
    R_2d     = safe_numpy(R_pix).reshape(N, N)
    Z_2d     = safe_numpy(Z_pix).reshape(N, N)

    ncols = min(M, 5)
    nrows = (M + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4 * ncols, 4.2 * nrows),
                              squeeze=False)

    vmax = preds_2d.max() or 1.0

    for m in range(M):
        ax  = axes[m // ncols][m % ncols]
        pcm = ax.pcolormesh(
            R_2d, Z_2d, preds_2d[m],
            vmin=0, vmax=vmax, cmap=cmap, shading="auto",
        )
        ax.set_title(f"Member {m + 1}", fontsize=10)
        ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
        ax.set_aspect("equal")
        plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for m in range(M, nrows * ncols):
        axes[m // ncols][m % ncols].set_visible(False)

    fig.suptitle(f"Ensemble Members — Profile {profile_idx}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 7.  Loss curves
# ═══════════════════════════════════════════════════════════════════════

def plot_loss_curves(
    hist      : dict,
    save_path : Optional[str] = None,
) -> plt.Figure:
    """
    Multi-panel loss-curve plot from the history dict accumulated by
    trainer.train().

    Panels shown (if the key exists in hist):
      • total        — overall weighted loss (log scale)
      • proj         — data-fidelity projection loss
      • nll          — ensemble NLL
      • smooth       — TV smoothness
      • diversity    — ensemble diversity (negative → rising = good)
      • isotropy     — flux-surface isotropy
      • boundary     — boundary enforcement

    Parameters
    ----------
    hist      : dict[str → list[float]]   from trainer.train()
    save_path : str, optional              if given, saves the figure

    Returns
    -------
    matplotlib Figure
    """
    KEYS   = ["total", "proj", "nll", "smooth",
              "diversity", "isotropy", "boundary"]
    COLORS = ["black", "tab:orange", "tab:blue", "tab:green",
              "tab:red",   "tab:purple", "tab:brown"]
    TITLES = ["Total loss", "Projection (data fidelity)",
              "Ensemble NLL", "TV smoothness",
              "Ensemble diversity", "Isotropy prior", "Boundary"]

    present = [(k, c, t) for k, c, t in zip(KEYS, COLORS, TITLES) if k in hist and len(hist[k]) > 0]
    n_plots = len(present)

    if n_plots == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No loss history available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    ncols = min(n_plots, 4)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 3.8 * nrows),
                              squeeze=False)

    for idx, (key, color, title) in enumerate(present):
        ax  = axes[idx // ncols][idx % ncols]
        arr = np.array(hist[key], dtype=np.float64)
        xs  = np.arange(len(arr))

        # Optionally clip extreme values for readability
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            ax.set_visible(False)
            continue

        ax.semilogy(xs, np.where(np.isfinite(arr), arr, np.nan),
                    lw=0.7, alpha=0.85, color=color)

        # Rolling mean (window = 5% of length)
        win = max(1, len(arr) // 20)
        if len(arr) >= win:
            rm = np.convolve(finite, np.ones(win) / win, mode="valid")
            xs_rm = np.linspace(win // 2, len(arr) - win // 2, len(rm))
            ax.semilogy(xs_rm, rm, lw=1.8, color=color, alpha=0.5, label="rolling mean")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel("loss (log scale)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n_plots, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("VICTOR v6.0 — Training loss curves",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Loss curves saved → {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════
# 8.  Summary metrics bar chart
# ═══════════════════════════════════════════════════════════════════════

def plot_metrics_summary(eval_bundles: list) -> plt.Figure:
    """
    Bar charts of PSNR, CC, and RelErr across all evaluated profiles.

    Parameters
    ----------
    eval_bundles : list[EvalBundle]

    Returns
    -------
    matplotlib Figure
    """
    idxs    = [b.metrics.profile_idx for b in eval_bundles]
    psnrs   = [b.metrics.psnr        for b in eval_bundles]
    ccs     = [b.metrics.cc          for b in eval_bundles]
    rel_errs= [b.metrics.rel_err     for b in eval_bundles]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(idxs))
    w = 0.65

    axes[0].bar(x, psnrs, width=w, color="steelblue", alpha=0.85)
    axes[0].set_title("PSNR [dB]  (↑ better)", fontsize=11)
    axes[0].set_xlabel("Profile")
    axes[0].set_xticks(x); axes[0].set_xticklabels(idxs)
    axes[0].axhline(np.mean(psnrs), color="red", lw=1.2,
                    linestyle="--", label=f"mean={np.mean(psnrs):.1f} dB")
    axes[0].legend(fontsize=9); axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, ccs, width=w, color="darkorange", alpha=0.85)
    axes[1].set_title("Cross-Correlation  (↑ better)", fontsize=11)
    axes[1].set_xlabel("Profile")
    axes[1].set_xticks(x); axes[1].set_xticklabels(idxs)
    axes[1].axhline(np.mean(ccs), color="red", lw=1.2,
                    linestyle="--", label=f"mean={np.mean(ccs):.3f}")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=9); axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(x, rel_errs, width=w, color="forestgreen", alpha=0.85)
    axes[2].set_title("Relative L2 Error  (↓ better)", fontsize=11)
    axes[2].set_xlabel("Profile")
    axes[2].set_xticks(x); axes[2].set_xticklabels(idxs)
    axes[2].axhline(np.mean(rel_errs), color="red", lw=1.2,
                    linestyle="--", label=f"mean={np.mean(rel_errs):.3f}")
    axes[2].legend(fontsize=9); axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("VICTOR v6.0 — Evaluation Metrics Summary",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 9.  Per-profile forward evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_profile(
    model,
    params    : dict,
    profile   : dict,
    w_ops,
    grids,
    rho_graph,
) -> EvalBundle:
    """
    Run a full forward pass on one profile and collect evaluation data.

    Parameters
    ----------
    model     : VICTOR_v6 instance
    params    : trained param tree
    profile   : dict from data_loader.load_profiles()
    w_ops     : WOperators
    grids     : PixelGrids
    rho_graph : RhoGraph

    Returns
    -------
    EvalBundle
    """
    from victor import config as cfg

    N = cfg.N_GRID
    M = cfg.N_ENS

    # Forward pass
    eps_out, mean, std, preds = model.apply(
        params,
        grids.R_PIX,
        grids.Z_PIX,
        profile["psi_n"],
        profile["bpol_n"],
        rho_graph.EDGES_SRC,
        rho_graph.EDGES_DST,
        rho_graph.EDGE_W,
        rho_graph.NODE_DEG,
        grids.RHO_2D,
    )

    eps_pred_np = safe_numpy(eps_out)            # (N, N)
    mean_np     = safe_numpy(mean).reshape(N, N) # (N, N)
    std_np      = safe_numpy(std).reshape(N, N)  # (N, N)
    preds_np    = safe_numpy(preds).reshape(M, N, N)  # (M, N, N)

    eps_gt_np   = profile["eps_n"]               # (N, N)  already numpy

    # Projected sinogram from prediction
    g_pred_jax  = w_ops.matvec(eps_out.flatten())
    g_pred_np   = safe_numpy(g_pred_jax)
    g_gt_np     = safe_numpy(profile["g_ideal"])

    # Active mask (all ones in eval mode — same treatment as verify_losses)
    active_mask = np.ones(128, dtype=np.float32)

    metrics = compute_all_metrics(
        eps_pred    = eps_pred_np,
        eps_gt      = eps_gt_np,
        g_pred      = g_pred_np,
        g_gt        = g_gt_np,
        active_mask = active_mask,
        profile_idx = int(profile.get("idx", 0)),
    )

    return EvalBundle(
        profile_idx = int(profile.get("idx", 0)),
        eps_pred    = eps_pred_np,
        eps_gt      = eps_gt_np,
        mean_2d     = mean_np,
        std_2d      = std_np,
        preds_2d    = preds_np,
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
    """
    Evaluate all profiles and return a list of EvalBundles.

    Parameters
    ----------
    model, params, w_ops, grids, rho_graph : as per evaluate_profile()
    profiles : list of profile dicts

    Returns
    -------
    list[EvalBundle]
    """
    bundles = []
    print(f"\nEvaluating {len(profiles)} profile(s)...")
    for i, prof in enumerate(profiles):
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

    # Print aggregate
    psnrs   = [b.metrics.psnr    for b in bundles]
    ccs     = [b.metrics.cc      for b in bundles]
    rerrs   = [b.metrics.rel_err for b in bundles]
    print(f"\n  ── Aggregate (n={len(bundles)}) ────────────────────────")
    print(f"     PSNR    : {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")
    print(f"     CC      : {np.mean(ccs):.4f} ± {np.std(ccs):.4f}")
    print(f"     RelErr  : {np.mean(rerrs):.4f} ± {np.std(rerrs):.4f}")
    return bundles


# ═══════════════════════════════════════════════════════════════════════
# 10.  Convenience: evaluate all profiles + save all figures
# ═══════════════════════════════════════════════════════════════════════

def plot_all_profiles(
    model,
    params      : dict,
    profiles    : list,
    w_ops,
    grids,
    rho_graph,
    results_dir : str = cfg.RESULTS_DIR,
    hist        : Optional[dict] = None,
) -> list:
    """
    Full evaluation pipeline for all profiles:
      1. Runs evaluate_all()
      2. Saves reconstruction, sinogram, radial, uncertainty, and ensemble
         figures for every profile into results_dir/
      3. Saves a metrics-summary bar chart
      4. Optionally saves loss curves (if hist is provided)
      5. Returns list[EvalBundle]

    Parameters
    ----------
    model, params, profiles, w_ops, grids, rho_graph : pipeline objects
    results_dir : str   where to write PNG files (created if absent)
    hist        : dict, optional   training history from trainer.train()

    Returns
    -------
    list[EvalBundle]
    """
    os.makedirs(results_dir, exist_ok=True)

    # ── Run evaluation ─────────────────────────────────────────────────
    eval_bundles = evaluate_all(model, params, profiles, w_ops, grids, rho_graph)

    # ── Per-profile figures ────────────────────────────────────────────
    active_mask = np.ones(128, dtype=np.float32)

    for eb in eval_bundles:
        pid = eb.profile_idx

        # Reconstruction (4-panel)
        fig = plot_reconstruction(
            eps_pred    = eb.eps_pred,
            eps_gt      = eb.eps_gt,
            std_2d      = eb.std_2d,
            rho_2d      = grids.RHO_2D,
            R_pix       = grids.R_PIX,
            Z_pix       = grids.Z_PIX,
            metrics     = eb.metrics,
            profile_idx = pid,
        )
        path = os.path.join(results_dir, f"reconstruction_profile_{pid:03d}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # Sinogram residual
        fig = plot_sinogram_residual(
            g_pred      = eb.g_pred,
            g_gt        = eb.g_gt,
            active_mask = active_mask,
            profile_idx = pid,
        )
        path = os.path.join(results_dir, f"sinogram_profile_{pid:03d}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # Radial profile
        fig = plot_radial_profile(
            eps_pred    = eb.eps_pred,
            eps_gt      = eb.eps_gt,
            std_2d      = eb.std_2d,
            rho_flat    = grids.RHO_FLAT,
            profile_idx = pid,
        )
        path = os.path.join(results_dir, f"radial_profile_{pid:03d}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # Uncertainty map
        fig = plot_uncertainty(
            std_2d      = eb.std_2d,
            rho_2d      = grids.RHO_2D,
            R_pix       = grids.R_PIX,
            Z_pix       = grids.Z_PIX,
            profile_idx = pid,
        )
        path = os.path.join(results_dir, f"uncertainty_profile_{pid:03d}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # Ensemble members
        fig = plot_ensemble_members(
            preds_2d    = eb.preds_2d,
            R_pix       = grids.R_PIX,
            Z_pix       = grids.Z_PIX,
            profile_idx = pid,
        )
        path = os.path.join(results_dir, f"ensemble_members_profile_{pid:03d}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── Metrics summary ────────────────────────────────────────────────
    fig  = plot_metrics_summary(eval_bundles)
    path = os.path.join(results_dir, "metrics_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Loss curves (optional) ─────────────────────────────────────────
    if hist is not None:
        path = os.path.join(results_dir, "loss_curves.png")
        fig  = plot_loss_curves(hist, save_path=path)
        plt.close(fig)

    # ── CSV summary ───────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "metrics_summary.csv")
    save_summary_csv(eval_bundles, csv_path)

    print(f"\nAll figures saved to: {results_dir}")
    return eval_bundles


# ═══════════════════════════════════════════════════════════════════════
# 11.  CSV export
# ═══════════════════════════════════════════════════════════════════════

def save_summary_csv(eval_bundles: list, path: str) -> None:
    """
    Write a CSV table of per-profile metrics.

    Columns: profile_idx, psnr_dB, cc, rel_err, proj_mse

    Parameters
    ----------
    eval_bundles : list[EvalBundle]
    path         : str   destination file path
    """
    import csv

    rows = [("profile_idx", "psnr_dB", "cc", "rel_err", "proj_mse")]
    for eb in eval_bundles:
        m = eb.metrics
        rows.append((
            m.profile_idx,
            round(m.psnr,    4),
            round(m.cc,      6),
            round(m.rel_err, 6),
            round(m.proj_mse,8),
        ))

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Metrics CSV saved → {path}")


# ── Module self-test ─────────────────────────────────────────────────

if __name__ == "__main__":
    """Quick sanity check with random arrays (no model or data needed)."""
    N, M = cfg.N_GRID, cfg.N_ENS
    rng  = np.random.default_rng(0)

    pred = rng.random((N, N)).astype(np.float32)
    gt   = rng.random((N, N)).astype(np.float32)
    std  = rng.random((N, N)).astype(np.float32) * 0.05

    lin      = np.linspace(-cfg.EXT, cfg.EXT, N)
    XX, YY   = np.meshgrid(lin, lin)
    rho_2d   = np.sqrt((XX / cfg.AP)**2 + (YY / cfg.BP)**2).astype(np.float32)
    R_pix    = (cfg.R0 + XX).flatten().astype(np.float32)
    Z_pix    = YY.flatten().astype(np.float32)
    rho_flat = rho_2d.flatten()

    psnr    = compute_psnr(pred, gt)
    cc      = compute_cc(pred, gt)
    rel_err = compute_rel_err(pred, gt)
    print(f"PSNR={psnr:.2f} dB  CC={cc:.4f}  RelErr={rel_err:.4f}")

    g_pred = rng.random(128).astype(np.float32)
    g_gt   = rng.random(128).astype(np.float32)
    mask   = np.ones(128, np.float32)

    m = compute_all_metrics(pred, gt, g_pred, g_gt, mask, profile_idx=0)
    print(m)

    fig = plot_reconstruction(pred, gt, std, rho_2d, R_pix, Z_pix, m, 0)
    plt.close(fig); print("plot_reconstruction OK")

    fig = plot_sinogram_residual(g_pred, g_gt, mask, 0)
    plt.close(fig); print("plot_sinogram_residual OK")

    fig = plot_radial_profile(pred, gt, std, rho_flat, 0)
    plt.close(fig); print("plot_radial_profile OK")

    fig = plot_uncertainty(std, rho_2d, R_pix, Z_pix, 0)
    plt.close(fig); print("plot_uncertainty OK")

    preds_2d = rng.random((M, N, N)).astype(np.float32)
    fig = plot_ensemble_members(preds_2d, R_pix, Z_pix, 0)
    plt.close(fig); print("plot_ensemble_members OK")

    hist_dummy = {
        "total": list(np.exp(-np.linspace(0, 5, 300))),
        "proj":  list(np.exp(-np.linspace(0, 5, 300)) * 0.8),
        "nll":   list(np.exp(-np.linspace(0, 4, 300)) * 0.3),
    }
    fig = plot_loss_curves(hist_dummy)
    plt.close(fig); print("plot_loss_curves OK")

    print("\nevaluate.py self-test PASSED")
