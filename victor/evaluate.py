# ============================================================
# VICTOR v8.2 — evaluate.py
# Standalone evaluation script: load checkpoint, run inference,
# compute metrics, produce publication-quality visualisations.
# ============================================================
#
# CHANGELOG v8.2
# --------------
#  [1] Uses differentiable lerp (build_eps2d_lerp) — no argmin.
#  [2] FIXED config: AP=1.0, BP=1.2, EXT=1.0 (geometry corrected).
#  [3] FIXED eps normalisation: (eps - min)/(max - min) → [0,1].
#  [4] Four figure types: per-profile panel, aggregate dashboard,
#      sinogram residual heatmap, gradient-flow diagnostic.
#  [5] Four metrics: MSE, PSNR, CC, RelError with core/edge splits.
#  [6] JSON export of all metrics.
#  [7] Unicode box table printed to stdout.
#
# USAGE
# -----
#   # Evaluate all profiles:
#   python evaluate.py --ckpt /path/to/checkpoints --dataset /path/to/data
#
#   # Single profile (quick debug):
#   python evaluate.py --profile_idx 0
#
#   # Save JSON:
#   python evaluate.py --save_metrics_json
#
# OUTPUTS (all written to --results_dir)
# ----------------------------------------
#   profile_XXXX_reconstruction.png  — per-profile 2×4 panel
#   evaluation_dashboard.png          — aggregate metrics across profiles
#   sinogram_residuals.png            — stacked sinogram heatmaps
#   gradient_flow_diagnostic.png      — gradient norms + skip gates
#   metrics.json                      — (optional) all numeric metrics
# ============================================================

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator, interp1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import jax
import jax.numpy as jnp
import flax.linen as nn

# Try orbax first (newer Flax), fall back to legacy checkpoints
try:
    import orbax.checkpoint as ocp
    _USE_ORBAX = True
except ImportError:
    from flax.training import checkpoints as flax_checkpoints
    _USE_ORBAX = False

from victor import config as cfg
from victor import geometry as geom
from victor import data_loader as dl
from victor import model as mdl
from victor import losses


# ── Dark theme constants ─────────────────────────────────────────────
_FIG_BG   = '#0d1117'
_AX_BG    = '#161b22'
_TEXT     = 'white'
_GRID_A   = 0.2
_LCFS_CLR = 'cyan'

# ── Metric thresholds ────────────────────────────────────────────────
_CC_GOOD      = 0.95
_CC_MARGINAL  = 0.85
_REL_GOOD     = 0.05
_REL_MARGINAL = 0.15
_PSNR_THRESH  = 25.0   # dB — publication threshold

# ── FIXED geometry (corrected from v8.1) ────────────────────────────
_R0    = 2.5
_AP    = 1.0
_BP    = 1.2
_EXT   = 1.0
_NGRID = 128


# ═══════════════════════════════════════════════════════════════════════
# 1.  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def build_eps2d_lerp(
    coeffs       : np.ndarray,   # (N_RADIAL, 1+2*H)
    rho_flat     : np.ndarray,   # (N_GRID²,)
    theta_flat   : np.ndarray,   # (N_GRID²,)
    rho_radial   : np.ndarray,   # (N_RADIAL,)
    lerp_idx_lo  : np.ndarray,   # (N_GRID²,) int
    lerp_idx_hi  : np.ndarray,   # (N_GRID²,) int
    lerp_frac    : np.ndarray,   # (N_GRID²,) float
    n_harmonics  : int = cfg.N_HARMONICS,
) -> np.ndarray:
    """
    Reconstruct 2D emissivity from Fourier-radial coefficients using
    differentiable linear interpolation (NO argmin).

    For each pixel i:
        eps[i] = lerp(coeffs[idx_lo[i]], coeffs[idx_hi[i]], frac[i])
               evaluated at (rho_i, theta_i)

    Pixels with rho > RHO_MAX are zeroed out.

    Parameters
    ----------
    coeffs      : (N_RADIAL, 1+2*H)  Fourier harmonic coefficients
    rho_flat    : (N_GRID²,)          pixel elliptic radius
    theta_flat  : (N_GRID²,)          pixel poloidal angle [rad]
    rho_radial  : (N_RADIAL,)         model radial axis
    lerp_idx_lo : (N_GRID²,) int      lower bin index per pixel
    lerp_idx_hi : (N_GRID²,) int      upper bin index per pixel
    lerp_frac   : (N_GRID²,) float    interpolation fraction in [0,1]
    n_harmonics : int                  number of poloidal harmonics H

    Returns
    -------
    eps_flat : (N_GRID²,) float32
    """
    # Linearly interpolate coefficients at each pixel's rho position
    # c_pix[i] = (1 - frac[i]) * coeffs[idx_lo[i]] + frac[i] * coeffs[idx_hi[i]]
    c_lo  = coeffs[lerp_idx_lo]                         # (N², NC)
    c_hi  = coeffs[lerp_idx_hi]                         # (N², NC)
    frac2 = lerp_frac[:, None]                          # (N², 1) broadcast
    c_pix = (1.0 - frac2) * c_lo + frac2 * c_hi        # (N², NC)

    # Symmetric component a0
    eps = c_pix[:, 0].copy()                            # (N²,)

    # Poloidal harmonics: a_h*cos(h*θ) + b_h*sin(h*θ)
    for h in range(1, n_harmonics + 1):
        a_col = 2 * h - 1
        b_col = 2 * h
        eps += c_pix[:, a_col] * np.cos(h * theta_flat)
        eps += c_pix[:, b_col] * np.sin(h * theta_flat)

    # Zero pixels outside the model's radial domain
    eps = np.where(rho_flat <= cfg.RHO_MAX, eps, 0.0)
    return eps.astype(np.float32)


def radial_profile(
    eps_flat : np.ndarray,
    rho_flat : np.ndarray,
    n_bins   : int   = 50,
    rho_max  : float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin eps_flat by rho and return mean ± std per bin.

    Only bins with >= 5 pixels are returned to avoid noisy tail estimates.

    Parameters
    ----------
    eps_flat : (N_GRID²,)  flat emissivity values
    rho_flat : (N_GRID²,)  corresponding rho values
    n_bins   : int          number of radial bins
    rho_max  : float        upper bound for binning

    Returns
    -------
    bin_centres : (M,)  rho bin centres
    means       : (M,)  mean emissivity per bin
    stds        : (M,)  std emissivity per bin
    """
    bins        = np.linspace(0.0, rho_max, n_bins + 1)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    bin_idx     = np.digitize(rho_flat, bins) - 1   # 0-indexed

    centres, means, stds = [], [], []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() >= 5:
            centres.append(bin_centres[i])
            means.append(eps_flat[mask].mean())
            stds.append(eps_flat[mask].std())

    return (np.array(centres, dtype=np.float32),
            np.array(means,   dtype=np.float32),
            np.array(stds,    dtype=np.float32))


def poloidal_slice(
    eps_flat    : np.ndarray,
    rho_flat    : np.ndarray,
    theta_flat  : np.ndarray,
    rho_target  : float = 0.5,
    rho_tol     : float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (theta, eps) for all pixels near rho = rho_target ± rho_tol.

    Parameters
    ----------
    eps_flat   : (N_GRID²,)  emissivity values
    rho_flat   : (N_GRID²,)  pixel rho values
    theta_flat : (N_GRID²,)  pixel theta values [rad]
    rho_target : float        target rho shell
    rho_tol    : float        half-width of shell

    Returns
    -------
    theta_vals : (M,)  poloidal angles of matching pixels
    eps_vals   : (M,)  emissivity of matching pixels
    """
    mask = np.abs(rho_flat - rho_target) < rho_tol
    # Sort by theta for cleaner line plots
    order = np.argsort(theta_flat[mask])
    return theta_flat[mask][order], eps_flat[mask][order]


def status_color(cc: float, rel_err: float) -> str:
    """
    Return a matplotlib hex colour based on reconstruction quality.

    Rules
    -----
    GOOD     : CC > 0.95  AND  RelError < 0.05   →  green  '#3fb950'
    MARGINAL : CC > 0.85  OR   RelError < 0.15   →  yellow '#e3b341'
    POOR     : otherwise                          →  red    '#f85149'

    Parameters
    ----------
    cc      : float  Pearson correlation coefficient
    rel_err : float  relative L2 error

    Returns
    -------
    hex colour string
    """
    if cc > _CC_GOOD and rel_err < _REL_GOOD:
        return '#3fb950'    # green — GOOD
    if cc > _CC_MARGINAL or rel_err < _REL_MARGINAL:
        return '#e3b341'    # yellow — MARGINAL
    return '#f85149'        # red — POOR


def status_label(cc: float, rel_err: float) -> str:
    """Return 'GOOD', 'MARGINAL', or 'POOR' string label."""
    if cc > _CC_GOOD and rel_err < _REL_GOOD:
        return 'GOOD'
    if cc > _CC_MARGINAL or rel_err < _REL_MARGINAL:
        return 'MARGINAL'
    return 'POOR'


def format_metrics_table(all_metrics: List[Dict]) -> str:
    """
    Format a unicode box-drawing table of per-profile metrics for stdout.

    Parameters
    ----------
    all_metrics : list of dicts, each with keys MSE, PSNR, CC, RelError, idx

    Returns
    -------
    str  multi-line table string
    """
    top    = '┌──────────┬──────────┬──────────┬──────────┬────────────┐'
    title  = '│  VICTOR v8.2 Evaluation Complete                        │'
    head   = '├──────────┬──────────┬──────────┬──────────┬────────────┤'
    cols   = '│ Profile  │   MSE    │   PSNR   │    CC    │  RelError  │'
    sep    = '├──────────┼──────────┼──────────┼──────────┼────────────┤'
    bot    = '└──────────┴──────────┴──────────┴──────────┴────────────┘'

    rows = [top, title, head, cols, sep]
    for m in all_metrics:
        idx = m.get('idx', 0)
        row = (f"│ {idx:04d}     "
               f"│ {m['MSE']:.2e} "
               f"│ {m['PSNR']:6.1f} dB "
               f"│ {m['CC']:.4f}   "
               f"│ {m['RelError']:.6f}   │")
        rows.append(row)

    rows.append(sep)

    # Mean row
    mses  = [m['MSE']      for m in all_metrics]
    psnrs = [m['PSNR']     for m in all_metrics]
    ccs   = [m['CC']       for m in all_metrics]
    rels  = [m['RelError'] for m in all_metrics]
    mean_row = (f"│ MEAN     "
                f"│ {np.mean(mses):.2e} "
                f"│ {np.mean(psnrs):6.1f} dB "
                f"│ {np.mean(ccs):.4f}   "
                f"│ {np.mean(rels):.6f}   │")
    rows.append(mean_row)
    rows.append(bot)
    return '\n'.join(rows)


# ═══════════════════════════════════════════════════════════════════════
# 2.  METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(
    eps_pred       : np.ndarray,          # (N_GRID²,) predicted emissivity
    eps_true       : np.ndarray,          # (N_GRID²,) ground truth emissivity
    rho_flat       : np.ndarray,          # (N_GRID²,) pixel rho
    active_mask_128: np.ndarray,          # (128,)     bool active chords
    W_norm         : sp.csr_matrix,       # (128, N_GRID²) row-normalised W
    g_true         : np.ndarray,          # (128,)     ground truth sinogram
) -> Dict[str, float]:
    """
    Compute all reconstruction quality metrics.

    All pixel-based metrics are restricted to pixels INSIDE the LCFS
    (rho_flat <= 1.0) to avoid contaminating scores with trivially-zero
    outside regions.  Core and edge sub-regions are also scored separately
    since they represent the hardest reconstruction challenges.

    Parameters
    ----------
    eps_pred        : (N_GRID²,)  model prediction, numpy array
    eps_true        : (N_GRID²,)  ground truth, numpy array, normalised [0,1]
    rho_flat        : (N_GRID²,)  elliptic radius per pixel
    active_mask_128 : (128,)      True for active bolometer chords
    W_norm          : scipy CSR   row-normalised projection matrix
    g_true          : (128,)      ground truth sinogram

    Returns
    -------
    dict with keys:
        MSE, MSE_core, MSE_edge,
        PSNR,
        CC, CC_core, CC_edge,
        RelError,
        sinogram_MSE, sinogram_RelError
    """
    # ── Region masks ──────────────────────────────────────────────────
    inside = rho_flat <= 1.0                              # full plasma interior
    core   = rho_flat <= 0.3                              # on-axis core
    edge   = (rho_flat >= 0.7) & (rho_flat <= 1.0)       # pedestal / edge

    def _region_metrics(mask, label='inside'):
        """Compute MSE and CC for a given pixel region mask."""
        p = eps_pred[mask]
        t = eps_true[mask]
        if len(p) == 0:
            return {'MSE': np.nan, 'CC': np.nan}

        # MSE: mean squared error inside region
        mse = float(np.mean((p - t) ** 2))

        # CC: Pearson correlation coefficient
        # cc = cov(p, t) / (std(p) * std(t) + eps)
        p_c = p - p.mean()
        t_c = t - t.mean()
        cc_num   = float(np.mean(p_c * t_c))
        cc_denom = float(np.std(p) * np.std(t)) + 1e-10
        cc = cc_num / cc_denom

        return {'MSE': mse, 'CC': cc}

    # ── Full interior metrics ─────────────────────────────────────────
    m_in   = _region_metrics(inside)
    m_core = _region_metrics(core)
    m_edge = _region_metrics(edge)

    mse = m_in['MSE']
    cc  = m_in['CC']

    # ── PSNR: 10 * log10(MAX_I^2 / MSE) ─────────────────────────────
    # MAX_I = peak ground truth inside LCFS
    if inside.sum() > 0 and mse > 0:
        max_i = float(np.max(eps_true[inside]))
        psnr  = float(10.0 * np.log10(max_i ** 2 / mse))
    else:
        psnr = np.nan

    # ── RelError: ||pred - true||_2 / ||true||_2 ─────────────────────
    if inside.sum() > 0:
        num = float(np.linalg.norm(eps_pred[inside] - eps_true[inside]))
        den = float(np.linalg.norm(eps_true[inside])) + 1e-10
        rel_err = num / den
    else:
        rel_err = np.nan

    # ── Sinogram metrics: project eps_pred and compare to g_true ──────
    # g_pred = W_norm @ eps_pred  (CPU sparse mat-vec)
    g_pred  = np.array(W_norm @ eps_pred.astype(np.float64)).flatten()
    act     = active_mask_128.astype(bool)

    g_p_act = g_pred[act]
    g_t_act = g_true[act]

    if act.sum() > 0:
        sino_mse = float(np.mean((g_p_act - g_t_act) ** 2))
        sino_rel = float(np.linalg.norm(g_p_act - g_t_act) /
                         (np.linalg.norm(g_t_act) + 1e-10))
    else:
        sino_mse = np.nan
        sino_rel = np.nan

    return {
        'MSE'             : mse,
        'MSE_core'        : m_core['MSE'],
        'MSE_edge'        : m_edge['MSE'],
        'PSNR'            : psnr,
        'CC'              : cc,
        'CC_core'         : m_core['CC'],
        'CC_edge'         : m_edge['CC'],
        'RelError'        : rel_err,
        'sinogram_MSE'    : sino_mse,
        'sinogram_RelError': sino_rel,
    }


# ═══════════════════════════════════════════════════════════════════════
# 3.  FIGURE 1 — Per-profile reconstruction panel
# ═══════════════════════════════════════════════════════════════════════

def _ax_style(ax, title: str, xlabel: str = '', ylabel: str = '') -> None:
    """Apply dark theme styling to a single axes."""
    ax.set_facecolor(_AX_BG)
    ax.set_title(title, color=_TEXT, fontsize=9, pad=4)
    ax.tick_params(colors=_TEXT, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    if xlabel:
        ax.set_xlabel(xlabel, color=_TEXT, fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, color=_TEXT, fontsize=7)
    ax.grid(alpha=_GRID_A, color='#30363d')


def _lcfs_contour(ax, rho_flat: np.ndarray, n_grid: int = _NGRID,
                  ext: float = _EXT, r0: float = _R0) -> None:
    """Overlay cyan LCFS contour (rho=1.0) on a 2D image axes."""
    rho_2d = rho_flat.reshape(n_grid, n_grid)
    R_lin  = np.linspace(r0 - ext, r0 + ext, n_grid)
    Z_lin  = np.linspace(-ext, ext, n_grid)
    ax.contour(R_lin, Z_lin, rho_2d,
               levels=[1.0], colors=[_LCFS_CLR], linewidths=1.2)


def plot_profile_panel(
    eps_pred    : np.ndarray,    # (N_GRID²,)
    eps_true    : np.ndarray,    # (N_GRID²,)
    coeffs      : np.ndarray,    # (N_RADIAL, NC)
    rho_flat    : np.ndarray,    # (N_GRID²,)
    theta_flat  : np.ndarray,    # (N_GRID²,)
    rho_radial  : np.ndarray,    # (N_RADIAL,)
    g_true      : np.ndarray,    # (128,)
    g_pred      : np.ndarray,    # (128,) projected sinogram
    active_mask : np.ndarray,    # (128,) bool
    metrics     : Dict[str, float],
    profile_idx : int,
    save_path   : str,
    n_grid      : int = _NGRID,
    ext         : float = _EXT,
    r0          : float = _R0,
) -> None:
    """
    Generate a 2×4 reconstruction diagnostic panel for one profile.

    Layout
    ------
    Row 0: [ground truth] [prediction] [|error|] [sinogram comparison]
    Row 1: [radial profile] [poloidal slice ρ=0.5] [coefficients] [metrics text]

    Parameters
    ----------
    eps_pred   : predicted 2D emissivity (flat)
    eps_true   : ground truth 2D emissivity (flat)
    coeffs     : model Fourier-radial coefficients
    ...        : geometry arrays
    metrics    : dict from compute_metrics()
    profile_idx: integer profile index
    save_path  : full output path for the PNG
    """
    fig = plt.figure(figsize=(20, 10), facecolor=_FIG_BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.40)

    extent  = [r0 - ext, r0 + ext, -ext, ext]
    vmin_e  = float(min(eps_true.min(), eps_pred.min()))
    vmax_e  = float(max(eps_true.max(), eps_pred.max()))

    mse     = metrics['MSE']
    psnr    = metrics['PSNR']
    cc      = metrics['CC']
    rel_err = metrics['RelError']
    sino_m  = metrics['sinogram_MSE']

    # ── [0,0] Ground truth ───────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0], facecolor=_AX_BG)
    im00 = ax00.imshow(eps_true.reshape(n_grid, n_grid),
                       origin='lower', cmap='inferno', aspect='auto',
                       extent=extent, vmin=vmin_e, vmax=vmax_e)
    plt.colorbar(im00, ax=ax00, fraction=0.046, pad=0.04)
    _lcfs_contour(ax00, rho_flat, n_grid, ext, r0)
    _ax_style(ax00, f'Ground Truth  profile_{profile_idx:04d}',
              'R [m]', 'Z [m]')

    # ── [0,1] Prediction ─────────────────────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1], facecolor=_AX_BG)
    im01 = ax01.imshow(eps_pred.reshape(n_grid, n_grid),
                       origin='lower', cmap='inferno', aspect='auto',
                       extent=extent, vmin=vmin_e, vmax=vmax_e)
    plt.colorbar(im01, ax=ax01, fraction=0.046, pad=0.04)
    _lcfs_contour(ax01, rho_flat, n_grid, ext, r0)
    _ax_style(ax01, f'Reconstruction  CC={cc:.4f}  PSNR={psnr:.2f} dB',
              'R [m]', 'Z [m]')

    # ── [0,2] Absolute error ─────────────────────────────────────────
    ax02 = fig.add_subplot(gs[0, 2], facecolor=_AX_BG)
    err  = np.abs(eps_pred - eps_true)
    im02 = ax02.imshow(err.reshape(n_grid, n_grid),
                       origin='lower', cmap='hot', aspect='auto',
                       extent=extent)
    plt.colorbar(im02, ax=ax02, fraction=0.046, pad=0.04)
    _lcfs_contour(ax02, rho_flat, n_grid, ext, r0)
    _ax_style(ax02,
              f'|Error|  MSE={mse:.2e}  RelErr={rel_err:.4f}',
              'R [m]', 'Z [m]')

    # ── [0,3] Sinogram comparison ─────────────────────────────────────
    ax03 = fig.add_subplot(gs[0, 3], facecolor=_AX_BG)
    act_idx = np.where(active_mask)[0]
    bar_w   = 0.4
    ax03.bar(act_idx - 0.2, g_true[act_idx], width=bar_w,
             color='#58a6ff', label='g_true', alpha=0.9)
    ax03.bar(act_idx + 0.2, g_pred[act_idx], width=bar_w,
             color='#f78166', label='g_pred', alpha=0.9)
    ax03.legend(fontsize=7, facecolor=_AX_BG, labelcolor=_TEXT,
                framealpha=0.8)
    _ax_style(ax03, f'Sinogram  sino_MSE={sino_m:.2e}',
              'chord index', 'normalised signal')

    # ── [1,0] Radial profile comparison ──────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0], facecolor=_AX_BG)
    rc_p, rm_p, rs_p = radial_profile(eps_pred, rho_flat)
    rc_t, rm_t, _    = radial_profile(eps_true, rho_flat)
    ax10.plot(rc_t, rm_t, color='#58a6ff', linewidth=1.8, label='true')
    ax10.plot(rc_p, rm_p, color='#f78166', linewidth=1.8, label='pred')
    ax10.fill_between(rc_p, rm_p - rs_p, rm_p + rs_p,
                      color='#f78166', alpha=0.2, label='±1σ pred')
    ax10.axvline(1.0, color=_LCFS_CLR, linewidth=1.0, linestyle='--',
                 label='LCFS')
    ax10.legend(fontsize=7, facecolor=_AX_BG, labelcolor=_TEXT,
                framealpha=0.8)
    _ax_style(ax10, 'Radial Profile (mean ± std)', 'ρ', 'emissivity')

    # ── [1,1] Poloidal profile at rho=0.5 ────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1], facecolor=_AX_BG)
    th_t, ep_t = poloidal_slice(eps_true, rho_flat, theta_flat)
    th_p, ep_p = poloidal_slice(eps_pred, rho_flat, theta_flat)
    ax11.scatter(th_t, ep_t, s=2, c='#58a6ff', label='true',  alpha=0.6)
    ax11.scatter(th_p, ep_p, s=2, c='#f78166', label='pred',  alpha=0.6)
    ax11.legend(fontsize=7, facecolor=_AX_BG, labelcolor=_TEXT,
                framealpha=0.8, markerscale=3)
    _ax_style(ax11, 'Poloidal Profile at ρ=0.5', 'θ [rad]', 'emissivity')

    # ── [1,2] Radial coefficient profiles ────────────────────────────
    ax12 = fig.add_subplot(gs[1, 2], facecolor=_AX_BG)
    coef_colors = ['#3fb950', '#f78166', '#58a6ff', '#e3b341', '#bc8cff',
               '#ff7b72', '#79c0ff', '#ffa657', '#d2a8ff', '#56d364',
               '#f0883e', '#338ef7', '#db61a2']
    nc = coeffs.shape[1]
    labels = ['a0'] + [f'{"ab"[i%2]}{i//2+1}' for i in range(nc - 1)]
    for ch in range(nc):
        ax12.plot(rho_radial, coeffs[:, ch],
                  color=coef_colors[ch % len(coef_colors)],
                  linewidth=1.5, label=labels[ch])
    ax12.axvline(1.0, color=_LCFS_CLR, linewidth=1.0, linestyle='--')
    ax12.legend(fontsize=7, facecolor=_AX_BG, labelcolor=_TEXT,
                framealpha=0.8)
    _ax_style(ax12, 'Fourier Coefficients vs ρ', 'ρ', 'amplitude')

    # ── [1,3] Metrics text box ────────────────────────────────────────
    ax13 = fig.add_subplot(gs[1, 3], facecolor=_AX_BG)
    ax13.axis('off')
    slabel = status_label(cc, rel_err)
    scol   = status_color(cc, rel_err)

    lines = [
        f"Profile: {profile_idx:04d}",
        "─────────────────────",
        f"MSE        : {mse:.4e}",
        f"MSE_core   : {metrics['MSE_core']:.4e}",
        f"MSE_edge   : {metrics['MSE_edge']:.4e}",
        f"PSNR       : {psnr:.2f} dB",
        f"CC         : {metrics['CC']:.6f}",
        f"CC_core    : {metrics['CC_core']:.6f}",
        f"CC_edge    : {metrics['CC_edge']:.6f}",
        f"RelError   : {rel_err:.6f}",
        f"Sino MSE   : {sino_m:.4e}",
        f"Sino RelErr: {metrics['sinogram_RelError']:.6f}",
        "─────────────────────",
        f"Status: {slabel}",
    ]
    text_str = '\n'.join(lines)
    ax13.text(0.05, 0.95, text_str,
              transform=ax13.transAxes,
              fontsize=9, verticalalignment='top',
              fontfamily='monospace', color=_TEXT,
              bbox=dict(boxstyle='round,pad=0.4',
                        facecolor='#21262d', edgecolor=scol,
                        linewidth=2.0))

    fig.suptitle(f'VICTOR v8.2 — Profile {profile_idx:04d} Reconstruction',
                 color=_TEXT, fontsize=12, y=1.01)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=_FIG_BG)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 4.  FIGURE 2 — Aggregate dashboard
# ═══════════════════════════════════════════════════════════════════════

def plot_dashboard(
    all_metrics  : List[Dict],
    save_path    : str,
) -> None:
    """
    Generate a 2×3 aggregate metrics dashboard across all evaluated profiles.

    Shows bar charts for MSE, CC, PSNR, RelError (colour-coded by quality),
    a CC_core vs CC_edge scatter, and a text summary table.

    Parameters
    ----------
    all_metrics : list of metric dicts (one per profile, from compute_metrics)
    save_path   : output PNG path
    """
    n     = len(all_metrics)
    idxs  = [m['idx']      for m in all_metrics]
    mses  = [m['MSE']      for m in all_metrics]
    psnrs = [m['PSNR']     for m in all_metrics]
    ccs   = [m['CC']       for m in all_metrics]
    rels  = [m['RelError'] for m in all_metrics]
    cc_c  = [m['CC_core']  for m in all_metrics]
    cc_e  = [m['CC_edge']  for m in all_metrics]
    sinos = [m['sinogram_MSE'] for m in all_metrics]

    colors = [status_color(c, r) for c, r in zip(ccs, rels)]
    x_pos  = np.arange(n)

    fig = plt.figure(figsize=(18, 10), facecolor=_FIG_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.40)

    # ── [0,0] MSE bar ─────────────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0], facecolor=_AX_BG)
    ax00.bar(x_pos, mses, color=colors, width=0.7, edgecolor='none')
    ax00.axhline(np.mean(mses), color='white', linewidth=1.0,
                 linestyle='--', label=f'mean={np.mean(mses):.2e}')
    ax00.set_xticks(x_pos); ax00.set_xticklabels([str(i) for i in idxs],
                                                   fontsize=7, color=_TEXT)
    ax00.legend(fontsize=7, facecolor=_AX_BG, labelcolor=_TEXT)
    _ax_style(ax00, 'MSE per profile', 'profile', 'MSE')

    # ── [0,1] CC bar ──────────────────────────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1], facecolor=_AX_BG)
    ax01.bar(x_pos, ccs, color=colors, width=0.7, edgecolor='none')
    ax01.axhline(_CC_GOOD, color='#3fb950', linewidth=1.2,
                 linestyle='--', label='CC=0.95 (target)')
    ax01.axhline(np.mean(ccs), color='white', linewidth=1.0,
                 linestyle=':', label=f'mean={np.mean(ccs):.4f}')
    ax01.set_ylim(0, 1.05)
    ax01.set_xticks(x_pos); ax01.set_xticklabels([str(i) for i in idxs],
                                                   fontsize=7, color=_TEXT)
    ax01.legend(fontsize=7, facecolor=_AX_BG, labelcolor=_TEXT)
    _ax_style(ax01, 'CC per profile', 'profile', 'CC')

    # ── [0,2] PSNR bar ────────────────────────────────────────────────
    ax02 = fig.add_subplot(gs[0, 2], facecolor=_AX_BG)
    ax02.bar(x_pos, psnrs, color=colors, width=0.7, edgecolor='none')
    ax02.axhline(_PSNR_THRESH, color='#e3b341', linewidth=1.2,
                 linestyle='--', label='25 dB (threshold)')
    ax02.axhline(np.mean(psnrs), color='white', linewidth=1.0,
                 linestyle=':', label=f'mean={np.mean(psnrs):.1f} dB')
    ax02.set_xticks(x_pos); ax02.set_xticklabels([str(i) for i in idxs],
                                                   fontsize=7, color=_TEXT)
    ax02.legend(fontsize=7, facecolor=_AX_BG, labelcolor=_TEXT)
    _ax_style(ax02, 'PSNR per profile', 'profile', 'PSNR [dB]')

    # ── [1,0] CC_core vs CC_edge scatter ─────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0], facecolor=_AX_BG)
    for i, (xv, yv, c, idx) in enumerate(zip(cc_c, cc_e, colors, idxs)):
        ax10.scatter(xv, yv, color=c, s=60, zorder=3)
        ax10.annotate(str(idx), (xv, yv), textcoords='offset points',
                      xytext=(4, 4), fontsize=7, color=_TEXT)
    # Diagonal reference line (x=y)
    lim_lo = min(min(cc_c), min(cc_e)) - 0.05
    lim_hi = max(max(cc_c), max(cc_e)) + 0.05
    ax10.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
              color='white', linewidth=1.0, linestyle='--', alpha=0.6)
    _ax_style(ax10, 'Core vs Edge Reconstruction Quality',
              'CC_core', 'CC_edge')

    # ── [1,1] RelError bar ────────────────────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1], facecolor=_AX_BG)
    ax11.bar(x_pos, rels, color=colors, width=0.7, edgecolor='none')
    ax11.axhline(_REL_GOOD, color='#3fb950', linewidth=1.2,
                 linestyle='--', label='0.05 (target)')
    ax11.axhline(np.mean(rels), color='white', linewidth=1.0,
                 linestyle=':', label=f'mean={np.mean(rels):.4f}')
    ax11.set_xticks(x_pos); ax11.set_xticklabels([str(i) for i in idxs],
                                                   fontsize=7, color=_TEXT)
    ax11.legend(fontsize=7, facecolor=_AX_BG, labelcolor=_TEXT)
    _ax_style(ax11, 'RelError per profile', 'profile', 'RelError')

    # ── [1,2] Summary text ────────────────────────────────────────────
    ax12 = fig.add_subplot(gs[1, 2], facecolor=_AX_BG)
    ax12.axis('off')

    n_good = sum(1 for c, r in zip(ccs, rels) if c > _CC_GOOD and r < _REL_GOOD)
    n_poor = sum(1 for c, r in zip(ccs, rels)
                 if c <= _CC_MARGINAL and r >= _REL_MARGINAL)
    n_marg = n - n_good - n_poor

    best_idx  = idxs[int(np.argmax(ccs))]
    worst_idx = idxs[int(np.argmin(ccs))]

    summary = (
        "VICTOR v8.2 — Evaluation Summary\n"
        "══════════════════════════════════\n"
        f"N profiles evaluated : {n}\n"
        f"GOOD  (CC>0.95)      : {n_good}\n"
        f"MARGINAL             : {n_marg}\n"
        f"POOR                 : {n_poor}\n"
        "──────────────────────────────────\n"
        f"Mean MSE     : {np.mean(mses):.4e} ± {np.std(mses):.4e}\n"
        f"Mean PSNR    : {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB\n"
        f"Mean CC      : {np.mean(ccs):.6f} ± {np.std(ccs):.6f}\n"
        f"Mean RelError: {np.mean(rels):.6f} ± {np.std(rels):.6f}\n"
        f"Mean Sino MSE: {np.mean(sinos):.4e}\n"
        "──────────────────────────────────\n"
        f"Best  profile: {best_idx} (CC={max(ccs):.4f})\n"
        f"Worst profile: {worst_idx} (CC={min(ccs):.4f})"
    )
    ax12.text(0.04, 0.96, summary,
              transform=ax12.transAxes,
              fontsize=8, verticalalignment='top',
              fontfamily='monospace', color=_TEXT,
              bbox=dict(boxstyle='round,pad=0.5',
                        facecolor='#21262d', edgecolor='#30363d'))

    fig.suptitle('VICTOR v8.2 — Evaluation Dashboard',
                 color=_TEXT, fontsize=13, y=1.01)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=_FIG_BG)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 5.  FIGURE 3 — Sinogram residual heatmap
# ═══════════════════════════════════════════════════════════════════════

def plot_sinogram_residuals(
    g_true_all : np.ndarray,   # (n_profiles, 128)
    g_pred_all : np.ndarray,   # (n_profiles, 128)
    save_path  : str,
) -> None:
    """
    Plot stacked sinogram heatmaps (ground truth, prediction, residual).

    Rows = profiles, columns = chord index.  Useful for identifying
    systematic chord-level biases across the profile ensemble.

    Parameters
    ----------
    g_true_all : (n_profiles, 128)  ground truth sinograms stacked
    g_pred_all : (n_profiles, 128)  predicted sinograms stacked
    save_path  : output PNG path
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                             facecolor=_FIG_BG)
    fig.subplots_adjust(wspace=0.35)

    vmin = min(g_true_all.min(), g_pred_all.min())
    vmax = max(g_true_all.max(), g_pred_all.max())
    n_p  = g_true_all.shape[0]

    def _hm(ax, data, title, cmap, vmin_=None, vmax_=None):
        ax.set_facecolor(_AX_BG)
        im = ax.imshow(data, origin='upper', aspect='auto',
                       cmap=cmap, vmin=vmin_, vmax=vmax_,
                       interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, color=_TEXT, fontsize=10)
        ax.set_xlabel('chord index', color=_TEXT, fontsize=8)
        ax.set_ylabel('profile index', color=_TEXT, fontsize=8)
        ax.tick_params(colors=_TEXT, labelsize=7)
        ax.set_yticks(np.arange(n_p))
        ax.set_yticklabels([str(i) for i in range(n_p)], fontsize=6)

    _hm(axes[0], g_true_all, 'Ground Truth Sinograms', 'inferno', vmin, vmax)
    _hm(axes[1], g_pred_all, 'Predicted Sinograms',    'inferno', vmin, vmax)
    _hm(axes[2], np.abs(g_pred_all - g_true_all),
        'Sinogram Residuals |g_pred − g_true|', 'hot')

    fig.suptitle('VICTOR v8.2 — Sinogram Residuals',
                 color=_TEXT, fontsize=12, y=1.01)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=_FIG_BG)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 6.  FIGURE 4 — Gradient flow diagnostic
# ═══════════════════════════════════════════════════════════════════════

def plot_gradient_flow(
    model          : nn.Module,
    params         : dict,
    profiles       : List[dict],
    grids          : geom.PixelGrids,
    w_bundle       : dl.WBundle,
    save_path      : str,
) -> None:
    """
    Verify that gradients flow through the decoder (argmin→lerp fix check).

    Computes the gradient of the projection loss w.r.t. every parameter
    leaf using JAX autodiff, then plots the mean gradient norm per named
    layer.  Dead layers (norm < 1e-6) shown in red.  Also extracts
    learned skip gate values from UFourierLayer1D layers.

    Parameters
    ----------
    model    : FourierDeepONetV8 instance
    params   : trained parameters
    profiles : list of profile dicts (uses first few for grad averaging)
    grids    : PixelGrids with LERP fields populated
    w_bundle : WBundle (for w_ops.matvec)
    save_path: output PNG path
    """
    # ── Compute gradient norms averaged over up to 3 profiles ─────────
    n_check = min(3, len(profiles))

    def _proj_loss(p, prof):
        """Projection loss only — tests gradient flow through decoder."""
        coeffs = model.apply(
            p,
            prof['g_ideal'],
            prof['psi_n'],
            grids.RHO_FLAT,
            prof['xi'],
        )
        eps = losses.build_eps2d(
            coeffs,
            grids.RHO_FLAT,
            grids.THETA_FLAT,
            grids.RHO_RADIAL,
            lerp_idx_lo = prof["lerp_idx_lo"],
            lerp_idx_hi = prof["lerp_idx_hi"],
            lerp_frac   = prof["lerp_frac"],
        )
        g_pred  = w_bundle.w_ops.matvec(eps)
        residual = (g_pred - prof['g_ideal']) * w_bundle.ACTIVE_MASK.astype(jnp.float32)
        return jnp.sum(residual ** 2) / jnp.maximum(w_bundle.ACTIVE_MASK.sum(), 1.0)
    # Accumulate gradient norms per leaf path
    norm_accum: Dict[str, List[float]] = {}

    for i in range(n_check):
        try:
            grads = jax.grad(_proj_loss)(params, profiles[i])
            for path, g_leaf in jax.tree_util.tree_leaves_with_path(grads):
                # Build a readable name from the path
                name = '/'.join(
                    str(k.key) if hasattr(k, 'key') else str(k)
                    for k in path
                )
                norm = float(jnp.linalg.norm(g_leaf))
                norm_accum.setdefault(name, []).append(norm)
        except Exception as exc:
            print(f"  [grad diagnostic] profile {i} failed: {exc}")

    if not norm_accum:
        print("  [gradient flow] No gradients computed — skipping Figure 4.")
        return

    # Average norms across profiles
    layer_names = list(norm_accum.keys())
    mean_norms  = [np.mean(norm_accum[k]) for k in layer_names]

    # Shorten names for display (keep last 2 path segments)
    short_names = ['/'.join(n.split('/')[-2:]) for n in layer_names]

    # ── Extract skip gate values ───────────────────────────────────────
    # Walk param tree looking for 'g_skip' keys in UFourierLayer1D
    gate_names  = []
    gate_values = []

    def _collect_gates(tree, prefix=''):
        if isinstance(tree, dict):
            for k, v in tree.items():
                _collect_gates(v, prefix + '/' + str(k) if prefix else str(k))
        elif hasattr(tree, 'items'):
            for k, v in tree.items():
                _collect_gates(v, prefix + '/' + str(k) if prefix else str(k))
        else:
            if 'g_skip' in prefix:
                gate_names.append(prefix)
                # gate = sigmoid(g_skip)
                gate_values.append(float(jax.nn.sigmoid(jnp.asarray(tree))))

    _collect_gates(params)

    # ── Figure ────────────────────────────────────────────────────────
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 6), facecolor=_FIG_BG)

    # [0] Gradient norm per layer
    bar_colors = ['#3fb950' if n > 1e-6 else '#f85149' for n in mean_norms]
    ax0.set_facecolor(_AX_BG)
    bars = ax0.bar(np.arange(len(mean_norms)), mean_norms,
                   color=bar_colors, edgecolor='none')
    ax0.set_xticks(np.arange(len(short_names)))
    ax0.set_xticklabels(short_names, rotation=45, ha='right',
                        fontsize=6, color=_TEXT)
    ax0.tick_params(colors=_TEXT, labelsize=7)
    ax0.set_title('Gradient Flow Through Decoder\n(red=dead  |  green=alive)',
                  color=_TEXT, fontsize=10)
    ax0.set_ylabel('mean |∇| norm', color=_TEXT, fontsize=8)
    ax0.set_xlabel('parameter leaf', color=_TEXT, fontsize=8)
    ax0.set_yscale('log')
    ax0.grid(alpha=_GRID_A, color='#30363d', axis='y')

    # Legend patches
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor='#3fb950', label='alive (norm > 1e-6)'),
                  Patch(facecolor='#f85149', label='dead  (norm ≤ 1e-6)')]
    ax0.legend(handles=legend_els, fontsize=8, facecolor=_AX_BG,
               labelcolor=_TEXT)

    # [1] Skip gate values
    ax1.set_facecolor(_AX_BG)
    if gate_values:
        g_colors = ['#e3b341' if 0.1 < v < 0.9 else
                    ('#f85149' if v <= 0.1 else '#f78166')
                    for v in gate_values]
        short_g = ['/'.join(n.split('/')[-3:]) for n in gate_names]
        ax1.bar(np.arange(len(gate_values)), gate_values,
                color=g_colors, edgecolor='none', width=0.6)
        ax1.set_xticks(np.arange(len(short_g)))
        ax1.set_xticklabels(short_g, rotation=30, ha='right',
                            fontsize=7, color=_TEXT)
        ax1.axhline(0.1, color='#f85149', linewidth=1.2,
                    linestyle='--', label='0.1 (collapsed off)')
        ax1.axhline(0.9, color='#f78166', linewidth=1.2,
                    linestyle='--', label='0.9 (collapsed on)')
        ax1.axhline(0.5, color='white', linewidth=0.8,
                    linestyle=':', alpha=0.6, label='0.5 (balanced)')
        ax1.set_ylim(0, 1.05)
        ax1.legend(fontsize=8, facecolor=_AX_BG, labelcolor=_TEXT)
    else:
        ax1.text(0.5, 0.5, 'No g_skip gates found\n(v8.1 model — gates not yet added)',
                 transform=ax1.transAxes, ha='center', va='center',
                 color=_TEXT, fontsize=10)

    ax1.set_title('Skip Gate Values\n(0=off, 1=always-on, 0.5=balanced)',
                  color=_TEXT, fontsize=10)
    ax1.set_ylabel('sigmoid(g_skip)', color=_TEXT, fontsize=8)
    ax1.tick_params(colors=_TEXT, labelsize=7)
    ax1.grid(alpha=_GRID_A, color='#30363d', axis='y')

    fig.suptitle('VICTOR v8.2 — Gradient Flow Diagnostic',
                 color=_TEXT, fontsize=12, y=1.01)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=_FIG_BG)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 7.  CHECKPOINT LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_checkpoint(ckpt_dir: str, model: nn.Module,
                    dummy_inputs: tuple) -> dict:
    """
    Load model parameters from a checkpoint directory.

    Tries orbax first (new Flax), then falls back to legacy
    flax.training.checkpoints.  Exits with code 1 if no checkpoint found.

    Parameters
    ----------
    ckpt_dir     : path to checkpoint directory or file
    model        : FourierDeepONetV8 instance (for structure)
    dummy_inputs : tuple of (g, psi_n, rho_flat, xi) dummy arrays
                   used to initialise parameter structure for restore

    Returns
    -------
    params : dict  restored parameter tree
    """
    if not os.path.exists(ckpt_dir):
        print(f"ERROR: Checkpoint path not found: {ckpt_dir}")
        sys.exit(1)

    if _USE_ORBAX:
        try:
            checkpointer = ocp.PyTreeCheckpointer()
            # Initialise target structure from a dummy forward pass
            key    = jax.random.PRNGKey(0)
            target = model.init(key, *dummy_inputs)
            params = checkpointer.restore(ckpt_dir, item=target)
            print(f"Loaded checkpoint (orbax) from {ckpt_dir}")
            return params
        except Exception as e:
            print(f"orbax restore failed ({e}), trying legacy checkpoints ...")

    # Legacy flax checkpoint
    try:
        key    = jax.random.PRNGKey(0)
        target = model.init(key, *dummy_inputs)
        params = flax_checkpoints.restore_checkpoint(ckpt_dir, target=target)
        if params is None:
            raise FileNotFoundError("restore_checkpoint returned None")
        print(f"Loaded checkpoint (flax legacy) from {ckpt_dir}")
        return params
    except Exception as e:
        print(f"ERROR: Could not load checkpoint from {ckpt_dir}: {e}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    CLI entry point for VICTOR v8.2 evaluation.

    Loads geometry + data, restores checkpoint, runs inference on all
    (or one) profiles, computes metrics, writes four figure types, and
    optionally dumps a JSON summary.
    """
    parser = argparse.ArgumentParser(
        description='VICTOR v8.2 — evaluate reconstruction quality',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--ckpt',        type=str, default=cfg.CKPT_DIR,
                        help='Checkpoint directory or file path')
    parser.add_argument('--dataset',     type=str, default=cfg.DATASET_DIR,
                        help='Dataset directory (profile_XXXX.npz + W_matrix.npz)')
    parser.add_argument('--results_dir', type=str, default=cfg.RESULTS_DIR,
                        help='Output directory for plots and JSON')
    parser.add_argument('--n_profiles',  type=int, default=cfg.N_PROFILES,
                        help='Number of profiles to evaluate')
    parser.add_argument('--noise_sigma', type=float, default=0.001,
                        help='Gaussian noise sigma for inference (0 = clean)')
    parser.add_argument('--save_metrics_json', action='store_true',
                        help='Write metrics.json to results_dir')
    parser.add_argument('--profile_idx', type=int, default=None,
                        help='Evaluate a single profile index only (debug mode)')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    print(f"\n{'═'*60}")
    print(f"  VICTOR v8.2  Evaluation")
    print(f"  checkpoint  : {args.ckpt}")
    print(f"  dataset     : {args.dataset}")
    print(f"  results_dir : {args.results_dir}")
    print(f"{'═'*60}\n")

    # ── 1. Load geometry and data ─────────────────────────────────────
    print("Loading geometry and profiles ...")
    w_bundle, grids, rays, rho_graph, profiles = dl.load_cell2(args.dataset)

    # Convert geometry arrays to numpy for metric/plot functions
    rho_flat_np   = np.array(grids.RHO_FLAT)
    theta_flat_np = np.array(grids.THETA_FLAT)
    rho_radial_np = np.array(grids.RHO_RADIAL)

    # Lerp indices — use from grids if available, else compute fallback
    if hasattr(grids, 'LERP_IDX_LO'):
        lerp_lo  = np.array(grids.LERP_IDX_LO)
        lerp_hi  = np.array(grids.LERP_IDX_HI)
        lerp_frac= np.array(grids.LERP_FRAC)
    else:
        # Fallback: compute lerp indices here if geometry.py not yet upgraded
        print("  [warn] grids missing LERP fields — computing fallback lerp ...")
        rho_c = np.clip(rho_flat_np, 0, rho_radial_np[-1])
        lerp_hi = np.searchsorted(rho_radial_np, rho_c, side='left')
        lerp_hi = np.clip(lerp_hi, 1, len(rho_radial_np) - 1)
        lerp_lo = lerp_hi - 1
        rho_lo  = rho_radial_np[lerp_lo]
        rho_hi_ = rho_radial_np[lerp_hi]
        lerp_frac = np.clip(
            (rho_c - rho_lo) / (rho_hi_ - rho_lo + 1e-8), 0.0, 1.0
        ).astype(np.float32)

    active_mask_np = np.array(w_bundle.ACTIVE_MASK_NP)   # (128,) bool

    # ── 2. Build model ────────────────────────────────────────────────
    print("Building model ...")
    # Example arrays for shape inference (first profile)
    ex_g   = profiles[0]['g_ideal']
    ex_psi = profiles[0]['psi_n']
    ex_xi  = profiles[0]['xi']

    model_inst = mdl.FourierDeepONetV8(
        branch_hidden = (256, 256),
        trunk_hidden  = (64, 128, 128),
        n_channels    = 64,
        n_modes       = 16,
        n_radial      = cfg.N_RADIAL,
        n_harmonics   = cfg.N_HARMONICS,
        n_eq_channels = cfg.N_EQ_CHANNELS,
        rff_features  = 64,
        rff_sigma     = 1.0,
    )

    dummy_inputs = (ex_g, ex_psi, grids.RHO_FLAT, ex_xi)

    # ── 3. Load checkpoint ────────────────────────────────────────────
    params = load_checkpoint(args.ckpt, model_inst, dummy_inputs)

    # ── 4. JIT-compile inference ──────────────────────────────────────
    @jax.jit
    def predict(p, g, psi_n, rho_flat_jax, xi):
        """Single-profile inference: returns (N_RADIAL, NC) coefficients."""
        return model_inst.apply(p, g, psi_n, rho_flat_jax, xi)

    # ── 5. Determine which profiles to evaluate ───────────────────────
    if args.profile_idx is not None:
        # Single-profile debug mode
        eval_profiles = [p for p in profiles if p['idx'] == args.profile_idx]
        if not eval_profiles:
            print(f"ERROR: profile_idx={args.profile_idx} not found in loaded profiles.")
            sys.exit(1)
        single_mode = True
    else:
        eval_profiles = profiles[:args.n_profiles]
        single_mode   = False

    # ── 6. Inference loop ─────────────────────────────────────────────
    all_metrics    : List[Dict]       = []
    g_true_all     : List[np.ndarray] = []
    g_pred_all     : List[np.ndarray] = []

    print(f"\nRunning inference on {len(eval_profiles)} profile(s) ...\n")

    for prof in eval_profiles:
        idx = prof['idx']

        # Guard: skip degenerate profiles (all-zero ground truth)
        eps_true_np = np.array(prof['eps_n']).flatten()
        if np.max(np.abs(eps_true_np)) < 1e-8:
            print(f"  [WARN] profile_{idx:04d}: eps_true is all zeros — skipping.")
            continue

        # ── a. Add noise to sinogram ─────────────────────────────────
        if args.noise_sigma > 0:
            key = jax.random.PRNGKey(idx)
            g_noisy = dl.inject_noise(prof['g_ideal'], args.noise_sigma, key)
        else:
            g_noisy = prof['g_ideal']

        # ── b. Predict coefficients ───────────────────────────────────
        coeffs_jax = predict(params, g_noisy, prof['psi_n'],
                             grids.RHO_FLAT, prof['xi'])
        coeffs_np  = np.array(coeffs_jax)   # (N_RADIAL, NC)

        # ── c. Reconstruct 2D emissivity via differentiable lerp ──────
        eps_pred_np = build_eps2d_lerp(
            coeffs_np,
            rho_flat_np,
            theta_flat_np,
            rho_radial_np,
            lerp_lo, lerp_hi, lerp_frac,
            n_harmonics=cfg.N_HARMONICS,
        )

        # ── d. Compute projected sinogram for metrics and plotting ────
        g_pred_np = np.array(w_bundle.w_ops.matvec(
            jnp.array(eps_pred_np)
        ))   # (128,)
        g_true_np = np.array(prof['g_ideal'])

        # ── e. Compute metrics ────────────────────────────────────────
        m = compute_metrics(
            eps_pred_np,
            eps_true_np,
            rho_flat_np,
            active_mask_np,
            w_bundle.W_norm,
            g_true_np,
        )
        m['idx'] = idx
        all_metrics.append(m)

        slabel = status_label(m['CC'], m['RelError'])
        print(f"  profile_{idx:04d}  "
              f"CC={m['CC']:.4f}  "
              f"PSNR={m['PSNR']:.1f}dB  "
              f"RelErr={m['RelError']:.4f}  "
              f"[{slabel}]")

        # ── f. Figure 1: per-profile reconstruction panel ────────────
        save_p = os.path.join(args.results_dir,
                              f'profile_{idx:04d}_reconstruction.png')
        plot_profile_panel(
            eps_pred    = eps_pred_np,
            eps_true    = eps_true_np,
            coeffs      = coeffs_np,
            rho_flat    = rho_flat_np,
            theta_flat  = theta_flat_np,
            rho_radial  = rho_radial_np,
            g_true      = g_true_np,
            g_pred      = g_pred_np,
            active_mask = active_mask_np,
            metrics     = m,
            profile_idx = idx,
            save_path   = save_p,
        )
        print(f"    → saved {save_p}")

        # Stash sinograms for Figure 3
        g_true_all.append(g_true_np)
        g_pred_all.append(g_pred_np)

    # ── 7. Multi-profile figures (skipped in single-profile mode) ─────
    if not single_mode and len(all_metrics) > 1:

        # Figure 2: aggregate dashboard
        dash_path = os.path.join(args.results_dir, 'evaluation_dashboard.png')
        plot_dashboard(all_metrics, dash_path)
        print(f"\n  → dashboard saved {dash_path}")

        # Figure 3: sinogram residuals
        sino_path = os.path.join(args.results_dir, 'sinogram_residuals.png')
        plot_sinogram_residuals(
            np.stack(g_true_all),
            np.stack(g_pred_all),
            sino_path,
        )
        print(f"  → sinogram residuals saved {sino_path}")

        # Figure 4: gradient flow diagnostic
        grad_path = os.path.join(args.results_dir, 'gradient_flow_diagnostic.png')
        plot_gradient_flow(
            model      = model_inst,
            params     = params,
            profiles   = eval_profiles,
            grids      = grids,
            w_bundle   = w_bundle,
            save_path  = grad_path,
        )
        print(f"  → gradient flow saved {grad_path}")

    # ── 8. JSON export ────────────────────────────────────────────────
    if args.save_metrics_json and all_metrics:
        # Build aggregate stats
        keys_agg = ['MSE', 'PSNR', 'CC', 'RelError',
                    'sinogram_MSE', 'sinogram_RelError']
        agg: Dict[str, float] = {}
        for k in keys_agg:
            vals = [m[k] for m in all_metrics if not np.isnan(m.get(k, np.nan))]
            if vals:
                agg[f'mean_{k}'] = float(np.mean(vals))
                agg[f'std_{k}']  = float(np.std(vals))

        json_dict = {
            f"profile_{m['idx']:04d}": {
                k: (float(v) if not isinstance(v, int) else v)
                for k, v in m.items()
            }
            for m in all_metrics
        }
        json_dict['aggregate'] = agg

        json_path = os.path.join(args.results_dir, 'metrics.json')
        with open(json_path, 'w') as f:
            json.dump(json_dict, f, indent=2)
        print(f"\n  → metrics JSON saved {json_path}")

    # ── 9. Print stdout table ─────────────────────────────────────────
    if all_metrics:
        print('\n' + format_metrics_table(all_metrics))

    print(f"\n{'═'*60}")
    print(f"  Evaluation complete.  "
          f"{len(all_metrics)} profile(s) processed.")
    print(f"{'═'*60}\n")


if __name__ == '__main__':
    main()
