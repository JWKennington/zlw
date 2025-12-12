#!/usr/bin/env python
"""
PSD-based MP–MP Validation with Minimum-Phase Whitening + Optimisation
======================================================================

High-level goal
---------------
We want a *numerical* benchmark for the MP–MP first-order geometric
corrections in a realistic PSD-drift scenario:

  - A baseline PSD S1(f) (power-law),
  - A perturbed PSD S2(f) = S1(f) * exp(2 ε a(f)) with a Gaussian bump,
  - Minimum-phase whiteners H1, H2 built from (S1, S2),
  - SPA-like intrinsic template |h_intr(f)| ∝ f^{-7/6},
  - True data d(f) = h(t0, φ0), templates h_{(t,φ)}.

We evaluate:

  - A φ-fixed *numerical* time and phase bias for each ε,
  - The full MP–MP first-order corrections δt^(1)(ε), δφ^(1)(ε),
  - The fractional errors as a function of ε.

To account for the known φ-fixed estimator bias in time, we introduce a
*calibration factor* c_t obtained from the smallest-ε points:

  c_t ≈ median[ Δt_full(ε) / δt^(1)(ε) ]  (ε in the perturbative regime).

We then define a calibrated time prediction:

  δt_cal(ε) = c_t * δt^(1)(ε),

and show that |δt_cal - Δt_full| / |Δt_full| decreases with ε.

We also optionally perform a *2D optimisation* over (t, φ) using SciPy to
minimise |ρ_MPMP(t, φ) - ρ_0(t, φ)|^2, providing a non-perturbative MP–MP
shift independent of the φ-fixed estimator.

Figures
-------
The script writes three main figures into --outdir:

  1) psd_snr_validation.png
     - Top: S1(f) vs S2(f) (for ε = ε_max).
     - Bottom: SNR²(t) near t0 for baseline vs MP–MP (for ε = ε_max).

  2) psd_mpmp_convergence_uncalibrated.png
     - Fractional error in time and phase vs ε (uncalibrated).

  3) psd_mpmp_convergence_calibrated.png
     - Fractional error in *calibrated* time and phase vs ε.

Usage (example)
---------------
Assuming this script is exposed as the `zlw-sim-psd` entrypoint:

    zlw-sim-psd \
        --f-min 20 \
        --fs 4096 \
        --n-fft 16384 \
        --t0 0.0 --phi0 0.0 \
        --psd-power 2.0 \
        --bump-amp 0.5 \
        --bump-f0 150 \
        --bump-sigma 40 \
        --eps-min 1e-4 \
        --eps-max 1e-1 \
        --n-eps 15 \
        --t-bracket 5e-3 \
        --outdir figs_psd \
        --optimize-2d

The defaults are chosen to be “reasonable”; you can tune bump-amp,
ε-range, and t-bracket to explore different regimes.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from zlw.fourier import NumpyFourierBackend
from zlw.kernels import MPWhiteningFilter

# Optional SciPy optimisation for 2D (t, φ) shift
try:
    from scipy import optimize as opt

    HAVE_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_SCIPY = False


# -------------------------------------------------------------------------
# Config + CLI
# -------------------------------------------------------------------------


@dataclass
class MpmpPsdConfig:
    f_min: float
    fs: float
    n_fft: int
    t0: float
    phi0: float
    psd_power: float
    bump_amp: float
    bump_f0: float
    bump_sigma: float
    eps_min: float
    eps_max: float
    n_eps: int
    t_bracket: float
    n_t: int
    outdir: str
    make_plots: bool
    optimize_2d: bool


def parse_args() -> MpmpPsdConfig:
    parser = argparse.ArgumentParser(
        description=(
            "PSD-based MP–MP validation using minimum-phase whitening, "
            "first-order geometric corrections, and optional 2D optimisation."
        )
    )

    parser.add_argument(
        "--f-min", type=float, default=20.0, help="Minimum SPA frequency [Hz]"
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=4096.0,
        help="Sampling rate fs [Hz] (Nyquist = fs/2, default 4096)",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=16384,
        help="FFT length (even, default 16384)",
    )

    parser.add_argument(
        "--t0", type=float, default=0.0, help="True coalescence time t0 [s]"
    )
    parser.add_argument(
        "--phi0", type=float, default=0.0, help="True coalescence phase φ0 [rad]"
    )

    parser.add_argument(
        "--psd-power",
        type=float,
        default=2.0,
        help="Baseline PSD power p in S1(f) ∝ f^p (default 2.0)",
    )

    parser.add_argument(
        "--bump-amp",
        type=float,
        default=0.5,
        help="Amplitude of Gaussian bump a(f) (default 0.5)",
    )
    parser.add_argument(
        "--bump-f0",
        type=float,
        default=150.0,
        help="Center frequency of Gaussian bump [Hz] (default 150)",
    )
    parser.add_argument(
        "--bump-sigma",
        type=float,
        default=40.0,
        help="Width σ of Gaussian bump [Hz] (default 40)",
    )

    parser.add_argument(
        "--eps-min",
        type=float,
        default=1e-4,
        help="Minimum ε for log-spaced sweep (default 1e-4)",
    )
    parser.add_argument(
        "--eps-max",
        type=float,
        default=1e-1,
        help="Maximum ε for log-spaced sweep (default 1e-1)",
    )
    parser.add_argument(
        "--n-eps",
        type=int,
        default=15,
        help="Number of ε samples (log-spaced, default 15)",
    )

    parser.add_argument(
        "--t-bracket",
        type=float,
        default=5e-3,
        help="Half-width around t0 for time scan [s] (default 5e-3 => ±5 ms)",
    )
    parser.add_argument(
        "--n-t",
        type=int,
        default=2001,
        help="Number of time samples in [t0 ± t_bracket] (default 2001)",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="figs_psd_mpmp",
        help="Output directory for figures (default figs_psd_mpmp)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="If set, do not generate any plots.",
    )
    parser.add_argument(
        "--optimize-2d",
        action="store_true",
        help="If set, attempt 2D (t,φ) optimisation using SciPy (if available).",
    )

    args = parser.parse_args()

    return MpmpPsdConfig(
        f_min=args.f_min,
        fs=args.fs,
        n_fft=args.n_fft,
        t0=args.t0,
        phi0=args.phi0,
        psd_power=args.psd_power,
        bump_amp=args.bump_amp,
        bump_f0=args.bump_f0,
        bump_sigma=args.bump_sigma,
        eps_min=args.eps_min,
        eps_max=args.eps_max,
        n_eps=args.n_eps,
        t_bracket=args.t_bracket,
        n_t=args.n_t,
        outdir=args.outdir,
        make_plots=not args.no_plots,
        optimize_2d=args.optimize_2d,
    )


# -------------------------------------------------------------------------
# Basic utilities
# -------------------------------------------------------------------------


def build_frequency_grid(cfg: MpmpPsdConfig) -> Tuple[np.ndarray, float]:
    fb = NumpyFourierBackend()
    freqs = fb.rfftfreq(cfg.n_fft, 1.0 / cfg.fs)
    df = freqs[1] - freqs[0] if freqs.size > 1 else 0.0
    return freqs, df


def spa_intrinsic_amplitude(freqs: np.ndarray, f_min: float) -> np.ndarray:
    """SPA-like amplitude |h_intr(f)| ∝ f^(-7/6) for f >= f_min."""
    amp = np.zeros_like(freqs)
    mask = freqs >= f_min
    amp[mask] = freqs[mask] ** (-7.0 / 6.0)
    return amp


def baseline_psd(freqs: np.ndarray, p: float, s0: float = 1.0) -> np.ndarray:
    """Baseline PSD S1(f) = s0 * max(f,f_floor)^p (strictly positive)."""
    f_floor = np.maximum(freqs, 10.0)
    return s0 * (f_floor**p)


def gaussian_bump(freqs: np.ndarray, amp: float, f0: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((freqs - f0) / sigma) ** 2)


def build_whitener(psd: np.ndarray, cfg: MpmpPsdConfig) -> np.ndarray:
    fb = NumpyFourierBackend()
    mp = MPWhiteningFilter(psd=psd, fs=cfg.fs, n_fft=cfg.n_fft, fb=fb)
    return mp.frequency_response()


def h_fd(freqs: np.ndarray, h_amp: np.ndarray, t: float, phi: float) -> np.ndarray:
    """Frequency-domain template h(f; t, φ) = |h(f)| exp(i (2π f t - φ))."""
    phase = 2.0 * np.pi * freqs * t - phi
    return h_amp * np.exp(1j * phase)


# -------------------------------------------------------------------------
# Matched filters + numeric estimators (1D, φ fixed)
# -------------------------------------------------------------------------


def rho_baseline(
    freqs: np.ndarray,
    df: float,
    H1: np.ndarray,
    h_amp: np.ndarray,
    t: float,
    phi: float,
    t0: float,
    phi0: float,
) -> complex:
    d_fd = h_fd(freqs, h_amp, t0, phi0)
    D1 = H1 * d_fd
    T1 = H1 * h_fd(freqs, h_amp, t, phi)
    return np.sum(D1 * np.conj(T1)) * df


def rho_mpmp(
    freqs: np.ndarray,
    df: float,
    H1: np.ndarray,
    H2: np.ndarray,
    h_amp: np.ndarray,
    t: float,
    phi: float,
    t0: float,
    phi0: float,
) -> complex:
    d_fd = h_fd(freqs, h_amp, t0, phi0)
    D2 = H2 * d_fd
    T1 = H1 * h_fd(freqs, h_amp, t, phi)
    return np.sum(D2 * np.conj(T1)) * df


def refine_peak(t_grid: np.ndarray, J: np.ndarray) -> Tuple[float, float]:
    """Parabolic refinement of a peak around argmax(J)."""
    idx = int(np.argmax(J))
    if idx == 0 or idx == len(J) - 1:
        return float(t_grid[idx]), float(J[idx])

    y0, y1, y2 = J[idx - 1], J[idx], J[idx + 1]
    t0, t1, t2 = t_grid[idx - 1], t_grid[idx], t_grid[idx + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return float(t1), float(y1)

    delta = 0.5 * (y0 - y2) / denom
    dt = t1 - t0
    t_peak = t1 + delta * dt
    J_peak = y1 - 0.25 * (y0 - y2) * delta
    return float(t_peak), float(J_peak)


def numeric_biases_1d(
    cfg: MpmpPsdConfig,
    freqs: np.ndarray,
    df: float,
    H1: np.ndarray,
    H2: np.ndarray,
    h_amp: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray, float, float]:
    """
    Compute full numeric (φ-fixed) MP–MP biases:

      - Δt_full from peak shift of J(t) = |ρ(t,φ0)|²,
      - Δφ_full from phase at (t0, φ0).

    Returns:
      Δt_full, Δφ_full, t_grid, J0(t), J_mpmp(t), t_peak0, t_peak_mp.
    """
    t_grid = np.linspace(cfg.t0 - cfg.t_bracket, cfg.t0 + cfg.t_bracket, cfg.n_t)

    rho0_t = np.array(
        [
            rho_baseline(freqs, df, H1, h_amp, t, cfg.phi0, cfg.t0, cfg.phi0)
            for t in t_grid
        ],
        dtype=complex,
    )
    rho_mp_t = np.array(
        [
            rho_mpmp(freqs, df, H1, H2, h_amp, t, cfg.phi0, cfg.t0, cfg.phi0)
            for t in t_grid
        ],
        dtype=complex,
    )

    J0 = np.abs(rho0_t) ** 2
    Jmp = np.abs(rho_mp_t) ** 2

    t_peak0, _ = refine_peak(t_grid, J0)
    t_peak_mp, _ = refine_peak(t_grid, Jmp)
    delta_t_full = t_peak_mp - t_peak0

    # Phase at true (t0, φ0)
    rho0_true = rho_baseline(freqs, df, H1, h_amp, cfg.t0, cfg.phi0, cfg.t0, cfg.phi0)
    rho_mp_true = rho_mpmp(freqs, df, H1, H2, h_amp, cfg.t0, cfg.phi0, cfg.t0, cfg.phi0)
    delta_phi_full = np.angle(rho_mp_true) - np.angle(rho0_true)
    delta_phi_full = float(np.arctan2(np.sin(delta_phi_full), np.cos(delta_phi_full)))

    return (
        float(delta_t_full),
        float(delta_phi_full),
        t_grid,
        J0,
        Jmp,
        float(t_peak0),
        float(t_peak_mp),
    )


# -------------------------------------------------------------------------
# Analytic first-order MP–MP corrections
# -------------------------------------------------------------------------


def analytic_full_first_order(
    freqs: np.ndarray,
    df: float,
    H1: np.ndarray,
    H2: np.ndarray,
    htilde_intr: np.ndarray,
) -> Tuple[float, float]:
    """
    Full first-order MP–MP corrections (δt^(1), δφ^(1)) for the given H1, H2.

    Using:

      W1 = H1, W2 = H2, dW = W2 - W1,
      h̃(f) = intrinsic template (no extrinsic phase),
      H1h = W1 h̃.

    Define:

      d_i_tt = - (2π)² ∫ f² |H1h|² df
      d_i_pp = - ∫ |H1h|² df

      d_di_t =  2π ∫ f Im[dW h̃ conj(H1h)] df
      d_di_p =      ∫ Im[dW h̃ conj(H1h)] df

    Then:

      δt^(1) = - d_di_t / d_i_tt
      δφ^(1) = - d_di_p / d_i_pp
    """
    W1 = H1
    W2 = H2
    dW = W2 - W1

    H1h = W1 * htilde_intr
    absH1h2 = np.abs(H1h) ** 2

    d_i_tt = -((2.0 * np.pi) ** 2) * np.sum((freqs**2) * absH1h2) * df
    d_i_pp = -np.sum(absH1h2) * df

    data_fd = htilde_intr
    conj_H1h = np.conj(H1h)

    integrand_t = freqs * np.imag(dW * data_fd * conj_H1h)
    d_di_t = 2.0 * np.pi * np.sum(integrand_t) * df

    integrand_p = np.imag(dW * data_fd * conj_H1h)
    d_di_p = np.sum(integrand_p) * df

    delta_t = -d_di_t / d_i_tt if d_i_tt != 0.0 else np.nan
    delta_phi = -d_di_p / d_i_pp if d_i_pp != 0.0 else np.nan
    return float(delta_t), float(delta_phi)


# -------------------------------------------------------------------------
# Optional 2D optimisation over (t, φ)
# -------------------------------------------------------------------------


def optimise_shift_2d(
    cfg: MpmpPsdConfig,
    freqs: np.ndarray,
    df: float,
    H1: np.ndarray,
    H2: np.ndarray,
    h_amp: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """
    Optional 2D optimisation: minimise |ρ_MPMP(t,φ) - ρ_0(t,φ)|² over (t, φ).

    This yields a non-perturbative MP–MP shift (Δt_opt, Δφ_opt) independent of
    the φ-fixed estimator.
    """
    if not HAVE_SCIPY:
        print("[sim_psd_opt] SciPy not available; skipping 2D optimisation.")
        return None

    def objective(theta: np.ndarray) -> float:
        t, phi = theta
        rho0 = rho_baseline(freqs, df, H1, h_amp, t, phi, cfg.t0, cfg.phi0)
        rho_mp = rho_mpmp(freqs, df, H1, H2, h_amp, t, phi, cfg.t0, cfg.phi0)
        diff = rho_mp - rho0
        return float(np.abs(diff) ** 2)

    x0 = np.array([cfg.t0, cfg.phi0], dtype=float)
    res = opt.minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 200, "xatol": 1e-10, "fatol": 1e-16},
    )

    if not res.success:
        print(f"[sim_psd_opt] 2D optimisation failed: {res.message}")
        return None

    t_opt, phi_opt = res.x
    dt_opt = float(t_opt - cfg.t0)
    dphi_opt = float(phi_opt - cfg.phi0)
    dphi_opt = float(np.arctan2(np.sin(dphi_opt), np.cos(dphi_opt)))

    print(
        f"[sim_psd_opt] 2D optimisation result: Δt_opt = {dt_opt:.6e} s, "
        f"Δφ_opt = {dphi_opt:.6e} rad"
    )
    return dt_opt, dphi_opt


# -------------------------------------------------------------------------
# Convergence plots + PSD/SNR figure
# -------------------------------------------------------------------------


def make_psd_snr_figure(
    cfg: MpmpPsdConfig,
    freqs: np.ndarray,
    S1: np.ndarray,
    S2_plot: np.ndarray,
    t_grid: np.ndarray,
    J0_plot: np.ndarray,
    Jmp_plot: np.ndarray,
    t_peak0: float,
    t_peak_mp: float,
) -> None:
    """
    Make the PSD + SNR^2(t) figure for the paper.

    Top panel:
      - Baseline S1(f)
      - S2(f) at eps = eps_max (orange dashed)
      - An exaggerated PSD S2_big(f) for eps = 0.5 (green dash–dot), to
        illustrate qualitatively how the Gaussian bump deformation looks.

    Bottom panel:
      - Normalised SNR^2(t) for baseline and MP–MP at eps = eps_max,
        zoomed around t0.
      - A horizontal dotted line at the first-order predicted peak
        reduction: J_pred / J0 ≈ 1 - 2 ε_max <a>_q.
    """
    os.makedirs(cfg.outdir, exist_ok=True)

    fig, (ax_psd, ax_snr) = plt.subplots(
        2, 1, figsize=(6.0, 6.0), constrained_layout=True
    )

    # ----------------------
    # PSD panel
    # ----------------------
    mask_band = freqs >= cfg.f_min

    # Baseline
    ax_psd.loglog(
        freqs[mask_band],
        S1[mask_band],
        label=r"$S_1(f)$ baseline",
    )

    # S2 at eps = eps_max (already passed in as S2_plot)
    eps_max = float(cfg.eps_max)
    ax_psd.loglog(
        freqs[mask_band],
        S2_plot[mask_band],
        ls="--",
        color="C1",
        label=rf"$S_2(f)$ perturbed, $\varepsilon = {eps_max:g}$",
    )

    # Reconstruct bump shape a(f) from S1 and S2 at eps_max:
    #   S2 = S1 * exp(2 ε a)  =>  a = (log(S2/S1)) / (2 ε)
    tiny = 1e-30
    ratio = np.ones_like(S1)
    np.divide(S2_plot, S1, out=ratio, where=S1 > 0.0)
    denom = 2.0 * max(eps_max, 1e-12)
    a_psd_recon = np.log(ratio) / denom

    # Exaggerated perturbation with ε_big = 0.5 (for qualitative illustration)
    eps_big = 1.0
    S2_big = S1 * np.exp(2.0 * eps_big * a_psd_recon)

    ax_psd.loglog(
        freqs[mask_band],
        S2_big[mask_band],
        ls="-.",
        color="C2",
        label=r"$S_2(f)$ perturbed, $\varepsilon = 0.5$",
    )

    ax_psd.set_xlabel(r"$f\,[\mathrm{Hz}]$")
    ax_psd.set_ylabel("PSD")
    ax_psd.set_title("Baseline and perturbed PSDs (PSD drift)")
    ax_psd.grid(True, which="both", ls=":", alpha=0.5)
    ax_psd.legend(fontsize=8)

    # ----------------------
    # SNR^2(t) panel
    # ----------------------
    # Normalise J by the baseline peak
    J0_max = float(np.max(J0_plot))
    J0_norm = J0_plot / J0_max
    Jmp_norm = Jmp_plot / J0_max
    t_ms = (t_grid - cfg.t0) * 1e3

    ax_snr.plot(
        t_ms,
        J0_norm,
        label=r"$J_0(t)$ (baseline)",
        lw=1.5,
        color="C0",
    )
    ax_snr.plot(
        t_ms,
        Jmp_norm,
        ls="--",
        lw=1.5,
        color="C1",
        label=rf"$J_{{\rm MP\text{{-}}MP}}(t)$, $\varepsilon = {eps_max:g}$",
    )
    ax_snr.axvline(
        (t_peak0 - cfg.t0) * 1e3,
        color="C0",
        ls=":",
        label="Peak (baseline)",
    )
    ax_snr.axvline(
        (t_peak_mp - cfg.t0) * 1e3,
        color="C1",
        ls=":",
        label="Peak (MP–MP)",
    )

    # --- First-order predicted peak drop: J_pred/J0 ≈ 1 - 2 ε_max <a>_q
    # Rebuild SPA amplitude and weights for <a>_q using the toy model.
    h_amp = spa_intrinsic_amplitude(freqs, cfg.f_min)
    # weight q ∝ |h(f)|^2 / S1(f)
    w_q = np.zeros_like(freqs)
    np.divide(h_amp**2, S1, out=w_q, where=S1 > 0.0)
    if np.any(w_q > 0.0):
        a_mean = float(np.sum(a_psd_recon * w_q) / np.sum(w_q))
        J_pred_over_J0 = 1.0 - 2.0 * eps_max * a_mean
        ax_snr.axhline(
            J_pred_over_J0,
            color="C3",
            ls=":",
            lw=1.2,
            label=r"First-order $J_{\rm pred}/J_0$",
        )

    # Extra zoom around t0
    zoom_width_ms = 0.2 * cfg.t_bracket * 1e3
    ax_snr.set_xlim(-zoom_width_ms, zoom_width_ms)

    mask_zoom = np.abs(t_ms) <= zoom_width_ms
    if np.any(mask_zoom):
        y_min = min(J0_norm[mask_zoom].min(), Jmp_norm[mask_zoom].min())
        y_max = max(J0_norm[mask_zoom].max(), Jmp_norm[mask_zoom].max())
    else:
        y_min, y_max = J0_norm.min(), Jmp_norm.max()
    margin = 5e-4
    ax_snr.set_ylim(y_min - margin, y_max + margin)

    ax_snr.set_xlabel(r"$t - t_0\,[\mathrm{ms}]$")
    ax_snr.set_ylabel(r"Normalised SNR$^2$")
    ax_snr.set_title(r"SNR$^2(t)$ near $t_0$ (MP–MP whitening, $\phi$ fixed)$")
    ax_snr.grid(True, ls=":", alpha=0.5)
    ax_snr.legend(fontsize=8)

    out_path = os.path.join(cfg.outdir, "psd_snr_validation.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[sim_psd_opt] Wrote {out_path}")


def make_convergence_figures(
    cfg: MpmpPsdConfig,
    eps_grid: np.ndarray,
    frac_t: np.ndarray,
    frac_t_cal: np.ndarray,
    frac_phi: np.ndarray,
) -> None:
    """
    Make convergence figures for first-order MP–MP corrections.

    Time errors are plotted as markers only (no connecting line), and
    points with NaN are automatically skipped. Phase errors are plotted
    as line+markers (they behave very cleanly).
    """
    os.makedirs(cfg.outdir, exist_ok=True)

    # -------------------------
    # Uncalibrated time vs phase
    # -------------------------
    fig1, ax1 = plt.subplots(figsize=(6.0, 4.5))

    # Time (uncalibrated): markers only
    ax1.loglog(
        eps_grid,
        frac_t,
        marker="o",
        ls="None",
        ms=5,
        label=r"Time shift: $|\delta t^{(1)}-\Delta t_{\rm full}|/|\Delta t_{\rm full}|$",
    )

    # Phase: line + markers
    ax1.loglog(
        eps_grid,
        frac_phi,
        marker="s",
        lw=1.5,
        label=r"Phase shift: $|\delta\phi^{(1)}-\Delta\phi_{\rm full}|/|\Delta\phi_{\rm full}|$",
    )

    ax1.set_xlabel(r"$\varepsilon$")
    ax1.set_ylabel("Fractional error")
    ax1.set_title(
        "Convergence of first-order PSD-drift corrections\n"
        "(uncalibrated time; log–log axes)"
    )
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend(fontsize=8)

    out_uncal = os.path.join(cfg.outdir, "psd_mpmp_convergence_uncalibrated.png")
    fig1.savefig(out_uncal, dpi=300)
    plt.close(fig1)
    print(f"[sim_psd_opt] Wrote {out_uncal}")

    # -------------------------
    # Calibrated time vs phase
    # -------------------------
    fig2, ax2 = plt.subplots(figsize=(6.0, 4.5))

    # Time (calibrated): markers only
    ax2.loglog(
        eps_grid,
        frac_t_cal,
        marker="o",
        ls="None",
        ms=5,
        label=r"Time shift (cal): $|\delta t_{\rm cal}-\Delta t_{\rm full}|/|\Delta t_{\rm full}|$",
    )

    # Phase: line + markers
    ax2.loglog(
        eps_grid,
        frac_phi,
        marker="s",
        lw=1.5,
        label=r"Phase shift: $|\delta\phi^{(1)}-\Delta\phi_{\rm full}|/|\Delta\phi_{\rm full}|$",
    )

    ax2.set_xlabel(r"$\varepsilon$")
    ax2.set_ylabel("Fractional error")
    ax2.set_title(
        "Convergence of first-order PSD-drift corrections\n"
        "(calibrated time; log–log axes)"
    )
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend(fontsize=8)

    out_cal = os.path.join(cfg.outdir, "psd_mpmp_convergence_calibrated.png")
    fig2.savefig(out_cal, dpi=300)
    plt.close(fig2)
    print(f"[sim_psd_opt] Wrote {out_cal}")


def make_snr_change_figure(
    cfg: MpmpPsdConfig,
    eps_grid: np.ndarray,
    amp_full: np.ndarray,
    amp_pred: np.ndarray,
) -> None:
    """
    Plot the fractional change in the peak of J(=SNR^2) as a function of ε.

    We show:
      - numerical   : [J_max(ε) - J0_max] / J0_max (markers)
      - first-order : -2 ε <a>_q          (solid line)
    """
    os.makedirs(cfg.outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    ax.semilogx(
        eps_grid,
        amp_full,
        marker="o",
        ls="None",
        ms=5,
        label=r"numeric $\,[J_{\max}(\varepsilon)-J_0]/J_0$",
    )
    ax.semilogx(
        eps_grid,
        amp_pred,
        lw=1.5,
        color="C3",
        label=r"first order $-2\,\varepsilon\langle a\rangle_q$",
    )

    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"Fractional change in peak $J$")
    ax.set_title(
        "Change in peak SNR$^2$ under PSD drift\n"
        r"(numeric vs. $-2\,\varepsilon\langle a\rangle_q$)"
    )
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)

    out_path = os.path.join(cfg.outdir, "psd_snr_change_vs_eps.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[sim-psd] Wrote {out_path}")


# -------------------------------------------------------------------------
# Main driver
# -------------------------------------------------------------------------


def main() -> None:
    cfg = parse_args()
    os.makedirs(cfg.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Frequency grid, intrinsic template, baseline PSD + whitener
    # ------------------------------------------------------------------
    freqs, df = build_frequency_grid(cfg)
    h_amp = spa_intrinsic_amplitude(freqs, cfg.f_min)
    # For analytic formulas we only need the (real) SPA amplitude; the
    # coalescence phase at (t0, phi0) is carried by exp(i φ) factors.
    htilde_intr = h_amp.astype(float)

    S1 = baseline_psd(freqs, cfg.psd_power)
    a_psd = gaussian_bump(freqs, cfg.bump_amp, cfg.bump_f0, cfg.bump_sigma)
    H1 = build_whitener(S1, cfg)

    # --- Analytic mean <a>_q for the SNR^2 peak amplitude change (App. A) ---
    # q(f) ∝ |W1(f) h(f; t0,φ0)|^2, evaluated at the true (t0,φ0).
    H1h = H1 * h_amp
    w_q = np.abs(H1h) ** 2
    a_mean = float(np.sum(a_psd * w_q) / np.sum(w_q))
    print(f"[info] Weighted mean <a>_q = {a_mean:.6e}")

    # ------------------------------------------------------------------
    # ε grid and storage
    # ------------------------------------------------------------------
    eps_grid = np.logspace(np.log10(cfg.eps_min), np.log10(cfg.eps_max), cfg.n_eps)

    dt_full = np.zeros_like(eps_grid)
    dphi_full = np.zeros_like(eps_grid)
    dt_geom = np.zeros_like(eps_grid)
    dphi_geom = np.zeros_like(eps_grid)

    # New: fractional change in peak J(=SNR^2) at φ=φ0:
    #   amp_full[i] = [J_max(ε_i) - J0_max]/J0_max (numeric)
    amp_full = np.zeros_like(eps_grid)

    # For plotting at ε = ε_max
    J0_plot = None
    Jmp_plot = None
    t_grid_plot = None
    t_peak0_plot = None
    t_peak_mp_plot = None
    S2_plot = None
    H2_for_opt = None

    # ------------------------------------------------------------------
    # Baseline J(t) and peak at ε = 0 (for amplitude normalisation)
    # ------------------------------------------------------------------
    H2_dummy = H1  # MP–MP = baseline when ε=0
    (
        _,
        _,
        t_grid0,
        J0_t0,
        _,
        t_peak0_baseline,
        _,
    ) = numeric_biases_1d(cfg, freqs, df, H1, H2_dummy, h_amp)
    J0_max = float(np.max(J0_t0))
    print(
        f"[info] Baseline peak J0_max = {J0_max:.6e}, "
        f"t_peak0 = {t_peak0_baseline:.6e} s (t0 = {cfg.t0:.6e} s)"
    )

    # ------------------------------------------------------------------
    # Sweep over ε: compute numeric + analytic corrections
    # ------------------------------------------------------------------
    for i, eps in enumerate(eps_grid):
        S2 = S1 * np.exp(2.0 * eps * a_psd)
        H2 = build_whitener(S2, cfg)

        # Analytic first-order time/phase corrections
        dt_geom[i], dphi_geom[i] = analytic_full_first_order(
            freqs, df, H1, H2, htilde_intr
        )

        # Numeric (φ-fixed) biases and J(t)
        (
            dt_full[i],
            dphi_full[i],
            t_grid,
            J0_t,
            Jmp_t,
            t_peak0,
            t_peak_mp,
        ) = numeric_biases_1d(cfg, freqs, df, H1, H2, h_amp)

        # Numeric fractional change in peak J at φ = φ0
        Jmax_eps = float(np.max(Jmp_t))
        amp_full[i] = (Jmax_eps - J0_max) / J0_max

        print(
            f"[eps={eps: .3e}] "
            f"Δt_full={dt_full[i]: .3e} s, δt^(1)={dt_geom[i]: .3e} s; "
            f"Δφ_full={dphi_full[i]: .3e} rad, δφ^(1)={dphi_geom[i]: .3e} rad; "
            f"ΔJ/J (num)={amp_full[i]: .3e}"
        )

        # Save plotting data for the largest ε (ε ≈ eps_max)
        if i == len(eps_grid) - 1:
            S2_plot = S2
            J0_plot = J0_t
            Jmp_plot = Jmp_t
            t_grid_plot = t_grid
            t_peak0_plot = t_peak0
            t_peak_mp_plot = t_peak_mp
            H2_for_opt = H2

    # ------------------------------------------------------------------
    # Fractional errors + time calibration (with masking)
    # ------------------------------------------------------------------
    tiny = 1e-20

    # Masks for nonzero true shifts
    mask_nonzero_t = np.abs(dt_full) > tiny
    mask_nonzero_phi = np.abs(dphi_full) > tiny

    # Initialise with NaNs so loglog() will skip invalid points
    frac_t = np.full_like(dt_full, np.nan, dtype=float)
    frac_t_cal = np.full_like(dt_full, np.nan, dtype=float)
    frac_phi = np.full_like(dphi_full, np.nan, dtype=float)

    # Reference scale for "meaningful" time shifts: median over a mid-ε window
    idx = np.arange(len(eps_grid))
    mid_mask = mask_nonzero_t & (idx >= 2) & (idx <= len(eps_grid) - 3)
    if np.any(mid_mask):
        ref_scale_t = float(np.median(np.abs(dt_full[mid_mask])))
    else:
        ref_scale_t = (
            float(np.median(np.abs(dt_full[mask_nonzero_t])))
            if np.any(mask_nonzero_t)
            else 0.0
        )

    # Threshold below which Δt_full is treated as effectively zero
    thresh_t = 1e-3 * ref_scale_t if ref_scale_t > 0.0 else np.inf
    mask_good_t = mask_nonzero_t & (np.abs(dt_full) >= thresh_t)

    # Uncalibrated fractional errors
    frac_t[mask_good_t] = np.abs(dt_geom[mask_good_t] - dt_full[mask_good_t]) / np.abs(
        dt_full[mask_good_t]
    )
    frac_phi[mask_nonzero_phi] = np.abs(
        dphi_geom[mask_nonzero_phi] - dphi_full[mask_nonzero_phi]
    ) / np.abs(dphi_full[mask_nonzero_phi])

    # Time calibration: use first few ε values in perturbative regime where Δt is reliable
    n_cal = min(5, len(eps_grid))
    mask_cal = mask_good_t & (idx < n_cal)
    if np.any(mask_cal):
        ratios = dt_full[mask_cal] / dt_geom[mask_cal]
        c_t = float(np.median(ratios))
    else:
        c_t = 1.0

    print(f"[info] Time calibration factor c_t ≈ {c_t:.6e}")

    dt_geom_cal = c_t * dt_geom
    frac_t_cal[mask_good_t] = np.abs(
        dt_geom_cal[mask_good_t] - dt_full[mask_good_t]
    ) / np.abs(dt_full[mask_good_t])

    # ------------------------------------------------------------------
    # Analytic prediction for peak SNR^2 change: ΔJ/J ≈ -2 ε <a>_q
    # ------------------------------------------------------------------
    amp_pred = -2.0 * eps_grid * a_mean

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    if cfg.make_plots:
        if (
            S2_plot is not None
            and J0_plot is not None
            and Jmp_plot is not None
            and t_grid_plot is not None
            and t_peak0_plot is not None
            and t_peak_mp_plot is not None
        ):
            make_psd_snr_figure(
                cfg,
                freqs,
                S1,
                S2_plot,
                t_grid_plot,
                J0_plot,
                Jmp_plot,
                t_peak0_plot,
                t_peak_mp_plot,
            )

        make_convergence_figures(cfg, eps_grid, frac_t, frac_t_cal, frac_phi)

        # New: SNR peak amplitude change vs ε (numeric vs analytic)
        make_snr_change_figure(cfg, eps_grid, amp_full, amp_pred)

    # ------------------------------------------------------------------
    # Optional 2D optimisation at ε = ε_max
    # ------------------------------------------------------------------
    if cfg.optimize_2d and H2_for_opt is not None:
        optimise_shift_2d(cfg, freqs, df, H1, H2_for_opt, h_amp)


if __name__ == "__main__":
    main()
