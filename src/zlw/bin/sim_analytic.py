#!/usr/bin/env python3
"""
SPA toy for validating first–order geometric phase corrections under
PSD–driven (whitening) drift.

This script *only* validates the phase bias. Time shifts in this pure
phase–perturbation model are formally O(ε²) and are not plotted.

Output figure (2 panels, stacked vertically for two-column layouts):

  (top)   J_0(t) and J_ε(t) near t₀ (SNR² vs t slice, φ fixed = φ₀)
  (bottom) fractional phase error
           |δφ^(1) − Δφ_full| / |Δφ_full| vs ε   (log–log axes)

Example usage (as before):

    zlw-sim-spa \\
      --f-min 20 --f-max 512 --n-freq 4000 \\
      --t0 0.5 --phi0 1.15 --psd-power 2.0 \\
      --phi-alpha 0.0 --phi-beta 3e-3 --phi-gamma 1e-4 \\
      --eps-min 1e-3 --eps-max 1e-1 --n-eps 15 \\
      --t-window 3e-4 --n-t 2001 \\
      --outdir figs_spa_nonlinear

"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class SpaConfig:
    f_min: float
    f_max: float
    n_freq: int

    t0: float
    phi0: float

    psd_power: float

    phi_alpha: float
    phi_beta: float
    phi_gamma: float

    eps_min: float
    eps_max: float
    n_eps: int

    t_window: float
    n_t: int

    outdir: str


def parse_args() -> SpaConfig:
    p = argparse.ArgumentParser(
        description=(
            "SPA toy for validating first–order geometric phase corrections "
            "under PSD/whitening drift (phase-only)."
        )
    )

    # Frequency band and resolution
    p.add_argument("--f-min", type=float, default=20.0, help="Min frequency [Hz]")
    p.add_argument("--f-max", type=float, default=512.0, help="Max frequency [Hz]")
    p.add_argument("--n-freq", type=int, default=4000, help="Number of freq bins")

    # True extrinsic parameters
    p.add_argument("--t0", type=float, default=0.0, help="True coalescence time [s]")
    p.add_argument(
        "--phi0", type=float, default=0.0, help="True coalescence phase [rad]"
    )

    # PSD shape: S1(f) = f^p
    p.add_argument(
        "--psd-power",
        type=float,
        default=2.0,
        help="PSD power p in S1(f) = f^p (default: 2.0)",
    )

    # Phase perturbation Φ_a(f) = α + β f + γ f²
    p.add_argument("--phi-alpha", type=float, default=0.0, help="α in Φ_a(f)")
    p.add_argument("--phi-beta", type=float, default=3e-3, help="β in Φ_a(f)")
    p.add_argument("--phi-gamma", type=float, default=0.0, help="γ in Φ_a(f)")

    # ε grid for convergence study
    p.add_argument(
        "--eps-min", type=float, default=1e-4, help="Min ε for convergence scan"
    )
    p.add_argument(
        "--eps-max", type=float, default=1e-1, help="Max ε for convergence scan"
    )
    p.add_argument(
        "--n-eps", type=int, default=15, help="Number of ε points (log-spaced)"
    )

    # Time slice around t0 for J(t) panel
    p.add_argument(
        "--t-window",
        type=float,
        default=1e-3,
        help="Half-width of time window around t0 [s] for J(t) slice",
    )
    p.add_argument(
        "--n-t",
        type=int,
        default=1001,
        help="Number of time samples in [t0−t_window, t0+t_window]",
    )

    # Output
    p.add_argument(
        "--outdir", type=str, default="figs_spa", help="Output directory for figures"
    )

    args = p.parse_args()

    return SpaConfig(
        f_min=args.f_min,
        f_max=args.f_max,
        n_freq=args.n_freq,
        t0=args.t0,
        phi0=args.phi0,
        psd_power=args.psd_power,
        phi_alpha=args.phi_alpha,
        phi_beta=args.phi_beta,
        phi_gamma=args.phi_gamma,
        eps_min=args.eps_min,
        eps_max=args.eps_max,
        n_eps=args.n_eps,
        t_window=args.t_window,
        n_t=args.n_t,
        outdir=args.outdir,
    )


# ----------------------------------------------------------------------
# Basic building blocks
# ----------------------------------------------------------------------


def build_frequency_grid(
    f_min: float, f_max: float, n_freq: int
) -> Tuple[np.ndarray, float]:
    f = np.linspace(f_min, f_max, n_freq)
    df = f[1] - f[0] if n_freq > 1 else 0.0
    return f, df


def spa_amplitude(freqs: np.ndarray) -> np.ndarray:
    """SPA-like amplitude |h(f)| ∝ f^{-7/6}."""
    return freqs ** (-7.0 / 6.0)


def powerlaw_psd(freqs: np.ndarray, p: float) -> np.ndarray:
    """S₁(f) = f^p (up to an irrelevant constant)."""
    return freqs**p


def phi_perturbation(
    freqs: np.ndarray, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Φ_a(f) = α + β f + γ f²."""
    return alpha + beta * freqs + gamma * freqs**2


# ----------------------------------------------------------------------
# Matched filter and biases
# ----------------------------------------------------------------------


def rho_epsilon(
    eps: float,
    freqs: np.ndarray,
    df: float,
    W: np.ndarray,
    phi_a: np.ndarray,
    t: float,
    t0: float,
    phi: float,
    phi0: float,
) -> complex:
    """
    Complex matched filter ρ_ε(t, φ) in the SPA toy.

    d(f) ∝ h(f) e^{i(2π f t0 − φ0)}, h_{t,φ}(f) ∝ h(f) e^{i(2π f t − φ)}
    PSD drift enters only through phase factor e^{i ε Φ_a(f)}.
    """
    dt = t0 - t
    dphi = phi0 - phi
    phase = 2.0 * np.pi * freqs * dt - dphi + eps * phi_a
    return np.sum(W * np.exp(1j * phase)) * df


def phase_bias_full(
    eps: float,
    freqs: np.ndarray,
    df: float,
    W: np.ndarray,
    phi_a: np.ndarray,
) -> float:
    """
    Exact phase bias Δφ_full(ε) at (t, φ) = (t0, φ0).

    At t = t0, φ = φ0, ρ_0 is real positive (phase 0), and:

      ρ_ε(t0, φ0) = ∫ W(f) e^{i ε Φ_a(f)} df
      Δφ_full(ε)   = arg ρ_ε(t0, φ0).
    """
    integrand = W * np.exp(1j * eps * phi_a)
    rho_eps = np.sum(integrand * df)
    return float(np.angle(rho_eps))


def phase_bias_geometric(
    eps: float,
    freqs: np.ndarray,
    df: float,
    W: np.ndarray,
    phi_a: np.ndarray,
) -> float:
    """
    First-order geometric phase bias δφ^(1)(ε) = ε * δφ₁, with

      δφ₁ = (∫ W Φ_a df) / (∫ W df).

    This is exactly the φ-component of g^{-1} k for the SPA toy.
    """
    num = np.sum(W * phi_a) * df
    den = np.sum(W) * df
    delta_phi_1 = num / den
    return float(eps * delta_phi_1), float(delta_phi_1)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


def make_plots(
    cfg: SpaConfig,
    freqs: np.ndarray,
    df: float,
    W: np.ndarray,
    phi_a: np.ndarray,
    eps_grid: np.ndarray,
    phase_full: np.ndarray,
    phase_geom: np.ndarray,
) -> None:
    """
    Generate a 2-panel figure:

      (top)  J(t) slice near t0 for a representative epsilon (phase-only drift),
             with baseline and perturbed peaks marked.

      (bottom) Fractional phase error |δφ^(1) − Δφ_full| / |Δφ_full| vs ε
               on log–log axes.

    Interface kept identical to the previous version.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(cfg.outdir, exist_ok=True)

    # --- choose a representative ε for the J(t) slice ---
    # Prefer the largest epsilon that is still "reasonably perturbative"
    # in phase (e.g. < 10% fractional error). If none, fall back to mid-grid.
    eps_grid = np.asarray(eps_grid)
    phase_full = np.asarray(phase_full)
    phase_geom = np.asarray(phase_geom)

    # fractional phase error (will be reused below)
    denom = np.maximum(np.abs(phase_full), 1e-12)
    frac_phase = np.abs(phase_geom - phase_full) / denom

    mask_good = frac_phase < 0.1  # 10% threshold; adjust if you like
    if np.any(mask_good):
        # pick the largest epsilon that still satisfies the threshold
        eps_plot = eps_grid[mask_good][-1]
    else:
        # fallback: geometric median of the grid
        eps_plot = eps_grid[len(eps_grid) // 2]

    # --- time grid and J(t) for ε = 0 and ε = eps_plot ---
    t_grid = np.linspace(cfg.t0 - cfg.t_window, cfg.t0 + cfg.t_window, cfg.n_t)

    rho0_t = np.array(
        [
            rho_epsilon(
                eps=0.0,
                freqs=freqs,
                df=df,
                W=W,
                phi_a=phi_a,
                t=t,
                t0=cfg.t0,
                phi=cfg.phi0,
                phi0=cfg.phi0,
            )
            for t in t_grid
        ],
        dtype=complex,
    )
    rhoeps_t = np.array(
        [
            rho_epsilon(
                eps=eps_plot,
                freqs=freqs,
                df=df,
                W=W,
                phi_a=phi_a,
                t=t,
                t0=cfg.t0,
                phi=cfg.phi0,
                phi0=cfg.phi0,
            )
            for t in t_grid
        ],
        dtype=complex,
    )

    J0 = np.abs(rho0_t) ** 2
    Jeps = np.abs(rhoeps_t) ** 2

    # Normalise both by the same reference max so the relative drop is visible.
    J0_max = float(J0.max()) if J0.size else 1.0
    if J0_max <= 0.0:
        J0_max = 1.0
    J0 /= J0_max
    Jeps /= J0_max

    # Locate peaks (for vertical markers)
    idx0 = int(np.argmax(J0)) if J0.size else 0
    idxe = int(np.argmax(Jeps)) if Jeps.size else 0
    t_peak0 = t_grid[idx0]
    t_peake = t_grid[idxe]

    # --- build stacked figure (2 panels) ---
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(5.0, 5.0),
        sharex=False,
        constrained_layout=True,
    )

    # Top panel: J(t) slice near t0
    t_ms = (t_grid - cfg.t0) * 1e3
    ax1.plot(t_ms, J0, label=r"$J_0(t)$ (baseline)", lw=1.5)
    ax1.plot(
        t_ms,
        Jeps,
        ls="--",
        lw=1.5,
        label=rf"$J_\varepsilon(t)$, $\varepsilon={eps_plot:.3g}$",
    )
    ax1.axvline(
        (t_peak0 - cfg.t0) * 1e3,
        ls=":",
        color="C0",
        lw=1.0,
        label=r"Peak $(\varepsilon=0)$",
    )
    ax1.axvline(
        (t_peake - cfg.t0) * 1e3,
        ls=":",
        color="C1",
        lw=1.0,
        label=rf"Peak $(\varepsilon={eps_plot:.3g})$",
    )

    ax1.set_ylabel(r"Normalised SNR$^2$")
    ax1.set_xlabel(r"$t - t_0\ \mathrm{[ms]}$")
    ax1.set_title(r"$J(t)$ near $t_0$ (SPA, phase-only drift)")
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.legend(fontsize=7, loc="lower center")

    # Optionally tighten x-limits around the peaks a bit
    # span_ms = max(np.abs(t_ms).max(), 1e-3)
    # ax1.set_xlim(-span_ms, span_ms)

    # Bottom panel: fractional phase error vs ε (log–log)
    ax2.loglog(
        eps_grid,
        frac_phase,
        marker="s",
        lw=1.5,
        label=r"Phase: $|\delta\phi^{(1)}-\Delta\phi_{\mathrm{full}}|/|\Delta\phi_{\mathrm{full}}|$",
    )
    ax2.set_xlabel(r"$\varepsilon$")
    ax2.set_ylabel("Fractional error")
    ax2.set_title(
        "Convergence of first-order geometric phase\n"
        r"(SPA, nonlinear $\Phi_a(f)$; log–log axes)"
    )
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend(fontsize=7, loc="upper left")

    # Optional panel labels (uncomment if you want "(a)" / "(b)" baked in)
    # ax1.text(0.02, 0.95, "(a)", transform=ax1.transAxes,
    #          fontsize=10, fontweight="bold", va="top")
    # ax2.text(0.02, 0.95, "(b)", transform=ax2.transAxes,
    #          fontsize=10, fontweight="bold", va="top")

    out_path = os.path.join(cfg.outdir, "spa_psddrift_validation.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Figure written to: {out_path}")


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------


def main() -> None:
    cfg = parse_args()
    os.makedirs(cfg.outdir, exist_ok=True)

    # Frequency grid and basic ingredients
    freqs, df = build_frequency_grid(cfg.f_min, cfg.f_max, cfg.n_freq)
    h_amp = spa_amplitude(freqs)
    S1 = powerlaw_psd(freqs, cfg.psd_power)
    W = 4.0 * (h_amp**2) / S1
    phi_a = phi_perturbation(freqs, cfg.phi_alpha, cfg.phi_beta, cfg.phi_gamma)

    # ε grid (log-spaced)
    eps_grid = np.logspace(np.log10(cfg.eps_min), np.log10(cfg.eps_max), cfg.n_eps)

    print("=== SPA PSD–drift Validation (phase-only) ===")
    print(
        f"Band: [{cfg.f_min:.3g}, {cfg.f_max:.3g}] Hz,  n_freq = {cfg.n_freq}, "
        f"p = {cfg.psd_power:.3g}"
    )
    print(f"t0 = {cfg.t0:.6g} s, phi0 = {cfg.phi0:.6g} rad")
    print(
        f"Phi_a(f) = alpha + beta f + gamma f^2 with "
        f"alpha = {cfg.phi_alpha:g}, beta = {cfg.phi_beta:g}, gamma = {cfg.phi_gamma:g}"
    )

    # Precompute the unit-ε coefficient δφ₁
    _, delta_phi_1 = phase_bias_geometric(eps=1.0, freqs=freqs, df=df, W=W, phi_a=phi_a)
    print(f"\nFirst-order phase coefficient (unit ε): δφ₁ = {delta_phi_1:.6e} rad")

    phase_full = np.zeros_like(eps_grid)
    phase_geom = np.zeros_like(eps_grid)

    print("\nε        Δφ_full[rad]   Δφ_geom[rad]   frac_err_phase")
    for i, eps in enumerate(eps_grid):
        # Exact phase bias at (t0, φ0)
        dphi_full = phase_bias_full(eps, freqs, df, W, phi_a)
        # First-order geometric prediction
        dphi_geom, _ = phase_bias_geometric(eps, freqs, df, W, phi_a)

        phase_full[i] = dphi_full
        phase_geom[i] = dphi_geom

        frac = abs(dphi_geom - dphi_full) / max(abs(dphi_full), 1e-30)
        print(f"{eps:9.3e}  {dphi_full: .6e}  {dphi_geom: .6e}  {frac: .3e}")

    # Produce stacked figure (J(t) slice + phase convergence)
    make_plots(cfg, freqs, df, W, phi_a, eps_grid, phase_full, phase_geom)


if __name__ == "__main__":
    main()
