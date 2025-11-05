#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_analytic_vs_zlw.py  — clean, minimal

Compares the CORRECT analytic first-order MP–MP formulas against your library:

  from zlw.corrections import MPMPCorrection
  from zlw.kernels     import MPWhiteningFilter

Analytic MP pair (closed-form):
    a(f)   = (ε/2) * ln(1 + (f/fc)^2)
    Φ_a(f) = ε * arctan(f/fc)

Weights (must match denominators in your code):
    W(f) = | W1(f) * h~(f) |^2,  with W1 built from S1 via MPWhiteningFilter

Analytic first-order (correct):
    δφ¹  =  (∫ W Φ_a df) / (∫ W df)
    δt¹  =  (∫ f W Φ_a df) / (2π ∫ f^2 W df)

Outputs: ./out_zlw_compare/{results.csv, compare_dt.png, compare_dphi.png, residuals_*.png, convergence_loglog.png}
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import rfftfreq

# ---- Import your classes (fallback shim only if zlw not importable here) ----
HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

def import_zlw_or_shim():
    try:
        from zlw.corrections import MPMPCorrection  # type: ignore
        from zlw.kernels import MPWhiteningFilter   # type: ignore
        return MPMPCorrection, MPWhiteningFilter
    except Exception:
        import types, importlib
        zlw_pkg = types.ModuleType("zlw"); sys.modules.setdefault("zlw", zlw_pkg)
        corr = importlib.import_module("corrections")
        kern = importlib.import_module("kernels")
        sys.modules["zlw.corrections"] = corr
        sys.modules["zlw.kernels"] = kern
        from zlw.corrections import MPMPCorrection  # type: ignore
        from zlw.kernels import MPWhiteningFilter   # type: ignore
        return MPMPCorrection, MPWhiteningFilter

MPMPCorrection, MPWhiteningFilter = import_zlw_or_shim()

OUTDIR = os.path.join(HERE, "out_zlw_compare3")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------- Grid, band, template ----------------
fs = 4096.0
N  = 2**16
f  = rfftfreq(N, d=1.0/fs)

fmin, fmax = 20.0, 512.0
band = (f >= fmin) & (f <= fmax)

# Inspiral-like template (SPA-ish amplitude; phase won’t affect W)
Mchirp = 1.2
const = (3.0/128.0) * (np.pi)**(-5.0/3.0)

amp = np.zeros_like(f)
amp[band] = np.power(f[band], -7.0/6.0) * np.exp(-(fmin / f[band])**8)
psi = np.zeros_like(f)
psi[band] = - const * np.power(Mchirp * f[band], -5.0/3.0)
h = amp * np.exp(1j * psi)

# Reference PSD
S1 = np.ones_like(f)

# Normalize so ∫_band |h|^2/S1 df ≈ 1 (stability)
norm = np.trapz(((np.abs(h)**2)/S1)[band], f[band])
if norm > 0:
    h = h / np.sqrt(norm)

# ---------------- Analytic MP pair (a, Φ_a) ----------------
fc = 50.0
def analytic_pair(fgrid, eps, fc_hz):
    a     = 0.5*eps*np.log1p((fgrid/fc_hz)**2)
    Phi_a = eps*np.arctan(fgrid/fc_hz)
    return a, Phi_a

# ---------------- Build W1 via your MPWhiteningFilter ----------------
def build_mp_kernel(psd, fs, n_fft):
    filt = MPWhiteningFilter(psd=psd, fs=fs, n_fft=n_fft)
    H = filt.frequency_response()  # complex array, len = n_fft//2+1
    if H.shape != psd.shape:
        raise RuntimeError(f"Kernel shape mismatch: {H.shape} vs {psd.shape}")
    return H

# ---------------- Correct analytic formulas (centroid with W=|W1 h|^2) ----------------
def analytic_dt1_dphi1(Phi_a, fgrid, W, bandmask):
    # δφ¹
    num_phi = np.trapz( (W[bandmask] * Phi_a[bandmask]), fgrid[bandmask] )
    den_phi = np.trapz(  W[bandmask],                    fgrid[bandmask] )
    dphi1   = num_phi/den_phi if den_phi != 0 else 0.0
    # δt¹  (NOTE: f to the FIRST POWER in numerator, f^2 in denominator)
    num_t   = np.trapz( (fgrid[bandmask] * W[bandmask] * Phi_a[bandmask]), fgrid[bandmask] )
    den_t   = np.trapz( ((fgrid[bandmask]**2) * W[bandmask]),             fgrid[bandmask] )
    dt1     = (num_t / (2.0*np.pi*den_t)) if den_t != 0 else 0.0
    return float(dt1), float(dphi1)

# ---------------- Sweep ε, compare analytic vs zlw ----------------
eps_list = np.linspace(0.0, 0.06, 13)
rows = []

# W1 depends only on S1 (fixed across ε)
n_fft = 2*(S1.size - 1)
W1 = build_mp_kernel(S1, fs, n_fft)

for eps in eps_list:
    a, Phi_a = analytic_pair(f, eps, fc)
    S2 = S1 * np.exp(2*a)

    # Weight for the analytic formula: W = |W1 h|^2  (matches denominators)
    W_weight = np.abs(W1 * h)**2

    # Analytic first-order (correct)
    dt1_an, dphi1_an = analytic_dt1_dphi1(Phi_a, f, W_weight, band)

    # Library (full)
    corr = MPMPCorrection(freqs=f, psd1=S1, psd2=S2, htilde=h, fs=fs)
    dt1_lib  = corr.dt1_full()
    dphi1_lib= corr.dphi1_full()

    rows.append({
        "epsilon": eps,
        "dt1_analytic":  dt1_an,
        "dphi1_analytic":dphi1_an,
        "dt1_zlw":       dt1_lib,
        "dphi1_zlw":     dphi1_lib,
        "dt1_residual":  dt1_lib - dt1_an,
        "dphi1_residual":dphi1_lib - dphi1_an,
    })

df = pd.DataFrame(rows)
csv_path = os.path.join(OUTDIR, "results.csv")
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

# ---------------- Plots ----------------
plt.figure(figsize=(7.2,4.6))
plt.plot(df["epsilon"], 1e6*df["dt1_analytic"], "o-", label="Analytic δt¹ (μs)")
plt.plot(df["epsilon"], 1e6*df["dt1_zlw"],      "s--", label="zlw δt¹ (μs)")
plt.xlabel("ε"); plt.ylabel("Time shift (μs)")
plt.title("δt¹: Analytic vs zlw")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "compare_dt.png"), dpi=150)

plt.figure(figsize=(7.2,4.6))
plt.plot(df["epsilon"], df["dphi1_analytic"], "o-", label="Analytic δφ¹ (rad)")
plt.plot(df["epsilon"], df["dphi1_zlw"],      "s--", label="zlw δφ¹ (rad)")
plt.xlabel("ε"); plt.ylabel("Phase shift (rad)")
plt.title("δφ¹: Analytic vs zlw")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "compare_dphi.png"), dpi=150)

plt.figure(figsize=(7.2,4.6))
plt.plot(df["epsilon"], 1e9*df["dt1_residual"],  "o-")
plt.axhline(0.0, color="k", lw=0.8, ls=":")
plt.xlabel("ε"); plt.ylabel("Residual δt¹ (ns)  [zlw - analytic]")
plt.title("Residual δt¹ (→ 0 and ~O(ε²) as ε → 0)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "residuals_dt.png"), dpi=150)

plt.figure(figsize=(7.2,4.6))
plt.plot(df["epsilon"], df["dphi1_residual"], "o-")
plt.axhline(0.0, color="k", lw=0.8, ls=":")
plt.xlabel("ε"); plt.ylabel("Residual δφ¹ (rad)  [zlw - analytic]")
plt.title("Residual δφ¹ (→ 0 and ~O(ε²) as ε → 0)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "residuals_dphi.png"), dpi=150)

# Convergence (log–log) for small ε
eps_nz = df["epsilon"].values[1:]
dt_abs = np.abs(df["dt1_residual"].values[1:])
dp_abs = np.abs(df["dphi1_residual"].values[1:])
plt.figure(figsize=(7.2,4.6))
plt.loglog(eps_nz, np.maximum(dt_abs, 1e-18), "o-", label="|δt¹ residual|")
plt.loglog(eps_nz, np.maximum(dp_abs, 1e-18), "s-", label="|δφ¹ residual|")
ref = (eps_nz**2) * max(dt_abs[0], dp_abs[0], 1e-18) / (eps_nz[0]**2)
plt.loglog(eps_nz, ref, "--", label="ε² reference")
plt.xlabel("ε"); plt.ylabel("Absolute residual (SI)")
plt.title("Convergence vs ε")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "convergence_loglog.png"), dpi=150)

print(f"Saved figures to: {OUTDIR}")
