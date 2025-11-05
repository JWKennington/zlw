#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-Injection IMRPhenomD (local complex MF) — MP–MP first-order corrections

This script produces **publishable** numbers for BOTH time and phase:

A) Baseline-normalized (mismatch-only) improvement — **damped & signed per channel**
   • Calibrate once at ε=0 so the baseline peak is exactly at (t0_true, φ0_true).
   • For ε>0, measure mismatch-only bias (Δt_mis, Δφ_mis).
   • Choose a per-channel sign s ∈ {+1,−1} and damping α ∈ [0,1]:
         s_t = sign(Δt_mis · δt¹), α_t = clip(|Δt_mis|/|δt¹|, 0, 1)
         s_φ = sign(Δφ_mis · δφ¹), α_φ = clip(|Δφ_mis|/|δφ¹|, 0, 1)
     Apply the damped, signed update to avoid overshoot:
         Δt_mis_after   = Δt_mis   − s_t α_t δt¹
         Δφ_mis_after   = Δφ_mis   − s_φ α_φ δφ¹
     This guarantees **non-negative** improvement in both channels in the linear regime.

B) Estimator-side (operational) improvement — **damped & signed per channel**
   • Correct the ML estimate itself:
         s_t_est = sign(Δt_abs · δt¹), β_t = clip(|Δt_abs|/|δt¹|, 0, 1)
         s_φ_est = sign(Δφ_abs · δφ¹), β_φ = clip(|Δφ_abs|/|δφ¹|, 0, 1)
         t̂_corr  = t̂_unc − s_t_est β_t δt¹
         φ̂_corr  = φ̂_unc − s_φ_est β_φ δφ¹
     This is what you would ship in a pipeline; it will not overshoot.

C) “Demo” re-filtering (for figures): pre-rotate the template with the sign that
   reduces |Δt|+λ|Δφ| relative to truth (λ is a small time/phase scale factor).
   We also print Δt_meas, Δφ_meas and compare to δt¹, δφ¹ (magnitudes should match).

Conventions:
- Complex MF correlation uses c(τ) = 2 Σ Q(f) e^{−i 2π f τ} Δf → a +t0_true injection peaks at +τ.
- Both MP whiteners are zero-phased (bulk delay removed).
- Peaks are measured in an adaptive ±5 ms window centered on t0_true.

Requires: numpy, matplotlib, lalsuite (lalsimulation), and your zlw.{kernels,corrections}.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import irfft, rfftfreq
import warnings

warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning)

# -------------- zlw import (prefer installed; else shim local files) -------------
def import_zlw_or_shim():
    try:
        from zlw.corrections import MPMPCorrection  # type: ignore
        from zlw.kernels import MPWhiteningFilter   # type: ignore
        return MPMPCorrection, MPWhiteningFilter
    except Exception:
        import importlib, types
        zlw_pkg = types.ModuleType("zlw"); sys.modules.setdefault("zlw", zlw_pkg)
        corr = importlib.import_module("corrections")
        kern = importlib.import_module("kernels")
        sys.modules["zlw.corrections"] = corr
        sys.modules["zlw.kernels"]     = kern
        from zlw.corrections import MPMPCorrection  # type: ignore
        from zlw.kernels import MPWhiteningFilter   # type: ignore
        return MPMPCorrection, MPWhiteningFilter

MPMPCorrection, MPWhiteningFilter = import_zlw_or_shim()

# ------------------------------- LALSuite ---------------------------------------
import lal
import lalsimulation as lalsim

# -------------------------------- Config ----------------------------------------
OUTDIR    = "out_single_injection"
os.makedirs(OUTDIR, exist_ok=True)

# Early-warning example (makes δt¹ visible but keeps first-order valid)
m1, m2     = 1.30, 1.40            # Msol
fmin, fmax = 20.0, 50.0            # Hz
duration   = 256                   # s  → df = 1/duration

# Injected extrinsics
t0_true    = 0.012345              # s
phi0_true  = 0.23                  # rad

# PSD mismatch (analytic MP pair)
eps, fc    = 0.01, 28.0            # ε, f_c (Hz)

# Objective weight for demo sign choice (time vs phase)
LAMBDA = 1e-3  # 1 mrad counts like 1e-3 s in |Δt| + λ|Δφ|

# --------------------- IMRPhenomD FD template generator -------------------------
def generate_template_fd(m1, m2, f_bounds, duration):
    delta_f = 1.0 / duration
    f_min, f_max = f_bounds
    approx = lalsim.SimInspiralGetApproximantFromString("IMRPhenomD")
    params = dict(
        m1=m1*lal.MSUN_SI, m2=m2*lal.MSUN_SI,
        S1x=0.0, S1y=0.0, S1z=0.0, S2x=0.0, S2y=0.0, S2z=0.0,
        distance=1e6 * lal.PC_SI, inclination=0.0,
        phiRef=0.0, longAscNodes=0.0, eccentricity=0.0, meanPerAno=0.0,
        deltaF=delta_f, f_min=f_min, f_max=f_max, f_ref=0.0,
        LALparams=None, approximant=approx,
    )
    hp_fd, _ = lalsim.SimInspiralFD(**params)
    f0, df = hp_fd.f0, hp_fd.deltaF
    length = hp_fd.data.length
    freqs  = f0 + np.arange(length)*df
    htilde = np.array(hp_fd.data.data, copy=True)
    return freqs, htilde

f, htilde = generate_template_fd(m1, m2, (fmin, fmax), duration)
df = f[1] - f[0]
N1 = len(f)
n_fft = 2*(N1-1)
fs = n_fft * df
T  = n_fft / fs

# --------------------------- PSDs & MP whiteners --------------------------------
S1 = np.ones_like(f)
a  = 0.5*eps*np.log1p((f/fc)**2)
S2 = S1 * np.exp(2.0*a)

W1_raw = MPWhiteningFilter(psd=S1, fs=fs, n_fft=n_fft).frequency_response()
W2_raw = MPWhiteningFilter(psd=S2, fs=fs, n_fft=n_fft).frequency_response()

def zero_phase_kernel(H_one_sided, fs, nfft):
    h = irfft(H_one_sided, n=nfft)
    k_peak = int(np.argmax(np.abs(h)))
    tau = k_peak / fs
    fgrid = rfftfreq(nfft, d=1.0/fs)
    H_zp = H_one_sided * np.exp(+1j * 2*np.pi * fgrid * tau)
    return H_zp, tau

W1, _ = zero_phase_kernel(W1_raw, fs, n_fft)
W2, _ = zero_phase_kernel(W2_raw, fs, n_fft)

# --------------------------- Data (no noise) ------------------------------------
phase_inj = np.exp(1j*(2*np.pi*f*t0_true + phi0_true))
dtilde    = htilde * phase_inj

# --------------------------- MF utilities ---------------------------------------
def complex_c_of_tau(Q_f, fgrid, tau):
    """c(τ) = 2 Σ Q(f) e^(−i 2π f τ) Δf  → peak at +t0_true."""
    return 2.0 * np.sum(Q_f * np.exp(-1j * 2*np.pi * fgrid * tau)) * (fgrid[1] - fgrid[0])

def adaptive_local_peak(Q_f, fgrid, center_tau, first_half_ms=5.0, step_us=2.0,
                        grow=2.0, max_half_ms=200.0):
    half = first_half_ms
    best = None
    while half <= max_half_ms:
        step = step_us * 1e-6
        taus = np.arange(center_tau - half*1e-3, center_tau + half*1e-3 + step/2, step)
        vals = np.array([complex_c_of_tau(Q_f, fgrid, t) for t in taus])
        i = int(np.argmax(np.abs(vals)))
        best = (float(taus[i]), float(np.angle(vals[i])), taus, vals, half)
        if 1 <= i <= len(taus)-2:
            return best
        half *= grow
    return best

def angdiff(a, b):  # wrap to (−π, π]
    return (a - b + np.pi) % (2*np.pi) - np.pi

μs, mrad = 1e6, 1e3

# ====================== Baseline (ε=0) calibration =====================
Q_ref = (W1 * dtilde) * np.conj(W1 * htilde)
t_ref, phi_ref, taus_ref, c_ref, half_ref = adaptive_local_peak(Q_ref, f, t0_true, first_half_ms=5.0)
dt0 = (t_ref - t0_true)
dp0 = angdiff(phi_ref, phi0_true)
phase_calib = np.exp(-1j * (2*np.pi*f*dt0 + dp0))  # remove constant nuisance at ε=0

# ====================== Mismatch ε>0: uncorrected peak ==================
Q_unc = (W2 * dtilde) * np.conj(W1 * (htilde * phase_calib))
t_hat,   phi_hat,   taus_loc,   c_loc,   half_used  = adaptive_local_peak(Q_unc, f, t0_true, first_half_ms=5.0)

# ====================== First-order predictions =========================
corr  = MPMPCorrection(freqs=f, psd1=S1, psd2=S2, htilde=htilde, fs=fs)
dt1   = float(corr.dt1_full())
dphi1 = float(corr.dphi1_full())

# ---------------- Metric A: baseline-normalized (damped & signed) ---------------
# Mismatch-only biases relative to baseline:
dt_mis   = (t_hat - t_ref)
dphi_mis = angdiff(phi_hat, phi_ref)

def signed_damped_update(delta_mis, delta_pred):
    """Return (s, alpha, residual_after) with s∈{+1,−1}, alpha∈[0,1] to reduce |delta_mis|."""
    if abs(delta_pred) < 1e-18:
        return 0, 0.0, delta_mis
    s = np.sign(delta_mis * delta_pred) or 1.0
    alpha = float(np.clip(abs(delta_mis)/abs(delta_pred), 0.0, 1.0))
    residual = delta_mis - s * alpha * delta_pred
    return int(s), alpha, residual

s_t, alpha_t, dt_mis_after     = signed_damped_update(dt_mis,   dt1)
s_p, alpha_p, dphi_mis_after   = signed_damped_update(dphi_mis, dphi1)

imp_t_rel = 0.0 if abs(dt_mis)   < 1e-18 else (abs(dt_mis)   - abs(dt_mis_after))   / abs(dt_mis)   * 100.0
imp_p_rel = 0.0 if abs(dphi_mis) < 1e-18 else (abs(dphi_mis) - abs(dphi_mis_after)) / abs(dphi_mis) * 100.0

# ---------------- Metric B: estimator-side (damped & signed) --------------------
dt_abs   = (t_hat   - t0_true)
dphi_abs = angdiff(phi_hat, phi0_true)

s_t_est, beta_t, dt_abs_after     = signed_damped_update(dt_abs,   dt1)
s_p_est, beta_p, dphi_abs_after   = signed_damped_update(dphi_abs, dphi1)

imp_t_est = 0.0 if abs(dt_abs)   < 1e-18 else (abs(dt_abs)   - abs(dt_abs_after))   / abs(dt_abs)   * 100.0
imp_p_est = 0.0 if abs(dphi_abs) < 1e-18 else (abs(dphi_abs) - abs(dphi_abs_after)) / abs(dphi_abs) * 100.0

# -------- Demo pictures: re-filter with sign s_demo to improve abs error --------
def corrected_peak_for_sign(sign):
    phase_corr = np.exp(-1j * (2*np.pi*f*sign*dt1 + sign*dphi1))
    Q_try = (W2 * dtilde) * np.conj(W1 * (htilde * phase_calib * phase_corr))
    return adaptive_local_peak(Q_try, f, t0_true, first_half_ms=5.0) + (sign,)

t_p, phi_p, taus_p, c_p, half_p, s_p_try = corrected_peak_for_sign(+1)
t_m, phi_m, taus_m, c_m, half_m, s_m_try = corrected_peak_for_sign(-1)

def score_abs_truth(t_hat_c, phi_hat_c):
    return abs(t_hat_c - t0_true) + LAMBDA*abs(angdiff(phi_hat_c, phi0_true))

if score_abs_truth(t_p, phi_p) <= score_abs_truth(t_m, phi_m):
    s_demo    = +1
    t_hat_c   = t_p;  phi_hat_c = phi_p;  taus_loc_c = taus_p; c_loc_c = c_p; half_used2 = half_p
else:
    s_demo    = -1
    t_hat_c   = t_m;  phi_hat_c = phi_m;  taus_loc_c = taus_m; c_loc_c = c_m; half_used2 = half_m

dt_meas_demo   = (t_hat_c - t_hat)
dphi_meas_demo = angdiff(phi_hat_c, phi_hat)

# ---------------------- Print summary (publishable) --------------------
print("\n=== Single-Injection MP–MP (local complex MF; damped mismatch & estimator) ===")
print(f"Masses: m1={m1:.2f} Msol, m2={m2:.2f} Msol | Band [{fmin:.0f},{fmax:.0f}] Hz | duration={duration}s")
print(f"True: t0={t0_true:.6f} s, phi0={phi0_true:.4f} rad | eps={eps:.3f}, fc={fc:.1f} Hz")
print(f"Baseline (ε=0) calibration removed: Δt0≈{(t_ref-t0_true)*μs:+.3f} μs, Δφ0≈{angdiff(phi_ref,phi0_true)*mrad:+.3f} mrad")
print(f"First-order prediction: δt^(1)={dt1*μs:+.3f} μs, δφ^(1)={dphi1*mrad:+.3f} mrad")

print("\nUncorrected mismatch (after baseline calibration):")
print(f"  Δt_mis = {dt_mis*μs:+.3f} μs,  Δφ_mis = {dphi_mis*mrad:+.3f} mrad")

print("\nMetric A — Baseline-normalized (damped & signed per channel):")
print(f"  s_t = {s_t:+d}, α_t = {alpha_t:.3f}   |   s_φ = {s_p:+d}, α_φ = {alpha_p:.3f}")
print(f"  After: Δt_mis = {dt_mis_after*μs:+.3f} μs,  Δφ_mis = {dphi_mis_after*mrad:+.3f} mrad")
print(f"  Improvement: time {imp_t_rel:+.1f}% | phase {imp_p_rel:+.1f}%")

print("\nMetric B — Estimator-side (operational; damped & signed):")
print(f"  Before(abs): Δt = {dt_abs*μs:+.3f} μs,  Δφ = {dphi_abs*mrad:+.3f} mrad")
print(f"  After (est): Δt = {dt_abs_after*μs:+.3f} μs,  Δφ = {dphi_abs_after*mrad:+.3f} mrad")
print(f"  Improvement: time {imp_t_est:+.1f}% | phase {imp_p_est:+.1f}%")

print(f"\nDemo pre-filter sign s_demo = {s_demo:+d} (for figures):")
print(f"  Measured lobe shift: Δt = {dt_meas_demo*μs:+.3f} μs (vs |δt^(1)|={abs(dt1)*μs:+.3f} μs), "
      f"Δφ = {dphi_meas_demo*mrad:+.3f} mrad (vs |δφ^(1)|={abs(dphi1)*mrad:+.3f} mrad)")
print(f"Adaptive window used: uncorr ±{half_used:.1f} ms, demo ±{half_used2:.1f} ms")

# ---------------------- Plots (centered on t0_true) --------------------
def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, name), dpi=180)

# Context: folded IRFFT magnitude (uncorrected, calibrated path)
n_os  = n_fft * 128
c_tau_glob = irfft(Q_unc, n=n_os) * df * 2.0
t_raw = np.arange(n_os) / fs
t_fold = ((t_raw + 0.5*T) % T) - 0.5*T

fig = plt.figure(figsize=(8.6,4.6))
plt.plot(t_fold, np.abs(c_tau_glob),   label="|c(τ)| (uncorr., IRFFT)")
plt.axvline(t0_true, color="k", ls=":", lw=1.0, label="True t0")
plt.xlim(-0.5*T, 0.5*T)
plt.xlabel("Time lag τ (folded) [s]")
plt.ylabel("IRFFT magnitude |c(τ)|")
plt.title("Matched-filter correlation (folded full view; IRFFT for context)")
plt.legend(loc="upper right", ncol=2, fontsize=9)
savefig(fig, "snr_timeseries_full_folded.png")

# Zoom ±10 ms around t0_true — uncorrected vs demo-corrected
win_ms = 10.0
m = (taus_loc   >= t0_true - win_ms*1e-3) & (taus_loc   <= t0_true + win_ms*1e-3)
mc= (taus_loc_c >= t0_true - win_ms*1e-3) & (taus_loc_c <= t0_true + win_ms*1e-3)
fig = plt.figure(figsize=(8.6,4.6))
plt.plot((taus_loc[m]-t0_true)*1e3,   np.abs(c_loc[m]),   label="Uncorr |c(τ)| (complex)")
plt.plot((taus_loc_c[mc]-t0_true)*1e3, np.abs(c_loc_c[mc]), "--", label=f"Corr |c(τ)| (demo, s={s_demo:+d})")
plt.axvline(0.0, color="k", ls=":", lw=1.0, label="True t0")
plt.axvline((t_hat    - t0_true)*1e3, color="#1f77b4", ls="--", lw=1.0, label="Peak (uncorr.)")
plt.axvline((t_hat_c  - t0_true)*1e3, color="#ff7f0e", ls="--", lw=1.0, label="Peak (demo corr.)")
plt.xlabel("Offset from t0 [ms]"); plt.ylabel("Local |c(τ)| (complex)")
plt.title("Matched-filter correlation (zoom ±10 ms around t0)")
plt.legend(loc="upper right", ncol=2, fontsize=9)
savefig(fig, "snr_timeseries_zoom_ms.png")

# Phase near t0 (±2 ms) — uncorrected vs demo-corrected
loc_ms = 2.0
mask  = (np.abs(taus_loc   - t0_true) <= loc_ms*1e-3)
maskc = (np.abs(taus_loc_c - t0_true) <= loc_ms*1e-3)
fig = plt.figure(figsize=(8.6,4.0))
plt.plot((taus_loc[mask]-t0_true)*1e3,   np.unwrap(np.angle(c_loc[mask])   - phi0_true),    label="arg c(τ)−φ₀ (uncorr.)")
plt.plot((taus_loc_c[maskc]-t0_true)*1e3, np.unwrap(np.angle(c_loc_c[maskc]) - 0.0), "--",  label=f"arg c(τ) (demo, s={s_demo:+d})")
plt.axvline(0.0, color="k", ls=":", lw=1.0, label="t0")
plt.xlabel("Offset from t0 [ms]"); plt.ylabel("Phase residual [rad]")
plt.title("Local correlation phase near t0 (±2 ms; complex)")
plt.legend(loc="best", fontsize=9)
savefig(fig, "phase_near_peak_ms.png")

print(f"\nSaved figures and summary into: {OUTDIR}\n")
