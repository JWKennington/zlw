
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNR contour geometry in (t, phi) for MP–MP whitening with a small PSD perturbation.
Now includes a third panel: Difference (perturbed − unperturbed).
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

def make_freq_grid(fs, nfft, fmin, fmax):
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
    band = (freqs >= max(1e-9, fmin)) & (freqs <= min(fs/2 - 1e-9, fmax))
    return freqs, band

def psd_lorentzian(freqs, fc=300.0, A=1.0):
    x = freqs / max(1e-9, fc)
    return A / (1.0 + x*x)

def psd_exponential(freqs, f0=300.0, A=1.0):
    return A * np.exp(-freqs / max(1e-9, f0))

def perturbation_shape(freqs, shape, **kw):
    if shape == "tilt":
        fmin = kw.get("fmin", freqs[1] if freqs[0]==0 else freqs[0])
        fmax = kw.get("fmax", freqs[-1])
        span = max(1e-9, fmax - fmin)
        p = (freqs - fmin)/span
        p[freqs < fmin] = 0.0
        p[freqs > fmax] = 1.0
        return p
    if shape == "bump":
        f0, s = kw.get("f0",150.0), kw.get("sigma",30.0)
        return np.exp(-0.5*((freqs - f0)/max(1e-9,s))**2)
    if shape == "dip":
        f0, s = kw.get("f0",300.0), kw.get("sigma",50.0)
        return -np.exp(-0.5*((freqs - f0)/max(1e-9,s))**2)
    # default power tilt around median
    alpha = kw.get("alpha", 0.5)
    med = np.median(freqs[freqs>0])
    return (freqs/max(1e-9,med))**alpha - 1.0

def minimum_phase_whitener_from_psd(S):
    n_rfft = S.shape[0]
    N = (n_rfft - 1) * 2
    L_half = -0.5 * np.log(np.clip(S, 1e-30, None))
    L_full = np.concatenate([L_half, L_half[-2:0:-1]])
    l_full = np.fft.ifft(L_full).real
    l_fold = np.zeros_like(l_full)
    l_fold[0] = l_full[0]
    l_fold[1:N//2] = 2.0*l_full[1:N//2]
    l_fold[N//2] = l_full[N//2]
    F_full = np.fft.fft(l_fold)
    return np.exp(F_full[:n_rfft])

def spa_newtonian_amplitude(freqs, fmin, fmax, A0=1.0):
    amp = A0 * np.where(freqs>0, freqs**(-7.0/6.0), 0.0)
    amp[(freqs<fmin) | (freqs>fmax)] = 0.0
    return amp

def first_order_bias(freqs, amp, S2, phi_mismatch):
    W = (amp**2) / np.clip(S2, 1e-30, None)
    num_t = np.trapz(freqs * W * phi_mismatch, freqs)
    den_t = np.trapz((freqs**2) * W, freqs)
    num_p = np.trapz(W * phi_mismatch, freqs)
    den_p = np.trapz(W, freqs)
    dt1 = num_t / (2.0*np.pi*den_t + 1e-30)
    dphi1 = num_p / (den_p + 1e-30)
    return float(dt1), float(dphi1)

def build_J_of_t(freqs, K, t_grid):
    phase = np.exp(1j * 2.0*np.pi * np.outer(freqs, t_grid))
    return np.trapz(K[:,None] * phase, freqs, axis=0)

def snr_surface_real(J_t, phi_grid):
    mag = np.abs(J_t)
    ang = np.angle(J_t)
    return mag[:,None] * np.cos(phi_grid[None,:] + ang[:,None])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", type=float, default=4096.0)
    ap.add_argument("--nfft", type=int, default=65536)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=512.0)
    ap.add_argument("--psd", choices=["lorentz","exp"], default="exp")
    ap.add_argument("--epsilon", type=float, default=0.9)
    ap.add_argument("--shape", choices=["tilt","bump","dip","power"], default="power")
    ap.add_argument("--twin", type=float, default=0.01)
    ap.add_argument("--tres", type=float, default=2.0e-4)
    ap.add_argument("--phi-res", type=int, default=181)
    ap.add_argument("--outfile", type=str, default="snr_contours_with_diff.png")
    args = ap.parse_args()

    freqs, band = make_freq_grid(args.fs, args.nfft, args.fmin, args.fmax)
    if not np.any(band):
        raise SystemExit("Empty frequency band; adjust fmin/fmax or fs/nfft.")

    S1 = psd_lorentzian(freqs) if args.psd == "lorentz" else psd_exponential(freqs)
    pshape = perturbation_shape(freqs, args.shape, fmin=args.fmin, fmax=args.fmax)
    S2 = np.clip(S1 * (1.0 + args.epsilon * pshape), 1e-30, None)
    S1 = np.clip(S1, 1e-30, None)

    W1 = minimum_phase_whitener_from_psd(S1)
    W2 = minimum_phase_whitener_from_psd(S2)
    phi_mis = np.unwrap(np.angle(W2) - np.angle(W1))
    amp = spa_newtonian_amplitude(freqs, args.fmin, args.fmax, A0=1.0)

    K0 = (np.abs(W1)**2) * (amp**2)
    Kp = (W2 * W1) * (amp**2)

    t_grid = np.arange(-args.twin, args.twin + 1e-12, args.tres)
    phi_grid = np.linspace(-np.pi, np.pi, args.phi_res)

    J0 = build_J_of_t(freqs, K0, t_grid)
    Jp = build_J_of_t(freqs, Kp, t_grid)

    I0 = snr_surface_real(J0, phi_grid)
    Ip = snr_surface_real(Jp, phi_grid)

    I0n = I0 / (np.max(np.abs(I0)) + 1e-30)
    Ipn = Ip / (np.max(np.abs(Ip)) + 1e-30)
    D = Ipn - I0n

    # Bias prediction (for annotation arrow)
    dt1, dphi1 = first_order_bias(freqs, amp, S2, phi_mis)
    t_idx0, phi_idx0 = np.unravel_index(np.argmax(I0n), I0n.shape)
    t0, phi0 = t_grid[t_idx0], phi_grid[phi_idx0]
    t_pred = t0 + dt1
    phi_pred = ((phi0 + dphi1 + np.pi) % (2*np.pi)) - np.pi

    # Plot 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(17,5), constrained_layout=True, sharey=True)
    levels = np.linspace(-1.0, 1.0, 21)

    im0 = axes[0].contourf(t_grid*1e3, phi_grid, I0n.T, levels=levels, cmap="RdBu_r")
    axes[0].contour(t_grid*1e3, phi_grid, I0n.T, levels=[0.5, 0.8], colors="k", linewidths=0.8)
    axes[0].plot([t0*1e3], [phi0], "ko")
    axes[0].set_title("Unperturbed (W2 = W1)")
    axes[0].set_xlabel("Time shift t [ms]")
    axes[0].set_ylabel("Coalescence phase φ [rad]")

    im1 = axes[1].contourf(t_grid*1e3, phi_grid, Ipn.T, levels=levels, cmap="RdBu_r")
    axes[1].contour(t_grid*1e3, phi_grid, Ipn.T, levels=[0.5, 0.8], colors="k", linewidths=0.8)
    axes[1].arrow(t0*1e3, phi0, (t_pred - t0)*1e3, (phi_pred - phi0), width=0.005, head_width=0.07,
                  length_includes_head=True, color="k")
    axes[1].set_title(f"Perturbed (ε={args.epsilon:.2f}, shape={args.shape})")
    axes[1].set_xlabel("Time shift t [ms]")

    im2 = axes[2].contourf(t_grid*1e3, phi_grid, D.T, levels=levels, cmap="RdBu_r")
    axes[2].contour(t_grid*1e3, phi_grid, D.T, levels=[0.0], colors="k", linewidths=1.0)
    axes[2].set_title("Difference: perturbed − unperturbed")
    axes[2].set_xlabel("Time shift t [ms]")

    fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.9, label="Re I(t, φ) (normalized / diff)")
    fig.suptitle("SNR contours in (t, φ): unperturbed, perturbed, and difference", y=1.02)

    plt.savefig(args.outfile, dpi=160)
    print(f"[OK] Wrote figure to {args.outfile}")
    print(f"[INFO] Δt₁={dt1*1e6:+.2f} μs, Δφ₁={dphi1:+.3f} rad")

if __name__ == "__main__":
    main()
