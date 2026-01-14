"""ZLW Example: Windowing Diagnostics & Spectral Verification.

This script demonstrates robust Zero-Latency Whitening (ZLW) using
Minimum-Phase filters.

It addresses the "Seismic Wall Artifact": Standard Welch PSD estimators
artificially suppress low-frequency noise due to windowing. This causes
naive whitening filters to develop unstable high gain at DC.

We fix this rigorously by 'extending the wall': we enforce that the PSD
at DC is at least as loud as the seismic peak. This restores the
mathematical admissibility of the PSD (bounded inverse) without using
arbitrary high-pass filters.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, butter, sosfilt, welch, correlate

from gwpy.timeseries import TimeSeries
from zlw.kernels import MPWhiteningFilter, LPWhiteningFilter
from zlw.window import Tukey, Hann


def fetch_h1_data(target_gps: float, duration: float = 64.0) -> TimeSeries:
    """Fetches open strain data for H1 with robust gap handling."""
    print(f"[Data] Fetching {duration}s of H1 data around GPS {target_gps}...")
    try:
        data = TimeSeries.fetch_open_data(
            "H1",
            int(target_gps - duration / 2),
            int(target_gps + duration / 2),
            verbose=False,
            cache=True,
        )
        nan_mask = np.isnan(data.value)
        if np.mean(nan_mask) > 0.05:
            raise ValueError("Data quality poor: >5% missing.")
        if np.any(nan_mask):
            data.value[nan_mask] = 0.0
        return data
    except Exception as e:
        print(f"[Error] Data fetch failed: {e}")
        raise


def preprocess_data(data: TimeSeries) -> TimeSeries:
    """Applies standard notch filters. No High-Pass."""
    print("[Prep] Notching power mains...")
    fs = data.sample_rate.value
    for freq in [60, 120, 180]:
        sos = butter(4, [freq - 1.0, freq + 1.0], btype="bandstop", fs=fs, output="sos")
        data.value[:] = sosfilt(sos, data.value)
    return data


def condition_seismic_wall(psd: np.ndarray, fs: float) -> np.ndarray:
    """Restores PSD admissibility by enforcing the Seismic Wall structure.

    Welch's method suppresses DC, creating a false "quiet" zone at f<10Hz.
    Inverting this creates a singularity (infinite gain).
    We fix this by extending the seismic peak value down to DC.
    """
    n = len(psd)
    freqs = np.fft.rfftfreq(2 * (n - 1), d=1.0 / fs)

    # 1. Find the Seismic Peak (typically 10-30 Hz)
    #    We search in the 5-50Hz band.
    mask_seismic = (freqs >= 5.0) & (freqs <= 50.0)
    if not np.any(mask_seismic):
        return psd  # Fallback if freq grid is weird

    idx_start = np.argmax(mask_seismic)  # First True
    # Find max within the band
    peak_idx_rel = np.argmax(psd[mask_seismic])
    peak_idx = idx_start + peak_idx_rel
    peak_val = psd[peak_idx]

    # 2. Extend the Wall
    #    For all f < f_peak, we enforce PSD >= peak_val.
    #    This ensures the whitener gain (1/sqrt(PSD)) is small and stable.
    psd[:peak_idx] = np.maximum(psd[:peak_idx], peak_val)

    return psd


def plot_debug_kernels(mp_kernel: np.ndarray, lp_kernel: np.ndarray, fs: float):
    """Visualizes the filter kernels."""
    plt.figure(figsize=(10, 6))
    t = np.arange(len(mp_kernel)) / fs

    # 1. MP Kernel (Should peak at t=0)
    plt.subplot(2, 1, 1)
    plt.plot(t, mp_kernel, label="MP Kernel (Causal Tukey)", color="#1f77b4", lw=1)
    plt.title("Minimum Phase Kernel (Zoomed Start)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-0.002, 0.05)

    # 2. LP Kernel
    plt.subplot(2, 1, 2)
    plt.plot(t, lp_kernel, label="LP Kernel (Symmetric Tukey)", color="#ff7f0e", lw=1)
    plt.title("Linear Phase Kernel (Full)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("debug_kernels.png", dpi=100)
    print("[Debug] Saved debug_kernels.png")


def plot_debug_spectra(strain: np.ndarray, mp_white: np.ndarray, fs: float):
    """Debug Plot 2: Verify spectral whitening."""
    f, p_raw = welch(strain, fs=fs, nperseg=4 * int(fs))
    f, p_white = welch(mp_white, fs=fs, nperseg=4 * int(fs))

    plt.figure(figsize=(10, 5))
    plt.loglog(f, p_raw, label="Raw Strain", alpha=0.5)

    idx_100 = np.argmin(np.abs(f - 100))
    scale = p_raw[idx_100] / p_white[idx_100]

    plt.loglog(f, p_white * scale, label="Whitened MP (Scaled)", alpha=0.8)

    plt.title("Spectral Whitening Verification")
    plt.xlabel("Frequency [Hz]")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.xlim(10, 2000)
    plt.savefig("debug_spectra.png", dpi=100)
    print("[Debug] Saved debug_spectra.png")


def whiten_with_windows(
    strain: np.ndarray, psd: np.ndarray, fs: float, kernel_duration: float
) -> dict:
    """Generates filters and whitens data."""
    n_fft = (len(psd) - 1) * 2

    # --- CRITICAL: Enforce Admissibility ---
    # Fix the estimator's low-freq artifact.
    safe_psd = condition_seismic_wall(psd.copy(), fs)

    print(f"[ZLW]  Building filters (n_fft={n_fft})...")
    mwf = MPWhiteningFilter(psd=safe_psd, fs=fs, n_fft=n_fft)

    center_delay = kernel_duration / 2.0
    lwf = LPWhiteningFilter(psd=safe_psd, fs=fs, n_fft=n_fft, delay=center_delay)

    # Use Tukey(alpha=0.1) for a robust taper.
    # MP gets Half-Window automatically.
    win_spec = Tukey(alpha=0.1)

    mp_kernel = mwf.impulse_response(window=win_spec)
    lp_kernel = lwf.impulse_response(window=win_spec)

    print("[ZLW]  Convolving...")
    mp_raw = fftconvolve(strain, mp_kernel, mode="same")
    lp_raw = fftconvolve(strain, lp_kernel, mode="same")

    # Normalize
    settle = int(kernel_duration * fs)
    valid_slice = slice(settle, -settle)

    mp_white = mp_raw / np.std(mp_raw[valid_slice])
    lp_white = lp_raw / np.std(lp_raw[valid_slice])

    return {
        "MP": mp_white,
        "LP": lp_white,
        "slice": valid_slice,
        "fs": fs,
        "mp_kernel": mp_kernel,
        "lp_kernel": lp_kernel,
    }


def plot_diagnostics(results: dict, gps_time: float):
    """Generates the standard diagnostic plot."""
    mp_data = results["MP"][results["slice"]]
    lp_data = results["LP"][results["slice"]]
    fs = results["fs"]

    # Metrics
    nperseg = int(4 * fs)
    f_mp, p_mp = welch(mp_data, fs=fs, nperseg=nperseg)
    f_lp, p_lp = welch(lp_data, fs=fs, nperseg=nperseg)

    cdf_mp = np.cumsum(p_mp)
    cdf_mp /= cdf_mp[-1]
    cdf_lp = np.cumsum(p_lp)
    cdf_lp /= cdf_lp[-1]

    n_corr = int(0.5 * fs)
    slice_mp = mp_data[: n_corr * 10]
    slice_lp = lp_data[: n_corr * 10]

    acf_mp = correlate(slice_mp, slice_mp, mode="full")
    acf_lp = correlate(slice_lp, slice_lp, mode="full")
    lags = np.arange(-len(slice_mp) + 1, len(slice_mp))

    center = len(acf_mp) // 2
    acf_mp /= acf_mp[center]
    acf_lp /= acf_lp[center]

    zoom = 100
    zoom_slice = slice(center - zoom, center + zoom)

    # Plot
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(
        f"H1 O3 Windowing Diagnostics (GPS {gps_time})", fontsize=16, fontweight="bold"
    )
    gs = fig.add_gridspec(2, 2)

    ax_time = fig.add_subplot(gs[0, :])
    t = np.arange(len(mp_data)) / fs
    t -= t[-1] / 2
    ax_time.plot(
        t, mp_data, label="MP (Causal Tukey)", color="#1f77b4", lw=0.1, alpha=0.8
    )
    ax_time.plot(
        t, lp_data, label="LP (Symmetric Tukey)", color="#ff7f0e", lw=0.1, alpha=0.6
    )
    ax_time.set_ylabel("Strain [sigma]")
    ax_time.set_xlabel("Time [s]")
    ax_time.set_xlim(-5, 5)
    ax_time.set_ylim(-6, 6)
    ax_time.legend(loc="upper right")
    ax_time.grid(alpha=0.3)

    ax_cdf = fig.add_subplot(gs[1, 0])
    ax_cdf.plot(f_mp, cdf_mp, label="MP CDF", color="#1f77b4")
    ax_cdf.plot(f_lp, cdf_lp, label="LP CDF", color="#ff7f0e")
    ax_cdf.plot([0, fs / 2], [0, 1], "k--", label="Theoretical", lw=1.5)
    ax_cdf.set_xlabel("Frequency [Hz]")
    ax_cdf.set_ylabel("Cumulative Power")
    ax_cdf.legend()
    ax_cdf.grid(alpha=0.3)
    ax_cdf.set_xlim(0, 2048)
    ax_cdf.set_ylim(0, 1.05)

    ax_acf = fig.add_subplot(gs[1, 1])
    ax_acf.plot(lags[zoom_slice], acf_mp[zoom_slice], label="MP ACF", color="#1f77b4")
    ax_acf.plot(
        lags[zoom_slice], acf_lp[zoom_slice], label="LP ACF", color="#ff7f0e", alpha=0.7
    )
    ax_acf.set_xlabel("Lag [samples]")
    ax_acf.set_ylabel("Correlation")
    ax_acf.legend()
    ax_acf.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("window_diagnostics.png", dpi=150)
    print("[Done] Plot saved to window_diagnostics.png")


def main():
    EVENT_GPS = 1239082262
    DURATION = 128.0

    data = fetch_h1_data(EVENT_GPS, DURATION)
    data = preprocess_data(data)
    fs = data.sample_rate.value

    # PSD Estimate
    psd_vals = data.psd(fftlength=8.0, method="median", window="hann").value
    psd_vals = np.maximum(psd_vals, 1e-48)

    # Whiten
    results = whiten_with_windows(data.value, psd_vals, fs, kernel_duration=8.0)

    plot_debug_kernels(results["mp_kernel"], results["lp_kernel"], fs)
    plot_debug_spectra(data.value, results["MP"], results["fs"])
    plot_diagnostics(results, EVENT_GPS)


if __name__ == "__main__":
    main()
