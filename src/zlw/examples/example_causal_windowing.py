"""ZLW Example: Windowing Diagnostics & Spectral Verification.

This script demonstrates the correct usage of Zero-Latency Whitening (ZLW)
using Minimum-Phase filters with matched-latency windowing.

It avoids artificial hacks (like forcing DC to infinity) and relies on the
physical properties of the PSD and the robustness of the homomorphic method.

Diagnostic Plots:
  1. Impulse Responses: Verifies MP is causal (peak at t=0) and LP is centered.
  2. Spectra: Verifies the whitening flattens the noise floor.
  3. Time/CDF/ACF: Standard checks for stationarity and whiteness.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, butter, sosfilt, welch, correlate

from gwpy.timeseries import TimeSeries
from zlw.kernels import MPWhiteningFilter, LPWhiteningFilter
from zlw.window import Tukey


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

        # Gap handling
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
    """Applies standard notch filters for power mains."""
    print("[Prep] Notching power mains...")
    fs = data.sample_rate.value
    for freq in [60, 120, 180]:
        sos = butter(4, [freq - 1.0, freq + 1.0], btype="bandstop", fs=fs, output="sos")
        data.value[:] = sosfilt(sos, data.value)
    return data


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
    plt.xlim(-0.002, 0.05)  # Zoom on the causal start

    # 2. LP Kernel (Should peak at delay)
    plt.subplot(2, 1, 2)
    plt.plot(t, lp_kernel, label="LP Kernel (Symmetric Tukey)", color="#ff7f0e", lw=1)
    plt.title("Linear Phase Kernel (Full)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("debug_kernels.png", dpi=100)
    print("[Debug] Saved debug_kernels.png")


def whiten_with_windows(
    strain: np.ndarray, psd: np.ndarray, fs: float, kernel_duration: float
) -> dict:
    """Generates filters and whitens data."""
    # FFT length must cover the kernel duration
    n_fft = (len(psd) - 1) * 2
    print(f"[ZLW]  Building filters (n_fft={n_fft})...")

    # 1. Instantiate Filters
    #    MP is causal (peak at 0).
    mwf = MPWhiteningFilter(psd=psd, fs=fs, n_fft=n_fft)

    #    LP is centered (peak at delay).
    center_delay = kernel_duration / 2.0
    lwf = LPWhiteningFilter(psd=psd, fs=fs, n_fft=n_fft, delay=center_delay)

    # 2. Compute Windowed Impulse Responses
    #    We use a Tukey window. The ZLW library handles the geometry:
    #    - MP: Gets the right-hand side of the window (starts at 1.0).
    #    - LP: Gets the full symmetric window centered on the delay.
    win_spec = Tukey(alpha=0.25)

    mp_kernel = mwf.impulse_response(window=win_spec)
    lp_kernel = lwf.impulse_response(window=win_spec)

    # 3. Convolve
    print("[ZLW]  Convolving...")
    mp_raw = fftconvolve(strain, mp_kernel, mode="same")
    lp_raw = fftconvolve(strain, lp_kernel, mode="same")

    # 4. Normalize
    #    Trim startup transients to get valid statistics
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

    # --- Metrics Calculation ---
    # CDF
    nperseg = int(4 * fs)
    f_mp, p_mp = welch(mp_data, fs=fs, nperseg=nperseg)
    f_lp, p_lp = welch(lp_data, fs=fs, nperseg=nperseg)

    cdf_mp = np.cumsum(p_mp)
    cdf_mp /= cdf_mp[-1]
    cdf_lp = np.cumsum(p_lp)
    cdf_lp /= cdf_lp[-1]

    # ACF (Correlation)
    n_corr = int(0.5 * fs)
    slice_mp = mp_data[: n_corr * 10]
    slice_lp = lp_data[: n_corr * 10]

    acf_mp = correlate(slice_mp, slice_mp, mode="full")
    acf_lp = correlate(slice_lp, slice_lp, mode="full")
    lags = np.arange(-len(slice_mp) + 1, len(slice_mp))

    # Normalize peak
    center = len(acf_mp) // 2
    acf_mp /= acf_mp[center]
    acf_lp /= acf_lp[center]

    # Zoom
    zoom = 100
    zoom_slice = slice(center - zoom, center + zoom)

    # --- Plotting ---
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(
        f"H1 O3 Windowing Diagnostics (GPS {gps_time})", fontsize=16, fontweight="bold"
    )
    gs = fig.add_gridspec(2, 2)

    # 1. Strain
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

    # 2. CDF
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

    # 3. ACF
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

    # 1. Fetch & Prep
    data = fetch_h1_data(EVENT_GPS, DURATION)
    data = preprocess_data(data)
    fs = data.sample_rate.value

    # 2. PSD Estimation
    #    We use a windowed median estimate to be robust against transients.
    #    We clamp to a physical floor to prevent division-by-zero, but we do
    #    NOT artificially modify the shape (no infinite DC).
    psd_vals = data.psd(fftlength=8.0, method="median", window="hann").value
    psd_vals = np.maximum(psd_vals, 1e-48)

    # 3. Whiten
    results = whiten_with_windows(data.value, psd_vals, fs, kernel_duration=8.0)

    # 4. Plots
    plot_debug_kernels(results["mp_kernel"], results["lp_kernel"], fs)
    plot_diagnostics(results, EVENT_GPS)


if __name__ == "__main__":
    main()
