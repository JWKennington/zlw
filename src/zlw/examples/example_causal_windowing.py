"""ZLW Example: Windowing Diagnostics & Spectral Verification.

This script validates the "Matched-Latency Windowing" logic in ZLW.
It demonstrates that applying the correct time-domain windows to whitening
filters (Causal "Half-Window" for MP vs Symmetric for LP) preserves the
spectral whitening properties while improving temporal localization.

We generate three key diagnostic plots:
  1. Whitened Strain (Time Domain): Visual check for stationarity/normalization.
  2. Cumulative Power Distribution (CDF): Verifies spectral flatness.
     Ideal white noise follows a perfect diagonal (y=x). Deviations indicate
     unwhitened colored noise or spectral leakage.
  3. Autocorrelation Function (ACF): Verifies temporal independence.
     Ideal white noise is a delta function. The width of the central peak
     and the structure of the "diamond" envelope reveal the filter's
     effective bandwidth and windowing side-effects.

Usage:
    python src/zlw/examples/example_window_diagnostics.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, butter, sosfilt, welch, correlate
import textwrap

# GWPY for easy access to open data
from gwpy.timeseries import TimeSeries

# ZLW imports
from zlw.kernels import MPWhiteningFilter, LPWhiteningFilter
from zlw.window import Tukey, Hann, WindowSpec


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

        # --- ROBUSTNESS CHECK: Data Quality ---
        # TimeSeries objects wrap numpy arrays in .value. We must operate there.
        nan_mask = np.isnan(data.value)
        nan_fraction = np.mean(nan_mask)

        if nan_fraction > 0.05:
            raise ValueError(
                f"Data quality poor: {nan_fraction:.1%} of data is missing (NaNs)."
            )

        if np.any(nan_mask):
            print(
                f"       [Warning] Found small gaps ({nan_fraction:.1%}). Filling with zeros."
            )
            data.value[nan_mask] = 0.0

        return data

    except Exception as e:
        print(f"[Error] Data fetch failed: {e}")
        raise


def preprocess_data(data: TimeSeries) -> TimeSeries:
    """Applies notch filters for 60Hz harmonics."""
    print("[Prep] Notching power mains...")
    fs = data.sample_rate.value
    for freq in [60, 120, 180]:
        sos = butter(4, [freq - 1.0, freq + 1.0], btype="bandstop", fs=fs, output="sos")
        data.value[:] = sosfilt(sos, data.value)
    return data


def whiten_with_windows(
    strain: np.ndarray, psd: np.ndarray, fs: float, kernel_duration: float
) -> dict:
    """Generates filters with matched windows and whitens data."""
    n_fft = (len(psd) - 1) * 2
    print(f"[ZLW]  Building filters (Length: {n_fft} taps)...")

    # 1. Instantiate Filters
    #    MP (Causal): peak_center = 0.0
    mwf = MPWhiteningFilter(psd=psd, fs=fs, n_fft=n_fft)

    #    LP (Acausal): peak_center = delay * fs
    center_delay = kernel_duration / 2.0
    lwf = LPWhiteningFilter(psd=psd, fs=fs, n_fft=n_fft, delay=center_delay)

    # 2. Compute Windowed Impulse Responses
    #    We use a generic Tukey(alpha=0.25).
    #    The ZLW library automatically aligns this window:
    #      - MP gets a "Half-Window" (1.0 at index 0, tapering right).
    #      - LP gets a Symmetric Window centered at delay.
    win_spec = Tukey(alpha=0.25)

    mp_kernel = mwf.impulse_response(window=win_spec)
    lp_kernel = lwf.impulse_response(window=win_spec)

    # 3. Convolve
    print("[ZLW]  Convolving...")
    mp_raw = fftconvolve(strain, mp_kernel, mode="same")
    lp_raw = fftconvolve(strain, lp_kernel, mode="same")

    # 4. Normalize (Generic trimming)
    settle = int(kernel_duration * fs)
    valid_slice = slice(settle, -settle)

    # Scale to unit variance on valid region
    mp_white = mp_raw / np.std(mp_raw[valid_slice])
    lp_white = lp_raw / np.std(lp_raw[valid_slice])

    return {"MP": mp_white, "LP": lp_white, "slice": valid_slice, "fs": fs}


def compute_diagnostics(data: np.ndarray, fs: float):
    """Computes CDF and Autocorrelation for plotting."""
    # 1. PSD / CDF
    nperseg = int(4 * fs)  # 4s FFT for good resolution
    freqs, Pxx = welch(data, fs=fs, nperseg=nperseg)

    # Normalize Cumulative Power (CDF)
    # Ideally should be a straight line from (0,0) to (Nyquist, 1)
    cdf = np.cumsum(Pxx)
    cdf /= cdf[-1]

    # 2. Autocorrelation (ACF)
    # We use FFT correlation on a shorter slice for speed
    n_corr = int(0.5 * fs)  # 0.5s correlation window
    slice_data = data[: n_corr * 10]  # Use enough data for statistics

    # Standard autocorrelation via FFT
    acf = correlate(slice_data, slice_data, mode="full")
    lags = np.arange(-len(slice_data) + 1, len(slice_data))

    # Normalize peak to 1
    center_idx = len(acf) // 2
    acf /= acf[center_idx]

    # Zoom in to +/- 100 lags for the plot
    zoom = 2000
    acf_zoom = acf[center_idx - zoom : center_idx + zoom]
    lags_zoom = lags[center_idx - zoom : center_idx + zoom]

    return freqs, cdf, lags_zoom, acf_zoom


def plot_diagnostics(results: dict, gps_time: float):
    """Generates the diagnostic figure."""
    mp_data = results["MP"][results["slice"]]
    lp_data = results["LP"][results["slice"]]
    fs = results["fs"]

    # Compute metrics
    f_mp, cdf_mp, lag_mp, acf_mp = compute_diagnostics(mp_data, fs)
    f_lp, cdf_lp, lag_lp, acf_lp = compute_diagnostics(lp_data, fs)

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(
        f"H1 O3 Windowing Diagnostics (GPS {gps_time})",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3)
    ax_time = fig.add_subplot(gs[0, :])
    ax_cdf = fig.add_subplot(gs[1, 0])
    ax_acf = fig.add_subplot(gs[1, 1])

    # --- 1. Whitened Strain ---
    t_axis = np.arange(len(mp_data)) / fs
    # Center time axis roughly
    t_axis -= t_axis[-1] / 2

    ax_time.plot(
        t_axis,
        mp_data,
        label="MP (Causal Window)",
        color="#1f77b4",
        linewidth=0.1,
        alpha=0.9,
    )
    ax_time.plot(
        t_axis,
        lp_data,
        label="LP (Symmetric Window)",
        color="#ff7f0e",
        linewidth=0.1,
        alpha=0.6,
    )
    ax_time.set_title("Whitened Strain (Zoomed)")
    ax_time.set_ylabel("Strain [sigma]")
    ax_time.set_xlabel("Time [s]")
    ax_time.set_ylim(-6, 6)
    ax_time.set_xlim(-10, 10)  # Zoom in
    ax_time.legend(loc="upper right")
    ax_time.grid(alpha=0.3)

    # --- 2. Spectral Flatness (CDF) ---
    ax_cdf.plot(f_mp, cdf_mp, label="MP CDF", color="#1f77b4")
    ax_cdf.plot(f_lp, cdf_lp, label="LP CDF", color="#ff7f0e")
    # Theoretical Diagonal
    ax_cdf.plot([0, fs / 2], [0, 1], "k--", label="Theoretical White", linewidth=1.5)

    ax_cdf.set_title("Normalized Cumulative Power (CDF)")
    ax_cdf.set_xlabel("Frequency [Hz]")
    ax_cdf.set_ylabel("Cumulative Power")
    ax_cdf.legend()
    ax_cdf.grid(alpha=0.3)
    ax_cdf.set_xlim(0, 2048)
    ax_cdf.set_ylim(0, 1.05)

    # --- 3. Autocorrelation (ACF) ---
    ax_acf.plot(lag_mp, acf_mp, label="MP ACF", color="#1f77b4", alpha=0.8)
    ax_acf.plot(lag_lp, acf_lp, label="LP ACF", color="#ff7f0e", alpha=0.5)

    ax_acf.set_title("Autocorrelation Function (ACF)")
    ax_acf.set_xlabel("Lag [samples]")
    ax_acf.set_ylabel("Normalized Correlation")
    ax_acf.legend()
    ax_acf.grid(alpha=0.3)
    # Zoom to show the "diamond" structure near zero lag
    ax_acf.set_xlim(-100, 100)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig("window_diagnostics.png", dpi=150)
    print("[Done] Plot saved to window_diagnostics.png")


def main():
    EVENT_GPS = 1239082262
    DURATION = 128.0

    # 1. Fetch
    data = fetch_h1_data(EVENT_GPS, DURATION)
    data = preprocess_data(data)

    # 2. PSD
    psd_vals = data.psd(fftlength=8.0, method="median", window="hann").value
    # Clamp PSD to avoid NaNs in filter
    psd_vals = np.maximum(psd_vals, 1e-50)

    # 3. Whiten with new Windowing logic
    results = whiten_with_windows(
        data.value, psd_vals, data.sample_rate.value, kernel_duration=8.0
    )

    # 4. Plot
    plot_diagnostics(results, EVENT_GPS)


if __name__ == "__main__":
    main()
