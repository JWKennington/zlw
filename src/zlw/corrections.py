from __future__ import annotations

import warnings
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter1d

from zlw.kernels import MPWhiteningFilter
from zlw.window import WindowSpec

#: Convenience container for returning timing / phase / SNR corrections.
# Added 'dsnr1' to the tuple
TimePhaseCorrection = namedtuple("TimePhaseCorrection", "dt1 dt2 dphi1 dphi2 dsnr1")


@dataclass
class PrtPsdDriftCorrection:
    r"""First–order perturbative MP–MP timing, phase, and SNR corrections for whitening
    mismatch.

    This class implements the *geometric* small–perturbation formulas for the
    MP–MP configuration.
    """

    freqs: np.ndarray
    psd1: np.ndarray
    psd2: np.ndarray
    h_tilde: np.ndarray
    fs: float

    # --- derived / cached quantities (populated in __post_init__) ---
    df: float = None
    n_fft: int = None
    wk1: np.ndarray = None
    wk2: np.ndarray = None
    w_simple: np.ndarray = None
    phi_diff: np.ndarray = None
    eps: float = None

    def __post_init__(self) -> None:
        """Validate inputs, build MP whiteners, and precompute weights."""
        # Coerce to numpy arrays
        self.freqs = np.asarray(self.freqs, dtype=float)
        self.psd1 = np.asarray(self.psd1, dtype=float)
        self.psd2 = np.asarray(self.psd2, dtype=float)
        self.h_tilde = np.asarray(self.h_tilde, dtype=complex)

        # Basic shape checks
        n = self.freqs.size
        if not (self.psd1.size == self.psd2.size == self.h_tilde.size == n):
            raise ValueError(
                "freqs, psd1, psd2, and htilde must all have the same length."
            )
        if n < 3:
            raise ValueError("Need at least 3 frequency bins for integration.")

        # Monotonic frequency grid
        if not np.all(np.diff(self.freqs) > 0):
            raise ValueError("freqs must be strictly increasing (one-sided grid).")

        # PSD sanity
        if np.any(self.psd1 <= 0) or np.any(self.psd2 <= 0):
            raise ValueError("psd1 and psd2 must be strictly positive everywhere.")

        # Frequency bin width
        self.df = float(self.freqs[1] - self.freqs[0])

        # Infer full FFT length
        self.n_fft = int((n - 1) * 2)

        # Build minimum-phase whitening filters
        self._build_mp_filters()

        # Precompute weights and phase mismatch
        self._precompute_weight_and_phase()

        # Perturbativity diagnostic
        eps_arr = np.sqrt(self.psd1 / self.psd2) - 1.0
        self.eps = float(np.max(np.abs(eps_arr)))

    def _build_mp_filters(self) -> None:
        """Construct MP whitening filters and cache one-sided responses."""
        mp1 = MPWhiteningFilter(self.psd1, self.fs, self.n_fft)
        mp2 = MPWhiteningFilter(self.psd2, self.fs, self.n_fft)

        self.wk1 = mp1.frequency_response()  # real >= 0
        self.wk2 = mp2.frequency_response()  # complex

    def _precompute_weight_and_phase(self) -> None:
        """Precompute w(f) and the whitening phase difference Φ(f)."""
        # Effective spectral weight: |W2(f) * h(f)|^2
        # (Uses W2 to reflect the actual data whitening in the specific realization)
        self.w_simple = np.abs(self.wk2 * self.h_tilde) ** 2

        # Whitening phase mismatch: Φ(f) = arg W2 − arg W1
        self.phi_diff = np.angle(self.wk2) - np.angle(self.wk1)

    def _integrate(self, arr: np.ndarray) -> float:
        """Numerically integrate ``arr(f)`` over ``self.freqs``."""
        arr = np.asarray(arr, dtype=float)
        n = arr.size
        if n % 2 == 1:
            return float(simpson(arr, self.freqs))
        else:
            # Handle NumPy 2.0+ removal of trapz
            if hasattr(np, "trapezoid"):
                return float(np.trapezoid(arr, self.freqs))
            else:
                return float(np.trapz(arr, self.freqs))

    def dt1(self) -> float:
        r"""First-order timing correction :math:`\delta t^{(1)}` (seconds)."""
        num = self._integrate(self.freqs * self.w_simple * self.phi_diff)
        den = self._integrate(self.freqs**2 * self.w_simple)
        if den == 0.0:
            return 0.0
        return (1.0 / (2.0 * np.pi)) * num / den

    def dphi1(self) -> float:
        r"""First-order phase correction :math:`\delta\phi^{(1)}` (radians)."""
        num = self._integrate(self.w_simple * self.phi_diff)
        den = self._integrate(self.w_simple)
        if den == 0.0:
            return 0.0
        return num / den

    def dsnr1(self) -> float:
        r"""First-order fractional SNR change :math:`\delta\rho^{(1)}/\rho`.

        Calculated as the power-weighted average of the log-magnitude difference:

        .. math::
            \frac{\delta\rho}{\rho} \approx
            \frac{\int w(f) \ln(|W_2(f)|/|W_1(f)|)\,df}
                 {\int w(f)\,df}

        where :math:`|W_2|/|W_1| = \sqrt{S_1/S_2}`. A negative value implies
        sensitivity loss due to the PSD mismatch.
        """
        # ln(|W2|) - ln(|W1|)
        log_mag_diff = np.log(np.abs(self.wk2)) - np.log(np.abs(self.wk1))

        num = self._integrate(self.w_simple * log_mag_diff)
        den = self._integrate(self.w_simple)

        if den == 0.0:
            return 0.0
        return num / den

    def correction(self) -> TimePhaseCorrection:
        """Return the first-order MP–MP corrections."""
        return TimePhaseCorrection(
            dt1=self.dt1(),
            dt2=0.0,
            dphi1=self.dphi1(),
            dphi2=0.0,
            dsnr1=self.dsnr1(),
        )


class AliasingWarning(UserWarning):
    """Warning raised when a kernel exhibits potential time-domain aliasing (wrap-around)."""

    pass


@dataclass
class ExtPsdDriftCorrection:
    """Exact correction kernels for PSD drift compensation.

    Computes the Minimum-Phase correction kernel K = W_live / W_ref that maps
    data whitened by the 'live' PSD back to the 'reference' PSD frame.

    Also supports computing the Adjoint kernel K^dagger for re-whitening applications.
    """

    freqs: np.ndarray
    psd_ref: np.ndarray
    psd_live: np.ndarray
    fs: float
    n_fft: Optional[int] = None

    def __post_init__(self):
        """Validate inputs."""
        self.freqs = np.asarray(self.freqs, dtype=float)
        self.psd_ref = np.asarray(self.psd_ref, dtype=float)
        self.psd_live = np.asarray(self.psd_live, dtype=float)

        if self.n_fft is None:
            self.n_fft = 2 * (self.freqs.size - 1)

        if not (self.psd_ref.size == self.psd_live.size == self.freqs.size):
            raise ValueError("PSD and frequency arrays must match in length.")

        # Basic sanitization to prevent div-by-zero or log(0) in kernels
        self.psd_ref = np.maximum(self.psd_ref, 1e-50)
        self.psd_live = np.maximum(self.psd_live, 1e-50)

    @property
    def df(self) -> float:
        """Frequency bin width."""
        if self.freqs.size > 1:
            return float(self.freqs[1] - self.freqs[0])
        return 1.0

    def _get_smoothed_ratio(self, smoothing_hz: float) -> np.ndarray:
        """Helper to compute smoothed PSD ratio P_live / P_ref."""
        ratio = self.psd_live / self.psd_ref
        if smoothing_hz > 0:
            sigma_bins = smoothing_hz / self.df
            ratio = gaussian_filter1d(ratio, sigma=sigma_bins, mode="nearest")
        return ratio

    def diagnose_time_aliasing(
        self, kernel: np.ndarray, threshold: float = 1e-3, tail_fraction: float = 0.05
    ) -> float:
        """Check if a causal kernel has wrapped energy at the end of the buffer.

        For a strictly causal filter computed via DFT, the end of the buffer
        (indices N-1, N-2...) corresponds to negative times. Significant energy
        here indicates the filter duration exceeds n_fft (Time-Domain Aliasing).

        Args:
            kernel: The time-domain impulse response.
            threshold: The warning threshold for the tail energy ratio.
            tail_fraction: The fraction of the buffer end to check (e.g. 0.05 = last 5%).

        Returns:
            float: The ratio of (Tail Energy / Total Energy).
        """
        N = len(kernel)
        n_tail = max(1, int(tail_fraction * N))

        # Calculate energy in the 'negative time' region (end of buffer)
        tail_energy = np.sum(kernel[-n_tail:] ** 2)
        total_energy = np.sum(kernel**2)

        if total_energy == 0:
            return 0.0

        ratio = tail_energy / total_energy

        if ratio > threshold:
            warnings.warn(
                f"Potential Time-Domain Aliasing detected! "
                f"Tail energy ratio {ratio:.2e} > threshold {threshold:.2e}. "
                f"The correction kernel may be wrapping around. "
                f"Consider increasing n_fft or smoothing inputs.",
                AliasingWarning,
            )

        return ratio

    def compute_correction_kernel(
        self,
        smoothing_hz: float = 0.0,
        window: Optional[WindowSpec] = None,
        truncate_samples: Optional[int] = None,
        check_aliasing: bool = True,
    ) -> np.ndarray:
        """Compute the causal correction kernel K(t).

        Algebra:
            K = W_live * W_ref^-1
            |K| = sqrt(P_ref / P_live)

        Args:
            smoothing_hz:
                Gaussian smoothing width in Hz applied to the ratio P_live/P_ref
                before kernel computation. Recommended to suppress noise artifacts.
            window:
                WindowSpec to apply to the time-domain kernel.
            truncate_samples:
                If set, truncates the kernel to this many samples. The kernel is
                assumed causal (starts at t=0), so this keeps indices [0, N].
            check_aliasing:
                If True (default), warns if the kernel has not decayed to zero
                at the end of the buffer (wrap-around risk).

        Returns:
            np.ndarray: Time-domain impulse response.
        """
        # 1. Compute Ratio and Kernel
        # MPWhiteningFilter(X) -> H = 1/sqrt(X). We want H = sqrt(P_ref/P_live).
        # Thus Input X = P_live / P_ref.
        ratio = self._get_smoothed_ratio(smoothing_hz)
        mp_filter = MPWhiteningFilter(psd=ratio, fs=self.fs, n_fft=self.n_fft)
        kernel = mp_filter.impulse_response()

        # 2. Check for Aliasing (before window/truncation hides it)
        if check_aliasing:
            self.diagnose_time_aliasing(kernel)

        # 3. Apply Truncation (Slicing for Causal)
        if truncate_samples is not None:
            if truncate_samples > len(kernel):
                raise ValueError(f"Truncation {truncate_samples} > N_FFT {len(kernel)}")
            kernel = kernel[:truncate_samples]

        # 4. Apply Window
        if window is not None:
            win_arr = window.make(len(kernel), center=0.0)
            kernel *= win_arr

        return kernel

    def compute_adjoint_kernel(
        self,
        smoothing_hz: float = 0.0,
        window: Optional[WindowSpec] = None,
        truncate_samples: Optional[int] = None,
        check_aliasing: bool = True,
    ) -> np.ndarray:
        """Compute the anti-causal adjoint correction kernel K^dagger(t).

        The adjoint is the time-reverse of the correction kernel: K^dagger(t) = K*(-t).
        In the frequency domain, this corresponds to conjugation: K^dagger(f) = K(f)*.

        Return Format:
            Returns the full N_FFT buffer suited for circular convolution / overlap-save.
            The "energy" of this anti-causal kernel is located at the beginning (index 0)
            and the end (indices N-1, N-2...) of the array.

        Args:
            smoothing_hz:
                Gaussian smoothing width in Hz.
            window:
                WindowSpec applied to the physical time axis (0, -dt, -2dt...).
                This effectively windows indices [0, N-1, N-2...] decaying away from 0.
            truncate_samples:
                If set, explicitly zeros out the "middle" of the buffer (the causal
                region), keeping only the 'truncate_samples' that correspond to the
                immediate future (t <= 0).
            check_aliasing:
                If True, checks the source causal kernel for aliasing before reversal.

        Returns:
            np.ndarray: Time-domain impulse response (length n_fft).
        """
        # 1. Compute Ratio
        ratio = self._get_smoothed_ratio(smoothing_hz)

        # 2. Get Causal Kernel in Frequency Domain
        mp_filter = MPWhiteningFilter(psd=ratio, fs=self.fs, n_fft=self.n_fft)

        # Check source aliasing implicitly by checking the time domain equivalent?
        # Efficiently: We can check the time domain kernel if requested.
        if check_aliasing:
            k_temp = mp_filter.impulse_response()
            self.diagnose_time_aliasing(k_temp)

        K_f = mp_filter.frequency_response()

        # 3. Compute Adjoint in Freq Domain (Conjugate)
        K_adj_f = np.conj(K_f)

        # 4. Transform to Time Domain (Full Buffer)
        k_adj_t = np.fft.irfft(K_adj_f, n=self.n_fft)

        # 5. Apply Window / Truncation Logic
        # For anti-causal, valid times are t=0, -1, -2...
        # In buffer indices: 0, N-1, N-2...
        # We construct a full-length window mask.

        N = len(k_adj_t)

        # Determine active region length
        L = truncate_samples if truncate_samples is not None else N

        # Build the mask/window array
        mask = np.zeros(N, dtype=float)

        if window is not None:
            # Generate window of length L (e.g. Tukey)
            # This window applies to time 0...L-1 (magnitude).
            # We map this to buffer indices 0, N-1, N-2...
            w_taps = window.make(L, center=0.0)

            # Apply w[0] to k[0]
            mask[0] = w_taps[0]

            # Apply w[1..L-1] to k[N-1..N-(L-1)] (reverse order in buffer)
            if L > 1:
                mask[N - (L - 1) :] = w_taps[1:][::-1]
        else:
            # Rectangular window (truncation only)
            mask[0] = 1.0
            if L > 1:
                mask[N - (L - 1) :] = 1.0

        # Apply mask
        k_adj_t *= mask

        return k_adj_t

    def compute_bias_measurements(
        self,
        h_tilde: np.ndarray,
        smoothing_hz: float = 0.0,
    ) -> TimePhaseCorrection:
        """Calculate exact scalar biases (dt, dphi, dsnr) by maximizing the
        inner product (Matched Filter overlap).

        This method avoids the perturbative approximation by explicitly
        reconstructing the time-domain overlap between the reference-whitened template
        and the live-whitened data. Uses sub-sample interpolation for dt.

        Args:
            h_tilde: Frequency domain waveform template (one-sided).
            smoothing_hz: Smoothing applied to the PSD ratio for stability.

        Returns:
            TimePhaseCorrection: A named tuple containing:
                dt1: Time shift (seconds).
                dphi1: Phase shift (radians).
                dsnr1: Fractional SNR change (Horizon distance shift).
                (dt2, dphi2 are zero).
        """
        if h_tilde.shape != self.freqs.shape:
            raise ValueError("Template h_tilde must match frequencies length.")

        # 1. Compute Correction Kernel Spectrum K(f)
        ratio = self._get_smoothed_ratio(smoothing_hz)
        mp = MPWhiteningFilter(ratio, self.fs, self.n_fft)
        K_f = mp.frequency_response()

        # 2. Construct Exact Overlap Integrand Z(f)
        # Z(f) = K(f) * |h(f)|^2 / P_ref(f)
        template_power_whitened = (np.abs(h_tilde) ** 2) / self.psd_ref
        Z_f = K_f * template_power_whitened

        # 3. Compute Complex Time-Domain Overlap (Analytic Signal)
        Z_full = np.zeros(self.n_fft, dtype=complex)
        Z_full[: len(Z_f)] = Z_f
        z_t = np.fft.ifft(Z_full)

        # 4. Find Peak Index and Magnitude
        abs_z = np.abs(z_t)
        idx_peak = np.argmax(abs_z)

        # 5. Sub-Sample Interpolation for dt
        # Quadratic interpolation around peak using magnitude
        y1 = abs_z[idx_peak - 1]
        y2 = abs_z[idx_peak]
        y3 = abs_z[(idx_peak + 1) % self.n_fft]

        # Parabolic peak location: delta in [-0.5, 0.5]
        denom = 2 * (2 * y2 - y1 - y3)
        if denom != 0:
            delta = (y1 - y3) / denom
        else:
            delta = 0.0

        exact_idx = idx_peak + delta

        # Handle wrap-around for negative times
        if exact_idx > self.n_fft // 2:
            dt_exact = (exact_idx - self.n_fft) / self.fs
        else:
            dt_exact = exact_idx / self.fs

        # 6. Extract Phase (at nearest peak is robust enough for comparison)
        dphi_exact = np.angle(z_t[idx_peak])

        # 7. Exact Fractional SNR Change (Comparison to Perturbative)
        # Perturbative dSNR calculates the Horizon Shift: (Rho_live - Rho_ref) / Rho_ref
        # Exact calculation:
        # Rho_ref = sqrt( integral |h|^2 / P_ref )
        # Rho_live = sqrt( integral |h|^2 / P_live )

        # We calculate these using the discrete sum consistent with FFT norm
        # Parseval: Sum |Z_f| ~ Integral... let's work in freq domain to be safe.
        df = self.df
        # Factor of 2 for one-sided to two-sided power, but ratio cancels it.

        # rho_sq_ref = 4 * sum( |h|^2 / P_ref ) * df
        # rho_sq_live = 4 * sum( |h|^2 / P_live ) * df

        # Avoid DC index 0 if needed, but array ops handle it.
        integ_ref = np.sum((np.abs(h_tilde) ** 2) / self.psd_ref)
        integ_live = np.sum((np.abs(h_tilde) ** 2) / self.psd_live)

        rho_ref = np.sqrt(integ_ref)
        rho_live = np.sqrt(integ_live)

        if rho_ref > 0:
            dsnr_exact = (rho_live / rho_ref) - 1.0
        else:
            dsnr_exact = 0.0

        return TimePhaseCorrection(
            dt1=float(dt_exact),
            dt2=0.0,
            dphi1=float(dphi_exact),
            dphi2=0.0,
            dsnr1=float(dsnr_exact),
        )
