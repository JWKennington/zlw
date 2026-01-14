"""
Convenience container for returning timing / phase / SNR corrections.
"""

from __future__ import annotations

import warnings
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, Any

import array_api_signal as aps
from zlw.kernels import MPWhiteningFilter
from zlw.window import WindowSpec

#: Convenience container for returning timing / phase / SNR corrections.
TimePhaseCorrection = namedtuple("TimePhaseCorrection", "dt1 dt2 dphi1 dphi2 dsnr1")


@dataclass
class PrtPsdDriftCorrection:
    r"""First–order perturbative MP–MP timing, phase, and SNR corrections for whitening
    mismatch.

    This class implements the *geometric* small–perturbation formulas for the
    MP–MP configuration.
    """

    freqs: Any
    psd1: Any
    psd2: Any
    h_tilde: Any
    fs: float

    # --- derived / cached quantities (populated in __post_init__) ---
    df: float = None
    n_fft: int = None
    wk1: Any = None
    wk2: Any = None
    w_simple: Any = None
    phi_diff: Any = None
    eps: float = None

    def __post_init__(self) -> None:
        """Validate inputs, build MP whiteners, and precompute weights."""
        # 1. Infer Backend
        self.xp = aps.array_namespace(self.freqs)

        # 2. Coerce inputs
        self.freqs = self.xp.asarray(self.freqs)
        self.psd1 = self.xp.asarray(self.psd1)
        self.psd2 = self.xp.asarray(self.psd2)
        self.h_tilde = self.xp.asarray(self.h_tilde)

        # Basic shape checks
        n = self.freqs.shape[0]
        if not (self.psd1.shape[0] == self.psd2.shape[0] == self.h_tilde.shape[0] == n):
            raise ValueError(
                "freqs, psd1, psd2, and htilde must all have the same length."
            )
        if n < 3:
            raise ValueError("Need at least 3 frequency bins for integration.")

        # Monotonic frequency grid
        # diff > 0 check
        diffs = self.xp.diff(self.freqs)
        if not self.xp.all(diffs > 0):
            raise ValueError("freqs must be strictly increasing (one-sided grid).")

        # PSD sanity
        if self.xp.any(self.psd1 <= 0) or self.xp.any(self.psd2 <= 0):
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
        eps_arr = self.xp.sqrt(self.psd1 / self.psd2) - 1.0
        self.eps = float(self.xp.max(self.xp.abs(eps_arr)))

    def _build_mp_filters(self) -> None:
        """Construct MP whitening filters and cache one-sided responses."""
        mp1 = MPWhiteningFilter(self.psd1, self.fs, self.n_fft)
        mp2 = MPWhiteningFilter(self.psd2, self.fs, self.n_fft)

        self.wk1 = mp1.frequency_response()  # real >= 0
        self.wk2 = mp2.frequency_response()  # complex

    def _precompute_weight_and_phase(self) -> None:
        """Precompute w(f) and the whitening phase difference Φ(f)."""
        # Effective spectral weight: |W2(f) * h(f)|^2
        self.w_simple = self.xp.abs(self.wk2 * self.h_tilde) ** 2

        # Whitening phase mismatch: Φ(f) = arg W2 − arg W1
        # Use aps.angle or fallback logic
        arg_wk2 = (
            self.xp.angle(self.wk2)
            if hasattr(self.xp, "angle")
            else self.xp.atan2(self.xp.imag(self.wk2), self.xp.real(self.wk2))
        )
        arg_wk1 = (
            self.xp.angle(self.wk1)
            if hasattr(self.xp, "angle")
            else self.xp.atan2(self.xp.imag(self.wk1), self.xp.real(self.wk1))
        )

        self.phi_diff = arg_wk2 - arg_wk1

    def _integrate(self, arr: Any) -> float:
        """Numerically integrate ``arr(f)`` over ``self.freqs``."""
        # Use array-api-signal's simpson
        return float(self.xp.simpson(arr, x=self.freqs))

    def dt1(self) -> float:
        r"""First-order timing correction :math:`\delta t^{(1)}` (seconds)."""
        num = self._integrate(self.freqs * self.w_simple * self.phi_diff)
        den = self._integrate(self.freqs**2 * self.w_simple)
        if den == 0.0:
            return 0.0
        # 1.0 / (2 * pi)
        val = (1.0 / (2.0 * 3.141592653589793)) * num / den
        return float(val)

    def dphi1(self) -> float:
        r"""First-order phase correction :math:`\delta\phi^{(1)}` (radians)."""
        num = self._integrate(self.w_simple * self.phi_diff)
        den = self._integrate(self.w_simple)
        if den == 0.0:
            return 0.0
        return float(num / den)

    def dsnr1(self) -> float:
        r"""First-order fractional SNR change :math:`\delta\rho^{(1)}/\rho`."""
        # ln(|W2|) - ln(|W1|)
        log_mag_diff = self.xp.log(self.xp.abs(self.wk2)) - self.xp.log(
            self.xp.abs(self.wk1)
        )

        num = self._integrate(self.w_simple * log_mag_diff)
        den = self._integrate(self.w_simple)

        if den == 0.0:
            return 0.0
        return float(num / den)

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
    """Warning raised when a kernel exhibits potential time-domain aliasing."""

    pass


@dataclass
class ExtPsdDriftCorrection:
    """Exact correction kernels for PSD drift compensation."""

    freqs: Any
    psd_ref: Any
    psd_live: Any
    fs: float
    n_fft: Optional[int] = None

    def __post_init__(self):
        """Validate inputs."""
        self.xp = aps.array_namespace(self.freqs)

        self.freqs = self.xp.asarray(self.freqs)
        self.psd_ref = self.xp.asarray(self.psd_ref)
        self.psd_live = self.xp.asarray(self.psd_live)

        if self.n_fft is None:
            self.n_fft = 2 * (self.freqs.shape[0] - 1)

        if not (self.psd_ref.shape[0] == self.psd_live.shape[0] == self.freqs.shape[0]):
            raise ValueError("PSD and frequency arrays must match in length.")

        # Basic sanitization (max with epsilon)
        # Using where: where(x < 1e-50, 1e-50, x)
        self.psd_ref = self.xp.where(self.psd_ref < 1e-50, 1e-50, self.psd_ref)
        self.psd_live = self.xp.where(self.psd_live < 1e-50, 1e-50, self.psd_live)

    @property
    def df(self) -> float:
        """Frequency bin width."""
        if self.freqs.shape[0] > 1:
            return float(self.freqs[1] - self.freqs[0])
        return 1.0

    def _get_smoothed_ratio(self, smoothing_hz: float) -> Any:
        """Helper to compute smoothed PSD ratio P_live / P_ref."""
        ratio = self.psd_live / self.psd_ref
        if smoothing_hz > 0:
            sigma_bins = smoothing_hz / self.df
            # Use array_api_signal's gaussian_filter1d
            ratio = self.xp.gaussian_filter1d(ratio, sigma=sigma_bins, mode="nearest")
        return ratio

    def diagnose_time_aliasing(
        self, kernel: Any, threshold: float = 1e-3, tail_fraction: float = 0.05
    ) -> float:
        """Check if a causal kernel has wrapped energy at the end of the buffer."""
        N = kernel.shape[0]
        n_tail = max(1, int(tail_fraction * N))

        # Calculate energy in the 'negative time' region (end of buffer)
        # slice: kernel[-n_tail:]
        tail_slice = kernel[-n_tail:]
        tail_energy = self.xp.sum(tail_slice**2)
        total_energy = self.xp.sum(kernel**2)

        if float(total_energy) == 0:
            return 0.0

        ratio = float(tail_energy / total_energy)

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
    ) -> Any:
        """Compute the causal correction kernel K(t)."""
        # 1. Compute Ratio and Kernel
        ratio = self._get_smoothed_ratio(smoothing_hz)
        mp_filter = MPWhiteningFilter(psd=ratio, fs=self.fs, n_fft=self.n_fft)
        kernel = mp_filter.impulse_response()

        # 2. Check for Aliasing
        if check_aliasing:
            self.diagnose_time_aliasing(kernel)

        # 3. Apply Truncation
        if truncate_samples is not None:
            if truncate_samples > kernel.shape[0]:
                raise ValueError(
                    f"Truncation {truncate_samples} > N_FFT {kernel.shape[0]}"
                )
            kernel = kernel[:truncate_samples]

        # 4. Apply Window
        if window is not None:
            # Pass xp context to window maker
            win_arr = window.make(kernel.shape[0], center=0.0, xp=self.xp)
            kernel *= win_arr

        return kernel

    def compute_adjoint_kernel(
        self,
        smoothing_hz: float = 0.0,
        window: Optional[WindowSpec] = None,
        truncate_samples: Optional[int] = None,
        check_aliasing: bool = True,
    ) -> Any:
        """Compute the anti-causal adjoint correction kernel K^dagger(t)."""
        # 1. Compute Ratio
        ratio = self._get_smoothed_ratio(smoothing_hz)

        # 2. Get Causal Kernel in Frequency Domain
        mp_filter = MPWhiteningFilter(psd=ratio, fs=self.fs, n_fft=self.n_fft)

        if check_aliasing:
            k_temp = mp_filter.impulse_response()
            self.diagnose_time_aliasing(k_temp)

        K_f = mp_filter.frequency_response()

        # 3. Compute Adjoint in Freq Domain (Conjugate)
        K_adj_f = self.xp.conj(K_f)

        # 4. Transform to Time Domain (Full Buffer)
        k_adj_t = self.xp.fft.irfft(K_adj_f, n=self.n_fft)

        # 5. Apply Window / Truncation Logic
        N = k_adj_t.shape[0]
        L = truncate_samples if truncate_samples is not None else N

        # Build the mask/window array
        mask = self.xp.zeros(N, dtype=k_adj_t.dtype)

        if window is not None:
            # Generate window of length L
            w_taps = window.make(L, center=0.0, xp=self.xp)

            # Apply w[0] to k[0]
            # Since array api doesn't support scalar item assignment easily for JAX,
            # we typically construct by parts or use .at[].set.
            # However, for 1D arrays, concatenation is safe.

            # We need to construct mask: [w[0], 0...0, w[L-1]...w[1]]

            # w[0]
            m0 = w_taps[0:1]

            if L > 1:
                # w[1:] reversed
                w_rev = self.xp.flip(w_taps[1:], axis=0)

                # zeros in middle
                # indices: 1 to N-(L-1) are zeros.
                # len = N - 1 - (L - 1) = N - L
                n_zeros = N - L
                zeros = self.xp.zeros(n_zeros, dtype=mask.dtype)

                mask = self.xp.concat([m0, zeros, w_rev], axis=0)
            else:
                # L=1, just w[0] and zeros
                zeros = self.xp.zeros(N - 1, dtype=mask.dtype)
                mask = self.xp.concat([m0, zeros], axis=0)

        else:
            # Rectangular window
            # mask[0] = 1, mask[-(L-1):] = 1
            one = self.xp.ones(1, dtype=mask.dtype)

            if L > 1:
                n_zeros = N - L
                zeros = self.xp.zeros(n_zeros, dtype=mask.dtype)
                # end part is ones of length L-1
                ones_end = self.xp.ones(L - 1, dtype=mask.dtype)
                mask = self.xp.concat([one, zeros, ones_end], axis=0)
            else:
                zeros = self.xp.zeros(N - 1, dtype=mask.dtype)
                mask = self.xp.concat([one, zeros], axis=0)

        # Apply mask
        k_adj_t *= mask

        return k_adj_t

    def compute_bias_measurements(
        self,
        h_tilde: Any,
        smoothing_hz: float = 0.0,
    ) -> TimePhaseCorrection:
        """Calculate exact scalar biases (dt, dphi, dsnr)."""
        # Ensure h_tilde is array
        h_tilde = self.xp.asarray(h_tilde)

        if h_tilde.shape[0] != self.freqs.shape[0]:
            raise ValueError("Template h_tilde must match frequencies length.")

        # 1. Compute Correction Kernel Spectrum K(f)
        ratio = self._get_smoothed_ratio(smoothing_hz)
        mp = MPWhiteningFilter(ratio, self.fs, self.n_fft)
        K_f = mp.frequency_response()

        # 2. Construct Exact Overlap Integrand Z(f)
        template_power_whitened = (self.xp.abs(h_tilde) ** 2) / self.psd_ref
        Z_f = K_f * template_power_whitened

        # 3. Compute Complex Time-Domain Overlap
        # Needs padding to full FFT size (Z_f is one-sided)
        # Z_full construction
        # Z_f: [DC, 1... Nyq]
        # We need to reconstruct full symmetric spectrum for IFFT
        # Or just use irfft if we treat it as real signal?
        # No, Z(f) is generally complex (phase shift).

        # We assume n_fft is even.
        # Z_f size is n_fft//2 + 1

        # NOTE: Using standard IFFT on one-sided data usually requires reconstruction.
        # But 'array-api-signal' doesn't have 'irfft_complex' helper.
        # We manually construct Z_full.
        # Z_full[0] = Z_f[0]
        # Z_full[1:mid] = Z_f[1:-1]
        # Z_full[mid:] = 0 ?? No.

        # Easier path: Z_t = IFFT(Z_full).
        # We construct Z_full explicitly.

        z_dc = Z_f[0:1]
        z_pos = Z_f[1:-1]  # Positive freqs
        z_nyq = Z_f[-1:]

        # Negative freqs?
        # Since h(t) and k(t) are real, Z(f) should be hermitian symmetric?
        # NO. h(t) is real, but if h_tilde is the template, we care about the overlap.
        # Wait, the method says "Analytic Signal".
        # If we want the analytic signal overlap, we usually zero out negative freqs.

        # "Z_full[: len(Z_f)] = Z_f; Z_full[others] = 0" implies analytic signal construction.
        # Let's replicate that logic.

        zeros_len = self.n_fft - Z_f.shape[0]
        zeros = self.xp.zeros(zeros_len, dtype=Z_f.dtype)
        Z_full = self.xp.concat([Z_f, zeros], axis=0)

        z_t = self.xp.fft.ifft(Z_full)

        # 4. Find Peak
        abs_z = self.xp.abs(z_t)
        idx_peak = int(self.xp.argmax(abs_z))

        # 5. Sub-Sample Interpolation for dt
        # Need scalar access
        y1 = float(abs_z[idx_peak - 1])
        y2 = float(abs_z[idx_peak])
        y3 = float(abs_z[(idx_peak + 1) % self.n_fft])

        denom = 2 * (2 * y2 - y1 - y3)
        if denom != 0:
            delta = (y1 - y3) / denom
        else:
            delta = 0.0

        exact_idx = idx_peak + delta

        if exact_idx > self.n_fft // 2:
            dt_exact = (exact_idx - self.n_fft) / self.fs
        else:
            dt_exact = exact_idx / self.fs

        # 6. Extract Phase
        dphi_exact = float(self.xp.angle(z_t[idx_peak]))

        # 7. Exact Fractional SNR Change
        integ_ref = self.xp.sum((self.xp.abs(h_tilde) ** 2) / self.psd_ref)
        integ_live = self.xp.sum((self.xp.abs(h_tilde) ** 2) / self.psd_live)

        rho_ref = float(self.xp.sqrt(integ_ref))
        rho_live = float(self.xp.sqrt(integ_live))

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
