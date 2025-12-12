from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import simpson

from zlw.kernels import MPWhiteningFilter

#: Convenience container for returning timing / phase / SNR corrections.
# Added 'dsnr1' to the tuple
TimePhaseCorrection = namedtuple("TimePhaseCorrection", "dt1 dt2 dphi1 dphi2 dsnr1")


@dataclass
class PertMpCorrection:
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
