"""
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import simpson
from zlw.kernels import MPWhiteningFilter

TimePhaseCorrection = namedtuple("TimePhaseCorrection", "dt1 dt2 dphi1 dphi2")


@dataclass
class MPMPCorrection:
    """Compute first- and second-order timing (dt1, dt2) and phase (dphi1, dphi2)
    corrections for the MP–MP whitening mismatch scheme.

    Args:
        freqs : np.ndarray
            One-sided frequency bins (Hz), length N_bins.
        psd1 : np.ndarray
            One-sided PSD array S1(f) used for the linear-phase (template) filter.
        psd2 : np.ndarray
            One-sided PSD array S2(f) used for the minimum-phase (data) filter.
        htilde : np.ndarray
            One-sided frequency-domain template waveform ˜h(f), length N_bins.
        fs : float
            Sampling rate (Hz).

    Details on the derivation can be found in [2].

    References:
        [2]:
            Eqs 4.7, 4.8, 4.9, 4.20, 4.21
    """

    freqs: np.ndarray
    psd1: np.ndarray
    psd2: np.ndarray
    htilde: np.ndarray
    fs: float

    def __post_init__(self):
        """Validate inputs and precompute common quantities."""
        # frequency bin width
        self.df = self.freqs[1] - self.freqs[0]
        # infer full FFT length from one-sided PSD length
        self.n_fft = int((self.psd1.size - 1) * 2)

        # build whitening filters
        # linear-phase filter: amplitude 1/sqrt(psd1), zero phase
        self.mp1 = MPWhiteningFilter(self.psd1, self.fs, self.n_fft)
        # minimum-phase filter
        self.mp2 = MPWhiteningFilter(self.psd2, self.fs, self.n_fft)

        # get one-sided frequency responses
        self.wk1 = self.mp1.frequency_response()  # real positive array
        self.wk2 = self.mp2.frequency_response()  # complex array
        self.dwk = self.wk2 - self.wk1  # complex array

        # Compute epsilon factor to determine if we are in the perturbative regime
        self._eps = np.sqrt(self.psd1 / self.psd2) - 1.0
        self.eps = np.max(np.abs(self._eps))

    def _integrate(self, arr: np.ndarray) -> float:
        """Numerically integrate arr(f) over self.freqs using:
          1) Composite Simpson's rule, if there are an odd number of points (even
          number of intervals).
          2) Otherwise, fallback to the trapezoidal rule.
        This improves on the simple Riemann sum by capturing curvature.
        """
        # Number of points
        n = arr.size

        # Simpson's rule requires an odd number of points (n = 2m+1)
        if n % 2 == 1:
            # simpson handles odd n automatically
            return float(simpson(arr, self.freqs))
        else:
            # fall back to trapezoid for even n
            return float(np.trapezoid(arr, self.freqs))

    def _d_i_tt(self):
        """Partial derivative of I0 w.r.t. t_c twice."""
        coeff = -((2 * np.pi) ** 2)
        integrand = self.freqs ** 2 * np.abs(self.wk1 * self.htilde) ** 2
        return coeff * self._integrate(integrand)

    def _d_i_pp(self):
        """Partial derivative of I0 w.r.t. phi_c twice."""
        coeff = -1
        integrand = np.abs(self.wk1 * self.htilde) ** 2
        return coeff * self._integrate(integrand)

    def _d_di_t(self, data: Optional[np.ndarray] = None):
        """Partial derivative of ΔI w.r.t. t_c.

        Args:
            data:
                ndarray, default None. if specified, must be the FFT of the
                time domain data, that should still be colored by psd2 (not whitened),
                must also be one-sided and same length as self.freqs. If None,
                assume data = htilde * exp(i phi0).
        """
        if data is None:
            data = self.htilde
        coeff = 2 * np.pi
        integrand = (
                self.freqs * np.imag(self.dwk) * data * np.conj(self.wk1 * self.htilde)
        )
        return coeff * self._integrate(integrand)

    def _d_di_p(self, data: Optional[np.ndarray] = None):
        """Partial derivative of ΔI w.r.t. phi_c.

        Args:
            data:
                ndarray, default None. if specified, must be the FFT of the
                time domain data, that should still be colored by psd2 (not whitened),
                must also be one-sided and same length as self.freqs. If None,
                assume data = htilde * exp(i phi0).
        """
        if data is None:
            data = self.htilde
        coeff = 1.0
        integrand = np.imag(self.dwk) * data * np.conj(self.wk1 * self.htilde)
        return coeff * self._integrate(integrand)

    def _d_di_tt(self, data: Optional[np.ndarray] = None):
        """Partial derivative of ΔI w.r.t. t_c twice."""
        if data is None:
            data = self.htilde
        coeff = -((2 * np.pi) ** 2)
        integrand = self.freqs ** 2 * self.dwk * data * np.conj(self.wk1 * self.htilde)
        return coeff * self._integrate(integrand)

    def _d_di_tp(self, data: Optional[np.ndarray] = None):
        """Partial derivative of ΔI w.r.t. t_c and phi_c."""
        if data is None:
            data = self.htilde
        coeff = -2 * np.pi
        integrand = self.freqs * self.dwk * data * np.conj(self.wk1 * self.htilde)
        return coeff * self._integrate(integrand)

    def _d_di_pp(self, data: Optional[np.ndarray] = None):
        """Partial derivative of ΔI w.r.t. phi_c twice."""
        if data is None:
            data = self.htilde
        coeff = -1.0
        integrand = self.dwk * data * np.conj(self.wk1 * self.htilde)
        return coeff * self._integrate(integrand)

    def dt1_full(self, data: Optional[np.ndarray] = None) -> float:
        """Compute the first-order timing correction δt^(1), with full precision."""
        num = self._d_di_t(data=data)
        den = self._d_i_tt()
        return num / den if den != 0 else 0.0

    def dphi1_full(self, data: Optional[np.ndarray] = None) -> float:
        """Compute the first-order phase correction δφ^(1), with full precision."""
        num = self._d_di_p(data=data)
        den = self._d_i_pp()
        return num / den if den != 0 else 0.0

    def dt1_simple(self) -> float:
        """Compute the first-order timing correction δt^(1), simplified form."""
        w_simple = np.abs(self.wk2 * self.htilde) ** 2
        phi_diff = np.angle(self.wk2) - np.angle(self.wk1)
        num = self._integrate(self.freqs * w_simple * phi_diff)
        den = self._integrate(self.freqs ** 2 * w_simple)
        coeff = 1 / (2 * np.pi)
        return 0 if den == 0 else coeff * num / den

    def dphi1_simple(self) -> float:
        """Compute the first-order phase correction δφ^(1), simplified form."""
        w_simple = np.abs(self.wk2 * self.htilde) ** 2
        phi_diff = np.angle(self.wk2) - np.angle(self.wk1)
        num = self._integrate(w_simple * phi_diff)
        den = self._integrate(w_simple)
        return 0 if den == 0 else num / den

    def dt2_full(
            self,
            data: Optional[np.ndarray] = None,
            dt1: Optional[float] = None,
            dphi1: Optional[float] = None,
    ) -> float:
        """Compute the second-order timing correction δt^(2), with full precision."""
        # TODO make these more efficient by optionally passing in all partial derivs
        if dt1 is None:
            dt1 = self.dt1_full(data=data)
        if dphi1 is None:
            dphi1 = self.dphi1_full(data=data)
        coeff = -1.0
        num = (
                np.real(self._d_di_tt(data=data)) * dt1
                + np.real(self._d_di_tp(data=data)) * dphi1
        )
        den = self._d_i_tt()
        return coeff * num / den if den != 0 else 0.0

    def dphi2_full(
            self,
            data: Optional[np.ndarray] = None,
            dt1: Optional[float] = None,
            dphi1: Optional[float] = None,
    ) -> float:
        """Compute the second-order phase correction δφ^(2), with full precision."""
        if dt1 is None:
            dt1 = self.dt1_full(data=data)
        if dphi1 is None:
            dphi1 = self.dphi1_full(data=data)
        coeff = -1.0
        num = (
                np.real(self._d_di_tp(data=data)) * dt1
                + np.real(self._d_di_pp(data=data)) * dphi1
        )
        den = self._d_i_pp()
        return coeff * num / den if den != 0 else 0.0

    def full_correction(
            self,
            data: Optional[np.ndarray] = None,
    ) -> TimePhaseCorrection:
        """Compute the general first- and second-order corrections.

        Returns:
            TimePhaseCorrection: (dt1, dt2, dphi1, dphi2)
        """
        dt1 = self.dt1_full(data=data)
        dphi1 = self.dphi1_full(data=data)
        dt2 = self.dt2_full(data=data, dt1=dt1, dphi1=dphi1)
        dphi2 = self.dphi2_full(data=data, dt1=dt1, dphi1=dphi1)

        return TimePhaseCorrection(
            dt1=dt1,
            dt2=dt2,
            dphi1=dphi1,
            dphi2=dphi2,
        )

    def simplified_correction(self) -> TimePhaseCorrection:
        """Compute the first-order corrections under small-perturbation approximation:

        Returns:
            TimePhaseCorrection: (dt1, 0.0, dphi1, 0.0)
        """
        dt1 = self.dt1_simple()
        dphi1 = self.dphi1_simple()

        return TimePhaseCorrection(
            dt1=dt1,
            dt2=0.0,
            dphi1=dphi1,
            dphi2=0.0,
        )
