"""Whitening filter kernel computation for LIGO data processing.


References:
    [1]: GstLAL Implementation: https://git.ligo.org/lscsoft/gstlal/-/blob/master
    /gstlal/python/kernels.py#L81
    [2]: Revised Implementation Derivations:
    https://www.overleaf.com/read/kjjbcqtwwfsj#3b5b59
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy

from zlw.fourier import FourierBackend, NumpyFourierBackend
from zlw.window import WindowSpec


class PSDAdmissibilityError(ValueError):
    """Raised when a PSD violates mathematical admissibility conditions."""

    pass


class PSDAdmissibilityWarning(UserWarning):
    """Warns when a PSD has suspicious properties (e.g. unnaturally quiet DC)."""

    pass


@dataclass
class WhiteningFilter:
    """Base class for whitening filters. Implements the shared amplitude response
    and leave phase response methods to be implemented by subclasses.

    Attributes:
        psd:
            np.ndarray, One-sided power spectral density, length n_fft//2 + 1.
        f_sample:
            Sampling frequency in Hz.
        n_fft:
            Optional[int], FFT length. if None, inferred as 2*(len(psd)-1).
    """

    psd: numpy.ndarray
    fs: float
    n_fft: Optional[int] = None
    fb: FourierBackend = field(default_factory=NumpyFourierBackend)

    def __post_init__(self):
        """Validate PSD length and infer n_fft if necessary."""
        # Check FFT length
        if self.n_fft is None:
            self.n_fft = 2 * (self.psd.size - 1)
        assert isinstance(self.n_fft, int)  # for typing checks

        # Coerce PSD to numpy array and ensure it's float type
        self.psd = numpy.asarray(self.psd, float)

        # Validate PSD length
        # TODO be able to handle raw physical PSDs and interpolate them
        #   to the expected length
        expected = self.n_fft // 2 + 1
        if self.psd.size != expected:
            raise ValueError(
                f"PSD length must be n_fft//2 + 1 ({expected}); got {self.psd.size}"
            )

        # Run rigorous checks
        self.validate_admissibility()

    def validate_admissibility(self) -> None:
        """Check if the PSD is mathematically and numerically admissible.

        Checks:
        1. Strict Positivity (Essential for inversion).
        2. Paley-Wiener Condition (Essential for MP factorization).
        3. Seismic Wall Integrity (Warns if DC is suspiciously quiet).
        """
        # 1. Check for Non-Positive Values
        min_val = numpy.min(self.psd)
        if min_val <= 0:
            raise PSDAdmissibilityError(
                f"PSD must be strictly positive. Found min value {min_val}. "
                "Whitening requires division by sqrt(PSD), which is undefined for <= 0."
            )

        # 2. Check for Numerical Underflow / Singularities
        # If PSD < 1e-48 (approx float64 limit for meaningful inverse),
        # the gain will exceed 1e24, causing integrator instability.
        critical_floor = 1e-48
        if min_val < critical_floor:
            warnings.warn(
                f"PSD contains extremely small values (< {critical_floor}). "
                f"Min value: {min_val:.2e}. This implies a whitening gain > 1e24, "
                "which may cause numerical instability or 'integrator drift'.",
                PSDAdmissibilityWarning,
            )

        # 3. Check for 'Seismic Wall' Artifacts (The "Welch Rolloff" Issue)
        # In physical ground-based GW data, low frequencies should be LOUD.
        # If DC is quieter than 20 Hz, it usually means the estimator (Welch)
        # artificially suppressed it via windowing.
        # We check if PSD[0] << PSD[at 20Hz].
        if self.fs > 40.0:  # Only meaningful if we resolve low freqs
            # Find bin for ~20Hz
            idx_20 = int(20.0 / (self.fs / self.n_fft))
            if idx_20 < len(self.psd):
                val_dc = self.psd[0]
                val_20 = self.psd[idx_20]

                # If DC is 100x quieter than 20Hz, that's suspicious for GW data
                if val_dc < 0.01 * val_20:
                    warnings.warn(
                        "Suspicious Low-Frequency Rolloff detected. "
                        f"PSD at DC ({val_dc:.2e}) is significantly lower than at 20Hz ({val_20:.2e}). "
                        "Real seismic noise should increase at low frequencies. "
                        "This may be an artifact of windowed spectral estimation. "
                        "Consider using a 'Seismic Wall' condition/fix.",
                        PSDAdmissibilityWarning,
                    )

    @property
    def peak_center(self) -> float:
        """The expected location (index) of the impulse response peak.
        Defaults to 0.0 (Minimum Phase / Causal).
        """
        return 0.0

    def amplitude_response(self) -> numpy.ndarray:
        """Compute the one-sided amplitude response |H(f)| = 1/sqrt(psd).

        Returns:
            np.ndarray,
                amplitude response of length n_fft//2 + 1.
        """
        return 1.0 / numpy.sqrt(self.psd)

    def frequency_response(self) -> numpy.ndarray:
        """Compute the one-sided complex frequency response H(f).
        To be implemented by subclasses.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("Subclasses must implement frequency_response")

    def phase_response(self) -> numpy.ndarray:
        """Compute the phase response φ(f) = arg[H(f)]. Can be overridden
        by subclasses if they have a specific phase response.

        Returns:
            np.ndarray: phase response of length n_fft//2 + 1.
        """
        H = self.frequency_response()
        return numpy.angle(H)

    def impulse_response(
        self, inverse: bool = False, window: Optional[WindowSpec] = None
    ) -> numpy.ndarray:
        """Compute the real-valued, time-domain impulse response via inverse FFT.

        Args:
            inverse:
                bool, default False. If True, compute the inverse filter
                impulse response, i.e., irfft(1/H).
            window:
                Optional[WindowSpec], default None. If provided, apply the specified
                time-domain window to the taps, preserving the L2 norm of the
                unwindowed taps.

        Returns:
            np.ndarray: impulse response of length n_fft.
        """
        H = self.frequency_response()

        if inverse:
            # Avoid division by zero by adding a small epsilon
            eps = 1e-12
            H_mag2 = H.real * H.real + H.imag * H.imag
            H_safe = numpy.where(H_mag2 > eps * eps, H, eps + 0j)
            H = 1.0 / H_safe

        h = self.fb.irfft(H, n=self.n_fft)
        if window is not None:
            w = window.make(h.size, center=self.peak_center)

            # Preserve energy of the unwindowed taps
            e0 = float(numpy.dot(h, h))
            if e0 > 0.0:
                hw = h * w
                e1 = float(numpy.dot(hw, hw))
                if e1 > 0.0:
                    h = hw * numpy.sqrt(e0 / e1)
                else:
                    h = hw
            else:
                h = h * w
        return h

    # FFT helpers encapsulate parameters and normalization
    def _rfft(self, x: numpy.ndarray) -> numpy.ndarray:
        return self.fb.rfft(x)

    def _irfft(self, X: numpy.ndarray) -> numpy.ndarray:
        return self.fb.irfft(X, n=self.n_fft)

    def _fft(self, x: numpy.ndarray) -> numpy.ndarray:
        return self.fb.fft(x)

    def _ifft(self, X: numpy.ndarray) -> numpy.ndarray:
        return self.fb.ifft(X, n=self.n_fft)


@dataclass
class LPWhiteningFilter(WhiteningFilter):
    """Linear-phase whitening filter.

    Implements H(f) = 1/sqrt(psd) * exp(-2πi f delay).
    """

    delay: float = 0.0

    @property
    def peak_center(self) -> float:
        """For Linear Phase, the peak is at the specified delay."""
        return self.delay * self.fs

    def phase_response(self) -> numpy.ndarray:
        """Compute the phase response φ(f) = -2πf delay.

        Returns:
            np.ndarray: phase response of length n_fft//2 + 1.
        """
        freqs = self.fb.rfftfreq(self.n_fft, 1.0 / self.fs)
        return -2 * numpy.pi * freqs * self.delay

    def frequency_response(self) -> numpy.ndarray:
        """Compute one-sided H(f) with a pure delay.

        Args:
            delay (float): time delay in seconds.

        Returns:
            np.ndarray: complex frequency response.
        """
        H = self.amplitude_response()

        # Cast to complex type for multiplication
        H = H.astype(numpy.complex128)

        if self.delay != 0.0:
            # Apply phase response for linear phase filter
            phase = self.phase_response()
            H *= numpy.exp(1j * phase)

        return H


@dataclass
class MPWhiteningFilter(WhiteningFilter):
    """Minimum-phase whitening filter via folded-cepstrum method."""

    clamp_log_min: float = -700.0
    clamp_log_max: float = 700.0

    def _dht_folded_cepstrum(
        self,
        data: numpy.ndarray,
        full_spectrum: bool = False,
    ) -> numpy.ndarray:
        """Helper method to compute the discrete Hilbert transform via
        the folded cepstrum method. Here we follow the steps proven
        in Thm A.3 in [2]

        Args:
            data:
                np.ndarray, real-valued data to transform.
            full_spectrum:
                bool, if True, return the combined spectrum of
                the original data and its

        Returns:
            np.ndarray: complex DHT of the input data.
        """
        # 1. Compute the real cepstrum via IFFT
        cepstrum = self._ifft(data)

        # 2. Fold the negative quefrencies
        folded = numpy.zeros_like(cepstrum)
        # Preserve the zero quefrency component (DC)
        folded[0] = cepstrum[0]
        # Preserve the Nyquist quefrency component (if applicable)
        folded[self.n_fft // 2] = cepstrum[self.n_fft // 2]
        # Double the positive quefrencies (preserving the energy of the signal)
        folded[1 : self.n_fft // 2] = 2 * cepstrum[1 : self.n_fft // 2]

        # 3. Compute the complex DHT via FFT
        freq_response = self._fft(folded)

        # Optionally return the full spectrum (useful as a shortcut for
        # computing the full complex spectrum)
        if full_spectrum:
            return freq_response

        # By default return only the imaginary part, which is the DHT
        return freq_response.imag

    def frequency_response(self) -> numpy.ndarray:
        """Compute one-sided minimum-phase frequency response H(f)
        by computing the discrete Hilbert transform of the log-spectrum.
        A brief summary of the steps is as follows:

          1. Log amplitude response L = -0.5*ln(psd)
          2. real cepstrum via full IFFT
          3. fold negative quefrencies
          4. FFT to get complex log-spectrum
          5. exponentiate

        For more detail on the process, see Thm A.3 in [2],
        as well as Section 2.2 in [2].

        Returns:
            np.ndarray: complex minimum-phase frequency response.
        """
        # 1. Compute the log-amplitude response
        #    We use careful error handling for zeros (log -> -inf)
        #    or infinities (log -> +inf).

        with numpy.errstate(divide="ignore", invalid="ignore"):
            log_psd = numpy.log(self.psd)

        # Clamp huge values to preserve FFT stability.
        # Clamping to +/- 700 preserves full float64 dynamic range without NaNs.
        log_psd = numpy.nan_to_num(
            log_psd, posinf=self.clamp_log_max, neginf=self.clamp_log_min
        )

        # 1. Compute the log-amplitude response
        #    Note that the below is equivalent to
        #    log_spectrum = np.log(self.amplitude_response())
        log_amp_res = -0.5 * log_psd

        # Compute two-sided log spectrum
        log_amp_res_full = numpy.zeros(self.n_fft, dtype=numpy.float64)
        log_amp_res_full[: self.n_fft // 2 + 1] = log_amp_res
        log_amp_res_full[self.n_fft // 2 + 1 :] = log_amp_res[1 : self.n_fft // 2][::-1]

        # 2. Compute the log frequency response
        #    via the folded cepstrum method
        log_freq_res = self._dht_folded_cepstrum(
            data=log_amp_res_full,
            full_spectrum=True,
        )

        # 3. exponentiate to get H_full
        freq_res = numpy.exp(log_freq_res)

        # Return the one-sided frequency response
        return freq_res[: self.n_fft // 2 + 1]
