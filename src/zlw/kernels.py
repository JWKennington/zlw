"""Whitening filter kernel computation for LIGO data processing.

References:
    [1]: GstLAL Implementation: https://git.ligo.org/lscsoft/gstlal/-/blob/master
    /gstlal/python/kernels.py#L81
    [2]: Revised Implementation Derivations:
    https://www.overleaf.com/read/kjjbcqtwwfsj#3b5b59
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Any

import array_api_signal as aps
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
            array, One-sided power spectral density, length n_fft//2 + 1.
        f_sample:
            Sampling frequency in Hz.
        n_fft:
            Optional[int], FFT length. if None, inferred as 2*(len(psd)-1).
    """

    psd: Any  # Supports any Array API array
    fs: float
    n_fft: Optional[int] = None

    def __post_init__(self):
        """Validate PSD length and infer n_fft if necessary."""
        # 1. Infer Backend
        self.xp = aps.array_namespace(self.psd)

        # 2. Coerce PSD
        self.psd = self.xp.asarray(self.psd)

        # 3. Infer FFT length if missing
        if self.n_fft is None:
            self.n_fft = 2 * (self.psd.shape[-1] - 1)

        # 4. Validate Length
        expected = self.n_fft // 2 + 1
        if self.psd.shape[-1] != expected:
            raise ValueError(
                f"PSD length must be n_fft//2 + 1 ({expected}); got {self.psd.shape[-1]}"
            )

        # 5. Run rigorous checks
        self.validate_admissibility()

    def validate_admissibility(self) -> None:
        """Check if the PSD is mathematically and numerically admissible."""
        # Note: We cast to float() to ensure we have Python scalars for comparisons/warnings
        # irrespective of whether the backend is Torch, JAX, or NumPy.
        min_val = float(self.xp.min(self.psd))

        # 1. Check for Non-Positive Values
        if min_val <= 0:
            raise PSDAdmissibilityError(
                f"PSD must be strictly positive. Found min value {min_val}. "
                "Whitening requires division by sqrt(PSD), which is undefined for <= 0."
            )

        # 2. Check for Numerical Underflow / Singularities
        critical_floor = 1e-48
        if min_val < critical_floor:
            warnings.warn(
                f"PSD contains extremely small values (< {critical_floor}). "
                f"Min value: {min_val:.2e}. This implies a whitening gain > 1e24, "
                "which may cause numerical instability or 'integrator drift'.",
                PSDAdmissibilityWarning,
            )

        # 3. Check for 'Seismic Wall' Artifacts
        if self.fs > 40.0:
            idx_20 = int(20.0 / (self.fs / self.n_fft))
            if idx_20 < self.psd.shape[-1]:
                val_dc = float(self.psd[0])
                val_20 = float(self.psd[idx_20])

                if val_dc < 0.01 * val_20:
                    warnings.warn(
                        "Suspicious Low-Frequency Rolloff detected. "
                        f"PSD at DC ({val_dc:.2e}) is significantly lower than at 20Hz ({val_20:.2e}). "
                        "Real seismic noise should increase at low frequencies.",
                        PSDAdmissibilityWarning,
                    )

    @property
    def peak_center(self) -> float:
        """The expected location (index) of the impulse response peak."""
        return 0.0

    def amplitude_response(self) -> Any:
        """Compute the one-sided amplitude response |H(f)| = 1/sqrt(psd)."""
        return 1.0 / self.xp.sqrt(self.psd)

    def frequency_response(self) -> Any:
        """Compute the one-sided complex frequency response H(f)."""
        raise NotImplementedError("Subclasses must implement frequency_response")

    def phase_response(self) -> Any:
        """Compute the phase response φ(f) = arg[H(f)]."""
        H = self.frequency_response()
        # aps.angle isn't standard yet in all backends, but usually available via xp.angle
        # If missing, we can implement via arctan2(imag, real)
        if hasattr(self.xp, "angle"):
            return self.xp.angle(H)
        return self.xp.atan2(self.xp.imag(H), self.xp.real(H))

    def impulse_response(
        self, inverse: bool = False, window: Optional[WindowSpec] = None
    ) -> Any:
        """Compute the real-valued, time-domain impulse response via inverse FFT."""
        H = self.frequency_response()

        if inverse:
            eps = 1e-12
            # H_safe inversion
            H_mag2 = self.xp.real(H) ** 2 + self.xp.imag(H) ** 2
            # Ensure complex type for eps addition
            safe_denom = self.xp.where(H_mag2 > eps**2, H, eps + 0j)
            H = 1.0 / safe_denom

        h = self.xp.fft.irfft(H, n=self.n_fft)

        if window is not None:
            # Pass our backend (self.xp) to the window generator
            w = window.make(h.shape[-1], center=self.peak_center, xp=self.xp)

            # Preserve energy of the unwindowed taps
            e0 = self.xp.sum(h * h)
            e0_val = float(e0)

            if e0_val > 0.0:
                hw = h * w
                e1 = self.xp.sum(hw * hw)
                e1_val = float(e1)
                if e1_val > 0.0:
                    h = hw * self.xp.sqrt(e0 / e1)
                else:
                    h = hw
            else:
                h = h * w

        return h


@dataclass
class LPWhiteningFilter(WhiteningFilter):
    """Linear-phase whitening filter.

    Implements H(f) = 1/sqrt(psd) * exp(-2πi f delay).
    """

    delay: float = 0.0

    @property
    def peak_center(self) -> float:
        return self.delay * self.fs

    def phase_response(self) -> Any:
        freqs = self.xp.fft.rfftfreq(self.n_fft, d=1.0 / self.fs)
        # Use explicit PI to match backend precision if needed
        pi = 3.141592653589793
        return -2 * pi * freqs * self.delay

    def frequency_response(self) -> Any:
        H = self.amplitude_response()
        # Ensure complex type
        if hasattr(self.xp, "complex128"):
            ctype = self.xp.complex128
        else:
            ctype = self.xp.complex64  # Fallback
        H = self.xp.astype(H, ctype)

        if self.delay != 0.0:
            phase = self.phase_response()
            # exp(1j * phase)
            rot = self.xp.exp(1j * phase)
            H = H * rot

        return H


@dataclass
class MPWhiteningFilter(WhiteningFilter):
    """Minimum-phase whitening filter via folded-cepstrum method."""

    clamp_log_min: float = -700.0
    clamp_log_max: float = 700.0

    def _dht_folded_cepstrum(self, data: Any, full_spectrum: bool = False) -> Any:
        """Helper method to compute the discrete Hilbert transform."""
        # 1. Compute the real cepstrum via IFFT (assumes 'data' is two-sided symmetric real)
        # However, the input 'data' here is constructed to be the full log spectrum.
        # Since it is real and symmetric, IFFT is real.
        cepstrum = self.xp.fft.ifft(data)

        # 2. Fold the negative quefrencies
        # We need to construct 'folded' array.
        # In Array API, in-place modification is frowned upon (and impossible in JAX).
        # We construct it via masking or concatenation.

        # cepstrum indices: [0, 1, ... N/2, ... N-1]
        n = self.n_fft
        mid = n // 2

        # 0 (DC) -> Keep
        # 1..mid-1 -> Double
        # mid (Nyquist) -> Keep (if even)
        # mid+1 .. end -> Zero

        c0 = cepstrum[..., 0:1]
        c_pos = cepstrum[..., 1:mid] * 2
        c_nyq = cepstrum[..., mid : mid + 1]
        c_neg = self.xp.zeros_like(cepstrum[..., mid + 1 :])

        folded = self.xp.concat([c0, c_pos, c_nyq, c_neg], axis=-1)

        # 3. Compute complex DHT via FFT
        freq_response = self.xp.fft.fft(folded)

        if full_spectrum:
            return freq_response

        return self.xp.imag(freq_response)

    def frequency_response(self) -> Any:
        """Compute one-sided minimum-phase frequency response H(f)."""
        # 1. Log Amplitude Response
        # Handle log(0) and clamp
        # log_psd = log(psd)
        # We avoid 'errstate' context managers as they are numpy-specific.
        # Instead we use safe masking.

        # Create a safe copy for log (replace <=0 with epsilon temporarily to avoid NaN)
        safe_psd = self.xp.where(self.psd <= 0, 1e-100, self.psd)
        log_psd = self.xp.log(safe_psd)

        # Now clamp logic
        # If original was <=0, log is -inf -> clamp_min
        log_psd = self.xp.where(
            log_psd < self.clamp_log_min, self.clamp_log_min, log_psd
        )
        log_psd = self.xp.where(
            log_psd > self.clamp_log_max, self.clamp_log_max, log_psd
        )

        # log |H| = -0.5 * log(PSD)
        log_amp_res = -0.5 * log_psd

        # 2. Construct Two-Sided Log Spectrum
        # [0, 1, ... mid] are given by log_amp_res
        # [mid+1 ... end] are reflections of [1 ... mid-1]
        # (Assuming real signal => even magnitude spectrum)

        # log_amp_res: [DC, 1, ... mid]
        # we want reflected: [mid-1, ... 1]

        # Slicing: 1 to -1 (exclude DC and Nyquist from the reflection source)
        # Then flip it.
        inner = log_amp_res[..., 1:-1]
        reflected = self.xp.flip(inner, axis=-1)

        log_amp_res_full = self.xp.concat([log_amp_res, reflected], axis=-1)

        # 3. Compute log frequency response (DHT)
        log_freq_res = self._dht_folded_cepstrum(
            data=log_amp_res_full, full_spectrum=True
        )

        # 4. Exponentiate
        freq_res = self.xp.exp(log_freq_res)

        # Return one-sided
        return freq_res[..., : self.n_fft // 2 + 1]
