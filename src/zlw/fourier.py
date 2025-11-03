"""
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy
from numpy.fft import fft, ifft, irfft, rfft, rfftfreq


@dataclass
class FourierBackend:
    """Base class for a fourier backend. This class provides methods for
    performing FFT and IFFT operations, which can be overridden by subclasses
    """

    def rfft(self, x: numpy.ndarray) -> numpy.ndarray:
        """Compute the real FFT of the input array."""
        raise NotImplementedError

    def irfft(self, X: numpy.ndarray, n: Optional[int] = None) -> numpy.ndarray:
        """Compute the inverse real FFT of the input array."""
        raise NotImplementedError

    def fft(self, x: numpy.ndarray) -> numpy.ndarray:
        """Compute the complex FFT of the input array."""
        raise NotImplementedError

    def ifft(self, X: numpy.ndarray, n: Optional[int] = None) -> numpy.ndarray:
        """Compute the inverse complex FFT of the input array."""
        raise NotImplementedError

    def rfftfreq(self, n: int, d: float = 1.0, device=None):
        """Compute the Discrete Fourier Transform sample frequencies"""
        raise NotImplementedError


@dataclass
class NumpyFourierBackend(FourierBackend):
    """Numpy implementation of the FourierBackend using numpy's FFT functions."""

    def rfft(self, x: numpy.ndarray) -> numpy.ndarray:
        return rfft(x)

    def irfft(self, X: numpy.ndarray, n: Optional[int] = None) -> numpy.ndarray:
        return irfft(X, n=n)

    def fft(self, x: numpy.ndarray) -> numpy.ndarray:
        return fft(x)

    def ifft(self, X: numpy.ndarray, n: Optional[int] = None) -> numpy.ndarray:
        return ifft(X, n=n)

    def rfftfreq(self, n: int, d: float=1.0, device=None):
        return rfftfreq(n, d=d, device=device)
