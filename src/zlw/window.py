"""
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.fft import fft, ifft, irfft, rfft, rfftfreq
from scipy.integrate import simpson
from scipy.signal.windows import hann as _hann_window
from scipy.signal.windows import tukey as _tukey_window

@dataclass
class WindowSpec:
    """Specification for an optional time-domain window to apply to FIR taps.

    Args:
        kind:
            One of {"tukey", "hann"}. Case-insensitive. Defaults to None (no window).
        alpha:
            For Tukey only, the shape parameter in [0, 1]. Fraction of the window
            inside the cosine tapered region (per SciPy definition). Ignored for Hann.
    """

    kind: Optional[str] = None
    alpha: float = 0.5

    def make(self, length: int) -> np.ndarray:
        """Construct the window array of given length.

        Returns a vector of ones if kind is None or unrecognized to preserve default
        behavior (no windowing).
        """
        if self.kind is None:
            return np.ones(length, dtype=float)
        k = self.kind.lower()
        if k == "tukey":
            a = float(self.alpha)
            a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
            return _tukey_window(length, alpha=a, sym=False).astype(float, copy=False)
        if k == "hann":
            return _hann_window(length, sym=False).astype(float, copy=False)
        # Fallback to identity window
        return np.ones(length, dtype=float)
