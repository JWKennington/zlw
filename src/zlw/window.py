"""
Time-domain windowing utilities for FIR filter design.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal.windows import hann as _hann_window
from scipy.signal.windows import tukey as _tukey_window


@dataclass
class WindowSpec(ABC):
    """Abstract base class for window specifications.

    Provides the machinery to generate time-shifted (asymmetric) window slices
    from a symmetric prototype. This allows producing "Half-Windows" for
    causal filters (center=0) or delayed windows for linear-phase filters.
    """

    def make(self, length: int, center: Optional[float] = None) -> np.ndarray:
        """Construct the window array of given length.

        If `center` is provided, the window is constructed such that its peak (1.0)
        is located at `center`. This allows for constructing "Half-Windows" for
        causal filters (center=0) or arbitrary asymmetric windows for delayed filters.

        Args:
            length: Number of samples in the window.
            center: Index of the window peak (maximum).
                    If None, defaults to (length - 1) / 2 (symmetric).

        Returns:
            np.ndarray: Window array of shape (length,).
        """
        # 1. Determine parameters
        if center is None:
            c = (length - 1) / 2.0
        else:
            c = float(center)

        # 2. Determine extent of the "virtual" symmetric window required
        #    to cover the requested range [0, length-1] with peak at c.
        #    We visualize a symmetric window centered at 'c'.
        dist_left = c  # distance from peak to 0
        dist_right = length - 1 - c  # distance from peak to end
        max_dist = max(dist_left, dist_right)

        # Virtual length must be at least 2*max_dist + 1 (odd) to be symmetric
        # and strictly contain the peak at a precise index.
        # We use ceil to ensure integer radius.
        reach = int(np.ceil(max_dist))
        L_virtual = 2 * reach + 1

        # In the virtual window, the peak is exactly at index 'reach'.
        peak_idx_virtual = reach

        # 3. Generate the base symmetric window
        w_base = self._generate_symmetric(L_virtual)

        # 4. Extract the relevant slice
        #    We map output index 'i' to virtual index 'k' such that:
        #    k - peak_idx_virtual = i - c
        #    => k = peak_idx_virtual - c + i

        start_idx = int(np.round(peak_idx_virtual - c))
        end_idx = start_idx + length

        # Handle edge cases (rare rounding issues)
        pad_pre = max(0, -start_idx)
        pad_post = max(0, end_idx - L_virtual)

        # Clamp slice indices
        s_start = max(0, start_idx)
        s_end = min(L_virtual, end_idx)

        slice_vals = w_base[s_start:s_end]

        if pad_pre > 0 or pad_post > 0:
            return np.pad(slice_vals, (pad_pre, pad_post), constant_values=0.0)

        return slice_vals

    @abstractmethod
    def _generate_symmetric(self, length: int) -> np.ndarray:
        """Generate a standard symmetric window of the given length."""
        pass


@dataclass
class IdentityWindow(WindowSpec):
    """No windowing (ones)."""

    def _generate_symmetric(self, length: int) -> np.ndarray:
        return np.ones(length, dtype=float)


@dataclass
class Hann(WindowSpec):
    """Hann window."""

    def _generate_symmetric(self, length: int) -> np.ndarray:
        return _hann_window(length, sym=True).astype(float, copy=False)


@dataclass
class Tukey(WindowSpec):
    """Tukey window with configurable alpha."""

    alpha: float = 0.5

    def _generate_symmetric(self, length: int) -> np.ndarray:
        a = float(self.alpha)
        a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
        return _tukey_window(length, alpha=a, sym=True).astype(float, copy=False)


@dataclass
class Epanechnikov(WindowSpec):
    """Epanechnikov window (parabolic).

    w(n) = 1 - (x)^2 for x in [-1, 1].
    """

    def _generate_symmetric(self, length: int) -> np.ndarray:
        if length <= 1:
            return np.ones(length, dtype=float)

        # Define support [-1, 1] over the window length
        # Peak at center.
        half_len = (length - 1) / 2.0
        n = np.arange(length, dtype=float)

        # Avoid division by zero
        if half_len == 0:
            return np.ones(length, dtype=float)

        x = (n - half_len) / half_len
        w = 1.0 - x**2

        # Clip to 0 (though analytic formula hits 0 at edges)
        w[w < 0] = 0.0
        return w
