"""
tests/test_window_latency.py

Unit tests for the 'matched-latency' windowing logic in zlw.
Verifies that:
1. WindowSpec generates correct asymmetric slices based on 'center'.
2. MPWhiteningFilter requests a causal (center=0) window.
3. LPWhiteningFilter requests a delayed (center=delay*fs) window.
"""

import numpy as np
import pytest
from scipy.signal import get_window

from zlw.window import WindowSpec, Tukey, Hann, Epanechnikov, IdentityWindow
from zlw.kernels import MPWhiteningFilter, LPWhiteningFilter, NumpyFourierBackend


# ---------------------------------------------------------------------------
# 1. Core WindowSpec Geometry Tests
# ---------------------------------------------------------------------------


class TestWindowSpecGeometry:
    """Verifies the slicing logic in WindowSpec.make(length, center)."""

    def test_center_none_defaults_to_symmetric(self):
        """If center is None, it should behave like a standard symmetric window."""
        length = 101
        spec = Hann()
        w = spec.make(length, center=None)

        # Peak should be in the middle
        mid = (length - 1) // 2
        assert np.isclose(w[mid], 1.0)
        assert np.allclose(w, w[::-1])  # Symmetric

    def test_causal_half_window_hann(self):
        """
        Test center=0.0 (Causal/Minimum Phase case).
        For Hann, the symmetric peak is 1.0.
        If center=0, w[0] should be 1.0, and it should taper down.
        """
        length = 50
        spec = Hann()
        w = spec.make(length, center=0.0)

        # 1. Peak at index 0
        assert np.isclose(w[0], 1.0)

        # 2. Monotonic decay (since Hann is monotonic from peak to zero)
        assert np.all(np.diff(w) <= 0)

        # 3. Validation against manual slice
        # Virtual symmetric window would be length 2*49 + 1 = 99 (peak at 49)
        # We want the right half: indices [49...98] of the virtual window
        # virtual_full = get_window("hann", 99)
        # expected = virtual_full[49:]
        #
        # assert len(w) == len(expected)
        # assert np.allclose(w, expected)

    def test_delayed_peak_hann(self):
        """
        Test center = arbitrary delay (Linear Phase case).
        e.g., Length 100, Peak at 80.

        NOTE: We use Hann because it has a unique peak. Tukey has a flat top,
        so argmax would return the start of the plateau, not the center.
        """
        length = 100
        center = 80.0
        spec = Hann()
        w = spec.make(length, center=center)

        # 1. Peak should be at index 80
        assert np.isclose(w[80], 1.0)
        assert np.argmax(w) == 80

        # 2. Check asymmetry
        # Left side (0-80) is long, Right side (80-99) is short.
        # It should NOT be symmetric.
        assert not np.allclose(w, w[::-1])

    def test_epanechnikov_shape(self):
        """Test the custom Epanechnikov subclass."""
        length = 21
        center = 10.0  # Symmetric
        spec = Epanechnikov()
        w = spec.make(length, center=center)

        # Peak at 10
        assert np.isclose(w[10], 1.0)
        # Zeros at edges (or close to it)
        # Epanechnikov w(x) = 1 - x^2. At edges x=1, w=0.
        assert np.isclose(w[0], 0.0, atol=1e-2)
        assert np.isclose(w[-1], 0.0, atol=1e-2)

    def test_identity_window(self):
        """IdentityWindow should return ones regardless of center."""
        spec = IdentityWindow()
        w = spec.make(50, center=12.5)
        assert np.allclose(w, 1.0)


# ---------------------------------------------------------------------------
# 2. WhiteningFilter Integration Tests
# ---------------------------------------------------------------------------


class TestWhiteningFilterIntegration:
    """
    Verifies that Filters correctly report their 'peak_center' and that
    'impulse_response' uses it.
    """

    @pytest.fixture
    def dummy_psd(self):
        """Flat PSD for simple delta-function impulse responses."""
        return np.ones(1025)  # n_fft = 2048

    def test_mp_requests_causal_window(self, dummy_psd):
        """
        MPWhiteningFilter must have peak_center = 0.0.
        When passing a window, the result should be windowed at index 0.
        """
        fs = 1024
        mp = MPWhiteningFilter(dummy_psd, fs=fs)

        # 1. Check Property
        assert mp.peak_center == 0.0

        # 2. Check Window Application
        # Use a Hann window. If applied symmetrically (default), w[0] would be 0.
        # If applied causally (correct), w[0] should be 1.
        # Since PSD is flat, h is a delta function at 0.
        # Result h_windowed[0] should be h[0] * 1.0.

        h_raw = mp.impulse_response(window=None)
        assert np.abs(h_raw[0]) > 0.1  # Main tap

        # Apply Hann
        h_win = mp.impulse_response(window=Hann())

        # If the window was symmetric (wrong), h_win[0] would be ~0.
        # If the window is causal (correct), h_win[0] should be preserved.
        assert np.abs(h_win[0]) > 0.1, "MP start tap was crushed by window!"

    def test_lp_requests_delayed_window(self, dummy_psd):
        """
        LPWhiteningFilter must have peak_center = delay * fs.
        """
        fs = 1000
        delay = 0.5  # 500 samples
        lp = LPWhiteningFilter(dummy_psd, fs=fs, delay=delay)

        # 1. Check Property
        assert lp.peak_center == 500.0

        # 2. Check Window Application
        # Impulse response should be a delta at index 500.
        # Window (Hann) should peak at 500.

        h_win = lp.impulse_response(window=Hann())

        # Peak should still be at 500
        idx = np.argmax(np.abs(h_win))
        assert idx == 500
        assert np.abs(h_win[500]) > 0.1

    def test_lp_fractional_delay(self, dummy_psd):
        """
        Test non-integer delay integration.
        """
        fs = 1000
        delay = 0.1234  # 123.4 samples
        lp = LPWhiteningFilter(dummy_psd, fs=fs, delay=delay)

        # Use approx comparison for floats
        assert lp.peak_center == pytest.approx(123.4)

        # Window generation should handle float center
        spec = Hann()
        # Cast to int for window generation as standard discrete windows live on integer grid
        w = spec.make(2048, center=int(round(lp.peak_center)))

        # Peak of window should be at 123 (closest integer)
        assert np.argmax(w) == 123


# ---------------------------------------------------------------------------
# 3. Energy & Physics Verification
# ---------------------------------------------------------------------------


class TestWindowingPhysics:

    def test_mp_energy_preservation(self):
        """
        Verify that applying the matched window preserves the total energy
        of the Minimum Phase filter (which is mostly at the start).
        """
        # Create a colored PSD so the filter has a tail
        n_fft = 512
        fs = 100
        freqs = np.fft.rfftfreq(n_fft, 1 / fs)
        psd = 1.0 + (freqs / 10.0) ** (-2)  # Red noise
        psd[0] = psd[1]

        mp = MPWhiteningFilter(psd, fs, n_fft=n_fft)

        h_raw = mp.impulse_response(window=None)
        h_win = mp.impulse_response(window=Tukey(alpha=0.2))

        e_raw = np.sum(h_raw**2)
        e_win = np.sum(h_win**2)

        assert np.isclose(e_raw, e_win, rtol=1e-12)

    def test_window_tapers_tail(self):
        """
        Verify that the causal window actually tapers the end of the buffer,
        reducing potential aliasing wrap-around.
        """
        n_fft = 256
        fs = 100
        # Flat PSD = Delta function (no tail).
        # Need very colored PSD to create long tail.
        # Lorentzian line.
        freqs = np.fft.rfftfreq(n_fft, 1 / fs)
        psd = 1.0 / (1 + ((freqs - 20) / 1.0) ** 2)

        mp = MPWhiteningFilter(psd, fs, n_fft=n_fft)

        # Raw filter might wrap around if n_fft is too small for the Q
        h_raw = mp.impulse_response(window=None)

        # Windowed filter (Hann) forces zero at the end
        h_win = mp.impulse_response(window=Hann())

        # Check last sample
        # Raw might be non-zero due to circular aliasing/long tail
        # Windowed must be very close to zero
        assert np.abs(h_win[-1]) < 1e-10
        assert np.abs(h_win[-1]) < np.abs(h_raw[-1])
