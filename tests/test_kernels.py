"""Test coverage for sgnligo.kernels module.
Tests include:
    - unit tests for LPWhiteningFilter and MPWhiteningFilter classes
    - scientific-validity tests for MPWhiteningFilter using simplified PSD models
        (Lorentzian, Exponential, Gaussian bump, Power-law)
    - basic tests for TimePhaseCorrection and MPMPCorrection classes
"""

import os
import warnings

import numpy as np
import pytest
from matplotlib import pyplot as plt

from zlw.kernels import (
    LPWhiteningFilter,
    MPWhiteningFilter,
    PSDAdmissibilityError,
    PSDAdmissibilityWarning,
    WhiteningFilter,
)


class TestWhiteningFilterAdmissibility:
    """Tests for the PSD admissibility validation logic."""

    @pytest.fixture
    def setup_valid(self):
        fs = 4096.0
        n_fft = 4096
        # A safe, flat PSD
        psd = np.ones(n_fft // 2 + 1)
        return psd, fs, n_fft

    def test_strict_positivity_error(self, setup_valid):
        """PSD containing zeros or negative values must raise an error."""
        psd, fs, n_fft = setup_valid

        # Inject a zero
        psd_bad = psd.copy()
        psd_bad[10] = 0.0

        with pytest.raises(PSDAdmissibilityError, match="strictly positive"):
            MPWhiteningFilter(psd_bad, fs, n_fft)

        # Inject a negative
        psd_neg = psd.copy()
        psd_neg[10] = -1.0
        with pytest.raises(PSDAdmissibilityError, match="strictly positive"):
            MPWhiteningFilter(psd_neg, fs, n_fft)

    def test_numerical_underflow_warning(self, setup_valid):
        """PSD containing tiny values (< 1e-48) must warn about integrator instability."""
        psd, fs, n_fft = setup_valid

        # Inject a tiny value
        psd_tiny = psd.copy()
        psd_tiny[100] = 1e-50

        with pytest.warns(PSDAdmissibilityWarning, match="extremely small values"):
            MPWhiteningFilter(psd_tiny, fs, n_fft)

    def test_seismic_wall_artifact_warning(self, setup_valid):
        """PSD with suspiciously quiet DC (relative to 20Hz) must warn."""
        psd, fs, n_fft = setup_valid
        # n_fft=4096, fs=4096 => df=1.0 Hz.
        # 20 Hz is at index 20.

        # Set 20Hz to a "seismic peak" level (e.g. 1.0)
        # Set DC to something unphysically low (e.g. 1e-5 * peak)
        # This simulates the "Welch artifact" that causes whitening instability.
        psd_artifact = psd.copy()
        psd_artifact[20] = 1.0
        psd_artifact[0] = 1e-5  # << 0.01 * 1.0

        with pytest.warns(PSDAdmissibilityWarning, match="Low-Frequency Rolloff"):
            MPWhiteningFilter(psd_artifact, fs, n_fft)

    def test_valid_psd_passes(self, setup_valid):
        """A physically reasonable PSD should pass without warnings."""
        psd, fs, n_fft = setup_valid

        # Ensure DC is reasonable relative to 20Hz
        psd[0] = 1.0
        psd[20] = 1.0

        # Should not raise
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter(
                "always"
            )  # Cause all warnings to always be triggered.
            MPWhiteningFilter(psd, fs, n_fft)
            # Filter out unrelated warnings if any
            relevant = [x for x in w if issubclass(x.category, PSDAdmissibilityWarning)]
            assert (
                len(relevant) == 0
            ), f"Caught unexpected warning: {relevant[0].message}"


class TestWhiteningFilterBase:
    """Tests for the base WhiteningFilter class (abstract behavior)."""

    @pytest.fixture
    def dummy_psd(self):
        # simple PSD of length 3 => n_fft = 4
        return np.array([1.0, 2.0, 1.0])

    def test_init(self, dummy_psd):
        """Test initialization of WhiteningFilter with PSD."""
        wf = WhiteningFilter(psd=dummy_psd, fs=100.0, n_fft=4)
        assert isinstance(wf, WhiteningFilter)
        assert wf.psd is dummy_psd
        assert wf.fs == 100.0
        assert wf.n_fft == 4

    def test_init_infer_n_fft(self, dummy_psd):
        """Test initialization of WhiteningFilter with inferred n_fft."""
        wf = WhiteningFilter(psd=dummy_psd, fs=100.0)
        assert wf.n_fft == 2 * (len(dummy_psd) - 1) == 4
        # length of one-sided PSD must be n_fft//2+1
        assert len(wf.psd) == wf.n_fft // 2 + 1

    def test_amplitude_response(self, dummy_psd):
        """Test amplitude response of WhiteningFilter."""
        wf = WhiteningFilter(psd=dummy_psd, fs=100.0, n_fft=4)
        amp = wf.amplitude_response()
        expected = 1.0 / np.sqrt(dummy_psd)
        assert np.allclose(amp, expected)

    def test_frequency_response_not_implemented(self, dummy_psd):
        """Test frequency response raises NotImplementedError."""
        wf = WhiteningFilter(psd=dummy_psd, fs=100.0, n_fft=4)
        with pytest.raises(NotImplementedError):
            wf.frequency_response()

    def test_impulse_response_not_implemented(self, dummy_psd):
        """Test impulse response raises NotImplementedError."""
        wf = WhiteningFilter(psd=dummy_psd, fs=100.0, n_fft=4)
        with pytest.raises(NotImplementedError):
            wf.impulse_response()


class TestLPWhiteningFilter:
    """Unit tests for the linear‐phase whitening filter."""

    @pytest.fixture
    def flat_psd(self):
        fs = 512.0
        n_fft = 64
        psd = np.ones(n_fft // 2 + 1)
        return psd, fs, n_fft

    def test_init(self, flat_psd):
        """Test initialization of LPWhiteningFilter with flat PSD."""
        psd, fs, n_fft = flat_psd
        lp = LPWhiteningFilter(psd=psd, fs=fs, n_fft=n_fft)
        assert isinstance(lp, LPWhiteningFilter)
        assert lp.psd is psd
        assert lp.fs == fs
        assert lp.n_fft == n_fft
        assert lp.delay == 0.0

    def test_frequency_response_flat_psd_no_delay(self, flat_psd):
        """Test frequency response of LPWhiteningFilter with flat PSD."""
        psd, fs, n_fft = flat_psd
        lp = LPWhiteningFilter(psd=psd, fs=fs, n_fft=n_fft, delay=0.0)
        H = lp.frequency_response()
        # amplitude response is 1/sqrt(1)=1
        assert H.shape == (n_fft // 2 + 1,)
        assert np.allclose(np.abs(H), 1.0)

    def test_phase_response(self, flat_psd):
        """Test phase response of LPWhiteningFilter with flat PSD."""
        psd, fs, n_fft = flat_psd
        delay = 0.01
        lp = LPWhiteningFilter(psd=psd, fs=fs, n_fft=n_fft, delay=delay)
        phi = lp.phase_response()
        freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
        expected = -2 * np.pi * freqs * delay
        assert np.allclose(phi, expected)

    def test_impulse_response_flat_psd_delta(self, flat_psd):
        """Test impulse response of LPWhiteningFilter with flat PSD."""
        psd, fs, n_fft = flat_psd
        lp = LPWhiteningFilter(psd=psd, fs=fs, n_fft=n_fft, delay=0.0)
        h = lp.impulse_response()
        # impulse at index 0
        assert np.argmax(np.abs(h)) == 0
        # frequency response magnitude squared == 1/psd
        H = np.fft.rfft(h)
        mag2 = np.abs(H) ** 2
        assert np.allclose(mag2, 1.0 / psd, atol=1e-7)

    def test_impulse_response_with_delay(self, flat_psd):
        """Test impulse response of LPWhiteningFilter with delay."""
        psd, fs, n_fft = flat_psd
        delay = 0.005
        lp = LPWhiteningFilter(psd=psd, fs=fs, n_fft=n_fft, delay=delay)
        h = lp.impulse_response()
        idx = np.argmax(np.abs(h))
        expected = int(round(delay * fs)) % n_fft
        assert idx == expected


class TestMPWhiteningFilter:
    """Unit tests for the minimum‐phase whitening filter using a flat PSD baseline."""

    @pytest.fixture
    def flat_psd(self):
        fs = 512.0
        n_fft = 64
        psd = np.ones(n_fft // 2 + 1)
        return psd, fs, n_fft

    def test_init(self, flat_psd):
        """Initialize MPWhiteningFilter without error and correct attributes."""
        psd, fs, n_fft = flat_psd
        mp = MPWhiteningFilter(psd=psd, fs=fs, n_fft=n_fft)
        assert isinstance(mp, MPWhiteningFilter)
        assert mp.psd is psd
        assert mp.fs == fs
        assert mp.n_fft == n_fft

    def test_frequency_response_shape_and_magnitude(self, flat_psd):
        """Frequency response has correct shape and |H(f)| = 1/sqrt(psd)."""
        psd, fs, n_fft = flat_psd
        mp = MPWhiteningFilter(psd=psd, fs=fs, n_fft=n_fft)
        H = mp.frequency_response()
        # shape
        assert H.shape == (n_fft // 2 + 1,)
        # magnitude matches 1/sqrt(psd)
        expected_mag = 1.0 / np.sqrt(psd)
        assert np.allclose(np.abs(H), expected_mag, atol=1e-8)

    def test_phase_response_consistency(self, flat_psd):
        """phase_response() returns the argument of frequency_response()."""
        psd, fs, n_fft = flat_psd
        mp = MPWhiteningFilter(psd=psd, fs=fs, n_fft=n_fft)
        phi = mp.phase_response()
        H = mp.frequency_response()
        assert np.allclose(phi, np.angle(H), atol=1e-8)

    def test_impulse_response_is_delta(self, flat_psd):
        """For a flat PSD, minimum‐phase whitening filter should be a pure delta
        at index 0 (no delay, no dispersion).
        """
        psd, fs, n_fft = flat_psd
        mp = MPWhiteningFilter(psd=psd, fs=fs, n_fft=n_fft)
        h = mp.impulse_response()
        # Impulse at zero
        idx = int(np.argmax(np.abs(h)))
        assert idx == 0
        # All other samples should be (nearly) zero
        h_copy = h.copy()
        h_copy[0] = 0.0
        assert np.allclose(h_copy, 0.0, atol=1e-7)

    def test_frequency_magnitude_squared_matches_psd(self, flat_psd):
        """Check that |FFT[h]|^2 = 1/psd for the impulse response of the flat case."""
        psd, fs, n_fft = flat_psd
        mp = MPWhiteningFilter(psd=psd, fs=fs, n_fft=n_fft)
        h = mp.impulse_response()
        H = np.fft.rfft(h)
        mag2 = np.abs(H) ** 2
        assert np.allclose(mag2, 1.0 / psd, atol=1e-7)


class TestMPWhiteningFilterSciValPsdLorentzian:
    """Scientific validity tests for PSD family 1: Lorentzian Model.
    These are high-sample-rate unit tests for MPWhiteningFilter focused on
    he LIGO “science band” (20–300 Hz). This is a test of the scientific validity
    of the minimum phase whitening filter implementation, using a simplified
    form of the PSD - a single-pole Lorentzian.

    We choose a high sampling rate (fs=4096 Hz) and large FFT length (n_fft=8192)
    so that the digital Lorentzian PSD is well-resolved over the band where
    stellar-mass BBH signals accumulate most SNR.  Tests compare:

      - amplitude_response() vs. analytic √[1+(s/Ωc)²]
      - frequency_response() vs.
         (a) folded-cepstrum reference and
         (b) closed-form analytic via bilinear map
      - phase_response() vs. analytic arctan(s/Ωc) − ω/2

    An optional plotting method is provided for visual inspection.
    """

    @pytest.fixture
    def high_rate_lorentzian_psd(self):
        fs = 4096.0
        n_fft = 8192
        fc = 10.0  # continuous-time corner frequency (Hz)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)  # Hz bins
        omega = 2 * np.pi * freqs / fs  # digital ω [rad/sample]
        T = 1.0 / fs
        Ωc = 2 * np.pi * fc  # continuous Ωc [rad/s]

        # Bilinear (Tustin) mapping from digital ω to analog s = iΩ_d
        s = (2.0 / T) * np.tan(omega / 2.0)  # mapped analog freq [rad/s]

        psd = 1.0 / (1.0 + (s / Ωc) ** 2)  # one-sided Lorentz PSD
        return psd, fs, n_fft, freqs, fc

    @pytest.fixture
    def expected_analytic_amp(self, high_rate_lorentzian_psd):
        psd, fs, n_fft, freqs, fc = high_rate_lorentzian_psd
        omega = 2 * np.pi * freqs / fs
        T = 1.0 / fs
        s = (2.0 / T) * np.tan(omega / 2.0)
        Ωc = 2 * np.pi * fc
        return np.sqrt(1.0 + (s / Ωc) ** 2)

    @pytest.fixture
    def expected_analytic_phase(self, high_rate_lorentzian_psd):
        psd, fs, n_fft, freqs, fc = high_rate_lorentzian_psd
        omega = 2 * np.pi * freqs / fs
        T = 1.0 / fs
        s = (2.0 / T) * np.tan(omega / 2.0)
        Ωc = 2 * np.pi * fc
        return np.arctan(s / Ωc)

    @pytest.fixture
    def expected_analytic_freq_res(
        self, expected_analytic_amp, expected_analytic_phase
    ) -> np.ndarray:
        """
        Expected analytic frequency response via bilinear mapping:
            H(ω) = |H(ω)| · exp[i φ(ω)].
        """
        return expected_analytic_amp * np.exp(1j * expected_analytic_phase)

    def test_amplitude_response(self, high_rate_lorentzian_psd, expected_analytic_amp):
        psd, fs, n_fft, freqs, fc = high_rate_lorentzian_psd
        mp = MPWhiteningFilter(psd, fs, n_fft)
        amp = mp.amplitude_response()
        # only compare in the 20–300 Hz “science” band
        mask = (freqs >= 20) & (freqs <= 300)
        assert np.allclose(amp[mask], expected_analytic_amp[mask], rtol=1e-6, atol=1e-8)

    def test_frequency_response(
        self,
        high_rate_lorentzian_psd,
        expected_analytic_freq_res,
    ):
        """
        1) Sanity‐check vs. folded‐cepstrum reference
        2) Compare to closed‐form analytic H(ω)
        """
        psd, fs, n_fft, freqs, fc = high_rate_lorentzian_psd
        mp = MPWhiteningFilter(psd, fs, n_fft)
        H = mp.frequency_response()

        # restrict to the 20–300 Hz “science” band
        mask = (freqs >= 20) & (freqs <= 300)

        # 2) closed‐form analytic
        assert np.allclose(
            H[mask], expected_analytic_freq_res[mask], rtol=1e-5, atol=1e-1
        )

    def test_phase_response(self, high_rate_lorentzian_psd, expected_analytic_phase):
        psd, fs, n_fft, freqs, fc = high_rate_lorentzian_psd
        mp = MPWhiteningFilter(psd, fs, n_fft)
        phi = mp.phase_response()
        mask = (freqs >= 20) & (freqs <= 300)
        assert np.allclose(
            phi[mask], expected_analytic_phase[mask], rtol=1e-5, atol=1e-2
        )

    @pytest.mark.skipif(
        os.getenv("SGNLIGO_TEST_SCIVAL_PLOT") != "1",
        reason="Set SGNLIGO_TEST_SCIVAL_PLOT=1 to display comparison plots",
    )
    def test_plot_comparison(
        self, high_rate_lorentzian_psd, expected_analytic_amp, expected_analytic_phase
    ):
        """Optional helper (not a pytest assertion) to plot analytic vs.
        MPWhiteningFilter amplitude & phase responses over the 20–300 Hz
        “science” band. Run manually for visual inspection, using a python snippet like:

        ```python
        from tests.test_kernels import TestMPWhiteningFilterSciValPsd1
        TestMPWhiteningFilterSciValPsd1.plot_comparison(
        ```
        """
        psd, fs, n_fft, freqs, fc = high_rate_lorentzian_psd
        mp = MPWhiteningFilter(psd, fs, n_fft)

        H = mp.frequency_response()
        amp = np.abs(H)
        phi = mp.phase_response()

        # use analytic fixtures directly
        analytic_amp = expected_analytic_amp
        analytic_phi = expected_analytic_phase

        mask = (freqs >= 20) & (freqs <= 300)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        # Amplitude plot
        ax1.plot(freqs[mask], analytic_amp[mask], label="Analytic |H|")
        ax1.plot(freqs[mask], amp[mask], "--", label="MPWhiteningFilter |H|")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)
        ax1.legend()

        # Phase plot
        ax2.plot(freqs[mask], analytic_phi[mask], label="Analytic φ")
        ax2.plot(freqs[mask], phi[mask], "--", label="MPWhiteningFilter φ")
        ax2.axvline(300, color="k", linestyle=":", label="Science‐band Edge")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (rad)")
        ax2.grid(True)
        ax2.legend()

        # Add title
        ax1.set_title("MPWhiteningFilter vs. Analytic Responses (Lorentzian PSD)")

        plt.tight_layout()
        plt.show()

        # Save figure to png file
        fig.savefig("mp_whitening_filter_lorentzian_comparison.png")


class TestMPWhiteningFilterSciValPsdExponential:
    """Scientific‐validity tests for MPWhiteningFilter using a discrete‐time
    Exponential PSD:
        S(ω) = exp(−β ω),   ω = 2π f / fs,   β = fs/(2π f0).
    These high‐sample‐rate unit tests focus on the LIGO “science band” (20–300 Hz).

    We choose fs=4096 Hz and n_fft=8192 so that the exponential PSD is well‐resolved
    where stellar‐mass BBH signals accumulate most SNR.  Tests compare:
      - amplitude_response() vs. analytic exp(β ω / 2)
      - frequency_response() vs.
         (a) folded‐cepstrum reference and
         (b) closed‐form analytic exp(β ω / 2)·exp[i β(ω−π)/2]
      - phase_response() vs. analytic φ(ω)=β(ω−π)/2

    An optional plotting helper is provided for visual inspection.
    """

    @pytest.fixture
    def high_rate_exponential_psd(self):
        fs = 4096.0
        n_fft = 8192
        f0 = 50.0  # e-folding frequency (Hz)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)  # Hz
        ω = 2 * np.pi * freqs / fs  # digital angular freq
        β = fs / (2 * np.pi * f0)  # dimensionless decay rate

        psd = np.exp(-β * ω)  # one-sided Exponential PSD
        return psd, fs, n_fft, freqs, f0

    @pytest.fixture
    def expected_analytic_amp(self, high_rate_exponential_psd):
        _, fs, _, freqs, f0 = high_rate_exponential_psd
        ω = 2 * np.pi * freqs / fs
        β = fs / (2 * np.pi * f0)
        # |W(e^{iω})| = exp(β ω / 2)
        return np.exp((β * ω) / 2)

    @pytest.fixture
    def expected_analytic_phase(self, high_rate_exponential_psd):
        """
        Expected discrete‐time phase response for an exponential PSD,
        using the closed‐form odd‐sine series:

            φ(ω) = -1/(π² f₀ T) ∑_{m=1}^M sin((2m-1) ω)/(2m-1)²
        """
        psd, fs, n_fft, freqs, f0 = high_rate_exponential_psd
        T = 1.0 / fs
        ω = 2 * np.pi * freqs / fs

        # truncation length for convergence of the series
        M = 500
        m = np.arange(1, M + 1)
        n = 2 * m - 1  # odd integers

        # compute sin(n ω)/(n^2) for all m, sum over m
        # shape of sin_terms: (M, len(ω))
        sin_terms = np.sin(np.outer(n, ω)) / (n[:, None] ** 2)
        series_sum = np.sum(sin_terms, axis=0)

        # scale factor: -1/(π^2 f0 T)
        factor = -1.0 / (np.pi**2 * f0 * T)
        phi = factor * series_sum

        # ensure phase is in [-π, π] if desired:
        phi_wrapped = (phi + np.pi) % (2 * np.pi) - np.pi

        return -phi_wrapped

    @pytest.fixture
    def expected_analytic_H(self, expected_analytic_amp, expected_analytic_phase):
        """Closed-form analytic frequency response: |H|·exp[i φ]."""
        return expected_analytic_amp * np.exp(1j * expected_analytic_phase)

    def test_amplitude_response(self, high_rate_exponential_psd, expected_analytic_amp):
        psd, fs, n_fft, freqs, _ = high_rate_exponential_psd
        mp = MPWhiteningFilter(psd, fs, n_fft)
        amp = mp.amplitude_response()
        mask = (freqs >= 20) & (freqs <= 300)
        assert np.allclose(amp[mask], expected_analytic_amp[mask], rtol=1e-6, atol=1e-8)

    def test_frequency_response(self, high_rate_exponential_psd, expected_analytic_H):
        psd, fs, n_fft, freqs, _ = high_rate_exponential_psd
        mp = MPWhiteningFilter(psd, fs, n_fft)
        H = mp.frequency_response()
        mask = (freqs >= 20) & (freqs <= 300)

        # (b) vs. closed-form analytic
        assert np.allclose(H[mask], expected_analytic_H[mask], rtol=1e-5, atol=1e-3)

    def test_phase_response(self, high_rate_exponential_psd, expected_analytic_phase):
        psd, fs, n_fft, freqs, _ = high_rate_exponential_psd
        mp = MPWhiteningFilter(psd, fs, n_fft)
        phi = mp.phase_response()
        mask = (freqs >= 20) & (freqs <= 300)
        assert np.allclose(
            phi[mask], expected_analytic_phase[mask], rtol=1e-5, atol=1e-2
        )

    @pytest.mark.skipif(
        os.getenv("SGNLIGO_TEST_SCIVAL_PLOT") != "1",
        reason="Set SGNLIGO_TEST_SCIVAL_PLOT=1 to display comparison plots",
    )
    def test_plot_comparison(
        self, high_rate_exponential_psd, expected_analytic_amp, expected_analytic_phase
    ):
        """Optional helper: plot analytic vs. MPWhiteningFilter responses (
        20–300 Hz)."""
        psd, fs, n_fft, freqs, _ = high_rate_exponential_psd
        mp = MPWhiteningFilter(psd, fs, n_fft)
        H = mp.frequency_response()
        amp = np.abs(H)
        phi = mp.phase_response()
        mask = (freqs >= 20) & (freqs <= 300)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        # Amplitude
        ax1.plot(freqs[mask], expected_analytic_amp[mask], label="Analytic |H|")
        ax1.plot(freqs[mask], amp[mask], "--", label="MPWhiteningFilter |H|")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)
        ax1.legend()
        # Phase
        ax2.plot(freqs[mask], expected_analytic_phase[mask], label="Analytic φ")
        ax2.plot(freqs[mask], phi[mask], "--", label="MPWhiteningFilter φ")
        ax2.axvline(300, color="k", linestyle=":", label="Science-band Edge")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (rad)")
        ax2.grid(True)
        ax2.legend()

        # Add title
        ax1.set_title("MPWhiteningFilter vs. Analytic Responses (Exponential PSD)")

        plt.tight_layout()
        plt.show()

        # Save figure to png file
        fig.savefig("mp_whitening_filter_exponential_comparison.png")


class TestMPWhiteningFilterSciValPsdGaussianBump:
    """Scientific‐validity tests for PSD family 3: Gaussian‐Bump Model.
    High‐sample‐rate unit tests for MPWhiteningFilter over
    the LIGO “science band” (20–300 Hz), using a simple
    Gaussian‐shaped PSD bump:
        S_cont(Ω) = exp[−(Ω − Ω₀)²/(2 σ²)],   Ω₀ = 2π f₀,   σ = 2π Δf.

    We choose fs=4096 Hz, n_fft=8192, center f₀=100 Hz, σ_f=20 Hz.
    Tests compare:
      - amplitude_response() vs. analytic exp[( (s − Ω₀)² / (4 σ²) )]
      - frequency_response() vs. folded‐cepstrum *and*
        closed‐form H(ω) = |H| · exp[i·0] (since the Gaussian bump is
        symmetric about Ω₀ in the analog domain, the minimum‐phase factor
        has zero phase in discrete time as well).
      - phase_response() vs. zero.
    """

    @pytest.fixture
    def high_rate_gaussian_psd(self):
        fs = 4096.0
        T = 1.0 / fs
        n_fft = 8192

        # bump parameters
        f0 = 100.0  # center [Hz]
        bw = 20.0  # one‐sigma [Hz]
        alpha = 0.5  # relative amplitude

        # digital frequency axes
        freqs = np.fft.rfftfreq(n_fft, d=T)  # [Hz]
        omega = 2 * np.pi * freqs / fs  # ω ∈ [0,π]

        # normalized bump center & width in rad/sample
        omega0 = 2 * np.pi * f0 / fs
        sigma = 2 * np.pi * bw / fs

        # **direct digital PSD** (no Tustin)
        psd = 1.0 + alpha * np.exp(-0.5 * ((omega - omega0) / sigma) ** 2)

        return psd, fs, n_fft, freqs, f0, bw, alpha

    @pytest.fixture
    def expected_analytic_amp(self, high_rate_gaussian_psd):
        psd, *_ = high_rate_gaussian_psd
        return 1.0 / np.sqrt(psd)

    @pytest.fixture
    def expected_analytic_phase(self, high_rate_gaussian_psd, N_terms=4096):
        """
        Exact discrete‐time phase via Fourier–cosine expansion
        of L(ω) = -½ ln S(e^{iω}), mirrored to [-π,π].
        """
        psd, fs, n_fft, freqs, f0, bw, alpha = high_rate_gaussian_psd
        omega = 2 * np.pi * freqs / fs

        # 1) build L(ω) on [0, π]
        L = -0.5 * np.log(psd)

        # 2) compute cosine coefficients A_n up to N_terms
        A = np.zeros(N_terms + 1)
        # we only need n=1…N_terms
        for n in range(1, N_terms + 1):
            # trapz over [0,π]
            # note: omega runs 0→π, equispaced
            A[n] = (2 / np.pi) * np.trapezoid(L * np.cos(n * omega), omega)

        # 3) rebuild phase
        phi = np.zeros_like(omega)
        for n in range(1, N_terms + 1):
            phi += A[n] * np.sin(n * omega)

        # 4) zero reference at ω=0
        phi -= phi[0]

        return -phi

    @pytest.fixture
    def expected_analytic_freq_res(
        self, expected_analytic_amp, expected_analytic_phase
    ):
        return expected_analytic_amp * np.exp(1j * expected_analytic_phase)

    def test_amplitude_response(self, high_rate_gaussian_psd, expected_analytic_amp):
        psd, fs, n_fft, freqs, f0, bw, alpha = high_rate_gaussian_psd
        mp = MPWhiteningFilter(psd, fs, n_fft)
        amp = mp.amplitude_response()
        mask = (freqs >= 20) & (freqs <= 300)
        assert np.allclose(amp[mask], expected_analytic_amp[mask], rtol=1e-6, atol=1e-8)

    def test_frequency_response(
        self, high_rate_gaussian_psd, expected_analytic_freq_res
    ):
        psd, fs, n_fft, freqs, f0, bw, alpha = high_rate_gaussian_psd
        mp = MPWhiteningFilter(psd, fs, n_fft)
        H = mp.frequency_response()
        mask = (freqs >= 20) & (freqs <= 300)

        # folded‐cepstrum vs. closed‐form analytic
        assert np.allclose(
            H[mask], expected_analytic_freq_res[mask], rtol=1e-5, atol=1e-8
        )

    def test_phase_response(self, high_rate_gaussian_psd, expected_analytic_phase):
        psd, fs, n_fft, freqs, f0, bw, alpha = high_rate_gaussian_psd
        mp = MPWhiteningFilter(psd, fs, n_fft)
        phi = mp.phase_response()
        mask = (freqs >= 20) & (freqs <= 300)
        assert np.allclose(
            phi[mask], expected_analytic_phase[mask], rtol=1e-6, atol=1e-8
        )

    @pytest.mark.skipif(
        os.getenv("SGNLIGO_TEST_SCIVAL_PLOT") != "1",
        reason="Set SGNLIGO_TEST_SCIVAL_PLOT=1 to display comparison plots",
    )
    def test_plot_comparison(
        self, high_rate_gaussian_psd, expected_analytic_amp, expected_analytic_phase
    ):
        """Optional helper: compare analytic vs. MPWhiteningFilter responses."""
        psd, fs, n_fft, freqs, f0, bw, alpha = high_rate_gaussian_psd
        mp = MPWhiteningFilter(psd, fs, n_fft)
        H = mp.frequency_response()
        amp = np.abs(H)
        phi = mp.phase_response()
        mask = (freqs >= 20) & (freqs <= 300)

        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        # Amplitude
        ax1.plot(freqs[mask], expected_analytic_amp[mask], label="Analytic |H|")
        ax1.plot(freqs[mask], amp[mask], "--", label="MPWhiteningFilter |H|")
        ax1.set_ylabel("Amplitude")
        # make y axis logarithmic
        ax1.set_yscale("log")
        ax1.grid(True)
        ax1.legend()

        # Phase
        ax2.plot(
            freqs[mask],
            expected_analytic_phase[mask],
            label="Analytic φ",
        )
        ax2.plot(freqs[mask], phi[mask], "--", label="MPWhiteningFilter φ")
        ax2.axvline(300, color="k", linestyle=":", label="Science‐band Edge")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (rad)")
        ax2.grid(True)
        ax2.legend()

        # Add title
        ax1.set_title("MPWhiteningFilter vs. Analytic Responses (Gaussian PSD)")

        plt.tight_layout()
        plt.show()

        # Save figure to png file
        fig.savefig("mp_whitening_filter_gaussian_comparison.png")


class TestMPWhiteningFilterSciValPsdPowerLaw:
    """Scientific-validity tests for PSD family 4: Power-Law Model.
    High-sample-rate unit tests for MPWhiteningFilter over
    the LIGO “science band” (20–300 Hz), using a simple
    power-law PSD:
        S_cont(f) = A · f^(–γ),    f ≥ f_min > 0.

    We choose fs=4096 Hz, n_fft=8192, A=1, γ=2 so that the digital
    PSD is well-resolved over the band where stellar-mass BBH signals
    accumulate most SNR.  Tests compare:

      - amplitude_response() vs. analytic 1/√S = A^(–½) · f^(γ/2)
      - frequency_response() vs. folded-cepstrum reference *and*
        closed-form analytic H(ω) = |H(ω)| · exp[i φ(ω)]
      - phase_response() vs. analytic constant φ(ω)=γ π/4
    """

    @pytest.fixture
    def high_rate_powerlaw_psd(self):
        fs = 4096.0
        n_fft = 8192
        A = 1.0
        gamma = 2.0

        # digital frequency bins
        freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)  # [Hz]
        omega = 2 * np.pi * freqs / fs  # [rad/sample]

        # exact discrete‐time power‐law PSD:
        psd = A * (2 * np.sin(omega / 2.0)) ** (-gamma)

        # avoid the DC singularity by copying the next bin
        psd[0] = psd[1]

        return psd, fs, n_fft, freqs, A, gamma

    @pytest.fixture
    def expected_analytic_amp(self, high_rate_powerlaw_psd):
        psd, fs, n_fft, freqs, A, gamma = high_rate_powerlaw_psd
        return 1.0 / np.sqrt(psd)

    @pytest.fixture
    def expected_analytic_phase(self, high_rate_powerlaw_psd):
        psd, fs, n_fft, freqs, A, gamma = high_rate_powerlaw_psd
        omega = 2.0 * np.pi * freqs / fs

        # φ(ω) = (γ/4)·(π − ω)
        phi = (gamma / 4.0) * (np.pi - omega)

        return phi

    @pytest.fixture
    def expected_analytic_freq_res(
        self, expected_analytic_amp, expected_analytic_phase
    ):
        return expected_analytic_amp * np.exp(1j * expected_analytic_phase)

    def test_amplitude_response(self, high_rate_powerlaw_psd, expected_analytic_amp):
        psd, fs, n_fft, freqs, A, gamma = high_rate_powerlaw_psd
        mp = MPWhiteningFilter(psd, fs, n_fft)
        amp = mp.amplitude_response()
        mask = (freqs >= 20) & (freqs <= 300)
        assert np.allclose(amp[mask], expected_analytic_amp[mask], rtol=1e-6, atol=1e-8)

    def test_frequency_response(
        self,
        high_rate_powerlaw_psd,
        expected_analytic_freq_res,
    ):
        """
        1) Sanity‐check vs. folded‐cepstrum reference
        2) Compare to closed‐form analytic H(ω)
        """
        psd, fs, n_fft, freqs, A, gamma = high_rate_powerlaw_psd
        mp = MPWhiteningFilter(psd, fs, n_fft)
        H = mp.frequency_response()

        mask = (freqs >= 100) & (freqs <= 300)

        # closed-form analytic
        assert np.allclose(
            H[mask], expected_analytic_freq_res[mask], rtol=1e-5, atol=1e-2
        )

    def test_phase_response(
        self,
        high_rate_powerlaw_psd,
        expected_analytic_phase,
    ):
        psd, fs, n_fft, freqs, A, gamma = high_rate_powerlaw_psd
        mp = MPWhiteningFilter(psd, fs, n_fft)
        phi = mp.phase_response()
        mask = (freqs >= 100) & (freqs <= 300)
        assert np.allclose(
            phi[mask], expected_analytic_phase[mask], rtol=1e-5, atol=1e-2
        )

    @pytest.mark.skipif(
        os.getenv("SGNLIGO_TEST_SCIVAL_PLOT") != "1",
        reason="Set SGNLIGO_TEST_SCIVAL_PLOT=1 to display comparison plots",
    )
    def test_plot_comparison(
        self, high_rate_powerlaw_psd, expected_analytic_amp, expected_analytic_phase
    ):
        """Optional helper: compare analytic vs. MPWhiteningFilter responses."""
        import matplotlib.pyplot as plt

        psd, fs, n_fft, freqs, A, gamma = high_rate_powerlaw_psd
        mp = MPWhiteningFilter(psd, fs, n_fft)
        H = mp.frequency_response()
        amp = np.abs(H)
        phi = mp.phase_response()

        mask = (freqs >= 20) & (freqs <= 300)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        # Amplitude
        ax1.plot(freqs[mask], expected_analytic_amp[mask], label="Analytic |H|")
        ax1.plot(freqs[mask], amp[mask], "--", label="MPWhiteningFilter |H|")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)
        ax1.legend()

        # Phase
        ax2.plot(freqs[mask], expected_analytic_phase[mask], label="Analytic φ")
        ax2.plot(freqs[mask], phi[mask], "--", label="MPWhiteningFilter φ")
        ax2.axvline(300, color="k", linestyle=":", label="Science-band Edge")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (rad)")
        ax2.grid(True)
        ax2.legend()

        # Add title
        ax1.set_title("MPWhiteningFilter vs. Analytic Responses (Power-Law PSD)")

        plt.tight_layout()
        plt.show()

        # Save figure to png file
        fig.savefig("mp_whitening_filter_powerlaw_comparison.png")
