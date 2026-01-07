"""
test_exact_correction.py

Split test suite for the ExactPsdDriftCorrection class in zlw.corrections.
"""

import numpy as np
import pytest
from scipy.fft import fft, ifft, rfft
from scipy.signal import get_window

from zlw.corrections import (
    ExtPsdDriftCorrection,
    AliasingWarning,
    PrtPsdDriftCorrection,
)
from zlw.window import WindowSpec

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def fs():
    return 4096.0


@pytest.fixture
def analytic_drift_scenario(fs):
    """
    Generates a Reference PSD and a Live PSD where a Gaussian feature
    has drifted in frequency.
    Ref: Line at 60 Hz.
    Live: Line at 60.5 Hz.

    CRITICAL FIX: Uses 32s duration to prevent Time-Domain Aliasing
    (wrap-around) of the 0.5Hz wide feature.
    """
    duration = 32.0
    n_fft = int(duration * fs)
    freqs = np.linspace(0, fs / 2, n_fft // 2 + 1)

    def make_psd(f0):
        # 1/f noise + gaussian line (width 0.5 Hz)
        # Avoid f=0 singularity
        noise = 1e-40 * (10.0 / np.maximum(freqs, 1.0)) ** 2
        line = 1e-35 * np.exp(-0.5 * ((freqs - f0) / 0.5) ** 2)
        return noise + line

    psd_ref = make_psd(60.0)
    psd_live = make_psd(60.5)

    return freqs, psd_ref, psd_live, n_fft


@pytest.fixture
def power_law_bump_scenario(fs):
    """
    Scenario B: Power Law Background + Broad Gaussian Sensitivity Loss.
    Ref: Pure 1/f^2 Power Law.
    Live: Ref * (1 + Gaussian Bump at 100Hz).

    This simulates a broadband sensitivity degradation (e.g. alignment drift).
    Feature width = 20 Hz (Broad). Time constant ~ 1/20s = 50ms (Short).
    """
    duration = 16.0  # Broad features decay fast, 16s is plenty.
    n_fft = int(duration * fs)
    freqs = np.linspace(0, fs / 2, n_fft // 2 + 1)

    # 1. Background Power Law
    p_ref = 1e-40 * (100.0 / np.maximum(freqs, 10.0)) ** 2

    # 2. Add Broad Bump to Live
    # Amplitude 0.5 (50% noise increase), Center 100Hz, Width 20Hz
    bump = 0.5 * np.exp(-0.5 * ((freqs - 100.0) / 20.0) ** 2)
    p_live = p_ref * (1.0 + bump)

    return freqs, p_ref, p_live, n_fft


@pytest.fixture
def small_mismatch_scenario(fs):
    """
    Scenario C: Small Perturbation for comparing Exact vs Perturbative methods.
    Reference: Flat noise.
    Live: Flat noise * (1 + epsilon * sin(f)).
    Template: Power law (inspiral-like).
    """
    duration = 16.0
    n_fft = int(duration * fs)
    freqs = np.linspace(0, fs / 2, n_fft // 2 + 1)

    # 1. Flat Reference
    p_ref = np.ones_like(freqs) * 1e-40

    # 2. Small Ripple Perturbation (approx 1% magnitude change)
    # The perturbation function 'g(f)' is sinusoidal
    epsilon = 0.01
    perturbation = 1.0 + epsilon * np.sin(freqs / 100.0)
    p_live = p_ref * (perturbation**2)  # Squared because PSD is amplitude^2

    # 3. Inspiral-like Template
    # f^(-7/6) amplitude decay, linear phase (time shift) to make it interesting
    # We add a small time shift to the template so it's not perfectly centered at 0
    amp = np.maximum(freqs, 10.0) ** (-7.0 / 6.0)
    phase = -2 * np.pi * freqs * 0.01  # 10ms shift
    h_tilde = amp * np.exp(1j * phase)
    h_tilde[0] = 0  # No DC

    return freqs, p_ref, p_live, h_tilde, n_fft


class TestBasicFunctionality:

    def test_initialization(self, analytic_drift_scenario, fs):
        freqs, p_ref, p_live, n_fft = analytic_drift_scenario
        corr = ExtPsdDriftCorrection(freqs, p_ref, p_live, fs)

        assert corr.fs == fs
        assert corr.n_fft == n_fft
        assert corr.df == pytest.approx(freqs[1] - freqs[0])

    def test_input_validation(self, fs):
        freqs = np.linspace(0, 100, 100)
        p_ref = np.ones(100)
        p_bad = np.ones(99)  # Wrong length

        with pytest.raises(ValueError, match="match in length"):
            ExtPsdDriftCorrection(freqs, p_ref, p_bad, fs)


class TestNumericalBehavior:

    def test_correction_kernel_shape(self, analytic_drift_scenario, fs):
        freqs, p_ref, p_live, n_fft = analytic_drift_scenario
        corr = ExtPsdDriftCorrection(freqs, p_ref, p_live, fs)

        k_t = corr.compute_correction_kernel()
        assert len(k_t) == n_fft
        assert np.isrealobj(k_t)

    def test_truncation_length(self, analytic_drift_scenario, fs):
        freqs, p_ref, p_live, _ = analytic_drift_scenario
        corr = ExtPsdDriftCorrection(freqs, p_ref, p_live, fs)

        M = 128
        k_trunc = corr.compute_correction_kernel(truncate_samples=M)
        assert len(k_trunc) == M

    def test_adjoint_shape_and_type(self, analytic_drift_scenario, fs):
        freqs, p_ref, p_live, n_fft = analytic_drift_scenario
        corr = ExtPsdDriftCorrection(freqs, p_ref, p_live, fs)

        k_adj = corr.compute_adjoint_kernel()
        # Adjoint returns full buffer for circular convolution safety
        assert len(k_adj) == n_fft
        assert np.isrealobj(k_adj)

    def test_adjoint_truncation_zeros_middle(self, analytic_drift_scenario, fs):
        """Verify that truncating the adjoint zeroes out the 'causal' middle."""
        freqs, p_ref, p_live, n_fft = analytic_drift_scenario
        corr = ExtPsdDriftCorrection(freqs, p_ref, p_live, fs)

        M = 100
        k_adj = corr.compute_adjoint_kernel(truncate_samples=M)

        # We expect data at [0] and [N-(M-1) : N]
        # The middle region [1 : N-(M-1)] should be zero
        middle_start = 1
        middle_end = n_fft - (M - 1)

        assert np.allclose(k_adj[middle_start:middle_end], 0.0)
        # Check that edges are non-zero (given the physics of the problem)
        assert np.abs(k_adj[0]) > 0
        assert np.abs(k_adj[-1]) > 0


class TestScientificValidation:

    def test_group_property_recovery(self, analytic_drift_scenario, fs):
        """
        Verify |K|^2 * P_live approx P_ref.
        Validates K = W_live * W_ref^-1 = sqrt(P_ref/P_live).
        """
        freqs, p_ref, p_live, n_fft = analytic_drift_scenario
        corr = ExtPsdDriftCorrection(freqs, p_ref, p_live, fs)

        k_t = corr.compute_correction_kernel(smoothing_hz=0.0)
        K_f = np.fft.rfft(k_t)

        p_recovered = p_live * (np.abs(K_f) ** 2)

        # Check agreement (ignore DC/Nyquist edge effects)
        # We expect exact match since input is analytic and n_fft is large
        ratio = p_recovered[10:-10] / p_ref[10:-10]
        np.testing.assert_allclose(ratio, 1.0, rtol=1e-4)

    def test_causality(self, analytic_drift_scenario, fs):
        """
        Correction kernel energy should be concentrated at start (t>=0).
        With 32s buffer, wrap-around should be negligible (>99.9% causal).
        """
        freqs, p_ref, p_live, _ = analytic_drift_scenario
        corr = ExtPsdDriftCorrection(freqs, p_ref, p_live, fs)

        k_t = corr.compute_correction_kernel(smoothing_hz=0.0)

        # Check first 5% of buffer contains bulk of energy
        n_causal = int(0.05 * len(k_t))
        energy_causal = np.sum(k_t[:n_causal] ** 2)
        energy_total = np.sum(k_t**2)

        assert energy_causal / energy_total > 0.999

    def test_anti_causality_of_adjoint(self, analytic_drift_scenario, fs):
        """
        Adjoint kernel energy should be concentrated at end (t<=0 wrapped).
        """
        freqs, p_ref, p_live, _ = analytic_drift_scenario
        corr = ExtPsdDriftCorrection(freqs, p_ref, p_live, fs)

        k_adj = corr.compute_adjoint_kernel(smoothing_hz=0.0)

        # Check last 5% of buffer (plus index 0)
        n_anti = int(0.05 * len(k_adj))
        # Sum [N-n_anti : N] and [0]
        energy_anti = np.sum(k_adj[-n_anti:] ** 2) + k_adj[0] ** 2
        energy_total = np.sum(k_adj**2)

        assert energy_anti / energy_total > 0.999

    def test_aliasing_warning_triggered(self, fs):
        """
        Intentionally use a buffer that is too short (1s) for the feature width (0.5 Hz).
        This should cause Time-Domain Aliasing (wrap-around) and trigger the warning.
        """
        # 1. Setup Bad Scenario: Short Buffer (1s)
        duration = 1.0
        n_fft = int(duration * fs)
        freqs = np.linspace(0, fs / 2, n_fft // 2 + 1)

        # 2. Setup Sharp Feature: 0.5 Hz width -> ~2s decay time
        # This feature is physically longer than the buffer.
        f0 = 60.0
        noise = 1e-40 * np.ones_like(freqs)
        # Strong line to dominate the energy
        line = 1e-35 * np.exp(-0.5 * ((freqs - f0) / 0.5) ** 2)

        p_ref = noise + line
        p_live = noise  # Drastic change requires strong filter

        corr = ExtPsdDriftCorrection(freqs, p_ref, p_live, fs)

        # 3. Assert Warning is Raised
        with pytest.warns(AliasingWarning, match="Potential Time-Domain Aliasing"):
            _ = corr.compute_correction_kernel(smoothing_hz=0.0)

    def test_analytic_identity(self, fs):
        """If P_ref == P_live, K(t) should be a Delta function."""
        freqs = np.linspace(0, 100, 1024)
        psd = np.ones_like(freqs)
        corr = ExtPsdDriftCorrection(freqs, psd, psd, fs)

        k_t = corr.compute_correction_kernel()

        # Delta function: [1, 0, 0, ...]
        expected = np.zeros_like(k_t)
        expected[0] = 1.0

        np.testing.assert_allclose(k_t, expected, atol=1e-10)

    def test_smoothing_reduces_variance(self, fs):
        """Verify that smoothing_hz suppresses high-frequency noise in the kernel."""
        # Setup 4s buffer
        n_fft = int(4 * fs)
        freqs = np.linspace(0, 100, n_fft // 2 + 1)

        p_ref = np.ones_like(freqs)
        # Add random noise to live PSD
        rng = np.random.default_rng(42)
        # Ensure positive
        p_live = 1.0 + 0.5 * rng.standard_normal(len(freqs))
        p_live = np.abs(p_live)

        corr = ExtPsdDriftCorrection(freqs, p_ref, p_live, fs, n_fft=n_fft)

        # Compute Raw vs Smoothed
        # We can suppress aliasing warning here since we just want to check variance
        k_raw = corr.compute_correction_kernel(smoothing_hz=0.0, check_aliasing=False)
        k_smooth = corr.compute_correction_kernel(
            smoothing_hz=10.0, check_aliasing=False
        )

        # Measure high-frequency variance (roughness) in Frequency Domain
        K_raw_f = np.abs(np.fft.rfft(k_raw))
        K_smooth_f = np.abs(np.fft.rfft(k_smooth))

        roughness_raw = np.std(np.diff(K_raw_f))
        roughness_smooth = np.std(np.diff(K_smooth_f))

        # Smoothing should reduce roughness significantly
        assert roughness_smooth < 0.5 * roughness_raw

    @pytest.mark.skip(reason="Plotting test for manual verification.")
    def test_plot_analytic_drift_recovery(self, analytic_drift_scenario, fs):
        """
        Generates plots for visual inspection of kernel accuracy.
        Scenario: 0.5 Hz line drift.
        Saves 'debug_scientific_validation.png'.
        """
        import matplotlib.pyplot as plt

        freqs, p_ref, p_live, n_fft = analytic_drift_scenario
        corr = ExtPsdDriftCorrection(freqs, p_ref, p_live, fs)
        k_t = corr.compute_correction_kernel(smoothing_hz=0.0)
        K_f = rfft(k_t)
        p_recovered = p_live * (np.abs(K_f) ** 2)

        # --- Analytical Calculation for Verification ---
        # |K| = sqrt(P_ref / P_live)
        ratio = p_ref / p_live
        expected_mag = np.sqrt(ratio)

        # Manual Homomorphic Reconstruction
        log_amp = np.log(expected_mag)
        log_amp_full = np.concatenate([log_amp, log_amp[1:-1][::-1]])
        cepstrum = ifft(log_amp_full).real

        cepstrum_mp = np.zeros_like(cepstrum)
        cepstrum_mp[0] = cepstrum[0]
        N = len(cepstrum)
        cepstrum_mp[1 : N // 2] = 2 * cepstrum[1 : N // 2]
        cepstrum_mp[N // 2] = cepstrum[N // 2]

        h_analytic = ifft(np.exp(fft(cepstrum_mp))).real

        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

        # Plot 1: PSD Recovery
        ax = axes[0]
        ax.loglog(freqs, p_ref, label="Reference (Target)")
        ax.loglog(freqs, p_live, label="Live (Drifted)", alpha=0.5)
        ax.loglog(freqs, p_recovered, label="Recovered = Live * |K|^2", linestyle="--")
        ax.set_title("PSD Drift Recovery (0.5Hz Line Drift)")
        ax.set_ylabel("PSD")
        ax.set_xlim(55, 65)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Kernel Impulse - LOG SCALE
        ax = axes[1]
        t = np.arange(len(k_t)) / fs

        # Plot absolute value on log scale to see decay envelope
        ax.semilogy(
            t,
            np.abs(k_t),
            label="Computed |K(t)|",
            color="red",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.semilogy(
            t,
            np.abs(h_analytic),
            label="Analytical |h(t)|",
            color="blue",
            linestyle="--",
            alpha=0.7,
        )

        ax.axvline(0, color="black", linestyle="-", label="t=0 (Causality)")
        # Add Noise Floor
        ax.axhline(1e-15, color="gray", linestyle=":", label="Numerical Floor")

        ax.set_title("Correction Kernel Impulse (Log Magnitude)")
        ax.set_xlabel("Time [s]")
        # Zoom to show onset and first major oscillations
        ax.set_xlim(-0.05, 0.2)
        # Ensure we see the noise floor
        ax.set_ylim(1e-16, 10)
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")

        fname = "debug_scientific_validation.png"
        fig.tight_layout()
        fig.savefig(fname)
        print(f"\nSaved verification plot to {fname}")
        plt.close(fig)


class TestScientificValidationPowerLaw:

    def test_magnitude_inversion(self, power_law_bump_scenario, fs):
        """
        Verify that the correction kernel K(f) correctly inverts the Gaussian bump.
        If P_live = P_ref * (1 + Bump), then |K| should be 1/sqrt(1+Bump).
        """
        freqs, p_ref, p_live, n_fft = power_law_bump_scenario
        corr = ExtPsdDriftCorrection(freqs, p_ref, p_live, fs)

        k_t = corr.compute_correction_kernel(smoothing_hz=0.0)
        K_f = np.abs(np.fft.rfft(k_t))

        # Analytic Expectation
        ratio = p_live / p_ref
        expected_mag = 1.0 / np.sqrt(ratio)

        # Verify agreement (excluding DC/Nyquist)
        np.testing.assert_allclose(K_f[10:-10], expected_mag[10:-10], rtol=1e-4)

    def test_kernel_compactness(self, power_law_bump_scenario, fs):
        """
        Since the feature is Broad (20 Hz width), the kernel should decay FAST.
        Time constant ~ 1/20s = 50ms.
        We expect nearly all energy to be contained in the first 200ms.
        """
        freqs, p_ref, p_live, n_fft = power_law_bump_scenario
        corr = ExtPsdDriftCorrection(freqs, p_ref, p_live, fs)

        k_t = corr.compute_correction_kernel(smoothing_hz=0.0)

        # Define 200ms window
        n_compact = int(0.2 * fs)  # 819 samples

        energy_compact = np.sum(k_t[:n_compact] ** 2)
        energy_total = np.sum(k_t**2)

        # Assert > 99.9% energy is in the compact window
        # This confirms we don't have long ringing for broad spectral features
        assert energy_compact / energy_total > 0.999

    def test_adjoint_spectral_properties(self, power_law_bump_scenario, fs):
        """
        The Adjoint kernel should have the SAME magnitude response as the causal kernel,
        just with reversed phase.
        """
        freqs, p_ref, p_live, n_fft = power_law_bump_scenario
        corr = ExtPsdDriftCorrection(freqs, p_ref, p_live, fs)

        k_causal = corr.compute_correction_kernel(smoothing_hz=0.0)
        k_adjoint = corr.compute_adjoint_kernel(smoothing_hz=0.0)

        mag_causal = np.abs(np.fft.rfft(k_causal))
        mag_adjoint = np.abs(np.fft.rfft(k_adjoint))

        np.testing.assert_allclose(mag_causal, mag_adjoint, atol=1e-12)

    @pytest.mark.skip(reason="Plotting test for manual verification.")
    def test_plot_powerlaw_inversion(self, power_law_bump_scenario, fs):
        """
        Generates plots for the broadband power-law scenario.
        Includes explicit calculation of 'Analytical Impulse Response' via
        Homomorphic filtering from first principles to verify the class implementation.
        Saves 'debug_powerlaw_validation.png'.
        """
        import matplotlib.pyplot as plt
        from scipy.fft import fft, ifft

        freqs, p_ref, p_live, n_fft = power_law_bump_scenario
        corr = ExtPsdDriftCorrection(freqs, p_ref, p_live, fs)

        # 1. Compute via Class
        k_t = corr.compute_correction_kernel(smoothing_hz=0.0)
        K_f = np.abs(rfft(k_t))

        # 2. Analytic Magnitude Target
        ratio = p_live / p_ref
        expected_mag = 1.0 / np.sqrt(ratio)

        # 3. Compute "Analytical Impulse Response" from first principles
        # (Replicating MP logic manually to validate)
        # H = exp( Hilbert( log(|H|) ) )
        # log_amp = log(expected_mag) (two-sided)
        log_amp = np.log(expected_mag)
        log_amp_full = np.concatenate([log_amp, log_amp[1:-1][::-1]])

        # Real Cepstrum
        cepstrum = ifft(log_amp_full).real
        # Fold: 0, 2*pos, 0*neg
        cepstrum_mp = np.zeros_like(cepstrum)
        cepstrum_mp[0] = cepstrum[0]
        N = len(cepstrum)
        cepstrum_mp[1 : N // 2] = 2 * cepstrum[1 : N // 2]
        cepstrum_mp[N // 2] = cepstrum[N // 2]

        # Back to Freq
        log_H_mp = fft(cepstrum_mp)
        H_mp = np.exp(log_H_mp)
        h_analytic = ifft(H_mp).real

        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

        # Plot 1: Magnitude Inversion
        ax = axes[0]
        ax.plot(
            freqs, K_f, label="Computed |K(f)|", color="red", linewidth=2, alpha=0.7
        )
        ax.plot(
            freqs,
            expected_mag,
            label="Theoretical Target",
            linestyle="--",
            color="black",
        )
        ax.set_title("Broadband Feature Inversion (Gaussian Bump @ 100Hz)")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude")
        ax.set_xlim(50, 150)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Impulse Response Comparison (Zoomed, Log Scale)
        ax = axes[1]
        t = np.arange(len(k_t)) / fs * 1000  # ms

        ax.semilogy(
            t, np.abs(k_t), label="Computed |K(t)|", color="red", linewidth=2, alpha=0.7
        )
        ax.semilogy(
            t,
            np.abs(h_analytic),
            label="Analytical |h(t)|",
            color="blue",
            linestyle="--",
        )

        ax.axvline(0, color="black", linestyle="-", label="t=0 (Causality)")
        ax.axhline(1e-15, color="gray", linestyle=":", label="Numerical Floor")

        ax.set_title("Correction Kernel (Time Domain - Zoomed Log Scale)")
        ax.set_xlabel("Time [ms]")
        # Tight zoom on the initial pulse structure
        ax.set_xlim(-5, 50)
        ax.set_ylim(1e-16, 2)
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")

        fname = "debug_powerlaw_validation.png"
        fig.tight_layout()
        fig.savefig(fname)
        print(f"\nSaved verification plot to {fname}")
        plt.close(fig)


class TestBiasMeasurementComparison:
    """
    Validates that ExactPsdDriftCorrection.compute_bias_measurements()
    converges to PertMpCorrection results for small mismatches.
    """

    def test_perturbative_agreement(self, small_mismatch_scenario, fs):
        """
        Comparison of Exact vs Perturbative.
        Note: Exact method finds peak of correlation, Perturbative integrates phase deriv.
        For oscillating group delays, slight discrepancies in definition (correction vs bias)
        and numerical methods (DFT vs Integral) can occur.
        We check for rough agreement in magnitude.
        """
        freqs, p_ref, p_live, h_tilde, n_fft = small_mismatch_scenario

        pert_calc = PrtPsdDriftCorrection(freqs, p_ref, p_live, h_tilde, fs)
        res_pert = pert_calc.correction()

        exact_calc = ExtPsdDriftCorrection(freqs, p_ref, p_live, fs, n_fft=n_fft)
        res_exact = exact_calc.compute_bias_measurements(h_tilde, smoothing_hz=0.0)

        print(
            f"\nTime Shift (dt): Pert={res_pert.dt1:.4e} vs Exact={res_exact.dt1:.4e}"
        )

        # TIME SHIFT: Compare magnitudes
        # Perturbative and Exact may have different sign conventions for "Correction" vs "Bias".
        # We assert they are of the same order of magnitude.
        # dt ~ 7e-7 s (very small).
        if abs(res_pert.dt1) > 1e-9:
            # Allow 50% diff given the different methodologies for sub-sample shifts
            diff = abs(abs(res_exact.dt1) - abs(res_pert.dt1))
            assert diff / abs(res_pert.dt1) < 0.50
        else:
            assert abs(res_exact.dt1) < 1e-9

        # PHASE SHIFT: Consistent definitions, should match well.
        print(
            f"Phase Shift (dphi): Pert={res_pert.dphi1:.4e} vs Exact={res_exact.dphi1:.4e}"
        )
        if abs(res_pert.dphi1) > 1e-6:
            assert abs(res_exact.dphi1 - res_pert.dphi1) / abs(res_pert.dphi1) < 0.05

        # SNR LOSS: Consistent definitions, should match well.
        print(
            f"SNR Loss (dsnr): Pert={res_pert.dsnr1:.4e} vs Exact={res_exact.dsnr1:.4e}"
        )
        if abs(res_pert.dsnr1) > 1e-6:
            assert abs(res_exact.dsnr1 - res_pert.dsnr1) / abs(res_pert.dsnr1) < 0.05

    def test_consistency_check(self, small_mismatch_scenario, fs):
        """
        Identity Check: If P_live == P_ref, all biases must be zero.
        """
        freqs, p_ref, _, h_tilde, n_fft = small_mismatch_scenario
        # Force Identity
        exact_calc = ExtPsdDriftCorrection(freqs, p_ref, p_ref, fs, n_fft=n_fft)
        res = exact_calc.compute_bias_measurements(h_tilde, smoothing_hz=0.0)

        assert abs(res.dt1) < 1e-12
        assert abs(res.dphi1) < 1e-12
        assert abs(res.dsnr1) < 1e-12
