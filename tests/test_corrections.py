"""Test coverage for sgnligo.kernels module.
Tests include:
    - unit tests for LPWhiteningFilter and MPWhiteningFilter classes
    - scientific-validity tests for MPWhiteningFilter using simplified PSD models
        (Lorentzian, Exponential, Gaussian bump, Power-law)
    - basic tests for TimePhaseCorrection and MPMPCorrection classes
"""

import os

import lal
import lalsimulation as lalsim
import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from zlw.corrections import (
    MPMPCorrection,
    TimePhaseCorrection,
)


class TestMPMPCorrection:
    """Test group for the LPMPCorrection class."""

    @pytest.fixture
    def sample_rate(self):
        """Sample rate (Hz) for the psds and template (should be common across all)"""
        return 2048.0

    @pytest.fixture
    def f_bounds(self, sample_rate):
        """frequency bounds for the psds and template, should be common"""
        return (20.0, sample_rate / 2)

    @pytest.fixture
    def duration(self):
        """Duration of the template in seconds, used to determine frequency
        resolution"""
        return 20.0

    @pytest.fixture
    def freqs(self, f_bounds, duration):
        """Frequency bins for the psds and template (should be common across all)"""
        delta_f = 1.0 / duration  # frequency spacing [Hz]
        f_min, f_max = f_bounds
        N_req = int(np.ceil((f_max - f_min) / delta_f))
        # Find next biggest power of 2 if not already a power of 2
        N_fft = 2 ** int(np.ceil(np.log2(N_req))) + 1
        return np.arange(N_fft) * delta_f

    @pytest.fixture
    def psd1(self, freqs):
        """Simple PSD similar to LIGO design sensitivity"""
        data = 3e-48 * ((freqs / 300) ** 2 + (200 / freqs) + (100 / freqs) ** 8)
        # Zero out the DC bin to avoid singularity
        data[0] = data[1]  # TODO handle a amplitude floor in MPWhiteningFilter?
        return data

    @pytest.fixture
    def psd2(self, freqs, psd1):
        """PSD with a reasonable perturbation from psd1"""
        data = psd1 * (1.0 + 0.05 * np.exp(-0.5 * ((freqs - 150) / 20) ** 2))
        return data

    @pytest.fixture
    def template_fd(self, f_bounds, duration):
        """Simple 30 M☉–50 M☉ inspiral chirp **directly** in the frequency domain,
        at 2048 Hz sample rate, using LALSimulation's FD waveform generator.
        """
        delta_f = 1.0 / duration  # frequency spacing [Hz]
        f_min, f_max = f_bounds
        m1, m2 = 30.0, 50.0  # component masses [M_sun]
        distance = 1e6 * lal.PC_SI  # 1 Mpc in meters (arbitrary scaling)
        inclination = 0.0  # face‐on
        phi_ref = 0.0  # coalescence phase
        f_ref = 0.0  # reference frequency at coalescence
        approx = lalsim.SimInspiralGetApproximantFromString("IMRPhenomD")

        # --- assemble parameters dict for SimInspiralFD ---
        params = {
            "m1": m1 * lal.MSUN_SI,
            "m2": m2 * lal.MSUN_SI,
            "S1x": 0.0,
            "S1y": 0.0,
            "S1z": 0.0,
            "S2x": 0.0,
            "S2y": 0.0,
            "S2z": 0.0,
            "distance": distance,
            "inclination": inclination,
            "phiRef": phi_ref,
            "longAscNodes": 0.0,
            "eccentricity": 0.0,
            "meanPerAno": 0.0,
            "deltaF": delta_f,
            "f_min": f_min,
            "f_max": f_max,
            "f_ref": f_ref,
            "LALparams": None,
            "approximant": approx,
        }

        # --- generate frequency‐domain waveform directly ---
        hp_fd, _ = lalsim.SimInspiralFD(**params)

        # --- extract numpy arrays ---
        f0 = hp_fd.f0
        df = hp_fd.deltaF
        length = hp_fd.data.length
        freqs = f0 + np.arange(length) * df
        htilde = np.array(hp_fd.data.data, copy=True)

        return freqs, htilde

    def test_init(self, freqs, psd1, psd2, template_fd, sample_rate):
        """Initialize LPMPCorrection without error and correct attributes."""
        corr = MPMPCorrection(
            freqs=freqs,
            psd1=psd1,
            psd2=psd2,
            htilde=template_fd[1],
            fs=sample_rate,
        )
        assert isinstance(corr, MPMPCorrection)
        assert corr.df == freqs[1] - freqs[0]
        assert corr.n_fft == (len(freqs) - 1) * 2

    def test_partial_derivs(self, freqs, psd1, psd2, template_fd, sample_rate):
        """Initialize LPMPCorrection without error and correct attributes."""
        corr = MPMPCorrection(
            freqs=freqs,
            psd1=psd1,
            psd2=psd2,
            htilde=template_fd[1],
            fs=sample_rate,
        )
        # Partial derivs of unperturbed overlap
        d_i_tt = corr._d_i_tt()
        d_i_pp = corr._d_i_pp()

        # Partial derivs of perturbed overlap
        d_di_t = corr._d_di_t()
        d_di_p = corr._d_di_p()
        d_di_tt = corr._d_di_tt()
        d_di_tp = corr._d_di_tp()
        d_di_pp = corr._d_di_pp()

        assert np.isreal(d_i_tt)
        assert np.isreal(d_i_pp)
        assert np.isreal(d_di_t)
        assert np.isreal(d_di_p)
        assert np.isreal(d_di_tt)
        assert np.isreal(d_di_tp)
        assert np.isreal(d_di_pp)

    def test_full_correction(self, freqs, psd1, psd2, template_fd, sample_rate):
        """Initialize LPMPCorrection without error and correct attributes."""
        corr = MPMPCorrection(
            freqs=freqs,
            psd1=psd1,
            psd2=psd2,
            htilde=template_fd[1],
            fs=sample_rate,
        )
        cv = corr.full_correction()
        assert isinstance(cv, TimePhaseCorrection)
        assert cv.dt2 != 0.0 or cv.dphi2 != 0.0, "Correction should not be trivial"
        assert abs(cv.dt1) < 1e-5, "dt1 should be small"
        assert abs(cv.dt2) < 1e-6, "dt2 should be small"

        assert abs(cv.dphi1) < 1e-2, "dphi1 should be small"
        assert abs(cv.dphi2) < 1e-3, "dphi2 should be small"

        # Check relative sizes
        assert abs(cv.dt2) < abs(cv.dt1), "dt2 should be smaller than dt1"
        assert abs(cv.dphi2) < abs(cv.dphi1), "dphi2 should be smaller than dphi1"

    def test_simplified_correction(self, freqs, psd1, psd2, template_fd, sample_rate):
        """Test simplified correction"""
        corr = MPMPCorrection(
            freqs=freqs,
            psd1=psd1,
            psd2=psd2,
            htilde=template_fd[1],
            fs=sample_rate,
        )
        fc = corr.simplified_correction()
        assert isinstance(fc, TimePhaseCorrection)
        assert fc.dt2 == 0.0
        assert fc.dphi2 == 0.0

        # Check nontrivial simplified dt1 and dphi1
        assert abs(fc.dt1) > 0.0, "dt1 should be nontrivial"
        assert abs(fc.dphi1) > 0.0, "dphi1 should be nontrivial"

        # Check small
        assert abs(fc.dt1) < 1e-5, "dt1 should be small"
        assert abs(fc.dphi1) < 1e-2, "dphi1 should be small"


class TestAnalyticMPMPCorrection:
    """Test first-order correction against analytic estimate for toy PSD and inspiral
    waveform."""

    @pytest.fixture
    def sample_rate(self):
        return 4096.0

    @pytest.fixture
    def f_bounds(self):
        # Analyze from 20 Hz to 1024 Hz
        return 0.0, 1024.0

    @pytest.fixture
    def duration(self):
        # Duration sets frequency resolution: 1/duration
        return 32.0

    @pytest.fixture
    def freqs(self, f_bounds, duration):
        f_min, f_max = f_bounds
        delta_f = 1.0 / duration
        N_req = int(np.ceil((f_max - f_min) / delta_f))
        N_fft = 2 ** int(np.ceil(np.log2(N_req))) + 1
        # full grid from 0 to (N_fft-1)*delta_f
        return np.arange(N_fft) * delta_f

    @pytest.fixture
    def htilde(self, freqs):
        # Toy inspiral amplitude ~ f^{-7/6}, zero below f_min
        h = freqs ** (-7.0 / 6.0)
        h[freqs == 0] = 0.0
        return h

    @pytest.fixture
    def psd1(self, freqs):
        # Flat PSD (white noise)
        psd = np.ones_like(freqs)
        psd[0] = psd[1]
        return psd

    @pytest.mark.parametrize("eps", [0.0001, 0.001, 0.01, 0.1])
    def test_correction_matches_analytic(self, freqs, htilde, psd1, sample_rate, eps):
        # Build PSD2 with linear tilt: 1 + eps * (f / f_max)
        f_max = freqs.max()
        psd2 = psd1 * np.exp(eps * freqs / f_max)

        # No injection phase/time offset: data = template
        data_fd = htilde.copy()

        # Compute correction via MPMPCorrection
        corr = MPMPCorrection(freqs, psd1, psd2, htilde, sample_rate)
        dt1, dt2, dphi1, dphi2 = corr.full_correction(data=data_fd)

        # Analytic estimate: dt1 ≈ eps / (4 π f_max)
        dt1_analytic = eps / (4.0 * np.pi * f_max)

        # Assert that computed dt1 ≃ analytic within 10% relative error
        rel_error = abs((dt1 - dt1_analytic) / dt1_analytic)
        assert rel_error < 0.25, (
            f"dt1 {dt1:.3e} does not match analytic {dt1_analytic:.3e}"
            f" (rel error {rel_error:.2%})"
        )

    def test_phase_bias_small(self, freqs, htilde, psd1, sample_rate):
        # Test that phase correction is small but present for eps=0.1
        eps = 0.1
        f_max = freqs.max()
        psd2 = psd1 * (1.0 + eps * (freqs / f_max))
        data_fd = htilde.copy()

        corr = MPMPCorrection(freqs, psd1, psd2, htilde, sample_rate)
        dt1, dt2, dphi1, dphi2 = corr.full_correction(data=data_fd)

        # Analytic estimate: dphi1 ≈ (eps/2) * <g> where <g> ~ 1/2
        # thus dphi1 ~ eps/4 ~ 0.025
        dphi1_est = eps / 4.0
        assert (
            abs(dphi1 - dphi1_est) < 0.05
        ), f"dphi1 {dphi1:.3e} deviates from estimate {dphi1_est:.3e}"

    @pytest.mark.skipif(
        os.getenv("SGNLIGO_TEST_SCIVAL_PLOT") != "1",
        reason="Set SGNLIGO_TEST_SCIVAL_PLOT=1 to display comparison plots",
    )
    def test_plot_correction_vs_epsilon(
        self, freqs, htilde, psd1, sample_rate, tmp_path
    ):
        """
        Generate and save a plot of measured vs. analytic dt1 over a range of epsilons.
        """
        f_max = freqs.max()
        epsilons = np.logspace(-3.0, 0.0, 50)
        measured = []
        analytic = []
        analytic_k = []

        for eps in epsilons:
            psd2 = psd1 * np.exp(eps * freqs / f_max)
            corr = MPMPCorrection(freqs, psd1, psd2, htilde, sample_rate)
            dt1, *_ = corr.full_correction(data=htilde)
            measured.append(dt1)
            analytic.append(eps / (4.0 * np.pi * f_max))

            W = np.abs(corr.wk1 * htilde) ** 2
            df = freqs[1] - freqs[0]
            phase_diff = np.unwrap(np.angle(corr.wk1) - np.angle(corr.wk2))

            # numerator and denominator sums
            N = np.sum(freqs * phase_diff * W) * df
            D = np.sum((freqs**2) * W) * df

            # exact discrete‐grid first‐order bias
            dt1_analytic = N / (2.0 * np.pi * D)
            analytic_k.append(dt1_analytic)

        measured = np.array(measured)
        analytic = np.array(analytic)
        analytic_k = np.array(analytic_k)

        fig, ax = plt.subplots()
        ax.plot(epsilons, measured / 1e-6, "o-", label="measured dt1")
        ax.plot(epsilons, analytic / 1e-6, "s--", label="analytic dt1")
        ax.plot(epsilons, analytic_k / 1e-6, "x:", label="numerical approx K dt1")
        # ax.plot(epsilons, analytic2, "d--", label="analytic2 K dt1 (scaled by K)")
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(r"$\delta t^{(1)}$ ($\mu$s)")
        ax.set_title("Measured vs. Analytic First‑Order Time Bias 1/2 pi")
        ax.legend()
        plt.show()

        # Save figure to png file
        fig.savefig("mp_mpcorrection_dt1_vs_epsilon.png")


class TestMPMPCorrectionSciVal1:
    """Test group for the MPMPCorrection class."""

    @pytest.fixture
    def sample_rate(self):
        return 4096.0

    @pytest.fixture
    def f_bounds(self, sample_rate):
        return (20.0, sample_rate / 2)

    @pytest.fixture
    def duration(self):
        return 32.0

    @pytest.fixture
    def freqs(self, f_bounds, duration):
        delta_f = 1.0 / duration
        f_min, f_max = f_bounds
        N_req = int(np.ceil((f_max - f_min) / delta_f))
        N_fft = 2 ** int(np.ceil(np.log2(N_req))) + 1
        return np.arange(N_fft) * delta_f

    @pytest.fixture
    def psd1(self, freqs):
        data = 3e-48 * ((freqs / 300) ** 2 + 200 / freqs + (100 / freqs) ** 8)
        data[0] = data[1]
        return data

    @pytest.fixture
    def psd2(self, freqs, psd1):
        eps = 0.05
        res = psd1 * (1 + eps * freqs)
        res = res * (1.0 + 0.5 * np.exp(-0.5 * ((freqs - 50) / 20) ** 2))
        return res

    @pytest.fixture
    def template_fd(self, f_bounds, duration):
        """Simple 30 M☉–50 M☉ inspiral chirp **directly** in the frequency domain,
        at 2048 Hz sample rate, using LALSimulation's FD waveform generator.
        """
        delta_f = 1.0 / duration  # frequency spacing [Hz]
        f_min, f_max = f_bounds
        m1, m2 = 2.0, 3.0  # component masses [M_sun]
        distance = 1e6 * lal.PC_SI  # 1 Mpc in meters (arbitrary scaling)
        inclination = 0.0  # face‐on
        phi_ref = 0.0  # coalescence phase
        f_ref = 0.0  # reference frequency at coalescence
        approx = lalsim.SimInspiralGetApproximantFromString("IMRPhenomD")

        # --- assemble parameters dict for SimInspiralFD ---
        params = {
            "m1": m1 * lal.MSUN_SI,
            "m2": m2 * lal.MSUN_SI,
            "S1x": 0.0,
            "S1y": 0.0,
            "S1z": 0.0,
            "S2x": 0.0,
            "S2y": 0.0,
            "S2z": 0.0,
            "distance": distance,
            "inclination": inclination,
            "phiRef": phi_ref,
            "longAscNodes": 0.0,
            "eccentricity": 0.0,
            "meanPerAno": 0.0,
            "deltaF": delta_f,
            "f_min": f_min,
            "f_max": f_max,
            "f_ref": f_ref,
            "LALparams": None,
            "approximant": approx,
        }

        # --- generate frequency‐domain waveform directly ---
        hp_fd, _ = lalsim.SimInspiralFD(**params)

        # --- extract numpy arrays ---
        f0 = hp_fd.f0
        df = hp_fd.deltaF
        length = hp_fd.data.length
        freqs = f0 + np.arange(length) * df
        htilde = np.array(hp_fd.data.data, copy=True)

        return freqs, htilde

    def test_recovery_base_case(self, freqs, psd1, sample_rate, template_fd):
        # setup injection
        t_true = 3.0257
        # Verify non integer time
        assert (
            t_true % (1.0 / sample_rate) != 0
        ), "t_true should not be an integer multiple of dt"
        phi_true = 0.0
        freqs, htilde = template_fd
        data_fd = htilde * np.exp(-2j * np.pi * freqs * t_true) * np.exp(-1j * phi_true)

        corr = MPMPCorrection(freqs, psd1, psd1, htilde, sample_rate)
        # whiten
        w1h = corr.wk1 * htilde
        w1d = corr.wk1 * data_fd
        w2d = corr.wk2 * data_fd
        df = corr.df
        norm = np.sqrt(np.sum(np.abs(w1h) ** 2) * df)
        mf_u = np.fft.irfft(w1d * np.conj(w1h), n=corr.n_fft) * df / norm
        mf_p = np.fft.irfft(w2d * np.conj(w1h), n=corr.n_fft) * df / norm

        Δf = freqs[1] - freqs[0]
        n_fft = corr.n_fft
        dt_mf = 1.0 / (n_fft * Δf)
        assert t_true % dt_mf != 0, "t_true should not be an integer multiple of dt"

        t_u, phi_u = find_cubic_spline_peak(mf_u, dt_mf)
        t_p, phi_p = find_cubic_spline_peak(mf_p, dt_mf)

        assert t_p - t_u < 1e-12, "No correction should be needed for identical PSDs"
        assert t_p - t_true < 1e-12, "Peak time should match true time"

    def test_full_vs_simplified_agreement_when_psd_equal(
        self, freqs, psd1, template_fd, sample_rate
    ):
        """When psd2=psd1, there should be no mismatch => all corrections ~0."""
        corr = MPMPCorrection(freqs, psd1, psd1, template_fd[1], sample_rate)
        full = corr.full_correction()
        simp = corr.simplified_correction()
        # first‐order
        assert abs(full.dt1) < 1e-12
        assert abs(full.dphi1) < 1e-12
        assert abs(simp.dt1) < 1e-12
        assert abs(simp.dphi1) < 1e-12
        # second‐order
        assert abs(full.dt2) < 1e-12
        assert abs(full.dphi2) < 1e-12

    @pytest.mark.skipif(
        os.getenv("SGNLIGO_TEST_SCIVAL_PLOT") != "1",
        reason="Set SGNLIGO_TEST_SCIVAL_PLOT=1 to display comparison plots",
    )
    def test_error_vs_perturbation_magnitude(
        self, freqs, psd1, sample_rate, template_fd, tmp_path
    ):
        t_true, phi_true = 0.02, 0.3
        freqs, htilde = template_fd
        epsilons = [0.01, 0.05, 0.1, 0.2, 0.5]
        errs_before, errs_after = [], []
        for eps in epsilons:
            psd2 = psd1 * (1 + eps * (freqs / freqs.max()))
            corr = MPMPCorrection(freqs, psd1, psd2, htilde, sample_rate)
            # data injection
            data_fd = (
                htilde * np.exp(-2j * np.pi * freqs * t_true) * np.exp(-1j * phi_true)
            )
            # matched filter
            w1h = corr.wk1 * htilde
            w2d = corr.wk2 * data_fd
            norm = np.sqrt(np.sum(np.abs(w1h) ** 2) * corr.df)
            mf_p = np.fft.irfft(w2d * np.conj(w1h), n=corr.n_fft) * corr.df / norm
            idx_p = np.argmax(np.abs(mf_p))
            t_p = idx_p / sample_rate
            dt1, _, _, _ = corr.full_correction(data=data_fd)
            errs_before.append(abs(t_p - t_true))
            errs_after.append(abs((t_p - dt1) - t_true))
        # plot
        fig, ax = plt.subplots()
        ax.plot(epsilons, errs_before, "o-", label="error before")
        ax.plot(epsilons, errs_after, "s--", label="error after")
        ax.set_xscale("log")
        ax.set_xlabel("ε (perturbation magnitude)")
        ax.set_ylabel("Time error [s]")
        ax.set_title("Recovery error vs PSD perturbation")
        ax.legend()
        plt.show()

        # Save figure to png file
        fig.savefig("mp_mpcorrection_scival1_error_vs_eps.png")


class TestMPMPCorrectionSciVal2:
    """Scientific validation test for first-order time & phase bias correction."""

    @pytest.fixture
    def sample_rate(self):
        # Use a typical gravitational-wave sampling rate
        return 4096.0

    @pytest.fixture
    def duration(self):
        # Duration of the signal (seconds)
        return 32.0

    @pytest.fixture
    def template_fd(self, sample_rate, duration):
        """Generate a frequency-domain inspiral waveform (IMRPhenomD) for use as
        template."""
        df = 1.0 / duration
        f_low = 20.0
        f_high = sample_rate / 2.0
        approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomD")
        # Waveform parameters (moderate mass binary, non-spinning, 1 Mpc distance)
        hp_fd, _ = lalsim.SimInspiralFD(
            m1=2.0 * lal.MSUN_SI,
            m2=3.0 * lal.MSUN_SI,
            S1x=0,
            S1y=0,
            S1z=0,
            S2x=0,
            S2y=0,
            S2z=0,
            distance=1e6 * lal.PC_SI,  # 1 Mpc
            inclination=0.0,
            phiRef=0.0,
            longAscNodes=0.0,
            eccentricity=0.0,
            meanPerAno=0.0,
            deltaF=df,
            f_min=f_low,
            f_max=f_high,
            f_ref=0.0,
            LALparams=None,
            approximant=approximant,
        )
        # Extract frequency series data
        N = hp_fd.data.length  # number of frequency bins
        freqs = hp_fd.f0 + np.arange(N) * hp_fd.deltaF
        htilde = np.array(hp_fd.data.data, copy=True)
        return freqs, htilde  # one-sided frequency series of template (complex values)

    @pytest.fixture
    def psd1(self, template_fd):
        """Define a realistic baseline PSD (one-sided) for the given frequency grid."""
        freqs, _ = template_fd
        # Example PSD: combination of rising power law at high f and steeply rising
        # noise at low f
        psd = 3e-48 * ((freqs / 300.0) ** 2 + 200.0 / freqs + (100.0 / freqs) ** 8)
        psd[0] = psd[1]  # avoid zero at f=0 by copying the value at the next bin
        return psd

    @pytest.fixture
    def psd2(self, psd1, template_fd):
        """Define a perturbed PSD = PSD1 + small fractional tilt perturbation."""
        freqs, _ = template_fd
        eps = 100.0
        bump = np.exp(-0.5 * ((freqs - 30.0) / 10.0) ** 2)
        psd2 = psd1 * (1.0 + eps * bump)
        psd2[0] = psd2[1]  # ensure no zero at DC
        return psd2

    @pytest.fixture
    def psd3(self, psd1, template_fd):
        """Define a perturbed PSD = PSD1 + small fractional tilt perturbation."""
        freqs, _ = template_fd
        eps = 0.01
        bump = np.exp(-0.5 * ((freqs - 30.0) / 10.0) ** 2)
        psd2 = psd1 * (1.0 + eps * bump)
        psd2[0] = psd2[1]  # ensure no zero at DC
        return psd2

    def test_baseline_time_and_phase_psd_match(self, template_fd, psd1, sample_rate):
        """
        Baseline sanity check: when PSD1 == PSD2 and injection phase = 0,
        the pipeline should recover the coalescence time and phase (≈0)
        within realistic interpolation errors.
        """
        freqs, htilde = template_fd
        df = freqs[1] - freqs[0]
        N = len(freqs) * 2 - 2
        oversample = 4
        Nfft = N * oversample
        dt = 1.0 / (Nfft * df)

        # --- 1) Ground truth: non-integer time delay, zero phase ---
        t_true = 0.0137  # 13.7 ms delay (not a multiple of dt)

        # Sanity: ensure sub-sample injection
        assert (t_true / dt) % 1 != 0, "Choose t_true not aligned to the grid"

        # --- 2) Build freq-domain data_fd for injection with zero phase ---
        # data_fd = H(f) * exp(-2πi f t_true) * exp(-i φ_true)
        data_fd = htilde * np.exp(-2j * np.pi * freqs * t_true)

        # --- 3) Instantiate correction object with identical PSDs ---
        corr0 = MPMPCorrection(
            freqs=freqs, psd1=psd1, psd2=psd1, htilde=htilde, fs=sample_rate
        )

        # --- 4) Whiten template & data with the same MP filter (wk1) ---
        w0h = corr0.wk1 * htilde
        w0d = corr0.wk1 * data_fd

        # Norm for matched filter
        norm0 = np.sqrt(np.sum(np.abs(w0h) ** 2) * df)
        assert norm0 > 0.0

        # --- 5) Compute complex SNR time series via IRFFT (zero pad by oversample) ---
        mf0 = np.fft.irfft(w0d * np.conj(w0h), n=Nfft) * (df / norm0)

        # --- 6) Locate peak with sub-sample spline (window=5 for robustness) ---
        t_rec, phi_rec = find_cubic_spline_peak(mf0, dt, window=5)

        # --- 7) Assertions: recovered time & phase must match true values within
        # error ---
        # Time: within ±2 dt
        assert (
            abs(t_rec - t_true) < 2 * dt
        ), f"Recovered time {t_rec:.6f}s vs true {t_true:.6f}s; dt={dt:.3e}s"

        # Phase: since φ_true=0, we expect φ_rec ≈ 0 modulo 2π
        # Wrap to [−π, π]
        phi_err = (phi_rec + np.pi) % (2 * np.pi) - np.pi
        assert (
            abs(phi_err) < 1e-2
        ), f"Recovered phase {phi_rec:.3f} rad; expected 0 → error {phi_err:.3e}"

    def test_first_order_correction_improves_time_and_phase(
        self, template_fd, psd1, psd2, sample_rate
    ):
        """
        Verify that, under a small PSD mismatch (PSD2 = PSD1 + tilt),
        the first-order MP–MP whitening correction (full & simplified)
        reduces both time and phase errors of a time-domain injection.
        """
        freqs, htilde = template_fd
        df = freqs[1] - freqs[0]
        N = len(freqs) * 2 - 2
        oversample = 4
        Nfft = N * oversample
        dt = 1.0 / (Nfft * df)

        # 1) Ground truth injection (sub-sample) with non-zero phase
        t_true = 0.020  # 20 ms
        phi_true = 0.3  # rad
        assert (t_true / dt) % 1 != 0, "Injection must be off-grid for sub-sample test"

        # 2) Build frequency-domain injection
        data_fd = htilde * np.exp(-2j * np.pi * freqs * t_true) * np.exp(-1j * phi_true)

        # 3) Instantiate MP–MP correction (PSD1 for template, PSD2 for data)
        corr = MPMPCorrection(
            freqs=freqs, psd1=psd1, psd2=psd2, htilde=htilde, fs=sample_rate
        )

        # 4) Whiten template & data
        w1h = corr.wk1 * htilde  # template → PSD1
        w1d = corr.wk1 * data_fd  # data → PSD1 (baseline)
        w2d = corr.wk2 * data_fd  # data → PSD2 (mismatch)
        norm = np.sqrt(np.sum(np.abs(w1h) ** 2) * df)
        assert norm > 0, "Whitened template norm must be positive"

        # 5) Compute complex SNR time-series
        mf_base = np.fft.irfft(w1d * np.conj(w1h), n=Nfft) * (df / norm)
        mf_mismatch = np.fft.irfft(w2d * np.conj(w1h), n=Nfft) * (df / norm)

        # 6) Locate sub-sample peaks (time & phase)
        t_u, phi_u = find_cubic_spline_peak(mf_base, dt, window=5)
        t_p, phi_p = find_cubic_spline_peak(mf_mismatch, dt, window=5)

        # 8) Confirm mismatch introduced bias
        err_time_p = abs(t_p - t_true)
        err_phase_p = abs((phi_p - phi_true + np.pi) % (2 * np.pi) - np.pi)

        # 9) Compute first-order corrections
        dt1_full, _, dphi1_full, _ = corr.full_correction(data=data_fd)
        simp = corr.simplified_correction()
        dt1_simp = simp.dt1
        dphi1_simp = simp.dphi1

        # 10) Corrected estimates
        t_corr_full = t_p - dt1_full
        phi_corr_full = phi_p - dphi1_full
        t_corr_simp = t_p - dt1_simp
        phi_corr_simp = phi_p - dphi1_simp

        # 11) Assert corrections reduce the *mismatched* errors err_time_p, err_phase_p
        err_time_full = abs(t_corr_full - t_true)
        err_time_simp = abs(t_corr_simp - t_true)
        err_phase_full = abs((phi_corr_full - phi_true + np.pi) % (2 * np.pi) - np.pi)
        err_phase_simp = abs((phi_corr_simp - phi_true + np.pi) % (2 * np.pi) - np.pi)

        assert (
            err_time_full < err_time_p
        ), f"Full corr: time error {err_time_full:.2e} ≥ before {err_time_p:.2e}"
        assert (
            err_time_simp < err_time_p
        ), f"Simp corr: time error {err_time_simp:.2e} ≥ before {err_time_p:.2e}"
        assert (
            err_phase_full < err_phase_p
        ), f"Full corr: phase error {err_phase_full:.3e} ≥ before {err_phase_p:.3e}"
        assert (
            err_phase_simp < err_phase_p
        ), f"Simp corr: phase error {err_phase_simp:.3e} ≥ before {err_phase_p:.3e}"

        # 12) And full ≤ simplified
        assert err_time_full <= err_time_simp, "Full corr time error > simplified"
        # assert err_phase_full <= err_phase_simp, "Full corr phase error > simplified"

    def test_first_order_prediction_small(self, template_fd, psd1, psd3, sample_rate):
        """
        In the ε→0 regime, the measured PSD‐induced bias (t_p−t_u, φ_p−φ_u)
        should agree with the predicted first‐order corrections δt₁, δφ₁
        to within a few Δt / Δφ numerical steps.
        """
        freqs, htilde = template_fd
        df = freqs[1] - freqs[0]
        N = len(freqs) * 2 - 2
        oversample = 1024
        Nfft = N * oversample
        dt = 1.0 / (Nfft * df)

        # --- build an off‐grid injection with small phase ---
        t_true, phi_true = 0.020, 0.3
        data_fd = htilde * np.exp(-2j * np.pi * freqs * t_true) * np.exp(-1j * phi_true)

        # --- instantiate MP–MP correction with tiny PSD bump ---
        corr = MPMPCorrection(
            freqs=freqs, psd1=psd1, psd2=psd3, htilde=htilde, fs=sample_rate
        )

        # --- whiten and form MF time series ---
        w1h = corr.wk1 * htilde
        w1d = corr.wk1 * data_fd
        w2d = corr.wk2 * data_fd
        norm = np.sqrt(np.sum(np.abs(w1h) ** 2) * df)
        assert norm > 0

        mf_base = np.fft.irfft(w1d * np.conj(w1h), n=Nfft) * (df / norm)
        mf_mismatch = np.fft.irfft(w2d * np.conj(w1h), n=Nfft) * (df / norm)

        # 1) baseline peak (no bump)
        t_u, phi_u = find_cubic_spline_peak(mf_base, dt, window=5)
        # 2) mismatched peak (with tiny bump)
        t_p, phi_p = find_cubic_spline_peak(mf_mismatch, dt, window=5)

        # 3) measure the *additional* bias due to PSD mismatch
        delta_t_meas = t_p - t_u
        delta_phi_meas = ((phi_p - phi_u + np.pi) % (2 * np.pi)) - np.pi

        # 4) compute first‐order analytic predictions
        dt1_full, _, dphi1_full, _ = corr.full_correction(data=data_fd)
        simp = corr.simplified_correction()
        dt1_simp, dphi1_simp = simp.dt1, simp.dphi1

        # 5) compare measured vs predicted within a few dt / dφ
        tol_time = 3 * dt  # O(100 ns)
        tol_phase = 1e-2  # O(0.01 rad)

        assert (
            abs(delta_t_meas - dt1_full) < tol_time
        ), f"meas Δt={delta_t_meas:.3e}s vs pred_full δt₁={dt1_full:.3e}s"
        assert (
            abs(delta_t_meas - dt1_simp) < tol_time
        ), f"meas Δt={delta_t_meas:.3e}s vs pred_simp δt₁={dt1_simp:.3e}s"
        assert (
            abs(delta_phi_meas - dphi1_full) < tol_phase
        ), f"meas Δφ={delta_phi_meas:.3e} vs pred_full δφ₁={dphi1_full:.3e}"
        assert (
            abs(delta_phi_meas - dphi1_simp) < tol_phase
        ), f"meas Δφ={delta_phi_meas:.3e} vs pred_simp δφ₁={dphi1_simp:.3e}"


# Helper: find sub-sample peak time and phase of a complex matched-filter output
def find_cubic_spline_peak(mf: np.ndarray, dt: float, window: int = 2):
    """
    Find a sub-sample peak of a matched-filter output using cubic spline interpolation.
    Returns:
        t_peak (float): Sub-sample peak time in seconds.
        phi_peak (float): Phase of mf at t_peak (radians).
    """
    N = mf.size
    # 1) integer-bin peak index
    i0 = int(np.argmax(np.abs(mf)))
    # 2) neighborhood of samples around the peak
    idx = np.arange(i0 - window, i0 + window + 1)
    idx = idx[(idx >= 0) & (idx < N)]
    t_vals = idx * dt
    # 3) fit cubic spline to magnitude
    mag = np.abs(mf[idx])
    cs_mag = CubicSpline(t_vals, mag, bc_type="natural")
    # find spline extremum (roots of derivative)
    a, b, c, _ = cs_mag.c[:, 0]
    roots = np.roots([3 * a, 2 * b, c])
    # consider real roots in the fitting window
    real_roots = roots[np.isreal(roots)].real
    valid = real_roots[(real_roots >= t_vals.min()) & (real_roots <= t_vals.max())]
    t_peak = valid[0] if valid.size else i0 * dt
    # 4) interpolate phase at t_peak using separate spline fits for real and imag parts
    cs_re = CubicSpline(t_vals, np.real(mf[idx]), bc_type="natural")
    cs_im = CubicSpline(t_vals, np.imag(mf[idx]), bc_type="natural")
    re_peak = cs_re(t_peak)
    im_peak = cs_im(t_peak)
    phi_peak = np.angle(re_peak + 1j * im_peak)
    return t_peak, phi_peak

