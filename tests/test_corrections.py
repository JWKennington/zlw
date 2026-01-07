"""
test_mpmp_correction.py

Pytest test suite for the MPMPCorrection class.

Covers:
- Input validation and initialization behavior.
- Numerical integration paths (Simpson vs trapezoid).
- Public API invariants and regression-style behavior.
- Scientific validation: simplified geometric dt1/dphi1 vs full
  first-order dt1/dphi1 derived from ΔI derivatives (as in the paper).
"""

import numpy as np
import pytest

from scipy.integrate import simpson

# Adjust these imports if your module lives somewhere else
from zlw.corrections import PrtPsdDriftCorrection, TimePhaseCorrection
from zlw.kernels import MPWhiteningFilter


# ---------------------------------------------------------------------------
# Helpers: full first-order corrections (reference implementation)
# ---------------------------------------------------------------------------


def _integrate(freqs: np.ndarray, arr: np.ndarray) -> float:
    """
    Integration helper that mirrors the implementation in MPMPCorrection:
    Simpson's rule for odd n, trapezoid for even n.
    """
    arr = np.asarray(arr, dtype=float)
    n = arr.size
    if n % 2 == 1:
        return float(simpson(arr, freqs))
    else:
        # Handle NumPy 2.0+ removal of trapz
        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(arr, freqs))
        else:
            return float(np.trapz(arr, freqs))


def compute_full_first_order_biases(
    freqs: np.ndarray,
    psd1: np.ndarray,
    psd2: np.ndarray,
    htilde: np.ndarray,
    fs: float,
):
    """
    Reference implementation of dt1_full, dphi1_full based on the original
    ΔI derivative formulas (full MP–MP expressions), using the real
    MPWhiteningFilter.

    This is only used inside tests as a scientific baseline to validate the
    simplified geometric formulas implemented in MPMPCorrection.
    """
    freqs = np.asarray(freqs, dtype=float)
    psd1 = np.asarray(psd1, dtype=float)
    psd2 = np.asarray(psd2, dtype=float)
    htilde = np.asarray(htilde, dtype=complex)

    n = freqs.size
    n_fft = 2 * (n - 1)

    mp1 = MPWhiteningFilter(psd1, fs, n_fft)
    mp2 = MPWhiteningFilter(psd2, fs, n_fft)

    wk1 = mp1.frequency_response()  # real, positive
    wk2 = mp2.frequency_response()  # complex
    dwk = wk2 - wk1

    # Precompute |W1 h|^2
    w1h = wk1 * htilde
    abs_w1h_sq = np.abs(w1h) ** 2

    # I0 second derivatives
    def d_i_tt():
        coeff = -((2 * np.pi) ** 2)
        integrand = freqs**2 * abs_w1h_sq
        return coeff * _integrate(freqs, integrand)

    def d_i_pp():
        coeff = -1.0
        integrand = abs_w1h_sq
        return coeff * _integrate(freqs, integrand)

    # ΔI derivatives (data = htilde, so mismatched template/data are same waveform)
    def d_di_t():
        coeff = 2 * np.pi
        integrand = freqs * np.imag(dwk * htilde * np.conj(w1h))
        return coeff * _integrate(freqs, integrand)

    def d_di_p():
        coeff = 1.0
        integrand = np.imag(dwk * htilde * np.conj(w1h))
        return coeff * _integrate(freqs, integrand)

    I_tt = d_i_tt()
    I_pp = d_i_pp()

    dt1_full = d_di_t() / I_tt if I_tt != 0.0 else 0.0
    dphi1_full = d_di_p() / I_pp if I_pp != 0.0 else 0.0

    return dt1_full, dphi1_full


# ---------------------------------------------------------------------------
# Fixtures: grids, PSDs, templates
# ---------------------------------------------------------------------------


@pytest.fixture
def fs():
    """Sampling rate (Hz) used in tests."""
    return 4096.0


@pytest.fixture
def freqs_odd():
    """Odd-length one-sided frequency grid (Simpson path)."""
    return np.linspace(20.0, 512.0, 513)  # 513 = 2*256+1


@pytest.fixture
def freqs_even():
    """Even-length one-sided frequency grid (trapezoid path)."""
    return np.linspace(20.0, 512.0, 512)  # 512 points


@pytest.fixture
def realistic_psds(freqs_odd):
    """
    A simple "realistic" pair of PSDs:
      psd1 ~ f^4 (steeply rising),
      psd2 = psd1 * (1 + eps * sin(log f))^2  with small eps.
    """
    f = freqs_odd
    # Avoid f=0, so f>0 always; power-law shape
    psd1 = (f / f[0]) ** 4

    eps = 0.1  # small-ish mismatch
    g = np.sin(np.log(f / f[0]))
    scale = 1.0 + eps * g
    psd2 = psd1 * scale**2

    return psd1, psd2


@pytest.fixture
def simple_psds(freqs_odd):
    """
    Very simple flat PSD pair; psd2 slightly scaled version of psd1.
    Good for eps diagnostics and symmetry tests.
    """
    n = freqs_odd.size
    psd1 = np.ones(n)
    psd2 = np.full(n, 1.21)  # (1+0.1)^2
    return psd1, psd2


@pytest.fixture
def inspiral_like_template(freqs_odd):
    """
    Simple inspiral-like template amplitude: |h(f)| ~ f^{-7/6}, purely real.
    """
    f = freqs_odd
    amp = (f / f[0]) ** (-7.0 / 6.0)
    return amp.astype(complex)


@pytest.fixture
def flat_template(freqs_odd):
    """Flat template: htilde = 1."""
    return np.ones(freqs_odd.size, dtype=complex)


# ---------------------------------------------------------------------------
# Initialization & Validation
# ---------------------------------------------------------------------------


class TestInitAndValidation:
    """Tests for __post_init__ input validation and basic attributes."""

    def test_init_valid_inputs_sets_attributes(
        self,
        freqs_odd,
        simple_psds,
        flat_template,
        fs,
    ):
        psd1, psd2 = simple_psds
        n = freqs_odd.size

        corr = PrtPsdDriftCorrection(
            freqs=freqs_odd,
            psd1=psd1,
            psd2=psd2,
            h_tilde=flat_template,
            fs=fs,
        )

        # Shapes & basic derived quantities
        assert corr.freqs.shape == (n,)
        assert corr.psd1.shape == (n,)
        assert corr.psd2.shape == (n,)
        assert corr.h_tilde.shape == (n,)
        assert corr.wk1.shape == (n,)
        assert corr.wk2.shape == (n,)
        assert corr.w_simple.shape == (n,)
        assert corr.phi_diff.shape == (n,)

        assert corr.df == pytest.approx(freqs_odd[1] - freqs_odd[0])
        assert corr.n_fft == 2 * (n - 1)

        # eps diagnostic
        expected_eps = abs(1.0 / np.sqrt(1.21) - 1.0)
        assert corr.eps == pytest.approx(expected_eps)

    def test_init_raises_on_mismatched_lengths(self, fs):
        freqs = np.linspace(20.0, 512.0, 50)
        psd1 = np.ones(51)
        psd2 = np.ones(50)
        htilde = np.ones(50, dtype=complex)

        with pytest.raises(ValueError, match="same length"):
            PrtPsdDriftCorrection(
                freqs=freqs, psd1=psd1, psd2=psd2, h_tilde=htilde, fs=fs
            )

    def test_init_raises_on_too_few_bins(self, fs):
        freqs = np.linspace(20.0, 30.0, 2)
        psd1 = np.ones(2)
        psd2 = np.ones(2)
        htilde = np.ones(2, dtype=complex)

        with pytest.raises(ValueError, match="at least 3"):
            PrtPsdDriftCorrection(
                freqs=freqs, psd1=psd1, psd2=psd2, h_tilde=htilde, fs=fs
            )

    def test_init_raises_on_nonmonotonic_freqs(self, fs):
        freqs = np.array([100.0, 50.0, 200.0])  # nonmonotonic
        psd1 = np.ones(3)
        psd2 = np.ones(3)
        htilde = np.ones(3, dtype=complex)

        with pytest.raises(ValueError, match="strictly increasing"):
            PrtPsdDriftCorrection(
                freqs=freqs, psd1=psd1, psd2=psd2, h_tilde=htilde, fs=fs
            )

    def test_init_raises_on_nonpositive_psd(self, fs):
        freqs = np.linspace(20.0, 30.0, 3)
        psd1 = np.array([1.0, 0.0, 1.0])  # zero element
        psd2 = np.ones(3)
        htilde = np.ones(3, dtype=complex)

        with pytest.raises(ValueError, match="strictly positive"):
            PrtPsdDriftCorrection(
                freqs=freqs, psd1=psd1, psd2=psd2, h_tilde=htilde, fs=fs
            )


# ---------------------------------------------------------------------------
# Scientific validation: simple symmetry & full vs simplified
# ---------------------------------------------------------------------------


class TestScientificValidation:
    """
    Scientific validation tests:

    1. Symmetry checks (psd1 == psd2 ⇒ zero phase mismatch ⇒ zero corrections).
    2. Full first-order dt1/dphi1 vs simplified geometric dt1/dphi1 for a
       small MP–MP mismatch, up to overall sign convention.
    """

    def test_no_mismatch_gives_zero_corrections(
        self,
        freqs_odd,
        flat_template,
        fs,
    ):
        """
        If psd1 == psd2, both whiteners are identical (same amplitude and
        phase), so Φ(f) = 0 ⇒ dt1 = dphi1 = 0 by construction.
        """
        n = freqs_odd.size
        psd1 = np.ones(n)
        psd2 = np.ones(n)
        htilde = flat_template

        corr = PrtPsdDriftCorrection(
            freqs=freqs_odd,
            psd1=psd1,
            psd2=psd2,
            h_tilde=htilde,
            fs=fs,
        )

        assert corr.eps == 0.0
        assert corr.dt1() == 0.0
        assert corr.dphi1() == 0.0

    def test_simplified_matches_full_first_order_for_small_mismatch(
        self,
        freqs_odd,
        realistic_psds,
        inspiral_like_template,
        fs,
    ):
        """
        For a small MP–MP mismatch, the simplified geometric formulas should
        agree with the full first-order dt1/dphi1 to within a modest fraction
        of the mismatch size, *up to an overall sign convention*.

        In practice, dt1 from the geometric formula and from the ΔI-based
        full expression can differ by a global sign depending on whether one
        defines the shift as (data − template) or (template − data). Here we
        test agreement in magnitude and allow for a possible sign flip.
        """
        psd1, psd2 = realistic_psds
        htilde = inspiral_like_template

        corr = PrtPsdDriftCorrection(
            freqs=freqs_odd,
            psd1=psd1,
            psd2=psd2,
            h_tilde=htilde,
            fs=fs,
        )

        dt1_simplified = corr.dt1()
        dphi1_simplified = corr.dphi1()

        dt1_full, dphi1_full = compute_full_first_order_biases(
            freqs=freqs_odd,
            psd1=psd1,
            psd2=psd2,
            htilde=htilde,
            fs=fs,
        )

        # If the full result is essentially zero, the simplified should also be small.
        # Otherwise, compare magnitudes and allow for a possible sign flip.
        rel_tol = 5e-2  # 5% relative tolerance on magnitude

        if abs(dt1_full) < 1e-10:
            assert abs(dt1_simplified) < 1e-8
        else:
            ratio_dt = abs(dt1_simplified) / abs(dt1_full)
            assert ratio_dt == pytest.approx(1.0, rel=rel_tol)

        if abs(dphi1_full) < 1e-10:
            assert abs(dphi1_simplified) < 1e-8
        else:
            ratio_dphi = abs(dphi1_simplified) / abs(dphi1_full)
            assert ratio_dphi == pytest.approx(1.0, rel=rel_tol)


# ---------------------------------------------------------------------------
# Integration behavior & invariances
# ---------------------------------------------------------------------------


class TestIntegrationAndInvariances:
    """
    Tests that:

    - Both Simpson and trapezoid paths are exercised.
    - dt1/dphi1 are invariant under trivial transformations of htilde
      (global amplitude and phase), as they should be from the formulas.
    """

    def test_even_grid_uses_trapezoid_path(
        self,
        freqs_even,
        simple_psds,
        flat_template,
        fs,
    ):
        """
        Smoke test on an even-length grid: just ensure dt1/dphi1 run and
        produce finite values (this exercises the trapezoid branch).
        """
        psd1, psd2 = simple_psds

        corr = PrtPsdDriftCorrection(
            freqs=freqs_even,
            psd1=psd1[: freqs_even.size],
            psd2=psd2[: freqs_even.size],
            h_tilde=flat_template[: freqs_even.size],
            fs=fs,
        )

        dt1 = corr.dt1()
        dphi1 = corr.dphi1()

        assert np.isfinite(dt1)
        assert np.isfinite(dphi1)

    def test_corrections_invariant_under_global_amplitude(
        self,
        freqs_odd,
        realistic_psds,
        inspiral_like_template,
        fs,
    ):
        """
        dt1 and dphi1 should be invariant under a global real amplitude rescaling
        of the template: htilde -> A * htilde.
        """
        psd1, psd2 = realistic_psds
        htilde = inspiral_like_template

        A = 2.5

        corr1 = PrtPsdDriftCorrection(
            freqs=freqs_odd,
            psd1=psd1,
            psd2=psd2,
            h_tilde=htilde,
            fs=fs,
        )
        corr2 = PrtPsdDriftCorrection(
            freqs=freqs_odd,
            psd1=psd1,
            psd2=psd2,
            h_tilde=A * htilde,
            fs=fs,
        )

        assert corr1.dt1() == pytest.approx(corr2.dt1())
        assert corr1.dphi1() == pytest.approx(corr2.dphi1())

    def test_corrections_invariant_under_global_phase(
        self,
        freqs_odd,
        realistic_psds,
        inspiral_like_template,
        fs,
    ):
        """
        dt1 and dphi1 should be invariant under a global phase rotation
        of the template: htilde -> exp(i phi0) * htilde.
        """
        psd1, psd2 = realistic_psds
        htilde = inspiral_like_template

        phi0 = 0.7

        corr1 = PrtPsdDriftCorrection(
            freqs=freqs_odd,
            psd1=psd1,
            psd2=psd2,
            h_tilde=htilde,
            fs=fs,
        )
        corr2 = PrtPsdDriftCorrection(
            freqs=freqs_odd,
            psd1=psd1,
            psd2=psd2,
            h_tilde=np.exp(1j * phi0) * htilde,
            fs=fs,
        )

        assert corr1.dt1() == pytest.approx(corr2.dt1())
        assert corr1.dphi1() == pytest.approx(corr2.dphi1())


# ---------------------------------------------------------------------------
# Public API & regression-style checks
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Tests focused on the public-facing API layer."""

    def test_correction_namedtuple_contents(
        self,
        freqs_odd,
        realistic_psds,
        inspiral_like_template,
        fs,
    ):
        """
        correction() should return a TimePhaseCorrection namedtuple with
        dt2=dphi2=0.0 and dt1/dphi1 consistent with direct calls.
        """
        psd1, psd2 = realistic_psds
        htilde = inspiral_like_template

        corr = PrtPsdDriftCorrection(
            freqs=freqs_odd,
            psd1=psd1,
            psd2=psd2,
            h_tilde=htilde,
            fs=fs,
        )

        dt1_direct = corr.dt1()
        dphi1_direct = corr.dphi1()
        dsnr1_direct = corr.dsnr1()

        result = corr.correction()
        assert isinstance(result, TimePhaseCorrection)

        dt1, dt2, dphi1, dphi2, dsnr = result

        assert dt1 == pytest.approx(dt1_direct)
        assert dphi1 == pytest.approx(dphi1_direct)
        assert dt2 == 0.0
        assert dphi2 == 0.0
        assert dsnr == pytest.approx(dsnr1_direct)
