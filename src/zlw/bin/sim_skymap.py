#!/usr/bin/env python3
"""
Skymap Impact of Per-Detector Time and Phase Biases

This script computes how small timing and phase biases at individual detectors
(H1, L1, V1) propagate into an error in the reconstructed sky position
(right ascension and declination) in a simple time-of-arrival triangulation
model.

It is intentionally decoupled from PSDs and whitening. The idea is:

    1. You obtain per-detector time and/or phase biases from some analysis
       (e.g. MP–MP PSD-based script).
    2. You feed those biases in here (as Δt_i in microseconds and/or Δφ_i
       in radians).
    3. This script computes the implied sky-shift (ΔRA, ΔDec) using a
       linearized inversion around the true direction.

Model:
    We consider a 3-detector network (H1, L1, V1) with fixed Earth-centered
    positions r_i in an Earth-fixed frame. For simplicity, we:

      * Use a spherical Earth with radius R ≈ 6371 km.
      * Use approximate detector geodetic locations (latitude, longitude).

    For a true sky direction n (unit vector), the arrival time at detector i
    relative to the Earth's center is:

        t_i = t_ref - (n · r_i) / c,

    where c is the speed of light and t_ref is an arbitrary reference time.

    If we introduce *small* timing biases Δt_i (seconds) that shift the arrival
    times to t_i^biased = t_i + Δt_i, we can linearize around n_true and
    write:

        Δt_i ≈ - (r_i · δn) / c.

    After subtracting the average over detectors (to remove the degeneracy with
    the global reference time), we can solve in least squares for δn and set

        n_fit ≈ normalize(n_true + δn).

    Converting n_fit to (RA, Dec) yields the biased sky position, and the
    angular separation between n_true and n_fit quantifies the skymap error.

Inputs:
    * True sky location (RA0, Dec0) in degrees.
    * Per-detector **time biases** dt_h1, dt_l1, dt_v1 in microseconds.
    * Optional per-detector **phase biases** dphi_h1, dphi_l1, dphi_v1
      in radians.
    * Optional effective frequency f_eff [Hz] to convert phase biases to time:

          Δt_phase_i = Δφ_i / (2π f_eff).

    Total time bias at each detector is:

        Δt_total_i = Δt_time_i + Δt_phase_i.

Outputs:
    * Printed:
        - True (RA0, Dec0),
        - Reconstructed (RA_fit, Dec_fit),
        - Angular separation between true and biased directions (deg, arcmin),
        - Per-detector total time biases (μs).

    * Plot (optional):
        - RA/Dec scatter showing true vs biased sky position, saved to
          outdir/ra_dec_shift.png.

Example:
    Example with ±20 μs timing biases between H1 and L1:

        zlw-sim-sky \\
            --ra0-deg 200.0 \\
            --dec0-deg -30.0 \\
            --dt-h1 20 --dt-l1 -20 --dt-v1 0 \\
            --outdir figs_sky_impact

    Example using phase biases at effective frequency 100 Hz:

        zlw-sim-sky \\
            --ra0-deg 200.0 --dec0-deg -30.0 \\
            --dt-h1 0 --dt-l1 0 --dt-v1 0 \\
            --dphi-h1 0.01 --dphi-l1 -0.01 --dphi-v1 0.0 \\
            --f-eff 100 \\
            --outdir figs_sky_phase
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

# Speed of light and Earth radius (spherical approximation).
C_LIGHT = 299792458.0  # m/s
R_EARTH = 6371e3  # m


@dataclass
class SkymapImpactConfig:
    """Configuration for the skymap impact computation.

    Attributes:
        ra0_deg: True RA of the source [deg].
        dec0_deg: True Dec of the source [deg].
        dt_h1: Time bias at H1 [microseconds].
        dt_l1: Time bias at L1 [microseconds].
        dt_v1: Time bias at V1 [microseconds].
        dphi_h1: Phase bias at H1 [rad].
        dphi_l1: Phase bias at L1 [rad].
        dphi_v1: Phase bias at V1 [rad].
        f_eff: Effective frequency [Hz] for converting phase biases to time.
        outdir: Directory to save plots.
        make_plots: Whether to generate plots.
    """

    ra0_deg: float
    dec0_deg: float
    dt_h1: float
    dt_l1: float
    dt_v1: float
    dphi_h1: float
    dphi_l1: float
    dphi_v1: float
    f_eff: float
    outdir: str
    make_plots: bool


def parse_args() -> SkymapImpactConfig:
    """Parse CLI arguments into a SkymapImpactConfig.

    Returns:
        SkymapImpactConfig: Parsed configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute how per-detector time/phase biases (H1,L1,V1) perturb "
            "the reconstructed sky position (RA, Dec)."
        )
    )

    parser.add_argument(
        "--ra0-deg",
        type=float,
        required=True,
        help="True RA of the source [deg], in an Earth-fixed equatorial frame.",
    )
    parser.add_argument(
        "--dec0-deg",
        type=float,
        required=True,
        help="True Dec of the source [deg], in an Earth-fixed equatorial frame.",
    )

    parser.add_argument(
        "--dt-h1",
        type=float,
        default=20.0,
        help="Timing bias at H1 [microseconds] (default: 20).",
    )
    parser.add_argument(
        "--dt-l1",
        type=float,
        default=-20.0,
        help="Timing bias at L1 [microseconds] (default: -20).",
    )
    parser.add_argument(
        "--dt-v1",
        type=float,
        default=0.0,
        help="Timing bias at V1 [microseconds] (default: 0).",
    )

    parser.add_argument(
        "--dphi-h1",
        type=float,
        default=0.0,
        help="Phase bias at H1 [rad] (default: 0.0).",
    )
    parser.add_argument(
        "--dphi-l1",
        type=float,
        default=0.0,
        help="Phase bias at L1 [rad] (default: 0.0).",
    )
    parser.add_argument(
        "--dphi-v1",
        type=float,
        default=0.0,
        help="Phase bias at V1 [rad] (default: 0.0).",
    )

    parser.add_argument(
        "--f-eff",
        type=float,
        default=0.0,
        help=(
            "Effective frequency [Hz] to convert phase bias Δφ to time via "
            "Δt = Δφ / (2π f_eff). If 0, phase biases are ignored for timing."
        ),
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="figs_sky_impact",
        help="Output directory for plots (default: figs_sky_impact).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="If set, do not generate any plots.",
    )

    args = parser.parse_args()

    return SkymapImpactConfig(
        ra0_deg=args.ra0_deg,
        dec0_deg=args.dec0_deg,
        dt_h1=args.dt_h1,
        dt_l1=args.dt_l1,
        dt_v1=args.dt_v1,
        dphi_h1=args.dphi_h1,
        dphi_l1=args.dphi_l1,
        dphi_v1=args.dphi_v1,
        f_eff=args.f_eff,
        outdir=args.outdir,
        make_plots=not args.no_plots,
    )


# ---------------------------------------------------------------------------
# Detector geometry (approximate)
# ---------------------------------------------------------------------------


def latlon_to_ecef(lat_deg: float, lon_deg: float, r: float = R_EARTH) -> np.ndarray:
    """Convert geodetic latitude/longitude to ECEF coordinates.

    Args:
        lat_deg: Latitude in degrees.
        lon_deg: Longitude in degrees.
        r: Earth radius [m].

    Returns:
        ECEF position vector [x, y, z] in meters.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    x = r * math.cos(lat) * math.cos(lon)
    y = r * math.cos(lat) * math.sin(lon)
    z = r * math.sin(lat)
    return np.array([x, y, z], dtype=float)


def get_detector_positions() -> dict[str, np.ndarray]:
    """Return approximate ECEF positions for H1, L1, V1.

    The positions are based on approximate lat/lon (degrees):

        H1: Hanford
        L1: Livingston
        V1: Virgo

    You may refine these to match your exact geometry.

    Returns:
        Mapping from detector name ("H1","L1","V1") to ECEF position vectors [m].
    """
    latlon = {
        "H1": (46.455, -119.408),
        "L1": (30.563, -90.774),
        "V1": (43.63, 10.5),
    }
    positions = {ifo: latlon_to_ecef(lat, lon) for ifo, (lat, lon) in latlon.items()}
    return positions


# ---------------------------------------------------------------------------
# Time-of-arrival model and inversion
# ---------------------------------------------------------------------------


def ra_dec_to_unit_vector(ra_deg: float, dec_deg: float) -> np.ndarray:
    """Convert (RA, Dec) in degrees to a unit vector in equatorial ECEF frame.

    Args:
        ra_deg: Right ascension [deg].
        dec_deg: Declination [deg].

    Returns:
        Unit vector n pointing to that sky position.
    """
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    x = math.cos(dec) * math.cos(ra)
    y = math.cos(dec) * math.sin(ra)
    z = math.sin(dec)
    v = np.array([x, y, z], dtype=float)
    return v / np.linalg.norm(v)


def unit_vector_to_ra_dec(n: np.ndarray) -> Tuple[float, float]:
    """Convert a unit vector to (RA, Dec) in degrees.

    Args:
        n: Unit vector [x, y, z] in ECEF equatorial frame.

    Returns:
        Tuple (ra_deg, dec_deg).
    """
    x, y, z = n
    r_xy = math.hypot(x, y)
    dec = math.atan2(z, r_xy)
    ra = math.atan2(y, x)
    if ra < 0.0:
        ra += 2.0 * math.pi
    return math.degrees(ra), math.degrees(dec)


def angular_separation_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    """Compute angular separation between two unit vectors in degrees.

    Args:
        n1: Unit direction vector.
        n2: Unit direction vector.

    Returns:
        Angular separation angle [deg].
    """
    dot = float(np.dot(n1, n2))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


# ---------------------------------------------------------------------------
# Linearized estimator for small timing biases
# ---------------------------------------------------------------------------


def compute_total_dt_seconds(cfg: SkymapImpactConfig) -> dict[str, float]:
    """Combine time and phase biases for each detector, returning Δt_i [s].

    Time biases are specified in microseconds and converted to seconds:

        Δt_time_i = dt_i * 1e-6.

    If f_eff > 0, phase biases are converted to effective time biases:

        Δt_phase_i = Δφ_i / (2π f_eff).

    Total per-detector time bias:

        Δt_total_i = Δt_time_i + Δt_phase_i.

    Args:
        cfg: Configuration containing per-detector biases and f_eff.

    Returns:
        Mapping from detector name to total time bias Δt_i [s].
    """
    dt_time_h1 = cfg.dt_h1 * 1e-6
    dt_time_l1 = cfg.dt_l1 * 1e-6
    dt_time_v1 = cfg.dt_v1 * 1e-6

    if cfg.f_eff > 0.0:
        dt_phase_h1 = cfg.dphi_h1 / (2.0 * math.pi * cfg.f_eff)
        dt_phase_l1 = cfg.dphi_l1 / (2.0 * math.pi * cfg.f_eff)
        dt_phase_v1 = cfg.dphi_v1 / (2.0 * math.pi * cfg.f_eff)
    else:
        dt_phase_h1 = dt_phase_l1 = dt_phase_v1 = 0.0

    dt_total = {
        "H1": dt_time_h1 + dt_phase_h1,
        "L1": dt_time_l1 + dt_phase_l1,
        "V1": dt_time_v1 + dt_phase_v1,
    }
    return dt_total


def estimate_direction_from_time_biases(
    dt_total: dict[str, float],
    positions: dict[str, np.ndarray],
    n_true: np.ndarray,
) -> np.ndarray:
    """Estimate perturbed direction n_fit given small time biases Δt_i [s].

    This uses the linearized model around n_true:

        Δt_i ≈ - (r_i · δn) / c,

    with the mean over detectors subtracted from both sides to remove the
    degeneracy with a global reference time shift. We then solve in least
    squares for δn and set

        n_fit ≈ normalize(n_true + δn).

    Args:
        dt_total: Mapping from detector name to total time bias Δt_i [s].
        positions: Mapping from detector name to ECEF position r_i [m].
        n_true: True unit direction vector.

    Returns:
        Unit vector n_fit representing the biased sky direction.
    """
    ifos = sorted(dt_total.keys())
    dt_arr = np.array([dt_total[ifo] for ifo in ifos], dtype=float)
    r_arr = np.vstack([positions[ifo] for ifo in ifos])

    # Subtract means: Δt_i' = Δt_i - mean_i Δt_i, r_i' = r_i - mean_i r_i
    dt_mean = np.mean(dt_arr)
    r_mean = np.mean(r_arr, axis=0)
    dt_prime = dt_arr - dt_mean
    r_prime = r_arr - r_mean

    # Linear system: A δn ≈ b, with b = -c Δt_prime
    A = r_prime  # (N_det, 3)
    b = -C_LIGHT * dt_prime  # (N_det,)

    delta_n, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Remove any component of delta_n along n_true (degenerate with t_ref).
    proj = float(np.dot(delta_n, n_true))
    delta_n = delta_n - proj * n_true

    n_fit = n_true + delta_n
    n_fit = n_fit / np.linalg.norm(n_fit)
    return n_fit


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_ra_dec_plot(
    cfg: SkymapImpactConfig,
    ra0_deg: float,
    dec0_deg: float,
    ra_fit_deg: float,
    dec_fit_deg: float,
) -> None:
    """Plot true and biased sky positions in RA/Dec.

    Args:
        cfg: Configuration for output directory.
        ra0_deg: True RA [deg].
        dec0_deg: True Dec [deg].
        ra_fit_deg: Reconstructed RA [deg].
        dec_fit_deg: Reconstructed Dec [deg].
    """
    os.makedirs(cfg.outdir, exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.scatter(ra0_deg, dec0_deg, c="C0", label="True", s=40)
    plt.scatter(ra_fit_deg, dec_fit_deg, c="C1", label="Biased", s=40, marker="x")

    ra_min = min(ra0_deg, ra_fit_deg) - 1.0
    ra_max = max(ra0_deg, ra_fit_deg) + 1.0
    dec_min = min(dec0_deg, dec_fit_deg) - 1.0
    dec_max = max(dec0_deg, dec_fit_deg) + 1.0

    plt.xlim(ra_min, ra_max)
    plt.ylim(dec_min, dec_max)
    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.title("Sky position: true vs biased")
    plt.grid(True, ls=":")
    plt.legend()
    # Invert RA axis so RA increases to the left (astronomy convention).
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.outdir, "ra_dec_shift.png"), dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the skymap impact computation."""
    cfg = parse_args()

    # True direction n_true
    n_true = ra_dec_to_unit_vector(cfg.ra0_deg, cfg.dec0_deg)

    # Detector positions
    positions = get_detector_positions()

    # Total Δt_i [s] for each detector (from timing + phase)
    dt_total = compute_total_dt_seconds(cfg)

    # Estimate direction from the incremental time biases
    n_fit = estimate_direction_from_time_biases(dt_total, positions, n_true)
    ra_fit_deg, dec_fit_deg = unit_vector_to_ra_dec(n_fit)

    # Angular separation between true and biased directions
    sep_deg = angular_separation_deg(n_true, n_fit)

    print("=== Skymap Impact of Time/Phase Biases ===")
    print(
        f"True sky position:    RA0 = {cfg.ra0_deg:.6f} deg, Dec0 = {cfg.dec0_deg:.6f} deg"
    )
    print(
        f"Biased sky position:  RA_fit = {ra_fit_deg:.6f} deg, Dec_fit = {dec_fit_deg:.6f} deg"
    )
    print(f"Angular separation:   {sep_deg:.6f} deg ({sep_deg * 60.0:.3f} arcmin)")
    print()
    print("Per-detector total time biases:")
    for ifo in ["H1", "L1", "V1"]:
        print(f"  Δt_{ifo} = {dt_total[ifo] * 1e6:.3f} μs")

    if cfg.make_plots:
        make_ra_dec_plot(cfg, cfg.ra0_deg, cfg.dec0_deg, ra_fit_deg, dec_fit_deg)


if __name__ == "__main__":
    main()
