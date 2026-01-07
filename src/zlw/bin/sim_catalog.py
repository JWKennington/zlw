#!/usr/bin/env python3
"""
Catalog-Level Heuristic Analysis of MP–MP Timing and Phase Biases (GWTC-4).

Updates:
1. **SNR Drift:** Computes and plots fractional SNR change (dsnr1) to quantify sensitivity loss.
2. **Monte Carlo Sky:** Uses random sky injection for sensitivity analysis.
3. **Drift Stats:** Separated plots for Mean/Median vs Max drift.

Usage:
    zlw-sim-cat --outdir gwtc4_results --verbose --ref-psd-offset 604800
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# --- External Libraries ---
try:
    from gwpy.table import EventTable
    from gwpy.timeseries import TimeSeries
    from gwosc.datasets import event_detectors
    from gwosc.timeline import get_segments
except ImportError:
    print("Error: gwpy/gwosc not installed.", file=sys.stderr)
    sys.exit(1)

try:
    import lal
    import lalsimulation as lalsim
except ImportError:
    print("Error: lalsuite not installed.", file=sys.stderr)
    sys.exit(1)

# --- Project Libraries ---
try:
    from zlw.corrections import PrtPsdDriftCorrection
except ImportError:
    print("Error: 'zlw' package not found in path.", file=sys.stderr)
    sys.exit(1)

# Configure Logging
log = logging.getLogger("zlw_catalog")

# Standard Channel Names
CHANNELS = {
    "H1": "H1:DCS-CALIB_STRAIN_C01",
    "L1": "L1:DCS-CALIB_STRAIN_C01",
    "V1": "V1:Hrec_hoft_16384Hz",
}

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class AnalysisConfig:
    outdir: str
    f_min: float
    f_max: float
    df: float
    data_duration: float
    fft_ref: float
    fft_data: float
    max_events: Optional[int]
    verbose: bool
    ref_psd_offset: float
    use_nds: bool


def parse_args() -> AnalysisConfig:
    parser = argparse.ArgumentParser(description="GWTC-4 MP-MP Bias Analysis")
    parser.add_argument("-o", "--outdir", type=str, default="gwtc4_analysis")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print debug logs")

    parser.add_argument("--f-min", type=float, default=20.0)
    parser.add_argument("--f-max", type=float, default=1024.0)
    parser.add_argument("--df", type=float, default=0.25)

    parser.add_argument("--data-duration", type=float, default=128.0)
    parser.add_argument(
        "--fft-ref", type=float, default=32.0, help="S1 (Reference) FFT s"
    )
    parser.add_argument("--fft-data", type=float, default=4.0, help="S2 (Data) FFT s")

    parser.add_argument(
        "--ref-psd-offset",
        type=float,
        default=604800.0,
        help="Target lag [s] for Reference PSD. Default: 1 week.",
    )

    parser.add_argument("--use-nds", action="store_true", help="Enable NDS2 fallback")
    parser.add_argument("--max-events", type=int, default=None)

    args = parser.parse_args()
    return AnalysisConfig(
        outdir=args.outdir,
        f_min=args.f_min,
        f_max=args.f_max,
        df=args.df,
        data_duration=args.data_duration,
        fft_ref=args.fft_ref,
        fft_data=args.fft_data,
        max_events=args.max_events,
        verbose=args.verbose,
        ref_psd_offset=args.ref_psd_offset,
        use_nds=args.use_nds,
    )


# ----------------------------------------------------------------------
# Sky Localization Geometry
# ----------------------------------------------------------------------


def get_lal_detector(name: str):
    """Retrieve LAL detector struct using robust index lookups."""
    prefix = name[:2]
    if prefix == "H1":
        idx = getattr(lal, "LALDetectorIndexLHO4K", 4)
    elif prefix == "L1":
        idx = getattr(lal, "LALDetectorIndexLLO4K", 5)
    elif prefix == "V1":
        idx = getattr(lal, "LALDetectorIndexVIRGODIFF", 1)
    else:
        raise ValueError(f"Unknown detector {name}")

    return lal.CachedDetectors[idx]


def solve_sky_shift(
    gps_time: float, ra_true: float, dec_true: float, biases: Dict[str, float]
) -> Tuple[float, float, float]:
    """Estimates shift in sky position (RA, Dec)."""
    if len(biases) < 2:
        return np.nan, np.nan, np.nan

    try:
        detectors = {det: get_lal_detector(det) for det in biases.keys()}
    except Exception as e:
        log.debug(f"Detector lookup failed: {e}")
        return np.nan, np.nan, np.nan

    # "True" delays relative to geocenter
    t_true_rel = {
        d: lal.TimeDelayFromEarthCenter(
            det_struct.location, ra_true, dec_true, gps_time
        )
        for d, det_struct in detectors.items()
    }

    # The "Observed" delay is True Delay + Whitening Bias
    t_obs_rel = {d: t_true_rel[d] + biases[d] for d in biases}

    def objective(x):
        ra_try, dec_try, dt_geo_try = x
        sq_err = 0.0
        for d, det_struct in detectors.items():
            pred_geo_delay = lal.TimeDelayFromEarthCenter(
                det_struct.location, ra_try, dec_try, gps_time
            )
            resid = pred_geo_delay - (t_obs_rel[d] - dt_geo_try)
            sq_err += resid**2
        return sq_err

    x0 = [ra_true, dec_true, 0.0]
    res = minimize(objective, x0, method="Nelder-Mead", tol=1e-8)
    ra_new, dec_new, _ = res.x

    # Angular distance
    d_ra = ra_new - ra_true
    d_dec = dec_new - dec_true
    d_ra = (d_ra + np.pi) % (2 * np.pi) - np.pi

    cos_theta = np.sin(dec_true) * np.sin(dec_new) + np.cos(dec_true) * np.cos(
        dec_new
    ) * np.cos(ra_new - ra_true)
    cos_theta = min(1.0, max(-1.0, cos_theta))
    angle_rad = np.arccos(cos_theta)

    return d_ra, d_dec, np.degrees(angle_rad)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def find_valid_reference_time(
    det: str, target_gps: float, duration: float, search_window: float = 172800.0
) -> float:
    try:
        search_start = int(target_gps - search_window)
        search_end = int(target_gps + search_window)
        flag = f"{det}_DATA"
        segments = get_segments(flag, search_start, search_end)

        for seg in segments:
            if seg[0] <= target_gps and target_gps + duration <= seg[1]:
                return target_gps

        best_time = None
        min_dist = float("inf")
        for seg in segments:
            if (seg[1] - seg[0]) > (duration + 20):
                valid_start = seg[0] + 10 + duration / 2
                valid_end = seg[1] - 10 - duration / 2
                if valid_start > valid_end:
                    continue

                clamped = max(valid_start, min(valid_end, target_gps))
                dist = abs(clamped - target_gps)
                if dist < min_dist:
                    min_dist = dist
                    best_time = clamped
        return best_time
    except Exception:
        return target_gps


def fetch_strain_robust(
    det: str, start: float, end: float, use_nds: bool = False
) -> TimeSeries:
    try:
        return TimeSeries.fetch_open_data(det, start, end, verbose=False, cache=True)
    except Exception as e_osc:
        if use_nds:
            try:
                chan = CHANNELS.get(det, f"{det}:GDS-CALIB_STRAIN")
                return TimeSeries.get(chan, start, end, verbose=False)
            except Exception as e_nds:
                raise RuntimeError(f"GWOSC & NDS2 failed.") from e_nds
        raise e_osc


def generate_lal_waveform(
    m1_sol: float, m2_sol: float, freqs: np.ndarray, df: float, f_min: float
) -> np.ndarray:
    m1_kg = m1_sol * lal.MSUN_SI
    m2_kg = m2_sol * lal.MSUN_SI
    dist = 1.0 * 1e6 * lal.PC_SI
    approximant = lalsim.GetApproximantFromString("IMRPhenomD")

    hp, hc = lalsim.SimInspiralFD(
        m1_kg,
        m2_kg,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        dist,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        df,
        f_min,
        freqs.max() + df,
        0.0,
        None,
        approximant,
    )

    lal_freqs = np.arange(hp.data.length) * hp.deltaF
    h_data = hp.data.data

    amp_interp = np.interp(freqs, lal_freqs, np.abs(h_data), left=0.0, right=0.0)
    return amp_interp.astype(complex)


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------


def get_metadata(event: pd.Series) -> dict:
    name = event.get("name", event.get("commonName", "Unknown"))
    if "GPS" in event:
        gps = event["GPS"]
    elif "tc" in event:
        gps = event["tc"]
    else:
        raise ValueError("No GPS")

    if "mass_1_source" not in event:
        raise ValueError("No mass")
    m1_src, m2_src = event["mass_1_source"], event["mass_2_source"]
    if pd.isna(m1_src):
        raise ValueError("Mass NaN")

    z = event.get("redshift", 0.0)
    if pd.isna(z):
        z = 0.0

    ra = event.get("right_ascension", event.get("ra", np.nan))
    dec = event.get("declination", event.get("dec", np.nan))

    return {
        "name": name,
        "gps": gps,
        "m1": m1_src * (1 + z),
        "m2": m2_src * (1 + z),
        "ra": ra,
        "dec": dec,
    }


def analyze_event(row: pd.Series, cfg: AnalysisConfig) -> Tuple[List[dict], dict]:
    try:
        meta = get_metadata(row)
    except ValueError:
        return [], {}

    dets = []
    try:
        dets = event_detectors(meta["name"])
        if not dets:
            dets = ["H1", "L1"]
    except:
        dets = ["H1", "L1"]

    freqs = np.arange(cfg.f_min, cfg.f_max, cfg.df)
    try:
        h_tilde = generate_lal_waveform(
            meta["m1"], meta["m2"], freqs, cfg.df, cfg.f_min
        )
    except:
        return [], {}

    det_biases = {}
    results = []

    for det in dets:
        try:
            target_ref = meta["gps"] - cfg.ref_psd_offset
            actual_ref = find_valid_reference_time(det, target_ref, cfg.data_duration)
            if actual_ref is None:
                continue

            s_ref = fetch_strain_robust(
                det,
                actual_ref - cfg.data_duration / 2,
                actual_ref + cfg.data_duration / 2,
                cfg.use_nds,
            )
            s_dat = fetch_strain_robust(
                det,
                meta["gps"] - cfg.data_duration / 2,
                meta["gps"] + cfg.data_duration / 2,
                cfg.use_nds,
            )

            psd1_gw = s_ref.psd(fftlength=cfg.fft_ref, method="median", window="hann")
            psd2_gw = s_dat.psd(fftlength=cfg.fft_data, method="median", window="hann")

            psd1 = np.interp(freqs, psd1_gw.frequencies.value, psd1_gw.value)
            psd2 = np.interp(freqs, psd2_gw.frequencies.value, psd2_gw.value)

            if np.any(psd1 <= 0) or np.any(psd2 <= 0) or np.any(np.isnan(psd1)):
                continue

            corrector = PrtPsdDriftCorrection(freqs, psd1, psd2, h_tilde, 4096)
            bias = corrector.correction()

            drift_arr = np.abs(np.sqrt(psd1 / psd2) - 1.0)

            det_results = {
                "event": meta["name"],
                "detector": det,
                "gps": meta["gps"],
                "dt_us": bias.dt1 * 1e6,
                "dphi_rad": bias.dphi1,
                "dsnr_frac": bias.dsnr1,  # New Field
                "eps_max": np.max(drift_arr),
                "eps_mean": np.mean(drift_arr),
                "eps_med": np.median(drift_arr),
                "lag_diff": (meta["gps"] - actual_ref) - cfg.ref_psd_offset,
            }
            results.append(det_results)
            det_biases[det] = bias.dt1

        except Exception:
            continue

    sky_res = {}
    if len(det_biases) >= 2:
        if not pd.isna(meta["ra"]) and not pd.isna(meta["dec"]):
            ra_use, dec_use = meta["ra"], meta["dec"]
            src_type = "catalog"
        else:
            ra_use = np.random.uniform(0, 2 * np.pi)
            dec_use = np.arcsin(np.random.uniform(-1, 1))
            src_type = "random_injection"

        try:
            dra, ddec, dang = solve_sky_shift(meta["gps"], ra_use, dec_use, det_biases)
            sky_res = {
                "event": meta["name"],
                "src_type": src_type,
                "dra_deg": np.degrees(dra),
                "ddec_deg": np.degrees(ddec),
                "total_shift_deg": dang,
                "num_dets": len(det_biases),
            }
        except Exception as e:
            if cfg.verbose:
                log.debug(f"Sky Solve Error ({meta['name']}): {e}")

    return results, sky_res


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


def make_plots(df: pd.DataFrame, df_sky: pd.DataFrame, outdir: str):

    # 1. Bias Hists
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    dt = df["dt_us"].dropna()
    dt = dt[dt.between(-300, 300)]
    ax[0].hist(dt, bins=30, color="#2c7bb6", edgecolor="k", alpha=0.7)
    ax[0].set_xlabel(r"Timing Bias $\delta t$ [$\mu$s]")
    ax[0].set_ylabel("Detector Pairs")
    ax[0].set_title(f"MP-MP Timing Bias (N={len(dt)})")
    ax[0].grid(alpha=0.3)

    dphi = df["dphi_rad"].dropna()
    dphi = dphi[dphi.between(-1.5, 1.5)]
    ax[1].hist(dphi, bins=30, color="#d7191c", edgecolor="k", alpha=0.7)
    ax[1].set_xlabel(r"Phase Bias $\delta \phi$ [rad]")
    ax[1].set_ylabel("Detector Pairs")
    ax[1].set_title(f"MP-MP Phase Bias (N={len(dphi)})")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "bias_distributions.png"))
    plt.close()

    # 2. Epsilon Stats (Log Scale)
    def plot_log_hist(data, name, color, fname):
        plt.figure(figsize=(7, 5))
        valid = data[data > 0]
        if len(valid) == 0:
            return
        log_data = np.log10(valid)
        plt.hist(log_data, bins=30, color=color, edgecolor="k", alpha=0.7)
        plt.xlabel(rf"$\log_{{10}}$({name})")
        plt.ylabel("Detector Pairs")
        plt.title(f"Spectral Drift: {name} (N={len(valid)})")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(outdir, fname))
        plt.close()

    plot_log_hist(df["eps_mean"], "Mean Epsilon", "blue", "epsilon_mean_log.png")
    plot_log_hist(df["eps_med"], "Median Epsilon", "green", "epsilon_median_log.png")
    plot_log_hist(df["eps_max"], "Max Epsilon", "purple", "epsilon_max_log.png")

    # 3. Sky Impact
    if not df_sky.empty:
        sky = df_sky[df_sky["total_shift_deg"] < 10.0]

        if not sky.empty:
            fig = plt.figure(figsize=(14, 5))
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])

            ax0 = fig.add_subplot(gs[0])
            ax0.hist(sky["dra_deg"], bins=20, color="teal", edgecolor="k", alpha=0.7)
            ax0.set_xlabel(r"$\Delta$ RA [deg]")
            ax0.set_ylabel("Events")
            ax0.set_title("RA Shift")
            ax0.grid(alpha=0.3)

            ax1 = fig.add_subplot(gs[1])
            ax1.hist(sky["ddec_deg"], bins=20, color="orange", edgecolor="k", alpha=0.7)
            ax1.set_xlabel(r"$\Delta$ Dec [deg]")
            ax1.set_title("Dec Shift")
            ax1.grid(alpha=0.3)

            ax2 = fig.add_subplot(gs[2])
            sc = ax2.scatter(
                sky["dra_deg"],
                sky["ddec_deg"],
                c=sky["total_shift_deg"],
                cmap="viridis",
                edgecolor="k",
                s=60,
                alpha=0.8,
            )
            plt.colorbar(sc, ax=ax2, label="Total Shift [deg]")
            ax2.axhline(0, c="k", ls=":", alpha=0.5)
            ax2.axvline(0, c="k", ls=":", alpha=0.5)
            ax2.set_xlabel(r"$\Delta$ RA")
            ax2.set_ylabel(r"$\Delta$ Dec")
            ax2.set_title("Localization Bias Sensitivity")
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "sky_impact_analysis.png"))
            plt.close()

    # 4. SNR Drift (New)
    plt.figure(figsize=(7, 5))
    dsnr = df["dsnr_frac"].dropna()
    # Filter outliers
    dsnr = dsnr[dsnr.between(-0.1, 0.1)]
    plt.hist(dsnr, bins=30, color="magenta", edgecolor="k", alpha=0.7)
    plt.xlabel(r"Fractional SNR Drift $\delta\rho/\rho$")
    plt.ylabel("Detector Pairs")
    plt.title(f"Sensitivity Loss (N={len(dsnr)})")
    plt.axvline(0, c="k", ls="--", alpha=0.5)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(outdir, "snr_drift.png"))
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.cur = 0
        self.start = time.time()

    def update(self, txt):
        self.cur += 1
        elapsed = time.time() - self.start
        rem = (self.total - self.cur) * (elapsed / self.cur) if self.cur > 0 else 0
        sys.stdout.write(f"\r[{self.cur}/{self.total}] ETA: {rem/60:.1f}m | {txt:<25}")
        sys.stdout.flush()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    cfg = parse_args()
    if cfg.verbose:
        log.setLevel(logging.DEBUG)
    os.makedirs(cfg.outdir, exist_ok=True)

    log.info("Fetching GWTC-4...")
    try:
        events = EventTable.fetch_open_data("GWTC-4.0")
        df_events = events.to_pandas()
        for c in ["mass_1_source", "mass_2_source", "redshift", "GPS"]:
            if c in df_events:
                df_events[c] = pd.to_numeric(df_events[c], errors="coerce")

        df_events = df_events[df_events["mass_1_source"].notna()]
        if cfg.max_events:
            df_events = df_events.head(cfg.max_events)
        log.info(f"Processing {len(df_events)} events.")
    except Exception as e:
        log.exception("Init failed")
        sys.exit(1)

    all_det_res = []
    all_sky_res = []
    tracker = ProgressTracker(len(df_events))

    print("-" * 60)
    for idx, row in df_events.iterrows():
        tracker.update(str(row.get("commonName", "Event")))
        d_res, s_res = analyze_event(row, cfg)
        all_det_res.extend(d_res)
        if s_res:
            all_sky_res.append(s_res)
    print("\n" + "-" * 60)

    if not all_det_res:
        sys.exit("No results.")

    df = pd.DataFrame(all_det_res)
    df_sky = pd.DataFrame(all_sky_res)

    df.to_csv(os.path.join(cfg.outdir, "bias_results.csv"), index=False)
    if not df_sky.empty:
        df_sky.to_csv(os.path.join(cfg.outdir, "sky_results.csv"), index=False)
        log.info(f"Sky results: {len(df_sky)} events analyzed.")
    else:
        log.warning("No sky results generated (check detector availability).")

    make_plots(df, df_sky, cfg.outdir)
    log.info("Done.")


if __name__ == "__main__":
    main()
