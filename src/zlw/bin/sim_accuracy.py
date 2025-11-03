"""QA: GraceDB Corrections, Datasets, and PSD Evolution — Concise Guide

What this tool does
- Build datasets from GraceDB (events + PSDs) and from rewhitening XML folders.
- Study realistic MPMP bias corrections and PSD evolution over time.
- Save all outputs under an output directory for reuse across subcommands.

How to run
- Module: python -m sgnligo.bin.qa_gdb_corrections <subcommand> [options]
- Console (if installed): sgn-ligo-qa-mpmp <subcommand> [options]

Subcommands (end-to-end workflow)
0) fetch-coinc
   - Download coinc.xml for a single GraceDB event for debugging.
   - Example:
     sgn-ligo-qa-mpmp fetch-coinc --graceid M540474 --out /tmp/M540474-coinc.xml
     --gracedb-playground --verbose

1) events
   - Query GraceDB and build the events dataset (metadata + PSDs).
   - Outputs: events_metadata.csv, event_psds.nc
   - Example:
     sgn-ligo-qa-mpmp events \
       --out-dir gdb_001 \
       --pipeline gstlal --search AllSky \
       --far-max 1e-6 \
       --time-start 2025-01-01 --time-end 2025-08-01 \
       --limit 10000

   - Performance options (optional):
   Add these to split the time window into weekly chunks and query in parallel, while
   combining and deduplicating by superevent:
       --events-chunk-days 7 --events-parallel 6 --events-per-chunk-limit 2000
     Also parallelize coinc.xml download/parsing when building the dataset:
       --events-process-parallel 6 --events-process-batch 100

   - Injection events example (using GraceDB labels):
     sgn-ligo-qa-mpmp events \
       --out-dir gdb_002 \
       --pipeline gstlal \
       --search MDC \
       --gracedb-playground \
       --verbose \
       --limit 10000 \
       --time-start 2025-01-01 --time-end 2025-08-01

2) rewhite
   - Build a rewhitening dataset by scanning local or remote folders of XMLs.
   - Filter by analysis name and time window if desired.
   - Outputs: rewhite_index.csv, rewhite_psds.nc
   - Local example:
     sgn-ligo-qa-mpmp rewhite \
       --out-dir gdb_001 \
       --rewhite-base-dir /data/whitening \
       --rewhite-start 2025-03-01 --rewhite-end 2025-07-01 \
       --rewhite-analysis allsky --rewhite-limit 200

   Remote examples (SSH)
   - Username/password (no key required; needs paramiko installed):
       export QA_SSH_HOST=login.cluster.org
       export QA_SSH_USER=myuser
       export QA_SSH_PASSWORD='mypassword'
       sgn-ligo-qa-mpmp rewhite \
         --out-dir gdb_001 \
         --rewhite-base-dir /lustre/whitening/archive
   - SSH key (recommended for automation):
       export QA_SSH_HOST=ldas-pcdev8.ligo.caltech.edu
       export QA_SSH_USER=james.kennington
       export QA_SSH_IDENTITY=$HOME/.ssh/id_ed25519_ligo
       sgn-ligo-qa-mpmp rewhite --out-dir gdb_001 --rewhite-base-dir
       /home/gstlalcbc.offline/observing/4/c/rewhiten/

   O4 structure note (a/b/c subruns)
   - On CIT, O4 rewhitening is organized as: /home/gstlalcbc.offline/observing/4/{
   a|b|c}/rewhiten
     Each rewhiten folder contains GPS-window subfolders like 1434412818-604800,
     and inside
     an analysis subfolder such as alice, bob, charlie (with median_psd).
   - Analysis name mapping used here:
     Early Warning -> alice; AllSky -> bob; SSM -> charlie; For O4a, legacy names
     Edward (AllSky CIT) and Jacob (AllSky ICDS) are also accepted and mapped to bob.
   - You can specify the observing root and subruns directly without writing each
   base dir:
       sgn-ligo-qa-mpmp rewhite \
         --out-dir gdb_001 \
         --remote-host ldas-pcdev8.ligo.caltech.edu \
         --rewhite-observing-root /home/gstlalcbc.offline/observing/4 \
         --rewhite-subruns a,b,c \
         --rewhite-analysis alice
   - Equivalent explicit base-dir usage for O4c only:
       sgn-ligo-qa-mpmp rewhite \
         --out-dir gdb_001 \
         --remote-host ldas-pcdev8.ligo.caltech.edu \
         --rewhite-base-dir /home/gstlalcbc.offline/observing/4/c/rewhiten \
         --rewhite-analysis alice
   - Example concrete file observed on CIT (O4c):
       /home/gstlalcbc.offline/observing/4/c/rewhiten/1434412818-604800/alice
       /median_psd/14344/H1L1-GSTLAL_MEDIAN_PSD-1434412818-604800.xml.gz

3) biases
   - Compute MPMP bias corrections for each event using the most recent
     rewhitening PSD (per IFO) before the event time.
   - Inputs required: events and rewhite datasets in the same --out-dir.
   - Outputs: bias_corrections.csv, hist_dt1.png, hist_dphi1.png
   - Example:
     sgn-ligo-qa-mpmp biases --out-dir gdb_001 --ifo H1 --fmin 20 --fmax 1024
     sgn-ligo-qa-mpmp biases --out-dir gdb_011 --fmin 20 --fmax 1024

4) psd-evolution
   - Build a date-indexed PSD history from events + rewhite and compute
    adjacent-date epsilon metrics over time.
   - Inputs required: events and rewhite datasets.
   - Output: psd_evolution_summary.csv
   - Example:
     sgn-ligo-qa-mpmp psd-evolution --out-dir gdb_001

5) skymap
   - Estimate first-order changes in sky localization (RA/Dec) for each event due to MPMP bias corrections.
   - Inputs required: events and rewhite datasets in the same --out-dir; bias_corrections.csv (computed automatically if missing).
   - Notes:
     - If RA/Dec are missing (common for real events), the tool auto-triangulates a baseline sky position from per-IFO arrival times in coinc.xml (H1/L1/V1/K1 end_time[_ns]) and Coinc_end_time.
     - Uses only first-order terms DT1 and DPhi1; converts phase via dt_phi = -DPhi1/(2*pi*fref).
     - Requires at least two IFOs with corrections per event; set --fref to choose the reference frequency (default 100 Hz).
   - Outputs: skymap_deltas.csv, hist_dRA_deg.png, hist_dDec_deg.png, hist_dAng_deg.png, scatter_dRA_dDec.png, scatter_dRA_dDec_colored_by_median_epsilon.png (requires PsdEpsilonMedian in bias_corrections.csv)
   - Example:
     sgn-ligo-qa-mpmp skymap --out-dir gdb_020 --fref 100

Skymap RA/Dec estimation (how it works)
- Baseline sky position:
  - If the event already has RA/Dec in coinc/sim tables, use them.
  - Otherwise triangulate from per-IFO arrival times parsed from coinc.xml SnglInspiral:
    coarse grid over (RA, Dec) using lal.TimeDelayFromEarthCenter for each IFO to
    minimize squared residuals of observed inter-detector delays, then refine with
    one linear least-squares step around the best grid point.
- First-order bias-induced shifts (this analysis):
  - Per IFO i, convert the MPMP corrections to an equivalent time shift:
    dt_equiv_i = DT1_i + (-DPhi1_i)/(2*pi*fref).
  - Work with time differences relative to a reference detector r to eliminate the
    unknown common clock term: b_i = dt_equiv_i - dt_equiv_r.
  - Compute partial derivatives of inter-detector delays wrt RA and Dec at the baseline
    sky using finite differences of lal.TimeDelayFromEarthCenter. This forms a small
    linear system A x = b with x = [dRA, dDec]. Solve by least squares.
  - Convert dRA, dDec to degrees and report a small-angle total shift dAng ≈ sqrt((dRA*cos(Dec))^2 + dDec^2).
- Justification/assumptions:
  - Uses standard light-travel-time delays between detector locations and the incoming
    plane wave; partials are well behaved locally, so a first-order linearization is
    accurate for small corrections. Phase-to-time equivalence follows dt = -dphi/(2*pi*fref).

6) analyze
   - Run both analyses (biases and psd-evolution) using existing datasets.
   - Example:
     sgn-ligo-qa-mpmp analyze --out-dir gdb_001 --ifo H1 --fmin 20 --fmax 1024

7) all
   - Full pipeline: events -> (optional) rewhite -> analyses.
   - If --rewhite-base-dir is omitted, only events are created and analyses are skipped.
   - Example:
     sgn-ligo-qa-mpmp all \
       --out-dir gdb_001 \
       --pipeline gstlal --search AllSky \
       --time-start 2025-03-01 --time-end 2025-07-01 \
       --rewhite-base-dir /data/whitening

GraceDB authentication
- Obtain a read-only token and ensure your environment is configured:
    htgettoken -a vault.ligo.org -i igwn --scopes=gracedb.read
- Docs: https://ligo-gracedb.readthedocs.io/en/latest/user_guide.html#getting-a-token

GraceDB server selection
- By default, queries go to production: https://gracedb.ligo.org/api/
- To use the GraceDB Playground (useful for MDC injections), pass --gracedb-playground.
  Example:
    sgn-ligo-qa-mpmp events \
      --out-dir gdb_002 \
      --gracedb-playground \
      --labels INJ \
      --time-start 2025-01-01 --time-end 2025-08-01

Remote access details
- Env vars: QA_SSH_HOST, QA_SSH_USER, QA_SSH_IDENTITY (key), QA_SSH_PASSWORD (password).
- Behavior:
  - The tool now reuses a single SSH session (Paramiko) for the entire rewhite scan
  when possible, greatly reducing Duo prompts.
  - If QA_SSH_PASSWORD is set and paramiko is available, the tool uses password-based
  SSH/SFTP.
  - If QA_SSH_IDENTITY is set (key) and Paramiko is available, it will also reuse one
  session.
  - Otherwise it uses your system ssh/scp, honoring QA_SSH_IDENTITY when provided.
- Multiplexing (CLI ssh/scp): set QA_SSH_CONTROL=1 to enable
ControlMaster/ControlPersist so only the first ssh prompts Duo.
  - Optional: QA_SSH_CONTROL_PATH to override control socket path (default:
  ~/.ssh/cm-%r@%h:%p)
- Only one representative XML is fetched per GPS folder (prefers MEDIAN_PSD and
matching analysis).

Reduce Duo prompts (recommended)
- Best: use a key and/or set QA_SSH_IDENTITY and let the tool reuse a single session.
- If you must use system ssh/scp, set QA_SSH_CONTROL=1 so later commands reuse the
first connection without re-prompting Duo.
- Password mode: set QA_SSH_PASSWORD and install paramiko; the tool opens one session
and reuses it for all queries and copies.

Data files written
- events_metadata.csv: per-event metadata/intrinsics (columns always include m1, m2,
chi1z, chi2z, MChirp, ChiEff; values are NaN if unavailable).
- event_psds.nc: PSD cube for events [event, ifo, frequency].
- rewhite_index.csv and rewhite_psds.nc: rewhitening records and PSD cube [record,
ifo, frequency].
- bias_corrections.csv: per-(event, IFO) MPMP corrections and PSD epsilon metrics (PsdEpsilonMean, PsdEpsilonMedian, PsdEpsilonMax).
- psd_evolution_summary.csv: epsilon statistics over time.

End-to-end quickstart
1) Build GraceDB events dataset (pick a time window)
2) Build rewhitening dataset (local or remote)
3) Run biases to get MPMP corrections
4) Run psd-evolution to summarize PSD changes
5) Optionally re-run analyze on different IFO/fmin/fmax without rebuilding datasets

Troubleshooting
- Missing datasets: run the appropriate subcommand(s) first using the same --out-dir.
- GraceDB auth errors: ensure a valid scitoken is installed.
- Heterogeneous frequency grids: interpolation to a common grid is automatic.
- Remote listing/copy failures: try setting QA_SSH_IDENTITY or QA_SSH_PASSWORD (
requires paramiko).
- Use --verbose to print every directory searched and each XML file found/selected
during rewhite scans (local and remote).
"""

import collections
import datetime
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse

import lal
import lalsimulation as lalsim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lal import series
from gwpy import time
from ligo.gracedb.rest import GraceDb

from igwn_ligolw import ligolw
from igwn_ligolw import lsctables
from igwn_ligolw import utils as ligolw_utils
from pandas import DataFrame
from xarray import DataArray
import xarray as xr

from zlw.kernels import MPMPCorrection

# Default output directory for results (can be overridden by CLI)
OUT_DIR: str = "gdb_001"
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------- GraceDB utilities --------------------


def _split_time_ranges(
    start: Optional[datetime.datetime], end: Optional[datetime.datetime], days: int
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """Split [start, end] into contiguous [a,b] ranges of length `days`.
    If days <= 0 or start/end missing, returns a single [(start, end)] if provided.
    """
    if (start is None) or (end is None) or days <= 0:
        return [(start, end)] if (start is not None and end is not None) else []
    if start > end:
        start, end = end, start
    rngs: List[Tuple[datetime.datetime, datetime.datetime]] = []
    cur = start
    delta = datetime.timedelta(days=days)
    while cur < end:
        nxt = min(cur + delta, end)
        rngs.append((cur, nxt))
        cur = nxt
    return rngs


def _reduce_and_limit_events(
    event_data: pd.DataFrame, limit: Optional[int], reduce_by_superevent: bool
) -> pd.DataFrame:
    """Apply superevent reduction, sort by time, and cap to `limit` rows.
    This helps better honor user-requested limits after deduplication.
    """
    df = event_data.copy()
    if reduce_by_superevent and "superevent" in df.columns:
        try:
            df = df.loc[df.groupby("superevent")["snr"].idxmax()].reset_index(drop=True)
        except Exception:
            # Fallback: drop_duplicates keeping first occurrence
            df = df.drop_duplicates(subset=["superevent"]).reset_index(drop=True)
    # Sort by time if present
    if "time" in df.columns:
        try:
            df = df.sort_values(by="time").reset_index(drop=True)
        except Exception:
            pass
    if limit is not None and limit > 0:
        df = df.head(limit)
    return df


def _get_gracedb_service_url(use_playground: bool = False) -> str:
    """Return the GraceDB service URL for production or playground.
    Production: https://gracedb.ligo.org/api/
    Playground: https://gracedb-playground.ligo.org/api/
    """
    return (
        "https://gracedb-playground.ligo.org/api/"
        if use_playground
        else "https://gracedb.ligo.org/api/"
    )


# -------------------- Persistence and CLI utilities --------------------


def _ensure_out_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _date_from_str(s: Optional[str]) -> Optional[datetime.datetime]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    # Accept YYYY-MM-DD or full ISO 8601
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
    # Fallback: try fromtimestamp if numeric
    try:
        return datetime.datetime.fromtimestamp(float(s))
    except Exception:
        pass
    raise ValueError(f"Unrecognized date format: {s}")


def _normalize_intrinsics_df(df: DataFrame) -> DataFrame:
    """Ensure standard intrinsic columns exist and are numeric.
    Adds missing columns [m1, m2, chi1z, chi2z, MChirp, ChiEff] with NaN,
    coerces to float, and computes derived MChirp/ChiEff if not present
    and inputs are available.
    """
    cols = ["m1", "m2", "chi1z", "chi2z", "MChirp", "ChiEff"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    # Coerce to numeric floats
    for c in ["m1", "m2", "chi1z", "chi2z", "MChirp", "ChiEff"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Derive if possible
    m1 = df["m1"].astype(float)
    m2 = df["m2"].astype(float)
    chi1 = df["chi1z"].astype(float)
    chi2 = df["chi2z"].astype(float)
    # Compute MChirp where missing and masses are finite
    need_mchirp = df["MChirp"].isna() & m1.notna() & m2.notna()
    with np.errstate(invalid="ignore", divide="ignore"):
        mchirp = np.power(m1 * m2, 3.0 / 5.0) / np.power(m1 + m2, 1.0 / 5.0)
    df.loc[need_mchirp, "MChirp"] = mchirp[need_mchirp]
    # Compute ChiEff where missing and inputs finite
    denom = m1 + m2
    need_chi = (
        df["ChiEff"].isna()
        & m1.notna()
        & m2.notna()
        & chi1.notna()
        & chi2.notna()
        & (denom > 0)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        chieff = (m1 * chi1 + m2 * chi2) / denom
    df.loc[need_chi, "ChiEff"] = chieff[need_chi]
    return df


def save_events_dataset(
    out_dir: str,
    events_df: DataFrame,
    event_psds: DataArray,
    timing_df: Optional[pd.DataFrame] = None,
) -> None:
    _ensure_out_dir(out_dir)
    events_df.to_csv(os.path.join(out_dir, "events_metadata.csv"), index=False)
    # Timing must come from coinc.xml (SnglInspiral); do not attempt to synthesize from events_df
    if timing_df is None or not isinstance(timing_df, pd.DataFrame) or timing_df.empty:
        raise ValueError(
            "events_timing is required and must be extracted from coinc.xml; timing_df is missing/empty"
        )
    timing_df.to_csv(os.path.join(out_dir, "events_timing.csv"), index=False)
    # Save full PSD cube to NetCDF for reuse in later steps
    event_psds.to_netcdf(os.path.join(out_dir, "event_psds.nc"))


def load_events_dataset(out_dir: str) -> Tuple[DataFrame, DataArray]:
    df_path = os.path.join(out_dir, "events_metadata.csv")
    nc_path = os.path.join(out_dir, "event_psds.nc")
    events_df = pd.read_csv(df_path) if os.path.exists(df_path) else pd.DataFrame()
    event_psds = (
        xr.load_dataarray(nc_path)
        if os.path.exists(nc_path)
        else DataArray(np.zeros((0, 0, 0)), dims=["event", "ifo", "frequency"])
    )  # type: ignore
    return events_df, event_psds


def load_events_timing(out_dir: str) -> DataFrame:
    """Load per-IFO timing data if available."""
    path = os.path.join(out_dir, "events_timing.csv")
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame(columns=["graceid", "ifo", "time_gps", "phase"])
    return pd.DataFrame(columns=["graceid", "ifo", "time_gps", "phase"])


def save_rewhite_dataset(
    out_dir: str, rewhite_df: DataFrame, rewhite_psds: DataArray
) -> None:
    _ensure_out_dir(out_dir)
    rewhite_df.to_csv(os.path.join(out_dir, "rewhite_index.csv"), index=False)
    rewhite_psds.to_netcdf(os.path.join(out_dir, "rewhite_psds.nc"))


def load_rewhite_dataset(out_dir: str) -> Tuple[DataFrame, DataArray]:
    df_path = os.path.join(out_dir, "rewhite_index.csv")
    nc_path = os.path.join(out_dir, "rewhite_psds.nc")
    rewhite_df = pd.read_csv(df_path) if os.path.exists(df_path) else pd.DataFrame()
    rewhite_psds = (
        xr.load_dataarray(nc_path)
        if os.path.exists(nc_path)
        else DataArray(np.zeros((0, 0, 0)), dims=["record", "ifo", "frequency"])
    )  # type: ignore
    return rewhite_df, rewhite_psds


GdbEvent = collections.namedtuple(
    "GdbEvent",
    [
        "graceid",
        "pipeline",
        "search",
        "far",
        "time",
        "labels",
        "superevent",
        "snr",
        "m1",
        "m2",
        "chi",
    ],
)


def build_gracedb_query(
    pipeline: Optional[str] = None,
    search: Optional[str] = None,
    far_min: Optional[float] = None,
    far_max: Optional[float] = None,
    time_start: Optional[datetime.datetime] = None,
    time_end: Optional[datetime.datetime] = None,
    labels: Optional[list[str]] = None,
) -> str:
    """
    Construct a GraceDB event query string from the given filters.

    Args:
        pipeline:
            str, default None, Name of the pipeline (e.g. "gstlal", "cwb").
        far_min:
            float, default None, Minimum FAR threshold (in inverse seconds).
        far_max:
            float, default None, Maximum FAR threshold (in inverse seconds).
        time_start:
            datetime, default None, Start time for the event search.
        time_end:
            datetime, default None, End time for the event search.
        labels:
            list of str, default None, Labels to filter events by.

    Returns:
        str, A valid GraceDB query string (space-joined).
    """
    parts: list[str] = []

    if pipeline:
        parts.append(f"pipeline: {pipeline}")

    if search:
        parts.append(f"search: {search}")

    if far_min is not None:
        parts.append(f"far >= {far_min}")
    if far_max is not None:
        parts.append(f"far < {far_max}")

    if time_start is not None and time_end is not None:
        parts.append(
            f"gpstime: {time.to_gps(time_start)} .." f" {time.to_gps(time_end)}"
        )
    elif time_start is not None:
        parts.append(f"gpstime >= {time.to_gps(time_start)}")
    elif time_end is not None:
        parts.append(f"gpstime <= {time.to_gps(time_end)}")

    if labels:
        if isinstance(labels, str):
            label_str = labels
        else:
            label_str = ",".join(labels)
        parts.append(f"label: {label_str}")

    # join with spaces → implicit AND
    return " ".join(parts)


def get_events(
    pipeline: Optional[str] = None,
    search: Optional[str] = None,
    far_min: Optional[float] = None,
    far_max: Optional[float] = None,
    time_start: Optional[datetime.datetime] = None,
    time_end: Optional[datetime.datetime] = None,
    labels: Optional[list[str]] = None,
    limit: int = 100,
    reduce_by_superevent: bool = True,
    verbose: bool = False,
    gracedb_playground: bool = False,
    events_chunk_days: int = 0,
    events_parallel: int = 4,
    events_per_chunk_limit: Optional[int] = None,
) -> DataFrame:
    """Query GraceDB for recorded events with optional chunking/parallelism.

    When verbose=True, print the server, queries, and per-chunk counts. By default,
    behavior is unchanged (single query). If events_chunk_days > 0 and both
    time_start/time_end are provided, the interval is split into chunks of the given
    size (days) and queried in parallel, then combined. Deduplication (by superevent)
    is applied over the union, and the final limit is applied after deduplication.
    """
    import sys

    service_url = _get_gracedb_service_url(gracedb_playground)

    def _fetch_range(
        t0: Optional[datetime.datetime],
        t1: Optional[datetime.datetime],
        idx: Optional[int] = None,
        total: Optional[int] = None,
    ) -> List[dict]:
        # Create client per worker to avoid cross-thread issues
        client = GraceDb(service_url=service_url)
        q = build_gracedb_query(
            pipeline=pipeline,
            search=search,
            far_min=far_min,
            far_max=far_max,
            time_start=t0,
            time_end=t1,
            labels=labels,
        )
        if verbose:
            # Avoid deprecated client attributes; use the known service_url
            prefix = (
                "[events] Chunk" if (t0 is not None and t1 is not None) else "[events]"
            )
            idx_str = (
                f" [{idx}/{total}]" if (idx is not None and total is not None) else ""
            )
            if t0 is not None and t1 is not None:
                print(
                    f"{prefix}{idx_str} query: {q} | max_results="
                    f"{events_per_chunk_limit or limit} | server={service_url}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"{prefix} query: {q} | max_results="
                    f"{events_per_chunk_limit or limit} | server={service_url}",
                    file=sys.stderr,
                )
        iterator = client.events(q, max_results=(events_per_chunk_limit or limit))
        return list(iterator)

    # Decide between single query or chunked
    raw_events: List[dict] = []
    if events_chunk_days > 0 and (time_start is not None) and (time_end is not None):
        ranges = _split_time_ranges(time_start, time_end, events_chunk_days)
        if verbose:
            print(
                f"[events] Chunking {len(ranges)} ranges of {events_chunk_days} day(s)",
                file=sys.stderr,
            )
        if events_parallel and events_parallel > 1 and len(ranges) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=events_parallel) as ex:
                total = len(ranges)
                futs = {}
                for idx, (r0, r1) in enumerate(ranges, start=1):
                    fut = ex.submit(_fetch_range, r0, r1, idx, total)
                    futs[fut] = (r0, r1, idx, total)
                for fut in as_completed(futs):
                    r0, r1, idx, total = futs[fut]
                    try:
                        evs = fut.result()
                        raw_events.extend(evs)
                        if verbose:
                            print(
                                f"[events] Chunk [{idx}/{total}] [{r0}..{r1}] -> "
                                f"{len(evs)} events",
                                file=sys.stderr,
                            )
                    except Exception as e:
                        if verbose:
                            print(
                                f"[events] Chunk [{idx}/{total}] [{r0}..{r1}] failed:"
                                f" {e}",
                                file=sys.stderr,
                            )
        else:
            total = len(ranges)
            for idx, (r0, r1) in enumerate(ranges, start=1):
                try:
                    evs = _fetch_range(r0, r1, idx, total)
                    raw_events.extend(evs)
                    if verbose:
                        print(
                            f"[events] Chunk [{idx}/{total}] [{r0}..{r1}] -> "
                            f"{len(evs)} events",
                            file=sys.stderr,
                        )
                except Exception as e:
                    if verbose:
                        print(
                            f"[events] Chunk [{idx}/{total}] [{r0}..{r1}] failed: {e}",
                            file=sys.stderr,
                        )
    else:
        # Single query path (default behavior)
        raw_events = _fetch_range(time_start, time_end)

    # Convert to GdbEvent list
    events = [
        GdbEvent(
            graceid=e["graceid"],
            pipeline=e.get("pipeline"),
            search=e.get("search"),
            far=e.get("far"),
            time=time.from_gps(e["gpstime"]) if "gpstime" in e else None,
            labels=e.get("labels", []),
            superevent=e.get("superevent", ""),
            snr=(e.get("extra_attributes", {}).get("CoincInspiral", {}).get("snr")),
            m1=None,
            m2=None,
            chi=None,
        )
        for e in raw_events
    ]

    event_data = pd.DataFrame(
        [list(event) for event in events], columns=GdbEvent._fields
    )

    # Reduce then final-limit to better satisfy requested count
    event_data = _reduce_and_limit_events(
        event_data, limit=limit, reduce_by_superevent=reduce_by_superevent
    )

    return event_data


def get_events_dataset(
    event_ids: List[str],
    verbose: bool = True,
    gracedb_playground: bool = False,
    events_process_parallel: int = 1,
    events_process_batch: int = 50,
) -> Tuple[DataFrame, DataArray, pd.DataFrame]:
    """Build datasets for a list of GraceDB events.

    Returns
    -------
    (events_df, psds_xr, timing_df)
        - events_df: pandas DataFrame with one row per event containing
          metadata and intrinsics (m1, m2, spins if available).
        - psds_xr: xarray DataArray with dims [event, ifo, frequency]
          containing the event PSDs. Coordinate 'event' matches events_df['graceid'].
        - timing_df: pandas DataFrame with per-IFO timing/phase extracted directly
          from coinc.xml SnglInspiral per event with columns [graceid, ifo, time_gps, phase].
    """
    # Storage
    rows: List[dict] = []
    # We'll build PSDs into a dict per event: {ifo: (freqs, psd)}
    event_psds: List[dict] = []
    # Per-event IFO timing rows extracted directly from coinc.xml
    timing_rows_map: Dict[str, List[dict]] = {}

    # Per-thread clients are created inside the worker to avoid cross-thread issues
    common_freqs: Optional[np.ndarray] = None
    ifo_list: List[str] = []

    # Parallelize per-event processing in batches for better throughput
    N = len(event_ids)
    if verbose:
        import sys

        print(
            f"[events-ds] Processing {N} events with "
            f"{max(1, int(events_process_parallel))}"
            f" workers (batch={max(1, int(events_process_batch))})",
            file=sys.stderr,
        )

    def _process_one(
        eid: str,
    ) -> Tuple[
        str,
        Optional[dict],
        Optional[np.ndarray],
        Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]],
        List[dict],
    ]:
        # One client per worker for thread-safety
        cli = GraceDb(service_url=_get_gracedb_service_url(gracedb_playground))
        try:
            files = cli.files(eid).json()
            meta_row: dict = {"graceid": eid}
            freqs: Optional[np.ndarray] = None
            packaged: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Try coinc.xml first for both PSDs and intrinsics
                if "coinc.xml" in files:
                    path = Path(tmpdirname) / eid / "coinc.xml"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "wb") as fid:
                        raw = cli.files(eid, "coinc.xml")
                        fid.write(raw.data)
                    xmldoc = ligolw_utils.load_filename(path.as_posix(), verbose=False)
                    try:
                        # Preferred path: extract PSDs and intrinsics together
                        meta_row_ex, freqs_ex, psd_by_ifo = extract_psd_data_from_coinc(
                            xmldoc
                        )
                        meta_row.update(meta_row_ex or {})
                        freqs = freqs_ex
                        packaged = {
                            ifo: (freqs_ex, psd_by_ifo[ifo]) for ifo in psd_by_ifo
                        }
                    except Exception:
                        # If PSD parsing fails, still salvage intrinsics from
                        # sngl/sim tables
                        try:
                            intr = extract_intrinsics_only(xmldoc)
                            meta_row.update(intr)
                        except Exception:
                            pass
                    # Build per-IFO timing/phase rows directly from meta_row (already populated from SnglInspiral)
                    try:
                        # Collect candidate IFOs from packaged PSDs and meta_row keys
                        ifos_from_psd = (
                            list(packaged.keys()) if isinstance(packaged, dict) else []
                        )
                        ifos_from_meta: list[str] = []
                        for k in list(meta_row.keys()):
                            if not isinstance(k, str):
                                continue
                            # Keys like 'ChirpTime_<IFO>'
                            if k.startswith("ChirpTime_"):
                                ifos_from_meta.append(k.split("_", 1)[1])
                            # Keys like '<IFO>_impulse_time', '<IFO>_end_time', '<IFO>_coa_phase'
                            parts = k.split("_", 1)
                            if len(parts) == 2 and parts[0] in ("H1", "L1", "V1", "K1"):
                                ifos_from_meta.append(parts[0])
                        ifos_all = sorted(
                            set([str(x) for x in (ifos_from_psd + ifos_from_meta)])
                        )

                        def _get(key: str):
                            return (
                                meta_row.get(key)
                                if isinstance(meta_row, dict)
                                else None
                            )

                        for ifo_val in ifos_all:
                            t_gps = None
                            # Prefer ChirpTime_<IFO>, then <IFO>_impulse_time_gps
                            for key in (
                                f"ChirpTime_{ifo_val}",
                                f"{ifo_val}_impulse_time_gps",
                            ):
                                v = _get(key)
                                if v is not None:
                                    try:
                                        t_gps = float(pd.to_numeric(v))
                                        if np.isfinite(t_gps):
                                            break
                                    except Exception:
                                        t_gps = None
                            # Next, compose from impulse_time + ns
                            if t_gps is None or not np.isfinite(t_gps):
                                try:
                                    sec = _get(f"{ifo_val}_impulse_time")
                                    nsec = _get(f"{ifo_val}_impulse_time_ns")
                                    if sec is not None:
                                        s = float(pd.to_numeric(sec))
                                        ns = (
                                            float(pd.to_numeric(nsec))
                                            if nsec is not None
                                            else 0.0
                                        )
                                        # treat (0,0) as missing
                                        if (s != 0.0) or (ns != 0.0):
                                            t_gps = s + ns * 1e-9
                                except Exception:
                                    t_gps = None
                            # Fallback to end_time (_gps or sec+ns)
                            if t_gps is None or not np.isfinite(t_gps):
                                try:
                                    v = _get(f"{ifo_val}_end_time_gps")
                                    if v is not None:
                                        t_gps = float(pd.to_numeric(v))
                                except Exception:
                                    t_gps = None
                                if t_gps is None or not np.isfinite(t_gps):
                                    try:
                                        sec = _get(f"{ifo_val}_end_time")
                                        nsec = _get(f"{ifo_val}_end_time_ns")
                                        if sec is not None:
                                            s = float(pd.to_numeric(sec))
                                            ns = (
                                                float(pd.to_numeric(nsec))
                                                if nsec is not None
                                                else 0.0
                                            )
                                            t_gps = s + ns * 1e-9
                                    except Exception:
                                        t_gps = None
                            if (
                                (t_gps is None)
                                or (not np.isfinite(t_gps))
                                or (float(t_gps) <= 1.0)
                            ):
                                continue
                            # Phase: prefer ChirpPhase_<IFO>, then <IFO>_coa_phase
                            ph = None
                            for key in (
                                f"ChirpPhase_{ifo_val}",
                                f"{ifo_val}_coa_phase",
                            ):
                                v = _get(key)
                                if v is not None:
                                    try:
                                        ph = float(pd.to_numeric(v))
                                    except Exception:
                                        ph = None
                                    break
                            timing_row = {
                                "graceid": eid,
                                "ifo": str(ifo_val),
                                "time_gps": float(t_gps),
                                "phase": (
                                    float(ph)
                                    if (ph is not None and np.isfinite(ph))
                                    else np.nan
                                ),
                            }
                            timing_rows_map.setdefault(eid, []).append(timing_row)
                    except Exception:
                        pass
                    # Fallback: if no timing rows were added from meta/psd keys, parse SnglInspiral table directly
                    try:
                        if (eid not in timing_rows_map) or (
                            not timing_rows_map.get(eid)
                        ):
                            sngl_tbl = lsctables.SnglInspiralTable.get_table(xmldoc)
                            if len(sngl_tbl) > 0:
                                for r in sngl_tbl:
                                    ifo_val = getattr(r, "ifo", None)
                                    if not ifo_val:
                                        continue
                                    # Prefer impulse_time unless it is (0,0); else end_time
                                    it = getattr(r, "impulse_time", None)
                                    itns = getattr(r, "impulse_time_ns", None)
                                    et = getattr(r, "end_time", None)
                                    etns = getattr(r, "end_time_ns", None)
                                    t_gps = None
                                    try:
                                        if (
                                            (it is not None)
                                            and (itns is not None)
                                            and (
                                                (float(it) != 0.0)
                                                or (float(itns) != 0.0)
                                            )
                                        ):
                                            t_gps = float(it) + float(itns) * 1e-9
                                        elif (it is not None) and (float(it) != 0.0):
                                            t_gps = float(it)
                                        elif (et is not None) and (etns is not None):
                                            t_gps = float(et) + float(etns) * 1e-9
                                        elif et is not None:
                                            t_gps = float(et)
                                    except Exception:
                                        t_gps = None
                                    if (
                                        (t_gps is None)
                                        or (not np.isfinite(t_gps))
                                        or (float(t_gps) <= 1.0)
                                    ):
                                        continue
                                    # Phase
                                    ph = None
                                    for name in ("coa_phase", "coaphase", "phase"):
                                        if hasattr(r, name):
                                            try:
                                                ph = float(
                                                    pd.to_numeric(getattr(r, name))
                                                )
                                            except Exception:
                                                ph = None
                                            break
                                    timing_rows_map.setdefault(eid, []).append(
                                        {
                                            "graceid": eid,
                                            "ifo": str(ifo_val),
                                            "time_gps": float(t_gps),
                                            "phase": (
                                                float(ph)
                                                if (ph is not None and np.isfinite(ph))
                                                else np.nan
                                            ),
                                        }
                                    )
                    except Exception:
                        pass
                # If intrinsics still missing, look for auxiliary inspiral attachments
                if (
                    ("m1" not in meta_row)
                    or ("m2" not in meta_row)
                    or ("chi1z" not in meta_row)
                    or ("chi2z" not in meta_row)
                ):
                    try:
                        fnames = list(files.keys()) if isinstance(files, dict) else []
                        cand = None
                        # Prefer sngl_inspiral over sim_inspiral/injection
                        for key in fnames:
                            k = str(key).lower()
                            if "sngl_inspiral" in k and (
                                k.endswith(".xml") or k.endswith(".xml.gz")
                            ):
                                cand = key
                                break
                        if cand is None:
                            for key in fnames:
                                k = str(key).lower()
                                if (
                                    "sim_inspiral" in k
                                    or "sim-inspiral" in k
                                    or "injection" in k
                                ) and (k.endswith(".xml") or k.endswith(".xml.gz")):
                                    cand = key
                                    break
                        if cand is not None:
                            aux_path = Path(tmpdirname) / eid / Path(str(cand)).name
                            aux_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(aux_path, "wb") as af:
                                aresp = cli.files(eid, cand)
                                af.write(aresp.data)
                            aux_doc = ligolw_utils.load_filename(
                                aux_path.as_posix(), verbose=False
                            )
                            meta_only = extract_intrinsics_only(aux_doc)
                            for k, v in meta_only.items():
                                meta_row.setdefault(k, v)
                    except Exception:
                        pass
            # Add event time if available
            try:
                evt_json = cli.event(eid).json()
                evt_gps = evt_json.get("gpstime")
                if evt_gps is not None:
                    meta_row.setdefault("event_time", time.from_gps(evt_gps))
            except Exception:
                pass
            return eid, meta_row, freqs, packaged, timing_rows_map.get(eid, [])
        except Exception:
            return eid, {"graceid": eid}, None, None, []

    # Batch submission and collection
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results_map: Dict[
        str,
        Tuple[
            Optional[dict],
            Optional[np.ndarray],
            Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]],
            List[dict],
        ],
    ] = {}
    if events_process_parallel is None or events_process_parallel < 1:
        events_process_parallel = 1
    if events_process_batch is None or events_process_batch < 1:
        events_process_batch = 50

    batches = [
        event_ids[i : i + events_process_batch]
        for i in range(0, N, events_process_batch)
    ]
    total_batches = len(batches)
    for bidx, batch in enumerate(batches, start=1):
        if verbose:
            import sys

            print(
                f"[events-ds] Batch [{bidx}/{total_batches}] size={len(batch)} submit",
                file=sys.stderr,
            )
        with ThreadPoolExecutor(max_workers=events_process_parallel) as ex:
            futs = {ex.submit(_process_one, eid): eid for eid in batch}
            ok = 0
            fail = 0
            for fut in as_completed(futs):
                eid = futs[fut]
                try:
                    eid_out, meta_row, freqs, packaged, trows = fut.result()
                    results_map[eid_out] = (meta_row, freqs, packaged, trows)
                    ok += 1
                except Exception:
                    results_map[eid] = ({"graceid": eid}, None, None, [])
                    fail += 1
            if verbose:
                import sys

                print(
                    f"[events-ds] Batch [{bidx}/{total_batches}] completed: {ok} ok, "
                    f"{fail} failed",
                    file=sys.stderr,
                )

    # Reconstruct in original order and harmonize frequency grids
    event_ids_with_psd: List[str] = []
    timing_rows_all: List[dict] = []
    for idx, event_id in enumerate(event_ids):
        meta_row, freqs, packaged, trows = results_map.get(
            event_id, ({"graceid": event_id}, None, None, [])
        )
        # Always keep the metadata row so intrinsics are persisted even if PSDs are
        # missing
        if meta_row is None:
            meta_row = {"graceid": event_id}
        rows.append(meta_row)
        # Accumulate timing rows (if any)
        try:
            if trows:
                # De-duplicate by (ifo, time_gps) per event
                seen = set()
                for tr in trows:
                    key = (
                        tr.get("ifo"),
                        float(pd.to_numeric(tr.get("time_gps"), errors="coerce")),
                    )
                    if (key[0] is None) or (not np.isfinite(key[1])):
                        continue
                    if key in seen:
                        continue
                    seen.add(key)
                    timing_rows_all.append(
                        {
                            "graceid": str(tr.get("graceid", event_id)),
                            "ifo": str(tr.get("ifo")),
                            "time_gps": float(key[1]),
                            "phase": (
                                float(pd.to_numeric(tr.get("phase"), errors="coerce"))
                                if pd.notna(tr.get("phase"))
                                else np.nan
                            ),
                        }
                    )
        except Exception:
            pass
        # Handle PSDs only when available
        if freqs is None or packaged is None:
            continue
        event_psds.append(packaged)
        event_ids_with_psd.append(str(meta_row.get("graceid", event_id)))
        if common_freqs is None:
            # Establish common grid and IFOs from first successful event
            common_freqs = freqs
            ifo_list = sorted(packaged.keys())
        else:
            same_grid = (len(freqs) == len(common_freqs)) and np.allclose(
                common_freqs, freqs
            )
            if not same_grid:
                # Interpolate this event's PSDs onto the established common grid
                new_psds = {}
                for ifo, (fr, psd) in packaged.items():
                    new_psds[ifo] = np.interp(
                        common_freqs, fr, psd, left=np.nan, right=np.nan
                    )
                event_psds[-1] = {
                    ifo: (common_freqs, new_psds[ifo]) for ifo in new_psds
                }
            # Harmonize IFO list over time
            ifo_list = sorted(set(ifo_list).union(packaged.keys()))

    # Build DataFrame
    events_df = pd.DataFrame(rows)
    # Normalize and guarantee intrinsic columns exist for downstream analyses
    events_df = _normalize_intrinsics_df(events_df)

    # Build timing DataFrame directly from coinc.xml extraction
    timing_df = pd.DataFrame(
        timing_rows_all, columns=["graceid", "ifo", "time_gps", "phase"]
    ).reset_index(drop=True)

    # Build DataArray [event, ifo, frequency]
    if common_freqs is None:
        # No events
        psds_xr = DataArray(
            np.zeros((0, 0, 0)), dims=["event", "ifo", "frequency"]
        )  # type: ignore
        return events_df, psds_xr, timing_df

    E = len(event_psds)
    F = len(common_freqs)
    I = len(ifo_list)
    data = np.full((E, I, F), np.nan, dtype=float)

    for eidx, psd_dict in enumerate(event_psds):
        for iidx, ifo in enumerate(ifo_list):
            if ifo in psd_dict:
                freqs_e, psd_e = psd_dict[ifo]
                same_grid = (len(freqs_e) == len(common_freqs)) and np.allclose(
                    freqs_e, common_freqs
                )
                if same_grid:
                    data[eidx, iidx, :] = psd_e
                else:
                    data[eidx, iidx, :] = np.interp(
                        common_freqs, freqs_e, psd_e, left=np.nan, right=np.nan
                    )

    psds_xr = DataArray(
        data,
        dims=["event", "ifo", "frequency"],
        coords={
            "event": event_ids_with_psd,
            "ifo": ifo_list,
            "frequency": common_freqs,
        },
        name="psd",
    )

    return events_df, psds_xr, timing_df


def extract_psd_data_from_coinc(
    xmldoc: ligolw.LIGO_LW,
) -> Tuple[dict, np.ndarray, Dict[str, np.ndarray]]:
    """Extract intrinsics and PSDs from a coinc.xml igwn-ligolw document.

    Returns
    -------
    meta_row: dict
        Contains event intrinsics if available (m1, m2, spins) and any
        other useful metadata that can be extracted.
    freqs: np.ndarray
        Frequency grid common to all PSDs (attempted). If different across
        IFOs, chooses the first and interpolates others.
    psd_by_ifo: Dict[str, np.ndarray]
        Mapping IFO -> PSD values on freqs grid.
    """
    meta_row: Dict[str, object] = {}

    # Read intrinsics: prefer SnglInspiral (loudest row), then fall back to
    # SimInspiral (common for MDC)
    try:
        sngl_tbl = lsctables.SnglInspiralTable.get_table(xmldoc)
        if len(sngl_tbl) > 0:
            row = max(sngl_tbl, key=lambda r: getattr(r, "snr", 0.0))
            # Preserve common intrinsics and useful derived values if present
            for k in (
                "mass1",
                "mass2",
                "spin1z",
                "spin2z",
                "snr",
                "event_time",
                # MDC-style extras we want to carry along when available
                "mchirp",
                "mtotal",
                "eta",
                "chi",
                "f_final",
            ):
                if hasattr(row, k):
                    v = getattr(row, k)
                    keymap = {
                        "mass1": "m1",
                        "mass2": "m2",
                        "spin1z": "chi1z",
                        "spin2z": "chi2z",
                        "event_time": "event_time",
                        "snr": "snr",
                        "mchirp": "MChirp",
                        "mtotal": "MTotal",
                        "eta": "Eta",
                        # "chi" is typically an effective spin proxy; map to ChiEff
                        # if not already set
                        "chi": "ChiEff",
                        "f_final": "F_final",
                    }
                    key_out = keymap.get(k, k)
                    # Do not overwrite if value was already filled from a preferred
                    # source
                    meta_row.setdefault(key_out, v)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read SnglInspiral table for event from " f"coinc.xml"
        ) from e

    # Compute derived parameters when possible
    try:
        m1 = float(meta_row.get("m1")) if "m1" in meta_row else float("nan")
        m2 = float(meta_row.get("m2")) if "m2" in meta_row else float("nan")
        chi1z = float(meta_row.get("chi1z")) if "chi1z" in meta_row else float("nan")
        chi2z = float(meta_row.get("chi2z")) if "chi2z" in meta_row else float("nan")
        if np.isfinite(m1) and np.isfinite(m2):
            mchirp = np.power(m1 * m2, 3.0 / 5.0) / np.power(m1 + m2, 1.0 / 5.0)
            meta_row.setdefault("MChirp", mchirp)
            if np.isfinite(chi1z) and np.isfinite(chi2z) and (m1 + m2) > 0:
                chieff = (m1 * chi1z + m2 * chi2z) / (m1 + m2)
                meta_row.setdefault("ChiEff", chieff)
    except Exception:
        pass

    # Also extract extrinsic parameters where possible (per-IFO and coinc)
    try:
        _fill_extrinsics_from_xml(xmldoc, meta_row)
    except Exception:
        pass

    # Read PSDs: try standard 'psd' root first; if empty, fall back to top-level
    # Some MDC/Playground files store PSD series at the top level (no 'psd' root).
    psd_dict = {}
    last_err: Optional[Exception] = None
    for root_name in ("psd", None, ""):
        try:
            # series.read_psd_xmldoc accepts root_name; None/"" should select top-level
            psd_dict = series.read_psd_xmldoc(
                xmldoc, root_name=root_name
            )  # type: ignore
        except Exception as e:
            last_err = e
            psd_dict = {}
        if psd_dict:
            break
    if not psd_dict:
        # Persist the offending document for troubleshooting
        try:
            with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmpf:
                ligolw_utils.write_filename(xmldoc, tmpf.name)
                path_saved = tmpf.name
        except Exception:
            path_saved = "<unavailable>"
        if last_err is not None:
            raise RuntimeError(
                f"Failed to read PSDs from coinc.xml (tried roots 'psd', None, "
                f"'') ; saved to {path_saved}"
            ) from last_err
        raise ValueError(
            "No PSDs found in coinc.xml document (after trying fallback roots)"
        )

    # Build frequency grid and PSD arrays
    ifos = sorted(psd_dict.keys())
    # Start with the first IFO's grid
    first = psd_dict[ifos[0]]
    f0 = getattr(first, "f0", 0.0)
    df = getattr(first, "deltaF", None) or getattr(
        first, "deltaT", None
    )  # deltaF expected
    if df is None:
        # Fallback: infer from array length and assume starts at f0
        arr = np.asarray(first.data.data)
        # default to 1 Hz grid if cannot infer
        df = 1.0 if len(arr) < 2 else (arr.shape[-1] and 1.0)
    length = first.data.length
    freqs = f0 + np.arange(length) * (first.deltaF)

    psd_by_ifo: Dict[str, np.ndarray] = {}
    for ifo in ifos:
        fs = psd_dict[ifo]
        f0_i = getattr(fs, "f0", 0.0)
        df_i = getattr(fs, "deltaF", first.deltaF)
        length_i = fs.data.length
        freqs_i = f0_i + np.arange(length_i) * df_i
        vals = np.array(fs.data.data, copy=True)
        # Interpolate to common grid if necessary (handle differing lengths safely)
        same_grid = (len(freqs_i) == len(freqs)) and np.allclose(freqs_i, freqs)
        if not same_grid:
            vals = np.interp(freqs, freqs_i, vals, left=np.nan, right=np.nan)
        psd_by_ifo[ifo] = vals

    return meta_row, freqs, psd_by_ifo


def extract_intrinsics_only(xmldoc: ligolw.LIGO_LW) -> dict:
    """Extract only intrinsic parameters from a LIGO_LW document.
    Reads directly from SnglInspiral (masses/spins present there); no fallbacks or
    recovery.
    Computes MChirp/ChiEff from masses/spins when possible.
    """
    meta: Dict[str, object] = {}
    # 1) Prefer SnglInspiral (typically has per-IFO masses/spins and SNR)
    try:
        sngl_tbl = lsctables.SnglInspiralTable.get_table(xmldoc)
        if len(sngl_tbl) > 0:
            row = max(sngl_tbl, key=lambda r: getattr(r, "snr", 0.0))
            for k in (
                "mass1",
                "mass2",
                "spin1z",
                "spin2z",
                "mchirp",
                "mtotal",
                "eta",
                "chi",
                "snr",
            ):
                if hasattr(row, k):
                    v = getattr(row, k)
                    keymap = {
                        "mass1": "m1",
                        "mass2": "m2",
                        "spin1z": "chi1z",
                        "spin2z": "chi2z",
                        "mchirp": "MChirp",
                        "mtotal": "MTotal",
                        "eta": "Eta",
                        "chi": "ChiEff",
                        "snr": "snr",
                    }
                    key_out = keymap.get(k, k)
                    meta.setdefault(key_out, v)
    except Exception:
        pass

    # 4) Derived parameters
    try:
        m1 = float(meta.get("m1")) if "m1" in meta else float("nan")
        m2 = float(meta.get("m2")) if "m2" in meta else float("nan")
        chi1z = float(meta.get("chi1z")) if "chi1z" in meta else float("nan")
        chi2z = float(meta.get("chi2z")) if "chi2z" in meta else float("nan")
        if np.isfinite(m1) and np.isfinite(m2):
            meta.setdefault(
                "MChirp", (np.power(m1 * m2, 3.0 / 5.0) / np.power(m1 + m2, 1.0 / 5.0))
            )
            if np.isfinite(chi1z) and np.isfinite(chi2z) and (m1 + m2) > 0:
                meta.setdefault("ChiEff", ((m1 * chi1z + m2 * chi2z) / (m1 + m2)))
    except Exception:
        pass
    # Also try to fill extrinsics
    try:
        _fill_extrinsics_from_xml(xmldoc, meta)
    except Exception:
        pass
    return meta


def generate_template_fd(
    m1: float, m2: float, f_bounds: Tuple[int, int], duration: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an IMRPhenomD template in the freq domain for given masses.

    Args:
        m1:
            float, mass of the first component in solar masses.
        m2:
            float, mass of the second component in solar masses.
        f_bounds:
            tuple of (f_min, f_max) in Hz, frequency bounds for the template.
        duration:
            int, duration in seconds for the template generation.

    Returns:
        freqs (np.ndarray), htilde (np.ndarray)
    """
    delta_f = 1.0 / duration
    f_min, f_max = f_bounds
    # set up LAL params
    approx = lalsim.SimInspiralGetApproximantFromString("IMRPhenomD")
    params = {
        "m1": m1 * lal.MSUN_SI,
        "m2": m2 * lal.MSUN_SI,
        "S1x": 0.0,
        "S1y": 0.0,
        "S1z": 0.0,
        "S2x": 0.0,
        "S2y": 0.0,
        "S2z": 0.0,
        "distance": 1e6 * lal.PC_SI,
        "inclination": 0.0,
        "phiRef": 0.0,
        "longAscNodes": 0.0,
        "eccentricity": 0.0,
        "meanPerAno": 0.0,
        "deltaF": delta_f,
        "f_min": f_min,
        "f_max": f_max,
        "f_ref": 0.0,
        "LALparams": None,
        "approximant": approx,
    }
    hp_fd, _ = lalsim.SimInspiralFD(**params)
    f0, df = hp_fd.f0, hp_fd.deltaF
    length = hp_fd.data.length
    freqs = f0 + np.arange(length) * df
    htilde = np.array(hp_fd.data.data, copy=True)
    return freqs, htilde


# -------------------- Rewhitening history utilities --------------------


def _normalize_analysis_name(analysis: Optional[str]) -> Optional[str]:
    """Map user-facing analysis keywords to rewhiten subfolder names.
    Accepts values like 'AllSky', 'Early Warning', 'SSM', or nicknames 'Bob',
    'Alice', 'Charlie'.
    Returns the canonical subfolder key to search for (lowercase), or None if not
    provided.
    Also supports deprecated O4a names Edward (AllSky CIT) and Jacob (AllSky ICDS).
    """
    if not analysis:
        return None
    a = str(analysis).strip().lower()
    # direct nicknames
    if a in {"alice", "bob", "charlie"}:
        return a
    # deprecated
    if a in {"edward", "jacob"}:
        return "bob"  # old AllSky variants should map under AllSky group
    # descriptive names
    if a in {"allsky", "all-sky", "all_sky"}:
        return "bob"
    if a in {"early warning", "early_warning", "early-warning", "ew"}:
        return "alice"
    if a in {"ssm", "single-stage-match", "single stage match"}:
        return "charlie"
    # unknown, pass through string as-is
    return a


def _path_contains_analysis(path_str: str, analysis_key: Optional[str]) -> bool:
    if not analysis_key:
        return True
    key = analysis_key.lower()
    p = path_str.lower()
    # prefer exact subfolder match, but accept anywhere in path
    return f"/{key}/" in p or p.endswith(f"/{key}") or (key in p)


def _run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    import subprocess

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, err = proc.communicate()
    return proc.returncode, out, err


class _RemoteSession:
    """Reusable SSH session using paramiko for reduced MFA prompts.
    Open once per run and reuse for all remote exec and file transfers.
    """

    def __init__(self, host: str, user: Optional[str] = None):
        self.host = host
        self.user = user
        self.client = None  # type: ignore
        self.sftp = None  # type: ignore

    def open(self) -> bool:
        try:
            import paramiko  # type: ignore
        except Exception:
            return False
        try:
            password = os.environ.get("QA_SSH_PASSWORD")
            identity = os.environ.get("QA_SSH_IDENTITY")
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # Build auth kwargs
            kwargs = {"hostname": self.host}
            if self.user:
                kwargs["username"] = self.user
            if password:
                kwargs["password"] = password
            if identity and not password:
                # Try loading key file (ed25519/rsa auto)
                try:
                    pkey = None
                    try:
                        pkey = paramiko.Ed25519Key.from_private_key_file(identity)
                    except Exception:
                        pkey = paramiko.RSAKey.from_private_key_file(identity)
                    kwargs["pkey"] = pkey
                except Exception:
                    pass
            # Connect and prepare SFTP
            self.client.connect(**kwargs)
            self.sftp = self.client.open_sftp()
            return True
        except Exception:
            self.close()
            return False

    def exec(self, command: str) -> tuple[int, str, str]:
        if not self.client:
            return 1, "", "no session"
        try:
            stdin, stdout, stderr = self.client.exec_command(command)
            out = stdout.read().decode()
            err = stderr.read().decode()
            rc = 0 if not err else 0  # remote tools often write warnings to stderr
            return rc, out, err
        except Exception as e:
            return 1, "", str(e)

    def sftp_get(self, remote_path: str, local_path: str) -> bool:
        if not self.sftp:
            return False
        try:
            self.sftp.get(remote_path, local_path)
            return True
        except Exception:
            return False

    def close(self) -> None:
        try:
            if self.sftp:
                self.sftp.close()
        except Exception:
            pass
        try:
            if self.client:
                self.client.close()
        except Exception:
            pass
        self.sftp = None
        self.client = None


def _ssh_control_opts() -> list[str]:
    if str(os.environ.get("QA_SSH_CONTROL", "")).lower() in {"1", "true", "yes"}:
        control_path = os.environ.get(
            "QA_SSH_CONTROL_PATH", os.path.expanduser("~/.ssh/cm-%r@%h:%p")
        )
        return [
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={control_path}",
            "-o",
            "ControlPersist=600",
        ]
    return []


def _ssh_list_gps_dirs(
    host: str,
    base_dir: str,
    user: Optional[str] = None,
    session: Optional[_RemoteSession] = None,
    verbose: bool = False,
) -> List[str]:
    """List immediate GPS-named directories under base_dir on remote host.
    Respects optional env QA_SSH_IDENTITY for SSH identity file.
    If a session is provided or QA_SSH_PASSWORD is set with paramiko available,
    uses a single SSH session to avoid repeated MFA prompts.
    """
    login = f"{user}@{host}" if user else host
    ident = os.environ.get("QA_SSH_IDENTITY")
    password = os.environ.get("QA_SSH_PASSWORD")

    if verbose:
        print(
            f"[rewhite][remote] Listing GPS directories under: {base_dir} on {host} ("
            f"user={user or ''})"
        )
    out = ""
    if session and session.client:
        shell = f'shopt -s nullglob; for d in {base_dir}/*/; do basename "$d"; done'
        rc, out, err = session.exec(shell)
        if rc != 0 and not out:
            raise RuntimeError(f"ssh list failed: {err.strip()}")
    elif password:
        try:
            import paramiko  # type: ignore

            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=host, username=user, password=password)
            shell = (
                f'shopt -s nullglob; for d in {base_dir}/*/; do basename "$d"; ' f"done"
            )
            stdin, stdout, stderr = client.exec_command(shell)
            out = stdout.read().decode()
            err = stderr.read().decode()
            rc = 0 if not err else 1 if not out else 0
            client.close()
            if rc != 0:
                raise RuntimeError(f"ssh list failed: {err.strip()}")
        except ImportError:
            password = None
        except Exception as e:
            raise RuntimeError(f"password SSH list failed: {e}")

    if not out and password is None and not (session and session.client):
        # Use POSIX find to list directories one level down via ssh CLI
        base_cmd = (
            ["ssh"]
            + _ssh_control_opts()
            + (["-i", ident] if ident else [])
            + [login, "bash", "-lc"]
        )
        cmd = base_cmd + [
            f'shopt -s nullglob; for d in {base_dir}/*/; do basename "$d"; done'
        ]
        rc, out, err = _run_cmd(cmd)
        if rc != 0:
            raise RuntimeError(f"ssh list failed: {err.strip() or rc}")

    names = [line.strip().rstrip("/") for line in out.splitlines() if line.strip()]
    if verbose:
        print(f"[rewhite][remote] Candidate directory names: {names}")
    # Keep only those with a 9-11 digit GPS substring
    import re

    keep: List[str] = []
    for name in names:
        if re.search(r"\d{9,11}", name):
            full = os.path.join(base_dir, name)
            keep.append(full)
            if verbose:
                print(f"[rewhite][remote] Keep: {full}")
        elif verbose:
            print(f"[rewhite][remote] Drop (no GPS token): {name}")
    return keep


def _ssh_find_xml_in_folder(
    host: str,
    folder: str,
    user: Optional[str],
    analysis_key: Optional[str],
    session: Optional[_RemoteSession] = None,
    verbose: bool = False,
) -> Optional[str]:
    """Find a representative XML(.gz) under the GPS folder on remote host.
    Prefer paths under the analysis subfolder if provided. Respects env QA_SSH_IDENTITY.
    If a session is provided, it is used to avoid repeated MFA prompts.
    """
    login = f"{user}@{host}" if user else host
    ident = os.environ.get("QA_SSH_IDENTITY")
    password = os.environ.get("QA_SSH_PASSWORD")

    # Find XML files under folder; prefer MEDIAN_PSD and analysis path
    # Escape parentheses so the shell passes them to find (avoid subshell grouping)
    find_cmd = (
        f"find {folder} -type f "
        + "\\( -name '*.xml' -o -name '*.xml.gz' \\) -printf '%p\\n' 2>/dev/null"
    )

    if verbose:
        print(f"[rewhite][remote] Searching XMLs under: {folder}")
    out = ""
    if session and session.client:
        rc, out, err = session.exec(find_cmd)
        if rc != 0 and not out:
            if verbose and err:
                print(f"[rewhite][remote] find error under {folder}: {err.strip()}")
            return None
    elif password:
        try:
            import paramiko  # type: ignore

            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=host, username=user, password=password)
            stdin, stdout, stderr = client.exec_command(find_cmd)
            out = stdout.read().decode()
            if verbose:
                err = stderr.read().decode()
                if err:
                    print(
                        f"[rewhite][remote] find stderr under {folder}: {err.strip()}"
                    )
            client.close()
        except ImportError:
            password = None
        except Exception as e:
            if verbose:
                print(f"[rewhite][remote] find exec failed under {folder}: {e}")
            return None

    if not out and password is None and not (session and session.client):
        base_cmd = (
            ["ssh"]
            + _ssh_control_opts()
            + (["-i", ident] if ident else [])
            + [login, "bash", "-lc"]
        )
        cmd = base_cmd + [find_cmd]
        rc, out_cli, err = _run_cmd(cmd)
        if rc != 0:
            if verbose and err:
                print(f"[rewhite][remote] find error under {folder}: {err.strip()}")
            return None
        out = out_cli

    files = [line.strip() for line in out.splitlines() if line.strip()]
    if verbose:
        print(f"[rewhite][remote] Found {len(files)} XML candidates under {folder}")
        if files:
            for f in files if len(files) <= 5 else files[:5]:
                print(f"[rewhite][remote]   - {f}")
            if len(files) > 5:
                print(f"[rewhite][remote]   ... (+{len(files) - 5} more)")
    if not files:
        return None

    # Score files: prefer ones matching analysis and MEDIAN_PSD
    def score(p: str) -> tuple[int, int, int]:
        s1 = 1 if _path_contains_analysis(p, analysis_key) else 0
        s2 = 1 if "median_psd" in p.lower() else 0
        s3 = 1 if "median" in p.lower() else 0
        return (s1, s2, s3)

    files.sort(key=score, reverse=True)
    if verbose and files:
        print(f"[rewhite][remote] Selected: {files[0]}")
    return files[0]


def _scp_copy(
    host: str,
    remote_path: str,
    local_path: str,
    user: Optional[str],
    session: Optional[_RemoteSession] = None,
    verbose: bool = False,
) -> bool:
    login = f"{user}@{host}" if user else host
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if verbose:
        print(f"[rewhite][remote] Copying: {host}:{remote_path} -> {local_path}")
    if session and session.sftp:
        ok = session.sftp_get(remote_path, local_path)
        if verbose:
            print(
                f"[rewhite][remote] SFTP copy {'OK' if ok else 'FAILED'} for "
                f"{remote_path}"
            )
        return ok
    password = os.environ.get("QA_SSH_PASSWORD")
    if password:
        try:
            import paramiko  # type: ignore

            transport = paramiko.Transport((host, 22))
            transport.connect(username=user, password=password)
            sftp = paramiko.SFTPClient.from_transport(transport)
            sftp.get(remote_path, local_path)
            sftp.close()
            transport.close()
            return True
        except Exception:
            return False
    ident = os.environ.get("QA_SSH_IDENTITY")
    cmd = (
        ["scp"]
        + _ssh_control_opts()
        + (["-i", ident] if ident else [])
        + [f"{login}:{remote_path}", local_path]
    )
    rc, out, err = _run_cmd(cmd)
    return rc == 0


def index_rewhitening_folders(
    base_dir: str, analysis: Optional[str] = None, verbose: bool = False
) -> DataFrame:
    """Index available rewhitening folders in a local/remote-like directory.

    Expects directory names that contain a GPS integer (e.g., 1371234567) and
    optionally analysis name substrings. Returns a DataFrame with columns:
      - folder: full path
      - gps: int
      - date: datetime
      - analysis: str (best-effort from folder name)
    """
    p = Path(base_dir)
    if verbose:
        print(f"[rewhite][local] Scanning base dir: {p}")
    rows = []
    for d in p.iterdir():
        if not d.is_dir():
            if verbose:
                print(f"[rewhite][local] Skipping non-directory: {d}")
            continue
        if analysis and analysis not in d.name:
            if verbose:
                print(
                    f"[rewhite][local] Skipping (analysis filter '{analysis}') dir: "
                    f"{d.name}"
                )
            continue
        # Extract first 10+ digit number as GPS
        import re

        m = re.search(r"(\d{9,11})", d.name)
        if not m:
            if verbose:
                print(f"[rewhite][local] No GPS-like token in: {d.name}")
            continue
        gps = int(m.group(1))
        try:
            dt = time.from_gps(gps)
        except Exception:
            dt = None
        if verbose:
            print(f"[rewhite][local] Found folder: {d.name} -> gps={gps} date={dt}")
        rows.append(
            {"folder": d.as_posix(), "gps": gps, "date": dt, "analysis": d.name}
        )
    df = pd.DataFrame(rows).sort_values("gps").reset_index(drop=True)
    if verbose:
        print(f"[rewhite][local] Indexed {len(df)} GPS folders under {p}")
    return df


def build_rewhitening_dataset(
    base_dir: object,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
    analysis: Optional[str] = None,
    limit: Optional[int] = None,
    remote_host: Optional[str] = None,
    remote_user: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[DataFrame, DataArray]:
    """Scan rewhitening folders and parse XMLs into a dataset similar to events.

    Supports remote access via SSH/SCP when remote_host (or env QA_SSH_HOST) is
    provided.
    In remote mode, the tool lists GPS folders on the remote host, finds a
    representative XML per folder (preferring analysis-specific and MEDIAN_PSD
    files), copies it to a temporary local file via scp, and parses PSDs.

    Returns (index_df, psds_xr) where index_df has columns [record_id, date, folder]
    and psds_xr has dims [record, ifo, frequency].
    """
    # Allow env to supply defaults
    remote_host = remote_host or os.environ.get("QA_SSH_HOST")
    remote_user = remote_user or os.environ.get("QA_SSH_USER")

    analysis_key = _normalize_analysis_name(analysis)

    # Normalize base_dirs as a list of strings
    if isinstance(base_dir, (list, tuple, set)):
        base_dirs: List[str] = [str(b) for b in base_dir]
    else:
        base_dirs = [str(base_dir)]

    if remote_host:
        # Prepare a reusable SSH session (if possible) to minimize MFA prompts
        session = _RemoteSession(remote_host, remote_user)
        have_session = session.open()
        if verbose:
            print(
                f"[rewhite] Remote host: {remote_host}, user: "
                f"{remote_user or ''}. "
                f"Session: {'yes' if have_session else 'no'}"
            )
        # Remote listing of GPS directories aggregated across base_dirs
        rows_idx = []
        import re

        for bd in base_dirs:
            try:
                remote_folders = _ssh_list_gps_dirs(
                    remote_host,
                    bd,
                    user=remote_user,
                    session=session if have_session else None,
                    verbose=verbose,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to list remote folders via ssh for {bd}: {e}"
                )
            if verbose:
                print(
                    f"[rewhite] Base dir {bd}: found {len(remote_folders)} candidate "
                    f"GPS folders"
                )
            for folder in remote_folders:
                m = re.search(r"(\d{9,11})", folder)
                if not m:
                    continue
                gps = int(m.group(1))
                try:
                    dt = time.from_gps(gps)
                except Exception:
                    dt = None
                rows_idx.append(
                    {
                        "folder": folder,
                        "gps": gps,
                        "date": dt,
                        "analysis": analysis_key or "",
                    }
                )
        idx = pd.DataFrame(rows_idx).sort_values("gps").reset_index(drop=True)
        if verbose:
            print(f"[rewhite] Total candidate folders after aggregation: {len(idx)}")
    else:
        # Local aggregation across base_dirs
        idx_parts: List[DataFrame] = []
        for bd in base_dirs:
            idx_parts.append(
                index_rewhitening_folders(bd, analysis=analysis, verbose=verbose)
            )
        idx = pd.concat(idx_parts, ignore_index=True) if idx_parts else pd.DataFrame()
        if not idx.empty:
            idx = idx.sort_values("gps").reset_index(drop=True)

    if start_date is not None:
        idx = idx[idx["date"] >= start_date]
    if end_date is not None:
        idx = idx[idx["date"] <= end_date]
    if limit is not None:
        idx = idx.head(limit)

    rows: List[dict] = []
    records_psds: List[dict] = []
    common_freqs: Optional[np.ndarray] = None
    ifo_list: List[str] = []

    # Work directory for remote files if needed
    tmpdir_cm = tempfile.TemporaryDirectory() if remote_host else None
    tmpdir = Path(tmpdir_cm.name) if tmpdir_cm else None

    try:
        for _, r in idx.iterrows():
            if remote_host:
                # Find a representative XML file path on remote, then copy it
                remote_folder = r["folder"]
                xml_remote = _ssh_find_xml_in_folder(
                    remote_host,
                    remote_folder,
                    user=remote_user,
                    analysis_key=analysis_key,
                    session=session if remote_host else None,
                    verbose=verbose,
                )
                if not xml_remote:
                    if verbose:
                        print(
                            f"[rewhite] No XML found under remote folder: "
                            f"{remote_folder}"
                        )
                    continue
                if verbose:
                    print(f"[rewhite] Selected representative XML: {xml_remote}")
                # Local destination path
                rec_name = Path(remote_folder).name
                assert tmpdir is not None
                # Preserve original filename (keep .gz if present) so igwn-ligolw can
                # auto-detect compression
                local_xml = tmpdir / Path(xml_remote).name
                # If remote file is .gz, still copy and let parser handle via
                # igwn-ligolw
                ok = _scp_copy(
                    remote_host,
                    xml_remote,
                    local_xml.as_posix(),
                    user=remote_user,
                    session=session if remote_host else None,
                )
                if not ok:
                    if verbose:
                        print(f"[rewhite] Copy failed: {xml_remote} -> {local_xml}")
                    continue
                try:
                    xmldoc = ligolw_utils.load_filename(
                        local_xml.as_posix(), verbose=False
                    )
                except Exception as e:
                    if verbose:
                        print(f"[rewhite] Parse failed for {local_xml}: {e}")
                    continue
                meta_row, freqs, psd_by_ifo = extract_psd_data_from_coinc(xmldoc)
                record_id = f"{rec_name}:{Path(xml_remote).name}"
                rows.append(
                    {
                        "record_id": record_id,
                        "date": r["date"],
                        "folder": remote_folder,
                        **meta_row,
                    }
                )
                records_psds.append(
                    {ifo: (freqs, psd_by_ifo[ifo]) for ifo in psd_by_ifo}
                )
            else:
                folder_path = Path(r["folder"])
                # Prefer xml and xml.gz files
                xml_files = list(folder_path.glob("*.xml")) + list(
                    folder_path.glob("*.xml.gz")
                )
                if verbose:
                    print(
                        f"[rewhite][local] Folder: {folder_path} -> found "
                        f"{len(xml_files)} XML(s)"
                    )
                    for f in xml_files if len(xml_files) <= 5 else xml_files[:5]:
                        print(f"[rewhite][local]   - {f}")
                    if len(xml_files) > 5:
                        print(f"[rewhite][local]   ... (+{len(xml_files) - 5} more)")
                if not xml_files:
                    continue
                xml_path = xml_files[0]
                if verbose:
                    print(f"[rewhite][local] Selected representative: {xml_path}")
                try:
                    xmldoc = ligolw_utils.load_filename(
                        xml_path.as_posix(), verbose=False
                    )
                except Exception:
                    continue
                meta_row, freqs, psd_by_ifo = extract_psd_data_from_coinc(xmldoc)
                record_id = f"{folder_path.name}:{xml_path.name}"
                rows.append(
                    {
                        "record_id": record_id,
                        "date": r["date"],
                        "folder": r["folder"],
                        **meta_row,
                    }
                )
                records_psds.append(
                    {ifo: (freqs, psd_by_ifo[ifo]) for ifo in psd_by_ifo}
                )

            if records_psds:
                freqs = next(iter(records_psds[-1].values()))[0]
                if common_freqs is None:
                    common_freqs = freqs
                    ifo_list = sorted({ifo for ifo in records_psds[-1].keys()})
                else:
                    same_grid = (len(freqs) == len(common_freqs)) and np.allclose(
                        common_freqs, freqs
                    )
                    if not same_grid:
                        new_psds = {}
                        for ifo, (fr, psd) in records_psds[-1].items():
                            new_psds[ifo] = np.interp(
                                common_freqs, fr, psd, left=np.nan, right=np.nan
                            )
                        records_psds[-1] = {
                            ifo: (common_freqs, new_psds[ifo]) for ifo in new_psds
                        }
                    ifo_list = sorted(set(ifo_list).union(records_psds[-1].keys()))
    finally:
        if tmpdir_cm is not None:
            tmpdir_cm.cleanup()
        try:
            if remote_host:
                # session may exist if remote_host set
                session.close()  # type: ignore
        except Exception:
            pass

    index_df = pd.DataFrame(rows)
    if common_freqs is None or len(records_psds) == 0:
        psds_xr = DataArray(
            np.zeros((0, 0, 0)), dims=["record", "ifo", "frequency"]
        )  # type: ignore
        return index_df, psds_xr

    R = len(records_psds)
    I = len(ifo_list)
    F = len(common_freqs)
    data = np.full((R, I, F), np.nan, dtype=float)
    for ridx, psd_dict in enumerate(records_psds):
        for iidx, ifo in enumerate(ifo_list):
            if ifo in psd_dict:
                freqs_e, psd_e = psd_dict[ifo]
                same_grid = (len(freqs_e) == len(common_freqs)) and np.allclose(
                    freqs_e, common_freqs
                )
                if same_grid:
                    data[ridx, iidx, :] = psd_e
                else:
                    data[ridx, iidx, :] = np.interp(
                        common_freqs, freqs_e, psd_e, left=np.nan, right=np.nan
                    )

    psds_xr = DataArray(
        data,
        dims=["record", "ifo", "frequency"],
        coords={
            "record": index_df["record_id"].tolist(),
            "ifo": ifo_list,
            "frequency": common_freqs,
        },
        name="psd",
    )
    return index_df, psds_xr


# -------------------- Studies --------------------


def _epsilon(psd1: np.ndarray, psd2: np.ndarray) -> Tuple[float, float]:
    """Compute simple epsilon metrics between two PSDs on the same grid.
    Returns (mean_abs, max_abs) of |sqrt(psd2/psd1) - 1| over finite points.
    Uses only bins where both PSDs are finite and strictly positive to avoid
    invalid divisions. If fewer than 3 valid bins exist, returns NaN.
    """
    # Ensure 1D float arrays to avoid shape/broadcast issues (e.g., row vectors)
    p1 = np.asarray(psd1, dtype=float).ravel()
    p2 = np.asarray(psd2, dtype=float).ravel()
    # If lengths mismatch, compare over the common prefix to be safe
    if p1.size != p2.size:
        n = int(min(p1.size, p2.size))
        p1 = p1[:n]
        p2 = p2[:n]
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.isfinite(p1) & np.isfinite(p2) & (p1 > 0) & (p2 > 0)
        if not np.any(m) or m.sum() < 3:
            return float("nan"), float("nan")
        ratio = np.sqrt(p2[m] / p1[m])
        eps = np.abs(ratio - 1.0)
        if eps.size == 0:
            return float("nan"), float("nan")
        return float(np.nanmean(eps)), float(np.nanmax(eps))


def _epsilon_robust(
    psd1: np.ndarray, psd2: np.ndarray, smooth_window: int = 7, pclip: float = 99.0
) -> Tuple[float, float]:
    """Robust epsilon against narrow line shifts.

    Strategy:
    - Compute eps = |sqrt(psd2/psd1) - 1| per-frequency bin.
    - Smooth eps with a short moving-average window to reduce the impact of
      single-bin (narrow-line) excursions that move slightly in frequency.
    - Report:
      * mean over the smoothed epsilon (EpsilonMeanRobust)
      * high-percentile of the raw epsilon (EpsilonPXX with pclip, default 99th),
        which is less sensitive than the absolute max.

    Uses only bins where both PSDs are finite and strictly positive. If fewer than
    3 valid bins exist, returns NaN for both metrics.

    Returns (mean_smoothed, pclip_percentile).
    """
    if smooth_window < 1:
        smooth_window = 1
    # Ensure 1D float arrays
    p1 = np.asarray(psd1, dtype=float).ravel()
    p2 = np.asarray(psd2, dtype=float).ravel()
    if p1.size != p2.size:
        n = int(min(p1.size, p2.size))
        p1 = p1[:n]
        p2 = p2[:n]
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.isfinite(p1) & np.isfinite(p2) & (p1 > 0) & (p2 > 0)
        if not np.any(m) or m.sum() < 3:
            return float("nan"), float("nan")
        ratio = np.sqrt(p2[m] / p1[m])
        eps = np.abs(ratio - 1.0)
        eps = eps[np.isfinite(eps)]
        if eps.size == 0:
            return float("nan"), float("nan")
        # Moving-average smoothing on epsilon itself (no SciPy dependency)
        if smooth_window > 1:
            k = np.ones(int(smooth_window), dtype=float)
            k /= k.sum()
            # Use full convolution then trim to original length via 'same' behavior
            eps_sm = np.convolve(eps, k, mode="same")
        else:
            eps_sm = eps
        mean_sm = float(np.nanmean(eps_sm))
        pxx = float(np.nanpercentile(eps, pclip))
        return mean_sm, pxx


def _align_and_band_psds(
    freq_target: np.ndarray,
    psd_event: np.ndarray,
    freq_other: np.ndarray,
    psd_other: np.ndarray,
    fmin: float,
    fmax: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate `psd_other` onto `freq_target`, then band-limit both PSDs.

    Returns (fb, psd_other_banded, psd_event_banded) where fb are the banded
    frequencies, and both PSD arrays are filtered to the same bins and to
    finite, positive values to avoid invalid ratios.
    """
    ft = np.asarray(freq_target)
    pe = np.asarray(psd_event)
    fo = np.asarray(freq_other)
    po = np.asarray(psd_other)

    same_grid = (len(fo) == len(ft)) and np.allclose(fo, ft)
    if same_grid:
        po_interp = po
    else:
        po_interp = np.interp(ft, fo, po, left=np.nan, right=np.nan)

    fmax_eff = float(ft.max()) if (fmax is None) else float(fmax)
    band = (ft >= float(fmin)) & (ft <= fmax_eff)
    if not np.any(band):
        return ft[:0], pe[:0], pe[:0]

    fb = ft[band]
    pe_b = pe[band]
    po_b = po_interp[band]
    # Keep only finite and positive bins
    m = np.isfinite(pe_b) & np.isfinite(po_b) & (pe_b > 0) & (po_b > 0)
    if not np.any(m):
        return fb[:0], pe_b[:0], pe_b[:0]
    # Return as (fb, rewhite_on_target, event_on_target)
    return fb[m], po_b[m], pe_b[m]


def compute_bias_corrections(
    events_df: DataFrame,
    events_psds: DataArray,
    rewhite_df: DataFrame,
    rewhite_psds: DataArray,
    ifo: Optional[str] = None,
    fmin: float = 20.0,
    fmax: Optional[float] = None,
    verbose: bool = False,
) -> DataFrame:
    """For each event, pick most recent rewhitening PSD and compute corrections.

    Returns a DataFrame with columns:
      EventID, EventDate, WhitenDate, Ifo, PsdEpsilonMean, PsdEpsilonMax, DT1, DT2,
      DPhi1, DPhi2
    """
    if events_psds.dims != ("event", "ifo", "frequency"):
        raise ValueError("events_psds must have dims ['event','ifo','frequency']")
    if rewhite_psds.dims != ("record", "ifo", "frequency"):
        raise ValueError("rewhite_psds must have dims ['record','ifo','frequency']")

    freq = events_psds.coords["frequency"].values
    rw_freq = rewhite_psds.coords["frequency"].values
    if fmax is None:
        fmax = float(freq.max()) if freq.size else 1024.0

    results = []
    ifos = [ifo] if ifo else list(events_psds.coords["ifo"].values)

    for eidx, eid in enumerate(events_psds.coords["event"].values):
        evt_row = events_df.loc[events_df["graceid"] == eid]
        evt_time = (
            evt_row["event_time"].iloc[0]
            if not evt_row.empty and "event_time" in evt_row
            else None
        )
        # Precompute candidate rewhitening records with date <= event_time
        if evt_time is not None and "date" in rewhite_df:
            candidates = rewhite_df[rewhite_df["date"] <= evt_time].copy()
            # Sort ascending by date so we can scan from most recent to oldest
            try:
                candidates = candidates.sort_values("date")
            except Exception:
                pass
        else:
            candidates = rewhite_df.copy()

        for ifoname in ifos:
            if ifoname not in events_psds.coords["ifo"]:
                continue
            psd2 = events_psds.sel(event=eid, ifo=ifoname).values
            # pick most recent rewhitening record that contains this IFO
            wrec_ifo = None
            if ifoname in rewhite_psds.coords.get("ifo", []):
                # iterate candidates from newest to oldest
                for _, rr in (
                    candidates.iloc[::-1].iterrows() if not candidates.empty else []
                ):
                    try:
                        arr = rewhite_psds.sel(
                            record=rr["record_id"], ifo=ifoname
                        ).values
                        # accept if any finite values exist
                        if np.isfinite(arr).any():
                            wrec_ifo = rr
                            break
                    except Exception:
                        continue
            if wrec_ifo is None:
                continue
            psd1 = rewhite_psds.sel(record=wrec_ifo["record_id"], ifo=ifoname).values
            # Align grids and band-limit on the events frequency grid
            fb, psd1b, psd2b = _align_and_band_psds(
                freq, psd2, rw_freq, psd1, fmin, fmax
            )
            if fb.size < 2:
                # Not enough valid bins to compute metrics
                continue
            # Build a dummy template using intrinsics if available
            if not evt_row.empty and ("m1" in evt_row) and ("m2" in evt_row):
                try:
                    m1 = float(pd.to_numeric(evt_row["m1"].iloc[0], errors="coerce"))
                    m2 = float(pd.to_numeric(evt_row["m2"].iloc[0], errors="coerce"))
                except Exception:
                    m1 = m2 = float("nan")
            else:
                m1 = m2 = float("nan")
            if np.isnan(m1) or np.isnan(m2):
                # fall back to representative masses
                m1, m2 = 30.0, 30.0
            # Generate template on band
            dur = (
                max(8, int(np.ceil(1.0 / np.diff(fb[:2]).item())))
                if fb.size > 1
                else 16
            )
            tfreqs, htilde = generate_template_fd(
                m1, m2, (int(fb.min()), int(fb.max())), duration=dur
            )
            # Interpolate template onto fb
            htilde_b = np.interp(fb, tfreqs, htilde.real) + 1j * np.interp(
                fb, tfreqs, htilde.imag
            )

            # Simplified epsilon
            eps_mean, eps_max = _epsilon(psd1b, psd2b)
            # Median epsilon across frequency bins
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.sqrt(psd2b / psd1b)
                    eps_bins = np.abs(ratio - 1.0)
                    eps_bins = eps_bins[np.isfinite(eps_bins)]
                    eps_median = (
                        float(np.nanmedian(eps_bins)) if eps_bins.size else float("nan")
                    )
            except Exception:
                eps_median = float("nan")

            # Full corrections via MPMP
            try:
                sr = int(2 * float(fb.max())) if fb.size else 2048
                corr = MPMPCorrection(fb, psd1b, psd2b, htilde_b, sr)
                dt1, dt2, dphi1, dphi2 = corr.full_correction(data=htilde_b)
            except Exception:
                dt1 = dt2 = dphi1 = dphi2 = float("nan")

            results.append(
                {
                    "EventID": eid,
                    "EventDate": evt_time,
                    "WhitenDate": wrec_ifo["date"] if wrec_ifo is not None else None,
                    "Ifo": ifoname,
                    "PsdEpsilonMean": eps_mean,
                    "PsdEpsilonMedian": eps_median,
                    "PsdEpsilonMax": eps_max,
                    "DT1": dt1,
                    "DT2": dt2,
                    "DPhi1": dphi1,
                    "DPhi2": dphi2,
                }
            )

    return pd.DataFrame(results)


def build_unified_psd_history(
    events_df: DataFrame,
    events_psds: DataArray,
    rewhite_df: DataFrame,
    rewhite_psds: DataArray,
) -> DataArray:
    """Combine event and rewhitening PSDs into a single history with dims
    [date, ifo, frequency], resampling each source onto a unified frequency grid.
    """
    # Decide unified frequency grid (prefer events if present)
    if "frequency" in events_psds.coords and events_psds.sizes.get("frequency", 0) > 0:
        freq = np.asarray(events_psds.coords["frequency"].values)
    else:
        freq = np.asarray(rewhite_psds.coords["frequency"].values)
    ifos = sorted(
        set(list(events_psds.coords.get("ifo", []).values))
        | set(list(rewhite_psds.coords.get("ifo", []).values))
    )

    # Build list of (date, source, key)
    entries: list[tuple[object, str, object]] = []
    if "event_time" in events_df:
        gid_to_dt: Dict[str, object] = {}
        try:
            for _, row in events_df.iterrows():
                gid = row.get("graceid")
                dt = row.get("event_time")
                if gid is not None and dt is not None:
                    gid_to_dt[str(gid)] = dt
        except Exception:
            gid_to_dt = {}
        try:
            event_ids_seq = list(events_psds.coords.get("event", []).values)
        except Exception:
            event_ids_seq = []
        for eid in event_ids_seq:
            dt = gid_to_dt.get(str(eid))
            if dt is not None:
                entries.append((dt, "event", eid))
    for rec_id, dt in zip(rewhite_df.get("record_id", []), rewhite_df.get("date", [])):
        entries.append((dt, "record", rec_id))
    entries = [e for e in entries if e[0] is not None]
    entries.sort(key=lambda x: x[0])

    D = len(entries)
    F = len(freq)
    I = len(ifos)
    data = np.full((D, I, F), np.nan, dtype=float)

    # Helper to fetch and resample onto freq
    def _resample(arr_da: DataArray) -> np.ndarray:
        try:
            src_f = np.asarray(arr_da.coords["frequency"].values)
            src_v = np.asarray(arr_da.values)
            same = (len(src_f) == len(freq)) and np.allclose(src_f, freq)
            if same:
                return src_v.astype(float, copy=False)
            # Interpolate; mask to NaN outside range
            return np.interp(freq, src_f, src_v, left=np.nan, right=np.nan)
        except Exception:
            return np.full_like(freq, np.nan, dtype=float)

    for didx, (dt, kind, key) in enumerate(entries):
        for iidx, ifo in enumerate(ifos):
            try:
                if kind == "event":
                    arr_da = events_psds.sel(event=key, ifo=ifo)
                else:
                    arr_da = rewhite_psds.sel(record=key, ifo=ifo)
                data[didx, iidx, :] = _resample(arr_da)
            except Exception:
                # keep NaNs when selection fails
                continue

    dates = [dt for dt, _, _ in entries]
    psd_hist_xr = DataArray(
        data,
        dims=["date", "ifo", "frequency"],
        coords={"date": dates, "ifo": ifos, "frequency": freq},
        name="psd",
    )
    return psd_hist_xr


def compute_psd_evolution_metrics(psd_hist: DataArray) -> DataFrame:
    """Compute adjacent PSD change metrics per IFO over the date dimension.

    Only compares consecutive dates in time order (i, i+1) for each detector
    independently; does not mix IFOs.

    Returns a DataFrame with columns:
    - StartDate, EndDate, TimeDeltaDays, Ifo
    - EpsilonMean, EpsilonMedian, EpsilonMax (classic)
    - EpsilonMeanRobust, EpsilonP99 (less sensitive to line shifts)
    """
    dates = list(psd_hist.coords["date"].values)
    ifos = list(psd_hist.coords["ifo"].values) if "ifo" in psd_hist.coords else []
    results = []
    # Adjacent-only comparisons per IFO
    for ifo in ifos:
        for i in range(len(dates) - 1):
            j = i + 1
            try:
                a = psd_hist.sel(date=dates[i], ifo=ifo).values
                b = psd_hist.sel(date=dates[j], ifo=ifo).values
            except Exception:
                continue
            mean_eps, max_eps = _epsilon(a, b)
            mean_sm, p99 = _epsilon_robust(a, b, smooth_window=7, pclip=99.0)
            # New: median epsilon across frequency bins
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    a1 = np.asarray(a, dtype=float).ravel()
                    b1 = np.asarray(b, dtype=float).ravel()
                    n = int(min(a1.size, b1.size))
                    a1 = a1[:n]
                    b1 = b1[:n]
                    m = np.isfinite(a1) & np.isfinite(b1) & (a1 > 0) & (b1 > 0)
                    if np.any(m) and m.sum() >= 3:
                        ratio = np.sqrt(b1[m] / a1[m])
                        eps = np.abs(ratio - 1.0)
                        med_eps = float(np.nanmedian(eps)) if eps.size else float("nan")
                    else:
                        med_eps = float("nan")
            except Exception:
                med_eps = float("nan")
            # Compute time delta in days (End - Start)
            try:
                dt_seconds = (
                    pd.to_datetime(dates[j]) - pd.to_datetime(dates[i])
                ).total_seconds()  # type: ignore
                dt_days = float(dt_seconds) / 86400.0
            except Exception:
                dt_days = float("nan")
            results.append(
                {
                    "StartDate": dates[i],
                    "EndDate": dates[j],
                    "TimeDeltaDays": dt_days,
                    "Ifo": ifo,
                    "EpsilonMean": mean_eps,
                    "EpsilonMedian": med_eps,
                    "EpsilonMax": max_eps,
                    "EpsilonMeanRobust": mean_sm,
                    "EpsilonP99": p99,
                }
            )
    return pd.DataFrame(results)


def _ifo_to_lal_location(ifo: str):
    """Return lal detector location for a short IFO code (H1, L1, V1, K1)."""
    ifo = str(ifo).upper()
    try:
        if ifo == "H1":
            return lal.CachedDetectors[lal.LHO_4K_DETECTOR].location
        if ifo == "L1":
            return lal.CachedDetectors[lal.LLO_4K_DETECTOR].location
        if ifo == "V1":
            return lal.CachedDetectors[lal.VIRGO_DETECTOR].location
        if ifo == "K1":
            return lal.CachedDetectors[lal.KAGRA_DETECTOR].location
    except Exception:
        return None
    return None


def _ifo_to_lal_response(ifo: str):
    """Return lal detector response for a short IFO code or None."""
    ifo = str(ifo).upper()
    try:
        if ifo == "H1":
            return lal.CachedDetectors[lal.LHO_4K_DETECTOR].response
        if ifo == "L1":
            return lal.CachedDetectors[lal.LLO_4K_DETECTOR].response
        if ifo == "V1":
            return lal.CachedDetectors[lal.VIRGO_DETECTOR].response
        if ifo == "K1":
            return lal.CachedDetectors[lal.KAGRA_DETECTOR].response
    except Exception:
        return None
    return None


def _event_epoch_gps(row: pd.Series) -> Optional[float]:
    """Pick a GPS epoch for sky calculations from an events_df row."""
    # Prefer Coinc_end_time_gps if present
    for k in ("Coinc_end_time_gps", "coinc_end_time_gps", "coincidence_end_time_gps"):
        if k in row and pd.notna(row[k]):
            try:
                return float(row[k])
            except Exception:
                pass
    # Fallback: event_time (UTC datetime) -> gps
    if "event_time" in row and pd.notna(row["event_time"]):
        try:
            return float(time.to_gps(pd.to_datetime(row["event_time"])))
        except Exception:
            return None
    return None


def _per_ifo_times_gps(
    row: pd.Series, restrict_ifos: Optional[List[str]] = None
) -> Dict[str, float]:
    """Extract per-IFO arrival times (GPS) from an events_df row.
    Preference order per IFO:
      1) ChirpTime_<IFO> (float GPS seconds) if present
      2) <IFO>_impulse_time_gps, else <IFO>_impulse_time + ns
      3) <IFO>_end_time_gps, else <IFO>_end_time + ns
    If restrict_ifos is provided, only returns times for those IFOs.
    """
    times: Dict[str, float] = {}
    # Common IFOs
    candidates = ["H1", "L1", "V1", "K1"]
    if restrict_ifos is not None:
        # Preserve original order of restrict_ifos
        candidates = [s.upper() for s in restrict_ifos if isinstance(s, str)]
    for ifo in candidates:
        tval: Optional[float] = None
        # 1) ChirpTime_<IFO>
        ckey = f"ChirpTime_{ifo}"
        if ckey in row and pd.notna(row[ckey]):
            try:
                tval = float(row[ckey])
            except Exception:
                tval = None
        # 2) impulse_time
        if tval is None:
            gkey_imp = f"{ifo}_impulse_time_gps"
            if gkey_imp in row and pd.notna(row[gkey_imp]):
                try:
                    tval = float(row[gkey_imp])
                except Exception:
                    tval = None
        if tval is None:
            s_imp = f"{ifo}_impulse_time"
            ns_imp = f"{ifo}_impulse_time_ns"
            if (s_imp in row) and pd.notna(row[s_imp]):
                try:
                    sec = float(row[s_imp])
                    ns = (
                        float(row.get(ns_imp, 0.0))
                        if pd.notna(row.get(ns_imp, np.nan))
                        else 0.0
                    )
                    tval = sec + ns * 1e-9
                except Exception:
                    tval = None
        # 3) end_time
        if tval is None:
            gkey = f"{ifo}_end_time_gps"
            if gkey in row and pd.notna(row[gkey]):
                try:
                    tval = float(row[gkey])
                except Exception:
                    tval = None
        if tval is None:
            s_key = f"{ifo}_end_time"
            ns_key = f"{ifo}_end_time_ns"
            if (s_key in row) and pd.notna(row[s_key]):
                try:
                    sec = float(row[s_key])
                    ns = (
                        float(row.get(ns_key, 0.0))
                        if pd.notna(row.get(ns_key, np.nan))
                        else 0.0
                    )
                    tval = sec + ns * 1e-9
                except Exception:
                    tval = None
        if (tval is not None) and np.isfinite(tval) and (float(tval) > 1.0):
            times[ifo] = float(tval)
    return times


def _wrap_ra(ra: float) -> float:
    return float(ra % (2.0 * np.pi))


def _clamp_dec(dec: float) -> float:
    return float(np.clip(dec, -0.5 * np.pi, 0.5 * np.pi))


def _triangulate_sky_from_times(
    ifos: List[str],
    times_gps: Dict[str, float],
    gps_epoch: float,
    verbose: bool = False,
) -> Optional[Tuple[float, float]]:
    """Estimate (ra, dec) from per-IFO arrival times via delay triangulation.
    Returns (ra, dec) in radians, or None on failure.

    Method (two stages):
    - Stage 1 (coarse grid seed): sample RA in [0, 2π) and Dec in [-π/2, π/2],
      compute model time delays for each detector using lal.TimeDelayFromEarthCenter,
      compare observed inter-detector time differences (relative to a fixed reference
      IFO) to modeled ones, and pick the (RA, Dec) with minimum squared residual.
    - Stage 2 (linear refinement): linearize the inter-detector delays around that
      seed; compute partial derivatives ∂ΔT/∂RA and ∂ΔT/∂Dec by finite differences
      of TimeDelayFromEarthCenter; form a small A x = b system for x=[dRA, dDec]
      and solve by least squares. Return the corrected (RA, Dec), wrapped/clamped
      to valid ranges.

    Assumptions/justification:
    - The Earth-fixed detector locations and light-travel-time model are accurate
      (provided by LALSuite). For small perturbations, the linearization is valid
      and robust. The reference-IFO differencing removes an arbitrary absolute time
      offset common to all sites.
    """
    if len(ifos) < 2:
        return None
    # Map to locations
    locs = {}
    for ifo in ifos:
        loc = _ifo_to_lal_location(ifo)
        if loc is None:
            return None
        locs[ifo] = loc
    # Choose reference IFO deterministically
    ref_idx = int(np.argsort(np.array(ifos))[0])
    ref_ifo = ifos[ref_idx]
    # Observed diffs relative to reference
    obs = {ifo: (times_gps[ifo] - times_gps[ref_ifo]) for ifo in ifos}

    def tdelay(ifo_code: str, ra: float, dec: float) -> float:
        try:
            return float(
                lal.TimeDelayFromEarthCenter(locs[ifo_code], ra, dec, gps_epoch)
            )
        except Exception:
            try:
                return float(
                    lal.TimeDelayFromEarthCenter(
                        locs[ifo_code], ra, dec, lal.LIGOTimeGPS(gps_epoch)
                    )
                )
            except Exception:
                return float("nan")

    # Coarse grid seed: 10 deg resolution
    ra_grid = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    dec_grid = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 18)
    best = (None, None, float("inf"))
    for ra in ra_grid:
        for dec in dec_grid:
            # model diffs
            t0 = tdelay(ref_ifo, ra, dec)
            if not np.isfinite(t0):
                continue
            err2 = 0.0
            ok = True
            for ifo in ifos:
                dt = tdelay(ifo, ra, dec)
                if not np.isfinite(dt):
                    ok = False
                    break
                model = dt - t0
                r = obs[ifo] - model
                err2 += r * r
            if ok and err2 < best[2]:
                best = (ra, dec, err2)
    ra0, dec0, _ = best
    if ra0 is None or dec0 is None:
        return None

    # Linear refinement with finite differences
    h = 1e-6
    # Model vector (excluding ref)
    A = []
    b = []
    t_ref = tdelay(ref_ifo, ra0, dec0)
    for ifo in ifos:
        if ifo == ref_ifo:
            continue
        t_i = tdelay(ifo, ra0, dec0)
        t_i_ra = tdelay(ifo, ra0 + h, dec0)
        t_ref_ra = tdelay(ref_ifo, ra0 + h, dec0)
        t_i_dec = tdelay(ifo, ra0, dec0 + h)
        t_ref_dec = tdelay(ref_ifo, ra0, dec0 + h)
        dT_dra = ((t_i_ra - t_i) - (t_ref_ra - t_ref)) / h
        dT_ddec = ((t_i_dec - t_i) - (t_ref_dec - t_ref)) / h
        A.append([dT_dra, dT_ddec])
        model = t_i - t_ref
        b.append(obs[ifo] - model)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    if A.size == 0 or not np.isfinite(A).all() or not np.isfinite(b).all():
        return (float(ra0), float(dec0))
    try:
        dx, *_ = np.linalg.lstsq(A, b, rcond=None)
        ra1 = _wrap_ra(ra0 + float(dx[0]))
        dec1 = _clamp_dec(dec0 + float(dx[1]))
        return (ra1, dec1)
    except Exception:
        return (float(ra0), float(dec0))


def compute_skymap_deltas(
    events_df: DataFrame,
    bias_df: DataFrame,
    fref: float = 100.0,
    verbose: bool = False,
    events_timing: Optional[pd.DataFrame] = None,
) -> DataFrame:
    """Compute first-order RA/Dec changes per event induced by MPMP corrections.

    Overview of method:
    - Convert per-IFO corrections to an equivalent arrival-time shift:
      dt_equiv_i = DT1_i + (-DPhi1_i)/(2*pi*fref). This uses the standard
      phase/time relation dt = -dphi/(2*pi*f) at a chosen reference frequency fref.
    - Work with time differences relative to a reference detector r to remove
      a common absolute offset: b_i = dt_equiv_i - dt_equiv_r.
    - Compute partial derivatives of the inter-detector delays (relative to r)
      with respect to RA and Dec at the baseline sky position (either provided
      in metadata or triangulated from per-IFO times). We evaluate these partials
      by finite differences using lal.TimeDelayFromEarthCenter.
    - Solve A x = b in the least-squares sense for x = [dRA, dDec]. Report
      both the component shifts and a small-angle separation
      dAng ≈ sqrt( (dRA*cos(Dec))^2 + dDec^2 ).

    Baseline sky and epoch:
    - If events_df provides RA/Dec, use them; otherwise triangulate from the per-IFO
      arrival times in events_timing (parsed directly from coinc.xml SnglInspiral).
      An event GPS epoch is required (Coinc_end_time if available, else event_time or
      mean of IFO times), because Earth rotation affects the geometric delays.

    Returns a DataFrame; diagnostics summary is stored in df.attrs['diagnostics'].
    """
    results = []
    diags = {
        "n_bias_rows": int(len(bias_df) if isinstance(bias_df, pd.DataFrame) else 0),
        "has_eventid_col": bool(
            isinstance(bias_df, pd.DataFrame) and ("EventID" in bias_df.columns)
        ),
        "unique_events_in_bias": 0,
        "events_with_2_ifo": 0,
        "events_with_ra_dec": 0,
        "events_with_gps": 0,
        "events_with_locations": 0,
        "events_with_finite_delays": 0,
        "events_solved": 0,
        "skip_examples": [],  # list of (EventID, reason)
    }
    if not isinstance(bias_df, pd.DataFrame) or bias_df.empty:
        if verbose:
            print("[skymap] bias_corrections is empty; cannot compute sky deltas")
        df_out = pd.DataFrame(results)
        df_out.attrs["diagnostics"] = diags
        return df_out
    if "EventID" not in bias_df.columns:
        if verbose:
            print(
                "[skymap] bias_corrections is missing 'EventID' column; cannot group by event"
            )
        df_out = pd.DataFrame(results)
        df_out.attrs["diagnostics"] = diags
        return df_out

    # Require events_timing from coinc.xml
    if not isinstance(events_timing, pd.DataFrame) or events_timing.empty:
        raise ValueError(
            "events_timing is required for skymap: parse from coinc.xml during events dataset build"
        )

    by_evt = bias_df.groupby("EventID")
    diags["unique_events_in_bias"] = int(len(by_evt))

    for eid, df_ev in by_evt:
        # Collect per-IFO total dt (s)
        ifos = []
        dt_total = []
        for _, r in df_ev.iterrows():
            ifo = str(r.get("Ifo", "")).upper()
            if not ifo:
                continue
            dt1 = float(pd.to_numeric(r.get("DT1"), errors="coerce"))
            dphi1 = float(pd.to_numeric(r.get("DPhi1"), errors="coerce"))
            if not (np.isfinite(dt1) or np.isfinite(dphi1)):
                continue
            dt_equiv = 0.0
            if np.isfinite(dt1):
                dt_equiv += dt1
            if np.isfinite(dphi1) and np.isfinite(fref) and fref > 0:
                dt_equiv += (-dphi1) / (2.0 * np.pi * fref)
            ifos.append(ifo)
            dt_total.append(dt_equiv)
        if len(ifos) < 2:
            diags["skip_examples"].append((eid, "<2 IFOs with finite DT1/DPhi1"))
            continue
        diags["events_with_2_ifo"] += 1
        # Baseline sky and epoch
        row = events_df.loc[events_df["graceid"] == eid]
        if row.empty:
            diags["skip_examples"].append((eid, "event not found in events_df"))
            continue
        row = row.iloc[0]
        # Determine GPS epoch for calculations (needed for triangulation and derivatives)
        gps = _event_epoch_gps(row)
        # Try to use RA/Dec if present; otherwise attempt triangulation fallback
        ra0 = float("nan")
        dec0 = float("nan")
        have_ra_dec_cols = ("ra" in row) and ("dec" in row)
        if have_ra_dec_cols:
            try:
                ra0 = float(pd.to_numeric(row.get("ra"), errors="coerce"))
                dec0 = float(pd.to_numeric(row.get("dec"), errors="coerce"))
            except Exception:
                ra0 = dec0 = float("nan")
        used_triang = False
        if not (np.isfinite(ra0) and np.isfinite(dec0)):
            # Triangulate strictly from events_timing (required)
            times: Dict[str, float] = {}
            try:
                sub = events_timing[events_timing["graceid"] == eid]
                if not sub.empty:
                    for ifo_name in ifos:
                        tvals = pd.to_numeric(
                            sub.loc[
                                sub["ifo"].str.upper() == ifo_name.upper(), "time_gps"
                            ],
                            errors="coerce",
                        )
                        tvals = tvals[np.isfinite(tvals)]
                        if not tvals.empty:
                            times[ifo_name] = float(tvals.iloc[0])
            except Exception:
                times = {}
            if len(times) >= 2:
                diags.setdefault("events_with_times", 0)
                diags["events_with_times"] += 1
                # If GPS epoch missing, use mean of per-IFO times
                if gps is None or not np.isfinite(gps):
                    try:
                        gps = float(np.mean(list(times.values())))
                    except Exception:
                        gps = None
                if gps is not None and np.isfinite(gps):
                    tri = _triangulate_sky_from_times(ifos, times, gps, verbose=verbose)
                    if tri is not None:
                        ra0, dec0 = tri
                        used_triang = True
                    else:
                        diags.setdefault("triangulation_failures", 0)
                        diags["triangulation_failures"] += 1
                        diags["skip_examples"].append((eid, "triangulation failed"))
                        continue
                else:
                    diags["skip_examples"].append(
                        (eid, "no valid GPS epoch and cannot infer from IFO times")
                    )
                    continue
            else:
                diags["skip_examples"].append(
                    (
                        eid,
                        "missing ra/dec and insufficient per-IFO times to triangulate",
                    )
                )
                continue
        # At this point we have baseline (ra0, dec0) and gps
        if gps is None or not np.isfinite(gps):
            diags["skip_examples"].append((eid, "no valid GPS epoch found"))
            continue
        diags["events_with_gps"] += 1
        if used_triang:
            diags.setdefault("events_triangulated", 0)
            diags["events_triangulated"] += 1
        else:
            diags["events_with_ra_dec"] += 1
        # Build derivative matrices using lal.TimeDelayFromEarthCenter
        # Use differences relative to a reference IFO to eliminate common time
        ref = np.argsort(np.array(ifos))[0]
        ifo_ref = ifos[ref]
        # Map IFOs to lal locations; skip if any missing
        locs = {}
        ok = True
        for ifo in ifos:
            loc = _ifo_to_lal_location(ifo)
            if loc is None:
                ok = False
                break
            locs[ifo] = loc
        if not ok:
            diags["skip_examples"].append((eid, "unable to map IFO to LAL location"))
            continue
        diags["events_with_locations"] += 1

        # Compute baseline delays
        def t_at(ra, dec, ifo_code):
            try:
                return float(lal.TimeDelayFromEarthCenter(locs[ifo_code], ra, dec, gps))
            except Exception:
                # Some versions take (location, ra, dec, epoch)
                try:
                    return float(
                        lal.TimeDelayFromEarthCenter(
                            locs[ifo_code], ra, dec, lal.LIGOTimeGPS(gps)
                        )
                    )
                except Exception:
                    return float("nan")

        t0 = {ifo: t_at(ra0, dec0, ifo) for ifo in ifos}
        if not all(np.isfinite(v) for v in t0.values()):
            diags["skip_examples"].append((eid, "non-finite baseline time delays"))
            continue
        # Finite differences
        h = 1e-6  # radians
        t_ra = {ifo: t_at(ra0 + h, dec0, ifo) for ifo in ifos}
        t_dec = {ifo: t_at(ra0, dec0 + h, ifo) for ifo in ifos}
        # Build A and b for (N-1) equations
        A = []
        b = []
        for i, ifo in enumerate(ifos):
            if i == ref:
                continue
            dT_dra = ((t_ra[ifo] - t0[ifo]) - (t_ra[ifo_ref] - t0[ifo_ref])) / h
            dT_ddec = ((t_dec[ifo] - t0[ifo]) - (t_dec[ifo_ref] - t0[ifo_ref])) / h
            A.append([dT_dra, dT_ddec])
            b.append((dt_total[i] - dt_total[ref]))
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        if A.size == 0 or not np.isfinite(A).all() or not np.isfinite(b).all():
            diags["skip_examples"].append(
                (eid, "ill-conditioned system or non-finite A/b")
            )
            continue
        try:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception as e:
            diags["skip_examples"].append((eid, f"lstsq failed: {e}"))
            continue
        d_ra, d_dec = float(x[0]), float(x[1])
        # Angular separation (small-angle approx)
        d_ang_rad = float(np.sqrt((d_ra * np.cos(dec0)) ** 2 + d_dec**2))
        results.append(
            {
                "EventID": eid,
                "NifoUsed": len(ifos),
                "dRA_rad": d_ra,
                "dDec_rad": d_dec,
                "dAng_rad": d_ang_rad,
                "dRA_deg": np.degrees(d_ra),
                "dDec_deg": np.degrees(d_dec),
                "dAng_deg": np.degrees(d_ang_rad),
            }
        )
        diags["events_with_finite_delays"] += 1
        diags["events_solved"] += 1

    df_out = pd.DataFrame(results)
    df_out.attrs["diagnostics"] = diags
    if verbose:
        # Print a concise summary
        print("[skymap] diagnostics:")
        for k, v in diags.items():
            if k != "skip_examples":
                print(f"  - {k}: {v}")
        if diags["skip_examples"]:
            # Show up to first 10 examples
            show = diags["skip_examples"][:10]
            print("  - skip_examples (first 10):")
            for eid, reason in show:
                print(f"    * {eid}: {reason}")
    return df_out


def parse_args() -> argparse.Namespace:
    # Define a parent parser that holds all common options. We will attach it to
    # both the top-level parser and each subparser so that options are accepted
    # whether they appear before or after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--out-dir", default=OUT_DIR, help="Output directory (default: %(default)s)"
    )

    # Common GraceDB query args
    common.add_argument(
        "--pipeline",
        default="gstlal",
        help="GraceDB pipeline filter (e.g., gstlal, pycbc)",
    )
    common.add_argument(
        "--search",
        default=None,
        help="GraceDB search filter (omitted if not specified)",
    )
    common.add_argument("--far-min", type=float, default=None, help="Minimum FAR (Hz)")
    common.add_argument("--far-max", type=float, default=1e-5, help="Maximum FAR (Hz)")
    common.add_argument(
        "--time-start", default=None, help="Start date (YYYY-MM-DD or ISO)"
    )
    common.add_argument("--time-end", default=None, help="End date (YYYY-MM-DD or ISO)")
    common.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max number of events to fetch (applied after dedupe " "when chunking)",
    )
    common.add_argument(
        "--events-chunk-days",
        type=int,
        default=0,
        help="Split the events time range into N-day chunks (" "0=disabled)",
    )
    common.add_argument(
        "--events-parallel",
        type=int,
        default=4,
        help="Max parallel chunk queries when chunking > 0",
    )
    common.add_argument(
        "--events-per-chunk-limit",
        type=int,
        default=None,
        help="Max results per chunk (defaults to --limit)",
    )
    common.add_argument(
        "--events-process-parallel",
        type=int,
        default=1,
        help="Parallel workers for building the events dataset ("
        "coinc.xml download & parse)",
    )
    common.add_argument(
        "--events-process-batch",
        type=int,
        default=50,
        help="Batch size for submitting events to workers (for " "logging/granularity)",
    )
    common.add_argument(
        "--gracedb-playground",
        action="store_true",
        help="Use GraceDB Playground ("
        "https://gracedb-playground.ligo.org/api/) instead of "
        "production",
    )
    common.add_argument(
        "--no-reduce-by-superevent",
        action="store_true",
        help="Do not reduce by superevent",
    )
    common.add_argument(
        "--labels",
        default=None,
        help="Comma-separated GraceDB labels to include (e.g., "
        "INJ or HardwareInjection; example: --labels INJ,CBC)",
    )

    # Rewhitening dataset args
    common.add_argument(
        "--rewhite-base-dir",
        default=None,
        help="Base directory containing rewhitening XMLs (local or "
        "remote path on SSH host)",
    )
    common.add_argument(
        "--rewhite-analysis",
        default=None,
        help="Substring to filter rewhitening analyses",
    )
    common.add_argument(
        "--rewhite-limit",
        type=int,
        default=None,
        help="Limit number of rewhitening folders",
    )
    common.add_argument(
        "--rewhite-start",
        default=None,
        help="Rewhitening start date (YYYY-MM-DD or ISO)",
    )
    common.add_argument(
        "--rewhite-end", default=None, help="Rewhitening end date (YYYY-MM-DD or ISO)"
    )
    common.add_argument(
        "--rewhite-observing-root",
        default=None,
        help="Observing run root (e.g., "
        "/home/gstlalcbc.offline/observing/4). \n"
        "When set with --rewhite-subruns, "
        "scans <root>/<sub>/rewhiten for given subruns.",
    )
    common.add_argument(
        "--rewhite-subruns",
        default=None,
        help="Comma-separated subruns (e.g., a,b,c). Used with "
        "--rewhite-observing-root.",
    )
    common.add_argument(
        "--remote-host",
        default=None,
        help="SSH hostname to fetch rewhitening files (fallback env " "QA_SSH_HOST)",
    )
    common.add_argument(
        "--remote-user", default=None, help="SSH username (fallback env QA_SSH_USER)"
    )

    # Analysis args
    common.add_argument(
        "--ifo", default=None, help="Restrict to a single IFO for analyses (e.g., H1)"
    )
    common.add_argument(
        "--fmin", type=float, default=20.0, help="Min frequency for analyses"
    )
    common.add_argument(
        "--fmax", type=float, default=None, help="Max frequency for analyses"
    )
    common.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging (remote scans, " "copies)",
    )
    common.add_argument(
        "--epsilon-min",
        type=float,
        default=1e-5,
        help="Minimum epsilon value for log-scale plots to avoid axis blowout (default: %(default)s)",
    )

    parser = argparse.ArgumentParser(
        description="QA: GraceDB datasets, bias corrections, and PSD evolution "
        "analyses",
        parents=[common],
        add_help=True,
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "events",
        parents=[common],
        help="Create events dataset from GraceDB and save to disk",
    )
    fetch = sub.add_parser(
        "fetch-coinc",
        parents=[common],
        help="Download coinc.xml for a single GraceDB event to a " "local path",
    )
    fetch.add_argument(
        "--graceid", required=True, help="GraceDB event ID (e.g., M540474)"
    )
    fetch.add_argument("--out", required=True, help="Local path to save coinc.xml")
    sub.add_parser(
        "rewhite",
        parents=[common],
        help="Create rewhitening dataset from local/remote folders and " "save to disk",
    )
    sub.add_parser(
        "biases",
        parents=[common],
        help="Analyze realistic bias correction distribution; requires "
        "both datasets",
    )
    sub.add_parser(
        "psd-evolution",
        parents=[common],
        help="Analyze PSD evolution across dates; requires both datasets",
    )
    # New skymap subcommand
    sk = sub.add_parser(
        "skymap",
        parents=[common],
        help="Estimate first-order changes in sky localization (RA/Dec) per event due to bias corrections; saves CSV and plots",
    )
    sk.add_argument(
        "--fref",
        type=float,
        default=100.0,
        help="Reference frequency [Hz] to convert phase corrections to equivalent time: dt_phi = -DPhi1/(2*pi*fref)",
    )
    sub.add_parser(
        "analyze",
        parents=[common],
        help="Run analyses using existing datasets in --out-dir (biases + "
        "psd-evolution)",
    )
    sub.add_parser(
        "all",
        parents=[common],
        help="Run full pipeline: events, optional rewhite, then analyses",
    )

    return parser.parse_args()


def _fetch_events_from_args(args: argparse.Namespace) -> DataFrame:
    t0 = _date_from_str(args.time_start)
    t1 = _date_from_str(args.time_end)
    # Parse labels if provided (comma-separated)
    labels_list = None
    if getattr(args, "labels", None):
        labels_list = [s.strip() for s in str(args.labels).split(",") if s.strip()]
    ev = get_events(
        pipeline=args.pipeline,
        search=args.search,
        far_min=args.far_min,
        far_max=args.far_max,
        time_start=t0,
        time_end=t1,
        labels=labels_list,
        limit=args.limit,
        reduce_by_superevent=not args.no_reduce_by_superevent,
        verbose=getattr(args, "verbose", False),
        gracedb_playground=getattr(args, "gracedb_playground", False),
        events_chunk_days=getattr(args, "events_chunk_days", 0),
        events_parallel=getattr(args, "events_parallel", 4),
        events_per_chunk_limit=getattr(args, "events_per_chunk_limit", None),
    )
    return ev


def _build_events_dataset_and_save(
    args: argparse.Namespace,
) -> Tuple[DataFrame, DataArray]:
    out_dir = _ensure_out_dir(args.out_dir)
    events = _fetch_events_from_args(args)
    print(f"Found {len(events)} events.")
    event_ids = events["graceid"].tolist()
    events_df, event_psds, timing_df = get_events_dataset(
        event_ids,
        gracedb_playground=getattr(args, "gracedb_playground", False),
        events_process_parallel=getattr(args, "events_process_parallel", 1),
        events_process_batch=getattr(args, "events_process_batch", 50),
        verbose=getattr(args, "verbose", False),
    )
    save_events_dataset(out_dir, events_df, event_psds, timing_df=timing_df)
    print(
        f"Event dataset saved: {len(events_df)} rows, psds s"
        f"hape {tuple(event_psds.shape)}"
    )
    return events_df, event_psds


def _build_rewhite_dataset_and_save(
    args: argparse.Namespace,
    default_time_window_from_events: Optional[
        Tuple[datetime.datetime, datetime.datetime]
    ] = None,
) -> Tuple[DataFrame, DataArray]:
    # Build list of base directories
    base_dirs: List[str] = []
    if getattr(args, "rewhite_observing_root", None) and getattr(
        args, "rewhite_subruns", None
    ):
        subs = [s.strip() for s in str(args.rewhite_subruns).split(",") if s.strip()]
        for s in subs:
            base_dirs.append(os.path.join(args.rewhite_observing_root, s, "rewhiten"))
    elif args.rewhite_base_dir:
        base_dirs = [args.rewhite_base_dir]
    else:
        raise ValueError(
            "Provide --rewhite-base-dir or (--rewhite-observing-root with "
            "--rewhite-subruns)"
        )

    out_dir = _ensure_out_dir(args.out_dir)
    start_dt = _date_from_str(args.rewhite_start) or (
        default_time_window_from_events[0] if default_time_window_from_events else None
    )
    end_dt = _date_from_str(args.rewhite_end) or (
        default_time_window_from_events[1] if default_time_window_from_events else None
    )
    rewhite_df, rewhite_psds = build_rewhitening_dataset(
        base_dirs,
        start_date=start_dt,
        end_date=end_dt,
        analysis=args.rewhite_analysis,
        limit=args.rewhite_limit,
        remote_host=args.remote_host,
        remote_user=args.remote_user,
        verbose=getattr(args, "verbose", False),
    )
    save_rewhite_dataset(out_dir, rewhite_df, rewhite_psds)
    print(
        f"Rewhitening dataset saved: {len(rewhite_df)} ro"
        f"ws, psds shape {tuple(rewhite_psds.shape)}"
    )
    return rewhite_df, rewhite_psds


def _require_datasets(
    out_dir: str,
) -> Tuple[DataFrame, DataArray, DataFrame, DataArray]:
    events_df, event_psds = load_events_dataset(out_dir)
    rewhite_df, rewhite_psds = load_rewhite_dataset(out_dir)
    if events_df.empty or event_psds.size == 0:
        raise FileNotFoundError(
            "Events dataset not found in out-dir; run 'events' or 'all' first"
        )
    if rewhite_df.empty or rewhite_psds.size == 0:
        raise FileNotFoundError(
            "Rewhitening dataset not found in out-dir; run 'rewhite' or 'all' first"
        )
    return events_df, event_psds, rewhite_df, rewhite_psds


def download_coinc(
    graceid: str, out: str, gracedb_playground: bool = False, verbose: bool = False
) -> str:
    """Download coinc.xml for a single GraceDB event and save to a local path.
    Returns the path written. Raises a ValueError if coinc.xml is not attached.
    """
    service_url = _get_gracedb_service_url(gracedb_playground)
    cli = GraceDb(service_url=service_url)
    files = cli.files(graceid).json()
    if verbose:
        available = list(files.keys()) if isinstance(files, dict) else []
        print(
            f"[fetch-coinc] GraceID={graceid} server={service_url} "
            f"available_files={available}"
        )
    if "coinc.xml" not in files:
        raise ValueError(
            f"Event {graceid} does not have coinc.xml (has: "
            f"{list(files.keys()) if isinstance(files, dict) else []})"
        )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fid:
        resp = cli.files(graceid, "coinc.xml")
        fid.write(resp.data)
    return out


def main():
    args = parse_args()
    # Override default OUT_DIR usage by CLI arg
    out_dir = _ensure_out_dir(args.out_dir)

    if args.command == "fetch-coinc":
        out_path = download_coinc(
            args.graceid,
            args.out,
            gracedb_playground=getattr(args, "gracedb_playground", False),
            verbose=getattr(args, "verbose", False),
        )
        print(f"Saved coinc.xml to {out_path}")

    elif args.command == "events":
        _build_events_dataset_and_save(args)

    elif args.command == "rewhite":
        # If available, infer default time window from existing events dataset
        ev_df, _ = load_events_dataset(out_dir)
        default_window = None
        if not ev_df.empty and ("event_time" in ev_df):
            default_window = (
                pd.to_datetime(ev_df["event_time"]).min().to_pydatetime(),
                pd.to_datetime(ev_df["event_time"]).max().to_pydatetime(),
            )
        _build_rewhite_dataset_and_save(args, default_window)

    elif args.command == "biases":
        events_df, event_psds, rewhite_df, rewhite_psds = _require_datasets(out_dir)
        bc_df = compute_bias_corrections(
            events_df,
            event_psds,
            rewhite_df,
            rewhite_psds,
            ifo=args.ifo,
            fmin=args.fmin,
            fmax=args.fmax,
        )
        bc_path = os.path.join(out_dir, "bias_corrections.csv")
        bc_df.to_csv(bc_path, index=False)
        # Diagnostic summary about IFO coverage
        try:
            events_ifos = (
                list(event_psds.coords["ifo"].values)
                if "ifo" in event_psds.coords
                else []
            )
            rewhite_ifos = (
                list(rewhite_psds.coords["ifo"].values)
                if "ifo" in rewhite_psds.coords
                else []
            )
            produced_ifos = (
                sorted(set(str(x) for x in bc_df["Ifo"].dropna().unique().tolist()))
                if "Ifo" in bc_df
                else []
            )
            print(f"Bias corrections saved: {bc_path} ({len(bc_df)} rows)")
            print(f"[biases] IFOs in events dataset: {events_ifos}")
            print(f"[biases] IFOs in rewhite dataset: {rewhite_ifos}")
            print(f"[biases] IFOs produced in corrections: {produced_ifos}")
            if args.ifo:
                print(
                    f"[biases] Note: --ifo filter applied ({args.ifo}); only that "
                    f"detector is processed."
                )
            else:
                expected = sorted(set(events_ifos) & set(rewhite_ifos))
                missing = sorted(set(expected) - set(produced_ifos))
                if missing:
                    print(
                        f"[biases] Missing IFOs in output: {missing}. Likely causes: "
                        f"no matching rewhitening record with finite PSD for that IFO "
                        f"before each event time, or the event PSD lacked that IFO. "
                        f"Try rewhite --verbose to confirm files and per-IFO content."
                    )
        except Exception:
            print(f"Bias corrections saved: {bc_path} ({len(bc_df)} rows)")
        # Histograms faceted by IFO
        unique_ifos = (
            sorted(set(str(x) for x in bc_df["Ifo"].dropna().unique().tolist()))
            if "Ifo" in bc_df
            else []
        )
        for col, fname, xlabel in [
            ("DT1", "hist_dt1.png", "dt1 [s]"),
            ("DPhi1", "hist_dphi1.png", "dphi1 [rad]"),
            ("DT2", "hist_dt2.png", "dt2 [s]"),
            ("DPhi2", "hist_dphi2.png", "dphi2 [rad]"),
        ]:
            try:
                n = max(1, len(unique_ifos))
                fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
                for idx, ifo_name in enumerate(unique_ifos or ["ALL"]):
                    ax = axes[0, idx]
                    sub = (
                        bc_df if ifo_name == "ALL" else bc_df[bc_df["Ifo"] == ifo_name]
                    )
                    vals = sub[col].values.astype(float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        ax.hist(vals, bins=30)
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel("Count")
                    title = f"{col} ({ifo_name})" if ifo_name != "ALL" else f"{col}"
                    ax.set_title(title)
                    ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, fname))
                plt.close(fig)
            except Exception as e:
                print(f"Warning: failed to write histogram {fname}: {e}")

        # Additional faceted scatter plots requested: m1 vs m2 and MChirp vs ChiEff
        # colored by DT/DPhi
        try:
            cols_present = set(events_df.columns)
            needed = {"graceid", "m1", "m2", "chi1z", "chi2z"}
            if needed.issubset(cols_present):
                # Merge masses/spins into bias corrections by EventID
                meta = events_df[["graceid", "m1", "m2", "chi1z", "chi2z"]].copy()
                merged = bc_df.merge(
                    meta, left_on="EventID", right_on="graceid", how="left"
                )
                # Compute derived parameters
                m1v = pd.to_numeric(merged["m1"], errors="coerce").astype(float)
                m2v = pd.to_numeric(merged["m2"], errors="coerce").astype(float)
                chi1 = pd.to_numeric(merged.get("chi1z"), errors="coerce").astype(float)
                chi2 = pd.to_numeric(merged.get("chi2z"), errors="coerce").astype(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    mchirp = np.power(m1v * m2v, 3.0 / 5.0) / np.power(
                        m1v + m2v, 1.0 / 5.0
                    )
                    chieff = (m1v * chi1 + m2v * chi2) / (m1v + m2v)
                merged["MChirp"] = mchirp
                merged["ChiEff"] = chieff

                def _scatter_faceted(
                    df_in: pd.DataFrame,
                    x: str,
                    y: str,
                    c: str,
                    fname_out: str,
                    xlabel: str,
                    ylabel: str,
                    cmap: str = "viridis",
                ) -> None:
                    # Drop rows with missing x/y/c
                    sub = df_in[["Ifo", x, y, c]].copy()
                    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(
                        subset=[x, y, c]
                    )
                    if sub.empty:
                        return
                    # Robust color clipping to P2-P98 to reduce outlier streaking
                    cvals = sub[c].values.astype(float)
                    cmin = np.nanpercentile(cvals, 2.0)
                    cmax = np.nanpercentile(cvals, 98.0)
                    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmin == cmax:
                        cmin, cmax = np.nanmin(cvals), np.nanmax(cvals)
                    ifos_plot = sorted(
                        set(str(xv) for xv in sub["Ifo"].dropna().unique().tolist())
                    ) or ["ALL"]
                    n = max(1, len(ifos_plot))
                    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
                    for idx, ifo_name in enumerate(ifos_plot):
                        ax = axes[0, idx]
                        dat = sub[sub["Ifo"] == ifo_name] if ifo_name != "ALL" else sub
                        sc = ax.scatter(
                            dat[x],
                            dat[y],
                            c=np.clip(dat[c], cmin, cmax),
                            s=14,
                            cmap=cmap,
                            vmin=cmin,
                            vmax=cmax,
                            alpha=0.7,
                            edgecolors="none",
                        )
                        ax.set_xlabel(xlabel)
                        ax.set_ylabel(ylabel)
                        title = (
                            f"{os.path.splitext(fname_out)[0]} ({ifo_name})"
                            if (ifo_name != "ALL")
                            else os.path.splitext(fname_out)[0]
                        )
                        ax.set_title(title)
                        ax.grid(True, alpha=0.3)
                        # Shared colorbar per subplot for readability
                        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                        cb.set_label(c)
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, fname_out))
                    plt.close(fig)

                # Generate plots for each correction metric
                for color_col, pretty in [
                    ("DT1", "dt1"),
                    ("DT2", "dt2"),
                    ("DPhi1", "dphi1"),
                    ("DPhi2", "dphi2"),
                ]:
                    if color_col in merged.columns:
                        _scatter_faceted(
                            merged,
                            "m1",
                            "m2",
                            color_col,
                            f"scatter_m1_m2_{pretty}.png",
                            "m1 [Msun]",
                            "m2 [Msun]",
                        )
                        _scatter_faceted(
                            merged,
                            "MChirp",
                            "ChiEff",
                            color_col,
                            f"scatter_mchirp_chieff_{pretty}.png",
                            "M_chirp [Msun]",
                            "chi_eff",
                        )
            else:
                print(
                    "[biases] Note: m1/m2/spins not found in events metadata; "
                    "skipping m1-m2 and Mchirp-ChiEff scatter plots."
                )
        except Exception as e:
            print(f"Warning: failed to write faceted scatter plots: {e}")
        # New: mass distribution plots and log-log joint scatter
        try:
            # Histograms of log10(m1) and log10(m2)
            m1_all = pd.to_numeric(
                events_df.get("m1", pd.Series(dtype=float)), errors="coerce"
            ).astype(float)
            m2_all = pd.to_numeric(
                events_df.get("m2", pd.Series(dtype=float)), errors="coerce"
            ).astype(float)
            m1_pos = m1_all[np.isfinite(m1_all) & (m1_all > 0)]
            m2_pos = m2_all[np.isfinite(m2_all) & (m2_all > 0)]
            if m1_pos.size:
                logm1 = np.log10(m1_pos)
                fig_h1, ax_h1 = plt.subplots(figsize=(6, 4))
                ax_h1.hist(logm1, bins=30)
                ax_h1.set_xlabel("log10(m1 [Msun])")
                ax_h1.set_ylabel("Count")
                ax_h1.set_title("Histogram of log10(m1)")
                ax_h1.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "hist_log_m1.png"))
                plt.close(fig_h1)
            if m2_pos.size:
                logm2 = np.log10(m2_pos)
                fig_h2, ax_h2 = plt.subplots(figsize=(6, 4))
                ax_h2.hist(logm2, bins=30)
                ax_h2.set_xlabel("log10(m2 [Msun])")
                ax_h2.set_ylabel("Count")
                ax_h2.set_title("Histogram of log10(m2)")
                ax_h2.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "hist_log_m2.png"))
                plt.close(fig_h2)
            # Joint scatter (log10 m1 vs log10 m2) with marginal histograms
            mask_joint = (
                np.isfinite(m1_all) & (m1_all > 0) & np.isfinite(m2_all) & (m2_all > 0)
            )
            if np.any(mask_joint):
                x = np.log10(m1_all[mask_joint])
                y = np.log10(m2_all[mask_joint])
                # Layout: top (hist x), right (hist y), main (scatter)
                import matplotlib.gridspec as gridspec

                fig = plt.figure(figsize=(6, 6))
                gs = gridspec.GridSpec(
                    2,
                    2,
                    width_ratios=[4, 1.4],
                    height_ratios=[1.4, 4],
                    hspace=0.05,
                    wspace=0.05,
                )
                ax_scatter = fig.add_subplot(gs[1, 0])
                ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
                ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
                ax_scatter.scatter(x, y, s=12, alpha=0.6, edgecolors="none")
                ax_scatter.set_xlabel("log10(m1 [Msun])")
                ax_scatter.set_ylabel("log10(m2 [Msun])")
                ax_scatter.grid(True, alpha=0.3)
                # Marginals
                ax_histx.hist(x, bins=30)
                ax_histy.hist(y, bins=30, orientation="horizontal")
                # Hide tick labels on shared axes for marginals
                plt.setp(ax_histx.get_xticklabels(), visible=False)
                plt.setp(ax_histy.get_yticklabels(), visible=False)
                ax_histx.tick_params(axis="x", which="both", length=0)
                ax_histy.tick_params(axis="y", which="both", length=0)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "scatter_log_m1_m2_joint.png"))
                plt.close(fig)
            # Scatter: log10(MChirp) vs Chi (ChiEff) with marginal histograms
            # Robustly derive MChirp and ChiEff from intrinsics when available
            m1_all = pd.to_numeric(
                events_df.get("m1", pd.Series(dtype=float)), errors="coerce"
            ).astype(float)
            m2_all = pd.to_numeric(
                events_df.get("m2", pd.Series(dtype=float)), errors="coerce"
            ).astype(float)
            chi1_all = pd.to_numeric(
                events_df.get("chi1z", pd.Series(dtype=float)), errors="coerce"
            ).astype(float)
            chi2_all = pd.to_numeric(
                events_df.get("chi2z", pd.Series(dtype=float)), errors="coerce"
            ).astype(float)
            # Prefer derived MChirp where masses are present; fall back to column
            with np.errstate(invalid="ignore", divide="ignore"):
                mchirp_der = np.power(m1_all * m2_all, 3.0 / 5.0) / np.power(
                    m1_all + m2_all, 1.0 / 5.0
                )
            mchirp_col = pd.to_numeric(
                events_df.get("MChirp", pd.Series(dtype=float)), errors="coerce"
            ).astype(float)
            mchirp_all = mchirp_der.copy()
            # Use column where derived is not finite
            mchirp_all[~np.isfinite(mchirp_all)] = mchirp_col[~np.isfinite(mchirp_all)]
            # Prefer derived ChiEff; fall back to existing ChiEff column
            with np.errstate(invalid="ignore", divide="ignore"):
                chieff_der = (m1_all * chi1_all + m2_all * chi2_all) / (m1_all + m2_all)
            chieff_col = pd.to_numeric(
                events_df.get("ChiEff", pd.Series(dtype=float)), errors="coerce"
            ).astype(float)
            chieff_all = chieff_der.copy()
            chieff_all[~np.isfinite(chieff_all)] = chieff_col[~np.isfinite(chieff_all)]
            # Physical clipping for effective spin
            chieff_all = np.clip(chieff_all, -1.0, 1.0)
            # Build mask and plot
            mask_mc = (
                np.isfinite(mchirp_all) & (mchirp_all > 0) & np.isfinite(chieff_all)
            )
            if np.any(mask_mc):
                x = np.log10(mchirp_all[mask_mc])
                y = chieff_all[mask_mc]
                import matplotlib.gridspec as gridspec

                fig = plt.figure(figsize=(6, 6))
                gs = gridspec.GridSpec(
                    2,
                    2,
                    width_ratios=[4, 1.4],
                    height_ratios=[1.4, 4],
                    hspace=0.05,
                    wspace=0.05,
                )
                ax_scatter = fig.add_subplot(gs[1, 0])
                ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
                ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
                ax_scatter.scatter(x, y, s=12, alpha=0.6, edgecolors="none")
                ax_scatter.set_xlabel("log10(M_chirp [Msun])")
                ax_scatter.set_ylabel("Chi (ChiEff)")
                ax_scatter.set_title("log10(M_chirp) vs Chi")
                ax_scatter.grid(True, alpha=0.3)
                # Marginals
                ax_histx.hist(x, bins=30)
                ax_histy.hist(y, bins=30, orientation="horizontal")
                # Hide tick labels on shared axes for marginals
                plt.setp(ax_histx.get_xticklabels(), visible=False)
                plt.setp(ax_histy.get_yticklabels(), visible=False)
                ax_histx.tick_params(axis="x", which="both", length=0)
                ax_histy.tick_params(axis="y", which="both", length=0)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "scatter_log_mchirp_vs_chi.png"))
                plt.close(fig)
        except Exception as e:
            print(f"Warning: failed to write mass distribution plots: {e}")

    elif args.command == "psd-evolution":
        events_df, event_psds, rewhite_df, rewhite_psds = _require_datasets(out_dir)
        psd_hist = build_unified_psd_history(
            events_df, event_psds, rewhite_df, rewhite_psds
        )
        evo_df = compute_psd_evolution_metrics(psd_hist)
        evo_path = os.path.join(out_dir, "psd_evolution_summary.csv")
        evo_df.to_csv(evo_path, index=False)
        print(f"PSD evolution summary saved: {evo_path} ({len(evo_df)} rows)")
        # Scatter: EpsilonMean vs time delay, faceted by IFO, log-log
        try:
            if "Ifo" in evo_df.columns:
                ifos_plot = sorted(
                    set(str(x) for x in evo_df["Ifo"].dropna().unique().tolist())
                )
            else:
                ifos_plot = ["ALL"]
            eps_min = max(0.0, float(getattr(args, "epsilon_min", 1e-5))) or 1e-5
            n = max(1, len(ifos_plot))
            fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
            for idx, ifo in enumerate(ifos_plot):
                ax = axes[0, idx]
                sub = evo_df if ifo == "ALL" else evo_df[evo_df["Ifo"] == ifo]
                # Filter positive time deltas for log-log plotting
                sub = sub[(sub["TimeDeltaDays"] > 0)]
                if not sub.empty:
                    if "EpsilonMean" in sub.columns:
                        sub1 = sub[
                            np.isfinite(
                                pd.to_numeric(sub["EpsilonMean"], errors="coerce")
                            )
                            & (sub["EpsilonMean"] > 0)
                        ]
                        if not sub1.empty:
                            x = sub1["TimeDeltaDays"]
                            y = np.clip(
                                sub1["EpsilonMean"].astype(float), eps_min, np.inf
                            )
                            ax.scatter(x, y, s=16, alpha=0.6, label="Mean")
                    if "EpsilonMeanRobust" in sub.columns:
                        sub2 = sub[
                            np.isfinite(
                                pd.to_numeric(sub["EpsilonMeanRobust"], errors="coerce")
                            )
                            & (sub["EpsilonMeanRobust"] > 0)
                        ]
                        if not sub2.empty:
                            x = sub2["TimeDeltaDays"]
                            y = np.clip(
                                sub2["EpsilonMeanRobust"].astype(float), eps_min, np.inf
                            )
                            ax.scatter(x, y, s=16, alpha=0.6, label="Mean (robust)")
                    if "EpsilonMedian" in sub.columns:
                        sub3 = sub[
                            np.isfinite(
                                pd.to_numeric(sub["EpsilonMedian"], errors="coerce")
                            )
                            & (sub["EpsilonMedian"] > 0)
                        ]
                        if not sub3.empty:
                            x = sub3["TimeDeltaDays"]
                            y = np.clip(
                                sub3["EpsilonMedian"].astype(float), eps_min, np.inf
                            )
                            ax.scatter(x, y, s=16, alpha=0.6, label="Median")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("Time delay [days] (End - Start)")
                ax.set_ylabel("PSD epsilon")
                title = (
                    f"PSD epsilon vs time delay ({ifo})"
                    if ifo != "ALL"
                    else "PSD epsilon vs time delay"
                )
                ax.set_title(title)
                ax.grid(True, which="both", alpha=0.3)
                ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "psd_epsilon_vs_dt.png"))
            plt.close(fig)
            # Dedicated: Median epsilon vs time delay (faceted by IFO)
            try:
                n = max(1, len(ifos_plot))
                figm, axesm = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
                for idx, ifo in enumerate(ifos_plot):
                    axm = axesm[0, idx]
                    sub = evo_df if ifo == "ALL" else evo_df[evo_df["Ifo"] == ifo]
                    sub = sub[(sub["TimeDeltaDays"] > 0)]
                    sub = sub[
                        np.isfinite(
                            pd.to_numeric(
                                sub.get("EpsilonMedian", pd.Series(dtype=float)),
                                errors="coerce",
                            )
                        )
                    ]
                    if not sub.empty:
                        x = sub["TimeDeltaDays"]
                        y = np.clip(sub["EpsilonMedian"].astype(float), eps_min, np.inf)
                        axm.scatter(x, y, s=16, alpha=0.6, label="Median")
                    axm.set_xscale("log")
                    axm.set_yscale("log")
                    axm.set_xlabel("Time delay [days] (End - Start)")
                    axm.set_ylabel("PSD epsilon (median)")
                    axm.set_title(
                        f"PSD median epsilon vs time delay ({ifo})"
                        if ifo != "ALL"
                        else "PSD median epsilon vs time delay"
                    )
                    axm.grid(True, which="both", alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "psd_epsilon_median_vs_dt.png"))
                plt.close(figm)
            except Exception as e:
                print(f"Warning: failed to write psd_epsilon_median_vs_dt.png: {e}")
            # Histograms of epsilon metrics (mean and median)
            for col, fname, xlabel in [
                ("EpsilonMean", "hist_epsilon_mean.png", "PSD epsilon (mean)"),
                ("EpsilonMedian", "hist_epsilon_median.png", "PSD epsilon (median)"),
            ]:
                vals = pd.to_numeric(
                    evo_df.get(col, pd.Series(dtype=float)), errors="coerce"
                ).astype(float)
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if vals.size == 0:
                    continue
                vals = np.clip(vals, eps_min, np.inf)
                q95 = float(np.nanpercentile(vals, 95.0))
                fig_h, ax_h = plt.subplots(figsize=(6, 4))
                ax_h.hist(vals, bins=30)
                ax_h.axvline(q95, color="r", linestyle="--", label=f"95th: {q95:.3g}")
                # Place text annotation near the top-right
                ylim = ax_h.get_ylim()
                ax_h.text(
                    q95,
                    0.9 * ylim[1],
                    f"95th: {q95:.3g}",
                    color="r",
                    rotation=90,
                    va="top",
                    ha="left",
                )
                ax_h.set_xlabel(xlabel)
                ax_h.set_ylabel("Count")
                ax_h.grid(True, alpha=0.3)
                ax_h.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, fname))
                plt.close(fig_h)
            # Faceted histograms by IFO for mean and median epsilons with 95th quantile line
            try:
                metrics = [
                    (
                        "EpsilonMean",
                        "hist_epsilon_mean_by_ifo.png",
                        "PSD epsilon (mean)",
                    ),
                    (
                        "EpsilonMedian",
                        "hist_epsilon_median_by_ifo.png",
                        "PSD epsilon (median)",
                    ),
                ]
                n = max(1, len(ifos_plot))
                for col, fname, xlabel in metrics:
                    figf, axesf = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
                    for idx, ifo in enumerate(ifos_plot):
                        axf = axesf[0, idx]
                        sub = evo_df if ifo == "ALL" else evo_df[evo_df["Ifo"] == ifo]
                        vals = pd.to_numeric(
                            sub.get(col, pd.Series(dtype=float)), errors="coerce"
                        ).astype(float)
                        vals = vals[np.isfinite(vals) & (vals > 0)]
                        if vals.size:
                            vals = np.clip(vals, eps_min, np.inf)
                            axf.hist(vals, bins=30)
                            q95 = float(np.nanpercentile(vals, 95.0))
                            axf.axvline(
                                q95, color="r", linestyle="--", label=f"95th: {q95:.3g}"
                            )
                            ylim = axf.get_ylim()
                            axf.text(
                                q95,
                                0.9 * ylim[1],
                                f"95th: {q95:.3g}",
                                color="r",
                                rotation=90,
                                va="top",
                                ha="left",
                            )
                        axf.set_xlabel(xlabel)
                        axf.set_ylabel("Count")
                        axf.set_title(
                            f"{xlabel} ({ifo})" if ifo != "ALL" else f"{xlabel}"
                        )
                        axf.grid(True, alpha=0.3)
                        axf.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, fname))
                    plt.close(figf)
                # New: exact copy of hist_epsilon_median_by_ifo with log-scale on x-axis
                try:
                    # Only generate for EpsilonMedian
                    fname_log = "hist_epsilon_median_by_ifo_logx.png"
                    figl, axl = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
                    for idx, ifo in enumerate(ifos_plot):
                        ax = axl[0, idx]
                        sub = evo_df if ifo == "ALL" else evo_df[evo_df["Ifo"] == ifo]
                        vals = pd.to_numeric(
                            sub.get("EpsilonMedian", pd.Series(dtype=float)),
                            errors="coerce",
                        ).astype(float)
                        vals = vals[np.isfinite(vals) & (vals > 0)]
                        if vals.size:
                            vals = np.clip(vals, eps_min, np.inf)
                            logv = np.log10(vals)
                            ax.hist(logv, bins=30)
                            # Compute 95th percentile on original scale, draw at log10(q95)
                            q95 = float(np.nanpercentile(vals, 95.0))
                            xq = (
                                np.log10(q95)
                                if np.isfinite(q95) and (q95 > 0)
                                else None
                            )
                            if xq is not None:
                                ax.axvline(
                                    xq,
                                    color="r",
                                    linestyle="--",
                                    label=f"95th: {q95:.3g}",
                                )
                                ylim = ax.get_ylim()
                                ax.text(
                                    xq,
                                    0.9 * ylim[1],
                                    f"95th: {q95:.3g}",
                                    color="r",
                                    rotation=90,
                                    va="top",
                                    ha="left",
                                )
                        ax.set_xlabel("log10(PSD epsilon (median))")
                        ax.set_ylabel("Count")
                        ax.set_title(
                            f"PSD epsilon (median) ({ifo})"
                            if ifo != "ALL"
                            else "PSD epsilon (median)"
                        )
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, fname_log))
                    plt.close(figl)
                except Exception as e:
                    print(f"Warning: failed to write {fname_log}: {e}")
            except Exception as e:
                print(f"Warning: failed to write faceted epsilon histograms: {e}")
        except Exception as e:
            print(f"Warning: failed to write psd_epsilon_vs_dt.png: {e}")

    elif args.command == "skymap":
        # Load existing datasets and bias corrections (compute if missing)
        events_df, event_psds, rewhite_df, rewhite_psds = _require_datasets(out_dir)
        bc_path = os.path.join(out_dir, "bias_corrections.csv")
        if os.path.exists(bc_path):
            bias_df = pd.read_csv(bc_path)
        else:
            bias_df = compute_bias_corrections(
                events_df,
                event_psds,
                rewhite_df,
                rewhite_psds,
                ifo=args.ifo,
                fmin=args.fmin,
                fmax=args.fmax,
            )
            bias_df.to_csv(bc_path, index=False)
            print(f"[skymap] Bias corrections computed and saved: {bc_path}")
        if getattr(args, "verbose", False):
            # Pre-run informative logging
            uniq_events = (
                sorted(set(bias_df["EventID"].dropna().unique().tolist()))
                if (isinstance(bias_df, pd.DataFrame) and ("EventID" in bias_df))
                else []
            )
            print(
                f"[skymap] Starting skymap analysis with fref={getattr(args, 'fref', 100.0)} Hz"
            )
            print(
                f"[skymap] bias_corrections rows: {len(bias_df)}; events in bias: {len(uniq_events)}"
            )
            have_ra = (
                set(
                    events_df.loc[
                        np.isfinite(
                            pd.to_numeric(
                                events_df.get("ra", pd.Series(index=events_df.index)),
                                errors="coerce",
                            )
                        )
                        & np.isfinite(
                            pd.to_numeric(
                                events_df.get("dec", pd.Series(index=events_df.index)),
                                errors="coerce",
                            )
                        ),
                        "graceid",
                    ].tolist()
                )
                if not events_df.empty
                else set()
            )
            print(f"[skymap] events with finite RA/Dec in events_df: {len(have_ra)}")
        # Load per-IFO timing (if available) and pass to skymap
        evt_timing_df = load_events_timing(out_dir)
        sky_df = compute_skymap_deltas(
            events_df,
            bias_df,
            fref=getattr(args, "fref", 100.0),
            verbose=getattr(args, "verbose", False),
            events_timing=evt_timing_df,
        )
        diags = (
            sky_df.attrs.get("diagnostics", {})
            if isinstance(sky_df, pd.DataFrame)
            else {}
        )
        # If nothing produced, raise an informative error (by request)
        if sky_df.empty:
            # Print diagnostics summary
            print("[skymap] No skymap deltas were produced. Diagnostics summary:")
            for k, v in diags.items():
                if k != "skip_examples":
                    print(f"  - {k}: {v}")
            if diags.get("skip_examples"):
                show = diags["skip_examples"][:10]
                print("  - skip_examples (first 10):")
                for eid, reason in show:
                    print(f"    * {eid}: {reason}")
            raise RuntimeError(
                "skymap produced 0 rows. See diagnostics above. Common causes: (1) no events with RA/Dec; (2) fewer than two IFO corrections per event; (3) missing GPS epoch; (4) detector location mapping failure."
            )
        sky_path = os.path.join(out_dir, "skymap_deltas.csv")
        sky_df.to_csv(sky_path, index=False)
        print(f"Skymap delta summary saved: {sky_path} ({len(sky_df)} rows)")
        # Plots: histograms of dRA_deg, dDec_deg, dAng_deg
        try:
            for col, fname, xlabel in [
                ("dRA_deg", "hist_dRA_deg.png", "dRA [deg]"),
                ("dDec_deg", "hist_dDec_deg.png", "dDec [deg]"),
                ("dAng_deg", "hist_dAng_deg.png", "Angular shift [deg]"),
            ]:
                vals = pd.to_numeric(
                    sky_df.get(col, pd.Series(dtype=float)), errors="coerce"
                ).astype(float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(vals, bins=30)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Count")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, fname))
                plt.close(fig)
            # Scatter dRA vs dDec
            if set(["dRA_deg", "dDec_deg"]).issubset(set(sky_df.columns)):
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.scatter(sky_df["dRA_deg"], sky_df["dDec_deg"], s=14, alpha=0.7)
                ax.set_xlabel("dRA [deg]")
                ax.set_ylabel("dDec [deg]")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "scatter_dRA_dDec.png"))
                plt.close(fig)
                # Colored scatter by per-event median epsilon (from bias_corrections)
                try:
                    if (
                        isinstance(bias_df, pd.DataFrame)
                        and ("EventID" in bias_df.columns)
                        and ("PsdEpsilonMedian" in bias_df.columns)
                    ):
                        # Aggregate per-event by taking the median across IFO rows
                        med_eps = (
                            bias_df.groupby("EventID")["PsdEpsilonMedian"]
                            .median()
                            .rename("PsdEpsilonMedianEvent")
                        )
                        merged = sky_df.merge(
                            med_eps, left_on="EventID", right_index=True, how="left"
                        )
                        sub = merged[
                            ["dRA_deg", "dDec_deg", "PsdEpsilonMedianEvent"]
                        ].copy()
                        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(
                            subset=["dRA_deg", "dDec_deg", "PsdEpsilonMedianEvent"]
                        )
                        if not sub.empty:
                            cvals = sub["PsdEpsilonMedianEvent"].astype(float).values
                            # Robust color clipping to P2-P98
                            cmin = (
                                float(np.nanpercentile(cvals, 2.0))
                                if np.isfinite(cvals).any()
                                else float("nan")
                            )
                            cmax = (
                                float(np.nanpercentile(cvals, 98.0))
                                if np.isfinite(cvals).any()
                                else float("nan")
                            )
                            if (
                                (not np.isfinite(cmin))
                                or (not np.isfinite(cmax))
                                or (cmin == cmax)
                            ):
                                cmin = float(np.nanmin(cvals))
                                cmax = float(np.nanmax(cvals))
                            # Joint layout with marginal histograms (top and right)
                            import matplotlib.gridspec as gridspec

                            fig2 = plt.figure(figsize=(6.5, 6.5))
                            gs = gridspec.GridSpec(
                                2,
                                2,
                                width_ratios=[4, 1.2],
                                height_ratios=[1.2, 4],
                                hspace=0.06,
                                wspace=0.06,
                            )
                            ax_scatter = fig2.add_subplot(gs[1, 0])
                            ax_histx = fig2.add_subplot(gs[0, 0], sharex=ax_scatter)
                            ax_histy = fig2.add_subplot(gs[1, 1], sharey=ax_scatter)
                            # Main colored scatter
                            sc = ax_scatter.scatter(
                                sub["dRA_deg"],
                                sub["dDec_deg"],
                                c=np.clip(
                                    sub["PsdEpsilonMedianEvent"].astype(float),
                                    cmin,
                                    cmax,
                                ),
                                s=18,
                                cmap="viridis",
                                vmin=cmin,
                                vmax=cmax,
                                alpha=0.8,
                                edgecolors="none",
                            )
                            ax_scatter.set_xlabel("dRA [deg]")
                            ax_scatter.set_ylabel("dDec [deg]")
                            ax_scatter.set_title(
                                "dDec vs dRA colored by PSD median epsilon"
                            )
                            ax_scatter.grid(True, alpha=0.3)
                            # Marginal histograms
                            try:
                                ax_histx.hist(
                                    pd.to_numeric(sub["dRA_deg"], errors="coerce")
                                    .astype(float)
                                    .values,
                                    bins=30,
                                )
                                ax_histy.hist(
                                    pd.to_numeric(sub["dDec_deg"], errors="coerce")
                                    .astype(float)
                                    .values,
                                    bins=30,
                                    orientation="horizontal",
                                )
                            except Exception:
                                pass
                            # Hide tick labels on shared axes for marginals
                            plt.setp(ax_histx.get_xticklabels(), visible=False)
                            plt.setp(ax_histy.get_yticklabels(), visible=False)
                            ax_histx.tick_params(axis="x", which="both", length=0)
                            ax_histy.tick_params(axis="y", which="both", length=0)
                            # Colorbar attached to main scatter
                            cb = plt.colorbar(
                                sc, ax=ax_scatter, fraction=0.046, pad=0.04
                            )
                            cb.set_label("PSD median epsilon (bias)")
                            plt.tight_layout()
                            plt.savefig(
                                os.path.join(
                                    out_dir,
                                    "scatter_dRA_dDec_colored_by_median_epsilon.png",
                                )
                            )
                            plt.close(fig2)
                            # Also produce a log-log version using absolute values and a small floor
                            try:
                                eps_min = (
                                    max(0.0, float(getattr(args, "epsilon_min", 1e-5)))
                                    or 1e-5
                                )
                            except Exception:
                                eps_min = 1e-5
                            sub_log = sub.copy()
                            sub_log["abs_dRA"] = np.clip(
                                np.abs(sub_log["dRA_deg"].astype(float)),
                                eps_min,
                                np.inf,
                            )
                            sub_log["abs_dDec"] = np.clip(
                                np.abs(sub_log["dDec_deg"].astype(float)),
                                eps_min,
                                np.inf,
                            )
                            # Joint layout with marginal histograms (log-log)
                            import matplotlib.gridspec as gridspec

                            fig3 = plt.figure(figsize=(6.5, 6.5))
                            gs3 = gridspec.GridSpec(
                                2,
                                2,
                                width_ratios=[4, 1.2],
                                height_ratios=[1.2, 4],
                                hspace=0.06,
                                wspace=0.06,
                            )
                            ax_sc3 = fig3.add_subplot(gs3[1, 0])
                            ax_hx3 = fig3.add_subplot(gs3[0, 0], sharex=ax_sc3)
                            ax_hy3 = fig3.add_subplot(gs3[1, 1], sharey=ax_sc3)
                            # Main scatter on log-log axes
                            sc2 = ax_sc3.scatter(
                                sub_log["abs_dRA"],
                                sub_log["abs_dDec"],
                                c=np.clip(
                                    sub_log["PsdEpsilonMedianEvent"].astype(float),
                                    cmin,
                                    cmax,
                                ),
                                s=18,
                                cmap="viridis",
                                vmin=cmin,
                                vmax=cmax,
                                alpha=0.8,
                                edgecolors="none",
                            )
                            ax_sc3.set_xscale("log")
                            ax_sc3.set_yscale("log")
                            ax_sc3.set_xlabel("|dRA| [deg]")
                            ax_sc3.set_ylabel("|dDec| [deg]")
                            ax_sc3.set_title(
                                "|dDec| vs |dRA| colored by PSD median epsilon (log-log)"
                            )
                            ax_sc3.grid(True, which="both", alpha=0.3)
                            # Marginal histograms with log-spaced bins to match axes
                            try:
                                xvals = sub_log["abs_dRA"].astype(float).values
                                yvals = sub_log["abs_dDec"].astype(float).values
                                xvals = xvals[np.isfinite(xvals) & (xvals > 0)]
                                yvals = yvals[np.isfinite(yvals) & (yvals > 0)]
                                if xvals.size > 0 and yvals.size > 0:
                                    xmin, xmax = float(np.nanmin(xvals)), float(
                                        np.nanmax(xvals)
                                    )
                                    ymin, ymax = float(np.nanmin(yvals)), float(
                                        np.nanmax(yvals)
                                    )
                                    # Ensure strictly positive ranges
                                    xmin = max(xmin, eps_min)
                                    ymin = max(ymin, eps_min)
                                    if xmin < xmax and ymin < ymax:
                                        bx = np.logspace(
                                            np.log10(xmin), np.log10(xmax), 30
                                        )
                                        by = np.logspace(
                                            np.log10(ymin), np.log10(ymax), 30
                                        )
                                        ax_hx3.hist(xvals, bins=bx)
                                        ax_hy3.hist(
                                            yvals, bins=by, orientation="horizontal"
                                        )
                                        ax_hx3.set_xscale("log")
                                        ax_hy3.set_yscale("log")
                            except Exception:
                                pass
                            # Hide tick labels on shared axes for marginals
                            plt.setp(ax_hx3.get_xticklabels(), visible=False)
                            plt.setp(ax_hy3.get_yticklabels(), visible=False)
                            ax_hx3.tick_params(axis="x", which="both", length=0)
                            ax_hy3.tick_params(axis="y", which="both", length=0)
                            # Colorbar on main scatter
                            cb2 = plt.colorbar(sc2, ax=ax_sc3, fraction=0.046, pad=0.04)
                            cb2.set_label("PSD median epsilon (bias)")
                            plt.tight_layout()
                            plt.savefig(
                                os.path.join(
                                    out_dir,
                                    "scatter_dRA_dDec_colored_by_median_epsilon_loglog.png",
                                )
                            )
                            plt.close(fig3)
                except Exception as e:
                    print(f"Warning: failed to write colored scatter plot: {e}")
        except Exception as e:
            print(f"Warning: failed to write skymap plots: {e}")
        # Summarize outputs for the user
        try:
            created = []
            expected = [
                "skymap_deltas.csv",
                "hist_dRA_deg.png",
                "hist_dDec_deg.png",
                "hist_dAng_deg.png",
                "scatter_dRA_dDec.png",
                "scatter_dRA_dDec_colored_by_median_epsilon.png",
                "scatter_dRA_dDec_colored_by_median_epsilon_loglog.png",
            ]
            for name in expected:
                p = os.path.join(out_dir, name)
                if os.path.exists(p):
                    created.append(name)
            print("Finished. Outputs in %s" % out_dir)
            if created:
                print("Skymap figures:")
                for nm in created:
                    if nm.endswith(".png"):
                        print(f"  - {nm}")
            # Explain when the colored scatter is missing
            colored = os.path.join(
                out_dir, "scatter_dRA_dDec_colored_by_median_epsilon.png"
            )
            if not os.path.exists(colored):
                try:
                    has_eps = isinstance(bias_df, pd.DataFrame) and (
                        "PsdEpsilonMedian" in bias_df.columns
                    )
                except Exception:
                    has_eps = False
                if not has_eps:
                    print(
                        "Note: colored scatter (by median epsilon) was not created because bias_corrections.csv lacks 'PsdEpsilonMedian'. Run 'biases' to regenerate bias_corrections (now includes PsdEpsilonMedian), then rerun skymap."
                    )
        except Exception:
            pass

    elif args.command == "analyze":
        # Run both analyses using existing datasets in out_dir
        ev_df, ev_psds, rw_df, rw_psds = _require_datasets(out_dir)
        bc_df = compute_bias_corrections(
            ev_df, ev_psds, rw_df, rw_psds, ifo=args.ifo, fmin=args.fmin, fmax=args.fmax
        )
        bc_path = os.path.join(out_dir, "bias_corrections.csv")
        bc_df.to_csv(bc_path, index=False)
        print(f"Bias corrections saved: {bc_path} ({len(bc_df)} rows)")
        psd_hist = build_unified_psd_history(ev_df, ev_psds, rw_df, rw_psds)
        evo_df = compute_psd_evolution_metrics(psd_hist)
        evo_path = os.path.join(out_dir, "psd_evolution_summary.csv")
        evo_df.to_csv(evo_path, index=False)
        print(f"PSD evolution summary saved: {evo_path} ({len(evo_df)} rows)")
        # Scatter: Epsilon vs time delay, faceted by IFO, log-log
        try:
            if "Ifo" in evo_df.columns:
                ifos_plot = sorted(
                    set(str(x) for x in evo_df["Ifo"].dropna().unique().tolist())
                )
            else:
                ifos_plot = ["ALL"]
            eps_min = max(0.0, float(getattr(args, "epsilon_min", 1e-5))) or 1e-5
            n = max(1, len(ifos_plot))
            fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
            for idx, ifo in enumerate(ifos_plot):
                ax = axes[0, idx]
                sub = evo_df if ifo == "ALL" else evo_df[evo_df["Ifo"] == ifo]
                sub = sub[(sub["TimeDeltaDays"] > 0)]
                if not sub.empty:
                    if "EpsilonMean" in sub.columns:
                        sub1 = sub[
                            np.isfinite(
                                pd.to_numeric(sub["EpsilonMean"], errors="coerce")
                            )
                            & (sub["EpsilonMean"] > 0)
                        ]
                        if not sub1.empty:
                            ax.scatter(
                                sub1["TimeDeltaDays"],
                                np.clip(
                                    sub1["EpsilonMean"].astype(float), eps_min, np.inf
                                ),
                                s=16,
                                alpha=0.6,
                                label="Mean",
                            )
                    if "EpsilonMeanRobust" in sub.columns:
                        sub2 = sub[
                            np.isfinite(
                                pd.to_numeric(sub["EpsilonMeanRobust"], errors="coerce")
                            )
                            & (sub["EpsilonMeanRobust"] > 0)
                        ]
                        if not sub2.empty:
                            ax.scatter(
                                sub2["TimeDeltaDays"],
                                np.clip(
                                    sub2["EpsilonMeanRobust"].astype(float),
                                    eps_min,
                                    np.inf,
                                ),
                                s=16,
                                alpha=0.6,
                                label="Mean (robust)",
                            )
                    if "EpsilonMedian" in sub.columns:
                        sub3 = sub[
                            np.isfinite(
                                pd.to_numeric(sub["EpsilonMedian"], errors="coerce")
                            )
                            & (sub["EpsilonMedian"] > 0)
                        ]
                        if not sub3.empty:
                            ax.scatter(
                                sub3["TimeDeltaDays"],
                                np.clip(
                                    sub3["EpsilonMedian"].astype(float), eps_min, np.inf
                                ),
                                s=16,
                                alpha=0.6,
                                label="Median",
                            )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("Time delay [days] (End - Start)")
                ax.set_ylabel("PSD epsilon")
                title = (
                    f"PSD epsilon vs time delay ({ifo})"
                    if ifo != "ALL"
                    else "PSD epsilon vs time delay"
                )
                ax.set_title(title)
                ax.grid(True, which="both", alpha=0.3)
                ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "psd_epsilon_vs_dt.png"))
            plt.close(fig)
            # Dedicated: Median epsilon vs time delay (faceted by IFO)
            try:
                n = max(1, len(ifos_plot))
                figm, axesm = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
                for idx, ifo in enumerate(ifos_plot):
                    axm = axesm[0, idx]
                    sub = evo_df if ifo == "ALL" else evo_df[evo_df["Ifo"] == ifo]
                    sub = sub[(sub["TimeDeltaDays"] > 0)]
                    sub = sub[
                        np.isfinite(
                            pd.to_numeric(
                                sub.get("EpsilonMedian", pd.Series(dtype=float)),
                                errors="coerce",
                            )
                        )
                    ]
                    if not sub.empty:
                        x = sub["TimeDeltaDays"]
                        y = np.clip(sub["EpsilonMedian"].astype(float), eps_min, np.inf)
                        axm.scatter(x, y, s=16, alpha=0.6, label="Median")
                    axm.set_xscale("log")
                    axm.set_yscale("log")
                    axm.set_xlabel("Time delay [days] (End - Start)")
                    axm.set_ylabel("PSD epsilon (median)")
                    axm.set_title(
                        f"PSD median epsilon vs time delay ({ifo})"
                        if ifo != "ALL"
                        else "PSD median epsilon vs time delay"
                    )
                    axm.grid(True, which="both", alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "psd_epsilon_median_vs_dt.png"))
                plt.close(figm)
            except Exception as e:
                print(f"Warning: failed to write psd_epsilon_median_vs_dt.png: {e}")
            # Histograms of epsilon metrics (mean and median)
            for col, fname, xlabel in [
                ("EpsilonMean", "hist_epsilon_mean.png", "PSD epsilon (mean)"),
                ("EpsilonMedian", "hist_epsilon_median.png", "PSD epsilon (median)"),
            ]:
                vals = pd.to_numeric(
                    evo_df.get(col, pd.Series(dtype=float)), errors="coerce"
                ).astype(float)
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if vals.size == 0:
                    continue
                vals = np.clip(vals, eps_min, np.inf)
                q95 = float(np.nanpercentile(vals, 95.0))
                fig_h, ax_h = plt.subplots(figsize=(6, 4))
                ax_h.hist(vals, bins=30)
                ax_h.axvline(q95, color="r", linestyle="--", label=f"95th: {q95:.3g}")
                ylim = ax_h.get_ylim()
                ax_h.text(
                    q95,
                    0.9 * ylim[1],
                    f"95th: {q95:.3g}",
                    color="r",
                    rotation=90,
                    va="top",
                    ha="left",
                )
                ax_h.set_xlabel(xlabel)
                ax_h.set_ylabel("Count")
                ax_h.grid(True, alpha=0.3)
                ax_h.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, fname))
                plt.close(fig_h)
            # Faceted histograms by IFO for mean and median epsilons with 95th quantile line
            try:
                metrics = [
                    (
                        "EpsilonMean",
                        "hist_epsilon_mean_by_ifo.png",
                        "PSD epsilon (mean)",
                    ),
                    (
                        "EpsilonMedian",
                        "hist_epsilon_median_by_ifo.png",
                        "PSD epsilon (median)",
                    ),
                ]
                n = max(1, len(ifos_plot))
                for col, fname, xlabel in metrics:
                    figf, axesf = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
                    for idx, ifo in enumerate(ifos_plot):
                        axf = axesf[0, idx]
                        sub = evo_df if ifo == "ALL" else evo_df[evo_df["Ifo"] == ifo]
                        vals = pd.to_numeric(
                            sub.get(col, pd.Series(dtype=float)), errors="coerce"
                        ).astype(float)
                        vals = vals[np.isfinite(vals) & (vals > 0)]
                        if vals.size:
                            vals = np.clip(vals, eps_min, np.inf)
                            axf.hist(vals, bins=30)
                            q95 = float(np.nanpercentile(vals, 95.0))
                            axf.axvline(
                                q95, color="r", linestyle="--", label=f"95th: {q95:.3g}"
                            )
                            ylim = axf.get_ylim()
                            axf.text(
                                q95,
                                0.9 * ylim[1],
                                f"95th: {q95:.3g}",
                                color="r",
                                rotation=90,
                                va="top",
                                ha="left",
                            )
                        axf.set_xlabel(xlabel)
                        axf.set_ylabel("Count")
                        axf.set_title(
                            f"{xlabel} ({ifo})" if ifo != "ALL" else f"{xlabel}"
                        )
                        axf.grid(True, alpha=0.3)
                        axf.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, fname))
                    plt.close(figf)
                # New: exact copy of hist_epsilon_median_by_ifo with log-scale on x-axis
                try:
                    fname_log = "hist_epsilon_median_by_ifo_logx.png"
                    figl, axl = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
                    for idx, ifo in enumerate(ifos_plot):
                        ax = axl[0, idx]
                        sub = evo_df if ifo == "ALL" else evo_df[evo_df["Ifo"] == ifo]
                        vals = pd.to_numeric(
                            sub.get("EpsilonMedian", pd.Series(dtype=float)),
                            errors="coerce",
                        ).astype(float)
                        vals = vals[np.isfinite(vals) & (vals > 0)]
                        if vals.size:
                            vals = np.clip(vals, eps_min, np.inf)
                            logv = np.log10(vals)
                            ax.hist(logv, bins=30)
                            q95 = float(np.nanpercentile(vals, 95.0))
                            xq = (
                                np.log10(q95)
                                if np.isfinite(q95) and (q95 > 0)
                                else None
                            )
                            if xq is not None:
                                ax.axvline(
                                    xq,
                                    color="r",
                                    linestyle="--",
                                    label=f"95th: {q95:.3g}",
                                )
                                ylim = ax.get_ylim()
                                ax.text(
                                    xq,
                                    0.9 * ylim[1],
                                    f"95th: {q95:.3g}",
                                    color="r",
                                    rotation=90,
                                    va="top",
                                    ha="left",
                                )
                        ax.set_xlabel("log10(PSD epsilon (median))")
                        ax.set_ylabel("Count")
                        ax.set_title(
                            f"PSD epsilon (median) ({ifo})"
                            if ifo != "ALL"
                            else "PSD epsilon (median)"
                        )
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, fname_log))
                    plt.close(figl)
                except Exception as e:
                    print(f"Warning: failed to write {fname_log}: {e}")
            except Exception as e:
                print(f"Warning: failed to write faceted epsilon histograms: {e}")
        except Exception as e:
            print(f"Warning: failed to write psd_epsilon_vs_dt.png: {e}")

    elif args.command == "all":
        ev_df, ev_psds = _build_events_dataset_and_save(args)
        if args.rewhite_base_dir:
            # derive event time window if possible
            time_min = ev_df["event_time"].min() if "event_time" in ev_df else None
            time_max = ev_df["event_time"].max() if "event_time" in ev_df else None
            default_window = (
                (
                    pd.to_datetime(time_min).to_pydatetime(),
                    pd.to_datetime(time_max).to_pydatetime(),
                )
                if (time_min is not None and time_max is not None)
                else None
            )
            rw_df, rw_psds = _build_rewhite_dataset_and_save(args, default_window)
            # Run analyses
            bc_df = compute_bias_corrections(
                ev_df,
                ev_psds,
                rw_df,
                rw_psds,
                ifo=args.ifo,
                fmin=args.fmin,
                fmax=args.fmax,
            )
            bc_df.to_csv(os.path.join(out_dir, "bias_corrections.csv"), index=False)
            psd_hist = build_unified_psd_history(ev_df, ev_psds, rw_df, rw_psds)
            evo_df = compute_psd_evolution_metrics(psd_hist)
            evo_df.to_csv(
                os.path.join(out_dir, "psd_evolution_summary.csv"), index=False
            )
            # Scatter in 'all' pipeline as well (faceted by IFO, log-log)
            try:
                if "Ifo" in evo_df.columns:
                    ifos_plot = sorted(
                        set(str(x) for x in evo_df["Ifo"].dropna().unique().tolist())
                    )
                else:
                    ifos_plot = ["ALL"]
                eps_min = max(0.0, float(getattr(args, "epsilon_min", 1e-5))) or 1e-5
                n = max(1, len(ifos_plot))
                fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
                for idx, ifo in enumerate(ifos_plot):
                    ax = axes[0, idx]
                    sub = evo_df if ifo == "ALL" else evo_df[evo_df["Ifo"] == ifo]
                    sub = sub[(sub["TimeDeltaDays"] > 0)]
                    if not sub.empty:
                        if "EpsilonMean" in sub.columns:
                            sub1 = sub[
                                np.isfinite(
                                    pd.to_numeric(sub["EpsilonMean"], errors="coerce")
                                )
                                & (sub["EpsilonMean"] > 0)
                            ]
                            if not sub1.empty:
                                ax.scatter(
                                    sub1["TimeDeltaDays"],
                                    np.clip(
                                        sub1["EpsilonMean"].astype(float),
                                        eps_min,
                                        np.inf,
                                    ),
                                    s=16,
                                    alpha=0.6,
                                    label="Mean",
                                )
                        if "EpsilonMeanRobust" in sub.columns:
                            sub2 = sub[
                                np.isfinite(
                                    pd.to_numeric(
                                        sub["EpsilonMeanRobust"], errors="coerce"
                                    )
                                )
                                & (sub["EpsilonMeanRobust"] > 0)
                            ]
                            if not sub2.empty:
                                ax.scatter(
                                    sub2["TimeDeltaDays"],
                                    np.clip(
                                        sub2["EpsilonMeanRobust"].astype(float),
                                        eps_min,
                                        np.inf,
                                    ),
                                    s=16,
                                    alpha=0.6,
                                    label="Mean (robust)",
                                )
                        if "EpsilonMedian" in sub.columns:
                            sub3 = sub[
                                np.isfinite(
                                    pd.to_numeric(sub["EpsilonMedian"], errors="coerce")
                                )
                                & (sub["EpsilonMedian"] > 0)
                            ]
                            if not sub3.empty:
                                ax.scatter(
                                    sub3["TimeDeltaDays"],
                                    np.clip(
                                        sub3["EpsilonMedian"].astype(float),
                                        eps_min,
                                        np.inf,
                                    ),
                                    s=16,
                                    alpha=0.6,
                                    label="Median",
                                )
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.set_xlabel("Time delay [days] (End - Start)")
                    ax.set_ylabel("PSD epsilon")
                    title = (
                        f"PSD epsilon vs time delay ({ifo})"
                        if ifo != "ALL"
                        else "PSD epsilon vs time delay"
                    )
                    ax.set_title(title)
                    ax.grid(True, which="both", alpha=0.3)
                    ax.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "psd_epsilon_vs_dt.png"))
                plt.close(fig)
                # Histograms of epsilon metrics (mean and median)
                for col, fname, xlabel in [
                    ("EpsilonMean", "hist_epsilon_mean.png", "PSD epsilon (mean)"),
                    (
                        "EpsilonMedian",
                        "hist_epsilon_median.png",
                        "PSD epsilon (median)",
                    ),
                ]:
                    vals = pd.to_numeric(
                        evo_df.get(col, pd.Series(dtype=float)), errors="coerce"
                    ).astype(float)
                    vals = vals[np.isfinite(vals) & (vals > 0)]
                    if vals.size == 0:
                        continue
                    vals = np.clip(vals, eps_min, np.inf)
                    q95 = float(np.nanpercentile(vals, 95.0))
                    fig_h, ax_h = plt.subplots(figsize=(6, 4))
                    ax_h.hist(vals, bins=30)
                    ax_h.axvline(
                        q95, color="r", linestyle="--", label=f"95th: {q95:.3g}"
                    )
                    ylim = ax_h.get_ylim()
                    ax_h.text(
                        q95,
                        0.9 * ylim[1],
                        f"95th: {q95:.3g}",
                        color="r",
                        rotation=90,
                        va="top",
                        ha="left",
                    )
                    ax_h.set_xlabel(xlabel)
                    ax_h.set_ylabel("Count")
                    ax_h.grid(True, alpha=0.3)
                    ax_h.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, fname))
                    plt.close(fig_h)
            except Exception as e:
                print(f"Warning: failed to write psd_epsilon_vs_dt.png: {e}")
        else:
            print(
                "No --rewhite-base-dir provided; skipping biases and PSD evolution "
                "analyses."
            )

    print(f"Finished. Outputs in {out_dir}")


if __name__ == "__main__":
    main()


def _safe_float(v: object) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _fill_extrinsics_from_xml(xmldoc: ligolw.LIGO_LW, meta: Dict[str, object]) -> None:
    """Populate extrinsic parameters into meta from xml when available.
    - Per-IFO from SnglInspiral: end_time/end_time_ns → *_end_time(_ns/_gps),
      coa_phase → *_coa_phase, snr → *_snr, eff_distance → *_eff_distance.
    - CoincInspiral: end_time(_ns/_gps) → Coinc_*
    - SimInspiral: sky params (ra/dec/distance/inclination/polarization) if present.
    """
    # 1) Per-IFO from SnglInspiral (select loudest per IFO)
    try:
        sngl_tbl = lsctables.SnglInspiralTable.get_table(xmldoc)
        if len(sngl_tbl) > 0:
            best_by_ifo: Dict[str, object] = {}
            for row in sngl_tbl:
                ifo = getattr(row, "ifo", None)
                if not ifo:
                    continue
                cur = best_by_ifo.get(ifo)
                try:
                    snr_row = getattr(row, "snr", 0.0) or 0.0
                    snr_cur = getattr(cur, "snr", 0.0) if cur is not None else -1.0
                except Exception:
                    snr_row, snr_cur = 0.0, -1.0
                if (cur is None) or (snr_row >= snr_cur):
                    best_by_ifo[ifo] = row
            for ifo, row in best_by_ifo.items():
                # end times
                et = getattr(row, "end_time", None)
                etns = getattr(row, "end_time_ns", None)
                if et is not None:
                    meta.setdefault(f"{ifo}_end_time", et)
                if etns is not None:
                    meta.setdefault(f"{ifo}_end_time_ns", etns)
                try:
                    gps = None
                    if et is not None and etns is not None:
                        gps = float(et) + float(etns) * 1e-9
                    elif et is not None:
                        gps = float(et)
                    if gps is not None:
                        meta.setdefault(f"{ifo}_end_time_gps", gps)
                except Exception:
                    pass
                # impulse/"chirp" times
                it = getattr(row, "impulse_time", None)
                itns = getattr(row, "impulse_time_ns", None)
                if it is not None:
                    meta.setdefault(f"{ifo}_impulse_time", it)
                if itns is not None:
                    meta.setdefault(f"{ifo}_impulse_time_ns", itns)
                try:
                    igps = None
                    # Treat (0, 0) as missing; only accept non-zero components
                    if it is not None and itns is not None:
                        if (float(it) != 0.0) or (float(itns) != 0.0):
                            igps = float(it) + float(itns) * 1e-9
                    elif it is not None:
                        if float(it) != 0.0:
                            igps = float(it)
                    if (igps is not None) and np.isfinite(igps) and (igps > 1.0):
                        meta.setdefault(f"{ifo}_impulse_time_gps", igps)
                        # Expose user-facing ChirpTime_<IFO>
                        meta.setdefault(f"ChirpTime_{ifo}", igps)
                except Exception:
                    pass
                # per-IFO SNR
                if hasattr(row, "snr"):
                    meta.setdefault(f"{ifo}_snr", getattr(row, "snr"))
                # coalescence phase
                for name in ("coa_phase", "coaphase", "phase"):
                    if hasattr(row, name):
                        phase_val = getattr(row, name)
                        meta.setdefault(f"{ifo}_coa_phase", phase_val)
                        # Expose user-facing ChirpPhase_<IFO>
                        meta.setdefault(f"ChirpPhase_{ifo}", phase_val)
                        break
                # effective distance
                for name in ("eff_distance", "effdist", "effective_distance"):
                    if hasattr(row, name):
                        meta.setdefault(f"{ifo}_eff_distance", getattr(row, name))
                        break
    except Exception:
        pass

    # 2) CoincInspiral end times
    try:
        coinc_tbl = lsctables.Table.get_table(
            xmldoc, lsctables.CoincInspiralTable.tableName
        )
        # type: ignore[attr-defined]
        if len(coinc_tbl) > 0:
            crow = coinc_tbl[0]
            cet = getattr(crow, "end_time", None)
            cetns = getattr(crow, "end_time_ns", None)
            if cet is not None:
                meta.setdefault("Coinc_end_time", cet)
            if cetns is not None:
                meta.setdefault("Coinc_end_time_ns", cetns)
            try:
                gps = None
                if cet is not None and cetns is not None:
                    gps = float(cet) + float(cetns) * 1e-9
                elif cet is not None:
                    gps = float(cet)
                if gps is not None:
                    meta.setdefault("Coinc_end_time_gps", gps)
            except Exception:
                pass
    except Exception:
        pass

    # 3) SimInspiral sky parameters (MDC/injections)
    try:
        sim_tbl = lsctables.Table.get_table(
            xmldoc, lsctables.SimInspiralTable.tableName
        )  #
        # type: ignore[attr-defined]
        if len(sim_tbl) > 0:
            row = sim_tbl[0]
            # Right ascension
            for name in ("ra", "right_ascension"):
                if hasattr(row, name):
                    meta.setdefault("ra", getattr(row, name))
                    break
            # Declination
            for name in ("dec", "declination"):
                if hasattr(row, name):
                    meta.setdefault("dec", getattr(row, name))
                    break
            # Distance
            for name in ("distance", "dist"):
                if hasattr(row, name):
                    meta.setdefault("distance", getattr(row, name))
                    break
            # Inclination
            for name in ("inclination", "iota"):
                if hasattr(row, name):
                    meta.setdefault("inclination", getattr(row, name))
                    break
            # Polarization
            for name in ("polarization", "psi"):
                if hasattr(row, name):
                    meta.setdefault("polarization", getattr(row, name))
                    break
    except Exception:
        pass
