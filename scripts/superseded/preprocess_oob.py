"""Preprocess: OOB year-block bootstrap KDE predictive comparison.

For each station, evaluates which KDE bandwidth strategy (adaptive vs
Silverman) achieves higher per-observation log-likelihood on held-out
hydrological years, using a year-block out-of-bag (OOB) bootstrap.

Algorithm (per station)
-----------------------
  1. Split the station's complete hydrological years into K year-arrays,
     each holding daily UAR observations for that Oct-Sep cycle.
  2. Draw B=1000 bootstrap replicates.  Each replicate samples K years
     with replacement.  The OOB set is the complement (approx 37%).
  3. For each replicate:
       a. Fit adaptive KDE and Silverman KDE on the in-bag UAR pool.
       b. Bin the OOB observations onto the shared log-UAR grid.
       c. Compute per-observation mean log-PDF for each estimator:
              ll_adp[b] = mean( log pmf_adp[j] - log dlog[j] )
              ll_sil[b] = mean( log pmf_sil[j] - log dlog[j] )
          where j is the bin index for each OOB observation.
       d. delta_ll[b] = ll_adp[b] - ll_sil[b]
          (delta_ll is invariant to the dlog[j] correction, so the sign
           of delta_ll[b] = sign of mean( log pmf_adp[j] - log pmf_sil[j] ))
  4. Replicates with fewer than MIN_OOB_OBS = 30 OOB observations are skipped.

Outputs (per region, written to cache/)
---------------------------------------
  {region}_oob_scores.parquet
      One row per station.  Columns:
        station_id, region, n_years, n_valid_replicates,
        delta_ll_mean, delta_ll_ci_lo, delta_ll_ci_hi   (2.5 / 97.5 pct)
        adp_ll_mean,   adp_ll_ci_lo,   adp_ll_ci_hi
        sil_ll_mean,   sil_ll_ci_lo,   sil_ll_ci_hi
        p_adp_better   (fraction of valid replicates with delta_ll > 0)

Usage
-----
    python preprocess_oob.py [region|index|all]

Prerequisites
-------------
    sensitivity_report/preprocess.py (step 0 + step 1 + step 1a) must be
    complete so that complete_year_stats.parquet and station_meta.parquet
    exist for the target region.

Notes
-----
    The OOB fraction is approximately 1 - (1 - 1/K)^K → 1 - 1/e ≈ 36.8 %.
    For stations with K < 5 complete years the expected OOB set per replicate
    may be smaller than MIN_OOB_OBS; such replicates are automatically skipped
    and n_valid_replicates will be < B.

    Drainage area (da_km2) is required to convert UAR to volumetric flow for
    the adaptive bandwidth error model.  Silverman's rule does not use da_km2,
    but the measurement-error floor applied to Silverman also requires it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR        = Path(__file__).resolve().parent
REPO_ROOT         = SCRIPT_DIR.parent.parent.parent
CACHE_DIR         = SCRIPT_DIR / "cache"
PARENT_CACHE_ROOT = SCRIPT_DIR.parent.parent / "cache"

sys.path.insert(0, str(REPO_ROOT))

from config import Config                       # noqa: E402
from utils.kde_estimator import KDEEstimator    # noqa: E402

CARAVAN_ROOT  = Path('/home/danbot/Documents/common_data/Caravan-csv')
TS_ROOT       = CARAVAN_ROOT / 'timeseries' / 'csv'
DATA_DIR      = REPO_ROOT / "docs" / "notebooks" / "data"
CY_STATS_DIR  = DATA_DIR / "complete_year_stats"

# ---------------------------------------------------------------------------
# Bootstrap / grid parameters
# ---------------------------------------------------------------------------

B             = 500
REF_BITRATE   = 8
MIN_OOB_OBS   = 30      # minimum OOB observations per replicate to be valid
LOG_EPS       = 1e-30   # PMF floor before log

# Minimum complete years required per region.  Regions not listed default to 2.
REGION_MIN_YEARS: dict[str, int] = {"hysets": 10, "hysets_bc": 10}


# ---------------------------------------------------------------------------
# Region selection
# ---------------------------------------------------------------------------

def _resolve_regions() -> list[str]:
    regions = sorted(d.name for d in PARENT_CACHE_ROOT.iterdir() if d.is_dir())
    if not regions:
        raise SystemExit(
            f"No processed regions found in:\n  {PARENT_CACHE_ROOT}\n"
            "Run sensitivity_report/preprocess.py first."
        )
    print("Available regions:")
    for i, name in enumerate(regions):
        print(f"  {i}: {name}")

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "all":
            print("Processing all regions.")
            return regions
        if arg.isdigit():
            idx = int(arg)
            if 0 <= idx < len(regions):
                print(f"Selected: {regions[idx]}")
                return [regions[idx]]
            raise ValueError(f"Index {idx} out of range (0-{len(regions)-1}).")
        if arg in regions:
            print(f"Selected: {arg}")
            return [arg]
        raise ValueError(f"Region '{arg}' not found. Available: {regions}")

    while True:
        raw = input(f"Enter number (0-{len(regions)-1}) or 'all': ").strip()
        if raw == "all":
            print("Processing all regions.")
            return regions
        if raw.isdigit():
            idx = int(raw)
            if 0 <= idx < len(regions):
                print(f"Selected: {regions[idx]}")
                return [regions[idx]]
        print(f"Invalid. Enter a number between 0 and {len(regions)-1}, or 'all'.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_stale(out: Path, *sources: Path) -> bool:
    if not out.exists():
        return True
    out_mtime = out.stat().st_mtime
    return any(s.exists() and s.stat().st_mtime > out_mtime for s in sources)


def _build_log_edges() -> np.ndarray:
    """Return (N_BINS+1,) bin edges consistent with the pipeline PMF grid."""
    n_bins      = 2 ** REF_BITRATE
    log_min     = np.log(Config.GLOBAL_MIN_UAR)
    log_max     = np.log(Config.GLOBAL_MAX_UAR)
    right_edges = np.linspace(log_min, log_max, n_bins)
    h           = right_edges[1] - right_edges[0]
    return np.concatenate([[right_edges[0] - h], right_edges])  # (N_BINS+1,)


def _load_year_arrays(
    region: str,
    stn_id: str,
    hyd_years: list[int],
) -> dict[int, np.ndarray]:
    """Load UAR arrays split by complete hydrological year.

    Returns dict mapping year (int) to a float32 array of positive UAR values
    (L/s/km²) for all days in that complete Oct-Sep cycle.  Years for which
    no timeseries data is available are omitted silently.

    UAR conversion: streamflow (mm/day) * 1e6 / 86400 (no drainage area needed;
    the drainage area is used separately by KDEEstimator for the bandwidth error
    model, not for the unit conversion).
    """
    ts_path = TS_ROOT / region / f"{stn_id}.csv"
    if not ts_path.exists():
        return {}

    try:
        ts = pd.read_csv(ts_path, parse_dates=["date"], index_col="date")
    except Exception:
        return {}

    if "streamflow" not in ts.columns:
        return {}

    sf  = ts["streamflow"].dropna()
    uar = (sf * (1e6 / 86400.0)).clip(lower=Config.GLOBAL_MIN_UAR)

    # Water year label: Oct-Sep cycle ending in year Y uses period label Y.
    periods = sf.index.to_period("Y-SEP")

    result: dict[int, np.ndarray] = {}
    for yr in hyd_years:
        mask   = periods.year == yr
        if not mask.any():
            continue
        yr_uar = uar[mask].values.astype(np.float32)
        yr_uar = yr_uar[yr_uar > 0]
        if len(yr_uar) > 0:
            result[yr] = yr_uar
    return result


def _oob_one_station(
    stn_id:      str,
    year_arrays: dict[int, np.ndarray],
    da_km2:      float,
    log_edges:   np.ndarray,
    seed:        int,
) -> dict:
    """Run B-replicate OOB bootstrap for one station.

    Returns a dict with all output columns; NaN values are used when no valid
    replicates were obtained (e.g. too few years).
    """
    NAN_ROW = {
        "station_id":         stn_id,
        "n_years":            len(year_arrays),
        "n_valid_replicates": 0,
        "delta_ll_mean":      np.nan,
        "delta_ll_ci_lo":     np.nan,
        "delta_ll_ci_hi":     np.nan,
        "adp_ll_mean":        np.nan,
        "adp_ll_ci_lo":       np.nan,
        "adp_ll_ci_hi":       np.nan,
        "sil_ll_mean":        np.nan,
        "sil_ll_ci_lo":       np.nan,
        "sil_ll_ci_hi":       np.nan,
        "p_adp_better":       np.nan,
    }

    years = sorted(year_arrays.keys())
    K     = len(years)
    if K < 2:
        return NAN_ROW

    n_bins = len(log_edges) - 1
    dlog   = np.diff(log_edges)              # (N_BINS,) bin widths in log space
    log_dlog = np.log2(dlog + LOG_EPS)       # pre-computed for log2-PDF correction (bits)

    kde = KDEEstimator(log_edges)
    rng = np.random.default_rng(seed)

    delta_ll_reps: list[float] = []
    adp_ll_reps:   list[float] = []
    sil_ll_reps:   list[float] = []

    for _ in range(B):
        in_bag_idx = rng.integers(0, K, size=K)
        in_bag_set = set(in_bag_idx.tolist())
        oob_idx    = [i for i in range(K) if i not in in_bag_set]

        if not oob_idx:
            continue

        oob_uar    = np.concatenate([year_arrays[years[i]] for i in oob_idx])
        if len(oob_uar) < MIN_OOB_OBS:
            continue

        in_bag_uar = np.concatenate([year_arrays[years[i]] for i in in_bag_idx])
        if len(in_bag_uar) < 2:
            continue

        try:
            pmf_adp, pmf_sil = kde.compute_both(in_bag_uar, da_km2)
        except Exception:
            continue

        # Bin OOB observations onto the log-UAR grid.
        log_oob = np.log(np.clip(oob_uar, np.exp(log_edges[0]), np.exp(log_edges[-1])))
        bin_j   = np.clip(
            np.searchsorted(log_edges, log_oob, side="right") - 1,
            0, n_bins - 1,
        )

        # Per-observation log2-PDF = log2(PMF) - log2(bin_width), units: bits/obs
        ll_adp = np.log2(pmf_adp[bin_j] + LOG_EPS) - log_dlog[bin_j]
        ll_sil = np.log2(pmf_sil[bin_j] + LOG_EPS) - log_dlog[bin_j]

        delta_ll_reps.append(float(np.mean(ll_adp - ll_sil)))
        adp_ll_reps.append(float(np.mean(ll_adp)))
        sil_ll_reps.append(float(np.mean(ll_sil)))

    if not delta_ll_reps:
        return NAN_ROW

    delta = np.array(delta_ll_reps)
    adp   = np.array(adp_ll_reps)
    sil   = np.array(sil_ll_reps)

    return {
        "station_id":         stn_id,
        "n_years":            K,
        "n_valid_replicates": len(delta),
        "delta_ll_mean":      float(delta.mean()),
        "delta_ll_ci_lo":     float(np.percentile(delta, 2.5)),
        "delta_ll_ci_hi":     float(np.percentile(delta, 97.5)),
        "adp_ll_mean":        float(adp.mean()),
        "adp_ll_ci_lo":       float(np.percentile(adp, 2.5)),
        "adp_ll_ci_hi":       float(np.percentile(adp, 97.5)),
        "sil_ll_mean":        float(sil.mean()),
        "sil_ll_ci_lo":       float(np.percentile(sil, 2.5)),
        "sil_ll_ci_hi":       float(np.percentile(sil, 97.5)),
        "p_adp_better":       float(np.mean(delta > 0)),
    }


# ---------------------------------------------------------------------------
# Region-level runner
# ---------------------------------------------------------------------------

def preprocess_oob_region(region: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"{region}_oob_scores.parquet"

    meta_path  = PARENT_CACHE_ROOT / region / "station_meta.parquet"
    cy_path    = CY_STATS_DIR / f"{region}_complete_year_stats.parquet"

    for p in (meta_path, cy_path):
        if not p.exists():
            print(f"  {region}: required file missing: {p}")
            print("  Run sensitivity_report/preprocess.py steps 0 and 1 first.")
            return

    sources = [Path(__file__), meta_path, cy_path]
    if not _is_stale(out_path, *sources):
        print(f"  {region}: OOB cache up to date, skipping.")
        return

    # Load metadata
    meta   = pd.read_parquet(meta_path, columns=["da_km2", "record_years"])
    cy_df  = pd.read_parquet(cy_path)

    min_yrs = REGION_MIN_YEARS.get(region, 2)

    # Build station list: must appear in both meta and cy_stats
    common_stns = meta.index.intersection(cy_df.index).tolist()
    print(f"  {region}: {len(common_stns)} stations in meta + cy_stats")

    log_edges = _build_log_edges()

    # Seed RNG once; generate per-station seeds for reproducible parallelism
    master_rng = np.random.default_rng(seed=42)
    seeds      = master_rng.integers(0, 2**31, size=len(common_stns))

    try:
        from joblib import Parallel, delayed

        def _worker(stn, seed):
            hyd_years = list(cy_df.loc[stn, "hyd_years"])
            if len(hyd_years) < min_yrs:
                return None
            da_km2     = float(meta.loc[stn, "da_km2"])
            year_arrays = _load_year_arrays(region, stn, hyd_years)
            if len(year_arrays) < 2:
                return None
            return _oob_one_station(stn, year_arrays, da_km2, log_edges, seed)

        print(f"  {region}: running OOB bootstrap (B={B}) with joblib …")
        results = Parallel(n_jobs=-1, verbose=5)(
            delayed(_worker)(stn, int(seeds[i]))
            for i, stn in enumerate(common_stns)
        )

    except ImportError:
        print(
            f"  {region}: joblib not available, running single-threaded "
            f"(B={B}) …"
        )
        results = []
        for i, stn in enumerate(common_stns):
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(common_stns)} …")
            hyd_years = list(cy_df.loc[stn, "hyd_years"])
            if len(hyd_years) < min_yrs:
                results.append(None)
                continue
            da_km2      = float(meta.loc[stn, "da_km2"])
            year_arrays = _load_year_arrays(region, stn, hyd_years)
            if len(year_arrays) < 2:
                results.append(None)
                continue
            results.append(_oob_one_station(stn, year_arrays, da_km2, log_edges, int(seeds[i])))

    rows = [r for r in results if r is not None]
    if not rows:
        print(f"  {region}: no valid stations, skipping output.")
        return

    df = pd.DataFrame(rows)
    df["region"] = region
    df.to_parquet(out_path, index=False)
    n_valid = (df["n_valid_replicates"] > 0).sum()
    pct_adp = (df["p_adp_better"] > 0.5).mean() * 100
    print(
        f"  {region}: {len(df)} stations, {n_valid} with valid replicates, "
        f"{pct_adp:.1f}% favour adaptive -> {out_path.name}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    regions = _resolve_regions()
    for region in regions:
        print(f"\nProcessing: {region}")
        preprocess_oob_region(region)


if __name__ == "__main__":
    main()
