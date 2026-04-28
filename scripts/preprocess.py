"""
KDE reference distribution computation pipeline.

Each step is independently cacheable. Run main() to execute the core
pipeline steps in dependency order, skipping steps whose cache files
already exist.  Optional and report-side computation functions are
defined in this file but not called from main(). See the comments in
main() for guidance.
"""
import os
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure repo root is on path so config and utils are importable from any cwd.
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import Config

from utils import apply_kld_limited_uniform_mixture, ensure_parquet_engine, filter_complete_years

from utils import pmf_to_log_quantiles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = REPO_ROOT / "cache"
BASELINE_DIR = DATA_DIR / "baseline_distributions"

# ---------------------------------------------------------------------------
# Caravan data root (timeseries and attributes per region sub-folder)
# ---------------------------------------------------------------------------
CARAVAN_ROOT = Path('/home/danbot/Documents/common_data/Caravan-csv')
TS_ROOT      = CARAVAN_ROOT / 'timeseries' / 'csv'
ATTR_ROOT    = CARAVAN_ROOT / 'attributes'

COMPLETE_YEAR_STATS = DATA_DIR / "complete_year_stats.npy"

# Active region name. Set once at startup via init_region().
REGION: str = ""

# Caravan/HydroATLAS column name -> canonical pipeline name.
_CARAVAN_COL_MAP = {
    'gauge_id':          'station_id',
    'gauge_lat':         'lat',
    'gauge_lon':         'lon',
    'area':              'da_km2',
    'p_mean':            'prcp',
    'tmp_dc_syr':        'tmean',
    'snw_pc_syr':        'swe',
    'high_prec_freq':    'high_prcp_freq',
    'low_prec_freq':     'low_prcp_freq',
    'high_prec_dur':     'high_prcp_duration',
    'low_prec_dur':      'low_prcp_duration',
    'slp_dg_sav':        'slope_deg',
    'ele_mt_sav':        'elevation_m',
    'for_pc_sse':        'land_use_forest_frac',
    'gla_pc_sse':        'land_use_snow_ice_frac',
}

# Shared data loaded once by load_shared_data() after build_station_meta() runs.
META:       pd.DataFrame | None = None
YEAR_STATS: dict | None = None
DA_DICT:    dict | None = None

REF_BITRATE = Config.DEFAULT_BITRATE
# ---------------------------------------------------------------------------
# Helper structures
# ---------------------------------------------------------------------------

# Estimators used at ref bitrate=8 for cross-estimator analyses
_REF_ESTIMATORS = ["bch", "bch_u", "ab_kde", "ab_kde_u", "sil_kde", "sil_kde_u",
                   "lnmle", "lnmle_u"]



def init_region(name: str) -> None:
    """Set the active region for the entire preprocess module.  Call once before
    invoking any pipeline step."""
    global REGION, COMPLETE_YEAR_STATS
    REGION = name
    COMPLETE_YEAR_STATS = DATA_DIR / "complete_year_stats" / f"{name}_complete_year_stats.parquet"


def _load_year_stats() -> dict:
    """Load the complete-year stats dict from the region-scoped parquet cache.

    Returns a dict keyed by station_id.  Each value is itself a dict with:
        hyd_years (list[int]), n_hyd_years, mean/median/stdev/mad stats, etc.
    """
    df = pd.read_parquet(COMPLETE_YEAR_STATS)
    result = {}
    for stn, row in df.iterrows():
        entry = row.to_dict()
        # hyd_years stored as Python list. Parquet should load correctly
        # but coerce elements to int for safe set/membership operations.
        entry['hyd_years'] = [int(y) for y in entry['hyd_years']]
        result[stn] = entry
    return result


def load_shared_data() -> None:
    """Load META, YEAR_STATS, and DA_DICT once into module globals.
    Call from main() after build_station_meta() has written its cache."""
    global META, YEAR_STATS, DA_DICT
    META       = pd.read_parquet(_cache("station_meta.parquet"))
    YEAR_STATS = _load_year_stats()
    DA_DICT    = META["da_km2"].to_dict()


def get_processed_regions() -> list[str]:
    """Return the list of regions to process.

    Pass a region name, index, or 'all' as a CLI argument to skip the prompt:
        python preprocess.py hysets_bc
        python preprocess.py 0
        python preprocess.py all
    """
    folders = sorted(d.name for d in TS_ROOT.iterdir() if d.is_dir())
    print("Available regions:")
    for i, folder in enumerate(folders):
        print(f"  {i}: {folder}")
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "all":
            print("Processing all regions.")
            return folders
        elif arg.isdigit():
            choice = int(arg)
            if 0 <= choice < len(folders):
                selected = folders[choice]
            else:
                raise ValueError(f"Region index {choice} out of range (0-{len(folders)-1})")
        else:
            if arg in folders:
                selected = arg
            else:
                raise ValueError(f"Region '{arg}' not found. Available: {folders}")
    else:
        while True:
            raw = input(f"Enter a number (0-{len(folders)-1}) or 'all': ").strip()
            if raw == "all":
                print("Processing all regions.")
                return folders
            if raw.isdigit():
                choice = int(raw)
                if 0 <= choice < len(folders):
                    break
            print(f"Invalid input. Enter a number between 0 and {len(folders)-1}, or 'all'.")
        selected = folders[choice]
    print(f"Selected region: {selected}")
    return [selected]


def _cache(name: str) -> Path:
    """Return path to a cache file, scoped to the active region."""
    region_cache = CACHE_DIR / REGION if REGION else CACHE_DIR
    region_cache.mkdir(parents=True, exist_ok=True)
    return region_cache / name


def _pmf_dir(bitrate: int) -> Path:
    """Return the baseline_distributions sub-directory for the active region and bitrate."""
    return BASELINE_DIR / REGION / f"{bitrate:02d}_bits" if REGION else BASELINE_DIR / f"{bitrate:02d}_bits"


def _load_pmf_source(est_key: str, bitrate: int=REF_BITRATE) -> pd.DataFrame:
    """Load a PMF DataFrame (index=log_x_uar, columns=station_id) for a given estimator."""
    d = _pmf_dir(bitrate)
    if est_key == "bch":
        return pd.read_csv(d / "pmf_obs.csv", index_col=0)
    if est_key == "bch_u":
        return pd.read_csv(d / "pmf_obs_mixture.csv", index_col=0)
    if est_key == "ab_kde":
        return pd.read_csv(d / "pmf_kde_adaptive.csv", index_col=0)
    if est_key == "ab_kde_u":
        return pd.read_csv(d / "pmf_kde_adaptive_mixture.csv", index_col=0)
    if est_key == "sil_kde":
        return pd.read_csv(d / "pmf_kde_silverman.csv", index_col=0)
    if est_key == "sil_kde_u":
        return pd.read_csv(d / "pmf_kde_silverman_mixture.csv", index_col=0)
    if est_key == "lnmle":
        return pd.read_csv(d / "pmf_lnmle.csv", index_col=0)
    if est_key == "lnmle_u":
        return pd.read_csv(d / "pmf_lnmle_mixture.csv", index_col=0)
    raise ValueError(f"Unknown est_key: {est_key!r}")


# ---------------------------------------------------------------------------
# Timeseries helpers (Caravan)
# ---------------------------------------------------------------------------

def load_timeseries_data(stn: str) -> "pd.DataFrame | None":
    """Load daily streamflow timeseries for a station from the Caravan store.

    The region sub-folder is taken from the module global REGION when the
    ``region`` argument is not supplied.

    """
    ts_path = TS_ROOT / REGION / f"{stn}.csv"
    if not ts_path.exists():
        raise Exception(f"Timeseries file not found: {ts_path}")
    
    df = pd.read_csv(ts_path, parse_dates=['date'], index_col='date')
    df = df.dropna(subset=['streamflow'])
    if len(df) < 365:
        return pd.DataFrame(), {}
    return df


def retrieve_and_preprocess_timeseries_discharge(
    stn: str,
    da: float,
) -> "tuple[pd.DataFrame, dict]":
    """Load and preprocess Caravan daily streamflow for one station.

    Parameters
    ----------
    stn:    Station ID (must match the CSV filename without the region prefix).
    da:     Drainage area in km².

    Returns
    -------
    df:    DataFrame indexed by date with columns:
               ``streamflow``  (mm/day, original)
               ``flow_m3s``    (m³/s)
               ``{stn}_uar``   (specific discharge, L/s/km²)
               ``{stn}_log_uar``
    stats: Dict with scalar summaries used by downstream steps.
    """
    zero_flow_threshold = Config.ZERO_FLOW_THRESHOLD
    assert da > 0, f"Drainage area for {stn} must be positive, got {da}"
    df = load_timeseries_data(stn)

    zero_flow_flag     = bool((df['streamflow'] <= zero_flow_threshold).any())
    min_nonzero_mmd    = float(df[df['streamflow'] > 0]['streamflow'].min())

    df['flow_m3s']     = df['streamflow'] * 1000.0 * da / (24 * 3600)
    min_nonzero_uar    = float(1000.0 * df[df['flow_m3s'] > 0]['flow_m3s'].min() / da)  # diagnostic only
    max_uar            = float(1000.0 * df['flow_m3s'].max() / da)

    min_uar            = 1000.0 * zero_flow_threshold / da
    df[f'{stn}_uar']     = 1000.0 * df['flow_m3s'] / da
    df[f'{stn}_log_uar'] = np.log(df[f'{stn}_uar'].clip(lower=min_uar))

    stats = {
        'zero_flow_flag':        zero_flow_flag,           # bool
        'min_nonzero_flow_m3s':  float(df[df['flow_m3s'] > 0]['flow_m3s'].min()),  # m³/s
        'min_nonzero_flow_mmd':  min_nonzero_mmd,          # mm/day (raw Caravan units)
        'max_flow_m3s':          float(df['flow_m3s'].max()),  # m³/s
        'min_nonzero_uar':       min_nonzero_uar,           # L/s/km²
        'max_uar':               max_uar,                   # L/s/km²
        'drainage_area_km2':     da,                        # km²
    }
    return df, stats


def _retrieve_discharge(stn: str, da_km2: float) -> tuple[pd.DataFrame, bool]:
    """Thin adapter: call retrieve_and_preprocess_timeseries_discharge and return
    (df_with_discharge_col, zero_flow_flag) for use with ReferenceDistribution."""
    df, stats = retrieve_and_preprocess_timeseries_discharge(stn, da=da_km2)
    if df.empty:
        return df, False
    return df.rename(columns={"flow_m3s": "discharge"}), stats["zero_flow_flag"]


# ---------------------------------------------------------------------------
# Shared probe grid and score helpers
# ---------------------------------------------------------------------------

# Probe grid for pairwise metric evaluation: 100 equally-spaced centile levels
# trimmed at Config.PROBE_ALPHA on each tail.  Used as the shared evaluation
# grid for NAE, RMSE, KGE, etc., not as annual quantile summaries.
# The Weibull reference in compute_weibull_quantiles stores each station's
# exact n sorted observations with their k/(n+1) plotting positions.
_PROBE_PROBS = Config.PROBE_PROBS

# ---------------------------------------------------------------------------
# Step 0: compute_complete_year_stats
# ---------------------------------------------------------------------------
def compute_complete_year_stats(force: bool = False) -> dict:
    """Compute complete hydrological year statistics for every station in the
    active Caravan region and cache as parquet.

    Streamflow is read from the Caravan timeseries (mm/day).  A hydrological
    year runs Oct-Sep (Config.HYD_MS = 'SEP').  A year is complete when every
    month has at least Config.MIN_DAYS_PER_MONTH valid observations.

    Output
    ------
    DATA_DIR / 'complete_year_stats' / '{REGION}_complete_year_stats.parquet'
        One row per station.  Columns:
            hyd_years (list[int])
            n_hyd_years, mean_annual_uar_hyd, mean_annual_log_uar_hyd,
            median_annual_uar_hyd, median_annual_log_uar_hyd,
            stdev_annual_uar_hyd, mad_annual_uar_hyd,
            min_nonzero_uar, max_uar,
            n_obs_hyd, n_unique_uar_hyd
    """

    out_dir = DATA_DIR / "complete_year_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = COMPLETE_YEAR_STATS

    if out_path.exists() and not force:
        print(f"compute_complete_year_stats: check cache ({out_path.name})")
        return _load_year_stats()
        
    print(f"compute_complete_year_stats: processing region '{REGION}'...")

    ts_dir   = TS_ROOT / REGION
    all_csv  = sorted(os.listdir(ts_dir))

    assert len(all_csv) > 0, f"No CSV files found in {ts_dir}"
    hyd_ms   = Config.HYD_MS
    min_days = Config.MIN_DAYS_PER_MONTH

    rows: list[dict] = []
    n_skip = 0
    for i, csv_name in enumerate(all_csv):
        stn = csv_name.split('.')[0]
        df = load_timeseries_data(stn)
        s = df['streamflow'].dropna().sort_index()

        # mm/day -> L/s/km²    (unit-area runoff, constant factor, no da needed)
        uar     = s * (1e6 / 86400.0)
        log_uar = np.log(uar.clip(lower=1e-9))

        mask_hyd, complete_hyd_years = filter_complete_years(s, hyd_ms, min_days)
        if len(complete_hyd_years) == 0:
            n_skip += 1
            continue

        hyd_uar     = uar[mask_hyd]
        hyd_log_uar = log_uar[mask_hyd]

        rows.append({
            'station_id':               stn,
            'hyd_years':                [int(y) for y in complete_hyd_years],
            'n_hyd_years':              int(len(complete_hyd_years)),
            'mean_annual_uar_hyd':      float(hyd_uar.mean()),
            'mean_annual_log_uar_hyd':  float(hyd_log_uar.mean()),
            'median_annual_uar_hyd':    float(hyd_uar.median()),
            'median_annual_log_uar_hyd': float(hyd_log_uar.median()),
            'stdev_annual_uar_hyd':     float(hyd_uar.std(ddof=1)),
            'mad_annual_uar_hyd':       float(np.mean(np.abs(hyd_uar.values - hyd_uar.mean()))),
            'min_nonzero_uar':          float(uar[uar > 0].min()) if (uar > 0).any() else np.nan,
            'max_uar':                  float(uar.max()),
            'n_obs_hyd':                int(hyd_uar.size),
            'n_unique_uar_hyd':         int(hyd_uar.nunique()),
        })

        if (i + 1) % 100 == 0:
            logger.info(f"  {i + 1}/{len(all_csv)} stations processed")
    df = pd.DataFrame(rows).set_index('station_id')
    df.to_parquet(out_path)
    print(f"  {len(df)} stations ({n_skip} skipped) → {out_path.name}")
    return _load_year_stats()


# ---------------------------------------------------------------------------
# Step 1: build_station_meta
# ---------------------------------------------------------------------------
def build_station_meta(force: bool = False) -> pd.DataFrame:
    """
    Build station metadata table and write cache/station_meta.parquet.

    Columns: station_id, lat, lon, da_km2, record_years, log_drainage_area_km2,
             prcp, tmean, swe, high_prcp_freq, low_prcp_freq, high_prcp_duration,
             low_prcp_duration, slope_deg, elevation_m,
             land_use_forest_frac, land_use_snow_ice_frac,
             zero_flow_frac  (filled with NaN here, updated by Step 1b)
    """
    out = _cache("station_meta.parquet")
    if out.exists() and not force:
        return pd.read_parquet(out)

    print("Step 1: building station_meta...")

    # -----------------------------------------------------------------------
    # Caravan attributes path: load and merge the three per-region attribute
    # files (other, caravan, hydroatlas), then apply the canonical column map.
    # -----------------------------------------------------------------------
    attr_dir = ATTR_ROOT / REGION
    region_tag = REGION.split('_')[0]  # e.g. 'hysets' from 'hysets_bc'

    other_df  = pd.read_csv(attr_dir / f"attributes_other_{region_tag}.csv",   dtype={'gauge_id': str})
    caravan_df = pd.read_csv(attr_dir / f"attributes_caravan_{region_tag}.csv", dtype={'gauge_id': str})
    hydro_df  = pd.read_csv(attr_dir / f"attributes_hydroatlas_{region_tag}.csv", dtype={'gauge_id': str})

    # Keep only the columns we need from each file before merging
    caravan_keep = ['gauge_id', 'p_mean', 'high_prec_freq', 'low_prec_freq',
                    'high_prec_dur', 'low_prec_dur'] # caravan attributes
    hydro_keep   = ['gauge_id', 'tmp_dc_syr', 'snw_pc_syr', 'slp_dg_sav',
                    'ele_mt_sav', 'for_pc_sse', 'gla_pc_sse'] # hydroatlas attributes
    other_keep   = ['gauge_id', 'gauge_lat', 'gauge_lon', 'area'] # "other" attributes from Caravan

    other_keep   = [c for c in other_keep   if c in other_df.columns]
    caravan_keep = [c for c in caravan_keep if c in caravan_df.columns]
    hydro_keep   = [c for c in hydro_keep   if c in hydro_df.columns]

    raw = (other_df[other_keep]
           .merge(caravan_df[caravan_keep], on='gauge_id', how='left')
           .merge(hydro_df[hydro_keep],     on='gauge_id', how='left'))

    raw.rename(columns=_CARAVAN_COL_MAP, inplace=True)
    raw = raw.set_index('station_id')

    # --- exclude regulated / QA failed stations ---
    raw = raw[~raw.index.isin(Config.EXCLUDED_STATIONS)].copy()

    # --- record years from complete_year_stats parquet ---
    year_stats = _load_year_stats()
    raw['record_years'] = pd.Series(
        {s: len(year_stats[s]['hyd_years']) for s in year_stats if s in raw.index},
        dtype=float,
    )
    raw['record_years'] = raw['record_years'].fillna(0)

    # --- filter on minimum record length ---
    raw = raw[raw['record_years'] >= Config.MIN_YEARS_OF_RECORD].copy()

    # --- filter stations with no non-zero flow in any complete year ---
    # min_nonzero_uar is NaN when uar > 0 is never true; drop those stations.
    nonzero_min = pd.Series(
        {s: year_stats[s]['min_nonzero_uar'] for s in year_stats if s in raw.index},
        dtype=float,
    )
    raw = raw[nonzero_min.reindex(raw.index).notna()].copy()

    # --- filter stations whose max UAR exceeds the global support right edge ---
    # Any station with max_uar > GLOBAL_MAX_UAR has observations that fall outside
    # the common log-space grid, corrupting PMF binning, CRPS, and all derived metrics.
    max_uar_series = pd.Series(
        {s: year_stats[s]['max_uar'] for s in year_stats if s in raw.index},
        dtype=float,
    )
    n_before = len(raw)
    raw = raw[max_uar_series.reindex(raw.index) <= Config.GLOBAL_MAX_UAR].copy()
    n_dropped = n_before - len(raw)
    if n_dropped > 0:
        logger.warning(
            "  %d station(s) dropped: max_uar exceeds GLOBAL_MAX_UAR (%.4g L/s/km²)",
            n_dropped, Config.GLOBAL_MAX_UAR,
        )

    # --- filter stations with too few unique complete-year UAR values ---
    # KDE and log-space PMF construction require non-degenerate variability.
    unique_uar_series = pd.Series(
        {s: year_stats[s].get('n_unique_uar_hyd', np.nan) for s in year_stats if s in raw.index},
        dtype=float,
    )
    n_before = len(raw)
    raw = raw[unique_uar_series.reindex(raw.index) >= 2].copy()
    n_dropped = n_before - len(raw)
    if n_dropped > 0:
        logger.warning(
            "  %d station(s) dropped: fewer than 2 unique UAR values in complete years",
            n_dropped,
        )

    # --- derived columns ---
    raw['log_drainage_area_km2'] = np.log(raw['da_km2'])
    raw['zero_flow_frac']        = np.nan   # filled by compute_zero_flow_fracs()

    # --- CV of flows (sigma_uar / mu_uar) from complete_year_stats ---
    raw['cv_flows'] = pd.Series(
        {
            s: (year_stats[s]['stdev_annual_uar_hyd'] / year_stats[s]['mean_annual_uar_hyd'])
            for s in year_stats
            if s in raw.index and year_stats[s]['mean_annual_uar_hyd'] > 0
        },
        dtype=float,
    )

    meta = raw
    meta.to_parquet(out)
    print(f"  {len(meta)} stations --> {out.name}")
    return meta

def _make_pmf_df(arr: np.ndarray, cols: list[str], log_x: np.ndarray) -> pd.DataFrame:
    """Construct a (nbins × nstations) PMF DataFrame with log_x_uar as the first column."""
    df = pd.DataFrame(arr, columns=cols)
    df.insert(0, "log_x_uar", log_x.astype(np.float32))
    return df


# ---------------------------------------------------------------------------
# Step 1a: compute_baseline_pmfs
# ---------------------------------------------------------------------------

def compute_baseline_pmfs(
    force: bool = False,
    baseline_bitrates: list[int] | None = None,
) -> None:
    """Compute observed and KDE-smoothed PMFs for every station in the active
    region at every supported bitrate.

    This mirrors the notebook 01_data_preprocessing section 7.  The step must
    run after build_station_meta() / load_shared_data().  It is independent
    of compute_zero_flow_fracs(), which reads from the raw timeseries.

    Outputs (per bitrate b, written to _pmf_dir(b))
    ------------------------------------------------
    pmf_obs.csv
    pmf_obs_mixture.csv
    pmf_kde_adaptive.csv
    pmf_kde_adaptive_mixture.csv
    pmf_kde_silverman.csv
    pmf_kde_silverman_mixture.csv
    kld_obs_vs_kde_adaptive_{b}.csv
    kld_obs_vs_kde_silverman_{b}.csv
    """
    from kde_estimator import KDEEstimator, InsufficientDataError
    from reference_distribution import ReferenceDistribution

    meta           = META if META is not None else pd.read_parquet(_cache("station_meta.parquet"))
    year_stats     = YEAR_STATS if YEAR_STATS is not None else _load_year_stats()
    da_dict_       = DA_DICT if DA_DICT is not None else meta["da_km2"].to_dict()
    study_stations = [s for s in meta.index if s in year_stats]

    target_bitrates = sorted(set(baseline_bitrates or [REF_BITRATE]))
    pending = []
    for b in target_bitrates:
        bd       = _pmf_dir(b)
        outputs  = [
            bd / "pmf_obs.csv",                  bd / "pmf_obs_mixture.csv",
            bd / "pmf_kde_adaptive.csv",          bd / "pmf_kde_adaptive_mixture.csv",
            bd / "pmf_kde_silverman.csv",         bd / "pmf_kde_silverman_mixture.csv",
        ]
        if force or not all(p.exists() for p in outputs):
            pending.append(b)

    if not pending:
        print(f"Step 1a: baseline PMFs already exist for target bitrates {target_bitrates}, skipping.")
        return

    for bitrate in pending:
        print(f"Step 1a: computing baseline PMFs for bitrate={bitrate} ...")
        log_edges_uar      = np.linspace(np.log(Config.GLOBAL_MIN_UAR), np.log(Config.GLOBAL_MAX_UAR), 2**bitrate)
        log_w              = np.diff(0.5 * (log_edges_uar[1:] + log_edges_uar[:-1]))
        log_edges_extended = np.concatenate(([log_edges_uar[0] - log_w[0]], log_edges_uar))
        log_x_extended     = 0.5 * (log_edges_extended[1:] + log_edges_extended[:-1])
        nbins, nstations   = int(2**bitrate), len(study_stations)

        processed_stns:  list[str]        = []
        obs_pmf_cols:    list[np.ndarray] = []
        obs_mix_cols:    list[np.ndarray] = []
        adp_pmf_cols:    list[np.ndarray] = []
        adp_mix_cols:    list[np.ndarray] = []
        sil_pmf_cols:    list[np.ndarray] = []
        sil_mix_cols:    list[np.ndarray] = []
        kld_adp_list: list[dict] = []
        kld_sil_list: list[dict] = []

        kde_estimator = KDEEstimator(log_edges_extended)

        for j, stn in enumerate(study_stations):
            da_km2 = float(da_dict_[stn])
            df, zero_flow_flag = _retrieve_discharge(stn, da_km2)
            if df.empty:
                logger.warning("  Station %s: empty timeseries, skipping.", stn)
                continue

            try:
                baseline = ReferenceDistribution(
                    df=df,
                    zero_flow_flag=zero_flow_flag,
                    drainage_area_km2=da_km2,
                    log_edges_extended=log_edges_extended,
                    kde_estimator=kde_estimator,
                    delta=Config.UNIFORM_MIXTURE_DELTA,
                )
                obs_pmf_stn, adp_pmf_stn = baseline.build_station_pmf()
                sil_pmf_stn = baseline.build_station_pmf_silverman()
            except InsufficientDataError as e:
                logger.warning("  Station %s: insufficient unique flow values, skipping. (%s)", stn, e)
                continue

            obs_mix_stn = baseline._compute_adjusted_distribution_with_mixed_uniform(obs_pmf_stn)
            processed_stns.append(stn)
            obs_pmf_cols.append(obs_pmf_stn)
            obs_mix_cols.append(obs_mix_stn)

            adp_mix = baseline._compute_adjusted_distribution_with_mixed_uniform(adp_pmf_stn)
            sil_mix = baseline._compute_adjusted_distribution_with_mixed_uniform(sil_pmf_stn)
            adp_pmf_cols.append(adp_pmf_stn)
            adp_mix_cols.append(adp_mix)
            sil_pmf_cols.append(sil_pmf_stn)
            sil_mix_cols.append(sil_mix)
            kld_adp_list.append({"station": stn, "bitrate": bitrate,
                    "D_KL":         baseline._compute_kl_divergence(obs_pmf_stn, adp_pmf_stn),
                    "D_KL_mixture": baseline._compute_kl_divergence(obs_mix_stn, adp_mix)})
            kld_sil_list.append({"station": stn, "bitrate": bitrate,
                    "D_KL":         baseline._compute_kl_divergence(obs_pmf_stn, sil_pmf_stn),
                    "D_KL_mixture": baseline._compute_kl_divergence(obs_mix_stn, sil_mix)})

            if (j + 1) % 100 == 0:
                logger.info("  %d / %d stations processed (bitrate=%d)", j + 1, nstations, bitrate)

        obs_pmf_arr      = np.column_stack(obs_pmf_cols)   if obs_pmf_cols else np.zeros((nbins, 0))
        obs_pmf_mixtures = np.column_stack(obs_mix_cols)   if obs_mix_cols else np.zeros((nbins, 0))
        adp_pmf_arr = np.column_stack(adp_pmf_cols) if adp_pmf_cols else np.zeros((nbins, 0))
        adp_mix_arr = np.column_stack(adp_mix_cols) if adp_mix_cols else np.zeros((nbins, 0))
        sil_pmf_arr = np.column_stack(sil_pmf_cols) if sil_pmf_cols else np.zeros((nbins, 0))
        sil_mix_arr = np.column_stack(sil_mix_cols) if sil_mix_cols else np.zeros((nbins, 0))

        n_skipped = nstations - len(processed_stns)
        if n_skipped:
            logger.warning("  %d station(s) skipped (empty or insufficient data).", n_skipped)

        bits_dir = _pmf_dir(bitrate)
        bits_dir.mkdir(parents=True, exist_ok=True)
        _make_pmf_df(obs_pmf_arr,      processed_stns, log_x_extended).to_csv(bits_dir / "pmf_obs.csv",         index=False)
        _make_pmf_df(obs_pmf_mixtures, processed_stns, log_x_extended).to_csv(bits_dir / "pmf_obs_mixture.csv", index=False)
        for arr, fname in [
            (adp_pmf_arr, "pmf_kde_adaptive.csv"),
            (adp_mix_arr, "pmf_kde_adaptive_mixture.csv"),
            (sil_pmf_arr, "pmf_kde_silverman.csv"),
            (sil_mix_arr, "pmf_kde_silverman_mixture.csv"),
        ]:
            pmf_df   = _make_pmf_df(arr, processed_stns, log_x_extended)
            stn_cols = [c for c in pmf_df.columns if c != "log_x_uar"]
            sums     = pmf_df[stn_cols].sum(axis=0).values
            if not np.allclose(sums, 1.0, atol=1e-4):
                bad = stn_cols[np.where(~np.isclose(sums, 1.0, atol=1e-4))[0]]
                raise ValueError(
                    f"{fname}: {len(bad)} station column(s) not normalised "
                    f"(min={sums.min():.6f}, max={sums.max():.6f}): {list(bad)[:10]}"
                )
            pmf_df.to_csv(bits_dir / fname, index=False)
        pd.DataFrame(kld_adp_list).to_csv(bits_dir / f"kld_obs_vs_kde_adaptive_{bitrate}.csv",  index=False)
        pd.DataFrame(kld_sil_list).to_csv(bits_dir / f"kld_obs_vs_kde_silverman_{bitrate}.csv", index=False)
        print(f"  Wrote baseline PMFs for bitrate={bitrate}: {len(processed_stns)} stations → {bits_dir}")

# ---------------------------------------------------------------------------
# Helper: complete-year UAR extraction
# ---------------------------------------------------------------------------

def _get_complete_year_uar(
    stn: str,
    da_km2: float,
    year_stats: dict,
) -> "tuple[np.ndarray, pd.DatetimeIndex, float] | None":
    """Return (uar_raw, dates, min_uar) for complete hydrological years, or None to skip.

    Applies the complete-year mask (Oct-Sep: month >= 10 maps to year+1) and
    a finiteness filter.  Negative UAR values raise ValueError.  All values at or below
    min_uar are sub-threshold (zero-flow) and must be handled consistently by the
    caller, matching how ReferenceDistribution routes sub-threshold observations
    to bin 0 via np.digitize.
    """
    flow_df, _ = retrieve_and_preprocess_timeseries_discharge(stn, da=da_km2)
    if flow_df.empty:
        return None
    x_raw      = flow_df[f"{stn}_uar"].values.astype("float64")
    time_index = pd.DatetimeIndex(flow_df.index)
    hyd_yr     = np.where(time_index.month >= 10, time_index.year + 1, time_index.year)
    mask       = np.isfinite(x_raw) & np.isin(hyd_yr, list(year_stats[stn]["hyd_years"]))
    uar_raw    = x_raw[mask]
    n_neg      = int((uar_raw < 0).sum())
    if n_neg > 0:
        raise ValueError(
            f"Station {stn!r}: {n_neg} negative UAR value(s) in complete hydrological "
            "years. Check the source timeseries for data quality issues."
        )
    n_exceed = int((uar_raw > Config.GLOBAL_MAX_UAR).sum())
    if n_exceed > 0:
        raise ValueError(
            f"Station {stn!r}: {n_exceed} UAR value(s) exceed GLOBAL_MAX_UAR "
            f"({Config.GLOBAL_MAX_UAR:.4g} L/s/km²). This station should have been "
            "excluded in build_station_meta. Re-run that step with force=True."
        )
    return uar_raw, time_index[mask], 1000.0 * Config.ZERO_FLOW_THRESHOLD / da_km2


# ---------------------------------------------------------------------------
# Step 1b (optional): compute_zero_flow_fracs
# ---------------------------------------------------------------------------

def compute_zero_flow_fracs(force: bool = False) -> pd.DataFrame:
    """
    Compute the empirical zero-flow fraction for each station from the raw
    daily timeseries, restricted to complete hydrological years.

    The zero-flow fraction is the proportion of complete-year daily observations
    whose UAR is at or below the station-specific threshold:

        min_uar = 1000 * Config.ZERO_FLOW_THRESHOLD / da_km2  [L/s/km²]

    This is a property of the observed streamflow record independent of PMF
    bin resolution.  The value should be numerically close to the integrated
    probability mass at or below the same threshold in the bch empirical PMF.
    Large discrepancies between them are a useful diagnostic.

    Output
    ------
    cache/zero_flow_fracs.parquet
        Columns: station_id, zero_flow_frac (float32).  One row per station.
    """
    out = _cache("zero_flow_fracs.parquet")
    if out.exists() and not force:
        return pd.read_parquet(out)

    print("Step 1b: computing empirical zero-flow fractions...")

    meta       = META if META is not None else pd.read_parquet(_cache("station_meta.parquet"))
    year_stats = YEAR_STATS if YEAR_STATS is not None else _load_year_stats()
    da_dict    = DA_DICT if DA_DICT is not None else meta["da_km2"].to_dict()

    study_stations = [s for s in meta.index if s in year_stats]

    records: list[dict] = []
    n_skip = 0
    for stn in study_stations:
        da_km2 = float(da_dict[stn])
        result = _get_complete_year_uar(stn, da_km2, year_stats)
        if result is None or len(result[0]) == 0:
            n_skip += 1
            continue
        uar_raw, _, min_uar = result
        records.append({
            "station_id":     stn,
            "zero_flow_frac": float(np.mean(uar_raw < min_uar)),  # strict <: consistent with bch/ab_kde/sil_kde
        })

    if not records:
        raise RuntimeError(
            "compute_zero_flow_fracs: no valid stations. "
            "Check that timeseries and complete_year_stats exist for the active region."
        )

    df_out = pd.DataFrame(records).astype({"zero_flow_frac": "float32"})
    df_out.to_parquet(out, index=False)
    print(f"  {len(df_out)} stations ({n_skip} skipped) → {out.name}")

    # Back-fill zero_flow_frac into station_meta.parquet.
    meta_path = _cache("station_meta.parquet")
    meta_df   = pd.read_parquet(meta_path)
    zff_map   = df_out.set_index("station_id")["zero_flow_frac"]
    meta_df["zero_flow_frac"] = zff_map.reindex(meta_df.index).values
    meta_df.to_parquet(meta_path)
    print(f"  updated zero_flow_frac in {meta_path.name} for {len(zff_map)} stations")

    return df_out


# ---------------------------------------------------------------------------
# Step 1c: compute_lnmle_pmfs
# ---------------------------------------------------------------------------
def compute_lnmle_pmfs(bitrates: list[int], force: bool = False) -> None:
    """
    Fit a two-parameter log-normal by MLE for each station and save the
    resulting PMF (and KLD-limited uniform mixture) at every supported bitrate.

    MLE parameters (mu, sigma) are estimated once from all daily UAR
    observations across complete hydrological years (Oct-Sep).  Zero-flow
    probability is handled identically to fdc_data._set_zero_flow_edges and
    parametric_estimator.compute_lognorm_pmf().  The PMF is then projected onto
    the log-UAR bin grid for each bitrate independently.

    Outputs (one pair per supported bitrate b)
    ------------------------------------------
    baseline_distributions/{b:02d}_bits/pmf_lnmle.csv
    baseline_distributions/{b:02d}_bits/pmf_lnmle_mixture.csv
    """

    bitrates = sorted(set(bitrates))
    if not bitrates:
        print("Step 1c: no target bitrates requested for LN-MLE, skipping.")
        return

    # Determine which bitrates still need generating
    pending = []
    for b in bitrates:
        bd = _pmf_dir(b)
        if force or not (bd / "pmf_lnmle.csv").exists() or not (bd / "pmf_lnmle_mixture.csv").exists():
            pending.append(b)
    if not pending:
        print("Step 1c: pmf_lnmle already exists for all bitrates, skipping.")
        return

    missing = [b for b in pending if not (_pmf_dir(b) / "pmf_obs.csv").is_file()]
    if missing:
        raise FileNotFoundError(
            "compute_lnmle_pmfs: missing prerequisite pmf_obs.csv for bitrate(s) "
            f"{missing}. Run compute_baseline_pmfs with run_basic=True and "
            f"baseline_bitrates including {missing}."
        )

    station_sets: dict[int, set[str]] = {}
    for b in pending:
        pmf_obs_df = pd.read_csv(_pmf_dir(b) / "pmf_obs.csv", index_col=0)
        station_sets[b] = set(pmf_obs_df.columns)

    ref_b = pending[0]
    ref_set = station_sets[ref_b]
    for b in pending[1:]:
        this_set = station_sets[b]
        if this_set != ref_set:
            missing_vs_ref = sorted(ref_set - this_set)
            extra_vs_ref = sorted(this_set - ref_set)
            raise RuntimeError(
                "compute_lnmle_pmfs: station set mismatch across bitrate caches. "
                f"reference bitrate={ref_b} has {len(ref_set)} stations, bitrate={b} "
                f"has {len(this_set)} stations. Missing({len(missing_vs_ref)}): "
                f"{missing_vs_ref[:5]}; Extra({len(extra_vs_ref)}): {extra_vs_ref[:5]}."
            )

    print(f"Step 1c: computing LN-MLE PMFs for bitrates {pending}...")

    meta       = META if META is not None else pd.read_parquet(_cache("station_meta.parquet"))
    year_stats = YEAR_STATS if YEAR_STATS is not None else _load_year_stats()
    da_dict    = DA_DICT if DA_DICT is not None else meta["da_km2"].to_dict()

    # Station list from the validated reference bitrate cache.
    ref_pmf = pd.read_csv(_pmf_dir(ref_b) / "pmf_obs.csv", index_col=0)
    study_stations = [s for s in ref_pmf.columns if s in meta.index]

    # Fit MLE params once per station (bitrate-independent).
    # Zero-flow probability is the empirical fraction at or below min_uar (hurdle model).
    # Tuple: (mu, sigma, log_mu, log_sigma, log_min_uar, p_zero)
    # mu/sigma are the linear-space mean and std (stored for diagnostics).
    # log_mu/log_sigma are the log-normal MLE parameters used for PMF construction.
    mle_params: dict[str, tuple[float, float, float, float, float, float]] = {}
    n_skip = 0
    for stn in study_stations:
        if stn not in year_stats:
            n_skip += 1
            continue
        da_km2 = float(da_dict[stn])
        result = _get_complete_year_uar(stn, da_km2, year_stats)
        if result is None:
            n_skip += 1
            continue
        uar_raw, _, min_uar = result
        pos    = uar_raw[uar_raw > min_uar]
        p_zero = float(np.mean(uar_raw < min_uar))  # strict <: consistent with bch/ab_kde/sil_kde

        if len(pos) < 10:
            n_skip += 1
            continue

        log_pos   = np.log(pos)
        mu        = float(np.mean(pos))
        sigma     = float(np.std(pos, ddof=1))
        log_mu    = float(np.mean(log_pos))
        log_sigma = float(np.std(log_pos, ddof=1))
        if log_sigma <= 0:
            n_skip += 1
            continue

        mle_params[stn] = (mu, sigma, log_mu, log_sigma, np.log(min_uar), p_zero)

    if not mle_params:
        raise RuntimeError("compute_lnmle_pmfs: no valid MLE fits. Check timeseries path.")

    print(f"  MLE params fitted: {len(mle_params)} stations ({n_skip} skipped)")
    stations = sorted(mle_params)

    # Rebin onto each bitrate's grid and save
    for b in pending:
        ref_path  = _pmf_dir(b) / "pmf_obs.csv"
        log_mids  = pd.read_csv(ref_path, usecols=[0]).iloc[:, 0].to_numpy(dtype=float)
        half_dlog = 0.5 * (log_mids[1] - log_mids[0])
        log_edges = np.concatenate([[log_mids[0] - half_dlog], log_mids + half_dlog])
        B         = len(log_mids)

        pmf_arr = np.zeros((B, len(stations)), dtype=float)
        for si, stn in enumerate(stations):
            mu, sigma, log_mu, log_sigma, log_min_uar, p_zero = mle_params[stn]

            # Positive-flow bins are 1..B-1.  For each bin, the effective lower
            # edge is max(log_min_uar, log_edges[k]) so that the portion of any
            # bin below the zero-flow threshold contributes nothing to positive
            # mass.  This handles the threshold-falls-inside-a-bin case (typical)
            # and the threshold-below-grid-floor case (large-DA stations) without
            # needing to locate which bin the threshold falls in.
            lowers    = np.maximum(log_min_uar, log_edges[1:B])   # (B-1,)
            uppers    = log_edges[2:B + 1]                         # (B-1,)
            pos_total = norm.cdf(log_edges[-1], log_mu, log_sigma) - norm.cdf(log_min_uar, log_mu, log_sigma)
            pmf       = np.zeros(B, dtype=float)
            pmf[0]    = p_zero
            if pos_total > 0:
                raw = norm.cdf(uppers, log_mu, log_sigma) - norm.cdf(lowers, log_mu, log_sigma)
                pmf[1:] = (1.0 - p_zero) * np.maximum(0.0, raw) / pos_total

            pmf = np.clip(pmf, 0.0, None)
            pmf /= pmf.sum()
            pmf_arr[:, si] = pmf

        pmf_df = pd.DataFrame(pmf_arr, index=log_mids, columns=stations)
        pmf_df.index.name = "log_x_uar"

        mix_arr = np.column_stack([
            apply_kld_limited_uniform_mixture(pmf_arr[:, i], Config.Metrics.KLD_DELTA_MAX)
            for i in range(len(stations))
        ])
        mix_df = pd.DataFrame(mix_arr, index=log_mids, columns=stations)
        mix_df.index.name = "log_x_uar"

        bits_dir = _pmf_dir(b)
        bits_dir.mkdir(parents=True, exist_ok=True)
        pmf_df.to_csv(bits_dir / "pmf_lnmle.csv")
        mix_df.to_csv(bits_dir / "pmf_lnmle_mixture.csv")
        print(f"  pmf_lnmle (b={b}): {pmf_df.shape} → pmf_lnmle.csv")


# ---------------------------------------------------------------------------
# Step 1d: compute_weibull_quantiles
# ---------------------------------------------------------------------------

def compute_weibull_quantiles(force: bool = False) -> pd.DataFrame:
    """
    Compute complete-year Weibull ECDF quantiles for every station and cache
    the result.

    For each station all valid daily UAR observations across complete
    hydrological years are clipped to Config.GLOBAL_MIN_UAR before ranking.
    All N observations are stored with Weibull plotting positions
    F_k = k / (N+1).  Sub-threshold observations collapse to exactly
    GLOBAL_MIN_UAR and form a flat plateau at the high-exceedance tail of the
    FDC.

    GLOBAL_MIN_UAR (rather than the station-specific min_uar) is used as the
    sentinel so that all zero-flow observations land on the same log_x grid
    position (bin 1) in _crps_one_vs_many_batch regardless of drainage area.

    This matches how PMF estimators represent "zero flows", i.e. as probability
    mass assigned to a designated bin to represent these flows.  Comparisons
    between estimators and this reference are therefore zero in the plateau
    region by construction, and residual errors only arise from differences in
    how estimators model positive flows.

    This is the purely empirical reference, it makes no distributional
    assumption and is not derived from any PMF estimator.

    Output
    ------
    cache/weibull_quantiles.parquet
        Long-format DataFrame sorted by station_id then F_k ascending.
        Columns:
          station_id  str           Station identifier.
          log_uar     float64       log(UAR) after clipping to GLOBAL_MIN_UAR.
          F_k         float64       Weibull plotting position k/(N+1).
          date        datetime64    Calendar date of the observation (retained
                                    for FDC seasonal diagnostics; not used by
                                    scorer functions).
        One row per valid complete-year observation (N total, after clipping).
        F_k spans 1/(N+1) ... N/(N+1).
    """
    out = _cache("weibull_quantiles.parquet")
    if out.exists() and not force:
        return pd.read_parquet(out)

    print("Step 1d: computing Weibull quantiles...")

    meta       = META if META is not None else pd.read_parquet(_cache("station_meta.parquet"))
    year_stats = YEAR_STATS if YEAR_STATS is not None else _load_year_stats()
    da_dict    = DA_DICT if DA_DICT is not None else meta["da_km2"].to_dict()

    # Station list: intersection of meta and year_stats
    study_stations = [s for s in meta.index if s in year_stats]

    station_frames: list[pd.DataFrame] = []
    n_skip = 0
    for stn in study_stations:
        da_km2 = float(da_dict[stn])
        result = _get_complete_year_uar(stn, da_km2, year_stats)
        if result is None:
            n_skip += 1
            continue
        uar_raw, dates, _ = result
        # Clip to GLOBAL_MIN_UAR so all zero-flow observations share the same
        # sentinel value across all drainage areas (see docstring).
        uar      = np.maximum(uar_raw, Config.GLOBAL_MIN_UAR)
        N        = len(uar)
        sort_idx = np.argsort(uar)
        uar_sorted  = uar[sort_idx]
        date_sorted = dates[sort_idx]
        station_frames.append(pd.DataFrame({
            "station_id": stn,
            "log_uar":    np.log(uar_sorted),
            "F_k":        np.arange(1, N + 1) / (N + 1.0),
            "date":       date_sorted,
        }))

    if not station_frames:
        raise RuntimeError("compute_weibull_quantiles: no valid stations. Check timeseries path.")

    df_out = pd.concat(station_frames, ignore_index=True)
    df_out = df_out.sort_values(["station_id", "F_k"]).reset_index(drop=True)
    df_out.to_parquet(out, index=False)
    n_stns = df_out["station_id"].nunique()
    print(f"  {n_stns} stations ({n_skip} skipped), {len(df_out):,} rows --> {out.name}")
    return df_out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_pipeline_for_region() -> None:
    """Execute PMF-only pipeline steps for the currently active region."""
    compute_complete_year_stats()
    build_station_meta()
    load_shared_data()
    compute_baseline_pmfs(baseline_bitrates=Config.SUPPORTED_BITRATES)
    compute_lnmle_pmfs(bitrates=[REF_BITRATE])
    compute_weibull_quantiles()
    print(
        "PMF-only pipeline complete for region",
        REGION,
        "with bitrates",
        Config.SUPPORTED_BITRATES,
    )



def main():
    CACHE_DIR.mkdir(exist_ok=True)
    ensure_parquet_engine()
    for region in get_processed_regions():
        print(f"\n{'='*60}\nProcessing region: {region}\n{'='*60}")
        init_region(region)
        _run_pipeline_for_region()


if __name__ == "__main__":
    main()
