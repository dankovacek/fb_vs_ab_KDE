"""Preprocess for 09_max_kde_diffs.

Loads pre-computed KDE PMFs from the main sensitivity_report pipeline cache
and computes per-station divergence scores between the adaptive bandwidth KDE
and the Silverman rule-of-thumb KDE at bitrates 6, 8, and 10.

Five divergence measures are computed for each station and bitrate:
  - KS statistic    : max_j |CDF_ab(x_j) - CDF_fb(x_j)|
  - Wasserstein     : sum_j |CDF_ab(x_j) - CDF_fb(x_j)| * delta_j  (L1 of CDFs, nats)
  - Energy distance : sqrt( 2 * sum_j (CDF_ab - CDF_fb)^2 * delta_j )  (L2 of CDFs)
  - ISD             : sum_j (pmf_ab - pmf_fb)^2 / delta_j  (L2 density distance)
  - KL divergence   : sum_j p_ab(x_j) * log2(p_ab(x_j) / p_fb(x_j))  (bits)

Note on spread dependence: Wasserstein scales as sigma_log and energy distance
as sqrt(sigma_log) for a fixed relative bandwidth discrepancy.  Normalised
variants (w1_norm = W1/sigma_log, ed_norm = ED/sqrt(sigma_log)) that are
spread-invariant are also written to the cache.  ISD scales inversely as
1/sigma_log, which is theoretically motivated: narrow distributions have higher
density values and errors in density estimation are correspondingly more
consequential.  The name 'ISD' (Integrated Squared Difference) is used rather
than 'MISE' because no reference truth distribution is involved; MISE is
reserved for comparisons against a known true density (see synthetic_test.py).

The global UAR range [Config.GLOBAL_MIN_UAR, Config.GLOBAL_MAX_UAR] is fixed
by the main pipeline and shared across all regions and bitrates.

Usage
-----
    python run_analysis.py [region|index|all]

Prerequisites
-------------
    preprocess.py (step 1a: compute_baseline_pmfs) must be complete so that
    pmf_kde_adaptive.csv and pmf_kde_silverman.csv exist.

Outputs (per region, written to cache/)
-----------
    {region}_kde_comparison.parquet
        One row per (station_id, bitrate).  Columns:
          station_id, bitrate, region, ks_stat, wasserstein, w1_norm,
          energy_distance, ed_norm, isd, kl_divergence, sigma_log, ...
    {region}_worst10_pmfs.parquet
        Long-format PMF data for the top-N stations per divergence metric at 8 bits.
        Columns: station_id, metric, rank, log_x, pmf_obs, pmf_ab, pmf_fb.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import diptest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR        = Path(__file__).resolve().parent
REPO_ROOT         = SCRIPT_DIR.parent
CACHE_DIR         = REPO_ROOT / "cache"
BASELINE_DIR      = REPO_ROOT / "data" / "baseline_distributions"
PARENT_CACHE_ROOT = REPO_ROOT / "cache"

sys.path.insert(0, str(SCRIPT_DIR))

from config import Config
# from utils import apply_kld_limited_uniform_mixture

BITRATES    = Config.SUPPORTED_BITRATES
REF_BITRATE = Config.DEFAULT_BITRATE
N_WORST     = 10
N_MEDIAN    = 8


# ---------------------------------------------------------------------------
# Region selection (mirrors energy_distance.py / argument_ideas/05 pattern)
# ---------------------------------------------------------------------------

def _resolve_regions() -> list[str]:
    if not PARENT_CACHE_ROOT.exists():
        raise SystemExit(
            f"Cache directory not found:\n  {PARENT_CACHE_ROOT}\n"
            "Run preprocess.py first."
        )
    regions = sorted(d.name for d in PARENT_CACHE_ROOT.iterdir() if d.is_dir())
    if not regions:
        raise SystemExit(
            f"No processed regions found in:\n  {PARENT_CACHE_ROOT}\n"
            "Run preprocess.py first."
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

def _pmf_dir(region: str, bitrate: int) -> Path:
    return BASELINE_DIR / region / f"{bitrate:02d}_bits"


def _is_stale(out: Path, *sources: Path) -> bool:
    if not out.exists():
        return True
    out_mtime = out.stat().st_mtime
    return any(s.exists() and s.stat().st_mtime > out_mtime for s in sources)


def _find_median_stations(scores8: pd.DataFrame) -> list[tuple[str, str]]:
    """Return 6 (station_id, metric_label) pairs: 2 closest to the median
    for each of KS, Wasserstein, and energy distance.  Stations that would
    appear more than once (i.e. near-median on multiple metrics) keep both
    entries so the caller can see which metric drove the selection.
    """
    pairs: list[tuple[str, str]] = []
    for col, label in [
        ("ks_stat",         "median KS"),
        ("wasserstein",     "median W\u2081"),
        ("energy_distance", "median ED"),
        ("kl_divergence",   "median KL"),
        ("isd",             "median ISD"),
    ]:
        med = scores8[col].median()
        closest = (
            scores8.assign(_dist=(scores8[col] - med).abs())
            .nsmallest(2, "_dist")["station_id"]
            .tolist()
        )
        for stn in closest:
            pairs.append((stn, label))
    return pairs


def _load_pmf(region: str, bitrate: int, name: str) -> pd.DataFrame:
    """Load a PMF CSV (index = log_x_uar, columns = station_id)."""
    path = _pmf_dir(region, bitrate) / name
    if not path.exists():
        raise FileNotFoundError(
            f"PMF file not found: {path}\n"
            "Run preprocess.py (compute_baseline_pmfs) first."
        )
    return pd.read_csv(path, index_col=0)


def _apply_mixture_cols(mat: np.ndarray, delta: float) -> np.ndarray:
    """Apply the KLD-limited uniform mixture to every column of mat (shape B×N).

    Finds, per column, the largest mixing weight λ such that
    KL(pmf_j ∥ (1-λ)·pmf_j + λ·U) ≤ delta (bits), where U = 1/B uniform.
    Uses a vectorised 60-iteration binary search so all N stations are
    processed in one NumPy call.
    """
    B, N = mat.shape
    col_sums = mat.sum(axis=0, keepdims=True)
    pmf = mat / np.where(col_sums > 0, col_sums, 1.0)
    pmf = np.clip(pmf, 1e-300, 1.0)
    pmf /= pmf.sum(axis=0, keepdims=True)
    U = 1.0 / B

    def _kl_cols(lam: np.ndarray) -> np.ndarray:
        q = (1.0 - lam[None, :]) * pmf + lam[None, :] * U   # (B, N)
        return np.sum(pmf * np.log2(pmf / q), axis=0)         # (N,)

    lo           = np.zeros(N)
    hi           = np.ones(N)
    needs_search = _kl_cols(hi) > delta
    if needs_search.any():
        for _ in range(60):
            mid     = 0.5 * (lo + hi)
            kl_mid  = _kl_cols(mid)
            lo      = np.where(needs_search & (kl_mid <= delta), mid, lo)
            hi      = np.where(needs_search & (kl_mid >  delta), mid, hi)
            if (hi - lo).max() < 1e-9:
                break
    lam_opt = np.where(needs_search, lo, 1.0)
    mixed   = (1.0 - lam_opt[None, :]) * pmf + lam_opt[None, :] * U
    mixed  /= mixed.sum(axis=0, keepdims=True)
    return mixed


def _compute_scores(
    ab_df: pd.DataFrame,
    fb_df: pd.DataFrame,
    obs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Vectorised divergence scores between AB-KDE and Silverman-KDE for all stations.

    Scores computed on the shared log-UAR grid:
      KS       : max_j |ΔCDF(j)|  -- spread-invariant worst-case bound
      W1       : sum_j |ΔCDF(j)| * dlog_j  -- L1 CDF distance (units: nats)
      ED       : sqrt(2 * sum_j ΔCDF(j)² * dlog_j)  -- L2 CDF distance
      ISD      : sum_j (pmf_ab_j - pmf_fb_j)² / dlog_j  -- L2 density distance
      KLD      : sum_j pmf_ab_j * log2(pmf_ab_j / pmf_fb_j)  (bits)

    NOTE on spread dependence:
      Silverman's bandwidth h ∝ σ_log, so the region of CDF disagreement also
      scales with σ_log.  Consequently W1 ∝ σ_log and ED ∝ sqrt(σ_log) for a
      fixed relative bandwidth discrepancy.  ISD scales as 1/σ_log (inversely).
      Cross-station ranking with these raw values confounds bandwidth accuracy
      with distributional spread.  Normalised variants are therefore provided:
        W1_norm = W1 / sigma_log
        ED_norm = ED / sqrt(sigma_log)
      These remove the first-order spread dependence, making cross-station
      comparison of bandwidth-choice impact meaningful.

    Quantile shift formula (log-space):
        delta_log_x = KS * dlog[j*] / pmf[j*]
    where j* is the bin of maximum |delta CDF| and pmf is evaluated in the given
    density basis.  The percentage shift is (exp(delta_log_x) - 1) * 100.
    When the density is zero the shift is left as NaN.
    """
    common = ab_df.columns.intersection(fb_df.columns)
    log_x  = ab_df.index.to_numpy(dtype=float)
    ab    = ab_df[common].values.astype(float)                    # (B, N)
    fb    = fb_df[common].values.astype(float)                    # (B, N)
    obs    = obs_df.reindex(columns=common).values.astype(float)    # (B, N)

    # Integration weights: gradient gives exact half-spacing at boundaries
    dlog = np.gradient(log_x)                      # (B,)

    # Apply the KLD-limited uniform mixture to the two KDE PMFs upfront.
    # This is the same treatment as the main pipeline and ensures every bin
    # carries a non-negligible probability, eliminating the need for any
    # per-metric ad-hoc guards.  The information-loss budget is
    # Config.Metrics.KLD_DELTA_MAX bits, matching the global constant.
    # The observed PMF (obs) is not mixed: its zero bins are genuine absences
    # in the training data and should produce NaN rather than a floor value.
    _kld_delta = Config.Metrics.KLD_DELTA_MAX
    ab_m = _apply_mixture_cols(ab, _kld_delta)   # (B, N)
    fb_m = _apply_mixture_cols(fb, _kld_delta)   # (B, N)

    cdf_ab = np.cumsum(ab_m, axis=0)               # (B, N)
    cdf_fb = np.cumsum(fb_m, axis=0)               # (B, N)
    delta   = cdf_ab - cdf_fb                    # (B, N)

    ks   = np.max(np.abs(delta), axis=0)
    wass = np.sum(np.abs(delta) * dlog[:, None], axis=0)
    enrg = np.sqrt(np.maximum(2.0 * np.sum(delta**2 * dlog[:, None], axis=0), 0.0))

    # ISD: integrated squared difference between the two density estimates in log-UAR space.
    # sum_j (pmf_AB_j - pmf_FB_j)^2 / dlog_j = sum_j (density_AB_j - density_FB_j)^2 * dlog_j
    # Dividing by dlog converts PMF probability mass to density units (mass/bin-width).
    # NOTE: named 'isd' (Integrated Squared Difference) because no reference truth is involved;
    # the classical MISE (Mean Integrated Squared Error) compares an estimator to the true density.
    # In synthetic_test.py where a true distribution IS available the name 'mise' is appropriate.
    pmf_diff = ab_m - fb_m                                            # (B, N)
    isd = np.sum(pmf_diff ** 2 / dlog[:, None], axis=0)              # (N,)

    # KL(ab_m || fb_m): both PMFs have been mixed toward uniform so all bins
    # are strictly positive; the log ratio is finite everywhere.
    kl = np.sum(ab_m * np.log2(ab_m / fb_m), axis=0)

    # --- KS-maximum bin information ---
    N           = ab_m.shape[1]
    ks_indices  = np.argmax(np.abs(delta), axis=0)   # (N,) per-station bin of max CDF gap
    stn_idx     = np.arange(N)
    dlog_at_max = dlog[ks_indices]                   # (N,) bin width at that bin
    ab_at_max  = ab_m[ks_indices, stn_idx]           # (N,) adaptive PMF at max-KS bin
    fb_at_max  = fb_m[ks_indices, stn_idx]           # (N,) Silverman PMF at max-KS bin
    obs_at_max  = obs[ks_indices, stn_idx]            # (N,) observed PMF at max-KS bin
    log_x_at_max = log_x[ks_indices]                 # (N,) log-UAR position of max-KS bin

    # Relative quantile shift (%) in each density basis:
    #   delta_log_x = KS * dlog / pmf  =>  shift_pct = (exp(delta_log_x) - 1) * 100
    # Mixed KDE PMFs (ab_m, fb_m) are always strictly positive so no guard is
    # needed.  The observed PMF may contain genuine zero bins (no data at that
    # bin); those are undefined and are returned as NaN.
    def _qshift_pct(pmf: np.ndarray, require_positive: bool = False) -> np.ndarray:
        if require_positive:
            valid     = pmf > 0.0
            safe_pmf  = np.where(valid, pmf, 1.0)
            delta_log = np.where(valid, ks * dlog_at_max / safe_pmf, np.nan)
        else:
            delta_log = ks * dlog_at_max / pmf
        # Clamp to float64 range before exponentiation; shifts above ~700 nats
        # exceed any physically meaningful flow ratio and are returned as NaN.
        delta_log = np.where(delta_log < 700.0, delta_log, np.nan)
        return np.expm1(delta_log) * 100.0

    # --- Full-support per-bin shift arrays (B, N) ---
    # shift_adp_full[j, n] = (exp(|ΔCDF[j,n]| * dlog[j] / ab_m[j,n]) - 1) * 100
    # shift_sil_full[j, n] analogously for fb_m.
    # Both KDE PMFs are mixture-floored so denominators are strictly positive.
    # The KS-bin scalar (ks_qshift_adp_pct) is a lower bound on these arrays:
    # a bin adjacent to j* with low density and a marginally smaller CDF gap
    # can produce a substantially larger implied shift.
    abs_delta      = np.abs(delta)                           # (B, N)
    _dl_adp        = abs_delta * dlog[:, None] / ab_m        # (B, N)
    _dl_sil        = abs_delta * dlog[:, None] / fb_m        # (B, N)
    _dl_adp        = np.where(_dl_adp < 700.0, _dl_adp, np.nan)
    _dl_sil        = np.where(_dl_sil < 700.0, _dl_sil, np.nan)
    shift_adp_full = np.expm1(_dl_adp) * 100.0              # (B, N)
    shift_sil_full = np.expm1(_dl_sil) * 100.0              # (B, N)

    # Per-station maximum and its bin index.
    # nanargmax on all-NaN columns would raise; guard with a finite-value check.
    _adp_finite    = np.isfinite(shift_adp_full)
    _sil_finite    = np.isfinite(shift_sil_full)
    max_qshift_adp_pct = np.where(_adp_finite.any(axis=0), np.nanmax(shift_adp_full, axis=0), np.nan)
    max_qshift_sil_pct = np.where(_sil_finite.any(axis=0), np.nanmax(shift_sil_full, axis=0), np.nan)
    max_qshift_adp_bin = np.where(
        _adp_finite.any(axis=0),
        np.argmax(np.where(_adp_finite, shift_adp_full, -np.inf), axis=0),
        -1,
    ).astype(int)
    max_qshift_sil_bin = np.where(
        _sil_finite.any(axis=0),
        np.argmax(np.where(_sil_finite, shift_sil_full, -np.inf), axis=0),
        -1,
    ).astype(int)

    # sigma from observed PMF moments in log-UAR space (not from the KDE,
    # so scale is independent of bandwidth choice)
    mu_obs   = np.sum(log_x[:, None] * obs, axis=0)                       # (N,)
    var_obs  = np.sum((log_x[:, None] - mu_obs[None, :])**2 * obs, axis=0)  # (N,)
    sigma    = np.sqrt(np.maximum(var_obs, 1e-12))                        # (N,)

    # Spread-normalised W1 and ED: remove the first-order σ_log dependence so
    # that cross-station comparisons reflect bandwidth-choice impact rather than
    # the spread of each station's flow distribution.
    #   W1 ∝ σ_log   → W1_norm = W1 / σ_log  (dimensionless)
    #   ED ∝ √σ_log  → ED_norm = ED / √σ_log  (units: √nats)
    w1_norm = wass / sigma                                              # (N,)
    ed_norm = enrg / np.sqrt(sigma)                                     # (N,)

    # --- Distributional shape descriptors from the observed PMF ---
    # Computed in log-UAR space.  Silverman (1986) states his rule of thumb
    # gives MISE within 10% of optimum for log-normals with skewness up to
    # ~1.8 and normal mixtures with component separation up to 3 sigma.
    # We record the sample moments and peak structure so those bounds can be
    # evaluated against the actual station population.
    #
    # Note: skewness computed here is in log-UAR space.  Log-transformation
    # compresses right-skew, so most stations appear more symmetric here
    # than their raw-discharge skewness would suggest.  A station exceeding
    # 1.8 in log-UAR space is an especially extreme case.
    m3_obs = np.sum(
        (log_x[:, None] - mu_obs[None, :]) ** 3 * obs, axis=0
    )                                                                    # (N,) 3rd central moment (weighted)
    m4_obs = np.sum(
        (log_x[:, None] - mu_obs[None, :]) ** 4 * obs, axis=0
    )                                                                    # (N,) 4th central moment
    obs_skewness = np.where(sigma > 0, m3_obs / (sigma ** 3), np.nan)  # standardised skewness

    # --- Peak structure from the AB KDE PMF (8-bit grid) ---
    # Peaks detected by prominence rather than absolute height.  A 5% absolute
    # height floor was previously used but suppressed most real bimodal
    # distributions: on a 256-bin grid a mode with 30% of total mass spread
    # over 20 bins reaches a peak height of only ~0.015, well below 0.05.
    # Prominence (height above the highest intervening saddle) is the correct
    # criterion for separating genuine modes from noise.
    #
    # Separation is normalised by the MEAN of the two component standard
    # deviations, not by the global sigma.  Silverman's (1986) 3-sigma bound
    # uses within-component sigma.  Global sigma is always larger than component
    # sigma for separated modes (mixture-variance decomposition:
    # sigma_global^2 = sigma_comp^2 + 0.25*delta^2), so dividing by global sigma
    # underestimates the true separation by ~44% at the 3-sigma boundary.
    # Component sigma is estimated by a valley-split: the inter-modal saddle bin
    # divides the PMF into two regions; the PMF-weighted std dev of each region
    # gives the two component sigmas.
    _PEAK_PROMINENCE = 0.005   # catches modes down to ~2% of total PMF mass
    n_peaks_arr          = np.zeros(N, dtype=int)
    peak_sep_sigma_arr   = np.full(N, np.nan)
    peak_sep_abs_arr     = np.full(N, np.nan)
    mode_sigma_left_arr  = np.full(N, np.nan)
    mode_sigma_right_arr = np.full(N, np.nan)
    spread_asymmetry_arr = np.full(N, np.nan)
    _bin_idx = np.arange(len(log_x))
    # ab columns sum to 1 (PMF); each column is one station
    for j in range(N):
        pmf_j    = ab[:, j]
        peaks_j, props_j = find_peaks(pmf_j, prominence=_PEAK_PROMINENCE)
        n_peaks_arr[j] = len(peaks_j)
        if len(peaks_j) >= 2:
            # Select the two tallest peaks by height (read directly from PMF)
            heights_j = pmf_j[peaks_j]
            top2_idx  = np.argpartition(heights_j, -2)[-2:]
            top2_bins = peaks_j[top2_idx]          # bin indices of the two tallest peaks
            # Sort so i1 < i2 (left peak first)
            i1, i2 = int(top2_bins.min()), int(top2_bins.max())
            peak_sep_abs_arr[j] = float(log_x[i2] - log_x[i1])
            # Valley: bin of minimum PMF value between the two peaks (inclusive)
            valley_bin = int(i1 + np.argmin(pmf_j[i1 : i2 + 1]))
            # Component sigma: PMF-weighted std dev on each side of the saddle.
            # s1 = left (low-Q) component; s2 = right (high-Q) component.
            # These are stored directly so that spread asymmetry (max/min) can
            # be evaluated per station independently of peak separation.
            def _comp_sigma(mask: np.ndarray) -> float:
                mass = float(pmf_j[mask].sum())
                if mass <= 0:
                    return np.nan
                mu_c  = float(np.dot(log_x[mask], pmf_j[mask])) / mass
                var_c = float(np.dot((log_x[mask] - mu_c) ** 2, pmf_j[mask])) / mass
                return float(np.sqrt(max(var_c, 1e-12)))
            s1 = _comp_sigma(_bin_idx <= valley_bin)
            s2 = _comp_sigma(_bin_idx >  valley_bin)
            if np.isfinite(s1) and np.isfinite(s2) and s1 > 0 and s2 > 0:
                peak_sep_sigma_arr[j]   = peak_sep_abs_arr[j] / (0.5 * (s1 + s2))
                mode_sigma_left_arr[j]  = s1
                mode_sigma_right_arr[j] = s2
                # Spread asymmetry: ratio of larger to smaller component sigma.
                # Value >= 1; equal to 1 when components have identical spread.
                # Silverman (1986) does not address this case; it is an
                # empirically identified failure mode of the fixed bandwidth.
                spread_asymmetry_arr[j] = max(s1, s2) / min(s1, s2)

    _df = pd.DataFrame({
        "station_id":          list(common),
        "ks_stat":             ks,
        "wasserstein":         wass,
        "w1_norm":             w1_norm,
        "energy_distance":     enrg,
        "ed_norm":             ed_norm,
        "isd":                 isd,
        "kl_divergence":       kl,
        "sigma_log":           sigma,
        "ks_bin_index":        ks_indices.astype(int),
        "ks_log_x":            log_x_at_max,
        "ks_pmf_adp":          ab_at_max,
        "ks_pmf_sil":          fb_at_max,
        "ks_pmf_obs":          obs_at_max,
        "ks_qshift_adp_pct":   _qshift_pct(ab_at_max, require_positive=False),
        "ks_qshift_sil_pct":   _qshift_pct(fb_at_max, require_positive=False),
        "ks_qshift_obs_pct":   _qshift_pct(obs_at_max, require_positive=True),
        "max_qshift_adp_pct":  max_qshift_adp_pct,
        "max_qshift_sil_pct":  max_qshift_sil_pct,
        "max_qshift_adp_bin":  max_qshift_adp_bin,
        "max_qshift_sil_bin":  max_qshift_sil_bin,
        "obs_skewness":        obs_skewness,
        "n_peaks":             n_peaks_arr,
        "peak_sep_abs":        peak_sep_abs_arr,
        "peak_sep_sigma":      peak_sep_sigma_arr,
        "mode_sigma_left":     mode_sigma_left_arr,
        "mode_sigma_right":    mode_sigma_right_arr,
        "spread_asymmetry":    spread_asymmetry_arr,
    })
    return _df, shift_adp_full, shift_sil_full, delta


# ---------------------------------------------------------------------------
# Bimodality (Hartigan dip test) and record-length helpers
# ---------------------------------------------------------------------------

def _compute_dip_flags(region: str, station_ids: list[str]) -> pd.DataFrame:
    """Run the Hartigan & Hartigan (1985) dip test on each station's log-UAR
    sample from weibull_quantiles.parquet.

    Returns a DataFrame with columns:
      station_id  : str
      bimodal     : int  (1 if dip p-value < 0.05, else 0)
      dip_pval    : float

    The dip test is applied to the raw daily log-UAR observations, not to the
    KDE or the PMF grid.  It requires no bandwidth or prominence parameter and
    has a known asymptotic null distribution under unimodality.
    """
    wq_path = PARENT_CACHE_ROOT / region / "weibull_quantiles.parquet"
    if not wq_path.exists():
        print(f"  WARNING: weibull_quantiles.parquet not found for {region}; "
              "bimodal/dip_pval set to NaN.")
        return pd.DataFrame({
            "station_id": station_ids,
            "bimodal":    [np.nan] * len(station_ids),
            "dip_pval":   [np.nan] * len(station_ids),
        })
    wq = pd.read_parquet(wq_path, columns=["station_id", "log_uar"])
    stn_set = set(station_ids)
    wq = wq[wq["station_id"].isin(stn_set)]
    rows: list[dict] = []
    for stn_id, grp in wq.groupby("station_id", sort=False):
        sample = grp["log_uar"].to_numpy(dtype=float)
        sample = sample[np.isfinite(sample)]
        if len(sample) < 10:
            rows.append({"station_id": stn_id, "bimodal": np.nan, "dip_pval": np.nan})
            continue
        _, pval = diptest.diptest(sample)
        rows.append({
            "station_id": stn_id,
            "bimodal":    int(pval < 0.05),
            "dip_pval":   float(pval),
        })
    df_dip = pd.DataFrame(rows)
    # Include any station_ids present in station_ids but not in wq
    missing = [s for s in station_ids if s not in set(df_dip["station_id"])]
    if missing:
        df_dip = pd.concat([
            df_dip,
            pd.DataFrame({"station_id": missing,
                          "bimodal":    [np.nan] * len(missing),
                          "dip_pval":   [np.nan] * len(missing)}),
        ], ignore_index=True)
    return df_dip


# ---------------------------------------------------------------------------
# Per-bin quantile-shift profile helper
# ---------------------------------------------------------------------------

def _compute_perbin_profile(
    shift_adp: np.ndarray,
    shift_sil: np.ndarray,
    log_x: np.ndarray,
    delta: np.ndarray,
) -> pd.DataFrame:
    """Percentile profile of the per-bin implied quantile shift across stations.

    Parameters
    ----------
    shift_adp : (B, N) array - per-bin AB implied shift % for each of N stations.
    shift_sil : (B, N) array - per-bin FB implied shift % for each of N stations.
    log_x     : (B,) log-UAR grid coordinates.
    delta     : (B, N) array - signed CDF difference CDF_AB - CDF_FB for each station.

    Returns
    -------
    DataFrame with B rows (one per bin) and columns:
        log_x, p2p5_adp, p50_adp, p97p5_adp, p2p5_sil, p50_sil, p97p5_sil,
        p2p5_delta, p50_delta, p97p5_delta

    The 2.5th-97.5th percentile interval is a 95% coverage band across the
    station population at each flow level, showing where on the support the
    bandwidth choice is most sensitive to local density.  The signed delta
    columns capture direction: positive values mean CDF_AB > CDF_FB (AB assigns
    more mass below x, so AB gives a lower flow estimate at that exceedance
    probability); negative values mean AB gives a higher flow estimate.
    """
    pcts = [2.5, 50.0, 97.5]
    # nanpercentile over stations (axis=1): output shape (3, B), transposed to (B, 3).
    pa   = np.nanpercentile(shift_adp, pcts, axis=1).T   # (B, 3)
    ps   = np.nanpercentile(shift_sil, pcts, axis=1).T   # (B, 3)
    pd_d = np.nanpercentile(delta,     pcts, axis=1).T   # (B, 3) signed CDF difference
    return pd.DataFrame({
        "log_x":       log_x,
        "p2p5_adp":    pa[:, 0],
        "p50_adp":     pa[:, 1],
        "p97p5_adp":   pa[:, 2],
        "p2p5_sil":    ps[:, 0],
        "p50_sil":     ps[:, 1],
        "p97p5_sil":   ps[:, 2],
        "p2p5_delta":  pd_d[:, 0],
        "p50_delta":   pd_d[:, 1],
        "p97p5_delta": pd_d[:, 2],
    })


# ---------------------------------------------------------------------------
# Bin concentration helper
# ---------------------------------------------------------------------------

def _compute_bin_concentration(obs_df: pd.DataFrame, valid_stns) -> pd.DataFrame:
    """Return per-station fraction of total PMF mass held in the 4 largest bins.

    For a perfectly spread uniform PMF every bin carries 1/n_bins of the mass;
    a quantized station with most observations landing in the same cell will
    have nearly all its mass in 1-4 bins.  ``top4_mass`` is the sum of the four
    highest PMF values for each station (not the four highest-valued flow bins;
    just the four bins with the most probability mass).

    Parameters
    ----------
    obs_df     : DataFrame (n_bins x n_stations) from pmf_obs.csv
    valid_stns : index/list of station IDs to evaluate

    Returns
    -------
    DataFrame with columns ``station_id`` and ``top4_mass``.
    """
    arr = obs_df[valid_stns].values.astype(float)          # (n_bins, n_stations)
    sorted_arr = np.sort(arr, axis=0)                      # ascending along bin axis
    top2 = sorted_arr[-2:, :].sum(axis=0)                  # (n_stations,)
    top3 = sorted_arr[-3:, :].sum(axis=0)
    top4 = sorted_arr[-4:, :].sum(axis=0)
    top5 = sorted_arr[-5:, :].sum(axis=0)
    return pd.DataFrame({
        "station_id": list(valid_stns),
        "top2_mass":  top2,
        "top3_mass":  top3,
        "top4_mass":  top4,
        "top5_mass":  top5,
    })


# ---------------------------------------------------------------------------
# Main preprocessing step
# ---------------------------------------------------------------------------

def preprocess_region(region: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_scores   = CACHE_DIR / region / f"{region}_kde_comparison.parquet"
    out_worst    = CACHE_DIR / region / f"{region}_worst10_pmfs.parquet"
    out_median   = CACHE_DIR / region / f"{region}_median_pmfs.parquet"
    out_conc     = CACHE_DIR / region / f"{region}_bin_concentration.parquet"
    out_profile  = CACHE_DIR / region / f"{region}_perbin_qshift_profile.parquet"

    long_record_stns = None
    meta_path = PARENT_CACHE_ROOT / region / "station_meta.parquet"

    # Stale check against source PMF CSVs at all bitrates
    sources = [
        Path(__file__),
        *[
            _pmf_dir(region, b) / fname
            for b in BITRATES
            for fname in ("pmf_kde_adaptive.csv", "pmf_kde_silverman.csv", "pmf_obs.csv")
        ]
    ]
    if (not _is_stale(out_scores, *sources) and not _is_stale(out_worst, *sources)
            and not _is_stale(out_median, *sources) and not _is_stale(out_conc, *sources)
            and not _is_stale(out_profile, *sources)):
        print(f"  {region}: cache up to date, skipping.")
        return

    score_rows: list[pd.DataFrame]      = []
    worst_pmf_rows: list[dict]          = []
    median_pmf_rows: list[dict]         = []
    conc_rows: list[pd.DataFrame]       = []
    _perbin_profile_df: pd.DataFrame | None = None

    for bitrate in BITRATES:
        try:
            ab_df = _load_pmf(region, bitrate, "pmf_kde_adaptive.csv")
            fb_df = _load_pmf(region, bitrate, "pmf_kde_silverman.csv")
            obs_df = _load_pmf(region, bitrate, "pmf_obs.csv")
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc}")
            continue

        # Drop stations with only 1 non-zero observed bin: their entire
        # observed mass sits in a single UAR cell, so the two KDE estimators
        # are responding to a near-Dirac mass and the divergence scores are
        # not meaningful comparisons of bandwidth strategy.
        common = ab_df.columns.intersection(fb_df.columns).intersection(obs_df.columns)
        n_nonzero_bins = (obs_df[common] > 0).sum(axis=0)
        valid_stns = n_nonzero_bins[n_nonzero_bins > 1].index
        n_dropped = len(common) - len(valid_stns)
        if n_dropped > 0:
            print(f"  {region} bitrate={bitrate}: dropped {n_dropped} single-bin station(s)")

        if long_record_stns is not None:
            before = len(valid_stns)
            valid_stns = valid_stns[valid_stns.isin(long_record_stns)]
            if (n_short := before - len(valid_stns)):
                print(f"  {region} bitrate={bitrate}: dropped {n_short} station(s) with < {min_yrs} yr")

        ab_df = ab_df[valid_stns]
        fb_df = fb_df[valid_stns]
        obs_df = obs_df[valid_stns]

        scores, _adp_full, _sil_full, _delta = _compute_scores(ab_df, fb_df, obs_df)
        scores["bitrate"] = bitrate
        scores["region"]  = region
        score_rows.append(scores)

        conc_b             = _compute_bin_concentration(obs_df, valid_stns)
        conc_b["bitrate"]  = bitrate
        conc_b["region"]   = region
        conc_rows.append(conc_b)

        if bitrate == REF_BITRATE:
            log_x  = ab_df.index.to_numpy(dtype=float)
            _perbin_profile_df = _compute_perbin_profile(_adp_full, _sil_full, log_x, _delta)
            _worst_metrics = ["ks_stat", "wasserstein", "energy_distance", "kl_divergence"]
            for metric_col in _worst_metrics:
                ranked = scores.nlargest(N_WORST, metric_col)["station_id"].tolist()
                for rank_1based, stn in enumerate(ranked, start=1):
                    pmf_o = obs_df[stn].values if stn in obs_df.columns else np.zeros(len(log_x))
                    pmf_a = ab_df[stn].values if stn in ab_df.columns else np.zeros(len(log_x))
                    pmf_s = fb_df[stn].values if stn in fb_df.columns else np.zeros(len(log_x))
                    for lx, po, pa, ps in zip(log_x, pmf_o, pmf_a, pmf_s):
                        worst_pmf_rows.append({
                            "station_id": stn,
                            "metric":     metric_col,
                            "rank":       rank_1based,
                            "log_x":      float(lx),
                            "pmf_obs":    float(po),
                            "pmf_adp":    float(pa),
                            "pmf_sil":    float(ps),
                        })

            median6 = _find_median_stations(scores)
            for stn, metric_label in median6:
                pmf_o = obs_df[stn].values if stn in obs_df.columns else np.zeros(len(log_x))
                pmf_a = ab_df[stn].values if stn in ab_df.columns else np.zeros(len(log_x))
                pmf_s = fb_df[stn].values if stn in fb_df.columns else np.zeros(len(log_x))
                for lx, po, pa, ps in zip(log_x, pmf_o, pmf_a, pmf_s):
                    median_pmf_rows.append({
                        "station_id":   stn,
                        "metric_group": metric_label,
                        "log_x":        float(lx),
                        "pmf_obs":      float(po),
                        "pmf_adp":      float(pa),
                        "pmf_sil":      float(ps),
                    })

    if not score_rows:
        print(f"  {region}: no bitrate data found, skipping.")
        return

    df_scores = pd.concat(score_rows, ignore_index=True)

    # --- Join bimodality flags (dip test on raw daily log-UAR sample) ---
    # Run the dip test once per station (not per bitrate) using the 8-bit
    # station set as the reference population.
    ref_stns = (
        df_scores.loc[df_scores["bitrate"] == REF_BITRATE, "station_id"]
        .unique().tolist()
    )
    df_dip = _compute_dip_flags(region, ref_stns)
    df_scores = df_scores.merge(df_dip, on="station_id", how="left")
    print(f"  {region}: dip test flags joined "
          f"({int(df_dip['bimodal'].sum())} bimodal / {len(df_dip)} stations)")

    # --- Join record_years from station_meta ---
    if meta_path.exists():
        _meta_ry = pd.read_parquet(meta_path, columns=["record_years"]).reset_index()
        _meta_ry.columns = ["station_id", "record_years"]
        _meta_ry["station_id"] = _meta_ry["station_id"].astype(str)
        df_scores = df_scores.merge(_meta_ry, on="station_id", how="left")
    else:
        df_scores["record_years"] = np.nan

    df_scores.to_parquet(out_scores, index=False)
    print(f"  {region}: {len(df_scores)} rows -> {out_scores.name}")

    if worst_pmf_rows:
        pd.DataFrame(worst_pmf_rows).to_parquet(out_worst, index=False)
        print(f"  {region}: worst-{N_WORST} PMFs -> {out_worst.name}")

    if median_pmf_rows:
        pd.DataFrame(median_pmf_rows).to_parquet(out_median, index=False)
        print(f"  {region}: median-{N_MEDIAN} PMFs -> {out_median.name}")

    if conc_rows:
        pd.concat(conc_rows, ignore_index=True).to_parquet(out_conc, index=False)
        print(f"  {region}: bin concentration -> {out_conc.name}")

    if _perbin_profile_df is not None:
        _perbin_profile_df.to_parquet(out_profile, index=False)
        print(f"  {region}: per-bin qshift profile ({len(_perbin_profile_df)} bins) -> {out_profile.name}")


def main() -> None:
    regions = _resolve_regions()
    for region in regions:
        print(f"\nProcessing: {region}")
        preprocess_region(region)


if __name__ == "__main__":
    main()
