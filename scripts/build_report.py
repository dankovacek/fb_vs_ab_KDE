"""Build a combined HTML report for 09_max_kde_diffs.

Usage
-----
    python build_report.py [region|index|all]

Reads only the pre-computed local cache (written by preprocess.py).
Run preprocess.py first if cache files are missing.

Output
------
    max_kde_diffs.html  --  all cached regions shown side by side
                            (one column per region in every figure)
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import norm as sp_norm, kendalltau as sp_kendalltau, pearsonr as sp_pearsonr, gaussian_kde as sp_gaussian_kde, linregress as sp_linregress, t as sp_t
from shapely.geometry import Point
from bokeh.embed import components
from bokeh.layouts import column, gridplot, row
from bokeh.models import BoxAnnotation, ColorBar, ColumnDataSource, Div, FactorRange, HoverTool, Label, Legend, LegendItem, LinearColorMapper, LogAxis, Range1d, Span as BkSpan, Title as BkTitle
from bokeh.palettes import Category10, RdBu, RdYlGn, Viridis256
from bokeh.plotting import figure as bk_figure
from bokeh.transform import dodge
from _plot_helpers import (
    _AB_COLOR, _HIST_COLOR, _FB_COLOR,
    _apply_theme, _style_legend,
    _BODY_FONT, _LEGEND_STYLE_KW, _SIDE_TITLE_KW,
    _apply_dotwhisker_axis_style,
    REGION_NAMES, _REGION_COLORS,
)
from jinja2 import Environment, FileSystemLoader

from synthetic_test import build_synthetic_section
from fig_silverman_bounds import (
    _fig_silverman_bounds, _fig_record_length_stability,
    _SKEW_THRESHOLD, _PEAK_SEP_THRESHOLD, _SPREAD_ASYM_THRESHOLD,
    _THRESHOLD_COLOR,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR     = Path(__file__).resolve().parent
REPO_ROOT      = SCRIPT_DIR.parent
CACHE_DIR      = REPO_ROOT / "cache"
BASELINE_DIR   = REPO_ROOT / "data" / "baseline_distributions"
TEMPLATES_ROOT = REPO_ROOT / 'report'

sys.path.insert(0, str(SCRIPT_DIR))
from config import Config

BITRATES    = Config.SUPPORTED_BITRATES
REF_BITRATE = Config.DEFAULT_BITRATE
N_WORST     = 10
N_MEDIAN    = 8

# Per-metric (metric_col, 1-based rank) pairs to render in the worst-stations grid.
WORST_SELECTIONS: list[tuple[str, int]] = [
    ("ks_stat",         1),
    ("ks_stat",         3),
    ("wasserstein",     1),
    ("wasserstein",     8),
    ("energy_distance", 1),
    ("energy_distance", 2),
    ("kl_divergence",   2),
    ("kl_divergence",   3),
]

BITRATE_COLORS = {6: "#2B0000", 7: "#7A0202", 8: "#d10404", 10: "#ff7c7c"}

# ---------------------------------------------------------------------------
# Metric display metadata: edit these dicts to update units everywhere.
#
# Metric display transforms applied at data-load time by _apply_metric_transforms().
#
# Transforms applied:
#   KS   :  × 100              (probability gap → pp)
#   W1   :  100*(exp(s) - 1)   (L1 CDF-gap area → % equivalent flow shift)
#   ED   :  ED²               (L2 CDF-gap area, log-flow units; same scale as W1)
#   ISD  :  unchanged          (∫(f_AB − f_FB)² d(ln x), log-flow units⁻¹)
#   KL   :  unchanged          (bits)
# ---------------------------------------------------------------------------
METRIC_UNITS: dict[str, str] = {
    "ks_stat":         "%",
    "wasserstein":     "%",
    "energy_distance": "log-flow units",
    "isd":             "log-flow units\u207b\u00b9",
    "kl_divergence":   "bit",
}

# Short display names used in legends, table headers, and correlation heatmaps.
METRIC_LABELS: dict[str, str] = {
    "ks_stat":         "KS",
    "wasserstein":     "W\u2081",
    "energy_distance": "ED\u00b2",
    "isd":             "ISD",
    "kl_divergence":   "KL",
}

# Axis labels combining short name and unit: "NAME (unit)".
# Reference this dict in all plot x_axis_label / y_axis_label arguments.
METRIC_AXIS_LABELS: dict[str, str] = {
    k: f"{METRIC_LABELS[k]} ({METRIC_UNITS[k]})"
    for k in METRIC_UNITS
}


def _apply_metric_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with divergence columns scaled to display units.

    KS   : × 100             # probability gap -> %
    W1   : 100*(exp(s) - 1)  # L1 CDF-gap area -> % equivalent flow shift
    ED   : ED²               # L2 CDF-gap area (log-flow units; same scale as W1)
    ISD  : unchanged         # integrated squared density difference (log-flow units⁻¹)
    KL   : unchanged         # bits
    """
    df = df.copy()
    # Backward-compat: older cache files stored this column as "mise".
    if "mise" in df.columns and "isd" not in df.columns:
        df = df.rename(columns={"mise": "isd"})
    if "ks_stat" in df.columns:
        df["ks_stat"] = df["ks_stat"] * 100.0
    if "wasserstein" in df.columns:
        df["wasserstein"] = 100.0 * np.expm1(df["wasserstein"].to_numpy(dtype=float))
    if "energy_distance" in df.columns:
        df["energy_distance"] = df["energy_distance"] ** 2
    return df

# Target widths per panel (keep figures manageable at many regions)
_W_SUMMARY = 280   # ECDF / scatter panel width
_H_SUMMARY = 220   # ECDF / scatter panel height
_W_STRAT   = 170   # stratum grid cell width  (small multiples)
_H_STRAT   = 130   # stratum grid cell height

# ---------------------------------------------------------------------------
# Rule Set C (bin-concentration) parameters: edit these to test thresholds
# ---------------------------------------------------------------------------
# _BIN_CONC_N_BINS : how many top bins to sum  (must be 2, 3, 4, or 5)
# _BIN_CONC_THRESHOLD : the fraction at-or-above which the rule fires
_BIN_CONC_N_BINS      = 4
_BIN_CONC_THRESHOLD   = 0.5
_W_WORST   = 320   # worst-10 panel width
_H_WORST   = 200   # worst-10 panel height
_MAP_W     = 320   # station map width
_MAP_H     = 320   # station map height
_TOOLS_MAP = "pan,wheel_zoom,reset,save"

# ---------------------------------------------------------------------------
# Region selection and metadata helpers
# ---------------------------------------------------------------------------

def _resolve_regions(region_arg: "str | None" = None) -> list[str]:
    regions = sorted(
        p.stem.replace("_kde_comparison", "")
        for p in CACHE_DIR.glob("*/*_kde_comparison.parquet")
    )
    if not regions:
        raise SystemExit(
            f"No cached results found in:\n  {CACHE_DIR}\n"
            "Run preprocess.py first."
        )
    print("Available regions (all will be combined):")
    for i, name in enumerate(regions):
        print(f"  {i}: {name}")

    if region_arg is not None:
        arg = region_arg
        if arg == "all":
            return regions
        if arg.isdigit():
            idx = int(arg)
            if 0 <= idx < len(regions):
                return [regions[idx]]
        if arg in regions:
            return [arg]

    # Default: use all cached regions
    print("Using all cached regions.")
    return regions


def _load_record_years(region: str) -> dict[str, int]:
    """Return {station_id: record_years} from station_meta for a region."""
    meta_path = CACHE_DIR / region / "station_meta.parquet"
    if not meta_path.exists():
        return {}
    meta = pd.read_parquet(meta_path, columns=["record_years"])
    return meta["record_years"].dropna().astype(int).to_dict()


def _load_drain_area(region: str) -> dict[str, float]:
    """Return {station_id: da_km2} from station_meta for a region."""
    meta_path = CACHE_DIR / region / "station_meta.parquet"
    if not meta_path.exists():
        return {}
    meta = pd.read_parquet(meta_path, columns=["da_km2"])
    return meta["da_km2"].dropna().to_dict()


def _load_attr_meta(region: str) -> pd.DataFrame:
    """Return DataFrame of display-ready catchment attributes indexed by station_id.

    Columns written (where present in the cache):
      da_km2               - drainage area (km²)
      prcp_mm_yr           - mean annual precipitation (mm/yr)  [source: prcp mm/day x 365]
      tmean_c              - mean annual temperature (°C)        [source: tmean 0.1 °C / 10]
      record_years         - record length (complete years)
      swe                  - mean annual snow cover (%)          [raw]
      zero_flow_frac       - fraction of days with zero flow     [raw]
      high_prcp_freq       - high-precipitation frequency        [raw]
      low_prcp_freq        - low-precipitation frequency         [raw]
      high_prcp_duration   - mean high-precip. duration (d)      [raw]
      low_prcp_duration    - mean low-precip. duration (d)       [raw]
      elevation_m          - mean basin elevation (m)            [raw]
      slope_deg            - mean basin slope (degrees)          [raw]
      land_use_forest_frac - forest cover fraction (0-100)       [raw]
      cv_flows             - CV of annual unit-area runoff        [raw]
    """
    meta_path = CACHE_DIR / region / "station_meta.parquet"
    if not meta_path.exists():
        return pd.DataFrame(columns=["da_km2", "prcp_mm_yr", "tmean_c", "record_years"])
    meta = pd.read_parquet(meta_path)
    out = pd.DataFrame(index=meta.index)
    if "da_km2" in meta.columns:
        out["da_km2"] = pd.to_numeric(meta["da_km2"], errors="coerce")
    if "prcp" in meta.columns:
        out["prcp_mm_yr"] = pd.to_numeric(meta["prcp"], errors="coerce") * 365.0
    if "tmean" in meta.columns:
        out["tmean_c"] = pd.to_numeric(meta["tmean"], errors="coerce") / 10.0
    if "record_years" in meta.columns:
        out["record_years"] = pd.to_numeric(meta["record_years"], errors="coerce")
    for _col in (
        "swe", "zero_flow_frac",
        "high_prcp_freq", "low_prcp_freq",
        "high_prcp_duration", "low_prcp_duration",
        "elevation_m", "slope_deg", "land_use_forest_frac", "cv_flows",
    ):
        if _col in meta.columns:
            out[_col] = pd.to_numeric(meta[_col], errors="coerce")
    return out


def _load_station_coords(region: str) -> pd.DataFrame:
    """Return DataFrame with columns [station_id, mx, my] in EPSG:3857."""
    meta_path = CACHE_DIR / region / "station_meta.parquet"
    if not meta_path.exists():
        return pd.DataFrame(columns=["station_id", "mx", "my"])
    meta = pd.read_parquet(meta_path, columns=["lat", "lon"]).reset_index()
    geometry = [Point(lon, lat) for lon, lat in zip(meta["lon"], meta["lat"])]
    gdf = gpd.GeoDataFrame(meta, geometry=geometry, crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3857")
    meta["mx"] = gdf.geometry.x
    meta["my"] = gdf.geometry.y
    return meta[["station_id", "mx", "my"]]


def _fig_station_map(
    region: str,
    df8: pd.DataFrame,
    coords: pd.DataFrame,
) -> object:
    """Station-location map coloured by the section-6.4.1 segmentation criterion.

    Dark grey: |skew| < _SKEW_THRESHOLD AND (unimodal OR peak_sep_sigma <= _PEAK_SEP_THRESHOLD).
    Red: all other stations.
    """
    label = REGION_NAMES.get(region, region)

    cols = ["station_id"]
    for _c in ("obs_skewness", "peak_sep_sigma"):
        if _c in df8.columns:
            cols.append(_c)
    merged = coords.merge(df8[cols], on="station_id", how="inner")
    if merged.empty:
        return Div(text=f"<em>No coordinate data for {region}.</em>", width=_MAP_W)

    skew = (
        merged["obs_skewness"].to_numpy(dtype=float)
        if "obs_skewness" in merged.columns
        else np.full(len(merged), np.nan)
    )
    ps = (
        merged["peak_sep_sigma"].to_numpy(dtype=float)
        if "peak_sep_sigma" in merged.columns
        else np.full(len(merged), np.nan)
    )
    ps_ok    = np.isnan(ps) | (ps <= _PEAK_SEP_THRESHOLD)
    in_group = np.isfinite(skew) & (np.abs(skew) < _SKEW_THRESHOLD) & ps_ok

    colors = np.where(in_group, "#555555", "#e73535").tolist()

    src = ColumnDataSource({
        "mx":         merged["mx"].tolist(),
        "my":         merged["my"].tolist(),
        "station_id": merged["station_id"].tolist(),
        "color":      colors,
    })

    _buf_frac = 0.08 if region in ("camelsgb", "camelsbr") else 0.0
    _mx = merged["mx"].to_numpy(dtype=float)
    _my = merged["my"].to_numpy(dtype=float)
    _range_kw: dict = {}
    if _buf_frac:
        # Expand data extents uniformly then fit the opposite axis so the
        # padded span respects the frame's pixel aspect ratio (_MAP_W / _MAP_H).
        _aspect = _MAP_W / _MAP_H
        _x_span = (_mx.max() - _mx.min()) * (1 + 2 * _buf_frac)
        _y_span = (_my.max() - _my.min()) * (1 + 2 * _buf_frac)
        if _x_span / _y_span > _aspect:
            # data is wider than frame: expand y to match
            _y_span = _x_span / _aspect
        else:
            # data is taller than frame: expand x to match
            _x_span = _y_span * _aspect
        _xc = (_mx.min() + _mx.max()) / 2
        _yc = (_my.min() + _my.max()) / 2
        _range_kw = {
            "x_range": Range1d(_xc - _x_span / 2, _xc + _x_span / 2),
            "y_range": Range1d(_yc - _y_span / 2, _yc + _y_span / 2),
        }

    fig = bk_figure(
        frame_width=_MAP_W, frame_height=_MAP_H,
        title=label,
        x_axis_type="mercator", y_axis_type="mercator",
        tools=_TOOLS_MAP,
        toolbar_location=None,
        **_range_kw,
    )
    _apply_theme(fig)
    fig.title.text_font_size = "13px"
    fig.add_tile("CartoDB Positron")

    _n_in  = int(np.sum(in_group))
    _n_out = int(np.sum(~in_group))

    _src_in  = ColumnDataSource({k: [v for v, f in zip(src.data[k], in_group) if f]     for k in ("mx", "my", "station_id")})
    _src_out = ColumnDataSource({k: [v for v, f in zip(src.data[k], in_group) if not f] for k in ("mx", "my", "station_id")})
    sz =2 if region in ("camelsgb", "camelsbr") else 1.4
    r_in  = fig.scatter("mx", "my", source=_src_in,  size=sz, marker="circle",  fill_alpha=0.8,
                        fill_color="#383838", line_color=None)
    r_out = fig.scatter("mx", "my", source=_src_out, size=sz, marker="circle", fill_alpha=0.8,
                        fill_color="#d63c3c", line_color=None)

    fig.add_tools(HoverTool(tooltips=[("Station", "@station_id")]))
    legend_loc = 'top_right'if region in ("camelsgb", "hysets") else 'bottom_left'

    legend = Legend(items=[
        LegendItem(label=f"\u03b3\u2081<{_SKEW_THRESHOLD}, unimodal \nor ps\u2264{_PEAK_SEP_THRESHOLD}\u03c3  (n = {_n_in})",  renderers=[r_in]),
        LegendItem(label=f"other  (n = {_n_out})", renderers=[r_out]),
    ], location=legend_loc, label_text_font_size="9px")
    fig.add_layout(legend)
    fig.legend.background_fill_alpha = 0.7
    fig.grid.visible = False
    return fig


def _fig_error_model() -> bk_figure:
    """Piecewise measurement-error bandwidth floor (Config.KDE)."""
    bps = np.array(Config.KDE.ERROR_MODEL_BREAKPOINTS, dtype=float)
    bvs = np.array(Config.KDE.ERROR_MODEL_VALUES, dtype=float)
    fig = bk_figure(
        width=520, height=280,
        x_axis_type="log",
        x_axis_label="Flow (m\u00b3/s)",
        y_axis_label="Relative error",
        title="Measurement-error bandwidth floor",
        toolbar_location=None,
    )
    _apply_theme(fig)
    fig.y_range = Range1d(0, 1.05)
    # fig.add_layout(BoxAnnotation(
    #     left=0.1, right=100.0,
    #     fill_color="#d4edda", fill_alpha=0.50,
    #     line_color=None,
    # ))
    # fig.add_layout(BkSpan(
    #     location=0.05, dimension="width",
    #     line_dash="dashed", line_color="#888888", line_width=1,
    # ))
    fig.line(bps, bvs, line_width=2.0, color="#222222")
    fig.scatter(bps, bvs, size=6, color="#222222")
    # fig.add_layout(Label(
    #     x=0.15, y=0.08,
    #     text="5 % nominal accuracy (0.1\u2013100 m\u00b3/s)",
    #     text_font="EB Garamond, serif", text_font_size="12px",
    #     text_color="#555555", x_units="data", y_units="data",
    # ))
    return fig


# ---------------------------------------------------------------------------
# Per-region figure builders (return single bk_figure, one column per region)
# ---------------------------------------------------------------------------

# (column_name, x-axis label, row title suffix)
_ECDF_METRICS = [
    ("isd",             METRIC_AXIS_LABELS["isd"],             "ISD",   (1e-6, 1e0)),
    ("ks_stat",         METRIC_AXIS_LABELS["ks_stat"],         "KS",    (0.04, 20)),
    ("wasserstein",     METRIC_AXIS_LABELS["wasserstein"],     "W\u2081", (0.04, 25)),
    ("energy_distance", METRIC_AXIS_LABELS["energy_distance"], "ED\u00b2", (2e-7, 1e-1)),
    ("kl_divergence",   METRIC_AXIS_LABELS["kl_divergence"],   "KL",    (1e-4, 7e0)),
]

_QSHIFT_METRICS = [
    ("ks_qshift_adp_pct", "AB quantile shift %"),
    ("ks_qshift_sil_pct", "FB quantile shift %"),
]

def _fig_ecdf_region(
    df: pd.DataFrame, region: str,
    metric: str = "ks_stat",
    x_label: str = METRIC_AXIS_LABELS["ks_stat"],
    x_range=None,
    y_label=True,
    show_title: bool = False,
    show_legend: bool = False,
    x_range_default=(1e-5, 2e-1),
) -> bk_figure:
    """ECDF of a divergence metric at each bitrate for one region."""
    label = REGION_NAMES.get(region, region)
    kw = dict(
        frame_width=_W_SUMMARY-50, frame_height=_H_SUMMARY-40,
        x_axis_label=x_label,
        y_axis_label="P(X \u2264 x)" if y_label else '',
        title=label if show_title else '',
        tools="pan,wheel_zoom,reset,save",
        x_axis_type="log",
    )
    kw["x_range"] = x_range if x_range is not None else Range1d(*x_range_default)
    if metric == 'kl_divergence':
        kw["x_range"] = Range1d(1e-4, 7)
        
    fig = bk_figure(**kw)
    _apply_theme(fig)
    for b in BITRATES:
        sub = df[df["bitrate"] == b][metric].dropna().sort_values().to_numpy()
        if len(sub) == 0:
            continue
        ecdf = np.arange(1, len(sub) + 1) / len(sub)
        line_dash = 'solid' if b == REF_BITRATE else 'dashed'
        fig.line(sub, ecdf, line_width=2.0, color=BITRATE_COLORS[b],
                 legend_label=f"{b}-bit", line_dash=line_dash)
        fig.scatter([float(np.median(sub))], [0.5], size=7,
                    color=BITRATE_COLORS[b], marker="circle")
    fig.add_layout(BkSpan(location=0.5, dimension="width",
                           line_color="#999999", line_dash="dashed", line_width=1))
    _style_legend(fig, "top_left", "12px")
    if not y_label:
        fig.yaxis.visible = False
    if not show_legend:
        fig.legend.visible = False
    return fig

def _fig_worst_panel(
    worst_df: pd.DataFrame,
    stations: list[str],
    score_map: dict,
    rec_years: dict,
    drain_area: dict,
    rank: int,
    region: str,
    median_labels: "list[str] | None" = None,
    show_y_label: bool = False,
    show_legend: bool = False,
) -> "bk_figure | None":
    """One worst-station density panel: rank-th worst in this region."""
    if rank >= len(stations):
        return None
    stn   = stations[rank]
    label = median_labels[rank] if median_labels is not None else None
    sdata = (
        worst_df[worst_df["station_id"] == stn]
        .drop_duplicates("log_x")
        .sort_values("log_x")
    )
    if sdata.empty:
        return None

    log_x   = sdata["log_x"].to_numpy(dtype=float)
    dlog    = np.maximum(np.gradient(log_x), 1e-12)
    lin_x   = np.exp(log_x)
    half    = np.exp(dlog / 2.0)

    pmf_obs_arr = sdata["pmf_obs"].to_numpy(dtype=float)
    pmf_adp_arr = sdata["pmf_adp"].to_numpy(dtype=float)
    pmf_sil_arr = sdata["pmf_sil"].to_numpy(dtype=float)

    yrs    = rec_years.get(stn, "?")
    da     = drain_area.get(stn, None)
    da_str = f", {da:,.0f} km\u00b2" if da is not None else ""

    # Station-specific zero-equivalent threshold (L/s/km²)
    uar_threshold = (0.1 / da) if (da is not None and da > 0) else None

    # Zero-equivalent bin: all bins whose right edge <= threshold.
    # Bin 0 always captures this mass; any bins in the dead zone are forced 0.
    # p_zero is the empirical probability of sub-threshold flow.
    p_zero_obs = float(pmf_obs_arr[0]) if len(pmf_obs_arr) > 0 else 0.0
    p_zero_ab  = float(pmf_adp_arr[0]) if len(pmf_adp_arr) > 0 else 0.0
    p_zero_fb  = float(pmf_sil_arr[0]) if len(pmf_sil_arr) > 0 else 0.0

    # Exclude bin 0 from the positive-flow density (bins 1 onward)
    dens_obs = pmf_obs_arr / dlog
    dens_ab  = pmf_adp_arr / dlog
    dens_fb  = pmf_sil_arr / dlog
    # Zero out bin 0 in the density arrays so it is not rendered as a bar
    dens_obs[0] = 0.0
    dens_ab[0]  = 0.0
    dens_fb[0]  = 0.0

    # Derive x-range from the adaptive KDE CDF so the axis tightly brackets
    # the actual distribution regardless of how sparse or spread the data are.
    # Skip bin 0 (zero-equivalent) when computing the visible range.
    pmf_ref   = pmf_adp_arr.copy()
    pmf_ref[0] = 0.0
    pmf_total = pmf_ref.sum()
    if pmf_total > 0:
        cum_norm = np.cumsum(pmf_ref) / pmf_total
        lo_idx   = int(np.searchsorted(cum_norm, 0.001))
        hi_idx   = int(np.searchsorted(cum_norm, 0.999))
        x_min    = lin_x[max(lo_idx - 1, 1)] / 3.0   # start at bin 1 minimum
        x_max    = lin_x[min(hi_idx, len(lin_x) - 1)] * 3.0
    else:
        x_min, x_max = 0.1, 10000

    # When there is appreciable zero-flow mass, extend x_min leftward so the
    # BoxAnnotation (zero-equivalent region shading) is visible on the plot.
    _max_pz = max(p_zero_obs, p_zero_ab, p_zero_fb)
    if uar_threshold is not None and _max_pz > 0.01:
        x_min = min(x_min, uar_threshold * 0.25)

    y_max = float(np.max(dens_ab)) * 1.10
    y_max = max(y_max, 1e-6)

    # First row of a column: add region label to title
    label_suffix   = f"  \u2014 {label}" if label is not None else ""
    title_str      = f"{stn}  (N = {yrs} yr{da_str}){label_suffix}"

    f = bk_figure(
        frame_width=_W_WORST, frame_height=_H_WORST,
        x_axis_type="log",
        x_axis_label="UAR (L/s/km\u00b2)",
        y_axis_label="Density",
        title=title_str,
        y_range=Range1d(0, y_max),
        x_range=Range1d(x_min, x_max),
        tools="pan,wheel_zoom,reset,save",
    )
    _apply_theme(f)
    f.title.text_font_size = "12px"

    f.quad(left=lin_x / half, right=lin_x * half, top=dens_obs, bottom=0,
           fill_color=_HIST_COLOR, line_color=None, alpha=0.85,
           legend_label="Empirical")
    f.line(lin_x, dens_ab, line_width=2.0, color=_AB_COLOR,
           legend_label="AB")
    f.line(lin_x, dens_fb, line_width=2.0, color=_FB_COLOR,
           legend_label="FB", line_dash="dashed")

    # Zero-equivalent region: shade everything left of the station threshold.
    # When p_zero is appreciable, x_min was already extended leftward so this
    # box is visible; for perennial streams the box is off the left edge.

    # p_zero label: show when any estimator has appreciable zero-flow mass.
    # Positioned just inside the right edge of the grey box.
    if _max_pz > 0.00 and uar_threshold is not None:
        # set the y coordinate to the probability mass of the zero-equivalent bin, 
        # but cap it at 98% of the y-axis max so it doesn't collide with the top edge of the plot
        y_pz = min(_max_pz, 0.98 * y_max)
        f.add_layout(BoxAnnotation(
            right=uar_threshold,
            top=min(y_pz, y_max * 0.95),
            fill_color="#727272", fill_alpha=0.65,
            line_color="#292929", line_dash="dotted", line_width=1.0,
        ))
        zero_equiv_label = Label(
            x=uar_threshold, y=_max_pz * 0.95,
                x_units="data", y_units="data",
                text=(
                    f"p(0-equiv.) = {p_zero_ab:.2f}\n"
                ),
                text_align="left", text_baseline="top",
                text_font="EB Garamond, serif", text_font_size="12px",
                text_color="#202020",
            )
        f.add_layout(zero_equiv_label)

    _style_legend(f, "top_right", "12px")
    if not show_legend:
        f.legend.visible = False
    f.yaxis.visible=show_y_label # hide axis if no legend, to save space    
    return f

# ---------------------------------------------------------------------------
# Catchment attribute CDF figure
# ---------------------------------------------------------------------------

_ATTR_PANEL_SPECS = [
    # (attr_col,     x_label,                     x_axis_type, show_y_label)
    ("da_km2",       "Drainage area (km\u00b2)",  "log",        True),
    ("prcp_mm_yr",   "Annual precip. (mm/yr)",    "linear",     False),
    ("tmean_c",      "Annual temp. (\u00b0C)",    "linear",     False),
    ("record_years", "Record length (yr)",         "linear",     False),
]

def _fig_attr_cdfs(regions: list[str], region_data: dict) -> object:
    """Four-panel ECDF comparing catchment characteristics across regions.

    One line per region in each panel; legend entries include the short region
    name and the scored-station count N (from the 8-bit dataset).
    """
    panels = []
    for ii, (col, x_label, x_axis_type, show_y_label) in enumerate(_ATTR_PANEL_SPECS):
        fig = bk_figure(
            frame_width=280, frame_height=220,
            x_axis_label=x_label,
            y_axis_label="P(X \u2264 x)" if show_y_label else "",
            x_axis_type=x_axis_type,
            toolbar_location=None,
            tools="",
        )
        _apply_theme(fig)
        if not show_y_label:
            fig.yaxis.visible = False
        for i, r in enumerate(regions):
            color = _REGION_COLORS.get(r, Category10[10][i % 10])
            attr_df = region_data[r].get("attr_meta", pd.DataFrame())
            n_total = region_data[r]["df8"]["station_id"].nunique()
            if col not in attr_df.columns:
                continue
            # Restrict to the scored station set for consistency with the rest
            # of the report
            scored_ids = set(region_data[r]["df8"]["station_id"].unique())
            vals = (
                attr_df.loc[attr_df.index.isin(scored_ids), col]
                .dropna()
                .sort_values()
                .to_numpy(dtype=float)
            )
            if len(vals) == 0:
                continue
            ecdf = np.arange(1, len(vals) + 1) / len(vals)
            short = REGION_NAMES.get(r, r).split("\u00b7")[0].strip()
            legend_label = f"{short}  (N\u202f=\u202f{n_total:,})"
            fig.line(
                vals, ecdf,
                line_width=2.0,
                color=color,
                legend_label=legend_label,
            )
        if ii == len(_ATTR_PANEL_SPECS) - 1:
            _style_legend(fig, "bottom_right")
        else:
            fig.legend.visible = False
        panels.append(fig)

    if not panels:
        return Div(text="<em>No catchment attribute data available.</em>")
    return gridplot([panels], merge_tools=False, toolbar_location=None)


# ---------------------------------------------------------------------------
# Pairwise metric correlation heatmaps
# ---------------------------------------------------------------------------

_CORR_METRIC_COLS = [
    "ks_stat", "wasserstein", "energy_distance", "isd", "kl_divergence",
]
_CORR_METRIC_LABELS = {k: METRIC_LABELS[k] for k in _CORR_METRIC_COLS}


def _fig_metric_corr_heatmaps(
    regions: list[str],
    region_data: dict,
) -> object:
    """Two side-by-side lower-triangle heatmaps of pairwise metric correlations.

    Left panel: Pearson r (linear).  Right panel: Kendall tau (rank).
    Per-station scores are pooled across all regions from the 8-bit dataset.
    Only the lower triangle is rendered; the upper triangle and diagonal are
    left blank as they encode no additional information.
    """
    parts = []
    for r in regions:
        df = region_data[r]["df8"]
        avail = [c for c in _CORR_METRIC_COLS if c in df.columns]
        if avail:
            parts.append(df[avail])
    if not parts:
        return Div(text="<em>No metric data available for correlation heatmaps.</em>")

    combined = pd.concat(parts, ignore_index=True)
    avail_cols = [c for c in _CORR_METRIC_COLS if c in combined.columns]
    combined = combined[avail_cols].dropna()
    n = len(avail_cols)
    labels = [_CORR_METRIC_LABELS[c] for c in avail_cols]
    data = combined.to_numpy(dtype=float)

    # Compute pairwise correlations for the lower triangle only
    pearson_mat = np.full((n, n), np.nan)
    kendall_mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i):
            valid = np.isfinite(data[:, i]) & np.isfinite(data[:, j])
            if valid.sum() < 3:
                continue
            r_val, _   = sp_pearsonr(data[valid, i],  data[valid, j])
            tau_val, _ = sp_kendalltau(data[valid, i], data[valid, j])
            pearson_mat[i, j] = r_val
            kendall_mat[i, j] = tau_val

    # Colour range driven by the actual data minimum, high fixed at 1
    _all_vals = np.concatenate([
        pearson_mat[np.isfinite(pearson_mat)],
        kendall_mat[np.isfinite(kendall_mat)],
    ])
    _low  = float(np.floor(_all_vals.min() * 10) / 10) if len(_all_vals) else 0.0
    _high = 1.0

    # Sequential palette: near-white (low correlation) → bright green (r = 1)
    _t = np.linspace(0, 1, 256)
    _pr = np.interp(_t, [0, 0.5, 1], [255, 204,   0]).astype(int)
    _pg = np.interp(_t, [0, 0.5, 1], [255, 255, 204]).astype(int)
    _pb = np.interp(_t, [0, 0.5, 1], [255, 204,  68]).astype(int)
    palette = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in zip(_pr, _pg, _pb)]
    mapper = LinearColorMapper(palette=palette, low=_low, high=_high, nan_color="#d0d0d0")

    # Axis factor ranges
    # x (columns): labels[0] .. labels[n-2]  (left to right)
    # y (rows): labels[1] .. labels[n-1], reversed so labels[1] is at the top
    x_factors = labels[:-1]
    y_factors  = list(reversed(labels[1:]))
    FW = 80 * (n - 1)
    FH = 80 * (n - 1)

    def _make_source(mat: np.ndarray) -> ColumnDataSource:
        xs, ys, vals, txts = [], [], [], []
        for i in range(n):
            for j in range(i):
                v = float(mat[i, j])
                xs.append(labels[j])
                ys.append(labels[i])
                vals.append(v)
                txts.append(f"{v:.2f}" if np.isfinite(v) else "")
        return ColumnDataSource(dict(x=xs, y=ys, val=vals, txt=txts))

    panels = []
    for title, mat, show_cbar in [
        ("Pearson r", pearson_mat, False),
        ("Kendall \u03c4", kendall_mat, True),
    ]:
        src = _make_source(mat)
        p = bk_figure(
            frame_width=FW, frame_height=FH,
            x_range=FactorRange(*x_factors),
            y_range=FactorRange(*y_factors),
            title=title,
            toolbar_location=None,
            tools="hover",
        )
        _apply_theme(p)
        p.grid.grid_line_color = None
        p.rect(
            x="x", y="y", width=0.995, height=0.995,
            source=src,
            fill_color={"field": "val", "transform": mapper},
            line_color=None,
        )
        p.text(
            x="x", y="y", text="txt",
            source=src,
            text_align="center",
            text_baseline="middle",
            text_font={"value": "EB Garamond, serif"},
            text_font_size={"value": "16px"},
            # text_font_style={"value": "bold"},
            text_color={"value": "#1b1919"},
        )
        p.select_one(HoverTool).tooltips = [
            ("Pair", "@y \u2013 @x"),
            ("Value", "@val{0.3f}"),
        ]
        if show_cbar:
            p.add_layout(
                ColorBar(
                    color_mapper=mapper,
                    width=14,
                    label_standoff=6,
                    major_label_text_font="EB Garamond, serif",
                    major_label_text_font_size="14px",
                    title="\u03c1",
                    title_text_font="EB Garamond, serif",
                    title_text_font_size="14px",
                ),
                "right",
            )
        panels.append(p)

    return gridplot([panels], merge_tools=False, toolbar_location=None)



# ---------------------------------------------------------------------------
# Skewness-stratum divergence figures
# ---------------------------------------------------------------------------

# _STRATUM_WITHIN  = "#3A9EA5"   # teal, stations within Silverman MISE bound |g| ≤ 1.8)
# _STRATUM_OUTSIDE = "#E07B39"   # orange, stations outside bound (|g| > 1.8)
_COLOR_BLACK = "#222222"
_COLOR_GREY  = "#888888"
_COLOR_BLUE  = "#2c6fad"

_STRATUM_METRICS: list[tuple[str, str, tuple]] = [
    ("isd",             METRIC_AXIS_LABELS["isd"],             (1e-6, 1e0)),
    ("ks_stat",         METRIC_AXIS_LABELS["ks_stat"],         (0.04, 20)),
    ("wasserstein",     METRIC_AXIS_LABELS["wasserstein"],     (0.04, 25)),
    ("energy_distance", METRIC_AXIS_LABELS["energy_distance"], (2e-7, 1e-1)),
    ("kl_divergence",   METRIC_AXIS_LABELS["kl_divergence"],   (1e-4, 7e0)),
]

# Metrics whose CI values span orders of magnitude: use log x-axis in dotwhisker.
# Other stratification figures already use log scale for all metrics.
_STRATUM_LOG_SCALE: dict[str, bool] = {
    "isd":             False,#True,
    "ks_stat":         False,
    "wasserstein":     False,
    "energy_distance": False,#True,
    "kl_divergence":   False,#True,
}


# ---------------------------------------------------------------------------
# Bootstrap mean + 95 % CI helpers for stratification tables
# ---------------------------------------------------------------------------
_N_BOOTSTRAP = 2000
_RNG_SEED = 42


def _fmt_stat(m: float, lo: float, hi: float) -> str:
    """Format a bootstrap statistic and its CI for HTML table cells.

    Uses 3 significant figures (:.3g) throughout so that values spanning
    many orders of magnitude (e.g. MISE 1e-6 to 1, KL 2e-4 to 7)
    are rendered with consistent information content rather than a fixed
    number of decimal places that would lose precision for small values.
    """
    if not np.isfinite(m):
        return "&ndash;"
    return f"{m:.2f}<br><small>[{lo:.2f},&thinsp;{hi:.2f}]</small>"


def _bootstrap_stat_ci(
    vals: np.ndarray,
    stat: str = "median",
    n_boot: int = _N_BOOTSTRAP,
    seed: int = _RNG_SEED,
    ci: float = 0.95,
) -> "tuple[float, float, float]":
    """Return (stat, ci_low, ci_high) via percentile bootstrap on *vals*.

    stat : 'median' (default) or 'mean'.  Median is preferred for skewed
    non-negative divergence distributions.
    Only finite, positive values are used.  Returns (nan, nan, nan) when
    fewer than 5 valid values are available.
    """
    v = vals[np.isfinite(vals) & (vals > 0)]
    if len(v) < 5:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    fn = np.median if stat == "median" else np.mean
    boot_stats = fn(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    alpha = 1.0 - ci
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2)))
    return (float(fn(v)), lo, hi)


def _bootstrap_median_ci(vals: np.ndarray, **kw) -> "tuple[float, float, float]":
    """Return (median, ci_low, ci_high) via percentile bootstrap on *vals*."""
    return _bootstrap_stat_ci(vals, stat="median", **kw)


def _ci_overlap(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
    """Return True when two 95 % CIs overlap (intervals [lo1,hi1] and [lo2,hi2])."""
    return lo1 <= hi2 and lo2 <= hi1


def _html_strat_bootstrap_table(
    regions: list[str],
    region_data: dict,
    rule_label: str,
    group_a_mask_fn,
    group_b_mask_fn,
    group_a_label: str,
    group_b_label: str,
) -> str:
    """Build an HTML summary table showing bootstrap median and 95 % CI for two groups.

    Parameters
    ----------
    regions : list[str]
        Active regions.
    region_data : dict
        Per-region data dict as built in build_report().
    rule_label : str
        Short label for the rule set (used in table caption).
    group_a_mask_fn : callable(df8, attr) -> np.ndarray[bool]
        Returns a boolean mask selecting group A rows from the 8-bit DataFrame.
    group_b_mask_fn : callable(df8, attr) -> np.ndarray[bool]
        Returns a boolean mask selecting group B rows.
    group_a_label, group_b_label : str
        Display labels for each group.

    Returns
    -------
    str
        Self-contained HTML <table> element (no Bokeh required).
    """
    _fmt_ci = _fmt_stat
    _overlap_cell = lambda lo1, hi1, lo2, hi2: (
        '<td style="color:#b55a00;">overlap</td>'
        if _ci_overlap(lo1, hi1, lo2, hi2)
        else '<td style="color:#1a6e1a;font-weight:600;">separated</td>'
    )

    metric_labels = {m: lbl for m, lbl, _ in _STRATUM_METRICS}

    rows_html = []
    for r in regions:
        short = REGION_NAMES.get(r, r).split("\u00b7")[0].strip()
        df8  = region_data[r]["df8"]
        attr = region_data[r].get("attr_meta", pd.DataFrame())
        mask_a = group_a_mask_fn(df8, attr)
        mask_b = group_b_mask_fn(df8, attr)
        n_a = int(mask_a.sum())
        n_b = int(mask_b.sum())

        first_row_in_region = True
        for metric, lbl, _ in _STRATUM_METRICS:
            if metric not in df8.columns:
                continue
            vals = df8[metric].to_numpy(dtype=float)
            m_a, lo_a, hi_a = _bootstrap_median_ci(vals[mask_a])
            m_b, lo_b, hi_b = _bootstrap_median_ci(vals[mask_b])

            region_cell = (
                f'<td rowspan="{len(_STRATUM_METRICS)}" '
                f'style="vertical-align:middle;font-weight:500;">'
                f'{short}</td>'
                if first_row_in_region else ""
            )
            rows_html.append(
                f"<tr>"
                f"{region_cell}"
                f"<td>{lbl}</td>"
                f'<td>{n_a}</td><td>{_fmt_ci(m_a, lo_a, hi_a)}</td>'
                f'<td>{n_b}</td><td>{_fmt_ci(m_b, lo_b, hi_b)}</td>'
                f'{_overlap_cell(lo_a, hi_a, lo_b, hi_b)}'
                f"</tr>"
            )
            first_row_in_region = False

    if not rows_html:
        return ""

    header = (
        f'<thead><tr>'
        f'<th>Region</th><th>Metric</th>'
        f'<th>N<sub>A</sub></th>'
        f'<th>{group_a_label}<br>median&nbsp;[95&thinsp;%&nbsp;CI]</th>'
        f'<th>N<sub>B</sub></th>'
        f'<th>{group_b_label}<br>median&nbsp;[95&thinsp;%&nbsp;CI]</th>'
        f'<th>CI separation</th>'
        f'</tr></thead>'
    )
    table_style = (
        'style="border-collapse:collapse;font-size:12.5px;'
        'font-family:\'EB Garamond\',Palatino,serif;margin-top:0.8em;'
        'max-width:820px;"'
    )
    cell_style = (
        '<style>'
        '.bstrap-tbl td, .bstrap-tbl th {'
        '  padding:3px 10px; border:1px solid #d0d0d0;'
        '  text-align:center; vertical-align:middle;'
        '}'
        '.bstrap-tbl th { background:#f4f4f4; font-weight:600; }'
        '.bstrap-tbl tr:nth-child(even) { background:#fafafa; }'
        '</style>'
    )
    caption = (
        f'<caption style="text-align:left;font-size:12px;color:#555;'
        f'padding-bottom:4px;">Bootstrap mean and 95&thinsp;% CI '
        f'({_N_BOOTSTRAP:,} resamples). {rule_label}.</caption>'
    )
    return (
        f'{cell_style}'
        f'<table class="bstrap-tbl" {table_style}>'
        f'{caption}{header}<tbody>'
        + "".join(rows_html)
        + '</tbody></table>'
    )


# ---------------------------------------------------------------------------
# Dot-whisker CI figure  (bootstrap median + 95 % CI, two groups per region)
# ---------------------------------------------------------------------------
def _fig_strat_dotwhisker(
    regions: list[str],
    region_data: dict,
    group_a_mask_fn,
    group_b_mask_fn,
    group_a_label: str,
    group_b_label: str,
    conc_data: "dict | None" = None,
) -> object:
    """Compact dot-whisker gridplot showing bootstrap median + 95 % CI for two groups.

    Layout: one row of N_metrics panels.  Each panel:
      - Y-axis: FactorRange of nested (region_short, group_label) categories.
        Group A is listed first (above) per region; Group B below.
      - X-axis: log-scale divergence value (range set from pooled data).
      - Filled circle  + horizontal segment = group A (within / symmetric).
      - Open circle (circle_open) + dashed segment = group B (outside / asymmetric).
      - Colour encodes region using the global _REGION_COLORS palette.
      - A thin dashed vertical grey line at x=median of the pooled population for reference.

    Designed for Tuftean minimalism: no grid lines, no border, no legend box
    (colour alone identifies regions; shape identifies the group).

    For Rule Set C the mask functions receive a df8 augmented with topN_mass
    from conc_data; callers should pass conc_data so that merge is applied here.
    """
    # Build y FactorRange: for each region, group A above group B.
    # Reversed so first region appears at the top of each panel.
    shorts = [REGION_NAMES.get(r, r).split("\u00b7")[0].strip() for r in regions]
    y_factors: list = []
    for s in reversed(shorts):
        y_factors.extend([(s, group_a_label), (s, group_b_label)])
    y_range = FactorRange(*y_factors)

    # Pre-compute per-region: augmented df8, attr, and masks.
    # The conc_data merge and mask evaluation are independent of metric, so
    # doing them here avoids repeating them inside the metric loop.
    _conc_col_local = f"top{_BIN_CONC_N_BINS}_mass"
    region_cache: dict = {}
    for r in regions:
        df8  = region_data[r]["df8"]
        attr = region_data[r].get("attr_meta", pd.DataFrame())
        if conc_data is not None:
            df8 = df8.copy()
            c_df = conc_data.get(r, pd.DataFrame())
            if not c_df.empty and _conc_col_local in c_df.columns:
                c8 = (
                    c_df[c_df["bitrate"] == REF_BITRATE][["station_id", _conc_col_local]]
                    if "bitrate" in c_df.columns
                    else c_df[["station_id", _conc_col_local]]
                )
                df8 = df8.merge(c8, on="station_id", how="left")
            else:
                df8[_conc_col_local] = np.nan
        mask_a = group_a_mask_fn(df8, attr)
        mask_b = group_b_mask_fn(df8, attr)
        region_cache[r] = dict(df8=df8, mask_a=mask_a, mask_b=mask_b)

    # Pre-compute bootstrap CIs for every (region, metric, group) combination.
    # Reused for both x-range determination and rendering, so each CI is
    # computed exactly once.
    ci_cache: dict = {}   # (r, metric, "a"|"b") -> (median, lo, hi)
    for r in regions:
        rc  = region_cache[r]
        df8 = rc["df8"]
        for metric, _, _ in _STRATUM_METRICS:
            if metric not in df8.columns:
                continue
            vals = df8[metric].to_numpy(float)
            ci_cache[(r, metric, "a")] = _bootstrap_stat_ci(vals[rc["mask_a"]])
            ci_cache[(r, metric, "b")] = _bootstrap_stat_ci(vals[rc["mask_b"]])

    panels = []
    for ii, (metric, x_label, x_rng_default) in enumerate(_STRATUM_METRICS):
        use_log = _STRATUM_LOG_SCALE.get(metric, False)
        # Derive x-range from the widest CI across all regions and groups.
        _ci_los = [ci_cache[(r, metric, g)][1] for r in regions
                   for g in ("a", "b") if (r, metric, g) in ci_cache
                   and np.isfinite(ci_cache[(r, metric, g)][1])
                   and ci_cache[(r, metric, g)][1] > 0]
        _ci_his = [ci_cache[(r, metric, g)][2] for r in regions
                   for g in ("a", "b") if (r, metric, g) in ci_cache
                   and np.isfinite(ci_cache[(r, metric, g)][2])
                   and ci_cache[(r, metric, g)][2] > 0]
        if _ci_los and _ci_his:
            if use_log:
                # half-decade buffer in log space
                x_lo = 10 ** (np.log10(min(_ci_los)) - 0.3)
                x_hi = 10 ** (np.log10(max(_ci_his)) + 0.3)
            else:
                buf  = (max(_ci_his) - min(_ci_los)) * 0.25
                x_lo = max(min(_ci_los) - buf, 0.0)
                x_hi = max(_ci_his) + buf
        else:
            x_lo, x_hi = x_rng_default

        n_rows = len(y_factors)
        _fig_kw: dict = dict(
            frame_width=180,
            frame_height=max(20 * n_rows + 80, 100),
            x_axis_label=x_label,
            y_range=y_range,
            x_range=Range1d(x_lo, x_hi),
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        if use_log:
            _fig_kw["x_axis_type"] = "log"
        fig = bk_figure(**_fig_kw)
        _apply_theme(fig)
        _apply_dotwhisker_axis_style(fig)
        if ii > 0:
            fig.yaxis.visible = False

        for r, s in zip(regions, shorts):
            if metric not in region_cache[r]["df8"].columns:
                continue
            color        = _REGION_COLORS.get(r, "#555555")
            rc           = region_cache[r]
            m_a, lo_a, hi_a = ci_cache[(r, metric, "a")]
            m_b, lo_b, hi_b = ci_cache[(r, metric, "b")]
            ya = (s, group_a_label)
            yb = (s, group_b_label)

            if np.isfinite(m_a) and np.isfinite(lo_a):
                fig.segment(
                    x0=[lo_a], x1=[hi_a], y0=[ya], y1=[ya],
                    line_color=color, line_width=2.0, line_alpha=0.9,
                )
                fig.scatter(
                    [m_a], [ya],
                    size=5, color=color, alpha=0.95,
                    marker="circle", line_color=color,
                )
            if np.isfinite(m_b) and np.isfinite(lo_b):
                fig.segment(
                    x0=[lo_b], x1=[hi_b], y0=[yb], y1=[yb],
                    line_color=color, line_width=2.0, line_alpha=0.9,
                    line_dash="solid",
                )
                fig.scatter(
                    [m_b], [yb],
                    size=5, color="#ffffff", alpha=1.0,
                    marker="circle", line_color=color, line_width=1.5,
                )

            # Separation annotation: thin line between medians + distance value
            if np.isfinite(m_a) and np.isfinite(m_b):
                gap = abs(m_b - m_a)
                # On log scale use geometric mean for x-midpoint of the label.
                if use_log and m_a > 0 and m_b > 0:
                    x_mid = float(np.sqrt(m_a * m_b))
                else:
                    x_mid = (m_a + m_b) / 2.0
                fig.segment(
                    x0=[m_a], x1=[m_b], y0=[s], y1=[s],
                    line_color="#5E5E5E", line_alpha=0.6, line_width=1.0,
                    line_dash="solid",
                )
                fig.scatter(
                    [m_a, m_b], [s, s],
                    marker="dash", size=8, angle=1.5708,
                    line_color="#5E5E5E", line_width=1.0, line_alpha=0.6,
                )
                fig.text(
                    x=[x_mid], y=[s],
                    text=[f"{gap:.3f}"],
                    text_align="center", text_baseline="middle",
                    background_fill_color="#ffffff", background_fill_alpha=0.9,
                    text_font_size="12px", text_color="#383838", text_alpha=0.8,
                )

        # Vertical grey reference line at the pooled sample median
        _all_vals = np.concatenate([
            region_cache[r]["df8"][metric].to_numpy(float)
            for r in regions if metric in region_cache[r]["df8"].columns
        ])
        _all_vals = _all_vals[np.isfinite(_all_vals)]
        if len(_all_vals) > 0:
            _pop_median = float(np.median(_all_vals))
            fig.add_layout(BkSpan(
                location=_pop_median, dimension="height",
                line_color="#aaaaaa", line_dash="dashed", line_width=1.0,
            ))

        # rotate x axis labels by 45° handled by _apply_dotwhisker_axis_style
        panels.append(fig)

    if not panels:
        return Div(text="<em>No data available for dot-whisker plot.</em>")
    return gridplot([panels], merge_tools=True, toolbar_location='above')


def _fig_divergence_by_skew_stratum(
    regions: list[str],
    region_data: dict,
) -> object:
    """Region × metric grid: ECDF of each divergence metric split by the FB applicability criterion.

    Rows = regions; columns = divergence metrics (_STRATUM_METRICS).

    Two series per panel:
      black solid  = in-group:  |γ₁| < _SKEW_THRESHOLD
                                AND (unimodal OR peak_sep_sigma ≤ _PEAK_SEP_THRESHOLD)
      black dashed = out-group: all valid stations that do not meet both criteria

    X-ranges linked within each column; Y-range shared across all panels.
    Per-row legend on the rightmost panel shows N counts for each group.
    """
    n_rows = len(regions)
    n_cols = len(_STRATUM_METRICS)
    grid: list[list] = []

    for i_r, r in enumerate(regions):
        short       = REGION_NAMES.get(r, r).split("\u00b7")[0].strip()
        is_last_row = (i_r == n_rows - 1)
        df8         = region_data[r]["df8"]
        attr        = region_data[r].get("attr_meta", pd.DataFrame())

        # --- Step 1: pull raw arrays (NaN where data are absent) ---------------
        skew = (
            df8["obs_skewness"].to_numpy(dtype=float)
            if "obs_skewness" in df8.columns
            else np.full(len(df8), np.nan)
        )
        ps = (
            df8["peak_sep_sigma"].to_numpy(dtype=float)
            if "peak_sep_sigma" in df8.columns
            else np.full(len(df8), np.nan)
            # NaN means unimodal (n_peaks < 2); treated as passing the separation test.
        )

        # --- Step 2: validity mask; rows usable for classification -------------
        valid = np.isfinite(skew)

        # --- Step 3: per-criterion boolean masks (applied only to valid rows) ---
        # Criterion A: skewness within Silverman's log-normal bound.
        skew_ok = np.abs(skew) < _SKEW_THRESHOLD

        # Criterion B: unimodal station OR bimodal-but-close peaks.
        #   peak_sep_sigma is NaN for unimodal stations → NaN rows pass automatically.
        ps_ok = np.isnan(ps) | (ps <= _PEAK_SEP_THRESHOLD)

        # --- Step 4: group assignment -------------------------------------------
        # "in-group": both criteria satisfied (these stations are FB-appropriate).
        blk_w = valid & skew_ok & ps_ok
        # "out-group": valid skewness data but at least one criterion fails.
        blk_o = valid & ~blk_w

        last_rends: dict = {}
        row_panels: list = []
        # Accumulate renderers across ALL columns so the per-row legend toggle
        # hides/shows a series in every panel simultaneously.
        row_all_rends: dict[str, list] = {k: [] for k in ("blk_w", "blk_o")}

        for ii, (metric, x_label, x_rng) in enumerate(_STRATUM_METRICS):
            show_y      = (ii == 0)
            is_last_col = (ii == n_cols - 1)

            fig = bk_figure(
                frame_width=_W_STRAT, frame_height=_H_STRAT,
                x_axis_label=x_label if is_last_row else "",
                y_axis_label="P(X \u2264 x)" if show_y else "",
                x_axis_type="log",
                x_range=Range1d(*x_rng),
                y_range=Range1d(0.0, 1.02),
                tools="pan,wheel_zoom,reset,save",
                min_border_top=4,
                min_border_bottom=30 if is_last_row else 6,
                min_border_left=40 if show_y else 6,
                min_border_right=4,
            )
            _apply_theme(fig)
            if not show_y:
                fig.yaxis.visible = False
            if not is_last_row:
                fig.xaxis.visible = False

            if metric in df8.columns:
                vals = df8[metric].to_numpy(dtype=float)

                def _ecdf(mask: np.ndarray, color: str, dash: str) -> "object | None":
                    v = np.sort(vals[np.isfinite(vals) & (vals > 0) & mask])
                    if len(v) == 0:
                        return None
                    e = np.arange(1, len(v) + 1) / len(v)
                    return fig.line(v, e, line_width=1.3, color=color, line_dash=dash, line_alpha=0.8)

                r_blk_w = _ecdf(blk_w, _COLOR_BLACK, "solid")
                r_blk_o = _ecdf(blk_o, _COLOR_BLACK, "dashed")

                for key, rend in [("blk_w", r_blk_w), ("blk_o", r_blk_o)]:
                    if rend is not None:
                        row_all_rends[key].append(rend)

                if is_last_col:
                    last_rends = {
                        "blk_w": (r_blk_w, int(np.sum(blk_w))),
                        "blk_o": (r_blk_o, int(np.sum(blk_o))),
                    }

            if i_r == 0:
                fig.title.text           = x_label.split(" (")[0]
                fig.title.text_font_size = "13px"
            if is_last_col:
                fig.add_layout(BkTitle(text=short, **_SIDE_TITLE_KW), "right")
            fig.grid.grid_line_alpha = 0.6
            row_panels.append(fig)

        # Per-row legend on rightmost panel; each item owns renderers from ALL columns
        # so click_policy="hide" toggles that series across the entire row.
        if last_rends:
            _n_blk_w = last_rends["blk_w"][1]
            _n_blk_o = last_rends["blk_o"][1]
            _legend_specs = [
                ("blk_w", f"\u03b3\u2081<{_SKEW_THRESHOLD} \u2009\u2229\u2009|\u0394\u03bc|<{_PEAK_SEP_THRESHOLD}\u03c3\n(n\u202f=\u202f{_n_blk_w})"),
                ("blk_o", f"others\n(n\u202f=\u202f{_n_blk_o})"),
            ]
            items = [
                LegendItem(label=lbl, renderers=row_all_rends[key])
                for key, lbl in _legend_specs
                if row_all_rends[key]
            ]
            if items:
                row_panels[-1].add_layout(
                    Legend(items=items, **_LEGEND_STYLE_KW),
                    "right",
                )

        grid.append(row_panels)

    if not grid:
        return Div(text="<em>No skewness data available.</em>")

    # Link x_ranges within each column
    for ii in range(n_cols):
        anchor_x = grid[0][ii].x_range
        for i_r in range(1, n_rows):
            grid[i_r][ii].x_range = anchor_x

    # Link all y_ranges to the top-left panel
    anchor_y = grid[0][0].y_range
    for i_r in range(n_rows):
        for ii in range(n_cols):
            if i_r == 0 and ii == 0:
                continue
            grid[i_r][ii].y_range = anchor_y

    return gridplot(grid, merge_tools=True, toolbar_location="above")


def _fig_divergence_by_spread_asymmetry_stratum(
    regions: list[str],
    region_data: dict,
) -> object:
    """Region x metric grid: ECDF of each divergence metric stratified by spread asymmetry.

    Restricted to bimodal stations: n_peaks >= 2.  Where the dip-test result is
    available (bimodal column present), the gate requires BOTH n_peaks >= 2 AND
    bimodal == 1 (dip p < 0.05) to reduce false positives from noisy KDE shoulders.
    Where bimodal is NaN for a station (e.g. insufficient sample size), n_peaks >= 2
    alone is used as a fallback.

    Two strata:
      black  (solid)  : spread_asymmetry <= _SPREAD_ASYM_THRESHOLD  (symmetric modes)
      black (dashed) : spread_asymmetry >  _SPREAD_ASYM_THRESHOLD  (asymmetric modes)

    Layout: same region x metric grid as _fig_divergence_by_skew_stratum.
    """
    n_rows = len(regions)
    n_cols = len(_STRATUM_METRICS)
    grid: list[list] = []

    for i_r, r in enumerate(regions):
        short       = REGION_NAMES.get(r, r).split("\u00b7")[0].strip()
        is_last_row = (i_r == n_rows - 1)
        df8         = region_data[r]["df8"]

        asym = (
            df8["spread_asymmetry"].to_numpy(dtype=float)
            if "spread_asymmetry" in df8.columns
            else np.full(len(df8), np.nan)
        )

        skew = (
            df8["obs_skewness"].to_numpy(dtype=float)
            if "obs_skewness" in df8.columns
            else np.full(len(df8), np.nan)
        )

        valid_skew = np.isfinite(skew)

        # Bimodal gate: where the dip test is available (bimodal not NaN) require
        # bimodal == 1 in addition to the KDE peak structure.  Where bimodal is NaN
        # for a station (sample too short for the dip test), fall back to the KDE
        # peak detector alone.  This avoids silently dropping stations that could
        # not be dip-tested.
        # Base population: all finite-skew stations, same as Rule Sets A/C/D.
        # Unimodal stations (NaN asym) pass the spread-asymmetry criterion by default.
        valid = valid_skew
        sym_w  = valid & (np.isnan(asym) | (asym <= _SPREAD_ASYM_THRESHOLD))
        sym_o  = valid & ~sym_w

        row_panels: list = []
        row_all_rends: dict[str, list] = {"sym_w": [], "sym_o": []}

        for ii, (metric, x_label, x_rng) in enumerate(_STRATUM_METRICS):
            show_y      = (ii == 0)
            is_last_col = (ii == n_cols - 1)

            fig = bk_figure(
                frame_width=_W_STRAT, frame_height=_H_STRAT,
                x_axis_label=x_label if is_last_row else "",
                y_axis_label="P(X \u2264 x)" if show_y else "",
                x_axis_type="log",
                x_range=Range1d(*x_rng),
                y_range=Range1d(0.0, 1.02),
                tools="pan,box_zoom,wheel_zoom,reset,save",
                min_border_top=4,
                min_border_bottom=30 if is_last_row else 6,
                min_border_left=40 if show_y else 6,
                min_border_right=4,
            )
            _apply_theme(fig)
            if not show_y:
                fig.yaxis.visible = False
            if not is_last_row:
                fig.xaxis.visible = False

            if metric in df8.columns:
                vals = df8[metric].to_numpy(dtype=float)

                def _ecdf(mask: np.ndarray, color: str, dash: str):
                    v = np.sort(vals[np.isfinite(vals) & (vals > 0) & mask])
                    if len(v) == 0:
                        return None
                    e = np.arange(1, len(v) + 1) / len(v)
                    return fig.line(v, e, line_width=1.3, color=color, line_dash=dash, line_alpha=0.8)

                r_sw = _ecdf(sym_w, _COLOR_BLACK,  "solid")
                r_so = _ecdf(sym_o, _COLOR_GREY, "dashed")
                if r_sw: row_all_rends["sym_w"].append(r_sw)
                if r_so: row_all_rends["sym_o"].append(r_so)

            if i_r == 0:
                fig.title.text           = x_label.split(" (")[0]
                fig.title.text_font_size = "13px"
            if is_last_col:
                fig.add_layout(BkTitle(text=short, **_SIDE_TITLE_KW), "right")

            fig.grid.grid_line_alpha = 0.6
            row_panels.append(fig)

        # Per-row legend on rightmost panel; each item owns renderers from ALL columns
        # so click_policy="hide" toggles that series across the entire row.
        n_sym  = int(sym_w.sum())
        n_asym = int(sym_o.sum())
        # f"\u03b3\u2081<{_SKEW_THRESHOLD} \u2009\u2229\u2009|\u0394\u03bc|<{_PEAK_SEP_THRESHOLD}\u03c3\n(n\u202f=\u202f{_n_blk_w})"),
        _legend_specs = [
            ("sym_w", f"unimodal or \u03c3-ratio \u2264 {_SPREAD_ASYM_THRESHOLD:.0f}\u00d7\n(n\u202f=\u202f{n_sym})"),
            ("sym_o", f"\u03c3-ratio > {_SPREAD_ASYM_THRESHOLD:.0f}\u00d7  (n\u202f=\u202f{n_asym})"),
        ]
        items = [
            LegendItem(label=lbl, renderers=row_all_rends[key])
            for key, lbl in _legend_specs
            if row_all_rends[key]
        ]
        if items:
            row_panels[-1].add_layout(
                Legend(items=items, **_LEGEND_STYLE_KW),
                "right",
            )

        # Link y-ranges within row
        if row_panels:
            anchor_y = row_panels[0].y_range
            for p in row_panels[1:]:
                p.y_range = anchor_y
        grid.append(row_panels)

    # Link x-ranges within each column
    for ii in range(n_cols):
        col_figs = [grid[ir][ii] for ir in range(n_rows) if grid[ir][ii] is not None]
        if col_figs:
            anchor_x = col_figs[0].x_range
            for f in col_figs[1:]:
                f.x_range = anchor_x

    return gridplot(grid, merge_tools=True, toolbar_location="above")


def _fig_divergence_by_bin_concentration_stratum(
    regions: list[str],
    region_data: dict,
    conc_data: "dict[str, pd.DataFrame]",
) -> object:
    """Region x metric grid: ECDF stratified by bin concentration rule.

    Rule: a station is flagged as 'concentrated' when >= 50 % of its total PMF
    mass (observed, 8-bit grid) falls into 5 or fewer bins.  This reflects the
    condition where the KDE is asked to smooth a near-Dirac comb.

    Two strata:
      black solid  : topN_mass < _BIN_CONC_THRESHOLD  (mass spread broadly, rule NOT triggered)
      black dashed : topN_mass >= _BIN_CONC_THRESHOLD  (rule triggered, concentrated)

    N and the threshold are controlled by the module-level constants
    _BIN_CONC_N_BINS and _BIN_CONC_THRESHOLD.

    Layout: same region x metric grid as other stratification figures.
    """
    n_rows = len(regions)
    n_cols = len(_STRATUM_METRICS)
    grid: list[list] = []

    for i_r, r in enumerate(regions):
        short       = REGION_NAMES.get(r, r).split("\u00b7")[0].strip()
        is_last_row = (i_r == n_rows - 1)
        df8         = region_data[r]["df8"]

        # Merge topN_mass from concentration cache (bitrate=REF_BITRATE) into df8.
        # N and threshold are set by _BIN_CONC_N_BINS / _BIN_CONC_THRESHOLD.
        _col = f"top{_BIN_CONC_N_BINS}_mass"
        conc_df = conc_data.get(r, pd.DataFrame())
        if not conc_df.empty and "bitrate" in conc_df.columns and _col in conc_df.columns:
            conc8 = conc_df[conc_df["bitrate"] == REF_BITRATE][["station_id", _col]].copy()
        elif not conc_df.empty and _col in conc_df.columns:
            conc8 = conc_df[["station_id", _col]].copy()
        else:
            conc8 = pd.DataFrame(columns=["station_id", _col])

        if "station_id" in df8.columns and not conc8.empty:
            merged = df8.merge(conc8, on="station_id", how="left")
        else:
            merged = df8.copy()
            merged[_col] = np.nan

        topN     = merged[_col].to_numpy(dtype=float)
        # Base population: all finite-skew stations, same as Rule Sets A/B/D.
        # Stations without concentration data pass by default (go into conc_o).
        _skew_c  = (df8["obs_skewness"].to_numpy(dtype=float)
                    if "obs_skewness" in df8.columns
                    else np.full(len(df8), np.nan))
        valid    = np.isfinite(_skew_c)
        conc_w   = valid & np.isfinite(topN) & (topN >= _BIN_CONC_THRESHOLD)   # concentrated
        conc_o   = valid & ~conc_w                                               # spread or no data

        row_panels: list = []
        row_all_rends: dict[str, list] = {"conc_w": [], "conc_o": []}

        for ii, (metric, x_label, x_rng) in enumerate(_STRATUM_METRICS):
            show_y      = (ii == 0)
            is_last_col = (ii == n_cols - 1)

            fig = bk_figure(
                frame_width=_W_STRAT, frame_height=_H_STRAT,
                x_axis_label=x_label if is_last_row else "",
                y_axis_label="P(X \u2264 x)" if show_y else "",
                x_axis_type="log",
                x_range=Range1d(*x_rng),
                y_range=Range1d(0.0, 1.02),
                tools="pan,box_zoom,wheel_zoom,reset,save",
                min_border_top=4,
                min_border_bottom=30 if is_last_row else 6,
                min_border_left=40 if show_y else 6,
                min_border_right=4,
            )
            _apply_theme(fig)
            if not show_y:
                fig.yaxis.visible = False
            if not is_last_row:
                fig.xaxis.visible = False

            if metric in merged.columns:
                vals = merged[metric].to_numpy(dtype=float)

                def _ecdf(mask: np.ndarray, color: str, dash: str):
                    v = np.sort(vals[np.isfinite(vals) & (vals > 0) & mask])
                    if len(v) == 0:
                        return None
                    e = np.arange(1, len(v) + 1) / len(v)
                    return fig.line(v, e, line_width=1.3, color=color, line_dash=dash, line_alpha=0.8)

                r_cw = _ecdf(conc_w, _COLOR_BLACK, "dashed")
                r_co = _ecdf(conc_o, _COLOR_BLACK, "solid")
                if r_cw: row_all_rends["conc_w"].append(r_cw)
                if r_co: row_all_rends["conc_o"].append(r_co)

            if i_r == 0:
                fig.title.text           = x_label.split(" (")[0]
                fig.title.text_font_size = "13px"
            if is_last_col:
                fig.add_layout(BkTitle(text=short, **_SIDE_TITLE_KW), "right")

            fig.grid.grid_line_alpha = 0.6
            row_panels.append(fig)

        n_conc  = int(conc_w.sum())
        n_sprd  = int(conc_o.sum())
        _legend_specs = [
            ("conc_o", f"top-{_BIN_CONC_N_BINS} mass < {_BIN_CONC_THRESHOLD}  (n\u202f=\u202f{n_sprd})"),
            ("conc_w", f"top-{_BIN_CONC_N_BINS} mass \u2265 {_BIN_CONC_THRESHOLD}  (n\u202f=\u202f{n_conc})"),
        ]
        items = [
            LegendItem(label=lbl, renderers=row_all_rends[key])
            for key, lbl in _legend_specs
            if row_all_rends[key]
        ]
        if items:
            row_panels[-1].add_layout(
                Legend(items=items, **_LEGEND_STYLE_KW),
                "right",
            )

        if row_panels:
            anchor_y = row_panels[0].y_range
            for p in row_panels[1:]:
                p.y_range = anchor_y
        grid.append(row_panels)

    for ii in range(n_cols):
        col_figs = [grid[ir][ii] for ir in range(n_rows) if grid[ir][ii] is not None]
        if col_figs:
            anchor_x = col_figs[0].x_range
            for f in col_figs[1:]:
                f.x_range = anchor_x

    if not grid:
        return Div(text="<em>No bin-concentration data available.</em>")
    return gridplot(grid, merge_tools=True, toolbar_location="above")


def _fig_divergence_by_combined_stratum(
    regions: list[str],
    region_data: dict,
    conc_data: "dict[str, pd.DataFrame] | None" = None,
) -> object:
    """Region x metric grid: ECDF stratified by a five-criterion combined gate.

    Group 1 (solid)   : all active criteria pass:
      |g| < _SKEW_THRESHOLD
      peak_sep_sigma <= _PEAK_SEP_THRESHOLD (or NaN = unimodal, passes by default)
      spread_asymmetry <= _SPREAD_ASYM_THRESHOLD (or NaN = unimodal, passes by default)
      top5_mass < 0.50  (bin-concentration rule; NaN passes by default)
      Optional (uncomment to enable):
        record_years > _RECORD_YEARS_THRESHOLD (or NaN passes by default)
        zero_flow_frac == 0 (NaN = unknown, fails)

    Group 2 (dashed) : at least one active criterion fails.

    Layout: same region x metric grid as other stratification figures.
    """
    n_rows = len(regions)
    n_cols = len(_STRATUM_METRICS)
    grid: list[list] = []

    for i_r, r in enumerate(regions):
        short       = REGION_NAMES.get(r, r).split("\u00b7")[0].strip()
        is_last_row = (i_r == n_rows - 1)
        df8         = region_data[r]["df8"]
        attr        = region_data[r].get("attr_meta", pd.DataFrame())

        rec = (
            df8["record_years"].to_numpy(dtype=float)
            if "record_years" in df8.columns
            else np.full(len(df8), np.nan)
        )

        skew = (
            df8["obs_skewness"].to_numpy(dtype=float)
            if "obs_skewness" in df8.columns
            else np.full(len(df8), np.nan)
        )
        ps = (
            df8["peak_sep_sigma"].to_numpy(dtype=float)
            if "peak_sep_sigma" in df8.columns
            else np.full(len(df8), np.nan)
        )
        asym = (
            df8["spread_asymmetry"].to_numpy(dtype=float)
            if "spread_asymmetry" in df8.columns
            else np.full(len(df8), np.nan)
        )
        if "zero_flow_frac" in attr.columns and "station_id" in df8.columns:
            zff = attr["zero_flow_frac"].reindex(df8["station_id"].values).to_numpy(dtype=float)
        else:
            zff = np.full(len(df8), np.nan)

        # Bin-concentration criterion: merge topN_mass from conc_data (bitrate=REF_BITRATE).
        # N and threshold are controlled by _BIN_CONC_N_BINS / _BIN_CONC_THRESHOLD.
        # NaN (unavailable) passes by default.
        _col = f"top{_BIN_CONC_N_BINS}_mass"
        if conc_data is not None:
            conc_df = conc_data.get(r, pd.DataFrame())
            if not conc_df.empty and "bitrate" in conc_df.columns and _col in conc_df.columns:
                conc8 = conc_df[conc_df["bitrate"] == REF_BITRATE][["station_id", _col]].copy()
            elif not conc_df.empty and _col in conc_df.columns:
                conc8 = conc_df[["station_id", _col]].copy()
            else:
                conc8 = pd.DataFrame(columns=["station_id", _col])
            if "station_id" in df8.columns and not conc8.empty:
                _merged = df8[["station_id"]].merge(conc8, on="station_id", how="left")
                topN = _merged[_col].to_numpy(dtype=float)
            else:
                topN = np.full(len(df8), np.nan)
        else:
            topN = np.full(len(df8), np.nan)
        # NaN = data unavailable = passes (cannot confirm violation)
        conc_ok = np.isnan(topN) | (topN < _BIN_CONC_THRESHOLD)

        # Criterion masks.  NaN on ps/asym = unimodal = passes those two criteria.
        # NaN on zff = unknown = fails the ZFF criterion (cannot confirm favourable).
        skew_ok = np.isfinite(skew) & (np.abs(skew) < _SKEW_THRESHOLD)
        ps_ok   = np.isnan(ps)   | (ps   <= _PEAK_SEP_THRESHOLD)
        asym_ok = np.isnan(asym) | (asym <= _SPREAD_ASYM_THRESHOLD)
        # Optional: record length gate; uncomment to require > _RECORD_YEARS_THRESHOLD years
        # NaN (unknown length) passes by default.
        # rl_ok = np.isnan(rec) | (rec > _RECORD_YEARS_THRESHOLD)
        # Optional: zero-flow gate; uncomment to require zero_flow_frac == 0.
        # NaN (unavailable) fails, since the favourable condition cannot be confirmed.
        # pzf_ok = np.isfinite(zff) & (zff == 0.0)

        valid   = np.isfinite(skew)
        fav     = valid & skew_ok & ps_ok & asym_ok #& conc_ok  # add & rl_ok or & pzf_ok to enable optional gates
        unfav   = valid & ~fav                                  # at least one criterion fails

        row_panels: list = []
        row_all_rends: dict[str, list] = {"fav": [], "unfav": []}

        for ii, (metric, x_label, x_rng) in enumerate(_STRATUM_METRICS):
            show_y      = (ii == 0)
            is_last_col = (ii == n_cols - 1)

            fig = bk_figure(
                frame_width=_W_STRAT, frame_height=_H_STRAT,
                x_axis_label=x_label if is_last_row else "",
                y_axis_label="P(X \u2264 x)" if show_y else "",
                x_axis_type="log",
                x_range=Range1d(*x_rng),
                y_range=Range1d(0.0, 1.02),
                tools="pan,wheel_zoom,reset,save",
                min_border_top=4,
                min_border_bottom=30 if is_last_row else 6,
                min_border_left=40 if show_y else 6,
                min_border_right=4,
            )
            _apply_theme(fig)
            if not show_y:
                fig.yaxis.visible = False
            if not is_last_row:
                fig.xaxis.visible = False

            if metric in df8.columns:
                vals = df8[metric].to_numpy(dtype=float)

                def _ecdf(mask: np.ndarray, color: str, dash: str):
                    v = np.sort(vals[np.isfinite(vals) & (vals > 0) & mask])
                    if len(v) == 0:
                        return None
                    e = np.arange(1, len(v) + 1) / len(v)
                    return fig.line(v, e, line_width=1.3, color=color, line_dash=dash, line_alpha=0.8)

                r_fav   = _ecdf(fav,   _COLOR_BLACK,  "solid")
                r_unfav = _ecdf(unfav, _COLOR_BLACK, "dashed")
                if r_fav:   row_all_rends["fav"].append(r_fav)
                if r_unfav: row_all_rends["unfav"].append(r_unfav)

            if i_r == 0:
                fig.title.text           = x_label.split(" (")[0]
                fig.title.text_font_size = "13px"
            if is_last_col:
                fig.add_layout(BkTitle(text=short, **_SIDE_TITLE_KW), "right")
            fig.grid.grid_line_alpha = 0.6
            row_panels.append(fig)

        # Per-row legend on rightmost panel; each item owns renderers from ALL columns
        # so click_policy="hide" toggles that series across the entire row.
        n_fav   = int(fav.sum())
        n_unfav = int(unfav.sum())
        _legend_specs = [
            ("fav",   f"all 5 pass\n(n\u202f=\u202f{n_fav})"),
            ("unfav", f"\u22651 fails\n(n\u202f=\u202f{n_unfav})"),
        ]
        items = [
            LegendItem(label=lbl, renderers=row_all_rends[key])
            for key, lbl in _legend_specs
            if row_all_rends[key]
        ]
        if items:
            row_panels[-1].add_layout(
                Legend(items=items, **_LEGEND_STYLE_KW),
                "right",
            )

        if row_panels:
            anchor_y = row_panels[0].y_range
            for p in row_panels[1:]:
                p.y_range = anchor_y
        grid.append(row_panels)

    # Link x-ranges within each column
    for ii in range(n_cols):
        col_figs = [grid[ir][ii] for ir in range(n_rows) if grid[ir][ii] is not None]
        if col_figs:
            anchor_x = col_figs[0].x_range
            for f in col_figs[1:]:
                f.x_range = anchor_x

    return gridplot(grid, merge_tools=True, toolbar_location="above")


def _fig_qshift_sensitivity_profile(
    regions: list[str],
    region_data: dict,
) -> "column":
    """Full-support implied quantile-shift profile: magnitude and direction.

    Two stacked panels per region, all sharing a common x-axis.

    Panel 1 (magnitude):
      Left y-axis  : relative implied shift % (log scale), capped at _Y_CAP.
      Right y-axis : absolute implied shift in L s^-1 km^-2 (log scale).
                     abs_shift[j] = x_uar[j] * rel_shift%[j] / 100

    Panel 2 (direction):
      Y-axis: cross-station median (and 95% band) of the signed CDF difference
              CDF_AB(x) - CDF_FB(x) (linear scale).  Positive values indicate
              that AB accumulates more probability mass below x than FB does,
              so the AB flow estimate at that exceedance probability is lower
              than the FB estimate.  Negative values indicate the AB estimate
              is higher.  A zero-crossing marks where the two estimators agree.

    Bins where f(x) -> 0 produce artefactually large shifts; values above
    _Y_CAP % on the relative axis are clipped and a red dashed line marks
    the cap.
    """
    description = Div(text="", width=_W_SUMMARY * 2 + 80, height=0)

    _Y_MIN_CLIP = 1e-2     # floor to keep log y-axis finite
    _Y_CAP      = 1000.0   # shifts above this are low-density artefacts
    _Y_MAX_VIEW = _Y_CAP * 3.0   # axis upper bound; cap line in lower third
    _ABS_RANGE_NAME = "abs_shift"

    x_range_ref = None
    panels: list = []

    for i_r, r in enumerate(sorted(regions)):
        prof = region_data[r].get("perbin_profile")
        if prof is None or prof.empty:
            continue
        if "p50_adp" not in prof.columns:
            continue

        color = _REGION_COLORS.get(r, Category10[10][i_r % 10])
        title = REGION_NAMES.get(r, r)
        log_x = prof["log_x"].to_numpy(dtype=float)
        x_uar = np.exp(log_x)   # nats -> UAR in L/s/km²

        # --- Relative (%) arrays, clipped ---
        med = np.clip(prof["p50_adp"].to_numpy(dtype=float),  _Y_MIN_CLIP, _Y_CAP)
        lo  = np.clip(prof["p2p5_adp"].to_numpy(dtype=float), _Y_MIN_CLIP, _Y_CAP)
        hi  = np.clip(prof["p97p5_adp"].to_numpy(dtype=float),_Y_MIN_CLIP, _Y_CAP)

        # --- Absolute shift arrays (L/s/km²) derived from clipped relative ---
        abs_med = x_uar * med / 100.0

        # Secondary axis range: span of the absolute median, with a floor
        _ABS_FLOOR = 1e-6
        abs_finite = abs_med[np.isfinite(abs_med) & (abs_med > 0)]
        abs_lo_rng = max(float(abs_finite.min()), _ABS_FLOOR) if abs_finite.size else _ABS_FLOOR
        abs_hi_rng = float(abs_finite.max()) * 3.0 if abs_finite.size else 1.0

        kwargs: dict = dict(
            frame_width=_W_SUMMARY * 2,
            frame_height=_H_SUMMARY,
            x_axis_label="",
            y_axis_label="Implied shift %",
            x_axis_type="log",
            y_axis_type="log",
            y_range=Range1d(_Y_MIN_CLIP, _Y_MAX_VIEW),
            extra_y_ranges={_ABS_RANGE_NAME: Range1d(abs_lo_rng, abs_hi_rng)},
            title=title,
            tools="pan,wheel_zoom,reset,save",
        )
        if x_range_ref is not None:
            kwargs["x_range"] = x_range_ref

        fig = bk_figure(**kwargs)
        _apply_theme(fig)
        fig.grid.grid_line_alpha = 0.5
        fig.xaxis.visible = False

        # Right (absolute) axis
        fig.add_layout(
            LogAxis(
                y_range_name=_ABS_RANGE_NAME,
                axis_label="Abs. shift (L s\u207b\u00b9 km\u207b\u00b2)",
            ),
            "right",
        )

        if x_range_ref is None:
            x_range_ref = fig.x_range

        # --- Left axis: band and median (relative %) ---
        valid = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(x_uar) & (x_uar > 0)
        if valid.any():
            fig.varea(
                x=x_uar[valid], y1=lo[valid], y2=hi[valid],
                fill_color=color, fill_alpha=0.20,
                legend_label="95% band",
            )

        fin_m = np.isfinite(med) & np.isfinite(x_uar) & (x_uar > 0)
        if fin_m.any():
            fig.line(x_uar[fin_m], med[fin_m], line_width=2.0,
                     color=color, line_alpha=0.9,
                     legend_label="Median (rel.)")

        # Red dashed cap line on relative axis
        fig.add_layout(BkSpan(
            location=_Y_CAP, dimension="width",
            line_color="crimson", line_width=1.5, line_dash="dashed",
        ))
        fig.line([np.nan], [np.nan], line_color="crimson", line_width=1.5,
                 line_dash="dashed", legend_label="1\u202f000\u2009% cap")

        # --- Right axis: absolute median (dashed) ---
        fin_abs = np.isfinite(abs_med) & np.isfinite(x_uar) & (x_uar > 0) & (abs_med > 0)
        if fin_abs.any():
            fig.line(
                x_uar[fin_abs], abs_med[fin_abs],
                line_width=1.5, color=color, line_dash="dotted", line_alpha=0.85,
                y_range_name=_ABS_RANGE_NAME,
                legend_label="Median (abs., right)",
            )

        _style_legend(fig, "top_right", "10px")
        fig.add_layout(fig.legend[0], "right")
        panels.append(fig)

        # --- Direction panel: signed median CDF(AB) - CDF(FB) ---
        if "p50_delta" not in prof.columns:
            fig.xaxis.visible = True   # no direction panel below; restore x-axis
            continue

        d_lo  = prof["p2p5_delta"].to_numpy(dtype=float)
        d_hi  = prof["p97p5_delta"].to_numpy(dtype=float)
        d_med = prof["p50_delta"].to_numpy(dtype=float)

        # Symmetric y-range from the widest band value, floored at +/-0.05
        _d_vals = np.concatenate([d_lo, d_hi])
        _d_vals = _d_vals[np.isfinite(_d_vals)]
        _y_half = float(np.abs(_d_vals).max()) * 1.25 if _d_vals.size else 0.5
        _y_half = max(_y_half, 0.05)

        dir_fig = bk_figure(
            frame_width=_W_SUMMARY * 2,
            frame_height=_H_SUMMARY,
            x_axis_label="UAR (L s\u207b\u00b9 km\u207b\u00b2)",
            y_axis_label="CDF(AB) \u2212 CDF(FB)",
            x_axis_type="log",
            y_range=Range1d(-_y_half, _y_half),
            x_range=x_range_ref,
            tools="pan,wheel_zoom,reset,save",
        )
        _apply_theme(dir_fig)
        dir_fig.grid.grid_line_alpha = 0.5

        # Zero reference: the line where both estimators agree
        dir_fig.add_layout(BkSpan(
            location=0.0, dimension="width",
            line_color="#666666", line_width=1.0, line_dash="solid",
        ))

        valid_d = (
            np.isfinite(d_lo) & np.isfinite(d_hi)
            & np.isfinite(x_uar) & (x_uar > 0)
        )
        if valid_d.any():
            dir_fig.varea(
                x=x_uar[valid_d], y1=d_lo[valid_d], y2=d_hi[valid_d],
                fill_color=color, fill_alpha=0.50,
                legend_label="95% band",
            )

        fin_d = np.isfinite(d_med) & np.isfinite(x_uar) & (x_uar > 0)
        if fin_d.any():
            dir_fig.line(
                x_uar[fin_d], d_med[fin_d],
                line_width=2.0, color=color, line_alpha=0.9,
                legend_label="Median delta",
            )

        _style_legend(dir_fig, "top_right", "10px")
        dir_fig.add_layout(dir_fig.legend[0], "right")
        panels.append(dir_fig)

    layout = [Div(text="<em>No quantile-shift profile data available.</em>")]
    if panels:
        fig_layout = gridplot(panels, ncols=1, merge_tools=True, toolbar_location="above")
        layout = column(description, fig_layout)

    return layout


def _fig_qshift_grid(
    regions: list[str],
    region_data: dict,
    x_lo: float,
    x_hi: float,
) -> object:
    """Three-panel quantile-shift summary.

    Left:   KS-point shift ECDF (lower bound), AB = dashed, FB = solid, one
            colour per region.  This is the shift implied at the single max-KS
            bin j* and is a lower bound on the worst-case shift across the
            full support.
    Centre: Max-over-support shift ECDF for the AB (adaptive) basis.  One
            ECDF per region.  Shows how much the KS-point bound is exceeded.
    Right:  Max-over-support shift ECDF for the FB (Silverman) basis.
    All three panels share the same x-axis range.
    """
    shared_range = Range1d(x_lo, x_hi)

    def _max_ecdf_panel(col: str, title: str, show_y: bool = False) -> "bk_figure":
        """ECDF of a max-shift column, one line per region."""
        fp = bk_figure(
            frame_width=_W_SUMMARY, frame_height=_H_SUMMARY,
            x_axis_label="Quantile shift %",
            y_axis_label="P(X \u2264 x)" if show_y else "",
            title=title,
            x_axis_type="log",
            x_range=shared_range,
            y_range=Range1d(0.0, 1.02),
            tools="pan,wheel_zoom,reset,save",
        )
        _apply_theme(fp)
        fp.grid.grid_line_alpha = 0.5
        fp.title.text_font_size = "12px"
        if not show_y:
            fp.yaxis.visible = False
        for i_r, r in enumerate(sorted(regions)[::-1]):
            df8   = region_data[r]["df8"]
            if col not in df8.columns:
                continue
            color = _REGION_COLORS.get(r, Category10[10][i_r % 10])
            short = REGION_NAMES.get(r, r).split("\u00b7")[0].strip()
            n_tot = len(df8)
            raw   = df8[col].to_numpy(dtype=float)
            vals  = np.sort(raw[np.isfinite(raw) & (raw > 0)])
            if len(vals) == 0:
                continue
            ecdf = np.arange(1, len(vals) + 1) / max(n_tot, 1)
            fp.line(vals, ecdf, line_width=1.8, color=color, line_alpha=0.9,
                    legend_label=short)
        fp.add_layout(BkSpan(location=0.5, dimension="width",
                             line_color="#999999", line_dash="dashed", line_width=1))
        _style_legend(fp, "bottom_right", "10px")
        fp.add_layout(fp.legend[0], "right")
        return fp

    # ---- Left panel: KS-point shift ECDF, AB=dashed, FB=solid ----
    f_ecdf = bk_figure(
        frame_width=_W_SUMMARY, frame_height=_H_SUMMARY,
        x_axis_label="Quantile shift %",
        y_axis_label="P(X \u2264 x)",
        title="KS-point shift (lower bound)",
        x_axis_type="log",
        x_range=shared_range,
        y_range=Range1d(0.0, 1.02),
        tools="pan,wheel_zoom,reset,save",
    )
    _apply_theme(f_ecdf)
    f_ecdf.grid.grid_line_alpha = 0.5
    f_ecdf.title.text_font_size = "12px"

    for i_r, r in enumerate(sorted(regions)[::-1]):
        df8   = region_data[r]["df8"]
        color = _REGION_COLORS.get(r, Category10[10][i_r % 10])
        short = REGION_NAMES.get(r, r).split("\u00b7")[0].strip()
        n_tot = len(df8)

        for e_col, dash, e_lbl in [
            ("ks_qshift_adp_pct", "dashed", "AB"),
            ("ks_qshift_sil_pct", "solid",  "FB"),
        ]:
            if e_col not in df8.columns:
                continue
            raw  = df8[e_col].to_numpy(dtype=float)
            vals = np.sort(raw[np.isfinite(raw) & (raw > 0)])
            if len(vals) == 0:
                continue
            ecdf = np.arange(1, len(vals) + 1) / max(n_tot, 1)
            lbl  = f"{short}" if e_lbl == "AB" else ""
            f_ecdf.line(vals, ecdf, line_width=1.8, color=color, line_dash=dash,
                        line_alpha=0.9, legend_label=lbl)

    # Style keys
    f_ecdf.line([np.nan], [np.nan], line_dash="dashed", color="#555555",
                line_width=1.8, legend_label="AB (dashed)")
    f_ecdf.line([np.nan], [np.nan], line_dash="solid",  color="#555555",
                line_width=1.8, legend_label="FB (solid)")
    f_ecdf.add_layout(BkSpan(location=0.5, dimension="width",
                             line_color="#999999", line_dash="dashed", line_width=1))

    # Vertical dotted line at the 90th percentile of pooled KS-point shifts
    _all_ks: list[np.ndarray] = []
    for _r in sorted(regions)[::-1]:
        _df8 = region_data[_r]["df8"]
        for _col in ("ks_qshift_adp_pct", "ks_qshift_sil_pct"):
            if _col not in _df8.columns:
                continue
            _raw = _df8[_col].to_numpy(dtype=float)
            _all_ks.append(_raw[np.isfinite(_raw) & (_raw > 0)])
    if _all_ks:
        _p90 = float(np.percentile(np.concatenate(_all_ks), 90))
        f_ecdf.add_layout(BkSpan(location=_p90, dimension="height",
                                 line_color="#999999", line_dash="dotted", line_width=1.2))

    _style_legend(f_ecdf, "bottom_right", "10px")
    f_ecdf.add_layout(f_ecdf.legend[0], "right")

    # ---- Right panel: max-over-support shift, AB density basis ----
    _has_max = any(
        "max_qshift_adp_pct" in region_data[r]["df8"].columns for r in regions
    )
    if _has_max:
        f_max_adp = _max_ecdf_panel("max_qshift_adp_pct", "Max shift over support (AB basis)")
        return row(f_ecdf, f_max_adp)
    return row(f_ecdf)

# ---------------------------------------------------------------------------
# Bin-concentration ECDF figure
# ---------------------------------------------------------------------------

def _fig_bin_concentration_sweep(
    regions: list[str],
    conc_data: "dict[str, pd.DataFrame]",
) -> object:
    """ECDF of per-station PMF mass concentration into the top-4 observed bins.

    One panel per bitrate (6, 8, 10 bits), one ECDF line per region.  The
    x-axis runs from 0 (mass completely spread across all bins, H0) to 1
    (all mass in four or fewer cells).  The y-axis is the standard cumulative
    fraction P(X ≤ x), matching the orientation of the skewness and
    spread-asymmetry ECDFs in sections 6.4.1 and 6.4.2.

    A HYSETS-specific quantization artifact produces a noticeably heavier
    right tail: stations whose observed flow record is so short, or whose
    annual runoff so strongly peaked at a particular value, that the 256-bin
    (8-bit) grid captures the bulk of their mass in just a handful of cells.
    This condition is not represented in the synthetic sweep.
    """
    # shared x range for all three panels

    panels = []
    for i_b, bitrate in enumerate(BITRATES):
        show_y = (i_b == 0)
        x_type = "log" if bitrate >= 8 else "linear"
        x_max = 2 if bitrate > 6 else 1.2
        x_range = Range1d(2e-2, x_max)
        fig = bk_figure(
            frame_width=_W_SUMMARY, frame_height=_H_SUMMARY,
            x_axis_label=f"Fraction of total mass in top-{_BIN_CONC_N_BINS} observed bins",
            y_axis_label="P(X \u2264 x)" if show_y else "",
            x_axis_type=x_type,
            title=f"{bitrate}-bit ({2**bitrate} bins)",
            tools="pan,wheel_zoom,reset,save",
            x_range=x_range,
        )
        _apply_theme(fig)
        fig.x_range = x_range

        for j, r in enumerate(regions):
            df_r = conc_data.get(r)
            if df_r is None:
                continue
            _col = f"top{_BIN_CONC_N_BINS}_mass"
            if _col not in df_r.columns:
                continue
            sub = df_r[df_r["bitrate"] == bitrate][_col].dropna().to_numpy(dtype=float)
            if len(sub) == 0:
                continue
            color = _REGION_COLORS.get(r, Category10[10][j % 10])
            short = REGION_NAMES.get(r, r).split("\u00b7")[0].strip()
            vals  = np.sort(sub)
            ecdf  = np.arange(1, len(vals) + 1) / len(vals)
            pct_hi = float(np.mean(sub > _BIN_CONC_THRESHOLD)) * 100
            label = f"{short} \n({pct_hi:.0f}% > {_BIN_CONC_THRESHOLD})"
            fig.line(vals, ecdf, line_width=2, color=color, legend_label=label)
            fig.scatter([float(np.median(vals))], [0.5], size=7,
                        color=color, marker="circle")

        # Reference threshold at 0.5
        fig.add_layout(BkSpan(
            location=0.5, dimension="height",
            line_color=_THRESHOLD_COLOR, line_dash="dashed", line_width=1.5,
        ))

        _style_legend(fig, "top_left", "12px")
        if not show_y:
            fig.yaxis.visible  = False
            # fig.legend.visible = False

        panels.append(fig)

    return gridplot([panels], merge_tools=True, toolbar_location="above")


# ---------------------------------------------------------------------------
# §6.4.7: Largest ISD within Silverman bounds
# ---------------------------------------------------------------------------

def _pmf_dir(region: str, bitrate: int) -> Path:
    """Path to the baseline-distribution directory for a region+bitrate."""
    return BASELINE_DIR / region / f"{bitrate:02d}_bits"


def _load_station_pmf(
    region: str,
    station_id: str,
    bitrate: int = 8,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None":
    """Return (log_x, pmf_obs, pmf_adp, pmf_sil) for one station or None."""
    base = _pmf_dir(region, bitrate)
    try:
        obs_df = pd.read_csv(base / "pmf_obs.csv",           index_col=0, usecols=["log_x_uar", station_id])
        adp_df = pd.read_csv(base / "pmf_kde_adaptive.csv",  index_col=0, usecols=["log_x_uar", station_id])
        sil_df = pd.read_csv(base / "pmf_kde_silverman.csv", index_col=0, usecols=["log_x_uar", station_id])
    except (FileNotFoundError, ValueError):
        return None
    log_x   = obs_df.index.to_numpy(dtype=float)
    pmf_obs = obs_df[station_id].to_numpy(dtype=float)
    pmf_adp = adp_df[station_id].to_numpy(dtype=float)
    pmf_sil = sil_df[station_id].to_numpy(dtype=float)
    return log_x, pmf_obs, pmf_adp, pmf_sil


def _fig_isd_paradox_panels(
    regions: list[str],
    region_data: dict,
    n_panels: int = 3,
) -> "object | None":
    """Row of *n_panels* density panels: the largest-ISD stations that pass
    all Silverman validity criteria (|γ₁| < 1.8 AND unimodal-or-separation ≤ 3σ).

    Each panel shows the empirical histogram, the AB KDE line, and the FB line,
    annotated with γ₁ and Δμ (or 'unimodal') plus PASS badges for each criterion.
    """
    # Pool 8-bit rows across all regions, keeping region info for PMF loading.
    frames: list[pd.DataFrame] = []
    for r in regions:
        df8 = region_data[r]["df8"].copy()
        df8["_region"] = r
        frames.append(df8)
    if not frames:
        return None
    all_df = pd.concat(frames, ignore_index=True)

    # Apply the Silverman within-bounds mask (same logic as _msk_a_in).
    skew = all_df["obs_skewness"].to_numpy(float) if "obs_skewness" in all_df.columns else np.full(len(all_df), np.nan)
    ps   = all_df["peak_sep_sigma"].to_numpy(float) if "peak_sep_sigma" in all_df.columns else np.full(len(all_df), np.nan)
    isd = all_df["isd"].to_numpy(float) if "isd" in all_df.columns else np.full(len(all_df), np.nan)
    pass_mask = np.isfinite(skew) & (np.abs(skew) < _SKEW_THRESHOLD) & (np.isnan(ps) | (ps <= _PEAK_SEP_THRESHOLD)) & np.isfinite(isd)
    passing = all_df[pass_mask].copy()
    if passing.empty:
        return None

    # Sort descending by ISD; deduplicate by station (keep highest ISD row).
    passing = (
        passing.sort_values("isd", ascending=False)
        .drop_duplicates("station_id")
        .head(n_panels)
        .reset_index(drop=True)
    )

    _BADGE_PASS  = "\u2713 PASS"    # ✓ PASS
    _BADGE_COL   = "#1a6e1a"        # green
    _ANNOT_FONT  = "12px"
    _ANNOT_ALPHA = 0.88

    panels: list[object] = []
    for idx, row in passing.iterrows():
        stn     = row["station_id"]
        region  = row["_region"]
        gamma1  = float(row["obs_skewness"])
        delta_mu_val = row.get("peak_sep_sigma", np.nan)
        delta_mu = float(delta_mu_val) if pd.notna(delta_mu_val) else np.nan
        n_peaks_val  = row.get("n_peaks", 1)
        n_peaks = int(n_peaks_val) if pd.notna(n_peaks_val) else 1
        isd_val = float(row["isd"])

        # Load PMF from source CSV.
        pmf_data = _load_station_pmf(region, stn, bitrate=REF_BITRATE)
        if pmf_data is None:
            continue
        log_x, pmf_obs, pmf_adp, pmf_sil = pmf_data

        dlog    = np.maximum(np.gradient(log_x), 1e-12)
        lin_x   = np.exp(log_x)
        half    = np.exp(dlog / 2.0)

        dens_obs = pmf_obs / dlog
        dens_adp = pmf_adp / dlog
        dens_sil = pmf_sil / dlog
        # Suppress bin-0 (zero-equivalent) in density
        dens_obs[0] = dens_adp[0] = dens_sil[0] = 0.0

        # X-range from AB CDF (skip bin 0)
        pmf_ref = pmf_adp.copy(); pmf_ref[0] = 0.0
        pmf_tot = pmf_ref.sum()
        if pmf_tot > 0:
            cum  = np.cumsum(pmf_ref) / pmf_tot
            lo_i = int(np.searchsorted(cum, 0.001))
            hi_i = int(np.searchsorted(cum, 0.999))
            x_min = lin_x[max(lo_i - 1, 1)] / 3.0
            x_max = lin_x[min(hi_i, len(lin_x) - 1)] * 3.0
        else:
            x_min, x_max = 0.1, 10000.0
        y_max = max(float(np.max(dens_adp)) * 1.10, 1e-6)

        yrs  = region_data[region]["rec_years"].get(stn, "?")
        da   = region_data[region]["drain_area"].get(stn, None)
        da_str = f", {da:,.0f} km\u00b2" if da is not None else ""

        title_str = f"{stn}  (N\u202f=\u202f{yrs}\u202fyr{da_str})"
        f = bk_figure(
            frame_width=_W_WORST, frame_height=_H_WORST,
            x_axis_type="log",
            x_axis_label="UAR (L/s/km\u00b2)",
            y_axis_label="Density" if idx == 0 else None,
            title=title_str,
            y_range=Range1d(0, y_max),
            x_range=Range1d(x_min, x_max),
            tools="pan,wheel_zoom,reset,save",
        )
        _apply_theme(f)
        f.title.text_font_size = "11px"
        if idx > 0:
            f.yaxis.visible = False

        f.quad(
            left=lin_x / half, right=lin_x * half,
            top=dens_obs, bottom=0,
            fill_color=_HIST_COLOR, line_color=None, alpha=0.85,
            legend_label="Empirical",
        )
        f.line(lin_x, dens_adp, line_width=2.0, color=_AB_COLOR, legend_label="AB")
        f.line(lin_x, dens_sil, line_width=2.0, color=_FB_COLOR, line_dash="dashed", legend_label="FB")

        # Criterion annotation (top-left corner, stacked lines)
        gamma_str   = f"\u03b3\u2081 = {gamma1:.2f}  {_BADGE_PASS}"
        if n_peaks >= 2 and np.isfinite(delta_mu):
            delta_str = f"\u0394\u03bc = {delta_mu:.2f}\u03c3  {_BADGE_PASS}"
        else:
            delta_str = f"unimodal  {_BADGE_PASS}"
        isd_str     = f"ISD = {isd_val:.2e}"

        # Stack three annotation labels inside the plot frame
        for line_idx, (ann_text, ann_color) in enumerate([
            (gamma_str,  _BADGE_COL),
            (delta_str,  _BADGE_COL),
            (isd_str,    "#333333"),
        ]):
            f.add_layout(Label(
                # set x to right align with the data window
                x=x_max * 0.95, 
                y=y_max * (0.97 - 0.12 * line_idx),
                x_units="data", y_units="data",
                text=ann_text,
                text_font=_BODY_FONT, text_font_size=_ANNOT_FONT,
                text_color=ann_color, text_alpha=_ANNOT_ALPHA,
                text_baseline="top",
                text_align="right",
            ))

        if idx == 0:
            _style_legend(f, "bottom_right", "12px")
        else:
            f.legend.visible = False

        panels.append(f)

    if not panels:
        return None
    return gridplot([panels], merge_tools=True, toolbar_location="right")


def _build_figures(
    regions: list[str],
    region_data: dict,
    oob_data: "dict[str, pd.DataFrame] | None" = None,
    conc_data: "dict[str, pd.DataFrame] | None" = None,
) -> dict[str, object]:
    """Return {name: gridplot} for each figure type."""

    # Build ECDF rows: one row per metric, x-ranges linked across regions within each row
    ecdf_rows = []
    for i, (col, x_label, _, x_rng_default) in enumerate(_ECDF_METRICS) :
        row_figs = [_fig_ecdf_region(region_data[regions[0]]["scores"], regions[0],
                                     metric=col, x_label=x_label,
                                     x_range_default=x_rng_default,
                                     show_title=(i == 0), show_legend=True)]
        for r in regions[1:]:
            row_figs.append(
                _fig_ecdf_region(region_data[r]["scores"], r,
                                 metric=col, x_label=x_label,
                                 x_range=row_figs[0].x_range,
                                 y_label=False, 
                                 x_range_default=x_rng_default,
                                 show_title=(i == 0))
            )
        ecdf_rows.append(row_figs)

    def _pick_unused(worst_index, metric_col, start_rank, used):
        """Return (station_id, rank) for the lowest rank >= start_rank not in used."""
        rank = start_rank
        while True:
            stn = worst_index.get((metric_col, rank))
            if stn is None:
                return None, rank
            if stn not in used:
                return stn, rank
            rank += 1

    # Worst panel grid: WORST_SELECTIONS rows x N_regions columns
    worst_rows = []
    used_stations: dict[str, set] = {r: set() for r in regions}
    for sel_idx, (metric_col, sel_rank) in enumerate(WORST_SELECTIONS):
        row = []
        for i, r in enumerate(regions):
            stn, rank = _pick_unused(
                region_data[r]["worst_index"], metric_col, sel_rank, used_stations[r]
            )
            if stn is None:
                row.append(None)
                continue
            used_stations[r].add(stn)
            wd      = region_data[r]["worst_df"]
            stn_pmf = wd[
                (wd["station_id"] == stn) &
                (wd["metric"]     == metric_col) &
                (wd["rank"]       == rank)
            ][["station_id", "log_x", "pmf_obs", "pmf_adp", "pmf_sil"]]
            row.append(_fig_worst_panel(
                stn_pmf,
                [stn],
                region_data[r]["score_map"],
                region_data[r]["rec_years"],
                region_data[r]["drain_area"],
                0,
                r,
                show_legend=((sel_idx == 0) and (i == 2)),  # only show legend in top-right panel
                show_y_label=(i % 3 == 0),
            ))
        worst_rows.append(row)

    # Median-5 grid: N_MEDIAN rows x N_regions columns
    median_rows = []
    for rank in range(N_MEDIAN):
        row = [
            _fig_worst_panel(
                region_data[r]["median_df"],
                [p[0] for p in region_data[r]["median5"]],
                region_data[r]["score_map"],
                region_data[r]["rec_years"],
                region_data[r]["drain_area"],
                rank,
                r,
                show_legend=((sel_idx == 0) and (i == 2)),  # only show legend in top-right panel
                show_y_label=(i % 3 == 0),
                median_labels=[p[1] for p in region_data[r]["median5"]],
            )
            for i, r in enumerate(regions)
        ]
        median_rows.append(row)

    # Station maps: one column per region, coloured by segmentation criterion (§6.4.1)
    map_figs = [
        _fig_station_map(
            r,
            region_data[r]["df8"],
            region_data[r]["coords"],
        )
        for r in regions
    ]

    # Quantile-shift ECDFs: one panel per density basis (ab, fb, obs).
    # Each panel shows one ECDF line per region using 8-bit scores only.
    # Build only when the new columns are present (i.e. after preprocess rerun).
    _has_qshift = all(
        _QSHIFT_METRICS[0][0] in region_data[r]["df8"].columns
        for r in regions
    )
    _qshift_grid = None
    if _has_qshift:
        # Shared x-range from the 1st-99th percentile pooled across all shift
        # columns (KS-point and max-over-support) so the three panels share an
        # axis that captures both the lower-bound and worst-case values.
        _shift_cols = [c for c, _ in _QSHIFT_METRICS] + [
            "max_qshift_adp_pct", "max_qshift_sil_pct",
        ]
        _all_qshift = np.concatenate([
            region_data[r]["df8"][c].to_numpy(dtype=float)
            for r in regions
            for c in _shift_cols
            if c in region_data[r]["df8"].columns
        ])
        _finite_pos = _all_qshift[np.isfinite(_all_qshift) & (_all_qshift > 0)]
        _x_lo = float(np.nanpercentile(_finite_pos, 1))  if len(_finite_pos) else 0.1
        _x_hi = float(np.nanpercentile(_finite_pos, 99)) if len(_finite_pos) else 1000.0
        _qshift_grid = _fig_qshift_grid(regions, region_data, _x_lo, _x_hi)

    out: dict = {
        "ecdfs":               gridplot(ecdf_rows,    merge_tools=True, toolbar_location="right"),
        "worst10":             gridplot(worst_rows,   merge_tools=True, toolbar_location="above"),
        "median5":             gridplot(median_rows,  merge_tools=True, toolbar_location="above"),
        "error_model":         _fig_error_model(),
        "duality":             _build_duality_diagram(),
        "maps":                gridplot([map_figs],   merge_tools=True, toolbar_location="above"),
        "attr_cdfs":           _fig_attr_cdfs(regions, region_data),
        "metric_corr_heatmaps": _fig_metric_corr_heatmaps(regions, region_data),
    }

    out.update(build_synthetic_section())
    if _qshift_grid is not None:
        out["qshift_combined"] = _qshift_grid
    if any(region_data[r].get("perbin_profile") is not None for r in regions):
        out["qshift_sensitivity"] = _fig_qshift_sensitivity_profile(regions, region_data)
    _has_skewness = all("obs_skewness" in region_data[r]["df8"].columns for r in regions)

    # Mask functions shared by ECDF stratum figures, dot-whisker figures, and tables.
    def _msk_a_in(df8, attr):
        skew = df8["obs_skewness"].to_numpy(float) if "obs_skewness" in df8.columns else np.full(len(df8), np.nan)
        ps   = df8["peak_sep_sigma"].to_numpy(float) if "peak_sep_sigma" in df8.columns else np.full(len(df8), np.nan)
        return np.isfinite(skew) & (np.abs(skew) < _SKEW_THRESHOLD) & (np.isnan(ps) | (ps <= _PEAK_SEP_THRESHOLD))
    def _msk_a_out(df8, attr):
        skew = df8["obs_skewness"].to_numpy(float) if "obs_skewness" in df8.columns else np.full(len(df8), np.nan)
        return np.isfinite(skew) & ~_msk_a_in(df8, attr)
    def _msk_b_in(df8, attr):
        skew = df8["obs_skewness"].to_numpy(float) if "obs_skewness" in df8.columns else np.full(len(df8), np.nan)
        asym = df8["spread_asymmetry"].to_numpy(float) if "spread_asymmetry" in df8.columns else np.full(len(df8), np.nan)
        return np.isfinite(skew) & (np.isnan(asym) | (asym <= _SPREAD_ASYM_THRESHOLD))
    def _msk_b_out(df8, attr):
        skew = df8["obs_skewness"].to_numpy(float) if "obs_skewness" in df8.columns else np.full(len(df8), np.nan)
        return np.isfinite(skew) & ~_msk_b_in(df8, attr)
    _conc_col_fig = f"top{_BIN_CONC_N_BINS}_mass"
    def _msk_c_in(df8, attr):
        skew = df8["obs_skewness"].to_numpy(float) if "obs_skewness" in df8.columns else np.full(len(df8), np.nan)
        topN = df8[_conc_col_fig].to_numpy(float) if _conc_col_fig in df8.columns else np.full(len(df8), np.nan)
        return np.isfinite(skew) & np.isfinite(topN) & (topN >= _BIN_CONC_THRESHOLD)
    def _msk_c_out(df8, attr):
        skew = df8["obs_skewness"].to_numpy(float) if "obs_skewness" in df8.columns else np.full(len(df8), np.nan)
        return np.isfinite(skew) & ~_msk_c_in(df8, attr)

    if _has_skewness:
        out["silverman_bounds"]           = _fig_silverman_bounds(regions, region_data)
        out["skew_stratified_divergence"] = _fig_divergence_by_skew_stratum(regions, region_data)
        out["dotwhisker_a"] = _fig_strat_dotwhisker(
            regions, region_data,
            group_a_mask_fn=_msk_a_in,
            group_b_mask_fn=_msk_a_out,
            group_a_label="within",
            group_b_label="outside",
        )
    _has_spread_asym = any(
        "spread_asymmetry" in region_data[r]["df8"].columns for r in regions
    )
    if _has_spread_asym:
        out["spread_asymmetry_stratified_divergence"] = _fig_divergence_by_spread_asymmetry_stratum(
            regions, region_data
        )
        out["dotwhisker_b"] = _fig_strat_dotwhisker(
            regions, region_data,
            group_a_mask_fn=_msk_b_in,
            group_b_mask_fn=_msk_b_out,
            group_a_label="sym.",
            group_b_label="asym.",
        )
    if _has_skewness:
        out["combined_stratum_divergence"] = _fig_divergence_by_combined_stratum(
            regions, region_data, conc_data if conc_data else None
        )
    if _has_skewness and all("isd" in region_data[r]["df8"].columns for r in regions):
        _isd_paradox = _fig_isd_paradox_panels(regions, region_data)
        if _isd_paradox is not None:
            out["isd_paradox"] = _isd_paradox
    _has_record_years = any("record_years" in region_data[r]["df8"].columns for r in regions)
    if _has_record_years:
        out["record_length_stability"] = _fig_record_length_stability(regions, region_data, metrics=_STRATUM_METRICS)
    if conc_data:
        out["bin_concentration"] = _fig_bin_concentration_sweep(regions, conc_data)
        out["bin_concentration_stratum"] = _fig_divergence_by_bin_concentration_stratum(regions, region_data, conc_data)
        out["dotwhisker_c"] = _fig_strat_dotwhisker(
            regions, region_data,
            group_a_mask_fn=_msk_c_out,
            group_b_mask_fn=_msk_c_in,
            group_a_label="spread",
            group_b_label="conc.",
            conc_data=conc_data,
        )
    # if oob_data:
    #     out["oob_ecdf"]      = _fig_oob_ecdf(oob_data, regions)
    #     out["oob_stability"] = _fig_oob_stability(oob_data, region_data, regions)
    return out


def _build_duality_diagram():
    """Two-panel geometric diagram: CDF space (energy distance) vs quantile space (W₂)."""
    # P: narrow, centred; Q: wider, shifted right, both in log-UAR space
    mu_p, sig_p = 0.0, 0.45
    mu_q, sig_q = 0.85, 0.75

    N = 400
    log_x = np.linspace(-2.0, 3.5, N)
    F_p   = sp_norm.cdf(log_x, loc=mu_p, scale=sig_p)
    F_q   = sp_norm.cdf(log_x, loc=mu_q, scale=sig_q)

    # Locate x-strip at the point of maximum CDF gap
    gap   = F_p - F_q
    i0    = int(np.argmax(gap))
    x0    = log_x[i0]
    Fp0, Fq0 = F_p[i0], F_q[i0]
    dx    = 0.30

    # Quantile curves
    p_vals = np.linspace(0.005, 0.995, N)
    Qp_inv = sp_norm.ppf(p_vals, loc=mu_p, scale=sig_p)
    Qq_inv = sp_norm.ppf(p_vals, loc=mu_q, scale=sig_q)

    # Corresponding probability level for the quantile-space strip
    p0     = float(sp_norm.cdf(x0, loc=mu_p, scale=sig_p))
    dp     = 0.06
    Qp0    = float(sp_norm.ppf(p0, loc=mu_p, scale=sig_p))
    Qq0    = float(sp_norm.ppf(p0, loc=mu_q, scale=sig_q))

    W, H = 310, 270
    STRIP_COLOR  = "#d05a10"
    STRIP_FILL   = "#e07030"
    FILL_COLOR   = "#9ad8f0"

    # ---- Left panel: CDF space ----------------------------------------
    cdf_fig = bk_figure(
        width=W, height=H,
        x_axis_label="log-UAR",
        y_axis_label="F(x)  [probability]",
        title="CDF space  \u2014  Cram\u00e9r / Energy",
        toolbar_location=None, tools="",
    )
    _apply_theme(cdf_fig)
    cdf_fig.title.text_font_size = "12px"
    cdf_fig.yaxis.bounds = (0, 1)

    # Shade enclosed area
    cdf_fig.patch(
        np.concatenate([log_x, log_x[::-1]]),
        np.concatenate([F_p,   F_q[::-1]]),
        fill_color=FILL_COLOR, fill_alpha=0.40, line_color=None,
    )
    cdf_fig.line(log_x, F_p, line_width=2.0, color=_AB_COLOR, legend_label="P")
    cdf_fig.line(log_x, F_q, line_width=2.0, color=_FB_COLOR,  legend_label="Q")

    # Representative strip
    cdf_fig.quad(
        left=x0 - dx/2, right=x0 + dx/2,
        top=Fp0, bottom=Fq0,
        fill_color=STRIP_FILL, fill_alpha=0.70,
        line_color=STRIP_COLOR, line_width=1.5,
    )
    cdf_fig.add_layout(Label(
        x=x0 + dx/2 + 0.12, y=(Fp0 + Fq0)/2 - 0.01,
        text="\u2195 \u0394F   (squared for Cram\u00e9r)",
        text_font="EB Garamond, serif", text_font_size="11px",
        text_color=STRIP_COLOR, x_units="data", y_units="data",
    ))
    _DIM_COLOR   = "#555555"
    _dim_y_cdf   = Fq0 - 0.07
    _tick_h_cdf  = 0.025
    cdf_fig.segment(
        x0=[x0 - dx/2], x1=[x0 + dx/2],
        y0=[_dim_y_cdf], y1=[_dim_y_cdf],
        line_color=_DIM_COLOR, line_width=1.2,
    )
    cdf_fig.segment(
        x0=[x0 - dx/2], x1=[x0 - dx/2],
        y0=[_dim_y_cdf - _tick_h_cdf/2], y1=[_dim_y_cdf + _tick_h_cdf/2],
        line_color=_DIM_COLOR, line_width=1.2,
    )
    cdf_fig.segment(
        x0=[x0 + dx/2], x1=[x0 + dx/2],
        y0=[_dim_y_cdf - _tick_h_cdf/2], y1=[_dim_y_cdf + _tick_h_cdf/2],
        line_color=_DIM_COLOR, line_width=1.2,
    )
    cdf_fig.add_layout(Label(
        x=x0, y=_dim_y_cdf - _tick_h_cdf - 0.015,
        text="dx",
        text_font="EB Garamond, serif", text_font_size="11px",
        text_color=_DIM_COLOR, text_align="center",
        x_units="data", y_units="data",
    ))
    _style_legend(cdf_fig, "top_left", "10px")

    # ---- Right panel: quantile space -------------------------------------
    q_fig = bk_figure(
        width=W, height=H,
        x_axis_label="Probability p",
        y_axis_label="Q\u207b\u00b9(p)  [log-UAR]",
        title="Quantile space  \u2014  W\u2082",
        toolbar_location=None, tools="",
        x_range=Range1d(0, 1),
    )
    _apply_theme(q_fig)
    q_fig.title.text_font_size = "12px"

    q_fig.patch(
        np.concatenate([p_vals, p_vals[::-1]]),
        np.concatenate([Qp_inv, Qq_inv[::-1]]),
        fill_color=FILL_COLOR, fill_alpha=0.40, line_color=None,
    )
    q_fig.line(p_vals, Qp_inv, line_width=2.0, color=_AB_COLOR, legend_label="P")
    q_fig.line(p_vals, Qq_inv, line_width=2.0, color=_FB_COLOR,  legend_label="Q")

    # Same strip, quantile space: same probability mass, axes transposed
    q_fig.quad(
        left=p0 - dp/2, right=p0 + dp/2,
        top=max(Qp0, Qq0), bottom=min(Qp0, Qq0),
        fill_color=STRIP_FILL, fill_alpha=0.70,
        line_color=STRIP_COLOR, line_width=1.5,
    )
    q_fig.add_layout(Label(
        x=p0 + dp/2 + 0.025, y=(Qp0 + Qq0)/2 - 0.05,
        text="\u2195 \u0394x   (squared for W\u2082)",
        text_font="EB Garamond, serif", text_font_size="11px",
        text_color=STRIP_COLOR, x_units="data", y_units="data",
    ))
    _dim_y_q   = min(Qp0, Qq0) - 0.18
    _tick_h_q  = 0.10
    q_fig.segment(
        x0=[p0 - dp/2], x1=[p0 + dp/2],
        y0=[_dim_y_q], y1=[_dim_y_q],
        line_color=_DIM_COLOR, line_width=1.2,
    )
    q_fig.segment(
        x0=[p0 - dp/2], x1=[p0 - dp/2],
        y0=[_dim_y_q - _tick_h_q/2], y1=[_dim_y_q + _tick_h_q/2],
        line_color=_DIM_COLOR, line_width=1.2,
    )
    q_fig.segment(
        x0=[p0 + dp/2], x1=[p0 + dp/2],
        y0=[_dim_y_q - _tick_h_q/2], y1=[_dim_y_q + _tick_h_q/2],
        line_color=_DIM_COLOR, line_width=1.2,
    )
    q_fig.add_layout(Label(
        x=p0, y=_dim_y_q - _tick_h_q/2 - 0.10,
        text="dp",
        text_font="EB Garamond, serif", text_font_size="11px",
        text_color=_DIM_COLOR, text_align="center",
        x_units="data", y_units="data",
    ))
    _style_legend(q_fig, "top_left", "10px")

    caption = Div(text="""
<div style="font-family:'EB Garamond',Palatino,serif; font-size:12.5px;
            max-width:640px; color:#333; line-height:1.6; margin-top:6px;">
  The <b>same shaded area</b> appears in both panels and equals W&#8321;
  (the region is merely sliced vertically on the left and horizontally on the right).
  For L2, squaring the strip <em>height</em> gives different results in each panel
  because height means probability in CDF space (&#916;F, dimensionless) and flow in
  quantile space (&#916;x, log-UAR units).
  The orange strips represent the same probability mass &#916;p at the same location;
  their aspect ratios differ by a factor of 1/f(x).
</div>
""", width=640)

    return column(row(cdf_fig, q_fig), caption)


# ---------------------------------------------------------------------------
# Build combined report
# ---------------------------------------------------------------------------
def _export_figures_as_png(figs_dict: dict, images_dir: Path) -> None:
    """Export each figure in figs_dict as a PNG file using Playwright via Bokeh."""
    try:
        from bokeh.io import export_png
    except ImportError as exc:
        print(f"  WARNING: bokeh.io.export_png not available ({exc}), skipping PNG export.")
        return
    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting {len(figs_dict)} figure(s) to {images_dir} ...")
    for name, fig in figs_dict.items():
        out_path = images_dir / f"{name}.png"
        print(f"  {name}.png ...", end=" ", flush=True)
        try:
            export_png(fig, filename=str(out_path))
            print("done")
        except Exception as exc:
            print(f"FAILED ({exc})")
    print("PNG export complete.")


def _load_all_region_data(regions: list[str]) -> dict:
    """Load all per-region cache files and return a populated region_data dict."""
    region_data: dict = {}
    for r in regions:
        cache_path = CACHE_DIR / r / f"{r}_kde_comparison.parquet"
        worst_path = CACHE_DIR / r / f"{r}_worst10_pmfs.parquet"
        if not cache_path.exists():
            print(f"  WARNING: cache missing for {r}, skipping.")
            continue
        df       = _apply_metric_transforms(pd.read_parquet(cache_path))
        worst_df = pd.read_parquet(worst_path) if worst_path.exists() else pd.DataFrame()
        df8         = df[df["bitrate"] == REF_BITRATE].copy()
        if not worst_df.empty and "metric" in worst_df.columns:
            worst_index: dict[tuple[str, int], str] = {
                (row.metric, row.rank): row.station_id
                for row in worst_df[["metric", "rank", "station_id"]]
                .drop_duplicates(["metric", "rank"])
                .itertuples(index=False)
            }
        else:
            worst_index = {}
        median_path = CACHE_DIR / r / f"{r}_median_pmfs.parquet"
        median_df   = pd.read_parquet(median_path) if median_path.exists() else pd.DataFrame()
        median5     = median_df[["station_id", "metric_group"]].drop_duplicates().values.tolist() if not median_df.empty else []
        profile_path = CACHE_DIR / r / f"{r}_perbin_qshift_profile.parquet"
        region_data[r] = {
            "scores":    df,
            "df8":       df8,
            "worst_df":  worst_df,
            "worst_index": worst_index,
            "median_df": median_df,
            "median5":   median5,
            "score_map": df8.set_index("station_id")[
                [c for c in ["ks_stat", "wasserstein", "energy_distance", "kl_divergence", "isd"]
                 if c in df8.columns]
            ].to_dict("index"),
            "rec_years": _load_record_years(r),
            "drain_area": _load_drain_area(r),
            "coords":    _load_station_coords(r),
            "attr_meta": _load_attr_meta(r),
            "perbin_profile": pd.read_parquet(profile_path) if profile_path.exists() else None,
        }
    return region_data


def _compute_report_metadata(active: list[str], region_data: dict) -> "tuple[dict, dict]":
    """Return (metadata, bimodality_stats) dicts for template context."""
    all_df8 = pd.concat([region_data[r]["df8"] for r in active], ignore_index=True)
    metadata = {
        "date":        date.today().isoformat(),
        "regions":     ", ".join(REGION_NAMES.get(r, r) for r in active),
        "n_stations":  int(all_df8["station_id"].nunique()),
        "median_ks":   float(all_df8["ks_stat"].median()),
        "max_ks":      float(all_df8["ks_stat"].max()),
        "median_enrg": float(all_df8["energy_distance"].median()),
        "max_enrg":    float(all_df8["energy_distance"].max()),
    }

    def _bm_stats(df: pd.DataFrame) -> dict:
        n_total = len(df)
        if n_total == 0:
            return {}
        np_arr  = df["n_peaks"].to_numpy(float) if "n_peaks" in df.columns else np.full(n_total, np.nan)
        bm_arr  = df["bimodal"].to_numpy(float) if "bimodal" in df.columns else np.full(n_total, np.nan)
        structural  = (np_arr >= 2) & np.isfinite(np_arr)
        dip_tested  = np.isfinite(bm_arr)
        dip_ok      = dip_tested & (bm_arr == 1)
        gate        = structural & (dip_ok | ~dip_tested)
        n_gate      = int(gate.sum())
        n_dip_avail = int(dip_tested.sum())
        n_dip_conf  = int(dip_ok.sum())
        asym_arr    = df["spread_asymmetry"].to_numpy(float) if "spread_asymmetry" in df.columns else np.full(n_total, np.nan)
        n_valid     = int((gate & np.isfinite(asym_arr)).sum())
        n_asym_out  = int((gate & np.isfinite(asym_arr) & (asym_arr > _SPREAD_ASYM_THRESHOLD)).sum())
        return {
            "n_total":     n_total,
            "n_gate":      n_gate,
            "pct_gate":    round(100.0 * n_gate / n_total, 1) if n_total else 0.0,
            "n_dip_avail": n_dip_avail,
            "n_dip_conf":  n_dip_conf,
            "n_valid":     n_valid,
            "n_asym_out":  n_asym_out,
            "pct_asym_out": round(100.0 * n_asym_out / n_valid, 1) if n_valid else 0.0,
        }

    bimodality_stats = {
        "pooled": _bm_stats(all_df8),
        "by_region": {r: _bm_stats(region_data[r]["df8"]) for r in active},
        "threshold": _SPREAD_ASYM_THRESHOLD,
    }
    return metadata, bimodality_stats


def _build_bootstrap_tables(
    active: list[str],
    region_data: dict,
    conc_data: "dict | None",
) -> "tuple[str, str, str]":
    """Return (bstrap_table_a, bstrap_table_b, bstrap_table_c) as HTML strings."""

    def _mask_a_in(df8: pd.DataFrame, attr: pd.DataFrame) -> np.ndarray:
        skew = df8["obs_skewness"].to_numpy(float) if "obs_skewness" in df8.columns else np.full(len(df8), np.nan)
        ps   = df8["peak_sep_sigma"].to_numpy(float) if "peak_sep_sigma" in df8.columns else np.full(len(df8), np.nan)
        valid = np.isfinite(skew)
        ps_ok = np.isnan(ps) | (ps <= _PEAK_SEP_THRESHOLD)
        return valid & (np.abs(skew) < _SKEW_THRESHOLD) & ps_ok

    def _mask_a_out(df8: pd.DataFrame, attr: pd.DataFrame) -> np.ndarray:
        return ~_mask_a_in(df8, attr) & np.isfinite(
            df8["obs_skewness"].to_numpy(float) if "obs_skewness" in df8.columns else np.full(len(df8), np.nan)
        )

    def _mask_b_in(df8: pd.DataFrame, attr: pd.DataFrame) -> np.ndarray:
        skew = df8["obs_skewness"].to_numpy(float) if "obs_skewness" in df8.columns else np.full(len(df8), np.nan)
        asym = df8["spread_asymmetry"].to_numpy(float) if "spread_asymmetry" in df8.columns else np.full(len(df8), np.nan)
        valid = np.isfinite(skew)
        return valid & (np.isnan(asym) | (asym <= _SPREAD_ASYM_THRESHOLD))

    def _mask_b_out(df8: pd.DataFrame, attr: pd.DataFrame) -> np.ndarray:
        return ~_mask_b_in(df8, attr) & np.isfinite(
            df8["obs_skewness"].to_numpy(float) if "obs_skewness" in df8.columns else np.full(len(df8), np.nan)
        )

    bstrap_table_a = _html_strat_bootstrap_table(
        active, region_data,
        rule_label=f"Rule Set A: \u03b3\u2081&lt;{_SKEW_THRESHOLD} and "
                   f"peak separation &le;&thinsp;{_PEAK_SEP_THRESHOLD:.0f}&sigma;",
        group_a_mask_fn=_mask_a_in,
        group_b_mask_fn=_mask_a_out,
        group_a_label="Within Silverman bounds",
        group_b_label="Outside Silverman bounds",
    ) if all("obs_skewness" in region_data[r]["df8"].columns for r in active) else ""

    bstrap_table_b = _html_strat_bootstrap_table(
        active, region_data,
        rule_label=f"Rule Set B: spread asymmetry &le;&thinsp;{_SPREAD_ASYM_THRESHOLD:.0f}&times;",
        group_a_mask_fn=_mask_b_in,
        group_b_mask_fn=_mask_b_out,
        group_a_label=f"Asymmetry &le;&thinsp;{_SPREAD_ASYM_THRESHOLD:.0f}&times;",
        group_b_label=f"Asymmetry &gt;&thinsp;{_SPREAD_ASYM_THRESHOLD:.0f}&times;",
    ) if any("spread_asymmetry" in region_data[r]["df8"].columns for r in active) else ""

    if not conc_data:
        return bstrap_table_a, bstrap_table_b, ""

    _conc_col = f"top{_BIN_CONC_N_BINS}_mass"
    _c_rows = []
    for r in active:
        short = REGION_NAMES.get(r, r).split("\u00b7")[0].strip()
        df8 = region_data[r]["df8"].copy()
        conc_df = conc_data.get(r, pd.DataFrame())
        if not conc_df.empty and _conc_col in conc_df.columns:
            conc8 = conc_df[conc_df["bitrate"] == REF_BITRATE][["station_id", _conc_col]] if "bitrate" in conc_df.columns else conc_df[["station_id", _conc_col]]
            df8 = df8.merge(conc8, on="station_id", how="left")
        else:
            df8[_conc_col] = np.nan
        skew = df8["obs_skewness"].to_numpy(float) if "obs_skewness" in df8.columns else np.full(len(df8), np.nan)
        valid = np.isfinite(skew)
        topN  = df8[_conc_col].to_numpy(float)
        mask_cw = valid & np.isfinite(topN) & (topN >= _BIN_CONC_THRESHOLD)
        mask_co = valid & ~mask_cw
        n_cw = int(mask_cw.sum())
        n_co = int(mask_co.sum())
        first = True
        for metric, lbl, _ in _STRATUM_METRICS:
            if metric not in df8.columns:
                continue
            vals = df8[metric].to_numpy(float)
            m_cw, lo_cw, hi_cw = _bootstrap_median_ci(vals[mask_cw])
            m_co, lo_co, hi_co = _bootstrap_median_ci(vals[mask_co])
            rc = (
                f'<td rowspan="{len(_STRATUM_METRICS)}" style="vertical-align:middle;font-weight:500;">{short}</td>'
                if first else ""
            )
            overlap = _ci_overlap(lo_cw, hi_cw, lo_co, hi_co)
            ov_cell = '<td style="color:#b55a00;">overlap</td>' if overlap else '<td style="color:#1a6e1a;font-weight:600;">separated</td>'
            _c_rows.append(
                f"<tr>{rc}<td>{lbl}</td>"
                f"<td>{n_co}</td><td>{_fmt_stat(m_co, lo_co, hi_co)}</td>"
                f"<td>{n_cw}</td><td>{_fmt_stat(m_cw, lo_cw, hi_cw)}</td>"
                f"{ov_cell}</tr>"
            )
            first = False

    _c_header = (
        "<thead><tr><th>Region</th><th>Metric</th>"
        f"<th>N<sub>A</sub></th>"
        f"<th>Spread (top-{_BIN_CONC_N_BINS}&nbsp;&lt;&thinsp;{_BIN_CONC_THRESHOLD})<br>median&nbsp;[95&thinsp;%&nbsp;CI]</th>"
        f"<th>N<sub>B</sub></th>"
        f"<th>Concentrated (top-{_BIN_CONC_N_BINS}&nbsp;&ge;&thinsp;{_BIN_CONC_THRESHOLD})<br>median&nbsp;[95&thinsp;%&nbsp;CI]</th>"
        "<th>CI separation</th></tr></thead>"
    )
    _c_style = (
        '<style>.bstrap-tbl td,.bstrap-tbl th{padding:3px 10px;border:1px solid #d0d0d0;'
        'text-align:center;vertical-align:middle;}'
        '.bstrap-tbl th{background:#f4f4f4;font-weight:600;}'
        '.bstrap-tbl tr:nth-child(even){background:#fafafa;}</style>'
    )
    bstrap_table_c = (
        f'{_c_style}<table class="bstrap-tbl" style="border-collapse:collapse;font-size:12.5px;'
        f'font-family:\'EB Garamond\',Palatino,serif;margin-top:0.8em;max-width:820px;">'
        f'<caption style="text-align:left;font-size:12px;color:#555;padding-bottom:4px;">'
        f'Bootstrap median and 95&thinsp;% CI ({_N_BOOTSTRAP:,} resamples). '
        f'Rule Set C: mass concentration into &le;&thinsp;{_BIN_CONC_N_BINS} bins &ge;&thinsp;{_BIN_CONC_THRESHOLD}.'
        f'</caption>{_c_header}<tbody>'
        + "".join(_c_rows) + "</tbody></table>"
    )
    return bstrap_table_a, bstrap_table_b, bstrap_table_c


def build_report(regions: list[str], save_png: bool = False) -> None:
    region_data = _load_all_region_data(regions)
    active = [r for r in regions if r in region_data]
    if not active:
        raise SystemExit("No valid region data found. Run preprocess.py first.")

    oob_data: dict[str, pd.DataFrame] = {}
    for r in active:
        oob_path = CACHE_DIR / f"{r}_oob_scores.parquet"
        if oob_path.exists():
            oob_data[r] = pd.read_parquet(oob_path)
            print(f"  Loaded OOB scores for {r}: {len(oob_data[r])} stations")
        else:
            print(f"  No OOB cache for {r} (run preprocess_oob.py to enable OOB section)")

    conc_data: dict[str, pd.DataFrame] = {}
    for r in active:
        conc_path = CACHE_DIR / r / f"{r}_bin_concentration.parquet"
        if conc_path.exists():
            conc_data[r] = pd.read_parquet(conc_path)
            print(f"  Loaded bin concentration for {r}: {len(conc_data[r])} rows")
        else:
            print(f"  No bin-concentration cache for {r} (run preprocess.py to enable)")

    figs_dict = _build_figures(active, region_data, oob_data if oob_data else None, conc_data if conc_data else None)

    if save_png:
        _export_figures_as_png(figs_dict, REPO_ROOT / "images")

    bokeh_script, sections = components(figs_dict)
    metadata, bimodality_stats = _compute_report_metadata(active, region_data)
    bstrap_table_a, bstrap_table_b, bstrap_table_c = _build_bootstrap_tables(
        active, region_data, conc_data if conc_data else None
    )

    context = {
        "metadata":          metadata,
        "bokeh_script":      bokeh_script,
        "sections":          sections,
        "has_worst10":       "worst10" in sections,
        "bimodality_stats":  bimodality_stats,
        "spread_asym_threshold": _SPREAD_ASYM_THRESHOLD,
        "bin_conc_n_bins":   _BIN_CONC_N_BINS,
        "bin_conc_threshold": _BIN_CONC_THRESHOLD,
        "bstrap_table_a":    bstrap_table_a,
        "bstrap_table_b":    bstrap_table_b,
        "bstrap_table_c":    bstrap_table_c,
    }

    env = Environment(
        loader=FileSystemLoader([str(SCRIPT_DIR), str(TEMPLATES_ROOT)]),
        autoescape=False,
    )
    html = env.get_template("templates/template.html").render(**context)
    out_path = REPO_ROOT / 'report' / "max_kde_diffs.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\nWrote: {out_path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Build the KDE comparison HTML report.",
    )
    parser.add_argument(
        "region",
        nargs="?",
        default=None,
        help="Region name, numeric index, or 'all' (default: all cached regions).",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        default=False,
        help=(
            "Export each figure as a PNG to the images/ directory. "
            "Requires Playwright (playwright install chromium)."
        ),
    )
    args = parser.parse_args()
    regions = _resolve_regions(args.region)
    build_report(regions, save_png=args.save_png)


if __name__ == "__main__":
    main()
