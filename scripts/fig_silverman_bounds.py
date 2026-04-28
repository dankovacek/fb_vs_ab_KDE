"""Silverman validity-bound characterization figures.

Two panels characterizing where the station sample falls relative to
Silverman's (1986) stated conditions for the rule-of-thumb bandwidth to be
MISE-adequate:

  1. Skewness ECDF (per region in log-UAR space, threshold at |s| = 1.8)
  2. Peak separation ECDF (stations with n_peaks >= 2; threshold at 3 sigma)

A separate function _fig_record_length_stability produces a four-panel row
showing all divergence metrics stratified by short vs. long record length.

Called from build_report.py via:

    from fig_silverman_bounds import _fig_silverman_bounds, _fig_record_length_stability
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bokeh.layouts import column, gridplot
from bokeh.models import (
    Div,
    Label,
    Legend,
    LegendItem,
    Range1d,
    Span as BkSpan,
)
from bokeh.palettes import Category10
from bokeh.plotting import figure as bk_figure

from _plot_helpers import (
    _apply_theme, _style_legend, _BODY_FONT,
    REGION_NAMES, _REGION_COLORS,
)

# Silverman (1986 / 2018 ed.) stated validity thresholds
_SKEW_THRESHOLD        = 1.8   # log-normal skewness bound (Silverman 3.31, p. 45)
_PEAK_SEP_THRESHOLD    = 3.0   # normal-mixture separation bound (in units of sigma)
_SPREAD_ASYM_THRESHOLD = 3.0     # mode spread-asymmetry threshold (sigma ratio)

# Annotation style constants
_THRESHOLD_COLOR      = "#cc4400"
_THRESHOLD_LABEL_SIZE = "14px"
_NO_DATA_LABEL_SIZE   = "13px"

_W = 320   # panel frame width
_H = 220   # panel frame height


# ---------------------------------------------------------------------------
# Shared low-level helpers
# ---------------------------------------------------------------------------

def _ecdf(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_vals, cumulative_probability) for *vals*."""
    s = np.sort(vals)
    return s, np.arange(1, len(s) + 1) / len(s)


def _add_vthreshold(fig, location: float, label_text: str | None = None) -> None:
    """Add a vertical dashed threshold span to *fig*, with an optional label."""
    fig.add_layout(BkSpan(
        location=location, dimension="height",
        line_color=_THRESHOLD_COLOR, line_dash="dashed", line_width=1.5,
    ))
    if label_text is not None:
        fig.add_layout(Label(
            x=location, y=0.02,
            x_units="data", y_units="data",
            text=label_text,
            text_font=_BODY_FONT, text_font_size=_THRESHOLD_LABEL_SIZE,
            text_color=_THRESHOLD_COLOR, x_offset=4,
        ))


def _make_ecdf_fig(x_label: str, title: str, **kwargs) -> object:
    """Return a Bokeh figure pre-configured for an ECDF panel."""
    return bk_figure(
        frame_width=_W, frame_height=_H,
        x_axis_label=x_label,
        y_axis_label="P(X \u2264 x)",
        title=title,
        tools="pan,wheel_zoom,reset,save",
        **kwargs,
    )


def _draw_ecdf_lines(fig, regions, region_data, get_vals, threshold, make_label, exc_fn=None):
    """Draw one ECDF line per region onto *fig*.

    get_vals(df8) -> np.ndarray | None  -- extract values; return None to skip region.
    make_label(short, pct_exc, n) -> str -- format the legend label.
    exc_fn(vals, threshold) -> bool arr -- defaults to ``vals > threshold``.
    """
    if exc_fn is None:
        exc_fn = lambda v, t: v > t  # noqa: E731
    for i, r in enumerate(regions):
        vals = get_vals(region_data[r]["df8"])
        if vals is None or len(vals) == 0:
            continue
        color = _REGION_COLORS.get(r, Category10[10][i % 10])
        short = REGION_NAMES.get(r, r).split("\u00b7")[0].strip()
        vals_s, ecdf = _ecdf(vals)
        n = len(vals)
        pct_exc = int(np.sum(exc_fn(vals, threshold))) / n * 100
        fig.line(vals_s, ecdf, line_width=2, color=color,
                 legend_label=make_label(short, pct_exc, n))


def _fig_silverman_bounds(regions: list[str], region_data: dict) -> object:
    """Return a side-by-side row of three panels characterizing Silverman validity bounds.

    Panel A: skewness ECDF per region with |s|=1.8 threshold.
    Panel B: peak separation ECDF per region (stations with n_peaks>=2) with 3sigma threshold.
    Panel C: spread-asymmetry ECDF per region (stations with n_peaks>=2).

    Uses df8 (8-bit reference bitrate) from each region's region_data entry.
    Expects columns: obs_skewness, n_peaks, peak_sep_sigma, spread_asymmetry.
    """
    # Collect 8-bit rows across all regions
    frames: list[pd.DataFrame] = []
    for i, r in enumerate(regions):
        df8 = region_data[r]["df8"].copy()
        df8["_color"] = _REGION_COLORS.get(r, Category10[10][i % 10])
        df8["_short"] = REGION_NAMES.get(r, r).split("\u00b7")[0].strip()
        frames.append(df8)

    if not frames:
        return Div(text="<em>No data available for Silverman bounds figure.</em>")

    all_df = pd.concat(frames, ignore_index=True)
    _multi = (
        all_df[all_df["n_peaks"] >= 2]
        if "n_peaks" in all_df.columns
        else pd.DataFrame()
    )
    n_multi = len(_multi)

    # ------------------------------------------------------------------
    # Panel A: Skewness ECDF per region
    # ------------------------------------------------------------------
    fig_a = _make_ecdf_fig(
        "Skewness of log-UAR distribution",
        "Skewness distribution (log-UAR space)",
        x_range=Range1d(-7.5, 7.5),
    )
    _apply_theme(fig_a)
    _draw_ecdf_lines(
        fig_a, regions, region_data,
        get_vals=lambda df8: (
            df8["obs_skewness"].dropna().to_numpy(dtype=float)
            if "obs_skewness" in df8.columns else None
        ),
        threshold=_SKEW_THRESHOLD,
        make_label=lambda s, p, n: f"{s} (n={n})\n   ({p:.0f}% >|1.8|)",
        exc_fn=lambda v, t: np.abs(v) > t,
    )
    _add_vthreshold(fig_a, _SKEW_THRESHOLD, "FB skewness\nbound (|s|> 1.8)  \u2192")
    _add_vthreshold(fig_a, -_SKEW_THRESHOLD)
    _style_legend(fig_a, "top_left")
    fig_a.legend.visible = True
    fig_a.grid.grid_line_alpha = 0.4

    # ------------------------------------------------------------------
    # Panel B: Peak separation ECDF (n_peaks >= 2 stations only)
    # ------------------------------------------------------------------
    fig_b = _make_ecdf_fig(
        "Mode separation (component-\u03c3 units, log-UAR)",
        "Mode separation distribution",
    )
    _apply_theme(fig_b)

    if n_multi > 0:
        _draw_ecdf_lines(
            fig_b, regions, region_data,
            get_vals=lambda df8: (
                df8[df8["n_peaks"] >= 2]["peak_sep_sigma"].dropna().to_numpy(dtype=float)
                if "n_peaks" in df8.columns and "peak_sep_sigma" in df8.columns else None
            ),
            threshold=_PEAK_SEP_THRESHOLD,
            make_label=lambda s, p, n: f"{s} (n={n})\n   ({p:.0f}% of bimodal >3\u03c3)",
        )
        _style_legend(fig_b, "top_right")
        fig_b.legend.visible = True
        fig_b.yaxis.visible = False
    else:
        fig_b.add_layout(Label(
            x=0.5, y=0.5, x_units="screen", y_units="screen",
            text="No stations with \u22652 detected peaks",
            text_font=_BODY_FONT, text_font_size=_NO_DATA_LABEL_SIZE,
            text_align="center", text_color="#999999",
        ))

    _sep_max = float(_multi["peak_sep_sigma"].max()) if n_multi > 0 else _PEAK_SEP_THRESHOLD * 1.5
    fig_b.x_range = Range1d(0.0, max(_sep_max * 1.2, _PEAK_SEP_THRESHOLD * 1.5))
    _add_vthreshold(fig_b, _PEAK_SEP_THRESHOLD, "FB separation\nbound (3\u03c3) \u2192")
    fig_b.grid.grid_line_alpha = 0.4

    # ------------------------------------------------------------------
    # Panel C: Spread-asymmetry ECDF (n_peaks >= 2 stations, bimodal gate)
    # ------------------------------------------------------------------
    fig_c = _make_ecdf_fig(
        "Spread asymmetry (\u03c3_left / \u03c3_right, larger/smaller)",
        "Mode spread asymmetry",
        x_axis_type="log",
    )
    _apply_theme(fig_c)

    if n_multi > 0 and "spread_asymmetry" in all_df.columns:
        _draw_ecdf_lines(
            fig_c, regions, region_data,
            get_vals=lambda df8: (
                df8[df8["n_peaks"] >= 2]["spread_asymmetry"].dropna().to_numpy(dtype=float)
                if "n_peaks" in df8.columns and "spread_asymmetry" in df8.columns else None
            ),
            threshold=_SPREAD_ASYM_THRESHOLD,
            make_label=lambda s, p, n: f"{s} (n={n})\n   ({p:.0f}% of bimodal >{_SPREAD_ASYM_THRESHOLD:.0f}\u00d7)",
        )
        _style_legend(fig_c, "top_right")
        fig_c.legend.visible = True
        _asym_max = (
            float(_multi["spread_asymmetry"].max())
            if "spread_asymmetry" in _multi.columns
            else _SPREAD_ASYM_THRESHOLD * 2.0
        )
        fig_c.x_range = Range1d(1.0, max(_asym_max * 1.1, _SPREAD_ASYM_THRESHOLD * 2.0))
    else:
        fig_c.add_layout(Label(
            x=0.5, y=0.5, x_units="screen", y_units="screen",
            text="No stations with \u22652 detected peaks or spread_asymmetry not yet computed",
            text_font=_BODY_FONT, text_font_size=_NO_DATA_LABEL_SIZE,
            text_align="center", text_color="#999999",
        ))
        fig_c.x_range = Range1d(1.0, _SPREAD_ASYM_THRESHOLD * 2.0)

    _add_vthreshold(
        fig_c, _SPREAD_ASYM_THRESHOLD,
        f"Asymmetry threshold\n({_SPREAD_ASYM_THRESHOLD:.0f}\u00d7) \u2192",
    )
    fig_c.grid.grid_line_alpha = 0.4
    fig_c.yaxis.visible = False
    return gridplot([[fig_a, fig_b, fig_c]], merge_tools=True, toolbar_location='above')


# ---------------------------------------------------------------------------
# Record-length stability figure
# ---------------------------------------------------------------------------

_W_RL = 280
_H_RL = 240


def _fig_record_length_stability(
    regions: list[str],
    region_data: dict,
    metrics: list[tuple[str, str, tuple[float, float]]] | None = None,
) -> object:
    """Four-panel row: ECDF of each divergence metric split by record length.

    Split at the global median record_years across all stations.
    Dashed lines = short records (<= median); solid lines = long records (> median).
    One line per region per group. Y-axis shown on leftmost panel only.
    A single shared legend on the right of the rightmost panel toggles series
    across all four panels simultaneously via click_policy='hide'.
    """
    # Compute global median record length
    all_rl: list[float] = []
    for r in regions:
        df8 = region_data[r]["df8"]
        if "record_years" in df8.columns:
            all_rl.extend(df8["record_years"].dropna().tolist())

    if all_rl:
        rl_split = float(np.median(all_rl))
        rl_label = f"{int(round(rl_split))}\u202fyr (median)"
    else:
        rl_split = 25.0
        rl_label = "25\u202fyr"

    _DASH_MAP  = {"short": "dashed", "long": "solid"}
    _GROUP_LBL = {
        "short": f"\u2264{int(round(rl_split))}\u202fyr",
        "long":  f">{int(round(rl_split))}\u202fyr",
    }

    _metrics = metrics if metrics is not None else []

    # Accumulate renderers by label to build a shared cross-panel legend
    renderers_by_label: dict[str, list] = {}

    panels: list = []
    for ii, (metric, x_label, x_range) in enumerate(_metrics):
        show_y = (ii == 0)
        fig = bk_figure(
            frame_width=_W_RL-100, frame_height=_H_RL-80,
            x_axis_label=x_label,
            x_range=Range1d(*x_range),
            y_range=Range1d(0, 1),
            y_axis_label="P(X \u2264 x)" if show_y else "",
            x_axis_type="log",
            title=f"Record-length split ({rl_label})" if ii == 0 else "",
            tools="pan,wheel_zoom,reset,save",
            min_border_top=4,
            min_border_bottom=55,
            min_border_left=40 if show_y else 6,
            min_border_right=4,
        )
        _apply_theme(fig)
        if not show_y:
            fig.yaxis.visible = False

        for i, r in enumerate(regions):
            df8 = region_data[r]["df8"]
            if "record_years" not in df8.columns or metric not in df8.columns:
                continue
            color = _REGION_COLORS.get(r, Category10[10][i % 10])
            short = REGION_NAMES.get(r, r).split("\u00b7")[0].strip()
            for group, mask_cond in [
                ("short", df8["record_years"].notna() & (df8["record_years"] <= rl_split)),
                ("long",  df8["record_years"].notna() & (df8["record_years"] >  rl_split)),
            ]:
                sub = df8.loc[mask_cond & (df8[metric] > 0), metric].dropna()
                if len(sub) < 3:
                    continue
                vals_s = np.sort(sub.to_numpy(dtype=float))
                ecdf   = np.arange(1, len(vals_s) + 1) / len(vals_s)
                lbl    = f"{short} {_GROUP_LBL[group]}"
                renderer = fig.line(vals_s, ecdf, line_width=1.8, color=color,
                                    line_dash=_DASH_MAP[group])
                if lbl not in renderers_by_label:
                    renderers_by_label[lbl] = []
                renderers_by_label[lbl].append(renderer)

        fig.add_layout(BkSpan(
            location=0.5, dimension="width",
            line_color="#999999", line_dash="dashed", line_width=1,
        ))
        # Suppress any auto-generated per-panel legend (guard: panel may have no data)
        if fig.legend:
            fig.legend.visible = False
        panels.append(fig)

    if not panels:
        return Div(text="<em>No record-length data available.</em>")

    # Single shared legend on the rightmost panel; click_policy='hide' sets
    # visible on all referenced renderers across all panels
    if renderers_by_label:
        legend_items = [
            LegendItem(label=lbl, renderers=rends)
            for lbl, rends in renderers_by_label.items()
        ]
        shared_legend = Legend(
            items=legend_items,
            click_policy="hide",
            label_text_font=_BODY_FONT,
            label_text_font_size="10px",  # intentionally smaller: denser legend
            background_fill_alpha=0.7,
        )
        panels[-1].add_layout(shared_legend, "right")

    return gridplot([panels], merge_tools=True, toolbar_location="above")
