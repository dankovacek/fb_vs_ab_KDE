"""Synthetic validation of adaptive vs. Silverman KDE bandwidth.

Pre-stated hypothesis
---------------------
Silverman's rule-of-thumb bandwidth is MISE-suboptimal when either of the two
conditions stated in Silverman (1986) is violated:

  (1) Log-space skewness |s| > 1.8
  (2) Mixture-component separation > 3\u03c3

Under condition violation, the adaptive estimator achieves lower MISE against the
known true distribution.  Where neither condition is violated, both estimators
agree (null).

Primary metric: MISE.  Silverman's rule is derived to minimise MISE, so this is
the pre-stated comparison criterion.  Secondary metrics (KS, W\u2081, ED\u00b2, KL) are
reported as diagnostics only and were not used to select scenarios.

Design
------
The three sweep analyses each vary one parameter from a null endpoint (where both
conditions are satisfied and AB \u2248 FB) toward increasing condition violation:

  5.2 Mode-separation   -- \u0394\u03bc from 0 \u2192 5  (condition 2: bimodality grows; \u0394\u03bc=0 is null)
  5.3 Spread asymmetry  -- \u03c3\u2081/\u03c3\u2082 from 1 \u2192 12  (condition 1: log-skewness grows; ratio=1 is null)

The two fixed scenarios (S1, S2) are cross-section snapshots at specific parameter
values chosen to represent conditions found in real streamflow data (see
fig_silverman_bounds.py for the empirical basis).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm as scipy_norm
from bokeh.layouts import gridplot, row, column
from bokeh.models import BoxAnnotation, Div, LinearAxis, Range1d
from bokeh.plotting import figure as bk_figure

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from kde_estimator import KDEEstimator, silverman_bandwidth
from utils import apply_kld_limited_uniform_mixture
from _plot_helpers import _AB_COLOR, _HIST_COLOR, _FB_COLOR, _apply_theme


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DA          = 1000.0   # synthetic drainage area [km²]  → Q[m³/s] = UAR[L/s/km²]
_BITS        = 8        # grid resolution
_N_SAMPLES   = 3650     # 10 years of daily observations
_SEED        = 42

from _plot_helpers import _AB_COLOR, _HIST_COLOR, _FB_COLOR
_TRUE_COLOR = "#1a1a1a"

_W_PDF  = 480
_H_PDF  = 320


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def _make_grid(bits: int = _BITS) -> KDEEstimator:
    log_edges = np.linspace(
        np.log(Config.GLOBAL_MIN_UAR),
        np.log(Config.GLOBAL_MAX_UAR),
        2**bits + 1,
    )
    return KDEEstimator(log_edges)


# ---------------------------------------------------------------------------
# Mixture helpers
# ---------------------------------------------------------------------------

def _mixture_pmf(grid: KDEEstimator, params: list[tuple]) -> np.ndarray:
    """True PMF on the grid from a mixture of log-normals.

    params: list of (weight, mu_log, sigma_log)
    mu_log and sigma_log are in log(UAR) space.
    """
    cdf_edges = np.zeros(len(grid.log_edges))
    for w, mu, sigma in params:
        cdf_edges += w * scipy_norm.cdf(grid.log_edges, loc=mu, scale=sigma)
    pmf = np.diff(cdf_edges)
    pmf = np.maximum(pmf, 0.0)
    pmf /= pmf.sum()
    return pmf


def _sample_mixture(
    n: int, params: list[tuple], rng: np.random.Generator
) -> np.ndarray:
    """Draw n UAR samples from a mixture of log-normals."""
    weights = np.array([p[0] for p in params])
    comps   = rng.choice(len(params), size=n, p=weights)
    samples = np.zeros(n)
    for i, (_, mu, sigma) in enumerate(params):
        mask = comps == i
        if mask.sum() > 0:
            samples[mask] = np.exp(rng.normal(mu, sigma, size=mask.sum()))
    # keep within global grid bounds
    samples = np.clip(
        samples,
        Config.GLOBAL_MIN_UAR * 1.001,
        Config.GLOBAL_MAX_UAR * 0.999,
    )
    return samples


# ---------------------------------------------------------------------------
# Divergence metrics
# ---------------------------------------------------------------------------

def _divergence(pmf_est: np.ndarray, pmf_true: np.ndarray, dlog: np.ndarray | None = None) -> tuple[float, float, float, float, float]:
    """KS, Wasserstein (L1), energy distance (L2), KL divergence, and MISE between two discrete PMFs.

    W1 and ED are integrated over the log-UAR axis using dlog as the integration weight.
    W1 is in log-UAR; ED is returned as sqrt(2·Σ ΔF²·δ) in \u221alog-UAR so that squaring gives
    ED² in log-UAR on the same scale as W1.  This matches the vectorised formula in
    preprocess.py.  When dlog is None a uniform bin width of 1 is assumed (dimensionless).

    Both PMFs are mixed toward uniform with the KLD-limited mixture before computing any
    metric.  This matches the upfront treatment in preprocess.py and ensures all bins carry
    strictly positive probability, eliminating the need for any per-metric guards.
    dlog from np.diff on a fixed linspace grid is always strictly positive by construction;
    no guard is applied so a zero value would fail loudly.
    """
    pmf_est_m  = apply_kld_limited_uniform_mixture(pmf_est,  Config.Metrics.KLD_DELTA_MAX)
    pmf_true_m = apply_kld_limited_uniform_mixture(pmf_true, Config.Metrics.KLD_DELTA_MAX)

    cdf_e = np.cumsum(pmf_est_m)
    cdf_t = np.cumsum(pmf_true_m)
    diff  = cdf_e - cdf_t
    ks    = float(np.max(np.abs(diff)))
    if dlog is not None and len(dlog) == len(pmf_est_m):
        w1 = float(np.sum(np.abs(diff) * dlog))
        ed = float(np.sqrt(2.0 * np.sum(diff ** 2 * dlog)))
    else:
        w1 = float(np.sum(np.abs(diff)))
        ed = float(np.sqrt(2.0 * np.sum(diff ** 2)))
    kl = float(np.sum(pmf_est_m * np.log2(pmf_est_m / pmf_true_m)))
    # MISE: integrated squared difference between the two density estimates in log-UAR space.
    # In discrete form: sum_j (pmf_est_j - pmf_true_j)^2 / dlog_j
    # dlog comes from np.diff on a fixed linspace grid and is by construction
    # strictly positive; no guard is applied so a zero would fail loudly.
    if dlog is not None and len(dlog) == len(pmf_est_m):
        mise = float(np.sum((pmf_est_m - pmf_true_m) ** 2 / dlog))
    else:
        mise = float(np.sum((pmf_est_m - pmf_true_m) ** 2))
    return ks, w1, ed, kl, mise


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _scenario_fig(
    title:             str,
    grid:              KDEEstimator,
    samples:           np.ndarray,
    true_pmf:          np.ndarray,
    ab_pmf:           np.ndarray,
    fb_pmf:           np.ndarray,
    da:                float = _DA,
    x_range=None,
    legend_loc: str = 'top_right',
    show_high_error_zone: bool = False,
) -> bk_figure:
    """Return the PDF figure for one scenario.

    All x-axis values are displayed as volumetric flow Q (m\u00b3/s):
        Q = UAR [L/s/km\u00b2] \u00d7 DA [km\u00b2] / 1000
    With DA = 1000 km\u00b2 the conversion factor is 1.0 so bin-PMF values are
    unchanged; only axis labels and coordinates reflect m\u00b3/s.
    """
    q_factor = da / 1000.0   # UAR -> Q (m\u00b3/s)
    q_lin_x   = grid.lin_x            * q_factor
    q_left    = grid.left_lin_edges   * q_factor
    q_right   = grid.right_lin_edges  * q_factor

    # Empirical histogram -- PMF values are pure probabilities (no unit change)
    hist_counts, _ = np.histogram(np.log(samples), bins=grid.log_edges)
    hist_pmf = hist_counts.astype(float) / hist_counts.sum()

    # Restrict x range to the region where data actually falls (in Q space)
    nonzero = np.where(hist_pmf > 0)[0]
    x_lo = float(q_lin_x[max(0, nonzero[0] - 8)])             if len(nonzero) else float(q_lin_x[0])
    x_hi = float(q_lin_x[min(len(q_lin_x) - 1, nonzero[-1] + 8)]) if len(nonzero) else float(q_lin_x[-1])
    data_x_range = Range1d(x_lo * 0.5, x_hi * 2.0)
    shared_x = x_range if x_range is not None else data_x_range

    # ---- PDF figure --------------------------------------------------------
    n_years = len(samples) / 365
    pdf_fig = bk_figure(
        width=_W_PDF, height=_H_PDF,
        x_axis_label="Q (m\u00b3 s\u207b\u00b9)",
        y_axis_label="Density",
        title=f"{title}  (N = 3650)",
        tools="pan,wheel_zoom,reset,save",
        x_axis_type="log",
        x_range=shared_x,
        y_range = (0, 0.5) if title.startswith('S1') else (0, 4),
    )
    _apply_theme(pdf_fig)

    # High-error zone shading (Q < 0.1 m\u00b3/s, error > 5%)
    if show_high_error_zone:
        pdf_fig.add_layout(
            BoxAnnotation(right=0.1, fill_color="#838383", fill_alpha=0.10, line_color=None)
        )

    # histogram bars
    pdf_fig.quad(
        top=hist_pmf / grid.log_w,
        bottom=0,
        left=q_left,
        right=q_right,
        color=_HIST_COLOR, alpha=0.50, line_color="#a3a3a3", line_alpha=0.4,
        line_width=0.5,
        legend_label="Sample histogram",
    )
    pdf_fig.line(q_lin_x, true_pmf / grid.log_w, line_width=2.0,
                 color=_TRUE_COLOR, legend_label="True PDF", line_dash='dotted')
    pdf_fig.line(q_lin_x, ab_pmf / grid.log_w, line_width=1.5,
                 color=_AB_COLOR, line_dash="solid",
                 legend_label="AB KDE")
    pdf_fig.line(q_lin_x, fb_pmf / grid.log_w, line_width=1.5,
                 color=_FB_COLOR, line_dash="solid",
                 legend_label="FB KDE")

    pdf_fig.legend.location             = legend_loc
    pdf_fig.legend.label_text_font      = "EB Garamond, serif"
    pdf_fig.legend.label_text_font_size = "12px"
    pdf_fig.legend.click_policy         = "hide"
    pdf_fig.legend.location = 'top_left'
    pdf_fig.legend.background_fill_alpha = 0.75

    # ---- Secondary y-axis: non-exceedance CDF (right side) ----------------
    cdf_ab = np.cumsum(ab_pmf)
    cdf_fb = np.cumsum(fb_pmf)
    pdf_fig.extra_y_ranges = {"cdf": Range1d(0.0, 1.0)}
    pdf_fig.add_layout(
        LinearAxis(
            y_range_name="cdf",
            axis_label="P(X \u2264 x)",
            axis_label_text_font="EB Garamond, Noto Serif, serif",
            axis_label_text_font_size="14px",
            major_label_text_font="EB Garamond, Noto Serif, serif",
            major_label_text_font_size="14px",
        ),
        "right",
    )
    pdf_fig.line(
        q_lin_x, cdf_ab,
        line_width=1.5, color='black', alpha=0.8,
        y_range_name="cdf", legend_label="AB CDF",
    )
    pdf_fig.line(
        q_lin_x, cdf_fb,
        line_width=1.5, color='grey', alpha=0.7,
        y_range_name="cdf", legend_label="FB CDF",
    )

    return pdf_fig


def _callout_div(
    formula_html: str,
    params_lines: list,
    note: str,
    scores: tuple,
    height: int = 400,
    width: int = _W_PDF,
) -> Div:
    """Prominent callout: formula, parameters, and score table with three comparison columns."""
    (ks_ab, w1_ab, ed_ab, kl_ab, mise_ab,
     ks_fb, w1_fb, ed_fb, kl_fb, mise_fb,
     ks_abfb, w1_abfb, ed_abfb, kl_abfb, mise_abfb) = scores

    def _cell(val_ab: float, val_fb: float, fmt: str = ".2f", thr: float = 0.025) -> tuple:
        a = format(val_ab, fmt)
        s = format(val_fb, fmt)
        if (val_fb - val_ab) > thr:
            return f"<b>{a}</b>", s
        elif (val_ab - val_fb) > thr:
            return a, f"<b>{s}</b>"
        else:
            return a, s

    def _v(val: float, fmt: str = ".2f") -> str:
        return format(val, fmt)

    w1_ab_d   = 100.0 * np.expm1(w1_ab)
    w1_fb_d   = 100.0 * np.expm1(w1_fb)
    w1_abfb_d = 100.0 * np.expm1(w1_abfb)
    ed_ab_d   = ed_ab  ** 2
    ed_fb_d   = ed_fb  ** 2
    ed_abfb_d = ed_abfb ** 2

    ks_a,   ks_s   = _cell(ks_ab,    ks_fb)
    w1_a,   w1_s   = _cell(w1_ab_d,  w1_fb_d,  fmt=".2f", thr=2.0)
    ed_a,   ed_s   = _cell(ed_ab_d,  ed_fb_d,  fmt=".4f", thr=0.001)
    kl_a,   kl_s   = _cell(kl_ab,    kl_fb)
    mise_a,  mise_s  = _cell(mise_ab,  mise_fb)

    bullets_html = "".join(
        f'<li style="margin:0 0 3px 0;">{p}</li>' for p in params_lines
    )

    html = f"""
<div style="
    width: {width}px;
    height: {height}px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    padding: 12px 18px;
    font-family: 'EB Garamond', Palatino, serif;
    color: #111111;
    background: #fffff8;
    box-sizing: border-box;
    overflow: auto;
">
  <div style="font-size:15px; font-weight:600; color:#111; line-height:1.4; margin-bottom:9px;">
    {formula_html}
  </div>
  <ul style="margin:0 0 9px 0; padding-left:14px; font-size:12px; color:#5c5c5c; line-height:1.5; list-style:disc;">
    {bullets_html}
  </ul>
  <div style="border-top:1px solid #d0d0c8; padding-top:5px; margin-bottom:4px;">
    <span style="color:#5c5c5c; font-size:11px; letter-spacing:0.07em; text-transform:uppercase;">
      Divergence scores
    </span>
  </div>
  <table style="border-collapse:collapse; width:100%; font-size:12px;">
    <thead>
      <tr>
        <th style="text-align:left; font-weight:normal; color:#5c5c5c; padding:1px 4px 3px 0;"></th>
        <th style="text-align:right; font-weight:normal; color:#4915ac; padding:1px 4px 3px 4px;">AB vs ref.</th>
        <th style="text-align:right; font-weight:normal; color:#c221d1; padding:1px 4px 3px 4px;">FB vs ref.</th>
        <th style="text-align:right; font-weight:normal; color:#555555; padding:1px 0 3px 4px;">D(AB, FB)</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background:#fff8e8;">
        <td style="color:#5c5c5c; padding:2px 4px 2px 0;">MISE &#x2605;</td>
        <td style="text-align:right; padding:2px 4px;">{mise_a}</td>
        <td style="text-align:right; padding:2px 4px;">{mise_s}</td>
        <td style="text-align:right; padding:2px 0;">{_v(mise_abfb)}</td>
      </tr>
      <tr>
        <td style="color:#5c5c5c; padding:2px 4px 2px 0;">KS</td>
        <td style="text-align:right; padding:2px 4px;">{ks_a}</td>
        <td style="text-align:right; padding:2px 4px;">{ks_s}</td>
        <td style="text-align:right; padding:2px 0;">{_v(ks_abfb)}</td>
      </tr>
      <tr style="background:#f4f4f0;">
        <td style="color:#5c5c5c; padding:2px 4px 2px 0;">W&#8321; (%)</td>
        <td style="text-align:right; padding:2px 4px;">{w1_a}</td>
        <td style="text-align:right; padding:2px 4px;">{w1_s}</td>
        <td style="text-align:right; padding:2px 0;">{_v(w1_abfb_d, ".2f")}</td>
      </tr>
      <tr>
        <td style="color:#5c5c5c; padding:2px 4px 2px 0;">ED&#x00b2;</td>
        <td style="text-align:right; padding:2px 4px;">{ed_a}</td>
        <td style="text-align:right; padding:2px 4px;">{ed_s}</td>
        <td style="text-align:right; padding:2px 0;">{_v(ed_abfb_d, ".4f")}</td>
      </tr>
      <tr style="background:#f4f4f0;">
        <td style="color:#5c5c5c; padding:2px 4px 2px 0;">KL</td>
        <td style="text-align:right; padding:2px 4px;">{kl_a}</td>
        <td style="text-align:right; padding:2px 4px;">{kl_s}</td>
        <td style="text-align:right; padding:2px 0;">{_v(kl_abfb)}</td>
      </tr>
    </tbody>
  </table>
  <div style="margin-top:7px; font-size:11px; color:#979797; line-height:1.35;">
    {note}<br>
    &#x2605; Primary metric. MISE (mean integrated squared error); Silverman&rsquo;s rule minimises MISE under a Gaussian reference.<br>
    AB/FB vs ref.: bold = closer to truth (KS, W&#8321;, ED&#x00b2;, KL are diagnostics).<br>
    D(AB, FB): divergence between the two estimators; true distribution not involved.
  </div>
</div>
"""
    return Div(text=html, width=width, height=height)


# ---------------------------------------------------------------------------
# Scenario data container
# ---------------------------------------------------------------------------

@dataclass
class ScenarioData:
    title:         str
    samples:       np.ndarray
    true_pmf:      np.ndarray
    ab_pmf:       np.ndarray
    fb_pmf:       np.ndarray
    formula_html:  str
    params_lines:  list[str]
    note:          str
    ks_ab:        float
    w1_ab:        float
    ed_ab:        float
    kl_ab:        float
    mise_ab:      float
    ks_fb:        float
    w1_fb:        float
    ed_fb:        float
    kl_fb:        float
    mise_fb:      float
    ks_ab_fb:     float
    w1_ab_fb:     float
    ed_ab_fb:     float
    kl_ab_fb:     float
    mise_ab_fb:   float


# ---------------------------------------------------------------------------
# Computation: one function per scenario
# ---------------------------------------------------------------------------
def pooled_skewness(w1: float, mu1: float, sig1: float, mu2: float, sig2: float) -> float:
    w2 = 1.0 - w1
    delta = mu1 - mu2
    mu3 = w1 * w2 * delta * ((1 - 2*w1) * delta**2 + 3*(sig1**2 - sig2**2))
    var = w1*(sig1**2 + w2**2*delta**2) + w2*(sig2**2 + w1**2*delta**2)
    return mu3 / var**1.5


def _compute_s1(grid: KDEEstimator, rng: np.random.Generator) -> ScenarioData:
    # Mixture parameters in UAR [L/s/km²] space.
    # With DA = 1000 km², UAR numerically equals Q in m³/s.
    wt1, mu1_q, sig1 = 0.2, 5,  0.3
    wt2, mu2_q, sig2 = 0.8, 25.0, 0.90
    # mu_gap = np.abs(np.log(mu2_q) - np.log(mu1_q)) / mean_spread
    params = [
        (wt1, np.log(mu1_q), sig1),
        (wt2, np.log(mu2_q), sig2),
    ]
    samples = _sample_mixture(_N_SAMPLES, params, rng)
    true    = _mixture_pmf(grid, params)
    ab, _  = grid.compute(samples, _DA)
    fb, _  = grid.compute_silverman(samples, _DA)
    dlog   = np.diff(grid.log_edges)
    ks_ab, w1_ab, ed_ab, kl_ab, mise_ab = _divergence(ab, true, dlog)
    ks_fb, w1_fb, ed_fb, kl_fb, mise_fb = _divergence(fb, true, dlog)
    ks_ab_fb, w1_ab_fb, ed_ab_fb, kl_ab_fb, mise_ab_fb = _divergence(ab, fb, dlog)
    # compute delta mu as a function of the arithmetic mean 
    # of component spread to match the 3 sigma Silverman threshold
    mean_spread = 0.5 * (sig1 + sig2)
    delta_mu = np.abs(np.log(mu2_q) - np.log(mu1_q)) / mean_spread

    return ScenarioData(
        title        = f"S1: \u0394\u03bc \u202f = {delta_mu:.2f} < \u202f3\u03c3",
        samples      = samples,
        true_pmf     = true,
        ab_pmf      = ab,
        fb_pmf      = fb,
        formula_html = f"f(x) = {wt1}\u00b7LN(\u03bc\u2081, \u03c3\u2081) + {wt2}\u00b7LN(\u03bc\u2082, \u03c3\u2082)",
        params_lines = [
            f"\u03bc\u2081 = {mu1_q}\u2009m\u00b3/s,  \u03c3\u2081 = {sig1}",
            f"\u03bc\u2082 = {mu2_q}\u2009m\u00b3/s,  \u03c3\u2082 = {sig2}",
        ],
        note         = "\u03bc = median Q (m\u00b3/s);  \u03c3 in natural-log space",
        ks_ab=ks_ab, w1_ab=w1_ab, ed_ab=ed_ab, kl_ab=kl_ab, mise_ab=mise_ab,
        ks_fb=ks_fb, w1_fb=w1_fb, ed_fb=ed_fb, kl_fb=kl_fb, mise_fb=mise_fb,
        ks_ab_fb=ks_ab_fb, w1_ab_fb=w1_ab_fb, ed_ab_fb=ed_ab_fb, kl_ab_fb=kl_ab_fb, mise_ab_fb=mise_ab_fb,
    )


def _compute_s2(grid: KDEEstimator, rng: np.random.Generator) -> ScenarioData:
    wt1, mu1_q, sig1 = 0.25, 0.025, 0.3   # low-Q pool, high-error zone
    wt2, mu2_q, sig2 = 0.75, 0.5,   0.1   # narrow high-flow mode: forces small Silverman h
    quant_thresh = 0.1    # Q below this is quantized [m³/s]
    quant_step   = 0.005   # rating-curve resolution step [m³/s]
    quant_step   = 0.01   # rating-curve resolution step [m³/s]
    params = [
        (wt1, np.log(mu1_q), sig1),
        (wt2, np.log(mu2_q), sig2),
    ]
    raw  = _sample_mixture(_N_SAMPLES, params, rng)
    true = _mixture_pmf(grid, params)
    pooled_skew = pooled_skewness(wt1, np.log(mu1_q), sig1, np.log(mu2_q), sig2)

    # Quantize low-Q values to the nearest quant_step.
    # DA = 1000 km² so UAR [L/s/km²] = Q [m³/s] numerically.
    samples = raw.copy()
    low_mask = samples < quant_thresh
    samples[low_mask] = np.round(samples[low_mask] / quant_step) * quant_step
    samples = np.clip(samples, Config.GLOBAL_MIN_UAR * 1.001, Config.GLOBAL_MAX_UAR * 0.999)

    ab, _ = grid.compute(samples, _DA)
    fb, _ = grid.compute_silverman(samples, _DA)
    dlog  = np.diff(grid.log_edges)
    ks_ab, w1_ab, ed_ab, kl_ab, mise_ab = _divergence(ab, true, dlog)
    ks_fb, w1_fb, ed_fb, kl_fb, mise_fb = _divergence(fb, true, dlog)
    ks_ab_fb, w1_ab_fb, ed_ab_fb, kl_ab_fb, mise_ab_fb = _divergence(ab, fb, dlog)
    return ScenarioData(
        title        = f"S2: |\u03b3\u2081|\u202f={np.abs(pooled_skew):.2f} < 1.8 with precision artifact",
        samples      = samples,
        true_pmf     = true,
        ab_pmf      = ab,
        fb_pmf      = fb,
        formula_html = f"f(x) = {wt1}\u00b7LN(\u03bc\u2081,\u03c3\u2081) + {wt2}\u00b7LN(\u03bc\u2082,\u03c3\u2082)",
        params_lines = [
            f"\u03bc\u2081 = {mu1_q}\u2009m\u00b3/s,  \u03c3\u2081 = {sig1}  (low-Q, high-error zone)",
            f"\u03bc\u2082 = {mu2_q}\u2009m\u00b3/s,  \u03c3\u2082 = {sig2}  (narrow, forces small FB h)",
            f"Q < {quant_thresh}\u2009m\u00b3/s rounded to nearest {quant_step}\u2009m\u00b3/s (rating precision floor)",
            "\u25a4 shaded region: error model > 5%  (Q < 0.1 m\u00b3/s)",
        ],
        note         = "\u03bc = median Q (m\u00b3/s);  \u03c3 in log space",
        ks_ab=ks_ab, w1_ab=w1_ab, ed_ab=ed_ab, kl_ab=kl_ab, mise_ab=mise_ab,
        ks_fb=ks_fb, w1_fb=w1_fb, ed_fb=ed_fb, kl_fb=kl_fb, mise_fb=mise_fb,
        ks_ab_fb=ks_ab_fb, w1_ab_fb=w1_ab_fb, ed_ab_fb=ed_ab_fb, kl_ab_fb=kl_ab_fb, mise_ab_fb=mise_ab_fb,
    )


# ---------------------------------------------------------------------------
# Rendering: scenario data → (figure, callout)
# ---------------------------------------------------------------------------

def _render_scenario(
    sc: ScenarioData,
    grid: KDEEstimator,
    da: float = _DA,
    x_range=None,
    show_high_error_zone: bool = False,
    legend_loc: str = 'top_right',

):
    """Return (fig, callout) for one ScenarioData."""
    fig = _scenario_fig(
        sc.title, grid, sc.samples, sc.true_pmf, sc.ab_pmf, sc.fb_pmf,
        da=da, x_range=x_range, show_high_error_zone=show_high_error_zone,
        legend_loc=legend_loc,
    )
    callout = _callout_div(
        formula_html = sc.formula_html,
        params_lines = sc.params_lines,
        note         = sc.note,
        scores       = (sc.ks_ab, sc.w1_ab, sc.ed_ab, sc.kl_ab, sc.mise_ab,
                        sc.ks_fb, sc.w1_fb, sc.ed_fb, sc.kl_fb, sc.mise_fb,
                        sc.ks_ab_fb, sc.w1_ab_fb, sc.ed_ab_fb, sc.kl_ab_fb, sc.mise_ab_fb),
    )
    return fig, callout


# ---------------------------------------------------------------------------
# Shared sweep-render helpers
# ---------------------------------------------------------------------------

_SWEEP_METRIC_SPECS = [
    ("MISE vs ref. (log-flow units\u207b\u00b9)",        "mise_ab",  "mise_fb",  None),
    ("KS vs ref.",                     "ks_ab",   "ks_fb",    None),
    ("W\u2081 vs ref. (%)",             "w1_ab",   "w1_fb",    lambda v: 100.0 * np.expm1(v)),
    ("ED\u00b2 vs ref. (log-flow units)", "ed_ab",  "ed_fb",    lambda v: v ** 2),
    ("KL vs ref. (bits)",              "kl_ab",   "kl_fb",    None),
]


def _sweep_xy_ranges(
    sweep:   list[dict],
    grid:    KDEEstimator,
    q_lin_x: np.ndarray,
) -> tuple:
    """Shared x- and y-range Range1d objects spanning all sweep panels."""
    all_lo, all_hi = [], []
    for d in sweep:
        hist_counts, _ = np.histogram(np.log(d["samples"]), bins=grid.log_edges)
        hist_pmf = hist_counts.astype(float) / hist_counts.sum()
        nz = np.where(hist_pmf > 0)[0]
        if len(nz):
            all_lo.append(nz[0])
            all_hi.append(nz[-1])
    if all_lo:
        x_lo = float(q_lin_x[max(0, min(all_lo) - 8)]) * 0.5
        x_hi = float(q_lin_x[min(len(q_lin_x) - 1, max(all_hi) + 8)]) * 2.0
    else:
        x_lo, x_hi = float(q_lin_x[0]), float(q_lin_x[-1])
    y_max_global = max(
        max(
            (d["ab"]   / grid.log_w).max(),
            (d["fb"]   / grid.log_w).max(),
            (d["true"] / grid.log_w).max(),
        )
        for d in sweep
    )
    return Range1d(x_lo, x_hi), Range1d(0.0, y_max_global * 1.10)


def _make_sweep_panel(
    title_str:  str,
    i:          int,
    n_panels:   int,
    q_lin_x:    np.ndarray,
    d:          dict,
    grid:       KDEEstimator,
    shared_x,
    shared_y,
    legend_loc: str = "top_left",
) -> object:
    """One small PDF-overlay panel for a sweep row, with a right-side CDF axis."""
    is_last = (i == n_panels - 1)
    fig = bk_figure(
        frame_width=206, frame_height=150,
        title=title_str,
        tools="pan,wheel_zoom,reset",
        x_axis_type="log",
        x_range=shared_x,
        x_axis_label="Q (m\u00b3 s\u207b\u00b9)",
        y_axis_label="Density" if i == 0 else "",
        y_range=shared_y,
    )
    _apply_theme(fig)
    fig.title.text_font_size               = "14px"
    fig.axis.axis_label_text_font_size     = "12px"
    fig.axis.major_label_text_font_size    = "12px"
    fig.line(q_lin_x, d["ab"]   / grid.log_w, line_width=1.25,
             color=_AB_COLOR, legend_label="AB")
    fig.line(q_lin_x, d["fb"]   / grid.log_w, line_width=1.25,
             color=_FB_COLOR,  legend_label="FB")
    fig.line(q_lin_x, d["true"] / grid.log_w, line_width=3,
             color=_TRUE_COLOR, line_dash="dotted", legend_label="Reference")
    # Secondary y-axis: non-exceedance CDF (right side)
    cdf_ab = np.cumsum(d["ab"])
    cdf_fb = np.cumsum(d["fb"])
    left_yaxis = fig.yaxis[0]   # capture before add_layout appends the right axis
    fig.extra_y_ranges = {"cdf": Range1d(0.0, 1.0)}
    right_axis = LinearAxis(
        y_range_name="cdf",
        axis_label="P(X \u2264 x)" if is_last else "",
        axis_label_text_font="EB Garamond, Noto Serif, serif",
        axis_label_text_font_size="11px",
        major_label_text_font="EB Garamond, Noto Serif, serif",
        major_label_text_font_size="13px",
        visible=is_last,
    )
    fig.add_layout(right_axis, "right")
    fig.line(q_lin_x, cdf_ab, line_width=1.5, color="black", alpha=0.8,
             y_range_name="cdf", legend_label="AB CDF")
    fig.line(q_lin_x, cdf_fb, line_width=1.5, color="grey",  alpha=0.7,
             y_range_name="cdf", legend_label="FB CDF")
    fig.legend.label_text_font       = "EB Garamond, serif"
    fig.legend.label_text_font_size  = "13px"
    fig.legend.click_policy          = "hide"
    fig.legend.background_fill_alpha = 0.75
    fig.toolbar.logo = None
    if not is_last:
        fig.legend.visible = False
    else:
        # fig.legend.location = legend_loc
        fig.add_layout(fig.legend[0], 'right')
    left_yaxis.visible = (i == 0)   # set after all axes added so only the left axis is affected
    fig.xgrid.grid_line_alpha = 0.4
    fig.ygrid.grid_line_alpha = 0.4
    return fig


def _link_sweep_legends(figs: list) -> None:
    """Register every panel's renderers in the last panel's visible legend.

    Non-last panels have legend.visible = False, so click_policy='hide' on the
    visible (last-panel) legend has no effect on them.  This extends each
    LegendItem's renderer list with the matching renderers from all hidden
    panels, keyed by label string, so a single click toggles lines everywhere.
    """
    if len(figs) < 2:
        return
    visible_legend = figs[-1].legend[0]

    def _label_str(lbl) -> str | None:
        """Extract the plain string from a Bokeh label spec (dict or Value object)."""
        if isinstance(lbl, dict):
            return lbl.get("value")
        val = getattr(lbl, "value", None)
        return val if isinstance(val, str) else None

    item_map = {}
    for item in visible_legend.items:
        lbl = _label_str(item.label)
        if lbl is not None:
            item_map[lbl] = item
    for fig in figs[:-1]:
        for legend in fig.legend:
            for item in legend.items:
                label = _label_str(item.label)
                if label in item_map:
                    item_map[label].renderers.extend(item.renderers)


def _make_score_charts(
    param_vals:   list,
    sweep:        list[dict],
    x_axis_label: str,
) -> list:
    """Five score-chart figures (MISE, KS, W\u2081, ED\u00b2, KL vs ref.) against a sweep parameter."""
    score_figs = []
    for i, (ylabel, ab_key, fb_key, tfm) in enumerate(_SWEEP_METRIC_SPECS):
        ab_v = [d[ab_key] for d in sweep]
        fb_v = [d[fb_key] for d in sweep]
        if tfm is not None:
            ab_v = [tfm(v) for v in ab_v]
            fb_v = [tfm(v) for v in fb_v]
        sf = bk_figure(
            width=225, height=210,
            x_axis_label=x_axis_label,
            y_axis_label=ylabel,
            title=ylabel,
            tools="",
        )
        _apply_theme(sf)
        sf.title.text_font_size = "12px"
        sf.line(param_vals, ab_v, line_width=2, color=_AB_COLOR, legend_label="AB")
        sf.line(param_vals, fb_v, line_width=2, color=_FB_COLOR, legend_label="FB")
        sf.scatter(param_vals, ab_v, size=5, color=_AB_COLOR)
        sf.scatter(param_vals, fb_v, size=5, color=_FB_COLOR)
        sf.legend.location              = "top_left"
        sf.legend.label_text_font       = "EB Garamond, serif"
        sf.legend.label_text_font_size  = "12px"
        sf.legend.background_fill_alpha = 0.75
        sf.toolbar.logo = None
        sf.xgrid.grid_line_alpha = 0.4
        sf.ygrid.grid_line_alpha = 0.4
        # if i > 0:
        sf.legend.visible = False
        score_figs.append(sf)
    return score_figs



# ---------------------------------------------------------------------------
# Spread-asymmetry sweep
# ---------------------------------------------------------------------------

def _compute_spread_asymmetry_sweep(
    grid:    KDEEstimator,
    rng:     np.random.Generator,
    mu1_log: float,
    mu2_log: float,
    w1:      float = 0.5,
    n_steps: int   = 5,
    ratio_min: float = 1.0,
    ratio_max: float = 8.0,
    sigma_hmean: float = 0.55,
) -> list[dict]:
    """Vary the ratio sigma_1 / sigma_2 from ratio_min to ratio_max on a log scale.

    Component *means* (mu1_log, mu2_log) and *weights* (w1, w2 = 1-w1) are fixed.
    The harmonic mean of (sigma_1, sigma_2) is held constant at sigma_hmean so that
    total mixture spread is comparable across panels and changes are purely in shape,
    not scale.  For a harmonic mean h and ratio r = sigma_1/sigma_2:

        sigma_1 = sigma_2 * r
        2 / h   = 1/sigma_1 + 1/sigma_2 = (1 + 1/r) / sigma_2
        => sigma_2 = h * (1 + 1/r) / 2
        => sigma_1 = r * sigma_2

    Returns a list of dicts (one per ratio value) each with keys:
        ratio, sigma_1, sigma_2, samples, true, ab, fb, h_fb,
        ks_ab, ks_fb, w1_ab, w1_fb, ed_ab, ed_fb, kl_ab, kl_fb,
        mise_ab, mise_fb.
    """
    ratio_vals = np.exp(
        np.linspace(np.log(ratio_min), np.log(ratio_max), n_steps)
    )
    w2 = 1.0 - w1
    results = []
    for ratio in ratio_vals:
        sig2 = sigma_hmean * (1.0 + 1.0 / ratio) / 2.0
        sig1 = ratio * sig2
        params = [
            (w1, mu1_log, sig1),
            (w2, mu2_log, sig2),
        ]
        samples = _sample_mixture(_N_SAMPLES, params, rng)
        true    = _mixture_pmf(grid, params)
        ab, _  = grid.compute(samples, _DA)
        fb, _  = grid.compute_silverman(samples, _DA)
        h_fb   = silverman_bandwidth(np.log(samples))
        dlog   = np.diff(grid.log_edges)
        ks_ab, w1_ab, ed_ab, kl_ab, mise_ab = _divergence(ab, true, dlog)
        ks_fb, w1_fb, ed_fb, kl_fb, mise_fb = _divergence(fb, true, dlog)
        skew = pooled_skewness(w1, mu1_log, sig1, mu2_log, sig2)
        results.append(dict(
            ratio=float(ratio), sigma_1=sig1, sigma_2=sig2,
            samples=samples, true=true, ab=ab, fb=fb, h_fb=h_fb,
            ks_ab=ks_ab, ks_fb=ks_fb,
            w1_ab=w1_ab, w1_fb=w1_fb,
            ed_ab=ed_ab, ed_fb=ed_fb,
            kl_ab=kl_ab, kl_fb=kl_fb,
            sample_skew=skew,
            mise_ab=mise_ab, mise_fb=mise_fb,
        ))
    return results


def _render_spread_asymmetry_sweep(
    sweep:  list[dict],
    grid:   KDEEstimator,
    mu1_q:  float,
    mu2_q:  float,
    w1:     float,
    da:     float = _DA,
) -> object:
    """PDF panels and score charts (MISE, KS, W\u2081, ED\u00b2, KL) vs \u03c3\u2081/\u03c3\u2082."""
    q_lin_x = grid.lin_x * (da / 1000.0)
    shared_x, shared_y = _sweep_xy_ranges(sweep, grid, q_lin_x)
    n_panels = len(sweep)
    skew_symbol = "\u03b3" 
    figs = [
        _make_sweep_panel(
            f"\u03c3\u2081/\u03c3\u2082 = {d['ratio']:.2f}   h\u2080 = {d['h_fb']:.2f} ({skew_symbol} = {d['sample_skew']:.2f})",
            i, n_panels, q_lin_x, d, grid, shared_x, shared_y,
        )
        for i, d in enumerate(sweep)
    ]
    _link_sweep_legends(figs)
    panel_grid = gridplot([figs], toolbar_location="above", merge_tools=True)
    ratio_vals = [d["ratio"] for d in sweep]
    ax_label = "\u03c3\u2081/\u03c3\u2082"
    score_figs = _make_score_charts(ratio_vals, sweep, ax_label)
    return column(panel_grid, gridplot(score_figs, ncols=5))


# ---------------------------------------------------------------------------
# Mode-separation sweep
# ---------------------------------------------------------------------------
def _compute_separation_sweep(
    grid:      KDEEstimator,
    rng:       np.random.Generator,
    mu1_log:   float,
    sigma:     float = 0.55,
    w1:        float = 0.5,
    n_steps:   int   = 6,
    delta_min: float = 0.0,
    delta_max: float = 5.0,
) -> list[dict]:
    """Vary the mode separation Δμ = mu2_log - mu1_log from delta_min to delta_max.

    Component *spreads* (sigma_1 = sigma_2 = sigma) and *weights* (w1, w2 = 1-w1)
    are fixed.  At delta=0 both components are identical, collapsing the mixture to
    a single log-normal where AB and FB are expected to agree (lower bound).  As
    delta grows the distribution becomes progressively more bimodal and Silverman's
    single bandwidth is stretched across the widening inter-modal gap.

    Returns a list of dicts (one per delta value) each with keys:
        delta, q_ratio, sigma, samples, true, ab, fb, h_fb,
        ks_ab, ks_fb, w1_ab, w1_fb, ed_ab, ed_fb, kl_ab, kl_fb,
        mise_ab, mise_fb.
    """
    delta_vals = np.linspace(delta_min, delta_max, n_steps)
    w2 = 1.0 - w1
    results = []
    for delta in delta_vals:
        mu2_log = mu1_log + delta
        params = [
            (w1, mu1_log, sigma),
            (w2, mu2_log, sigma),
        ]
        samples = _sample_mixture(_N_SAMPLES, params, rng)
        true    = _mixture_pmf(grid, params)
        ab, _  = grid.compute(samples, _DA)
        fb, _  = grid.compute_silverman(samples, _DA)
        h_fb   = silverman_bandwidth(np.log(samples))
        dlog   = np.diff(grid.log_edges)
        ks_ab, w1_ab, ed_ab, kl_ab, mise_ab = _divergence(ab, true, dlog)
        ks_fb, w1_fb, ed_fb, kl_fb, mise_fb = _divergence(fb, true, dlog)
        results.append(dict(
            delta=float(delta), q_ratio=float(np.exp(delta)), sigma=sigma,
            samples=samples, true=true, ab=ab, fb=fb, h_fb=h_fb,
            ks_ab=ks_ab, ks_fb=ks_fb,
            w1_ab=w1_ab, w1_fb=w1_fb,
            ed_ab=ed_ab, ed_fb=ed_fb,
            kl_ab=kl_ab, kl_fb=kl_fb,
            mise_ab=mise_ab, mise_fb=mise_fb,
        ))
    return results


def _render_separation_sweep(
    sweep:  list[dict],
    grid:   KDEEstimator,
    mu1_q:  float,
    sigma:  float,
    w1:     float,
    da:     float = _DA,
) -> object:
    """PDF panels and score charts (MISE, KS, W\u2081, ED\u00b2, KL) vs \u0394\u03bc (log-space separation)."""
    q_lin_x = grid.lin_x * (da / 1000.0)
    shared_x, shared_y = _sweep_xy_ranges(sweep, grid, q_lin_x)
    n_panels = len(sweep)
    figs = []
    for i, d in enumerate(sweep):
        delta, q_ratio, h_val = d["delta"], d["q_ratio"], d["h_fb"]
        if delta < 1e-9:
            title_str = "\u0394\u03bc = 0.00  (unimodal)"
        else:
            title_str = f"\u0394\u03bc = {delta:.2f}  Q\u2082/Q\u2081 = {q_ratio:.0f}\u00d7  h\u2080 = {h_val:.3f}"
        figs.append(
            _make_sweep_panel(title_str, i, n_panels, q_lin_x, d, grid, shared_x, shared_y, legend_loc="top_right")
        )
    _link_sweep_legends(figs)
    panel_grid = gridplot([figs], toolbar_location="above", merge_tools=True)
    delta_vals = [d["delta"] for d in sweep]
    ax_label = "\u0394\u03bc"
    score_figs = _make_score_charts(delta_vals, sweep, f"{ax_label}")
    return column(panel_grid, gridplot(score_figs, ncols=5))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_synthetic_section() -> dict:
    """Return a dict of four Bokeh layout objects, one per synthetic subsection.

    Keys
    ----
    synthetic_scenarios      : PDF figures + callout panels for Scenarios 1 and 2.
    synthetic_separation_sweep: Mode-separation sweep (equal weights, equal sigmas;
                                isolates Δμ as the sole varying parameter).
    synthetic_spread_sweep   : Spread-asymmetry sweep (equal weights, fixed Δμ;
                                isolates σ1/σ2 as the sole varying parameter).
    """
    rng  = np.random.default_rng(_SEED)
    grid = _make_grid()
    n_sweep_steps = 5

    s1 = _compute_s1(grid, rng)
    s2 = _compute_s2(grid, rng)  # rating-curve precision floor

    f1, c1 = _render_scenario(s1, grid, da=_DA, x_range=(5e-2, 1e3))
    f3, c3 = _render_scenario(s2, grid, da=_DA, show_high_error_zone=True, legend_loc='top_left')


    # Mode-separation sweep: σ1=σ2=0.55, w1=w2=0.5 fixed; only Δμ varies.
    # At Δμ=0 both components are identical (lower bound: AB≈FB).
    # At Δμ=5 the modes span a Q2/Q1 ratio of exp(5)≈148×.
    # Anchor: mu1_q=3.0 m³/s so the grid safely contains both modes at all
    # separations (3σ extent of component 2 reaches log(445)=6.10,
    # well within GLOBAL_MAX_UAR=log(10000)=9.21).
    _sep_mu1_q   = 3.0    # m³/s
    _sep_sigma   = 0.55
    _sep_w1      = 0.5
    sep_sweep        = _compute_separation_sweep(
        grid, rng, np.log(_sep_mu1_q),
        sigma=_sep_sigma, w1=_sep_w1, n_steps=n_sweep_steps, delta_min=0.0, delta_max=5.0,
    )
    sep_sweep_layout = _render_separation_sweep(
        sep_sweep, grid, _sep_mu1_q, _sep_sigma, _sep_w1,
    )

    # Spread-asymmetry sweep: dedicated mode positions with separation ≈ 2.5 × sigma_hmean
    # so that at ratio=1 FB has a fair chance, and the asymmetry failure emerges cleanly
    # as ratio grows.  ratio_max=12 extends the sweep into the plateau region where FB
    # error stabilises, giving a visible upper bound.
    # Derivation: sigma_hmean=0.55, target sep = 2.5 × 0.55 = 1.375
    #   mu1_q=2.0, mu2_q=8.0  (sep=log(4)≈1.386)
    _asym_mu1_q      = 1    # m³/s
    _asym_mu2_q      = 5.0    # m³/s
    _asym_w1         = 0.7
    _asym_sigma_hmean = 0.65  # harmonic mean of component sigmas (function default)
    asym_sweep        = _compute_spread_asymmetry_sweep(
        grid, rng, np.log(_asym_mu1_q), np.log(_asym_mu2_q),
        w1=_asym_w1, n_steps=n_sweep_steps, ratio_min=1.0, ratio_max=12.0,
        sigma_hmean=_asym_sigma_hmean,
    )
    asym_sweep_layout = _render_spread_asymmetry_sweep(
        asym_sweep, grid, _asym_mu1_q, _asym_mu2_q, _asym_w1,
    )

    result = {
        "synthetic_scenarios":       column(
            gridplot([f1, f3], ncols=2, toolbar_location="above", merge_tools=True),
            row(c1, c3),
        ),
        "synthetic_separation_sweep": sep_sweep_layout,
        "synthetic_spread_sweep":    asym_sweep_layout,
    }
    result.update(build_mise_vs_ed_section())
    return result


# ---------------------------------------------------------------------------
# MISE vs. ED: minimum illustrative examples
# ---------------------------------------------------------------------------

def _toy_pmf_cdf_fig(
    bins:          np.ndarray,
    true_pmf:      np.ndarray,
    pmf_a:         np.ndarray,
    pmf_b:         np.ndarray,
    label_a:       str,
    label_b:       str,
    title:         str,
    x_range,
    width:         int  = 290,
    height:        int  = 170,
    show_y:        bool = True,
    show_cdf_axis: bool = True,
) -> bk_figure:
    """PMF step lines (left axis) and optional CDF lines (right axis) for two toy estimators."""
    cdf_t = np.cumsum(true_pmf)
    cdf_a = np.cumsum(pmf_a)
    cdf_b = np.cumsum(pmf_b)


    # y limits: centre on the true PMF value, show ±2× the perturbation range
    pmf_lo = min(true_pmf.min(), pmf_a.min(), pmf_b.min()) * 0.5
    pmf_hi = max(true_pmf.max(), pmf_a.max(), pmf_b.max()) * 1.1

    fig = bk_figure(
            frame_width=width, frame_height=height, title=title,
            tools="pan,wheel_zoom,reset",
            x_axis_label="Bin index", y_axis_label="PMF",
            x_range=x_range, y_range=Range1d(pmf_lo, pmf_hi),
    )

    # Right axis: CDF (only on the right-column figure)
    if show_cdf_axis:
        fig.extra_y_ranges = {"cdf": Range1d(0, 1.1)}
        fig.add_layout(
            LinearAxis(
                y_range_name="cdf",
                axis_label="P(X \u2264 x)",
                axis_label_text_font="EB Garamond, Noto Serif, serif",
                axis_label_text_font_size="12px",
                major_label_text_font="EB Garamond, Noto Serif, serif",
                major_label_text_font_size="12px",
            ),
            "right",
        )
        fig.line(bins, cdf_t, line_width=1.2, color=_TRUE_COLOR, line_dash="dotted",
                 y_range_name="cdf", legend_label="True CDF")
        fig.line(bins, cdf_a, line_width=1.8, color="black", alpha=0.8,
                 y_range_name="cdf", legend_label=f"{label_a} CDF")
        fig.line(bins, cdf_b, line_width=1.8, color="grey", alpha=0.7,
                 y_range_name="cdf", legend_label=f"{label_b} CDF")

    _apply_theme(fig)
    fig.step(bins, true_pmf, line_width=1.2, color=_TRUE_COLOR,
             line_dash="dotted", legend_label="True distribution", mode="center")
    fig.step(bins, pmf_a, line_width=1.8, color=_AB_COLOR,
             legend_label=label_a, mode="center")
    fig.step(bins, pmf_b, line_width=1.8, color=_FB_COLOR,
             legend_label=label_b, mode="center")

    fig.legend.location              = "top_left"
    fig.legend.label_text_font       = "EB Garamond, serif"
    fig.legend.label_text_font_size  = "10px"
    fig.legend.background_fill_alpha = 0.75
    fig.legend.visible               = True
    fig.toolbar.logo = None
    fig.xgrid.grid_line_alpha = 0.3
    fig.legend.click_policy = "hide"
    fig.ygrid.grid_line_alpha = 0.3
    if not show_y:
        fig.yaxis.visible = False
    return fig


def _toy_residual_fig(
    bins:    np.ndarray,
    res_a:   np.ndarray,
    res_b:   np.ndarray,
    label_a: str,
    label_b: str,
    title:   str,
    x_range,
    width:   int = 290,
    height:  int = 170,
    show_y: bool = True,

) -> bk_figure:
    """Bar chart of PMF residuals (est - true) for two estimators, half-bin wide each."""
    fig = bk_figure(
        frame_width=width, frame_height=height, title=title, tools="pan,wheel_zoom,reset",
        x_axis_label="Bin index", y_axis_label="PMF residual",
        x_range=x_range,
    )
    _apply_theme(fig)
    fig.quad(top=res_a, bottom=0, left=bins - 0.45, right=bins,
             color=_AB_COLOR, alpha=0.8, legend_label=label_a)
    fig.quad(top=res_b, bottom=0, left=bins, right=bins + 0.45,
             color=_FB_COLOR, alpha=0.8, legend_label=label_b)
    fig.line([float(bins[0]), float(bins[-1])], [0.0, 0.0],
             line_width=0.8, color="#999999")
    fig.legend.location              = "bottom_left"
    fig.legend.label_text_font       = "EB Garamond, serif"
    fig.legend.label_text_font_size  = "11px"
    fig.legend.background_fill_alpha = 0.75
    fig.toolbar.logo = None
    fig.xgrid.grid_line_alpha = 0.3
    fig.ygrid.grid_line_alpha = 0.3
    if not show_y:
        fig.yaxis.visible = False
    return fig


def _toy_cdf_gap_fig(
    bins:    np.ndarray,
    gap_a:   np.ndarray,
    gap_b:   np.ndarray,
    label_a: str,
    label_b: str,
    title:   str,
    x_range,
    width:   int = 290,
    height:  int = 170,
    show_y: bool = True,
) -> bk_figure:
    """Line plot of CDF gap (CDF_est - CDF_true) for two estimators."""
    fig = bk_figure(
        frame_width=width, frame_height=height, title=title, tools="pan,wheel_zoom,reset",
        x_axis_label="Bin index", y_axis_label="CDF gap",
        x_range=x_range,
    )
    _apply_theme(fig)
    fig.line(bins, gap_a, line_width=2.0, color=_AB_COLOR, legend_label=label_a)
    fig.line(bins, gap_b, line_width=2.0, color=_FB_COLOR,  legend_label=label_b)
    fig.line([float(bins[0]), float(bins[-1])], [0.0, 0.0],
             line_width=0.8, color="#999999")
    fig.legend.location              = "bottom_center"
    fig.legend.label_text_font       = "EB Garamond, serif"
    fig.legend.label_text_font_size  = "11px"
    fig.legend.background_fill_alpha = 0.75
    fig.toolbar.logo = None
    fig.xgrid.grid_line_alpha = 0.3
    fig.ygrid.grid_line_alpha = 0.3
    if not show_y:
        fig.yaxis.visible = False
    return fig


def _toy_score_div(
    heading:  str,
    note:     str,
    mise_a:   float,
    mise_b:   float,
    ed_a:     float,
    ed_b:     float,
    label_a:  str,
    label_b:  str,
    width:    int = 230,
    height:   int = 170,
) -> Div:
    """Score table (MISE and ED\u00b2 only) for one toy example; bold marks the lower score."""
    # Square ed values so display is in log-flow units (same scale as W1).
    ed_a = ed_a ** 2
    ed_b = ed_b ** 2

    def _fmt(v: float) -> str:
        return f"{v:.4f}" if abs(v) < 0.001 else f"{v:.4f}"

    def _cell(va: float, vb: float) -> tuple:
        a, b = _fmt(va), _fmt(vb)
        if va < vb * 0.92:
            return f"<b>{a}</b>", b
        if vb < va * 0.92:
            return a, f"<b>{b}</b>"
        return a, b

    mise_a_s, mise_b_s = _cell(mise_a, mise_b)
    ed_a_s,   ed_b_s   = _cell(ed_a,   ed_b)
    html = f"""<div style="
  width:{width}px; height:{height}px; padding:10px 14px;
  font-family:'EB Garamond',Palatino,serif; background:#fffff8;
  box-sizing:border-box; font-size:12px; color:#111;
  display:flex; flex-direction:column;">
  <div style="font-weight:600; font-size:13px; margin-bottom:4px;">{heading}</div>
  <div style="color:#666; font-size:11px; margin-bottom:8px; line-height:1.4;">{note}</div>
  <table style="border-collapse:collapse; width:100%; font-size:12px;">
    <thead><tr>
      <th style="text-align:left; color:#888; font-weight:normal; padding:1px 0;"></th>
      <th style="text-align:right; color:#4915ac; font-weight:normal; padding:1px 6px;">{label_a}</th>
      <th style="text-align:right; color:#c221d1; font-weight:normal; padding:1px 0;">{label_b}</th>
    </tr></thead>
    <tbody>
      <tr style="background:#fff8e8;">
        <td style="color:#555; padding:2px 0;">MISE &#x2605;</td>
        <td style="text-align:right; padding:2px 6px;">{mise_a_s}</td>
        <td style="text-align:right; padding:2px 0;">{mise_b_s}</td>
      </tr>
      <tr>
        <td style="color:#555; padding:2px 0;">ED&#x00b2;</td>
        <td style="text-align:right; padding:2px 6px;">{ed_a_s}</td>
        <td style="text-align:right; padding:2px 0;">{ed_b_s}</td>
      </tr>
    </tbody>
  </table>
  <div style="color:#aaa; font-size:10px; margin-top:8px; line-height:1.35;">
    Bold = lower score (gap &gt; 8%).<br>
    MISE = &Sigma;(PMF&#x2009;residual)&#x00b2;.&#x2003;ED&#x00b2; = 2&middot;&Sigma;(CDF&#x2009;gap)&#x00b2;&middot;&delta;.
  </div>
</div>"""
    return Div(text=html, width=width, height=height)


def build_mise_vs_ed_section() -> dict:
    """Two examples using the Scenario 2 bimodal distribution contrasting MISE and ED.

    The true distribution is f(x) = 0.25·LN(log(0.025), 0.3) + 0.75·LN(log(0.5), 0.1),
    the same as Scenario 2 (narrow high-Q mode + broader low-Q pool in log space).

    Example 1 (same MISE, different ED):
        Both estimators displace mass ε at the narrow peak.  A deposits it at the
        adjacent bin; B deposits it at the low-Q mode (~40 bins away).  PMF residuals
        are identical in magnitude so MISE is the same.  The CDF gap for B persists
        across the full inter-modal distance, making ED ≈ 6.3× larger.

    Example 2 (MISE ratio 5:1, ED within ~4%):
        Both C and D move mass M from the low-Q pool to the narrow peak so the overall
        CDF gap profile is nearly identical and ED agrees within ~4%.  C concentrates
        the added mass at a single bin (spike); D spreads it over 5 adjacent bins.
        MISE penalises C's height concentration 5× more.
    """
    grid = _make_grid()
    dlog = np.diff(grid.log_edges)

    # Scenario 2 parameters (matches _compute_s2, without quantization artifact)
    wt1, mu1_q, sig1 = 0.25, 0.025, 0.3   # low-Q pool (broader in log space)
    wt2, mu2_q, sig2 = 0.75, 0.5,   0.1   # narrow high-Q mode
    params = [
        (wt1, np.log(mu1_q), sig1),
        (wt2, np.log(mu2_q), sig2),
    ]
    true = _mixture_pmf(grid, params)
    N    = len(true)
    bins = np.arange(N, dtype=float)

    # Bin indices at each mode's peak
    log_centers = 0.5 * (grid.log_edges[:-1] + grid.log_edges[1:])
    peak_bin = int(np.argmin(np.abs(log_centers - np.log(mu2_q))))  # narrow peak
    low_bin  = int(np.argmin(np.abs(log_centers - np.log(mu1_q))))  # low-Q mode

    # Shared x-range covering both modes
    xr = Range1d(float(low_bin - 12), float(peak_bin + 15))

    # ---- Example 1: same MISE, different ED --------------------------------
    # eps: fraction of the narrow-peak PMF.  Peak is the source so the sink bins only
    # gain mass and cannot go negative.  This allows a much larger perturbation than
    # constraining on the small pool-bin PMF.
    eps = true[peak_bin] * 0.2
    # A: -eps at peak, +eps at the adjacent bin (left of peak)
    est_a = true.copy(); est_a[peak_bin] -= eps; est_a[peak_bin - 1] += eps
    # B: -eps at peak, +eps at the low-Q mode bin (far away)
    est_b = true.copy(); est_b[peak_bin] -= eps; est_b[low_bin]      += eps

    _, _, ed_a, _, mise_a = _divergence(est_a, true, dlog)
    _, _, ed_b, _, mise_b = _divergence(est_b, true, dlog)
    res_a1 = est_a - true
    res_b1 = est_b - true
    gap_a1 = np.cumsum(est_a) - np.cumsum(true)
    gap_b1 = np.cumsum(est_b) - np.cumsum(true)

    pmf_fig1 = _toy_pmf_cdf_fig(
        bins, true, est_a, est_b, "A (adjacent)", "B (distant)",
        "S2 distribution", xr,
        show_cdf_axis=False,
    )
    res_fig1 = _toy_residual_fig(
        bins, res_a1, res_b1, "A (adjacent)", "B (distant)",
        "PMF residual  (est \u2212 true)", xr,
    )
    gap_fig1 = _toy_cdf_gap_fig(
        bins, gap_a1, gap_b1, "A (adjacent)", "B (distant)",
        "CDF gap  (CDF\u2009est \u2212 CDF\u2009true)", xr,
    )
    score1 = _toy_score_div(
        "Example 1 scores",
        f"\u2212\u03b5 at Q\u2009\u2248\u2009{mu2_q}\u2009m\u00b3/s.  "
        f"+\u03b5 at adj bin (A) or Q\u2009\u2248\u2009{mu1_q}\u2009m\u00b3/s pool (B).",
        mise_a, mise_b, ed_a, ed_b, "A", "B",
    )

    # ---- Example 2: spike vs. spread ----------------------------------------
    span = 5
    half = span // 2
    # M: fraction of the minimum PMF in the 5-bin peak window.  Peak region is the
    # source so the low-Q sink bins only gain mass (no non-negativity concern).
    M = np.min(true[peak_bin - half : peak_bin + half + 1]) * 0.25
    # C: single-bin removal at peak, single-bin addition at low-Q mode (spike)
    est_c = true.copy(); est_c[peak_bin] -= M; est_c[low_bin] += M
    # D: same mass spread over 5 bins at each location (spread)
    est_d = true.copy()
    for j in range(peak_bin - half, peak_bin + half + 1): est_d[j] -= M / span
    for j in range(low_bin  - half, low_bin  + half + 1): est_d[j] += M / span

    _, _, ed_c, _, mise_c = _divergence(est_c, true, dlog)
    _, _, ed_d, _, mise_d = _divergence(est_d, true, dlog)
    res_c2 = est_c - true
    res_d2 = est_d - true
    gap_c2 = np.cumsum(est_c) - np.cumsum(true)
    gap_d2 = np.cumsum(est_d) - np.cumsum(true)

    pmf_fig2 = _toy_pmf_cdf_fig(
        bins, true, est_c, est_d, "C (spike)", "D (spread)",
        "S2 distribution", xr,
        show_y=False
    )
    res_fig2 = _toy_residual_fig(
        bins, res_c2, res_d2, "C (spike)", "D (spread)",
        "PMF residual  (est \u2212 true)", xr,
        show_y = False,
    )
    gap_fig2 = _toy_cdf_gap_fig(
        bins, gap_c2, gap_d2, "C (spike)", "D (spread)",
        "CDF gap  (CDF\u2009est \u2212 CDF\u2009true)", xr,
        show_y = False
    )
    score2 = _toy_score_div(
        "Example 2 scores",
        f"\u2212M at Q\u2009\u2248\u2009{mu2_q}\u2009m\u00b3/s, +M at Q\u2009\u2248\u2009{mu1_q}\u2009m\u00b3/s.  "
        f"C: 1-bin spike; D: {span}-bin spread.",
        mise_c, mise_d, ed_c, ed_d, "C", "D",
    )

    layout = column(
        row(score1, score2),
        gridplot(
            [[pmf_fig1, pmf_fig2],
             [res_fig1, res_fig2],
             [gap_fig1, gap_fig2]],
            toolbar_location="right",
            merge_tools=True,
        ),
    )
    return {"mise_vs_ed_illustration": layout}


# ---------------------------------------------------------------------------
# CLI quick-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from bokeh.io import output_file, save
    from bokeh.layouts import column as bk_column
    output_file("/tmp/synthetic_test.html")
    p = build_synthetic_section()
    save(bk_column(*p.values()))
    print("Wrote /tmp/synthetic_test.html")
