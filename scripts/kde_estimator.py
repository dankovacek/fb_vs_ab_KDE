"""KDE bandwidth strategies and density estimator for log-space flow data.

Contains the Silverman rule-of-thumb bandwidth, the adaptive measurement-error
bandwidth, the kernel density evaluator, and the KDEEstimator class that
wraps both strategies for pipeline use.
"""
import os
import sys
from pathlib import Path
import numpy as np

# Add repository root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import Config


class InsufficientDataError(ValueError):
    """Raised when a station has too few unique flow values for KDE estimation."""


# ---------- Bandwidth Strategies ----------

def silverman_bandwidth(log_data: np.ndarray) -> float:
    q75, q25 = np.percentile(log_data, [75, 25])
    stdev = np.std(log_data)
    A = min(stdev, (q75 - q25) / 1.34)
    return 1.06 * A / log_data.shape[0] ** 0.2


def measurement_error_bandwidth_function(x: np.ndarray) -> np.ndarray:
    error_points = np.array(Config.KDE.ERROR_MODEL_BREAKPOINTS)
    error_values = np.array(Config.KDE.ERROR_MODEL_VALUES)
    return np.interp(x, error_points, error_values, left=1.0, right=0.25)


def adaptive_bandwidths(uar: np.ndarray, da: float) -> np.ndarray:
    """Adaptive KDE bandwidth derived from a flow-magnitude-dependent measurement
    uncertainty model. The bandwidth floor replaces Silverman's constant log-variance
    assumption with a piecewise model grounded in WSC hydrometric accuracy standards
    (ECCC, 2016; En37-464-2016). See Config.KDE for breakpoint values.
    """
    flow_data = uar * da / 1000
    unique_q = np.unique(np.array(flow_data))
    
    # compute the measurement error informed bandwidth
    # units must be volumetric flow
    error_model = measurement_error_bandwidth_function(unique_q)
    unique_UAR = (1000 / da) * unique_q
    # minimum bandwidth from measurement error model: log(1 + relative_error)
    err_widths_UAR = np.log1p(error_model)

    if len(unique_UAR) < 2:
        raise InsufficientDataError(
            f"Station has only {len(unique_UAR)} unique flow value(s); cannot estimate KDE bandwidth."
        )

    # Compute midpoints and bandwidths entirely in log space to avoid unit mixing
    log_unique_UAR = np.log(unique_UAR)
    log_midpoints = 0.5 * (log_unique_UAR[:-1] + log_unique_UAR[1:])  # midpoints in log space
    # mirror boundaries: reflect the first/last log-space gap outward
    left_mirror  = log_unique_UAR[0]  - (log_midpoints[0]  - log_unique_UAR[0])
    right_mirror = log_unique_UAR[-1] + (log_unique_UAR[-1] - log_midpoints[-1])
    log_midpoints = np.concatenate(([left_mirror], log_midpoints, [right_mirror]))
    log_diffs = np.diff(log_midpoints) / 2  # half-width of each Voronoi cell in log space
    # bandwidth = max(data-spacing half-width, measurement-error floor)
    bw_vals = np.maximum(log_diffs, err_widths_UAR)
            
    idx = np.searchsorted(unique_UAR, uar).clip(0, len(bw_vals) - 1)
    return bw_vals[idx]



def kde_kernel(log_data, bw_values, log_grid):
    H = bw_values[:, None]  # (N, 1)
    U = (log_grid[None, :] - log_data) / H  # (N, M)
    K = np.exp(-0.5 * U**2) / (H * np.sqrt(2 * np.pi))
    return K.sum(axis=0) / log_data.shape[0]


def kde_full(uar_data, da, log_x, log_w):
    bw_values = adaptive_bandwidths(uar_data, da)
    log_data = np.log(uar_data)[:, None]
    pdf = kde_kernel(log_data, bw_values, log_x)
    pdf /= np.trapezoid(pdf, x=log_x)
    pmf = pdf * log_w
    assert np.all(np.isfinite(pmf)), "KDE PMF contains non-finite values"
    assert np.all(pmf >= 0), "KDE PMF contains negative values"
    pmf /= np.sum(pmf)
    return pmf, pdf


class KDEEstimator:
    """
    Adaptive kernel density estimator using a measurement-error-informed bandwidth.

    Attributes
    ----------
    log_grid : np.ndarray
        Grid in log space over which to evaluate the KDE.
    dx : np.ndarray
        Spacing between grid points (gradient of log_grid).
    cache : dict
        Optional cache to store previously computed KDE results.
    """
    def __init__(self, log_edges):
        self.log_edges = np.asarray(log_edges, dtype=np.float32)
        # get the midpoints in log space
        self.log_x = 0.5 * (log_edges[:-1] + log_edges[1:])
        self.lin_x = np.exp(self.log_x)
        self.log_w = np.diff(log_edges) # widths of each bin in log space
        self.left_log_edges = log_edges[:-1]
        self.right_log_edges = log_edges[1:]
        self.left_lin_edges = np.exp(self.left_log_edges)
        self.right_lin_edges = np.exp(self.right_log_edges)


    def compute(self, uar_data, da):
        # Input is already filtered to positive flows by reference_distribution.py;
        # zero-flow mass is handled there. Just run KDE on the provided data.
        uar_data = np.asarray(uar_data, dtype=np.float32)
        da = float(da)
        assert np.all(np.isfinite(uar_data)), "Input UAR data contains non-finite values"
        assert np.all(uar_data > 0), "Input UAR data contains non-positive values"
        return kde_full(uar_data, da, self.log_x, self.log_w)


    def compute_silverman(self, uar_data, da):
        """KDE with a global Silverman rule-of-thumb bandwidth.

        Uses the same grid, kernel, and normalization as compute(), replacing
        the adaptive per-sample bandwidths with a single scalar bandwidth from
        Silverman's rule applied to log(uar_data): h = 1.06 * min(sigma, IQR/1.34) * n^(-1/5).
        Support handling is identical to compute().
        """
        # Input is already filtered to positive flows by reference_distribution.py;
        # zero-flow mass is handled there. Just run Silverman KDE on the provided data.
        uar_data = np.asarray(uar_data, dtype=np.float32)
        da = float(da)
        assert np.all(np.isfinite(uar_data)), "Input UAR data contains non-finite values"
        assert np.all(uar_data > 0), "Input UAR data contains non-positive values"
        log_data = np.log(uar_data)
        h_silverman = silverman_bandwidth(log_data)
        # Apply same measurement-error floor as adaptive_bandwidths to prevent h=0
        # when all remaining positive values are near-constant (e.g. ephemeral stations)
        mean_q = float(np.mean(uar_data)) * da / 1000
        bw_floor = float(np.log1p(measurement_error_bandwidth_function(mean_q)))
        h = max(h_silverman, bw_floor)
        bw_values = np.full(log_data.shape[0], h)
        pdf = kde_kernel(log_data[:, None], bw_values, self.log_x)
        pdf /= np.trapezoid(pdf, x=self.log_x)
        pmf = pdf * self.log_w
        assert np.all(np.isfinite(pmf)), "Silverman KDE PMF contains non-finite values"
        assert np.all(pmf >= 0), "Silverman KDE PMF contains negative values"
        pmf /= np.sum(pmf)
        return pmf, pdf


    def compute_both(self, uar_data, da):
        """Adaptive and Silverman PMFs sharing one log(uar_data) and diff matrix.

        Equivalent to compute() followed by compute_silverman(), but the
        log-transformed observations and (N, M) distance matrix are computed
        once and reused for both bandwidth strategies.
        Returns (pmf_adp, pmf_sil).
        """
        uar_data = np.asarray(uar_data, dtype=np.float32)
        da = float(da)
        assert np.all(np.isfinite(uar_data)), "Input UAR data contains non-finite values"
        assert np.all(uar_data > 0), "Input UAR data contains non-positive values"

        log_data = np.log(uar_data)          # (N,) -- computed once
        N = log_data.shape[0]
        diff = self.log_x[None, :] - log_data[:, None]   # (N, M) -- computed once

        # --- Adaptive ---
        bw_adp = adaptive_bandwidths(uar_data, da)   # (N,)
        H_adp = bw_adp[:, None]
        pdf_adp = (np.exp(-0.5 * (diff / H_adp) ** 2) / (H_adp * np.sqrt(2 * np.pi))).sum(axis=0) / N
        pdf_adp /= np.trapezoid(pdf_adp, x=self.log_x)
        pmf_adp = pdf_adp * self.log_w
        assert np.all(np.isfinite(pmf_adp)), "Adaptive KDE PMF contains non-finite values"
        assert np.all(pmf_adp >= 0), "Adaptive KDE PMF contains negative values"
        pmf_adp /= np.sum(pmf_adp)

        # --- Silverman (reuses diff) ---
        h_silverman = silverman_bandwidth(log_data)
        mean_q = float(np.mean(uar_data)) * da / 1000
        bw_floor = float(np.log1p(measurement_error_bandwidth_function(mean_q)))
        h = max(h_silverman, bw_floor)
        pdf_sil = (np.exp(-0.5 * (diff / h) ** 2) / (h * np.sqrt(2 * np.pi))).sum(axis=0) / N
        pdf_sil /= np.trapezoid(pdf_sil, x=self.log_x)
        pmf_sil = pdf_sil * self.log_w
        assert np.all(np.isfinite(pmf_sil)), "Silverman KDE PMF contains non-finite values"
        assert np.all(pmf_sil >= 0), "Silverman KDE PMF contains negative values"
        pmf_sil /= np.sum(pmf_sil)

        return pmf_adp, pmf_sil


    