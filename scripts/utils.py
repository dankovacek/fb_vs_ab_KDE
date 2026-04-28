"""Shared utilities for the KDE pipeline.

Provides parquet engine checks, ECDF computation, PMF inversion,
uniform mixture application, and hydrological year filtering.
All functions are stateless and importable from any pipeline script.
"""
"""Shared utilities for the KDE pipeline.

Provides parquet engine checks, ECDF computation, PMF inversion,
uniform mixture application, and hydrological year filtering.
All functions are stateless and importable from any pipeline script.
"""
import numpy as np
import pandas as pd
import sys
import importlib.util
import subprocess
from config import Config


def ensure_parquet_engine() -> None:
    """Ensure pandas parquet support is available before any parquet I/O.

    Prefers pyarrow, then fastparquet. In interactive runs, offers to install
    pyarrow into the current interpreter environment.
    """
    if importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet"):
        return

    msg = (
        "Parquet support is required but no engine was found.\n"
        "Install one of: pyarrow (recommended) or fastparquet.\n"
        f"Current interpreter: {sys.executable}\n"
        "Suggested commands:\n"
        "  1) uv sync   (installs pyproject dependencies, including pyarrow)\n"
        "  2) uv add pyarrow\n"
        "  3) python -m pip install pyarrow  (fallback)"
    )

    if sys.stdin.isatty():
        print(msg)
        choice = input("Install pyarrow now in this environment? [Y/n]: ").strip().lower()
        if choice in {"", "y", "yes"}:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "pyarrow"],
                    check=True,
                )
            except Exception as exc:
                raise ImportError(
                    "Failed to install pyarrow automatically. "
                    "Run: uv sync (preferred), or uv add pyarrow, "
                    "or python -m pip install pyarrow"
                ) from exc

            if importlib.util.find_spec("pyarrow"):
                return

    raise ImportError(msg)


def find_percentile_x(x_vals, y_vals, p):
    """Find x value where CDF crosses p (e.g., 0.5 for median)."""
    if len(x_vals) == 0:
        return None
    idx = np.searchsorted(y_vals, p)
    if idx >= len(x_vals):
        return x_vals[-1]
    if idx == 0:
        return x_vals[0]
    return x_vals[idx]


def compute_ecdf(values):
    """
    Compute empirical cumulative distribution function.
    
    Parameters
    ----------
    values : array-like
        Observed data values.
    
    Returns
    -------
    tuple of np.ndarray
        (sorted_values, cumulative_probabilities)
        - sorted_values: Values sorted in ascending order
        - cumulative_probabilities: ECDF values at each point (i/n for i-th sorted value)
    """
    values = np.asarray(values)
    # Remove NaN values
    values = values[~np.isnan(values)]
    n = len(values)
    
    if n == 0:
        return np.array([]), np.array([])
    
    # Sort values
    sorted_vals = np.sort(values)
    # Compute cumulative probabilities: i/n for plotting position
    cum_probs = np.arange(1, n + 1) / n
    
    return sorted_vals, cum_probs


def pmf_to_log_quantiles(
    pmf_mat: np.ndarray,
    log_x:   np.ndarray,
    probs:   np.ndarray,
) -> np.ndarray:
    """
    Invert a (B, N) column-normalised PMF array to log-quantile values at each
    probability level in probs via piecewise-linear CDF interpolation.

    The CDF is evaluated at bin midpoints (cdf - 0.5*pmf) rather than right
    edges so that each bin's probability mass is centred on log_x[j],
    eliminating the ~half-bin-width systematic downward bias.

    Returns (L, N) in the same units as log_x.
    """
    cdf     = np.cumsum(pmf_mat, axis=0)
    cdf    /= cdf[-1:, :]          # guard against float drift
    cdf_mid = cdf - 0.5 * pmf_mat  # mid-bin CDF: centres each bin on log_x[j]
    return np.column_stack(
        [np.interp(probs, cdf_mid[:, n], log_x) for n in range(pmf_mat.shape[1])]
    )  # (L, N)


def apply_kld_limited_uniform_mixture(pmf: np.ndarray, delta: float) -> np.ndarray:
    """
    Mix pmf toward uniform using the largest lambda that keeps
    D_bits(pmf || (1-lambda)*pmf + lambda*uniform) <= delta.

    Extracted from FDC_Data.compute_optimal_delta_limited_lambda /
    mix_with_uniform so it can be used without a full estimator object.

    Parameters
    ----------
    pmf : np.ndarray
        Probability mass function (must sum to 1, non-negative).
    delta : float
        Maximum allowed KL divergence in bits (e.g. Config.Metrics.KLD_DELTA_MAX).

    Returns
    -------
    np.ndarray
        Mixed PMF, renormalised to sum to 1.
    """
    pmf = np.asarray(pmf, dtype=float)
    pmf = np.clip(pmf / pmf.sum(), 1e-300, 1.0)
    N = pmf.size
    U = 1.0 / N

    def _d_bits(lam: float) -> float:
        q = (1.0 - lam) * pmf + lam * U
        return float(np.sum(pmf * (np.log2(pmf) - np.log2(q))))

    lo, hi = 0.0, 1.0
    if _d_bits(hi) > delta:
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if _d_bits(mid) <= delta:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-6:
                break
        lam_opt = lo
    else:
        lam_opt = hi

    mixed = (1.0 - lam_opt) * pmf + lam_opt * U
    mixed /= mixed.sum()
    return mixed


def filter_complete_years(
    series: pd.Series,
    year_end: str,
    min_days_per_month: int,
) -> tuple[np.ndarray, pd.Index]:
    """Return a boolean row-mask and year labels for complete hydrological years.

    A year is *complete* when every calendar month within it has at least
    ``min_days_per_month`` valid (non-NaN) daily observations.

    Parameters
    ----------
    series : pd.Series
        Daily timeseries with a :class:`pandas.DatetimeIndex`.
    year_end : str
        Pandas year-end month code (e.g. ``"SEP"`` for an Oct-Sep
        hydrological year, ``"DEC"`` for a calendar year).
    min_days_per_month : int
        Minimum number of valid observations required per month.

    Returns
    -------
    mask : np.ndarray of bool
        Boolean array aligned to *series* (after ``sort_index``); ``True``
        for rows that fall inside a complete year.
    complete_years : pd.Index
        Integer year labels of all complete years found.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("filter_complete_years requires a DatetimeIndex.")

    s = series.sort_index()
    ok_year = (
        s.resample("MS")
        .count()
        .ge(min_days_per_month)
        .groupby(pd.Grouper(freq=f"YE-{year_end}"))
        .sum()
        .eq(12)
    )
    ok_year.index = ok_year.index.to_period(f"Y-{year_end}")
    complete_years = ok_year[ok_year].index.year
    per_year = s.index.to_period(f"Y-{year_end}")
    mask = ok_year.reindex(per_year, fill_value=False).to_numpy()
    return mask, complete_years