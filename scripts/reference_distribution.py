"""Per-station PMF construction from daily discharge records.

The ReferenceDistribution class handles complete-year filtering, UAR
digitization, KDE and empirical PMF computation, KL-divergence-limited
uniform mixing, and optional log-normal MLE fitting for a single station.
"""
import numpy as np
import pandas as pd

from config import Config
from utils import filter_complete_years


class ReferenceDistribution:
    """Compute observed and KDE-smoothed PMFs for a single station's daily discharge series.

    Parameters
    ----------
    df : pd.DataFrame
        Daily timeseries with a ``"discharge"`` column (m³/s) and a DatetimeIndex.
    zero_flow_flag : bool
        True if the station has intermittent zero-flow periods.
    drainage_area_km2 : float
        Catchment drainage area in km².
    log_edges_extended : np.ndarray
        Log-space bin edges (length = nbins + 1).
    kde_estimator : KDEEstimator
        Pre-initialised estimator shared across stations.
    delta : float
        Maximum KL-divergence budget for the uniform mixture step.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        zero_flow_flag: bool,
        drainage_area_km2: float,
        log_edges_extended: np.ndarray,
        kde_estimator,
        delta: float,
    ):
        self.df                = df.copy()
        self.zero_flow_flag    = zero_flow_flag
        self.da                = drainage_area_km2
        self.log_edges_extended = log_edges_extended
        self.kde_estimator     = kde_estimator
        self.delta             = delta

        self.df["uar"] = 1000.0 * self.df["discharge"] / self.da
        self._filter_complete_hydrological_years()
        self._digitize_uar_series()

    
    def _filter_complete_hydrological_years(
        self,
        year_end: str = Config.HYD_MS,
        min_days_threshold: int = Config.MIN_DAYS_PER_MONTH,
    ) -> tuple[np.ndarray, pd.Index]:
        """Filter ``self.df`` to rows that fall within complete hydrological years.

        Parameters
        ----------
        year_end : str
            Year-end month code for pandas offsets (e.g. ``"SEP"``, ``"DEC"``).
            Defaults to ``Config.HYD_MS`` (Oct-Sep hydrological year).
        min_days_threshold : int
            Minimum number of daily observations per month for that month to be
            counted as complete.

        Returns
        -------
        mask : np.ndarray of bool
            Row-wise boolean mask aligned to ``self.df`` (sorted by index).
        complete_years : pd.Index
            Integer year labels of all complete years found.
        """
        series = self.df["uar"].sort_index()
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("_filter_complete_hydrological_years requires a DatetimeIndex.")

        monthly_counts = series.resample("MS").count()
        ok_year = (
            monthly_counts.ge(min_days_threshold)
            .groupby(pd.Grouper(freq=f"YE-{year_end}"))
            .sum()
            .eq(12)
        )
        ok_year.index = ok_year.index.to_period(f"Y-{year_end}")
        complete_years = ok_year[ok_year].index.year

        per_year = series.index.to_period(f"Y-{year_end}")
        mask = ok_year.reindex(per_year, fill_value=False).to_numpy()

        s = self.df[["discharge"]].sort_index()
        self.hyd_df = s[mask].copy()
        self.hyd_df["uar"] = 1000.0 * self.hyd_df["discharge"] / self.da
        self.hyd_df.dropna(subset=["uar"], inplace=True)

        return mask, complete_years



    def _digitize_uar_series(self):
        lin_edges_extended = np.exp(self.log_edges_extended)
        self.minimum_uar_threshold = float(1000.0 * Config.ZERO_FLOW_THRESHOLD / self.da)
        self.hyd_df["uar_bin"] = (
            np.digitize(self.hyd_df["uar"], lin_edges_extended, right=False) - 1
        )
        self.lin_x_extended = np.exp(
            0.5 * (self.log_edges_extended[1:] + self.log_edges_extended[:-1])
        )
        self.hyd_df["uar_discrete"] = self.lin_x_extended[
            self.hyd_df["uar_bin"].clip(0, np.inf).astype(int)
        ]
        assert self.hyd_df["uar_bin"].max() < len(self.lin_x_extended), (
            f"uar_bin index out of range: {self.hyd_df['uar_bin'].max()} >= {len(self.lin_x_extended)}"
        )
        self.zero_bin_index = max(
            0,
            np.digitize(self.minimum_uar_threshold, lin_edges_extended, right=False) - 1,
        )
        self.hyd_df["uar_bin_adjusted"] = self.hyd_df["uar_bin"].copy()
        self.hyd_df["uar_zero_adjusted"] = self.hyd_df["uar"].copy()

        # Step 1 (always): any observation that falls off the LEFT edge of the
        # global grid (uar=0 or sub-GLOBAL_MIN_UAR flow) produces uar_bin = -1
        # via np.digitize.  Always clip those to bin 0 so that pmf indexing is
        # safe (Python's negative indexing would otherwise place zero-flow mass
        # at the *last* bin (the highest-UAR cell), which is clearly wrong).
        self.hyd_df.loc[self.hyd_df["uar_bin_adjusted"] < 0, "uar_bin_adjusted"] = 0

        # Step 2 (DA-specific threshold, when it falls inside the grid): also
        # sweep any sub-threshold bins into bin 0 so that the OBS PMF places
        # near-zero flow in the same cell as the KDE zero-flow mass.
        if self.zero_bin_index > 0 and (self.hyd_df["uar_bin"] < self.zero_bin_index).any():
            min_uar = self.lin_x_extended[self.zero_bin_index - 1]
            self.hyd_df.loc[self.hyd_df["uar_bin"] < 0, "uar_discrete"] = np.float32(min_uar)
            self.hyd_df.loc[
                self.hyd_df["uar_bin_adjusted"] < self.zero_bin_index, "uar_bin_adjusted"
            ] = 0
            self.hyd_df.loc[
                self.hyd_df["uar_bin_adjusted"] < self.zero_bin_index, "uar_zero_adjusted"
            ] = np.float32(min_uar)


    def _compute_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """KL divergence D_KL(p || q) in bits."""
        p, q = p.astype(float), q.astype(float)
        mask = (p > 0) & (q > 0)
        terms = np.zeros_like(p)
        terms[mask] = p[mask] * (np.log2(p[mask]) - np.log2(q[mask]))
        return float(np.sum(terms[mask]))

    def _apply_zero_bin_adjustment(self, kde_counts: np.ndarray, N_n: int) -> np.ndarray:
        """Sweep sub-threshold KDE density and zero-flow counts into bin 0.

        Matches the OBS PMF convention: bins 0 through zero_bin_index-1 are
        swept into bin 0; bin zero_bin_index and above are kept as-is.
        (The OBS clips uar_bin < zero_bin_index → 0, so it keeps zero_bin_index.)
        """
        pmf = np.zeros_like(kde_counts)
        # Accumulate all sub-threshold KDE mass (bins 0..zero_bin_index-1) plus
        # the observed zero-flow count into bin 0.
        low_mass = N_n + kde_counts[: self.zero_bin_index].sum()
        pmf[self.zero_bin_index :] = kde_counts[self.zero_bin_index :]
        pmf[0] = low_mass
        return pmf

    def build_station_pmf(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (obs_pmf, kde_adaptive_pmf), each normalised to sum=1."""
        unique_bins, counts = np.unique(self.hyd_df["uar_bin_adjusted"].values, return_counts=True)
        pmf = np.zeros(len(self.lin_x_extended))
        pmf[unique_bins] = counts.astype(int)
        assert pmf.sum() == len(self.hyd_df), (
            f"PMF counts {pmf.sum()} != observations {len(self.hyd_df)}"
        )
        pmf /= pmf.sum()

        positive_values = self.hyd_df[self.hyd_df["uar"] >= self.minimum_uar_threshold]["uar"].values
        N_p, N_n = len(positive_values), len(self.hyd_df) - len(positive_values)
        pmf_kde_raw, _ = self.kde_estimator.compute(positive_values, self.da)
        kde_counts = pmf_kde_raw * N_p
        assert len(kde_counts) == len(self.lin_x_extended), (
            f"KDE counts length {len(kde_counts)} != PMF length {len(self.lin_x_extended)}"
        )

        pmf_kde = self._apply_zero_bin_adjustment(kde_counts, N_n) if N_n > 0 else pmf_kde_raw * len(self.hyd_df)
        assert np.isclose(pmf_kde.sum(), len(self.hyd_df)), (
            f"KDE counts {pmf_kde.sum()} != observations {len(self.hyd_df)} after zero-bin adjustment"
        )
        pmf_kde /= pmf_kde.sum()
        assert np.isclose(pmf_kde.sum(), 1.0), f"KDE PMF does not sum to 1: {pmf_kde.sum()}"
        assert np.isclose(pmf.sum(), 1.0),     f"Discrete PMF does not sum to 1: {pmf.sum()}"
        return pmf, pmf_kde

    def build_station_pmf_silverman(self) -> np.ndarray:
        """Return the Silverman-bandwidth KDE PMF, normalised to sum=1."""
        positive_values = self.hyd_df[self.hyd_df["uar"] >= self.minimum_uar_threshold]["uar"].values
        N_p, N_n = len(positive_values), len(self.hyd_df) - len(positive_values)
        pmf_sil_raw, _ = self.kde_estimator.compute_silverman(positive_values, self.da)
        sil_counts = pmf_sil_raw * N_p
        assert len(sil_counts) == len(self.lin_x_extended), (
            f"Silverman counts length {len(sil_counts)} != PMF length {len(self.lin_x_extended)}"
        )

        pmf_sil = self._apply_zero_bin_adjustment(sil_counts, N_n) if N_n > 0 else pmf_sil_raw * len(self.hyd_df)
        assert np.isclose(pmf_sil.sum(), len(self.hyd_df)), (
            f"Silverman PMF counts {pmf_sil.sum()} != observations {len(self.hyd_df)}"
        )
        pmf_sil /= pmf_sil.sum()
        assert np.isclose(pmf_sil.sum(), 1.0), f"Silverman PMF does not sum to 1: {pmf_sil.sum()}"
        return pmf_sil

    # ------------------------------------------------------------------
    # Uniform-mixture utilities
    # ------------------------------------------------------------------

    def D_bits_Q_to_Qlam(self, Q: np.ndarray, lam: float) -> float:
        """D_bits(Q || Q_λ) where Q_λ = (1-λ)Q + λU."""
        Q = np.clip(np.asarray(Q, dtype=float) / Q.sum(), 1e-300, 1.0)
        Qlam = (1.0 - lam) * Q + lam / Q.size
        return float(np.sum(Q * (np.log2(Q) - np.log2(Qlam))))

    def compute_optimal_delta_limited_lambda(self, Q: np.ndarray, maxit: int = 100, tol: float = 1e-6) -> float:
        """Largest λ ∈ [0,1] s.t. D_bits(Q || Q_λ) ≤ δ (monotone bisection)."""
        lo, hi = 0.0, 1.0
        if self.D_bits_Q_to_Qlam(Q, hi) <= self.delta:
            return hi
        for _ in range(maxit):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.D_bits_Q_to_Qlam(Q, mid) > self.delta else (lo, mid)
            if hi - lo < tol:
                break
        return lo

    def mix_with_uniform(self, Q: np.ndarray, lam: float) -> np.ndarray:
        """Q_λ = (1-λ)Q + λU."""
        return (1.0 - lam) * Q + lam / len(Q)

    def _compute_adjusted_distribution_with_mixed_uniform(self, pmf: np.ndarray) -> np.ndarray:
        """Mix pmf toward uniform at the largest λ that keeps D_KL ≤ delta."""
        lam = self.compute_optimal_delta_limited_lambda(pmf)
        pmf_mixed = self.mix_with_uniform(pmf, lam)
        pmf_mixed /= pmf_mixed.sum()
        assert np.isclose(pmf_mixed.sum(), 1.0), f"Mixed PMF does not sum to 1: {pmf_mixed.sum()}"
        return pmf_mixed
