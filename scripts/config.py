"""
Comprehensive Configuration for Flow Duration Curve Distribution Estimation

This module consolidates all configuration constants, assumptions, and parameters
used throughout the distribution estimation pipeline. All hardcoded values should
be defined here to maintain a single source of truth.

Usage:
    from config import Config
    
    bitrates = Config.SUPPORTED_BITRATES
    min_uar = Config.GLOBAL_MIN_UAR
"""

import os
from pathlib import Path
from typing import Union, Optional
import numpy as np


class Config:
    """
    Centralized configuration for FDC estimation pipeline.
    All hardcoded constants should be defined here.
    """
    
    # ==================== DATA PATHS ====================
    DATA_DIR = Path('data')
    BASELINE_DIST_DIR = DATA_DIR / 'baseline_distributions/'
    
    # ==================== DISCRETIZATION ====================
    DEFAULT_BITRATE = 8
    SUPPORTED_BITRATES = [6, 7, 8, 10]

    @staticmethod
    def n_bins(bitrate: Optional[int] = None) -> int:
        """Calculate number of bins from bitrate."""
        if bitrate is None:
            bitrate = Config.DEFAULT_BITRATE
        return 2 ** bitrate
    
    # ==================== TEMPORAL PARAMETERS ====================
    MIN_YEARS_OF_RECORD = 5  # years
    MIN_DAYS_PER_MONTH = 20  # days (for complete month)

    # Hydrological year definition
    HYDRO_YEAR_START_MONTH = 10  # October
    HYDRO_YEAR_START_DAY = 1
    HYD_MS = 'SEP'  # Hydrological year end month (Oct-Sep cycle)
    
    # ==================== FLOW THRESHOLDS ====================
    ZERO_FLOW_THRESHOLD = 1e-4  # m³/s
    
    # UAR (Unit Area Runoff) bounds
    GLOBAL_MIN_UAR = 5e-5  # L/s/km²
    GLOBAL_MAX_UAR = 1e4   # L/s/km²

    # Smoothing/numerical stability
    # KLD_SMOOTHING_CONSTANT = 1e-10
    
    # ==================== KDE PARAMETERS ====================
    class KDE:
        """KDE-specific configuration parameters."""
        # Minimum bandwidth floor as a fraction of flow magnitude (m³/s).
        # Flat 5% band (0.1-100 m³/s) = WSC nominal measurement accuracy (ECCC, 2016;
        # En37-464-2016). Rises at extremes: low flows are poorly constrained by the
        # stage-discharge relation; high flows are typically extrapolated beyond the
        # rating curve calibration range. This is a physical prior on smoothing width,
        # not an MISE-optimal bandwidth.
        ERROR_MODEL_BREAKPOINTS = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.,
                                            1e1, 1e2, 1e3, 1e4, 1e5])
        ERROR_MODEL_VALUES = np.array([1.0, 0.5, 0.2, 0.05, 0.05,
                                       0.05, 0.05, 0.1, 0.15, 0.2])
        
        # Silverman bandwidth coefficient
        SILVERMAN_COEF = 1.06
        SILVERMAN_EXPONENT = 0.2
        
        # Log scaling options
        LOG_DIFF_SCALING = 2.0  # Divisor for log differences
        LOG_DIFF_SCALING_ALT = 1.15  # Alternative (commented in original code)
        
        # Regularization mode
        REGULARIZATION_TYPES = ['kde', 'discrete']
        DEFAULT_REGULARIZATION = 'kde'
        
        # Kernel options
        AVAILABLE_KERNELS = ['gaussian', 'epanechnikov', 'top_hat']
        DEFAULT_KERNEL = 'gaussian'
    
    # ==================== STATION FILTERING ====================
    # Stations to exclude (regulated/problematic) - from diagnostic page processing
    EXCLUDED_STATIONS = [
        '08FA009', '08GA037', '08NC003', '12052500', '12090480', '12107950', 
        '12108450', '12119300', '12119450', '12200684', '12200762', '12203000', 
        '12409500', '15056070', '15081510', '12323760', '12143700', '12143900', 
        '12398000', '12058800', '12137800', '12100000',
        '08HB075', '10ED009', '08HA026', '12202310', '10CD005', '12202420'
    ]
    
    # Caravan/HydroATLAS variant: same attributes, but land use columns are
    # unversioned (no _2010 suffix) as produced by _CARAVAN_COL_MAP renaming.
    CARAVAN_DESCRIPTOR_COLS = [
        # climate
        'prcp', 'tmean', 'swe',
        'high_prcp_freq', 'low_prcp_freq', 'high_prcp_duration', 'low_prcp_duration',
        # terrain
        'log_drainage_area_km2', 'slope_deg', 'elevation_m',
        # land use
        'land_use_forest_frac', 'land_use_snow_ice_frac',
    ]
    
    # Terrain attributes for analysis
    # TERRAIN_ATTRIBUTES = [
    #     'slope_deg', 'aspect_deg', 'elevation_m', 'log_drainage_area_km2'
    # ]
    
    # ==================== EVALUATION METRICS ====================
    class Metrics:
        """Evaluation metric configuration and thresholds."""
        # Metric tolerance limits (thresholds for "perfect")
        LIMITS = {
            'kld': 0.001,
            'emd': 0.05,  # L/s/km²
            'nse': 1 - 0.001,  # Flipped: 1.0 is perfect
            'kge': 1 - 0.001,  # Flipped: 1.0 is perfect
            'mean_error': 0.01,
            'pct_vol_bias': 0.01,
            'mean_abs_rel_error': 0.01,
            'rmse': 0.01
        }
        
        # KL divergence delta (max uncertainty from uniform mixture)
        KLD_DELTA_MAX = 0.001
    
    # ==================== MIXTURE MODELS ====================
    UNIFORM_MIXTURE_EPSILON = 0.01  # For PMF smoothing
    UNIFORM_MIXTURE_DELTA = 0.001   # Default delta for KL-constrained mixture

    # ==================== GRID TRIMMING ====================
    PROBE_ALPHA = 1e-3  # Edge-trimming fraction for the quantile evaluation grid
    PROBE_PROBS = np.linspace(PROBE_ALPHA, 1.0 - PROBE_ALPHA, 100)

    # ==================== UNIT CONVERSIONS ====================
    @staticmethod
    def flow_to_uar(flow_m3s: Union[float, np.ndarray], drainage_area_km2: float) -> Union[float, np.ndarray]:
        """
        Convert flow (m³/s) to unit area runoff (L/s/km²).
        
        Parameters
        ----------
        flow_m3s : float or array-like
            Flow in m³/s
        drainage_area_km2 : float
            Drainage area in km²
            
        Returns
        -------
        float or array-like
            Unit area runoff in L/s/km²
        """
        return 1000.0 * flow_m3s / drainage_area_km2
    
    @staticmethod
    def uar_to_flow(uar_l_s_km2: Union[float, np.ndarray], drainage_area_km2: float) -> Union[float, np.ndarray]:
        """
        Convert unit area runoff (L/s/km²) to flow (m³/s).
        
        Parameters
        ----------
        uar_l_s_km2 : float or array-like
            Unit area runoff in L/s/km²
        drainage_area_km2 : float
            Drainage area in km²
            
        Returns
        -------
        float or array-like
            Flow in m³/s
        """
        return uar_l_s_km2 * drainage_area_km2 / 1000.0
