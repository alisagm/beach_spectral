"""
Feature extraction module for spectral transect analysis.

Supports 3-band (RGB or CIR) and 4-band (RGBN) imagery with automatic
detection and conditional feature computation.
"""

import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import linregress
from spectral_classifier.config import EPSILON, THRESHOLDS
from ..utils import band_mode_from_indices

logger = logging.getLogger(__name__)

def compute_all(data: pd.DataFrame, band_indices: dict) -> pd.DataFrame:
    """
    Compute all spectral features for a single transect profile.

    Args:
        data:         DataFrame with columns [distance, red, green, blue, nir].
                      nir column must be present but may be NaN for RGB imagery.
        band_indices: Output of resolve_band_indices(year_config). Determines
                      which features are computed — no column scanning performed.
    Returns:
        Input DataFrame with all feature columns appended.
    """
    result = data.copy()
    has_nir  = band_indices["nir"]  is not None
    has_blue = band_indices["blue"] is not None
    band_mode = band_mode_from_indices(band_indices)

    result["band_mode"] = band_mode
    # ... delegate to spectral_indices, derivatives, window_features, detection_features
    return result

class SpectralFeatures:
    """
    Compatibility shim for existing callers. Prefer compute_all() directly.

    Callers should pass band_indices (from resolve_band_indices) rather than
    band_mode. Passing band_mode as a string is still accepted for now but
    bypasses config validation.
    """
    def __init__(self, spectral_data: pd.DataFrame,
                 band_indices: dict = None,
                 band_mode: str = None):   # kept for backward compat only
        self.data = spectral_data.copy()
        if band_indices is not None:
            self._band_indices = band_indices
        elif band_mode is not None:
            # Reconstruct a minimal band_indices from band_mode string
            # so compute_all receives a consistent interface
            self._band_indices = _band_indices_from_mode(band_mode)
        else:
            raise ValueError(
                "Provide band_indices (preferred) or band_mode (legacy)."
            )

    def compute_all(self):
        return compute_all(self.data, self._band_indices)
        


        # ================================================================
        # Spectral indices - some require NIR
        # ================================================================
        
        if self.has_nir:
            self.data['ndvi'] = self._compute_ndvi()
            self.data['ndwi'] = self._compute_ndwi()
            self.data['nir_ratio'] = self._compute_nir_ratio()
        else:
            # RGB-only: set NIR-dependent indices to NaN
            self.data['ndvi'] = np.nan
            self.data['ndwi'] = np.nan
            self.data['nir_ratio'] = np.nan
            logger.debug("NIR not available - NDVI, NDWI, nir_ratio set to NaN")
        
        # RGB-based indices (always computed)
        self.data['brightness'] = self._compute_brightness()
        self.data['blue_red_ratio'] = self._compute_blue_red_ratio()
        self.data['red_green_ratio'] = self._compute_red_green_ratio()
        
        # RGB-only brightness (excludes NIR even when available)
        self.data['brightness_rgb'] = self._compute_brightness_rgb()

        # ================================================================
        # Derivative features - PRIMARY for boundary detection
        # ================================================================
        
        if self.has_nir:
            # NIR derivatives (primary signal for 4-band)
            self.data['nir_d1'] = self._compute_nir_derivative()
            self.data['nir_d1_smooth'] = self._compute_nir_derivative_smooth()
        else:
            # RGB-only: use brightness derivative as primary signal
            self.data['nir_d1'] = np.nan
            self.data['nir_d1_smooth'] = np.nan
            logger.debug("NIR not available - using brightness derivatives as primary signal")
        
        # Brightness derivative (always computed, primary for RGB mode)
        self.data['brightness_d1'] = self._compute_brightness_derivative()
        self.data['brightness_d1_smooth'] = self._compute_brightness_derivative_smooth()
        
        # RGB brightness derivative (excludes NIR)
        self.data['brightness_rgb_d1'] = self._compute_brightness_rgb_derivative()
        self.data['brightness_rgb_d1_smooth'] = self._compute_brightness_rgb_derivative_smooth()

        # Multi-band derivatives for consensus detection (always computed)
        self.data['red_d1'] = self._compute_red_derivative()
        self.data['red_d1_smooth'] = self._compute_red_derivative_smooth()
        self.data['green_d1'] = self._compute_green_derivative()
        self.data['green_d1_smooth'] = self._compute_green_derivative_smooth()
        
        if self.has_blue:
            self.data['blue_d1'] = self._compute_blue_derivative()
            self.data['blue_d1_smooth'] = self._compute_blue_derivative_smooth()
        else:
            self.data['blue_d1'] = np.nan
            self.data['blue_d1_smooth'] = np.nan

        # R/G ratio derivatives for shell-line detection (always available)
        self.data['rg_ratio_d1'] = self._compute_rg_ratio_derivative()
        self.data['rg_ratio_d1_smooth'] = self._compute_rg_ratio_derivative_smooth()

        # ================================================================
        # Multi-scale derivatives (Phase 2A)
        # ================================================================
        
        if self.has_nir:
            multiscale = self._compute_nir_derivative_multiscale()
            for key, series in multiscale.items():
                self.data[key] = series
        else:
            # RGB-only: compute brightness-based multiscale derivatives
            multiscale_rgb = self._compute_brightness_rgb_derivative_multiscale()
            for key, series in multiscale_rgb.items():
                self.data[key] = series
            # Set NIR multiscale to NaN for compatibility
            for w in [5, 7, 9, 11]:
                self.data[f'nir_d1_w{w}'] = np.nan

        # ================================================================
        # Second derivatives (Phase 2A) - for inflection detection
        # ================================================================
        
        if self.has_nir:
            self.data['nir_d2_w5'] = self._compute_nir_derivative_second(window_size=5)
            self.data['nir_d2_w7'] = self._compute_nir_derivative_second(window_size=7)
            self.data['nir_d2_w9'] = self._compute_nir_derivative_second(window_size=9)
        else:
            # RGB-only: use brightness second derivatives
            self.data['nir_d2_w5'] = self._compute_brightness_rgb_derivative_second(window_size=5)
            self.data['nir_d2_w7'] = self._compute_brightness_rgb_derivative_second(window_size=7)
            self.data['nir_d2_w9'] = self._compute_brightness_rgb_derivative_second(window_size=9)

        # ================================================================
        # RGB foam peak detection (Phase 2A) - always available
        # ================================================================
        self.data['has_rgb_foam_peak'] = self._detect_rgb_foam_peaks()

        # ================================================================
        # Statistical features (sliding window) - always available
        # ================================================================
        self.data['variability'] = self._compute_variability()
        self.data['slope'] = self._compute_slope()
        self.data['curvature'] = self._compute_curvature()
        self.data['local_range'] = self._compute_local_range()

        # ================================================================
        # Shape features
        # ================================================================
        self.data['spectral_angle'] = self._compute_spectral_angle()
        self.data['has_oscillations'] = self._detect_oscillations()

        # Count computed features
        n_features = len(self.data.columns) - 6  # Subtract original columns + band_mode
        logger.info(f"Computed {n_features} features (band_mode={self.band_mode})")

        return self.data

    # ========================================================================
    # Summary methods
    # ========================================================================

    def get_feature_summary(self) -> Dict:
        """Get summary statistics of computed features."""
        if 'brightness' not in self.data.columns:
            raise ValueError("Features not computed yet. Call compute_all() first.")

        # Define features to summarize (only those that are computed)
        feature_cols = ['brightness', 'blue_red_ratio', 'red_green_ratio',
                       'variability', 'slope', 'curvature', 'local_range', 'spectral_angle']
        
        if self.has_nir:
            feature_cols.extend(['ndvi', 'ndwi', 'nir_ratio'])

        summary = {
            'band_mode': self.band_mode,
            'has_nir': self.has_nir,
            'has_blue': self.has_blue,
            'n_samples': len(self.data)
        }
        
        for col in feature_cols:
            if col in self.data.columns:
                summary[col] = {
                    'mean': self.data[col].mean(),
                    'std': self.data[col].std(),
                    'min': self.data[col].min(),
                    'max': self.data[col].max()
                }

        return summary