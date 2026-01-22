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
from .config import EPSILON, THRESHOLDS
from .data_io import detect_band_mode_from_dataframe

logger = logging.getLogger(__name__)


class SpectralFeatures:
    """Extract spectral indices and statistical features from transect data."""

    def __init__(self, spectral_data: pd.DataFrame, band_mode: str = None):
        """
        Initialize with spectral data.

        Args:
            spectral_data: DataFrame with columns [distance, red, green, blue, nir]
                          (nir may be NaN for 3-band RGB imagery)
            band_mode: Optional band mode string ('4band', 'cir', 'rgb')
                      If None, auto-detected from data
        """
        self.data = spectral_data.copy()
        self.window_size = THRESHOLDS['window_size']
        
        # Detect band mode if not provided
        if band_mode is None:
            self.band_mode = self._detect_band_mode()
        else:
            self.band_mode = band_mode
        
        # Check NIR availability
        self.has_nir = self._check_nir_available()
        self.has_blue = self._check_blue_available()
        
        logger.debug(f"SpectralFeatures initialized: band_mode={self.band_mode}, "
                    f"has_nir={self.has_nir}, has_blue={self.has_blue}")

        # Validate required columns
        required = ['distance', 'red', 'green', 'blue']
        if not all(col in self.data.columns for col in required):
            raise ValueError(f"Data must contain columns: {required}")
        
        # NIR is optional for 3-band data
        if 'nir' not in self.data.columns:
            self.data['nir'] = np.nan

    def _detect_band_mode(self) -> str:
        """Auto-detect band mode from data contents."""
        return detect_band_mode_from_dataframe(self.data)

    def _check_nir_available(self) -> bool:
        """Check if NIR band has valid (non-NaN) values."""
        if 'nir' not in self.data.columns:
            return False
        return not self.data['nir'].isna().all()
    
    def _check_blue_available(self) -> bool:
        """Check if Blue band has valid (non-NaN) values."""
        if 'blue' not in self.data.columns:
            return False
        return not self.data['blue'].isna().all()

    def compute_all(self) -> pd.DataFrame:
        """
        Compute all features and return as DataFrame.
        
        For RGB-only imagery, NIR-dependent features are either:
        - Set to NaN (for indices like NDVI, NDWI)
        - Replaced with RGB-based alternatives (for derivatives)

        Returns:
            DataFrame with original data plus all computed features
        """
        logger.debug(f"Computing spectral features (band_mode={self.band_mode}, has_nir={self.has_nir})")

        # Store band mode in DataFrame for downstream use
        self.data['band_mode'] = self.band_mode

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
    # NIR-based indices (require NIR)
    # ========================================================================

    def _compute_ndvi(self) -> pd.Series:
        """Normalized Difference Vegetation Index."""
        nir = self.data['nir']
        red = self.data['red']
        return (nir - red) / (nir + red + EPSILON)

    def _compute_ndwi(self) -> pd.Series:
        """Normalized Difference Water Index."""
        green = self.data['green']
        nir = self.data['nir']
        return (green - nir) / (green + nir + EPSILON)

    def _compute_nir_ratio(self) -> pd.Series:
        """NIR prominence relative to visible bands."""
        visible_mean = self.data[['red', 'green', 'blue']].mean(axis=1)
        return self.data['nir'] / (visible_mean + EPSILON)

    # ========================================================================
    # RGB-based indices (always available)
    # ========================================================================

    def _compute_brightness(self) -> pd.Series:
        """Overall reflectance (mean of all available bands)."""
        if self.has_nir:
            return self.data[['red', 'green', 'blue', 'nir']].mean(axis=1)
        else:
            return self._compute_brightness_rgb()
    
    def _compute_brightness_rgb(self) -> pd.Series:
        """RGB-only brightness (excludes NIR)."""
        return self.data[['red', 'green', 'blue']].mean(axis=1)

    def _compute_blue_red_ratio(self) -> pd.Series:
        """Blue to red ratio (water indicator)."""
        if not self.has_blue:
            return pd.Series(np.nan, index=self.data.index)
        return self.data['blue'] / (self.data['red'] + EPSILON)

    def _compute_red_green_ratio(self) -> pd.Series:
        """
        Red to green ratio - diagnostic for beach zone classification.

        Expected patterns:
        - VEGETATED_DUNES: Variable, generally R > G (ratio > 1.0)
        - BEACH_DRY: Generally R ~ G (ratio ~ 1.0)
        - BEACH_WET: Generally R > G (ratio > 1.0)
        - WATER: Generally R < G (ratio < 1.0), except wave crests

        Returns:
            Series with red/green ratio values
        """
        return self.data['red'] / (self.data['green'] + EPSILON)

    # ========================================================================
    # NIR derivatives
    # ========================================================================

    def _compute_nir_derivative(self) -> pd.Series:
        """
        First derivative of NIR band (rate of change).

        Primary signal for boundary detection in 4-band mode.
        Negative values indicate NIR decreasing (land->water direction).

        Returns:
            Series with NIR derivative in units per meter
        """
        nir = self.data['nir'].values
        distance = self.data['distance'].values
        nir_d1 = np.gradient(nir, distance)
        return pd.Series(nir_d1, index=self.data.index)

    def _compute_nir_derivative_smooth(self) -> pd.Series:
        """Smoothed NIR first derivative to reduce noise."""
        from scipy.ndimage import uniform_filter1d
        nir_d1 = self._compute_nir_derivative().values
        nir_d1_smooth = uniform_filter1d(nir_d1, size=3, mode='nearest')
        return pd.Series(nir_d1_smooth, index=self.data.index)

    def _compute_nir_derivative_multiscale(self) -> Dict[str, pd.Series]:
        """Compute NIR derivatives at multiple smoothing scales."""
        from scipy.ndimage import uniform_filter1d
        
        scales = [5, 7, 9, 11]
        derivatives = {}
        
        for window in scales:
            nir_smooth = uniform_filter1d(
                self.data['nir'].values,
                size=window,
                mode='nearest'
            )
            nir_d1 = np.gradient(nir_smooth, self.data['distance'].values)
            derivatives[f'nir_d1_w{window}'] = pd.Series(nir_d1, index=self.data.index)
        
        return derivatives

    def _compute_nir_derivative_second(self, window_size: int = 7) -> pd.Series:
        """Second derivative of NIR (curvature/inflection detection)."""
        from scipy.ndimage import uniform_filter1d
        
        nir_smooth = uniform_filter1d(
            self.data['nir'].values,
            size=window_size,
            mode='nearest'
        )
        distances = self.data['distance'].values
        nir_d1 = np.gradient(nir_smooth, distances)
        nir_d2 = np.gradient(nir_d1, distances)
        return pd.Series(nir_d2, index=self.data.index)

    # ========================================================================
    # Brightness derivatives (primary for RGB mode)
    # ========================================================================

    def _compute_brightness_derivative(self) -> pd.Series:
        """First derivative of brightness for confirmation."""
        brightness = self._compute_brightness().values
        distance = self.data['distance'].values
        brightness_d1 = np.gradient(brightness, distance)
        return pd.Series(brightness_d1, index=self.data.index)
    
    def _compute_brightness_derivative_smooth(self) -> pd.Series:
        """Smoothed brightness derivative."""
        from scipy.ndimage import uniform_filter1d
        brightness_d1 = self._compute_brightness_derivative().values
        brightness_d1_smooth = uniform_filter1d(brightness_d1, size=3, mode='nearest')
        return pd.Series(brightness_d1_smooth, index=self.data.index)

    def _compute_brightness_rgb_derivative(self) -> pd.Series:
        """First derivative of RGB-only brightness."""
        brightness_rgb = self._compute_brightness_rgb().values
        distance = self.data['distance'].values
        brightness_d1 = np.gradient(brightness_rgb, distance)
        return pd.Series(brightness_d1, index=self.data.index)
    
    def _compute_brightness_rgb_derivative_smooth(self) -> pd.Series:
        """Smoothed RGB-only brightness derivative."""
        from scipy.ndimage import uniform_filter1d
        brightness_d1 = self._compute_brightness_rgb_derivative().values
        brightness_d1_smooth = uniform_filter1d(brightness_d1, size=3, mode='nearest')
        return pd.Series(brightness_d1_smooth, index=self.data.index)

    def _compute_brightness_rgb_derivative_multiscale(self) -> Dict[str, pd.Series]:
        """Compute RGB brightness derivatives at multiple smoothing scales."""
        from scipy.ndimage import uniform_filter1d
        
        scales = [5, 7, 9, 11]
        derivatives = {}
        brightness_rgb = self._compute_brightness_rgb().values
        
        for window in scales:
            brightness_smooth = uniform_filter1d(
                brightness_rgb,
                size=window,
                mode='nearest'
            )
            brightness_d1 = np.gradient(brightness_smooth, self.data['distance'].values)
            # Store with 'brightness_rgb_d1_w' prefix for RGB mode
            derivatives[f'brightness_rgb_d1_w{window}'] = pd.Series(brightness_d1, index=self.data.index)
        
        return derivatives

    def _compute_brightness_rgb_derivative_second(self, window_size: int = 7) -> pd.Series:
        """Second derivative of RGB brightness (curvature/inflection detection)."""
        from scipy.ndimage import uniform_filter1d
        
        brightness_rgb = self._compute_brightness_rgb().values
        brightness_smooth = uniform_filter1d(
            brightness_rgb,
            size=window_size,
            mode='nearest'
        )
        distances = self.data['distance'].values
        brightness_d1 = np.gradient(brightness_smooth, distances)
        brightness_d2 = np.gradient(brightness_d1, distances)
        return pd.Series(brightness_d2, index=self.data.index)

    # ========================================================================
    # RGB band derivatives
    # ========================================================================

    def _compute_red_derivative(self) -> pd.Series:
        """First derivative of red band (rate of change)."""
        red = self.data['red'].values
        distance = self.data['distance'].values
        red_d1 = np.gradient(red, distance)
        return pd.Series(red_d1, index=self.data.index)

    def _compute_red_derivative_smooth(self) -> pd.Series:
        """Smoothed red first derivative to reduce noise."""
        from scipy.ndimage import uniform_filter1d
        red_d1 = self._compute_red_derivative().values
        red_d1_smooth = uniform_filter1d(red_d1, size=3, mode='nearest')
        return pd.Series(red_d1_smooth, index=self.data.index)

    def _compute_green_derivative(self) -> pd.Series:
        """First derivative of green band (rate of change)."""
        green = self.data['green'].values
        distance = self.data['distance'].values
        green_d1 = np.gradient(green, distance)
        return pd.Series(green_d1, index=self.data.index)

    def _compute_green_derivative_smooth(self) -> pd.Series:
        """Smoothed green first derivative to reduce noise."""
        from scipy.ndimage import uniform_filter1d
        green_d1 = self._compute_green_derivative().values
        green_d1_smooth = uniform_filter1d(green_d1, size=3, mode='nearest')
        return pd.Series(green_d1_smooth, index=self.data.index)

    def _compute_blue_derivative(self) -> pd.Series:
        """First derivative of blue band (rate of change)."""
        blue = self.data['blue'].values
        distance = self.data['distance'].values
        blue_d1 = np.gradient(blue, distance)
        return pd.Series(blue_d1, index=self.data.index)

    def _compute_blue_derivative_smooth(self) -> pd.Series:
        """Smoothed blue first derivative to reduce noise."""
        from scipy.ndimage import uniform_filter1d
        blue_d1 = self._compute_blue_derivative().values
        blue_d1_smooth = uniform_filter1d(blue_d1, size=3, mode='nearest')
        return pd.Series(blue_d1_smooth, index=self.data.index)

    # ========================================================================
    # R/G ratio derivatives (Phase 1)
    # ========================================================================

    def _compute_rg_ratio_derivative(self) -> pd.Series:
        """First derivative of R/G ratio (rate of change)."""
        rg_ratio = self._compute_red_green_ratio().values
        distance = self.data['distance'].values
        rg_d1 = np.gradient(rg_ratio, distance)
        return pd.Series(rg_d1, index=self.data.index)

    def _compute_rg_ratio_derivative_smooth(self) -> pd.Series:
        """Smoothed R/G ratio derivative to reduce noise."""
        from scipy.ndimage import uniform_filter1d
        rg_d1 = self._compute_rg_ratio_derivative().values
        rg_d1_smooth = uniform_filter1d(rg_d1, size=5, mode='nearest')
        return pd.Series(rg_d1_smooth, index=self.data.index)

    # ========================================================================
    # Statistical features (sliding window)
    # ========================================================================

    def _compute_variability(self) -> pd.Series:
        """Standard deviation in sliding window."""
        brightness = self._compute_brightness()
        return brightness.rolling(
            window=self.window_size,
            center=True,
            min_periods=1
        ).std()

    def _compute_slope(self) -> pd.Series:
        """Linear regression slope in sliding window."""
        brightness = self._compute_brightness()
        slopes = []

        for i in range(len(brightness)):
            start = max(0, i - self.window_size // 2)
            end = min(len(brightness), i + self.window_size // 2 + 1)

            window_data = brightness.iloc[start:end]
            window_x = np.arange(len(window_data))

            if len(window_data) < 3:
                slopes.append(0.0)
            else:
                try:
                    result = linregress(window_x, window_data)
                    slopes.append(result.slope)
                except:
                    slopes.append(0.0)

        return pd.Series(slopes, index=brightness.index)

    def _compute_curvature(self) -> pd.Series:
        """Second derivative approximation."""
        brightness = self._compute_brightness()
        first_deriv = np.gradient(brightness)
        second_deriv = np.gradient(first_deriv)
        return pd.Series(second_deriv, index=brightness.index)

    def _compute_local_range(self) -> pd.Series:
        """Max - min in sliding window."""
        brightness = self._compute_brightness()

        local_max = brightness.rolling(
            window=self.window_size,
            center=True,
            min_periods=1
        ).max()

        local_min = brightness.rolling(
            window=self.window_size,
            center=True,
            min_periods=1
        ).min()

        return local_max - local_min

    # ========================================================================
    # Shape features
    # ========================================================================

    def _compute_spectral_angle(self) -> pd.Series:
        """
        Spectral angle between consecutive points.
        
        For RGB-only mode, uses 3-band vectors instead of 4-band.
        """
        angles = [0.0]  # First point has no predecessor

        for i in range(1, len(self.data)):
            if self.has_nir:
                # 4-band spectral vector
                vec1 = self.data.iloc[i-1][['red', 'green', 'blue', 'nir']].values.astype(np.float64)
                vec2 = self.data.iloc[i][['red', 'green', 'blue', 'nir']].values.astype(np.float64)
            else:
                # 3-band spectral vector (RGB only)
                vec1 = self.data.iloc[i-1][['red', 'green', 'blue']].values.astype(np.float64)
                vec2 = self.data.iloc[i][['red', 'green', 'blue']].values.astype(np.float64)

            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 > 0 and norm2 > 0:
                cos_angle = dot_product / (norm1 * norm2 + EPSILON)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
            else:
                angle = 0.0

            angles.append(angle)

        return pd.Series(angles, index=self.data.index)

    def _detect_oscillations(self) -> pd.Series:
        """
        Detect presence of oscillations using peak detection.
        Returns boolean series indicating oscillatory regions.
        """
        brightness = self._compute_brightness()

        peaks, _ = signal.find_peaks(
            brightness,
            distance=2,
            prominence=5
        )

        has_osc = np.zeros(len(brightness), dtype=bool)

        for peak_idx in peaks:
            start = max(0, peak_idx - self.window_size // 2)
            end = min(len(brightness), peak_idx + self.window_size // 2)
            has_osc[start:end] = True

        return pd.Series(has_osc, index=brightness.index)

    def _detect_rgb_foam_peaks(self) -> pd.Series:
        """
        Detect RGB peaks indicating foam from breaking surf.
        
        PHASE 2 FIX: Stricter prominence threshold to reduce wave crest false positives

        Returns:
            Boolean series indicating foam peak locations
        """
        from scipy.ndimage import uniform_filter1d

        rgb_avg = self.data[['red', 'green', 'blue']].mean(axis=1)
        rgb_smooth = uniform_filter1d(rgb_avg.values, size=7, mode='nearest')

        prominence = THRESHOLDS.get('boundary_thresholds', {}).get(
            'surf_zone', {}
        ).get('rgb_peak_prominence', 10.0)

        peaks, properties = signal.find_peaks(
            rgb_smooth,
            prominence=prominence,
            width=2
        )

        has_foam = pd.Series(False, index=self.data.index)
        has_foam.iloc[peaks] = True

        logger.debug(f"Detected {len(peaks)} RGB foam peaks (prominence >= {prominence})")

        return has_foam

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