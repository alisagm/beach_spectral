"""
Feature extraction module for spectral transect analysis.
"""

import logging
from typing import Dict
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import linregress
from .config import EPSILON, THRESHOLDS

logger = logging.getLogger(__name__)


class SpectralFeatures:
    """Extract spectral indices and statistical features from transect data."""

    def __init__(self, spectral_data: pd.DataFrame):
        """
        Initialize with spectral data.

        Args:
            spectral_data: DataFrame with columns [distance, red, green, blue, nir]
        """
        self.data = spectral_data.copy()
        self.window_size = THRESHOLDS['window_size']

        # Ensure we have required columns
        required = ['distance', 'red', 'green', 'blue', 'nir']
        if not all(col in self.data.columns for col in required):
            raise ValueError(f"Data must contain columns: {required}")

    def compute_all(self) -> pd.DataFrame:
        """
        Compute all features and return as DataFrame.

        Returns:
            DataFrame with original data plus all computed features
        """
        logger.debug("Computing spectral features")

        # Spectral indices
        self.data['ndvi'] = self._compute_ndvi()
        self.data['ndwi'] = self._compute_ndwi()
        self.data['brightness'] = self._compute_brightness()
        self.data['nir_ratio'] = self._compute_nir_ratio()
        self.data['blue_red_ratio'] = self._compute_blue_red_ratio()
        self.data['red_green_ratio'] = self._compute_red_green_ratio()

        # Derivative features (PRIMARY for boundary detection)
        self.data['nir_d1'] = self._compute_nir_derivative()
        self.data['nir_d1_smooth'] = self._compute_nir_derivative_smooth()
        self.data['brightness_d1'] = self._compute_brightness_derivative()

        # Multi-band derivatives for consensus detection
        self.data['red_d1'] = self._compute_red_derivative()
        self.data['red_d1_smooth'] = self._compute_red_derivative_smooth()
        self.data['green_d1'] = self._compute_green_derivative()
        self.data['green_d1_smooth'] = self._compute_green_derivative_smooth()
        self.data['blue_d1'] = self._compute_blue_derivative()
        self.data['blue_d1_smooth'] = self._compute_blue_derivative_smooth()

        # PHASE 1: R/G ratio derivatives for shell-line detection
        self.data['rg_ratio_d1'] = self._compute_rg_ratio_derivative()
        self.data['rg_ratio_d1_smooth'] = self._compute_rg_ratio_derivative_smooth()

        # PHASE 2A: Multi-scale derivatives for boundary-type-specific detection
        multiscale = self._compute_nir_derivative_multiscale()
        for key, series in multiscale.items():
            self.data[key] = series

        # PHASE 2A: Second derivatives (curvature/inflection detection for VEG_DUNES)
        self.data['nir_d2_w5'] = self._compute_nir_derivative_second(window_size=5)
        self.data['nir_d2_w7'] = self._compute_nir_derivative_second(window_size=7)
        self.data['nir_d2_w9'] = self._compute_nir_derivative_second(window_size=9)

        # PHASE 2A: RGB foam peak detection (for BEACH_WET→WATER boundaries)
        self.data['has_rgb_foam_peak'] = self._detect_rgb_foam_peaks()

        # Statistical features (sliding window)
        self.data['variability'] = self._compute_variability()
        self.data['slope'] = self._compute_slope()
        self.data['curvature'] = self._compute_curvature()
        self.data['local_range'] = self._compute_local_range()

        # Shape features
        self.data['spectral_angle'] = self._compute_spectral_angle()
        self.data['has_oscillations'] = self._detect_oscillations()

        logger.info(f"Computed {len(self.data.columns) - 5} features")

        return self.data

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

    def _compute_brightness(self) -> pd.Series:
        """Overall reflectance (mean of all bands)."""
        return self.data[['red', 'green', 'blue', 'nir']].mean(axis=1)

    def _compute_nir_ratio(self) -> pd.Series:
        """NIR prominence relative to visible bands."""
        visible_mean = self.data[['red', 'green', 'blue']].mean(axis=1)
        return self.data['nir'] / (visible_mean + EPSILON)

    def _compute_blue_red_ratio(self) -> pd.Series:
        """Blue to red ratio (water indicator)."""
        return self.data['blue'] / (self.data['red'] + EPSILON)

    def _compute_red_green_ratio(self) -> pd.Series:
        """
        Red to green ratio - diagnostic for beach zone classification.

        Expected patterns:
        - VEGETATED_DUNES: Variable, generally R > G (ratio > 1.0)
        - BEACH_DRY: Generally R ≈ G (ratio ≈ 1.0)
        - BEACH_WET: Generally R > G (ratio > 1.0)
        - WATER: Generally R < G (ratio < 1.0), except wave crests

        Returns:
            Series with red/green ratio values
        """
        return self.data['red'] / (self.data['green'] + EPSILON)

    def _compute_rg_ratio_derivative(self) -> pd.Series:
        """
        First derivative of R/G ratio (rate of change).

        Can help detect zone transitions where R/G pattern changes.

        Returns:
            Series with R/G ratio derivative
        """
        rg_ratio = self._compute_red_green_ratio().values
        distance = self.data['distance'].values

        # Calculate derivative using numpy gradient
        rg_d1 = np.gradient(rg_ratio, distance)

        return pd.Series(rg_d1, index=self.data.index)

    def _compute_rg_ratio_derivative_smooth(self) -> pd.Series:
        """
        Smoothed R/G ratio derivative to reduce noise.

        Returns:
            Series with smoothed R/G ratio derivative
        """
        from scipy.ndimage import uniform_filter1d

        rg_d1 = self._compute_rg_ratio_derivative().values

        # Smooth with uniform filter (window=5 for stability)
        rg_d1_smooth = uniform_filter1d(rg_d1, size=5, mode='nearest')

        return pd.Series(rg_d1_smooth, index=self.data.index)

    def _compute_nir_derivative(self) -> pd.Series:
        """
        First derivative of NIR band (rate of change).

        Primary signal for boundary detection.
        Negative values indicate NIR decreasing (land->water direction).

        Returns:
            Series with NIR derivative in units per meter
        """
        nir = self.data['nir'].values
        distance = self.data['distance'].values

        # Calculate derivative using numpy gradient
        nir_d1 = np.gradient(nir, distance)

        return pd.Series(nir_d1, index=self.data.index)

    def _compute_nir_derivative_smooth(self) -> pd.Series:
        """
        Smoothed NIR first derivative to reduce noise.

        Uses uniform filter (moving average) with window size 3.
        Better for detecting sustained drops vs. single-point noise.

        Returns:
            Series with smoothed NIR derivative
        """
        from scipy.ndimage import uniform_filter1d

        nir_d1 = self._compute_nir_derivative().values

        # Smooth with uniform filter (window=3)
        nir_d1_smooth = uniform_filter1d(nir_d1, size=3, mode='nearest')

        return pd.Series(nir_d1_smooth, index=self.data.index)

    def _compute_brightness_derivative(self) -> pd.Series:
        """
        First derivative of brightness for confirmation.

        Returns:
            Series with brightness derivative in units per meter
        """
        brightness = self._compute_brightness().values
        distance = self.data['distance'].values

        # Calculate derivative
        brightness_d1 = np.gradient(brightness, distance)

        return pd.Series(brightness_d1, index=self.data.index)

    def _compute_red_derivative(self) -> pd.Series:
        """
        First derivative of red band (rate of change).

        Used for multi-band consensus in boundary detection.

        Returns:
            Series with red derivative in units per meter
        """
        red = self.data['red'].values
        distance = self.data['distance'].values

        # Calculate derivative using numpy gradient
        red_d1 = np.gradient(red, distance)

        return pd.Series(red_d1, index=self.data.index)

    def _compute_red_derivative_smooth(self) -> pd.Series:
        """
        Smoothed red first derivative to reduce noise.

        Uses uniform filter (moving average) with window size 3.

        Returns:
            Series with smoothed red derivative
        """
        from scipy.ndimage import uniform_filter1d

        red_d1 = self._compute_red_derivative().values

        # Smooth with uniform filter (window=3)
        red_d1_smooth = uniform_filter1d(red_d1, size=3, mode='nearest')

        return pd.Series(red_d1_smooth, index=self.data.index)

    def _compute_green_derivative(self) -> pd.Series:
        """
        First derivative of green band (rate of change).

        Used for multi-band consensus in boundary detection.

        Returns:
            Series with green derivative in units per meter
        """
        green = self.data['green'].values
        distance = self.data['distance'].values

        # Calculate derivative using numpy gradient
        green_d1 = np.gradient(green, distance)

        return pd.Series(green_d1, index=self.data.index)

    def _compute_green_derivative_smooth(self) -> pd.Series:
        """
        Smoothed green first derivative to reduce noise.

        Uses uniform filter (moving average) with window size 3.

        Returns:
            Series with smoothed green derivative
        """
        from scipy.ndimage import uniform_filter1d

        green_d1 = self._compute_green_derivative().values

        # Smooth with uniform filter (window=3)
        green_d1_smooth = uniform_filter1d(green_d1, size=3, mode='nearest')

        return pd.Series(green_d1_smooth, index=self.data.index)

    def _compute_blue_derivative(self) -> pd.Series:
        """
        First derivative of blue band (rate of change).

        Used for multi-band consensus in boundary detection.

        Returns:
            Series with blue derivative in units per meter
        """
        blue = self.data['blue'].values
        distance = self.data['distance'].values

        # Calculate derivative using numpy gradient
        blue_d1 = np.gradient(blue, distance)

        return pd.Series(blue_d1, index=self.data.index)

    def _compute_blue_derivative_smooth(self) -> pd.Series:
        """
        Smoothed blue first derivative to reduce noise.

        Uses uniform filter (moving average) with window size 3.

        Returns:
            Series with smoothed blue derivative
        """
        from scipy.ndimage import uniform_filter1d

        blue_d1 = self._compute_blue_derivative().values

        # Smooth with uniform filter (window=3)
        blue_d1_smooth = uniform_filter1d(blue_d1, size=3, mode='nearest')

        return pd.Series(blue_d1_smooth, index=self.data.index)

    def _compute_variability(self) -> pd.Series:
        """Standard deviation in sliding window."""
        # Use brightness for variability calculation
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
            # Define window bounds
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

        # Use numpy gradient for second derivative
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

    def _compute_spectral_angle(self) -> pd.Series:
        """
        Spectral angle between consecutive points.

        Returns angle in degrees between spectral vectors.
        """
        angles = [0.0]  # First point has no predecessor

        for i in range(1, len(self.data)):
            # Create spectral vectors [R, G, B, NIR]
            # Convert to float64 to prevent overflow in dot product calculations
            vec1 = self.data.iloc[i-1][['red', 'green', 'blue', 'nir']].values.astype(np.float64)
            vec2 = self.data.iloc[i][['red', 'green', 'blue', 'nir']].values.astype(np.float64)

            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 > 0 and norm2 > 0:
                cos_angle = dot_product / (norm1 * norm2 + EPSILON)
                # Clamp to valid range for arccos
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

        # Find peaks
        peaks, _ = signal.find_peaks(
            brightness,
            distance=2,  # Minimum distance between peaks
            prominence=5  # Minimum prominence
        )

        # Create boolean array
        has_osc = np.zeros(len(brightness), dtype=bool)

        # Mark regions around peaks as oscillatory
        for peak_idx in peaks:
            start = max(0, peak_idx - self.window_size // 2)
            end = min(len(brightness), peak_idx + self.window_size // 2)
            has_osc[start:end] = True

        return pd.Series(has_osc, index=brightness.index)

    def get_feature_summary(self) -> Dict:
        """
        Get summary statistics of computed features.

        Returns:
            Dictionary with feature statistics
        """
        if 'ndvi' not in self.data.columns:
            raise ValueError("Features not computed yet. Call compute_all() first.")

        feature_cols = [
            'ndvi', 'ndwi', 'brightness', 'nir_ratio', 'blue_red_ratio',
            'variability', 'slope', 'curvature', 'local_range', 'spectral_angle'
        ]

        summary = {}
        for col in feature_cols:
            summary[col] = {
                'mean': self.data[col].mean(),
                'std': self.data[col].std(),
                'min': self.data[col].min(),
                'max': self.data[col].max()
            }

        return summary

    # ========================================================================
    # PHASE 2A: Boundary-Type-Specific Features
    # ========================================================================

    def _compute_nir_derivative_multiscale(self) -> Dict[str, pd.Series]:
        """
        Compute NIR derivatives at multiple smoothing scales.

        Different boundary types require different smoothing levels:
        - VEG_DUNES→BEACH_DRY: window 7-9 (coarse, smooth vegetation noise)
        - BEACH_WET→WATER: window 7-11 (coarse, for foam detection)
        - BEACH_DRY→BEACH_WET: window 5-9 (medium, preserve sharpness)

        Returns:
            Dictionary mapping feature names to derivative series
        """
        from scipy.ndimage import uniform_filter1d

        scales = [5, 7, 9, 11]  # Different smoothing windows
        derivatives = {}

        for window in scales:
            # Smooth NIR signal
            nir_smooth = uniform_filter1d(
                self.data['nir'].values,
                size=window,
                mode='nearest'
            )
            # Compute derivative
            nir_d1 = np.gradient(nir_smooth, self.data['distance'].values)
            derivatives[f'nir_d1_w{window}'] = pd.Series(nir_d1, index=self.data.index)

        return derivatives

    def _compute_nir_derivative_second(self, window_size: int = 7) -> pd.Series:
        """
        Second derivative of NIR (curvature/inflection detection).

        Critical for VEG_DUNES→BEACH_DRY boundaries which show inflection points,
        not simple first derivative drops.

        Visual signature of VEG_DUNES boundary:
        - Steep increase → steep drop → **inflection at 0** → increase → shallow

        Args:
            window_size: Smoothing window for first derivative (default 7)

        Returns:
            Series with second derivative (curvature)
        """
        from scipy.ndimage import uniform_filter1d

        # Smooth NIR signal first
        nir_smooth = uniform_filter1d(
            self.data['nir'].values,
            size=window_size,
            mode='nearest'
        )

        # First derivative
        distances = self.data['distance'].values
        nir_d1 = np.gradient(nir_smooth, distances)

        # Second derivative
        nir_d2 = np.gradient(nir_d1, distances)

        return pd.Series(nir_d2, index=self.data.index)

    def _detect_rgb_foam_peaks(self) -> pd.Series:
        """
        Detect RGB peaks indicating foam from breaking surf.

        Critical for BEACH_WET→WATER boundaries which often have:
        - Small RGB bump from white foam at breaking surf
        - Gradual NIR transition (not always a sharp drop)

        Visual signature:
        - Local maximum in RGB bands
        - Visible at smoothing windows 7-11
        - Often the DEFINING feature (not NIR drop)

        PHASE 2 FIX: Stricter prominence threshold to reduce wave crest false positives

        Returns:
            Boolean series indicating foam peak locations
        """
        from scipy.ndimage import uniform_filter1d
        from .config import THRESHOLDS

        # Compute RGB average with smoothing (window=7 optimal for foam detection)
        rgb_avg = self.data[['red', 'green', 'blue']].mean(axis=1)
        rgb_smooth = uniform_filter1d(rgb_avg.values, size=7, mode='nearest')

        # Get prominence threshold from config (PHASE 2 FIX: increased from 5.0 to 10.0)
        prominence = THRESHOLDS.get('boundary_thresholds', {}).get(
            'surf_zone', {}
        ).get('rgb_peak_prominence', 10.0)

        # Find peaks using scipy (stricter threshold)
        peaks, properties = signal.find_peaks(
            rgb_smooth,
            prominence=prominence,  # PHASE 2 FIX: Increased from 5 to 10 (only strong foam)
            width=2                 # Minimum peak width in points
        )

        # Create boolean series
        has_foam = pd.Series(False, index=self.data.index)
        has_foam.iloc[peaks] = True

        logger.debug(f"Detected {len(peaks)} RGB foam peaks (prominence >= {prominence})")

        return has_foam
