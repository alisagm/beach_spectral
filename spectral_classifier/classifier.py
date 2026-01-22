"""
Landcover classification module using rule-based approach.

Supports 3-band (RGB or CIR) and 4-band (RGBN) imagery.
For RGB-only imagery, classification is skipped (returns UNKNOWN)
as NIR-based features are required for reliable classification.
"""

import logging
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from .config import THRESHOLDS, LANDCOVER_CLASSES
from .data_io import detect_band_mode_from_dataframe

logger = logging.getLogger(__name__)


class LandcoverClassifier:
    """Rule-based classifier for spectral transect data."""

    def __init__(self, thresholds: Dict = None):
        """
        Initialize classifier with threshold parameters.

        Args:
            thresholds: Optional custom threshold dictionary (uses config default if None)
        """
        self.thresholds = thresholds if thresholds is not None else THRESHOLDS

    def classify(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Classify landcover for each point using rule-based logic.
        
        For RGB-only imagery (no NIR), returns UNKNOWN for all points
        as NIR-based features are required for reliable classification.

        Args:
            features: DataFrame with computed features

        Returns:
            DataFrame with added 'predicted_class' and 'confidence' columns
        """
        logger.debug("Classifying landcover")

        # Detect band mode and NIR availability
        band_mode = self._detect_band_mode(features)
        has_nir = self._check_nir_available(features)
        
        if not has_nir:
            logger.info(f"RGB-only mode detected (band_mode={band_mode}) - "
                       "classification skipped (returning UNKNOWN for all points)")
            return self._classify_rgb_mode(features)

        # Standard NIR-based classification
        # Ensure required features exist
        required = ['brightness', 'variability', 'ndvi', 'ndwi',
                    'nir_ratio', 'blue_red_ratio', 'has_oscillations']
        missing = [f for f in required if f not in features.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # Check for optional Phase 1-3 features (R/G ratio)
        has_rg_ratio = 'red_green_ratio' in features.columns
        if has_rg_ratio:
            logger.debug("R/G ratio feature available for classification")

        # Apply classification rules
        classes = []
        confidences = []

        for idx, row in features.iterrows():
            label, confidence = self._classify_point(row)
            classes.append(label)
            confidences.append(confidence)

        features['predicted_class'] = classes
        features['confidence'] = confidences

        # Log class distribution
        class_counts = features['predicted_class'].value_counts()
        logger.info(f"Classification complete: {dict(class_counts)}")

        return features

    def _detect_band_mode(self, features: pd.DataFrame) -> str:
        """Detect band mode from features DataFrame."""
        return detect_band_mode_from_dataframe(features)

    def _check_nir_available(self, features: pd.DataFrame) -> bool:
        """Check if NIR band has valid (non-NaN) values."""
        if 'nir' not in features.columns:
            return False
        return not features['nir'].isna().all()

    def _classify_rgb_mode(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Return UNKNOWN classification for RGB-only mode.
        
        NIR-based features (nir_ratio, NDVI, NDWI) are required for
        reliable landcover classification. Without NIR, we cannot
        distinguish between dry beach, wet beach, vegetation, etc.
        
        Args:
            features: DataFrame with computed features
            
        Returns:
            DataFrame with 'UNKNOWN' for all points
        """
        features['predicted_class'] = 'UNKNOWN'
        features['confidence'] = 0.30  # Low confidence for RGB-only
        
        logger.info(f"RGB classification: {len(features)} points -> UNKNOWN")
        
        return features

    def _classify_point(self, row: pd.Series) -> tuple:
        """
        Classify a single point using hierarchical rules.

        Classification hierarchy based on NIR ratio (primary discriminator):
        1. WATER:      nir_ratio < 0.35
        2. BEACH_WET:  0.40 < nir_ratio < 0.70
        3. DRY_BEACH:  nir_ratio > 0.80, brightness > 195, variability < 10
        4. VEG_DUNES:  nir_ratio > 0.80, variability > 15, brightness < 190
        5. WAVE_CRESTS: Within water zones with high variability
        6. UNKNOWN:    Fallback

        Args:
            row: Series with feature values

        Returns:
            Tuple of (class_label, confidence_score)
        """
        brightness = row['brightness']
        variability = row['variability']
        ndvi = row['ndvi']
        ndwi = row['ndwi']
        nir_ratio = row['nir_ratio']
        blue_red = row['blue_red_ratio']
        has_osc = row['has_oscillations']

        # Handle NaN values (shouldn't happen if _check_nir_available worked)
        if pd.isna(nir_ratio):
            return 'UNKNOWN', 0.30

        # Rule 1: WATER (primary: very low NIR ratio)
        if self._is_water(row):
            confidence = 0.85
            if ndwi > self.thresholds['ndwi_water_min']:
                confidence += 0.05
            if blue_red > self.thresholds['blue_red_water_min']:
                confidence += 0.05
            confidence = min(confidence, 0.95)
            
            # Check for wave crests within water
            if self._is_wave_crest(row):
                return 'WAVE_CRESTS', 0.65
            
            return 'WATER', confidence

        # Rule 2: BEACH_WET (intermediate NIR ratio)
        if self._is_wet_beach(row):
            confidence = 0.60
            if ndwi > self.thresholds['ndwi_wet_beach_min']:
                confidence += 0.15
            if variability > 5:
                confidence += 0.10
            confidence = min(confidence, 0.85)
            return 'BEACH_WET', confidence

        # Rule 3: DRY_BEACH (high NIR, high brightness, low variability)
        if self._is_dry_beach(row):
            confidence = 0.80
            if variability < 5:
                confidence += 0.05
            if brightness > 200:
                confidence += 0.05
            confidence = min(confidence, 0.90)
            return 'DRY_BEACH', confidence

        # Rule 4: VEG_DUNES (high NIR, high variability, lower brightness)
        if self._is_veg_dunes(row):
            confidence = 0.70
            if variability > 20:
                confidence += 0.10
            if brightness < 160:
                confidence += 0.05
            confidence = min(confidence, 0.85)
            return 'VEG_DUNES', confidence

        # Note: ALL_LAND class removed - areas beyond dunes will be UNKNOWN or VEG_DUNES
        return 'UNKNOWN', 0.30

    def _is_water(self, row: pd.Series) -> bool:
        """Check if point matches WATER criteria."""
        nir_ratio = row['nir_ratio']
        ndwi = row['ndwi']
        
        if pd.isna(nir_ratio):
            return False
        
        # Primary: very low NIR ratio
        if nir_ratio < self.thresholds['nir_ratio_water_max']:
            return True
        
        # Secondary: high NDWI and blue/red ratio
        if not pd.isna(ndwi) and ndwi > self.thresholds['ndwi_water_min']:
            if row['blue_red_ratio'] > self.thresholds['blue_red_water_min']:
                return True
        
        return False

    def _is_wave_crest(self, row: pd.Series) -> bool:
        """
        Check if point is a wave crest (within water zone).
        
        Note: This is a refinement within water zones
        """
        # Must be water-like NIR ratio
        if row['nir_ratio'] > self.thresholds['nir_ratio_water_max']:
            return False
        
        # Higher variability than typical water
        if row['variability'] > 25:
            return True
        
        # Oscillatory pattern detected
        if row['has_oscillations']:
            return True
        
        return False

    def _is_wet_beach(self, row: pd.Series) -> bool:
        """Check if point matches BEACH_WET criteria."""
        nir_ratio = row['nir_ratio']
        
        if pd.isna(nir_ratio):
            return False
        
        # Intermediate NIR ratio
        min_ratio = self.thresholds['nir_ratio_wet_beach_min']
        max_ratio = self.thresholds['nir_ratio_wet_beach_max']
        
        return min_ratio <= nir_ratio <= max_ratio

    def _is_dry_beach(self, row: pd.Series) -> bool:
        """Check if point matches DRY_BEACH criteria."""
        nir_ratio = row['nir_ratio']
        brightness = row['brightness']
        variability = row['variability']
        
        if pd.isna(nir_ratio):
            return False
        
        # High NIR ratio
        if nir_ratio < self.thresholds['nir_ratio_dry_min']:
            return False
        
        # High brightness
        if brightness < self.thresholds['dry_beach_brightness_min']:
            return False
        
        # Low variability
        if variability > self.thresholds['dry_beach_variability_max']:
            return False
        
        return True

    def _is_veg_dunes(self, row: pd.Series) -> bool:
        """Check if point matches VEG_DUNES criteria."""
        nir_ratio = row['nir_ratio']
        brightness = row['brightness']
        variability = row['variability']
        
        if pd.isna(nir_ratio):
            return False
        
        # High NIR ratio (similar to dry beach)
        if nir_ratio < self.thresholds['nir_ratio_dry_min']:
            return False
        
        # High variability (different from dry beach)
        if variability < self.thresholds['veg_variability_min']:
            return False
        
        # Lower brightness than dry beach (optional)
        if brightness > self.thresholds['veg_dune_brightness_max']:
            return False
        
        return True

    def apply_spatial_smoothing(
        self,
        landcover: pd.DataFrame,
        window: int = 3
    ) -> pd.DataFrame:
        """
        Apply spatial median filter to reduce classification noise.

        Args:
            landcover: DataFrame with predicted classes
            window: Size of median filter window

        Returns:
            DataFrame with smoothed classifications
        """
        logger.debug(f"Applying spatial smoothing (window={window})")

        # Convert classes to numeric for filtering
        class_to_num = {cls: i for i, cls in enumerate(LANDCOVER_CLASSES)}
        num_to_class = {i: cls for cls, i in class_to_num.items()}

        numeric = landcover['predicted_class'].map(class_to_num).values
        
        # Handle any unmapped classes
        numeric = np.nan_to_num(numeric, nan=class_to_num.get('UNKNOWN', 5))

        # Apply median filter
        smoothed = median_filter(numeric.astype(float), size=window)
        smoothed = smoothed.astype(int)

        # Convert back to class labels
        landcover['predicted_class'] = [num_to_class.get(int(n), 'UNKNOWN') for n in smoothed]

        return landcover

    def get_classification_summary(self, landcover: pd.DataFrame) -> Dict:
        """
        Get summary statistics of classification results.

        Args:
            landcover: DataFrame with predicted classes

        Returns:
            Dictionary with classification statistics
        """
        summary = {
            'total_points': len(landcover),
            'class_distribution': landcover['predicted_class'].value_counts().to_dict(),
            'mean_confidence': landcover['confidence'].mean(),
            'confidence_by_class': landcover.groupby('predicted_class')['confidence'].mean().to_dict()
        }

        return summary