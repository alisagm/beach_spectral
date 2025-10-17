"""
Landcover classification module using rule-based approach.
"""

import logging
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from .config import THRESHOLDS, LANDCOVER_CLASSES

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

        Args:
            features: DataFrame with computed features

        Returns:
            DataFrame with added 'predicted_class' and 'confidence' columns
        """
        logger.debug("Classifying landcover")

        # Ensure required features exist
        required = ['brightness', 'variability', 'ndvi', 'ndwi',
                    'nir_ratio', 'blue_red_ratio', 'has_oscillations']
        missing = [f for f in required if f not in features.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # PHASE 5: Check for optional Phase 1-3 features (R/G ratio)
        # R/G ratio is used in boundary detection (Phase 3) but optional for classification
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

    def _classify_point(self, row: pd.Series) -> tuple:
        """
        Classify a single point using hierarchical rules.

        Classification hierarchy based on NIR ratio (primary discriminator):
        1. WATER:      nir_ratio < 0.35
        2. BEACH_WET:  0.40 < nir_ratio < 0.70
        3. DRY_BEACH:  nir_ratio > 0.80, brightness > 195, variability < 10
        4. VEG_DUNES:  nir_ratio > 0.80, variability > 15, brightness < 190
        5. WAVE_CRESTS: Within water zones with high variability
        6. ALL_LAND:   High brightness fallback

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

        # Rule 1: WATER (primary: very low NIR ratio)
        if self._is_water(row):
            # Confidence based on how strongly conditions are met
            confidence = 0.85
            if ndwi > self.thresholds['ndwi_water_min']:
                confidence += 0.05
            if blue_red > self.thresholds['blue_red_water_min']:
                confidence += 0.05
            if nir_ratio < 0.25:  # Very strong water signal
                confidence += 0.05
            return 'WATER', min(confidence, 0.95)

        # Rule 2: BEACH_WET (intermediate NIR ratio)
        if self._is_wet_beach(row):
            confidence = 0.70
            # Higher confidence if NDWI is clearly elevated
            if ndwi > 0.35:
                confidence += 0.10
            # Lower confidence near boundaries
            if nir_ratio < 0.45 or nir_ratio > 0.65:
                confidence -= 0.10
            return 'BEACH_WET', min(max(confidence, 0.60), 0.85)

        # Rule 3: DRY_BEACH (high NIR, very bright, uniform)
        if self._is_dry_beach(row):
            confidence = 0.80
            # Higher confidence if very bright and very uniform
            if brightness > 205:
                confidence += 0.05
            if variability < 5:
                confidence += 0.05
            return 'DRY_BEACH', min(confidence, 0.90)

        # Rule 4: VEG_DUNES (high NIR, high variability, lower brightness)
        if self._is_veg_dunes(row):
            confidence = 0.70
            # Higher confidence if variability very high
            if variability > 25:
                confidence += 0.10
            # Confirmation from oscillations
            if has_osc:
                confidence += 0.05
            return 'VEG_DUNES', min(confidence, 0.85)

        # Rule 5: WAVE_CRESTS (water-like NIR but high variability)
        # Note: This is a refinement within water zones
        if (nir_ratio < self.thresholds['nir_ratio_water_max'] and
            variability > 20 and has_osc):
            return 'WAVE_CRESTS', 0.65

        # Default: UNKNOWN if no rules matched
        # Note: ALL_LAND class removed - areas beyond dunes will be UNKNOWN or VEG_DUNES
        return 'UNKNOWN', 0.30

    def _is_water(self, row: pd.Series) -> bool:
        """
        Water is characterized by very low NIR ratio.
        Empirical: NIR ratio = 0.21 +- 0.10, NDWI = 0.75 +- 0.12
        """
        nir_ratio = row['nir_ratio']
        ndwi = row['ndwi']
        blue_red = row['blue_red_ratio']

        # Primary: Very low NIR (water absorbs NIR)
        if nir_ratio > self.thresholds['nir_ratio_water_max']:
            return False

        # Confirmation: High NDWI or blue > red
        if ndwi > self.thresholds['ndwi_water_min']:
            return True
        if blue_red > self.thresholds['blue_red_water_min']:
            return True

        # Moderate confidence if NIR very low even without confirmation
        return nir_ratio < 0.30

    def _is_wet_beach(self, row: pd.Series) -> bool:
        """
        Wet beach has intermediate NIR ratio (transitional zone).
        Empirical: NIR ratio = 0.56 +- 0.13, NDWI = 0.37 +- 0.13
        """
        nir_ratio = row['nir_ratio']
        ndwi = row['ndwi']
        brightness = row['brightness']

        # NIR ratio in intermediate range
        if not (self.thresholds['nir_ratio_wet_beach_min'] < nir_ratio <
                self.thresholds['nir_ratio_wet_beach_max']):
            return False

        # NDWI elevated but not as high as water
        if ndwi < self.thresholds['ndwi_wet_beach_min']:
            return False

        # Brightness check (not too bright like dry beach)
        if brightness > self.thresholds['dry_beach_brightness_min']:
            return False

        return True

    def _is_dry_beach(self, row: pd.Series) -> bool:
        """
        Dry beach is very bright, uniform, high NIR ratio.
        Empirical: Brightness = 209 +- 6, NIR ratio = 0.89 +- 0.04, Variability = 4 +- 2
        """
        nir_ratio = row['nir_ratio']
        brightness = row['brightness']
        variability = row['variability']

        # High NIR ratio (dry sand)
        if nir_ratio < self.thresholds['nir_ratio_dry_min']:
            return False

        # Very high brightness
        if brightness < self.thresholds['dry_beach_brightness_min']:
            return False

        # Low variability (uniform surface)
        if variability > self.thresholds['dry_beach_variability_max']:
            return False

        return True

    def _is_veg_dunes(self, row: pd.Series) -> bool:
        """
        Vegetated dunes have high variability, high NIR ratio, lower brightness.
        Empirical: Variability = 28 +- 11, NIR ratio = 0.89 +- 0.08, Brightness = 145 +- 37

        NOTE: NDVI is NOT used as primary criterion (empirical NDVI = -0.12 +- 0.05, negative!)
        """
        nir_ratio = row['nir_ratio']
        brightness = row['brightness']
        variability = row['variability']

        # High NIR ratio (dry substrate)
        if nir_ratio < self.thresholds['nir_ratio_dry_min']:
            return False

        # High variability (vegetation/dune structure)
        if variability < self.thresholds['veg_variability_min']:
            return False

        # Lower brightness than dry beach
        if brightness > self.thresholds['veg_dune_brightness_max']:
            return False

        return True

    def apply_spatial_smoothing(
        self,
        features: pd.DataFrame,
        window_size: int = None
    ) -> pd.DataFrame:
        """
        Apply median filter to reduce isolated misclassifications.

        Args:
            features: DataFrame with 'predicted_class' column
            window_size: Size of smoothing window (uses config default if None)

        Returns:
            DataFrame with smoothed classifications
        """
        if 'predicted_class' not in features.columns:
            raise ValueError("Must run classify() before smoothing")

        if window_size is None:
            window_size = self.thresholds['smoothing_window']

        logger.debug(f"Applying spatial smoothing (window={window_size})")

        # Convert class labels to numeric codes
        class_to_code = {label: i for i, label in enumerate(LANDCOVER_CLASSES)}
        code_to_class = {i: label for label, i in class_to_code.items()}

        codes = features['predicted_class'].map(class_to_code).values

        # Apply median filter
        smoothed_codes = median_filter(
            codes,
            size=window_size,
            mode='nearest'
        )

        # Round to nearest integer (median filter may produce floats)
        smoothed_codes = np.round(smoothed_codes).astype(int)

        # Convert back to class labels
        features['predicted_class'] = [
            code_to_class.get(code, 'UNKNOWN')
            for code in smoothed_codes
        ]

        logger.debug("Spatial smoothing complete")

        return features

    def apply_monotonic_smoothing(
        self,
        features: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        DEPRECATED: Monotonic smoothing was disabled in Phase 6D due to
        catastrophic class collapse (93% UNKNOWN in some transects).

        This method is kept for API compatibility but returns features unchanged.

        For historical implementation, see:
        spectral_classifier/archive/deprecated_smoothing.py

        Args:
            features: DataFrame with 'predicted_class' column
            **kwargs: Ignored (formerly min_span_m, use_relaxed)

        Returns:
            DataFrame unchanged
        """
        logger.warning(
            "apply_monotonic_smoothing() is deprecated and disabled. "
            "It caused class collapse in Phase 6D testing. "
            "See docs/IMPROVEMENT_HISTORY.md for details."
        )
        return features  # Return unchanged

    def apply_transition_based_classification(
        self,
        features: pd.DataFrame,
        transitions: List[Dict]
    ) -> pd.DataFrame:
        """
        Use detected transitions to refine classifications.

        Strategy:
        1. Use transitions to define zone boundaries
        2. Classify entire zones based on majority vote or median features
        3. Ensures consistency within zones while respecting transition boundaries

        Args:
            features: DataFrame with 'predicted_class' column
            transitions: List of transition dictionaries from TransitionDetector

        Returns:
            DataFrame with refined classifications
        """
        if not transitions:
            logger.debug("No transitions provided, skipping transition-based classification")
            return features

        if 'predicted_class' not in features.columns:
            raise ValueError("Must run classify() before transition-based refinement")

        logger.debug(f"Refining classifications using {len(transitions)} transitions")

        classes = features['predicted_class'].values.copy()
        distances = features['distance'].values

        # Sort transitions by distance
        sorted_transitions = sorted(transitions, key=lambda x: x['distance'])

        # Define zone boundaries from transitions
        zone_boundaries = [0] + [t['index'] for t in sorted_transitions] + [len(classes)]

        # Process each zone
        for i in range(len(zone_boundaries) - 1):
            start_idx = zone_boundaries[i]
            end_idx = zone_boundaries[i + 1]

            if start_idx >= end_idx:
                continue

            zone_classes = classes[start_idx:end_idx]

            # Get majority class (excluding UNKNOWN)
            non_unknown = [c for c in zone_classes if c != 'UNKNOWN']
            if non_unknown:
                from collections import Counter
                majority_class = Counter(non_unknown).most_common(1)[0][0]

                # Apply majority class to entire zone
                classes[start_idx:end_idx] = majority_class

        features['predicted_class'] = classes

        # Log results
        class_counts = pd.Series(classes).value_counts()
        logger.info(f"Transition-based refinement complete: {dict(class_counts)}")

        return features

    def apply_boundary_aware_correction(
        self,
        features: pd.DataFrame,
        shell_line_distance: float,
        correction_mode: str = 'strict',
        correction_buffer: float = 2.0
    ) -> pd.DataFrame:
        """
        PHASE 7B: Correct classifications to respect detected shell line boundary.

        Enforces spatial consistency so that landward classes (VEG_DUNES, DRY_BEACH)
        only appear before the shell line, and seaward classes (BEACH_WET, WATER)
        only appear after it. This eliminates background shading inconsistencies
        in visualizations.

        Args:
            features: DataFrame with 'predicted_class' and 'distance' columns
            shell_line_distance: Distance (m) of detected DRY->WET boundary
            correction_mode:
                'strict' - Reclassify violations to expected class
                'soft' - Mark violations as UNKNOWN
            correction_buffer: Distance buffer (m) around boundary where corrections
                              are skipped to avoid over-correction near transition

        Returns:
            DataFrame with corrected 'predicted_class' column
        """
        if 'predicted_class' not in features.columns:
            raise ValueError("Must run classify() before boundary correction")

        if 'distance' not in features.columns:
            raise ValueError("Features must have 'distance' column")

        classes = features['predicted_class'].values.copy()
        distances = features['distance'].values

        # Define class groups
        landward_classes = ['VEG_DUNES', 'DRY_BEACH']
        seaward_classes = ['BEACH_WET', 'WATER', 'WAVE_CRESTS']

        corrections_made = 0
        corrections_log = []

        for i, (dist, cls) in enumerate(zip(distances, classes)):
            # Skip UNKNOWN (can appear anywhere)
            if cls == 'UNKNOWN':
                continue

            # Skip points within buffer zone of boundary
            if abs(dist - shell_line_distance) <= correction_buffer:
                continue

            # Check for violations
            if dist < shell_line_distance:
                # LANDWARD of shell line: only landward classes allowed
                if cls in seaward_classes:
                    original_class = cls

                    if correction_mode == 'strict':
                        classes[i] = 'DRY_BEACH'
                    else:  # soft mode
                        classes[i] = 'UNKNOWN'

                    corrections_made += 1
                    corrections_log.append({
                        'distance': dist,
                        'original': original_class,
                        'corrected': classes[i],
                        'reason': 'seaward_class_before_boundary'
                    })

            else:
                # SEAWARD of shell line: only seaward classes allowed
                if cls in landward_classes:
                    original_class = cls

                    if correction_mode == 'strict':
                        classes[i] = 'BEACH_WET'
                    else:  # soft mode
                        classes[i] = 'UNKNOWN'

                    corrections_made += 1
                    corrections_log.append({
                        'distance': dist,
                        'original': original_class,
                        'corrected': classes[i],
                        'reason': 'landward_class_after_boundary'
                    })

        features['predicted_class'] = classes

        # Log results
        logger.info(f"Boundary-aware correction: {corrections_made} points reclassified "
                   f"(shell line at {shell_line_distance:.1f}m, mode={correction_mode})")
        if corrections_made > 0 and corrections_made <= 10:
            # Log details for small number of corrections
            for corr in corrections_log:
                logger.debug(f"  {corr['distance']:.1f}m: {corr['original']} -> {corr['corrected']} ({corr['reason']})")
        elif corrections_made > 10:
            logger.debug(f"  First 3 corrections: {corrections_log[:3]}")

        return features

    def validate_sequence(self, features: pd.DataFrame) -> Dict:
        """
        Validate landcover sequence makes geographic sense.

        Expected sequence from land to water:
        VEG_DUNES -> DRY_BEACH -> BEACH_WET -> WATER

        Note: VEG_DUNES is optional (sequence may start with DRY_BEACH)
              BEACH_WET is optional (may go directly DRY_BEACH -> WATER)

        Args:
            features: DataFrame with 'predicted_class' column

        Returns:
            Dictionary with validation results and warnings
        """
        classes = features['predicted_class'].values
        warnings = []

        # Check for valid transitions (ALL_LAND removed)
        valid_transitions = {
            'VEG_DUNES': ['DRY_BEACH', 'VEG_DUNES'],
            'DRY_BEACH': ['BEACH_WET', 'VEG_DUNES', 'DRY_BEACH', 'WATER'],
            'BEACH_WET': ['WATER', 'DRY_BEACH', 'BEACH_WET', 'WAVE_CRESTS'],
            'WATER': ['WAVE_CRESTS', 'WATER', 'BEACH_WET'],
            'WAVE_CRESTS': ['WATER', 'BEACH_WET', 'WAVE_CRESTS'],
            'UNKNOWN': LANDCOVER_CLASSES  # Can transition to anything
        }

        # Check each transition
        for i in range(len(classes) - 1):
            current = classes[i]
            next_class = classes[i + 1]

            if next_class not in valid_transitions.get(current, []):
                warnings.append(
                    f"Unusual transition at point {i}: {current} -> {next_class}"
                )

        # Check for isolated single-point classes
        for i in range(1, len(classes) - 1):
            if classes[i] != classes[i-1] and classes[i] != classes[i+1]:
                warnings.append(
                    f"Isolated classification at point {i}: {classes[i]}"
                )

        return {
            'is_valid': len(warnings) == 0,
            'warnings': warnings,
            'num_warnings': len(warnings)
        }
