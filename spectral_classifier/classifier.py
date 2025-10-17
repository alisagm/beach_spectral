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
        min_span_m: float = None,
        use_relaxed: bool = True
    ) -> pd.DataFrame:
        """
        Apply monotonic sequence constraint to classifications.

        Enforces the expected land-to-water sequence:
        VEG_DUNES -> DRY_BEACH -> BEACH_WET -> WATER

        Strategy:
        1. Find major zones (segments >= min_span_m)
        2. Build monotonic sequence from major zones
        3. If relaxed mode, preserve original classifications when no major zones found

        Rules:
        - Each category (except UNKNOWN) appears at most once
        - VEG_DUNES is optional (sequence may start with DRY_BEACH)
        - BEACH_WET is optional (may go DRY_BEACH -> WATER directly)
        - UNKNOWN can appear multiple times anywhere

        Args:
            features: DataFrame with 'predicted_class' and 'distance' columns
            min_span_m: Minimum span in meters (uses config default if None)
            use_relaxed: If True, only enforce monotonicity without strict span requirements

        Returns:
            DataFrame with monotonic sequence enforced
        """
        if 'predicted_class' not in features.columns:
            raise ValueError("Must run classify() before monotonic smoothing")

        if min_span_m is None:
            min_span_m = self.thresholds.get('min_category_span_m', 2.5)  # Relaxed from 5.0

        logger.debug(f"Applying monotonic smoothing (min_span={min_span_m}m, relaxed={use_relaxed})")

        classes = features['predicted_class'].values.copy()
        distances = features['distance'].values

        # Find all segments
        segments = self._build_segments(classes, distances)

        if use_relaxed:
            # Relaxed mode: only enforce monotonicity, don't filter by span
            monotonic_classes = self._enforce_monotonic_sequence(classes, distances)
        else:
            # Strict mode: filter by span and build from major zones
            major_zones = [s for s in segments if s['span'] >= min_span_m or s['class'] == 'UNKNOWN']

            # Fallback: if no major zones (except UNKNOWN), just enforce monotonicity
            non_unknown_zones = [z for z in major_zones if z['class'] != 'UNKNOWN']
            if len(non_unknown_zones) == 0:
                logger.warning("No major zones found, falling back to relaxed monotonic enforcement")
                monotonic_classes = self._enforce_monotonic_sequence(classes, distances)
            else:
                monotonic_classes = self._build_monotonic_from_zones(major_zones, len(classes), distances)

        features['predicted_class'] = monotonic_classes

        # Log results
        class_counts = pd.Series(monotonic_classes).value_counts()
        logger.info(f"Monotonic smoothing complete: {dict(class_counts)}")

        return features

    def _build_segments(
        self,
        classes: np.ndarray,
        distances: np.ndarray
    ) -> List[Dict]:
        """Build list of segments from classified data."""
        segments = []
        start_idx = 0
        current_class = classes[0]

        for i in range(1, len(classes)):
            if classes[i] != current_class:
                span = distances[i-1] - distances[start_idx]
                segments.append({
                    'class': current_class,
                    'start': start_idx,
                    'end': i - 1,
                    'span': span,
                    'start_dist': distances[start_idx],
                    'end_dist': distances[i-1]
                })
                start_idx = i
                current_class = classes[i]

        # Add final segment
        span = distances[-1] - distances[start_idx]
        segments.append({
            'class': current_class,
            'start': start_idx,
            'end': len(classes) - 1,
            'span': span,
            'start_dist': distances[start_idx],
            'end_dist': distances[-1]
        })

        return segments

    def _build_monotonic_from_zones(
        self,
        zones: List[Dict],
        total_points: int,
        distances: np.ndarray
    ) -> np.ndarray:
        """
        Build monotonic sequence from major zones.

        Args:
            zones: List of zone dictionaries
            total_points: Total number of points
            distances: Distance array

        Returns:
            Array of class labels
        """
        # Initialize all as UNKNOWN
        result = np.array(['UNKNOWN'] * total_points)

        # Sequence order
        sequence_order = {
            'VEG_DUNES': 0,
            'DRY_BEACH': 1,
            'BEACH_WET': 2,
            'WATER': 3,
            'WAVE_CRESTS': 3,
        }

        # Track which class types we've placed (ignore UNKNOWN)
        placed_classes = set()
        max_order_placed = -1

        for zone in zones:
            if zone['class'] == 'UNKNOWN':
                # UNKNOWN zones are always allowed
                result[zone['start']:zone['end']+1] = 'UNKNOWN'
                continue

            zone_order = sequence_order.get(zone['class'], -1)

            # Check if this class maintains monotonic sequence
            if zone['class'] in placed_classes:
                # Already placed this class - can't repeat
                result[zone['start']:zone['end']+1] = 'UNKNOWN'
                continue

            if zone_order < max_order_placed:
                # Going backward - not allowed
                result[zone['start']:zone['end']+1] = 'UNKNOWN'
                continue

            # Valid zone - place it
            result[zone['start']:zone['end']+1] = zone['class']
            placed_classes.add(zone['class'])
            max_order_placed = zone_order

        return result

    def _remove_short_segments(
        self,
        classes: np.ndarray,
        distances: np.ndarray,
        min_span_m: float
    ) -> np.ndarray:
        """
        Remove segments shorter than minimum span by merging with neighbors.

        Strategy:
        1. Build segments and identify short ones
        2. Iteratively merge short segments with neighbors
        3. Keep UNKNOWN segments as-is (they can be any length)

        Args:
            classes: Array of class labels
            distances: Array of distances
            min_span_m: Minimum span in meters

        Returns:
            Array with short segments removed
        """
        classes = classes.copy()
        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Identify current segments
            segments = []
            start_idx = 0
            current_class = classes[0]

            for i in range(1, len(classes)):
                if classes[i] != current_class:
                    # Segment ends
                    span = distances[i-1] - distances[start_idx]
                    segments.append({
                        'class': current_class,
                        'start': start_idx,
                        'end': i - 1,
                        'span': span
                    })
                    start_idx = i
                    current_class = classes[i]

            # Add final segment
            span = distances[-1] - distances[start_idx]
            segments.append({
                'class': current_class,
                'start': start_idx,
                'end': len(classes) - 1,
                'span': span
            })

            # Find first short segment (that's not UNKNOWN)
            short_seg_idx = None
            for idx, seg in enumerate(segments):
                if seg['span'] < min_span_m and seg['class'] != 'UNKNOWN':
                    short_seg_idx = idx
                    break

            # If no short segments found, we're done
            if short_seg_idx is None:
                break

            seg = segments[short_seg_idx]

            # Find adjacent segments (direct neighbors)
            prev_seg = segments[short_seg_idx - 1] if short_seg_idx > 0 else None
            next_seg = segments[short_seg_idx + 1] if short_seg_idx < len(segments) - 1 else None

            # Decide which neighbor to merge with
            # Prefer merging with non-UNKNOWN neighbors
            # If both are UNKNOWN, pick the larger one
            merge_target = None

            if next_seg and next_seg['class'] != 'UNKNOWN':
                merge_target = next_seg['class']
            elif prev_seg and prev_seg['class'] != 'UNKNOWN':
                merge_target = prev_seg['class']
            elif next_seg and prev_seg:
                # Both UNKNOWN, merge with larger
                if next_seg['span'] > prev_seg['span']:
                    merge_target = 'UNKNOWN'  # Merge into next
                else:
                    merge_target = 'UNKNOWN'  # Merge into prev
            elif next_seg:
                merge_target = next_seg['class']
            elif prev_seg:
                merge_target = prev_seg['class']
            else:
                # Isolated segment, mark as UNKNOWN
                merge_target = 'UNKNOWN'

            # Apply merge
            classes[seg['start']:seg['end']+1] = merge_target

        return classes

    def _enforce_monotonic_sequence(
        self,
        classes: np.ndarray,
        distances: np.ndarray
    ) -> np.ndarray:
        """
        Enforce monotonic land-to-water sequence.

        Expected order: VEG_DUNES -> DRY_BEACH -> BEACH_WET -> WATER

        Args:
            classes: Array of class labels
            distances: Array of distances

        Returns:
            Array with monotonic sequence enforced
        """
        classes = classes.copy()

        # Define sequence order (lower number = more landward)
        sequence_order = {
            'VEG_DUNES': 0,
            'DRY_BEACH': 1,
            'BEACH_WET': 2,
            'WATER': 3,
            'WAVE_CRESTS': 3,  # Same as water
            'UNKNOWN': -1  # Can appear anywhere
        }

        # Track which classes we've seen (except UNKNOWN)
        seen_classes = set()
        max_order_seen = -1

        for i in range(len(classes)):
            current_class = classes[i]

            if current_class == 'UNKNOWN':
                continue  # UNKNOWN can appear anywhere

            current_order = sequence_order.get(current_class, -1)

            # Check if we've seen this class before (violation of monotonic rule)
            if current_class in seen_classes:
                # Already seen this class - should not repeat
                # Convert to UNKNOWN or adjacent valid class
                classes[i] = 'UNKNOWN'
                continue

            # Check if this class is out of sequence (going backward)
            if current_order < max_order_seen:
                # Going backward - not allowed
                # Convert to the max class we've seen so far
                for cls, order in sequence_order.items():
                    if order == max_order_seen:
                        classes[i] = cls
                        break
            else:
                # Valid forward progression
                seen_classes.add(current_class)
                max_order_seen = current_order

        return classes

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
