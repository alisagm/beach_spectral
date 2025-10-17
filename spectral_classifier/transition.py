"""
Transition zone detection module for identifying beach-water interfaces.
"""

import logging
from typing import List, Dict
import numpy as np
import pandas as pd
from .config import THRESHOLDS, EDGE_BUFFER_M

logger = logging.getLogger(__name__)


class TransitionDetector:
    """Detect transition zones between landcover classes."""

    def __init__(self, thresholds: Dict = None):
        """
        Initialize detector with threshold parameters.

        Args:
            thresholds: Optional custom threshold dictionary (uses config default if None)
        """
        self.thresholds = thresholds if thresholds is not None else THRESHOLDS

    def find_transitions(
        self,
        features: pd.DataFrame,
        landcover: pd.DataFrame = None
    ) -> List[Dict]:
        """
        Identify zone boundary transitions using boundary-type-specific detection methods.

        PHASE 2 APPROACH: Different boundary types have different spectral signatures:
        1. VEG_DUNES->BEACH_DRY: Inflection point detection (second derivative)
        2. BEACH_WET->WATER: RGB foam peak detection
        3. BEACH_DRY->BEACH_WET: Derivative magnitude (current method, works well)

        Strategy:
        1. Run all three specialized detectors in parallel
        2. Merge nearby detections (keep highest confidence within 5m)
        3. Add class transition information if available
        4. Filter by minimum confidence threshold (0.60)

        Args:
            features: DataFrame with spectral features (must include Phase 2 features)
            landcover: Optional DataFrame with classified landcover

        Returns:
            List of transition dictionaries with location, type, and confidence
        """
        if landcover is None:
            landcover = features

        logger.debug("Detecting transitions using boundary-type-specific methods (Phase 2)")

        all_transitions = []

        # Method 1: VEG_DUNES boundaries (inflection detection)
        veg_transitions = self._detect_vegetation_boundaries(features, landcover)
        all_transitions.extend(veg_transitions)

        # Method 2: Surf zone boundaries (RGB foam detection)
        surf_transitions = self._detect_surf_zone_boundaries(features, landcover)
        all_transitions.extend(surf_transitions)

        # Method 3: Dry/wet boundaries (derivative magnitude - current method)
        dw_transitions = self._detect_dry_wet_boundaries(features, landcover)

        # PHASE 6B: Apply relative ranking for shell line selection
        # This ensures we pick the BEST candidate even if weak, and enforces uniqueness
        dw_transitions = self._select_best_shell_line_candidate(dw_transitions, features)

        all_transitions.extend(dw_transitions)

        logger.debug(f"Total candidates from all methods: {len(all_transitions)}")

        # Remove duplicates (merge detections within 5m)
        transitions = self._merge_nearby_detections(all_transitions, min_separation=5.0)

        # Add class information if available
        if 'predicted_class' in landcover.columns:
            transitions = self._add_class_transition_info(transitions, landcover)

        # Filter and validate
        validated_transitions = self._filter_transitions(
            transitions,
            features,
            landcover
        )

        logger.info(f"Detected {len(validated_transitions)} zone boundaries (Phase 2)")
        if validated_transitions:
            # Log detection method breakdown
            method_counts = {}
            for t in validated_transitions:
                method = t.get('detection_method', 'unknown')
                method_counts[method] = method_counts.get(method, 0) + 1
            logger.info(f"  Detection methods: {method_counts}")

        return validated_transitions

    def _check_sustainability(
        self,
        derivative: pd.Series,
        index: int,
        threshold: float,
        min_consecutive: int = None
    ) -> bool:
        """
        Check if derivative drop is sustained over multiple consecutive points.

        This helps distinguish true boundaries (sustained drops) from transient spikes
        caused by wave crests or noise.

        Args:
            derivative: Derivative series (e.g., nir_d1_smooth)
            index: Index to check
            threshold: Threshold value (negative for drops, e.g., -4.38)
            min_consecutive: Minimum consecutive points below threshold
                           (uses config default if None)

        Returns:
            True if sustained drop detected, False otherwise
        """
        if min_consecutive is None:
            min_consecutive = self.thresholds.get('derivative_sustainability_points', 3)

        # Check if we have enough points after this index
        if index + min_consecutive > len(derivative):
            return False

        # Count consecutive points below threshold starting from index
        consecutive_count = 0
        for i in range(index, min(index + min_consecutive + 2, len(derivative))):
            if derivative.iloc[i] < threshold:
                consecutive_count += 1
                if consecutive_count >= min_consecutive:
                    return True
            else:
                # Break in consecutive sequence
                break

        return False

    def _check_multi_band_consensus(
        self,
        features: pd.DataFrame,
        index: int,
        nir_threshold: float = None,
        min_bands: int = None
    ) -> tuple:
        """
        Check if multiple spectral bands show derivative drops (consensus).

        This helps distinguish true boundaries (multiple bands agree) from
        single-band artifacts or noise.

        Args:
            features: DataFrame with derivative features (must include smoothed derivatives)
            index: Index to check
            nir_threshold: NIR derivative threshold (uses config default if None)
            min_bands: Minimum number of bands that must agree (uses config default if None)

        Returns:
            Tuple of (consensus_reached: bool, num_bands_agreeing: int, band_magnitudes: dict)
        """
        if nir_threshold is None:
            nir_threshold = self.thresholds.get('nir_derivative_threshold', -4.38)

        if min_bands is None:
            min_bands = self.thresholds.get('multi_band_consensus_required', 2)

        # Get scaling factors for other bands
        scaling = self.thresholds.get('derivative_scaling', {
            'red': 0.85,
            'green': 0.80,
            'blue': 0.75
        })

        # Check each band's derivative at this index
        bands_agreeing = 0
        band_magnitudes = {}

        # NIR is always included (primary signal)
        if 'nir_d1_smooth' in features.columns:
            nir_val = features['nir_d1_smooth'].iloc[index]
            band_magnitudes['nir'] = nir_val
            if nir_val < nir_threshold:
                bands_agreeing += 1

        # Check Red band
        if 'red_d1_smooth' in features.columns:
            red_threshold = nir_threshold * scaling['red']
            red_val = features['red_d1_smooth'].iloc[index]
            band_magnitudes['red'] = red_val
            if red_val < red_threshold:
                bands_agreeing += 1

        # Check Green band
        if 'green_d1_smooth' in features.columns:
            green_threshold = nir_threshold * scaling['green']
            green_val = features['green_d1_smooth'].iloc[index]
            band_magnitudes['green'] = green_val
            if green_val < green_threshold:
                bands_agreeing += 1

        # Check Blue band
        if 'blue_d1_smooth' in features.columns:
            blue_threshold = nir_threshold * scaling['blue']
            blue_val = features['blue_d1_smooth'].iloc[index]
            band_magnitudes['blue'] = blue_val
            if blue_val < blue_threshold:
                bands_agreeing += 1

        consensus_reached = bands_agreeing >= min_bands

        return consensus_reached, bands_agreeing, band_magnitudes

    def _validate_rg_pattern_for_boundary_type(
        self,
        features: pd.DataFrame,
        index: int,
        boundary_type: str
    ) -> tuple:
        """
        Validate R/G ratio pattern matches expected boundary type signature.

        PHASE 3: Zone-specific pattern validation using R/G ratio.
        Different boundary types have different expected R/G signatures.

        Args:
            features: DataFrame with R/G ratio feature
            index: Index of boundary candidate
            boundary_type: Type of boundary ('dry_wet', 'veg', 'surf')

        Returns:
            Tuple of (is_valid: bool, confidence_adjustment: float)
        """
        if 'red_green_ratio' not in features.columns:
            return True, 0.0  # No validation if feature unavailable

        rg_ratio = features['red_green_ratio'].iloc[index]

        # Define expected patterns by boundary type
        if boundary_type == 'dry_wet':
            # BEACH_DRY->BEACH_WET: R ~ G (ratio near 1.0)
            expected_min, expected_max = 0.95, 1.15
            if expected_min <= rg_ratio <= expected_max:
                # Perfect match - boost confidence
                confidence_boost = 0.05
                return True, confidence_boost
            elif 0.90 <= rg_ratio <= 1.20:
                # Acceptable range - neutral
                return True, 0.0
            else:
                # Outside expected range - reject
                return False, 0.0

        elif boundary_type == 'veg':
            # VEG_DUNES->BEACH_DRY: Variable R>G typical, but wide range acceptable
            # Don't enforce strict pattern (too variable)
            return True, 0.0

        elif boundary_type == 'surf':
            # BEACH_WET->WATER: R>G transitioning to R<G
            # Look for decreasing R/G trend if derivative available
            if 'rg_ratio_d1_smooth' in features.columns:
                rg_deriv = features['rg_ratio_d1_smooth'].iloc[index]
                if rg_deriv < -0.01:  # Decreasing R/G (toward water)
                    return True, 0.03  # Small confidence boost
            return True, 0.0

        return True, 0.0

    # ========================================================================
    # LEGACY DETECTION METHODS
    # Phase 1 detection methods were removed in Phase 8 (2025-10-17).
    # These were superseded by Phase 2+ boundary-type-specific methods:
    #   - _detect_vegetation_boundaries() (inflection detection)
    #   - _detect_surf_zone_boundaries() (RGB foam detection)
    #   - _detect_dry_wet_boundaries() (derivative magnitude)
    # Historical code available in git history before Phase 8.
    # ========================================================================

    def _add_class_transition_info(
        self,
        transitions: List[Dict],
        landcover: pd.DataFrame
    ) -> List[Dict]:
        """
        Add class transition information to detected boundaries.

        Args:
            transitions: List of transitions from NIR derivative detection
            landcover: DataFrame with predicted classes

        Returns:
            Updated transitions with from_class and to_class info
        """
        if 'predicted_class' not in landcover.columns:
            return transitions

        classes = landcover['predicted_class'].values

        for transition in transitions:
            idx = transition['index']

            # Get classes before and after transition
            if idx > 0 and idx < len(classes) - 1:
                # Look back to find non-UNKNOWN class
                from_class = 'UNKNOWN'
                for i in range(idx, max(-1, idx-10), -1):
                    if classes[i] != 'UNKNOWN':
                        from_class = classes[i]
                        break

                # Look forward to find non-UNKNOWN class
                to_class = 'UNKNOWN'
                for i in range(idx+1, min(len(classes), idx+10)):
                    if classes[i] != 'UNKNOWN':
                        to_class = classes[i]
                        break

                transition['from_class'] = from_class
                transition['to_class'] = to_class
                transition['boundary_type'] = f"{from_class}->{to_class}"

                # Boost confidence if it's a major boundary
                if self._is_beach_water_transition(from_class, to_class):
                    transition['confidence'] = min(transition['confidence'] + 0.1, 0.95)

        return transitions

    def _is_beach_water_transition(
        self,
        class1: str,
        class2: str
    ) -> bool:
        """
        Check if transition represents beach-water interface.

        Important transitions for boundary detection:
        - DRY_BEACH <-> BEACH_WET (wet/dry boundary)
        - BEACH_WET <-> WATER (beach/water boundary)
        - DRY_BEACH <-> WATER (sharp boundary, less common)
        """
        # Land classes (ordered from inland to ocean)
        land_classes = ['VEG_DUNES', 'DRY_BEACH']
        # Transitional zone
        transitional_classes = ['BEACH_WET']
        # Water classes
        water_classes = ['WATER', 'WAVE_CRESTS']

        # Critical transitions (high priority for boundary detection)
        # 1. Wet beach to water (primary beach/water boundary)
        if (class1 == 'BEACH_WET' and class2 in water_classes) or \
           (class2 == 'BEACH_WET' and class1 in water_classes):
            return True

        # 2. Dry beach to wet beach (wet/dry sand boundary)
        if (class1 == 'DRY_BEACH' and class2 == 'BEACH_WET') or \
           (class2 == 'DRY_BEACH' and class1 == 'BEACH_WET'):
            return True

        # 3. Dry land directly to water (sharp boundary, less common)
        if (class1 in land_classes and class2 in water_classes) or \
           (class2 in land_classes and class1 in water_classes):
            return True

        return False

    def _filter_transitions(
        self,
        transitions: List[Dict],
        features: pd.DataFrame,
        landcover: pd.DataFrame
    ) -> List[Dict]:
        """
        Filter transitions to keep only high-quality zone boundaries.

        Filtering criteria:
        - Minimum confidence: 0.75 (PHASE 2 FIX: increased from 0.60)
        - PHASE 2 FIX: Zone-aware filtering (reject within-zone transitions)
        - Avoid edge effects (within 5m of transect start/end)
        - Already enforced minimum separation in detection

        Args:
            transitions: List of detected transitions
            features: Feature DataFrame
            landcover: Landcover DataFrame

        Returns:
            Filtered list of validated transitions
        """
        if not transitions:
            return []

        # Get transect extent
        min_dist = features['distance'].min()
        max_dist = features['distance'].max()
        edge_buffer = EDGE_BUFFER_M  # PHASE 2 FIX: Increased from 5.0m to 10.0m

        # PHASE 2 FIX: Increased confidence threshold to reduce false positives
        # PHASE 6C: Relax threshold for fallback mode detections
        min_confidence = self.thresholds.get('min_boundary_confidence', 0.75)

        filtered = []

        for transition in transitions:
            # PHASE 6C: Use relaxed threshold for fallback mode candidates
            detection_mode = transition.get('detection_mode', 'strict')
            if detection_mode == 'fallback':
                effective_min_confidence = 0.50  # Relaxed for fallback
            else:
                effective_min_confidence = min_confidence

            # Filter by confidence (stricter threshold)
            if transition['confidence'] < effective_min_confidence:
                continue

            # Filter edge effects
            if (transition['distance'] < min_dist + edge_buffer or
                transition['distance'] > max_dist - edge_buffer):
                continue

            # PHASE 2 FIX: Zone-aware filtering - reject within-zone transitions
            # These are the primary source of false positives (wave crests, vegetation noise)
            from_class = transition.get('from_class', 'UNKNOWN')
            to_class = transition.get('to_class', 'UNKNOWN')

            # Reject transitions within the same zone (not real boundaries)
            if from_class == to_class and from_class != 'UNKNOWN':
                logger.debug(f"Rejecting within-zone transition at {transition['distance']:.1f}m "
                           f"({from_class}->{to_class})")
                continue

            # PHASE 4 ENHANCEMENT: Stricter filtering for dry beach zone
            # Check if this is a DRY_BEACH->DRY_BEACH transition by examining spectral variability
            if from_class == 'DRY_BEACH' and to_class == 'DRY_BEACH':
                # Check spectral variability in vicinity
                idx = transition['index']
                window_start = max(0, idx - 5)
                window_end = min(len(features), idx + 5)

                # Check NIR variability in window
                if 'nir' in features.columns:
                    nir_std = features['nir'].iloc[window_start:window_end].std()

                    # Low variability threshold from config (default 15.0)
                    min_variability = self.thresholds.get('min_nir_variability', 15.0)

                    if nir_std < min_variability:
                        # Low variability = not a real boundary
                        logger.debug(f"Rejecting low-variability DRY->DRY at {transition['distance']:.1f}m "
                                   f"(NIR std={nir_std:.1f})")
                        continue

            # Also reject UNKNOWN->UNKNOWN (edge artifacts)
            if from_class == 'UNKNOWN' and to_class == 'UNKNOWN':
                continue

            # Add transition flag to landcover
            idx = transition['index']
            if 0 <= idx < len(landcover):
                landcover.loc[idx, 'transition_flag'] = True

            filtered.append(transition)

        logger.debug(f"Zone-aware filtering: {len(transitions)} -> {len(filtered)} transitions")

        return filtered

    def _validate_context(
        self,
        index: int,
        features: pd.DataFrame,
        landcover: pd.DataFrame
    ) -> bool:
        """
        Validate transition using neighboring points.

        Args:
            index: Index of transition point
            features: Feature DataFrame
            landcover: Landcover DataFrame

        Returns:
            True if context supports transition
        """
        window = 3

        # Check if we have enough context
        if index < window or index >= len(landcover) - window:
            return False

        # Check for consistent class pattern before/after
        before_classes = landcover.iloc[index-window:index]['predicted_class'].values
        after_classes = landcover.iloc[index+1:index+window+1]['predicted_class'].values

        # Transition is valid if classes are consistent on both sides
        before_consistent = len(set(before_classes)) <= 2
        after_consistent = len(set(after_classes)) <= 2

        return before_consistent and after_consistent

    # ========================================================================
    # PHASE 2B: Boundary-Type-Specific Detection Methods
    # ========================================================================

    def _detect_vegetation_boundaries(
        self,
        features: pd.DataFrame,
        landcover: pd.DataFrame
    ) -> List[Dict]:
        """
        Detect VEG_DUNES->BEACH_DRY boundaries using inflection point detection.

        Visual signature: Curvature change (second derivative zero-crossing)
        Optimal smoothing: window 7-9
        Pattern: steep increase -> steep drop -> **inflection at 0** -> increase -> shallow

        Key insight: These boundaries are NOT characterized by first derivative drops,
        but by INFLECTION POINTS where the curvature changes sign.

        Args:
            features: DataFrame with computed features
            landcover: DataFrame with predicted classes

        Returns:
            List of transition dictionaries
        """
        # Use coarse-scale second derivative (smooths vegetation noise)
        nir_d2 = features.get('nir_d2_w7', features.get('curvature'))
        distance = features['distance']
        nir = features['nir']

        transitions = []

        # Configuration from THRESHOLDS (PHASE 2 FIX: stricter thresholds)
        veg_config = self.thresholds.get('boundary_thresholds', {}).get('veg_boundaries', {})
        threshold = veg_config.get('second_deriv_threshold', 1.0)  # PHASE 2 FIX: 0.5 -> 1.0
        trend_window = veg_config.get('trend_change_window', 10)
        min_nir_change = veg_config.get('min_nir_change', 20)     # PHASE 2 FIX: 10 -> 20

        # Find zero-crossings (inflection points)
        for i in range(1, len(nir_d2) - 1):
            # Check for sign change in second derivative
            if nir_d2.iloc[i-1] * nir_d2.iloc[i+1] < 0:
                # Verify this is a significant inflection
                curvature_change = abs(nir_d2.iloc[i-1] - nir_d2.iloc[i+1])

                if curvature_change > threshold:
                    # Verify trend pattern: check if we have the expected
                    # increase -> decrease -> increase pattern
                    window_start = max(0, i - trend_window)
                    window_end = min(len(nir), i + trend_window)

                    if window_end - window_start < trend_window:
                        continue

                    # Simple pattern check: NIR before should be different from after
                    nir_before = nir.iloc[window_start:i].mean()
                    nir_after = nir.iloc[i:window_end].mean()

                    # VEG_DUNES typically has higher NIR than BEACH_DRY initially, then drops
                    # PHASE 2 FIX: Require larger NIR difference (10 -> 20 units)
                    if nir_before > nir_after + min_nir_change:
                        # Calculate confidence based on curvature change magnitude
                        # PHASE 2 FIX: More conservative confidence scaling
                        confidence = min(0.65 + (curvature_change / 10.0), 0.85)

                        # PHASE 3: Zone-specific R/G pattern validation
                        # For vegetation boundaries, R/G pattern is variable, so validation is permissive
                        is_valid, conf_adjustment = self._validate_rg_pattern_for_boundary_type(
                            features, i, 'veg'
                        )
                        if not is_valid:
                            logger.debug(f"  Rejected at {distance.iloc[i]:.1f}m: "
                                       f"R/G pattern doesn't match veg signature")
                            continue
                        confidence = min(confidence + conf_adjustment, 0.90)

                        transitions.append({
                            'index': i,
                            'distance': distance.iloc[i],
                            'type': 'vegetation_inflection',
                            'confidence': confidence,
                            'magnitude': curvature_change,
                            'nir_value': nir.iloc[i],
                            'detection_method': 'inflection_point'
                        })

        logger.debug(f"Found {len(transitions)} vegetation boundary candidates (inflection detection)")

        return transitions

    def _detect_surf_zone_boundaries(
        self,
        features: pd.DataFrame,
        landcover: pd.DataFrame
    ) -> List[Dict]:
        """
        Detect BEACH_WET->WATER boundaries using RGB foam detection.

        Visual signature: Small RGB bump from breaking surf
        Optimal smoothing: window 7-11

        Key insight: The foam signature in RGB bands is often MORE IMPORTANT
        than the NIR drop for this boundary type.

        Args:
            features: DataFrame with computed features
            landcover: DataFrame with predicted classes

        Returns:
            List of transition dictionaries
        """
        # Look for RGB peaks + NIR drop combination
        has_foam = features.get('has_rgb_foam_peak', pd.Series(False, index=features.index))
        nir_d1_smooth = features.get('nir_d1_w7', features.get('nir_d1_smooth'))
        distance = features['distance']
        nir = features['nir']

        transitions = []

        # Configuration from THRESHOLDS (PHASE 2 FIX: require NIR drop)
        surf_config = self.thresholds.get('boundary_thresholds', {}).get('surf_zone', {})
        nir_threshold = surf_config.get('nir_threshold', -2.0)
        require_nir_drop = surf_config.get('require_nir_drop', True)  # PHASE 2 FIX: Now required

        for i in range(len(features)):
            # Primary signal: RGB foam peak
            if has_foam.iloc[i]:
                # Secondary confirmation: moderate NIR drop
                nir_deriv = nir_d1_smooth.iloc[i] if i < len(nir_d1_smooth) else 0

                # PHASE 2 FIX: Require NIR drop for confirmation (removed "or True")
                # This prevents detecting every wave crest as a boundary
                if require_nir_drop:
                    # NIR drop is REQUIRED
                    if nir_deriv < nir_threshold:
                        # Both foam and NIR signals agree - high confidence
                        confidence = 0.82
                    else:
                        # Foam peak without NIR drop - likely wave crest, skip it
                        continue
                else:
                    # Legacy behavior: accept foam-only (less strict)
                    if nir_deriv < nir_threshold:
                        confidence = 0.80  # Both signals agree
                    else:
                        confidence = 0.70  # Only foam signal

                # PHASE 3: Zone-specific R/G pattern validation
                # For surf zone, look for decreasing R/G trend (R>G to R<G transition)
                is_valid, conf_adjustment = self._validate_rg_pattern_for_boundary_type(
                    features, i, 'surf'
                )
                if not is_valid:
                    logger.debug(f"  Rejected at {distance.iloc[i]:.1f}m: "
                               f"R/G pattern doesn't match surf signature")
                    continue
                confidence = min(confidence + conf_adjustment, 0.90)

                transitions.append({
                    'index': i,
                    'distance': distance.iloc[i],
                    'type': 'surf_zone_foam',
                    'confidence': confidence,
                    'magnitude': nir_deriv,
                    'nir_value': nir.iloc[i],
                    'detection_method': 'rgb_foam_peak'
                })

        logger.debug(f"Found {len(transitions)} surf zone boundary candidates (foam detection)")

        return transitions

    def _detect_dry_wet_boundaries(
        self,
        features: pd.DataFrame,
        landcover: pd.DataFrame
    ) -> List[Dict]:
        """
        Detect BEACH_DRY->BEACH_WET boundaries using derivative magnitude.

        Visual signature: Slope steepening
        Optimal smoothing: window 5-9

        PHASE 6: Enhanced with comprehensive context validation and composite scoring
        based on empirical analysis of 20 manually classified transects.

        Args:
            features: DataFrame with computed features
            landcover: DataFrame with predicted classes

        Returns:
            List of transition dictionaries
        """
        # Use medium-scale smoothing (preserves sharpness)
        nir_d1 = features.get('nir_d1_w5', features.get('nir_d1_smooth'))
        distance = features['distance']
        nir = features['nir']

        transitions = []

        # Configuration from THRESHOLDS
        config = self.thresholds.get('boundary_thresholds', {}).get('dry_wet', {})
        threshold = config.get('nir_threshold', -8.0)  # PHASE 6: Tightened from -4.0
        min_nir_drop_abs = config.get('min_nir_drop_absolute', 39.0)  # PHASE 6C: Relaxed to 5th percentile (was 40)
        brightness_min = config.get('brightness_before_min', 175)  # PHASE 6C: Relaxed from 190 (5th%=183, using 175 for safety)
        nir_before_min = config.get('nir_before_min', 135)  # PHASE 6C: Relaxed from 160 (5th%=139, using 135 for safety)
        var_ratio_min = config.get('variability_ratio_min', 1.5)  # PHASE 6C: Relaxed from 2.0 (some shell lines have lower ratios)
        expected_loc_min = config.get('expected_location_min', 40)  # PHASE 6: New
        expected_loc_max = config.get('expected_location_max', 120)  # PHASE 6: New
        require_sustainability = config.get('require_sustainability', True)
        require_consensus = config.get('require_consensus', True)
        min_nir = 50  # Avoid noise in deep water

        # PHASE 6C: Debug logging to verify config loading
        logger.debug(f"Dry/wet detection config: NIR_drop_min={min_nir_drop_abs}, "
                    f"brightness_min={brightness_min}, NIR_before_min={nir_before_min}, "
                    f"var_ratio_min={var_ratio_min}, expected_zone={expected_loc_min}-{expected_loc_max}m")

        # Find candidate transition points
        for i in range(len(nir_d1)):
            # Check if this is a significant negative slope
            if not (nir_d1.iloc[i] < threshold and nir.iloc[i] > min_nir):
                continue

            # Check if local minimum in derivative (peak negative slope)
            is_local_min = True
            if i > 0 and nir_d1.iloc[i] > nir_d1.iloc[i-1]:
                is_local_min = False
            if i < len(nir_d1) - 1 and nir_d1.iloc[i] > nir_d1.iloc[i+1]:
                is_local_min = False

            if not is_local_min:
                continue

            # PHASE 6D: Context Validation - SOFT PENALTIES (not hard rejections)
            # Compute context features for confidence scoring
            nir_drop_abs = 0.0
            if i >= 5:
                nir_before = nir.iloc[i-5:i].mean()
                nir_at = nir.iloc[i]
                nir_drop_abs = nir_before - nir_at

            brightness_before = 0.0
            if i >= 5 and 'brightness' in features.columns:
                brightness_before = features['brightness'].iloc[i-5:i].mean()

            nir_mean_before = 0.0
            if i >= 5:
                nir_mean_before = nir.iloc[i-5:i].mean()

            var_ratio = 0.0
            if i >= 5 and i < len(features) - 5 and 'variability' in features.columns:
                var_before = features['variability'].iloc[i-5:i].mean()
                var_after = features['variability'].iloc[i:i+5].mean()
                var_ratio = var_after / (var_before + 1e-6)  # Avoid divide by zero

            magnitude = abs(nir_d1.iloc[i])

            # PHASE 6: Composite Confidence Scoring
            # Base confidence from derivative magnitude (stricter scaling for -8.0 threshold)
            base_confidence = 0.50 + (magnitude - 8.0) / 30.0
            base_confidence = max(0.50, min(base_confidence, 0.70))  # Clamp [0.50, 0.70]

            confidence = base_confidence

            # Bonus 1: Strong derivative (empirical: -13.8 units/m)
            if magnitude > 12.0:
                confidence += 0.10

            # Bonus 2: Large absolute NIR drop (empirical: 55 units)
            if nir_drop_abs > 50:
                confidence += 0.08

            # Bonus 3: Large brightness drop
            if i >= 5 and 'brightness' in features.columns:
                brightness_at = features['brightness'].iloc[i]
                brightness_drop = brightness_before - brightness_at
                if brightness_drop > 25:
                    confidence += 0.05

            # Bonus 4: High variability ratio (empirical: 5.9x)
            if var_ratio > 4.0:
                confidence += 0.07

            # Bonus 5: Sustained drop (existing check)
            is_sustained = False
            if require_sustainability:
                is_sustained = self._check_sustainability(nir_d1, i, threshold)
                if is_sustained:
                    confidence += 0.08

            # Bonus 6: Multi-band consensus (existing check)
            consensus_reached = False
            num_bands = 1
            band_mags = {}
            if require_consensus:
                consensus_reached, num_bands, band_mags = self._check_multi_band_consensus(
                    features, i, threshold
                )
                if num_bands >= 3:
                    confidence += 0.08

            # Bonus 7: R/G ratio near 1.0 (empirical: 1.02+-0.04)
            use_rg_ratio = config.get('use_rg_ratio', True)
            if use_rg_ratio:
                is_valid, conf_adjustment = self._validate_rg_pattern_for_boundary_type(
                    features, i, 'dry_wet'
                )
                if not is_valid:
                    logger.debug(f"  Rejected at {distance.iloc[i]:.1f}m: "
                               f"R/G pattern doesn't match dry_wet signature")
                    continue
                confidence += conf_adjustment

            # Bonus 8: Expected location (40-120m from start)
            dist = distance.iloc[i]
            in_expected_zone = expected_loc_min <= dist <= expected_loc_max
            if in_expected_zone:
                confidence += 0.05

            # PHASE 6C: VEG->DRY discrimination - Apply penalty for candidates in VEG zone
            # VEG->DRY boundaries typically occur at 40-70m, shell lines at 90-120m
            veg_zone_min = config.get('expected_location_veg_dry_min', 40)
            veg_zone_max = config.get('expected_location_veg_dry_max', 70)

            if veg_zone_min <= dist <= veg_zone_max:
                # Candidate is in typical VEG->DRY zone
                # PHASE 6E FIX 2: Hard rejection for low var_ratio in VEG zone
                # VEG->DRY boundaries have low spectral variability, shell lines have high variability
                if var_ratio < 1.0:
                    logger.debug(f"  REJECTED at {dist:.1f}m: VEG zone with low var_ratio ({var_ratio:.2f})")
                    continue  # Hard rejection

                # Apply penalty unless it has very strong shell line signature
                if nir_drop_abs < 50:  # Shell lines typically >50 units
                    confidence -= 0.20  # Heavy penalty
                    logger.debug(f"  Penalty at {dist:.1f}m: likely VEG->DRY boundary (in zone {veg_zone_min}-{veg_zone_max}m, NIR_drop={nir_drop_abs:.1f})")

            # PHASE 6D: SOFT PENALTIES for weak context features
            # Instead of hard rejection, apply graduated penalties based on how far below thresholds

            # Penalty 1: Weak NIR drop (target: 39+, empirical: 55)
            if nir_drop_abs < min_nir_drop_abs:
                penalty = (min_nir_drop_abs - nir_drop_abs) / min_nir_drop_abs  # 0.0-1.0 scale
                confidence -= 0.15 * penalty  # Up to -0.15 for very weak drops
                logger.debug(f"  Soft penalty: NIR_drop={nir_drop_abs:.1f} < {min_nir_drop_abs} (penalty={0.15*penalty:.2f})")

            # Penalty 2: Low brightness before (target: 175+, empirical: 202)
            if brightness_before > 0 and brightness_before < brightness_min:
                penalty = (brightness_min - brightness_before) / brightness_min
                confidence -= 0.10 * penalty  # Up to -0.10 for very dark sand
                logger.debug(f"  Soft penalty: brightness={brightness_before:.1f} < {brightness_min} (penalty={0.10*penalty:.2f})")

            # Penalty 3: Low NIR before (target: 135+, empirical: 172)
            if nir_mean_before > 0 and nir_mean_before < nir_before_min:
                penalty = (nir_before_min - nir_mean_before) / nir_before_min
                confidence -= 0.10 * penalty  # Up to -0.10 for low NIR
                logger.debug(f"  Soft penalty: NIR_before={nir_mean_before:.1f} < {nir_before_min} (penalty={0.10*penalty:.2f})")

            # Penalty 4: Low variability ratio (target: 1.5+, empirical: 5.9)
            if var_ratio > 0 and var_ratio < var_ratio_min:
                penalty = (var_ratio_min - var_ratio) / var_ratio_min
                confidence -= 0.08 * penalty  # Up to -0.08 for smooth transitions
                logger.debug(f"  Soft penalty: var_ratio={var_ratio:.2f} < {var_ratio_min} (penalty={0.08*penalty:.2f})")

            confidence = min(confidence, 0.95)  # Cap at 0.95
            confidence = max(confidence, 0.30)  # Floor at 0.30 (very weak candidates)

            # PHASE 6: Enhanced logging for accepted candidates
            logger.debug(f"  Candidate at {distance.iloc[i]:.1f}m: "
                       f"conf={confidence:.2f}, NIR_drop={nir_drop_abs:.1f}, "
                       f"deriv={magnitude:.2f}, var_ratio={var_ratio:.2f}, "
                       f"brightness_before={brightness_before:.1f}, "
                       f"in_zone={in_expected_zone}")

            # Only add if confidence meets minimum threshold
            # Will be filtered later by _filter_transitions (min_confidence = 0.75)
            if confidence >= 0.60:
                transitions.append({
                    'index': i,
                    'distance': distance.iloc[i],
                    'type': 'dry_wet_derivative',
                    'confidence': confidence,
                    'magnitude': nir_d1.iloc[i],
                    'nir_value': nir.iloc[i],
                    'nir_drop_abs': nir_drop_abs,
                    'brightness_before': brightness_before,
                    'variability_ratio': var_ratio,
                    'in_expected_zone': in_expected_zone,
                    'is_sustained': is_sustained,
                    'num_bands_agreeing': num_bands,
                    'band_magnitudes': band_mags,
                    'detection_method': 'derivative_magnitude'
                })

        logger.debug(f"Found {len(transitions)} dry/wet boundary candidates (derivative magnitude)")

        return transitions

    def _merge_nearby_detections(
        self,
        transitions: List[Dict],
        min_separation: float = 5.0
    ) -> List[Dict]:
        """
        Merge detections that are within min_separation meters of each other.

        PHASE 6C FIX: Only merge detections of the SAME TYPE (detection_method).
        Different boundary types (inflection_point vs derivative_magnitude) represent
        different physical phenomena and should not be merged together.

        Args:
            transitions: List of all detected transitions
            min_separation: Minimum distance between boundaries (meters)

        Returns:
            Filtered list with nearby detections merged
        """
        if not transitions:
            return []

        # Sort by distance
        sorted_transitions = sorted(transitions, key=lambda x: x['distance'])

        merged = []
        i = 0

        while i < len(sorted_transitions):
            current = sorted_transitions[i]
            current_method = current.get('detection_method', 'unknown')
            candidates = [current]

            # Collect all transitions within separation window WITH SAME DETECTION METHOD
            j = i + 1
            while j < len(sorted_transitions):
                next_trans = sorted_transitions[j]
                distance_diff = next_trans['distance'] - current['distance']

                if distance_diff < min_separation:
                    # PHASE 6C FIX: Only merge if same detection method
                    next_method = next_trans.get('detection_method', 'unknown')
                    if next_method == current_method:
                        candidates.append(next_trans)
                        logger.debug(f"Merging {next_method} detections at {current['distance']:.1f}m and {next_trans['distance']:.1f}m")
                    else:
                        # Different detection type - keep both, don't merge
                        logger.debug(f"NOT merging {current_method} at {current['distance']:.1f}m with {next_method} at {next_trans['distance']:.1f}m (different types)")
                        # Don't add to candidates, but also don't advance i past it
                        # We'll process it in the next iteration
                    j += 1
                else:
                    break

            # Keep the one with highest confidence from candidates of same type
            if len(candidates) > 1:
                best = max(candidates, key=lambda x: x['confidence'])
                logger.debug(f"Selected best {current_method} with conf={best['confidence']:.2f} from {len(candidates)} candidates")
            else:
                best = candidates[0]

            merged.append(best)

            # Move to next unprocessed detection
            # Only skip candidates that were actually merged (same type)
            i += len(candidates)

        logger.debug(f"Merged {len(transitions)} detections into {len(merged)} final boundaries (type-aware merging)")

        return merged

    def _select_best_shell_line_candidate(
        self,
        candidates: List[Dict],
        features: pd.DataFrame
    ) -> List[Dict]:
        """
        Select the single best shell line candidate from dry/wet boundary detections.

        PHASE 7A: GUARANTEED DETECTION with progressive fallback strategy.
        Always returns exactly ONE shell line boundary using:
        1. STRICT MODE: Phase 6E logic (conf >= 0.75, in expected zone)
        2. RELAXED MODE: Fallback detection (conf >= 0.50, relaxed thresholds)
        3. BEST-AVAILABLE MODE: Highest confidence from any derivative minimum
        4. LAST-RESORT MODE: Expected zone median (95m default)

        This ensures 100% detection rate for Phase 7B boundary correction.

        Args:
            candidates: List of dry/wet boundary candidates from strict mode
            features: Feature DataFrame for context

        Returns:
            List with exactly ONE shell line candidate (guaranteed non-empty)
        """
        config = self.thresholds.get('boundary_thresholds', {}).get('dry_wet', {})
        expected_min = config.get('expected_location_min', 40)
        expected_max = config.get('expected_location_max', 120)

        # PHASE 6E FIX 4: Hard distance bounds to prevent extreme edge artifacts
        # Shell lines cannot occur within 30m of start or beyond 250m (deep water)
        distance_min = 30.0
        distance_max = 250.0

        # MODE 1: Try strict mode selection
        strict_result = self._try_strict_selection(
            candidates, expected_min, expected_max, distance_min, distance_max
        )
        if strict_result:
            return strict_result

        # MODE 2: Try relaxed mode selection
        relaxed_result = self._try_relaxed_selection(
            features, expected_min, expected_max, distance_min, distance_max
        )
        if relaxed_result:
            return relaxed_result

        # MODE 3: Best-available mode (any derivative minimum)
        best_available = self._find_any_derivative_minimum(
            features, distance_min, distance_max
        )
        if best_available:
            logger.warning(f"Using best-available mode: shell line at {best_available['distance']:.1f}m "
                          f"(conf={best_available['confidence']:.2f}, low quality)")
            return [best_available]

        # MODE 4: Last resort - use expected zone median
        fallback = self._create_fallback_boundary(features, expected_min, expected_max)
        logger.warning(f"Using last-resort mode: shell line at {fallback['distance']:.1f}m "
                      f"(expected zone median, very low quality)")
        return [fallback]

    def _try_strict_selection(
        self,
        candidates: List[Dict],
        expected_min: float,
        expected_max: float,
        distance_min: float,
        distance_max: float
    ) -> List[Dict]:
        """
        PHASE 7A MODE 1: Try strict mode selection (Phase 6E logic).

        Returns:
            List with one candidate, or empty list if strict mode fails
        """
        # Filter candidates by hard distance bounds
        if candidates:
            candidates = [c for c in candidates
                          if distance_min <= c['distance'] <= distance_max]
            if not candidates:
                logger.debug(f"Strict mode: No candidates within distance bounds ({distance_min}-{distance_max}m)")
                return []

        # Check if we have candidates in the expected zone
        in_zone = [c for c in candidates
                   if expected_min <= c['distance'] <= expected_max] if candidates else []

        if not in_zone:
            logger.debug(f"Strict mode: No candidates in expected zone ({expected_min}-{expected_max}m)")
            return []

        # Mark as strict mode
        for cand in in_zone:
            cand['detection_mode'] = 'strict'

        # Sort by confidence (highest first)
        in_zone.sort(key=lambda c: c['confidence'], reverse=True)
        best = in_zone[0]

        # Check minimum confidence
        min_confidence = 0.60
        if best['confidence'] < min_confidence:
            logger.debug(f"Strict mode: Best candidate confidence too low ({best['confidence']:.2f} < {min_confidence})")
            return []

        logger.info(f"Selected shell line at {best['distance']:.1f}m "
                   f"(mode=strict, confidence={best['confidence']:.2f}, "
                   f"rank=1/{len(in_zone)} in zone)")
        return [best]

    def _try_relaxed_selection(
        self,
        features: pd.DataFrame,
        expected_min: float,
        expected_max: float,
        distance_min: float,
        distance_max: float
    ) -> List[Dict]:
        """
        PHASE 7A MODE 2: Try relaxed mode selection (fallback detection).

        Returns:
            List with one candidate, or empty list if relaxed mode fails
        """
        logger.debug("Attempting relaxed mode detection...")
        relaxed_candidates = self._detect_with_relaxed_thresholds(features)

        if not relaxed_candidates:
            logger.debug("Relaxed mode: No candidates found")
            return []

        # Filter by distance bounds
        relaxed_candidates = [c for c in relaxed_candidates
                             if distance_min <= c['distance'] <= distance_max]

        if not relaxed_candidates:
            logger.debug("Relaxed mode: No candidates within distance bounds")
            return []

        # Mark as fallback mode
        for cand in relaxed_candidates:
            cand['detection_mode'] = 'fallback'

        # Prefer candidates in expected zone
        in_zone = [c for c in relaxed_candidates
                   if expected_min <= c['distance'] <= expected_max]

        if not in_zone:
            logger.debug("Relaxed mode: No candidates in expected zone, using best anywhere")
            in_zone = relaxed_candidates

        # Sort by confidence
        in_zone.sort(key=lambda c: c['confidence'], reverse=True)
        best = in_zone[0]

        # Check minimum confidence
        min_confidence = 0.50
        if best['confidence'] < min_confidence:
            logger.debug(f"Relaxed mode: Best candidate confidence too low ({best['confidence']:.2f} < {min_confidence})")
            return []

        logger.info(f"Selected shell line at {best['distance']:.1f}m "
                   f"(mode=fallback, confidence={best['confidence']:.2f})")
        return [best]

    def _find_any_derivative_minimum(
        self,
        features: pd.DataFrame,
        distance_min: float,
        distance_max: float
    ) -> Dict:
        """
        PHASE 7A MODE 3: Find strongest derivative minimum regardless of context.

        Returns:
            Single candidate dictionary, or None if no minima found
        """
        logger.debug("Attempting best-available mode (any derivative minimum)...")

        nir_d1 = features.get('nir_d1_w5', features.get('nir_d1_smooth'))
        distance = features['distance']
        nir = features['nir']

        # Relaxed threshold for this mode
        threshold = -3.0  # Looser than -8.0
        min_nir = 30  # Looser than 50

        candidates = []

        for i in range(len(nir_d1)):
            # Check if this is a negative slope
            if not (nir_d1.iloc[i] < threshold and nir.iloc[i] > min_nir):
                continue

            # Check if local minimum
            is_local_min = True
            if i > 0 and nir_d1.iloc[i] > nir_d1.iloc[i-1]:
                is_local_min = False
            if i < len(nir_d1) - 1 and nir_d1.iloc[i] > nir_d1.iloc[i+1]:
                is_local_min = False

            if not is_local_min:
                continue

            dist = distance.iloc[i]

            # Apply hard distance bounds
            if dist < distance_min or dist > distance_max:
                continue

            magnitude = abs(nir_d1.iloc[i])

            candidates.append({
                'index': i,
                'distance': dist,
                'type': 'dry_wet_derivative',
                'confidence': 0.40,  # Low confidence for best-available
                'magnitude': nir_d1.iloc[i],
                'nir_value': nir.iloc[i],
                'detection_mode': 'best_available',
                'detection_method': 'derivative_magnitude_any'
            })

        if not candidates:
            logger.debug("Best-available mode: No derivative minima found")
            return None

        # Select strongest derivative magnitude
        best = max(candidates, key=lambda c: abs(c['magnitude']))
        logger.debug(f"Best-available mode: Found candidate at {best['distance']:.1f}m (magnitude={best['magnitude']:.2f})")
        return best

    def _create_fallback_boundary(
        self,
        features: pd.DataFrame,
        expected_min: float,
        expected_max: float
    ) -> Dict:
        """
        PHASE 7A MODE 4: Create last-resort boundary at expected zone median.

        Returns:
            Single candidate dictionary (always succeeds)
        """
        target_distance = (expected_min + expected_max) / 2.0  # 95m for 40-120 range
        logger.debug(f"Creating last-resort boundary at target {target_distance:.1f}m")

        distance = features['distance']
        nir = features['nir']

        # Find closest point to target with valid NIR
        valid_indices = []
        for i, (dist, nir_val) in enumerate(zip(distance, nir)):
            if nir_val > 30:  # Minimum NIR threshold
                valid_indices.append(i)

        if not valid_indices:
            # Extreme edge case: use midpoint anyway
            i = len(distance) // 2
            logger.warning("Last-resort: No valid NIR values, using midpoint")
        else:
            # Find closest to target
            best_i = min(valid_indices, key=lambda i: abs(distance.iloc[i] - target_distance))
            i = best_i

        return {
            'index': i,
            'distance': float(distance.iloc[i]),
            'type': 'dry_wet_derivative',
            'confidence': 0.30,  # Very low confidence
            'magnitude': 0.0,
            'nir_value': float(nir.iloc[i]),
            'detection_mode': 'last_resort',
            'detection_method': 'expected_zone_median'
        }

    def _detect_with_relaxed_thresholds(
        self,
        features: pd.DataFrame
    ) -> List[Dict]:
        """
        Re-run dry/wet detection with significantly relaxed thresholds for fallback mode.

        PHASE 6C: This method enables detection on difficult transects where shell lines
        have weak spectral signatures that don't meet strict Phase 6 criteria.

        Relaxation strategy:
        - NIR drop absolute: 39 -> 15 (allow much weaker drops)
        - Brightness before: 175 -> 160 (allow darker sand)
        - NIR before: 135 -> 100 (allow much lower NIR values)
        - Variability ratio: 1.5 -> 1.0 (allow smoother transitions)

        Args:
            features: Feature DataFrame with spectral features

        Returns:
            List of candidate dictionaries with relaxed thresholds
        """
        logger.debug("Running fallback detection with relaxed thresholds...")

        # Use medium-scale smoothing (preserves sharpness)
        nir_d1 = features.get('nir_d1_w5', features.get('nir_d1_smooth'))
        distance = features['distance']
        nir = features['nir']

        transitions = []

        # Configuration - RELAXED thresholds for fallback
        config = self.thresholds.get('boundary_thresholds', {}).get('dry_wet', {})
        threshold = config.get('nir_threshold', -8.0)

        # PHASE 6D FALLBACK: Relaxed thresholds with hard minimums to prevent edge artifacts
        min_nir_drop_abs = 20.0  # Was 39, now 20 (PHASE 6D: increased from 15 to prevent edge artifacts)
        brightness_min = 160  # Was 175, now 160
        nir_before_min = 100  # Was 135, now 100 (26% reduction)
        var_ratio_min = 1.2  # Was 1.5, now 1.2 (PHASE 6D: increased from 1.0 to prevent edge artifacts)

        # PHASE 6D: Distance bounds to prevent edge artifacts (0.0m detections)
        distance_min = 30.0  # No shell lines within 30m of transect start
        distance_max = 250.0  # No shell lines beyond 250m (deep water)

        expected_loc_min = config.get('expected_location_min', 40)
        expected_loc_max = config.get('expected_location_max', 120)
        min_nir = 50

        # PHASE 6D: Debug logging for fallback mode config
        logger.debug(f"Fallback detection config: NIR_drop_min={min_nir_drop_abs}, "
                    f"brightness_min={brightness_min}, NIR_before_min={nir_before_min}, "
                    f"var_ratio_min={var_ratio_min}, distance_bounds={distance_min}-{distance_max}m, "
                    f"expected_zone={expected_loc_min}-{expected_loc_max}m")
        logger.debug(f"  Fallback thresholds: NIR_drop>={min_nir_drop_abs}, "
                    f"brightness>={brightness_min}, NIR_before>={nir_before_min}, "
                    f"var_ratio>={var_ratio_min}, dist={distance_min}-{distance_max}m")

        # Find candidate transition points with relaxed criteria
        for i in range(len(nir_d1)):
            # Check if this is a significant negative slope
            if not (nir_d1.iloc[i] < threshold and nir.iloc[i] > min_nir):
                continue

            # Check if local minimum in derivative
            is_local_min = True
            if i > 0 and nir_d1.iloc[i] > nir_d1.iloc[i-1]:
                is_local_min = False
            if i < len(nir_d1) - 1 and nir_d1.iloc[i] > nir_d1.iloc[i+1]:
                is_local_min = False

            if not is_local_min:
                continue

            # Relaxed context validation
            nir_drop_abs = 0.0
            if i >= 5:
                nir_before = nir.iloc[i-5:i].mean()
                nir_at = nir.iloc[i]
                nir_drop_abs = nir_before - nir_at

                if nir_drop_abs < min_nir_drop_abs:
                    continue

            brightness_before = 0.0
            if i >= 5 and 'brightness' in features.columns:
                brightness_before = features['brightness'].iloc[i-5:i].mean()
                if brightness_before < brightness_min:
                    continue

            nir_mean_before = 0.0
            if i >= 5:
                nir_mean_before = nir.iloc[i-5:i].mean()
                if nir_mean_before < nir_before_min:
                    continue

            var_ratio = 0.0
            if i >= 5 and i < len(features) - 5 and 'variability' in features.columns:
                var_before = features['variability'].iloc[i-5:i].mean()
                var_after = features['variability'].iloc[i:i+5].mean()
                var_ratio = var_after / (var_before + 1e-6)

                if var_ratio < var_ratio_min:
                    continue

            magnitude = abs(nir_d1.iloc[i])

            # Confidence scoring - more lenient for fallback mode
            base_confidence = 0.40 + (magnitude - 8.0) / 40.0  # Lower base, gentler slope
            base_confidence = max(0.40, min(base_confidence, 0.60))

            confidence = base_confidence

            # Bonuses (same as strict mode but apply to weaker signals)
            if magnitude > 10.0:
                confidence += 0.08
            if nir_drop_abs > 30:
                confidence += 0.06
            if var_ratio > 3.0:
                confidence += 0.05

            # PHASE 6D: Hard constraint - distance bounds to prevent edge artifacts
            dist = distance.iloc[i]
            if dist < distance_min or dist > distance_max:
                logger.debug(f"  Rejected at {dist:.1f}m: outside valid distance bounds ({distance_min}-{distance_max}m)")
                continue

            # Location bonus
            in_expected_zone = expected_loc_min <= dist <= expected_loc_max
            if in_expected_zone:
                confidence += 0.05

            # VEG zone penalty (same as strict mode)
            veg_zone_min = config.get('expected_location_veg_dry_min', 40)
            veg_zone_max = config.get('expected_location_veg_dry_max', 70)

            if veg_zone_min <= dist <= veg_zone_max:
                if nir_drop_abs < 40:  # Slightly relaxed from 50
                    confidence -= 0.15  # Slightly less harsh penalty

            confidence = min(confidence, 0.85)  # Lower cap for fallback mode

            logger.debug(f"  Fallback candidate at {dist:.1f}m: "
                        f"conf={confidence:.2f}, NIR_drop={nir_drop_abs:.1f}, "
                        f"deriv={magnitude:.2f}, var_ratio={var_ratio:.2f}")

            if confidence >= 0.40:  # Very low threshold for fallback
                transitions.append({
                    'index': i,
                    'distance': distance.iloc[i],
                    'type': 'dry_wet_derivative',
                    'confidence': confidence,
                    'magnitude': nir_d1.iloc[i],
                    'nir_value': nir.iloc[i],
                    'nir_drop_abs': nir_drop_abs,
                    'brightness_before': brightness_before,
                    'variability_ratio': var_ratio,
                    'in_expected_zone': in_expected_zone,
                    'is_sustained': False,  # Not checking in fallback mode
                    'num_bands_agreeing': 1,  # Not checking in fallback mode
                    'band_magnitudes': {},
                    'detection_method': 'derivative_magnitude_relaxed'
                })

        logger.debug(f"Found {len(transitions)} dry/wet boundary candidates (fallback/relaxed)")

        return transitions

    # ========================================================================
    # Boundary Type Classification Helpers
    # ========================================================================

    @staticmethod
    def is_shore_boundary(transition: Dict) -> bool:
        """
        Check if transition represents a shore boundary (BEACH_DRY->BEACH_WET).

        This is the swash line or shell line - the wet/dry sand boundary.
        Most geomorphologically important boundary for beach analysis.

        PHASE 6C FIX: Also accepts transitions based on detection_method when
        class labels are unavailable (e.g., when monotonic smoothing collapses classes).

        Args:
            transition: Transition dictionary with from_class/to_class or detection_method

        Returns:
            True if this is a shore boundary
        """
        from_class = transition.get('from_class', 'UNKNOWN')
        to_class = transition.get('to_class', 'UNKNOWN')
        detection_method = transition.get('detection_method', '')

        # Method 1: Class-based detection (preferred when available)
        if (from_class == 'DRY_BEACH' and to_class == 'BEACH_WET') or \
           (from_class == 'BEACH_WET' and to_class == 'DRY_BEACH'):
            return True

        # Method 2: Detection method fallback (when classes are UNKNOWN)
        # Dry/wet boundaries are shore boundaries by definition
        if detection_method in ['derivative_magnitude', 'derivative_magnitude_relaxed']:
            return True

        return False

    @staticmethod
    def is_water_boundary(transition: Dict) -> bool:
        """
        Check if transition represents a water boundary (BEACH_WET->WATER).

        This is the waterline - the boundary between wet sand and open water.

        Args:
            transition: Transition dictionary with from_class and to_class

        Returns:
            True if this is a water boundary
        """
        from_class = transition.get('from_class', 'UNKNOWN')
        to_class = transition.get('to_class', 'UNKNOWN')

        # Water boundary: BEACH_WET -> WATER
        water_classes = ['WATER', 'WAVE_CRESTS']
        return (from_class == 'BEACH_WET' and to_class in water_classes) or \
               (to_class == 'BEACH_WET' and from_class in water_classes)

    @staticmethod
    def is_vegetation_boundary(transition: Dict) -> bool:
        """
        Check if transition represents a vegetation boundary (VEG_DUNES->BEACH_DRY).

        This is the vegetation/dune to bare sand boundary.

        Args:
            transition: Transition dictionary with from_class and to_class

        Returns:
            True if this is a vegetation boundary
        """
        from_class = transition.get('from_class', 'UNKNOWN')
        to_class = transition.get('to_class', 'UNKNOWN')

        # Vegetation boundary: VEG_DUNES -> DRY_BEACH or BEACH_WET
        beach_classes = ['DRY_BEACH', 'BEACH_WET']
        return (from_class == 'VEG_DUNES' and to_class in beach_classes) or \
               (to_class == 'VEG_DUNES' and from_class in beach_classes)

    @staticmethod
    def filter_by_boundary_type(transitions: List[Dict], boundary_types: str) -> List[Dict]:
        """
        Filter transitions based on requested boundary types.

        PHASE 6C FIX: Enhanced to handle transitions where class labels are UNKNOWN
        by falling back to detection_method field.

        Args:
            transitions: List of detected transitions
            boundary_types: One of 'shore_only', 'waterline', or 'all'

        Returns:
            Filtered list of transitions matching requested types
        """
        if boundary_types == 'all':
            return transitions

        filtered = []
        rejected = []  # Track rejections for debug logging

        for t in transitions:
            accepted = False

            if boundary_types == 'shore_only':
                # Only shore boundaries (BEACH_DRY->BEACH_WET)
                if TransitionDetector.is_shore_boundary(t):
                    filtered.append(t)
                    accepted = True
                else:
                    rejected.append(t)

            elif boundary_types == 'waterline':
                # Shore + water boundaries
                if TransitionDetector.is_shore_boundary(t) or \
                   TransitionDetector.is_water_boundary(t):
                    filtered.append(t)
                    accepted = True
                else:
                    rejected.append(t)

        # Debug logging for rejections
        if rejected:
            logger.debug(f"Rejected {len(rejected)} transitions during boundary_type filtering:")
            for t in rejected:
                from_class = t.get('from_class', 'UNKNOWN')
                to_class = t.get('to_class', 'UNKNOWN')
                method = t.get('detection_method', 'unknown')
                dist = t.get('distance', 0)
                logger.debug(f"  - {dist:.1f}m: {from_class}->{to_class} (method={method})")

        logger.info(f"Filtered {len(transitions)} -> {len(filtered)} transitions "
                   f"(boundary_types={boundary_types})")

        return filtered
