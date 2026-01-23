"""
Shell line detection and transition orchestration.

This module contains:
- TransitionDetector: Main class that orchestrates boundary detection
- Shell line selection logic (best candidate selection)
- Transition filtering and merging
- Static utility methods for boundary classification

The shell line (BEACH_DRY -> BEACH_WET boundary) is the PRIMARY output
of this system. Other boundary types (vegetation, surf zone) are optional
and handled by the optional/boundary_types.py module.
"""

import logging
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

from ..config import THRESHOLDS, EDGE_BUFFER_M, DEFAULT_BOUNDARY_TYPES
from ..utils.data_io import (
    BAND_CONFIG_4BAND, BAND_CONFIG_CIR, BAND_CONFIG_RGB,
    detect_band_mode_from_dataframe
)

# Import detection methods from submodules
from .nir import (
    detect_dry_wet_boundaries,
    detect_vegetation_boundaries,
    detect_surf_zone_boundaries,
    _get_max_confidence,
    _get_min_acceptance_confidence,
)
from .rgb import (
    detect_dry_wet_boundaries_rgb,
    detect_with_relaxed_thresholds_rgb,
    find_any_derivative_minimum_rgb,
)

logger = logging.getLogger(__name__)


class TransitionDetector:
    """
    Detect transition zones between landcover classes.
    
    Primary output: Shell line (BEACH_DRY -> BEACH_WET boundary)
    Optional outputs: Vegetation boundary, surf zone boundary
    
    Automatically selects detection method based on band availability:
    - 4band/CIR: NIR-based detection (full capability)
    - RGB: Brightness-based detection (degraded mode, lower confidence)
    
    Example usage:
        detector = TransitionDetector()
        transitions = detector.find_transitions(features, landcover)
        
        # Filter to shell line only (default)
        shell_lines = TransitionDetector.filter_by_boundary_type(
            transitions, 'shore_only'
        )
    """

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
        landcover: pd.DataFrame = None,
        include_optional_boundaries: bool = False
    ) -> List[Dict]:
        """
        Identify zone boundary transitions.

        Automatically selects detection method based on band availability:
        - 4band/CIR: NIR-based detection (full capability, CIR slightly reduced confidence)
        - RGB: Brightness-based detection (degraded mode, lower confidence)

        Args:
            features: DataFrame with spectral features
            landcover: Optional DataFrame with classified landcover
            include_optional_boundaries: If True, also detect veg and surf boundaries

        Returns:
            List of transition dictionaries with location, type, and confidence
        """
        if landcover is None:
            landcover = features

        # Detect band mode
        band_mode = self._detect_band_mode(features)
        has_nir = band_mode in [BAND_CONFIG_4BAND, BAND_CONFIG_CIR]
        
        logger.debug(f"Detecting transitions (band_mode={band_mode}, has_nir={has_nir})")

        all_transitions = []

        if has_nir:
            # Full NIR-based detection (works for both 4-band and CIR)
            
            # Optional: Vegetation boundaries
            if include_optional_boundaries:
                veg_transitions = detect_vegetation_boundaries(
                    features, landcover, band_mode, self.thresholds
                )
                all_transitions.extend(veg_transitions)

                # Optional: Surf zone boundaries
                surf_transitions = detect_surf_zone_boundaries(
                    features, landcover, band_mode, self.thresholds
                )
                all_transitions.extend(surf_transitions)

            # Primary: Dry/wet boundaries (shell line)
            dw_transitions = detect_dry_wet_boundaries(
                features, landcover, band_mode, self.thresholds
            )
        else:
            # RGB-only mode - degraded detection
            logger.info("Using RGB-only detection mode (no NIR available)")
            
            # Optional: Surf zone boundaries (still works with RGB)
            if include_optional_boundaries:
                surf_transitions = detect_surf_zone_boundaries(
                    features, landcover, band_mode, self.thresholds
                )
                all_transitions.extend(surf_transitions)
            
            # Primary: Dry/wet boundaries using brightness (degraded)
            dw_transitions = detect_dry_wet_boundaries_rgb(
                features, landcover, self.thresholds
            )

        # Apply shell line selection (same for both modes)
        dw_transitions = self._select_best_shell_line_candidate(
            dw_transitions, features, band_mode
        )

        # Log selected transitions (debug level to avoid console spam)
        logger.debug(f"_select_best_shell_line_candidate returned {len(dw_transitions)} transitions")
        for i, t in enumerate(dw_transitions):
            logger.debug(f"  [{i}] dist={t['distance']:.1f}m, conf={t['confidence']:.2f}, "
                        f"guaranteed={t.get('guaranteed_shell_line', False)}, "
                        f"mode={t.get('detection_mode', 'unknown')}")

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

        logger.debug(f"Detected {len(validated_transitions)} zone boundaries "
                    f"(band_mode={band_mode})")
        
        if validated_transitions:
            method_counts = {}
            for t in validated_transitions:
                method = t.get('detection_method', 'unknown')
                method_counts[method] = method_counts.get(method, 0) + 1
            logger.debug(f"  Detection methods: {method_counts}")

        return validated_transitions

    def _detect_band_mode(self, features: pd.DataFrame) -> str:
        """Detect band mode from features DataFrame."""
        return detect_band_mode_from_dataframe(features)

    # ========================================================================
    # SHELL LINE SELECTION
    # ========================================================================

    def _select_best_shell_line_candidate(
        self,
        candidates: List[Dict],
        features: pd.DataFrame,
        band_mode: str = None
    ) -> List[Dict]:
        """
        Select the single best shell line candidate from dry/wet boundary detections.
        
        Uses a cascading strategy:
        1. Strict mode: High-confidence candidates in expected zone
        2. Relaxed mode: Re-run detection with lower thresholds
        3. Best-available: Strongest derivative regardless of context
        4. Last resort: Expected zone median
        
        Args:
            candidates: List of candidate transitions
            features: DataFrame with spectral features
            band_mode: Band configuration string
            
        Returns:
            List containing single best shell line (or empty if none found)
        """
        if band_mode is None:
            band_mode = self._detect_band_mode(features)
            
        config = self.thresholds.get('boundary_thresholds', {}).get('dry_wet', {})
        expected_min = config.get('expected_location_min', 40)
        expected_max = config.get('expected_location_max', 120)

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
            features, expected_min, expected_max, distance_min, distance_max, band_mode
        )
        if relaxed_result:
            return relaxed_result

        # MODE 3: Best-available mode
        best_available = self._find_any_derivative_minimum(
            features, distance_min, distance_max, band_mode
        )
        if best_available:
            logger.warning(f"Using best-available mode: shell line at {best_available['distance']:.1f}m "
                          f"(conf={best_available['confidence']:.2f}, band_mode={band_mode})")
            return [best_available]

        # MODE 4: Last resort
        fallback = self._create_fallback_boundary(features, expected_min, expected_max, band_mode)
        logger.warning(f"Using last-resort mode: shell line at {fallback['distance']:.1f}m")
        return [fallback]

    def _try_strict_selection(
        self,
        candidates: List[Dict],
        expected_min: float,
        expected_max: float,
        distance_min: float,
        distance_max: float
    ) -> List[Dict]:
        """Try strict mode selection - high confidence candidates in expected zone."""
        if candidates:
            candidates = [c for c in candidates
                          if distance_min <= c['distance'] <= distance_max]
            if not candidates:
                return []

        in_zone = [c for c in candidates
                   if expected_min <= c['distance'] <= expected_max] if candidates else []

        if not in_zone:
            return []

        for cand in in_zone:
            cand['detection_mode'] = 'strict'
            cand['guaranteed_shell_line'] = True

        in_zone.sort(key=lambda c: c['confidence'], reverse=True)
        best = in_zone[0]

        min_confidence = 0.60
        if best['confidence'] < min_confidence:
            return []

        logger.info(f"Selected shell line at {best['distance']:.1f}m "
                   f"(mode=strict, confidence={best['confidence']:.2f})")
        return [best]

    def _try_relaxed_selection(
        self,
        features: pd.DataFrame,
        expected_min: float,
        expected_max: float,
        distance_min: float,
        distance_max: float,
        band_mode: str = None
    ) -> List[Dict]:
        """Try relaxed mode selection - lower thresholds for edge cases."""
        logger.debug("Attempting relaxed mode detection...")
        relaxed_candidates = self._detect_with_relaxed_thresholds(features, band_mode)

        if not relaxed_candidates:
            return []

        relaxed_candidates = [c for c in relaxed_candidates
                             if distance_min <= c['distance'] <= distance_max]

        if not relaxed_candidates:
            return []

        for cand in relaxed_candidates:
            cand['detection_mode'] = 'fallback'
            cand['guaranteed_shell_line'] = True

        in_zone = [c for c in relaxed_candidates
                   if expected_min <= c['distance'] <= expected_max]

        if not in_zone:
            in_zone = relaxed_candidates

        in_zone.sort(key=lambda c: c['confidence'], reverse=True)
        best = in_zone[0]

        min_confidence = _get_min_acceptance_confidence(band_mode) if band_mode else 0.50
        if best['confidence'] < min_confidence:
            return []

        logger.info(f"Selected shell line at {best['distance']:.1f}m "
                   f"(mode=fallback, confidence={best['confidence']:.2f})")
        return [best]

    def _detect_with_relaxed_thresholds(
        self,
        features: pd.DataFrame,
        band_mode: str = None
    ) -> List[Dict]:
        """Re-run detection with relaxed thresholds for fallback mode."""
        if band_mode is None:
            band_mode = self._detect_band_mode(features)
            
        max_confidence = _get_max_confidence(band_mode)
            
        if band_mode == BAND_CONFIG_RGB:
            return detect_with_relaxed_thresholds_rgb(features, self.thresholds)
        
        # NIR-based relaxed detection
        d1_col = 'nir_d1_w5'
        if d1_col not in features.columns:
            d1_col = 'nir_d1_smooth'
        value_col = 'nir'
        threshold = -4.0  # Relaxed from -8.0
        min_value = 40

        if d1_col not in features.columns:
            return []
            
        derivative = features[d1_col]
        distance = features['distance']
        values = features.get(value_col, features.get('brightness'))

        transitions = []

        for i in range(len(derivative)):
            if pd.isna(derivative.iloc[i]):
                continue
            if not (derivative.iloc[i] < threshold and values.iloc[i] > min_value):
                continue

            is_local_min = True
            if i > 0 and not pd.isna(derivative.iloc[i-1]) and derivative.iloc[i] > derivative.iloc[i-1]:
                is_local_min = False
            if i < len(derivative) - 1 and not pd.isna(derivative.iloc[i+1]) and derivative.iloc[i] > derivative.iloc[i+1]:
                is_local_min = False

            if not is_local_min:
                continue

            magnitude = abs(derivative.iloc[i])
            confidence = 0.50 + (magnitude - 4.0) / 20.0
            confidence = max(0.50, min(confidence, 0.70))
            confidence = min(confidence, max_confidence)

            transitions.append({
                'index': i,
                'distance': distance.iloc[i],
                'type': 'dry_wet_derivative',
                'confidence': confidence,
                'magnitude': derivative.iloc[i],
                'detection_method': 'nir_relaxed_derivative',
                'band_mode': band_mode
            })

        logger.debug(f"Relaxed detection found {len(transitions)} candidates")

        return transitions

    def _find_any_derivative_minimum(
        self,
        features: pd.DataFrame,
        distance_min: float,
        distance_max: float,
        band_mode: str = None
    ) -> Optional[Dict]:
        """Find strongest derivative minimum regardless of context."""
        logger.debug("Attempting best-available mode...")

        if band_mode == BAND_CONFIG_RGB:
            return find_any_derivative_minimum_rgb(
                features, distance_min, distance_max, self.thresholds
            )

        # NIR-based best-available
        d1_col = 'nir_d1_w5'
        if d1_col not in features.columns:
            d1_col = 'nir_d1_smooth'
        value_col = 'nir'
        threshold = -3.0
        min_value = 30

        if d1_col not in features.columns:
            return None
            
        derivative = features[d1_col]
        distance = features['distance']
        values = features.get(value_col, features.get('brightness'))

        candidates = []

        for i in range(len(derivative)):
            if pd.isna(derivative.iloc[i]):
                continue
            if not (derivative.iloc[i] < threshold and values.iloc[i] > min_value):
                continue

            is_local_min = True
            if i > 0 and not pd.isna(derivative.iloc[i-1]) and derivative.iloc[i] > derivative.iloc[i-1]:
                is_local_min = False
            if i < len(derivative) - 1 and not pd.isna(derivative.iloc[i+1]) and derivative.iloc[i] > derivative.iloc[i+1]:
                is_local_min = False

            if not is_local_min:
                continue

            dist = distance.iloc[i]

            if dist < distance_min or dist > distance_max:
                continue

            candidates.append({
                'index': i,
                'distance': dist,
                'type': 'dry_wet_derivative',
                'confidence': 0.40,
                'magnitude': derivative.iloc[i],
                'detection_mode': 'best_available',
                'detection_method': 'nir_derivative_any',
                'guaranteed_shell_line': True,
                'band_mode': band_mode or BAND_CONFIG_4BAND
            })

        if not candidates:
            return None

        best = max(candidates, key=lambda c: abs(c['magnitude']))
        return best

    def _create_fallback_boundary(
        self,
        features: pd.DataFrame,
        expected_min: float,
        expected_max: float,
        band_mode: str = None
    ) -> Dict:
        """Create last-resort boundary at expected zone median."""
        target_distance = (expected_min + expected_max) / 2.0
        logger.debug(f"Creating last-resort boundary at target {target_distance:.1f}m")

        distance = features['distance']
        
        if band_mode == BAND_CONFIG_RGB:
            values = features.get('brightness', features.get('brightness_rgb'))
        else:
            values = features.get('nir', features.get('brightness'))
        
        min_val = 30

        valid_indices = []
        for i, (dist, val) in enumerate(zip(distance, values)):
            if pd.notna(val) and val > min_val:
                valid_indices.append(i)

        if not valid_indices:
            i = len(distance) // 2
        else:
            best_i = min(valid_indices, key=lambda i: abs(distance.iloc[i] - target_distance))
            i = best_i

        conf = 0.25 if band_mode == BAND_CONFIG_RGB else 0.30

        return {
            'index': i,
            'distance': float(distance.iloc[i]),
            'type': 'dry_wet_derivative',
            'confidence': conf,
            'magnitude': 0.0,
            'detection_mode': 'last_resort',
            'detection_method': 'expected_zone_median',
            'guaranteed_shell_line': True,
            'band_mode': band_mode or BAND_CONFIG_4BAND
        }

    # ========================================================================
    # FILTERING AND MERGING
    # ========================================================================

    def _merge_nearby_detections(
        self,
        transitions: List[Dict],
        min_separation: float = 5.0
    ) -> List[Dict]:
        """
        Merge detections within min_separation meters, keeping highest confidence.
        
        Multiple detection methods may identify the same boundary. This merges
        duplicates within the separation threshold, preferring guaranteed
        shell lines and higher confidence detections.
        """
        if not transitions:
            return []

        sorted_transitions = sorted(transitions, key=lambda x: x['distance'])

        merged = []
        i = 0

        while i < len(sorted_transitions):
            current = sorted_transitions[i]
            current_method = current.get('detection_method', 'unknown')
            candidates = [current]

            j = i + 1
            while j < len(sorted_transitions):
                next_trans = sorted_transitions[j]
                distance_diff = next_trans['distance'] - current['distance']

                if distance_diff < min_separation:
                    next_method = next_trans.get('detection_method', 'unknown')
                    if next_method == current_method:
                        candidates.append(next_trans)
                    j += 1
                else:
                    break

            if len(candidates) > 1:
                # Prefer guaranteed shell lines
                guaranteed_candidates = [c for c in candidates if c.get('guaranteed_shell_line', False)]
                if guaranteed_candidates:
                    best = max(guaranteed_candidates, key=lambda x: x['confidence'])
                else:
                    best = max(candidates, key=lambda x: x['confidence'])
            else:
                best = candidates[0]

            merged.append(best)
            i += len(candidates)

        logger.debug(f"Merged {len(transitions)} detections into {len(merged)} final boundaries")

        return merged

    def _add_class_transition_info(
        self,
        transitions: List[Dict],
        landcover: pd.DataFrame
    ) -> List[Dict]:
        """Add class transition information to detected boundaries."""
        for transition in transitions:
            idx = transition['index']

            if idx > 0:
                transition['from_class'] = landcover.iloc[idx - 1].get('predicted_class', 'UNKNOWN')
            else:
                transition['from_class'] = 'UNKNOWN'

            if idx < len(landcover):
                transition['to_class'] = landcover.iloc[idx].get('predicted_class', 'UNKNOWN')
            else:
                transition['to_class'] = 'UNKNOWN'

        return transitions

    def _filter_transitions(
        self,
        transitions: List[Dict],
        features: pd.DataFrame,
        landcover: pd.DataFrame
    ) -> List[Dict]:
        """
        Filter transitions by confidence, edge effects, and zone consistency.
        
        Removes:
        - Transitions below confidence threshold
        - Transitions within edge buffer of transect start/end
        - Same-class transitions (false positives)
        """
        if not transitions:
            return []

        min_dist = features['distance'].min()
        max_dist = features['distance'].max()
        edge_buffer = EDGE_BUFFER_M

        min_confidence = self.thresholds.get('min_boundary_confidence', 0.75)

        filtered = []

        logger.debug(f"_filter_transitions: Received {len(transitions)} transitions")
        for i, t in enumerate(transitions):
            logger.debug(f"  [{i}] dist={t['distance']:.1f}m, conf={t['confidence']:.2f}, "
                        f"method={t.get('detection_method', '?')}, "
                        f"guaranteed={t.get('guaranteed_shell_line', False)}")

        for transition in transitions:
            is_guaranteed = transition.get('guaranteed_shell_line', False)

            # Always preserve guaranteed shell lines
            if is_guaranteed:
                logger.debug(f"Preserving guaranteed shell line at {transition['distance']:.1f}m")
                idx = transition['index']
                if 0 <= idx < len(landcover):
                    landcover.loc[idx, 'transition_flag'] = True
                filtered.append(transition)
                continue

            # Get band-mode-specific minimum confidence
            band_mode = transition.get('band_mode', BAND_CONFIG_4BAND)
            detection_mode = transition.get('detection_mode', 'strict')
            
            if band_mode == BAND_CONFIG_RGB:
                effective_min_confidence = _get_min_acceptance_confidence(BAND_CONFIG_RGB)
            elif band_mode == BAND_CONFIG_CIR:
                effective_min_confidence = _get_min_acceptance_confidence(BAND_CONFIG_CIR)
            else:
                if detection_mode == 'fallback':
                    effective_min_confidence = 0.50
                else:
                    effective_min_confidence = min_confidence

            # Filter by confidence
            if transition['confidence'] < effective_min_confidence:
                continue

            # Filter by edge buffer
            if (transition['distance'] < min_dist + edge_buffer or
                transition['distance'] > max_dist - edge_buffer):
                continue

            # Filter same-class transitions
            from_class = transition.get('from_class', 'UNKNOWN')
            to_class = transition.get('to_class', 'UNKNOWN')

            if from_class == to_class and from_class != 'UNKNOWN':
                continue

            if from_class == 'UNKNOWN' and to_class == 'UNKNOWN':
                continue

            # Mark transition flag
            idx = transition['index']
            if 0 <= idx < len(landcover):
                landcover.loc[idx, 'transition_flag'] = True

            filtered.append(transition)

        logger.debug(f"Filtered: {len(transitions)} -> {len(filtered)} transitions")

        return filtered

    # ========================================================================
    # STATIC UTILITY METHODS
    # ========================================================================

    @staticmethod
    def filter_by_boundary_type(
        transitions: List[Dict],
        boundary_types: str = 'shore_only'
    ) -> List[Dict]:
        """
        Filter transitions by type.
        
        Args:
            transitions: List of transition dictionaries
            boundary_types: 'shore_only', 'waterline', or 'all'
            
        Returns:
            Filtered list of transitions
        """
        if boundary_types == 'all':
            return transitions
            
        filtered = []
        for t in transitions:
            t_type = t.get('type', '')
            
            if boundary_types == 'shore_only':
                if 'dry_wet' in t_type:
                    filtered.append(t)
            elif boundary_types == 'waterline':
                if 'dry_wet' in t_type or 'surf' in t_type:
                    filtered.append(t)

        return filtered

    @staticmethod
    def is_shore_boundary(transition: Dict) -> bool:
        """
        Check if a transition represents a shore boundary (shell line).

        Shore boundaries are BEACH_DRY->BEACH_WET transitions detected
        via the dry_wet_derivative method.

        Args:
            transition: Transition dictionary with 'type' key

        Returns:
            True if this is a shore boundary transition
        """
        t_type = transition.get('type', '')
        return 'dry_wet' in t_type