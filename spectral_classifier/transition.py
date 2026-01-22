"""
Transition zone detection module for identifying beach-water interfaces.

Supports 3-band (RGB or CIR) and 4-band (RGBN) imagery with automatic
detection method selection based on band availability.
"""

import logging
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from .config import THRESHOLDS, EDGE_BUFFER_M
from .data_io import (
    BAND_CONFIG_4BAND, BAND_CONFIG_CIR, BAND_CONFIG_RGB,
    detect_band_mode_from_dataframe
)

logger = logging.getLogger(__name__)


def _get_max_confidence(band_mode: str) -> float:
    """Get maximum confidence cap for a given band mode from config."""
    if band_mode == BAND_CONFIG_RGB:
        return THRESHOLDS.get('rgb_thresholds', {}).get('max_confidence', 0.70)
    elif band_mode == BAND_CONFIG_CIR:
        return THRESHOLDS.get('cir_thresholds', {}).get('max_confidence', 0.90)
    else:  # 4-band
        return 0.95  # Full confidence for 4-band


def _get_min_acceptance_confidence(band_mode: str) -> float:
    """Get minimum acceptance confidence for a given band mode from config."""
    if band_mode == BAND_CONFIG_RGB:
        return THRESHOLDS.get('rgb_thresholds', {}).get('min_acceptance_confidence', 0.45)
    elif band_mode == BAND_CONFIG_CIR:
        return THRESHOLDS.get('cir_thresholds', {}).get('min_acceptance_confidence', 0.55)
    else:  # 4-band
        return 0.60


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

        Automatically selects detection method based on band availability:
        - 4band/CIR: NIR-based detection (full capability, CIR slightly reduced confidence)
        - RGB: Brightness-based detection (degraded mode, lower confidence)

        Args:
            features: DataFrame with spectral features
            landcover: Optional DataFrame with classified landcover

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
            
            # Method 1: VEG_DUNES boundaries (inflection detection)
            veg_transitions = self._detect_vegetation_boundaries(features, landcover, band_mode)
            all_transitions.extend(veg_transitions)

            # Method 2: Surf zone boundaries (RGB foam detection)
            surf_transitions = self._detect_surf_zone_boundaries(features, landcover, band_mode)
            all_transitions.extend(surf_transitions)

            # Method 3: Dry/wet boundaries (NIR derivative magnitude)
            dw_transitions = self._detect_dry_wet_boundaries(features, landcover, band_mode)
        else:
            # RGB-only mode - degraded detection
            logger.info("Using RGB-only detection mode (no NIR available)")
            
            # Method 1: Surf zone boundaries (RGB foam detection) - still works
            surf_transitions = self._detect_surf_zone_boundaries(features, landcover, band_mode)
            all_transitions.extend(surf_transitions)
            
            # Method 3: Dry/wet boundaries using brightness (degraded)
            dw_transitions = self._detect_dry_wet_boundaries_rgb(features, landcover)

        # Apply shell line selection (same for both modes)
        dw_transitions = self._select_best_shell_line_candidate(
            dw_transitions, features, band_mode
        )

        # Log selected transitions
        logger.info(f"_select_best_shell_line_candidate returned {len(dw_transitions)} transitions")
        for i, t in enumerate(dw_transitions):
            logger.info(f"  [{i}] dist={t['distance']:.1f}m, conf={t['confidence']:.2f}, "
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

        logger.info(f"Detected {len(validated_transitions)} zone boundaries "
                   f"(band_mode={band_mode})")
        
        if validated_transitions:
            method_counts = {}
            for t in validated_transitions:
                method = t.get('detection_method', 'unknown')
                method_counts[method] = method_counts.get(method, 0) + 1
            logger.info(f"  Detection methods: {method_counts}")

        return validated_transitions

    def _detect_band_mode(self, features: pd.DataFrame) -> str:
        """Detect band mode from features DataFrame."""
        return detect_band_mode_from_dataframe(features)

    # ========================================================================
    # RGB-ONLY DETECTION METHODS
    # ========================================================================

    def _detect_dry_wet_boundaries_rgb(
        self,
        features: pd.DataFrame,
        landcover: pd.DataFrame
    ) -> List[Dict]:
        """
        Detect BEACH_DRY->BEACH_WET boundaries using brightness derivatives (RGB-only mode).

        This is a degraded detection method when NIR is not available.
        Uses RGB brightness drop as proxy for the wet/dry boundary.
        
        Key differences from NIR-based detection:
        - Uses brightness_rgb_d1_w5 instead of nir_d1_w5
        - Lower thresholds (brightness changes are smaller)
        - Lower maximum confidence (capped at config value)
        - Still uses R/G ratio validation

        Args:
            features: DataFrame with computed features
            landcover: DataFrame with predicted classes

        Returns:
            List of transition dictionaries
        """
        # Use RGB brightness derivative
        brightness_d1_col = 'brightness_rgb_d1_smooth'
        if brightness_d1_col not in features.columns:
            brightness_d1_col = 'brightness_d1_smooth'
        
        if brightness_d1_col not in features.columns:
            logger.warning("No brightness derivative available for RGB detection")
            return []
        
        brightness_d1 = features[brightness_d1_col]
        distance = features['distance']
        brightness = features.get('brightness_rgb', features.get('brightness'))
        
        transitions = []

        # RGB-specific thresholds (more lenient than NIR)
        rgb_config = self.thresholds.get('rgb_thresholds', {})
        threshold = rgb_config.get('brightness_derivative_threshold', -5.0)
        min_brightness_drop_abs = rgb_config.get('min_brightness_drop_absolute', 25.0)
        brightness_before_min = rgb_config.get('brightness_before_min', 150)
        var_ratio_min = rgb_config.get('variability_ratio_min', 1.3)
        max_confidence = _get_max_confidence(BAND_CONFIG_RGB)
        
        # Use same expected location as NIR mode
        config = self.thresholds.get('boundary_thresholds', {}).get('dry_wet', {})
        expected_loc_min = config.get('expected_location_min', 40)
        expected_loc_max = config.get('expected_location_max', 120)
        
        min_brightness = 50  # Avoid noise in deep water

        logger.debug(f"RGB dry/wet detection: threshold={threshold}, "
                    f"brightness_drop_min={min_brightness_drop_abs}, "
                    f"expected_zone={expected_loc_min}-{expected_loc_max}m")

        # Find candidate transition points
        for i in range(len(brightness_d1)):
            # Check if this is a significant negative slope
            if not (brightness_d1.iloc[i] < threshold and brightness.iloc[i] > min_brightness):
                continue

            # Check if local minimum in derivative (peak negative slope)
            is_local_min = True
            if i > 0 and brightness_d1.iloc[i] > brightness_d1.iloc[i-1]:
                is_local_min = False
            if i < len(brightness_d1) - 1 and brightness_d1.iloc[i] > brightness_d1.iloc[i+1]:
                is_local_min = False

            if not is_local_min:
                continue

            # Context validation
            brightness_drop_abs = 0.0
            if i >= 5:
                brightness_before = brightness.iloc[i-5:i].mean()
                brightness_at = brightness.iloc[i]
                brightness_drop_abs = brightness_before - brightness_at

            brightness_mean_before = 0.0
            if i >= 5:
                brightness_mean_before = brightness.iloc[i-5:i].mean()

            var_ratio = 0.0
            if i >= 5 and i < len(features) - 5 and 'variability' in features.columns:
                var_before = features['variability'].iloc[i-5:i].mean()
                var_after = features['variability'].iloc[i:i+5].mean()
                var_ratio = var_after / (var_before + 1e-6)

            magnitude = abs(brightness_d1.iloc[i])

            # RGB-specific confidence scoring (lower baseline)
            base_confidence = 0.40 + (magnitude - 5.0) / 25.0
            base_confidence = max(0.40, min(base_confidence, 0.55))

            confidence = base_confidence

            # Bonus: Strong derivative
            if magnitude > 8.0:
                confidence += 0.08

            # Bonus: Large brightness drop
            if brightness_drop_abs > 30:
                confidence += 0.06

            # Bonus: High variability ratio
            if var_ratio > 3.0:
                confidence += 0.05

            # Bonus: Multi-band consensus (RGB only)
            consensus_reached, num_bands, band_mags = self._check_multi_band_consensus_rgb(
                features, i, threshold
            )
            if num_bands >= 2:
                confidence += 0.06

            # Bonus: R/G ratio validation (still available in RGB mode)
            use_rg_ratio = config.get('use_rg_ratio', True)
            if use_rg_ratio:
                is_valid, conf_adjustment = self._validate_rg_pattern_for_boundary_type(
                    features, i, 'dry_wet'
                )
                if not is_valid:
                    logger.debug(f"  RGB rejected at {distance.iloc[i]:.1f}m: R/G pattern mismatch")
                    continue
                confidence += conf_adjustment

            # Bonus: Expected location
            dist = distance.iloc[i]
            in_expected_zone = expected_loc_min <= dist <= expected_loc_max
            if in_expected_zone:
                confidence += 0.05

            # Penalties for weak context
            if brightness_drop_abs < min_brightness_drop_abs:
                penalty = (min_brightness_drop_abs - brightness_drop_abs) / min_brightness_drop_abs
                confidence -= 0.12 * penalty

            if brightness_mean_before > 0 and brightness_mean_before < brightness_before_min:
                penalty = (brightness_before_min - brightness_mean_before) / brightness_before_min
                confidence -= 0.08 * penalty

            if var_ratio > 0 and var_ratio < var_ratio_min:
                penalty = (var_ratio_min - var_ratio) / var_ratio_min
                confidence -= 0.06 * penalty

            # Cap confidence for RGB mode
            confidence = min(confidence, max_confidence)
            confidence = max(confidence, 0.25)

            logger.debug(f"  RGB candidate at {distance.iloc[i]:.1f}m: "
                        f"conf={confidence:.2f}, brightness_drop={brightness_drop_abs:.1f}, "
                        f"deriv={magnitude:.2f}, var_ratio={var_ratio:.2f}")

            min_acceptance = _get_min_acceptance_confidence(BAND_CONFIG_RGB)
            if confidence >= min_acceptance:
                transitions.append({
                    'index': i,
                    'distance': distance.iloc[i],
                    'type': 'dry_wet_derivative',
                    'confidence': confidence,
                    'magnitude': brightness_d1.iloc[i],
                    'brightness_value': brightness.iloc[i],
                    'brightness_drop_abs': brightness_drop_abs,
                    'brightness_before': brightness_mean_before,
                    'variability_ratio': var_ratio,
                    'in_expected_zone': in_expected_zone,
                    'num_bands_agreeing': num_bands,
                    'band_magnitudes': band_mags,
                    'detection_method': 'rgb_brightness_derivative',
                    'band_mode': BAND_CONFIG_RGB
                })

        logger.debug(f"Found {len(transitions)} RGB dry/wet boundary candidates")

        return transitions

    def _check_multi_band_consensus_rgb(
        self,
        features: pd.DataFrame,
        index: int,
        threshold: float
    ) -> tuple:
        """
        Check if multiple RGB bands show derivative drops (consensus) - RGB-only mode.

        Args:
            features: DataFrame with derivative features
            index: Index to check
            threshold: Brightness derivative threshold

        Returns:
            Tuple of (consensus_reached: bool, num_bands_agreeing: int, band_magnitudes: dict)
        """
        scaling = {
            'red': 1.0,
            'green': 0.95,
            'blue': 0.90
        }

        bands_agreeing = 0
        band_magnitudes = {}

        # Check Red band
        if 'red_d1_smooth' in features.columns:
            red_threshold = threshold * scaling['red']
            red_val = features['red_d1_smooth'].iloc[index]
            band_magnitudes['red'] = red_val
            if red_val < red_threshold:
                bands_agreeing += 1

        # Check Green band
        if 'green_d1_smooth' in features.columns:
            green_threshold = threshold * scaling['green']
            green_val = features['green_d1_smooth'].iloc[index]
            band_magnitudes['green'] = green_val
            if green_val < green_threshold:
                bands_agreeing += 1

        # Check Blue band
        if 'blue_d1_smooth' in features.columns:
            blue_threshold = threshold * scaling['blue']
            blue_val = features['blue_d1_smooth'].iloc[index]
            band_magnitudes['blue'] = blue_val
            if blue_val < blue_threshold:
                bands_agreeing += 1

        consensus_reached = bands_agreeing >= 2

        return consensus_reached, bands_agreeing, band_magnitudes

    # ========================================================================
    # STANDARD NIR-BASED DETECTION METHODS
    # ========================================================================

    def _check_sustainability(
        self,
        derivative: pd.Series,
        index: int,
        threshold: float,
        min_consecutive: int = None
    ) -> bool:
        """
        Check if derivative drop is sustained over multiple consecutive points.
        """
        if min_consecutive is None:
            min_consecutive = self.thresholds.get('derivative_sustainability_points', 3)

        if index + min_consecutive > len(derivative):
            return False

        consecutive_count = 0
        for i in range(index, min(index + min_consecutive + 2, len(derivative))):
            if derivative.iloc[i] < threshold:
                consecutive_count += 1
                if consecutive_count >= min_consecutive:
                    return True
            else:
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
        """
        if nir_threshold is None:
            nir_threshold = self.thresholds.get('nir_derivative_threshold', -4.38)

        if min_bands is None:
            min_bands = self.thresholds.get('multi_band_consensus_required', 2)

        scaling = self.thresholds.get('derivative_scaling', {
            'red': 0.85,
            'green': 0.80,
            'blue': 0.75
        })

        bands_agreeing = 0
        band_magnitudes = {}

        # NIR is always included (primary signal)
        if 'nir_d1_smooth' in features.columns:
            nir_val = features['nir_d1_smooth'].iloc[index]
            if pd.notna(nir_val):
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
            if pd.notna(blue_val):
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
        """
        if 'red_green_ratio' not in features.columns:
            return True, 0.0

        rg_ratio = features['red_green_ratio'].iloc[index]

        if boundary_type == 'dry_wet':
            expected_min, expected_max = 0.95, 1.15
            if expected_min <= rg_ratio <= expected_max:
                return True, 0.05
            elif 0.90 <= rg_ratio <= 1.20:
                return True, 0.0
            else:
                return False, 0.0

        elif boundary_type == 'veg':
            return True, 0.0

        elif boundary_type == 'surf':
            if 'rg_ratio_d1_smooth' in features.columns:
                rg_deriv = features['rg_ratio_d1_smooth'].iloc[index]
                if rg_deriv < -0.01:
                    return True, 0.03
            return True, 0.0

        return True, 0.0

    def _detect_vegetation_boundaries(
        self,
        features: pd.DataFrame,
        landcover: pd.DataFrame,
        band_mode: str = BAND_CONFIG_4BAND
    ) -> List[Dict]:
        """
        Detect VEG_DUNES->BEACH_DRY boundaries using inflection point detection.
        """
        nir_d2 = features.get('nir_d2_w7', features.get('curvature'))
        if nir_d2 is None or nir_d2.isna().all():
            logger.debug("No second derivative available for vegetation boundary detection")
            return []
            
        distance = features['distance']
        nir = features.get('nir', features.get('brightness'))

        transitions = []
        max_confidence = _get_max_confidence(band_mode)

        veg_config = self.thresholds.get('boundary_thresholds', {}).get('veg_boundaries', {})
        threshold = veg_config.get('second_deriv_threshold', 1.0)
        trend_window = veg_config.get('trend_change_window', 10)
        min_nir_change = veg_config.get('min_nir_change', 20)

        for i in range(1, len(nir_d2) - 1):
            if nir_d2.iloc[i-1] * nir_d2.iloc[i+1] < 0:
                curvature_change = abs(nir_d2.iloc[i-1] - nir_d2.iloc[i+1])

                if curvature_change > threshold:
                    window_start = max(0, i - trend_window)
                    window_end = min(len(nir), i + trend_window)

                    if window_end - window_start < trend_window:
                        continue

                    nir_before = nir.iloc[window_start:i].mean()
                    nir_after = nir.iloc[i:window_end].mean()

                    if nir_before > nir_after + min_nir_change:
                        confidence = min(0.65 + (curvature_change / 10.0), 0.85)
                        
                        # Apply band-mode confidence cap
                        confidence = min(confidence, max_confidence)

                        is_valid, conf_adjustment = self._validate_rg_pattern_for_boundary_type(
                            features, i, 'veg'
                        )
                        confidence += conf_adjustment
                        confidence = min(confidence, max_confidence)

                        transitions.append({
                            'index': i,
                            'distance': distance.iloc[i],
                            'type': 'veg_inflection',
                            'confidence': confidence,
                            'curvature_change': curvature_change,
                            'nir_change': nir_before - nir_after,
                            'detection_method': 'inflection_point',
                            'band_mode': band_mode
                        })

        logger.debug(f"Found {len(transitions)} vegetation boundary candidates")

        return transitions

    def _detect_surf_zone_boundaries(
        self,
        features: pd.DataFrame,
        landcover: pd.DataFrame,
        band_mode: str = BAND_CONFIG_4BAND
    ) -> List[Dict]:
        """
        Detect BEACH_WET->WATER boundaries using RGB foam peak detection.
        """
        if 'has_rgb_foam_peak' not in features.columns:
            return []

        foam_peaks = features['has_rgb_foam_peak']
        distance = features['distance']
        
        # Get NIR if available for confirmation
        nir_d1 = features.get('nir_d1_w7')
        has_nir = nir_d1 is not None and not nir_d1.isna().all()

        transitions = []
        max_confidence = _get_max_confidence(band_mode)

        surf_config = self.thresholds.get('boundary_thresholds', {}).get('surf_zone', {})
        require_nir_drop = surf_config.get('require_nir_drop', True)
        nir_threshold = surf_config.get('nir_threshold', -2.0)

        for i in range(len(foam_peaks)):
            if foam_peaks.iloc[i]:
                # Confirm with NIR drop if available
                if has_nir and require_nir_drop:
                    if nir_d1.iloc[i] >= nir_threshold:
                        continue

                confidence = 0.65 if has_nir else 0.55
                
                # Apply band-mode confidence cap
                confidence = min(confidence, max_confidence)

                is_valid, conf_adjustment = self._validate_rg_pattern_for_boundary_type(
                    features, i, 'surf'
                )
                confidence += conf_adjustment
                confidence = min(confidence, max_confidence)

                transitions.append({
                    'index': i,
                    'distance': distance.iloc[i],
                    'type': 'surf_foam',
                    'confidence': confidence,
                    'has_nir_confirmation': has_nir,
                    'detection_method': 'rgb_foam_peak',
                    'band_mode': band_mode
                })

        logger.debug(f"Found {len(transitions)} surf zone boundary candidates")

        return transitions

    def _detect_dry_wet_boundaries(
        self,
        features: pd.DataFrame,
        landcover: pd.DataFrame,
        band_mode: str = BAND_CONFIG_4BAND
    ) -> List[Dict]:
        """
        Detect BEACH_DRY->BEACH_WET boundaries using NIR derivative magnitude.
        
        Works for both 4-band and CIR modes (both have NIR).
        """
        nir_d1 = features.get('nir_d1_w5', features.get('nir_d1_smooth'))
        if nir_d1 is None or nir_d1.isna().all():
            logger.warning("No NIR derivative available, cannot run NIR-based detection")
            return []
            
        distance = features['distance']
        nir = features['nir']

        transitions = []
        max_confidence = _get_max_confidence(band_mode)
        min_acceptance = _get_min_acceptance_confidence(band_mode)

        config = self.thresholds.get('boundary_thresholds', {}).get('dry_wet', {})
        threshold = config.get('nir_threshold', -8.0)
        min_nir_drop_abs = config.get('min_nir_drop_absolute', 39.0)
        brightness_min = config.get('brightness_before_min', 175)
        nir_before_min = config.get('nir_before_min', 135)
        var_ratio_min = config.get('variability_ratio_min', 1.5)
        expected_loc_min = config.get('expected_location_min', 40)
        expected_loc_max = config.get('expected_location_max', 120)
        require_sustainability = config.get('require_sustainability', True)
        require_consensus = config.get('require_consensus', True)
        min_nir = 50

        logger.debug(f"Dry/wet detection config: NIR_drop_min={min_nir_drop_abs}, "
                    f"brightness_min={brightness_min}, NIR_before_min={nir_before_min}, "
                    f"expected_zone={expected_loc_min}-{expected_loc_max}m, "
                    f"band_mode={band_mode}, max_conf={max_confidence}")

        for i in range(len(nir_d1)):
            if not (nir_d1.iloc[i] < threshold and nir.iloc[i] > min_nir):
                continue

            is_local_min = True
            if i > 0 and nir_d1.iloc[i] > nir_d1.iloc[i-1]:
                is_local_min = False
            if i < len(nir_d1) - 1 and nir_d1.iloc[i] > nir_d1.iloc[i+1]:
                is_local_min = False

            if not is_local_min:
                continue

            # Context validation
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
                var_ratio = var_after / (var_before + 1e-6)

            magnitude = abs(nir_d1.iloc[i])

            # Composite confidence scoring
            base_confidence = 0.50 + (magnitude - 8.0) / 30.0
            base_confidence = max(0.50, min(base_confidence, 0.70))

            confidence = base_confidence

            if magnitude > 12.0:
                confidence += 0.10

            if nir_drop_abs > 50:
                confidence += 0.08

            if i >= 5 and 'brightness' in features.columns:
                brightness_at = features['brightness'].iloc[i]
                brightness_drop = brightness_before - brightness_at
                if brightness_drop > 25:
                    confidence += 0.05

            if var_ratio > 4.0:
                confidence += 0.07

            is_sustained = False
            if require_sustainability:
                is_sustained = self._check_sustainability(nir_d1, i, threshold)
                if is_sustained:
                    confidence += 0.08

            consensus_reached = False
            num_bands = 1
            band_mags = {}
            if require_consensus:
                consensus_reached, num_bands, band_mags = self._check_multi_band_consensus(
                    features, i, threshold
                )
                if num_bands >= 3:
                    confidence += 0.08

            use_rg_ratio = config.get('use_rg_ratio', True)
            if use_rg_ratio:
                is_valid, conf_adjustment = self._validate_rg_pattern_for_boundary_type(
                    features, i, 'dry_wet'
                )
                if not is_valid:
                    logger.debug(f"  Rejected at {distance.iloc[i]:.1f}m: R/G pattern mismatch")
                    continue
                confidence += conf_adjustment

            dist = distance.iloc[i]
            in_expected_zone = expected_loc_min <= dist <= expected_loc_max
            if in_expected_zone:
                confidence += 0.05

            # VEG->DRY discrimination
            veg_zone_min = config.get('expected_location_veg_dry_min', 40)
            veg_zone_max = config.get('expected_location_veg_dry_max', 70)

            if veg_zone_min <= dist <= veg_zone_max:
                if var_ratio < 1.0:
                    logger.debug(f"  REJECTED at {dist:.1f}m: VEG zone with low var_ratio ({var_ratio:.2f})")
                    continue
                if nir_drop_abs < 50:
                    confidence -= 0.20

            # Penalties
            if nir_drop_abs < min_nir_drop_abs:
                penalty = (min_nir_drop_abs - nir_drop_abs) / min_nir_drop_abs
                confidence -= 0.15 * penalty

            if brightness_before > 0 and brightness_before < brightness_min:
                penalty = (brightness_min - brightness_before) / brightness_min
                confidence -= 0.10 * penalty

            if nir_mean_before > 0 and nir_mean_before < nir_before_min:
                penalty = (nir_before_min - nir_mean_before) / nir_before_min
                confidence -= 0.10 * penalty

            if var_ratio > 0 and var_ratio < var_ratio_min:
                penalty = (var_ratio_min - var_ratio) / var_ratio_min
                confidence -= 0.08 * penalty

            # Apply band-mode confidence cap
            confidence = min(confidence, max_confidence)
            confidence = max(confidence, 0.30)

            logger.debug(f"  Candidate at {distance.iloc[i]:.1f}m: "
                        f"conf={confidence:.2f}, NIR_drop={nir_drop_abs:.1f}, "
                        f"deriv={magnitude:.2f}, var_ratio={var_ratio:.2f}")

            if confidence >= min_acceptance:
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
                    'detection_method': 'derivative_magnitude',
                    'band_mode': band_mode
                })

        logger.debug(f"Found {len(transitions)} dry/wet boundary candidates (NIR, mode={band_mode})")

        return transitions

    # ========================================================================
    # FILTERING AND SELECTION METHODS
    # ========================================================================

    def _merge_nearby_detections(
        self,
        transitions: List[Dict],
        min_separation: float = 5.0
    ) -> List[Dict]:
        """Merge detections that are within min_separation meters of each other."""
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
        """Filter transitions by confidence, edge effects, and zone consistency."""
        if not transitions:
            return []

        min_dist = features['distance'].min()
        max_dist = features['distance'].max()
        edge_buffer = EDGE_BUFFER_M

        min_confidence = self.thresholds.get('min_boundary_confidence', 0.75)

        filtered = []

        logger.info(f"_filter_transitions: Received {len(transitions)} transitions")
        for i, t in enumerate(transitions):
            logger.info(f"  [{i}] dist={t['distance']:.1f}m, conf={t['confidence']:.2f}, "
                       f"method={t.get('detection_method', '?')}, "
                       f"guaranteed={t.get('guaranteed_shell_line', False)}")

        for transition in transitions:
            is_guaranteed = transition.get('guaranteed_shell_line', False)

            if is_guaranteed:
                logger.info(f"Preserving guaranteed shell line at {transition['distance']:.1f}m")
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

            if transition['confidence'] < effective_min_confidence:
                continue

            if (transition['distance'] < min_dist + edge_buffer or
                transition['distance'] > max_dist - edge_buffer):
                continue

            from_class = transition.get('from_class', 'UNKNOWN')
            to_class = transition.get('to_class', 'UNKNOWN')

            if from_class == to_class and from_class != 'UNKNOWN':
                continue

            if from_class == 'UNKNOWN' and to_class == 'UNKNOWN':
                continue

            idx = transition['index']
            if 0 <= idx < len(landcover):
                landcover.loc[idx, 'transition_flag'] = True

            filtered.append(transition)

        logger.debug(f"Filtered: {len(transitions)} -> {len(filtered)} transitions")

        return filtered

    # ========================================================================
    # SHELL LINE SELECTION (handles both NIR and RGB modes)
    # ========================================================================

    def _select_best_shell_line_candidate(
        self,
        candidates: List[Dict],
        features: pd.DataFrame,
        band_mode: str = None
    ) -> List[Dict]:
        """
        Select the single best shell line candidate from dry/wet boundary detections.
        
        Handles both NIR and RGB modes with appropriate fallback strategies.
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
        """Try strict mode selection."""
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
        """Try relaxed mode selection."""
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

    def _find_any_derivative_minimum(
        self,
        features: pd.DataFrame,
        distance_min: float,
        distance_max: float,
        band_mode: str = None
    ) -> Dict:
        """Find strongest derivative minimum regardless of context."""
        logger.debug("Attempting best-available mode...")

        # Choose derivative based on band mode
        if band_mode == BAND_CONFIG_RGB:
            d1_col = 'brightness_rgb_d1_smooth'
            if d1_col not in features.columns:
                d1_col = 'brightness_d1_smooth'
            value_col = 'brightness'
            threshold = -3.0
            min_value = 30
        else:
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

            magnitude = abs(derivative.iloc[i])
            
            # Lower confidence for RGB mode
            base_conf = 0.35 if band_mode == BAND_CONFIG_RGB else 0.40

            candidates.append({
                'index': i,
                'distance': dist,
                'type': 'dry_wet_derivative',
                'confidence': base_conf,
                'magnitude': derivative.iloc[i],
                'detection_mode': 'best_available',
                'detection_method': 'derivative_magnitude_any',
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
        
        # Use appropriate value column
        if band_mode == BAND_CONFIG_RGB:
            values = features.get('brightness', features.get('brightness_rgb'))
            min_val = 30
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

        # Lower confidence for RGB mode
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
            # Use RGB brightness detection with relaxed thresholds
            d1_col = 'brightness_rgb_d1_smooth'
            if d1_col not in features.columns:
                d1_col = 'brightness_d1_smooth'
            value_col = 'brightness'
            threshold = -3.0  # Relaxed from -5.0
            min_value = 40
        else:
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
            
            # Confidence calculation
            if band_mode == BAND_CONFIG_RGB:
                confidence = 0.40 + (magnitude - 3.0) / 20.0
                confidence = max(0.40, min(confidence, 0.60))
            else:
                confidence = 0.50 + (magnitude - 4.0) / 20.0
                confidence = max(0.50, min(confidence, 0.70))
            
            # Apply band-mode confidence cap
            confidence = min(confidence, max_confidence)

            transitions.append({
                'index': i,
                'distance': distance.iloc[i],
                'type': 'dry_wet_derivative',
                'confidence': confidence,
                'magnitude': derivative.iloc[i],
                'detection_method': 'relaxed_derivative',
                'band_mode': band_mode
            })

        logger.debug(f"Relaxed detection found {len(transitions)} candidates")

        return transitions

    # ========================================================================
    # STATIC FILTERING METHOD
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