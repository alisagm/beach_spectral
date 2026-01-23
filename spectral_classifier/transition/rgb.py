"""
RGB-only boundary detection methods (fallback/degraded mode).

This module contains detection methods for when NIR band is not available,
working with standard RGB [R,G,B] imagery only.

Key limitations:
    - Lower confidence than NIR-based detection
    - Uses brightness as proxy for NIR
    - Brightness changes are smaller and less distinctive
    - Higher false positive rate

When to use:
    - 3-band RGB imagery without NIR
    - When CIR detection fails or is unavailable

Detection approach:
    Brightness derivative magnitude - Similar to NIR detection but using
    RGB mean brightness as the signal. Less reliable because brightness
    differences between wet and dry sand are smaller than NIR differences.
"""

import logging
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np

from ..config import THRESHOLDS
from ..utils.data_io import BAND_CONFIG_RGB
from .nir import validate_rg_pattern_for_boundary_type

logger = logging.getLogger(__name__)


def _get_max_confidence_rgb() -> float:
    """Get maximum confidence cap for RGB mode."""
    return THRESHOLDS.get('rgb_thresholds', {}).get('max_confidence', 0.70)


def _get_min_acceptance_confidence_rgb() -> float:
    """Get minimum acceptance confidence for RGB mode."""
    return THRESHOLDS.get('rgb_thresholds', {}).get('min_acceptance_confidence', 0.45)


# ============================================================================
# MULTI-BAND CONSENSUS (RGB-ONLY)
# ============================================================================

def check_multi_band_consensus_rgb(
    features: pd.DataFrame,
    index: int,
    threshold: float,
    thresholds: Dict = None
) -> Tuple[bool, int, Dict]:
    """
    Check if multiple RGB bands show derivative drops (consensus) - RGB-only mode.
    
    Similar to NIR consensus but without NIR band. Uses band-specific
    scaling factors to account for different sensitivities.

    Args:
        features: DataFrame with derivative features
        index: Index to check
        threshold: Brightness derivative threshold
        thresholds: Optional thresholds dict

    Returns:
        Tuple of (consensus_reached, num_bands_agreeing, band_magnitudes)
    """
    if thresholds is None:
        thresholds = THRESHOLDS
        
    # RGB-specific scaling (relative to brightness threshold)
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


# ============================================================================
# DRY/WET BOUNDARY DETECTION (RGB FALLBACK)
# ============================================================================

def detect_dry_wet_boundaries_rgb(
    features: pd.DataFrame,
    landcover: pd.DataFrame = None,
    thresholds: Dict = None
) -> List[Dict]:
    """
    Detect BEACH_DRY -> BEACH_WET boundaries using brightness derivatives (RGB-only mode).

    This is a DEGRADED detection method when NIR is not available.
    Uses RGB brightness drop as proxy for the wet/dry boundary.
    
    Key differences from NIR-based detection:
    - Uses brightness_rgb_d1_w5 instead of nir_d1_w5
    - Lower thresholds (brightness changes are smaller)
    - Lower maximum confidence (capped at config value)
    - Still uses R/G ratio validation

    Args:
        features: DataFrame with computed features
        landcover: Optional DataFrame with predicted classes
        thresholds: Optional thresholds dict

    Returns:
        List of transition dictionaries
    """
    if thresholds is None:
        thresholds = THRESHOLDS
        
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
    rgb_config = thresholds.get('rgb_thresholds', {})
    threshold = rgb_config.get('brightness_derivative_threshold', -5.0)
    min_brightness_drop_abs = rgb_config.get('min_brightness_drop_absolute', 25.0)
    brightness_before_min = rgb_config.get('brightness_before_min', 150)
    var_ratio_min = rgb_config.get('variability_ratio_min', 1.3)
    max_confidence = _get_max_confidence_rgb()
    min_acceptance = _get_min_acceptance_confidence_rgb()
    
    # Use same expected location as NIR mode
    config = thresholds.get('boundary_thresholds', {}).get('dry_wet', {})
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

        # === Context Validation ===
        
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

        # === Confidence Scoring (lower baseline for RGB) ===
        
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
        consensus_reached, num_bands, band_mags = check_multi_band_consensus_rgb(
            features, i, threshold, thresholds
        )
        if num_bands >= 2:
            confidence += 0.06

        # R/G ratio validation (still available in RGB mode)
        use_rg_ratio = config.get('use_rg_ratio', True)
        if use_rg_ratio:
            is_valid, conf_adjustment = validate_rg_pattern_for_boundary_type(
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

        # === Penalties for Weak Context ===
        
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


# ============================================================================
# RELAXED DETECTION (FALLBACK)
# ============================================================================

def detect_with_relaxed_thresholds_rgb(
    features: pd.DataFrame,
    thresholds: Dict = None
) -> List[Dict]:
    """
    Re-run RGB detection with relaxed thresholds for fallback mode.
    
    Used when standard detection finds no candidates. Lowers the derivative
    threshold to capture weaker transitions that might still be valid.

    Args:
        features: DataFrame with computed features
        thresholds: Optional thresholds dict

    Returns:
        List of transition dictionaries
    """
    if thresholds is None:
        thresholds = THRESHOLDS
        
    max_confidence = _get_max_confidence_rgb()
        
    # Use RGB brightness detection with relaxed thresholds
    d1_col = 'brightness_rgb_d1_smooth'
    if d1_col not in features.columns:
        d1_col = 'brightness_d1_smooth'
    value_col = 'brightness'
    threshold = -3.0  # Relaxed from -5.0
    min_value = 40

    if d1_col not in features.columns:
        return []
        
    derivative = features[d1_col]
    distance = features['distance']
    values = features.get(value_col, features.get('brightness_rgb'))

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
        
        # Lower confidence for relaxed mode
        confidence = 0.40 + (magnitude - 3.0) / 20.0
        confidence = max(0.40, min(confidence, 0.60))
        confidence = min(confidence, max_confidence)

        transitions.append({
            'index': i,
            'distance': distance.iloc[i],
            'type': 'dry_wet_derivative',
            'confidence': confidence,
            'magnitude': derivative.iloc[i],
            'detection_method': 'rgb_relaxed_derivative',
            'band_mode': BAND_CONFIG_RGB
        })

    logger.debug(f"RGB relaxed detection found {len(transitions)} candidates")

    return transitions


def find_any_derivative_minimum_rgb(
    features: pd.DataFrame,
    distance_min: float,
    distance_max: float,
    thresholds: Dict = None
) -> Dict:
    """
    Find strongest brightness derivative minimum regardless of context (RGB mode).
    
    Last-resort fallback when no candidates pass validation. Finds the
    strongest derivative minimum within the expected zone.

    Args:
        features: DataFrame with computed features
        distance_min: Minimum distance to search
        distance_max: Maximum distance to search
        thresholds: Optional thresholds dict

    Returns:
        Transition dictionary or None if no minimum found
    """
    if thresholds is None:
        thresholds = THRESHOLDS
        
    d1_col = 'brightness_rgb_d1_smooth'
    if d1_col not in features.columns:
        d1_col = 'brightness_d1_smooth'
    value_col = 'brightness'
    threshold = -3.0
    min_value = 30

    if d1_col not in features.columns:
        return None
        
    derivative = features[d1_col]
    distance = features['distance']
    values = features.get(value_col, features.get('brightness_rgb'))

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
            'confidence': 0.35,  # Low confidence for best-available
            'magnitude': derivative.iloc[i],
            'detection_mode': 'best_available',
            'detection_method': 'rgb_derivative_any',
            'guaranteed_shell_line': True,
            'band_mode': BAND_CONFIG_RGB
        })

    if not candidates:
        return None

    best = max(candidates, key=lambda c: abs(c['magnitude']))
    return best