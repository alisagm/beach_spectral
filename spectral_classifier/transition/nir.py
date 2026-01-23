"""
NIR-based boundary detection methods.

This module contains detection methods that require NIR band data,
working with both 4-band (RGBN) and CIR [NIR,R,G] imagery.

Primary detection approach:
    NIR derivative magnitude - The NIR band shows strong contrast between
    dry sand (high reflectance) and wet sand/water (low reflectance due to
    absorption). Sharp negative derivatives indicate beach-water transitions.

Key methods:
    - detect_dry_wet_boundaries: BEACH_DRY -> BEACH_WET (shell line)
    - detect_vegetation_boundaries: VEG_DUNES -> BEACH_DRY
    - check_sustainability: Verify sustained derivative drops
    - check_multi_band_consensus: Verify multiple bands agree

These methods are adaptable to other edge detection applications where
NIR contrast is significant (e.g., forest edges, water body boundaries).
"""

import logging
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np

from ..config import THRESHOLDS
from ..utils.data_io import BAND_CONFIG_4BAND, BAND_CONFIG_CIR, BAND_CONFIG_RGB

logger = logging.getLogger(__name__)


def _get_max_confidence(band_mode: str) -> float:
    """Get maximum confidence cap for a given band mode."""
    if band_mode == BAND_CONFIG_RGB:
        return THRESHOLDS.get('rgb_thresholds', {}).get('max_confidence', 0.70)
    elif band_mode == BAND_CONFIG_CIR:
        return THRESHOLDS.get('cir_thresholds', {}).get('max_confidence', 0.90)
    else:  # 4-band
        return 0.95


def _get_min_acceptance_confidence(band_mode: str) -> float:
    """Get minimum acceptance confidence for a given band mode."""
    if band_mode == BAND_CONFIG_RGB:
        return THRESHOLDS.get('rgb_thresholds', {}).get('min_acceptance_confidence', 0.45)
    elif band_mode == BAND_CONFIG_CIR:
        return THRESHOLDS.get('cir_thresholds', {}).get('min_acceptance_confidence', 0.55)
    else:  # 4-band
        return 0.60


# ============================================================================
# MULTI-BAND VALIDATION
# ============================================================================

def check_sustainability(
    derivative: pd.Series,
    index: int,
    threshold: float,
    min_consecutive: int = None,
    thresholds: Dict = None
) -> bool:
    """
    Check if derivative drop is sustained over multiple consecutive points.
    
    A sustained drop indicates a true boundary rather than noise. This helps
    filter out spurious spikes in the derivative signal.
    
    Args:
        derivative: Series of derivative values
        index: Index to check
        threshold: Derivative threshold (negative for drops)
        min_consecutive: Minimum consecutive points below threshold
        thresholds: Optional thresholds dict (uses config default if None)
        
    Returns:
        True if drop is sustained for min_consecutive points
        
    REUSABLE: This pattern works for any derivative-based edge detection.
    """
    if thresholds is None:
        thresholds = THRESHOLDS
        
    if min_consecutive is None:
        min_consecutive = thresholds.get('derivative_sustainability_points', 3)

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


def check_multi_band_consensus(
    features: pd.DataFrame,
    index: int,
    nir_threshold: float = None,
    min_bands: int = None,
    thresholds: Dict = None
) -> Tuple[bool, int, Dict]:
    """
    Check if multiple spectral bands show derivative drops (consensus).
    
    Multi-band consensus increases confidence that a detected transition
    is real rather than an artifact of a single noisy band.
    
    Args:
        features: DataFrame with derivative features (*_d1_smooth columns)
        index: Index to check
        nir_threshold: NIR derivative threshold (uses config default if None)
        min_bands: Minimum bands required for consensus
        thresholds: Optional thresholds dict
        
    Returns:
        Tuple of (consensus_reached, num_bands_agreeing, band_magnitudes)
        
    REUSABLE: This pattern works for any multi-band edge detection.
    """
    if thresholds is None:
        thresholds = THRESHOLDS
        
    if nir_threshold is None:
        nir_threshold = thresholds.get('nir_derivative_threshold', -4.38)

    if min_bands is None:
        min_bands = thresholds.get('multi_band_consensus_required', 2)

    # Band-specific scaling factors (relative to NIR threshold)
    scaling = thresholds.get('derivative_scaling', {
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


def validate_rg_pattern_for_boundary_type(
    features: pd.DataFrame,
    index: int,
    boundary_type: str
) -> Tuple[bool, float]:
    """
    Validate R/G ratio pattern matches expected boundary type signature.
    
    Different boundary types have characteristic R/G ratio patterns:
    - dry_wet: R/G ≈ 1.0 (sand is spectrally neutral)
    - veg: Variable (depends on vegetation type)
    - surf: R/G derivative < 0 (foam has characteristic signature)
    
    Args:
        features: DataFrame with 'red_green_ratio' column
        index: Index to check
        boundary_type: 'dry_wet', 'veg', or 'surf'
        
    Returns:
        Tuple of (is_valid, confidence_adjustment)
    """
    if 'red_green_ratio' not in features.columns:
        return True, 0.0

    rg_ratio = features['red_green_ratio'].iloc[index]

    if boundary_type == 'dry_wet':
        # Shell line: expect R/G ≈ 1.0 (neutral sand)
        expected_min, expected_max = 0.95, 1.15
        if expected_min <= rg_ratio <= expected_max:
            return True, 0.05  # Bonus for matching pattern
        elif 0.90 <= rg_ratio <= 1.20:
            return True, 0.0   # Acceptable but no bonus
        else:
            return False, 0.0  # Reject

    elif boundary_type == 'veg':
        # Vegetation boundary: no strong R/G constraint
        return True, 0.0

    elif boundary_type == 'surf':
        # Surf zone: look for R/G derivative drop
        if 'rg_ratio_d1_smooth' in features.columns:
            rg_deriv = features['rg_ratio_d1_smooth'].iloc[index]
            if rg_deriv < -0.01:
                return True, 0.03
        return True, 0.0

    return True, 0.0


# ============================================================================
# VEGETATION BOUNDARY DETECTION (OPTIONAL)
# ============================================================================

def detect_vegetation_boundaries(
    features: pd.DataFrame,
    landcover: pd.DataFrame = None,
    band_mode: str = BAND_CONFIG_4BAND,
    thresholds: Dict = None
) -> List[Dict]:
    """
    Detect VEG_DUNES -> BEACH_DRY boundaries using inflection point detection.
    
    Vegetation boundaries are characterized by:
    - Sign change in second derivative (inflection point)
    - NIR decreasing from vegetation to bare sand
    - Higher curvature change magnitude
    
    This is an OPTIONAL boundary type - not needed for shell line detection.
    
    Args:
        features: DataFrame with spectral features
        landcover: Optional DataFrame with classified landcover
        band_mode: Band configuration string
        thresholds: Optional thresholds dict
        
    Returns:
        List of transition dictionaries
    """
    if thresholds is None:
        thresholds = THRESHOLDS
        
    nir_d2 = features.get('nir_d2_w7', features.get('curvature'))
    if nir_d2 is None or nir_d2.isna().all():
        logger.debug("No second derivative available for vegetation boundary detection")
        return []
        
    distance = features['distance']
    nir = features.get('nir', features.get('brightness'))

    transitions = []
    max_confidence = _get_max_confidence(band_mode)

    veg_config = thresholds.get('boundary_thresholds', {}).get('veg_boundaries', {})
    threshold = veg_config.get('second_deriv_threshold', 1.0)
    trend_window = veg_config.get('trend_change_window', 10)
    min_nir_change = veg_config.get('min_nir_change', 20)

    for i in range(1, len(nir_d2) - 1):
        # Check for sign change (inflection point)
        if nir_d2.iloc[i-1] * nir_d2.iloc[i+1] < 0:
            curvature_change = abs(nir_d2.iloc[i-1] - nir_d2.iloc[i+1])

            if curvature_change > threshold:
                window_start = max(0, i - trend_window)
                window_end = min(len(nir), i + trend_window)

                if window_end - window_start < trend_window:
                    continue

                nir_before = nir.iloc[window_start:i].mean()
                nir_after = nir.iloc[i:window_end].mean()

                # Vegetation has higher NIR than bare sand
                if nir_before > nir_after + min_nir_change:
                    confidence = min(0.65 + (curvature_change / 10.0), 0.85)
                    confidence = min(confidence, max_confidence)

                    is_valid, conf_adjustment = validate_rg_pattern_for_boundary_type(
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


# ============================================================================
# SURF ZONE BOUNDARY DETECTION (OPTIONAL)
# ============================================================================

def detect_surf_zone_boundaries(
    features: pd.DataFrame,
    landcover: pd.DataFrame = None,
    band_mode: str = BAND_CONFIG_4BAND,
    thresholds: Dict = None
) -> List[Dict]:
    """
    Detect BEACH_WET -> WATER boundaries using RGB foam peak detection.
    
    Surf zone boundaries are characterized by:
    - RGB brightness peaks (white foam from breaking waves)
    - NIR drop confirmation (water absorbs NIR)
    
    This is an OPTIONAL boundary type - not needed for shell line detection.
    
    Args:
        features: DataFrame with spectral features
        landcover: Optional DataFrame with classified landcover
        band_mode: Band configuration string
        thresholds: Optional thresholds dict
        
    Returns:
        List of transition dictionaries
    """
    if thresholds is None:
        thresholds = THRESHOLDS
        
    if 'has_rgb_foam_peak' not in features.columns:
        return []

    foam_peaks = features['has_rgb_foam_peak']
    distance = features['distance']
    
    # Get NIR if available for confirmation
    nir_d1 = features.get('nir_d1_w7')
    has_nir = nir_d1 is not None and not nir_d1.isna().all()

    transitions = []
    max_confidence = _get_max_confidence(band_mode)

    surf_config = thresholds.get('boundary_thresholds', {}).get('surf_zone', {})
    require_nir_drop = surf_config.get('require_nir_drop', True)
    nir_threshold = surf_config.get('nir_threshold', -2.0)

    for i in range(len(foam_peaks)):
        if foam_peaks.iloc[i]:
            # Confirm with NIR drop if available
            if has_nir and require_nir_drop:
                if nir_d1.iloc[i] >= nir_threshold:
                    continue

            confidence = 0.65 if has_nir else 0.55
            confidence = min(confidence, max_confidence)

            is_valid, conf_adjustment = validate_rg_pattern_for_boundary_type(
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


# ============================================================================
# DRY/WET BOUNDARY DETECTION (PRIMARY - SHELL LINE)
# ============================================================================

def detect_dry_wet_boundaries(
    features: pd.DataFrame,
    landcover: pd.DataFrame = None,
    band_mode: str = BAND_CONFIG_4BAND,
    thresholds: Dict = None
) -> List[Dict]:
    """
    Detect BEACH_DRY -> BEACH_WET boundaries using NIR derivative magnitude.
    
    This is the PRIMARY detection method for shell line identification.
    Works for both 4-band (RGBN) and CIR modes (both have NIR).
    
    Detection criteria:
    1. Strong negative NIR derivative (below threshold)
    2. Local minimum in derivative (peak negative slope)
    3. Context validation:
       - Sufficient NIR drop magnitude
       - High brightness before transition
       - Variability ratio increase
    4. R/G ratio pattern validation
    
    Args:
        features: DataFrame with spectral features including 'nir_d1_w5' or 'nir_d1_smooth'
        landcover: Optional DataFrame with classified landcover
        band_mode: Band configuration string
        thresholds: Optional thresholds dict
        
    Returns:
        List of transition dictionaries with:
            - index: Position in transect
            - distance: Distance along transect (meters)
            - type: 'dry_wet_derivative'
            - confidence: Detection confidence (0-1)
            - Various diagnostic values
            
    REUSABLE: This derivative-based approach adapts to other edge detection
    applications - replace NIR with any spectral band showing contrast at the
    boundary of interest.
    """
    if thresholds is None:
        thresholds = THRESHOLDS
        
    nir_d1 = features.get('nir_d1_w5', features.get('nir_d1_smooth'))
    if nir_d1 is None or nir_d1.isna().all():
        logger.warning("No NIR derivative available, cannot run NIR-based detection")
        return []
        
    distance = features['distance']
    nir = features['nir']

    transitions = []
    max_confidence = _get_max_confidence(band_mode)
    min_acceptance = _get_min_acceptance_confidence(band_mode)

    # Get configuration
    config = thresholds.get('boundary_thresholds', {}).get('dry_wet', {})
    threshold = config.get('nir_threshold', -8.0)
    min_nir_drop_abs = config.get('min_nir_drop_absolute', 39.0)
    brightness_min = config.get('brightness_before_min', 175)
    nir_before_min = config.get('nir_before_min', 135)
    var_ratio_min = config.get('variability_ratio_min', 1.5)
    expected_loc_min = config.get('expected_location_min', 40)
    expected_loc_max = config.get('expected_location_max', 120)
    require_sustainability = config.get('require_sustainability', True)
    require_consensus = config.get('require_consensus', True)
    min_nir = 50  # Avoid noise in deep water

    logger.debug(f"Dry/wet detection config: NIR_drop_min={min_nir_drop_abs}, "
                f"brightness_min={brightness_min}, NIR_before_min={nir_before_min}, "
                f"expected_zone={expected_loc_min}-{expected_loc_max}m, "
                f"band_mode={band_mode}, max_conf={max_confidence}")

    for i in range(len(nir_d1)):
        # Basic filter: significant negative derivative above water noise floor
        if not (nir_d1.iloc[i] < threshold and nir.iloc[i] > min_nir):
            continue

        # Check for local minimum in derivative
        is_local_min = True
        if i > 0 and nir_d1.iloc[i] > nir_d1.iloc[i-1]:
            is_local_min = False
        if i < len(nir_d1) - 1 and nir_d1.iloc[i] > nir_d1.iloc[i+1]:
            is_local_min = False

        if not is_local_min:
            continue

        # === Context Validation ===
        
        # NIR drop magnitude
        nir_drop_abs = 0.0
        if i >= 5:
            nir_before = nir.iloc[i-5:i].mean()
            nir_at = nir.iloc[i]
            nir_drop_abs = nir_before - nir_at

        # Brightness before transition
        brightness_before = 0.0
        if i >= 5 and 'brightness' in features.columns:
            brightness_before = features['brightness'].iloc[i-5:i].mean()

        # Mean NIR before transition
        nir_mean_before = 0.0
        if i >= 5:
            nir_mean_before = nir.iloc[i-5:i].mean()

        # Variability ratio (texture change)
        var_ratio = 0.0
        if i >= 5 and i < len(features) - 5 and 'variability' in features.columns:
            var_before = features['variability'].iloc[i-5:i].mean()
            var_after = features['variability'].iloc[i:i+5].mean()
            var_ratio = var_after / (var_before + 1e-6)

        magnitude = abs(nir_d1.iloc[i])

        # === Confidence Scoring ===
        
        # Base confidence from derivative magnitude
        base_confidence = 0.50 + (magnitude - 8.0) / 30.0
        base_confidence = max(0.50, min(base_confidence, 0.70))
        confidence = base_confidence

        # Bonus: Strong derivative
        if magnitude > 12.0:
            confidence += 0.10

        # Bonus: Large NIR drop
        if nir_drop_abs > 50:
            confidence += 0.08

        # Bonus: Brightness drop
        if i >= 5 and 'brightness' in features.columns:
            brightness_at = features['brightness'].iloc[i]
            brightness_drop = brightness_before - brightness_at
            if brightness_drop > 25:
                confidence += 0.05

        # Bonus: High variability ratio
        if var_ratio > 4.0:
            confidence += 0.07

        # Bonus: Sustained derivative drop
        is_sustained = False
        if require_sustainability:
            is_sustained = check_sustainability(nir_d1, i, threshold, thresholds=thresholds)
            if is_sustained:
                confidence += 0.08

        # Bonus: Multi-band consensus
        consensus_reached = False
        num_bands = 1
        band_mags = {}
        if require_consensus:
            consensus_reached, num_bands, band_mags = check_multi_band_consensus(
                features, i, threshold, thresholds=thresholds
            )
            if num_bands >= 3:
                confidence += 0.08

        # R/G ratio validation
        use_rg_ratio = config.get('use_rg_ratio', True)
        if use_rg_ratio:
            is_valid, conf_adjustment = validate_rg_pattern_for_boundary_type(
                features, i, 'dry_wet'
            )
            if not is_valid:
                logger.debug(f"  Rejected at {distance.iloc[i]:.1f}m: R/G pattern mismatch")
                continue
            confidence += conf_adjustment

        # Bonus: Expected location
        dist = distance.iloc[i]
        in_expected_zone = expected_loc_min <= dist <= expected_loc_max
        if in_expected_zone:
            confidence += 0.05

        # === VEG->DRY Discrimination ===
        # Penalize candidates in the vegetation zone
        veg_zone_min = config.get('expected_location_veg_dry_min', 40)
        veg_zone_max = config.get('expected_location_veg_dry_max', 70)

        if veg_zone_min <= dist <= veg_zone_max:
            if var_ratio < 1.0:
                logger.debug(f"  REJECTED at {dist:.1f}m: VEG zone with low var_ratio ({var_ratio:.2f})")
                continue
            if nir_drop_abs < 50:
                confidence -= 0.20

        # === Penalties for Weak Context ===
        
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
                    f"deriv={magnitude:.2f}, var_ratio={var_ratio:.2f}, "
                    f"sustained={is_sustained}, consensus={num_bands} bands")

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
                'nir_before': nir_mean_before,
                'variability_ratio': var_ratio,
                'is_sustained': is_sustained,
                'in_expected_zone': in_expected_zone,
                'num_bands_agreeing': num_bands,
                'band_magnitudes': band_mags,
                'detection_method': 'nir_derivative',
                'band_mode': band_mode
            })

    logger.debug(f"Found {len(transitions)} dry/wet boundary candidates")

    return transitions