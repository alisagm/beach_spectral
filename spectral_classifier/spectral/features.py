"""
Feature extraction module for spectral transect analysis.

Supports 3-band (RGB or CIR) and 4-band (RGBN) imagery with automatic
detection and conditional feature computation.
"""

import logging
from typing import Dict
import numpy as np
import pandas as pd
from spectral_classifier.config import THRESHOLDS

from .spectral_indices import (
    compute_ndvi, compute_ndwi, compute_nir_ratio,
    compute_brightness, compute_blue_red_ratio, compute_red_green_ratio,
    compute_grvi,
)
from .derivatives import (
    compute_band_derivative, compute_derivative_smooth,
    compute_derivative_multiscale, compute_second_derivative,
)
from .window_features import (
    compute_variability, compute_slope, compute_curvature, compute_local_range,
)
from .detection_features import (
    compute_spectral_angle, detect_oscillations, detect_rgb_foam_peaks,
)

logger = logging.getLogger(__name__)


def compute_all(data: pd.DataFrame, band_indices: dict) -> pd.DataFrame:
    """
    Compute all spectral features for a single transect profile.

    Args:
        data:         DataFrame with columns [TransectID, distance, x, y, red, green, blue, nir].
                      blue column must be present but may be NaN for CIR imagery.
                      nir column must be present but may be NaN for RGB imagery.
        band_indices: Output of resolve_band_indices(year_config). Determines
                      which features are computed — no column scanning performed.

    Returns:
        Copy of input DataFrame with all feature columns appended.

    Brightness columns by band configuration:

        column           RGBN        RGB         CIR
        ───────────────────────────────────────────────
        brightness       R+G+B+NIR   R+G+B       R+G+NIR
        brightness_rgb   R+G+B       R+G+B       NaN
        brightness_cir   R+G+NIR     NaN         R+G+NIR

    interpret.py should always use the `brightness` column for boundary
    detection; brightness_rgb and brightness_cir are supplementary features.
    """
    result = data.copy()
    has_nir  = band_indices["nir"]  is not None
    has_blue = band_indices["blue"] is not None
    window_size = THRESHOLDS['window_size']

    # ----------------------------------------------------------------
    # Extract raw arrays once — all downstream functions receive arrays,
    # not DataFrame column references.
    # ----------------------------------------------------------------

    distance = result['distance'].values
    red      = result['red'].values
    green    = result['green'].values
    blue     = result['blue'].values  # present but NaN-filled for CIR
    nir      = result['nir'].values   # present but NaN-filled for RGB

    # ----------------------------------------------------------------
    # Spectral indices
    # ----------------------------------------------------------------

    # CHANGED: brightness_rgb only valid when blue is available.
    if has_blue:
        brightness_rgb = compute_brightness(np.stack([red, green, blue], axis=1))
    else:
        brightness_rgb = np.full_like(red, np.nan, dtype=float)

    # CHANGED: brightness_cir — R+G+NIR, valid for CIR and RGBN years.
    if has_nir:
        brightness_cir = compute_brightness(np.stack([red, green, nir], axis=1))
    else:
        brightness_cir = np.full_like(red, np.nan, dtype=float)

    # CHANGED: brightness uses the full available band set per config:
    #   RGBN -> R+G+B+NIR,  RGB -> brightness_rgb,  CIR -> brightness_cir
    if has_blue and has_nir:
        brightness = compute_brightness(np.stack([red, green, blue, nir], axis=1))
    elif has_blue:   # RGB
        brightness = brightness_rgb
    else:            # CIR
        brightness = brightness_cir

    result['brightness']     = brightness
    result['brightness_rgb'] = brightness_rgb
    result['brightness_cir'] = brightness_cir
    result['red_green_ratio'] = compute_red_green_ratio(red, green)
    result['grvi'] = compute_grvi(green, red)

    if has_nir:
        result['ndvi']      = compute_ndvi(nir, red)
        result['ndwi']      = compute_ndwi(green, nir)
        result['nir_ratio'] = compute_nir_ratio(nir, brightness_rgb)
    else:
        result['ndvi']      = np.nan
        result['ndwi']      = np.nan
        result['nir_ratio'] = np.nan
        logger.debug("NIR not available — ndvi, ndwi, nir_ratio set to NaN")

    if has_blue:
        result['blue_red_ratio'] = compute_blue_red_ratio(blue, red)
    else:
        result['blue_red_ratio'] = np.nan

    # ----------------------------------------------------------------
    # Derivatives
    # ----------------------------------------------------------------

    if has_nir:
        result['nir_d1']        = compute_band_derivative(nir, distance)
        result['nir_d1_smooth'] = compute_derivative_smooth(nir, distance)
    else:
        result['nir_d1']        = np.nan
        result['nir_d1_smooth'] = np.nan
        logger.debug("NIR not available — brightness derivatives are the primary signal")

    result['brightness_d1']        = compute_band_derivative(brightness, distance)
    result['brightness_d1_smooth'] = compute_derivative_smooth(brightness, distance)

    # CHANGED: brightness_rgb derivatives only when blue available.
    if has_blue:
        result['brightness_rgb_d1']        = compute_band_derivative(brightness_rgb, distance)
        result['brightness_rgb_d1_smooth'] = compute_derivative_smooth(brightness_rgb, distance)
    else:
        result['brightness_rgb_d1']        = np.nan
        result['brightness_rgb_d1_smooth'] = np.nan

    # CHANGED: brightness_cir derivatives only when NIR available.
    if has_nir:
        result['brightness_cir_d1']        = compute_band_derivative(brightness_cir, distance)
        result['brightness_cir_d1_smooth'] = compute_derivative_smooth(brightness_cir, distance)
    else:
        result['brightness_cir_d1']        = np.nan
        result['brightness_cir_d1_smooth'] = np.nan

    result['red_d1']          = compute_band_derivative(red, distance)
    result['red_d1_smooth']   = compute_derivative_smooth(red, distance)
    result['green_d1']        = compute_band_derivative(green, distance)
    result['green_d1_smooth'] = compute_derivative_smooth(green, distance)

    if has_blue:
        result['blue_d1']        = compute_band_derivative(blue, distance)
        result['blue_d1_smooth'] = compute_derivative_smooth(blue, distance)
    else:
        result['blue_d1']        = np.nan
        result['blue_d1_smooth'] = np.nan

    # R/G ratio derivative — reuse already-computed column rather than
    # calling compute_red_green_ratio a second time.
    rg_ratio = result['red_green_ratio'].values
    result['rg_ratio_d1']        = compute_band_derivative(rg_ratio, distance)
    result['rg_ratio_d1_smooth'] = compute_derivative_smooth(rg_ratio, distance)

    # Multiscale derivatives — NIR if available, brightness fallback.
    # CIR years have NIR so take the NIR path correctly; RGB falls back to brightness.
    if has_nir:
        for key, arr in compute_derivative_multiscale(nir, distance, 'nir').items():
            result[key] = arr
    else:
        for key, arr in compute_derivative_multiscale(
                brightness, distance, 'brightness_rgb').items():
            result[key] = arr
        for w in [5, 7, 9, 11]:
            result[f'nir_d1_w{w}'] = np.nan

    # Second derivatives — same primary/fallback logic.
    primary = nir if has_nir else brightness
    for w in [5, 7, 9]:
        result[f'nir_d2_w{w}'] = compute_second_derivative(primary, distance, window_size=w)

    # ----------------------------------------------------------------
    # Statistical window features — operate on brightness as pd.Series
    # so rolling() retains the DataFrame index.
    # brightness is always valid (RGBN, RGB, or CIR path above).
    # ----------------------------------------------------------------
    brightness_s = pd.Series(brightness, index=result.index)

    result['variability'] = compute_variability(brightness_s, window_size)
    result['slope']       = compute_slope(brightness_s, window_size)
    result['curvature']   = compute_curvature(brightness_s)
    result['local_range'] = compute_local_range(brightness_s, window_size)

    # ----------------------------------------------------------------
    # Detection features
    # ----------------------------------------------------------------

    # CHANGED: band_cols excludes blue for CIR, excludes nir for RGB.
    if has_blue and has_nir:
        band_cols = ['red', 'green', 'blue', 'nir']
    elif has_blue:
        band_cols = ['red', 'green', 'blue']
    else:  # CIR
        band_cols = ['red', 'green', 'nir']

    brightness_rgb_s = pd.Series(brightness_rgb, index=result.index)

    result['spectral_angle']   = compute_spectral_angle(result[band_cols])
    result['has_oscillations'] = detect_oscillations(brightness_s, window_size)

    # CHANGED: foam peak detection requires blue — NaN for CIR.
    if has_blue:
        result['has_rgb_foam_peak'] = detect_rgb_foam_peaks(brightness_rgb_s)
    else:
        result['has_rgb_foam_peak'] = np.nan

    n_features = len(result.columns) - len(data.columns)
    logger.info(f"Computed {n_features} features")

    return result