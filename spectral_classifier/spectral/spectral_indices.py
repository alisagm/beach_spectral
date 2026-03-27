"""
Spectral index computation on band arrays.
"""

import numpy as np
from spectral_classifier.config import EPSILON

# ========================================================================
# NIR-based indices (require NIR)
# ========================================================================

def compute_ndvi(nir, red, epsilon=EPSILON) -> np.ndarray:
    """Normalized Difference Vegetation Index."""
    ndvi = (nir - red) / (nir + red + epsilon)
    return ndvi

def compute_ndwi(green, nir, epsilon=EPSILON) -> np.ndarray:
    """Normalized Difference Water Index."""
    ndwi = (green - nir) / (green + nir + epsilon)
    return ndwi

def compute_nir_ratio(nir, brightness_rgb, epsilon=EPSILON) -> np.ndarray:
    """NIR prominence relative to visible bands."""
    nir_ratio = nir / (brightness_rgb + epsilon)
    return nir_ratio

# ========================================================================
# RGB-based indices (always available)
# ========================================================================

def compute_brightness(bands: np.ndarray) -> np.ndarray:
    """Overall reflectance (mean of all available bands)."""
    brightness = bands.mean(axis=1)
    return brightness

def compute_blue_red_ratio(blue, red, epsilon=EPSILON) -> np.ndarray:
    """Blue to red ratio (water indicator)."""
    br_ratio = blue / (red + epsilon)
    return br_ratio

def compute_red_green_ratio(red, green, epsilon=EPSILON) -> np.ndarray:
    """
    Red to green ratio - diagnostic for beach zone classification.

    Expected patterns:
    - VEGETATED_DUNES: Variable, generally R > G (ratio > 1.0)
    - BEACH_DRY: Generally R ~ G (ratio ~ 1.0)
    - BEACH_WET: Generally R > G (ratio > 1.0)
    - WATER: Generally R < G (ratio < 1.0), except wave crests

    Returns:
    Series with red/green ratio values
    """
    rg_ratio = red / (green + epsilon)
    return rg_ratio