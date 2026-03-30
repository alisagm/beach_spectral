"""
Spectral derivative computations.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

def compute_band_derivative(values: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """First derivative of spectral band."""
    nan_mask = np.isnan(values)

    if nan_mask.all():
        return np.full_like(values, np.nan, dtype=float)

    # Fill NaN positions with nearest valid neighbour so that
    # uniform_filter1d doesn't bleed NaN through the kernel window.
    filled = values.copy().astype(float)
    if nan_mask.any():
        idx = np.arange(len(filled))
        valid_idx = idx[~nan_mask]
        filled = np.interp(idx, valid_idx, filled[valid_idx])
    
    result = np.gradient(filled, distance)
    result[nan_mask] = np.nan
    return result

def compute_derivative_smooth(
    values: np.ndarray,
    distance: np.ndarray,
    smooth_size: int = 3,
) -> np.ndarray:
    """Smoothed first derivative of spectral band (NaN-safe)."""
    nan_mask = np.isnan(values)

    if nan_mask.all():
        return np.full_like(values, np.nan, dtype=float)

    # Fill NaN positions with nearest valid neighbour so that
    # uniform_filter1d doesn't bleed NaN through the kernel window.
    filled = values.copy().astype(float)
    if nan_mask.any():
        idx = np.arange(len(filled))
        valid_idx = idx[~nan_mask]
        filled = np.interp(idx, valid_idx, filled[valid_idx])

    smooth = uniform_filter1d(filled, size=smooth_size, mode="nearest")
    result = np.gradient(smooth, distance)

    # Re-apply the original NaN mask so callers know where data was absent.
    result[nan_mask] = np.nan
    return result

def compute_derivative_multiscale(values: np.ndarray, distance: np.ndarray, 
                                  name: str, scales = (5,7,9,11)) -> dict[str, np.ndarray]:
    """Derivative of spectral band at multiple smoothing scales.
        Args:
        name: Band or signal name used as dict key prefix, e.g. 'nir' or
              'brightness_rgb'. Output keys will be '{name}_d1_w{window}'.
    """
    derivatives = {}
    nan_mask = np.isnan(values)

    if nan_mask.all():
        return {f'{name}_d1_w{w}': np.full_like(values, np.nan, dtype=float)
            for w in scales}

    # Fill NaN positions with nearest valid neighbour so that
    # uniform_filter1d doesn't bleed NaN through the kernel window.
    filled = values.copy().astype(float)
    if nan_mask.any():
        idx = np.arange(len(filled))
        valid_idx = idx[~nan_mask]
        filled = np.interp(idx, valid_idx, filled[valid_idx])

    for window in scales:
        smooth = uniform_filter1d(filled, size=window, mode='nearest')
        arr = np.gradient(smooth, distance)
        arr[nan_mask] = np.nan
        derivatives[f'{name}_d1_w{window}'] = arr  
    return derivatives

def compute_second_derivative(values: np.ndarray, distance: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Second derivative of spectral band (curvature/inflection detection)."""
    nan_mask = np.isnan(values)

    if nan_mask.all():
        return np.full_like(values, np.nan, dtype=float)

    # Fill NaN positions with nearest valid neighbour so that
    # uniform_filter1d doesn't bleed NaN through the kernel window.
    filled = values.copy().astype(float)
    if nan_mask.any():
        idx = np.arange(len(filled))
        valid_idx = idx[~nan_mask]
        filled = np.interp(idx, valid_idx, filled[valid_idx])
       
    smooth = uniform_filter1d(filled, size=window_size, mode='nearest')
    smooth_d1 = np.gradient(smooth, distance)
    smooth_d2 = np.gradient(smooth_d1, distance)
    smooth_d2[nan_mask] = np.nan

    return smooth_d2