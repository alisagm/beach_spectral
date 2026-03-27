"""
Spectral derivative computations.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

def compute_band_derivative(values: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """First derivative of spectral band."""
    d1 = np.gradient(values, distance)
    return d1

def compute_derivative_smooth(values: np.ndarray, distance: np.ndarray, smooth_size: int=3) -> np.ndarray:
    """Smoothed first derivative of spectral band."""
    smooth = uniform_filter1d(values, size=smooth_size, mode='nearest')
    return np.gradient(smooth, distance)

def compute_derivative_multiscale(values: np.ndarray, distance: np.ndarray, 
                                  name: str, scales = (5,7,9,11)) -> dict[str, np.ndarray]:
    """Derivative of spectral band at multiple smoothing scales.
        Args:
        name: Band or signal name used as dict key prefix, e.g. 'nir' or
              'brightness_rgb'. Output keys will be '{name}_d1_w{window}'.
    """
    derivatives = {}
    for window in scales:
        smooth = uniform_filter1d(values, size=window, mode='nearest')
        derivatives[f'{name}_d1_w{window}'] = np.gradient(smooth, distance)    
    return derivatives

def compute_second_derivative(values: np.ndarray, distance: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Second derivative of spectral band (curvature/inflection detection)."""
    smooth = uniform_filter1d(values, size=window_size, mode='nearest')
    smooth_d1 = np.gradient(smooth, distance)
    smooth_d2 = np.gradient(smooth_d1, distance)
    return smooth_d2