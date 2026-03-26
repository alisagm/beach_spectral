"""Statistical features using sliding-window pattern."""

import pandas as pd
import numpy as np
from scipy.stats import linregress

def compute_variability(brightness: pd.Series, window_size: int) -> pd.Series:
    """Standard deviation in sliding window."""
    return brightness.rolling(
        window=window_size,
        center=True,
        min_periods=1
    ).std()

def compute_slope(brightness: pd.Series, window_size: int) -> pd.Series:
    """Linear regression slope in sliding window."""
    slopes = []

    for i in range(len(brightness)):
        start = max(0, i-window_size//2)
        end = min(len(brightness), i + window_size//2+1)

        window_data = brightness.iloc[start:end]
        window_x = np.arange(len(window_data))

        if len(window_data) < 3:
            slopes.append(0.0)
        else:
            try:
                result = linregress(window_x, window_data)
                slopes.append(result.slope)
            except:
                slopes.append(0.0)
    
    return pd.Series(slopes, index=brightness.index)

def compute_curvature(brightness: pd.Series) -> pd.Series:
    """Second derivative approximation."""
    first_deriv = np.gradient(brightness)
    second_deriv = np.gradient(first_deriv)
    return pd.Series(second_deriv, index=brightness.index)

def compute_local_range(brightness: pd.Series, window_size: int) -> pd.Series:
    """Max - min in sliding window."""

    local_max = brightness.rolling(
        window=window_size,
        center=True,
        min_periods=1
    ).max()

    local_min = brightness.rolling(
        window=window_size,
        center=True,
        min_periods=1
    ).min()

    return local_max - local_min