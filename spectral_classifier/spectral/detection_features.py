"""Categorical detection functions."""

import logging
import pandas as pd
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from spectral_classifier.config import EPSILON, THRESHOLDS

logger = logging.getLogger(__name__)

def compute_spectral_angle(band_df: pd.DataFrame) -> np.ndarray:
    """
    Spectral angle between consecutive points.
    band_df: DataFrame with one column per band (e.g. ['red','green','blue'] or
            ['red','green','blue','nir']). Shape (n_samples, n_bands).
    """
    angles = [0.0]

    for i in range(1, len(band_df)):
        vec1 = band_df.iloc[i-1].values.astype(np.float64)
        vec2 = band_df.iloc[i].values.astype(np.float64)

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 > 0 and norm2 > 0:
            cos_angle = dot_product / (norm1 * norm2 + EPSILON)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
        else:
            angle = 0.0

        angles.append(angle)

    return pd.Series(angles, index=band_df.index)

def detect_oscillations(brightness: pd.Series, window_size: int) -> pd.Series:
    """
    Detect presence of oscillations using peak detection.
    Returns boolean series indicating oscillatory regions.
    """
    peaks, _ = signal.find_peaks(
        brightness,
        distance=2,
        prominence=5
        )

    has_osc = np.zeros(len(brightness), dtype=bool)

    for peak_idx in peaks:
        start = max(0, peak_idx - window_size // 2)
        end = min(len(brightness), peak_idx + window_size // 2)
        has_osc[start:end] = True

    return pd.Series(has_osc, index=brightness.index)

def detect_rgb_foam_peaks(brightness_rgb: pd.Series) -> pd.Series:
    """
    Detect RGB peaks indicating foam from breaking surf.
    Returns boolean series indicating foam peak locations.
    rgb_avg: pre-computed mean of R, G, B bands as a Series.
    """
    rgb_smooth = uniform_filter1d(brightness_rgb.values, size=7, mode='nearest')

    prominence = THRESHOLDS.get('boundary_thresholds', {}).get(
         'surf_zone', {}
        ).get('rgb_peak_prominence', 10.0)

    peaks, _ = signal.find_peaks(
        rgb_smooth,
        prominence=prominence,
        width=2
    )

    has_foam = pd.Series(False, index=brightness_rgb.index)
    has_foam.iloc[peaks] = True

    logger.debug(f"Detected {len(peaks)} RGB foam peaks (prominence >= {prominence})")

    return has_foam