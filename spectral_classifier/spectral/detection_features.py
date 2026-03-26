"""Categorical detection functions."""

import logging
import pandas as pd
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from spectral_classifier.config import EPSILON, THRESHOLDS

logger = logging.getLogger(__name__)

def compute_spectral_angle(values: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """
    Spectral angle between consecutive points.
    """
    angles = [0.0]  # First point has no predecessor

    for i in range(1, len(distance)):
        vec1 = values.iloc[i-1].astype(np.float64)
        vec2 = values.iloc[i].astype(np.float64)

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

    return angles

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

def detect_rgb_foam_peaks(bands: list[np.ndarray]) -> pd.Series:
    """
    Detect RGB peaks indicating foam from breaking surf.
    Returns boolean series indicating foam peak locations.
    """
    rgb_avg = bands.mean(axis=1)
    rgb_smooth = uniform_filter1d(rgb_avg.values, size=7, mode='nearest')

    prominence = THRESHOLDS.get('boundary_thresholds', {}).get(
         'surf_zone', {}
        ).get('rgb_peak_prominence', 10.0)

    peaks = signal.find_peaks(
        rgb_smooth,
        prominence=prominence,
        width=2
    )

    has_foam = pd.Series(False, index=rgb_avg.index)
    has_foam.iloc[peaks] = True

    logger.debug(f"Detected {len(peaks)} RGB foam peaks (prominence >= {prominence})")

    return has_foam