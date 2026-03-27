"""
Spectral analysis module for transect data extraction and feature computation.

This module handles:
- Spectral value extraction along transects (sampler)
- Feature computation (indices, derivatives, statistics)

Submodules:
    - sampler: Extract spectral values at points along transects
    - features: Computation wrapper of spectral features for classification/detection

    Internal modules (not exported):
        - spectral_indices: Spectral index functions
        - derivatives: Band derivatives with multiscale smoothing
        - window_features: Statistical features using sliding windows
        - detection_features: Categorical detection of spectral angle, oscillation, foam peaks
"""

from .sampler import (
    sample_transect,
    interpolate_points_along_line,
    sample_raster_at_point
)

from .features import compute_all

__all__ = [
    'sample_transect',
    'interpolate_points_along_line',
    'sample_raster_at_point',
    'compute_all',
]