"""
Spectral analysis module for transect data extraction and feature computation.

This module handles:
- Spectral value extraction along transects (sampler)
- Feature computation (indices, derivatives, statistics)

Submodules:
    - sampler: Extract spectral values at points along transects
    - features: Compute spectral features for classification/detection
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