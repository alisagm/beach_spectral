"""
Spectral analysis module for transect data extraction and feature computation.

This module handles:
- Spectral value extraction along transects (sampler)
- Feature computation (indices, derivatives, statistics)

Submodules:
    - sampler: Extract spectral values at points along transects
    - features: Compute spectral features for classification/detection

The SpectralFeatures class contains methods that are REUSABLE for other
spectral edge detection applications (e.g., forest boundary detection).
See features.py docstrings for details on adaptable methods.
"""

from .sampler import (
    sample_transect,
    validate_spectral_data,
    interpolate_points_along_line,
    sample_raster_at_point
)

from .features import SpectralFeatures

__all__ = [
    'sample_transect',
    'validate_spectral_data',
    'interpolate_points_along_line',
    'sample_raster_at_point',
    'SpectralFeatures',
]