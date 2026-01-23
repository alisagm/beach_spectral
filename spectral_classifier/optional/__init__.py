"""
Optional modules for spectral transect classification.

These modules are NOT required for shell line detection but provide
additional functionality for extended analysis.

Submodules:
    - landcover_classifier: Rule-based landcover classification
    - boundary_types: Detection of optional boundary types (veg, waterline)
"""

from .boundary_types import (
    detect_all_boundaries,
    get_boundary_description,
    BOUNDARY_TYPES
)

from .landcover_classifier import LandcoverClassifier

__all__ = [
    'detect_all_boundaries',
    'get_boundary_description',
    'BOUNDARY_TYPES',
]