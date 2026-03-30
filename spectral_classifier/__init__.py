"""
Spectral Transect Classification System

A Python-based system to automatically classify spectral profiles from cross-shore
transects and identify transition zones (beach-water interfaces).

The system analyzes multi-band spectral data (Red, Green, Blue, NIR) along transects
extracted from GeoTIFF rasters and GeoJSON geometries.

Main modules:
    - data_io: Raster and GeoJSON loading with CRS handling
    - sampler: Spectral value extraction along transects
    - features: Spectral feature extraction (indices, statistics)
    - classifier: Rule-based landcover classification
    - transition: Transition zone detection
    - visualization: Plotting and visualization
    - utils: Helper functions and utilities
    - main: Pipeline orchestration

Example usage:
    from spectral_classifier.main import analyze_all_transects

    results = analyze_all_transects(
        raster_dir='path/to/rasters',
        transect_geojson='path/to/transects.geojson',
        output_dir='path/to/output',
        num_visualize=5
    )
"""

__version__ = '1.0.0'
__author__ = 'Spectral Analysis Team'

# Import main components for convenience
from .utils import (
    build_raster_index,
    load_transects,
    reproject_if_needed,
    find_overlapping_rasters  
)
from .spectral import sample_transect
from .config import THRESHOLDS, LANDCOVER_CLASSES, LANDCOVER_COLORS

__all__ = [
    # Data I/O
    'build_raster_index',
    'load_transects',
    'reproject_if_needed',
    'find_overlapping_rasters',

    # Processing
    'sample_transect',

    # Configuration
    'THRESHOLDS',
    'LANDCOVER_CLASSES',
    'LANDCOVER_COLORS',
]
