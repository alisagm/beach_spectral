"""
Utility modules for spectral transect classification system.

Submodules:
    - data_io: Raster and GeoJSON loading with CRS handling
    - export: CSV/JSON export and statistics
    - logging_config: Logging setup
    - batch: Batch processing utilities
    - footprint_clip: Imagery footprint computation
"""

from .logging_config import setup_logging, get_logger
from .export import (
    export_results_to_csv,
    export_summary_json,
    calculate_processing_stats,
    print_processing_summary,
    select_representative_transects,
    validate_output_directory
)
from .batch import (
    extract_year_from_path,
    extract_capture_date_from_path,
    group_rasters_by_year,
    resolve_year_band_config,
    classify_raster
)
from .data_io import (
    build_raster_index,
    load_transects,
    reproject_if_needed,
    find_overlapping_rasters,
    RasterIndex,
    BAND_CONFIG_4BAND,
    BAND_CONFIG_CIR,
    BAND_CONFIG_RGB,
    detect_band_configuration,
    detect_band_mode_from_dataframe
)
from .helpers import detect_transect_direction

__all__ = [
    # Logging
    'setup_logging',
    'get_logger',
    
    # Export
    'export_results_to_csv',
    'export_summary_json',
    'calculate_processing_stats',
    'print_processing_summary',
    'select_representative_transects',
    'validate_output_directory',
    
    # Batch
    'extract_year_from_path',
    'extract_capture_date_from_path',
    'group_rasters_by_year',
    'resolve_year_band_config',
    'classify_raster',
    
    # Data I/O
    'build_raster_index',
    'load_transects',
    'reproject_if_needed',
    'find_overlapping_rasters',
    'RasterIndex',
    'BAND_CONFIG_4BAND',
    'BAND_CONFIG_CIR',
    'BAND_CONFIG_RGB',
    'detect_band_configuration',
    'detect_band_mode_from_dataframe',
]