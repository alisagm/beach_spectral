"""
Utilities package for spectral transect classification system.

Re-exports commonly used utilities from submodules for convenient importing.
"""

# Data I/O
from .data_io import (
    build_raster_index,
    load_transects,
    reproject_if_needed,
    find_overlapping_rasters,
    RasterIndex,
    BAND_CONFIG_4BAND,
    BAND_CONFIG_CIR,
    BAND_CONFIG_RGB,
    detect_band_mode_from_dataframe,
    validate_raster_bands
)

# Export utilities
from .export import (
    export_results_to_csv,
    export_summary_json,
    select_representative_transects,
    validate_output_directory,
    calculate_processing_stats,
    print_processing_summary,
    export_shell_line_geojson,
    bundle_shell_lines_to_gpkg
)

# Logging configuration
from .logging_config import (
    setup_logging,
    get_logger,
    ProgressTracker
)

# Miscellaneous helpers
from .helpers import detect_transect_direction

# Batch processing utilities
from .batch import (
    extract_year_from_path,
    extract_capture_date_from_path,
    group_rasters_by_year,
    resolve_year_band_config,
    detect_cir_from_filename
)

# Footprint clipping
from .footprint_clip import (
    clip_line_to_footprint,
    get_raster_valid_footprint
)

__all__ = [
    # Data I/O
    'build_raster_index',
    'load_transects',
    'reproject_if_needed',
    'find_overlapping_rasters',
    'RasterIndex',
    'BAND_CONFIG_4BAND',
    'BAND_CONFIG_CIR',
    'BAND_CONFIG_RGB',
    'detect_band_mode_from_dataframe',
    'validate_raster_bands',
    # Export
    'export_results_to_csv',
    'export_summary_json',
    'select_representative_transects',
    'validate_output_directory',
    'calculate_processing_stats',
    'print_processing_summary',
    'export_shell_line_geojson',
    'bundle_shell_lines_to_gpkg',
    # Logging
    'setup_logging',
    'get_logger',
    'ProgressTracker',
    # Helpers
    'detect_transect_direction',
    # Batch
    'extract_year_from_path',
    'extract_capture_date_from_path',
    'group_rasters_by_year',
    'resolve_year_band_config',
    'detect_cir_from_filename',
    # Footprint
    'clip_line_to_footprint',
    'get_raster_valid_footprint',
]