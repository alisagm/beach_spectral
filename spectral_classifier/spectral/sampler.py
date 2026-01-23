"""
Spectral sampling module for extracting values along transects.

Supports 3-band (RGB or CIR) and 4-band (RGBN) imagery with automatic
band remapping based on detected configuration.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import LineString, Point
from ..config import SAMPLING_INTERVAL
from ..utils.data_io import RasterIndex, BAND_CONFIG_4BAND, BAND_CONFIG_CIR, BAND_CONFIG_RGB

logger = logging.getLogger(__name__)


def interpolate_points_along_line(
    line: LineString,
    interval: float = SAMPLING_INTERVAL
) -> List[Tuple[Point, float]]:
    """
    Generate points at fixed intervals along a LineString.

    Args:
        line: Shapely LineString geometry
        interval: Distance between points in same units as line CRS

    Returns:
        List of (Point, distance_from_start) tuples
    """
    points = []
    total_length = line.length
    distance = 0.0

    while distance <= total_length:
        point = line.interpolate(distance)
        points.append((point, distance))
        distance += interval

    # Always include the end point
    if points[-1][1] < total_length:
        end_point = line.interpolate(total_length)
        points.append((end_point, total_length))

    logger.debug(
        f"Generated {len(points)} sample points along {total_length:.1f}m transect"
    )

    return points


def sample_raster_at_point(
    point: Point,
    raster_datasets: List[rasterio.DatasetReader],
    band_configs: Dict[Path, str] = None
) -> Tuple[np.ndarray, str]:
    """
    Sample spectral bands at a point location using bilinear interpolation.
    
    Handles different band configurations:
    - 4band: Returns [red, green, blue, nir] directly
    - cir: Remaps [NIR, Red, Green] to [red, green, blue, nir] with blue=None marker
    - rgb: Returns [red, green, blue, None] with nir=None marker

    Args:
        point: Shapely Point geometry
        raster_datasets: List of open rasterio datasets
        band_configs: Dict mapping dataset paths to band configuration strings

    Returns:
        Tuple of (array of [red, green, blue, nir] values, band_config string)
        For 3-band imagery, missing bands are set to np.nan

    Raises:
        ValueError: If point is outside all rasters
    """
    x, y = point.x, point.y
    band_configs = band_configs or {}

    # Try each raster dataset
    for dataset in raster_datasets:
        # Check if point is within raster bounds
        if not (
            dataset.bounds.left <= x <= dataset.bounds.right and
            dataset.bounds.bottom <= y <= dataset.bounds.top
        ):
            continue

        # Convert geographic coordinates to pixel coordinates
        row, col = dataset.index(x, y)

        # Check if within raster dimensions
        if not (0 <= row < dataset.height and 0 <= col < dataset.width):
            continue

        # Determine band configuration
        # Try to get from band_configs dict using dataset name
        dataset_path = Path(dataset.name) if dataset.name else None
        band_config = band_configs.get(dataset_path, None)
        
        # Fallback: detect from band count
        if band_config is None:
            if dataset.count >= 4:
                band_config = BAND_CONFIG_4BAND
            else:
                # Default to RGB if we can't determine
                band_config = BAND_CONFIG_RGB
                logger.debug(f"No band config for {dataset_path}, defaulting to RGB")

        try:
            if band_config == BAND_CONFIG_4BAND:
                # Standard 4-band: read all 4 bands
                values = next(dataset.sample([(x, y)], indexes=[1, 2, 3, 4]))
                
                # Check for nodata values
                if dataset.nodata is not None and np.any(values == dataset.nodata):
                    continue
                    
                return values.astype(float), band_config
                
            elif band_config == BAND_CONFIG_CIR:
                # CIR [NIR, Red, Green] stored as bands 1,2,3
                # Remap to [Red, Green, Blue=nan, NIR]
                raw_values = next(dataset.sample([(x, y)], indexes=[1, 2, 3]))
                
                # Check for nodata
                if dataset.nodata is not None and np.any(raw_values == dataset.nodata):
                    continue
                
                # Remap: band1=NIR, band2=Red, band3=Green
                nir = float(raw_values[0])
                red = float(raw_values[1])
                green = float(raw_values[2])
                blue = np.nan  # No blue band in CIR
                
                values = np.array([red, green, blue, nir])
                return values, band_config
                
            else:  # BAND_CONFIG_RGB
                # True RGB [Red, Green, Blue] - no NIR
                raw_values = next(dataset.sample([(x, y)], indexes=[1, 2, 3]))
                
                # Check for nodata
                if dataset.nodata is not None and np.any(raw_values == dataset.nodata):
                    continue
                
                red = float(raw_values[0])
                green = float(raw_values[1])
                blue = float(raw_values[2])
                nir = np.nan  # No NIR band in RGB
                
                values = np.array([red, green, blue, nir])
                return values, band_config

        except Exception as e:
            # Log to file but don't spam console
            logger.debug(f"Error sampling at ({x}, {y}): {e}")
            continue

    # If we get here, point is outside all rasters or in nodata region
    raise ValueError(
        f"Point ({x:.2f}, {y:.2f}) is outside all raster coverage or in nodata region"
    )


def sample_transect(
    transect_row,
    overlapping_rasters: List[Path],
    direction: str = 'west_to_east',
    raster_index: RasterIndex = None
) -> Tuple[pd.DataFrame, int]:
    """
    Extract spectral values along a transect from rasters.

    Args:
        transect_row: Row from transects GeoDataFrame (must have geometry and TransectID)
        overlapping_rasters: List of raster file paths that intersect the transect
        direction: 'west_to_east' or 'east_to_west' - determines distance sign
        raster_index: Optional RasterIndex for band configuration lookup

    Returns:
        Tuple of:
            - DataFrame with columns: [TransectID, distance, red, green, blue, nir]
              (nir may be NaN for RGB-only imagery, blue may be NaN for CIR imagery)
            - int: Number of points skipped (outside coverage or nodata)
        
    Note:
        Band mode is determined at the dataset level via raster_index.get_primary_band_mode(),
        not per-transect. This keeps the return signature simple.
    """
    transect_id = transect_row.TransectID
    geometry = transect_row.geometry

    logger.debug(f"Sampling transect {transect_id}")

    # Generate sample points along transect
    sample_points = interpolate_points_along_line(geometry, SAMPLING_INTERVAL)

    # Get max distance for reversal if needed
    max_distance = sample_points[-1][1] if sample_points else 0

    # Build band config lookup
    band_configs = {}
    if raster_index is not None:
        band_configs = raster_index.band_configs

    # Open all overlapping rasters
    datasets = []
    skipped_count = 0  # Track skipped points for aggregated reporting
    
    try:
        for raster_path in overlapping_rasters:
            datasets.append(rasterio.open(raster_path))

        # Sample spectral values at each point
        data = []
        for point, distance in sample_points:
            try:
                spectral_values, band_config = sample_raster_at_point(
                    point, datasets, band_configs
                )

                data.append({
                    'TransectID': transect_id,
                    'distance': distance,  # Keep original distance for now
                    'red': spectral_values[0],
                    'green': spectral_values[1],
                    'blue': spectral_values[2],
                    'nir': spectral_values[3]  # May be NaN for RGB
                })

            except ValueError:
                # Point outside coverage - count but don't log each one
                skipped_count += 1
                continue

        # For east_to_west transects, reverse the data order so westmost point comes first
        # This ensures plotting from left to right shows west->east
        if direction == 'east_to_west':
            data.reverse()
            # Recalculate distances from 0 to max in the new order
            for i, point_data in enumerate(data):
                point_data['distance'] = max_distance - point_data['distance']

    finally:
        # Close all raster datasets
        for dataset in datasets:
            dataset.close()

    if not data:
        raise ValueError(
            f"No valid spectral samples obtained for transect {transect_id}"
        )

    df = pd.DataFrame(data)
    
    # Log sampling summary (to file, not console)
    has_nir = not df['nir'].isna().all()
    has_blue = not df['blue'].isna().all()
    
    # Determine band mode for logging
    if has_nir and has_blue:
        band_mode = BAND_CONFIG_4BAND
    elif has_nir and not has_blue:
        band_mode = BAND_CONFIG_CIR
    else:
        band_mode = BAND_CONFIG_RGB
    
    # Summary log includes skip count
    total_points = len(sample_points)
    sampled_points = len(df)
    logger.debug(
        f"Transect {transect_id}: sampled {sampled_points}/{total_points} points "
        f"over {df['distance'].max():.1f}m (band_mode={band_mode}, skipped={skipped_count})"
    )

    return df, skipped_count


def validate_spectral_data(df: pd.DataFrame, require_nir: bool = False) -> bool:
    """
    Validate spectral data for quality issues.

    Args:
        df: DataFrame with spectral band columns
        require_nir: If True, require non-NaN NIR values

    Returns:
        True if valid, raises ValueError otherwise

    Raises:
        ValueError: If data quality issues detected
    """
    # Determine which bands to validate
    if require_nir:
        bands_to_check = ['red', 'green', 'blue', 'nir']
    else:
        # For 3-band data, only check RGB (allow NaN in NIR)
        bands_to_check = ['red', 'green', 'blue']
    
    # Check for missing values in required bands
    for band in bands_to_check:
        if df[band].isnull().any():
            if band == 'nir' and not require_nir:
                continue  # Allow NaN in NIR for 3-band data
            if band == 'blue':
                # Allow NaN in Blue for CIR data
                continue
            raise ValueError(f"Spectral data contains NaN values in {band} band")

    # Check for negative values (invalid reflectance)
    for band in bands_to_check:
        if (df[band].dropna() < 0).any():
            raise ValueError(f"Spectral data contains negative values in {band} band")

    # Check for unreasonable values (assuming 8-bit or 16-bit data)
    for band in bands_to_check:
        max_val = df[band].dropna().max()
        if max_val > 65535:
            logger.warning(
                f"Spectral values in {band} exceed expected range (max={max_val})"
            )

    # Check distance monotonicity (should be increasing regardless of sign)
    if not df['distance'].is_monotonic_increasing:
        raise ValueError("Distance values are not monotonically increasing")

    return True