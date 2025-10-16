"""
Spectral sampling module for extracting values along transects.
"""

import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import LineString, Point
from .config import SAMPLING_INTERVAL

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
    raster_datasets: List[rasterio.DatasetReader]
) -> np.ndarray:
    """
    Sample all 4 bands at a point location using bilinear interpolation.

    Args:
        point: Shapely Point geometry
        raster_datasets: List of open rasterio datasets

    Returns:
        Array of [red, green, blue, nir] values, or NaN if outside all rasters

    Raises:
        ValueError: If point is outside all rasters
    """
    x, y = point.x, point.y

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

        # Sample all 4 bands with bilinear interpolation
        try:
            # Use rasterio's sampling - returns array of [r, g, b, nir] for the point
            values = next(dataset.sample([(x, y)], indexes=[1, 2, 3, 4]))

            # Check for nodata values
            if dataset.nodata is not None:
                if np.any(values == dataset.nodata):
                    continue

            return values

        except Exception as e:
            logger.warning(f"Error sampling at ({x}, {y}): {e}")
            continue

    # If we get here, point is outside all rasters or in nodata region
    raise ValueError(
        f"Point ({x:.2f}, {y:.2f}) is outside all raster coverage or in nodata region"
    )


def sample_transect(
    transect_row,
    overlapping_rasters: List[Path],
    direction: str = 'west_to_east'
) -> pd.DataFrame:
    """
    Extract spectral values along a transect from rasters.

    Args:
        transect_row: Row from transects GeoDataFrame (must have geometry and TransectID)
        overlapping_rasters: List of raster file paths that intersect the transect
        direction: 'west_to_east' or 'east_to_west' - determines distance sign

    Returns:
        DataFrame with columns: [TransectID, distance, red, green, blue, nir]
    """
    transect_id = transect_row.TransectID
    geometry = transect_row.geometry

    logger.debug(f"Sampling transect {transect_id}")

    # Generate sample points along transect
    sample_points = interpolate_points_along_line(geometry, SAMPLING_INTERVAL)

    # Get max distance for reversal if needed
    max_distance = sample_points[-1][1] if sample_points else 0

    # Open all overlapping rasters
    datasets = []
    try:
        for raster_path in overlapping_rasters:
            datasets.append(rasterio.open(raster_path))

        # Sample spectral values at each point
        data = []
        for point, distance in sample_points:
            try:
                spectral_values = sample_raster_at_point(point, datasets)

                data.append({
                    'TransectID': transect_id,
                    'distance': distance,  # Keep original distance for now
                    'red': spectral_values[0],
                    'green': spectral_values[1],
                    'blue': spectral_values[2],
                    'nir': spectral_values[3]
                })

            except ValueError as e:
                logger.warning(
                    f"Skipping point at distance {distance:.1f}m: {e}"
                )
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

    logger.info(
        f"Transect {transect_id}: sampled {len(df)} points over "
        f"{df['distance'].max():.1f}m"
    )

    return df


def validate_spectral_data(df: pd.DataFrame) -> bool:
    """
    Validate spectral data for quality issues.

    Args:
        df: DataFrame with spectral band columns

    Returns:
        True if valid, raises ValueError otherwise

    Raises:
        ValueError: If data quality issues detected
    """
    # Check for missing values
    if df[['red', 'green', 'blue', 'nir']].isnull().any().any():
        raise ValueError("Spectral data contains NaN values")

    # Check for negative values (invalid reflectance)
    if (df[['red', 'green', 'blue', 'nir']] < 0).any().any():
        raise ValueError("Spectral data contains negative values")

    # Check for unreasonable values (assuming 8-bit or 16-bit data)
    max_val = df[['red', 'green', 'blue', 'nir']].max().max()
    if max_val > 65535:
        logger.warning(
            f"Spectral values exceed expected range (max={max_val})"
        )

    # Check distance monotonicity (should be increasing regardless of sign)
    if not df['distance'].is_monotonic_increasing:
        raise ValueError("Distance values are not monotonically increasing")

    return True
