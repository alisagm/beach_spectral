"""
Spectral sampling module for extracting values along transects.

Supports 3-band (RGB or CIR) and 4-band (RGBNIR) imagery.
Band assignments are read from band_config.json rather than detected at runtime.

Distance convention
-------------------
``distance`` is always arc-length measured from the **lowest-easting vertex** of
the transect as defined in the source GeoJSON.  For the standard PAIS east-to-west
transect orientation this means:

    distance = 0   →  landward / west end
    distance ≈ 300 →  seaward  / east end

No directional flag is needed anywhere in the pipeline.  Visualization code can
plot ``distance`` directly on a west-left / east-right x-axis.
"""

import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import LineString, Point
from ..config import SAMPLING_INTERVAL
from utils import resolve_band_indices

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def interpolate_points_along_line(
    line: LineString,
    interval: float = SAMPLING_INTERVAL,
) -> List[Tuple[Point, float]]:
    """
    Generate points at fixed intervals along a LineString.

    Distances are arc-length from the **first vertex** of ``line`` as defined
    in the source geometry.  Callers that need distances measured from the
    lowest-easting vertex should apply the flip in ``sample_transect``.

    Args:
        line: Shapely LineString geometry
        interval: Distance between points in same units as line CRS

    Returns:
        List of (Point, distance_from_first_vertex) tuples
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


# ---------------------------------------------------------------------------
# Core sampling functions
# ---------------------------------------------------------------------------

def sample_raster_at_point(
    point: Point,
    raster_datasets: List[rasterio.DatasetReader],
    year_config: dict,
) -> np.ndarray:
    """
    Sample spectral bands at a point location using the year's band configuration.

    All band routing is driven by year_config (from band_config.json) via
    resolve_band_indices(). No per-file detection is performed.

    Args:
        point: Shapely Point geometry
        raster_datasets: List of open rasterio datasets to search
        year_config: Single year entry from band_config.json

    Returns:
        np.ndarray of shape (4,): [red, green, blue, nir]
        Missing bands (blue for CIR, nir for RGB) are np.nan.

    Raises:
        ValueError: If point is outside all rasters or in a nodata region
    """
    x, y = point.x, point.y
    indices = resolve_band_indices(year_config)

    # Collect the 1-based band indices we actually need to read
    bands_to_read = [
        idx for idx in [
            indices["red"], indices["green"], indices["blue"], indices["nir"]
        ]
        if idx is not None
    ]

    for dataset in raster_datasets:
        # Bounds check
        if not (
            dataset.bounds.left <= x <= dataset.bounds.right
            and dataset.bounds.bottom <= y <= dataset.bounds.top
        ):
            continue

        # Pixel check (guards against floating-point edge cases)
        row, col = dataset.index(x, y)
        if not (0 <= row < dataset.height and 0 <= col < dataset.width):
            continue

        try:
            raw = next(dataset.sample([(x, y)], indexes=bands_to_read))

            # Nodata check across all read bands
            if dataset.nodata is not None and np.any(raw == dataset.nodata):
                continue

            # DN=0 sentinel check (black fill / unwritten tiles)
            if np.all(raw == 0):
                continue

            # Map raw values back to [red, green, blue, nir]
            band_map = {
                band_idx: float(val)
                for band_idx, val in zip(bands_to_read, raw)
            }

            return np.array([
                band_map.get(indices["red"],   np.nan),
                band_map.get(indices["green"],  np.nan),
                band_map.get(indices["blue"],   np.nan) if indices["blue"]  is not None else np.nan,
                band_map.get(indices["nir"],    np.nan) if indices["nir"]   is not None else np.nan,
            ])

        except Exception as e:
            logger.debug(f"Error sampling at ({x:.2f}, {y:.2f}): {e}")
            continue

    raise ValueError(
        f"Point ({x:.2f}, {y:.2f}) is outside all raster coverage or in nodata region"
    )


def sample_transect(
    transect_row,
    overlapping_rasters: List[Path],
    year_config: dict,
) -> Tuple[pd.DataFrame, int]:
    """
    Extract spectral values along a transect from rasters.

    Distance convention
    -------------------
    ``distance`` is arc-length from the **lowest-easting vertex** of the transect
    geometry, regardless of how the vertices are ordered in the source GeoJSON.
    For PAIS transects (east-to-west in the GeoJSON) this means:

        distance = 0   →  west / landward end
        distance ≈ 300 →  east / seaward end

    Nodata / out-of-coverage points are recorded as NaN rows and then linearly
    interpolated over interior gaps (bounded by valid data on both sides).
    Leading/trailing NaNs at the transect ends are left as NaN.

    Args:
        transect_row: Row from transects GeoDataFrame (needs .geometry and .TransectID)
        overlapping_rasters: Raster paths that intersect the transect
        year_config: Single year entry from band_config.json

    Returns:
        Tuple of:
            - DataFrame: [TransectID, distance, x, y, red, green, blue, nir]
              sorted by distance (west → east).
              (blue is NaN for CIR, nir is NaN for RGB)
            - int: Number of points that hit nodata and were interpolated over
    """
    transect_id = transect_row.TransectID
    geometry = transect_row.geometry

    logger.debug(f"Sampling transect {transect_id}")

    sample_points = interpolate_points_along_line(geometry, SAMPLING_INTERVAL)

    # Determine whether distances need flipping so that the lowest-easting
    # end equals distance=0.  For standard PAIS east-to-west transects the
    # first vertex is the easternmost, so we flip.
    start_easting = geometry.coords[0][0]
    end_easting   = geometry.coords[-1][0]
    flip_distances = start_easting > end_easting

    max_distance = sample_points[-1][1] if sample_points else 0.0

    datasets = []
    nodata_count = 0

    try:
        for raster_path in overlapping_rasters:
            datasets.append(rasterio.open(raster_path))

        data = []
        for point, arc_distance in sample_points:
            distance = (max_distance - arc_distance) if flip_distances else arc_distance

            try:
                spectral_values = sample_raster_at_point(point, datasets, year_config)
                data.append({
                    "TransectID": transect_id,
                    "distance":   distance,
                    "x":          point.x,
                    "y":          point.y,
                    "red":        spectral_values[0],
                    "green":      spectral_values[1],
                    "blue":       spectral_values[2],
                    "nir":        spectral_values[3],
                })
            except ValueError:
                # Point is outside coverage or in nodata — insert NaN row
                # for interpolation below rather than silently dropping it.
                data.append({
                    "TransectID": transect_id,
                    "distance":   distance,
                    "x":          point.x,
                    "y":          point.y,
                    "red":        np.nan,
                    "green":      np.nan,
                    "blue":       np.nan,
                    "nir":        np.nan,
                })
                nodata_count += 1

    finally:
        for dataset in datasets:
            dataset.close()

    if not data:
        raise ValueError(
            f"No sample points generated for transect {transect_id}"
        )

    # Sort by distance (west → east) so interpolation and downstream consumers
    # always see a monotonically increasing distance column.
    df = pd.DataFrame(data).sort_values("distance").reset_index(drop=True)

    # Interpolate over interior nodata gaps (tile seams, missing tiles, etc.)
    # limit_area='inside' means only gaps bounded by valid data on both sides
    # are filled — we never extrapolate off the ends of the transect.
    spectral_cols = ["red", "green", "blue", "nir"]
    df[spectral_cols] = df[spectral_cols].interpolate(
        method="linear",
        limit_area="inside",
    )

    fmt = year_config["format"]
    logger.debug(
        f"Transect {transect_id}: {len(sample_points) - nodata_count}/{len(sample_points)} "
        f"valid points, {nodata_count} interpolated, format={fmt}, "
        f"distance range={df['distance'].min():.1f}–{df['distance'].max():.1f}m"
    )

    return df, nodata_count