"""
Data I/O module for loading rasters and transects with CRS handling.
"""

import logging
from pathlib import Path
from typing import List
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from shapely.geometry import box
import pandas as pd

logger = logging.getLogger(__name__)

class RasterIndex:
    """Spatial index for raster files with lazy loading."""

    def __init__(self, raster_paths: List[Path], crs: CRS):
        """
        Initialize raster index.

        Args:
            raster_paths: List of paths to raster files
            crs: Common CRS for all rasters
        """
        self.raster_paths = raster_paths
        self.crs = crs
        self.bounds = {}  # {path: (minx, miny, maxx, maxy)}
        self.geometries = {}  # {path: shapely.geometry.box}
    
    def __len__(self):
        return len(self.raster_paths)

def build_raster_index(raster_paths: List[Path]) -> RasterIndex:
    """
    Build spatial index from a pre-validated list of raster paths.
    CRS consistency is assumed (enforced by preprocessing).
    """
    if not raster_paths:
        raise ValueError("No raster paths provided")

    # Grab CRS from first file only
    with rasterio.open(raster_paths[0]) as src:
        common_crs = src.crs

    if common_crs is None:
        raise ValueError(f"Could not read CRS from {raster_paths[0]}")
    if common_crs.is_geographic:
        raise ValueError(
            f"Raster CRS is geographic ({common_crs}). "
            "Reproject to a projected CRS (e.g. EPSG:26914) before running."
        )

    # Build spatial index — one open per file, bounds only
    index = RasterIndex(raster_paths, common_crs)
    for path in raster_paths:
        with rasterio.open(path) as src:
            bounds = src.bounds
            index.bounds[path] = bounds
            index.geometries[path] = box(
                bounds.left, bounds.bottom, bounds.right, bounds.top
            )
        logger.debug(f"Indexed {path.name}: bounds={bounds}")

    logger.info(f"Built raster index: {len(raster_paths)} rasters, CRS: {common_crs}")
    return index


def load_transects(geojson_path: Path) -> gpd.GeoDataFrame:
    """
    Load transects from GeoJSON file.

    Args:
        geojson_path: Path to GeoJSON file

    Returns:
        GeoDataFrame with transects sorted by TransectID

    Raises:
        ValueError: If file is invalid or missing required fields
    """
    geojson_path = Path(geojson_path)

    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

    # Load GeoJSON
    gdf = gpd.read_file(geojson_path)

    # Validate geometry type
    if not all(gdf.geometry.type == 'LineString'):
        raise ValueError("All transect geometries must be LineString")

    # Validate TransectID field
    if 'TransectID' not in gdf.columns:
        raise ValueError("GeoJSON must contain 'TransectID' field")

    # Sort by TransectID
    gdf = gdf.sort_values('TransectID').reset_index(drop=True)

    logger.info(
        f"Loaded {len(gdf)} transects from {geojson_path.name} "
        f"(CRS: {gdf.crs})"
    )

    return gdf


def reproject_if_needed(
    transects: gpd.GeoDataFrame,
    target_crs: CRS
) -> gpd.GeoDataFrame:
    """
    Reproject transects to target CRS if needed.

    Args:
        transects: GeoDataFrame of transects
        target_crs: Target CRS (from raster index)

    Returns:
        GeoDataFrame in target CRS
    """
    if transects.crs is None:
        raise ValueError("Transect GeoDataFrame has no CRS defined")

    # Compare CRS
    if transects.crs == target_crs:
        logger.info("Transect CRS matches raster CRS - no reprojection needed")
        return transects

    # Reproject
    logger.warning(
        f"Reprojecting transects from {transects.crs} to {target_crs}"
    )
    transects_reprojected = transects.to_crs(target_crs)

    return transects_reprojected


def find_overlapping_rasters(
    transect_geometry,
    raster_index: RasterIndex
) -> List[Path]:
    """
    Find all rasters that intersect with transect geometry.

    Args:
        transect_geometry: Shapely LineString geometry
        raster_index: RasterIndex object

    Returns:
        List of raster paths that intersect the transect (may be empty)
    """
    overlapping = []

    for raster_path, raster_geom in raster_index.geometries.items():
        if transect_geometry.intersects(raster_geom):
            overlapping.append(raster_path)

    if overlapping:
        logger.debug(
            f"Found {len(overlapping)} overlapping rasters for transect"
        )
    else:
        logger.debug(
            f"No rasters overlap with transect (bounds: {transect_geometry.bounds})"
        )

    return overlapping

# ---------------------------------------------------------------------------
# Parquet merge helper
# ---------------------------------------------------------------------------

def merge_profiles_and_features(
    profiles_df: pd.DataFrame,
    features_df: pd.DataFrame,
    columns: list = None,
) -> pd.DataFrame:
    """
    Join profiles and features parquets on (transect_id, distance).

    The two parquets share the join key ``(transect_id, distance)`` but
    contain disjoint column sets (raw bands vs derived features).  This
    helper merges them into a single DataFrame so that downstream code
    can access both raw DN values and computed indices per sample point.

    Args:
        profiles_df: Raw band samples (transect_id, distance, x, y,
                     red, green, blue, nir).
        features_df: Derived features (transect_id, distance, ndwi,
                     nir_ratio, derivatives, ...).
        columns:     Optional list of feature column names to keep.
                     If None, all feature columns are included.
                     ``transect_id`` and ``distance`` are always kept.

    Returns:
        Merged DataFrame with all profile columns plus selected feature
        columns, one row per sample point.
    """
    join_keys = ["transect_id", "distance"]

    if columns is not None:
        # Always include join keys in the feature slice.
        keep = list(set(join_keys) | set(columns))
        features_subset = features_df[
            [c for c in keep if c in features_df.columns]
        ]
    else:
        features_subset = features_df

    merged = profiles_df.merge(features_subset, on=join_keys, how="inner")

    logger.info(
        "Merged profiles (%d rows) + features (%d rows) -> %d rows, %d columns",
        len(profiles_df), len(features_df), len(merged), len(merged.columns),
    )

    return merged