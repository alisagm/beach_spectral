"""
Data I/O module for loading rasters and transects with CRS handling.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from shapely.geometry import box
import warnings

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

    def __repr__(self):
        return f"RasterIndex(n_rasters={len(self.raster_paths)}, crs={self.crs})"


def build_raster_index(raster_dir: Path) -> RasterIndex:
    """
    Build spatial index of all GeoTIFF files in directory.

    Args:
        raster_dir: Directory containing GeoTIFF files

    Returns:
        RasterIndex object with spatial metadata

    Raises:
        ValueError: If no rasters found or CRS mismatch detected
    """
    raster_dir = Path(raster_dir)
    raster_paths = list(raster_dir.glob('*.tif')) + list(raster_dir.glob('*.tiff'))

    if not raster_paths:
        raise ValueError(f"No GeoTIFF files found in {raster_dir}")

    logger.info(f"Found {len(raster_paths)} raster files")

    # Extract metadata from first raster
    with rasterio.open(raster_paths[0]) as src:
        common_crs = src.crs
        if common_crs is None:
            raise ValueError(f"Raster {raster_paths[0]} has no CRS defined")

    # Build index
    index = RasterIndex(raster_paths, common_crs)

    for raster_path in raster_paths:
        with rasterio.open(raster_path) as src:
            # Validate CRS consistency
            if src.crs != common_crs:
                raise ValueError(
                    f"CRS mismatch: {raster_path} has {src.crs}, "
                    f"expected {common_crs}"
                )

            # Validate band count
            if src.count != 4:
                raise ValueError(
                    f"Expected 4 bands, got {src.count} in {raster_path}"
                )

            # Store bounds and geometry
            bounds = src.bounds
            index.bounds[raster_path] = bounds
            index.geometries[raster_path] = box(
                bounds.left, bounds.bottom,
                bounds.right, bounds.top
            )

        logger.debug(f"Indexed {raster_path.name}: bounds={bounds}")

    logger.info(f"Built raster index with CRS: {common_crs}")
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
        List of raster paths that intersect the transect

    Raises:
        ValueError: If no overlapping rasters found
    """
    overlapping = []

    for raster_path, raster_geom in raster_index.geometries.items():
        if transect_geometry.intersects(raster_geom):
            overlapping.append(raster_path)

    if not overlapping:
        raise ValueError(
            f"No rasters overlap with transect geometry. "
            f"Transect bounds: {transect_geometry.bounds}"
        )

    logger.debug(
        f"Found {len(overlapping)} overlapping rasters for transect"
    )

    return overlapping


def validate_raster_bands(raster_path: Path) -> Dict[str, int]:
    """
    Validate raster has correct band structure.

    Args:
        raster_path: Path to raster file

    Returns:
        Dictionary with band metadata

    Raises:
        ValueError: If band structure is invalid
    """
    with rasterio.open(raster_path) as src:
        if src.count != 4:
            raise ValueError(
                f"Expected 4 bands (R,G,B,NIR), got {src.count} "
                f"in {raster_path}"
            )

        metadata = {
            'count': src.count,
            'dtype': src.dtypes[0],
            'width': src.width,
            'height': src.height,
            'crs': src.crs
        }

    return metadata
