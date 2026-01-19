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
        self.skipped = {}  # {path: reason} - track skipped files

    def __repr__(self):
        return f"RasterIndex(n_rasters={len(self.raster_paths)}, skipped={len(self.skipped)}, crs={self.crs})"
    
    def __len__(self):
        return len(self.raster_paths)


def build_raster_index(raster_dir: Path, require_nir: bool = True, recursive: bool = False) -> RasterIndex:
    """
    Build spatial index of all supported raster files in directory.

    Args:
        raster_dir: Directory containing raster files
        require_nir: If True, skip files without 4 bands. If False, include 3-band files.
        recursive: If True, scan subdirectories recursively.

    Returns:
        RasterIndex object with spatial metadata

    Raises:
        ValueError: If no usable rasters found or CRS mismatch detected
    """
    raster_dir = Path(raster_dir)
    
    # Scan for supported formats (recursive or flat)
    if recursive:
        all_raster_paths = (
            list(raster_dir.rglob('*.tif')) + 
            list(raster_dir.rglob('*.tiff')) +
            list(raster_dir.rglob('*.jp2'))
        )
    else:
        all_raster_paths = (
            list(raster_dir.glob('*.tif')) + 
            list(raster_dir.glob('*.tiff')) +
            list(raster_dir.glob('*.jp2'))
        )

    if not all_raster_paths:
        raise ValueError(f"No supported raster files (.tif, .tiff, .jp2) found in {raster_dir}")

    logger.info(f"Found {len(all_raster_paths)} raster files" + (" (recursive)" if recursive else ""))

    # Find first valid raster to establish CRS
    common_crs = None
    for raster_path in all_raster_paths:
        try:
            with rasterio.open(raster_path) as src:
                if src.crs is not None:
                    common_crs = src.crs
                    break
        except Exception as e:
            logger.warning(f"Could not open {raster_path.name}: {e}")
            continue
    
    if common_crs is None:
        raise ValueError("No rasters with valid CRS found")

    # Build index, tracking skipped files
    valid_paths = []
    skipped = {}

    for raster_path in all_raster_paths:
        try:
            with rasterio.open(raster_path) as src:
                # Check CRS consistency
                if src.crs is None:
                    skipped[raster_path] = "No CRS defined"
                    logger.warning(f"Skipping {raster_path.name}: No CRS defined")
                    continue
                
                if src.crs != common_crs:
                    skipped[raster_path] = f"CRS mismatch ({src.crs} != {common_crs})"
                    logger.warning(f"Skipping {raster_path.name}: CRS mismatch ({src.crs} != {common_crs})")
                    continue

                # Check band count
                if require_nir and src.count < 4:
                    skipped[raster_path] = f"Missing NIR ({src.count} bands, need 4)"
                    logger.warning(f"Skipping {raster_path.name}: {src.count} bands (NIR required)")
                    continue
                
                if src.count < 3:
                    skipped[raster_path] = f"Insufficient bands ({src.count})"
                    logger.warning(f"Skipping {raster_path.name}: Only {src.count} bands")
                    continue

                # Valid raster - store metadata
                valid_paths.append(raster_path)
                
        except Exception as e:
            skipped[raster_path] = f"Read error: {e}"
            logger.warning(f"Skipping {raster_path.name}: Could not read ({e})")
            continue

    if not valid_paths:
        raise ValueError(
            f"No usable rasters found. {len(skipped)} files skipped. "
            f"Reasons: {set(skipped.values())}"
        )

    # Create index with valid rasters
    index = RasterIndex(valid_paths, common_crs)
    index.skipped = skipped

    # Build spatial index for valid rasters
    for raster_path in valid_paths:
        with rasterio.open(raster_path) as src:
            bounds = src.bounds
            index.bounds[raster_path] = bounds
            index.geometries[raster_path] = box(
                bounds.left, bounds.bottom,
                bounds.right, bounds.top
            )
        logger.debug(f"Indexed {raster_path.name}: bounds={bounds}")

    # Summary log
    logger.info(
        f"Built raster index: {len(valid_paths)} usable, {len(skipped)} skipped, CRS: {common_crs}"
    )
    
    if skipped:
        skip_summary = {}
        for reason in skipped.values():
            skip_summary[reason] = skip_summary.get(reason, 0) + 1
        logger.info(f"Skip summary: {skip_summary}")

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
        if src.count < 3:
            raise ValueError(
                f"Expected at least 3 bands (R,G,B), got {src.count} "
                f"in {raster_path}"
            )

        metadata = {
            'count': src.count,
            'dtype': src.dtypes[0],
            'width': src.width,
            'height': src.height,
            'crs': src.crs,
            'has_nir': src.count >= 4
        }

    return metadata