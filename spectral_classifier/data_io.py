"""
Data I/O module for loading rasters and transects with CRS handling.

Supports 3-band (RGB or CIR) and 4-band (RGBN) imagery with automatic
band configuration detection.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.crs import CRS
from shapely.geometry import box
import warnings

from .config import BAND_DETECTION

logger = logging.getLogger(__name__)


# Band configuration constants
BAND_CONFIG_4BAND = '4band'      # Standard RGBN (4 bands)
BAND_CONFIG_CIR = 'cir'          # Color Infrared [NIR, Red, Green] stored as "RGB"
BAND_CONFIG_RGB = 'rgb'          # True color [Red, Green, Blue]


def detect_band_mode_from_dataframe(df: pd.DataFrame) -> str:
    """
    Detect band mode from a DataFrame containing spectral data or features.

    Resolution order:
    1. Use 'band_mode' column if present and valid
    2. Check NIR and Blue band availability to infer mode

    Args:
        df: DataFrame with spectral data (may contain 'band_mode', 'nir', 'blue' columns)

    Returns:
        Band configuration string: '4band', 'cir', or 'rgb'
    """
    # Check if band_mode column exists
    if 'band_mode' in df.columns:
        mode = df['band_mode'].iloc[0]
        if pd.notna(mode):
            return mode

    # Fallback: check NIR availability
    if 'nir' in df.columns and not df['nir'].isna().all():
        # Check if we also have blue
        if 'blue' in df.columns and not df['blue'].isna().all():
            return BAND_CONFIG_4BAND
        return BAND_CONFIG_CIR
    return BAND_CONFIG_RGB


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
        self.band_configs = {}  # {path: band_config} - track band configuration per raster

    def __repr__(self):
        # Summarize band configs
        config_counts = {}
        for config in self.band_configs.values():
            config_counts[config] = config_counts.get(config, 0) + 1
        config_str = ", ".join(f"{k}={v}" for k, v in config_counts.items())
        return (f"RasterIndex(n_rasters={len(self.raster_paths)}, "
                f"skipped={len(self.skipped)}, crs={self.crs}, bands=[{config_str}])")
    
    def __len__(self):
        return len(self.raster_paths)
    
    def get_band_config(self, raster_path: Path) -> str:
        """Get band configuration for a specific raster."""
        return self.band_configs.get(raster_path, BAND_CONFIG_RGB)
    
    def get_band_config_summary(self) -> Dict[str, int]:
        """Get summary of band configurations across all rasters."""
        summary = {}
        for config in self.band_configs.values():
            summary[config] = summary.get(config, 0) + 1
        return summary
    
    def get_primary_band_mode(self) -> str:
        """
        Get the primary (most common) band mode across all rasters.
        
        Returns:
            Band mode string: '4band', 'cir', or 'rgb'
        """
        summary = self.get_band_config_summary()
        if not summary:
            return BAND_CONFIG_RGB
        return max(summary, key=summary.get)


def detect_band_configuration(raster_path: Path, sample_size: int = None) -> str:
    """
    Detect whether a 3-band raster is CIR [NIR,R,G] or RGB [R,G,B].
    
    Uses statistical heuristics based on band variance ratios:
    - In CIR imagery, Band 1 (NIR) has higher variance than Band 2 (Red)
      due to strong vegetation/water contrast in NIR
    - In RGB imagery, Band 1 (Red) has similar or lower variance than Band 2 (Green)
    
    Empirical observation from PAIS imagery:
    - CIR: Band1_StdDev / Band2_StdDev â‰ˆ 1.14 (NIR more variable)
    - RGB: Band1_StdDev / Band2_StdDev â‰ˆ 0.97 (Red less variable than Green)
    
    Args:
        raster_path: Path to raster file
        sample_size: Number of random pixels to sample for statistics
                    (uses BAND_DETECTION['sample_size'] if None)
        
    Returns:
        '4band' if 4+ bands present
        'cir' if detected as [NIR, Red, Green]
        'rgb' if detected as [Red, Green, Blue]
    """
    # Get configuration from config.py
    if sample_size is None:
        sample_size = BAND_DETECTION.get('sample_size', 10000)
    cir_threshold = BAND_DETECTION.get('cir_variance_ratio_threshold', 1.05)
    min_valid_samples = BAND_DETECTION.get('min_valid_samples', 100)
    
    with rasterio.open(raster_path) as src:
        # 4-band imagery - standard RGBN
        if src.count >= 4:
            return BAND_CONFIG_4BAND
        
        if src.count < 3:
            raise ValueError(f"Insufficient bands ({src.count}) in {raster_path}")
        
        # 3-band imagery - need to distinguish CIR from RGB
        # Sample random pixels to compute statistics
        height, width = src.height, src.width
        
        # Generate random sample indices
        np.random.seed(42)  # Reproducible
        n_samples = min(sample_size, height * width)
        sample_rows = np.random.randint(0, height, n_samples)
        sample_cols = np.random.randint(0, width, n_samples)
        
        # Read band data at sample locations
        band1_values = []
        band2_values = []
        
        # Read in windows for efficiency
        band1 = src.read(1)
        band2 = src.read(2)
        
        for r, c in zip(sample_rows, sample_cols):
            v1 = band1[r, c]
            v2 = band2[r, c]
            # Skip nodata
            if src.nodata is not None and (v1 == src.nodata or v2 == src.nodata):
                continue
            # Skip zero values (likely nodata)
            if v1 == 0 or v2 == 0:
                continue
            band1_values.append(v1)
            band2_values.append(v2)
        
        if len(band1_values) < min_valid_samples:
            logger.warning(f"Insufficient valid samples ({len(band1_values)}) for band detection in {raster_path.name}")
            # Default to RGB if we can't determine
            return BAND_CONFIG_RGB
        
        # Compute statistics
        band1_std = np.std(band1_values)
        band2_std = np.std(band2_values)
        
        # Compute variance ratio
        variance_ratio = band1_std / (band2_std + 1e-6)
        
        # Heuristic threshold from config
        # CIR: NIR (band1) has higher variance than Red (band2) â†’ ratio > threshold
        # RGB: Red (band1) has similar/lower variance than Green (band2) â†’ ratio â‰¤ threshold
        
        if variance_ratio > cir_threshold:
            detected = BAND_CONFIG_CIR
            logger.info(f"Detected CIR imagery: {raster_path.name} "
                       f"(Band1/Band2 StdDev ratio = {variance_ratio:.3f} > {cir_threshold})")
        else:
            detected = BAND_CONFIG_RGB
            logger.info(f"Detected RGB imagery: {raster_path.name} "
                       f"(Band1/Band2 StdDev ratio = {variance_ratio:.3f} â‰¤ {cir_threshold})")
        
        return detected


def build_raster_index(
    raster_dir: Path, 
    require_nir: bool = False,  # Changed default to False for 3-band support
    recursive: bool = False,
    detect_bands: bool = True,
    band_config_override: str = None
) -> RasterIndex:
    """
    Build spatial index of all supported raster files in directory.

    Args:
        raster_dir: Directory containing raster files
        require_nir: If True, skip files without 4 bands. If False, include 3-band files.
        recursive: If True, scan subdirectories recursively.
        detect_bands: If True, detect band configuration (CIR vs RGB) for 3-band files.
        band_config_override: If provided, use this config for ALL 3-band rasters instead
            of per-tile detection. Valid values: 'cir', 'rgb'. 4-band rasters always
            use '4band' regardless of override.

    Returns:
        RasterIndex object with spatial metadata and band configurations

    Raises:
        ValueError: If no usable rasters found, CRS mismatch detected, or CRS is geographic
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

    # Check if CRS is geographic (angular units) - pipeline requires projected CRS (meters)
    if common_crs.is_geographic:
        raise ValueError(
            f"Raster CRS is geographic ({common_crs}). "
            f"This pipeline requires projected coordinates (meters) for accurate sampling. "
            f"Please reproject your rasters to a projected CRS like EPSG:26914 (UTM Zone 14N) using:\n"
            f"  gdalwarp -t_srs EPSG:26914 input.tif output.tif"
        )

    # Build index, tracking skipped files
    valid_paths = []
    skipped = {}
    band_configs = {}

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
                
                # Determine band configuration
                if src.count >= 4:
                    # 4-band is always RGBN regardless of override
                    band_configs[raster_path] = BAND_CONFIG_4BAND
                elif band_config_override is not None:
                    # 3-band with year-level override - use override instead of per-tile detection
                    band_configs[raster_path] = band_config_override
                    logger.debug(f"Using override band config '{band_config_override}' for {raster_path.name}")
                elif detect_bands:
                    # 3-band without override - detect per-tile
                    try:
                        band_config = detect_band_configuration(raster_path)
                        band_configs[raster_path] = band_config
                    except Exception as e:
                        logger.warning(f"Could not detect band config for {raster_path.name}: {e}")
                        band_configs[raster_path] = BAND_CONFIG_RGB
                else:
                    # No detection, no override - default to RGB
                    band_configs[raster_path] = BAND_CONFIG_RGB
                
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
    index.band_configs = band_configs

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
    band_summary = index.get_band_config_summary()
    logger.info(
        f"Built raster index: {len(valid_paths)} usable, {len(skipped)} skipped, "
        f"CRS: {common_crs}, bands: {band_summary}"
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


def validate_raster_bands(raster_path: Path) -> Dict[str, any]:
    """
    Validate raster has correct band structure and detect configuration.

    Args:
        raster_path: Path to raster file

    Returns:
        Dictionary with band metadata including detected configuration

    Raises:
        ValueError: If band structure is invalid
    """
    with rasterio.open(raster_path) as src:
        if src.count < 3:
            raise ValueError(
                f"Expected at least 3 bands (R,G,B), got {src.count} "
                f"in {raster_path}"
            )

        # Detect band configuration
        band_config = detect_band_configuration(raster_path)

        metadata = {
            'count': src.count,
            'dtype': src.dtypes[0],
            'width': src.width,
            'height': src.height,
            'crs': src.crs,
            'has_nir': src.count >= 4 or band_config == BAND_CONFIG_CIR,
            'band_config': band_config
        }

    return metadata