"""
Footprint-based shoreline clipping module (v2).

Clips shorelines to valid imagery coverage, with optional edge buffering.
Handles nodata regions (black=0) and creates merged footprints from multiple rasters.

Key improvements over v1:
- MUCH faster: uses raster bounds by default, only does per-pixel analysis when needed
- Smarter nodata: only treats pure black (0,0,0) as nodata by default
- White nodata: only enabled if detected in image corners
- Progress logging for batch processing
- Minimum region size filter to avoid flagging small bright spots

Usage:
    from footprint_clip import clip_shoreline_to_footprint, compute_merged_footprint
    
    # Get merged footprint for a year's imagery
    footprint = compute_merged_footprint(raster_paths, edge_buffer_m=25)
    
    # Clip shoreline
    clipped = clip_shoreline_to_footprint(shoreline_geom, footprint)
"""

import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Tuple
import logging
import shapely

logger = logging.getLogger(__name__)


def get_raster_bounds_footprint(raster_path: Path) -> Optional[Tuple['shapely.geometry.Polygon', any]]:
    """
    Get simple rectangular footprint from raster bounds (FAST).
    
    Use this when rasters are rectangular without significant nodata regions.
    
    Args:
        raster_path: Path to raster file
        
    Returns:
        Tuple of (Polygon, CRS) or None if error
    """
    import rasterio
    from shapely.geometry import box
    
    try:
        with rasterio.open(raster_path) as src:
            bounds = src.bounds
            crs = src.crs
            footprint = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            return footprint, crs
    except Exception as e:
        logger.error(f"Error reading bounds from {raster_path}: {e}")
        return None


def check_corner_nodata(data: np.ndarray, corner_size: int = 50) -> dict:
    """
    Check corners of raster for nodata patterns.
    
    Returns dict with 'has_black_nodata' and 'has_white_nodata' flags.
    """
    h, w = data.shape[1], data.shape[2]
    corner_size = min(corner_size, h // 4, w // 4)
    
    if corner_size < 10:
        return {'has_black_nodata': False, 'has_white_nodata': False}
    
    # Sample corners
    corners = [
        data[:, :corner_size, :corner_size],           # top-left
        data[:, :corner_size, -corner_size:],          # top-right
        data[:, -corner_size:, :corner_size],          # bottom-left
        data[:, -corner_size:, -corner_size:],         # bottom-right
    ]
    
    has_black = False
    has_white = False
    
    for corner in corners:
        # Check if corner has significant black (all bands = 0)
        black_mask = np.all(corner == 0, axis=0)
        if np.mean(black_mask) > 0.5:  # >50% of corner is black
            has_black = True
        
        # Check if corner has significant white (all bands = 255)
        white_mask = np.all(corner == 255, axis=0)
        if np.mean(white_mask) > 0.5:  # >50% of corner is white
            has_white = True
    
    return {'has_black_nodata': has_black, 'has_white_nodata': has_white}


def get_raster_valid_footprint(
    raster_path: Path,
    check_nodata: bool = True,
    min_nodata_region_pixels: int = 10000
) -> Optional[Tuple['shapely.geometry.Polygon', any]]:
    """
    Compute the valid data footprint for a single raster.
    
    Identifies nodata regions as:
    - Pure black pixels (all bands = 0) - always checked
    - Pure white pixels (all bands = 255) - only if detected in corners
    
    Args:
        raster_path: Path to raster file
        check_nodata: If False, just return rectangular bounds (fast mode)
        min_nodata_region_pixels: Ignore nodata regions smaller than this
        
    Returns:
        Tuple of (Polygon, CRS) of valid data extent, or None if error
    """
    import rasterio
    from rasterio import features
    from shapely.geometry import shape, box
    from shapely.ops import unary_union
    
    # Fast mode: just use bounds
    if not check_nodata:
        return get_raster_bounds_footprint(raster_path)
    
    try:
        with rasterio.open(raster_path) as src:
            # Read all bands
            data = src.read()  # Shape: (bands, height, width)
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            
            # Check corners to determine nodata type
            corner_check = check_corner_nodata(data)
            
            # If no corner nodata detected, use simple bounds
            if not corner_check['has_black_nodata'] and not corner_check['has_white_nodata']:
                logger.debug(f"{raster_path.name}: No corner nodata, using bounds")
                footprint = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                return footprint, crs
            
            # Build nodata mask
            # Always check for pure black
            nodata_mask = np.all(data == 0, axis=0)
            
            # Only check for pure white if detected in corners
            if corner_check['has_white_nodata']:
                white_mask = np.all(data == 255, axis=0)
                nodata_mask = nodata_mask | white_mask
                logger.debug(f"{raster_path.name}: Detected white nodata in corners")
            
            valid_mask = ~nodata_mask
            
            # If almost all valid, use simple bounds
            valid_ratio = np.mean(valid_mask)
            if valid_ratio > 0.98:
                logger.debug(f"{raster_path.name}: {valid_ratio:.1%} valid, using bounds")
                footprint = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                return footprint, crs
            
            # Vectorize the valid data regions
            valid_uint8 = valid_mask.astype(np.uint8)
            
            shapes_gen = features.shapes(
                valid_uint8,
                mask=valid_mask,
                transform=transform
            )
            
            # Collect polygons, filtering small regions
            polygons = []
            for geom, value in shapes_gen:
                if value == 1:
                    poly = shape(geom)
                    # Rough pixel count estimate
                    pixel_area = abs(transform.a * transform.e)  # pixel size in CRS units
                    region_pixels = poly.area / pixel_area if pixel_area > 0 else float('inf')
                    
                    if region_pixels >= min_nodata_region_pixels:
                        polygons.append(poly)
            
            if not polygons:
                logger.warning(f"No valid polygons from {raster_path.name}, using bounds")
                footprint = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                return footprint, crs
            
            # Union all polygons
            if len(polygons) == 1:
                footprint = polygons[0]
            else:
                footprint = unary_union(polygons)
            
            # Simplify to reduce vertex count
            footprint = footprint.simplify(1.0, preserve_topology=True)
            
            logger.debug(f"{raster_path.name}: {valid_ratio:.1%} valid, {footprint.geom_type}")
            
            return footprint, crs
            
    except Exception as e:
        logger.error(f"Error processing {raster_path}: {e}")
        return None


def compute_merged_footprint(
    raster_paths: List[Path],
    edge_buffer_m: float = 25.0,
    fast_mode: bool = True,
    progress_interval: int = 10
) -> Optional[Tuple['shapely.geometry.base.BaseGeometry', any]]:
    """
    Compute merged valid-data footprint from multiple rasters.
    
    Creates a single (multi)polygon representing the union of all raster
    valid-data extents, then applies an inward buffer to the OUTER edges only.
    
    Args:
        raster_paths: List of paths to raster files
        edge_buffer_m: Inward buffer distance in meters (default 25)
        fast_mode: If True, use bounds-only for speed (default True)
        progress_interval: Log progress every N rasters
        
    Returns:
        Tuple of (buffered_footprint, crs) or None if no valid rasters
    """
    from shapely.ops import unary_union
    
    footprints = []
    crs = None
    total = len(raster_paths)
    
    logger.info(f"Computing footprints for {total} rasters (fast_mode={fast_mode})...")
    
    for i, raster_path in enumerate(raster_paths):
        # Progress logging
        if (i + 1) % progress_interval == 0 or (i + 1) == total:
            logger.info(f"  Processing raster {i+1}/{total}: {Path(raster_path).name}")
        
        if fast_mode:
            result = get_raster_bounds_footprint(Path(raster_path))
        else:
            result = get_raster_valid_footprint(Path(raster_path), check_nodata=True)
        
        if result is not None:
            footprint, raster_crs = result
            footprints.append(footprint)
            if crs is None:
                crs = raster_crs
    
    if not footprints:
        logger.error("No valid footprints computed from rasters")
        return None
    
    logger.info(f"Merging {len(footprints)} footprints...")
    
    # Union all footprints - this merges adjacent tiles seamlessly
    merged = unary_union(footprints)
    
    logger.info(f"Merged footprint: {merged.geom_type}, area={merged.area/1e6:.2f} km²")
    
    # Apply negative buffer (erosion) to outer edges
    if edge_buffer_m > 0:
        buffered = merged.buffer(-edge_buffer_m)
        
        if buffered.is_empty:
            logger.warning(f"Edge buffer of {edge_buffer_m}m eliminated all coverage!")
            return merged, crs
        
        logger.info(f"Applied {edge_buffer_m}m edge buffer, area now {buffered.area/1e6:.2f} km²")
        return buffered, crs
    
    return merged, crs


def clip_shoreline_to_footprint(
    shoreline: 'shapely.geometry.LineString',
    footprint: 'shapely.geometry.base.BaseGeometry',
    min_segment_length_m: float = 100.0
) -> 'shapely.geometry.base.BaseGeometry':
    """
    Clip a shoreline to the valid imagery footprint.
    
    Returns intersection of shoreline with footprint, which may be:
    - LineString (if continuous coverage)
    - MultiLineString (if gaps exist)
    - Empty geometry (if no overlap)
    
    Args:
        shoreline: Input shoreline geometry
        footprint: Valid imagery footprint polygon
        min_segment_length_m: Drop segments shorter than this (default 100m)
        
    Returns:
        Clipped shoreline geometry (LineString or MultiLineString)
    """
    from shapely.geometry import LineString, MultiLineString
    
    if shoreline is None or shoreline.is_empty:
        logger.warning("Input shoreline is empty")
        return shoreline
    
    if footprint is None or footprint.is_empty:
        logger.warning("Footprint is empty, returning original shoreline")
        return shoreline
    
    # Perform intersection
    clipped = shoreline.intersection(footprint)
    
    if clipped.is_empty:
        logger.warning("Shoreline does not intersect footprint!")
        return clipped
    
    # Filter out short segments if result is MultiLineString
    if isinstance(clipped, MultiLineString):
        valid_segments = [
            geom for geom in clipped.geoms 
            if geom.length >= min_segment_length_m
        ]
        
        dropped = len(clipped.geoms) - len(valid_segments)
        if dropped > 0:
            logger.info(f"Dropped {dropped} segments shorter than {min_segment_length_m}m")
        
        if len(valid_segments) == 0:
            logger.warning("All segments dropped due to minimum length filter!")
            return MultiLineString([])
        elif len(valid_segments) == 1:
            clipped = valid_segments[0]
        else:
            clipped = MultiLineString(valid_segments)
    
    # Log results
    if isinstance(clipped, MultiLineString):
        total_length = sum(g.length for g in clipped.geoms)
        logger.info(f"Clipped to {len(clipped.geoms)} segments, total length {total_length:.0f}m")
    else:
        logger.info(f"Clipped shoreline: {clipped.length:.0f}m")
    
    return clipped


def analyze_coverage_gaps(
    shoreline: 'shapely.geometry.LineString',
    footprint: 'shapely.geometry.base.BaseGeometry'
) -> dict:
    """
    Analyze gaps between shoreline and imagery coverage.
    """
    from shapely.geometry import MultiLineString
    
    outside = shoreline.difference(footprint)
    
    if outside.is_empty:
        return {
            'has_gaps': False,
            'gap_count': 0,
            'total_gap_length_m': 0,
            'gaps': []
        }
    
    gaps = []
    if isinstance(outside, MultiLineString):
        for i, geom in enumerate(outside.geoms):
            gaps.append({
                'index': i,
                'length_m': geom.length,
                'start_coord': list(geom.coords[0]),
                'end_coord': list(geom.coords[-1])
            })
    else:
        gaps.append({
            'index': 0,
            'length_m': outside.length,
            'start_coord': list(outside.coords[0]),
            'end_coord': list(outside.coords[-1])
        })
    
    return {
        'has_gaps': True,
        'gap_count': len(gaps),
        'total_gap_length_m': sum(g['length_m'] for g in gaps),
        'gaps': gaps
    }


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == '__main__':
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if len(sys.argv) < 4:
        print("Usage: python footprint_clip.py <shellline.geojson> <raster_dir> <output.geojson> [buffer_m] [--slow]")
        print("\nOptions:")
        print("  buffer_m   Edge buffer in meters (default: 25)")
        print("  --slow     Use per-pixel nodata detection instead of fast bounds mode")
        print("\nExample:")
        print("  python footprint_clip.py shellline_1995.geojson imagery/19950102/ shellline_1995_clipped.geojson 25")
        sys.exit(1)
    
    import geopandas as gpd
    
    shellline_path = Path(sys.argv[1])
    raster_dir = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    buffer_m = float(sys.argv[4]) if len(sys.argv) > 4 and not sys.argv[4].startswith('--') else 25.0
    fast_mode = '--slow' not in sys.argv
    
    # Find rasters
    raster_paths = list(raster_dir.glob('*.tif')) + list(raster_dir.glob('*.tiff'))
    print(f"Found {len(raster_paths)} rasters")
    
    # Load shellline
    gdf = gpd.read_file(shellline_path)
    original_geom = gdf.geometry.iloc[0]
    
    # Compute footprint
    result = compute_merged_footprint(raster_paths, edge_buffer_m=buffer_m, fast_mode=fast_mode)
    
    if result is None:
        print("ERROR: Failed to compute footprint")
        sys.exit(1)
    
    footprint, footprint_crs = result
    
    # Handle CRS mismatch
    if gdf.crs != footprint_crs:
        print(f"Reprojecting footprint from {footprint_crs} to {gdf.crs}...")
        import pyproj
        from shapely.ops import transform
        transformer = pyproj.Transformer.from_crs(footprint_crs, gdf.crs, always_xy=True)
        footprint = transform(transformer.transform, footprint)
    
    # Clip
    clipped = clip_shoreline_to_footprint(original_geom, footprint)
    
    # Save
    from shapely.geometry import MultiLineString
    output_data = gdf.iloc[0].to_dict()
    output_data['geometry'] = clipped
    output_data['original_length_m'] = original_geom.length
    output_data['clipped_length_m'] = clipped.length if not clipped.is_empty else 0
    
    if isinstance(clipped, MultiLineString):
        output_data['num_segments'] = len(clipped.geoms)
    else:
        output_data['num_segments'] = 1 if not clipped.is_empty else 0
    
    out_gdf = gpd.GeoDataFrame([output_data], crs=gdf.crs)
    out_gdf.to_file(output_path, driver='GeoJSON')
    
    print(f"\n=== Results ===")
    print(f"Original length: {original_geom.length/1000:.2f} km")
    print(f"Clipped length:  {output_data['clipped_length_m']/1000:.2f} km")
    print(f"Segments:        {output_data['num_segments']}")
    print(f"Mode:            {'fast (bounds)' if fast_mode else 'slow (per-pixel)'}")
    print(f"Saved to:        {output_path}")