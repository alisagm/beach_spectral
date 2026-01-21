"""
Footprint-based shoreline clipping module.

Clips shorelines to valid imagery coverage, with optional edge buffering.
Handles nodata regions (black=0 or white=255) and creates merged footprints
from multiple rasters.

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


def get_raster_valid_footprint(
    raster_path: Path,
    nodata_threshold: int = 5,
    band_check: str = 'all'
) -> Optional['shapely.geometry.Polygon']:
    """
    Compute the valid data footprint for a single raster.
    
    Identifies nodata regions as pixels where all bands are either:
    - Near black (all values < nodata_threshold)
    - Near white (all values > 255 - nodata_threshold for 8-bit)
    
    Args:
        raster_path: Path to raster file
        nodata_threshold: Tolerance for black/white detection (default 5)
        band_check: 'all' requires all bands to be nodata, 'any' requires any band
        
    Returns:
        Polygon of valid data extent, or None if no valid data
    """
    import rasterio
    from rasterio import features
    from shapely.geometry import shape, MultiPolygon
    from shapely.ops import unary_union
    
    try:
        with rasterio.open(raster_path) as src:
            # Read all bands
            data = src.read()  # Shape: (bands, height, width)
            transform = src.transform
            crs = src.crs
            
            num_bands = data.shape[0]
            
            # Determine bit depth from dtype
            if data.dtype == np.uint8:
                max_val = 255
            elif data.dtype == np.uint16:
                max_val = 65535
            else:
                max_val = 255  # Assume 8-bit
            
            # Create mask: True = valid data, False = nodata
            # Nodata is where ALL bands are near black OR ALL bands are near white
            near_black = np.all(data < nodata_threshold, axis=0)
            near_white = np.all(data > (max_val - nodata_threshold), axis=0)
            nodata_mask = near_black | near_white
            valid_mask = ~nodata_mask
            
            # If no valid data, return None
            if not np.any(valid_mask):
                logger.warning(f"No valid data in {raster_path.name}")
                return None
            
            # Convert mask to uint8 for vectorization
            valid_uint8 = valid_mask.astype(np.uint8)
            
            # Vectorize the valid data regions
            shapes_gen = features.shapes(
                valid_uint8,
                mask=valid_mask,
                transform=transform
            )
            
            # Collect all valid polygons (value == 1)
            polygons = []
            for geom, value in shapes_gen:
                if value == 1:
                    polygons.append(shape(geom))
            
            if not polygons:
                logger.warning(f"No polygons extracted from {raster_path.name}")
                return None
            
            # Union all polygons into single geometry
            if len(polygons) == 1:
                footprint = polygons[0]
            else:
                footprint = unary_union(polygons)
            
            # Simplify slightly to reduce vertex count (tolerance in CRS units)
            footprint = footprint.simplify(1.0, preserve_topology=True)
            
            logger.debug(f"Footprint for {raster_path.name}: {footprint.geom_type}, area={footprint.area:.0f}")
            
            return footprint, crs
            
    except Exception as e:
        logger.error(f"Error processing {raster_path}: {e}")
        return None


def compute_merged_footprint(
    raster_paths: List[Path],
    edge_buffer_m: float = 25.0,
    nodata_threshold: int = 5
) -> Optional[Tuple['shapely.geometry.base.BaseGeometry', any]]:
    """
    Compute merged valid-data footprint from multiple rasters.
    
    Creates a single (multi)polygon representing the union of all raster
    valid-data extents, then applies an inward buffer to the OUTER edges only.
    
    Args:
        raster_paths: List of paths to raster files
        edge_buffer_m: Inward buffer distance in meters (default 25)
        nodata_threshold: Tolerance for nodata detection
        
    Returns:
        Tuple of (buffered_footprint, crs) or None if no valid rasters
    """
    from shapely.ops import unary_union
    
    footprints = []
    crs = None
    
    for raster_path in raster_paths:
        result = get_raster_valid_footprint(
            Path(raster_path),
            nodata_threshold=nodata_threshold
        )
        if result is not None:
            footprint, raster_crs = result
            footprints.append(footprint)
            if crs is None:
                crs = raster_crs
    
    if not footprints:
        logger.error("No valid footprints computed from rasters")
        return None
    
    logger.info(f"Merging {len(footprints)} raster footprints...")
    
    # Union all footprints - this merges adjacent tiles seamlessly
    merged = unary_union(footprints)
    
    logger.info(f"Merged footprint: {merged.geom_type}, area={merged.area/1e6:.2f} km²")
    
    # Apply negative buffer (erosion) to outer edges
    if edge_buffer_m > 0:
        buffered = merged.buffer(-edge_buffer_m)
        
        # buffer() can return empty geometry if buffer is too large
        if buffered.is_empty:
            logger.warning(f"Edge buffer of {edge_buffer_m}m eliminated all coverage!")
            return merged, crs  # Return unbuffered
        
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
        # Keep only segments longer than minimum
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
    
    Returns statistics about where the shoreline falls outside coverage.
    Useful for diagnostics.
    
    Args:
        shoreline: Input shoreline geometry
        footprint: Valid imagery footprint polygon
        
    Returns:
        Dictionary with gap statistics
    """
    from shapely.geometry import MultiLineString
    
    # Find parts of shoreline outside footprint
    outside = shoreline.difference(footprint)
    
    if outside.is_empty:
        return {
            'has_gaps': False,
            'gap_count': 0,
            'total_gap_length_m': 0,
            'gaps': []
        }
    
    # Collect gap info
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
# Convenience function for batch processing
# =============================================================================

def clip_shellline_to_year_coverage(
    shellline_geojson: Path,
    raster_dir: Path,
    output_geojson: Path,
    edge_buffer_m: float = 25.0,
    min_segment_length_m: float = 100.0,
    raster_extensions: tuple = ('.tif', '.tiff')
) -> dict:
    """
    Clip an existing shellline GeoJSON to the coverage of rasters in a directory.
    
    Main entry point for post-processing existing shellline outputs.
    
    Args:
        shellline_geojson: Path to input shellline GeoJSON
        raster_dir: Directory containing rasters for this year
        output_geojson: Path for clipped output
        edge_buffer_m: Inward buffer from imagery edges
        min_segment_length_m: Drop segments shorter than this
        raster_extensions: File extensions to look for
        
    Returns:
        Dictionary with processing statistics
    """
    import geopandas as gpd
    from shapely.geometry import mapping
    
    # Find rasters
    raster_paths = []
    for ext in raster_extensions:
        raster_paths.extend(raster_dir.glob(f'*{ext}'))
    
    if not raster_paths:
        raise ValueError(f"No rasters found in {raster_dir}")
    
    logger.info(f"Found {len(raster_paths)} rasters in {raster_dir}")
    
    # Load shellline
    gdf = gpd.read_file(shellline_geojson)
    if len(gdf) == 0:
        raise ValueError(f"No features in {shellline_geojson}")
    
    original_geom = gdf.geometry.iloc[0]
    original_length = original_geom.length
    
    # Compute merged footprint
    result = compute_merged_footprint(
        raster_paths,
        edge_buffer_m=edge_buffer_m
    )
    
    if result is None:
        raise ValueError("Failed to compute imagery footprint")
    
    footprint, footprint_crs = result
    
    # Check CRS compatibility
    if gdf.crs != footprint_crs:
        logger.warning(f"CRS mismatch: shellline={gdf.crs}, footprint={footprint_crs}")
        # Could reproject here if needed
    
    # Analyze gaps before clipping
    gap_analysis = analyze_coverage_gaps(original_geom, footprint)
    
    # Clip shoreline
    clipped = clip_shoreline_to_footprint(
        original_geom,
        footprint,
        min_segment_length_m=min_segment_length_m
    )
    
    # Prepare output
    output_data = gdf.iloc[0].to_dict()
    output_data['geometry'] = clipped
    output_data['original_length_m'] = original_length
    output_data['clipped_length_m'] = clipped.length if not clipped.is_empty else 0
    output_data['edge_buffer_m'] = edge_buffer_m
    
    # Count segments
    from shapely.geometry import MultiLineString
    if isinstance(clipped, MultiLineString):
        output_data['num_segments'] = len(clipped.geoms)
    else:
        output_data['num_segments'] = 1 if not clipped.is_empty else 0
    
    # Create output GeoDataFrame
    out_gdf = gpd.GeoDataFrame([output_data], crs=gdf.crs)
    out_gdf.to_file(output_geojson, driver='GeoJSON')
    
    logger.info(f"Saved clipped shellline to {output_geojson}")
    
    return {
        'original_length_m': original_length,
        'clipped_length_m': clipped.length if not clipped.is_empty else 0,
        'num_segments': output_data['num_segments'],
        'num_rasters': len(raster_paths),
        'edge_buffer_m': edge_buffer_m,
        'gap_analysis': gap_analysis
    }


# =============================================================================
# Testing / Demo
# =============================================================================

if __name__ == '__main__':
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if len(sys.argv) < 4:
        print("Usage: python footprint_clip.py <shellline.geojson> <raster_dir> <output.geojson> [buffer_m]")
        print("\nExample:")
        print("  python footprint_clip.py shellline_1995.geojson imagery/19950102/ shellline_1995_clipped.geojson 25")
        sys.exit(1)
    
    shellline_path = Path(sys.argv[1])
    raster_dir = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    buffer_m = float(sys.argv[4]) if len(sys.argv) > 4 else 25.0
    
    result = clip_shellline_to_year_coverage(
        shellline_path,
        raster_dir,
        output_path,
        edge_buffer_m=buffer_m
    )
    
    print("\n=== Results ===")
    print(f"Original length: {result['original_length_m']/1000:.2f} km")
    print(f"Clipped length:  {result['clipped_length_m']/1000:.2f} km")
    print(f"Segments:        {result['num_segments']}")
    print(f"Rasters used:    {result['num_rasters']}")
    
    if result['gap_analysis']['has_gaps']:
        print(f"\nGaps outside coverage:")
        for gap in result['gap_analysis']['gaps']:
            print(f"  Gap {gap['index']}: {gap['length_m']:.0f}m")