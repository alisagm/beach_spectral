"""
Export utilities for spectral transect analysis results.

Handles CSV and JSON export of transect analysis results,
processing statistics, and summary reports.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def export_results_to_csv(
    all_results: List[Dict],
    output_dir: Path
) -> Path:
    """
    Export all transect results to CSV file.

    Args:
        all_results: List of transect analysis results, each containing:
            - 'transect_id': Transect identifier
            - 'data': DataFrame with spectral values
            - 'landcover': DataFrame with classification results
        output_dir: Directory to save CSV

    Returns:
        Path to saved CSV file
    """
    logger.debug("Exporting results to CSV")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = []

    for result in all_results:
        transect_id = result['transect_id']
        data = result['data']
        landcover = result['landcover']

        # Merge data and classification
        merged = data.copy()
        merged['predicted_class'] = landcover['predicted_class'].values
        merged['confidence'] = landcover['confidence'].values

        # Add transition flags
        if 'transition_flag' in landcover.columns:
            merged['transition_flag'] = landcover['transition_flag'].values
        else:
            merged['transition_flag'] = False

        all_data.append(merged)

    # Concatenate all transects
    combined = pd.concat(all_data, ignore_index=True)

    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'transect_analysis_{timestamp}.csv'

    combined.to_csv(output_path, index=False)

    logger.info(f"Exported {len(combined)} points to {output_path}")

    return output_path


def export_summary_json(
    all_results: List[Dict],
    output_dir: Path,
    processing_metadata: Optional[Dict] = None
) -> Path:
    """
    Export summary statistics as JSON.

    Args:
        all_results: List of transect analysis results
        output_dir: Directory to save JSON
        processing_metadata: Optional metadata about processing configuration

    Returns:
        Path to saved JSON file
    """
    logger.debug("Exporting summary to JSON")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_transects': len(all_results),
            'processing_info': processing_metadata or {}
        },
        'transects': []
    }

    for result in all_results:
        transect_summary = {
            'transect_id': str(result['transect_id']),
            'num_points': len(result['data']),
            'length_m': float(result['data']['distance'].max()),
            'class_distribution': result['landcover']['predicted_class'].value_counts().to_dict(),
            'transitions': [
                {
                    'distance': float(t['distance']),
                    'type': t['type'],
                    'confidence': float(t['confidence'])
                }
                for t in result.get('transitions', [])
            ],
            'num_transitions': len(result.get('transitions', []))
        }

        summary['transects'].append(transect_summary)

    # Overall statistics
    all_classes = pd.concat([
        r['landcover']['predicted_class']
        for r in all_results
    ])
    summary['overall_class_distribution'] = all_classes.value_counts().to_dict()

    total_transitions = sum(len(r.get('transitions', [])) for r in all_results)
    summary['total_transitions'] = total_transitions

    # Save to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'summary_{timestamp}.json'

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Exported summary to {output_path}")

    return output_path


def calculate_processing_stats(all_results: List[Dict]) -> Dict:
    """
    Calculate processing statistics across all transects.

    Args:
        all_results: List of transect analysis results

    Returns:
        Dictionary with processing statistics including:
            - num_transects: Total transects processed
            - total_sample_points: Total points sampled
            - total_length_m: Combined transect length
            - avg_points_per_transect: Mean points per transect
            - avg_length_per_transect_m: Mean transect length
            - total_transitions: Total boundaries detected
            - classification_confidence: Confidence statistics
    """
    total_points = sum(len(r['data']) for r in all_results)
    total_length = sum(r['data']['distance'].max() for r in all_results)
    total_transitions = sum(len(r.get('transitions', [])) for r in all_results)

    # Confidence statistics
    all_confidences = []
    for r in all_results:
        if 'confidence' in r['landcover'].columns:
            all_confidences.extend(r['landcover']['confidence'].values)

    stats = {
        'num_transects': len(all_results),
        'total_sample_points': total_points,
        'total_length_m': total_length,
        'avg_points_per_transect': total_points / len(all_results),
        'avg_length_per_transect_m': total_length / len(all_results),
        'total_transitions': total_transitions,
        'avg_transitions_per_transect': total_transitions / len(all_results),
        'classification_confidence': {
            'mean': float(np.mean(all_confidences)),
            'std': float(np.std(all_confidences)),
            'min': float(np.min(all_confidences)),
            'max': float(np.max(all_confidences))
        } if all_confidences else None
    }

    return stats


def print_processing_summary(stats: Dict) -> None:
    """
    Print formatted processing summary to console.

    Args:
        stats: Processing statistics dictionary from calculate_processing_stats()
    """
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Number of transects processed: {stats['num_transects']}")
    print(f"Total sample points: {stats['total_sample_points']}")
    print(f"Total transect length: {stats['total_length_m']:.1f} m")
    print(f"Average points per transect: {stats['avg_points_per_transect']:.1f}")
    print(f"Average transect length: {stats['avg_length_per_transect_m']:.1f} m")
    print(f"\nTotal transitions detected: {stats['total_transitions']}")
    print(f"Average transitions per transect: {stats['avg_transitions_per_transect']:.2f}")

    if stats['classification_confidence']:
        conf = stats['classification_confidence']
        print(f"\nClassification confidence:")
        print(f"  Mean: {conf['mean']:.3f}")
        print(f"  Std:  {conf['std']:.3f}")
        print(f"  Range: [{conf['min']:.3f}, {conf['max']:.3f}]")

    print("=" * 60 + "\n")


def select_representative_transects(
    all_results: List[Dict],
    num_to_select: int = 5
) -> List[Dict]:
    """
    Select representative transects for visualization.

    Selects evenly-spaced transects from the full set to provide
    a representative sample for diagnostic plots.

    Args:
        all_results: List of all transect results
        num_to_select: Number of transects to select

    Returns:
        List of selected transect results
    """
    n_transects = len(all_results)

    if n_transects <= num_to_select:
        logger.info(f"Selecting all {n_transects} transects for visualization")
        return all_results

    # Select evenly-spaced indices
    indices = np.linspace(0, n_transects - 1, num_to_select, dtype=int)

    selected = [all_results[i] for i in indices]

    logger.info(
        f"Selected {len(selected)} representative transects: "
        f"IDs = {[r['transect_id'] for r in selected]}"
    )

    return selected


def validate_output_directory(output_dir: Path) -> Path:
    """
    Validate and create output directory if needed.

    Args:
        output_dir: Path to output directory

    Returns:
        Validated Path object

    Raises:
        ValueError: If path exists and is not a directory
    """
    output_dir = Path(output_dir)

    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"{output_dir} exists but is not a directory")

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir

# ============================================================================
# SHELL LINE GEOJSON EXPORT
# ============================================================================

def export_shell_line_geojson(
    all_results: List[Dict],
    output_dir: Path,
    year: str,
    source_crs: str,
    gap_threshold_m: float = 500.0
) -> Optional[Path]:
    """
    Export shell line positions to GeoJSON for a single year.
    
    Creates a LineString (or MultiLineString if coverage gaps exist) from
    shell line detections across all transects. Gaps smaller than the
    threshold are connected; larger gaps start new line segments.
    
    Args:
        all_results: List of transect analysis results, each containing:
            - 'transect_id': Transect identifier
            - 'data': DataFrame with x, y coordinates
            - 'transitions': List of detected transitions
        output_dir: Directory to save GeoJSON
        year: Year string for filename (e.g., '2020')
        source_crs: CRS of the source coordinates (e.g., 'EPSG:26914')
        gap_threshold_m: Maximum gap (meters) to bridge; larger gaps
            create MultiLineString segments (default: 500m)
    
    Returns:
        Path to saved GeoJSON file, or None if no shell lines detected
    """
    try:
        import geopandas as gpd
        from shapely.geometry import LineString, MultiLineString, Point
    except ImportError as e:
        logger.error(f"Missing dependency for GeoJSON export: {e}")
        return None
    
    logger.info(f"Exporting shell line GeoJSON for year {year}")
    
    # Extract shell line points from results
    shell_points = _extract_shell_line_points(all_results)
    
    if not shell_points:
        logger.warning(f"No shell lines detected for year {year} - skipping GeoJSON export")
        return None
    
    logger.debug(f"Found {len(shell_points)} shell line points")
    
    # Sort by transect ID to ensure proper ordering along shore
    shell_points.sort(key=lambda p: p['transect_id'])
    
    # Build geometry with gap handling
    geometry = _build_shell_line_geometry(shell_points, gap_threshold_m)
    
    if geometry is None or geometry.is_empty:
        logger.warning(f"Could not build shell line geometry for year {year}")
        return None
    
    # Create GeoDataFrame in source CRS
    gdf = gpd.GeoDataFrame(
        [{'year': year, 'num_points': len(shell_points)}],
        geometry=[geometry],
        crs=source_crs
    )
    
    # Transform to EPSG:4326 for output
    gdf_wgs84 = gdf.to_crs('EPSG:4326')
    
    # Save to GeoJSON
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'shellline_{year}.geojson'
    
    gdf_wgs84.to_file(output_path, driver='GeoJSON')
    
    # Log geometry type for diagnostics
    geom_type = gdf_wgs84.geometry.iloc[0].geom_type
    if geom_type == 'MultiLineString':
        num_parts = len(gdf_wgs84.geometry.iloc[0].geoms)
        logger.info(f"Exported shell line to {output_path} ({num_parts} segments due to coverage gaps)")
    else:
        logger.info(f"Exported shell line to {output_path} (continuous)")
    
    return output_path


def _extract_shell_line_points(all_results: List[Dict]) -> List[Dict]:
    """
    Extract shell line point coordinates from analysis results.
    
    Filters transitions to find dry_wet boundaries (shell lines) and
    retrieves their coordinates from the spectral data.
    
    Args:
        all_results: List of transect analysis results
        
    Returns:
        List of dicts with 'transect_id', 'x', 'y', 'distance', 'confidence'
    """
    shell_points = []
    
    for result in all_results:
        transect_id = result['transect_id']
        data = result['data']
        transitions = result.get('transitions', [])
        
        # Find shell line transitions (dry_wet type)
        for t in transitions:
            t_type = t.get('type', '')
            if 'dry_wet' not in t_type:
                continue
            
            # Get coordinates from data using index
            idx = t.get('index')
            if idx is None or idx >= len(data):
                logger.warning(f"Invalid transition index {idx} for transect {transect_id}")
                continue
            
            # Check if coordinates are available
            if 'x' not in data.columns or 'y' not in data.columns:
                logger.error(f"Coordinates not found in data for transect {transect_id}")
                continue
            
            row = data.iloc[idx]
            
            shell_points.append({
                'transect_id': transect_id,
                'x': row['x'],
                'y': row['y'],
                'distance': t.get('distance', row.get('distance', 0)),
                'confidence': t.get('confidence', 0)
            })
    
    return shell_points


def _build_shell_line_geometry(
    shell_points: List[Dict],
    gap_threshold_m: float
):
    """
    Build LineString or MultiLineString from shell line points.
    
    Consecutive points closer than gap_threshold are connected.
    Gaps larger than the threshold start new line segments.
    
    Args:
        shell_points: List of point dicts with 'x', 'y' (sorted by transect_id)
        gap_threshold_m: Maximum distance to bridge between consecutive points
        
    Returns:
        LineString if continuous, MultiLineString if gaps exist, None if < 2 points
    """
    from shapely.geometry import LineString, MultiLineString
    
    if len(shell_points) < 2:
        return None
    
    # Build segments, splitting at gaps
    segments = []
    current_segment = [shell_points[0]]
    
    for i in range(1, len(shell_points)):
        prev = shell_points[i - 1]
        curr = shell_points[i]
        
        # Calculate distance between consecutive points
        dx = curr['x'] - prev['x']
        dy = curr['y'] - prev['y']
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > gap_threshold_m:
            # Gap detected - save current segment if valid
            if len(current_segment) >= 2:
                coords = [(p['x'], p['y']) for p in current_segment]
                segments.append(LineString(coords))
            # Start new segment
            current_segment = [curr]
        else:
            # Continue current segment
            current_segment.append(curr)
    
    # Don't forget the last segment
    if len(current_segment) >= 2:
        coords = [(p['x'], p['y']) for p in current_segment]
        segments.append(LineString(coords))
    
    # Return appropriate geometry type
    if not segments:
        return None
    elif len(segments) == 1:
        return segments[0]
    else:
        return MultiLineString(segments)


def bundle_shell_lines_to_gpkg(
    output_root: Path,
    years: List[str],
    output_filename: str = 'shelllines_all_years.gpkg',
    driver: str = 'GPKG',
) -> Optional[Path]:
    """
    Bundle multiple years' shell line GeoJSONs into a single file.
 
    Reads individual shellline_{year}.geojson files and combines them into
    one vector file where each feature is one year's shell line geometry.
    Features are sorted chronologically and carry a 'year' attribute.
 
    Args:
        output_root:     Root output directory containing year subdirectories.
        years:           List of year strings to include, e.g. ['1995', '2004'].
        output_filename: Name of the output file.  Extension should match
                         driver (e.g. '.gpkg' for GPKG, '.geojson' for GeoJSON).
        driver:          Fiona driver string.  'GPKG' (default) or 'GeoJSON'.
 
    Returns:
        Path to the saved file, or None if no valid per-year GeoJSONs were found.
    """
    try:
        import geopandas as gpd
    except ImportError as e:
        logger.error(f"Missing geopandas for bundle export: {e}")
        return None
 
    logger.info(f"Bundling shell lines for {len(years)} year(s): {sorted(years)}")
 
    output_root = Path(output_root)
    gdfs = []
 
    for year in sorted(years):          # chronological order
        geojson_path = output_root / year / f'shellline_{year}.geojson'
 
        if not geojson_path.exists():
            logger.warning(f"Shell line not found for year {year}: {geojson_path}")
            continue
 
        try:
            gdf = gpd.read_file(geojson_path)
            # Stamp year on every feature (export_shell_line_geojson already
            # writes a 'year' column, but re-stamping is harmless and ensures
            # the value is correct even if the file was edited externally).
            gdf['year'] = year
            gdfs.append(gdf)
            logger.debug(f"  Loaded {geojson_path} ({len(gdf)} feature(s))")
        except Exception as e:
            logger.error(f"Failed to load shell line for year {year}: {e}")
            continue
 
    if not gdfs:
        logger.warning("No valid shell line GeoJSONs found — skipping bundle")
        return None
 
    combined = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        crs=gdfs[0].crs,
    )
 
    output_path = output_root / output_filename
    combined.to_file(output_path, driver=driver)
 
    logger.info(
        f"Bundled {len(gdfs)}/{len(years)} year(s) → {output_path} "
        f"({len(combined)} feature(s), driver={driver})"
    )
 
    return output_path

def bundle_shell_lines_to_geojson(
    output_root: Path,
    years: List[str],
    output_filename: str = 'shelllines_all_years.geojson',
) -> Optional[Path]:
    """
    Bundle multiple years' shell lines into a single GeoJSON file.
 
    Convenience wrapper around bundle_shell_lines_to_gpkg with driver='GeoJSON'.
    Each feature represents one year's shell line geometry and carries a
    'year' attribute.
 
    Args:
        output_root:     Root output directory containing year subdirectories.
        years:           List of year strings to include.
        output_filename: Output filename (default: 'shelllines_all_years.geojson').
 
    Returns:
        Path to the saved GeoJSON, or None if no shell lines were found.
    """
    return bundle_shell_lines_to_gpkg(
        output_root=output_root,
        years=years,
        output_filename=output_filename,
        driver='GeoJSON',
    )