# compare_manual_vs_algorithm.py
"""
Validate algorithmic shoreline detection against manually digitized shorelines.

Methodology:
    For each transect, find where it intersects both the manual and algorithmic
    shorelines, then measure the along-transect distance between intersection points.
    
Sign convention:
    Positive distance = algorithm detected shellline LANDWARD of manual
    Negative distance = algorithm detected shellline SEAWARD of manual

Outputs:
    - Per-year statistics (mean, std, median, directional bias)
    - CSV with per-transect comparison data
    - Diagnostic spectral plots for best/worst/representative transects
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import nearest_points
import matplotlib.pyplot as plt


# === CONFIGURATION ===

# Update these paths for your system
MANUAL_SHORELINE_DIR = Path(r"C:\Users\alisa\Desktop\SIP\git\beach_spectral\PAIS_shorelines\features")
MANUAL_SHORELINE_BASE = "PAIS_shoreline"  # Will look for .shp file

ALGORITHM_OUTPUT_ROOT = Path(r"C:\Users\alisa\Desktop\SIP\git\beach_spectral\OUTPUT")

TRANSECT_FILE = Path(r"C:\Users\alisa\Desktop\SIP\git\beach_spectral\INPUT\shorelineTransPais.json")

OUTPUT_DIR = Path(r"C:\Users\alisa\Desktop\SIP\git\beach_spectral\validation\manual_comparison")

# Years to compare (must exist in both manual and algorithm outputs)
YEARS = ["2022"] 

# Maximum distance to consider a valid intersection (meters)
MAX_INTERSECTION_DISTANCE = 50.0  # If shoreline doesn't cross transect within this buffer, skip

# Maximum time gap (minutes) between CSV files to be considered part of the same run
MAX_CSV_GROUP_GAP_MINUTES = 20

# Transect to use for diagnostic plots
DIAGNOSTIC_TRANSECT_ID = 1000


# === HELPER FUNCTIONS ===

def count_vertices(geom) -> int:
    """
    Count total vertices in a LineString or MultiLineString.
    
    Args:
        geom: LineString or MultiLineString geometry
        
    Returns:
        Total number of coordinate vertices
    """
    if geom is None or geom.is_empty:
        return 0
    
    if isinstance(geom, MultiLineString):
        return sum(len(part.coords) for part in geom.geoms)
    elif isinstance(geom, LineString):
        return len(geom.coords)
    else:
        return 0


def count_segments(geom) -> int:
    """
    Count number of segments (parts) in a geometry.
    
    Returns:
        1 for LineString, N for MultiLineString with N parts, 0 for empty/invalid
    """
    if geom is None or geom.is_empty:
        return 0
    
    if isinstance(geom, MultiLineString):
        return len(geom.geoms)
    elif isinstance(geom, LineString):
        return 1
    else:
        return 0


def load_manual_shorelines(shapefile_dir: Path, base_name: str) -> gpd.GeoDataFrame:
    """
    Load manual shorelines from shapefile.
    
    Expected structure:
        - Shapefile with LineString geometries
        - 'Year' field containing year as integer or string
    
    Returns:
        GeoDataFrame with columns: geometry, Year
    """
    shp_path = shapefile_dir / f"{base_name}.shp"
    
    if not shp_path.exists():
        raise FileNotFoundError(f"Manual shoreline shapefile not found: {shp_path}")
    
    gdf = gpd.read_file(shp_path)
    
    # Ensure Year column exists
    if 'Year' not in gdf.columns:
        raise ValueError(f"'Year' column not found in shapefile. Columns: {gdf.columns.tolist()}")
    
    # Convert Year to string for consistent matching
    # Handle case where Year is stored as float (e.g., 1995.0 -> "1995")
    gdf['Year'] = gdf['Year'].astype(float).astype(int).astype(str)
    
    print(f"Loaded manual shorelines: {len(gdf)} features")
    print(f"  Years available: {sorted(gdf['Year'].unique())}")
    print(f"  CRS: {gdf.crs}")
    
    return gdf


def load_algorithm_shoreline(output_root: Path, year: str) -> Optional[gpd.GeoDataFrame]:
    """
    Load algorithmic shellline GeoJSON for a specific year.
    
    Expected path: OUTPUT/{year}/shellline_{year}.geojson
    
    Returns:
        GeoDataFrame with single row, or None if not found
    """
    geojson_path = output_root / year / f"shellline_{year}.geojson"
    
    if not geojson_path.exists():
        print(f"  Warning: Algorithm output not found for {year}: {geojson_path}")
        return None
    
    gdf = gpd.read_file(geojson_path)
    
    if gdf.empty:
        print(f"  Warning: Empty algorithm output for {year}")
        return None
    
    return gdf


def find_results_csv_group(output_root: Path, year: str, 
                           max_gap_minutes: int = MAX_CSV_GROUP_GAP_MINUTES) -> List[Path]:
    """
    Find all transect_analysis CSV files from the most recent run.
    
    Files are grouped by timestamp proximity: consecutive files (when sorted by time)
    within max_gap_minutes of each other are considered part of the same run.
    This handles MultiLineString outputs where segments are processed sequentially.
    
    Args:
        output_root: Root output directory
        year: Year string
        max_gap_minutes: Maximum time gap between consecutive files to be considered same run
        
    Returns:
        List of Paths to CSV files from the most recent run (sorted chronologically),
        or empty list if no matching files found
    """
    year_dir = output_root / year
    
    if not year_dir.exists():
        return []
    
    # Find all matching CSV files with their timestamps
    pattern = re.compile(r'transect_analysis_(\d{8})_(\d{6})\.csv')
    
    files_with_timestamps = []
    for f in year_dir.glob('transect_analysis_*.csv'):
        match = pattern.match(f.name)
        if match:
            date_str = match.group(1)  # YYYYMMDD
            time_str = match.group(2)  # HHMMSS
            # Parse to datetime for proper comparison
            dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            files_with_timestamps.append((dt, f))
    
    if not files_with_timestamps:
        return []
    
    # Sort by timestamp (oldest to newest)
    files_with_timestamps.sort(key=lambda x: x[0])
    
    # Group consecutive files that are within max_gap_minutes of each other
    groups: List[List[Tuple[datetime, Path]]] = []
    current_group = [files_with_timestamps[0]]
    
    for i in range(1, len(files_with_timestamps)):
        current_dt, current_file = files_with_timestamps[i]
        prev_dt, _ = current_group[-1]
        
        gap_minutes = (current_dt - prev_dt).total_seconds() / 60
        
        if gap_minutes <= max_gap_minutes:
            # Same group - add to current
            current_group.append((current_dt, current_file))
        else:
            # New group - save current and start fresh
            groups.append(current_group)
            current_group = [(current_dt, current_file)]
    
    # Don't forget the last group
    groups.append(current_group)
    
    # Select the most recent group (last one since we sorted oldest to newest)
    most_recent_group = groups[-1]
    
    # Extract just the paths (already in chronological order)
    result = [f for _, f in most_recent_group]
    
    # Log what we found
    if len(result) == 1:
        print(f"  Found 1 analysis file: {result[0].name}")
    else:
        print(f"  Found {len(result)} analysis files from same run:")
        for f in result:
            print(f"    - {f.name}")
    
    return result


def load_combined_results(csv_paths: List[Path]) -> Optional[pd.DataFrame]:
    """
    Load and combine multiple transect analysis CSV files into a single DataFrame.
    
    Args:
        csv_paths: List of paths to CSV files
        
    Returns:
        Combined DataFrame with all transects, or None if no valid data loaded
    """
    if not csv_paths:
        return None
    
    dfs = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            # Track source file for debugging
            df['_source_file'] = path.name
            dfs.append(df)
        except Exception as e:
            print(f"    Warning: Could not load {path.name}: {e}")
    
    if not dfs:
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Check for duplicate transects (shouldn't happen, but warn if it does)
    if 'TransectID' in combined.columns:
        duplicates = combined['TransectID'].duplicated().sum()
        if duplicates > 0:
            print(f"    Warning: Found {duplicates} duplicate TransectID entries across CSV files")
            # Show which files have overlapping transects
            dup_tids = combined[combined['TransectID'].duplicated(keep=False)]['TransectID'].unique()
            print(f"    Duplicate TransectIDs: {sorted(dup_tids)[:10]}{'...' if len(dup_tids) > 10 else ''}")
    
    return combined


# Legacy function for backwards compatibility
def find_most_recent_results_csv(output_root: Path, year: str) -> Optional[Path]:
    """
    Find the most recent transect_analysis CSV file for a given year.
    
    DEPRECATED: Use find_results_csv_group() for MultiLineString support.
    
    Returns:
        Path to most recent CSV, or None if no matching files found
    """
    csv_group = find_results_csv_group(output_root, year)
    if csv_group:
        return csv_group[-1]  # Return the last (most recent) file
    return None


def load_transects(transect_path: Path) -> gpd.GeoDataFrame:
    """
    Load transects from GeoJSON.
    
    Expected structure:
        - LineString geometries
        - 'TransectID' field
    
    Returns:
        GeoDataFrame with columns: geometry, TransectID
    """
    gdf = gpd.read_file(transect_path)
    
    if 'TransectID' not in gdf.columns:
        raise ValueError(f"'TransectID' column not found. Columns: {gdf.columns.tolist()}")
    
    # Ensure TransectID is integer for sorting
    gdf['TransectID'] = gdf['TransectID'].astype(int)
    gdf = gdf.sort_values('TransectID').reset_index(drop=True)
    
    print(f"Loaded transects: {len(gdf)} features")
    print(f"  TransectID range: {gdf['TransectID'].min()} to {gdf['TransectID'].max()}")
    print(f"  CRS: {gdf.crs}")
    
    return gdf


def get_transect_direction(transect_line: LineString) -> str:
    """
    Determine transect direction based on x-coordinates.
    
    Returns:
        'east_to_west' if start is east of end (higher x)
        'west_to_east' if start is west of end (lower x)
    """
    coords = list(transect_line.coords)
    start_x = coords[0][0]
    end_x = coords[-1][0]
    
    return 'east_to_west' if start_x > end_x else 'west_to_east'


def find_intersection_distance(transect_line: LineString, 
                                shoreline: LineString,
                                max_distance: float = MAX_INTERSECTION_DISTANCE) -> Optional[float]:
    """
    Find where a transect intersects a shoreline and return distance from west end.
    
    If no direct intersection, finds nearest point on shoreline to transect.
    
    Args:
        transect_line: Cross-shore transect LineString
        shoreline: Shoreline LineString (manual or algorithm)
        max_distance: Maximum perpendicular distance to consider valid
    
    Returns:
        Distance along transect from west (land) end, or None if no valid intersection
    """
    # Handle MultiLineString (sometimes shapefiles have these)
    if isinstance(shoreline, MultiLineString):
        # Find closest sub-linestring
        min_dist = float('inf')
        closest_shore = None
        for geom in shoreline.geoms:
            dist = transect_line.distance(geom)
            if dist < min_dist:
                min_dist = dist
                closest_shore = geom
        shoreline = closest_shore
    
    # Check for direct intersection
    intersection = transect_line.intersection(shoreline)
    
    if not intersection.is_empty:
        # Get intersection point(s)
        if intersection.geom_type == 'Point':
            intersect_point = intersection
        elif intersection.geom_type == 'MultiPoint':
            # Multiple intersections - take the one closest to middle of transect
            mid_point = transect_line.interpolate(0.5, normalized=True)
            intersect_point = min(intersection.geoms, 
                                   key=lambda p: p.distance(mid_point))
        elif intersection.geom_type == 'LineString':
            # Shoreline runs along transect - take midpoint
            intersect_point = intersection.interpolate(0.5, normalized=True)
        else:
            # Complex intersection - use centroid
            intersect_point = intersection.centroid
    else:
        # No direct intersection - find nearest point on shoreline
        nearest_on_transect, nearest_on_shore = nearest_points(transect_line, shoreline)
        
        # Check if close enough to be considered valid
        perp_distance = nearest_on_transect.distance(nearest_on_shore)
        if perp_distance > max_distance:
            return None
        
        intersect_point = nearest_on_transect
    
    # Calculate distance along transect from START
    distance_from_start = transect_line.project(intersect_point)
    
    # Normalize to "distance from west"
    direction = get_transect_direction(transect_line)
    
    if direction == 'east_to_west':
        # Start is seaward (east), so distance from west = length - distance from start
        distance_from_west = transect_line.length - distance_from_start
    else:
        # Start is landward (west)
        distance_from_west = distance_from_start
    
    return distance_from_west


def compare_shorelines_for_year(transects: gpd.GeoDataFrame,
                                  manual_shoreline: LineString,
                                  algorithm_shoreline: LineString,
                                  year: str) -> pd.DataFrame:
    """
    Compare manual vs algorithm shoreline positions for all transects.
    
    Returns:
        DataFrame with columns:
            TransectID, manual_distance, algorithm_distance, difference, year
        
        difference = algorithm_distance - manual_distance
            Positive = algorithm is landward of manual
            Negative = algorithm is seaward of manual
    """
    results = []
    
    for _, row in transects.iterrows():
        tid = row['TransectID']
        transect_line = row.geometry
        
        # Get intersection distances
        manual_dist = find_intersection_distance(transect_line, manual_shoreline)
        algo_dist = find_intersection_distance(transect_line, algorithm_shoreline)
        
        # Calculate difference if both valid
        if manual_dist is not None and algo_dist is not None:
            diff = algo_dist - manual_dist
        else:
            diff = None
        
        results.append({
            'TransectID': tid,
            'manual_distance': manual_dist,
            'algorithm_distance': algo_dist,
            'difference': diff,
            'year': year
        })
    
    return pd.DataFrame(results)


def calculate_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate summary statistics for shoreline comparison.
    
    Returns dict with:
        - n_valid: Number of transects with valid comparison
        - n_total: Total transects
        - mean_diff: Mean difference (bias direction)
        - std_diff: Standard deviation
        - median_diff: Median difference
        - mean_abs_diff: Mean absolute difference
        - pct_landward: Percentage where algorithm is landward
        - pct_seaward: Percentage where algorithm is seaward
        - worst_landward_tid: TransectID with largest landward difference
        - worst_seaward_tid: TransectID with largest seaward difference
        - best_agreement_tid: TransectID with smallest absolute difference
    """
    valid = df.dropna(subset=['difference'])
    
    if len(valid) == 0:
        return {
            'n_valid': 0,
            'n_total': len(df),
            'mean_diff': None,
            'std_diff': None,
            'median_diff': None,
            'mean_abs_diff': None,
            'pct_landward': None,
            'pct_seaward': None,
            'worst_landward_tid': None,
            'worst_seaward_tid': None,
            'best_agreement_tid': None
        }
    
    diffs = valid['difference']
    
    # Find extreme cases
    worst_landward_idx = diffs.idxmax()
    worst_seaward_idx = diffs.idxmin()
    best_idx = diffs.abs().idxmin()
    
    return {
        'n_valid': len(valid),
        'n_total': len(df),
        'mean_diff': diffs.mean(),
        'std_diff': diffs.std(),
        'median_diff': diffs.median(),
        'mean_abs_diff': diffs.abs().mean(),
        'pct_landward': (diffs > 0).sum() / len(diffs) * 100,
        'pct_seaward': (diffs < 0).sum() / len(diffs) * 100,
        'worst_landward_tid': int(valid.loc[worst_landward_idx, 'TransectID']),
        'worst_landward_diff': diffs.loc[worst_landward_idx],
        'worst_seaward_tid': int(valid.loc[worst_seaward_idx, 'TransectID']),
        'worst_seaward_diff': diffs.loc[worst_seaward_idx],
        'best_agreement_tid': int(valid.loc[best_idx, 'TransectID']),
        'best_agreement_diff': diffs.loc[best_idx]
    }


def select_diagnostic_transects(df: pd.DataFrame, stats: Dict) -> List[int]:
    """
    Select transects for diagnostic spectral profile plots.
    
    Returns list of TransectIDs:
        - Best agreement
        - Worst landward difference (if algorithm detected)
        - Worst seaward difference (if algorithm detected)
        - 1-2 representative (near median) transects
    """
    selected = []
    
    # Best agreement
    if stats['best_agreement_tid'] is not None:
        selected.append(stats['best_agreement_tid'])
    
    # Worst cases (only if algorithm has detection)
    valid = df.dropna(subset=['algorithm_distance', 'difference'])
    
    if stats['worst_landward_tid'] is not None:
        tid = stats['worst_landward_tid']
        if tid in valid['TransectID'].values:
            selected.append(tid)
    
    if stats['worst_seaward_tid'] is not None:
        tid = stats['worst_seaward_tid']
        if tid in valid['TransectID'].values:
            selected.append(tid)
    
    # Representative (near median)
    if len(valid) > 0:
        median_diff = stats['median_diff']
        valid = valid.copy()
        valid['dist_from_median'] = (valid['difference'] - median_diff).abs()
        sorted_by_median = valid.sort_values('dist_from_median')
        
        # Add 1-2 near median that aren't already selected
        for _, row in sorted_by_median.iterrows():
            tid = int(row['TransectID'])
            if tid not in selected:
                selected.append(tid)
                if len(selected) >= 5:  # Cap at 5 diagnostic transects
                    break
    
    return selected


def plot_difference_histogram(df: pd.DataFrame, year: str, output_path: Path):
    """
    Create histogram of differences between manual and algorithmic shorelines.
    """
    valid = df.dropna(subset=['difference'])
    
    if len(valid) == 0:
        print(f"  No valid data for histogram ({year})")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    diffs = valid['difference']
    
    # Histogram
    bins = np.linspace(diffs.min() - 5, diffs.max() + 5, 50)
    ax.hist(diffs, bins=bins, edgecolor='black', alpha=0.7)
    
    # Add mean and median lines
    mean_diff = diffs.mean()
    median_diff = diffs.median()
    
    ax.axvline(mean_diff, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_diff:.1f}m')
    ax.axvline(median_diff, color='orange', linestyle='-', linewidth=2,
               label=f'Median: {median_diff:.1f}m')
    ax.axvline(0, color='green', linestyle=':', linewidth=2, label='Zero (perfect match)')
    
    ax.set_xlabel('Difference (Algorithm - Manual) [m]', fontsize=12)
    ax.set_ylabel('Number of Transects', fontsize=12)
    ax.set_title(f'Shoreline Position Comparison - {year}\n'
                 f'Positive = Algorithm Landward, Negative = Algorithm Seaward',
                 fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = (f'N = {len(valid)}\n'
                  f'Mean: {mean_diff:.2f}m\n'
                  f'Std: {diffs.std():.2f}m\n'
                  f'MAE: {diffs.abs().mean():.2f}m')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  Saved histogram: {output_path.name}")


def plot_difference_by_transect(df: pd.DataFrame, year: str, output_path: Path):
    """
    Create scatter plot showing difference vs TransectID (spatial pattern).
    """
    valid = df.dropna(subset=['difference'])
    
    if len(valid) == 0:
        print(f"  No valid data for scatter plot ({year})")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Color by sign of difference
    colors = ['blue' if d < 0 else 'red' for d in valid['difference']]
    
    ax.scatter(valid['TransectID'], valid['difference'], c=colors, alpha=0.5, s=10)
    
    # Add zero line
    ax.axhline(0, color='green', linestyle='--', linewidth=1.5, label='Zero')
    
    # Add rolling mean to show trend
    window = min(50, len(valid) // 10)
    if window > 1:
        rolling_mean = valid.set_index('TransectID')['difference'].rolling(window, center=True).mean()
        ax.plot(rolling_mean.index, rolling_mean.values, color='black', linewidth=2,
                label=f'Rolling mean (n={window})')
    
    ax.set_xlabel('TransectID', fontsize=12)
    ax.set_ylabel('Difference (Algorithm - Manual) [m]', fontsize=12)
    ax.set_title(f'Shoreline Position Difference by Location - {year}\n'
                 f'Red = Algorithm Landward, Blue = Algorithm Seaward',
                 fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  Saved spatial plot: {output_path.name}")


def plot_diagnostic_spectral_profile(transect_id: int,
                                       year: str,
                                       manual_distance: float,
                                       algo_distance: float,
                                       difference: float,
                                       description: str,
                                       output_dir: Path,
                                       spectral_data: Optional[pd.DataFrame] = None):
    """
    Create a spectral profile plot for a diagnostic transect.
    
    Shows:
        - Spectral values (R, G, B, NIR) along transect
        - Manual shoreline position (green vertical line)
        - Algorithm shoreline position (red vertical line)
        - NIR derivative and classification
    
    Args:
        transect_id: TransectID to plot
        year: Year string
        manual_distance: Distance from west where manual shoreline intersects
        algo_distance: Distance from west where algorithm detected shellline
        difference: algo_distance - manual_distance
        description: e.g., 'best_agreement', 'worst_landward'
        output_dir: Directory to save plot
        spectral_data: Pre-loaded combined DataFrame with spectral data for all transects.
                       If None, will attempt to load from disk (legacy behavior).
    """
    # Use provided data or load from disk
    if spectral_data is None:
        # Legacy behavior: load from most recent CSV group
        csv_paths = find_results_csv_group(ALGORITHM_OUTPUT_ROOT, year)
        if not csv_paths:
            print(f"    Warning: No spectral data found for transect {transect_id} ({year})")
            return None
        spectral_data = load_combined_results(csv_paths)
        if spectral_data is None:
            print(f"    Warning: Could not load spectral data for {year}")
            return None
    
    # Filter to this transect
    transect_data = spectral_data[spectral_data['TransectID'] == transect_id]
    
    if transect_data.empty:
        print(f"    Warning: No data for transect {transect_id} in results")
        return None
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    
    distance = transect_data['distance'].values
    
    # === Top panel: Spectral profiles ===
    
    # Plot bands
    if 'red' in transect_data.columns:
        ax1.plot(distance, transect_data['red'], 'r-', label='Red', alpha=0.8)
    if 'green' in transect_data.columns:
        ax1.plot(distance, transect_data['green'], 'g-', label='Green', alpha=0.8)
    if 'blue' in transect_data.columns:
        ax1.plot(distance, transect_data['blue'], 'b-', label='Blue', alpha=0.8)
    if 'nir' in transect_data.columns:
        ax1.plot(distance, transect_data['nir'], 'k-', label='NIR', linewidth=2)
    
    # Add shoreline markers
    ax1.axvline(manual_distance, color='green', linestyle='--', linewidth=2.5,
                label=f'Manual: {manual_distance:.1f}m')
    ax1.axvline(algo_distance, color='red', linestyle='-', linewidth=2.5,
                label=f'Algorithm: {algo_distance:.1f}m')
    
    # Fill between to show difference
    ymin, ymax = ax1.get_ylim()
    if difference > 0:
        # Algorithm is landward (positive)
        ax1.axvspan(manual_distance, algo_distance, alpha=0.2, color='orange',
                    label=f'Diff: +{difference:.1f}m (landward)')
    else:
        ax1.axvspan(algo_distance, manual_distance, alpha=0.2, color='cyan',
                    label=f'Diff: {difference:.1f}m (seaward)')
    
    ax1.set_xlabel('Distance from West (m)', fontsize=11)
    ax1.set_ylabel('DN Value', fontsize=11)
    ax1.set_title(f'Transect {transect_id} - {year} - {description.replace("_", " ").title()}\n'
                  f'Difference: {difference:+.1f}m '
                  f'{"(algorithm landward)" if difference > 0 else "(algorithm seaward)"}',
                  fontsize=13)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # === Bottom panel: Classification and NIR derivative ===
    
    if 'predicted_class' in transect_data.columns:
        # Create color map for classes
        class_colors = {
            'VEG_DUNES': '#228B22',
            'BEACH_DRY': '#F4A460', 
            'BEACH_WET': '#D2B48C',
            'WATER': '#4169E1',
            'WAVE_CRESTS': '#87CEEB',
            'UNKNOWN': '#808080'
        }
        
        # Plot classification as background
        for i, (_, row) in enumerate(transect_data.iterrows()):
            cls = row.get('predicted_class', 'UNKNOWN')
            color = class_colors.get(cls, '#808080')
            if i < len(transect_data) - 1:
                next_dist = transect_data['distance'].iloc[i + 1]
                ax2.axvspan(row['distance'], next_dist, alpha=0.3, color=color)
    
    # Plot NIR derivative if available
    if 'nir_d1' in transect_data.columns:
        ax2.plot(distance, transect_data['nir_d1'], 'k-', label='NIR d1', linewidth=1.5)
    elif 'nir' in transect_data.columns:
        # Compute simple derivative
        nir = transect_data['nir'].values
        nir_d1 = np.gradient(nir, distance)
        ax2.plot(distance, nir_d1, 'k-', label='NIR d1 (computed)', linewidth=1.5)
    
    # Add shoreline markers
    ax2.axvline(manual_distance, color='green', linestyle='--', linewidth=2.5)
    ax2.axvline(algo_distance, color='red', linestyle='-', linewidth=2.5)
    
    ax2.axhline(0, color='gray', linestyle=':', linewidth=1)
    
    ax2.set_xlabel('Distance from West (m)', fontsize=11)
    ax2.set_ylabel('NIR Derivative', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'diagnostic_{transect_id}_{description}_{year}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved diagnostic plot: {output_path.name}")
    return output_path


def format_statistics_report(stats: Dict, year: str) -> str:
    """Format statistics as readable report string."""
    if stats['n_valid'] == 0:
        return f"\n=== {year} ===\nNo valid comparisons.\n"
    
    report = f"""
=== {year} ===
Coverage: {stats['n_valid']} / {stats['n_total']} transects ({100*stats['n_valid']/stats['n_total']:.1f}%)

DIRECTIONAL BIAS:
  Mean difference: {stats['mean_diff']:.2f} m {'(algorithm landward)' if stats['mean_diff'] > 0 else '(algorithm seaward)'}
  Median difference: {stats['median_diff']:.2f} m
  
MAGNITUDE:
  Std deviation: {stats['std_diff']:.2f} m
  Mean absolute error: {stats['mean_abs_diff']:.2f} m

DISTRIBUTION:
  Landward (positive): {stats['pct_landward']:.1f}%
  Seaward (negative): {stats['pct_seaward']:.1f}%

EXTREMES:
  Best agreement: Transect {stats['best_agreement_tid']} ({stats['best_agreement_diff']:.2f} m)
  Worst landward: Transect {stats['worst_landward_tid']} (+{stats['worst_landward_diff']:.2f} m)
  Worst seaward: Transect {stats['worst_seaward_tid']} ({stats['worst_seaward_diff']:.2f} m)
"""
    return report


# === MAIN ===

def main():
    print("=" * 60)
    print("MANUAL vs ALGORITHM SHORELINE COMPARISON")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    
    try:
        manual_gdf = load_manual_shorelines(MANUAL_SHORELINE_DIR, MANUAL_SHORELINE_BASE)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    
    try:
        transects = load_transects(TRANSECT_FILE)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    
    # Ensure CRS match
    if manual_gdf.crs != transects.crs:
        print(f"\nReprojecting manual shorelines from {manual_gdf.crs} to {transects.crs}")
        manual_gdf = manual_gdf.to_crs(transects.crs)
    
    # Process each year
    all_results = []
    all_stats = {}
    report_lines = []
    
    for year in YEARS:
        print(f"\n{'='*60}")
        print(f"Processing {year}...")
        print('='*60)
        
        # Get manual shoreline for this year
        manual_year = manual_gdf[manual_gdf['Year'] == year]
        if len(manual_year) == 0:
            print(f"  Warning: No manual shoreline for {year}")
            continue
        
        manual_shoreline = manual_year.geometry.iloc[0]
        
        # Get algorithm shoreline for this year
        algo_gdf = load_algorithm_shoreline(ALGORITHM_OUTPUT_ROOT, year)
        if algo_gdf is None:
            continue
        
        algo_shoreline = algo_gdf.geometry.iloc[0]
        
        # Ensure CRS match
        if algo_gdf.crs != transects.crs:
            algo_gdf = algo_gdf.to_crs(transects.crs)
            algo_shoreline = algo_gdf.geometry.iloc[0]
        
        print(f"  Manual shoreline: {manual_shoreline.geom_type} with {count_vertices(manual_shoreline)} vertices")
        print(f"  Algorithm shoreline: {algo_shoreline.geom_type} ({count_segments(algo_shoreline)} segments, {count_vertices(algo_shoreline)} vertices)")
        
        # Load combined spectral data for this year (once, before comparison)
        csv_paths = find_results_csv_group(ALGORITHM_OUTPUT_ROOT, year)
        spectral_data = load_combined_results(csv_paths) if csv_paths else None
        
        if spectral_data is not None:
            unique_transects = spectral_data['TransectID'].nunique()
            print(f"  Spectral data: {len(spectral_data)} samples across {unique_transects} transects")
        
        # Compare shorelines
        print(f"  Comparing {len(transects)} transects...")
        df = compare_shorelines_for_year(transects, manual_shoreline, algo_shoreline, year)
        
        # Calculate statistics
        stats = calculate_statistics(df)
        all_stats[year] = stats
        
        # Print and store report
        report = format_statistics_report(stats, year)
        print(report)
        report_lines.append(report)
        
        # Store results
        all_results.append(df)
        
        # Create visualizations
        year_output_dir = OUTPUT_DIR / year
        year_output_dir.mkdir(exist_ok=True)
        
        plot_difference_histogram(df, year, year_output_dir / f'histogram_{year}.png')
        plot_difference_by_transect(df, year, year_output_dir / f'spatial_{year}.png')
        
        # Save per-transect data
        csv_path = year_output_dir / f'comparison_{year}.csv'
        df.to_csv(csv_path, index=False)
        print(f"  Saved comparison data: {csv_path.name}")
        
        # Select diagnostic transects
        diagnostic_tids = select_diagnostic_transects(df, stats)
        print(f"  Diagnostic transects for spectral plots: {diagnostic_tids}")
        
        # Generate diagnostic spectral profile plots
        print(f"  Generating diagnostic spectral plots...")
        for tid in diagnostic_tids:
            # Get data for this transect
            row = df[df['TransectID'] == tid]
            if row.empty:
                continue
            
            row = row.iloc[0]
            manual_dist = row['manual_distance']
            algo_dist = row['algorithm_distance']
            diff = row['difference']
            
            # Skip if missing data
            if pd.isna(manual_dist) or pd.isna(algo_dist):
                print(f"    Skipping transect {tid}: missing intersection data")
                continue
            
            # Determine description
            if tid == stats['best_agreement_tid']:
                desc = 'best_agreement'
            elif tid == stats['worst_landward_tid']:
                desc = 'worst_landward'
            elif tid == stats['worst_seaward_tid']:
                desc = 'worst_seaward'
            else:
                desc = 'representative'
            
            plot_diagnostic_spectral_profile(
                transect_id=tid,
                year=year,
                manual_distance=manual_dist,
                algo_distance=algo_dist,
                difference=diff,
                description=desc,
                output_dir=year_output_dir,
                spectral_data=spectral_data  # Pass pre-loaded data
            )
        
        # Save diagnostic transect list
        diagnostic_path = year_output_dir / f'diagnostic_transects_{year}.json'
        with open(diagnostic_path, 'w') as f:
            json.dump({
                'year': year,
                'transect_ids': diagnostic_tids,
                'descriptions': {
                    str(stats['best_agreement_tid']): 'best_agreement',
                    str(stats['worst_landward_tid']): 'worst_landward',
                    str(stats['worst_seaward_tid']): 'worst_seaward'
                }
            }, f, indent=2)
    
    # Save combined report
    report_path = OUTPUT_DIR / 'comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write("MANUAL vs ALGORITHM SHORELINE COMPARISON\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"\nManual shorelines: {MANUAL_SHORELINE_DIR / MANUAL_SHORELINE_BASE}.shp\n")
        f.write(f"Algorithm outputs: {ALGORITHM_OUTPUT_ROOT}\n")
        f.write(f"Transects: {TRANSECT_FILE}\n")
        f.write("\n" + "=" * 60 + "\n")
        for line in report_lines:
            f.write(line)
    
    print(f"\nSaved report: {report_path}")
    
    # Save combined CSV
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_csv = OUTPUT_DIR / 'comparison_all_years.csv'
        combined_df.to_csv(combined_csv, index=False)
        print(f"Saved combined CSV: {combined_csv}")
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())