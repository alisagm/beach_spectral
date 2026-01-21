# batch_by_year.py
"""
Batch process all imagery (3-band and 4-band), grouped by year.

Supports:
- 4-band (RGBN): Full NIR-based detection
- 3-band CIR [NIR,R,G]: NIR-based detection (auto-detected)
- 3-band RGB [R,G,B]: Brightness-based detection (degraded mode)

When duplicate transects exist (same year), 4-band results are preferred.

Outputs: OUTPUT/{year}/summary_{year}.json, shellline_{year}.geojson, spectral_profile_{year}.png
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import matplotlib.pyplot as plt
import rasterio

from spectral_classifier.main import analyze_all_transects
from spectral_classifier.data_io import BAND_CONFIG_4BAND, BAND_CONFIG_CIR, BAND_CONFIG_RGB
from spectral_classifier.footprint_clip import compute_merged_footprint, clip_shoreline_to_footprint

# === CONFIGURATION ===
# Update this path for your system
IMAGERY_ROOT = Path(r"C:\Users\alisa\Desktop\SIP\git\beach_spectral\PAIS_shorelines\imagery")
TRANSECT_FILE = Path("INPUT/shorelineTransPais.json")
OUTPUT_ROOT = Path("OUTPUT")

# Supported raster formats
# Note: .sid (MrSID) and .ers require GDAL with special drivers
SUPPORTED_EXTENSIONS = ['.tif', '.tiff', '.jp2']

# Extensions that require special GDAL drivers (may not work without OSGeo4W)
OPTIONAL_EXTENSIONS = ['.sid', '.ers', '.ecw']

# Target transect for diagnostic plot
TARGET_TRANSECT_ID = 1000

# Minimum bands required (changed from 4 to 3 for 3-band support)
MIN_BANDS = 3

# Footprint clipping settings
EDGE_BUFFER_M = 25.0  # Buffer inward from imagery edges
MIN_SEGMENT_LENGTH_M = 100.0  # Drop segments shorter than this


# === HELPER FUNCTIONS ===

def extract_year_from_path(path: Path) -> str:
    """
    Extract year from path like 'imagery/2016/20160122/file.tif' or 'imagery/200605/file.tif'
    
    Handles:
        - 4 digits: YYYY (e.g., '2016' → '2016')
        - 6 digits: YYYYMM (e.g., '200605' → '2006')
        - 8 digits: YYYYMMDD (e.g., '20160122' → '2016')
    """
    for part in path.parts:
        # Check for 4-digit year (YYYY)
        if re.match(r'^\d{4}$', part):
            return part
        # Check for 6-digit date (YYYYMM)
        if re.match(r'^\d{6}$', part):
            return part[:4]
        # Check for 8-digit date (YYYYMMDD)
        if re.match(r'^\d{8}$', part):
            return part[:4]
    return "unknown"


def get_band_count_safe(path: Path) -> tuple:
    """
    Safely get band count from a raster file.
    
    Returns:
        Tuple of (band_count, error_message)
        If successful: (band_count, None)
        If failed: (0, error_message)
    """
    try:
        with rasterio.open(path) as src:
            return src.count, None
    except Exception as e:
        return 0, str(e)


def classify_raster(path: Path) -> dict:
    """
    Classify a raster by band count and infer band mode.
    
    Returns:
        Dict with keys: 'path', 'band_count', 'band_mode', 'year', 'error'
    """
    band_count, error = get_band_count_safe(path)
    
    if error:
        return {
            'path': path,
            'band_count': 0,
            'band_mode': None,
            'year': extract_year_from_path(path),
            'error': error
        }
    
    # Classify band mode
    if band_count >= 4:
        band_mode = BAND_CONFIG_4BAND
    elif band_count == 3:
        # Will be auto-detected as CIR or RGB by data_io.detect_band_configuration
        band_mode = '3band'  # Placeholder - actual detection happens later
    else:
        band_mode = None  # Insufficient bands
    
    return {
        'path': path,
        'band_count': band_count,
        'band_mode': band_mode,
        'year': extract_year_from_path(path),
        'error': None
    }


def group_rasters_by_year(imagery_root: Path, include_optional: bool = True) -> dict:
    """
    Find all rasters (3+ bands) and group by year.
    
    Args:
        imagery_root: Root directory to scan
        include_optional: If True, try to include .sid/.ers/.ecw formats
        
    Returns:
        Dict mapping year -> list of raster info dicts
    """
    # Build extension list
    extensions = SUPPORTED_EXTENSIONS.copy()
    if include_optional:
        extensions.extend(OPTIONAL_EXTENSIONS)
    
    # Find all files
    all_files = []
    for ext in extensions:
        all_files.extend(imagery_root.rglob(f"*{ext}"))
    
    print(f"Found {len(all_files)} raster files to check")
    
    by_year = defaultdict(list)
    skipped = []
    band_4_count = 0
    band_3_count = 0
    
    for path in all_files:
        info = classify_raster(path)
        
        if info['error']:
            skipped.append((path, info['error']))
            continue
        
        if info['band_count'] < MIN_BANDS:
            skipped.append((path, f"{info['band_count']} bands (need {MIN_BANDS}+)"))
            continue
        
        # Track band counts
        if info['band_count'] >= 4:
            band_4_count += 1
        else:
            band_3_count += 1
        
        by_year[info['year']].append(info)
    
    # Summary
    total_usable = sum(len(v) for v in by_year.values())
    print(f"\nRaster Summary:")
    print(f"  Usable: {total_usable} ({band_4_count} 4-band, {band_3_count} 3-band)")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Years: {sorted(by_year.keys())}")
    
    if skipped:
        # Group skip reasons
        skip_reasons = defaultdict(int)
        for _, reason in skipped:
            skip_reasons[reason] += 1
        print(f"  Skip reasons: {dict(skip_reasons)}")
    
    return dict(by_year)


def find_nearest_transect(results: list, target_id: int) -> dict:
    """
    Find the result with TransectID closest to target_id.
    
    Returns the result dict, or None if results is empty.
    """
    if not results:
        return None
    
    def get_tid(result):
        tid = result.get('transect_id', 0)
        try:
            return int(tid)
        except (ValueError, TypeError):
            return 0
    
    closest = min(results, key=lambda r: abs(get_tid(r) - target_id))
    return closest


def plot_spectral_profile(result: dict, output_path: Path, year: str):
    """
    Create a spectral profile plot showing R, G, B, NIR values vs distance.
    Includes a marker for detected shell line location if present.
    Handles missing NIR (3-band RGB) gracefully.
    
    Args:
        result: Dict with 'transect_id', 'data' (DataFrame), and 'transitions' (list)
        output_path: Path to save the PNG
        year: Year string for title
    """
    data = result.get('data')
    if data is None or data.empty:
        print(f"  Warning: No data available for spectral profile plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    distance = data['distance']
    
    # Plot RGB (always available)
    ax.plot(distance, data['red'], 'r-', label='Red', alpha=0.8)
    ax.plot(distance, data['green'], 'g-', label='Green', alpha=0.8)
    ax.plot(distance, data['blue'], 'b-', label='Blue', alpha=0.8)
    
    # Plot NIR if available
    if 'nir' in data.columns and data['nir'].notna().any():
        ax.plot(distance, data['nir'], 'k-', label='NIR', linewidth=2)
    
    # Mark shell line transitions
    transitions = result.get('transitions', [])
    for trans in transitions:
        if trans.get('type') == 'dry_wet_derivative':
            shell_dist = trans.get('distance', 0)
            ax.axvline(x=shell_dist, color='orange', linestyle='--', linewidth=2, 
                      label=f'Shell Line ({shell_dist:.1f}m)')
    
    ax.set_xlabel('Distance from West (m)')
    ax.set_ylabel('DN Value')
    ax.set_title(f'Spectral Profile - Year {year}, Transect {result.get("transect_id", "?")}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  Saved spectral profile: {output_path.name}")


def merge_summaries(summary_files: list, output_path: Path, year: str, prefer_4band: bool = True):
    """
    Merge multiple summary.json files, keeping best result per transect.
    
    When prefer_4band=True, 4-band detections take precedence over 3-band.
    """
    all_transects = {}  # TransectID -> best result
    band_mode_counts = defaultdict(int)
    
    for sf in summary_files:
        with open(sf) as f:
            summary = json.load(f)
        
        for t in summary.get('transects', []):
            tid = t['transect_id']
            band_mode = t.get('band_mode', 'unknown')
            
            # Check if we should replace existing
            if tid not in all_transects:
                all_transects[tid] = t
            elif prefer_4band:
                existing_mode = all_transects[tid].get('band_mode', 'unknown')
                # Replace if new is 4-band and existing is not
                if band_mode == '4band' and existing_mode != '4band':
                    all_transects[tid] = t
                # Also replace if same mode but new has transitions and existing doesn't
                elif band_mode == existing_mode:
                    existing_trans = all_transects[tid].get('transitions', [])
                    new_trans = t.get('transitions', [])
                    if len(new_trans) > len(existing_trans):
                        all_transects[tid] = t
    
    # Count final band modes
    for t in all_transects.values():
        mode = t.get('band_mode', 'unknown')
        if '4band' in str(mode):
            band_mode_counts['4band'] += 1
        elif 'cir' in str(mode).lower():
            band_mode_counts['3band_cir'] += 1
        elif 'rgb' in str(mode).lower():
            band_mode_counts['3band_rgb'] += 1
        else:
            band_mode_counts['3band'] += 1
    
    # Build merged summary
    merged = {
        'metadata': {
            'year': year,
            'merged_from': len(summary_files),
            'timestamp': datetime.now().isoformat()
        },
        'transects': sorted(all_transects.values(), key=lambda x: int(x['transect_id']) if str(x['transect_id']).isdigit() else 0),
        'band_mode_summary': dict(band_mode_counts),
        'total_transitions': sum(len(t.get('transitions', [])) for t in all_transects.values())
    }
    
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"  Merged {len(summary_files)} summaries -> {output_path.name}")
    print(f"    Total transects: {len(merged['transects'])}")
    print(f"    Band modes: {merged['band_mode_summary']}")
    print(f"    Total transitions: {merged['total_transitions']}")
    
    return output_path


def export_shellline_geojson(
    summary_file: Path, 
    transect_file: Path, 
    output_file: Path, 
    raster_paths: list = None,  # CHANGED: Now accepts list of all raster paths
    edge_buffer_m: float = 25.0,
    min_segment_length_m: float = 100.0
):
    """
    Convert summary.json transitions to a LineString GeoJSON.
    
    Optionally clips to imagery footprint if raster_paths provided.
    
    Args:
        summary_file: Path to merged summary JSON
        transect_file: Path to transects GeoJSON
        output_file: Path for output shellline GeoJSON
        raster_paths: List of all raster file paths for footprint clipping (optional)
        edge_buffer_m: Inward buffer from imagery edges (default 25m)
        min_segment_length_m: Drop clipped segments shorter than this (default 100m)
    """
    
    with open(summary_file) as f:
        summary = json.load(f)
    
    transects = gpd.read_file(transect_file)
    transect_lookup = {str(row.TransectID): row.geometry for _, row in transects.iterrows()}
    
    points_with_order = []
    
    for t in summary["transects"]:
        tid = t["transect_id"]
        
        for trans in t["transitions"]:
            if trans["type"] != "dry_wet_derivative":
                continue
            
            if tid not in transect_lookup:
                continue
            
            line = transect_lookup[tid]
            
            # === FIX: Account for transect direction ===
            # The classifier reports distance from WEST (land side)
            # But the GeoJSON stores transects EAST to WEST
            # So we need to measure from the END of the LineString, not the start
            
            coords = list(line.coords)
            start_x = coords[0][0]   # First point easting
            end_x = coords[-1][0]    # Last point easting
            
            if start_x > end_x:
                # Transect runs east-to-west (start is seaward)
                # Distance from west = line.length - distance_from_start
                interp_distance = line.length - trans["distance"]
            else:
                # Transect runs west-to-east (start is landward)
                interp_distance = trans["distance"]
            
            # Clamp to valid range
            interp_distance = max(0, min(interp_distance, line.length))
            point = line.interpolate(interp_distance)
            # === END FIX ===
            
            points_with_order.append({
                "transect_id": int(tid) if tid.isdigit() else 0,
                "point": point,
                "confidence": trans["confidence"]
            })

    if len(points_with_order) < 2:
        print(f"  Warning: Only {len(points_with_order)} points, skipping LineString export")
        return None
    
    # Sort by TransectID and create LineString
    points_with_order.sort(key=lambda x: x["transect_id"])
    coords = [p["point"] for p in points_with_order]
    
    shoreline = LineString(coords)
    original_length = shoreline.length
    avg_confidence = sum(p["confidence"] for p in points_with_order) / len(points_with_order)
    
    # === FOOTPRINT CLIPPING ===
    num_segments = 1
    clipped_length = original_length
    
    if raster_paths and len(raster_paths) > 0:
        print(f"  Computing imagery footprint from {len(raster_paths)} rasters...")
        
        footprint_result = compute_merged_footprint(
            raster_paths, 
            edge_buffer_m=edge_buffer_m
        )
        
        if footprint_result:
            footprint, footprint_crs = footprint_result
            
            # Handle CRS mismatch by reprojecting footprint to match transects
            if transects.crs != footprint_crs:
                print(f"  Reprojecting footprint from {footprint_crs} to {transects.crs}...")
                import pyproj
                from shapely.ops import transform
                
                transformer = pyproj.Transformer.from_crs(
                    footprint_crs, 
                    transects.crs, 
                    always_xy=True
                )
                footprint = transform(transformer.transform, footprint)
            
            # Clip shoreline to footprint
            shoreline = clip_shoreline_to_footprint(
                shoreline, 
                footprint, 
                min_segment_length_m=min_segment_length_m
            )
            
            clipped_length = shoreline.length if not shoreline.is_empty else 0
            
            # Count segments
            if isinstance(shoreline, MultiLineString):
                num_segments = len(shoreline.geoms)
            elif shoreline.is_empty:
                num_segments = 0
            else:
                num_segments = 1
            
            print(f"  Clipped: {original_length/1000:.1f}km → {clipped_length/1000:.1f}km ({num_segments} segments)")
    # === END FOOTPRINT CLIPPING ===
    
    # Include year and band mode from summary metadata
    year = summary.get("metadata", {}).get("year", "unknown")
    band_summary = summary.get("band_mode_summary", {})
    
    gdf = gpd.GeoDataFrame([{
        "geometry": shoreline,
        "year": year,
        "num_points": len(coords),
        "avg_confidence": avg_confidence,
        "band_4_count": band_summary.get('4band', 0),
        "band_3_count": band_summary.get('3band', 0),
        "original_length_m": original_length,
        "clipped_length_m": clipped_length,
        "num_segments": num_segments,
        "edge_buffer_m": edge_buffer_m
    }], crs=transects.crs)
    
    gdf.to_file(output_file, driver="GeoJSON")
    print(f"  Exported shellline: {len(coords)} vertices, avg confidence: {avg_confidence:.3f}")
    
    return output_file


# === MAIN ===

def main():
    print("=" * 60)
    print("BATCH PROCESSING BY YEAR")
    print("=" * 60)
    print(f"Imagery root: {IMAGERY_ROOT}")
    print(f"Transect file: {TRANSECT_FILE}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Minimum bands: {MIN_BANDS}")
    print(f"Edge buffer: {EDGE_BUFFER_M}m")
    print()
    
    # Verify paths exist
    if not IMAGERY_ROOT.exists():
        print(f"ERROR: Imagery root not found: {IMAGERY_ROOT}")
        return 1
    
    if not TRANSECT_FILE.exists():
        print(f"ERROR: Transect file not found: {TRANSECT_FILE}")
        return 1
    
    # Group rasters by year
    by_year = group_rasters_by_year(IMAGERY_ROOT)
    
    if not by_year:
        print("ERROR: No usable rasters found")
        return 1
    
    for year in sorted(by_year.keys()):
        raster_infos = by_year[year]
        
        # Separate 4-band and 3-band
        four_band = [r for r in raster_infos if r['band_count'] >= 4]
        three_band = [r for r in raster_infos if r['band_count'] == 3]
        
        print()
        print("=" * 60)
        print(f"YEAR: {year}")
        print(f"  Rasters: {len(raster_infos)} total ({len(four_band)} 4-band, {len(three_band)} 3-band)")
        print("=" * 60)
        
        # Create output directory
        output_dir = OUTPUT_ROOT / year
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get unique parent directories (date folders) for these rasters
        raster_dirs = sorted(set(r['path'].parent for r in raster_infos))
        
        # Sort directories to process 4-band first (so they get priority in merge)
        def dir_priority(d):
            """Return sort key: 4-band dirs first, then by name"""
            has_4band = any(r['band_count'] >= 4 for r in raster_infos if r['path'].parent == d)
            return (0 if has_4band else 1, str(d))
        
        raster_dirs = sorted(raster_dirs, key=dir_priority)
        
        # *** COLLECT ALL RASTER PATHS FOR THIS YEAR ***
        all_raster_paths_for_year = [r['path'] for r in raster_infos]
        
        # Track summary files created for this year
        date_summaries = []
        
        # Track the best result for target transect (for diagnostic plot)
        best_target_result = None
        best_target_is_4band = False
        
        # Process each date folder within the year
        for raster_dir in raster_dirs:
            # Check band composition of this directory
            dir_rasters = [r for r in raster_infos if r['path'].parent == raster_dir]
            n_4band = sum(1 for r in dir_rasters if r['band_count'] >= 4)
            n_3band = sum(1 for r in dir_rasters if r['band_count'] == 3)
            
            relative_path = raster_dir.relative_to(IMAGERY_ROOT) if raster_dir.is_relative_to(IMAGERY_ROOT) else raster_dir
            print(f"\n  Processing: {relative_path}")
            print(f"    ({n_4band} 4-band, {n_3band} 3-band)")
            
            try:
                results = analyze_all_transects(
                    raster_dir=raster_dir,
                    transect_geojson=TRANSECT_FILE,
                    output_dir=output_dir,
                    num_visualize=0,  # Skip plots for batch
                    verbose=False,
                    boundary_types='shore_only'
                )
                print(f"    Processed {len(results)} transects")
                
                # Find the summary file just created
                latest_summary = max(
                    output_dir.glob("summary_*.json"),
                    key=lambda p: p.stat().st_mtime
                )
                date_summaries.append(latest_summary)
                
                # Determine if this directory had 4-band
                dir_is_4band = n_4band > 0
                
                # Look for target transect for diagnostic plot
                candidate = find_nearest_transect(results, TARGET_TRANSECT_ID)
                if candidate:
                    candidate_tid = int(candidate.get('transect_id', 0))
                    
                    # Keep if: we don't have one yet, OR this is 4-band and prev wasn't,
                    # OR same band status but closer to target
                    should_replace = False
                    
                    if best_target_result is None:
                        should_replace = True
                    elif dir_is_4band and not best_target_is_4band:
                        should_replace = True
                    elif dir_is_4band == best_target_is_4band:
                        current_tid = int(best_target_result.get('transect_id', 0))
                        if abs(candidate_tid - TARGET_TRANSECT_ID) < abs(current_tid - TARGET_TRANSECT_ID):
                            should_replace = True
                    
                    if should_replace:
                        best_target_result = candidate
                        best_target_is_4band = dir_is_4band
                
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
        
        # Merge all date summaries into year summary
        if date_summaries:
            print(f"\n  Merging {len(date_summaries)} date summaries...")
            
            merged_summary = output_dir / f"summary_{year}.json"
            merge_summaries(date_summaries, merged_summary, year, prefer_4band=True)
            
            # Export shellline from merged summary WITH footprint clipping
            shellline_file = output_dir / f"shellline_{year}.geojson"
            export_shellline_geojson(
                merged_summary, 
                TRANSECT_FILE, 
                shellline_file,
                raster_paths=all_raster_paths_for_year,  # Pass ALL rasters for the year
                edge_buffer_m=EDGE_BUFFER_M,
                min_segment_length_m=MIN_SEGMENT_LENGTH_M
            )
            
            # Generate diagnostic spectral profile plot
            if best_target_result:
                profile_file = output_dir / f"spectral_profile_{year}.png"
                plot_spectral_profile(best_target_result, profile_file, year)
            else:
                print(f"  Warning: No transect data available for diagnostic plot")
            
            # Clean up individual date summaries
            for sf in date_summaries:
                sf.unlink()
        
        print(f"\n  Year {year} complete.")

    print()
    print("=" * 60)
    print("BATCH COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())