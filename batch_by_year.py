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
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import rasterio

from spectral_classifier.main import analyze_all_transects
from spectral_classifier.data_io import BAND_CONFIG_4BAND, BAND_CONFIG_CIR, BAND_CONFIG_RGB

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


# === HELPER FUNCTIONS ===

def extract_year_from_path(path: Path) -> str:
    """Extract year from path like 'imagery/2016/20160122/file.tif'"""
    for part in path.parts:
        if re.match(r'^\d{4}$', part):
            return part
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
        print(f"  Warning: No spectral data for transect {result.get('transect_id')}, skipping plot")
        return None
    
    transect_id = result.get('transect_id', 'unknown')
    transitions = result.get('transitions', [])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    distance = data['distance']
    
    # Check which bands are available
    has_nir = 'nir' in data.columns and not data['nir'].isna().all()
    has_blue = 'blue' in data.columns and not data['blue'].isna().all()
    
    # Determine band mode for title
    if has_nir and has_blue:
        band_mode = '4-band'
    elif has_nir:
        band_mode = 'CIR'
    else:
        band_mode = 'RGB'
    
    # Plot each band with distinct colors
    ax.plot(distance, data['red'], color='#e41a1c', linewidth=1.5, label='Red', alpha=0.8)
    ax.plot(distance, data['green'], color='#4daf4a', linewidth=1.5, label='Green', alpha=0.8)
    
    if has_blue:
        ax.plot(distance, data['blue'], color='#377eb8', linewidth=1.5, label='Blue', alpha=0.8)
    
    if has_nir:
        ax.plot(distance, data['nir'], color='#984ea3', linewidth=1.5, label='NIR', alpha=0.8)
    
    # Find shell line transition (dry_wet_derivative type)
    shell_line = None
    for t in transitions:
        if t.get('type') == 'dry_wet_derivative':
            shell_line = t
            break
    
    # Add shell line marker if detected
    if shell_line:
        shell_dist = shell_line['distance']
        shell_conf = shell_line.get('confidence', 0)
        
        # Vertical line at shell line location
        ax.axvline(x=shell_dist, color='#ff7f00', linewidth=2, linestyle='--', 
                   label=f'Shell Line ({shell_dist:.1f}m)', alpha=0.9)
        
        # Annotation with confidence
        bands_to_check = ['red', 'green']
        if has_blue:
            bands_to_check.append('blue')
        if has_nir:
            bands_to_check.append('nir')
        
        y_max = max(data[bands_to_check].max())
        ax.annotate(f'conf: {shell_conf:.2f}', 
                    xy=(shell_dist, y_max * 0.95),
                    xytext=(shell_dist + 5, y_max * 0.95),
                    fontsize=9, color='#ff7f00',
                    ha='left', va='top')
    
    # Labels and title
    ax.set_xlabel('Distance from West (m)', fontsize=11)
    ax.set_ylabel('DN Value', fontsize=11)
    
    # Title indicates band mode and detection status
    detection_status = "detected" if shell_line else "no detection"
    ax.set_title(f'Spectral Profile - Year {year}, Transect {transect_id}\n'
                 f'({band_mode}, {detection_status})', 
                 fontsize=12, fontweight='bold')
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Grid for readability
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.set_axisbelow(True)
    
    # Set reasonable y-axis limits
    bands_to_check = ['red', 'green']
    if has_blue:
        bands_to_check.append('blue')
    if has_nir:
        bands_to_check.append('nir')
    
    y_max = max(data[bands_to_check].max())
    ax.set_ylim(0, y_max * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved spectral profile: {output_path.name} (Transect {transect_id}, {band_mode}, {detection_status})")
    return output_path


def merge_summaries(summary_files: list, output_path: Path, year: str, 
                    prefer_4band: bool = True):
    """
    Merge multiple summary JSON files into one year-level summary.
    
    Handles duplicate TransectIDs by:
    1. Preferring 4-band results over 3-band
    2. If same band count, keeping higher confidence
    
    Args:
        summary_files: List of summary file paths
        output_path: Output path for merged summary
        year: Year string
        prefer_4band: If True, prefer 4-band results for duplicates
    """
    merged = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "year": year,
            "source_summaries": [str(p.name) for p in summary_files],
            "processing_info": {}
        },
        "transects": [],
        "overall_class_distribution": defaultdict(int),
        "total_transitions": 0,
        "band_mode_summary": defaultdict(int)
    }
    
    # Track transects by ID to handle duplicates
    transects_by_id = {}
    transect_band_modes = {}  # Track band mode per transect
    
    for summary_file in summary_files:
        with open(summary_file) as f:
            data = json.load(f)
        
        # Get band mode from metadata if available
        proc_info = data.get("metadata", {}).get("processing_info", {})
        primary_band_mode = proc_info.get("primary_band_mode", "unknown")
        
        # Track source info
        raster_dir = proc_info.get("raster_dir", "unknown")
        merged["metadata"]["processing_info"][raster_dir] = {
            "num_transects": data["metadata"].get("num_transects", 0),
            "band_mode": primary_band_mode
        }
        
        # Merge transects
        for t in data.get("transects", []):
            tid = t["transect_id"]
            
            # Determine if this result is from 4-band
            # (We can infer from class distribution - 4-band has less UNKNOWN)
            is_4band = primary_band_mode == '4band'
            
            if tid in transects_by_id:
                existing = transects_by_id[tid]
                existing_is_4band = transect_band_modes.get(tid, False)
                
                # Priority: 4-band > 3-band > lower confidence
                should_replace = False
                
                if prefer_4band:
                    if is_4band and not existing_is_4band:
                        should_replace = True
                    elif is_4band == existing_is_4band:
                        # Same band mode - compare confidence
                        existing_conf = max([tr["confidence"] for tr in existing["transitions"]], default=0)
                        new_conf = max([tr["confidence"] for tr in t["transitions"]], default=0)
                        if new_conf > existing_conf:
                            should_replace = True
                
                if should_replace:
                    transects_by_id[tid] = t
                    transect_band_modes[tid] = is_4band
            else:
                transects_by_id[tid] = t
                transect_band_modes[tid] = is_4band
    
    # Build final transect list (sorted by ID)
    merged["transects"] = [
        transects_by_id[tid] 
        for tid in sorted(transects_by_id.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    ]
    
    # Recalculate overall stats
    for t in merged["transects"]:
        merged["total_transitions"] += t.get("num_transitions", 0)
        for cls, count in t.get("class_distribution", {}).items():
            merged["overall_class_distribution"][cls] += count
    
    # Band mode summary
    for tid, is_4band in transect_band_modes.items():
        mode = '4band' if is_4band else '3band'
        merged["band_mode_summary"][mode] += 1
    
    # Convert defaultdicts to regular dicts for JSON
    merged["overall_class_distribution"] = dict(merged["overall_class_distribution"])
    merged["band_mode_summary"] = dict(merged["band_mode_summary"])
    
    # Write merged summary
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"  Merged {len(summary_files)} summaries -> {output_path.name}")
    print(f"    Total transects: {len(merged['transects'])}")
    print(f"    Band modes: {merged['band_mode_summary']}")
    print(f"    Total transitions: {merged['total_transitions']}")
    
    return output_path


def export_shellline_geojson(summary_file: Path, transect_file: Path, output_file: Path):
    """Convert summary.json transitions to a LineString GeoJSON."""
    
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
    avg_confidence = sum(p["confidence"] for p in points_with_order) / len(points_with_order)
    
    # Include year and band mode from summary metadata
    year = summary.get("metadata", {}).get("year", "unknown")
    band_summary = summary.get("band_mode_summary", {})
    
    gdf = gpd.GeoDataFrame([{
        "geometry": shoreline,
        "year": year,
        "num_points": len(coords),
        "avg_confidence": avg_confidence,
        "band_4_count": band_summary.get('4band', 0),
        "band_3_count": band_summary.get('3band', 0)
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
            
            # Export shellline from merged summary
            shellline_file = output_dir / f"shellline_{year}.geojson"
            export_shellline_geojson(merged_summary, TRANSECT_FILE, shellline_file)
            
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