# batch_by_year.py
"""
Batch process all 4-band imagery, grouped by year.
Outputs: OUTPUT/{year}/summary_{year}.json, shellline.geojson, spectral_profile_{year}.png
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import geopandas as gpd
from shapely.geometry import LineString
import matplotlib.pyplot as plt

from spectral_classifier.main import analyze_all_transects

# === CONFIGURATION ===
IMAGERY_ROOT = Path("PAIS_shorelines/imagery")
TRANSECT_FILE = Path("INPUT/shorelineTransPais.json")
OUTPUT_ROOT = Path("OUTPUT")

# Target transect for diagnostic plot
TARGET_TRANSECT_ID = 1000


# === HELPER FUNCTIONS ===

def extract_year_from_path(path: Path) -> str:
    """Extract year from path like 'imagery/2016/20160122/file.tif'"""
    for part in path.parts:
        if re.match(r'^\d{4}$', part):
            return part
        if re.match(r'^\d{8}$', part):
            return part[:4]
    return "unknown"


def group_rasters_by_year(imagery_root: Path) -> dict:
    """Find all 4-band rasters and group by year."""
    import rasterio
    
    all_files = (
        list(imagery_root.rglob("*.tif")) + 
        list(imagery_root.rglob("*.tiff")) + 
        list(imagery_root.rglob("*.jp2"))
    )
    
    by_year = defaultdict(list)
    skipped = []
    
    for path in all_files:
        try:
            with rasterio.open(path) as src:
                if src.count >= 4:
                    year = extract_year_from_path(path)
                    by_year[year].append(path)
                else:
                    skipped.append((path, f"{src.count} bands"))
        except Exception as e:
            skipped.append((path, str(e)))
    
    print(f"Found {sum(len(v) for v in by_year.values())} usable rasters across {len(by_year)} years")
    print(f"Skipped {len(skipped)} files (3-band or unreadable)")
    
    return dict(by_year)


def find_nearest_transect(results: list, target_id: int) -> dict:
    """
    Find the result with TransectID closest to target_id.
    
    Returns the result dict, or None if results is empty.
    """
    if not results:
        return None
    
    # Extract transect IDs as integers
    def get_tid(result):
        tid = result.get('transect_id', 0)
        try:
            return int(tid)
        except (ValueError, TypeError):
            return 0
    
    # Find closest to target
    closest = min(results, key=lambda r: abs(get_tid(r) - target_id))
    return closest


def plot_spectral_profile(result: dict, output_path: Path, year: str):
    """
    Create a simple spectral profile plot showing R, G, B, NIR values vs distance.
    Includes a marker for detected shell line location if present.
    
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
    
    # Plot each band with distinct colors
    ax.plot(distance, data['red'], color='#e41a1c', linewidth=1.5, label='Red', alpha=0.8)
    ax.plot(distance, data['green'], color='#4daf4a', linewidth=1.5, label='Green', alpha=0.8)
    ax.plot(distance, data['blue'], color='#377eb8', linewidth=1.5, label='Blue', alpha=0.8)
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
        y_max = max(data[['red', 'green', 'blue', 'nir']].max())
        ax.annotate(f'conf: {shell_conf:.2f}', 
                    xy=(shell_dist, y_max * 0.95),
                    xytext=(shell_dist + 5, y_max * 0.95),
                    fontsize=9, color='#ff7f00',
                    ha='left', va='top')
    
    # Labels and title
    ax.set_xlabel('Distance from West (m)', fontsize=11)
    ax.set_ylabel('DN Value', fontsize=11)
    
    # Title indicates if shell line was detected
    detection_status = "detected" if shell_line else "no detection"
    ax.set_title(f'Spectral Profile - Year {year}, Transect {transect_id} ({detection_status})', 
                 fontsize=12, fontweight='bold')
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Grid for readability
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.set_axisbelow(True)
    
    # Set reasonable y-axis limits (handle 8-bit and 16-bit imagery)
    y_max = max(data[['red', 'green', 'blue', 'nir']].max())
    ax.set_ylim(0, y_max * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved spectral profile: {output_path.name} (Transect {transect_id}, {detection_status})")
    return output_path


def merge_summaries(summary_files: list, output_path: Path, year: str):
    """
    Merge multiple summary JSON files into one year-level summary.
    
    Handles duplicate TransectIDs by keeping the one with higher confidence transitions.
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
        "total_transitions": 0
    }
    
    # Track transects by ID to handle duplicates
    transects_by_id = {}
    
    for summary_file in summary_files:
        with open(summary_file) as f:
            data = json.load(f)
        
        # Merge metadata (keep track of all raster dirs)
        if "processing_info" in data.get("metadata", {}):
            raster_dir = data["metadata"]["processing_info"].get("raster_dir", "unknown")
            merged["metadata"]["processing_info"][raster_dir] = {
                "num_transects": data["metadata"].get("num_transects", 0)
            }
        
        # Merge transects
        for t in data.get("transects", []):
            tid = t["transect_id"]
            
            # If we've seen this transect, keep the one with better detection
            if tid in transects_by_id:
                existing = transects_by_id[tid]
                
                # Compare: prefer one with transitions, or higher confidence
                existing_conf = max([tr["confidence"] for tr in existing["transitions"]], default=0)
                new_conf = max([tr["confidence"] for tr in t["transitions"]], default=0)
                
                if new_conf > existing_conf:
                    transects_by_id[tid] = t
            else:
                transects_by_id[tid] = t
    
    # Build final transect list (sorted by ID)
    merged["transects"] = [
        transects_by_id[tid] 
        for tid in sorted(transects_by_id.keys(), key=lambda x: int(x))
    ]
    
    # Recalculate overall stats
    for t in merged["transects"]:
        merged["total_transitions"] += t.get("num_transitions", 0)
        for cls, count in t.get("class_distribution", {}).items():
            merged["overall_class_distribution"][cls] += count
    
    # Convert defaultdict to regular dict for JSON
    merged["overall_class_distribution"] = dict(merged["overall_class_distribution"])
    
    # Write merged summary
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"  Merged {len(summary_files)} summaries -> {output_path.name}")
    print(f"    Total transects: {len(merged['transects'])}")
    print(f"    Total transitions: {merged['total_transitions']}")
    
    return output_path


def export_shellline_geojson(summary_file: Path, transect_file: Path, output_file: Path):
    """Convert summary.json transitions to a LineString GeoJSON."""
    
    with open(summary_file) as f:
        summary = json.load(f)
    
    transects = gpd.read_file(transect_file)
    transect_lookup = {str(row.TransectID): row.geometry for _, row in transects.iterrows()}
    
    # Collect shell line points
    points_with_order = []
    
    for t in summary["transects"]:
        tid = t["transect_id"]
        
        for trans in t["transitions"]:
            if trans["type"] != "dry_wet_derivative":
                continue
            
            if tid not in transect_lookup:
                continue
            
            line = transect_lookup[tid]
            distance = min(trans["distance"], line.length)
            point = line.interpolate(distance)
            
            points_with_order.append({
                "transect_id": int(tid),
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
    
    # Include year from summary metadata if available
    year = summary.get("metadata", {}).get("year", "unknown")
    
    gdf = gpd.GeoDataFrame([{
        "geometry": shoreline,
        "year": year,
        "num_points": len(coords),
        "avg_confidence": avg_confidence
    }], crs=transects.crs)
    
    gdf.to_file(output_file, driver="GeoJSON")
    print(f"  Exported shellline: {len(coords)} vertices, avg confidence: {avg_confidence:.3f}")
    
    return output_file


# === MAIN ===

def main():
    print("=" * 60)
    print("BATCH PROCESSING BY YEAR")
    print("=" * 60)
    
    # Group rasters by year
    by_year = group_rasters_by_year(IMAGERY_ROOT)
    
    for year in sorted(by_year.keys()):
        raster_paths = by_year[year]
        print()
        print("=" * 60)
        print(f"YEAR: {year} ({len(raster_paths)} rasters)")
        print("=" * 60)
        
        # Create output directory
        output_dir = OUTPUT_ROOT / year
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get unique parent directories (date folders) for these rasters
        raster_dirs = sorted(set(p.parent for p in raster_paths))
        
        # Track summary files created for this year
        date_summaries = []
        
        # Track the best result for target transect (for diagnostic plot)
        best_target_result = None
        
        # Process each date folder within the year
        for raster_dir in raster_dirs:
            date_str = raster_dir.name
            print(f"\n  Processing: {raster_dir.relative_to(IMAGERY_ROOT)}")
            
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
                # (it will have a timestamp close to now)
                latest_summary = max(
                    output_dir.glob("summary_*.json"),
                    key=lambda p: p.stat().st_mtime
                )
                date_summaries.append(latest_summary)
                
                # Look for target transect (1000 or nearest) for diagnostic plot
                candidate = find_nearest_transect(results, TARGET_TRANSECT_ID)
                if candidate:
                    candidate_tid = int(candidate.get('transect_id', 0))
                    
                    # Keep if: we don't have one yet, OR this one is closer to target
                    if best_target_result is None:
                        best_target_result = candidate
                    else:
                        current_tid = int(best_target_result.get('transect_id', 0))
                        if abs(candidate_tid - TARGET_TRANSECT_ID) < abs(current_tid - TARGET_TRANSECT_ID):
                            best_target_result = candidate
                
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
        
        # Merge all date summaries into year summary
        if date_summaries:
            print(f"\n  Merging {len(date_summaries)} date summaries...")
            
            merged_summary = output_dir / f"summary_{year}.json"
            merge_summaries(date_summaries, merged_summary, year)
            
            # Export shellline from merged summary
            shellline_file = output_dir / f"shellline_{year}.geojson"
            export_shellline_geojson(merged_summary, TRANSECT_FILE, shellline_file)
            
            # Generate diagnostic spectral profile plot
            if best_target_result:
                profile_file = output_dir / f"spectral_profile_{year}.png"
                plot_spectral_profile(best_target_result, profile_file, year)
            else:
                print(f"  Warning: No transect data available for diagnostic plot")
            
            # Optionally: clean up individual date summaries
            for sf in date_summaries:
                 sf.unlink()
        
        print(f"\n  Year {year} complete.")

    print()
    print("=" * 60)
    print("BATCH COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()