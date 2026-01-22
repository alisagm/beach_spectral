"""
Diagnostic script: Analyze band detection consistency across tiles within each year.

This script checks whether detect_band_configuration() gives consistent results
for all tiles belonging to the same year. If tiles from the same source are
being classified differently (some as CIR, some as RGB), this will identify
where and why.

Usage:
    python diagnose_band_detection.py [imagery_root]
    
Output:
    - Console summary of inconsistencies
    - CSV with per-tile statistics for further analysis
"""

import sys
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import rasterio

# === CONFIGURATION ===
# Update this if running standalone
DEFAULT_IMAGERY_ROOT = Path(r"C:\Users\alisa\Desktop\SIP\git\beach_spectral\PAIS_shorelines\imagery")
SUPPORTED_EXTENSIONS = ['.tif', '.tiff', '.jp2', '.sid', '.ers', '.ecw']

# Band detection parameters (should match config.py)
CIR_VARIANCE_RATIO_THRESHOLD = 1.05
SAMPLE_SIZE = 10000
MIN_VALID_SAMPLES = 100


def extract_year_from_path(path: Path) -> str:
    """Extract year from path like 'imagery/2016/20160122/file.tif'"""
    for part in path.parts:
        if re.match(r'^\d{4}$', part):
            return part
        if re.match(r'^\d{6}$', part):
            return part[:4]
        if re.match(r'^\d{8}$', part):
            return part[:4]
    return "unknown"


def compute_band_statistics(raster_path: Path) -> dict:
    """
    Compute the statistics used by detect_band_configuration().
    
    Returns dict with:
        - band_count: number of bands
        - band1_std: standard deviation of band 1
        - band2_std: standard deviation of band 2
        - variance_ratio: band1_std / band2_std
        - classification: '4band', 'cir', or 'rgb'
        - error: error message if any
    """
    result = {
        'path': str(raster_path),
        'filename': raster_path.name,
        'year': extract_year_from_path(raster_path),
        'band_count': None,
        'band1_std': None,
        'band2_std': None,
        'variance_ratio': None,
        'valid_samples': None,
        'classification': None,
        'error': None
    }
    
    try:
        with rasterio.open(raster_path) as src:
            result['band_count'] = src.count
            
            # 4-band is straightforward
            if src.count >= 4:
                result['classification'] = '4band'
                return result
            
            if src.count < 3:
                result['error'] = f"Only {src.count} bands"
                result['classification'] = 'invalid'
                return result
            
            # 3-band: compute variance ratio
            height, width = src.height, src.width
            
            np.random.seed(42)  # Reproducible
            n_samples = min(SAMPLE_SIZE, height * width)
            sample_rows = np.random.randint(0, height, n_samples)
            sample_cols = np.random.randint(0, width, n_samples)
            
            band1 = src.read(1)
            band2 = src.read(2)
            
            band1_values = []
            band2_values = []
            
            for r, c in zip(sample_rows, sample_cols):
                v1 = band1[r, c]
                v2 = band2[r, c]
                
                # Skip nodata
                if src.nodata is not None and (v1 == src.nodata or v2 == src.nodata):
                    continue
                if v1 == 0 or v2 == 0:
                    continue
                    
                band1_values.append(v1)
                band2_values.append(v2)
            
            result['valid_samples'] = len(band1_values)
            
            if len(band1_values) < MIN_VALID_SAMPLES:
                result['error'] = f"Only {len(band1_values)} valid samples"
                result['classification'] = 'rgb'  # Default fallback
                return result
            
            band1_std = np.std(band1_values)
            band2_std = np.std(band2_values)
            variance_ratio = band1_std / (band2_std + 1e-6)
            
            result['band1_std'] = round(band1_std, 3)
            result['band2_std'] = round(band2_std, 3)
            result['variance_ratio'] = round(variance_ratio, 4)
            
            if variance_ratio > CIR_VARIANCE_RATIO_THRESHOLD:
                result['classification'] = 'cir'
            else:
                result['classification'] = 'rgb'
                
    except Exception as e:
        result['error'] = str(e)
        result['classification'] = 'error'
    
    return result


def find_all_rasters(imagery_root: Path) -> list:
    """Find all raster files recursively."""
    all_files = []
    for ext in SUPPORTED_EXTENSIONS:
        all_files.extend(imagery_root.rglob(f"*{ext}"))
    return sorted(all_files)


def analyze_consistency(results: list) -> dict:
    """
    Analyze results for consistency within each year.
    
    Returns dict with:
        - years_consistent: list of years where all tiles agree
        - years_inconsistent: dict of year -> {classifications, tiles}
        - summary: overall statistics
    """
    # Group by year
    by_year = defaultdict(list)
    for r in results:
        by_year[r['year']].append(r)
    
    consistent = []
    inconsistent = {}
    
    for year in sorted(by_year.keys()):
        tiles = by_year[year]
        
        # Get unique classifications (excluding errors)
        classifications = set(
            t['classification'] for t in tiles 
            if t['classification'] not in ('error', 'invalid')
        )
        
        if len(classifications) <= 1:
            consistent.append(year)
        else:
            # Inconsistency found - gather details
            inconsistent[year] = {
                'classifications': list(classifications),
                'tiles': tiles,
                'breakdown': defaultdict(list)
            }
            for t in tiles:
                inconsistent[year]['breakdown'][t['classification']].append(t)
    
    return {
        'years_consistent': consistent,
        'years_inconsistent': inconsistent,
        'total_years': len(by_year),
        'total_tiles': len(results)
    }


def print_report(analysis: dict, results: list):
    """Print human-readable diagnostic report."""
    print("=" * 70)
    print("BAND DETECTION CONSISTENCY REPORT")
    print("=" * 70)
    print(f"Total tiles analyzed: {analysis['total_tiles']}")
    print(f"Total years: {analysis['total_years']}")
    print(f"Consistent years: {len(analysis['years_consistent'])}")
    print(f"Inconsistent years: {len(analysis['years_inconsistent'])}")
    print()
    
    if not analysis['years_inconsistent']:
        print("✓ All years have consistent band detection!")
        print()
        # Still show the classification breakdown
        by_year = defaultdict(list)
        for r in results:
            by_year[r['year']].append(r)
        
        print("Classification by year:")
        for year in sorted(by_year.keys()):
            tiles = by_year[year]
            cls = tiles[0]['classification'] if tiles else 'unknown'
            print(f"  {year}: {cls} ({len(tiles)} tiles)")
        return
    
    print("=" * 70)
    print("INCONSISTENT YEARS (tiles classified differently)")
    print("=" * 70)
    
    for year, info in sorted(analysis['years_inconsistent'].items()):
        print(f"\n### YEAR {year} ###")
        print(f"Classifications found: {info['classifications']}")
        print()
        
        for cls, tiles in info['breakdown'].items():
            print(f"  [{cls.upper()}] - {len(tiles)} tiles:")
            for t in tiles[:5]:  # Show first 5
                ratio_str = f"ratio={t['variance_ratio']:.4f}" if t['variance_ratio'] else "N/A"
                print(f"    - {t['filename']} ({ratio_str})")
            if len(tiles) > 5:
                print(f"    ... and {len(tiles) - 5} more")
        
        # Show variance ratio distribution for this year
        ratios = [t['variance_ratio'] for t in info['tiles'] if t['variance_ratio'] is not None]
        if ratios:
            print(f"\n  Variance ratio stats for {year}:")
            print(f"    Min:    {min(ratios):.4f}")
            print(f"    Max:    {max(ratios):.4f}")
            print(f"    Mean:   {np.mean(ratios):.4f}")
            print(f"    Median: {np.median(ratios):.4f}")
            print(f"    Threshold: {CIR_VARIANCE_RATIO_THRESHOLD}")
            
            # Show how many are near the threshold
            near_threshold = [r for r in ratios if abs(r - CIR_VARIANCE_RATIO_THRESHOLD) < 0.1]
            if near_threshold:
                print(f"    ⚠ {len(near_threshold)} tiles have ratio within ±0.1 of threshold!")


def save_csv(results: list, output_path: Path):
    """Save detailed results to CSV for further analysis."""
    import csv
    
    fieldnames = [
        'year', 'filename', 'band_count', 'classification',
        'variance_ratio', 'band1_std', 'band2_std', 'valid_samples', 'error', 'path'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x['year'], x['filename'])):
            writer.writerow({k: r.get(k) for k in fieldnames})
    
    print(f"\nDetailed results saved to: {output_path}")


def main():
    # Get imagery root from command line or use default
    if len(sys.argv) > 1:
        imagery_root = Path(sys.argv[1])
    else:
        imagery_root = DEFAULT_IMAGERY_ROOT
    
    print(f"Scanning: {imagery_root}")
    print()
    
    if not imagery_root.exists():
        print(f"ERROR: Directory not found: {imagery_root}")
        return 1
    
    # Find all rasters
    raster_files = find_all_rasters(imagery_root)
    print(f"Found {len(raster_files)} raster files")
    
    if not raster_files:
        print("No raster files found!")
        return 1
    
    # Analyze each file
    print("Analyzing band statistics...")
    results = []
    for i, raster_path in enumerate(raster_files):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(raster_files)}...", end='\r')
        results.append(compute_band_statistics(raster_path))
    print(f"  Processed {len(raster_files)}/{len(raster_files)} tiles")
    print()
    
    # Analyze consistency
    analysis = analyze_consistency(results)
    
    # Print report
    print_report(analysis, results)
    
    # Save CSV
    output_csv = Path("band_detection_diagnostic.csv")
    save_csv(results, output_csv)
    
    return 0 if not analysis['years_inconsistent'] else 1


if __name__ == "__main__":
    sys.exit(main())