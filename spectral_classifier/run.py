#!/usr/bin/env python3
"""
Spectral Shoreline Detection System - Main Entry Point

This script provides the primary interface for batch processing imagery
to detect shell lines (beach-water boundaries) using spectral analysis.

Usage:
    python run.py --imagery-root /path/to/imagery --output /path/to/output
    python run.py --config config.yaml
    python run.py --year 2020 --imagery-root /path/to/imagery

Example:
    python run.py \\
        --imagery-root ./PAIS_shorelines/imagery \\
        --transects ./INPUT/shorelineTransPais.json \\
        --output ./OUTPUT \\
        --year 2016

Output:
    For each year processed:
    - OUTPUT/{year}/shellline_{year}.geojson  <- PRIMARY OUTPUT
    - OUTPUT/{year}/summary_{year}.json       <- Processing metadata
    - OUTPUT/{year}/spectral_profile_{year}.png <- Diagnostic plot
    - OUTPUT/{year}/processing.log            <- Debug log
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Import from spectral_classifier package
from spectral_classifier.main import analyze_all_transects
from spectral_classifier.utils import (
    setup_logging,
    group_rasters_by_year,
    resolve_year_band_config,
    validate_output_directory,
)
from spectral_classifier.config import DEFAULT_BOUNDARY_TYPES


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Spectral Shoreline Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        '--imagery-root', '-i',
        type=Path,
        required=True,
        help='Root directory containing imagery organized by year'
    )
    
    parser.add_argument(
        '--transects', '-t',
        type=Path,
        default=Path('INPUT/shorelineTransPais.json'),
        help='Path to transects GeoJSON file (default: INPUT/shorelineTransPais.json)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('OUTPUT'),
        help='Output directory (default: OUTPUT)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--year', '-y',
        type=str,
        nargs='*',
        help='Specific year(s) to process (default: all years found)'
    )
    
    parser.add_argument(
        '--band-mode',
        choices=['auto', '4band', 'cir', 'rgb'],
        default='auto',
        help='Force band mode for 3-band imagery (default: auto-detect)'
    )
    
    parser.add_argument(
        '--boundary-types',
        choices=['shore_only', 'waterline', 'all'],
        default=DEFAULT_BOUNDARY_TYPES,
        help=f'Which boundaries to detect (default: {DEFAULT_BOUNDARY_TYPES})'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--diagnostic-transect',
        type=int,
        default=1000,
        help='Transect ID for diagnostic spectral plot (default: 1000)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List years/imagery found without processing'
    )
    
    return parser.parse_args()


def process_year(
    year: str,
    raster_paths: List[Path],
    transect_file: Path,
    output_root: Path,
    band_mode_override: Optional[str],
    boundary_types: str,
    verbose: bool,
    diagnostic_transect_id: int
) -> bool:
    """
    Process a single year's imagery.
    
    Args:
        year: Year string (e.g., '2020')
        raster_paths: List of raster files for this year
        transect_file: Path to transects GeoJSON
        output_root: Root output directory
        band_mode_override: Force band mode (None for auto)
        boundary_types: Which boundaries to detect
        verbose: Enable verbose logging
        diagnostic_transect_id: Transect ID for diagnostic plot
        
    Returns:
        True if processing succeeded, False otherwise
    """
    year_output_dir = output_root / year
    
    print(f"\n{'='*60}")
    print(f"Processing Year: {year}")
    print(f"  Rasters: {len(raster_paths)}")
    print(f"  Output: {year_output_dir}")
    print(f"{'='*60}")
    
    try:
        # Resolve band configuration for this year
        # Priority: 4band > cir > rgb (favors NIR-based detection)
        if band_mode_override and band_mode_override != 'auto':
            band_config = band_mode_override
            valid_paths = raster_paths
        else:
            band_config, valid_paths = resolve_year_band_config(raster_paths)
        
        print(f"  Band mode: {band_config}")
        print(f"  Valid rasters: {len(valid_paths)}/{len(raster_paths)}")
        
        if not valid_paths:
            print(f"  ERROR: No valid rasters for year {year}")
            return False
        
        # Create temporary directory with symlinks or use first raster's parent
        # For now, assume all rasters are in same directory structure
        raster_dir = valid_paths[0].parent
        
        # Run analysis
        results = analyze_all_transects(
            raster_dir=raster_dir,
            transect_geojson=transect_file,
            output_dir=year_output_dir,
            verbose=verbose,
            boundary_types=boundary_types,
            band_config_override=band_config if band_config != '4band' else None
        )
        
        print(f"  Processed: {len(results)} transects")
        
        # Count shell lines detected
        shell_line_count = sum(
            1 for r in results 
            if any('dry_wet' in t.get('type', '') for t in r.get('transitions', []))
        )
        print(f"  Shell lines: {shell_line_count}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        logging.exception(f"Failed to process year {year}")
        return False


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_file = args.output / 'batch_processing.log'
    setup_logging(verbose=args.verbose, log_file=log_file)
    
    print("\n" + "="*60)
    print("SPECTRAL SHORELINE DETECTION SYSTEM")
    print("="*60)
    print(f"Imagery root: {args.imagery_root}")
    print(f"Transects: {args.transects}")
    print(f"Output: {args.output}")
    print(f"Band mode: {args.band_mode}")
    print(f"Boundary types: {args.boundary_types}")
    
    # Validate inputs
    if not args.imagery_root.exists():
        print(f"\nERROR: Imagery root not found: {args.imagery_root}")
        sys.exit(1)
        
    if not args.transects.exists():
        print(f"\nERROR: Transects file not found: {args.transects}")
        sys.exit(1)
    
    # Group imagery by year
    print("\nScanning imagery...")
    years_data = group_rasters_by_year(args.imagery_root, recursive=True)
    
    if not years_data:
        print("\nERROR: No imagery found")
        sys.exit(1)
    
    print(f"Found {len(years_data)} years: {sorted(years_data.keys())}")
    
    # Filter to requested years
    if args.year:
        years_to_process = [y for y in args.year if y in years_data]
        if not years_to_process:
            print(f"\nERROR: Requested years not found: {args.year}")
            print(f"Available years: {sorted(years_data.keys())}")
            sys.exit(1)
    else:
        years_to_process = sorted(years_data.keys())
    
    print(f"\nYears to process: {years_to_process}")
    
    # Dry run - just show what would be processed
    if args.dry_run:
        print("\n--- DRY RUN ---")
        for year in years_to_process:
            paths = years_data[year]
            print(f"\n{year}: {len(paths)} rasters")
            for p in paths[:5]:
                print(f"  {p.name}")
            if len(paths) > 5:
                print(f"  ... and {len(paths) - 5} more")
        sys.exit(0)
    
    # Create output directory
    validate_output_directory(args.output)
    
    # Process each year
    start_time = datetime.now()
    results = {}
    
    for year in years_to_process:
        success = process_year(
            year=year,
            raster_paths=years_data[year],
            transect_file=args.transects,
            output_root=args.output,
            band_mode_override=args.band_mode if args.band_mode != 'auto' else None,
            boundary_types=args.boundary_types,
            verbose=args.verbose,
            diagnostic_transect_id=args.diagnostic_transect
        )
        results[year] = success
    
    # Summary
    elapsed = datetime.now() - start_time
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total time: {elapsed}")
    print(f"Successful: {successful}/{len(results)}")
    
    if failed > 0:
        print(f"Failed years: {[y for y, v in results.items() if not v]}")
        sys.exit(1)
    
    print("\nOutputs:")
    for year in years_to_process:
        if results[year]:
            geojson = args.output / year / f"shellline_{year}.geojson"
            if geojson.exists():
                print(f"  {geojson}")


if __name__ == '__main__':
    main()