#!/usr/bin/env python3
"""
Spectral Shoreline Detection System - Compute Step

Samples spectral profiles along all transects and extracts feature statistics.
Writes two Parquet checkpoints per year that are consumed by interpret.py and plot.py.

Usage:
    python run.py --imagery-root /path/to/imagery --output /path/to/output
    python run.py --year 2016 --imagery-root /path/to/imagery

Output:
    For each year processed:
    - OUTPUT/{year}/profiles_{year}.parquet  <- raw band samples + coordinates
    - OUTPUT/{year}/features_{year}.parquet  <- derived feature arrays
    - OUTPUT/{year}/compute.log              <- debug log

    Existing checkpoints are skipped unless --force is passed.
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List

from .compute import run_compute
from spectral_classifier.utils import (
    setup_logging,
    group_rasters_by_year,
    validate_output_directory,
)
from spectral_classifier.config import BAND_CONFIG_PATH


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Spectral Shoreline Detection System — Compute Step',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--imagery-root', '-i',
        type=Path,
        required=True,
        help='Root directory containing imagery organised by year'
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

    parser.add_argument(
        '--band-config',
        type=Path,
        default=BAND_CONFIG_PATH,
        help=f'Path to band_config.json (default: {BAND_CONFIG_PATH})'
    )

    parser.add_argument(
        '--year', '-y',
        type=str,
        nargs='*',
        help='Specific year(s) to process (default: all years found)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-run compute even if Parquet checkpoints already exist'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose debug logging'
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
    band_config_path: Path,
    verbose: bool,
    force: bool,
) -> bool:
    """
    Run the compute step for a single year.

    Skips years whose Parquet checkpoints already exist unless force=True.

    Args:
        year:             Year string (e.g. '2016').
        raster_paths:     Raster tile paths for this year.
        transect_file:    Path to transects GeoJSON.
        output_root:      Root output directory.
        band_config_path: Path to band_config.json.
        verbose:          Enable debug logging.
        force:            Overwrite existing checkpoints.

    Returns:
        True if processing succeeded or was skipped, False on error.
    """
    year_output_dir = output_root / year
    profiles_path = year_output_dir / f"profiles_{year}.parquet"
    features_path = year_output_dir / f"features_{year}.parquet"

    print(f"\n{'='*60}")
    print(f"Year: {year}  ({len(raster_paths)} rasters)")
    print(f"Output: {year_output_dir}")
    print(f"{'='*60}")

    # Skip if both checkpoints exist and --force not set.
    if not force and profiles_path.exists() and features_path.exists():
        print(f"  Skipping — checkpoints exist (use --force to overwrite)")
        return True

    try:
        profiles_path, features_path = run_compute(
            year=year,
            raster_paths=raster_paths,
            transect_geojson=transect_file,
            output_dir=year_output_dir,
            band_config_path=band_config_path,
            verbose=verbose,
        )
        print(f"  Profiles: {profiles_path}")
        print(f"  Features: {features_path}")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        logging.exception(f"Compute failed for year {year}")
        return False


def main():
    """Main entry point."""
    args = parse_args()

    # Top-level log captures batch-level events; per-year logs are in year dirs.
    validate_output_directory(args.output)
    setup_logging(
        verbose=args.verbose,
        log_file=args.output / 'compute_batch.log'
    )

    print("\n" + "="*60)
    print("SPECTRAL SHORELINE DETECTION — COMPUTE STEP")
    print("="*60)
    print(f"Imagery root: {args.imagery_root}")
    print(f"Transects:    {args.transects}")
    print(f"Band config:  {args.band_config}")
    print(f"Output:       {args.output}")

    # Validate inputs.
    if not args.imagery_root.exists():
        print(f"\nERROR: Imagery root not found: {args.imagery_root}")
        sys.exit(1)

    if not args.transects.exists():
        print(f"\nERROR: Transects file not found: {args.transects}")
        sys.exit(1)

    if not args.band_config.exists():
        print(f"\nERROR: band_config.json not found: {args.band_config}")
        sys.exit(1)

    # Discover imagery grouped by year.
    print("\nScanning imagery...")
    years_data = group_rasters_by_year(args.imagery_root, recursive=True)

    if not years_data:
        print("\nERROR: No imagery found")
        sys.exit(1)

    print(f"Found {len(years_data)} years: {sorted(years_data.keys())}")

    # Filter to requested years.
    if args.year:
        years_to_process = [y for y in args.year if y in years_data]
        missing = [y for y in args.year if y not in years_data]
        if missing:
            print(f"\nWARNING: Requested years not found in imagery: {missing}")
        if not years_to_process:
            print(f"Available years: {sorted(years_data.keys())}")
            sys.exit(1)
    else:
        years_to_process = sorted(years_data.keys())

    print(f"Years to process: {years_to_process}")

    # Dry run — show what would be processed without running.
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

    # Process each year.
    start_time = datetime.now()
    results = {}

    for year in years_to_process:
        results[year] = process_year(
            year=year,
            raster_paths=years_data[year],
            transect_file=args.transects,
            output_root=args.output,
            band_config_path=args.band_config,
            verbose=args.verbose,
            force=args.force,
        )

    # Batch summary.
    elapsed = datetime.now() - start_time
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful

    print("\n" + "="*60)
    print("COMPUTE COMPLETE")
    print("="*60)
    print(f"Total time:  {elapsed}")
    print(f"Successful:  {successful}/{len(results)}")

    if failed > 0:
        print(f"Failed years: {[y for y, v in results.items() if not v]}")
        sys.exit(1)


if __name__ == '__main__':
    main()