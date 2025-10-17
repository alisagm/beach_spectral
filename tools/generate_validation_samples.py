"""
Validation sample generation script for manual annotation and classifier testing.

This script randomly selects transects and generates:
- Clean spectral plots for manual annotation
- Classified plots for comparison
- Spectral data CSV for validation

Used to create ground truth datasets for validating classifier performance.
"""

import logging
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict
from datetime import datetime

import numpy as np
import pandas as pd

from spectral_classifier.config import VERBOSE, DEFAULT_BOUNDARY_TYPES
from spectral_classifier.data_io import (
    build_raster_index,
    load_transects,
    reproject_if_needed,
    find_overlapping_rasters
)
from spectral_classifier.sampler import sample_transect, validate_spectral_data
from spectral_classifier.features import SpectralFeatures
from spectral_classifier.classifier import LandcoverClassifier
from spectral_classifier.transition import TransitionDetector
from spectral_classifier.visualization import plot_transect_analysis, plot_spectral_only
from spectral_classifier.utils import (
    setup_logging,
    validate_output_directory,
    detect_transect_direction
)

logger = logging.getLogger(__name__)


def generate_training_sample(
    raster_dir: Path,
    transect_geojson: Path,
    output_dir: Path,
    num_samples: int = 10,
    seed: int = None,
    verbose: bool = VERBOSE,
    boundary_types: str = DEFAULT_BOUNDARY_TYPES
) -> Dict:
    """
    Generate training data by randomly sampling transects.

    Args:
        raster_dir: Directory containing GeoTIFF rasters
        transect_geojson: Path to GeoJSON file with transects
        output_dir: Directory to save outputs
        num_samples: Number of transects to randomly sample
        seed: Random seed for reproducibility
        verbose: Enable verbose logging
        boundary_types: Which boundary types to detect
            - 'shore_only': BEACH_DRY->BEACH_WET only (default)
            - 'waterline': Shore + BEACH_WET->WATER boundaries
            - 'all': All boundaries including VEG_DUNES->BEACH_DRY

    Returns:
        Dictionary with sampling metadata and results
    """
    # Setup
    output_dir = validate_output_directory(output_dir)
    log_file = output_dir / 'training_data.log'
    setup_logging(verbose=verbose, log_file=log_file)

    logger.info("=" * 60)
    logger.info("TRAINING DATA GENERATION")
    logger.info("=" * 60)
    logger.info(f"Boundary detection mode: {boundary_types}")

    # Create output subdirectories
    clean_dir = output_dir / 'clean_plots'
    classified_dir = output_dir / 'classified_plots'
    clean_dir.mkdir(exist_ok=True)
    classified_dir.mkdir(exist_ok=True)

    # Set random seed for reproducibility
    if seed is None:
        seed = random.randint(0, 999999)
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed: {seed}")

    # Step 1: Build raster spatial index
    logger.info("Step 1: Building raster spatial index...")
    raster_index = build_raster_index(raster_dir)

    # Step 2: Load transects
    logger.info("Step 2: Loading transects...")
    transects = load_transects(transect_geojson)
    total_transects = len(transects)
    logger.info(f"Total transects available: {total_transects}")

    # Validate sample size
    if num_samples > total_transects:
        logger.warning(
            f"Requested {num_samples} samples but only {total_transects} "
            f"transects available. Using all transects."
        )
        num_samples = total_transects

    # Step 3: Random sampling
    logger.info(f"Step 3: Randomly sampling {num_samples} transects...")
    sample_indices = random.sample(range(total_transects), num_samples)
    sample_indices.sort()  # Sort for easier tracking
    sampled_transects = transects.iloc[sample_indices].copy()

    logger.info(f"Selected transect indices: {sample_indices}")
    logger.info(f"Selected transect IDs: {sampled_transects['TransectID'].tolist()}")

    # Step 4: CRS handling
    logger.info("Step 4: Checking CRS compatibility...")
    sampled_transects = reproject_if_needed(sampled_transects, raster_index.crs)

    # Detect transect direction
    first_transect = sampled_transects.iloc[0]
    direction = detect_transect_direction(first_transect.geometry)
    logger.info(f"Detected transect direction: {direction}")

    # Step 5: Process each sampled transect
    logger.info(f"Step 5: Processing {num_samples} sampled transects...")
    results = []
    all_spectral_data = []

    for idx, (_, transect_row) in enumerate(sampled_transects.iterrows(), 1):
        transect_id = transect_row.TransectID

        logger.info(f"[{idx}/{num_samples}] Processing transect {transect_id}")

        try:
            # Process transect
            result = process_transect_for_training(
                transect_row,
                raster_index,
                direction,
                boundary_types
            )

            # Generate clean plot (for manual annotation)
            clean_path = plot_spectral_only(
                transect_id=transect_id,
                data=result['data'],
                output_dir=clean_dir,
                direction=direction
            )

            # Generate classified plot (for comparison)
            classified_path = plot_transect_analysis(
                result=result,
                output_dir=classified_dir,
                show_transitions=True
            )

            # Store result
            result['clean_plot'] = str(clean_path)
            result['classified_plot'] = str(classified_path)
            results.append(result)

            # Collect spectral data for CSV export
            # Note: TransectID is already in the data from sample_transect()
            all_spectral_data.append(result['data'].copy())

            logger.info(f"  [OK] Generated plots for transect {transect_id}")

        except Exception as e:
            logger.error(
                f"Failed to process transect {transect_id}: {e}",
                exc_info=True
            )
            continue

    if not results:
        raise RuntimeError("No transects were successfully processed")

    logger.info(f"Successfully processed {len(results)}/{num_samples} transects")

    # Step 6: Export spectral data to CSV
    logger.info("Step 6: Exporting spectral data...")
    combined_data = pd.concat(all_spectral_data, ignore_index=True)
    csv_path = output_dir / 'sampled_spectral_data.csv'
    combined_data.to_csv(csv_path, index=False)
    logger.info(f"Saved spectral data to {csv_path}")

    # Step 7: Export metadata
    logger.info("Step 7: Exporting metadata...")
    metadata = {
        'generation_timestamp': datetime.now().isoformat(),
        'random_seed': seed,
        'num_samples_requested': num_samples,
        'num_samples_processed': len(results),
        'total_transects_available': total_transects,
        'sampled_indices': sample_indices,
        'sampled_transect_ids': [r['transect_id'] for r in results],
        'direction': direction,
        'raster_dir': str(raster_dir),
        'transect_file': str(transect_geojson),
        'raster_crs': str(raster_index.crs),
        'output_directories': {
            'clean_plots': str(clean_dir),
            'classified_plots': str(classified_dir)
        },
        'transect_summary': [
            {
                'transect_id': r['transect_id'],
                'num_points': len(r['data']),
                'distance_range_m': [
                    float(r['data']['distance'].min()),
                    float(r['data']['distance'].max())
                ],
                'class_distribution': r['landcover']['predicted_class'].value_counts().to_dict(),
                'num_transitions': len(r['transitions'])
            }
            for r in results
        ]
    }

    json_path = output_dir / 'training_sample_info.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Sampled transects: {len(results)}")
    print(f"Random seed: {seed}")
    print(f"\nOutput locations:")
    print(f"  Clean plots:      {clean_dir}")
    print(f"  Classified plots: {classified_dir}")
    print(f"  Spectral data:    {csv_path}")
    print(f"  Metadata:         {json_path}")
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("1. Review clean plots in 'clean_plots/' directory")
    print("2. Manually annotate regions on printed/digital copies")
    print("3. Record annotations in CSV format (distance ranges + classes)")
    print("4. Use annotations for threshold tuning and validation")
    print("=" * 60 + "\n")

    return metadata


def process_transect_for_training(
    transect_row,
    raster_index,
    direction: str = 'west_to_east',
    boundary_types: str = DEFAULT_BOUNDARY_TYPES
) -> Dict:
    """
    Process a single transect through the full pipeline.

    Args:
        transect_row: Row from transects GeoDataFrame
        raster_index: RasterIndex object
        direction: Transect direction
        boundary_types: Which boundary types to detect
            - 'shore_only': BEACH_DRY->BEACH_WET only (default)
            - 'waterline': Shore + BEACH_WET->WATER boundaries
            - 'all': All boundaries including VEG_DUNES->BEACH_DRY

    Returns:
        Dictionary with analysis results
    """
    transect_id = transect_row.TransectID

    # Find intersecting rasters
    overlapping_rasters = find_overlapping_rasters(
        transect_row.geometry,
        raster_index
    )

    logger.debug(f"  Found {len(overlapping_rasters)} overlapping rasters")

    # Sample spectral values
    spectral_data = sample_transect(transect_row, overlapping_rasters, direction)
    validate_spectral_data(spectral_data)

    logger.debug(f"  Sampled {len(spectral_data)} points")

    # Extract features
    feature_extractor = SpectralFeatures(spectral_data)
    features = feature_extractor.compute_all()

    logger.debug(f"  Computed {len(features.columns)} features")

    # Classify landcover
    classifier = LandcoverClassifier()
    landcover = classifier.classify(features)

    # Apply spatial smoothing (basic median filter)
    landcover = classifier.apply_spatial_smoothing(landcover)

    # PHASE 6D: Conditionally apply monotonic smoothing (disabled by default due to class collapse)
    from spectral_classifier.config import THRESHOLDS
    if THRESHOLDS.get('enable_monotonic_smoothing', False):
        landcover = classifier.apply_monotonic_smoothing(landcover)
        logger.debug("  Monotonic smoothing applied")
    else:
        logger.debug("  Monotonic smoothing skipped (disabled in config)")

    # Log class distribution
    class_counts = landcover['predicted_class'].value_counts()
    logger.debug(f"  Classification: {dict(class_counts)}")

    # Detect transitions
    detector = TransitionDetector()
    all_transitions = detector.find_transitions(features, landcover)

    # PHASE 7B: Extract shell line for boundary correction
    # Phase 7A guarantees at least one shell line detection
    shell_lines = [t for t in all_transitions
                   if TransitionDetector.is_shore_boundary(t)]

    if shell_lines:
        # Take the first (and should be only) shell line
        shell_line = shell_lines[0]
        shell_line_distance = shell_line['distance']

        # Apply boundary-aware classification correction
        landcover = classifier.apply_boundary_aware_correction(
            landcover,
            shell_line_distance,
            correction_mode='strict',  # Can be made configurable
            correction_buffer=2.0
        )

        # Re-log class distribution after correction
        class_counts_after = landcover['predicted_class'].value_counts()
        logger.debug(f"  Classification after boundary correction: {dict(class_counts_after)}")
    else:
        logger.warning(f"  No shell line detected for transect {transect_id} - skipping boundary correction")

    # Filter transitions by boundary type
    transitions = TransitionDetector.filter_by_boundary_type(all_transitions, boundary_types)

    logger.debug(f"  Detected {len(all_transitions)} transitions, "
                f"returned {len(transitions)} (mode: {boundary_types})")

    # Return results
    return {
        'transect_id': transect_id,
        'data': spectral_data,
        'features': features,
        'landcover': landcover,
        'transitions': transitions,
        'direction': direction
    }


def main():
    """Command-line interface for validation sample generation."""
    parser = argparse.ArgumentParser(
        description='Generate validation samples for spectral classifier testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m tools.generate_validation_samples \\
    --rasters C:\\Users\\alisa\\Desktop\\SIP\\input_data\\NAIP22 \\
    --transects C:\\Users\\alisa\\Desktop\\SIP\\input_data\\PAIS_measured \\
    --output data/output/321197_run_001 \\
    --num-samples 10 \\
    --seed 321197

This will:
  1. Randomly sample 10 transects
  2. Generate clean spectral plots for manual annotation
  3. Generate classified plots for comparison
  4. Export spectral data and metadata
        """
    )

    parser.add_argument(
        '--rasters',
        type=Path,
        required=True,
        help='Directory containing GeoTIFF raster files'
    )

    parser.add_argument(
        '--transects',
        type=Path,
        required=True,
        help='Path to GeoJSON file with transect LineStrings'
    )

    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Directory to save training data outputs'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of transects to randomly sample (default: 10)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: random)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug logging'
    )

    parser.add_argument(
        '--boundary-types',
        type=str,
        choices=['shore_only', 'waterline', 'all'],
        default=DEFAULT_BOUNDARY_TYPES,
        help=(
            'Which boundary types to detect and return '
            '(default: shore_only for BEACH_DRY->BEACH_WET swash line only)'
        )
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.rasters.exists():
        print(f"ERROR: Raster directory does not exist: {args.rasters}")
        return 1

    if not args.transects.exists():
        print(f"ERROR: Transect file does not exist: {args.transects}")
        return 1

    # Run training data generation
    try:
        metadata = generate_training_sample(
            raster_dir=args.rasters,
            transect_geojson=args.transects,
            output_dir=args.output,
            num_samples=args.num_samples,
            seed=args.seed,
            verbose=args.verbose,
            boundary_types=args.boundary_types
        )

        return 0

    except Exception as e:
        logger.error(f"Training data generation failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
