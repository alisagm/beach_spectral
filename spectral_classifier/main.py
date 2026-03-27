"""
Main pipeline orchestration for spectral transect classification system.

Supports 3-band (RGB or CIR) and 4-band (RGBN) imagery with automatic
band configuration detection.
"""

import logging
import argparse
from pathlib import Path
from typing import List, Dict

from spectral_classifier.config import NUM_TRANSECTS_TO_VISUALIZE, VERBOSE, DEFAULT_BOUNDARY_TYPES, BAND_CONFIG_PATH
from .utils import (
    load_band_config,
    build_raster_index,
    load_transects,
    reproject_if_needed,
    find_overlapping_rasters,
    RasterIndex,
    setup_logging,
    export_results_to_csv,
    export_summary_json,
    select_representative_transects,
    validate_output_directory,
    calculate_processing_stats,
    print_processing_summary,
    ProgressTracker,
    export_shell_line_geojson,
    extract_year_from_path,
    resolve_band_indices
)
from .spectral import sample_transect, compute_all
from .transition import TransitionDetector
from .visualization.plotting import plot_transect_analysis
from .optional import LandcoverClassifier

logger = logging.getLogger(__name__)


def analyze_all_transects(
    raster_dir: Path,
    transect_geojson: Path,
    output_dir: Path,
    band_config_path: Path = BAND_CONFIG_PATH,
    num_visualize: int = NUM_TRANSECTS_TO_VISUALIZE,
    verbose: bool = VERBOSE,
    boundary_types: str = DEFAULT_BOUNDARY_TYPES,
) -> List[Dict]:
    """
    Main pipeline to analyze all transects.

    Supports both 3-band (RGB or CIR) and 4-band (RGBN) imagery.
    Band configuration is auto-detected per raster.

    Args:
        raster_dir: Directory containing raster files (.tif, .tiff, .jp2)
        transect_geojson: Path to GeoJSON file with transects
        output_dir: Directory to save outputs
        num_visualize: Number of transects to visualize
        verbose: Enable verbose logging
        boundary_types: Which boundary types to return
            - 'shore_only': BEACH_DRY->BEACH_WET only (swash/shell line)
            - 'waterline': Shore + BEACH_WET->WATER boundaries
            - 'all': All boundaries including VEG_DUNES->BEACH_DRY

    Returns:
        List of all transect analysis results
    """
    # Setup
    output_dir = validate_output_directory(output_dir)
    log_file = output_dir / 'processing.log'
    setup_logging(verbose=verbose, log_file=log_file)

    logger.info("=" * 60)
    logger.info("SPECTRAL TRANSECT CLASSIFICATION SYSTEM")
    logger.info("=" * 60)
    logger.info(f"Boundary detection mode: {boundary_types}")

    # Step 1: Build raster spatial index 
    logger.info("Step 1: Building raster spatial index...")
    print("Building raster spatial index...")
    raster_index = build_raster_index(raster_dir)
    year = extract_year_from_path(raster_dir)

    # Step 2: Load band configuration from band_config.json
    year_config = load_band_config(BAND_CONFIG_PATH, year)
    logger.info(f"Band config for {year}: {year_config}")
    
    # Print raster summary to console
    print(f"Found {len(raster_index)} valid rasters")
    if raster_index.skipped:
        print(f"Skipped {len(raster_index.skipped)} invalid rasters")

    # Step 3: Load transects and handle CRS
    logger.info("Step 2: Loading transects...")
    transects = load_transects(transect_geojson)

    logger.info("Step 3: Checking CRS compatibility...")
    transects = reproject_if_needed(transects, raster_index.crs)

    # Step 4: Process each transect with progress tracking
    logger.info(f"Step 4: Processing {len(transects)} transects...")
    print(f"\nProcessing {len(transects)} transects...")
    
    # Initialize progress tracker
    tracker = ProgressTracker(total_items=len(transects), item_noun="transect")
    
    all_results = []

    for idx, transect in transects.iterrows():
        # Update progress display
        tracker.update(idx + 1)
        
        try:
            result, skipped_points = process_single_transect(
                transect,
                raster_index,
                idx + 1,
                len(transects),
                year_config,
                boundary_types
            )
            all_results.append(result)
            tracker._processed_count += 1
            
            # Aggregate skipped points
            if skipped_points > 0:
                tracker.add_skipped("sample points (outside coverage)", skipped_points)

        except NoCoverageError:
            # Expected with gappy imagery - track quietly, don't log as error
            tracker.add_skipped("transects (no raster coverage)", 1)
            continue
            
        except Exception as e:
            logger.error(
                f"Failed to process transect {transect.TransectID}: {e}",
                exc_info=True
            )
            tracker.add_error(f"Transect {transect.TransectID}: {e}")
            continue

    # Finish progress display
    tracker.finish(success_count=len(all_results))
    
    if not all_results:
        raise RuntimeError("No transects were successfully processed")

    logger.info(f"Successfully processed {len(all_results)}/{len(transects)} transects")

    # Step 5: Export results
    logger.info("Step 5: Exporting results...")
    print("\nExporting results...")
    csv_path = export_results_to_csv(all_results, output_dir)

    processing_stats = calculate_processing_stats(all_results)
    json_path = export_summary_json(
        all_results,
        output_dir,
        processing_metadata={
            'raster_dir': str(raster_dir),
            'transect_file': str(transect_geojson),
            'raster_crs': str(raster_index.crs),
            'band_config': year_config
        }
    )

    # Export shell line GeoJSON (primary output)
    if year:
        geojson_path = export_shell_line_geojson(
            all_results,
            output_dir,
            year=year,
            source_crs=str(raster_index.crs)
        )
        if geojson_path:
            print(f"  Shell line: {geojson_path}")
    else:
        logger.warning("Year not specified - skipping shell line GeoJSON export")
        
    # Step 6: Generate visualizations for selected transects
    logger.info(f"Step 6: Generating visualizations for {num_visualize} transects...")
    print(f"Generating {num_visualize} diagnostic plots...")
    selected_transects = select_representative_transects(all_results, num_visualize)

    for result in selected_transects:
        try:
            plot_transect_analysis(result, output_dir)
        except Exception as e:
            logger.error(
                f"Failed to plot transect {result['transect_id']}: {e}",
                exc_info=True
            )
            tracker.add_error(f"Plot {result['transect_id']}: {e}")

    # Print summaries
    tracker.print_summary()
    print_processing_summary(processing_stats)

    logger.info("Processing complete!")
    logger.info(f"Results saved to: {output_dir}")

    return all_results


class NoCoverageError(Exception):
    """Raised when a transect has no raster coverage (expected with gappy imagery)."""
    pass


def process_single_transect(
    transect_row,
    raster_index: RasterIndex,
    current_idx: int,
    total: int,
    year_config: dict,
    boundary_types: str = DEFAULT_BOUNDARY_TYPES
) -> tuple:
    """
    Process a single transect through the analysis pipeline.

    Args:
        transect_row: Row from transects GeoDataFrame
        raster_index: RasterIndex object (includes band configurations)
        current_idx: Current transect number
        total: Total number of transects
        boundary_types: Which boundary types to return
            - 'shore_only': BEACH_DRY->BEACH_WET only (swash/shell line)
            - 'waterline': Shore + BEACH_WET->WATER boundaries
            - 'all': All boundaries including VEG_DUNES->BEACH_DRY

    Returns:
        Tuple of (result_dict, skipped_point_count)
        
    Raises:
        NoCoverageError: If transect has no raster coverage (expected, not a real error)
    """
    transect_id = transect_row.TransectID

    logger.debug(f"[{current_idx}/{total}] Processing transect {transect_id}")

    # Find intersecting rasters
    overlapping_rasters = find_overlapping_rasters(
        transect_row.geometry,
        raster_index
    )
    
    # Handle no coverage case (expected with gappy imagery)
    if not overlapping_rasters:
        raise NoCoverageError(f"Transect {transect_id} has no raster coverage")

    logger.debug(f"  Found {len(overlapping_rasters)} overlapping rasters")

    # Sample spectral values (returns tuple with skip count)
    spectral_data, skipped_points = sample_transect(
        transect_row, 
        overlapping_rasters, 
        year_config,
        raster_index=raster_index
    )

    logger.debug(f"  Sampled {len(spectral_data)} points (skipped {skipped_points})")

    # Extract features
    features = compute_all(spectral_data, resolve_band_indices(year_config))

    logger.debug(f"  Computed {len(features.columns)} features")

    # Classify landcover
    classifier = LandcoverClassifier()
    landcover = classifier.classify(features)

    # Apply spatial smoothing
    landcover = classifier.apply_spatial_smoothing(landcover)

    # Log class distribution
    class_counts = landcover['predicted_class'].value_counts()
    logger.debug(f"  Classification: {dict(class_counts)}")

    # Detect transitions
    detector = TransitionDetector()
    all_transitions = detector.find_transitions(features, landcover)

    # Filter transitions by boundary type
    transitions = TransitionDetector.filter_by_boundary_type(all_transitions, boundary_types)

    logger.debug(f"  Detected {len(all_transitions)} transitions, "
                f"returned {len(transitions)} (mode: {boundary_types})")

    # Return results with skip count
    result = {
        'transect_id': transect_id,
        'data': spectral_data,
        'features': features,
        'landcover': landcover,
        'transitions': transitions
    }
    
    return result, skipped_points


def main():
    """Command-line interface for the spectral classifier."""
    parser = argparse.ArgumentParser(
        description='Spectral Transect Classification System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m spectral_classifier.main \\
    --rasters /path/to/rasters \\
    --transects /path/to/transects.geojson \\
    --output /path/to/output \\
    --num-visualize 5

Supports 3-band (RGB, CIR) and 4-band (RGBN) imagery.
Band configuration is auto-detected per raster file.
        """
    )

    parser.add_argument(
        '--rasters',
        type=Path,
        required=True,
        help='Directory containing raster files (.tif, .tiff, .jp2)'
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
        help='Directory to save output files'
    )

    parser.add_argument(
        '--band-config',
        type=Path,
        default=Path('INPUT/band_config.json'),   # conventional location
        help='Path to band_config.json (default: INPUT/band_config.json)'
    )

    parser.add_argument(
        '--year',
        type=str,
        required=True,
        help='Year string matching a key in band_config.json (e.g. "2016")'
    )

    parser.add_argument(
        '--num-visualize',
        type=int,
        default=NUM_TRANSECTS_TO_VISUALIZE,
        help=f'Number of transects to visualize (default: {NUM_TRANSECTS_TO_VISUALIZE})'
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

    # Run pipeline
    try:
        results = analyze_all_transects(
            raster_dir=args.rasters,
            transect_geojson=args.transects,
            output_dir=args.output,
            num_visualize=args.num_visualize,
            verbose=args.verbose,
            boundary_types=args.boundary_types
        )

        print(f"\nSuccess! Processed {len(results)} transects.")
        print(f"Results saved to: {args.output}")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())