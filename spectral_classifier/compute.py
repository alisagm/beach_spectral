"""
Compute step: sample spectral profiles and extract features.

Reads imagery and transects, writes two Parquet checkpoints:
  profiles_{year}.parquet  — raw band samples (transect_id, distance, red, green, blue, nir)
  features_{year}.parquet  — derived feature arrays (transect_id, distance, + all features)

Run this step once per year. Re-run only if imagery or band config changes.
Interpretation (classify, detect transitions) is handled separately in interpret.py.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from spectral_classifier.config import BAND_CONFIG_PATH
from spectral_classifier.utils import (
    build_raster_index,
    load_transects,
    reproject_if_needed,
    find_overlapping_rasters,
    RasterIndex,
    setup_logging,
    validate_output_directory,
    ProgressTracker,
    load_band_config,
    resolve_band_indices,
)
from spectral_classifier.spectral import sample_transect, compute_all

logger = logging.getLogger(__name__)

# Columns written to profiles parquet — raw band samples + coordinates.
PROFILE_COLS = ['transect_id', 'distance', 'x', 'y', 'red', 'green', 'blue', 'nir']


class NoCoverageError(Exception):
    """Raised when a transect has no overlapping rasters (expected with gappy imagery)."""
    pass


def process_single_transect(
    transect_row,
    raster_index: RasterIndex,
    current_idx: int,
    total: int,
    year_config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Sample and extract features for one transect.

    Stamps transect_id onto both returned DataFrames so rows remain
    identifiable after concatenation across all transects.

    Args:
        transect_row: Row from transects GeoDataFrame.
        raster_index: Built raster spatial index.
        current_idx:  1-based position in the processing loop (for logging).
        total:        Total transect count (for logging).
        year_config:  Entry from band_config.json for this year.

    Returns:
        Tuple of (profiles_df, features_df, skipped_point_count).
        profiles_df: raw band samples — PROFILE_COLS only.
        features_df: all derived feature columns + transect_id + distance.

    Raises:
        NoCoverageError: If no raster covers this transect. Expected with
                         gappy imagery — callers should catch and skip quietly.
    """
    transect_id = transect_row.TransectID
    logger.debug(f"[{current_idx}/{total}] Transect {transect_id}")

    overlapping = find_overlapping_rasters(transect_row.geometry, raster_index)
    if not overlapping:
        raise NoCoverageError(f"Transect {transect_id}: no raster coverage")

    # Sample raw spectral values along the transect.
    spectral_data, skipped_points = sample_transect(
        transect_row,
        overlapping,
        year_config,
    )

    # Compute all feature arrays. Returns a copy of spectral_data with
    # feature columns appended — raw bands are still present.
    band_indices = resolve_band_indices(year_config)
    full_df = compute_all(spectral_data, band_indices)

    # Rename sampler's TransectID → transect_id (snake_case consistency).
    # Do this before slicing so both output files get the same column name.
    full_df = full_df.rename(columns={'TransectID': 'transect_id'})

    # Profiles: raw band samples only. Cast boolean-like columns explicitly
    # to avoid schema drift when concatenating across transects.
    profiles_df = full_df[PROFILE_COLS].copy()

    # Features: everything except the raw band columns (already in profiles).
    # Keep transect_id and distance so the two files can be joined later.
    feature_cols = (
        ['transect_id', 'distance']
        + [c for c in full_df.columns if c not in PROFILE_COLS]
    )
    features_df = full_df[feature_cols].copy()

    # Explicit bool cast — prevents mixed int/bool schema across years.
    for col in ('has_oscillations', 'has_rgb_foam_peak'):
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(bool)

    logger.debug(
        f"  {len(full_df)} points sampled, {skipped_points} skipped, "
        f"{len(features_df.columns) - 2} feature columns"
    )
    return profiles_df, features_df, skipped_points


def run_compute(
    year: str,
    raster_paths: List[Path],
    transect_geojson: Path,
    output_dir: Path,
    band_config_path: Path = BAND_CONFIG_PATH,
    verbose: bool = False,
) -> Tuple[Path, Path]:
    """
    Run the compute step for one year: sample all transects and save checkpoints.

    Args:
        year:             Year string matching a key in band_config.json.
        raster_paths:     List of raster tile paths for this year.
        transect_geojson: Path to transects GeoJSON.
        output_dir:       Directory to write Parquet files.
        band_config_path: Path to band_config.json.
        verbose:          Enable debug logging.

    Returns:
        Tuple of (profiles_path, features_path).

    Raises:
        RuntimeError: If no transects were successfully processed.
    """
    output_dir = validate_output_directory(output_dir)
    setup_logging(verbose=verbose, log_file=output_dir / 'compute.log')

    logger.info(f"Compute step — year {year}")

    # Build spatial index over raster tiles.
    raster_index = build_raster_index(raster_paths)
    logger.info(f"Indexed {len(raster_index)} rasters")

    # Load band configuration for this year.
    year_config = load_band_config(band_config_path, year)
    logger.info(f"Band config: {year_config}")

    # Load transects and align CRS to raster index.
    transects = load_transects(transect_geojson)
    transects = reproject_if_needed(transects, raster_index.crs)
    logger.info(f"Loaded {len(transects)} transects")

    tracker = ProgressTracker(total_items=len(transects), item_noun="transect")
    all_profiles: List[pd.DataFrame] = []
    all_features: List[pd.DataFrame] = []

    for idx, transect in transects.iterrows():
        tracker.update(idx + 1)

        try:
            profiles_df, features_df, skipped_pts = process_single_transect(
                transect, raster_index, idx + 1, len(transects), year_config
            )
            all_profiles.append(profiles_df)
            all_features.append(features_df)
            tracker._processed_count += 1

            if skipped_pts > 0:
                tracker.add_skipped("sample points (outside coverage)", skipped_pts)

        except NoCoverageError:
            tracker.add_skipped("transects (no raster coverage)", 1)

        except Exception as e:
            logger.error(f"Transect {transect.TransectID} failed: {e}", exc_info=True)
            tracker.add_error(f"Transect {transect.TransectID}: {e}")

    tracker.finish(success_count=len(all_profiles))

    if not all_profiles:
        raise RuntimeError(f"Year {year}: no transects were successfully processed")

    logger.info(f"Processed {len(all_profiles)}/{len(transects)} transects")

    # Concatenate and write checkpoints.
    profiles_path = output_dir / f"profiles_{year}.parquet"
    features_path = output_dir / f"features_{year}.parquet"

    profiles = pd.concat(all_profiles, ignore_index=True)
    features = pd.concat(all_features, ignore_index=True)

    profiles.to_parquet(profiles_path, index=False)
    features.to_parquet(features_path, index=False)

    logger.info(f"Profiles: {profiles_path}  ({len(profiles)} rows)")
    logger.info(f"Features: {features_path}  ({len(features)} rows)")

    return profiles_path, features_path