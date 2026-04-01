"""
Interpret step: run transition detection on precomputed feature Parquets.

Reads:
    features_{year}.parquet  — derived feature arrays (transect_id, distance, …)
    profiles_{year}.parquet  — raw band samples with x, y coordinates

Writes:
    transitions_{year}.parquet  — flat table of all detected transitions
                                  (one row per transition, all diagnostic fields
                                  flattened; nested band_magnitudes expanded to
                                  band_mag_{band} columns)
    shellline_{year}.geojson    — shell line geometry exported to EPSG:4326

Re-run this step whenever detection thresholds or transition logic change.
The compute step (profile sampling + feature extraction) does not need to be
repeated unless the imagery or band config changes.

NOTE: This step does not yet include a landcover classifier. A stub
landcover is synthesised from a distance-based split so that
TransitionDetector._filter_transitions does not silently discard all
candidates. See _make_stub_landcover for details and the TODO therein.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from spectral_classifier.utils import setup_logging, validate_output_directory
from spectral_classifier.transition import TransitionDetector
from spectral_classifier.utils.export import export_shell_line_geojson

logger = logging.getLogger(__name__)

# CRS of x/y coordinates written by compute.py (UTM Zone 14N).
SOURCE_CRS = "EPSG:26914"


# ── Landcover stub ─────────────────────────────────────────────────────────────

def _make_stub_landcover(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesise a minimal landcover DataFrame for transition filtering.

    TransitionDetector._filter_transitions rejects transitions where both
    from_class and to_class are 'UNKNOWN'. Until a real classifier is
    integrated, we assign classes by distance:

        [0, 60% of max_dist)  → BEACH_DRY   (landward / dry sand)
        [60% of max_dist, …]  → BEACH_WET   (seaward  / wet/water)

    This prior is reasonable for PAIS transects: the dry beach is typically
    wider than the wet intertidal zone.

    TODO: Replace with output from a trained landcover classifier. The
    classifier should produce a 'predicted_class' column aligned to the
    same (transect_id, distance) index as features_df.

    Args:
        features_df: Feature rows for a single transect, reset index.

    Returns:
        DataFrame with columns ['transect_id', 'distance', 'predicted_class'].
    """
    lc = features_df[["transect_id", "distance"]].copy()
    split_dist = features_df["distance"].max() * 0.60
    lc["predicted_class"] = "BEACH_DRY"
    lc.loc[features_df["distance"] >= split_dist, "predicted_class"] = "BEACH_WET"
    return lc


# ── Transition flattening ──────────────────────────────────────────────────────

# Known scalar keys present in transition dicts across all detection methods.
# Listed explicitly so the Parquet column order is deterministic.
_CORE_TRANSITION_COLS = [
    "transect_id",
    "index",
    "distance",
    "x",
    "y",
    "type",
    "confidence",
    "detection_method",
    "band_mode",
    "magnitude",
    "from_class",
    "to_class",
    "guaranteed_shell_line",
    "in_expected_zone",
    "num_bands_agreeing",
    # NIR-specific
    "nir_value",
    "nir_drop_abs",
    "nir_before",
    "variability_ratio",
    "rg_valid",
    "rg_adjustment",
    # RGB-specific
    "brightness_value",
    "brightness_drop_abs",
    "brightness_before",
    # Surf-specific
    "has_nir_confirmation",
    # Veg-specific
    "curvature_change",
    "nir_change",
]


def _flatten_transition(
    transect_id,
    t: Dict,
    profiles_group: pd.DataFrame,
) -> Dict:
    """
    Flatten a single transition dict into a tabular row.

    - Stamps transect_id.
    - Looks up x, y from profiles_group using t['index'].
    - Expands nested band_magnitudes dict → band_mag_{band} columns.
    - Any unrecognised extra keys are preserved as-is.

    Args:
        transect_id: Identifier for the parent transect.
        t:           Transition dict from TransitionDetector.find_transitions().
        profiles_group: Profile rows for this transect (reset index).

    Returns:
        Flat dict suitable for pd.DataFrame construction.
    """
    row: Dict = {"transect_id": transect_id}
    row.update(t)

    # Expand nested band_magnitudes → band_mag_nir, band_mag_red, …
    band_mags = row.pop("band_magnitudes", None)
    if isinstance(band_mags, dict):
        for band, val in band_mags.items():
            row[f"band_mag_{band}"] = val

    # Join x, y from profiles using the integer index within the transect.
    idx = t.get("index")
    if idx is not None and 0 <= idx < len(profiles_group):
        row["x"] = profiles_group.iloc[idx]["x"]
        row["y"] = profiles_group.iloc[idx]["y"]
    else:
        logger.warning(
            "Transect %s: transition index %s out of range (profiles len=%d)",
            transect_id, idx, len(profiles_group),
        )
        row["x"] = None
        row["y"] = None

    return row


def _transitions_to_dataframe(rows: List[Dict]) -> pd.DataFrame:
    """
    Convert a list of flattened transition rows to a well-ordered DataFrame.

    Core columns appear first in a fixed order; any extra diagnostic columns
    from future detection methods appear after, sorted alphabetically.

    Args:
        rows: List of flat dicts from _flatten_transition().

    Returns:
        DataFrame ready for Parquet serialisation.
    """
    df = pd.DataFrame(rows)

    # Determine final column order: core cols first (if present), then extras.
    present_core = [c for c in _CORE_TRANSITION_COLS if c in df.columns]
    extra_cols = sorted(c for c in df.columns if c not in _CORE_TRANSITION_COLS)
    df = df[present_core + extra_cols]

    # Explicit type enforcement for columns that must be stable across years.
    for col in ("transect_id", "index"):
        if col in df.columns:
            df[col] = df[col].astype("Int64")   # nullable int (survives NaN)

    for col in ("confidence", "distance", "x", "y"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ── Per-transect processing ────────────────────────────────────────────────────

def _process_transect(
    transect_id,
    features_grp: pd.DataFrame,
    profiles_grp: pd.DataFrame,
    detector: TransitionDetector,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run detection for one transect.

    Args:
        transect_id: Identifier for this transect.
        features_grp: Feature rows for this transect (reset index).
        profiles_grp: Profile rows for this transect (reset index).
        detector:     Shared TransitionDetector instance.

    Returns:
        Tuple of (flat_rows, transition_dicts):
            flat_rows:       Flattened dicts for Parquet output.
            transition_dicts: Raw dicts for GeoJSON export.
    """
    stub_lc = _make_stub_landcover(features_grp)

    try:
        transitions = detector.find_transitions(features_grp, stub_lc)
    except Exception:
        logger.exception("Transect %s: detection raised an exception", transect_id)
        transitions = []

    flat_rows = [
        _flatten_transition(transect_id, t, profiles_grp)
        for t in transitions
    ]

    return flat_rows, transitions


# ── Public API ─────────────────────────────────────────────────────────────────

def run_interpret(
    year: str,
    features_path: Path,
    profiles_path: Path,
    output_dir: Path,
    verbose: bool = False,
) -> Tuple[Path, Optional[Path]]:
    """
    Run the interpret step for one year.

    Loads feature and profile Parquets written by compute.py, runs
    TransitionDetector on each transect, and writes two outputs:

      transitions_{year}.parquet  — full diagnostic table (one row per transition)
      shellline_{year}.geojson    — shell line geometry in EPSG:4326

    Args:
        year:          Year string matching a key in band_config.json.
        features_path: Path to features_{year}.parquet.
        profiles_path: Path to profiles_{year}.parquet.
        output_dir:    Directory to write output files.
        verbose:       Enable debug logging.

    Returns:
        Tuple of (transitions_path, shellline_path).
        shellline_path is None if no shell lines were detected.

    Raises:
        FileNotFoundError: If either input Parquet does not exist.
        RuntimeError:      If features Parquet contains no transects.
    """
    output_dir = validate_output_directory(output_dir)
    setup_logging(verbose=verbose, log_file=output_dir / "interpret.log")

    logger.info("Interpret step — year %s", year)

    # ── Load inputs ──────────────────────────────────────────────────────────
    for path in (features_path, profiles_path):
        if not Path(path).exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    features_all = pd.read_parquet(features_path)
    profiles_all = pd.read_parquet(profiles_path)

    transect_ids = sorted(features_all["transect_id"].unique())
    if not transect_ids:
        raise RuntimeError(f"features Parquet for {year} contains no transects")

    logger.info(
        "Loaded %d feature rows across %d transects",
        len(features_all), len(transect_ids),
    )

    # ── Detect transitions ───────────────────────────────────────────────────
    detector = TransitionDetector()

    all_flat_rows: List[Dict] = []
    all_results: List[Dict] = []          # legacy format for export_shell_line_geojson

    for tid in transect_ids:
        features_grp = (
            features_all[features_all["transect_id"] == tid]
            .reset_index(drop=True)
        )
        profiles_grp = (
            profiles_all[profiles_all["transect_id"] == tid]
            .reset_index(drop=True)
        )

        flat_rows, transitions = _process_transect(
            tid, features_grp, profiles_grp, detector
        )

        all_flat_rows.extend(flat_rows)

        # Build the dict structure expected by export_shell_line_geojson.
        all_results.append({
            "transect_id": tid,
            "data": profiles_grp,        # must have x, y columns
            "transitions": transitions,
        })

    logger.info(
        "Detection complete: %d transitions across %d transects",
        len(all_flat_rows), len(transect_ids),
    )

    # ── Write transitions Parquet ────────────────────────────────────────────
    transitions_path = output_dir / f"transitions_{year}.parquet"

    if all_flat_rows:
        transitions_df = _transitions_to_dataframe(all_flat_rows)
        transitions_df.to_parquet(transitions_path, index=False)
        logger.info("Wrote %d rows to %s", len(transitions_df), transitions_path)
    else:
        # Write a typed empty Parquet so downstream steps can read it safely.
        empty_df = pd.DataFrame(
            columns=["transect_id", "distance", "type", "confidence", "x", "y"]
        )
        empty_df.to_parquet(transitions_path, index=False)
        logger.warning("No transitions detected for year %s", year)

    # ── Export shell line GeoJSON ────────────────────────────────────────────
    shellline_path = export_shell_line_geojson(
        all_results=all_results,
        output_dir=output_dir,
        year=year,
        source_crs=SOURCE_CRS,
    )

    return transitions_path, shellline_path