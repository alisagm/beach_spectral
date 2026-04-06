#!/usr/bin/env python3
"""
Analyse spectral profiles at manually digitised shoreline pick locations.

For each (transect_id, year) in manual_transitions.parquet:
  1. Extracts spectral feature values at the nearest sample point to the pick
  2. Computes windowed statistics around the pick:
       ndwi_gradient_offset  — where is the steepest NDWI gradient relative
                               to the pick? negative = landward, positive = seaward
       ndwi_contrast         — mean NDWI seaward of pick minus landward
                               (positive = wetter seaward, as expected)
       pick_fraction         — manual_distance / transect_length
                               (criteria drift diagnostic across years)
  3. Joins old-detector and PELT distances from comparison_{year}.parquet
     and computes signed errors against the manual pick.

NDWI derivative is computed fresh from the full profile (not pre-stored in
features), using compute_derivative_smooth from spectral_classifier.spectral.
Computing on the full profile before windowing avoids uniform_filter1d edge
effects that would corrupt window-edge values.

Output
------
  pick_analysis.parquet — one row per (transect_id, year) pick

Schema
------
  transect_id           int
  year                  str
  manual_distance       float   metres from landward end (manual pick)
  n_intersections       int     from compute_manual_transitions.py; >1 = inspect
  moisture_index_source str     which index was used: 'ndwi' (CIR/RGBN),
                                'grvi' (RGB fallback), or 'none' (all-NaN)
                                stratify by this column before comparing
                                contrast magnitudes across years
  ndwi_at_pick          float   moisture index value at nearest sample to pick
                                (NDWI for NIR years, GRVI for 2022 RGB)
  ndvi_at_pick          float   NDVI at nearest sample point (NaN for RGB)
  ndwi_gradient_offset  float   offset (m) of max |moisture index d1| from pick
                                within window; negative = landward of pick
  ndwi_contrast         float   mean moisture index in [pick, pick+window_m]
                                minus mean in [pick-window_m, pick]
                                positive = wetter/greener seaward of pick
                                NOTE: GRVI contrast is not directly comparable
                                to NDWI contrast — stratify by moisture_index_source
  pick_fraction         float   manual_distance / transect_length
  old_distance          float   threshold-detector pick distance (NaN if undetected)
  new_distance          float   PELT pick distance (NaN if undetected)
  old_error             float   old_distance - manual_distance; + = seaward of truth
  new_error             float   new_distance - manual_distance
  old_none              bool    threshold detector found no pick
  new_none              bool    PELT found no pick
  agreement_category    str     from compare_detectors.py
  band_mode             str

Usage
-----
  python scripts/analyse_picks.py \\
      --manual-transitions  OUTPUT/manual_transitions.parquet \\
      --output-root         OUTPUT \\
      --comparison-dir      OUTPUT/comp \\
      --output              OUTPUT/pick_analysis.parquet

  # Adjust window half-width (default 30 m either side of pick):
  python scripts/analyse_picks.py ... --window-m 20

  # Skip the comparison join (e.g. comparison parquets not yet available):
  python scripts/analyse_picks.py ... --skip-comparison
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Derivative function — imported from the package so we stay consistent with
# what was used to build features_{year}.parquet.
# Deferred inside main() to avoid hard import failure if running outside venv.

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_M = 30.0


# ── Data loading ──────────────────────────────────────────────────────────────

def _normalise_year(series: pd.Series) -> pd.Series:
    """
    Coerce a year column to plain string ('1995', not '1995.0' or 1995).

    Handles float (shapefile attribute read-back), int (comparison parquet),
    and string inputs uniformly.
    """
    return series.astype(float).astype(int).astype(str)


def load_manual_transitions(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["year"] = _normalise_year(df["year"])
    logger.info("Manual transitions: %d rows, years: %s",
                len(df), sorted(df["year"].unique()))
    return df


def load_features_for_years(output_root: Path, years: list[str]) -> pd.DataFrame:
    """
    Load and concatenate features_{year}.parquet for each requested year.

    Stamps a 'year' column (str) onto each before concatenating.
    Missing files emit a WARNING and are skipped — the downstream loop will
    produce NaN rows for affected transects rather than crashing.
    """
    frames = []
    for y in sorted(years):
        path = output_root / y / f"features_{y}.parquet"
        if not path.exists():
            logger.warning("Features parquet not found — skipping year %s: %s", y, path)
            continue
        df = pd.read_parquet(path)
        df["year"] = y
        frames.append(df)
        logger.info("  Loaded features %s: %d rows, %d columns", y, len(df), df.shape[1])

    if not frames:
        raise FileNotFoundError(
            f"No features parquets found under {output_root} "
            f"for years {years}.  Check --output-root."
        )
    return pd.concat(frames, ignore_index=True)


def load_comparison_for_years(
    comparison_dir: Path,
    years: list[str],
) -> Optional[pd.DataFrame]:
    """
    Load and concatenate comparison_{year}.parquet for each requested year.

    Returns None if comparison_dir is None or no files are found.
    """
    if comparison_dir is None:
        return None

    frames = []
    for y in sorted(years):
        path = comparison_dir / f"comparison_{y}.parquet"
        if not path.exists():
            logger.warning(
                "Comparison parquet not found — year %s will have NaN errors: %s", y, path
            )
            continue
        df = pd.read_parquet(path)
        df["year"] = _normalise_year(df["year"])
        frames.append(df)
        logger.info("  Loaded comparison %s: %d rows", y, len(df))

    if not frames:
        logger.warning("No comparison parquets found — error columns will be NaN.")
        return None

    keep_cols = [
        "transect_id", "year",
        "old_distance", "new_distance",
        "old_none", "new_none",
        "agreement_category", "band_mode",
    ]
    comp = pd.concat(frames, ignore_index=True)
    missing = [c for c in keep_cols if c not in comp.columns]
    if missing:
        logger.warning("Comparison parquet missing expected columns: %s", missing)
    return comp[[c for c in keep_cols if c in comp.columns]]


# ── Per-pick statistics ───────────────────────────────────────────────────────

def _compute_pick_stats(
    feat: pd.DataFrame,
    manual_distance: float,
    window_m: float,
) -> dict:
    """
    Compute point values and windowed statistics for one (transect_id, year) pick.

    Parameters
    ----------
    feat :
        Feature rows for this transect/year, sorted by distance.
        Must contain 'distance' and at least one moisture index:
          'ndwi'  — preferred; available for CIR and RGBN years
          'grvi'  — fallback for RGB-only years (2022)
        'ndvi' used if present.
    manual_distance :
        Pick position in metres from the landward end.
    window_m :
        Half-width of analysis window in metres.

    Returns
    -------
    dict with keys matching the output schema.
    """
    from spectral_classifier.spectral.derivatives import compute_derivative_smooth

    dist_arr = feat["distance"].values

    # ── Select moisture index — NDWI preferred, GRVI fallback for RGB ────
    # NDWI requires NIR; 2022 is RGB-only so GRVI is substituted.
    # All downstream stats are computed identically regardless of which
    # index is used. moisture_index_source is recorded so the caller can
    # stratify results — GRVI contrast is not directly comparable to NDWI.
    _NAN_STATS = {
        "ndwi_at_pick":          np.nan,
        "ndvi_at_pick":          np.nan,
        "ndwi_gradient_offset":  np.nan,
        "ndwi_contrast":         np.nan,
        "pick_fraction":         np.nan,
        "moisture_index_source": "none",
        "_snapped_distance":     np.nan,
        "_snap_delta_m":         np.nan,
    }

    if "ndwi" in feat.columns and feat["ndwi"].notna().any():
        moisture_arr          = feat["ndwi"].values
        moisture_index_source = "ndwi"
    elif "grvi" in feat.columns and feat["grvi"].notna().any():
        moisture_arr          = feat["grvi"].values
        moisture_index_source = "grvi"
    else:
        logger.warning(
            "No moisture index (ndwi / grvi) found at manual_distance=%.1f — "
            "both absent or all-NaN. Stat columns will be NaN for this pick.",
            manual_distance,
        )
        return _NAN_STATS

    # ── Point values at nearest sample ───────────────────────────────────
    offsets    = np.abs(dist_arr - manual_distance)
    nearest_i  = int(np.argmin(offsets))
    snap_dist  = float(dist_arr[nearest_i])
    snap_delta = float(offsets[nearest_i])  # QA: how far did we snap?

    ndwi_at_pick = (
        float(moisture_arr[nearest_i])
        if not np.isnan(moisture_arr[nearest_i]) else np.nan
    )
    ndvi_at_pick = (
        float(feat["ndvi"].values[nearest_i])
        if "ndvi" in feat.columns and not np.isnan(feat["ndvi"].values[nearest_i])
        else np.nan
    )

    # ── Moisture index derivative on FULL profile (before windowing) ──────
    # Computing on the full profile avoids uniform_filter1d edge effects
    # that would corrupt derivative values near the window boundary.
    moisture_d1 = compute_derivative_smooth(moisture_arr, dist_arr)

    # ── Window mask ───────────────────────────────────────────────────────
    in_window   = (dist_arr >= manual_distance - window_m) & \
                  (dist_arr <= manual_distance + window_m)
    landward_m  = (dist_arr >= manual_distance - window_m) & \
                  (dist_arr <  manual_distance)
    seaward_m   = (dist_arr >  manual_distance) & \
                  (dist_arr <= manual_distance + window_m)

    # ── ndwi_gradient_offset ─────────────────────────────────────────────
    # Position of the largest absolute moisture index gradient within the
    # window, expressed as signed offset from the pick (negative = landward).
    if in_window.sum() > 0:
        window_d1    = moisture_d1[in_window]
        window_dists = dist_arr[in_window]
        peak_i       = int(np.nanargmax(np.abs(window_d1)))
        ndwi_gradient_offset = float(window_dists[peak_i] - manual_distance)
    else:
        ndwi_gradient_offset = np.nan

    # ── ndwi_contrast ─────────────────────────────────────────────────────
    # Seaward mean minus landward mean within the window.
    # For NDWI: positive = wetter seaward, as expected at the shell line.
    # For GRVI: sign is less reliable — dry/wet contrast is weaker in
    # green/red than green/NIR. Stratify by moisture_index_source.
    mi_landward   = float(np.nanmean(moisture_arr[landward_m])) if landward_m.sum() > 0 else np.nan
    mi_seaward    = float(np.nanmean(moisture_arr[seaward_m]))  if seaward_m.sum()  > 0 else np.nan
    ndwi_contrast = (
        (mi_seaward - mi_landward)
        if not (np.isnan(mi_landward) or np.isnan(mi_seaward))
        else np.nan
    )

    # ── pick_fraction ─────────────────────────────────────────────────────
    transect_length = float(np.nanmax(dist_arr))
    pick_fraction   = manual_distance / transect_length if transect_length > 0 else np.nan

    return {
        "ndwi_at_pick":           ndwi_at_pick,
        "ndvi_at_pick":           ndvi_at_pick,
        "ndwi_gradient_offset":   ndwi_gradient_offset,
        "ndwi_contrast":          ndwi_contrast,
        "pick_fraction":          pick_fraction,
        "moisture_index_source":  moisture_index_source,
        # QA columns — useful for spotting bad snaps or sparse profiles
        "_snapped_distance":      snap_dist,
        "_snap_delta_m":          snap_delta,
    }


# ── Main assembly ─────────────────────────────────────────────────────────────

def build_analysis(
    manual: pd.DataFrame,
    features: pd.DataFrame,
    comparison: Optional[pd.DataFrame],
    window_m: float,
) -> pd.DataFrame:
    """
    Join manual picks with spectral statistics and algorithm errors.

    Iterates over (transect_id, year) pairs in manual; for each, runs
    _compute_pick_stats against the corresponding features slice.

    Missing features (transect not in that year's parquet) produce a row
    with NaN stat columns rather than being silently dropped.
    """
    # Build a (transect_id, year) → feature-rows lookup for fast access
    feat_groups = {
        key: grp.sort_values("distance").reset_index(drop=True)
        for key, grp in features.groupby(["transect_id", "year"])
    }

    records = []
    n_missing_feat = 0

    for _, row in manual.iterrows():
        tid  = int(row["transect_id"])
        year = str(row["year"])
        key  = (tid, year)

        base = {
            "transect_id":     tid,
            "year":            year,
            "manual_distance": float(row["distance"]),
            "n_intersections": int(row["n_intersections"]),
        }

        feat = feat_groups.get(key)
        if feat is None or feat.empty:
            logger.debug(
                "No features for transect %d year %s — stat columns will be NaN.", tid, year
            )
            n_missing_feat += 1
            stats = {k: np.nan for k in (
                "ndwi_at_pick", "ndvi_at_pick",
                "ndwi_gradient_offset", "ndwi_contrast",
                "pick_fraction", "moisture_index_source",
                "_snapped_distance", "_snap_delta_m",
            )}
            stats["moisture_index_source"] = "none"
        else:
            stats = _compute_pick_stats(feat, float(row["distance"]), window_m)

        records.append({**base, **stats})

    if n_missing_feat:
        logger.warning(
            "%d picks had no matching features entry — "
            "check that features parquets cover all years in manual_transitions.",
            n_missing_feat,
        )

    analysis = pd.DataFrame(records)

    # ── Join algorithm distances and compute signed errors ────────────────
    if comparison is not None:
        analysis = analysis.merge(
            comparison,
            on=["transect_id", "year"],
            how="left",   # keep all manual picks even if algo had no detection
        )
        # Signed error convention: positive = algorithm is seaward of truth
        analysis["old_error"] = analysis["old_distance"] - analysis["manual_distance"]
        analysis["new_error"] = analysis["new_distance"] - analysis["manual_distance"]
    else:
        for col in ("old_distance", "new_distance", "old_error", "new_error",
                    "old_none", "new_none", "agreement_category", "band_mode"):
            analysis[col] = np.nan

    return analysis


def print_summary(df: pd.DataFrame) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  PICK ANALYSIS — SUMMARY")
    print(sep)
    print(f"  Total picks:           {len(df)}")
    print(f"  Years:                 {sorted(df['year'].unique())}")

    if "moisture_index_source" in df.columns:
        print(f"\n  Moisture index used (stratify before comparing contrast magnitudes):")
        for src, grp in df.groupby("moisture_index_source"):
            print(f"    {src:<20}  {len(grp):>5} picks  "
                  f"years: {sorted(grp['year'].unique())}")

    print(f"\n  Spectral context (all picks):")
    for col in ("ndwi_at_pick", "ndwi_contrast", "ndwi_gradient_offset", "pick_fraction"):
        if col in df.columns:
            vals = df[col].dropna()
            print(f"    {col:<26}  n={len(vals):>5}  "
                  f"mean={vals.mean():>7.3f}  "
                  f"std={vals.std():>6.3f}  "
                  f"[{vals.min():.3f}, {vals.max():.3f}]")

    if "old_error" in df.columns and df["old_error"].notna().any():
        print(f"\n  Algorithm errors (positive = seaward of truth):")
        for label, col in [("old (threshold)", "old_error"), ("new (PELT)", "new_error")]:
            vals = df[col].dropna()
            if len(vals) == 0:
                print(f"    {label:<20}  no data")
                continue
            print(f"    {label:<20}  n={len(vals):>5}  "
                  f"mean={vals.mean():>+7.2f} m  "
                  f"std={vals.std():>6.2f}  "
                  f"median={vals.median():>+7.2f} m")

    snap_issues = (df["_snap_delta_m"] > 2.0).sum() if "_snap_delta_m" in df.columns else 0
    if snap_issues:
        print(f"\n  WARNING: {snap_issues} picks snapped >2 m from nearest sample point "
              f"— check profile coverage for these transects.")

    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse spectral profiles at manual pick locations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--manual-transitions", "-m",
        type=Path,
        default=Path("OUTPUT/manual_transitions.parquet"),
        help="Output of compute_manual_transitions.py (default: OUTPUT/manual_transitions.parquet)",
    )
    parser.add_argument(
        "--output-root", "-r",
        type=Path,
        default=Path("OUTPUT"),
        help="Root dir containing OUTPUT/{year}/features_{year}.parquet (default: OUTPUT)",
    )
    parser.add_argument(
        "--comparison-dir", "-c",
        type=Path,
        default=None,
        help="Dir containing comparison_{year}.parquet from compare_detectors.py",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("OUTPUT/pick_analysis.parquet"),
        help="Output Parquet path (default: OUTPUT/pick_analysis.parquet)",
    )
    parser.add_argument(
        "--window-m",
        type=float,
        default=DEFAULT_WINDOW_M,
        metavar="M",
        help=f"Half-width of analysis window in metres (default: {DEFAULT_WINDOW_M})",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip loading comparison parquets (error columns will be NaN).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)

    # ── Validate ──────────────────────────────────────────────────────────
    if not args.manual_transitions.exists():
        logger.error("manual_transitions not found: %s", args.manual_transitions)
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────
    logger.info("Loading manual transitions...")
    manual = load_manual_transitions(args.manual_transitions)
    years  = sorted(manual["year"].unique())

    logger.info("Loading features parquets...")
    features = load_features_for_years(args.output_root, years)

    comparison = None
    if not args.skip_comparison:
        comp_dir = args.comparison_dir or (args.output_root / "comp")
        logger.info("Loading comparison parquets from %s...", comp_dir)
        comparison = load_comparison_for_years(comp_dir, years)

    # ── Build analysis ────────────────────────────────────────────────────
    logger.info(
        "Computing pick statistics (window=±%.0f m) for %d picks...",
        args.window_m, len(manual),
    )
    analysis = build_analysis(manual, features, comparison, args.window_m)

    # ── Summarise and save ────────────────────────────────────────────────
    print_summary(analysis)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_parquet(args.output, index=False)
    logger.info("Saved %d rows → %s", len(analysis), args.output)


if __name__ == "__main__":
    main()