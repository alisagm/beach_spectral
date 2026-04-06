#!/usr/bin/env python3
"""
tools/compare_detectors.py — Side-by-side comparison of the old and PELT detectors.

Old detector results are loaded from pre-existing transitions_{year}.parquet
files (written by interpret.py) rather than re-running TransitionDetector.
This avoids the landcover-stub dependency and ensures the comparison is against
the actual production outputs of the old detector.

The PELT detector is run fresh from features_{year}.parquet.

Outputs (written to --comparison-dir)
--------------------------------------
  comparison_{year}.parquet          — one row per transect, all metrics
  comparison_all.parquet             — concatenation of all years
  old/{year}/shellline_{year}.geojson  — old detector shell line per year
  pelt/{year}/shellline_{year}.geojson — PELT shell line per year
  shelllines_old.gpkg                — all old shell lines bundled (QGIS-ready)
  shelllines_pelt.gpkg               — all PELT shell lines bundled (QGIS-ready)

Usage
-----
  python tools/compare_detectors.py \\
      --features-dir    OUTPUT/ \\
      --transitions-dir OUTPUT/ \\
      --profiles-dir    OUTPUT/ \\
      --comparison-dir  OUTPUT/comparison/ \\
      --band-config     INPUT/band_config.json

  # Skip GeoJSON/GPKG export (parquet only):
  python tools/compare_detectors.py ... --skip-geojson

  # Single year:
  python tools/compare_detectors.py ... --year 2009
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Agreement category thresholds (metres) ───────────────────────────────────
CLOSE_THRESHOLD    = 5.0   # < 5 m  — effectively the same position
MODERATE_THRESHOLD = 20.0  # 5–20 m — meaningful but not gross disagreement
# ≥ 20 m → "divergent"

_FORMAT_TO_BAND_MODE = {
    "CIR":  "cir",
    "RGBN": "4band",
    "RGB":  "rgb",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_band_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _band_mode_for_year(year: str, band_config: dict) -> str:
    fmt = band_config[year]["format"].upper()
    return _FORMAT_TO_BAND_MODE.get(fmt, "unknown")


def _load_old_shell_lines(transitions_path: Path) -> dict:
    """
    Load old detector shell line positions from transitions_{year}.parquet.

    Filters to dry_wet transitions, then selects the best candidate per
    transect using the same preference order as the old pipeline:
      1. guaranteed_shell_line=True — explicit fallback marker
      2. Highest confidence among remaining dry_wet candidates

    Args:
        transitions_path: Path to transitions_{year}.parquet from interpret.py.

    Returns:
        Dict mapping transect_id → {'distance': float, 'confidence': float}.
        Transects with no dry_wet transition are absent from the dict.
    """
    df = pd.read_parquet(transitions_path)

    dry_wet = df[df["type"].str.contains("dry_wet", na=False)].copy()
    if dry_wet.empty:
        logger.warning("No dry_wet transitions found in %s", transitions_path)
        return {}

    result = {}
    for tid, grp in dry_wet.groupby("transect_id"):
        if "guaranteed_shell_line" in grp.columns:
            guaranteed = grp[grp["guaranteed_shell_line"].astype(bool)]
        else:
            guaranteed = pd.DataFrame()
        pool = guaranteed if not guaranteed.empty else grp
        best = pool.loc[pool["confidence"].idxmax()]
        result[tid] = {
            "distance":   float(best["distance"]),
            "confidence": float(best["confidence"]),
        }

    return result


# ── GeoJSON export ────────────────────────────────────────────────────────────

# CRS of x/y coordinates written by compute.py (UTM Zone 14N).
_SOURCE_CRS = "EPSG:26914"


def _export_detector_geojsons(
    comparison_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    comparison_dir: Path,
    year: str,
) -> None:
    """
    Write per-year shell line GeoJSONs for both detectors.

    Layout matches what bundle_shell_lines_to_gpkg expects:
        comparison_dir/old/{year}/shellline_{year}.geojson
        comparison_dir/pelt/{year}/shellline_{year}.geojson

    Coordinates are looked up from profiles_df by nearest distance match.
    Points where either the distance is NaN (detector returned None) or the
    nearest profile match is a NaN coordinate are silently skipped — the
    gap-splitting logic in _build_shell_line_geometry then handles any
    resulting holes in coverage.

    Args:
        comparison_df: Output of run_year(), one row per transect.
        profiles_df:   profiles_{year}.parquet for this year (x, y columns required).
        comparison_dir: Root comparison output directory.
        year:          Year string, used for filenames and the GeoJSON 'year' attribute.
    """
    # Private helper from export.py — dependency is intentional; if the
    # function is renamed or moved, update this import accordingly.
    from spectral_classifier.utils.export import _build_shell_line_geometry  # noqa: PLC0415

    try:
        import geopandas as gpd  # noqa: PLC0415
        from shapely.geometry import Point  # noqa: PLC0415
    except ImportError as exc:
        logger.error("geopandas / shapely not available — skipping GeoJSON export: %s", exc)
        return

    def _make_shell_points(dist_col: str, conf_col: str) -> list:
        """
        Build the shell_points list for one detector from the comparison DataFrame.

        For each transect with a valid detected distance, finds the profile
        row with the closest distance value and extracts its x, y.
        """
        pts = []
        for _, row in comparison_df.iterrows():
            dist = row[dist_col]
            if np.isnan(dist):
                continue  # detector returned None for this transect

            tid = row["transect_id"]
            tprof = profiles_df[profiles_df["transect_id"] == tid]
            if tprof.empty:
                logger.debug("No profile rows for transect %d — skipping", tid)
                continue

            # Nearest profile point by distance value.
            nearest = tprof.iloc[(tprof["distance"] - dist).abs().argmin()]
            x, y = nearest["x"], nearest["y"]

            if pd.isna(x) or pd.isna(y):
                logger.debug(
                    "Transect %d: nearest profile point has NaN coordinates — skipping",
                    tid,
                )
                continue

            pts.append({
                "transect_id": int(tid),
                "x":           float(x),
                "y":           float(y),
                "distance":    float(dist),
                "confidence":  float(row[conf_col]) if not np.isnan(row[conf_col]) else 0.0,
            })
        return pts

    for detector_name, dist_col, conf_col in [
        ("old",  "old_distance",  "old_confidence"),
        ("pelt", "new_distance",  "new_confidence"),
    ]:
        shell_points = _make_shell_points(dist_col, conf_col)

        if len(shell_points) < 2:
            logger.warning(
                "Year %s %s detector: fewer than 2 shell points — skipping GeoJSON",
                year, detector_name,
            )
            continue

        geometry = _build_shell_line_geometry(
            shell_points,
            gap_threshold_m=500.0,  # matches interpret.py default
        )
        if geometry is None:
            logger.warning(
                "Year %s %s detector: _build_shell_line_geometry returned None",
                year, detector_name,
            )
            continue

        out_dir = comparison_dir / detector_name / year
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"shellline_{year}.geojson"

        gdf = gpd.GeoDataFrame(
            [{"year": year, "detector": detector_name, "num_points": len(shell_points)}],
            geometry=[geometry],
            crs=_SOURCE_CRS,
        )
        gdf.to_crs("EPSG:4326").to_file(out_path, driver="GeoJSON")

        geom_type = gdf.geometry.iloc[0].geom_type
        n_parts = len(gdf.geometry.iloc[0].geoms) if geom_type == "MultiLineString" else 1
        logger.info(
            "  %s shell line → %s  (%d points, %d segment(s))",
            detector_name, out_path, len(shell_points), n_parts,
        )


def _categorise(row: pd.Series) -> str:
    """Assign agreement_category from a comparison row."""
    if row["old_none"] and row["new_none"]:
        return "both_none"
    if row["old_none"]:
        return "new_only"
    if row["new_none"]:
        return "old_only"
    abs_d = row["abs_delta_m"]
    if abs_d < CLOSE_THRESHOLD:
        return "close"
    if abs_d < MODERATE_THRESHOLD:
        return "moderate"
    return "divergent"


# ── Per-year processing ───────────────────────────────────────────────────────

def run_year(
    year: str,
    features_path: Path,
    transitions_path: Path,
    band_mode: str,
    comparison_dir: Path,
    profiles_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compare old and PELT detectors for all transects in one year.

    Old shell lines are loaded from transitions_path (interpret.py output).
    PELT is run fresh from features_path.

    If profiles_path is provided and exists, also exports per-detector shell
    line GeoJSONs under comparison_dir/old/ and comparison_dir/pelt/.

    Returns a DataFrame with one row per transect.
    """
    # Deferred import: only needed at detect-time.
    # TODO: promote to top-level once interpret.py stabilises.
    from spectral_classifier.transition.pelt import detect_shell_line_pelt  # noqa: PLC0415

    logger.info("Year %s  band_mode=%s", year, band_mode)

    features_df = pd.read_parquet(features_path)
    all_ids = sorted(features_df["transect_id"].unique())
    logger.info("  %d transects in features parquet", len(all_ids))

    # Load old results once for the whole year — one parquet read.
    old_shell_lines = _load_old_shell_lines(transitions_path)
    logger.info("  %d transects with old shell line", len(old_shell_lines))

    rows = []
    for tid in all_ids:
        tfeatures = features_df[features_df["transect_id"] == tid].copy()

        # ── Old detector (from parquet) ───────────────────────────────────
        old_entry = old_shell_lines.get(tid)
        old_dist  = old_entry["distance"]   if old_entry else np.nan
        old_conf  = old_entry["confidence"] if old_entry else np.nan
        old_none  = old_entry is None

        # ── New detector (PELT, run fresh) ────────────────────────────────
        try:
            new_result = detect_shell_line_pelt(tfeatures, band_mode=band_mode)
        except Exception as exc:
            logger.debug("PELT failed on transect %d: %s", tid, exc)
            new_result = None

        new_dist = float(new_result["distance"])   if new_result else np.nan
        new_conf = float(new_result["confidence"]) if new_result else np.nan
        new_none = new_result is None

        delta = (new_dist - old_dist) if (not old_none and not new_none) else np.nan
        abs_d = abs(delta)            if not np.isnan(delta)             else np.nan

        partial = new_result.get("partial_coverage", False) if new_result else False

        rows.append({
            "transect_id":          tid,
            "year":                 int(year),
            "band_mode":            band_mode,
            "old_distance":         old_dist,
            "new_distance":         new_dist,
            "delta_m":              delta,
            "abs_delta_m":          abs_d,
            "old_confidence":       old_conf,
            "new_confidence":       new_conf,
            "old_none":             old_none,
            "new_none":             new_none,
            "both_none":            old_none and new_none,
            "new_partial_coverage": partial,
        })

    df = pd.DataFrame(rows)
    df["agreement_category"] = df.apply(_categorise, axis=1)

    out_path = comparison_dir / f"comparison_{year}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("  Wrote %s", out_path)

    cats      = df["agreement_category"].value_counts().to_dict()
    none_rate = df["new_none"].mean() * 100
    logger.info(
        "  Summary: %s | new_none=%.1f%%",
        "  ".join(f"{k}={v}" for k, v in sorted(cats.items())),
        none_rate,
    )

    # ── GeoJSON export (optional — requires profiles parquet) ─────────────
    if profiles_path is not None and profiles_path.exists():
        profiles_df = pd.read_parquet(profiles_path)
        _export_detector_geojsons(df, profiles_df, comparison_dir, year)
    elif profiles_path is not None:
        logger.warning("profiles_path supplied but not found: %s — skipping GeoJSON", profiles_path)

    return df



# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", required=True, type=Path,
        help="Directory containing features_{year}.parquet files.",
    )
    parser.add_argument(
        "--transitions-dir", required=True, type=Path,
        help="Directory containing transitions_{year}.parquet files (interpret.py output).",
    )
    parser.add_argument(
        "--profiles-dir", default=None, type=Path,
        help=(
            "Directory containing profiles_{year}.parquet files.  "
            "Required for GeoJSON/GPKG export.  If omitted, export is skipped."
        ),
    )
    parser.add_argument(
        "--comparison-dir", required=True, type=Path,
        help="Output directory for comparison parquets and GeoJSON exports.",
    )
    parser.add_argument(
        "--band-config", required=True, type=Path,
        help="Path to INPUT/band_config.json.",
    )
    parser.add_argument(
        "--year", type=str, default=None,
        help="Process a single year only (default: all available years).",
    )
    parser.add_argument(
        "--skip-geojson", action="store_true",
        help="Skip GeoJSON and GPKG export; write comparison parquets only.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    args.comparison_dir.mkdir(parents=True, exist_ok=True)

    band_config = _load_band_config(args.band_config)

    if args.year:
        years = [args.year]
    else:
        years = sorted(
            p.stem.replace("features_", "")
            for p in args.features_dir.rglob("features_*.parquet")
        )

    if not years:
        logger.error("No features_*.parquet files found under %s", args.features_dir)
        return

    logger.info("Years to process: %s", years)

    do_geojson = (not args.skip_geojson) and (args.profiles_dir is not None)
    if not args.skip_geojson and args.profiles_dir is None:
        logger.warning(
            "--profiles-dir not supplied; GeoJSON/GPKG export will be skipped.  "
            "Pass --profiles-dir to enable it, or --skip-geojson to silence this warning."
        )

    all_dfs = []
    exported_years = []   # years for which GeoJSON export succeeded (both detectors)

    for year in years:
        # Support both flat and year-subdirectory layouts.
        feat_candidates = [
            args.features_dir / f"features_{year}.parquet",
            args.features_dir / year / f"features_{year}.parquet",
        ]
        trans_candidates = [
            args.transitions_dir / f"transitions_{year}.parquet",
            args.transitions_dir / year / f"transitions_{year}.parquet",
        ]

        features_path    = next((p for p in feat_candidates  if p.exists()), None)
        transitions_path = next((p for p in trans_candidates if p.exists()), None)

        if features_path is None:
            logger.warning("features_%s.parquet not found — skipping.", year)
            continue
        if transitions_path is None:
            logger.warning("transitions_%s.parquet not found — skipping.", year)
            continue
        if year not in band_config:
            logger.warning("Year %s not in band_config — skipping.", year)
            continue

        profiles_path = None
        if do_geojson:
            prof_candidates = [
                args.profiles_dir / f"profiles_{year}.parquet",
                args.profiles_dir / year / f"profiles_{year}.parquet",
            ]
            profiles_path = next((p for p in prof_candidates if p.exists()), None)
            if profiles_path is None:
                logger.warning(
                    "profiles_%s.parquet not found under %s — GeoJSON skipped for this year.",
                    year, args.profiles_dir,
                )

        band_mode = _band_mode_for_year(year, band_config)
        df = run_year(
            year, features_path, transitions_path, band_mode,
            args.comparison_dir, profiles_path,
        )
        all_dfs.append(df)

        # Track years where both per-detector GeoJSONs were written.
        old_geojson  = args.comparison_dir / "old"  / year / f"shellline_{year}.geojson"
        pelt_geojson = args.comparison_dir / "pelt" / year / f"shellline_{year}.geojson"
        if old_geojson.exists() and pelt_geojson.exists():
            exported_years.append(year)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = args.comparison_dir / "comparison_all.parquet"
        combined.to_parquet(combined_path, index=False)
        logger.info("Combined parquet written → %s", combined_path)

        print("\n── Agreement by band mode ─────────────────────────────────")
        pivot = (
            combined
            .groupby(["band_mode", "agreement_category"])
            .size()
            .unstack(fill_value=0)
        )
        print(pivot.to_string())
        print()

    # ── Bundle GeoJSONs into per-detector GeoPackages ─────────────────────
    if exported_years:
        from spectral_classifier.utils.export import bundle_shell_lines_to_gpkg  # noqa: PLC0415

        for detector_name in ("old", "pelt"):
            gpkg_path = bundle_shell_lines_to_gpkg(
                output_root=args.comparison_dir / detector_name,
                years=exported_years,
                output_filename=f"shelllines_{detector_name}.gpkg",
            )
            if gpkg_path:
                # Move the GPKG up to comparison_dir root for easy access in QGIS.
                dest = args.comparison_dir / f"shelllines_{detector_name}.gpkg"
                gpkg_path.rename(dest)
                logger.info("GPKG → %s", dest)

        logger.info(
            "Exported shell lines for %d/%d year(s).  "
            "Load shelllines_old.gpkg and shelllines_pelt.gpkg in QGIS to compare.",
            len(exported_years), len(years),
        )

    # ── Per-year statistics ───────────────────────────────────────────────
    if all_dfs:
        stats_df = _compute_yearly_stats(combined)
        stats_path = args.comparison_dir / "yearly_stats.csv"
        stats_df.to_csv(stats_path, index=False, float_format="%.2f")
        logger.info("Per-year stats → %s", stats_path)
        print("\n── Per-year lateral shift statistics (delta_m = PELT − old, positive = seaward) ──")
        print(stats_df.to_string(index=False))
        print()


# ── Per-year statistics ───────────────────────────────────────────────────────

def _compute_yearly_stats(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-year summary statistics from the combined comparison DataFrame.

    Focuses on signed delta_m (new_distance − old_distance) to surface
    systematic lateral bias between the two detectors.  Positive values
    indicate PELT places the shell line further seaward than the old detector.

    Only rows where both detectors fired (old_none=False, new_none=False) are
    used for the distance/delta statistics, since NaN distances would skew
    means.  Coverage counts (n_old_only, n_new_only, etc.) are computed over
    all transects.

    Columns in output
    -----------------
    year, band_mode
    n_transects         — total transects processed
    n_both             — both detectors fired
    n_old_only         — old fired, PELT returned None
    n_new_only         — PELT fired, old returned None
    n_both_none        — neither fired
    pct_new_none       — % of transects where PELT returned None
    mean_old_dist      — mean shell line position from old detector (m from landward end)
    mean_new_dist      — mean shell line position from PELT (m from landward end)
    mean_delta_m       — mean signed shift (positive = PELT more seaward)
    median_delta_m     — median signed shift
    std_delta_m        — spread of shifts
    p25_delta_m        — 25th percentile
    p75_delta_m        — 75th percentile
    pct_close          — % of both-fired rows with abs_delta < 5 m
    pct_moderate       — % of both-fired rows with 5 ≤ abs_delta < 20 m
    pct_divergent      — % of both-fired rows with abs_delta ≥ 20 m
    """
    records = []

    for (year, band_mode), grp in combined.groupby(["year", "band_mode"]):
        n_total    = len(grp)
        n_old_only = int(((grp["old_none"] == False) & (grp["new_none"] == True)).sum())  # noqa: E712
        n_new_only = int(((grp["old_none"] == True)  & (grp["new_none"] == False)).sum())  # noqa: E712
        n_both_none = int(grp["both_none"].sum())

        # Subset where both detectors fired — used for all distance statistics.
        both = grp[~grp["old_none"] & ~grp["new_none"]]
        n_both = len(both)

        if n_both > 0:
            delta      = both["delta_m"]
            mean_old   = both["old_distance"].mean()
            mean_new   = both["new_distance"].mean()
            mean_d     = delta.mean()
            median_d   = delta.median()
            std_d      = delta.std()
            p25        = delta.quantile(0.25)
            p75        = delta.quantile(0.75)

            cats = both["agreement_category"].value_counts()
            pct_close     = 100 * cats.get("close",     0) / n_both
            pct_moderate  = 100 * cats.get("moderate",  0) / n_both
            pct_divergent = 100 * cats.get("divergent", 0) / n_both
        else:
            mean_old = mean_new = mean_d = median_d = std_d = p25 = p75 = float("nan")
            pct_close = pct_moderate = pct_divergent = float("nan")

        pct_new_none = 100 * grp["new_none"].mean()

        records.append({
            "year":           int(year),
            "band_mode":      band_mode,
            "n_transects":    n_total,
            "n_both":         n_both,
            "n_old_only":     n_old_only,
            "n_new_only":     n_new_only,
            "n_both_none":    n_both_none,
            "pct_new_none":   round(pct_new_none, 1),
            "mean_old_dist":  mean_old,
            "mean_new_dist":  mean_new,
            "mean_delta_m":   mean_d,
            "median_delta_m": median_d,
            "std_delta_m":    std_d,
            "p25_delta_m":    p25,
            "p75_delta_m":    p75,
            "pct_close":      pct_close,
            "pct_moderate":   pct_moderate,
            "pct_divergent":  pct_divergent,
        })

    return pd.DataFrame(records).sort_values("year").reset_index(drop=True)


if __name__ == "__main__":
    main()