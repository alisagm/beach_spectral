#!/usr/bin/env python3
"""
Compute transect intercept distances from manually digitised shorelines.

Reads a shoreline shapefile (PAIS_alldata/features/PAIS_shoreline.shp) and
the transects GeoJSON, intersects each transect with each shoreline feature,
and records the intercept distance (metres from the landward/west end).

Output: manual_transitions.parquet
Schema
------
    transect_id      int    — matches transect_id in profiles/features Parquets
    year             str    — from --year-col attribute, or --year fallback
    distance         float  — metres from landward (lowest-easting) end
    x                float  — easting of intercept point (source CRS units)
    y                float  — northing of intercept point (source CRS units)
    n_intersections  int    — number of raw intersections found before
                              keeping the most-seaward; >1 flags geometry
                              worth inspecting (oblique crossings, digitising
                              artefacts, or MultiLineString components that
                              both crossed the same transect)

Notes
-----
- Distance convention matches sampler.py: distance=0 is the lowest-easting
  (landward/west) end of the transect.  shapely.project() measures from
  coords[0], which may be the seaward end — the direction check and optional
  flip in _intercept_distance() corrects for this.
- The loop is (transect × shapefile row).  Each shapefile row is treated as
  an independent geometry, so coverage-gap segments for the same year produce
  separate rows in the intersect result — they are concatenated, not merged.
  Since gap segments don't overlap transect-wise you will get at most one row
  per (transect_id, year) pair.  The duplicate-check in the diagnostics block
  will surface any violation of that assumption.
- If no --year-col exists in the shapefile and no --year fallback is supplied,
  year is set to 'unknown' and a WARNING is emitted.

Usage
-----
# Typical call — shapefile has a 'Year' attribute column:
python scripts/compute_manual_transitions.py \\
    --shoreline  PAIS_alldata/features/PAIS_shoreline.shp \\
    --transects  INPUT/shorelineTransPais.json \\
    --output     OUTPUT/manual_transitions.parquet \\
    --year-col   Year

# Shapefile has no year attribute; tag everything with a single year:
python scripts/compute_manual_transitions.py \\
    --shoreline  PAIS_alldata/features/PAIS_shoreline.shp \\
    --transects  INPUT/shorelineTransPais.json \\
    --output     OUTPUT/manual_transitions.parquet \\
    --year       2014

# Override working CRS (default EPSG:26914 = UTM Zone 14N, NAD83):
python scripts/compute_manual_transitions.py \\
    --shoreline  PAIS_alldata/features/PAIS_shoreline.shp \\
    --transects  INPUT/shorelineTransPais.json \\
    --output     OUTPUT/manual_transitions.parquet \\
    --year-col   Year \\
    --crs        EPSG:26914
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPoint, Point

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_TRANSECTS  = Path("INPUT/shorelineTransPais.json")
DEFAULT_OUTPUT     = Path("OUTPUT/manual_transitions.parquet")
DEFAULT_CRS        = "EPSG:26914"   # UTM Zone 14N / NAD83 — matches PAIS imagery
TRANSECT_ID_COL    = "TransectID"   # PascalCase in the GeoJSON properties


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _landward_is_first(line) -> bool:
    """
    Return True if coords[0] is the landward (lowest-easting) end.

    shapely.project() always measures distance from coords[0].
    In shorelineTransPais.json the first vertex is seaward (higher easting),
    so this will typically return False — the caller must flip accordingly.
    """
    coords = list(line.coords)
    return coords[0][0] < coords[-1][0]


def _intercept_distance(transect_geom, point_geom) -> float:
    """
    Distance (m) from the LANDWARD end of the transect to the intersection point.

    Corrects for the fact that shapely.project() measures from coords[0],
    which is the seaward end in shorelineTransPais.json.
    """
    raw = transect_geom.project(point_geom)
    if _landward_is_first(transect_geom):
        return raw
    else:
        return transect_geom.length - raw


def _collect_points(intersection) -> List[Point]:
    """
    Flatten a shapely intersection result into a list of Points.

    Handles:
        Point           — single clean crossing
        MultiPoint      — multiple crossings (transect crosses line >1 times)
        LineString /
        GeometryCollection — degenerate overlap; use centroid and warn
    """
    geom_type = intersection.geom_type

    if geom_type == "Point":
        return [intersection]

    if geom_type == "MultiPoint":
        return list(intersection.geoms)

    # GeometryCollection or LineString — almost always a digitising artefact
    # (shoreline runs nearly parallel to transect for a stretch).
    # Use the centroid as a single representative point.
    logger.warning(
        "Intersection returned %s (expected Point/MultiPoint). "
        "Using centroid as representative point — inspect this transect.",
        geom_type,
    )
    return [intersection.centroid]


# ── Core logic ────────────────────────────────────────────────────────────────

def compute_intercepts(
    shorelines: gpd.GeoDataFrame,
    transects:  gpd.GeoDataFrame,
    year_col:   Optional[str],
    year_fallback: Optional[str],
) -> pd.DataFrame:
    """
    Intersect every transect against every shoreline row.

    Parameters
    ----------
    shorelines :
        Shoreline GeoDataFrame, already reprojected to the working CRS.
    transects :
        Transect GeoDataFrame, already reprojected to the working CRS.
    year_col :
        Attribute column in *shorelines* that contains the year string.
        If None, *year_fallback* is used for all rows.
    year_fallback :
        Year string to use when *year_col* is None or missing from a row.

    Returns
    -------
    pd.DataFrame with columns:
        transect_id, year, distance, x, y, n_intersections
    """
    n_transects  = len(transects)
    n_shorelines = len(shorelines)
    logger.info(
        "Intersecting %d transects × %d shoreline features",
        n_transects, n_shorelines,
    )

    records: List[dict] = []
    skipped_empty = 0

    for sl_idx, sl_row in shorelines.iterrows():
        # Resolve year label for this shoreline segment
        if year_col and year_col in sl_row.index and pd.notna(sl_row[year_col]):
            year_label = str(sl_row[year_col])
        elif year_fallback:
            year_label = year_fallback
        else:
            year_label = "unknown"
            logger.warning(
                "Shoreline row %s has no year attribute and no --year fallback "
                "was supplied; tagging as 'unknown'.",
                sl_idx,
            )

        for tx_idx, tx_row in transects.iterrows():
            intersection = tx_row.geometry.intersection(sl_row.geometry)

            if intersection.is_empty:
                skipped_empty += 1
                continue

            points = _collect_points(intersection)
            n      = len(points)

            distances = [_intercept_distance(tx_row.geometry, p) for p in points]

            # Retain the most-seaward intercept (largest distance from landward end).
            # For clean data n==1 almost always; n>1 is flagged via n_intersections.
            best_idx = max(range(n), key=lambda i: distances[i])
            pt       = points[best_idx]

            records.append({
                "transect_id":     int(tx_row[TRANSECT_ID_COL]),
                "year":            year_label,
                "distance":        float(distances[best_idx]),
                "x":               float(pt.x),
                "y":               float(pt.y),
                "n_intersections": n,
            })

    logger.info(
        "Found %d intercepts; %d (transect, shoreline) pairs had no crossing.",
        len(records), skipped_empty,
    )
    return pd.DataFrame(records)


# ── Diagnostics ───────────────────────────────────────────────────────────────

def print_diagnostics(df: pd.DataFrame, n_transects: int) -> None:
    """Print a summary of intersection results to stdout."""
    sep = "─" * 60

    print(f"\n{sep}")
    print("  MANUAL TRANSITIONS — INTERSECTION SUMMARY")
    print(sep)

    if df.empty:
        print("  WARNING: no intersections found — check CRS alignment.")
        print(sep)
        return

    n_intersecting = df["transect_id"].nunique()
    n_missing      = n_transects - n_intersecting
    n_multi        = (df["n_intersections"] > 1).sum()

    print(f"  Total transects          : {n_transects}")
    print(f"  Transects with intercept : {n_intersecting}  "
          f"({100 * n_intersecting / n_transects:.1f} %)")
    print(f"  Transects without        : {n_missing}  "
          f"(expected — imagery coverage gaps)")
    print(f"  Rows with n_intersections > 1 : {n_multi}  "
          f"← inspect if unexpectedly high")

    # Per-year breakdown
    print(f"\n  Per-year breakdown:")
    year_summary = (
        df.groupby("year")
        .agg(
            n_intercepts   = ("transect_id", "count"),
            dist_mean      = ("distance", "mean"),
            dist_min       = ("distance", "min"),
            dist_max       = ("distance", "max"),
            multi_flag     = ("n_intersections", lambda s: (s > 1).sum()),
        )
        .reset_index()
        .sort_values("year")
    )
    for _, row in year_summary.iterrows():
        print(
            f"    {row['year']}  "
            f"n={row['n_intercepts']:>5}  "
            f"dist mean={row['dist_mean']:>6.1f} m  "
            f"[{row['dist_min']:.1f}, {row['dist_max']:.1f}]  "
            f"multi={row['multi_flag']}"
        )

    # Duplicate (transect_id, year) check — should always be 0
    dupes = df.groupby(["transect_id", "year"]).size()
    n_dupes = (dupes > 1).sum()
    if n_dupes > 0:
        print(
            f"\n  WARNING: {n_dupes} (transect_id, year) pairs have >1 row — "
            "overlapping shoreline segments detected.  "
            "Most-seaward point was kept per intersection, but check your "
            "shapefile for segment overlaps."
        )
    else:
        print(f"\n  No duplicate (transect_id, year) pairs — OK")

    print(f"\n  Distance distribution across all years (m):")
    desc = df["distance"].describe()
    for stat in ("count", "mean", "std", "min", "25%", "50%", "75%", "max"):
        print(f"    {stat:>6}  {desc[stat]:>8.2f}")

    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute transect intercepts from manually digitised shorelines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--shoreline", "-s",
        type=Path,
        default=Path("PAIS_alldata/features/PAIS_shoreline.shp"),
        help=(
            "Path to manually digitised shoreline shapefile "
            "(default: PAIS_alldata/features/PAIS_shoreline.shp)"
        ),
    )
    parser.add_argument(
        "--transects", "-t",
        type=Path,
        default=DEFAULT_TRANSECTS,
        help=f"Path to transects GeoJSON (default: {DEFAULT_TRANSECTS})",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output Parquet path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--year-col",
        type=str,
        default=None,
        metavar="COL",
        help=(
            "Shapefile attribute column containing the year string "
            "(e.g. 'Year').  Takes priority over --year."
        ),
    )
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        metavar="YYYY",
        help=(
            "Fallback year tag to apply to all features when --year-col "
            "is not supplied or the attribute is missing."
        ),
    )
    parser.add_argument(
        "--crs",
        type=str,
        default=DEFAULT_CRS,
        help=(
            f"Working CRS for intersection (default: {DEFAULT_CRS}). "
            "Both inputs are reprojected to this CRS before intersecting."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging to stderr.",
    )

    return parser.parse_args()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)

    # ── Validate inputs ───────────────────────────────────────────────────
    if not args.shoreline.exists():
        logger.error("Shoreline shapefile not found: %s", args.shoreline)
        sys.exit(1)
    if not args.transects.exists():
        logger.error("Transects GeoJSON not found: %s", args.transects)
        sys.exit(1)
    if args.year_col is None and args.year is None:
        logger.warning(
            "Neither --year-col nor --year supplied.  "
            "Year will be recorded as 'unknown' for all features."
        )

    # ── Load ──────────────────────────────────────────────────────────────
    logger.info("Loading shorelines: %s", args.shoreline)
    shorelines = gpd.read_file(args.shoreline)
    logger.info(
        "  %d feature(s), CRS: %s, geometry types: %s",
        len(shorelines),
        shorelines.crs,
        shorelines.geometry.geom_type.unique().tolist(),
    )

    logger.info("Loading transects:  %s", args.transects)
    transects = gpd.read_file(args.transects)
    logger.info(
        "  %d transect(s), CRS: %s",
        len(transects),
        transects.crs,
    )

    # Log available shoreline columns so the user can pick --year-col easily
    if args.year_col is None:
        non_geom = [c for c in shorelines.columns if c != "geometry"]
        logger.info(
            "Shoreline attribute columns (use --year-col to tag by year): %s",
            non_geom,
        )

    # ── Reproject to working CRS ──────────────────────────────────────────
    logger.info("Reprojecting both layers to working CRS: %s", args.crs)
    shorelines = shorelines.to_crs(args.crs)
    transects  = transects.to_crs(args.crs)

    # ── Sanity-check: verify direction convention on first transect ───────
    sample_coords = list(transects.iloc[0].geometry.coords)
    first_x, last_x = sample_coords[0][0], sample_coords[-1][0]
    if first_x > last_x:
        logger.debug(
            "First transect: coords[0].x=%.1f > coords[-1].x=%.1f "
            "(seaward-first layout confirmed — intercept distances will be flipped).",
            first_x, last_x,
        )
    else:
        logger.debug(
            "First transect: coords[0].x=%.1f < coords[-1].x=%.1f "
            "(landward-first layout — no flip needed).",
            first_x, last_x,
        )

    # ── Intersect ─────────────────────────────────────────────────────────
    df = compute_intercepts(
        shorelines    = shorelines,
        transects     = transects,
        year_col      = args.year_col,
        year_fallback = args.year,
    )

    # ── Print diagnostics ─────────────────────────────────────────────────
    print_diagnostics(df, n_transects=len(transects))

    if df.empty:
        logger.error(
            "No intersections found — output not written.  "
            "Check that the shoreline and transect files overlap spatially "
            "and that the CRS is correct."
        )
        sys.exit(1)

    # ── Save ──────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    logger.info("Saved %d rows → %s", len(df), args.output)


if __name__ == "__main__":
    main()