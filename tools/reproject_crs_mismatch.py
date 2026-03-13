#!/usr/bin/env python3
"""
tools/reproject_crs_mismatch.py

One-time pre-processing step: reprojects raster tiles that are in the wrong
CRS to EPSG:26914 (NAD83 / UTM zone 14N) before running the main clipping
pipeline.

Background
----------
This script targets tiles flagged as "Wrong CRS" by clip_imagery_to_footprint.py
--dry-run.  For PAIS data this is typically the 2020 imagery (EPSG:6343,
NAD83(2011) / UTM zone 14N), which shares the same zone and axis orientation
as EPSG:26914 but uses a different datum realisation.  The horizontal shift
is centimetre-scale — negligible at 0.5–1 m pixel resolution.

Approach
--------
* Uses rasterio.warp.calculate_default_transform to derive an output pixel
  grid that fully encloses the reprojected source extent, avoiding coverage
  gaps at tile edges.
* Resamples with bilinear interpolation — correct for continuous spectral
  data; avoids staircase artefacts from nearest-neighbour.
* Outputs are written to <imagery_root>/<year>_reprojected/ mirroring the
  source directory structure, so raw tiles are never modified.
* Skips files that already exist in the output location (safe to re-run).
* Preserves all rasterio-readable metadata (band count, dtype, nodata,
  GeoTIFF tags) via src.meta.copy() + dst.update_tags().

Limitations
-----------
JP2-embedded XML metadata (acquisition date, sensor model) is not carried
through — rasterio does not expose it.  If provenance from those fields is
needed, extract them separately with gdalinfo before running this script.

Usage
-----
  # Dry run: list files that would be reprojected
  python tools/reproject_crs_mismatch.py \\
      --imagery-root PAIS_shorelines/imagery \\
      --source-crs 6343 \\
      --dry-run

  # Reproject
  python tools/reproject_crs_mismatch.py \\
      --imagery-root PAIS_shorelines/imagery \\
      --source-crs 6343

  # Reproject a specific year subdirectory only
  python tools/reproject_crs_mismatch.py \\
      --imagery-root PAIS_shorelines/imagery/2020 \\
      --source-crs 6343

  # Overwrite existing outputs (default is skip)
  python tools/reproject_crs_mismatch.py \\
      --imagery-root PAIS_shorelines/imagery \\
      --source-crs 6343 \\
      --overwrite
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import rasterio
from pyproj import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_CRS = CRS.from_epsg(26914)

# Raster formats this script will attempt to reproject.
READABLE_FORMATS = {".tif", ".tiff", ".jp2"}

# Subdirectory suffix appended to the immediate parent of each source tile.
# e.g.  imagery/2020/tile.tif  →  imagery/2020_reprojected/tile.tif
OUTPUT_SUFFIX = "_reprojected"


# ---------------------------------------------------------------------------
# Core reprojection
# ---------------------------------------------------------------------------

def reproject_tile(
    src_path: Path,
    dst_path: Path,
    target_crs: CRS,
    resampling: Resampling = Resampling.bilinear,
) -> None:
    """
    Reproject a single raster tile and write the result as a GeoTIFF.

    calculate_default_transform derives an output pixel grid that fully
    encloses the reprojected source extent, so no edge pixels are clipped.

    Parameters
    ----------
    src_path:
        Input raster (any GDAL-readable format).
    dst_path:
        Output path.  Parent directories are created as needed.
    target_crs:
        Target coordinate reference system.
    resampling:
        Resampling algorithm.  Default is bilinear, which is appropriate for
        continuous spectral data.  Use Resampling.nearest only for classified
        or categorical rasters.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
        )

        meta = src.meta.copy()
        meta.update(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            driver="GTiff",
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        # Remove JP2-specific keys that GTiff does not recognise.
        for key in ("REVERSIBLE", "QUALITY", "BLOCKXSIZE", "BLOCKYSIZE"):
            meta.pop(key, None)

        with rasterio.open(dst_path, "w", **meta) as dst:
            # Carry over any rasterio-readable tags (e.g. AREA_OR_POINT).
            dst.update_tags(**src.tags())

            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                )


# ---------------------------------------------------------------------------
# Discovery + output path resolution
# ---------------------------------------------------------------------------

def find_source_tiles(
    imagery_root: Path,
    source_crs: CRS,
) -> list[tuple[Path, str]]:
    """
    Return all raster files under imagery_root whose CRS matches source_crs.

    Returns a list of (path, crs_string) tuples.  Files that cannot be opened
    are skipped with a warning.
    """
    all_files = sorted(imagery_root.rglob("*"))
    candidates = [f for f in all_files if f.suffix.lower() in READABLE_FORMATS]

    matches: list[tuple[Path, str]] = []

    for path in candidates:
        try:
            with rasterio.open(path) as src:
                file_crs = CRS(src.crs.to_wkt())
                if file_crs == source_crs:
                    matches.append((path, src.crs.to_string()))
        except Exception as exc:
            print(f"  WARNING  could not open {path.name}: {exc}")

    return matches


def output_path_for(src_path: Path, imagery_root: Path) -> Path:
    """
    Derive the output path for a source tile.

    The immediate year directory (first component of the relative path) gets
    OUTPUT_SUFFIX appended.  All deeper structure is preserved.

    Examples
    --------
    imagery_root = Path("imagery")
    src_path     = Path("imagery/2020/block_A/tile_001.tif")
    →              Path("imagery/2020_reprojected/block_A/tile_001.tif")

    If the source is directly under imagery_root (no subdirectory), the output
    is placed in imagery_root / ("_reprojected") / tile.tif.
    """
    rel = src_path.relative_to(imagery_root)
    parts = rel.parts  # e.g. ("2020", "block_A", "tile_001.tif")

    if len(parts) == 1:
        # File sits directly under imagery_root — no year directory.
        out_dir = imagery_root / ("_reprojected")
    else:
        # Append suffix to the first path component (year folder).
        out_dir = imagery_root / (parts[0] + OUTPUT_SUFFIX) / Path(*parts[1:-1])

    return out_dir / src_path.name


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run(
    imagery_root: Path,
    source_crs: CRS,
    target_crs: CRS,
    dry_run: bool,
    overwrite: bool,
) -> None:
    print(f"Scanning for tiles in {source_crs.to_epsg() or source_crs} under: {imagery_root}")
    tiles = find_source_tiles(imagery_root, source_crs)

    if not tiles:
        print("No matching tiles found.  Check --source-crs and --imagery-root.")
        sys.exit(0)

    # Classify: skip already-done, reproject the rest.
    to_reproject: list[tuple[Path, Path]] = []
    skipped_existing = 0

    for src_path, _ in tiles:
        dst_path = output_path_for(src_path, imagery_root)
        if dst_path.exists() and not overwrite:
            skipped_existing += 1
        else:
            to_reproject.append((src_path, dst_path))

    # ── Report ──────────────────────────────────────────────────────────────
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"Reprojection report  ({len(tiles)} matching tiles found)")
    print(sep)
    print(f"  Source CRS  : {source_crs.to_epsg() or source_crs}")
    print(f"  Target CRS  : {target_crs.to_epsg() or target_crs}")
    print(f"  Resampling  : bilinear")
    print(f"  To reproject: {len(to_reproject)}")
    print(f"  Already done: {skipped_existing}  (use --overwrite to redo)")
    if dry_run:
        print("\n  Dry run — files that WOULD be reprojected:")
        for src_path, dst_path in to_reproject[:20]:
            print(f"    {src_path.name}  →  .../{dst_path.parent.name}/{dst_path.name}")
        if len(to_reproject) > 20:
            print(f"    … and {len(to_reproject) - 20} more")
    print(f"{sep}\n")

    if dry_run:
        print("Dry run complete.  No files were written.")
        return

    if not to_reproject:
        print("Nothing to do.")
        return

    # ── Reproject ────────────────────────────────────────────────────────────
    succeeded = 0
    failed = 0
    t0 = time.time()
    n = len(to_reproject)

    for i, (src_path, dst_path) in enumerate(to_reproject, 1):
        try:
            reproject_tile(src_path, dst_path, target_crs)
            succeeded += 1
        except Exception as exc:
            print(f"\n  ERROR  {src_path.name}: {exc}")
            failed += 1

        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0
        eta = (n - i) / rate if rate > 0 else 0
        print(
            f"\r  [{i:4d}/{n}]  ok={succeeded}  fail={failed}  "
            f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
            end="",
            flush=True,
        )

    print()
    print(f"\nReprojection complete: {succeeded} succeeded, {failed} failed.")

    if succeeded:
        # Show where outputs landed.
        sample_src, sample_dst = to_reproject[0]
        print(
            f"\nOutputs written under: .../{sample_dst.parent.parent.name}/\n"
            f"Point clip_imagery_to_footprint.py --imagery-root at that directory\n"
            f"(or include it alongside your existing tiles in a merged root)."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--imagery-root", "-i",
        type=Path,
        required=True,
        help="Root directory to scan for wrong-CRS tiles (searched recursively)",
    )
    p.add_argument(
        "--source-crs",
        type=int,
        default=6343,
        metavar="EPSG",
        help="EPSG code of the CRS to convert FROM (default: 6343)",
    )
    p.add_argument(
        "--target-crs",
        type=int,
        default=26914,
        metavar="EPSG",
        help="EPSG code of the CRS to convert TO (default: 26914)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be reprojected without writing anything",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (default is skip)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.imagery_root.exists():
        print(f"ERROR: imagery root does not exist: {args.imagery_root}")
        sys.exit(1)

    source_crs = CRS.from_epsg(args.source_crs)
    target_crs = CRS.from_epsg(args.target_crs)

    run(
        imagery_root=args.imagery_root,
        source_crs=source_crs,
        target_crs=target_crs,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()