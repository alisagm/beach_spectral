#!/usr/bin/env python3
"""
tools/clip_imagery_to_footprint.py

Pre-processing step: clips all imagery tiles to the transect analysis footprint
before running the main shoreline detection pipeline.

This script is intended to be run ONCE (or when the buffer extent changes).
Downstream processing then works on the clipped imagery, which is much faster
to load and reduces memory overhead.

Footprint definition
--------------------
For each transect in the input GeoJSON:
  1. Identify the eastern (seaward) and western (landward) endpoints by easting.
  2. Extend the transect 1000 m east and 100 m west along the transect axis.
  3. Buffer the extended line by half the transect spacing (25 m) → polygon.
  4. Union all per-transect polygons → analysis footprint.

The footprint is saved to disk so it can be inspected in QGIS and reused
without rebuilding it on every run.

Imagery requirements
--------------------
Files in .tif, .tiff, or .jp2 format in the target CRS (default EPSG:26914)
are processed.  All outputs are written as GeoTIFF regardless of input format,
so JP2 tiles are converted implicitly during clipping.

Files in other formats (.sid, .ecw) or other CRSes trigger a preflight
error.  For CRS mismatches, use tools/reproject_crs_mismatch.py to
reproject to EPSG:26914, then rerun this script.  For other formats,
convert with gdal_translate first.

Usage
-----
  python tools/clip_imagery_to_footprint.py \\
      --imagery-root PAIS_shorelines/imagery \\
      --transects INPUT/shorelineTransPais.json \\
      --output PAIS_shorelines/imagery_clipped

  # Rebuild footprint even if it already exists on disk
  python tools/clip_imagery_to_footprint.py ... --rebuild-footprint

  # Adjust buffer distances (metres)
  python tools/clip_imagery_to_footprint.py ... --west-buffer 200 --east-buffer 1500

  # Dry run: show preflight report without clipping anything
  python tools/clip_imagery_to_footprint.py ... --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import rasterio.mask
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds
from pyproj import CRS
from shapely.geometry import LineString, box as shapely_box, shape
from shapely.ops import unary_union


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_CRS = CRS.from_epsg(26914)

# Raster formats that rasterio/GDAL can read directly.  All outputs are
# written as GeoTIFF regardless of input format, so JP2 tiles are converted
# implicitly by clip_tile.  Formats NOT in this set (e.g. .sid, .ecw) are
# flagged as wrong_format — they require gdal_translate pre-conversion.
READABLE_FORMATS = {".tif", ".tiff", ".jp2"}

# Half the transect spacing (50 m apart → 25 m half-width).
# Buffers adjacent transect polygons so their union forms a continuous corridor.
DEFAULT_HALF_WIDTH_M = 25.0

DEFAULT_WEST_BUFFER_M = 100.0
DEFAULT_EAST_BUFFER_M = 1000.0

# Files whose boolean mask array would exceed this threshold are routed to
# gdalwarp (streaming, constant RAM) instead of rasterio.mask.mask() (which
# allocates the full mask up-front).  Adjust via --max-ram-gb.
DEFAULT_MAX_MASK_GB = 4.0

FOOTPRINT_FILENAME = "analysis_footprint.geojson"


# ---------------------------------------------------------------------------
# Step 1: Build analysis footprint
# ---------------------------------------------------------------------------

def _transect_endpoints(geom: LineString) -> tuple[tuple, tuple]:
    """
    Return the (western, eastern) endpoints of a transect by easting value.

    DSAS exports often duplicate the first coordinate, so we deduplicate
    before selecting endpoints.
    """
    coords = list(geom.coords)

    # Remove consecutive duplicates (DSAS artifact)
    unique: list[tuple] = [coords[0]]
    for c in coords[1:]:
        if c != unique[-1]:
            unique.append(c)

    pt_west = min(unique, key=lambda c: c[0])  # lowest easting
    pt_east = max(unique, key=lambda c: c[0])  # highest easting
    return pt_west, pt_east


def _extend_transect(
    pt_west: tuple,
    pt_east: tuple,
    west_m: float,
    east_m: float,
) -> LineString:
    """
    Extend a transect line beyond both endpoints along the transect axis.

    The extension direction is computed from the transect's own orientation,
    so the buffer follows the shoreline curvature rather than the cardinal grid.
    """
    dx = pt_east[0] - pt_west[0]
    dy = pt_east[1] - pt_west[1]
    length = np.hypot(dx, dy)

    if length < 1.0:
        raise ValueError(
            f"Degenerate transect: endpoints are {length:.2f} m apart — "
            "check for duplicate coordinates in the GeoJSON."
        )

    # Unit vector pointing west → east (seaward direction)
    ux, uy = dx / length, dy / length

    ext_east = (pt_east[0] + east_m * ux, pt_east[1] + east_m * uy)
    ext_west = (pt_west[0] - west_m * ux, pt_west[1] - west_m * uy)

    return LineString([ext_west, ext_east])


def build_analysis_footprint(
    transect_path: Path,
    west_m: float = DEFAULT_WEST_BUFFER_M,
    east_m: float = DEFAULT_EAST_BUFFER_M,
    half_width_m: float = DEFAULT_HALF_WIDTH_M,
    close_dist_m: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """
    Build the analysis footprint as a union of per-transect asymmetric buffers.

    Parameters
    ----------
    transect_path:
        Path to the transect GeoJSON (each feature is a LineString).
    west_m:
        Extension distance beyond the western (landward) endpoint.
    east_m:
        Extension distance beyond the eastern (seaward) endpoint.
    half_width_m:
        Buffer radius applied to each extended transect line.
        Should be ≥ half the transect spacing to avoid gaps.
    close_dist_m:
        Morphological buffer for closing gaps between transects of different azimuths.
        Default is half_width_m * 1.2.

    Returns
    -------
    Single-row GeoDataFrame containing the footprint polygon, in the
    same CRS as the transect file.
    """
    print(f"Loading transects from: {transect_path}")
    gdf = gpd.read_file(transect_path)

    if gdf.crs is None:
        raise ValueError("Transect GeoJSON has no CRS. Set it before running this script.")

    print(f"  {len(gdf)} transects loaded (CRS: {gdf.crs})")
    print(f"  Buffer: {west_m} m west, {east_m} m east, ±{half_width_m} m width")

    polys = []
    skipped = 0

    for row in gdf.itertuples():
        try:
            pt_west, pt_east = _transect_endpoints(row.geometry)
            extended = _extend_transect(pt_west, pt_east, west_m, east_m)
            # cap_style=2 → flat end caps (no semicircles at the tips)
            poly = extended.buffer(half_width_m, cap_style=2)
            polys.append(poly)
        except ValueError as exc:
            print(f"  WARNING transect {getattr(row, 'TransectID', row.Index)}: {exc}")
            skipped += 1

    if not polys:
        raise RuntimeError("No valid transect polygons were created. Check the input geometry.")

    if skipped:
        print(f"  Skipped {skipped} degenerate transects.")

    print(f"  Merging {len(polys)} per-transect polygons…")
    raw_union = unary_union(polys)

    area_km2 = raw_union.area / 1e6
    print(f"  Raw union area: {area_km2:.2f} km²  ({raw_union.geom_type})")

    # Morphological closing: fill gaps at orientation-change junctions.
    close_dist = close_dist_m if close_dist_m is not None else half_width_m * 1.2
    print(f"  Closing distance: {close_dist:.1f} m")
    footprint = raw_union.buffer(close_dist).buffer(-close_dist)

    print(f"  Footprint area (after closing): {footprint.area / 1e6:.2f} km²")
    
    return gpd.GeoDataFrame(
        {
            "west_buffer_m": [west_m], 
            "east_buffer_m": [east_m],
            "half_width_m": [half_width_m],
            "close_dist_m": [close_dist_m]
        },
        geometry=[footprint],
        crs=gdf.crs,
    )


# ---------------------------------------------------------------------------
# Step 2: Preflight — validate imagery before clipping
# ---------------------------------------------------------------------------

class PreflightResult:
    """Aggregates preflight findings across all imagery files."""

    def __init__(self) -> None:
        self.valid: list[Path] = []           # ready to clip
        self.wrong_format: list[Path] = []   # not in READABLE_FORMATS
        self.wrong_crs: list[tuple] = []     # (path, actual_crs)
        self.no_overlap: list[Path] = []     # .tif but outside footprint
        self.unreadable: list[tuple] = []    # (path, error_msg)

    @property
    def has_errors(self) -> bool:
        return bool(self.wrong_format or self.wrong_crs or self.unreadable)

    def print_report(self) -> None:
        total = (
            len(self.valid)
            + len(self.wrong_format)
            + len(self.wrong_crs)
            + len(self.no_overlap)
            + len(self.unreadable)
        )
        print(f"\n{'─'*60}")
        print(f"Preflight report  ({total} files found)")
        print(f"{'─'*60}")
        print(f"  Ready to clip  : {len(self.valid)}")
        print(f"  No overlap     : {len(self.no_overlap)}  (will skip)")
        print(f"  Wrong format   : {len(self.wrong_format)}  (need GDAL conversion)")
        print(f"  Wrong CRS      : {len(self.wrong_crs)}  (need reprojection)")
        print(f"  Unreadable     : {len(self.unreadable)}  (check file integrity)")

        if self.wrong_format:
            print("\n  Unsupported format — convert with gdal_translate first:")
            for p in self.wrong_format[:10]:
                print(f"    {p.name}")
            if len(self.wrong_format) > 10:
                print(f"    … and {len(self.wrong_format) - 10} more")
            print("  Supported formats: .tif, .tiff, .jp2")

        if self.wrong_crs:
            print("\n  Wrong-CRS files — reproject with tools/reproject_crs_mismatch.py:")
            for p, actual in self.wrong_crs[:10]:
                print(f"    {p.name}  →  {actual}")
            if len(self.wrong_crs) > 10:
                print(f"    … and {len(self.wrong_crs) - 10} more")

        if self.unreadable:
            print("\n  Unreadable files:")
            for p, err in self.unreadable[:5]:
                print(f"    {p.name}: {err}")
            if any(p.suffix.lower() == ".jp2" for p, _ in self.unreadable[:5]):
                print(
                    "\n  NOTE: JP2 files appearing here usually mean GDAL is missing\n"
                    "  the OpenJPEG codec.  Check with: gdalinfo --formats | grep JP2\n"
                    "  OSGeo4W users: install the 'gdal-ecw' or 'gdal-mrsid' package."
                )

        print(f"{'─'*60}\n")


def preflight_imagery(
    imagery_root: Path,
    footprint_geom,
    footprint_crs: CRS,
    target_crs: CRS = TARGET_CRS,
) -> PreflightResult:
    """
    Scan all raster files under imagery_root and classify them.

    Only files in READABLE_FORMATS, in target_crs, and overlapping the
    footprint will end up in result.valid.

    Overlap check ordering
    ----------------------
    Wrong-CRS files are tested for footprint overlap *before* being flagged
    for reprojection.  This avoids reporting tiles for conversion that fall
    entirely outside the analysis corridor — those go straight to no_overlap.
    """
    all_files = sorted(imagery_root.rglob("*"))
    # All recognised raster extensions — superset of READABLE_FORMATS so that
    # unsupported formats (.sid, .ecw) are still detected and reported.
    raster_extensions = {".tif", ".tiff", ".sid", ".ecw", ".img", ".jp2"}
    raster_files = [f for f in all_files if f.suffix.lower() in raster_extensions]

    print(f"Scanning {len(raster_files)} raster files under: {imagery_root}")

    result = PreflightResult()

    for path in raster_files:
        # ── Format check ───────────────────────────────────────────────────
        if path.suffix.lower() not in READABLE_FORMATS:
            result.wrong_format.append(path)
            continue

        # ── Readability + CRS + overlap checks ─────────────────────────────
        try:
            with rasterio.open(path) as src:
                raster_crs = CRS(src.crs.to_wkt())
                raster_box = shapely_box(
                    src.bounds.left, src.bounds.bottom,
                    src.bounds.right, src.bounds.top,
                )

                if raster_crs != target_crs:
                    # Run overlap check in the raster's own CRS before
                    # flagging for reprojection — no point converting tiles
                    # that fall outside the footprint entirely.
                    fp_gdf = gpd.GeoDataFrame(geometry=[footprint_geom], crs=footprint_crs)
                    fp_in_raster_crs = fp_gdf.to_crs(raster_crs).geometry.iloc[0]
                    if not fp_in_raster_crs.intersects(raster_box):
                        result.no_overlap.append(path)
                    else:
                        result.wrong_crs.append((path, src.crs.to_string()))
                    continue

                # CRS matches — check overlap in the shared CRS.
                fp_geom = (
                    gpd.GeoDataFrame(geometry=[footprint_geom], crs=footprint_crs)
                    .to_crs(raster_crs).geometry.iloc[0]
                    if footprint_crs != raster_crs
                    else footprint_geom
                )
                if not fp_geom.intersects(raster_box):
                    result.no_overlap.append(path)
                    continue

        except Exception as exc:
            result.unreadable.append((path, str(exc)))
            continue

        result.valid.append(path)

    return result


# ---------------------------------------------------------------------------
# Step 3: Clip valid tiles to footprint
# ---------------------------------------------------------------------------

def _output_photometric(band_count: int) -> str:
    """Return the GeoTIFF photometric creation option for the given band count.

    For 1-band imagery → 'minisblack' (grayscale).
    For 3- or 4-band imagery (RGB or RGBN) → 'rgb'.

    Without this override GDAL writes ExtraSamples=AssocAlpha on band 4 of
    any 4-band GeoTIFF that lacks an explicit photometric tag.  QGIS then
    treats band 4 as a transparency (alpha) channel, making the imagery
    appear invisible even though band 4 is NIR data.  Setting photometric
    to 'rgb' suppresses the alpha inference.
    """
    return "minisblack" if band_count == 1 else "rgb"


def estimate_mask_gb(src_path: Path) -> float:
    """
    Estimate the peak RAM that rasterio.mask.mask() would allocate for a file.

    rasterio.mask.mask() builds a boolean mask array with the same spatial
    dimensions as the *full* source raster before any cropping occurs.
    Peak allocation ≈ bands × height × width × 1 byte (bool).

    Opens only the file header — no pixel data is read.
    """
    with rasterio.open(src_path) as src:
        return (src.count * src.height * src.width) / 1e9

def clip_tile(
    src_path: Path,
    footprint_geom,
    footprint_crs: CRS,
    output_root: Path,
    imagery_root: Path,
    footprint_path: Optional[Path] = None,
    max_mask_gb: float = DEFAULT_MAX_MASK_GB,
) -> Optional[Path]:
    """
    Clip a single raster tile to the footprint and write the result.

    For files whose boolean mask array would exceed max_mask_gb, the clip is
    delegated to gdalwarp (via clip_tile_gdalwarp) which streams the output
    in blocks and keeps RAM usage roughly constant.  footprint_path must be
    provided for the gdalwarp path; if it is None and the threshold is
    exceeded, the file is skipped with an error message.

    Output paths mirror the input directory structure under output_root
    so that the existing batch processing logic (group_rasters_by_year)
    still works unchanged.

    Returns the output path on success, None on failure.
    """
    # Preserve relative directory structure
    rel = src_path.relative_to(imagery_root)
    out_path = output_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Open file header once: get CRS, bounds, and mask-size estimate ───────
    # We need the tile's own bounding box to clip the footprint down to just
    # the portion that overlaps this tile.  Without this intersection, crop=True
    # (and gdalwarp -crop_to_cutline) trims to the *footprint's* bounding box,
    # which spans the entire study corridor and is larger than any single tile,
    # so the output dimensions are identical to the input.
    try:
        with rasterio.open(src_path) as src:
            raster_crs = CRS(src.crs.to_wkt())
            tile_bbox = shapely_box(
                src.bounds.left, src.bounds.bottom,
                src.bounds.right, src.bounds.top,
            )
            mask_gb = (src.count * src.height * src.width) / 1e9
            # Capture these here — src.nodata / src.count accessed after the
            # with-block exits would rely on rasterio caching closed datasets,
            # which is fragile.  Capture explicitly before the context exits.
            src_nodata = src.nodata if src.nodata is not None else 0
            src_count  = src.count
    except Exception as exc:
        print(f"  ERROR clipping {src_path.name}: could not read header: {exc}")
        return None

    # ── Reproject footprint to raster CRS ────────────────────────────────────
    if footprint_crs != raster_crs:
        fp_gdf = gpd.GeoDataFrame(geometry=[footprint_geom], crs=footprint_crs)
        fp_geom = fp_gdf.to_crs(raster_crs).geometry.iloc[0]
    else:
        fp_geom = footprint_geom

    # ── Intersect footprint with this tile's extent ───────────────────────────
    # This is the core fix: crop=True (and gdalwarp -crop_to_cutline) both trim
    # to the bounding box of the geometry they receive.  By intersecting first,
    # that bounding box is tight to the footprint fragment inside this tile.
    tile_geom = fp_geom.intersection(tile_bbox)
    if tile_geom.is_empty:
        print(f"  WARNING {src_path.name}: footprint intersection is empty — skipping.")
        return None

    # ── Memory guard: route large files to windowed rasterio path ────────────
    if mask_gb > max_mask_gb:
        print(
            f"  INFO {src_path.name}: mask estimate {mask_gb:.1f} GB "
            f"> {max_mask_gb:.1f} GB threshold — using windowed I/O"
        )
        return clip_tile_windowed(src_path, tile_geom, out_path,
                                  nodata=src_nodata, src_count=src_count)


    # ── Standard path: rasterio.mask.mask() ──────────────────────────────────
    try:
        with rasterio.open(src_path) as src:
            clipped, transform = rasterio.mask.mask(
                src,
                [tile_geom],
                crop=True,
                nodata=src_nodata,
                all_touched=True,
            )

            meta = src.meta.copy()
            meta.update(
                {
                    "driver": "GTiff",
                    "height": clipped.shape[1],
                    "width": clipped.shape[2],
                    "transform": transform,
                    "nodata": src_nodata,
                    "photometric": _output_photometric(src_count),
                    "compress": "lzw",
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                }
            )

            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(clipped)

        return out_path

    except Exception as exc:
        print(f"  ERROR clipping {src_path.name}: {exc}")
        return None


def clip_tile_windowed(
    src_path: Path,
    tile_geom,
    out_path: Path,
    nodata: float = 0,
    src_count: Optional[int] = None,
    block_size: int = 256,
) -> Optional[Path]:
    """
    Clip a single raster tile using rasterio windowed I/O.

    Used as a fallback for files whose boolean mask array would exceed the
    configured RAM threshold.  Instead of allocating a mask the size of the
    full source raster, this function:

      1. Computes the output window from tile_geom's bounding box.
      2. Opens the output file with the cropped dimensions.
      3. Iterates over block_size x block_size tiles, reading one block at a
         time, burning the polygon mask into that block only, then writing it.

    Peak RAM is proportional to a single block (block_size^2 x bands x dtype),
    not to the full source raster — identical to the streaming behaviour of
    gdalwarp, but with no external dependency.

    Returns the output path on success, None on failure (soft error).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(src_path) as src:
            # ── Compute output window tight to tile_geom bbox ─────────────────
            minx, miny, maxx, maxy = tile_geom.bounds
            win = window_from_bounds(minx, miny, maxx, maxy, src.transform)
            win = win.round_lengths().round_offsets()
            out_height = int(win.height)
            out_width  = int(win.width)
            out_transform = src.window_transform(win)

            meta = src.meta.copy()
            meta.update({
                "driver":      "GTiff",
                "height":      out_height,
                "width":       out_width,
                "transform":   out_transform,
                "nodata":      nodata,
                "photometric": _output_photometric(src_count if src_count is not None else src.count),
                "compress":    "lzw",
                "tiled":       True,
                "blockxsize":  block_size,
                "blockysize":  block_size,
            })

            with rasterio.open(out_path, "w", **meta) as dst:
                # Iterate over block-sized tiles within the output window
                for row_off in range(0, out_height, block_size):
                    block_h = min(block_size, out_height - row_off)
                    for col_off in range(0, out_width, block_size):
                        block_w = min(block_size, out_width - col_off)

                        # Window in source-raster space
                        src_win = Window(
                            col_off=int(win.col_off) + col_off,
                            row_off=int(win.row_off) + row_off,
                            width=block_w,
                            height=block_h,
                        )
                        # Corresponding window in output-raster space
                        dst_win = Window(col_off, row_off, block_w, block_h)

                        data = src.read(window=src_win)

                        # Burn polygon mask for this block only.
                        # geometry_mask returns True where the geometry is ABSENT
                        # (i.e. pixels to be set to nodata).
                        outside = rasterio.features.geometry_mask(
                            [tile_geom],
                            out_shape=(block_h, block_w),
                            transform=src.window_transform(src_win),
                            all_touched=True,
                        )
                        data[:, outside] = nodata
                        dst.write(data, window=dst_win)

        return out_path

    except Exception as exc:
        print(f"  ERROR clipping {src_path.name} (windowed): {exc}")
        return None


def clip_all(
    valid_paths: list[Path],
    footprint_geom,
    footprint_crs: CRS,
    output_root: Path,
    imagery_root: Path,
    footprint_path: Optional[Path] = None,
    max_mask_gb: float = DEFAULT_MAX_MASK_GB,
) -> None:
    """Clip all valid tiles, printing a single-line progress indicator."""
    n = len(valid_paths)
    succeeded = 0
    failed = 0
    t0 = time.time()

    for i, path in enumerate(valid_paths, 1):
        result = clip_tile(
            path,
            footprint_geom,
            footprint_crs,
            output_root,
            imagery_root,
            footprint_path=footprint_path,
            max_mask_gb=max_mask_gb,
        )

        if result:
            succeeded += 1
        else:
            failed += 1

        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0
        eta = (n - i) / rate if rate > 0 else 0
        print(
            f"\r  [{i:4d}/{n}]  "
            f"ok={succeeded}  fail={failed}  "
            f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
            end="",
            flush=True,
        )

    print()  # newline after progress bar
    print(f"\nClipping complete: {succeeded} succeeded, {failed} failed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--transects", "-t",
        type=Path,
        default=Path("INPUT/shorelineTransPais.json"),
        help="Transect GeoJSON (default: INPUT/shorelineTransPais.json)",
    )
    p.add_argument(
        "--imagery-root", "-i",
        type=Path,
        required=True,
        help="Root directory containing raw imagery tiles",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Root directory for clipped imagery output",
    )
    p.add_argument(
        "--footprint-dir",
        type=Path,
        default=Path("INPUT"),
        help="Directory to save/load the footprint GeoJSON (default: INPUT/)",
    )
    p.add_argument(
        "--west-buffer",
        type=float,
        default=DEFAULT_WEST_BUFFER_M,
        help=f"Landward buffer distance in metres (default: {DEFAULT_WEST_BUFFER_M})",
    )
    p.add_argument(
        "--east-buffer",
        type=float,
        default=DEFAULT_EAST_BUFFER_M,
        help=f"Seaward buffer distance in metres (default: {DEFAULT_EAST_BUFFER_M})",
    )
    p.add_argument(
        "--half-width",
        type=float,
        default=DEFAULT_HALF_WIDTH_M,
        help=f"Per-transect buffer width in metres (default: {DEFAULT_HALF_WIDTH_M})",
    )
    p.add_argument(
        "--rebuild-footprint",
        action="store_true",
        help="Rebuild the footprint even if a saved version exists",
    )
    p.add_argument(
        "--max-ram-gb",
        type=float,
        default=DEFAULT_MAX_MASK_GB,
        help=(
            f"RAM threshold in GB for the boolean mask array (default: {DEFAULT_MAX_MASK_GB}). "
            "Files whose mask would exceed this are clipped with gdalwarp instead of "
            "rasterio.mask.mask(), which streams the output in blocks and avoids "
            "large up-front allocations."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight checks and print report without clipping anything",
    )
    p.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Proceed with clipping even if wrong-format or wrong-CRS files are found",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Step 1: Footprint ──────────────────────────────────────────────────
    footprint_path = args.footprint_dir / FOOTPRINT_FILENAME

    if footprint_path.exists() and not args.rebuild_footprint:
        print(f"Loading existing footprint from: {footprint_path}")
        fp_gdf = gpd.read_file(footprint_path)
    else:
        fp_gdf = build_analysis_footprint(
            args.transects,
            west_m=args.west_buffer,
            east_m=args.east_buffer,
            half_width_m=args.half_width,
        )
        args.footprint_dir.mkdir(parents=True, exist_ok=True)
        fp_gdf.to_file(footprint_path, driver="GeoJSON")
        print(f"Footprint saved to: {footprint_path}")

    footprint_geom = fp_gdf.geometry.iloc[0]
    footprint_crs = CRS(fp_gdf.crs.to_wkt())

    print(f"\nLoad this file in QGIS to verify the footprint before proceeding:")
    print(f"  {footprint_path.resolve()}\n")

    # ── Step 2: Preflight ─────────────────────────────────────────────────
    preflight = preflight_imagery(
        args.imagery_root,
        footprint_geom,
        footprint_crs,
    )
    preflight.print_report()

    if preflight.has_errors and not args.ignore_errors:
        print(
            "Preflight found files that need conversion or reprojection.\n"
            "Fix them first, then rerun.  Use --ignore-errors to clip "
            "only the ready files anyway."
        )
        sys.exit(1)

    if args.dry_run:
        print("Dry run complete. No files were clipped.")
        return

    if not preflight.valid:
        print("No valid tiles found to clip. Exiting.")
        sys.exit(0)

    # ── Step 3: Clip ──────────────────────────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Clipping {len(preflight.valid)} tiles → {args.output}")
    clip_all(
        preflight.valid,
        footprint_geom,
        footprint_crs,
        args.output,
        args.imagery_root,
        footprint_path=footprint_path,
        max_mask_gb=args.max_ram_gb,
    )


if __name__ == "__main__":
    main()