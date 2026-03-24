#!/usr/bin/env python3
"""
tools/compute_band_cv.py  —  Script 1 of 2: Band CV computation

Computes per-band coefficient of variation (CV) for each acquisition year by
building a virtual mosaic (gdalbuildvrt) of all same-CRS tiles in the year and
computing stats on the mosaic.  Using the full-year mosaic avoids the problem
of small edge tiles that contain only water or only sand and therefore produce
misleading CV ratios.  One output row is emitted per year (not per tile).

Outputs
-------
--output-csv   Per-year summary table (one row per year).
               Columns: filepath, year, band_count, dtype,
                        cv_b1 … cv_b4, max_cv, second_cv, cv_ratio,
                        otsu_threshold, sand_pixels, total_pixels,
                        sand_fraction, proposed_format, confidence,
                        included_tiles, skipped_tiles, error
--plot-file    Two-panel diagnostic figure (optional):
               Left  — cv_ratio vs max_cv scatter for the full dataset,
                       coloured by proposed_format.
               Right — per-band CV bar chart for a chosen example year
                       (defaults to the year closest to the median cv_ratio).

Usage
-----
# minimal
python tools/compute_band_cv.py \\
    --imagery-dir  PAIS_shorelines/ \\
    --output-csv   OUTPUT/band_cv_summary.csv

# with diagnostics, examining a specific year
python tools/compute_band_cv.py \\
    --imagery-dir  PAIS_shorelines/ \\
    --output-csv   OUTPUT/band_cv_summary.csv \\
    --plot-file    OUTPUT/cv_diagnostics.png \\
    --example-year 2010

# tune the NIR detection threshold (default 1.8)
python tools/compute_band_cv.py \\
    --imagery-dir  PAIS_shorelines/ \\
    --output-csv   OUTPUT/band_cv_summary.csv \\
    --cv-ratio-threshold 2.0

# override the in-memory load threshold (default 2.0 GB)
python tools/compute_band_cv.py \\
    --imagery-dir  PAIS_shorelines/ \\
    --output-csv   OUTPUT/band_cv_summary.csv \\
    --max-load-gb  4.0
"""

import argparse
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from skimage.filters import threshold_otsu
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Full path to gdalbuildvrt.exe (OSGeo4W install).
# subprocess uses this directly so gdalbuildvrt does not need to be on PATH.
GDALBUILDVRT_EXE = Path(r"C:\Users\alisa\AppData\Local\Programs\OSGeo4W\bin\gdalbuildvrt.exe")

RASTER_EXTENSIONS = {".tif", ".tiff", ".jp2"}

# Maximum valid value for normalisation to [0, 1].
# float32/float64 rasters are assumed already normalised; if not, the
# fallback in normalize_band() uses the actual per-file maximum.
DTYPE_MAX: dict[str, float] = {
    "uint8":  255.0,
    "uint16": 65535.0,
    "uint32": 4294967295.0,
    "int16":  32767.0,
}

# Files whose estimated in-memory load exceeds this threshold are routed to
# compute_file_stats_windowed(), which streams data in rasterio blocks and
# uses Welford's parallel algorithm so peak RAM is proportional to one block
# rather than the whole raster.  Adjust via --max-load-gb.
DEFAULT_MAX_LOAD_GB = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_year(filepath: Path) -> str | None:
    """
    Return the 4-digit year from the first four characters of the file's
    immediate parent folder name.

    Folder structure is expected to be:
        …/{year}/{YYYYMMDD}/{filename}
    so the immediate parent is always a date folder whose first four digits
    are the acquisition year.

    Falls back to searching ancestor directory names (excluding the filename)
    if the immediate parent does not start with a recognisable year, so that
    files stored directly in a plain year folder (e.g. ``2010/tile.tif``)
    are still handled correctly.
    """
    for part in filepath.parts[-2::-1]:   # parent, grandparent, … (skip filename)
        m = re.search(r"(19|20)\d{2}", part)
        if m:
            return m.group()
    return None


def normalize_band(data: np.ma.MaskedArray, dtype: str) -> np.ma.MaskedArray:
    """
    Scale pixel values to [0, 1].

    Uses the known dtype maximum for integer rasters.  For float rasters
    (or unknown dtypes) falls back to the actual maximum of valid pixels so
    that downstream CV values remain comparable.
    """
    norm_factor = DTYPE_MAX.get(dtype)
    if norm_factor is None:
        valid = data.compressed()
        norm_factor = float(valid.max()) if len(valid) > 0 and valid.max() > 0 else 1.0
    return data / norm_factor


def compute_cv(valid_pixels: np.ndarray) -> float:
    """CV = std / mean.  Returns 0.0 when mean <= 0 (avoids division by zero)."""
    mean = valid_pixels.mean()
    return float(valid_pixels.std() / mean) if mean > 0 else 0.0


def read_band_masked(
    src, band_index: int, dtype: str, window=None
) -> np.ma.MaskedArray:
    """
    Read one raster band and mask ALL NoData indicators:
      1. The metadata-registered nodata value (if any).
      2. Pure-black pixels (DN = 0)          — conventional tile-edge fill.
      3. Pure-white pixels (DN = dtype max)  — alternate fill in some sensors.

    For float rasters without a known dtype max, exact 0.0 and 1.0 are masked
    as reasonable sentinels.  Note: this could clip genuinely bright pixels in
    rare cases, but is preferable to white NoData borders being classified as
    sand.

    The ``window`` parameter is forwarded to rasterio.DatasetReader.read() so
    the same function works in both full-load and block I/O contexts.
    """
    raw = src.read(band_index, masked=True, window=window)

    if src.nodata is not None:
        raw = np.ma.masked_equal(raw, src.nodata)

    # Mask implicit sentinel values regardless of metadata registration.
    raw = np.ma.masked_equal(raw, 0)
    dtype_max = DTYPE_MAX.get(dtype)
    if dtype_max is not None:
        raw = np.ma.masked_equal(raw, int(dtype_max))
    else:
        # Float rasters: treat exact 0.0 and 1.0 as likely fill values.
        raw = np.ma.masked_equal(raw, 0.0)
        raw = np.ma.masked_equal(raw, 1.0)

    return raw


# ---------------------------------------------------------------------------
# Empty result template
# ---------------------------------------------------------------------------

def _make_empty_result(filepath: Path) -> dict:
    """
    Return a result dict pre-populated with filepath, year, and NaN/None
    defaults.  Shared by both the standard and windowed compute paths so
    the output schema is always identical regardless of which path ran.
    """
    return {
        "filepath":        str(filepath),
        "year":            extract_year(filepath),
        "band_count":      None,
        "dtype":           None,
        "cv_b1":           np.nan,
        "cv_b2":           np.nan,
        "cv_b3":           np.nan,
        "cv_b4":           np.nan,
        "max_cv":          np.nan,
        "second_cv":       np.nan,
        "cv_ratio":        np.nan,
        "otsu_threshold":  np.nan,
        "sand_pixels":     np.nan,
        "total_pixels":    np.nan,
        "sand_fraction":   np.nan,
        "proposed_format": "ERROR",
        "confidence":      "LOW",
        "included_tiles":  None,
        "skipped_tiles":   None,
        "error":           None,
    }


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

def estimate_load_gb(src) -> float:
    """
    Estimate the RAM required to load all bands as their native dtype.

    Opens only the file header — no pixel data is read.  Actual peak usage
    may be slightly higher due to Python object overhead, but this estimate
    is conservative enough to gate the windowed path reliably.
    """
    item_bytes = np.dtype(src.dtypes[0]).itemsize
    return (src.count * src.height * src.width * item_bytes) / 1e9


# ---------------------------------------------------------------------------
# Welford parallel accumulator  (windowed CV computation)
# ---------------------------------------------------------------------------

def _welford_combine(
    n1: int, mean1: float, M2_1: float,
    n2: int, mean2: float, M2_2: float,
) -> tuple[int, float, float]:
    """
    Combine two independent Welford accumulators in parallel.

    M2 is the sum of squared deviations from the running mean, so
    variance = M2 / n and std = sqrt(M2 / n).  The combination formula is
    exact (no approximation) and numerically stable for large n.
    """
    n = n1 + n2
    if n == 0:
        return 0, 0.0, 0.0
    delta = mean2 - mean1
    mean  = mean1 + delta * n2 / n
    M2    = M2_1 + M2_2 + delta**2 * n1 * n2 / n
    return n, mean, M2


def _welford_add_block(
    n: int, mean: float, M2: float, values: np.ndarray
) -> tuple[int, float, float]:
    """Add a 1-D array of new values into an existing Welford accumulator."""
    if len(values) == 0:
        return n, mean, M2
    n2    = len(values)
    mean2 = float(values.mean())
    M2_2  = float(values.var()) * n2   # population variance × n = sum of sq. dev.
    return _welford_combine(n, mean, M2, n2, mean2, M2_2)


# ---------------------------------------------------------------------------
# Windowed CV computation  (large-file path)
# ---------------------------------------------------------------------------

def compute_file_stats_windowed(
    filepath: Path, cv_ratio_threshold: float
) -> dict:
    """
    Block-wise variant of compute_file_stats for rasters too large to load
    in a single allocation.

    Algorithm
    ---------
    Per-band CV:
        Welford's parallel algorithm over successive rasterio native blocks.
        Peak RAM ≈ one block × bands × dtype bytes instead of the full raster.

    Otsu threshold:
        A 256-bin histogram is accumulated across blocks; threshold_otsu is
        called once on the final normalised histogram (skimage >= 0.19 API).

    For float rasters with no known dtype max, values are normalised by 1.0
    (i.e. treated as already in [0, 1]).  CV is scale-invariant, so band
    ranking is unaffected; Otsu accuracy may degrade if values fall outside
    [0, 1], but float rasters are rare in this dataset.

    The proposed_format decision logic is identical to the standard path.
    """
    result = _make_empty_result(filepath)

    try:
        with rasterio.open(filepath) as src:
            band_count  = src.count
            dtype       = src.dtypes[0]
            norm_factor = DTYPE_MAX.get(dtype)   # None for float rasters
            result["band_count"] = band_count
            result["dtype"]      = dtype

            # Per-band Welford accumulators: (n, mean, M2)
            accum = [(0, 0.0, 0.0)] * band_count
            # 256-bin histograms in normalised [0, 1] space (for Otsu)
            hists = [np.zeros(256, dtype=np.int64) for _ in range(band_count)]

            for _, win in src.block_windows(1):
                for b in range(band_count):
                    raw   = read_band_masked(src, b + 1, dtype, window=win)
                    valid = raw.compressed()
                    if len(valid) == 0:
                        continue

                    nf = norm_factor if norm_factor is not None else 1.0
                    valid_norm = (valid / nf).astype(np.float32)

                    accum[b] = _welford_add_block(*accum[b], valid_norm)

                    scaled = (valid_norm * 255).clip(0, 255).astype(np.uint8)
                    hist_block, _ = np.histogram(scaled, bins=256, range=(0, 256))
                    hists[b] += hist_block

        # --- derive CV from Welford accumulators ---
        cvs = []
        for b in range(band_count):
            n, mean, M2 = accum[b]
            if n < 2 or mean <= 0:
                cvs.append(0.0)
            else:
                cvs.append(float(np.sqrt(M2 / n) / mean))

        for i, cv in enumerate(cvs):
            result[f"cv_b{i + 1}"] = cv

        sorted_idx = np.argsort(cvs)[::-1]
        max_cv    = cvs[sorted_idx[0]]
        second_cv = cvs[sorted_idx[1]] if len(cvs) > 1 else 0.0
        ratio     = max_cv / second_cv if second_cv > 0 else np.inf

        result["max_cv"]    = max_cv
        result["second_cv"] = second_cv
        result["cv_ratio"]  = ratio

        nir_detected = ratio > cv_ratio_threshold
        if band_count == 4:
            result["proposed_format"] = "4BAND" if nir_detected else "4BAND_WEAK_NIR"
        else:
            result["proposed_format"] = "CIR" if nir_detected else "RGB"

        # --- Otsu from histogram of anchor band ---
        anchor_idx  = int(sorted_idx[0])
        hist_counts = hists[anchor_idx]
        total_valid = int(hist_counts.sum())

        if total_valid > 0:
            hist_f      = hist_counts.astype(float) / total_valid
            bin_centers = np.linspace(0.0, 1.0, 256)
            otsu_thresh = float(threshold_otsu(hist=(hist_f, bin_centers)))
            result["otsu_threshold"] = otsu_thresh

            # Sand fraction: valid pixels above Otsu threshold.
            # total_valid already excludes masked NoData cells.
            cutoff_bin = int(np.round(otsu_thresh * 255))
            sand_px    = int(hist_counts[cutoff_bin:].sum())
            sand_frac  = sand_px / total_valid
            result["sand_pixels"]   = sand_px
            result["total_pixels"]  = total_valid
            result["sand_fraction"] = sand_frac
            result["confidence"]    = "LOW" if sand_frac < 0.05 else "HIGH"

    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Per-file statistics  (standard path)
# ---------------------------------------------------------------------------

def compute_file_stats(
    filepath: Path, cv_ratio_threshold: float, max_load_gb: float
) -> dict:
    """
    Open one raster file, compute per-band CVs, run Otsu on the anchor band,
    and return a dictionary ready for a DataFrame row.

    Files whose estimated in-memory footprint exceeds max_load_gb are routed
    to compute_file_stats_windowed() which streams data in rasterio blocks.

    The 'proposed_format' field is the format decision point that links
    this script to script 2:
        'CIR'             — 3-band, NIR detected (NIR-R-G)
        'RGB'             — 3-band, no NIR signal
        '4BAND'           — 4-band, NIR detected
        '4BAND_WEAK_NIR'  — 4-band, CV ratio below threshold (flag for review)
        'ERROR'           — could not read file
    """
    result = _make_empty_result(filepath)

    try:
        with rasterio.open(filepath) as src:
            band_count = src.count
            dtype      = src.dtypes[0]
            load_gb    = estimate_load_gb(src)
            # --- memory guard: delegate large files to the windowed path ---
            if load_gb > max_load_gb:
                print(
                    f"\n  INFO {filepath.name}: estimated load {load_gb:.1f} GB "
                    f"> {max_load_gb:.1f} GB threshold — using windowed I/O"
                )
                return compute_file_stats_windowed(filepath, cv_ratio_threshold)

            # Standard path only — windowed path populates these itself.
            result["band_count"] = band_count
            result["dtype"]      = dtype

            # --- standard path: load all bands at once ---
            bands_flat: list[np.ndarray] = []
            for i in range(1, band_count + 1):
                raw        = read_band_masked(src, i, dtype)
                normalised = normalize_band(raw, dtype)
                bands_flat.append(normalised.compressed())  # 1-D valid pixels only

        # --- CV per band ---
        cvs = [compute_cv(b) for b in bands_flat]
        for i, cv in enumerate(cvs):
            result[f"cv_b{i + 1}"] = cv

        sorted_idx = np.argsort(cvs)[::-1]          # descending CV order
        max_cv    = cvs[sorted_idx[0]]
        second_cv = cvs[sorted_idx[1]] if len(cvs) > 1 else 0.0
        ratio     = max_cv / second_cv if second_cv > 0 else np.inf

        result["max_cv"]    = max_cv
        result["second_cv"] = second_cv
        result["cv_ratio"]  = ratio

        # --- format decision ---
        nir_detected = ratio > cv_ratio_threshold
        if band_count == 4:
            proposed = "4BAND" if nir_detected else "4BAND_WEAK_NIR"
        else:
            proposed = "CIR" if nir_detected else "RGB"
        result["proposed_format"] = proposed

        # --- Otsu on anchor band ---
        anchor_pixels = bands_flat[sorted_idx[0]]
        try:
            otsu_thresh = float(threshold_otsu(anchor_pixels))
        except Exception as oe:
            otsu_thresh = np.nan
            result["error"] = f"Otsu failed: {oe}"

        result["otsu_threshold"] = otsu_thresh

        if not np.isnan(otsu_thresh):
            sand_px   = int((anchor_pixels > otsu_thresh).sum())
            # anchor_pixels is already .compressed() — valid pixels only.
            # Using len() here (not mask_2d.size) avoids counting NoData cells
            # in total_pixels and keeps sand_fraction accurate.
            total_px  = len(anchor_pixels)
            sand_frac = sand_px / total_px if total_px > 0 else 0.0
            result["sand_pixels"]   = sand_px
            result["total_pixels"]  = total_px
            result["sand_fraction"] = sand_frac
            result["confidence"]    = "LOW" if sand_frac < 0.05 else "HIGH"

    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Year-level mosaic computation
# ---------------------------------------------------------------------------

def get_file_crs(filepath: Path) -> str | None:
    """
    Return the CRS of a raster as a normalised WKT string, or None on failure.

    Using to_wkt() rather than to_epsg() avoids None returns for CRS that are
    valid but not registered in the EPSG database (e.g. custom projections or
    non-standard authority codes), which would silently drop files from the
    majority-CRS calculation.
    """
    try:
        with rasterio.open(filepath) as src:
            return src.crs.to_wkt() if src.crs else None
    except Exception:
        return None


def build_year_vrt(filepaths: list[Path], vrt_path: Path) -> tuple[bool, str]:
    """
    Call gdalbuildvrt to create a virtual mosaic from filepaths.

    gdalbuildvrt is used rather than rasterio.merge so that no pixel data is
    read at this stage — the VRT is just a descriptor file.  Pixel I/O happens
    later in the windowed compute path.

    Returns (success: bool, stderr_message: str).
    """
    cmd = [str(GDALBUILDVRT_EXE), str(vrt_path)] + [str(f) for f in filepaths]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stderr.strip()


def compute_year_stats(
    year: str,
    filepaths: list[Path],
    cv_ratio_threshold: float,
    max_load_gb: float,
) -> dict:
    """
    Build a virtual mosaic for one year and compute CV / Otsu statistics.

    Algorithm
    ---------
    1. Read the CRS of every tile; tiles that cannot be opened are counted as
       skipped (not as errors, since individual tile failures are expected for
       edge segments).
    2. Identify the majority CRS.  Tiles on a different CRS are skipped with a
       warning — reprojecting on the fly before mosaicking would introduce
       resampling artefacts and is best handled as a separate preflight step.
    3. Build a .vrt in a temporary directory via gdalbuildvrt.  The VRT is a
       lightweight XML descriptor; no pixel data is read here.
    4. Run compute_file_stats() on the .vrt.  Because the mosaic spans the full
       year footprint its estimated size almost always exceeds max_load_gb, so
       the windowed streaming path is used automatically.
    5. Override the year and filepath fields (the VRT lives in a temp dir and
       yields no useful path information) and attach tile-count metadata.

    Returns
    -------
    A result dict with the same schema as compute_file_stats(), plus:
        included_tiles  — number of tiles that entered the mosaic
        skipped_tiles   — tiles dropped (CRS mismatch or unreadable)
    """
    result = _make_empty_result(Path(f"<mosaic:{year}>"))
    result["year"]    = year
    result["filepath"] = f"<mosaic:{year}>"

    # --- CRS census ---
    crs_by_file: dict[Path, str | None] = {fp: get_file_crs(fp) for fp in filepaths}
    readable   = {fp: crs for fp, crs in crs_by_file.items() if crs is not None}
    unreadable = len(filepaths) - len(readable)

    if not readable:
        result["error"]         = "No readable tiles in year group"
        result["skipped_tiles"] = len(filepaths)
        return result

    majority_crs = Counter(readable.values()).most_common(1)[0][0]
    included     = [fp for fp, crs in readable.items() if crs == majority_crs]
    skipped_crs  = [fp for fp, crs in readable.items() if crs != majority_crs]

    if skipped_crs:
        print(
            f"\n  WARN year {year}: skipping {len(skipped_crs)} tile(s) with "
            f"non-majority CRS — {[p.name for p in skipped_crs]}"
        )

    result["included_tiles"] = len(included)
    result["skipped_tiles"]  = len(skipped_crs) + unreadable

    # --- build VRT and compute stats ---
    with tempfile.TemporaryDirectory() as tmpdir:
        vrt_path = Path(tmpdir) / f"{year}_mosaic.vrt"
        ok, stderr = build_year_vrt(included, vrt_path)
        if not ok:
            result["error"] = f"gdalbuildvrt failed: {stderr}"
            return result

        stats = compute_file_stats(vrt_path, cv_ratio_threshold, max_load_gb)

    # Override fields that are meaningless for a temp-dir VRT.
    stats["year"]           = year
    stats["filepath"]       = f"<mosaic:{year}>"
    stats["included_tiles"] = result["included_tiles"]
    stats["skipped_tiles"]  = result["skipped_tiles"]
    return stats


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------

def plot_diagnostics(
    df: pd.DataFrame,
    example_year: str | None,
    cv_ratio_threshold: float,
    output_path: Path,
) -> None:
    """
    Two-panel figure saved to output_path.

    Left panel  — cv_ratio vs max_cv for all years (scatter), with the
                  cv_ratio_threshold drawn as a horizontal reference line.
                  Points coloured by proposed_format.

    Right panel — per-band CV bar chart for the example year (defaults to
                  the year closest to the median cv_ratio across the dataset).
    """
    FORMAT_COLORS = {
        "CIR":            "#1D9E75",
        "RGB":            "#378ADD",
        "4BAND":          "#D85A30",
        "4BAND_WEAK_NIR": "#BA7517",
        "ERROR":          "#888780",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- left: dataset scatter ---
    ax = axes[0]
    for fmt, group in df.groupby("proposed_format"):
        color = FORMAT_COLORS.get(fmt, "#888780")
        ax.scatter(
            group["max_cv"],
            group["cv_ratio"],
            label=fmt,
            color=color,
            alpha=0.7,
            edgecolors="none",
            s=30,
        )
    ax.axhline(
        y=cv_ratio_threshold,
        color="#E24B4A",
        linestyle="--",
        linewidth=1,
        label=f"threshold ({cv_ratio_threshold}×)",
    )
    ax.set_xlabel("Max CV (anchor band)")
    ax.set_ylabel("CV ratio (max / second)")
    ax.set_title("CV ratio — full dataset")
    ax.legend(fontsize=9)

    # --- right: per-band bar chart for example file ---
    ax2 = axes[1]

    if example_year is not None:
        row = df[df["year"].astype(str) == str(example_year)]
    else:
        # pick the file closest to the median cv_ratio
        median_ratio = df["cv_ratio"].median()
        row = df.iloc[(df["cv_ratio"] - median_ratio).abs().argsort()[:1]]

    if not row.empty:
        cv_cols = [c for c in ["cv_b1", "cv_b2", "cv_b3", "cv_b4"]
                   if not pd.isna(row[c].values[0])]
        cv_vals    = [row[c].values[0] for c in cv_cols]
        bar_labels = [f"Band {i + 1}" for i in range(len(cv_cols))]
        bar_colors = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD"][: len(cv_cols)]

        ax2.bar(bar_labels, cv_vals, color=bar_colors, edgecolor="none")
        ax2.set_ylabel("Coefficient of variation")

        fname  = f"Year {row['year'].values[0]}"
        fmt    = row["proposed_format"].values[0]
        ratio  = row["cv_ratio"].values[0]
        ax2.set_title(f"{fname}\nProposed: {fmt}  |  ratio: {ratio:.2f}")

        min_nir_cv = row["second_cv"].values[0] * cv_ratio_threshold
        ax2.axhline(
            y=min_nir_cv,
            color="#E24B4A",
            linestyle="--",
            linewidth=1,
            label="min CV for NIR at threshold",
        )
        ax2.legend(fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Diagnostic plot saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-band CV statistics for all raster files in an imagery "
            "directory, grouped by year.  A virtual mosaic (gdalbuildvrt) is built "
            "per year so that edge tiles containing only water or only sand do not "
            "distort the CV ratios used for NIR detection."
        )
    )
    parser.add_argument(
        "--imagery-dir", required=True, type=Path,
        help="Root directory containing raster files (searched recursively).",
    )
    parser.add_argument(
        "--output-csv", required=True, type=Path,
        help="Path for the per-year summary CSV.",
    )
    parser.add_argument(
        "--plot-file", type=Path, default=None,
        help="(Optional) Path to save the two-panel diagnostic figure.",
    )
    parser.add_argument(
        "--example-year", type=str, default=None,
        help="(Optional) Year to feature in the per-band CV bar chart (e.g. 2010). "
             "Defaults to the year closest to the dataset median cv_ratio.",
    )
    parser.add_argument(
        "--cv-ratio-threshold", type=float, default=1.8,
        help="CV ratio (max_cv / second_cv) above which a band is classified "
             "as NIR.  Default: 1.8.  Tune this using the scatter plot output.",
    )
    parser.add_argument(
        "--max-load-gb", type=float, default=DEFAULT_MAX_LOAD_GB,
        help=(
            f"RAM threshold in GB for loading all bands at once "
            f"(default: {DEFAULT_MAX_LOAD_GB}).  Files whose estimated "
            "footprint exceeds this value are processed with block I/O "
            "(Welford's algorithm) so peak RAM stays roughly constant."
        ),
    )
    args = parser.parse_args()

    # --- preflight: gdalbuildvrt must exist at the configured path ---
    if not GDALBUILDVRT_EXE.exists():
        print(
            f"ERROR: gdalbuildvrt not found at configured path:\n  {GDALBUILDVRT_EXE}\n"
            "Update the GDALBUILDVRT_EXE constant at the top of this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- discover files ---
    files = sorted(
        f for f in args.imagery_dir.rglob("*")
        if f.suffix.lower() in RASTER_EXTENSIONS
    )
    if not files:
        print(f"ERROR: no raster files found under {args.imagery_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} raster files.")

    # --- group by year ---
    year_groups: dict[str, list[Path]] = defaultdict(list)
    for fp in files:
        yr = extract_year(fp) or "UNKNOWN"
        year_groups[yr].append(fp)

    print(f"Grouped into {len(year_groups)} year(s): {sorted(year_groups)}")
    if "UNKNOWN" in year_groups:
        print(
            f"  WARN: {len(year_groups['UNKNOWN'])} file(s) could not be assigned "
            "a year from their folder names and will be processed as group 'UNKNOWN'.",
            file=sys.stderr,
        )

    # --- process one mosaic per year ---
    rows = []
    for year in tqdm(sorted(year_groups), desc="Processing years"):
        rows.append(
            compute_year_stats(
                year,
                year_groups[year],
                args.cv_ratio_threshold,
                args.max_load_gb,
            )
        )

    df = pd.DataFrame(rows)
    df.sort_values("year", inplace=True, ignore_index=True)

    # --- save ---
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Summary table saved → {args.output_csv}")

    # --- console summary ---
    print("\n--- Proposed format breakdown ---")
    print(df["proposed_format"].value_counts().to_string())
    print(f"\nLow-confidence years (sand_fraction < 5%): {(df['confidence'] == 'LOW').sum()}")
    print(f"Years with errors:                          {df['error'].notna().sum()}")
    print(f"Total tiles skipped (CRS mismatch):         {df['skipped_tiles'].sum()}")
    print("\nCV ratio statistics (all years):")
    print(df["cv_ratio"].describe().round(3).to_string())

    # --- plot ---
    if args.plot_file:
        plot_diagnostics(df, args.example_year, args.cv_ratio_threshold, args.plot_file)


if __name__ == "__main__":
    main()