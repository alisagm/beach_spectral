#!/usr/bin/env python3
"""
tools/assign_band_config.py  —  Band configuration assignment for PAIS imagery

Replaces compute_band_cv.py + assign_band_labels.py with a single script that
applies a PAIS-specific decision tree instead of a generic CV-ranking heuristic.

Decision tree
-------------
  4-band image  →  format = RGBN,  NIR = B4          (positional rule, always)
  3-band image  →  compute CV ratio:  cv_b1 / max(cv_b2, cv_b3)
                     ratio > threshold  →  format = CIR,  NIR = B1
                     ratio ≤ threshold  →  format = RGB,  NIR = none

The CV ratio threshold defaults to 1.2 — empirically validated against the
confirmed PAIS ground truth.  The gap between the lowest confirmed-CIR ratio
(~1.35, year 2007) and the highest confirmed-RGB ratio (~1.06, year 2022)
gives comfortable margin at 1.2.

Primary output: --output-json
    JSON config keyed by year, directly consumable by the detection pipeline.
    {
        "1995": {"format": "CIR",  "nir_band": 1, "otsu_threshold": 0.302, "band_count": 3},
        "2010": {"format": "RGBN", "nir_band": 4, "otsu_threshold": 0.337, "band_count": 4},
        "2022": {"format": "RGB",  "nir_band": null, "otsu_threshold": 0.392, "band_count": 3}
    }

Optional output: --audit-csv
    Full per-year statistics table for QA, including raw CV values and
    sand_fraction so threshold choices can be re-evaluated.

Usage
-----
# minimal
python tools/assign_band_config.py \\
    --imagery-dir  PAIS_shorelines/ \\
    --output-json  OUTPUT/band_config.json

# with full audit table and diagnostic plot
python tools/assign_band_config.py \\
    --imagery-dir  PAIS_shorelines/ \\
    --output-json  OUTPUT/band_config.json \\
    --audit-csv    OUTPUT/band_config_audit.csv \\
    --plot-file    OUTPUT/band_config_cv.png

# adjust threshold (e.g. when extending to non-PAIS datasets)
python tools/assign_band_config.py \\
    --imagery-dir  PAIS_shorelines/ \\
    --output-json  OUTPUT/band_config.json \\
    --cv-ratio-threshold 1.5
"""

import argparse
import json
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

GDALBUILDVRT_EXE = Path(r"C:\Users\alisa\AppData\Local\Programs\OSGeo4W\bin\gdalbuildvrt.exe")

RASTER_EXTENSIONS = {".tif", ".tiff", ".jp2"}

DTYPE_MAX: dict[str, float] = {
    "uint8":  255.0,
    "uint16": 65535.0,
    "uint32": 4294967295.0,
    "int16":  32767.0,
}

DEFAULT_MAX_LOAD_GB     = 2.0
DEFAULT_CV_RATIO_THRESHOLD = 1.2   # empirically validated for PAIS; see module docstring


# ---------------------------------------------------------------------------
# Helpers  (read_band_masked and friends — carried forward from compute_band_cv)
# ---------------------------------------------------------------------------

def extract_year(filepath: Path) -> str | None:
    """Return the 4-digit year from the first year-like directory in the path."""
    for part in filepath.parts[-2::-1]:
        m = re.search(r"(19|20)\d{2}", part)
        if m:
            return m.group()
    return None


def normalize_band(data: np.ma.MaskedArray, dtype: str) -> np.ma.MaskedArray:
    """Scale pixel values to [0, 1] using known dtype maxima."""
    norm_factor = DTYPE_MAX.get(dtype)
    if norm_factor is None:
        valid = data.compressed()
        norm_factor = float(valid.max()) if len(valid) > 0 and valid.max() > 0 else 1.0
    return data / norm_factor


def compute_cv(valid_pixels: np.ndarray) -> float:
    """CV = std / mean.  Returns 0.0 when mean <= 0."""
    mean = valid_pixels.mean()
    return float(valid_pixels.std() / mean) if mean > 0 else 0.0


def read_band_masked(src, band_index: int, dtype: str, window=None) -> np.ma.MaskedArray:
    """
    Read one band and mask all NoData indicators:
      - metadata-registered nodata value
      - DN = 0   (black tile-edge fill)
      - DN = dtype_max  (white fill in some sensors)
    """
    raw = src.read(band_index, masked=True, window=window)
    if src.nodata is not None:
        raw = np.ma.masked_equal(raw, src.nodata)
    raw = np.ma.masked_equal(raw, 0)
    dtype_max = DTYPE_MAX.get(dtype)
    if dtype_max is not None:
        raw = np.ma.masked_equal(raw, int(dtype_max))
    else:
        raw = np.ma.masked_equal(raw, 0.0)
        raw = np.ma.masked_equal(raw, 1.0)
    return raw


def estimate_load_gb(src) -> float:
    item_bytes = np.dtype(src.dtypes[0]).itemsize
    return (src.count * src.height * src.width * item_bytes) / 1e9


def get_file_crs(filepath: Path) -> str | None:
    try:
        with rasterio.open(filepath) as src:
            return src.crs.to_wkt() if src.crs else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Welford parallel accumulator  (unchanged from compute_band_cv)
# ---------------------------------------------------------------------------

def _welford_combine(n1, mean1, M2_1, n2, mean2, M2_2):
    n = n1 + n2
    if n == 0:
        return 0, 0.0, 0.0
    delta = mean2 - mean1
    mean  = mean1 + delta * n2 / n
    M2    = M2_1 + M2_2 + delta**2 * n1 * n2 / n
    return n, mean, M2


def _welford_add_block(n, mean, M2, values):
    if len(values) == 0:
        return n, mean, M2
    n2    = len(values)
    mean2 = float(values.mean())
    M2_2  = float(values.var()) * n2
    return _welford_combine(n, mean, M2, n2, mean2, M2_2)


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------

def classify_format(
    band_count: int,
    cv_b1: float,
    cv_b2: float,
    cv_b3: float,
    threshold: float,
) -> tuple[str, int | None]:
    """
    Apply the PAIS-specific decision tree.

    Parameters
    ----------
    band_count  : number of bands in the raster (3 or 4)
    cv_b1..b3   : per-band coefficients of variation (normalised [0,1] pixels)
    threshold   : CV ratio threshold for CIR detection (default 1.2)

    Returns
    -------
    (format, nir_band)
        format    : "RGBN" | "CIR" | "RGB"
        nir_band  : 1-indexed band slot for NIR, or None for RGB
    """
    if band_count == 4:
        return "RGBN", 4

    # 3-band: B1 is NIR (CIR) if it has distinctly higher CV than B2 and B3.
    cv_ratio = cv_b1 / max(cv_b2, cv_b3) if max(cv_b2, cv_b3) > 0 else 0.0
    if cv_ratio > threshold:
        return "CIR", 1
    return "RGB", None


# ---------------------------------------------------------------------------
# Core computation  (windowed — used for all year mosaics)
# ---------------------------------------------------------------------------

def compute_year_stats_windowed(
    filepath: Path,
    cv_ratio_threshold: float,
) -> dict:
    """
    Stream through the raster in rasterio native blocks; accumulate per-band
    CV via Welford and a 256-bin NIR histogram for Otsu.

    The NIR band for the Otsu calculation is selected *after* the decision
    tree runs, so for RGBN imagery the histogram is always built from B4,
    not from whichever band happened to have the highest CV.
    """
    result: dict = {
        "year":            None,
        "band_count":      None,
        "dtype":           None,
        "format":          "ERROR",
        "nir_band":        None,
        "cv_b1":           np.nan,
        "cv_b2":           np.nan,
        "cv_b3":           np.nan,
        "cv_b4":           np.nan,
        "cv_ratio_3band":  np.nan,
        "otsu_threshold":  np.nan,
        "sand_pixels":     np.nan,
        "total_pixels":    np.nan,
        "sand_fraction":   np.nan,
        "included_tiles":  None,
        "skipped_tiles":   None,
        "error":           None,
    }

    try:
        with rasterio.open(filepath) as src:
            band_count  = src.count
            dtype       = src.dtypes[0]
            norm_factor = DTYPE_MAX.get(dtype)

            result["band_count"] = band_count
            result["dtype"]      = dtype

            accum = [(0, 0.0, 0.0)] * band_count   # Welford: (n, mean, M2)
            hists = [np.zeros(256, dtype=np.int64) for _ in range(band_count)]

            for _, win in src.block_windows(1):
                for b in range(band_count):
                    raw   = read_band_masked(src, b + 1, dtype, window=win)
                    valid = raw.compressed()
                    if len(valid) == 0:
                        continue
                    nf         = norm_factor if norm_factor is not None else 1.0
                    valid_norm = (valid / nf).astype(np.float32)
                    accum[b]   = _welford_add_block(*accum[b], valid_norm)
                    scaled     = (valid_norm * 255).clip(0, 255).astype(np.uint8)
                    hist_block, _ = np.histogram(scaled, bins=256, range=(0, 256))
                    hists[b]  += hist_block

        # --- CV from Welford ---
        cvs = []
        for b in range(band_count):
            n, mean, M2 = accum[b]
            cvs.append(float(np.sqrt(M2 / n) / mean) if n >= 2 and mean > 0 else 0.0)

        for i, cv in enumerate(cvs):
            result[f"cv_b{i + 1}"] = cv

        # --- decision tree ---
        fmt, nir_band = classify_format(
            band_count,
            cv_b1=cvs[0],
            cv_b2=cvs[1],
            cv_b3=cvs[2] if len(cvs) > 2 else 0.0,
            threshold=cv_ratio_threshold,
        )
        result["format"]   = fmt
        result["nir_band"] = nir_band

        # record the ratio that drove the 3-band decision (useful for audit)
        if band_count == 3:
            denom = max(cvs[1], cvs[2]) if len(cvs) > 2 else cvs[1]
            result["cv_ratio_3band"] = cvs[0] / denom if denom > 0 else np.inf

        # --- Otsu on the designated NIR band (or highest-CV band for RGB) ---
        if nir_band is not None:
            hist_idx = nir_band - 1          # convert 1-indexed → 0-indexed
        else:
            hist_idx = int(np.argmax(cvs))   # RGB: use highest-CV band as proxy

        hist_counts = hists[hist_idx]
        total_valid = int(hist_counts.sum())

        if total_valid > 0:
            hist_f      = hist_counts.astype(float) / total_valid
            bin_centers = np.linspace(0.0, 1.0, 256)
            otsu_thresh = float(threshold_otsu(hist=(hist_f, bin_centers)))
            cutoff_bin  = int(np.round(otsu_thresh * 255))
            sand_px     = int(hist_counts[cutoff_bin:].sum())

            result["otsu_threshold"] = otsu_thresh
            result["sand_pixels"]    = sand_px
            result["total_pixels"]   = total_valid
            result["sand_fraction"]  = sand_px / total_valid

    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# VRT build + year-level orchestration
# ---------------------------------------------------------------------------

def build_year_vrt(filepaths: list[Path], vrt_path: Path) -> tuple[bool, str]:
    cmd = [str(GDALBUILDVRT_EXE), str(vrt_path)] + [str(f) for f in filepaths]
    r   = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stderr.strip()


def compute_year_stats(
    year: str,
    filepaths: list[Path],
    cv_ratio_threshold: float,
) -> dict:
    """
    Build a per-year virtual mosaic and compute band configuration stats.

    Uses the majority-CRS tile set so that one off-CRS stray tile does not
    silently degrade the mosaic statistics.  Off-CRS tiles should be
    reprojected beforehand via tools/reproject_crs_mismatch.py.
    """
    # CRS census
    crs_by_file = {fp: get_file_crs(fp) for fp in filepaths}
    readable    = {fp: crs for fp, crs in crs_by_file.items() if crs is not None}

    if not readable:
        return {
            "year": year, "format": "ERROR",
            "error": "No readable tiles", "skipped_tiles": len(filepaths),
        }

    majority_crs = Counter(readable.values()).most_common(1)[0][0]
    included     = [fp for fp, crs in readable.items() if crs == majority_crs]
    skipped_crs  = [fp for fp, crs in readable.items() if crs != majority_crs]

    if skipped_crs:
        print(
            f"\n  WARN year {year}: skipping {len(skipped_crs)} tile(s) "
            f"with non-majority CRS — run tools/reproject_crs_mismatch.py first"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        vrt_path = Path(tmpdir) / f"{year}_mosaic.vrt"
        ok, stderr = build_year_vrt(included, vrt_path)
        if not ok:
            return {
                "year": year, "format": "ERROR",
                "error": f"gdalbuildvrt failed: {stderr}",
                "included_tiles": len(included), "skipped_tiles": len(skipped_crs),
            }
        stats = compute_year_stats_windowed(vrt_path, cv_ratio_threshold)

    stats["year"]           = year
    stats["included_tiles"] = len(included)
    stats["skipped_tiles"]  = len(skipped_crs) + (len(filepaths) - len(readable))
    return stats


# ---------------------------------------------------------------------------
# Diagnostic plot  (optional, simplified)
# ---------------------------------------------------------------------------

def plot_diagnostics(rows: list[dict], threshold: float, output_path: Path) -> None:
    """
    Single-panel scatter: cv_ratio_3band vs cv_b1 for 3-band years.
    4-band years are shown as a reference strip on the right.
    The threshold line makes it immediately visible how much margin exists.
    """
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))

    FORMAT_COLORS = {"CIR": "#1D9E75", "RGB": "#378ADD", "RGBN": "#D85A30", "ERROR": "#888780"}
    MARKERS       = {"CIR": "o",        "RGB": "s",        "RGBN": "^",       "ERROR": "x"}

    for fmt, grp in df.groupby("format"):
        if fmt == "RGBN":
            # 4-band: plot cv_b4 on x, label on y=0 (no ratio concept)
            ax.scatter(
                grp["cv_b4"], [0] * len(grp),
                label=f"RGBN (n={len(grp)})",
                color=FORMAT_COLORS.get(fmt, "#888780"),
                marker=MARKERS.get(fmt, "o"),
                s=60, alpha=0.8, zorder=3,
            )
        else:
            ax.scatter(
                grp["cv_b1"], grp["cv_ratio_3band"],
                label=f"{fmt} (n={len(grp)})",
                color=FORMAT_COLORS.get(fmt, "#888780"),
                marker=MARKERS.get(fmt, "o"),
                s=60, alpha=0.8, zorder=3,
            )

    ax.axhline(
        y=threshold, color="#E24B4A", linestyle="--", linewidth=1.2,
        label=f"CIR threshold ({threshold}×)",
    )

    # annotate each point with year
    for _, row in df.iterrows():
        if row["format"] != "RGBN":
            ax.annotate(
                str(row["year"]),
                xy=(row["cv_b1"], row["cv_ratio_3band"]),
                xytext=(3, 3), textcoords="offset points",
                fontsize=7, color="#555555",
            )

    ax.set_xlabel("CV — Band 1")
    ax.set_ylabel("CV ratio  (cv_b1 / max(cv_b2, cv_b3))")
    ax.set_title(
        f"Band 1 CV vs ratio — PAIS years\n"
        f"Threshold = {threshold}× | 4-band years shown at ratio = 0"
    )
    ax.legend(fontsize=9)
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
            "Assign band format (RGBN / CIR / RGB) and NIR band slot for each "
            "acquisition year in PAIS_shorelines/, using a PAIS-specific decision "
            "tree.  Outputs a JSON config file for the detection pipeline."
        )
    )
    parser.add_argument(
        "--imagery-dir", required=True, type=Path,
        help="Root directory containing raster files (searched recursively).",
    )
    parser.add_argument(
        "--output-json", required=True, type=Path,
        help="Path for the JSON band configuration file (primary output).",
    )
    parser.add_argument(
        "--audit-csv", type=Path, default=None,
        help="(Optional) Path for a full per-year statistics CSV for QA.",
    )
    parser.add_argument(
        "--plot-file", type=Path, default=None,
        help="(Optional) Path to save the CV diagnostic scatter plot.",
    )
    parser.add_argument(
        "--cv-ratio-threshold", type=float, default=DEFAULT_CV_RATIO_THRESHOLD,
        help=(
            f"CV ratio (cv_b1 / max(cv_b2, cv_b3)) above which a 3-band image "
            f"is classified as CIR.  Default: {DEFAULT_CV_RATIO_THRESHOLD}.  "
            "Validated for PAIS; adjust if extending to other datasets."
        ),
    )
    parser.add_argument(
        "--max-load-gb", type=float, default=DEFAULT_MAX_LOAD_GB,
        help=f"RAM threshold for windowed I/O (default: {DEFAULT_MAX_LOAD_GB} GB). "
             "Year mosaics almost always exceed this; the windowed path is the norm.",
    )
    args = parser.parse_args()

    if not GDALBUILDVRT_EXE.exists():
        print(
            f"ERROR: gdalbuildvrt not found at:\n  {GDALBUILDVRT_EXE}\n"
            "Update GDALBUILDVRT_EXE at the top of this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- discover & group files ---
    files = sorted(
        f for f in args.imagery_dir.rglob("*")
        if f.suffix.lower() in RASTER_EXTENSIONS
    )
    if not files:
        print(f"ERROR: no raster files found under {args.imagery_dir}", file=sys.stderr)
        sys.exit(1)

    year_groups: dict[str, list[Path]] = defaultdict(list)
    for fp in files:
        year_groups[extract_year(fp) or "UNKNOWN"].append(fp)

    print(f"Found {len(files)} files across {len(year_groups)} year(s): {sorted(year_groups)}")

    # --- process years ---
    rows = []
    for year in tqdm(sorted(year_groups), desc="Processing years"):
        row = compute_year_stats(year, year_groups[year], args.cv_ratio_threshold)
        rows.append(row)
        status = row.get("format", "ERROR")
        nir    = row.get("nir_band")
        ratio  = row.get("cv_ratio_3band", float("nan"))
        print(
            f"  {year}:  {status:<6}  nir_band={nir}  "
            + (f"cv_ratio={ratio:.3f}" if not np.isnan(ratio) else "")
        )

    # --- build JSON config ---
    config = {}
    for row in rows:
        year = str(row.get("year", "UNKNOWN"))
        config[year] = {
            "format":         row.get("format"),
            "nir_band":       row.get("nir_band"),
            "otsu_threshold": (
                round(float(row["otsu_threshold"]), 6)
                if row.get("otsu_threshold") is not None
                   and not (isinstance(row["otsu_threshold"], float) and np.isnan(row["otsu_threshold"]))
                else None
            ),
            "band_count":     row.get("band_count"),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as fh:
        json.dump(config, fh, indent=2)
    print(f"\nBand config saved → {args.output_json}")

    # --- console summary ---
    formats = [r.get("format", "ERROR") for r in rows]
    print("\n--- Format breakdown ---")
    for fmt, count in Counter(formats).most_common():
        print(f"  {fmt:<10} {count}")

    errors = [r for r in rows if r.get("error")]
    if errors:
        print(f"\n  WARN: {len(errors)} year(s) with errors:")
        for r in errors:
            print(f"    {r.get('year')}: {r.get('error')}")

    # --- optional audit CSV ---
    if args.audit_csv:
        df = pd.DataFrame(rows)
        col_order = [
            "year", "format", "nir_band", "band_count", "dtype",
            "cv_b1", "cv_b2", "cv_b3", "cv_b4", "cv_ratio_3band",
            "otsu_threshold", "sand_pixels", "total_pixels", "sand_fraction",
            "included_tiles", "skipped_tiles", "error",
        ]
        df = df[[c for c in col_order if c in df.columns]]
        args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.audit_csv, index=False)
        print(f"Audit CSV saved → {args.audit_csv}")

    # --- optional plot ---
    if args.plot_file:
        plot_diagnostics(rows, args.cv_ratio_threshold, args.plot_file)


if __name__ == "__main__":
    main()