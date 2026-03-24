#!/usr/bin/env python3
"""
tools/assign_band_labels.py  —  Script 2 of 2: Band label assignment

For each raster file recorded in the CV summary CSV (output of script 1),
this script:
  1. Loads the raster and re-identifies the anchor band by CV.
  2. Applies the Otsu threshold (from script 1, or recomputed if needed)
     to build a binary sand mask.
  3. Ranks remaining bands by their mean over sand-mask pixels (seam-immune).
  4. Assigns a label (NIR / Red / Green / Blue) to every band slot.
  5. Saves a diagnostic mask image cropped to the water/sand boundary.

The rank-order physics, invariant across all sensor generations:
  Over dry sand (ascending):  Blue < Green < Red < NIR
  Over CIR imagery (2 bands): Green < Red        (NIR already identified)

Outputs
-------
--output-csv   Band label assignment table (primary output).
               Columns: filepath, year, proposed_format, year_nir_detected,
                        band_1_label … band_4_label,
                        otsu_threshold, sand_pixels, total_pixels,
                        sand_fraction, error
--mask-dir     One PNG per file showing anchor band + binary mask,
               cropped to the water/sand boundary zone.

Usage
-----
# standard run
python tools/assign_band_labels.py \\
    --cv-csv    OUTPUT/band_cv_summary.csv \\
    --output-csv OUTPUT/band_labels.csv \\
    --mask-dir   OUTPUT/masks/

# apply manual format corrections before ranking
python tools/assign_band_labels.py \\
    --cv-csv     OUTPUT/band_cv_summary.csv \\
    --output-csv OUTPUT/band_labels.csv \\
    --mask-dir   OUTPUT/masks/ \\
    --corrections tools/format_corrections.csv

# skip mask images (faster, labels only)
python tools/assign_band_labels.py \\
    --cv-csv     OUTPUT/band_cv_summary.csv \\
    --output-csv OUTPUT/band_labels.csv \\
    --mask-dir   OUTPUT/masks/ \\
    --skip-masks

# process one year at a time for debugging
python tools/assign_band_labels.py \\
    --cv-csv     OUTPUT/band_cv_summary.csv \\
    --output-csv OUTPUT/band_labels_2010.csv \\
    --mask-dir   OUTPUT/masks/2010/ \\
    --year       2010

Corrections CSV format
----------------------
Two columns, no header required beyond:
    filepath,corrected_format
    /path/to/file.tif,CIR
    /path/to/other.tif,RGB
"""

import argparse
import re
import sys
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

DTYPE_MAX: dict[str, float] = {
    "uint8":  255.0,
    "uint16": 65535.0,
    "uint32": 4294967295.0,
    "int16":  32767.0,
}

# Band label rank order over sand (ascending mean): physics-invariant.
# Index 0 = band with the LOWEST mean over the sand mask.
RANK_LABELS: dict[str, list[str]] = {
    "4BAND":          ["Blue", "Green", "Red"],   # NIR already identified; 3 remain
    "4BAND_WEAK_NIR": ["Blue", "Green", "Red"],   # treat as 4BAND; flag in output
    "CIR":            ["Green", "Red"],           # NIR identified; no Blue present
    "RGB":            ["Blue", "Green"],          # Red is anchor; 2 remain
}

# Minimum pixel fraction to trust the sand mask for ranking
MIN_SAND_FRACTION = 0.02

# Crop half-width (pixels, post-downsampling) around the boundary for diagnostics
BOUNDARY_CROP_HALF = 150


# ---------------------------------------------------------------------------
# Helpers (shared with script 1 — could be factored into utils if desired)
# ---------------------------------------------------------------------------

def extract_year(filepath: Path) -> str | None:
    for part in reversed(filepath.parts):
        m = re.search(r"(19|20)\d{2}", part)
        if m:
            return m.group()
    return None


def normalize_band(data: np.ma.MaskedArray, dtype: str) -> np.ma.MaskedArray:
    norm_factor = DTYPE_MAX.get(dtype)
    if norm_factor is None:
        valid = data.compressed()
        norm_factor = float(valid.max()) if len(valid) > 0 and valid.max() > 0 else 1.0
    return data / norm_factor


def compute_cv(valid_pixels: np.ndarray) -> float:
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
# Sand-mask boundary crop (seam-resistant diagnostic region)
# ---------------------------------------------------------------------------

def find_boundary_axis_and_position(
    mask_2d: np.ndarray,
    valid_2d: np.ndarray,
) -> tuple[str, int]:
    """
    Determine whether the water/sand boundary runs approximately horizontally
    (N-S in image space) or vertically (E-W), and return the index of the
    row or column where the transition is most concentrated.

    Strategy: for each axis, compute per-row (or per-column) sand fractions
    *over valid pixels only*.  Fractions are computed as (sand pixels) /
    (valid pixels) per line, so the NoData void does not contribute — without
    this, the irregular void edge creates steep artificial gradients that
    dominate over the true water/sand transition.

    The axis with the steeper maximum gradient is the one perpendicular to
    the shoreline — that is the axis we should crop along.

    Parameters
    ----------
    mask_2d  : 2-D uint8 binary sand mask  (1 = sand, 0 = other)
    valid_2d : 2-D bool array              (True = valid pixel, not NoData)

    Returns ('col', boundary_col) or ('row', boundary_row).
    """
    # Avoid division by zero in lines that are entirely NoData.
    valid_col_counts = valid_2d.sum(axis=0).clip(min=1)   # shape (W,)
    valid_row_counts = valid_2d.sum(axis=1).clip(min=1)   # shape (H,)

    # Sand fraction per line, computed over valid pixels only.
    col_fracs = (mask_2d * valid_2d).sum(axis=0) / valid_col_counts
    row_fracs = (mask_2d * valid_2d).sum(axis=1) / valid_row_counts

    col_gradient = np.abs(np.diff(col_fracs)).max() if len(col_fracs) > 1 else 0.0
    row_gradient = np.abs(np.diff(row_fracs)).max() if len(row_fracs) > 1 else 0.0

    if col_gradient >= row_gradient:
        # boundary is roughly vertical — crop horizontally around boundary column
        boundary = int(np.argmin(np.abs(col_fracs - 0.5)))
        return "col", boundary
    else:
        # boundary is roughly horizontal — crop vertically around boundary row
        boundary = int(np.argmin(np.abs(row_fracs - 0.5)))
        return "row", boundary


def crop_to_boundary(
    anchor_2d: np.ndarray,
    mask_2d: np.ndarray,
    valid_2d: np.ndarray,
    half_width: int = BOUNDARY_CROP_HALF,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Return (anchor_crop, mask_crop, description) — a strip centered on the
    water/sand boundary, which is the region most likely to expose mask errors.

    Parameters
    ----------
    anchor_2d : 2-D float array for the anchor band
    mask_2d   : 2-D uint8 binary sand mask (1 = sand, 0 = other)
    valid_2d  : 2-D bool validity mask (True = not NoData), forwarded to
                find_boundary_axis_and_position so the NoData void does not
                pull the detected boundary away from the true shoreline.
    """
    h, w = anchor_2d.shape
    axis, pos = find_boundary_axis_and_position(mask_2d, valid_2d)

    if axis == "col":
        lo = max(0, pos - half_width)
        hi = min(w, pos + half_width)
        return anchor_2d[:, lo:hi], mask_2d[:, lo:hi], f"column {pos}"
    else:
        lo = max(0, pos - half_width)
        hi = min(h, pos + half_width)
        return anchor_2d[lo:hi, :], mask_2d[lo:hi, :], f"row {pos}"


def adaptive_downsample_factor(h: int, w: int, target_max: int = 1500) -> int:
    """
    Choose a downsampling factor so that neither dimension exceeds target_max.
    Returns 1 (no downsampling) for small images.
    """
    return max(1, int(np.ceil(max(h, w) / target_max)))


# ---------------------------------------------------------------------------
# Mask diagnostic image
# ---------------------------------------------------------------------------

def save_mask_diagnostic(
    filepath: Path,
    anchor_2d: np.ndarray,
    mask_2d: np.ndarray,
    valid_2d: np.ndarray,
    otsu_thresh: float,
    band_labels: dict[int, str],
    mask_dir: Path,
) -> Path:
    """
    Save a side-by-side PNG:
      Left  — anchor band (greyscale), cropped to the water/sand boundary.
              The Otsu threshold is shown in the colorbar for reference.
      Right — binary sand mask (green = sand / above threshold).

    The crop centres on the boundary zone — the region where mask errors
    (e.g. wave crests mistaken for sand) are most likely to appear.

    Parameters
    ----------
    valid_2d : 2-D bool validity mask for the anchor band, used by
               crop_to_boundary to exclude the NoData void when locating
               the true water/sand boundary.
    """
    h, w = anchor_2d.shape
    ds = adaptive_downsample_factor(h, w)
    anchor_ds = anchor_2d[::ds, ::ds]
    mask_ds   = mask_2d[::ds, ::ds]
    valid_ds  = valid_2d[::ds, ::ds]

    anchor_crop, mask_crop, boundary_desc = crop_to_boundary(
        anchor_ds, mask_ds, valid_ds
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # left: anchor band
    ax1 = axes[0]
    vmin = float(np.percentile(anchor_crop, 2))
    vmax = float(np.percentile(anchor_crop, 98))
    im = ax1.imshow(anchor_crop, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    cb = plt.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
    cb.ax.axhline(y=otsu_thresh, color="#E24B4A", linewidth=1.5, label="Otsu")
    if vmax > vmin:
        cb.ax.axhline(
            y=(otsu_thresh - vmin) / (vmax - vmin),
            color="#E24B4A",
            linewidth=1.5,
        )
    anchor_label = next(
        (v for v in band_labels.values() if v in ("NIR", "Red")), "anchor"
    )
    ax1.set_title(
        f"Anchor band ({anchor_label})\n"
        f"Otsu threshold: {otsu_thresh:.4f}  |  boundary at {boundary_desc}"
    )
    ax1.axis("off")

    # right: binary mask
    ax2 = axes[1]
    ax2.imshow(mask_crop, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    sand_pct = int(mask_crop.mean() * 100)
    ax2.set_title(f"Sand mask  (green = sand)\n{sand_pct}% sand in crop")
    ax2.axis("off")

    labels_str = "  |  ".join(
        f"B{slot}: {lbl}" for slot, lbl in sorted(band_labels.items())
    )
    fig.suptitle(f"{filepath.name}\n{labels_str}", fontsize=10, y=1.01)

    plt.tight_layout()
    mask_dir.mkdir(parents=True, exist_ok=True)
    out_path = mask_dir / f"{filepath.stem}_mask.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Per-file assignment
# ---------------------------------------------------------------------------

def assign_labels_for_file(
    filepath: Path,
    proposed_format: str,
    otsu_threshold_hint: float | None = None,
) -> tuple[dict, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Load the raster, build the sand mask, and rank bands to assign labels.

    Parameters
    ----------
    filepath            : path to the raster file
    proposed_format     : format string from script 1 (CIR / RGB / 4BAND / …)
    otsu_threshold_hint : Otsu value from script 1 CSV; recomputed if NaN/None

    Returns
    -------
    result    : dict suitable for a DataFrame row
    anchor_2d : 2-D float array for the anchor band (for visualisation)
    mask_2d   : 2-D uint8 binary sand mask (1 = sand, 0 = water/other)
    valid_2d  : 2-D bool array (True = valid pixel, not NoData) for anchor band
    """
    result = {
        "filepath":        str(filepath),
        "proposed_format": proposed_format,
        "band_1_label":    None,
        "band_2_label":    None,
        "band_3_label":    None,
        "band_4_label":    None,
        "otsu_threshold":  np.nan,
        "sand_pixels":     np.nan,
        "total_pixels":    np.nan,
        "sand_fraction":   np.nan,
        "low_sand_warning": False,
        "error":           None,
    }
    anchor_2d: np.ndarray | None = None
    mask_2d:   np.ndarray | None = None
    valid_2d:  np.ndarray | None = None

    try:
        with rasterio.open(filepath) as src:
            dtype      = src.dtypes[0]
            band_count = src.count

            bands_2d:    list[np.ndarray] = []   # 2-D (NoData filled 0), for mask & viz
            bands_flat:  list[np.ndarray] = []   # 1-D valid pixels only, for stats
            bands_masks: list[np.ndarray] = []   # 2-D bool, True = valid pixel

            for i in range(1, band_count + 1):
                raw  = read_band_masked(src, i, dtype)
                norm = normalize_band(raw, dtype)
                bands_2d.append(norm.filled(0.0))
                bands_flat.append(norm.compressed())
                # Capture the validity mask before filling — used by boundary
                # detection to exclude the NoData void from gradient calculation.
                bands_masks.append(~np.ma.getmaskarray(norm))

        # --- re-identify anchor band by CV ---
        cvs        = [compute_cv(b) for b in bands_flat]
        sorted_idx = np.argsort(cvs)[::-1]
        anchor_idx = int(sorted_idx[0])       # 0-indexed

        anchor_2d = bands_2d[anchor_idx]
        valid_2d  = bands_masks[anchor_idx]   # True where pixel is not NoData

        # --- Otsu threshold: use hint from script 1 if valid, else recompute ---
        if otsu_threshold_hint is not None and not np.isnan(otsu_threshold_hint):
            otsu_thresh = float(otsu_threshold_hint)
        else:
            otsu_thresh = float(threshold_otsu(bands_flat[anchor_idx]))

        result["otsu_threshold"] = otsu_thresh

        # --- binary sand mask ---
        mask_2d = (anchor_2d > otsu_thresh).astype(np.uint8)

        # Count only valid pixels, not the NoData-filled zeros in anchor_2d.
        # bands_flat[anchor_idx] is already .compressed() so len() gives the
        # correct valid pixel count; mask_2d.size would inflate total_pixels
        # by including the NoData border cells.
        sand_pixels  = int((mask_2d.astype(bool) & valid_2d).sum())
        total_pixels = int(valid_2d.sum())
        sand_frac    = sand_pixels / total_pixels if total_pixels > 0 else 0.0
        result["sand_pixels"]      = sand_pixels
        result["total_pixels"]     = total_pixels
        result["sand_fraction"]    = sand_frac
        result["low_sand_warning"] = sand_frac < MIN_SAND_FRACTION

        # --- assign anchor label ---
        band_labels: dict[int, str] = {}   # 1-indexed slot → label
        if proposed_format in ("CIR", "4BAND", "4BAND_WEAK_NIR"):
            band_labels[anchor_idx + 1] = "NIR"
        else:
            # RGB: anchor is the highest-CV visible band → Red
            band_labels[anchor_idx + 1] = "Red"

        # --- rank remaining bands by mean over valid sand pixels ---
        # Using valid_2d as an additional gate ensures NoData cells that happen
        # to be above otsu_thresh (e.g. white fill pixels not caught by masking)
        # do not influence the mean.
        sand_mask_bool = mask_2d.astype(bool) & valid_2d
        remaining_idx  = [i for i in range(band_count) if i != anchor_idx]

        means_over_sand = {
            i: float(bands_2d[i][sand_mask_bool].mean())
            for i in remaining_idx
        }

        # sort ascending — lowest mean gets the first label in RANK_LABELS
        remaining_sorted = sorted(remaining_idx, key=lambda i: means_over_sand[i])
        rank_labels      = RANK_LABELS.get(proposed_format, [])

        if len(remaining_sorted) != len(rank_labels):
            result["error"] = (
                f"Band count mismatch: format '{proposed_format}' expects "
                f"{len(rank_labels)} remaining bands, found {len(remaining_sorted)}."
            )
        else:
            for band_idx, label in zip(remaining_sorted, rank_labels):
                band_labels[band_idx + 1] = label

        # --- write labels into result ---
        for slot, label in band_labels.items():
            result[f"band_{slot}_label"] = label

    except Exception as e:
        result["error"] = str(e)

    return result, anchor_2d, mask_2d, valid_2d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign band labels via sand-mask ranking. Reads format decisions "
                    "from the CV summary CSV produced by compute_band_cv.py."
    )
    parser.add_argument(
        "--cv-csv", required=True, type=Path,
        help="CSV from compute_band_cv.py.",
    )
    parser.add_argument(
        "--output-csv", required=True, type=Path,
        help="Path for the band label assignment output CSV.",
    )
    parser.add_argument(
        "--mask-dir", required=True, type=Path,
        help="Directory for diagnostic mask PNG images.",
    )
    parser.add_argument(
        "--corrections", type=Path, default=None,
        help="(Optional) CSV with manual format overrides. "
             "Two columns: filepath, corrected_format.",
    )
    parser.add_argument(
        "--skip-masks", action="store_true",
        help="Skip generating mask images (faster; labels only).",
    )
    parser.add_argument(
        "--year", type=str, default=None,
        help="(Optional) Process only files from this year.",
    )
    args = parser.parse_args()

    # --- load CV summary ---
    df_cv = pd.read_csv(args.cv_csv)
    print(f"Loaded {len(df_cv)} records from {args.cv_csv}")

    # optional year filter
    if args.year:
        df_cv = df_cv[df_cv["year"].astype(str) == str(args.year)]
        if df_cv.empty:
            print(f"ERROR: no records found for year {args.year}", file=sys.stderr)
            sys.exit(1)
        print(f"Filtered to year {args.year}: {len(df_cv)} files.")

    # --- apply manual corrections ---
    if args.corrections and args.corrections.exists():
        corrections = pd.read_csv(args.corrections).set_index("filepath")["corrected_format"]
        df_cv["proposed_format"] = df_cv.apply(
            lambda r: corrections.get(r["filepath"], r["proposed_format"]), axis=1
        )
        print(f"Applied manual corrections from {args.corrections}.")
    elif args.corrections:
        print(f"Warning: corrections file not found at {args.corrections}; skipping.")

    # skip files that errored in script 1
    df_valid  = df_cv[df_cv["error"].isna()].copy()
    n_skipped = len(df_cv) - len(df_valid)
    if n_skipped:
        print(f"Skipping {n_skipped} file(s) flagged as errors in script 1.")

    # --- main loop ---
    rows: list[dict] = []

    for _, record in tqdm(df_valid.iterrows(), total=len(df_valid), desc="Assigning labels"):
        filepath        = Path(record["filepath"])
        proposed_format = record["proposed_format"]
        otsu_hint       = record.get("otsu_threshold", np.nan)

        result, anchor_2d, mask_2d, valid_2d = assign_labels_for_file(
            filepath, proposed_format, otsu_hint
        )

        # save diagnostic image
        if (
            not args.skip_masks
            and anchor_2d is not None
            and mask_2d is not None
            and valid_2d is not None
        ):
            band_labels = {
                slot: result[f"band_{slot}_label"]
                for slot in range(1, 5)
                if result[f"band_{slot}_label"] is not None
            }
            save_mask_diagnostic(
                filepath, anchor_2d, mask_2d, valid_2d,
                result["otsu_threshold"], band_labels, args.mask_dir,
            )

        rows.append(result)

    # --- build output DataFrame ---
    df_out = pd.DataFrame(rows)

    # re-attach year from CV csv
    year_map       = df_cv.set_index("filepath")["year"].to_dict()
    df_out["year"] = df_out["filepath"].map(year_map)

    # year-level NIR consistency flag:
    # if ANY file in a year produced a CIR or 4BAND decision, the entire
    # year is presumed to be that format — flag RGB outliers for review.
    nir_formats  = {"CIR", "4BAND", "4BAND_WEAK_NIR"}
    year_has_nir = (
        df_out.groupby("year")["proposed_format"]
        .apply(lambda s: bool(set(s) & nir_formats))
        .to_dict()
    )
    df_out["year_nir_detected"] = df_out["year"].map(year_has_nir)

    # column order
    col_order = [
        "filepath", "year", "proposed_format", "year_nir_detected",
        "band_1_label", "band_2_label", "band_3_label", "band_4_label",
        "otsu_threshold", "sand_pixels", "total_pixels", "sand_fraction",
        "low_sand_warning", "error",
    ]
    df_out = df_out[[c for c in col_order if c in df_out.columns]]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)
    print(f"\nBand labels saved → {args.output_csv}")

    # --- console summary ---
    print("\n--- Assignment summary ---")
    print(df_out["proposed_format"].value_counts().to_string())

    low_sand = df_out["low_sand_warning"].sum()
    if low_sand:
        print(f"\nLow-sand-fraction warnings (< {MIN_SAND_FRACTION * 100:.0f}%): {low_sand} file(s).")
        print("  Rank-ordering may be unreliable for these — inspect their mask images.")

    inconsistent = df_out[
        df_out["year_nir_detected"] & df_out["proposed_format"].isin(["RGB"])
    ]
    if len(inconsistent):
        print(
            f"\nYear-consistency flag: {len(inconsistent)} file(s) assigned RGB "
            f"but belong to a year where NIR was detected elsewhere."
        )
        print("  These are candidates for format_corrections.csv.")

    n_errors = df_out["error"].notna().sum()
    if n_errors:
        print(f"\nFiles with assignment errors: {n_errors}")
        print(df_out.loc[df_out["error"].notna(), ["filepath", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()