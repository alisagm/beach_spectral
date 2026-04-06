#!/usr/bin/env python3
"""
tools/plot_comparison.py — Stratified comparison plots for old vs PELT detector.

Reads comparison_all.parquet (from compare_detectors.py) and produces one plot
per selected transect.  Transects are selected by stratified sampling across
(agreement_category × band_mode) cells, with year diversity enforced within
each cell so no single year dominates.

The plot for each transect shows:
  - NDWI proxy signal along the transect (primary axis)
  - Variability signal (secondary axis)
  - Vertical line: old detector position (orange dashed)
  - Vertical line: new (PELT) detector position (blue solid)
  - Title: transect_id, year, band_mode, delta_m, category

Outputs (written to --output-dir)
-----------------------------------
  {band_mode}/{category}/transect_{id}_{year}.png

Usage
-----
  python tools/plot_comparison.py \\
      --comparison-dir  OUTPUT/comparison/ \\
      --features-dir    OUTPUT/ \\
      --output-dir      OUTPUT/comparison_plots/ \\
      --n-per-cell      4

  # Specific category only:
  python tools/plot_comparison.py ... --category divergent
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Sampling constants ────────────────────────────────────────────────────────
ALL_CATEGORIES = ["close", "moderate", "divergent", "new_only", "old_only", "both_none"]
ALL_BAND_MODES = ["cir", "4band", "rgb"]

# Colours
_OLD_COLOUR  = "#E87722"   # orange — old detector
_NEW_COLOUR  = "#1D6FA4"   # blue   — PELT
_NDWI_COLOUR = "#2E9E6B"   # green  — NDWI proxy signal
_VAR_COLOUR  = "#9B6BB5"   # purple — variability signal


# ── Stratified sampler ────────────────────────────────────────────────────────

def stratified_sample(
    df: pd.DataFrame,
    n_per_cell: int,
    category_filter: Optional[str] = None,
    band_mode_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Sample up to n_per_cell rows per (agreement_category × band_mode) cell,
    maximising year diversity within each cell.

    Returns a DataFrame of selected rows with columns needed for plotting.
    """
    cats  = [category_filter] if category_filter else ALL_CATEGORIES
    modes = [band_mode_filter] if band_mode_filter else ALL_BAND_MODES

    selected = []
    for mode in modes:
        for cat in cats:
            cell = df[(df["band_mode"] == mode) & (df["agreement_category"] == cat)]
            if cell.empty:
                continue
            n_available = len(cell)
            if n_available <= n_per_cell:
                selected.append(cell)
                logger.debug(
                    "Cell (%s, %s): taking all %d rows", mode, cat, n_available
                )
                continue

            # Sample with year diversity: round-robin across years
            years = sorted(cell["year"].unique())
            chosen = []
            year_cycle = (years * ((n_per_cell // len(years)) + 2))[:n_per_cell * 2]
            for y in year_cycle:
                if len(chosen) >= n_per_cell:
                    break
                year_pool = cell[cell["year"] == y]
                already_chosen_ids = {r["transect_id"] for r in chosen}
                remaining = year_pool[~year_pool["transect_id"].isin(already_chosen_ids)]
                if remaining.empty:
                    continue
                chosen.append(remaining.sample(1, random_state=42).iloc[0].to_dict())

            logger.debug(
                "Cell (%s, %s): sampled %d from %d rows", mode, cat, len(chosen), n_available
            )
            selected.append(pd.DataFrame(chosen))

    if not selected:
        return pd.DataFrame()
    return pd.concat(selected, ignore_index=True)


# ── Feature loading ───────────────────────────────────────────────────────────

def _find_features_path(features_dir: Path, year: int) -> Optional[Path]:
    """Support flat and year-subdirectory layouts."""
    candidates = [
        features_dir / f"features_{year}.parquet",
        features_dir / str(year) / f"features_{year}.parquet",
    ]
    return next((p for p in candidates if p.exists()), None)


def _load_transect_features(
    features_dir: Path,
    year: int,
    transect_id: int,
    band_mode: str,
) -> Optional[pd.DataFrame]:
    path = _find_features_path(features_dir, year)
    if path is None:
        logger.warning("features_%d.parquet not found — skipping transect %d", year, transect_id)
        return None
    df = pd.read_parquet(path)
    tdf = df[df["transect_id"] == transect_id].copy()
    if tdf.empty:
        logger.warning("Transect %d not found in features_%d.parquet", transect_id, year)
        return None
    return tdf.sort_values("distance").reset_index(drop=True)


def _ndwi_proxy(tdf: pd.DataFrame, band_mode: str) -> pd.Series:
    """Return the NDWI proxy series (same sign convention as pelt.py)."""
    if band_mode == "rgb":
        # grvi negated so higher = more water-like (matches NDWI direction)
        if "grvi" in tdf.columns:
            return -tdf["grvi"]
        logger.warning("grvi column missing for RGB transect; falling back to NaN series")
        return pd.Series(np.nan, index=tdf.index)
    else:
        if "ndwi" in tdf.columns:
            return tdf["ndwi"]
        logger.warning("ndwi column missing; returning NaN series")
        return pd.Series(np.nan, index=tdf.index)


# ── Single-transect plot ──────────────────────────────────────────────────────

def plot_transect(
    row: pd.Series,
    tdf: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Render and save the comparison plot for one transect.

    Layout:
      Primary axis  — NDWI proxy (left y-axis)
      Secondary axis — variability (right y-axis)
      Vertical lines — old detector (orange dashed), PELT (blue solid)
    """
    band_mode = row["band_mode"]
    dist      = tdf["distance"].to_numpy()
    ndwi_prx  = _ndwi_proxy(tdf, band_mode).to_numpy()
    var_       = tdf["variability"].to_numpy() if "variability" in tdf.columns else None

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    # NDWI proxy
    ax1.plot(dist, ndwi_prx, color=_NDWI_COLOUR, lw=1.4, alpha=0.85,
             label="NDWI proxy" if band_mode != "rgb" else "−grvi proxy")
    ax1.set_ylabel(
        "NDWI" if band_mode != "rgb" else "−grvi (water proxy)",
        color=_NDWI_COLOUR,
    )
    ax1.tick_params(axis="y", colors=_NDWI_COLOUR)

    # Variability
    if var_ is not None:
        ax2.plot(dist, var_, color=_VAR_COLOUR, lw=1.0, alpha=0.55,
                 linestyle="--", label="variability")
        ax2.set_ylabel("variability (window std)", color=_VAR_COLOUR)
        ax2.tick_params(axis="y", colors=_VAR_COLOUR)

    # Detector positions
    ymin, ymax = ax1.get_ylim()

    if not row["old_none"] and not np.isnan(row["old_distance"]):
        ax1.axvline(
            row["old_distance"], color=_OLD_COLOUR,
            linestyle="--", lw=1.6, alpha=0.9,
            label=f"old  {row['old_distance']:.1f} m  (conf {row['old_confidence']:.2f})",
        )

    if not row["new_none"] and not np.isnan(row["new_distance"]):
        ax1.axvline(
            row["new_distance"], color=_NEW_COLOUR,
            linestyle="-", lw=1.6, alpha=0.9,
            label=f"PELT {row['new_distance']:.1f} m  (conf {row['new_confidence']:.2f})",
        )

    # Directional labels
    ax1.set_xlabel("distance from landward end (m)  →  seaward")
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(10))

    # Build title
    delta_str = (
        f"Δ = {row['delta_m']:+.1f} m"
        if not np.isnan(row["delta_m"])
        else "Δ = n/a"
    )
    partial_tag = "  ⚠ partial coverage" if row.get("new_partial_coverage") else ""
    ax1.set_title(
        f"Transect {int(row['transect_id'])}  |  {int(row['year'])}  |  "
        f"{band_mode.upper()}  |  {row['agreement_category']}  |  {delta_str}"
        f"{partial_tag}",
        fontsize=10,
    )

    # Combined legend from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-dir", required=True, type=Path,
        help="Directory containing comparison_all.parquet.",
    )
    parser.add_argument(
        "--features-dir", required=True, type=Path,
        help="Directory containing features_{year}.parquet files.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Root directory for output plots.",
    )
    parser.add_argument(
        "--n-per-cell", type=int, default=4,
        help="Max transects to plot per (category × band_mode) cell (default: 4).",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        choices=ALL_CATEGORIES,
        help="Restrict to one agreement category (default: all).",
    )
    parser.add_argument(
        "--band-mode", type=str, default=None,
        choices=ALL_BAND_MODES,
        help="Restrict to one band mode (default: all).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    combined_path = args.comparison_dir / "comparison_all.parquet"
    if not combined_path.exists():
        logger.error("comparison_all.parquet not found at %s", combined_path)
        return

    df = pd.read_parquet(combined_path)
    logger.info("Loaded %d rows from comparison_all.parquet", len(df))

    selected = stratified_sample(df, args.n_per_cell, args.category, args.band_mode)
    if selected.empty:
        logger.warning("No rows selected — nothing to plot.")
        return

    logger.info(
        "Plotting %d transects across %d cells",
        len(selected),
        selected.groupby(["band_mode", "agreement_category"]).ngroups,
    )

    n_ok = 0
    n_skip = 0
    for _, row in selected.iterrows():
        tdf = _load_transect_features(
            args.features_dir,
            int(row["year"]),
            int(row["transect_id"]),
            row["band_mode"],
        )
        if tdf is None:
            n_skip += 1
            continue

        out_path = (
            args.output_dir
            / row["band_mode"]
            / row["agreement_category"]
            / f"transect_{int(row['transect_id'])}_{int(row['year'])}.png"
        )
        plot_transect(row, tdf, out_path)
        n_ok += 1

    logger.info("Done.  %d plots written, %d skipped.", n_ok, n_skip)
    print(f"\nPlots written to: {args.output_dir}")


if __name__ == "__main__":
    main()