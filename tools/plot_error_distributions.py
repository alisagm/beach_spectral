#!/usr/bin/env python3
"""
Plot 2 — Error distribution comparison: old (threshold) vs PELT detector.

Primary diagnostic: is old_error bimodal?
  If the threshold detector correctly finds the shell line on most transects
  but falls back to the vegetation/beach boundary on a subset, old_error
  will show two peaks: one near 0 m and one at ~-40 to -80 m (landward).

Secondary: how does the shape of old_error compare to new_error?
  new_error is expected to be roughly unimodal centred near +25 m (seaward
  bias from PELT's distributional break mechanism, established in Plot 1).

Layout (one figure, NDWI years; GRVI shown separately)
-------------------------------------------------------
  Top-left   : overlaid KDE + histogram — old_error vs new_error
               Reference lines at 0 m and each distribution's median.
               Non-detection rates annotated explicitly so the n asymmetry
               is not overlooked.
  Top-right  : old_error only, stratified by band_mode (cir vs 4band)
               Tests whether vegetation-confusion failures are sensor-specific.
  Bottom     : per-year median errors for both detectors as a dot+whisker
               plot, ordered chronologically.
               Useful for spotting whether the old detector's failures are
               concentrated in specific years (e.g. 1995 with partial coverage).

Signed error convention (inherited from analyse_picks.py)
---------------------------------------------------------
  error = algorithm_distance - manual_distance
  positive = algorithm is seaward of truth
  negative = algorithm is landward of truth

Usage
-----
  python scripts/plot_error_distributions.py \\
      --input   OUTPUT/pick_analysis.parquet \\
      --output  OUTPUT/plots/analysis
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)

# ── Style constants ───────────────────────────────────────────────────────────

OLD_COLOUR  = "#2166ac"   # blue  — old (threshold) detector
NEW_COLOUR  = "#d6604d"   # red   — new (PELT) detector
BAND_COLOURS = {"cir": "#4dac26", "4band": "#b8860b"}

GRID_ALPHA  = 0.25
ANNOT_KW    = dict(fontsize=8.5, va="top")
KDE_LW      = 2.0

# x-axis limits for error histograms — wide enough to show the vegetation
# boundary failures (~-80 m) without compressing the main distribution
ERROR_XLIM  = (-120, 80)


# ── Helper ────────────────────────────────────────────────────────────────────

def _kde_line(ax, values, colour, label, xlim=ERROR_XLIM, n_pts=500):
    """Overlay a KDE curve on the current axes (secondary y on right)."""
    finite = values[np.isfinite(values)]
    if len(finite) < 10:
        return
    kde      = gaussian_kde(finite, bw_method="scott")
    x_grid   = np.linspace(xlim[0], xlim[1], n_pts)
    density  = kde(x_grid)
    # Normalise to peak = 1 so CIR/4band or old/new overlay on the same scale
    density /= density.max()
    ax.plot(x_grid, density, color=colour, linewidth=KDE_LW, label=label, zorder=3)


def _annotate_stats(ax, values, colour, label, y_frac=0.97, x_frac=0.97):
    """Text box with n, median, mean, std."""
    finite = values[np.isfinite(values)]
    ax.text(
        x_frac, y_frac,
        f"{label}\n"
        f"n      = {len(finite):,}\n"
        f"median = {np.median(finite):+.1f} m\n"
        f"mean   = {np.mean(finite):+.1f} m\n"
        f"std    = {np.std(finite):.1f} m",
        transform=ax.transAxes, ha="right",
        color=colour,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85,
                  edgecolor=colour, linewidth=0.8),
        **ANNOT_KW,
    )


# ── Panel functions ───────────────────────────────────────────────────────────


def _panel_old_by_bandmode(ax, df_ndwi):
    """
    Top-right: old_error stratified by band_mode.

    Tests whether vegetation-boundary failures (large negative errors)
    are concentrated in a specific sensor type.
    """
    bins = np.linspace(ERROR_XLIM[0], ERROR_XLIM[1], 70)

    for bm, colour in sorted(BAND_COLOURS.items()):
        sub    = df_ndwi[df_ndwi["band_mode"] == bm]["old_error"].dropna()
        if sub.empty:
            continue
        ax.hist(sub.values, bins=bins, density=True,
                color=colour, alpha=0.35, edgecolor="none")
        _kde_line(ax, sub.values, colour, label=f"{bm}  (n={len(sub):,})")
        med = float(np.median(sub))
        ax.axvline(med, color=colour, linestyle="--", linewidth=1.2, alpha=0.8,
                   label=f"{bm} median = {med:+.1f} m")

    ax.axvline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.6)
    ax.set_xlim(ERROR_XLIM)
    ax.set_xlabel("old_error  (m)     +ve = seaward of truth", fontsize=10)
    ax.set_ylabel("Density  (KDE, peak-normalised)", fontsize=10)
    ax.set_title("Old detector error by band mode", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=GRID_ALPHA)


def _panel_per_year(ax, df):
    """
    Bottom: per-year median + IQR for both detectors.

    Points = median, whiskers = 25th–75th percentile.
    Years on x-axis, ordered chronologically.
    Plotted side by side with a small x-offset so both detectors are visible.
    n per year annotated below each year label.
    """
    years = sorted(df["year"].unique(), key=lambda y: int(y))
    x     = np.arange(len(years))
    offset = 0.15

    for dx, col, colour, label in [
        (-offset, "old_error", OLD_COLOUR, "old (threshold)"),
        (+offset, "new_error", NEW_COLOUR, "new (PELT)"),
    ]:
        medians, q25s, q75s, ns = [], [], [], []
        for y in years:
            vals = df[df["year"] == y][col].dropna().values
            ns.append(len(vals))
            if len(vals) == 0:
                medians.append(np.nan); q25s.append(np.nan); q75s.append(np.nan)
            else:
                medians.append(float(np.median(vals)))
                q25s.append(float(np.percentile(vals, 25)))
                q75s.append(float(np.percentile(vals, 75)))

        medians = np.array(medians)
        q25s    = np.array(q25s)
        q75s    = np.array(q75s)

        ax.scatter(x + dx, medians, color=colour, zorder=4,
                   s=60, label=label)
        ax.vlines(x + dx,
                  ymin=q25s, ymax=q75s,
                  color=colour, linewidth=2.5, alpha=0.6, zorder=3)

    ax.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.6,
               label="truth (0 m error)")

    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=10)
    ax.set_ylabel("Signed error  (m)\n+ve = seaward of truth", fontsize=10)
    ax.set_title("Per-year median error  (whiskers = IQR)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=GRID_ALPHA, axis="y")

    # Annotate n below x-axis
    for i, (y, n_old, n_new) in enumerate(zip(
        years,
        [df[(df["year"] == y)]["old_error"].notna().sum() for y in years],
        [df[(df["year"] == y)]["new_error"].notna().sum() for y in years],
    )):
        ax.annotate(
            f"n={n_old}",
            xy=(i - offset, ax.get_ylim()[0]),
            xytext=(i - offset, ax.get_ylim()[0] - 3),
            ha="center", fontsize=7, color=OLD_COLOUR,
        )


# ── Figure assembly ───────────────────────────────────────────────────────────

def make_figure(df: pd.DataFrame, moisture_source: str, output_path: Path) -> None:
    """Build and save the three-panel figure for one moisture index group."""
    sub = df[df["moisture_index_source"] == moisture_source].copy()
    if len(sub) < 20:
        logger.warning("Skipping %s — fewer than 20 rows.", moisture_source)
        return

    old_err = sub["old_error"].values
    new_err = sub["new_error"].values

    none_rate_old = float(sub["old_none"].fillna(True).mean())
    none_rate_new = float(sub["new_none"].fillna(True).mean())

    label_map = {
        "ndwi": "CIR + RGBN years  (NDWI)",
        "grvi": "2022 RGB year  (GRVI fallback)",
    }
    label = label_map.get(moisture_source, moisture_source)

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Detector error distributions — {label}\n"
        f"Signed error = algorithm_distance − manual_distance  "
        f"(+ve seaward, −ve landward)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1, 0.75],
        hspace=0.40,
        wspace=0.32,
    )
    ax_overlay  = fig.add_subplot(gs[0, 0])
    ax_bandmode = fig.add_subplot(gs[0, 1])
    ax_peryear  = fig.add_subplot(gs[1, :])

    # ── Patch the none_rate label issue in _panel_overlay ────────────────
    # Build the overlay panel with correct rates passed explicitly
    bins = np.linspace(ERROR_XLIM[0], ERROR_XLIM[1], 80)
    for err, colour, label_det, none_rate in [
        (old_err, OLD_COLOUR, "old (threshold)", none_rate_old),
        (new_err, NEW_COLOUR, "new (PELT)",      none_rate_new),
    ]:
        finite = err[np.isfinite(err)]
        ax_overlay.hist(finite, bins=bins, density=True,
                        color=colour, alpha=0.30, edgecolor="none")
        kde      = gaussian_kde(finite, bw_method="scott")
        x_grid   = np.linspace(ERROR_XLIM[0], ERROR_XLIM[1], 500)
        density  = kde(x_grid)
        density /= density.max()
        ax_overlay.plot(
            x_grid, density, color=colour, linewidth=KDE_LW, zorder=3,
            label=f"{label_det}  (n={len(finite):,},  none={none_rate*100:.1f}%)",
        )
        med = float(np.nanmedian(err))
        ax_overlay.axvline(med, color=colour, linestyle="--",
                           linewidth=1.2, alpha=0.8)

    ax_overlay.axvline(0, color="black", linestyle="-",
                       linewidth=1.0, alpha=0.6, label="truth (0 m error)")
    ax_overlay.text(
        0.03, 0.97,
        f"Non-detection rates:\n"
        f"  old  = {none_rate_old*100:.1f}%\n"
        f"  PELT = {none_rate_new*100:.1f}%\n"
        f"Errors on detections only.\n"
        f"Dashed lines = medians.",
        transform=ax_overlay.transAxes,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85),
        **ANNOT_KW,
    )
    ax_overlay.set_xlim(ERROR_XLIM)
    ax_overlay.set_xlabel("Signed error  (m)     +ve = seaward of truth", fontsize=10)
    ax_overlay.set_ylabel("Density  (KDE, peak-normalised)", fontsize=10)
    ax_overlay.set_title("Error distributions — old vs PELT", fontsize=11)
    ax_overlay.legend(fontsize=8, loc="upper left",
                      bbox_to_anchor=(0.0, 0.73))
    ax_overlay.grid(True, alpha=GRID_ALPHA)

    _panel_old_by_bandmode(ax_bandmode, sub)
    _panel_per_year(ax_peryear, sub)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot error distributions for old vs PELT detector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("OUTPUT/pick_analysis.parquet"),
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("OUTPUT/plots/analysis"),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.input.exists():
        logger.error("pick_analysis.parquet not found: %s", args.input)
        sys.exit(1)

    df = pd.read_parquet(args.input)
    logger.info("Loaded %d rows", len(df))

    for moisture_source, fname in [
        ("ndwi", "error_distributions_ndwi.png"),
        ("grvi", "error_distributions_grvi.png"),
    ]:
        make_figure(
            df=df,
            moisture_source=moisture_source,
            output_path=args.output / fname,
        )

    print(f"\nFigures written to: {args.output}")


if __name__ == "__main__":
    main()