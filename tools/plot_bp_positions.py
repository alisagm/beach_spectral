#!/usr/bin/env python3
"""
Plot breakpoint positions relative to manual picks.

Reads all_breakpoints.parquet and histograms bp_offset (= bp_distance −
manual_distance) to show where the change-point model places its breaks
relative to the true shell line.

Layout
------
  Top row:  n_bkps=2 (left) and n_bkps=3 (right), rank model only,
            coloured by bp_rank.  NIR years only.
  Bottom:   n_bkps=3 / rank only, faceted by band_mode.

Usage
-----
  python tools/plot_breakpoint_positions.py \\
      --input  OUTPUT/all_breakpoints.parquet \\
      --output OUTPUT/plots/breakpoint_positions.png
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

RANK_COLOURS = ["#1b7837", "#e6ab02", "#d6604d", "#984ea3"]
RANK_LABELS  = ["break 0 (most landward)", "break 1", "break 2", "break 3"]
BAND_COLOURS = {"cir": "#4dac26", "4band": "#b8860b"}

OFFSET_XLIM = (-150, 120)
KDE_LW      = 2.0
GRID_ALPHA  = 0.25


def _kde_curve(values, xlim=OFFSET_XLIM, n_pts=500):
    finite = values[np.isfinite(values)]
    if len(finite) < 10:
        return None, None
    kde     = gaussian_kde(finite, bw_method="scott")
    x_grid  = np.linspace(xlim[0], xlim[1], n_pts)
    density = kde(x_grid)
    density /= density.max()
    return x_grid, density


def _panel_by_rank(ax, df, n_bkps, model="rank"):
    """Histogram of bp_offset coloured by bp_rank for one (n_bkps, model)."""
    sub = df[(df["n_bkps"] == n_bkps) & (df["model"] == model)
             & df["band_mode"].isin(["cir", "4band"])]

    bins = np.linspace(OFFSET_XLIM[0], OFFSET_XLIM[1], 80)

    for rank in sorted(sub["bp_rank"].unique()):
        vals = sub[sub["bp_rank"] == rank]["bp_offset"].dropna().values
        if len(vals) < 5:
            continue
        colour = RANK_COLOURS[rank % len(RANK_COLOURS)]
        label  = RANK_LABELS[rank] if rank < len(RANK_LABELS) else f"break {rank}"

        ax.hist(vals, bins=bins, density=True,
                color=colour, alpha=0.25, edgecolor="none")
        x, density = _kde_curve(vals)
        if x is not None:
            ax.plot(x, density, color=colour, linewidth=KDE_LW,
                    label=f"{label}  (n={len(vals):,}, med={np.median(vals):+.0f}m)",
                    zorder=3)
        ax.axvline(np.median(vals), color=colour, linestyle="--",
                   linewidth=1.0, alpha=0.6)

    ax.axvline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.7,
               label="manual pick (0 m)")
    ax.set_xlim(OFFSET_XLIM)
    ax.set_xlabel("Offset from manual pick (m)   +ve = seaward", fontsize=10)
    ax.set_ylabel("Density (peak-normalised)", fontsize=10)
    ax.set_title(f"n_bkps={n_bkps}, model={model} — NIR years", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=GRID_ALPHA)


def _panel_3bkps_by_bandmode(ax, df, model="rank"):
    """
    n_bkps=3, rank model: overlay bp_rank=1 (the middle break) per band_mode.
    Tests whether CIR and 4band place the middle break differently.
    """
    sub = df[(df["n_bkps"] == 3) & (df["model"] == model) & (df["bp_rank"] == 1)]
    bins = np.linspace(OFFSET_XLIM[0], OFFSET_XLIM[1], 70)

    for bm, colour in sorted(BAND_COLOURS.items()):
        vals = sub[sub["band_mode"] == bm]["bp_offset"].dropna().values
        if len(vals) < 5:
            continue
        ax.hist(vals, bins=bins, density=True,
                color=colour, alpha=0.30, edgecolor="none")
        x, density = _kde_curve(vals)
        if x is not None:
            ax.plot(x, density, color=colour, linewidth=KDE_LW,
                    label=f"{bm}  (n={len(vals):,}, med={np.median(vals):+.0f}m)",
                    zorder=3)
        ax.axvline(np.median(vals), color=colour, linestyle="--",
                   linewidth=1.0, alpha=0.6)

    ax.axvline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.7,
               label="manual pick (0 m)")
    ax.set_xlim(OFFSET_XLIM)
    ax.set_xlabel("Offset from manual pick (m)   +ve = seaward", fontsize=10)
    ax.set_ylabel("Density (peak-normalised)", fontsize=10)
    ax.set_title(f"Middle break (rank 1) by band mode — n_bkps=3, {model}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=GRID_ALPHA)


def make_figure(df: pd.DataFrame, output_path: Path) -> None:
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        "Where does Dynp place its breakpoints relative to the shell line?\n"
        "offset = bp_distance − manual_distance   (+ve = seaward)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 0.85],
                           hspace=0.38, wspace=0.30)

    ax_2bkps    = fig.add_subplot(gs[0, 0])
    ax_3bkps    = fig.add_subplot(gs[0, 1])
    ax_bandmode = fig.add_subplot(gs[1, :])

    _panel_by_rank(ax_2bkps, df, n_bkps=2)
    _panel_by_rank(ax_3bkps, df, n_bkps=3)
    _panel_3bkps_by_bandmode(ax_bandmode, df)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path,
                        default=Path("OUTPUT/all_breakpoints.parquet"))
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("OUTPUT/plots/breakpoint_positions.png"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    df = pd.read_parquet(args.input)
    logger.info("Loaded %d rows", len(df))
    make_figure(df, args.output)


if __name__ == "__main__":
    main()