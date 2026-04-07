#!/usr/bin/env python3
"""
Plot breakpoint contrast vs offset from manual pick.

Key question: do breakpoints near the shell line (offset ≈ 0) have
systematically weaker contrast than the veg/beach and water-edge breaks?

Layout
------
  Left:   hex density plot of bp_contrast vs bp_offset (n_bkps=3, rank, NIR).
          If the shell line exists as a detectable shift, there should be a
          cluster of low-but-positive-contrast points near offset=0.
  Right:  contrast distributions for three offset zones:
            "veg/beach"   (offset < -20 m)
            "shell line"  (|offset| ≤ 15 m)
            "water edge"  (offset > +20 m)
          If shell-line breaks have distinctly lower contrast than flanking
          breaks, that's evidence the signal exists but is drowned out by
          the stronger boundaries.

Uses n_bkps=3 / rank / NIR only — that's where all three boundaries appear.

Usage
-----
  python tools/plot_contrast_vs_offset.py \\
      --input  OUTPUT/all_breakpoints.parquet \\
      --output OUTPUT/plots/contrast_vs_offset.png
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)

GRID_ALPHA = 0.25
KDE_LW     = 2.0

# Offset zone boundaries (metres from manual pick)
VEG_BEACH_UPPER  = -20.0
SHELL_LINE_HALF  =  15.0
WATER_EDGE_LOWER =  20.0

ZONE_STYLES = {
    "veg/beach (offset < −20m)":  {"color": "#1b7837", "range": (-np.inf, VEG_BEACH_UPPER)},
    "shell line (|offset| ≤ 15m)": {"color": "#2166ac", "range": (-SHELL_LINE_HALF, SHELL_LINE_HALF)},
    "water edge (offset > +20m)":  {"color": "#d6604d", "range": (WATER_EDGE_LOWER, np.inf)},
}


def _kde_curve(values, xlim, n_pts=400):
    finite = values[np.isfinite(values)]
    if len(finite) < 10:
        return None, None
    kde     = gaussian_kde(finite, bw_method="scott")
    x_grid  = np.linspace(xlim[0], xlim[1], n_pts)
    density = kde(x_grid)
    density /= density.max()
    return x_grid, density


def _panel_hex(ax, df):
    """Hex density: bp_contrast vs bp_offset."""
    offsets   = df["bp_offset"].values
    contrasts = df["bp_contrast"].values

    hb = ax.hexbin(
        offsets, contrasts,
        gridsize=60,
        cmap="YlOrBr",
        mincnt=1,
        extent=[-150, 120, -0.6, 0.6],
    )
    plt.colorbar(hb, ax=ax, label="count", shrink=0.8)

    ax.axvline(0, color="black", linewidth=1.2, alpha=0.7, label="manual pick")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5,
               label="zero contrast")

    # Zone bands
    ax.axvspan(-150, VEG_BEACH_UPPER, alpha=0.06, color="#1b7837")
    ax.axvspan(-SHELL_LINE_HALF, SHELL_LINE_HALF, alpha=0.08, color="#2166ac")
    ax.axvspan(WATER_EDGE_LOWER, 120, alpha=0.06, color="#d6604d")

    ax.set_xlim(-150, 120)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("Offset from manual pick (m)   +ve = seaward", fontsize=10)
    ax.set_ylabel("Breakpoint contrast\n(seaward NDWI mean − landward)", fontsize=10)
    ax.set_title("All breakpoints — contrast vs position\nn_bkps=3, rank, NIR years", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=GRID_ALPHA)


def _panel_zone_contrasts(ax, df):
    """KDE of bp_contrast for each offset zone."""
    contrast_xlim = (-0.5, 0.5)

    for zone_label, style in ZONE_STYLES.items():
        lo, hi = style["range"]
        vals = df[(df["bp_offset"] >= lo) & (df["bp_offset"] < hi)]["bp_contrast"].dropna().values

        if len(vals) < 5:
            continue

        x, density = _kde_curve(vals, contrast_xlim)
        if x is not None:
            ax.plot(x, density, color=style["color"], linewidth=KDE_LW,
                    label=f"{zone_label}  (n={len(vals):,}, med={np.median(vals):+.3f})",
                    zorder=3)
            ax.fill_between(x, density, alpha=0.12, color=style["color"])

    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlim(contrast_xlim)
    ax.set_xlabel("Breakpoint contrast (NDWI units)", fontsize=10)
    ax.set_ylabel("Density (peak-normalised)", fontsize=10)
    ax.set_title("Contrast distribution by offset zone", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=GRID_ALPHA)


def _panel_scatter_near_shell(ax, df):
    """
    Zoomed scatter of breaks within ±40m of pick, coloured by contrast.
    Helps see whether the near-zero breaks form a distinct cluster or are
    just the tails of the flanking distributions.
    """
    near = df[(df["bp_offset"].abs() <= 40)].copy()
    if near.empty:
        ax.text(0.5, 0.5, "No data in ±40m zone", transform=ax.transAxes, ha="center")
        return

    sc = ax.scatter(
        near["bp_offset"], near["bp_contrast"],
        c=near["bp_contrast"], cmap="RdYlGn", vmin=-0.3, vmax=0.3,
        s=8, alpha=0.4, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="contrast", shrink=0.8)

    ax.axvline(0, color="black", linewidth=1.2, alpha=0.7)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlim(-40, 40)
    ax.set_ylim(-0.35, 0.35)
    ax.set_xlabel("Offset from manual pick (m)", fontsize=10)
    ax.set_ylabel("Breakpoint contrast", fontsize=10)
    ax.set_title("Zoomed: breaks within ±40m of shell line", fontsize=11)
    ax.grid(True, alpha=GRID_ALPHA)


def make_figure(df: pd.DataFrame, output_path: Path) -> None:
    # Filter to n_bkps=3, rank, NIR
    sub = df[
        (df["n_bkps"] == 3) & (df["model"] == "rank")
        & df["band_mode"].isin(["cir", "4band"])
    ].copy()

    logger.info("Plotting %d breakpoint rows (n_bkps=3, rank, NIR)", len(sub))

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Breakpoint contrast vs position — is the shell line detectable?\n"
        "n_bkps=3, model=rank, NIR years only",
        fontsize=13, fontweight="bold", y=1.01,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 0.85],
                           hspace=0.38, wspace=0.30)

    ax_hex    = fig.add_subplot(gs[0, 0])
    ax_zones  = fig.add_subplot(gs[0, 1])
    ax_zoom   = fig.add_subplot(gs[1, :])

    _panel_hex(ax_hex, sub)
    _panel_zone_contrasts(ax_zones, sub)
    _panel_scatter_near_shell(ax_zoom, sub)

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
                        default=Path("OUTPUT/plots/contrast_vs_offset.png"))
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