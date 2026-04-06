#!/usr/bin/env python3
"""
Plot 1 — What is PELT anchoring on?

Tests the hypothesis that PELT finds the steepest moisture-index gradient
rather than the shell line position.

If true: new_error ≈ ndwi_gradient_offset, so points cluster around y = x.
The residual (new_error − ndwi_gradient_offset) measures how far PELT
overshoots even the gradient peak — a positive residual mean would
implicate foam pulling detections further seaward.

Output figures
--------------
  gradient_anchoring_ndwi.png   — NDWI years (CIR + RGBN), main analysis
  gradient_anchoring_grvi.png   — GRVI year (2022 RGB), shown separately

Layout (each figure)
--------------------
  Left  : hexbin — ndwi_gradient_offset (x) vs new_error (y)
           y = x reference line (PELT = gradient peak hypothesis)
           y = 0 reference line (PELT = manual pick)
           annotated with Pearson r and regression line
  Right : histogram — residual (new_error − ndwi_gradient_offset)
           dashed line at 0 (perfect gradient-anchoring prediction)
           annotated with mean and std

Usage
-----
  python scripts/plot_gradient_anchoring.py \\
      --input   OUTPUT/pick_analysis.parquet \\
      --output  OUTPUT/plots/analysis
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ── Style constants ───────────────────────────────────────────────────────────

HEXBIN_CMAP   = "YlOrRd"
HEXBIN_BINS   = "log"          # log-scale counts handle the density range well
GRID_ALPHA    = 0.25
REFLINE_KW    = dict(linewidth=1.2, alpha=0.8)
ANNOT_KW      = dict(fontsize=9, va="top")


# ── Core plot functions ───────────────────────────────────────────────────────

def _hexbin_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    moisture_label: str,
    window_m: float,
) -> None:
    """
    Hexbin of gradient_offset (x) vs new_error (y) with reference lines.

    The axis limits are set symmetrically to ±window_m so the y=x line
    is visually anchored to the plot corners.
    """
    lim = window_m * 1.05

    hb = ax.hexbin(
        x, y,
        gridsize=40,
        bins=HEXBIN_BINS,
        cmap=HEXBIN_CMAP,
        extent=[-lim, lim, -lim, lim],
    )
    plt.colorbar(hb, ax=ax, label="log₁₀(count)", pad=0.02)

    # Reference lines
    ax.axhline(0,  color="steelblue",  linestyle="--", label="PELT = manual pick (y=0)", **REFLINE_KW)
    ax.axvline(0,  color="grey",       linestyle=":",  alpha=0.5, linewidth=1)
    ref_x = np.array([-lim, lim])
    ax.plot(ref_x, ref_x, color="firebrick", linestyle="-",
            label="PELT = gradient peak (y=x)", **REFLINE_KW)

    # OLS regression line + Spearman r (on finite values only)
    # NOTE: Pearson r and OLS slope are unreliable here because
    # ndwi_gradient_offset is clipped at ±window_m while new_error is
    # uncapped. Outliers at x≈+30, y>>30 dominate OLS and produce a
    # misleading negative slope even when the bulk of data lies on y=x.
    #
    # Full-range Spearman is also compromised: the hard clip at ±window_m
    # creates rank ties at both walls. Every point where the gradient peak
    # falls outside the window receives the same maximum rank regardless
    # of its actual new_error value, injecting rank noise that cancels the
    # true positive signal in the interior.
    #
    # Interior-only Spearman (|x| < 0.9 × window_m) excludes wall-clipped
    # points and gives a credible effect-size estimate for the anchoring
    # hypothesis where the measurement is actually meaningful.
    INTERIOR_FRAC = 0.9
    mask          = np.isfinite(x) & np.isfinite(y)
    interior_mask = mask & (np.abs(x) < window_m * INTERIOR_FRAC)

    if mask.sum() > 10:
        slope, intercept, r_pearson, _, _ = stats.linregress(x[mask], y[mask])
        rho_full, p_full         = stats.spearmanr(x[mask],          y[mask])
        rho_int,  p_int          = stats.spearmanr(x[interior_mask], y[interior_mask])

        fit_x = np.array([-lim, lim])
        ax.plot(fit_x, slope * fit_x + intercept,
                color="black", linestyle="-.", linewidth=1.0, alpha=0.7,
                label=f"OLS fit  (Pearson r={r_pearson:.2f}, ⚠ outlier-sensitive)")

        # Shade the interior region so the reader can see which points
        # the interior Spearman is computed from
        interior_lim = window_m * INTERIOR_FRAC
        ax.axvspan(-interior_lim, interior_lim,
                   color="grey", alpha=0.07, zorder=0,
                   label=f"interior  (|offset| < {interior_lim:.0f} m, n={interior_mask.sum():,})")

        def _p_str(p):
            return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"

        ax.text(
            0.03, 0.97,
            f"n (all)      = {mask.sum():,}\n"
            f"n (interior) = {interior_mask.sum():,}  "
            f"(|offset| < {interior_lim:.0f} m)\n"
            f"\n"
            f"Spearman ρ  interior  = {rho_int:+.3f}  (p={_p_str(p_int)})  ← use this\n"
            f"Spearman ρ  full      = {rho_full:+.3f}  (p={_p_str(p_full)})  ⚠ wall-clipping\n"
            f"Pearson r   full      = {r_pearson:+.3f}  ⚠ outlier-sensitive\n"
            f"OLS slope = {slope:.2f}  intercept = {intercept:.1f} m",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85),
            **ANNOT_KW,
        )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("ndwi_gradient_offset  (m from manual pick,\n+ve = gradient peak is seaward)", fontsize=10)
    ax.set_ylabel("new_error  (m from manual pick,\n+ve = PELT is seaward of truth)", fontsize=10)
    ax.set_title(f"PELT error vs gradient peak offset\n({moisture_label})", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_aspect("equal")


def _residual_panel(
    ax: plt.Axes,
    residual: np.ndarray,
    moisture_label: str,
    band_modes: "pd.Series | None" = None,
) -> None:
    """
    Histogram of residual = new_error − ndwi_gradient_offset.

    A residual near zero means PELT lands at the gradient peak.
    A positive mean means PELT overshoots even the gradient peak (foam pull).
    A negative mean means PELT undershoots the gradient peak.

    If band_modes is supplied (NDWI years only), overlays separate histograms
    per band_mode group so CIR and 4band residuals can be compared directly.
    A larger residual mean for CIR would implicate turbid-water NIR inflation
    as an additional foam-pull mechanism on top of the gradient anchoring.
    """
    finite_mask = np.isfinite(residual)

    # ── Colour palette for band_mode groups ──────────────────────────────
    GROUP_COLOURS = {"cir": "steelblue", "4band": "darkorange"}

    if band_modes is not None:
        # Overlay one histogram per band_mode group
        groups = {}
        for bm in band_modes[finite_mask].unique():
            grp_mask = finite_mask & (band_modes == bm).values
            groups[bm] = residual[grp_mask]

        # Shared bin edges across groups for visual comparability
        all_finite = residual[finite_mask]
        bin_edges  = np.linspace(
            max(all_finite.min(), -200),
            min(all_finite.max(),  200),
            61,
        )

        for bm, vals in sorted(groups.items()):
            colour = GROUP_COLOURS.get(bm, "grey")
            ax.hist(vals, bins=bin_edges, alpha=0.55, color=colour,
                    edgecolor="none", label=f"{bm}  (n={len(vals):,})")
            mean_v = float(np.mean(vals))
            ax.axvline(mean_v, color=colour, linestyle="--", linewidth=1.4,
                       label=f"{bm} mean = {mean_v:+.1f} m")

        ax.axvline(0, color="firebrick", linestyle="-", linewidth=1.2,
                   label="perfect gradient-anchoring (residual=0)")

        # Global summary annotation
        mean_all = float(np.mean(all_finite))
        std_all  = float(np.std(all_finite))
        ax.text(
            0.97, 0.97,
            f"overall n = {len(all_finite):,}\n"
            f"mean = {mean_all:+.1f} m\n"
            f"std  = {std_all:.1f} m",
            transform=ax.transAxes, ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8),
            **ANNOT_KW,
        )

    else:
        # Single-group path (used for GRVI)
        finite  = residual[finite_mask]
        mean_r  = float(np.mean(finite))
        std_r   = float(np.std(finite))

        ax.hist(finite, bins=60, color="steelblue", alpha=0.75, edgecolor="none")
        ax.axvline(0,      color="firebrick", linestyle="-",  linewidth=1.2,
                   label="perfect gradient-anchoring prediction")
        ax.axvline(mean_r, color="black",     linestyle="--", linewidth=1.2,
                   label=f"mean = {mean_r:+.1f} m")

        foam_str = (
            "← PELT overshoots gradient"  if mean_r >  2 else
            "← PELT undershoots gradient" if mean_r < -2 else
            "≈ PELT lands at gradient peak"
        )
        ax.text(
        0.97, 0.97,
        f"n = {len(finite):,}\n"
        f"mean = {mean_r:+.1f} m\n"
        f"std  = {std_r:.1f} m\n"
        f"{foam_str}",
        transform=ax.transAxes, ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8),
        **ANNOT_KW,
    )

    ax.set_xlabel("Residual: new_error − gradient_offset  (m)\n"
                  "+ve = PELT seaward of gradient peak  |  −ve = landward", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Residual after gradient-peak correction\n({moisture_label})", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=GRID_ALPHA, axis="y")


def make_figure(
    df: pd.DataFrame,
    moisture_source: str,
    output_path: Path,
    window_m: float,
) -> None:
    """
    Build and save the two-panel figure for one moisture index group.
    """
    # Filter to this moisture source and rows where PELT produced a detection
    mask = (
        (df["moisture_index_source"] == moisture_source) &
        df["new_error"].notna() &
        df["ndwi_gradient_offset"].notna()
    )
    sub = df[mask].copy()

    if len(sub) < 20:
        logger.warning(
            "Only %d valid rows for moisture_source=%s — skipping figure.",
            len(sub), moisture_source,
        )
        return

    x        = sub["ndwi_gradient_offset"].values
    y        = sub["new_error"].values
    residual = y - x

    label_map = {
        "ndwi": "CIR + RGBN years  (NDWI)",
        "grvi": "2022 RGB year  (GRVI fallback)",
    }
    label = label_map.get(moisture_source, moisture_source)

    fig, (ax_hex, ax_res) = plt.subplots(
        1, 2,
        figsize=(13, 6),
        gridspec_kw={"width_ratios": [1.1, 0.9]},
    )
    fig.suptitle(
        "Is PELT anchoring on the steepest spectral gradient?\n"
        "Hypothesis: new_error ≈ gradient_offset  →  points cluster on y = x",
        fontsize=12, fontweight="bold", y=1.01,
    )

    _hexbin_panel(ax_hex, x, y, label, window_m)
    band_modes = sub["band_mode"] if moisture_source == "ndwi" else None
    _residual_panel(ax_res, residual, label, band_modes=band_modes)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PELT error vs gradient offset to test anchoring hypothesis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("OUTPUT/pick_analysis.parquet"),
        help="pick_analysis.parquet from analyse_picks.py (default: OUTPUT/pick_analysis.parquet)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("OUTPUT/plots/analysis"),
        help="Output directory for PNGs (default: OUTPUT/plots/analysis)",
    )
    parser.add_argument(
        "--window-m",
        type=float,
        default=30.0,
        help="Analysis window half-width used when building pick_analysis "
             "(controls axis limits; default: 30)",
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
    logger.info("Loaded %d rows from %s", len(df), args.input)

    # Log a quick look at what we have before plotting
    pelt_detected = df["new_error"].notna().sum()
    logger.info(
        "PELT detections: %d / %d  (%.1f %% none-rate)",
        pelt_detected, len(df), 100 * (1 - pelt_detected / len(df)),
    )

    for moisture_source, fname in [
        ("ndwi", "gradient_anchoring_ndwi.png"),
        ("grvi", "gradient_anchoring_grvi.png"),
    ]:
        make_figure(
            df=df,
            moisture_source=moisture_source,
            output_path=args.output / fname,
            window_m=args.window_m,
        )

    print(f"\nFigures written to: {args.output}")


if __name__ == "__main__":
    main()