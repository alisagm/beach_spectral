#!/usr/bin/env python3
"""
Plot error distributions from dynp_experiment.parquet.

Focuses on the most diagnostic comparisons given experiment results:
  1. KDE overlay of key detectors (old, 2bkps_rank_landward, 2bkps_rank_max_contrast)
  2. Winner (2bkps_rank_landward) error stratified by band_mode — is CIR bimodal?
  3. Per-year median + IQR for old vs winner

Reads long-format parquet with columns:
  transect_id, year, band_mode, detector, detected_distance,
  manual_distance, signed_error, confidence, n_bkps, model, selection

Usage
-----
  python tools/plot_dynp_errors.py \\
      --input  OUTPUT/dynp_experiment.parquet \\
      --output OUTPUT/plots/dynp_errors.png
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

# ── Style ─────────────────────────────────────────────────────────────────────

DETECTOR_STYLES = {
    "old_threshold":              {"color": "#2166ac", "label": "old (threshold)"},
    "dynp_2bkps_rank_landward":   {"color": "#1b7837", "label": "2bkps / rank / landward"},
    "dynp_2bkps_rank_max_contrast": {"color": "#d6604d", "label": "2bkps / rank / max_contrast"},
    "dynp_2bkps_rbf_landward":    {"color": "#984ea3", "label": "2bkps / rbf / landward"},
    "dynp_3bkps_rank_middle":     {"color": "#e6ab02", "label": "3bkps / rank / middle"},
    "recursive_rank":             {"color": "#ff7f00", "label": "recursive / rank"},
    "recursive_rank_smooth11":    {"color": "#a65628", "label": "recursive / rank / smooth 11m"},
    "recursive_rank_smooth21":    {"color": "#f781bf", "label": "recursive / rank / smooth 21m"},
}

BAND_COLOURS = {
    "cir":   "#4dac26",
    "4band": "#b8860b",
    "rgb":   "#999999",
}

ERROR_XLIM = (-120, 80)
KDE_LW     = 2.0
GRID_ALPHA = 0.25
ANNOT_KW   = dict(fontsize=8.5, va="top")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kde_curve(values, xlim=ERROR_XLIM, n_pts=500):
    """Return (x, density) arrays, peak-normalised. None if too few points."""
    finite = values[np.isfinite(values)]
    if len(finite) < 10:
        return None, None
    kde     = gaussian_kde(finite, bw_method="scott")
    x_grid  = np.linspace(xlim[0], xlim[1], n_pts)
    density = kde(x_grid)
    density /= density.max()
    return x_grid, density


def _stat_text(values, label):
    """One-line stat summary for legend or annotation."""
    finite = values[np.isfinite(values)]
    return (
        f"{label}  n={len(finite):,}  "
        f"med={np.median(finite):+.1f}  "
        f"mean={np.mean(finite):+.1f}  "
        f"std={np.std(finite):.1f}"
    )


# ── Panel 1: KDE overlay of key detectors ────────────────────────────────────

def _panel_kde_overlay(ax, df):
    """
    Overlaid KDE + histogram for key detectors (NDWI years only).
    """
    # Exclude RGB — too noisy, separate problem
    df_nir = df[df["band_mode"].isin(["cir", "4band"])]

    bins = np.linspace(ERROR_XLIM[0], ERROR_XLIM[1], 80)

    stats_lines = []
    for det_key, style in DETECTOR_STYLES.items():
        errs = df_nir[df_nir["detector"] == det_key]["signed_error"].dropna().values
        if len(errs) < 10:
            continue

        ax.hist(errs, bins=bins, density=True,
                color=style["color"], alpha=0.15, edgecolor="none")

        x, density = _kde_curve(errs)
        if x is not None:
            ax.plot(x, density, color=style["color"], linewidth=KDE_LW,
                    label=style["label"], zorder=3)

        med = float(np.median(errs))
        ax.axvline(med, color=style["color"], linestyle="--",
                   linewidth=1.2, alpha=0.7)

        stats_lines.append(_stat_text(errs, style["label"]))

    ax.axvline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.6)
    ax.set_xlim(ERROR_XLIM)
    ax.set_xlabel("Signed error (m)   +ve = seaward of truth", fontsize=10)
    ax.set_ylabel("Density (peak-normalised)", fontsize=10)
    ax.set_title("Error distributions — NIR years only (CIR + 4band)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=GRID_ALPHA)

    # Stats annotation box
    ax.text(
        0.97, 0.97,
        "\n".join(stats_lines),
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85,
                  edgecolor="#888888", linewidth=0.6),
    )


# ── Panel 2: Winner error by band_mode ───────────────────────────────────────

def _panel_winner_by_bandmode(ax, df, winner="recursive_rank_smooth21"):
    """
    Error distribution for the best config, stratified by band_mode.
    Key diagnostic: is CIR bimodal (vegetation confusion on a subset)?
    """
    sub = df[df["detector"] == winner]
    bins = np.linspace(ERROR_XLIM[0], ERROR_XLIM[1], 70)

    for bm, colour in sorted(BAND_COLOURS.items()):
        errs = sub[sub["band_mode"] == bm]["signed_error"].dropna().values
        if len(errs) < 5:
            continue

        ax.hist(errs, bins=bins, density=True,
                color=colour, alpha=0.30, edgecolor="none")

        x, density = _kde_curve(errs)
        if x is not None:
            ax.plot(x, density, color=colour, linewidth=KDE_LW,
                    label=f"{bm}  (n={len(errs):,})", zorder=3)

        med = float(np.median(errs))
        ax.axvline(med, color=colour, linestyle="--", linewidth=1.2, alpha=0.7,
                   label=f"{bm} median = {med:+.1f} m")

    ax.axvline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.6)
    ax.set_xlim(ERROR_XLIM)
    ax.set_xlabel("Signed error (m)   +ve = seaward of truth", fontsize=10)
    ax.set_ylabel("Density (peak-normalised)", fontsize=10)

    winner_label = DETECTOR_STYLES.get(winner, {}).get("label", winner)
    ax.set_title(f"{winner_label} — error by band mode", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=GRID_ALPHA)


# ── Panel 3: Per-year median + IQR ───────────────────────────────────────────

def _panel_per_year(ax, df, detectors=None):
    """
    Per-year median + IQR whiskers for selected detectors.
    NIR years only (RGB excluded).
    """
    if detectors is None:
        detectors = ["old_threshold", "dynp_2bkps_rank_landward", "recursive_rank", "recursive_rank_smooth21"]

    df_nir = df[df["band_mode"].isin(["cir", "4band"])]
    years = sorted(df_nir["year"].unique(), key=lambda y: int(y))
    x = np.arange(len(years))

    n_det = len(detectors)
    offsets = np.linspace(-0.15, 0.15, n_det)

    for dx, det_key in zip(offsets, detectors):
        style = DETECTOR_STYLES.get(det_key, {"color": "#333333", "label": det_key})
        medians, q25s, q75s = [], [], []

        for y in years:
            vals = df_nir[
                (df_nir["year"] == y) & (df_nir["detector"] == det_key)
            ]["signed_error"].dropna().values

            if len(vals) == 0:
                medians.append(np.nan)
                q25s.append(np.nan)
                q75s.append(np.nan)
            else:
                medians.append(float(np.median(vals)))
                q25s.append(float(np.percentile(vals, 25)))
                q75s.append(float(np.percentile(vals, 75)))

        medians = np.array(medians)
        q25s    = np.array(q25s)
        q75s    = np.array(q75s)

        ax.scatter(x + dx, medians, color=style["color"], s=50,
                   label=style["label"], zorder=4)
        ax.vlines(x + dx, ymin=q25s, ymax=q75s,
                  color=style["color"], linewidth=2.5, alpha=0.6, zorder=3)

    ax.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.6,
               label="truth (0 m)")
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=10)
    ax.set_ylabel("Signed error (m)\n+ve = seaward", fontsize=10)
    ax.set_title("Per-year median error — NIR years  (whiskers = IQR)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=GRID_ALPHA, axis="y")


# ── Figure assembly ───────────────────────────────────────────────────────────

def make_figure(df: pd.DataFrame, output_path: Path) -> None:
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        "Dynp experiment — error distributions\n"
        "error = detected − manual   (+ve = seaward of truth)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[1, 0.8],
        hspace=0.38, wspace=0.30,
    )
    ax_kde      = fig.add_subplot(gs[0, 0])
    ax_bandmode = fig.add_subplot(gs[0, 1])
    ax_peryear  = fig.add_subplot(gs[1, :])

    _panel_kde_overlay(ax_kde, df)
    _panel_winner_by_bandmode(ax_bandmode, df)
    _panel_per_year(ax_peryear, df)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path,
                        default=Path("OUTPUT/dynp_experiment.parquet"))
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("OUTPUT/plots/dynp_errors.png"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    df = pd.read_parquet(args.input)
    logger.info("Loaded %d rows, detectors: %s",
                len(df), sorted(df["detector"].unique()))

    make_figure(df, args.output)


if __name__ == "__main__":
    main()