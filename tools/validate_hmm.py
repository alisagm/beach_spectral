"""
Validate HMM shell line detection on a single year.

Fits a 4-state 2D (NDWI + NIR) Gaussian HMM, runs detection on all
transects, and produces diagnostic plots for visual inspection.

Usage:
    python tools/validate_hmm.py --year 2010
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spectral_classifier.transition.hmm import (
    fit_hmm_for_year,
    detect_shell_line_hmm,
    run_hmm_detection,
    VEGETATION, DRY_BEACH, WET_BEACH, WATER,
    STATE_LABELS, N_STATES, FEATURE_COLS,
)
from spectral_classifier.utils.data_io import merge_profiles_and_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
STATE_COLORS = {
    VEGETATION: "#97C459",   # green
    DRY_BEACH:  "#5DCAA5",   # teal
    WET_BEACH:  "#F0997B",   # coral / sandy
    WATER:      "#85B7EB",   # blue
}
NDWI_COLOR        = "#3C3489"
SHELL_LINE_COLOR  = "#D85A30"
WATERLINE_COLOR   = "#185FA5"
VEG_BOUNDARY_COLOR = "#27500A"
MANUAL_COLOR      = "#639922"


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_transect_hmm(
    ax, distances, ndwi, states, shell_line_dist, confidence,
    transect_id, compact=False,
    manual_dist=None, waterline_dist=None, veg_boundary_dist=None,
):
    """Render one transect's NDWI profile with 4-state HMM overlay."""
    fontsize_title = 9 if compact else 12
    fontsize_tick  = 6 if compact else 8

    # State background shading.
    for i in range(len(distances) - 1):
        color = STATE_COLORS.get(states[i], "#ccc")
        ax.axvspan(distances[i], distances[i + 1], alpha=0.25, color=color,
                   linewidth=0)

    # NDWI profile.
    ax.plot(distances, ndwi, color=NDWI_COLOR, linewidth=1.2, alpha=0.9)

    # Boundary markers.
    if veg_boundary_dist is not None and np.isfinite(veg_boundary_dist):
        ax.axvline(veg_boundary_dist, color=VEG_BOUNDARY_COLOR, linewidth=1.0,
                   linestyle="-.", alpha=0.6)

    if np.isfinite(shell_line_dist):
        ax.axvline(shell_line_dist, color=SHELL_LINE_COLOR, linewidth=1.5,
                   linestyle="--")

    if waterline_dist is not None and np.isfinite(waterline_dist):
        ax.axvline(waterline_dist, color=WATERLINE_COLOR, linewidth=1.2,
                   linestyle=":")

    if manual_dist is not None and np.isfinite(manual_dist):
        ax.axvline(manual_dist, color=MANUAL_COLOR, linewidth=1.5,
                   linestyle="-", alpha=0.8)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)

    # Title.
    title = f"T{transect_id}"
    if np.isfinite(shell_line_dist):
        title += f"  |  SL={shell_line_dist:.1f} m  conf={confidence:.2f}"
        if manual_dist is not None and np.isfinite(manual_dist):
            offset = shell_line_dist - manual_dist
            title += f"  \u0394={offset:+.1f}"
    else:
        title += "  |  no detection"
    ax.set_title(title, fontsize=fontsize_title, fontweight="medium")
    ax.tick_params(labelsize=fontsize_tick)

    if not compact:
        ax.set_xlabel("Distance along transect (m)", fontsize=10)
        ax.set_ylabel("NDWI", fontsize=10)


def plot_overview_grid(
    merged_df, model, sample_ids, output_path,
    band_mode="unknown", ncols=4, nrows=5, manual_df=None,
):
    """Produce a multi-panel overview grid."""
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.5, nrows * 2.8),
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    for idx, tid in enumerate(sample_ids):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        grp = merged_df[merged_df["transect_id"] == tid].sort_values("distance")
        valid = grp[FEATURE_COLS].dropna()

        if len(valid) < 10:
            ax.text(0.5, 0.5, "Too few points", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="gray")
            ax.set_title(f"T{tid}", fontsize=9)
            continue

        distances = grp.loc[valid.index, "distance"].values
        obs = valid.values
        ndwi_arr = valid["ndwi"].values

        result = detect_shell_line_hmm(model, obs, distances, band_mode)

        m_dist = None
        if manual_df is not None and tid in manual_df["transect_id"].values:
            m_dist = float(
                manual_df.loc[manual_df["transect_id"] == tid, "distance"].iloc[0]
            )

        _plot_transect_hmm(
            ax, distances, ndwi_arr,
            result["states"], result["distance"], result["confidence"],
            transect_id=tid, compact=True,
            manual_dist=m_dist,
            waterline_dist=result.get("waterline_distance"),
            veg_boundary_dist=result.get("veg_boundary_distance"),
        )

    for idx in range(len(sample_ids), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Legend.
    handles = [
        mpatches.Patch(color=STATE_COLORS[VEGETATION], alpha=0.35, label="Vegetation"),
        mpatches.Patch(color=STATE_COLORS[DRY_BEACH], alpha=0.35, label="Dry beach"),
        mpatches.Patch(color=STATE_COLORS[WET_BEACH], alpha=0.35, label="Wet beach"),
        mpatches.Patch(color=STATE_COLORS[WATER], alpha=0.35, label="Water"),
        plt.Line2D([0], [0], color=VEG_BOUNDARY_COLOR, linewidth=1.0,
                   linestyle="-.", label="Veg boundary"),
        plt.Line2D([0], [0], color=SHELL_LINE_COLOR, linewidth=1.5,
                   linestyle="--", label="Shell line (HMM)"),
        plt.Line2D([0], [0], color=WATERLINE_COLOR, linewidth=1.2,
                   linestyle=":", label="Waterline (HMM)"),
        plt.Line2D([0], [0], color=MANUAL_COLOR, linewidth=1.5,
                   linestyle="-", label="Manual shell line"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=7,
               framealpha=0.9, ncol=2)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Overview grid saved: {output_path}")
    return output_path


def plot_individual_transect(
    merged_df, model, transect_id, output_path,
    band_mode="unknown", profiles_df=None, manual_df=None,
):
    """Produce a detailed single-transect diagnostic plot."""
    grp = merged_df[merged_df["transect_id"] == transect_id].sort_values("distance")
    valid = grp[FEATURE_COLS].dropna()
    distances = grp.loc[valid.index, "distance"].values
    obs = valid.values
    ndwi_arr = valid["ndwi"].values

    result = detect_shell_line_hmm(model, obs, distances, band_mode)

    m_dist = None
    if manual_df is not None and transect_id in manual_df["transect_id"].values:
        m_dist = float(
            manual_df.loc[manual_df["transect_id"] == transect_id, "distance"].iloc[0]
        )

    has_profiles = (
        profiles_df is not None
        and transect_id in profiles_df["transect_id"].values
    )

    if has_profiles:
        fig, (ax_bands, ax_ndwi) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True,
            gridspec_kw={"height_ratios": [1, 1.2]},
        )
        prof = profiles_df[
            profiles_df["transect_id"] == transect_id
        ].sort_values("distance")

        band_colors = {"red": "red", "green": "green", "blue": "blue", "nir": "#888"}
        for band, color in band_colors.items():
            if band in prof.columns and not prof[band].isna().all():
                ax_bands.plot(prof["distance"], prof[band],
                              color=color, linewidth=1, alpha=0.7, label=band.upper())

        for dist, color, ls in [
            (result["distance"], SHELL_LINE_COLOR, "--"),
            (result.get("waterline_distance"), WATERLINE_COLOR, ":"),
            (result.get("veg_boundary_distance"), VEG_BOUNDARY_COLOR, "-."),
            (m_dist, MANUAL_COLOR, "-"),
        ]:
            if dist is not None and np.isfinite(dist):
                ax_bands.axvline(dist, color=color, linewidth=1.3,
                                 linestyle=ls, alpha=0.6)

        ax_bands.set_ylabel("DN", fontsize=10)
        ax_bands.set_title(f"Transect {transect_id} -- raw bands", fontsize=11)
        ax_bands.legend(fontsize=8, loc="upper right")
    else:
        fig, ax_ndwi = plt.subplots(figsize=(12, 5))

    _plot_transect_hmm(
        ax_ndwi, distances, ndwi_arr,
        result["states"], result["distance"], result["confidence"],
        transect_id=transect_id, compact=False,
        manual_dist=m_dist,
        waterline_dist=result.get("waterline_distance"),
        veg_boundary_dist=result.get("veg_boundary_distance"),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Individual plot saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_fit_summary(model, results_df):
    """Print emission parameters and detection statistics."""
    print("\n" + "=" * 60)
    print("HMM FIT SUMMARY (4-state, 2D emissions)")
    print("=" * 60)

    print(f"\n  Emission parameters (per-year EM estimates):")
    print(f"    {'state':12s}   {'NDWI mean':>10s} {'NDWI std':>10s}"
          f"   {'NIR mean':>10s} {'NIR std':>10s}")
    for sid, label in STATE_LABELS.items():
        ndwi_mean = model.means_[sid, 0]
        nir_mean  = model.means_[sid, 1]
        ndwi_std  = float(np.sqrt(model.covars_[sid].flat[0]))
        nir_std   = float(np.sqrt(model.covars_[sid].flat[1]))
        print(f"    {label:12s}   {ndwi_mean:+10.4f} {ndwi_std:10.4f}"
              f"   {nir_mean:10.1f} {nir_std:10.1f}")

    print(f"\n  Transition matrix:")
    header = "                " + "".join(f"{STATE_LABELS[j]:>12s}" for j in range(N_STATES))
    print(header)
    for i in range(N_STATES):
        row = f"    {STATE_LABELS[i]:12s}"
        row += "".join(f"{model.transmat_[i, j]:12.3f}" for j in range(N_STATES))
        print(row)

    n_total = len(results_df)
    n_detected = results_df["shell_line_distance"].notna().sum()

    print(f"\n  Shell line detection:")
    print(f"    Detected: {n_detected}/{n_total} ({100*n_detected/n_total:.1f}%)")

    detected = results_df[results_df["shell_line_distance"].notna()]
    if len(detected) > 0:
        sl = detected["shell_line_distance"]
        print(f"    Mean: {sl.mean():.1f} m   Median: {sl.median():.1f} m   "
              f"Std: {sl.std():.1f} m   Range: [{sl.min():.1f}, {sl.max():.1f}]")
        c = detected["confidence"]
        print(f"    Confidence — Mean: {c.mean():.3f}   Median: {c.median():.3f}   "
              f"Range: [{c.min():.3f}, {c.max():.3f}]")

    n_wl = results_df["waterline_distance"].notna().sum()
    print(f"\n  Waterline detected: {n_wl}/{n_total}")

    n_veg = results_df["veg_boundary_distance"].notna().sum()
    print(f"  Veg boundary detected: {n_veg}/{n_total}")

    both = results_df.dropna(subset=["shell_line_distance", "waterline_distance"])
    if len(both) > 0:
        widths = both["waterline_distance"] - both["shell_line_distance"]
        print(f"\n  Wet beach zone width:")
        print(f"    Mean: {widths.mean():.1f} m   Median: {widths.median():.1f} m   "
              f"Std: {widths.std():.1f} m")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate 4-state 2D HMM shell line detection."
    )
    parser.add_argument("--year", type=str, default="2010")
    parser.add_argument("--features-path", type=Path, default=None)
    parser.add_argument("--profiles-path", type=Path, default=None)
    parser.add_argument("--manual-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-overview", type=int, default=20)
    parser.add_argument("--n-individual", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    year = args.year

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(name)s  %(message)s",
    )

    output_root = PROJECT_ROOT / "OUTPUT"

    features_path = args.features_path or output_root / year / f"features_{year}.parquet"
    profiles_path = args.profiles_path or output_root / year / f"profiles_{year}.parquet"
    output_dir = args.output_dir or output_root / year / "hmm_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load and merge -----------------------------------------------------
    print(f"\nLoading features: {features_path}")
    if not features_path.exists():
        print(f"  ERROR: {features_path} not found"); sys.exit(1)
    features_df = pd.read_parquet(features_path)

    print(f"Loading profiles: {profiles_path}")
    if not profiles_path.exists():
        print(f"  ERROR: {profiles_path} not found"); sys.exit(1)
    profiles_df = pd.read_parquet(profiles_path)

    print("Merging profiles + features...")
    merged_df = merge_profiles_and_features(
        profiles_df, features_df, columns=["ndwi", "nir_d1_smooth"]
    )
    all_ids = sorted(merged_df["transect_id"].unique())
    print(f"  {len(merged_df)} rows, {len(all_ids)} transects")

    # -- Load manual --------------------------------------------------------
    manual_path = args.manual_path or PROJECT_ROOT / "OUTPUT" / "manual_transitions.parquet"
    manual_df = None
    if manual_path.exists():
        manual_all = pd.read_parquet(manual_path)
        manual_all["year"] = manual_all["year"].str.replace(r"\.0$", "", regex=True)
        manual_df = manual_all[manual_all["year"] == year].copy()
        if manual_df.empty:
            print(f"  Manual lines found but none for year {year}.")
            manual_df = None
        else:
            print(f"  Loaded {len(manual_df)} manual shell lines for year {year}")
    else:
        print(f"  No manual transitions file -- skipping comparison.")

    # -- Fit HMM ------------------------------------------------------------
    print(f"\nFitting 4-state 2D HMM for year {year}...")
    model = fit_hmm_for_year(merged_df)

    # -- Run detection ------------------------------------------------------
    print("Running detection on all transects...")
    results_df = run_hmm_detection(model, merged_df)

    # -- Summary ------------------------------------------------------------
    print_fit_summary(model, results_df)

    if manual_df is not None:
        m = results_df.merge(
            manual_df[["transect_id", "distance"]].rename(
                columns={"distance": "manual_distance"}
            ),
            on="transect_id", how="inner",
        ).dropna(subset=["shell_line_distance"])
        if len(m) > 0:
            offsets = m["shell_line_distance"] - m["manual_distance"]
            print("  Offset vs manual (HMM - manual):")
            print(f"    N={len(offsets)}  Mean={offsets.mean():+.1f} m  "
                  f"Median={offsets.median():+.1f} m  Std={offsets.std():.1f} m  "
                  f"MAE={offsets.abs().mean():.1f} m")
            print(f"    Range: [{offsets.min():+.1f}, {offsets.max():+.1f}] m\n")

    # -- Sample transects ---------------------------------------------------
    rng = np.random.default_rng(args.seed)
    detected_ids = results_df.loc[
        results_df["shell_line_distance"].notna(), "transect_id"
    ].tolist()
    missed_ids = results_df.loc[
        results_df["shell_line_distance"].isna(), "transect_id"
    ].tolist()

    n_ov = min(args.n_overview, len(all_ids))
    n_miss = min(2, len(missed_ids), n_ov // 4)
    n_det  = n_ov - n_miss

    grid_ids = []
    if n_det > 0 and detected_ids:
        grid_ids += rng.choice(detected_ids, min(n_det, len(detected_ids)),
                               replace=False).tolist()
    if n_miss > 0 and missed_ids:
        grid_ids += rng.choice(missed_ids, min(n_miss, len(missed_ids)),
                               replace=False).tolist()
    rng.shuffle(grid_ids)

    n_ind = min(args.n_individual, len(detected_ids))
    ind_pool = [t for t in detected_ids if t not in grid_ids] or detected_ids
    individual_ids = rng.choice(ind_pool, min(n_ind, len(ind_pool)),
                                replace=False).tolist()

    # -- Plot ---------------------------------------------------------------
    print(f"\nPlotting overview grid ({len(grid_ids)} transects)...")
    plot_overview_grid(merged_df, model, grid_ids,
                       output_dir / "hmm_overview_grid.png",
                       manual_df=manual_df)

    for tid in individual_ids:
        print(f"Plotting transect {tid}...")
        plot_individual_transect(merged_df, model, tid,
                                 output_dir / f"hmm_transect_{tid}.png",
                                 profiles_df=profiles_df,
                                 manual_df=manual_df)

    results_df.to_csv(output_dir / "hmm_results.csv", index=False)
    print(f"\nAll outputs in: {output_dir}")
    print("Done.\n")


if __name__ == "__main__":
    main()