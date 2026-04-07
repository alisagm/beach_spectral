#!/usr/bin/env python3
"""
Diagnostic: classify transect sample points as vegetation / beach / water
using NDVI and NDWI zero crossings, then evaluate whether the "beach" zone
reliably brackets the manual shell line pick.

Classification rule (physically grounded, no tunable thresholds):
  vegetation : NDVI > 0          (NIR > Red → chlorophyll)
  water      : NDWI > 0 & ~veg   (Green > NIR → water absorption)
  beach      : NDVI ≤ 0 & NDWI ≤ 0  (bare substrate)

Outputs
-------
  zone_diagnostics.parquet — one row per (transect_id, year) with:
      manual_distance, beach_zone_start, beach_zone_end, beach_zone_width,
      pick_in_beach (bool), pick_offset_from_beach_edge (signed, + = inside),
      n_beach_fragments, band_mode

  zone_summary.png — three-panel figure:
      1. Histogram of pick_offset_from_beach_edge (does the beach zone bracket the pick?)
      2. Beach zone width distribution (is the zone wide enough for Dynp pass 2?)
      3. Per-year success rate (% of picks inside beach zone)

  zone_profiles_{N}.png — N example transect profiles with zone shading and
      manual pick marker, for visual sanity checking.

Usage
-----
  python tools/zone_diagnostic.py \\
      --manual-transitions  OUTPUT/manual_transitions.parquet \\
      --output-root         OUTPUT/ \\
      --band-config         INPUT/band_config.json \\
      --output-dir          OUTPUT/plots/zones
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)

_FORMAT_TO_BAND_MODE = {"CIR": "cir", "RGBN": "4band", "RGB": "rgb"}

ZONE_COLOURS = {
    "vegetation": "#2d6a2d",
    "beach":      "#d4a843",
    "water":      "#2166ac",
    "unclassified": "#cccccc",
}


# ── Classification ────────────────────────────────────────────────────────────

def _smooth_uniform(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Uniform (boxcar) smoothing with NaN-aware handling.

    Uses a centred window.  Edge values are computed with a truncated window
    (no padding artifacts).  NaN positions in the input produce NaN in the
    output only if the entire window is NaN.

    Physical justification: geomorphic zones (vegetation, beach, surf) are
    spatially coherent at scales of tens of metres.  Smoothing at that scale
    suppresses pixel-level noise while preserving real zone transitions.
    The window size should approximate the minimum plausible zone width.

    Args:
        arr:    1-D array of index values (may contain NaN).
        window: Window width in samples (odd recommended; even is rounded up).

    Returns:
        Smoothed array, same length as input.
    """
    if window <= 1:
        return arr.copy()
    from scipy.ndimage import uniform_filter1d
    # uniform_filter1d doesn't handle NaN — use nanmean via pandas
    s = pd.Series(arr)
    return s.rolling(window, center=True, min_periods=1).mean().values


def classify_points(feat: pd.DataFrame, smooth_window: int = 0) -> np.ndarray:
    """
    Classify each sample point using NDVI/NDWI zero crossings.

    Args:
        feat:           Feature DataFrame for one transect, sorted by distance.
        smooth_window:  If > 1, smooth NDVI and NDWI with a uniform window
                        of this width (in samples, ≈ metres at 1m spacing)
                        before applying zero-crossing classification.
                        0 or 1 = no smoothing.

    Returns array of zone labels, same length as feat.
    Points missing NDVI or NDWI get 'unclassified'.
    """
    n = len(feat)
    zones = np.full(n, "unclassified", dtype=object)

    has_ndvi = "ndvi" in feat.columns
    has_ndwi = "ndwi" in feat.columns

    if not has_ndvi or not has_ndwi:
        logger.debug("Missing ndvi or ndwi — all points unclassified")
        return zones

    ndvi = feat["ndvi"].values.copy()
    ndwi = feat["ndwi"].values.copy()

    if smooth_window > 1:
        ndvi = _smooth_uniform(ndvi, smooth_window)
        ndwi = _smooth_uniform(ndwi, smooth_window)

    veg_mask   = ndvi > 0
    water_mask = (ndwi > 0) & ~veg_mask
    beach_mask = (ndvi <= 0) & (ndwi <= 0)

    zones[veg_mask]   = "vegetation"
    zones[water_mask]  = "water"
    zones[beach_mask]  = "beach"

    # NaN positions stay unclassified
    nan_mask = np.isnan(ndvi) | np.isnan(ndwi)
    zones[nan_mask] = "unclassified"

    return zones


def find_largest_beach_zone(distances: np.ndarray, zones: np.ndarray):
    """
    Find the largest contiguous run of 'beach' points.

    Returns (start_dist, end_dist, n_points) for the largest fragment,
    and n_fragments (total number of contiguous beach runs).
    Returns (NaN, NaN, 0, 0) if no beach points exist.
    """
    is_beach = (zones == "beach")

    if not is_beach.any():
        return np.nan, np.nan, 0, 0

    # Find contiguous runs
    diffs = np.diff(is_beach.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends   = np.where(diffs == -1)[0] + 1

    # Handle edge cases: beach starts at index 0 or ends at last index
    if is_beach[0]:
        starts = np.insert(starts, 0, 0)
    if is_beach[-1]:
        ends = np.append(ends, len(is_beach))

    n_fragments = len(starts)

    # Find longest run
    lengths    = ends - starts
    best       = np.argmax(lengths)
    best_start = starts[best]
    best_end   = ends[best] - 1  # inclusive index

    return (
        float(distances[best_start]),
        float(distances[best_end]),
        int(lengths[best]),
        n_fragments,
    )


# ── Per-transect analysis ─────────────────────────────────────────────────────

def analyse_transect(
    feat: pd.DataFrame,
    manual_distance: float,
    smooth_window: int = 0,
) -> dict:
    """Classify one transect and compute zone diagnostics relative to manual pick."""
    feat = feat.sort_values("distance").reset_index(drop=True)
    distances = feat["distance"].values
    zones     = classify_points(feat, smooth_window=smooth_window)

    beach_start, beach_end, beach_n, n_frags = find_largest_beach_zone(distances, zones)

    # Is the manual pick inside the largest beach zone?
    if np.isnan(beach_start):
        pick_in_beach = False
        pick_offset   = np.nan
    else:
        pick_in_beach = beach_start <= manual_distance <= beach_end
        # Signed offset: positive = pick is inside zone (distance from nearest edge)
        # Negative = pick is outside zone
        if pick_in_beach:
            pick_offset = min(manual_distance - beach_start, beach_end - manual_distance)
        else:
            # How far outside? Negative = outside
            if manual_distance < beach_start:
                pick_offset = manual_distance - beach_start  # negative (landward)
            else:
                pick_offset = manual_distance - beach_end    # positive (seaward)

    beach_width = beach_end - beach_start if not np.isnan(beach_start) else np.nan

    # Zone fractions
    n_total = len(zones)
    zone_counts = {z: int((zones == z).sum()) for z in ["vegetation", "beach", "water", "unclassified"]}

    return {
        "manual_distance":     manual_distance,
        "beach_zone_start":    beach_start,
        "beach_zone_end":      beach_end,
        "beach_zone_width":    beach_width,
        "beach_zone_n_points": beach_n,
        "pick_in_beach":       pick_in_beach,
        "pick_offset":         pick_offset,
        "n_beach_fragments":   n_frags,
        "n_veg":               zone_counts["vegetation"],
        "n_beach":             zone_counts["beach"],
        "n_water":             zone_counts["water"],
        "n_unclassified":      zone_counts["unclassified"],
        "n_total":             n_total,
    }


# ── Example profile plots ────────────────────────────────────────────────────

def plot_example_profiles(
    manual: pd.DataFrame,
    features_by_year: dict[str, pd.DataFrame],
    band_config: dict,
    output_dir: Path,
    smooth_windows: list[int],
    n_examples: int = 8,
    seed: int = 42,
):
    """
    Plot example transect profiles with zone shading for each smoothing window.
    Each row is one transect; each column is one smoothing window.
    """
    rng = np.random.default_rng(seed)

    # Stratified sample: half CIR, half 4band
    candidates = []
    for _, row in manual.iterrows():
        year = str(row["year"])
        fmt  = band_config.get(year, {}).get("format", "").upper()
        bm   = _FORMAT_TO_BAND_MODE.get(fmt)
        if bm in ("cir", "4band"):
            candidates.append((int(row["transect_id"]), year, bm,
                               float(row["distance"])))

    if not candidates:
        logger.warning("No NIR candidates for example profiles.")
        return

    cir_pool   = [c for c in candidates if c[2] == "cir"]
    fband_pool = [c for c in candidates if c[2] == "4band"]

    n_each = n_examples // 2
    samples = []
    if cir_pool:
        idx = rng.choice(len(cir_pool), size=min(n_each, len(cir_pool)), replace=False)
        samples.extend([cir_pool[i] for i in idx])
    if fband_pool:
        idx = rng.choice(len(fband_pool), size=min(n_each, len(fband_pool)), replace=False)
        samples.extend([fband_pool[i] for i in idx])

    n_rows = len(samples)
    n_cols = len(smooth_windows)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.0 * n_cols, 3.0 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        "Zone classification — smoothing comparison\n"
        "green = vegetation   |   yellow = beach   |   blue = water   |   "
        "red line = manual pick",
        fontsize=12, fontweight="bold", y=1.01,
    )

    # Column headers
    for j, sw in enumerate(smooth_windows):
        label = "raw (no smoothing)" if sw <= 1 else f"smooth = {sw}m"
        axes[0, j].set_title(f"{label}", fontsize=10, fontweight="bold")

    for i, (tid, year, bm, man_dist) in enumerate(samples):
        feat = features_by_year[year]
        t_feat = feat[feat["transect_id"] == tid].sort_values("distance")
        if t_feat.empty:
            for j in range(n_cols):
                axes[i, j].set_visible(False)
            continue

        distances = t_feat["distance"].values

        for j, sw in enumerate(smooth_windows):
            ax = axes[i, j]
            zones = classify_points(t_feat, smooth_window=sw)

            # Zone shading
            for zone_name, colour in ZONE_COLOURS.items():
                mask = (zones == zone_name)
                if not mask.any():
                    continue
                for start, end in _contiguous_runs(mask):
                    ax.axvspan(
                        distances[start], distances[min(end, len(distances) - 1)],
                        alpha=0.22, color=colour, linewidth=0,
                    )

            # Plot smoothed indices (or raw if sw <= 1)
            ndwi_raw = t_feat["ndwi"].values if "ndwi" in t_feat.columns else None
            ndvi_raw = t_feat["ndvi"].values if "ndvi" in t_feat.columns else None

            if ndwi_raw is not None:
                ndwi_plot = _smooth_uniform(ndwi_raw, sw) if sw > 1 else ndwi_raw
                ax.plot(distances, ndwi_plot, color="#2166ac",
                        linewidth=1.0, label="NDWI", alpha=0.8)
            if ndvi_raw is not None:
                ndvi_plot = _smooth_uniform(ndvi_raw, sw) if sw > 1 else ndvi_raw
                ax.plot(distances, ndvi_plot, color="#2d6a2d",
                        linewidth=1.0, label="NDVI", alpha=0.8)

            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

            # Manual pick
            ax.axvline(man_dist, color="#d62728", linewidth=1.8, linestyle="-",
                       alpha=0.9)

            # Beach zone bracket
            bz_start, bz_end, _, n_frags = find_largest_beach_zone(distances, zones)
            if not np.isnan(bz_start):
                ax.axvline(bz_start, color="#d4a843", linewidth=1.2,
                           linestyle=":", alpha=0.7)
                ax.axvline(bz_end, color="#d4a843", linewidth=1.2,
                           linestyle=":", alpha=0.7)

            ax.set_xlim(distances[0], distances[-1])
            ax.set_ylim(-0.7, 0.7)
            ax.grid(True, alpha=0.2)

            # Row label on leftmost column
            if j == 0:
                ax.set_ylabel(f"T{tid}\n({year} {bm})", fontsize=9)

            # Only show legend on first row
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    out_path = output_dir / "zone_profiles.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out_path)


def _contiguous_runs(mask: np.ndarray):
    """Yield (start, end) index pairs for contiguous True runs."""
    diffs  = np.diff(mask.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends   = np.where(diffs == -1)[0] + 1
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask))
    for s, e in zip(starts, ends):
        yield s, e - 1


# ── Summary figure ────────────────────────────────────────────────────────────

def plot_summary(all_diags: dict[int, pd.DataFrame], output_dir: Path):
    """
    Comparison summary across smoothing windows (NIR years only).

    all_diags: {smooth_window: diag_df}

    Layout: one row of panels per metric, one column per smoothing window.
    Row 1: pick offset histograms
    Row 2: beach zone width
    Row 3: per-year success rate
    """
    windows = sorted(all_diags.keys())
    n_cols  = len(windows)

    fig, axes = plt.subplots(3, n_cols, figsize=(5.5 * n_cols, 12))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        "Zone diagnostic — smoothing comparison (NIR years)\n"
        "Classification: smoothed NDVI > 0 → veg, smoothed NDWI > 0 → water, "
        "both ≤ 0 → beach",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for j, sw in enumerate(windows):
        diag = all_diags[sw]
        nir  = diag[diag["band_mode"].isin(["cir", "4band"])].copy()

        label = "raw" if sw <= 1 else f"smooth={sw}m"
        pct_in = 100 * nir["pick_in_beach"].mean() if len(nir) > 0 else 0

        # ── Row 1: pick offset ────────────────────────────────────────────
        ax1 = axes[0, j]
        inside  = nir[nir["pick_in_beach"]]["pick_offset"].dropna()
        outside = nir[~nir["pick_in_beach"]]["pick_offset"].dropna()

        bins = np.linspace(-60, 100, 65)
        if len(inside) > 0:
            ax1.hist(inside.values, bins=bins, color="#2d6a2d", alpha=0.5,
                     label=f"inside (n={len(inside):,})", density=True)
        if len(outside) > 0:
            ax1.hist(outside.values, bins=bins, color="#d62728", alpha=0.5,
                     label=f"outside (n={len(outside):,})", density=True)

        ax1.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax1.set_title(f"{label} — {pct_in:.1f}% inside", fontsize=11,
                      fontweight="bold")
        ax1.set_xlabel("Offset from beach edge (m)", fontsize=9)
        ax1.set_ylabel("Density", fontsize=9)
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.2)

        # ── Row 2: beach width ────────────────────────────────────────────
        ax2 = axes[1, j]
        widths = nir["beach_zone_width"].dropna()
        ax2.hist(widths.values, bins=60, color="#d4a843", alpha=0.6,
                 edgecolor="none", range=(0, 200))
        med_w = widths.median()
        ax2.axvline(med_w, color="black", linewidth=1.5, linestyle="--",
                    label=f"median = {med_w:.0f}m")
        ax2.axvline(20, color="red", linewidth=1, linestyle=":",
                    label="min for Dynp (20m)")
        ax2.set_xlabel("Beach zone width (m)", fontsize=9)
        ax2.set_ylabel("Count", fontsize=9)
        ax2.set_title(f"Beach zone width — {label}", fontsize=10)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.2)

        # ── Row 3: per-year success rate ──────────────────────────────────
        ax3 = axes[2, j]
        yearly = (
            nir.groupby("year")["pick_in_beach"]
            .agg(["mean", "count"])
            .reset_index()
        )
        yearly["year_int"] = yearly["year"].astype(int)
        yearly = yearly.sort_values("year_int")

        ax3.bar(range(len(yearly)), yearly["mean"] * 100,
                color="#2166ac", alpha=0.7, edgecolor="none")
        ax3.set_xticks(range(len(yearly)))
        ax3.set_xticklabels(yearly["year"].values, fontsize=9)
        ax3.set_ylabel("% picks inside", fontsize=9)
        ax3.set_title(f"Per-year — {label}", fontsize=10)
        ax3.set_ylim(0, 105)
        ax3.grid(True, alpha=0.2, axis="y")
        for i, (_, row) in enumerate(yearly.iterrows()):
            ax3.text(i, row["mean"] * 100 + 2, f"n={int(row['count'])}",
                     ha="center", fontsize=7, color="#555555")

    fig.tight_layout()
    out_path = output_dir / "zone_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manual-transitions", "-m", type=Path,
                        default=Path("OUTPUT/manual_transitions.parquet"))
    parser.add_argument("--output-root", "-r", type=Path,
                        default=Path("OUTPUT"))
    parser.add_argument("--band-config", "-b", type=Path,
                        default=Path("INPUT/band_config.json"))
    parser.add_argument("--output-dir", "-o", type=Path,
                        default=Path("OUTPUT/plots/zones"))
    parser.add_argument(
        "--smooth", type=int, nargs="+", default=[0, 11, 21],
        metavar="W",
        help=(
            "Smoothing window widths in samples (≈ metres at 1m spacing). "
            "0 or 1 = no smoothing.  Odd values recommended.  "
            "Default: 0 11 21 (raw, ~10m, ~20m)."
        ),
    )
    parser.add_argument("--n-examples", type=int, default=8)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load ──────────────────────────────────────────────────────────────
    manual = pd.read_parquet(args.manual_transitions)
    manual["year"] = manual["year"].astype(float).astype(int).astype(str)
    years = sorted(manual["year"].unique())

    with open(args.band_config) as f:
        band_config = json.load(f)

    features_by_year = {}
    for y in years:
        path = args.output_root / y / f"features_{y}.parquet"
        if not path.exists():
            logger.warning("Features not found for %s", y)
            continue
        features_by_year[y] = pd.read_parquet(path)
        logger.info("  Loaded features %s: %d rows", y, len(features_by_year[y]))

    # ── Analyse each pick at each smoothing window ─────────────────────────
    all_diags = {}  # {smooth_window: DataFrame}

    for sw in args.smooth:
        records = []
        for _, row in manual.iterrows():
            tid  = int(row["transect_id"])
            year = str(row["year"])
            manual_dist = float(row["distance"])

            fmt = band_config.get(year, {}).get("format", "").upper()
            bm  = _FORMAT_TO_BAND_MODE.get(fmt, "unknown")

            feat = features_by_year.get(year)
            if feat is None:
                continue
            t_feat = feat[feat["transect_id"] == tid].sort_values("distance")
            if t_feat.empty:
                continue

            stats = analyse_transect(t_feat, manual_dist, smooth_window=sw)
            stats["transect_id"]   = tid
            stats["year"]          = year
            stats["band_mode"]     = bm
            stats["smooth_window"] = sw
            records.append(stats)

        diag = pd.DataFrame(records)
        all_diags[sw] = diag
        logger.info("smooth=%d: analysed %d picks", sw, len(diag))

        # Print per-window summary
        nir = diag[diag["band_mode"].isin(["cir", "4band"])]
        pct_in    = 100 * nir["pick_in_beach"].mean() if len(nir) > 0 else 0
        med_width = nir["beach_zone_width"].median()  if len(nir) > 0 else 0
        med_frags = nir["n_beach_fragments"].median()  if len(nir) > 0 else 0

        label = "raw" if sw <= 1 else f"smooth={sw}m"
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  ZONE DIAGNOSTIC — {label} (NIR years)")
        print(sep)
        print(f"  Picks analysed:        {len(nir):,}")
        print(f"  Picks inside beach:    {pct_in:.1f}%")
        print(f"  Median beach width:    {med_width:.0f} m")
        print(f"  Median fragments:      {med_frags:.0f}")
        if len(nir) > 0:
            outside = nir[~nir["pick_in_beach"]]
            if len(outside) > 0:
                print(f"\n  Outside picks (n={len(outside)}):")
                print(f"    Median offset:       {outside['pick_offset'].median():+.1f} m")
                print(f"    Landward (offset<0): {(outside['pick_offset'] < 0).sum()}")
                print(f"    Seaward  (offset>0): {(outside['pick_offset'] > 0).sum()}")
        print(sep)

    # ── Save and plot ─────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save combined parquet with all windows
    combined = pd.concat(all_diags.values(), ignore_index=True)
    parquet_path = args.output_dir / "zone_diagnostics.parquet"
    combined.to_parquet(parquet_path, index=False)
    logger.info("Saved %d rows → %s", len(combined), parquet_path)

    plot_summary(all_diags, args.output_dir)
    plot_example_profiles(
        manual, features_by_year, band_config, args.output_dir,
        smooth_windows=args.smooth,
        n_examples=args.n_examples,
    )

    print(f"\nOutputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()