#!/usr/bin/env python3
"""
tools/dynp_experiment.py — Benchmark Dynp configurations against manual picks.

Runs a matrix of (n_bkps, model, selection_rule) configurations on every
transect that has a manual pick, then computes signed error against truth.

The old-detector baseline is loaded from pre-existing transitions parquets
rather than re-run.

Experiment matrix
-----------------
  n_bkps    : [2, 3]
  model     : ["rbf", "rank"]
  selection : ["max_contrast", "landward", "middle"]

  → 12 Dynp configurations + 1 old-detector baseline = 13 labels per transect.

  "max_contrast" is the current pelt.py selection rule (pick the breakpoint
  with the largest positive seaward NDWI contrast).

  "landward" picks the most-landward breakpoint that still has positive
  seaward contrast — the hypothesis being that PELT's seaward bias comes
  from selecting the water edge rather than the shell line.

  "middle" sorts positive-contrast breaks by distance and takes the one
  at the median index.  For n_bkps=3 with all three breaks positive,
  this is the true interior break (the shell line, if the three-zone
  model holds: veg/beach, shell line, water edge).  For n_bkps=2 this
  degrades to the seaward break — the interesting case is n_bkps=3.

Selection rule details
----------------------
  max_contrast : identical to current pelt.py step 5 — breakpoint whose
                 right-segment mean exceeds left-segment mean by the largest
                 margin.  This is the rule that produces the +25 m seaward bias.

  landward     : among breakpoints with *positive* seaward contrast, take the
                 one closest to distance=0 (most landward).  Falls back to
                 max_contrast if no breakpoint has positive contrast, so the
                 failure mode is the same as today rather than silently worse.

Output
------
  dynp_experiment.parquet — long format, one row per (transect, year, config)

  Columns:
    transect_id       int
    year              str
    band_mode         str
    detector          str     e.g. "dynp_2bkps_rank_landward", "old_threshold"
    detected_distance float   metres from landward end (NaN if no detection)
    manual_distance   float   manual pick position
    signed_error      float   detected - manual (positive = seaward of truth)
    confidence        float   detector confidence (NaN for old if unavailable)
    n_bkps            int     (NaN for old)
    model             str     (NaN for old)
    selection         str     (NaN for old)

Usage
-----
  python tools/dynp_experiment.py \\
      --manual-transitions OUTPUT/manual_transitions.parquet \\
      --output-root        OUTPUT/ \\
      --transitions-dir    OUTPUT/ \\
      --band-config        INPUT/band_config.json \\
      --output             OUTPUT/dynp_experiment.parquet
"""

import argparse
import json
import logging
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Experiment matrix ─────────────────────────────────────────────────────────
N_BKPS_VALUES   = [2, 3]
MODEL_VALUES     = ["rbf", "rank"]
SELECTION_VALUES = ["max_contrast", "landward", "middle"]

_FORMAT_TO_BAND_MODE = {
    "CIR":  "cir",
    "RGBN": "4band",
    "RGB":  "rgb",
}


# ── Detection wrapper ─────────────────────────────────────────────────────────
# This duplicates the core of pelt.py steps 1–4 but returns ALL breakpoint
# positions so the caller can apply different selection rules.  We do NOT
# modify pelt.py itself — that stays stable until a winner is chosen.

def _detect_breakpoints(
    features: pd.DataFrame,
    band_mode: str,
    model: str,
    n_bkps: int,
    min_size: int = 20,
    smooth_window: int = 0,
) -> dict | None:
    """
    Run Dynp and return raw breakpoint info without applying a selection rule.

    Args:
        smooth_window: If > 1, smooth ndwi_proxy and variability with a
                       centred rolling mean before z-scoring.  Smoothing is
                       applied AFTER NaN masking on the valid-only arrays.

    Returns
    -------
    dict with keys:
        bkps            : list[int]   — breakpoint indices into valid-only arrays
        ndwi_valid      : np.ndarray  — moisture proxy values (NaN-masked, smoothed if requested)
        var_valid        : np.ndarray  — variability (NaN-masked, smoothed if requested)
        dist_valid      : np.ndarray  — distances (NaN-masked)
        original_indices: np.ndarray  — mapping back to full DataFrame rows
        n_valid         : int
        partial_coverage: bool
        ndwi_range      : float       — full-transect proxy range (for confidence)
        is_rgb          : bool
    Returns None on hard skip.
    """
    import ruptures  # noqa: PLC0415

    is_rgb = band_mode == "rgb"

    if is_rgb:
        if "grvi" not in features.columns:
            return None
        ndwi_proxy = -features["grvi"].to_numpy(dtype=float)
    else:
        if "ndwi" not in features.columns:
            return None
        ndwi_proxy = features["ndwi"].to_numpy(dtype=float)

    variability = features["variability"].to_numpy(dtype=float)
    distances   = features["distance"].to_numpy(dtype=float)

    valid_mask = ~(np.isnan(ndwi_proxy) | np.isnan(variability))
    n_total    = len(valid_mask)
    n_valid    = int(valid_mask.sum())

    if n_valid < 3 * min_size:
        return None

    partial_coverage = (n_valid / n_total) < 0.75 if n_total > 0 else True

    original_indices = np.where(valid_mask)[0]
    ndwi_valid       = ndwi_proxy[valid_mask]
    var_valid        = variability[valid_mask]
    dist_valid       = distances[valid_mask]

    # Optional smoothing — applied on valid-only arrays before z-scoring.
    if smooth_window > 1:
        import pandas as _pd  # noqa: PLC0415
        ndwi_valid = _pd.Series(ndwi_valid).rolling(smooth_window, center=True, min_periods=1).mean().values
        var_valid  = _pd.Series(var_valid).rolling(smooth_window, center=True, min_periods=1).mean().values

    def _zscore(arr):
        s = float(arr.std())
        if s == 0.0:
            return np.zeros_like(arr)
        return (arr - arr.mean()) / s

    signal = np.column_stack([_zscore(ndwi_valid), _zscore(var_valid)])

    try:
        raw_bkps = (
            ruptures.Dynp(model=model, min_size=min_size)
            .fit(signal)
            .predict(n_bkps=n_bkps)
        )
    except Exception as exc:
        logger.debug("Dynp(model=%s, n_bkps=%d) failed: %s", model, n_bkps, exc)
        return None

    bkps = [b for b in raw_bkps if b < n_valid]
    if not bkps:
        return None

    ndwi_range = float(np.nanmax(ndwi_proxy) - np.nanmin(ndwi_proxy))

    return {
        "bkps":             bkps,
        "ndwi_valid":       ndwi_valid,
        "var_valid":        var_valid,
        "dist_valid":       dist_valid,
        "original_indices": original_indices,
        "n_valid":          n_valid,
        "partial_coverage": partial_coverage,
        "ndwi_range":       ndwi_range,
        "is_rgb":           is_rgb,
    }


def _apply_selection(
    info: dict,
    selection: str,
) -> dict | None:
    """
    Apply a selection rule to raw breakpoints and return a result dict.

    Returns dict with 'distance', 'confidence', 'index', or None if no
    breakpoint satisfies the selection criteria.
    """
    bkps       = info["bkps"]
    ndwi_valid = info["ndwi_valid"]
    dist_valid = info["dist_valid"]
    n_valid    = info["n_valid"]

    # Build segment boundaries: [0, bp0, bp1, ..., n_valid]
    boundaries = [0] + bkps + [n_valid]

    # Compute contrast for each breakpoint
    bp_info = []
    for i, bp in enumerate(bkps):
        left_mean  = ndwi_valid[boundaries[i] : bp].mean()
        right_mean = ndwi_valid[bp : boundaries[i + 2]].mean()
        contrast   = right_mean - left_mean
        bp_info.append({
            "bp_idx":   bp,
            "distance": float(dist_valid[min(bp, n_valid - 1)]),
            "contrast": contrast,
        })

    # Filter to positive-contrast breakpoints (water is wetter than sand)
    positive = [b for b in bp_info if b["contrast"] > 0.0]

    if not positive:
        return None

    if selection == "max_contrast":
        chosen = max(positive, key=lambda b: b["contrast"])
    elif selection == "landward":
        chosen = min(positive, key=lambda b: b["distance"])
    elif selection == "middle":
        # Sort positive-contrast breaks by distance; take the one nearest
        # the median position.  For 3+ breaks this picks the interior one.
        # For 2 breaks this picks the seaward one (index 1 of 2) — expected
        # to behave like max_contrast there; the interesting case is n_bkps=3.
        # For 1 break it's the only option (same as landward/max_contrast).
        by_dist = sorted(positive, key=lambda b: b["distance"])
        mid_idx = len(by_dist) // 2
        # For even count, // 2 gives the seaward of the central pair.
        # For n_bkps=3 with all 3 positive, // 2 = 1 → true middle.
        chosen = by_dist[mid_idx]
    else:
        raise ValueError(f"Unknown selection rule: {selection!r}")

    # Confidence
    ndwi_range = info["ndwi_range"]
    confidence = float(np.clip(chosen["contrast"] / ndwi_range, 0.0, 1.0)) if ndwi_range > 0 else 0.0
    if info["is_rgb"]:
        confidence *= 0.7

    original_idx = int(info["original_indices"][chosen["bp_idx"]])

    return {
        "distance":   chosen["distance"],
        "confidence": confidence,
        "index":      original_idx,
    }


def _detect_recursive(
    band_mode: str,
    model: str,
    bp_cache: dict,
    min_size_pass2: int = 10,
) -> dict | None:
    """
    Two-pass recursive detection using cached pass-1 breakpoints.

    Pass 1: reuses bp_cache[(2, model)] from the main experiment loop.
    Pass 2: Dynp(n_bkps=1) on the signal between the two pass-1 breaks,
            re-z-scored within the constrained window.

    Returns dict with 'distance', 'confidence', 'index', or None.
    """
    import ruptures  # noqa: PLC0415

    # Retrieve pass-1 results from cache
    info = bp_cache.get((2, model))
    if info is None:
        return None

    bkps = info["bkps"]
    if len(bkps) < 2:
        return None

    ndwi_valid       = info["ndwi_valid"]
    var_valid        = info["var_valid"]
    dist_valid       = info["dist_valid"]
    original_indices = info["original_indices"]
    n_valid          = info["n_valid"]

    bp_lo, bp_hi = sorted(bkps[:2])

    # Check window is wide enough for pass 2
    window_len = bp_hi - bp_lo
    if window_len < 2 * min_size_pass2:
        return None

    # Slice to constrained window and re-z-score
    ndwi_window = ndwi_valid[bp_lo:bp_hi]
    var_window  = var_valid[bp_lo:bp_hi]
    dist_window = dist_valid[bp_lo:bp_hi]

    def _zscore(arr):
        s = float(arr.std())
        if s == 0.0:
            return np.zeros_like(arr)
        return (arr - arr.mean()) / s

    signal_p2 = np.column_stack([_zscore(ndwi_window), _zscore(var_window)])

    # Fit pass 2
    try:
        raw_bkps = (
            ruptures.Dynp(model=model, min_size=min_size_pass2)
            .fit(signal_p2)
            .predict(n_bkps=1)
        )
    except Exception:
        return None

    bkps_p2 = [b for b in raw_bkps if b < window_len]
    if not bkps_p2:
        return None

    # Map back to full valid-array index
    bp_in_window = bkps_p2[0]
    bp_in_valid  = bp_lo + bp_in_window

    # Contrast: computed on original NDWI within the pass-1 window
    left_mean  = ndwi_valid[bp_lo:bp_in_valid].mean()
    right_mean = ndwi_valid[bp_in_valid:bp_hi].mean()
    contrast   = right_mean - left_mean

    # Confidence
    ndwi_range = info["ndwi_range"]
    confidence = float(np.clip(contrast / ndwi_range, 0.0, 1.0)) if (ndwi_range > 0 and contrast > 0) else 0.0
    if info["is_rgb"]:
        confidence *= 0.7

    bp_distance  = float(dist_valid[min(bp_in_valid, n_valid - 1)])
    original_idx = int(original_indices[bp_in_valid])

    return {
        "distance":   bp_distance,
        "confidence": confidence,
        "index":      original_idx,
    }


# ── Old detector loader (from compare_detectors.py) ──────────────────────────

def _load_old_shell_lines(transitions_path: Path) -> dict:
    """Load old detector picks from transitions_{year}.parquet."""
    df = pd.read_parquet(transitions_path)
    dry_wet = df[df["type"].str.contains("dry_wet", na=False)].copy()
    if dry_wet.empty:
        return {}
    result = {}
    for tid, grp in dry_wet.groupby("transect_id"):
        if "guaranteed_shell_line" in grp.columns:
            guaranteed = grp[grp["guaranteed_shell_line"].astype(bool)]
        else:
            guaranteed = pd.DataFrame()
        pool = guaranteed if not guaranteed.empty else grp
        best = pool.loc[pool["confidence"].idxmax()]
        result[tid] = {
            "distance":   float(best["distance"]),
            "confidence": float(best["confidence"]),
        }
    return result


# ── Main experiment loop ──────────────────────────────────────────────────────

def collect_all_breakpoints(
    manual: pd.DataFrame,
    features_by_year: dict[str, pd.DataFrame],
    band_config: dict,
) -> pd.DataFrame:
    """
    For every manual pick, run each (n_bkps, model) config and record ALL
    breakpoint positions — not just the selected one.

    Returns one row per (transect, year, n_bkps, model, bp_rank) where
    bp_rank is 0-indexed landward-to-seaward.

    Columns:
        transect_id, year, band_mode, manual_distance,
        n_bkps, model, bp_rank, bp_distance, bp_offset,
        bp_contrast, n_breakpoints_returned
    """
    combos = list(product(N_BKPS_VALUES, MODEL_VALUES))
    records = []

    for _, pick_row in manual.iterrows():
        tid  = int(pick_row["transect_id"])
        year = str(pick_row["year"])
        manual_dist = float(pick_row["distance"])

        fmt = band_config.get(year, {}).get("format", "").upper()
        band_mode = _FORMAT_TO_BAND_MODE.get(fmt, "unknown")

        features = features_by_year.get(year)
        if features is None:
            continue
        t_feat = features[features["transect_id"] == tid].sort_values("distance")
        if t_feat.empty:
            continue

        for n_bkps, model in combos:
            info = _detect_breakpoints(t_feat, band_mode, model, n_bkps)
            if info is None:
                continue

            bkps       = info["bkps"]
            ndwi_valid = info["ndwi_valid"]
            dist_valid = info["dist_valid"]
            n_valid    = info["n_valid"]
            boundaries = [0] + bkps + [n_valid]

            for rank, bp in enumerate(bkps):
                bp_dist    = float(dist_valid[min(bp, n_valid - 1)])
                left_mean  = ndwi_valid[boundaries[rank] : bp].mean()
                right_mean = ndwi_valid[bp : boundaries[rank + 2]].mean()

                records.append({
                    "transect_id":      tid,
                    "year":             year,
                    "band_mode":        band_mode,
                    "manual_distance":  manual_dist,
                    "n_bkps":           n_bkps,
                    "model":            model,
                    "bp_rank":          rank,
                    "bp_distance":      bp_dist,
                    "bp_offset":        bp_dist - manual_dist,
                    "bp_contrast":      float(right_mean - left_mean),
                    "n_breakpoints_returned": len(bkps),
                })

    return pd.DataFrame(records)


def run_experiment(
    manual: pd.DataFrame,
    features_by_year: dict[str, pd.DataFrame],
    old_shell_lines_by_year: dict[str, dict],
    band_config: dict,
) -> pd.DataFrame:
    """
    Run all Dynp configurations + old baseline for every manual pick.

    Returns long-format DataFrame: one row per (transect, year, detector).
    """
    configs = list(product(N_BKPS_VALUES, MODEL_VALUES, SELECTION_VALUES))
    logger.info(
        "Experiment: %d Dynp configs × %d picks + old baseline",
        len(configs), len(manual),
    )

    records = []

    for _, pick_row in manual.iterrows():
        tid  = int(pick_row["transect_id"])
        year = str(pick_row["year"])
        manual_dist = float(pick_row["distance"])

        fmt = band_config.get(year, {}).get("format", "").upper()
        band_mode = _FORMAT_TO_BAND_MODE.get(fmt, "unknown")

        features = features_by_year.get(year)
        if features is None:
            continue

        t_feat = features[features["transect_id"] == tid].sort_values("distance")
        if t_feat.empty:
            continue

        base = {
            "transect_id":     tid,
            "year":            year,
            "band_mode":       band_mode,
            "manual_distance": manual_dist,
        }

        # ── Old detector baseline ─────────────────────────────────────────
        old_lines = old_shell_lines_by_year.get(year, {})
        old_entry = old_lines.get(tid)
        records.append({
            **base,
            "detector":          "old_threshold",
            "detected_distance": old_entry["distance"]   if old_entry else np.nan,
            "confidence":        old_entry["confidence"] if old_entry else np.nan,
            "signed_error":      (old_entry["distance"] - manual_dist) if old_entry else np.nan,
            "n_bkps":            np.nan,
            "model":             np.nan,
            "selection":         np.nan,
        })

        # ── Dynp configurations ───────────────────────────────────────────
        # Cache raw breakpoints per (n_bkps, model) to avoid redundant fits.
        # Selection is applied separately — it's just a filter on the same bkps.
        bp_cache = {}

        for n_bkps, model, selection in configs:
            cache_key = (n_bkps, model)
            if cache_key not in bp_cache:
                bp_cache[cache_key] = _detect_breakpoints(
                    t_feat, band_mode, model, n_bkps,
                )

            info = bp_cache[cache_key]
            if info is None:
                result = None
            else:
                result = _apply_selection(info, selection)

            det_dist = result["distance"]   if result else np.nan
            conf     = result["confidence"] if result else np.nan
            error    = (det_dist - manual_dist) if result else np.nan

            label = f"dynp_{n_bkps}bkps_{model}_{selection}"
            records.append({
                **base,
                "detector":          label,
                "detected_distance": det_dist,
                "confidence":        conf,
                "signed_error":      error,
                "n_bkps":            n_bkps,
                "model":             model,
                "selection":         selection,
            })

        # ── Recursive two-pass detector ───────────────────────────────────
        # Uses the pass-1 breakpoints from the bp_cache (n_bkps=2) to
        # define the constrained window, then fits n_bkps=1 inside it.
        for rec_model in MODEL_VALUES:
            result = _detect_recursive(band_mode, rec_model, bp_cache)

            det_dist = result["distance"]   if result else np.nan
            conf     = result["confidence"] if result else np.nan
            error    = (det_dist - manual_dist) if result else np.nan

            records.append({
                **base,
                "detector":          f"recursive_{rec_model}",
                "detected_distance": det_dist,
                "confidence":        conf,
                "signed_error":      error,
                "n_bkps":            np.nan,
                "model":             rec_model,
                "selection":         "recursive",
            })

        # ── Smoothed recursive two-pass detector ─────────────────────────
        # Same as recursive, but pass 1 uses smoothed features.
        # Pass 2 also operates on the smoothed signal within the window.
        for sw in (11, 21):
            smooth_bp_cache = {}
            for rec_model in MODEL_VALUES:
                cache_key = (2, rec_model)
                if cache_key not in smooth_bp_cache:
                    smooth_bp_cache[cache_key] = _detect_breakpoints(
                        t_feat, band_mode, rec_model, n_bkps=2,
                        smooth_window=sw,
                    )

                result = _detect_recursive(band_mode, rec_model, smooth_bp_cache)

                det_dist = result["distance"]   if result else np.nan
                conf     = result["confidence"] if result else np.nan
                error    = (det_dist - manual_dist) if result else np.nan

                records.append({
                    **base,
                    "detector":          f"recursive_{rec_model}_smooth{sw}",
                    "detected_distance": det_dist,
                    "confidence":        conf,
                    "signed_error":      error,
                    "n_bkps":            np.nan,
                    "model":             rec_model,
                    "selection":         f"recursive_smooth{sw}",
                })

    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame) -> None:
    """Print per-detector error summary, stratified by band_mode."""
    sep = "─" * 72
    print(f"\n{sep}")
    print("  DYNP EXPERIMENT — ERROR SUMMARY  (positive = seaward of truth)")
    print(sep)

    for detector, grp in df.groupby("detector"):
        valid = grp["signed_error"].dropna()
        if valid.empty:
            print(f"\n  {detector:<40}  no data")
            continue
        print(
            f"\n  {detector:<40}  n={len(valid):>5}  "
            f"mean={valid.mean():>+7.2f} m  "
            f"median={valid.median():>+7.2f} m  "
            f"std={valid.std():>6.2f} m"
        )
        # Per band_mode breakdown
        for bm, bgrp in grp.groupby("band_mode"):
            bvalid = bgrp["signed_error"].dropna()
            if bvalid.empty:
                continue
            print(
                f"    {bm:<12}  n={len(bvalid):>5}  "
                f"mean={bvalid.mean():>+7.2f}  "
                f"median={bvalid.median():>+7.2f}  "
                f"std={bvalid.std():>6.2f}"
            )

    print(f"\n{sep}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manual-transitions", "-m", type=Path,
        default=Path("OUTPUT/manual_transitions.parquet"),
    )
    parser.add_argument(
        "--output-root", "-r", type=Path,
        default=Path("OUTPUT"),
        help="Root dir containing OUTPUT/{year}/features_{year}.parquet",
    )
    parser.add_argument(
        "--transitions-dir", "-t", type=Path,
        default=Path("OUTPUT"),
        help="Dir containing transitions_{year}.parquet (old detector output)",
    )
    parser.add_argument(
        "--band-config", "-b", type=Path,
        default=Path("INPUT/band_config.json"),
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        default=Path("OUTPUT/dynp_experiment.parquet"),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--all-breakpoints", action="store_true",
        help="Also save a parquet with ALL breakpoint positions (for diagnostic histograms).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load manual picks ─────────────────────────────────────────────────
    manual = pd.read_parquet(args.manual_transitions)
    manual["year"] = manual["year"].astype(float).astype(int).astype(str)
    years = sorted(manual["year"].unique())
    logger.info("Manual picks: %d rows, years: %s", len(manual), years)

    # ── Load band config ──────────────────────────────────────────────────
    with open(args.band_config) as f:
        band_config = json.load(f)

    # ── Load features ─────────────────────────────────────────────────────
    features_by_year = {}
    for y in years:
        path = args.output_root / y / f"features_{y}.parquet"
        if not path.exists():
            logger.warning("Features not found for %s: %s", y, path)
            continue
        features_by_year[y] = pd.read_parquet(path)
        logger.info("  Loaded features %s: %d rows", y, len(features_by_year[y]))

    # ── Load old detector transitions ─────────────────────────────────────
    old_shell_lines_by_year = {}
    for y in years:
        candidates = [
            args.transitions_dir / f"transitions_{y}.parquet",
            args.transitions_dir / y / f"transitions_{y}.parquet",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path:
            old_shell_lines_by_year[y] = _load_old_shell_lines(path)
            logger.info("  Old shell lines %s: %d transects", y, len(old_shell_lines_by_year[y]))
        else:
            logger.warning("No transitions parquet for %s — old baseline will be NaN", y)

    # ── Run experiment ────────────────────────────────────────────────────
    results = run_experiment(manual, features_by_year, old_shell_lines_by_year, band_config)

    # ── Save and summarise ────────────────────────────────────────────────
    print_summary(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(args.output, index=False)
    logger.info("Saved %d rows → %s", len(results), args.output)

    # ── All-breakpoints diagnostic ────────────────────────────────────────
    if args.all_breakpoints:
        logger.info("Collecting all breakpoint positions...")
        bp_df = collect_all_breakpoints(
            manual, features_by_year, band_config,
        )
        bp_path = args.output.with_name("all_breakpoints.parquet")
        bp_df.to_parquet(bp_path, index=False)
        logger.info("Saved %d breakpoint rows → %s", len(bp_df), bp_path)


if __name__ == "__main__":
    main()