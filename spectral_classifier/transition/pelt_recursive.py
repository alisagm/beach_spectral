"""
Change-point shell line detection using ruptures.

Provides two public detection functions:

  detect_shell_line_pelt       — single-pass, original algorithm (n_bkps=2,
                                 max-contrast selection).
  detect_shell_line_recursive  — two-pass recursive algorithm:
                                 pass 1 finds veg/beach and water edge,
                                 pass 2 finds the shell line between them.

Both return the same output dict structure for pipeline compatibility.

Algorithm — single pass (detect_shell_line_pelt)
-------------------------------------------------
Feature vector : [ndwi_proxy, variability], independently z-scored per transect.
Fitting        : ruptures.Dynp or ruptures.Binseg with n_bkps=2 → 3 segments.
Selection rule : among the two breakpoints, the shell line is the one where
                 the seaward segment has higher mean ndwi_proxy than the
                 landward segment.  This encodes only the physical fact that
                 water is wetter than sand, nothing about where along the
                 transect the boundary should fall.

Algorithm — recursive (detect_shell_line_recursive)
----------------------------------------------------
Pass 1 : Dynp(model, n_bkps=2) on full transect → two breakpoints that
         typically correspond to veg/beach and beach/water boundaries.
Pass 2 : Dynp(model, n_bkps=1) on the signal BETWEEN the two pass-1 breaks.
         Re-z-scored within the constrained window so the dry/wet contrast
         dominates without competition from vegetation or open water.
         The single breakpoint is the shell line candidate.

The recursive approach exploits the empirical finding that the shell line
has strong local NDWI contrast (~+0.25) but is outcompeted by flanking
boundaries in a global optimization.  Constraining the search domain
removes the competing populations.

Band mode handling
------------------
NIR years (4band, cir) : ndwi_proxy = ndwi  (positive toward water)
RGB years  (2022)      : ndwi_proxy = -grvi  (grvi negated so sign matches ndwi)

    grvi = (green - red) / (green + red)   [Green-Red Vegetation Index]

    Negation is required because grvi decreases toward water while ndwi
    increases toward water.  The negation is applied *before* z-scoring so
    the selection rule ("seaward mean higher") is identical for all years.

    Note: grvi contrast is compressed by Gulf Coast turbidity; RGB-year
    confidence scores are attenuated by a fixed factor to reflect this.

variability = window std of brightness (mean of available bands, window=7).
    In RGB years NIR is excluded from brightness, so variability is not
    strictly the same quantity as in NIR years.  This is noted in output
    metadata but is not a blocker — both capture spatial roughness.

Partial coverage
----------------
Transects that extend beyond raster bounds show as trailing NaN blocks.
Detection is attempted on the valid subset with a partial_coverage flag
in the output.  Hard skip (return None) only if valid points < 3 * min_size,
which means there is not enough data to support three segments at minimum size.

Output
------
Returns a transition dict compatible with the rest of the pipeline
(same keys used by TransitionDetector outputs), or None on hard skip.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fraction of points that must be valid before partial_coverage is flagged.
# Does NOT trigger a skip on its own — only affects the output flag.
_PARTIAL_COVERAGE_THRESHOLD = 0.75

# RGB attenuation factor applied to confidence scores.
# Reflects compressed grvi contrast in Gulf Coast turbid water.
_RGB_CONFIDENCE_SCALE = 0.7


# ── Internal helpers ──────────────────────────────────────────────────────────

def _zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score normalisation; returns zeros for flat signals."""
    std = float(arr.std())
    if std == 0.0:
        logger.debug("Feature has zero std; z-score returns zeros.")
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def _prepare_signal(
    features: pd.DataFrame,
    band_mode: str,
    min_size: int,
) -> dict | None:
    """
    Steps 1–3: select ndwi proxy, mask NaNs, z-score.

    Returns a dict with all arrays needed by downstream steps, or None
    on hard skip (insufficient valid points).

    Keys:
        ndwi_proxy       — raw proxy values, full length (for confidence calc)
        ndwi_valid       — proxy values, NaN-masked
        var_valid        — variability values, NaN-masked
        dist_valid       — distances, NaN-masked
        signal           — z-scored [ndwi, variability] matrix (n_valid, 2)
        original_indices — mapping from valid-only indices to full DataFrame
        n_valid          — count of valid points
        partial_coverage — bool flag
        is_rgb           — whether this is an RGB year
    """
    is_rgb = band_mode == "rgb"

    if is_rgb:
        if "grvi" not in features.columns:
            raise KeyError(
                "band_mode='rgb' requires column 'grvi' in features DataFrame. "
                "Ensure features.py computes grvi for RGB-only years."
            )
        ndwi_proxy = -features["grvi"].to_numpy(dtype=float)
    else:
        if "ndwi" not in features.columns:
            raise KeyError("Column 'ndwi' not found in features DataFrame.")
        ndwi_proxy = features["ndwi"].to_numpy(dtype=float)

    variability = features["variability"].to_numpy(dtype=float)
    distances   = features["distance"].to_numpy(dtype=float)

    valid_mask = ~(np.isnan(ndwi_proxy) | np.isnan(variability))
    n_total    = len(valid_mask)
    n_valid    = int(valid_mask.sum())

    partial_coverage = (
        (n_valid / n_total) < _PARTIAL_COVERAGE_THRESHOLD
        if n_total > 0
        else True
    )

    hard_skip_threshold = 3 * min_size
    if n_valid < hard_skip_threshold:
        logger.warning(
            "Skipping transect: %d valid points is below hard-skip threshold "
            "(%d = 3 × min_size=%d).",
            n_valid, hard_skip_threshold, min_size,
        )
        return None

    if partial_coverage:
        logger.debug(
            "Partial coverage: %d/%d valid (%.0f%%).",
            n_valid, n_total, 100 * n_valid / n_total,
        )

    original_indices = np.where(valid_mask)[0]
    ndwi_valid       = ndwi_proxy[valid_mask]
    var_valid        = variability[valid_mask]
    dist_valid       = distances[valid_mask]

    signal = np.column_stack([_zscore(ndwi_valid), _zscore(var_valid)])

    return {
        "ndwi_proxy":       ndwi_proxy,
        "ndwi_valid":       ndwi_valid,
        "var_valid":        var_valid,
        "dist_valid":       dist_valid,
        "signal":           signal,
        "original_indices": original_indices,
        "n_valid":          n_valid,
        "partial_coverage": partial_coverage,
        "is_rgb":           is_rgb,
    }


def _fit_breakpoints(
    signal: np.ndarray,
    n_bkps: int,
    model: str,
    min_size: int,
    algorithm: str = "Dynp",
) -> list[int] | None:
    """
    Step 4: fit ruptures and return interior breakpoint indices.

    Returns list of breakpoint indices (terminal sentinel stripped),
    or None on failure.
    """
    import ruptures  # noqa: PLC0415

    try:
        algo_cls = getattr(ruptures, algorithm)
    except AttributeError:
        raise ValueError(
            f"Unknown ruptures algorithm '{algorithm}'. "
            "Expected 'Dynp' or 'Binseg'."
        )

    n_samples = signal.shape[0]

    try:
        raw_bkps = (
            algo_cls(model=model, min_size=min_size)
            .fit(signal)
            .predict(n_bkps=n_bkps)
        )
    except Exception as exc:
        logger.error("ruptures %s(model=%s) failed: %s", algorithm, model, exc)
        return None

    bkps = [b for b in raw_bkps if b < n_samples]

    if not bkps:
        logger.warning("No usable breakpoints after dropping terminal.")
        return None

    if len(bkps) < n_bkps:
        logger.warning(
            "ruptures returned %d breakpoints, fewer than requested %d.",
            len(bkps), n_bkps,
        )

    return bkps


def _select_max_contrast(
    bkps: list[int],
    ndwi_valid: np.ndarray,
    dist_valid: np.ndarray,
    n_valid: int,
) -> tuple[int, float] | None:
    """
    Step 5: select the breakpoint with maximum positive seaward contrast.

    Returns (bp_index_in_valid, contrast) or None if no breakpoint has
    positive contrast.
    """
    boundaries = [0] + bkps + [n_valid]
    best_bp    = None
    best_c     = -np.inf

    for i, bp in enumerate(bkps):
        left_mean  = ndwi_valid[boundaries[i] : bp].mean()
        right_mean = ndwi_valid[bp : boundaries[i + 2]].mean()
        contrast   = right_mean - left_mean

        logger.debug(
            "bp=%d  dist≈%.1fm  left=%.4f  right=%.4f  contrast=%.4f",
            bp, float(dist_valid[min(bp, n_valid - 1)]),
            left_mean, right_mean, contrast,
        )

        if contrast > best_c:
            best_c  = contrast
            best_bp = bp

    if best_bp is None or best_c <= 0.0:
        logger.warning(
            "No breakpoint with positive seaward contrast (best=%.4f).",
            best_c,
        )
        return None

    return best_bp, best_c


def _build_result(
    bp_idx_in_valid: int,
    contrast: float,
    prep: dict,
    band_mode: str,
    algorithm: str,
    model: str,
    detection_method: str,
) -> dict:
    """
    Steps 6–7: map back to original index, compute confidence, build output dict.
    """
    original_idx   = int(prep["original_indices"][bp_idx_in_valid])
    shell_distance = float(prep["dist_valid"][bp_idx_in_valid])

    ndwi_range = float(
        np.nanmax(prep["ndwi_proxy"]) - np.nanmin(prep["ndwi_proxy"])
    )
    if ndwi_range > 0.0:
        confidence = float(np.clip(contrast / ndwi_range, 0.0, 1.0))
    else:
        confidence = 0.0

    if prep["is_rgb"]:
        confidence *= _RGB_CONFIDENCE_SCALE

    return {
        "index":                 original_idx,
        "distance":              shell_distance,
        "type":                  "dry_wet_pelt",
        "confidence":            round(confidence, 4),
        "detection_method":      detection_method,
        "band_mode":             band_mode,
        "partial_coverage":      prep["partial_coverage"],
        "guaranteed_shell_line": True,
        "ndwi_contrast":         round(float(contrast), 6),
        "n_valid_points":        prep["n_valid"],
        "algorithm":             algorithm,
        "model":                 model,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def detect_shell_line_pelt(
    features: pd.DataFrame,
    band_mode: str,
    algorithm: Literal["Dynp", "Binseg"] = "Dynp",
    model: Literal["rbf", "rank"] = "rbf",
    n_bkps: int = 2,
    min_size: int = 20,
) -> dict | None:
    """
    Detect the shell line using single-pass change-point analysis.

    This is the original algorithm.  For the improved two-pass recursive
    approach, see detect_shell_line_recursive.

    Args:
        features:  Single-transect slice of features_{year}.parquet, sorted
                   by distance ascending.  Required columns: 'distance',
                   'variability', and either 'ndwi' or 'grvi'.
        band_mode: One of 'rgb', '4band', 'cir'.
        algorithm: 'Dynp' (exact, recommended) or 'Binseg' (greedy, faster).
        model:     Cost function — 'rbf' or 'rank'.
        n_bkps:    Number of breakpoints (default 2 → 3 segments).
        min_size:  Minimum segment length in samples.

    Returns:
        Transition dict or None on hard skip.
    """
    prep = _prepare_signal(features, band_mode, min_size)
    if prep is None:
        return None

    bkps = _fit_breakpoints(
        prep["signal"], n_bkps, model, min_size, algorithm,
    )
    if bkps is None:
        return None

    selected = _select_max_contrast(
        bkps, prep["ndwi_valid"], prep["dist_valid"], prep["n_valid"],
    )
    if selected is None:
        return None

    bp_idx, contrast = selected
    return _build_result(
        bp_idx, contrast, prep, band_mode, algorithm, model,
        detection_method=f"pelt_{algorithm.lower()}",
    )


def detect_shell_line_recursive(
    features: pd.DataFrame,
    band_mode: str,
    model: Literal["rbf", "rank"] = "rank",
    min_size_pass1: int = 20,
    min_size_pass2: int = 10,
) -> dict | None:
    """
    Detect the shell line using two-pass recursive change-point analysis.

    Pass 1: Dynp(n_bkps=2) on the full transect to locate the veg/beach
            boundary (landward break) and beach/water boundary (seaward break).
    Pass 2: Dynp(n_bkps=1) on the signal between the two pass-1 breaks,
            re-z-scored within the constrained window.  The single breakpoint
            is the shell line candidate.

    The recursive approach addresses the empirical finding that the shell line
    has strong local NDWI contrast (+0.25 median) but is outcompeted by the
    flanking veg/beach and beach/water transitions in a global optimization.
    Constraining the search domain removes the competing populations.

    Args:
        features:       Single-transect slice, sorted by distance ascending.
        band_mode:      One of 'rgb', '4band', 'cir'.
        model:          Cost function — 'rank' recommended for cross-sensor
                        invariance.
        min_size_pass1: Minimum segment length for pass 1 (full transect).
        min_size_pass2: Minimum segment length for pass 2 (constrained window).
                        Smaller than pass 1 because the window is narrower.

    Returns:
        Transition dict or None on hard skip.  The 'detection_method' field
        is 'recursive_dynp' to distinguish from single-pass results.
    """
    # ── Pass 1: find flanking boundaries on full transect ─────────────────
    prep = _prepare_signal(features, band_mode, min_size_pass1)
    if prep is None:
        return None

    bkps_pass1 = _fit_breakpoints(
        prep["signal"], n_bkps=2, model=model,
        min_size=min_size_pass1, algorithm="Dynp",
    )
    if bkps_pass1 is None:
        return None

    if len(bkps_pass1) < 2:
        logger.warning(
            "Pass 1 returned %d breakpoints, need 2 for recursive. "
            "Falling back to single-pass max-contrast.",
            len(bkps_pass1),
        )
        selected = _select_max_contrast(
            bkps_pass1, prep["ndwi_valid"], prep["dist_valid"], prep["n_valid"],
        )
        if selected is None:
            return None
        bp_idx, contrast = selected
        return _build_result(
            bp_idx, contrast, prep, band_mode, "Dynp", model,
            detection_method="recursive_dynp_fallback",
        )

    # Sort breakpoints landward-to-seaward.
    bp_lo, bp_hi = sorted(bkps_pass1[:2])

    logger.debug(
        "Pass 1 breaks: bp_lo=%d (dist≈%.1fm), bp_hi=%d (dist≈%.1fm)",
        bp_lo, float(prep["dist_valid"][min(bp_lo, prep["n_valid"] - 1)]),
        bp_hi, float(prep["dist_valid"][min(bp_hi, prep["n_valid"] - 1)]),
    )

    # ── Pass 2: search for shell line between the two flanking breaks ─────
    window_len = bp_hi - bp_lo
    if window_len < 2 * min_size_pass2:
        logger.warning(
            "Pass 2 window too narrow: %d samples (need %d = 2 × min_size_pass2). "
            "Falling back to pass 1 landward break.",
            window_len, 2 * min_size_pass2,
        )
        # Fallback: landward break with positive contrast check
        selected = _select_max_contrast(
            bkps_pass1, prep["ndwi_valid"], prep["dist_valid"], prep["n_valid"],
        )
        if selected is None:
            return None
        bp_idx, contrast = selected
        return _build_result(
            bp_idx, contrast, prep, band_mode, "Dynp", model,
            detection_method="recursive_dynp_narrow_fallback",
        )

    # Slice arrays to the constrained window and re-z-score.
    # Re-z-scoring is critical: it removes the influence of the vegetation
    # and open-water populations from the normalisation, so the dry/wet
    # contrast dominates the cost function.
    ndwi_window = prep["ndwi_valid"][bp_lo:bp_hi]
    var_window  = prep["var_valid"][bp_lo:bp_hi]
    dist_window = prep["dist_valid"][bp_lo:bp_hi]

    signal_pass2 = np.column_stack([
        _zscore(ndwi_window),
        _zscore(var_window),
    ])

    bkps_pass2 = _fit_breakpoints(
        signal_pass2, n_bkps=1, model=model,
        min_size=min_size_pass2, algorithm="Dynp",
    )
    if bkps_pass2 is None:
        logger.warning("Pass 2 found no breakpoint. Falling back to pass 1.")
        selected = _select_max_contrast(
            bkps_pass1, prep["ndwi_valid"], prep["dist_valid"], prep["n_valid"],
        )
        if selected is None:
            return None
        bp_idx, contrast = selected
        return _build_result(
            bp_idx, contrast, prep, band_mode, "Dynp", model,
            detection_method="recursive_dynp_pass2_fallback",
        )

    # Map pass-2 breakpoint back to full-transect valid-array index.
    bp_in_window   = bkps_pass2[0]
    bp_in_valid    = bp_lo + bp_in_window

    # Compute contrast at the shell line break using the ORIGINAL (un-windowed)
    # NDWI values.  This gives a contrast that is comparable to single-pass
    # results and to the manual-pick analysis.
    left_mean  = prep["ndwi_valid"][bp_lo:bp_in_valid].mean()
    right_mean = prep["ndwi_valid"][bp_in_valid:bp_hi].mean()
    contrast   = right_mean - left_mean

    logger.debug(
        "Pass 2 shell line: bp=%d (dist≈%.1fm), contrast=%.4f",
        bp_in_valid,
        float(prep["dist_valid"][min(bp_in_valid, prep["n_valid"] - 1)]),
        contrast,
    )

    # Accept even negative contrast in pass 2 — within the constrained
    # domain the signal might legitimately be weak (e.g. very dry conditions).
    # But flag it in confidence: negative contrast → confidence = 0.
    if contrast <= 0.0:
        logger.debug(
            "Pass 2 breakpoint has non-positive contrast (%.4f). "
            "Accepting with zero confidence.",
            contrast,
        )
        contrast = max(contrast, 0.0)

    return _build_result(
        bp_in_valid, contrast, prep, band_mode, "Dynp", model,
        detection_method="recursive_dynp",
    )