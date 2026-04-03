"""
Change-point shell line detection using ruptures.

Replaces the gradient-peak detector in transition/shell_line.py with a
distributional-shift approach that requires no positional priors or
hardcoded thresholds.

Algorithm
---------
Feature vector : [ndwi_proxy, variability], independently z-scored per transect.
Fitting        : ruptures.Dynp or ruptures.Binseg with n_bkps=2 → 3 segments.
Selection rule : among the two breakpoints, the shell line is the one where
                 the seaward segment has higher mean ndwi_proxy than the
                 landward segment.  This encodes only the physical fact that
                 water is wetter than sand, nothing about where along the
                 transect the boundary should fall.

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


def detect_shell_line_pelt(
    features: pd.DataFrame,
    band_mode: str,
    algorithm: Literal["Dynp", "Binseg"] = "Dynp",
    model: Literal["rbf", "rank"] = "rbf",
    n_bkps: int = 2,
    min_size: int = 20,
) -> dict | None:
    """
    Detect the shell line (dry/wet beach boundary) using change-point analysis.

    Args:
        features:  Single-transect slice of features_{year}.parquet, sorted
                   by distance ascending.  Required columns: 'distance',
                   'variability', and either 'ndwi' (NIR years) or 'grvi'
                   (RGB years).  Column 'grvi' must be present when
                   band_mode='rgb'; see features.py.
        band_mode: Band configuration string — one of 'rgb', '4band', 'cir'.
                   Determines which ndwi_proxy column is used.
        algorithm: 'Dynp' (exact dynamic programming, recommended) or
                   'Binseg' (greedy binary segmentation, faster on long signals).
        model:     Cost function passed to ruptures.
                   'rbf'  — Gaussian kernel, detects distributional shifts
                            in multivariate space (recommended for [ndwi, var]).
                   'rank' — rank-based, stronger radiometric invariance across
                            years with different sensors or brightness offsets.
        n_bkps:    Number of breakpoints (default 2 → 3 segments).
        min_size:  Minimum segment length in samples.  At ~1 m/sample the
                   default of 20 corresponds to a 20 m minimum zone width.

    Returns:
        dict with keys:
            index                 int    — row index into *features* DataFrame
            distance              float  — distance from landward end (m)
            type                  str    — 'dry_wet_pelt'
            confidence            float  — [0, 1], NDWI-contrast-based
            detection_method      str    — 'pelt_dynp' or 'pelt_binseg'
            band_mode             str
            partial_coverage      bool   — True if > 25% of points were NaN
            guaranteed_shell_line bool   — always True (matches pipeline contract)
            ndwi_contrast         float  — raw seaward–landward NDWI difference
            n_valid_points        int    — points used after NaN masking
            algorithm             str
            model                 str
        Returns None on hard skip (insufficient valid points).

    Raises:
        KeyError  : if a required column is missing from features.
        ValueError: if an unknown algorithm name is supplied.
    """
    # Deferred import: ruptures is only needed at detect-time, and keeping
    # it deferred avoids eager import failures if the package is unavailable
    # in environments that only run compute.py.
    # TODO: move to top-level import once interpret.py stabilises.
    import ruptures  # noqa: PLC0415

    # ── 1. Select ndwi proxy ──────────────────────────────────────────────
    is_rgb = band_mode == "rgb"

    if is_rgb:
        # grvi = (G-R)/(G+R); negated so sign convention matches ndwi.
        # Higher value → more water-like, matching the selection rule below.
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

    # ── 2. NaN masking ────────────────────────────────────────────────────
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
            "(%d = 3 × min_size=%d).  Possible cause: transect extends beyond "
            "raster coverage bounds.",
            n_valid, hard_skip_threshold, min_size,
        )
        return None

    if partial_coverage:
        logger.debug(
            "Partial coverage detected: %d/%d valid points (%.0f%%).  "
            "Detection will proceed with partial_coverage=True.",
            n_valid, n_total, 100 * n_valid / n_total,
        )

    # Slice to valid rows.  We preserve the mapping back to the original
    # index so the output 'index' key is correct for the full DataFrame.
    original_indices = np.where(valid_mask)[0]
    ndwi_valid       = ndwi_proxy[valid_mask]
    var_valid        = variability[valid_mask]
    dist_valid       = distances[valid_mask]

    # ── 3. Z-score normalisation (per-transect, per-feature) ──────────────
    # Both features are independently normalised so that the scale difference
    # between a normalised index ([-1, 1] range) and a window std (DN units,
    # possibly [0, 30]) does not let one feature dominate the cost function.
    def _zscore(arr: np.ndarray) -> np.ndarray:
        std = float(arr.std())
        if std == 0.0:
            # Flat feature — no information; return zeros rather than NaN.
            logger.debug("Feature has zero std across transect; z-score returns zeros.")
            return np.zeros_like(arr)
        return (arr - arr.mean()) / std

    signal = np.column_stack([_zscore(ndwi_valid), _zscore(var_valid)])
    # signal shape: (n_valid, 2)

    # ── 4. Change-point detection ─────────────────────────────────────────
    try:
        algo_cls = getattr(ruptures, algorithm)
    except AttributeError:
        raise ValueError(
            f"Unknown ruptures algorithm '{algorithm}'. "
            "Expected 'Dynp' or 'Binseg'."
        )

    try:
        raw_bkps = (
            algo_cls(model=model, min_size=min_size)
            .fit(signal)
            .predict(n_bkps=n_bkps)
        )
    except Exception as exc:
        logger.error("ruptures %s(model=%s) failed: %s", algorithm, model, exc)
        return None

    # ruptures always appends the terminal index (== n_valid) as a sentinel;
    # drop it so bkps contains only genuine interior breakpoints.
    bkps = [b for b in raw_bkps if b < n_valid]

    if len(bkps) == 0:
        logger.warning(
            "ruptures returned no usable breakpoints after dropping terminal.  "
            "Signal may be too short or flat."
        )
        return None

    if len(bkps) < n_bkps:
        logger.warning(
            "ruptures returned %d breakpoints, fewer than requested %d.  "
            "Proceeding with what was returned.",
            len(bkps), n_bkps,
        )

    # ── 5. Shell line selection ───────────────────────────────────────────
    # Segments are the intervals between consecutive boundaries.
    # For n_bkps=2: boundaries = [0, bp0, bp1, n_valid] → 3 segments.
    #
    # Selection rule (no positional prior):
    #   shell line = the breakpoint where the seaward (higher-distance)
    #   segment's mean ndwi_proxy exceeds the landward segment's mean by
    #   the greatest margin.
    #
    # Degenerate case: if no breakpoint has positive contrast (NDWI
    # decreases toward the sea — e.g. very dry year with no wet zone
    # visible), return None rather than return a wrong answer.
    boundaries = [0] + bkps + [n_valid]
    shell_idx_in_valid = None
    best_contrast      = -np.inf

    for i, bp in enumerate(bkps):
        seg_left_mean  = ndwi_valid[boundaries[i] : bp].mean()
        seg_right_mean = ndwi_valid[bp : boundaries[i + 2]].mean()
        contrast       = seg_right_mean - seg_left_mean

        logger.debug(
            "bp=%d  dist≈%.1fm  landward_mean=%.4f  seaward_mean=%.4f  contrast=%.4f",
            bp,
            float(dist_valid[min(bp, n_valid - 1)]),
            seg_left_mean,
            seg_right_mean,
            contrast,
        )

        if contrast > best_contrast:
            best_contrast      = contrast
            shell_idx_in_valid = bp

    if shell_idx_in_valid is None or best_contrast <= 0.0:
        logger.warning(
            "No breakpoint showed positive seaward NDWI contrast "
            "(best=%.4f).  Transect may lack a discernible wet zone.",
            best_contrast,
        )
        return None

    # ── 6. Map back to original DataFrame index and distance ──────────────
    # shell_idx_in_valid is an index into the NaN-masked sub-arrays.
    # original_indices[shell_idx_in_valid] is the corresponding row in
    # the original features DataFrame — this is what the pipeline uses
    # to look up coordinates in profiles_{year}.parquet.
    original_idx   = int(original_indices[shell_idx_in_valid])
    shell_distance = float(dist_valid[shell_idx_in_valid])

    # ── 7. Confidence ─────────────────────────────────────────────────────
    # Normalise NDWI contrast by the full-transect NDWI range so the score
    # reflects how clearly the shell line stands out *relative to the
    # transect's own dynamic range*, not just absolute NDWI magnitude.
    # This makes scores more comparable across years with different sensors.
    ndwi_range = float(np.nanmax(ndwi_proxy) - np.nanmin(ndwi_proxy))
    if ndwi_range > 0.0:
        confidence = float(np.clip(best_contrast / ndwi_range, 0.0, 1.0))
    else:
        confidence = 0.0

    if is_rgb:
        # Attenuate: grvi contrast is compressed by Gulf Coast turbid water
        # relative to NDWI, so raw scores would overstate reliability.
        confidence *= _RGB_CONFIDENCE_SCALE

    return {
        "index":                 original_idx,
        "distance":              shell_distance,
        "type":                  "dry_wet_pelt",
        "confidence":            round(confidence, 4),
        "detection_method":      f"pelt_{algorithm.lower()}",
        "band_mode":             band_mode,
        "partial_coverage":      partial_coverage,
        "guaranteed_shell_line": True,
        # ── Diagnostic fields ──────────────────────────────────────────
        # Consumed by the Phase 2 comparison harness and plots.
        # Not used by export.py or the GeoJSON builder.
        "ndwi_contrast":         round(float(best_contrast), 6),
        "n_valid_points":        n_valid,
        "algorithm":             algorithm,
        "model":                 model,
    }