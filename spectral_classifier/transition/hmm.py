"""
HMM-based shell line detection.

Four-state Gaussian HMM on (NDWI, NIR) profiles to identify the dry
beach / wet beach boundary (shell line).  The four states model the
physical zonation along a land-to-sea transect:

    vegetation -> dry_beach -> wet_beach -> water

Two-dimensional emission vectors (NDWI, NIR) give the model two
complementary signals:
    - NIR:  strong drop at the shell line (dry sand -> wet sand)
    - NDWI: strong jump at the waterline (wet sand -> standing water)

Emission parameters are estimated per-year via EM; the transition
matrix is fixed to enforce the ordering constraint (no backward
transitions).  The shell line is the first dry_beach -> wet_beach
transition in the Viterbi path.

Requires both NDWI and NIR -- not applicable to RGB-only years.

Usage:
    from spectral_classifier.transition.hmm import (
        fit_hmm_for_year,
        detect_shell_line_hmm,
        run_hmm_detection,
    )
    from spectral_classifier.utils.data_io import merge_profiles_and_features

    profiles_df = pd.read_parquet("profiles_2010.parquet")
    features_df = pd.read_parquet("features_2010.parquet")
    merged = merge_profiles_and_features(profiles_df, features_df, columns=["ndwi"])

    model = fit_hmm_for_year(merged)
    results = run_hmm_detection(model, merged)
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Deferred import -- hmmlearn is an optional dependency.
# TODO: migrate to top-level import once hmmlearn is a stable dependency.

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# States -- ordered land to sea.
VEGETATION = 0
DRY_BEACH  = 1
WET_BEACH  = 2
WATER      = 3

STATE_LABELS = {
    VEGETATION: "vegetation",
    DRY_BEACH:  "dry_beach",
    WET_BEACH:  "wet_beach",
    WATER:      "water",
}

N_STATES = 4
N_FEATURES = 2   # (NDWI, NIR)

# Feature column names expected in the merged DataFrame.
FEATURE_COLS = ["ndwi", "nir_d1_smooth"]

# Minimum number of valid sample points per transect.
MIN_TRANSECT_LENGTH = 10

# Default initial emission guesses -- (NDWI, NIR).
# NIR values are raw DN; typical PAIS range is ~30-200.
#
#   vegetation:  negative/low NDWI, high NIR  (foliage reflects NIR)
#   dry_beach:   low-positive NDWI, high NIR  (bright sand)
#   wet_beach:   moderate NDWI, medium NIR    (wet sand absorbs NIR)
#   water:       high NDWI, low NIR           (water absorbs NIR)
_DEFAULT_MEANS = np.array([
    [-0.05, 0.0],   # vegetation
    [ 0.05, -0.5],   # dry_beach
    [ 0.18, -5.0],   # wet_beach
    [ 0.50, -0.5],   # water
])

_DEFAULT_COVARS = np.array([
    [0.005, 4.0],   # vegetation
    [0.003, 2.0],   # dry_beach
    [0.008, 15.0],   # wet_beach
    [0.015, 4.0],   # water
])

# Fixed transition matrix -- ordered, no backslip.
#
#                   veg     dry     wet     water
#   veg       ->  [ 0.85,   0.15,   0.00,   0.00 ]
#   dry_beach ->  [ 0.00,   0.90,   0.10,   0.00 ]
#   wet_beach ->  [ 0.00,   0.00,   0.85,   0.15 ]
#   water     ->  [ 0.00,   0.00,   0.00,   1.00 ]
_TRANSITION_MATRIX = np.array([
    [0.85, 0.15, 0.00, 0.00],
    [0.00, 0.90, 0.10, 0.00],
    [0.00, 0.00, 0.85, 0.15],
    [0.00, 0.00, 0.00, 1.00],
])

_START_PROB = np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_hmmlearn():
    """Deferred import of hmmlearn.hmm.GaussianHMM."""
    try:
        from hmmlearn.hmm import GaussianHMM
        return GaussianHMM
    except ImportError as exc:
        raise ImportError(
            "hmmlearn is required for HMM-based detection.  "
            "Install it with:  pip install hmmlearn"
        ) from exc


def _prepare_sequences(
    merged_df: pd.DataFrame,
    min_length: int = MIN_TRANSECT_LENGTH,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Extract per-transect (NDWI, NIR) arrays from a merged DataFrame.

    Returns:
        X:            (N, 2) concatenated feature array for hmmlearn.
        lengths:      List of per-transect sequence lengths.
        transect_ids: Corresponding transect IDs (same order as lengths).

    Raises:
        ValueError: If required columns are missing or no usable transects.
    """
    for col in FEATURE_COLS:
        if col not in merged_df.columns:
            raise ValueError(
                f"Merged DataFrame missing column '{col}'.  "
                f"Required columns: {FEATURE_COLS}.  "
                f"Did you pass a merged profiles+features DataFrame?"
            )

    sequences: List[np.ndarray] = []
    lengths: List[int] = []
    transect_ids: List[int] = []

    for tid, grp in merged_df.groupby("transect_id"):
        grp_sorted = grp.sort_values("distance")
        valid = grp_sorted[FEATURE_COLS].dropna()
        if len(valid) >= min_length:
            sequences.append(valid.values)       # (T_i, 2)
            lengths.append(len(valid))
            transect_ids.append(tid)

    if not sequences:
        raise ValueError(
            f"No transects with >= {min_length} valid (NDWI, NIR) samples."
        )

    X = np.concatenate(sequences)    # (N, 2)
    return X, lengths, transect_ids


def _ensure_label_order(model) -> bool:
    """
    Verify states are ordered by ascending NDWI mean (feature index 0):
        vegetation < dry_beach < wet_beach < water.

    If out of order, permute all model parameters in-place.

    Returns:
        True if a permutation was applied.
    """
    ndwi_means = model.means_[:, 0]
    sorted_order = np.argsort(ndwi_means)

    if np.array_equal(sorted_order, np.arange(N_STATES)):
        return False

    logger.info(
        "HMM label reorder needed.  NDWI means: %s -> sorted: %s",
        [f"{m:.3f}" for m in ndwi_means],
        sorted_order.tolist(),
    )

    model.means_    = model.means_[sorted_order].copy()
    model.covars_   = model.covars_[sorted_order].copy()
    model.transmat_ = model.transmat_[sorted_order][:, sorted_order].copy()
    model.startprob_ = model.startprob_[sorted_order].copy()

    return True


def _compute_confidence(
    model,
    obs_seq: np.ndarray,
    transition_idx: int,
) -> float:
    """
    Estimate detection confidence at the shell line transition.

    Uses posterior probabilities: geometric mean of P(dry_beach) just
    before and P(wet_beach) at the transition point.
    """
    try:
        posteriors = model.predict_proba(obs_seq)
        wet_prob = float(posteriors[transition_idx, WET_BEACH])
        dry_prob_before = float(posteriors[transition_idx - 1, DRY_BEACH])
        confidence = np.sqrt(wet_prob * dry_prob_before)
        return round(float(np.clip(confidence, 0.0, 1.0)), 4)
    except Exception:
        logger.debug("Confidence computation failed; defaulting to 0.5")
        return 0.5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_hmm_for_year(
    merged_df: pd.DataFrame,
    n_iter: int = 50,
    min_length: int = MIN_TRANSECT_LENGTH,
) -> object:
    """
    Fit a 4-state Gaussian HMM from all transects in one year.

    Emission means and covariances are estimated via EM from pooled
    (NDWI, NIR) observations.  The transition matrix and start
    probabilities are fixed to enforce land-to-sea ordering.

    Args:
        merged_df:  Merged profiles + features DataFrame with columns
                    ``transect_id``, ``distance``, ``ndwi``, ``nir``.
        n_iter:     Maximum EM iterations (default 50).
        min_length: Minimum valid samples per transect.

    Returns:
        Fitted GaussianHMM with states:
            0 = vegetation, 1 = dry_beach, 2 = wet_beach, 3 = water.
    """
    GaussianHMM = _import_hmmlearn()

    X, lengths, tids = _prepare_sequences(merged_df, min_length)
    logger.info(
        "Fitting 4-state 2D HMM on %d transects (%d total samples)",
        len(lengths), len(X),
    )

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=n_iter,
        init_params="",
        params="mc",       # EM refines means + covariances; transmat fixed
    )

    model.startprob_ = _START_PROB.copy()
    model.transmat_  = _TRANSITION_MATRIX.copy()
    model.means_     = _DEFAULT_MEANS.copy()
    model.covars_    = _DEFAULT_COVARS.copy()

    model.fit(X, lengths)

    _ensure_label_order(model)

    for sid, label in STATE_LABELS.items():
        logger.info(
            "  %s  NDWI=%.3f  NIR_d1=%.1f",
            label, model.means_[sid, 0], model.means_[sid, 1],
        )

    return model


def detect_shell_line_hmm(
    model,
    obs: np.ndarray,
    distances: np.ndarray,
    band_mode: str = "unknown",
) -> Dict:
    """
    Decode one transect and return the shell line position.

    Args:
        model:     Fitted GaussianHMM.
        obs:       (T, 2) array of [NDWI, NIR] observations.
        distances: 1-D distance array (same length as obs).
        band_mode: Band config string for output metadata.

    Returns:
        Dict with: distance, index, type, confidence, detection_method,
        band_mode, states, waterline_distance, veg_boundary_distance.
    """
    states = model.predict(obs)

    # Label-order safety net.
    ndwi_means = model.means_[:, 0]
    if not all(ndwi_means[i] <= ndwi_means[i + 1] for i in range(N_STATES - 1)):
        order = np.argsort(ndwi_means)
        remap = np.empty(N_STATES, dtype=int)
        for new_idx, old_idx in enumerate(order):
            remap[old_idx] = new_idx
        states = np.array([remap[s] for s in states])

    # Shell line: first dry_beach -> wet_beach.
    shell_line_dist = np.nan
    shell_line_idx  = -1
    shell_line_conf = 0.0

    for i in range(1, len(states)):
        if states[i - 1] == DRY_BEACH and states[i] == WET_BEACH:
            shell_line_dist = float(distances[i])
            shell_line_idx  = int(i)
            shell_line_conf = _compute_confidence(model, obs, i)
            break

    # Waterline: first wet_beach -> water.
    waterline_dist = np.nan
    for i in range(1, len(states)):
        if states[i - 1] == WET_BEACH and states[i] == WATER:
            waterline_dist = float(distances[i])
            break

    # Vegetation boundary: first vegetation -> dry_beach.
    veg_boundary_dist = np.nan
    for i in range(1, len(states)):
        if states[i - 1] == VEGETATION and states[i] == DRY_BEACH:
            veg_boundary_dist = float(distances[i])
            break

    if shell_line_idx == -1:
        logger.debug("No dry_beach -> wet_beach transition found.")

    return {
        "distance": shell_line_dist,
        "index": shell_line_idx,
        "type": "dry_wet_derivative",
        "confidence": shell_line_conf,
        "detection_method": "hmm_gaussian_4state_2d",
        "band_mode": band_mode,
        "states": states,
        "waterline_distance": waterline_dist,
        "veg_boundary_distance": veg_boundary_dist,
    }


def run_hmm_detection(
    model,
    merged_df: pd.DataFrame,
    band_mode: str = "unknown",
    min_length: int = MIN_TRANSECT_LENGTH,
) -> pd.DataFrame:
    """
    Run HMM detection on all transects in a merged DataFrame.

    Returns:
        DataFrame with columns: transect_id, shell_line_distance,
        shell_line_index, confidence, detection_method,
        waterline_distance, veg_boundary_distance.
    """
    rows: List[Dict] = []

    for tid, grp in merged_df.groupby("transect_id"):
        grp_sorted = grp.sort_values("distance")
        valid = grp_sorted[FEATURE_COLS].dropna()

        if len(valid) < min_length:
            rows.append({
                "transect_id": tid,
                "shell_line_distance": np.nan,
                "shell_line_index": -1,
                "confidence": 0.0,
                "detection_method": "hmm_skipped_short",
                "waterline_distance": np.nan,
                "veg_boundary_distance": np.nan,
            })
            continue

        distances = grp_sorted.loc[valid.index, "distance"].values
        obs = valid.values    # (T, 2)

        result = detect_shell_line_hmm(model, obs, distances, band_mode)

        rows.append({
            "transect_id": tid,
            "shell_line_distance": result["distance"],
            "shell_line_index": result["index"],
            "confidence": result["confidence"],
            "detection_method": result["detection_method"],
            "waterline_distance": result["waterline_distance"],
            "veg_boundary_distance": result["veg_boundary_distance"],
        })

    results_df = pd.DataFrame(rows)
    n_detected = results_df["shell_line_distance"].notna().sum()
    logger.info(
        "HMM detection complete: %d/%d transects yielded a shell line.",
        n_detected, len(results_df),
    )

    return results_df