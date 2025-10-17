"""
DEPRECATED: Monotonic Smoothing Feature

This module contains the monotonic smoothing functionality that was removed
from the main codebase in Phase 8 (2025-10-17).

**Reason for Deprecation**: Phase 6D testing revealed that monotonic smoothing
caused catastrophic class collapse, with 93% of points being reclassified as
UNKNOWN in some transects. This made the feature unusable in production.

**DO NOT USE THESE METHODS** - They are preserved here only for historical
reference and to prevent breaking changes if old code still calls them.

For details on the failure, see:
- docs/IMPROVEMENT_HISTORY.md - Phase 6D section
- validation/outputs/phase6d/ - Diagnostic outputs

Historical code preserved from commit before Phase 8.
"""

import logging
from typing import List, Dict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_monotonic_smoothing(
    features: pd.DataFrame,
    thresholds: Dict,
    min_span_m: float = None,
    use_relaxed: bool = True
) -> pd.DataFrame:
    """
    Apply monotonic sequence constraint to classifications.

    DEPRECATED: This method is no longer functional and will return the
    input DataFrame unchanged. It has been disabled due to catastrophic
    class collapse discovered in Phase 6D testing.

    Enforces the expected land-to-water sequence:
    VEG_DUNES -> DRY_BEACH -> BEACH_WET -> WATER

    Strategy:
    1. Find major zones (segments >= min_span_m)
    2. Build monotonic sequence from major zones
    3. If relaxed mode, preserve original classifications when no major zones found

    Rules:
    - Each category (except UNKNOWN) appears at most once
    - VEG_DUNES is optional (sequence may start with DRY_BEACH)
    - BEACH_WET is optional (may go DRY_BEACH -> WATER directly)
    - UNKNOWN can appear multiple times anywhere

    Args:
        features: DataFrame with 'predicted_class' and 'distance' columns
        thresholds: Threshold dictionary from config
        min_span_m: Minimum span in meters (uses config default if None)
        use_relaxed: If True, only enforce monotonicity without strict span requirements

    Returns:
        DataFrame with monotonic sequence enforced
    """
    logger.warning(
        "apply_monotonic_smoothing() is DEPRECATED and disabled. "
        "It caused 93% UNKNOWN class collapse in Phase 6D testing. "
        "Returning features unchanged. See docs/IMPROVEMENT_HISTORY.md for details."
    )
    return features


def _apply_monotonic_smoothing_legacy(
    features: pd.DataFrame,
    thresholds: Dict,
    min_span_m: float = None,
    use_relaxed: bool = True
) -> pd.DataFrame:
    """
    LEGACY IMPLEMENTATION - DO NOT USE

    This is the original implementation preserved for reference only.
    """
    if 'predicted_class' not in features.columns:
        raise ValueError("Must run classify() before monotonic smoothing")

    if min_span_m is None:
        min_span_m = thresholds.get('min_category_span_m', 2.5)  # Relaxed from 5.0

    logger.debug(f"Applying monotonic smoothing (min_span={min_span_m}m, relaxed={use_relaxed})")

    classes = features['predicted_class'].values.copy()
    distances = features['distance'].values

    # Find all segments
    segments = _build_segments(classes, distances)

    if use_relaxed:
        # Relaxed mode: only enforce monotonicity, don't filter by span
        monotonic_classes = _enforce_monotonic_sequence(classes, distances)
    else:
        # Strict mode: filter by span and build from major zones
        major_zones = [s for s in segments if s['span'] >= min_span_m or s['class'] == 'UNKNOWN']

        # Fallback: if no major zones (except UNKNOWN), just enforce monotonicity
        non_unknown_zones = [z for z in major_zones if z['class'] != 'UNKNOWN']
        if len(non_unknown_zones) == 0:
            logger.warning("No major zones found, falling back to relaxed monotonic enforcement")
            monotonic_classes = _enforce_monotonic_sequence(classes, distances)
        else:
            monotonic_classes = _build_monotonic_from_zones(major_zones, len(classes), distances)

    features['predicted_class'] = monotonic_classes

    # Log results
    class_counts = pd.Series(monotonic_classes).value_counts()
    logger.info(f"Monotonic smoothing complete: {dict(class_counts)}")

    return features


def _build_segments(
    classes: np.ndarray,
    distances: np.ndarray
) -> List[Dict]:
    """Build list of segments from classified data."""
    segments = []
    start_idx = 0
    current_class = classes[0]

    for i in range(1, len(classes)):
        if classes[i] != current_class:
            span = distances[i-1] - distances[start_idx]
            segments.append({
                'class': current_class,
                'start': start_idx,
                'end': i - 1,
                'span': span,
                'start_dist': distances[start_idx],
                'end_dist': distances[i-1]
            })
            start_idx = i
            current_class = classes[i]

    # Add final segment
    span = distances[-1] - distances[start_idx]
    segments.append({
        'class': current_class,
        'start': start_idx,
        'end': len(classes) - 1,
        'span': span,
        'start_dist': distances[start_idx],
        'end_dist': distances[-1]
    })

    return segments


def _build_monotonic_from_zones(
    zones: List[Dict],
    total_points: int,
    distances: np.ndarray
) -> np.ndarray:
    """
    Build monotonic sequence from major zones.

    Args:
        zones: List of zone dictionaries
        total_points: Total number of points
        distances: Distance array

    Returns:
        Array of class labels
    """
    # Initialize all as UNKNOWN
    result = np.array(['UNKNOWN'] * total_points)

    # Sequence order
    sequence_order = {
        'VEG_DUNES': 0,
        'DRY_BEACH': 1,
        'BEACH_WET': 2,
        'WATER': 3,
        'WAVE_CRESTS': 3,
    }

    # Track which class types we've placed (ignore UNKNOWN)
    placed_classes = set()
    max_order_placed = -1

    for zone in zones:
        if zone['class'] == 'UNKNOWN':
            # UNKNOWN zones are always allowed
            result[zone['start']:zone['end']+1] = 'UNKNOWN'
            continue

        zone_order = sequence_order.get(zone['class'], -1)

        # Check if this class maintains monotonic sequence
        if zone['class'] in placed_classes:
            # Already placed this class - can't repeat
            result[zone['start']:zone['end']+1] = 'UNKNOWN'
            continue

        if zone_order < max_order_placed:
            # Going backward - not allowed
            result[zone['start']:zone['end']+1] = 'UNKNOWN'
            continue

        # Valid zone - place it
        result[zone['start']:zone['end']+1] = zone['class']
        placed_classes.add(zone['class'])
        max_order_placed = zone_order

    return result


def _remove_short_segments(
    classes: np.ndarray,
    distances: np.ndarray,
    min_span_m: float
) -> np.ndarray:
    """
    Remove segments shorter than minimum span by merging with neighbors.

    Strategy:
    1. Build segments and identify short ones
    2. Iteratively merge short segments with neighbors
    3. Keep UNKNOWN segments as-is (they can be any length)

    Args:
        classes: Array of class labels
        distances: Array of distances
        min_span_m: Minimum span in meters

    Returns:
        Array with short segments removed
    """
    classes = classes.copy()
    max_iterations = 10  # Prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Identify current segments
        segments = []
        start_idx = 0
        current_class = classes[0]

        for i in range(1, len(classes)):
            if classes[i] != current_class:
                # Segment ends
                span = distances[i-1] - distances[start_idx]
                segments.append({
                    'class': current_class,
                    'start': start_idx,
                    'end': i - 1,
                    'span': span
                })
                start_idx = i
                current_class = classes[i]

        # Add final segment
        span = distances[-1] - distances[start_idx]
        segments.append({
            'class': current_class,
            'start': start_idx,
            'end': len(classes) - 1,
            'span': span
        })

        # Find first short segment (that's not UNKNOWN)
        short_seg_idx = None
        for idx, seg in enumerate(segments):
            if seg['span'] < min_span_m and seg['class'] != 'UNKNOWN':
                short_seg_idx = idx
                break

        # If no short segments found, we're done
        if short_seg_idx is None:
            break

        seg = segments[short_seg_idx]

        # Find adjacent segments (direct neighbors)
        prev_seg = segments[short_seg_idx - 1] if short_seg_idx > 0 else None
        next_seg = segments[short_seg_idx + 1] if short_seg_idx < len(segments) - 1 else None

        # Decide which neighbor to merge with
        # Prefer merging with non-UNKNOWN neighbors
        # If both are UNKNOWN, pick the larger one
        merge_target = None

        if next_seg and next_seg['class'] != 'UNKNOWN':
            merge_target = next_seg['class']
        elif prev_seg and prev_seg['class'] != 'UNKNOWN':
            merge_target = prev_seg['class']
        elif next_seg and prev_seg:
            # Both UNKNOWN, merge with larger
            if next_seg['span'] > prev_seg['span']:
                merge_target = 'UNKNOWN'  # Merge into next
            else:
                merge_target = 'UNKNOWN'  # Merge into prev
        elif next_seg:
            merge_target = next_seg['class']
        elif prev_seg:
            merge_target = prev_seg['class']
        else:
            # Isolated segment, mark as UNKNOWN
            merge_target = 'UNKNOWN'

        # Apply merge
        classes[seg['start']:seg['end']+1] = merge_target

    return classes


def _enforce_monotonic_sequence(
    classes: np.ndarray,
    distances: np.ndarray
) -> np.ndarray:
    """
    Enforce monotonic land-to-water sequence.

    Expected order: VEG_DUNES -> DRY_BEACH -> BEACH_WET -> WATER

    Args:
        classes: Array of class labels
        distances: Array of distances

    Returns:
        Array with monotonic sequence enforced
    """
    classes = classes.copy()

    # Define sequence order (lower number = more landward)
    sequence_order = {
        'VEG_DUNES': 0,
        'DRY_BEACH': 1,
        'BEACH_WET': 2,
        'WATER': 3,
        'WAVE_CRESTS': 3,  # Same as water
        'UNKNOWN': -1  # Can appear anywhere
    }

    # Track which classes we've seen (except UNKNOWN)
    seen_classes = set()
    max_order_seen = -1

    for i in range(len(classes)):
        current_class = classes[i]

        if current_class == 'UNKNOWN':
            continue  # UNKNOWN can appear anywhere

        current_order = sequence_order.get(current_class, -1)

        # Check if we've seen this class before (violation of monotonic rule)
        if current_class in seen_classes:
            # Already seen this class - should not repeat
            # Convert to UNKNOWN or adjacent valid class
            classes[i] = 'UNKNOWN'
            continue

        # Check if this class is out of sequence (going backward)
        if current_order < max_order_seen:
            # Going backward - not allowed
            # Convert to the max class we've seen so far
            for cls, order in sequence_order.items():
                if order == max_order_seen:
                    classes[i] = cls
                    break
        else:
            # Valid forward progression
            seen_classes.add(current_class)
            max_order_seen = current_order

    return classes
