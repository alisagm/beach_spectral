# spectral_classifier/visualization/selector.py
"""
Column resolution and transect sampling for plot configuration.

Public API
----------
COLUMN_GROUPS          dict  — shorthand group names → column lists
COLUMN_REGISTRY        dict  — per-column metadata (source, scale, plot style)

resolve_plot_columns   Validates + expands a user feature list into a PlotSpec.
list_available_columns Returns {group: [col, ...]} for columns that are present
                       and non-all-NaN in the supplied DataFrames. Safe to call
                       outside any plotting context (e.g. from a diagnostic
                       script or run.py's --list-features flag).
print_feature_table    Pretty-prints list_available_columns output to stdout.
sample_transects       Unified transect sampler: random-N, stride, or all.

Private helpers
---------------
_assign_axes           Partition column lists into primary / secondary axes
                       based on scale-family mix (see docstring for rules).
_axis_label            Return a y-axis label string for a list of columns.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ColumnMeta:
    """Metadata for a single plottable column."""
    source: Literal["profiles", "features", "transitions"]
    group:  str                                # key in COLUMN_GROUPS
    scale:  Literal["dn", "norm", "deriv", "binary"]
    style:  dict = field(default_factory=dict) # matplotlib kwargs for ax.plot()


@dataclass
class PlotSpec:
    """
    Resolved plotting specification produced by resolve_plot_columns.

    Attributes
    ----------
    profile_cols    Columns to pull from profiles_df.
    feature_cols    Columns to pull from features_df.
    transition_cols Columns to pull from transitions_df (stub, always []).
    primary_cols    Columns assigned to the primary (left) y-axis.
    secondary_cols  Columns assigned to the secondary (right) y-axis.

    Note: binary columns (has_oscillations, has_rgb_foam_peak) are excluded
    from both axis lists and must be handled separately.
    TODO: render binary columns as axvspan shaded regions once the rendering
    layer supports mixed continuous + discrete series.
    """
    profile_cols:    list[str]
    feature_cols:    list[str]
    transition_cols: list[str]
    primary_cols:    list[str]
    secondary_cols:  list[str]


# ── Column registry ────────────────────────────────────────────────────────────
# Each entry maps a column name to its source parquet, semantic group,
# scale family, and default matplotlib plot style.
#
# Scale families:
#   dn     — raw digital numbers (~0–255); primary axis candidate
#   norm   — normalized ratios / indices (roughly –1 to 1)
#   deriv  — first/second derivatives in DN/m or DN/m²
#   binary — boolean flags; excluded from line plots (see PlotSpec note)

COLUMN_REGISTRY: dict[str, ColumnMeta] = {

    # ── Band profiles (source: profiles_df) ───────────────────────────────
    "red":   ColumnMeta("profiles", "bands", "dn",
                        dict(color="red",        linewidth=1.5, alpha=0.85, label="Red")),
    "green": ColumnMeta("profiles", "bands", "dn",
                        dict(color="green",      linewidth=1.5, alpha=0.85, label="Green")),
    "blue":  ColumnMeta("profiles", "bands", "dn",
                        dict(color="steelblue",  linewidth=1.5, alpha=0.85, label="Blue")),
    "nir":   ColumnMeta("profiles", "bands", "dn",
                        dict(color="darkred",    linewidth=1.5, alpha=0.85, label="NIR",
                             linestyle="--")),

    # ── Spectral indices (source: features_df) ────────────────────────────
    "ndwi":            ColumnMeta("features", "indices", "norm",
                                  dict(color="#1a78c2", linewidth=1.5, alpha=0.85,
                                       label="NDWI")),
    "ndvi":            ColumnMeta("features", "indices", "norm",
                                  dict(color="#2ca02c", linewidth=1.5, alpha=0.85,
                                       label="NDVI")),
    "nir_ratio":       ColumnMeta("features", "indices", "norm",
                                  dict(color="#8B0000", linewidth=1.2, alpha=0.75,
                                       label="NIR ratio", linestyle="--")),
    "brightness_rgb":  ColumnMeta("features", "indices", "dn",
                                  dict(color="#888888", linewidth=1.2, alpha=0.75,
                                       label="Brightness (RGB)")),
    "brightness":      ColumnMeta("features", "indices", "dn",
                                  dict(color="#555555", linewidth=1.2, alpha=0.75,
                                       label="Brightness")),
    "blue_red_ratio":  ColumnMeta("features", "indices", "norm",
                                  dict(color="#6a3d9a", linewidth=1.2, alpha=0.75,
                                       label="Blue/Red ratio")),
    "red_green_ratio": ColumnMeta("features", "indices", "norm",
                                  dict(color="#e6550d", linewidth=1.2, alpha=0.75,
                                       label="Red/Green ratio")),

    # ── First derivatives — NIR (features_df) ─────────────────────────────
    # nir_d1 and nir_d1_smooth come from compute_band_derivative /
    # compute_derivative_smooth applied directly to the NIR band.
    # nir_d1_wN come from compute_derivative_multiscale (NIR years only).
    "nir_d1":          ColumnMeta("features", "derivatives", "deriv",
                                  dict(color="#ff7f0e", linewidth=1.2, alpha=0.70,
                                       linestyle="-.", label="NIR d/dx")),
    "nir_d1_smooth":   ColumnMeta("features", "derivatives", "deriv",
                                  dict(color="darkorange", linewidth=1.5, alpha=0.85,
                                       linestyle="-.", label="NIR d/dx (smooth)")),
    "nir_d1_w5":       ColumnMeta("features", "derivatives", "deriv",
                                  dict(color="#d62728", linewidth=0.9, alpha=0.60,
                                       linestyle=":", label="NIR d/dx w5")),
    "nir_d1_w7":       ColumnMeta("features", "derivatives", "deriv",
                                  dict(color="#e377c2", linewidth=0.9, alpha=0.60,
                                       linestyle=":", label="NIR d/dx w7")),
    "nir_d1_w9":       ColumnMeta("features", "derivatives", "deriv",
                                  dict(color="#7f7f7f", linewidth=0.9, alpha=0.60,
                                       linestyle=":", label="NIR d/dx w9")),
    "nir_d1_w11":      ColumnMeta("features", "derivatives", "deriv",
                                  dict(color="#bcbd22", linewidth=0.9, alpha=0.60,
                                       linestyle=":", label="NIR d/dx w11")),

    # ── Second derivatives — NIR (features_df) ────────────────────────────
    "nir_d2_w5":       ColumnMeta("features", "derivatives", "deriv",
                                  dict(color="#17becf", linewidth=0.9, alpha=0.60,
                                       linestyle=":", label="NIR d²/dx² w5")),
    "nir_d2_w7":       ColumnMeta("features", "derivatives", "deriv",
                                  dict(color="#9467bd", linewidth=0.9, alpha=0.60,
                                       linestyle=":", label="NIR d²/dx² w7")),
    "nir_d2_w9":       ColumnMeta("features", "derivatives", "deriv",
                                  dict(color="#8c564b", linewidth=0.9, alpha=0.60,
                                       linestyle=":", label="NIR d²/dx² w9")),

    # ── Brightness derivatives (features_df) ──────────────────────────────
    # brightness_rgb_d1/smooth: always present (RGB-mode fallback signal).
    # brightness_rgb_d1_wN: from compute_derivative_multiscale when NIR
    # is unavailable; keys use the 'brightness_rgb' prefix.
    "brightness_d1":             ColumnMeta("features", "derivatives", "deriv",
                                            dict(color="#aec7e8", linewidth=1.0, alpha=0.65,
                                                 linestyle="-.", label="Brightness d/dx")),
    "brightness_d1_smooth":      ColumnMeta("features", "derivatives", "deriv",
                                            dict(color="#1f77b4", linewidth=1.2, alpha=0.75,
                                                 linestyle="-.", label="Brightness d/dx (smooth)")),
    "brightness_rgb_d1":         ColumnMeta("features", "derivatives", "deriv",
                                            dict(color="#c5b0d5", linewidth=1.0, alpha=0.65,
                                                 linestyle="-.", label="Brightness RGB d/dx")),
    "brightness_rgb_d1_smooth":  ColumnMeta("features", "derivatives", "deriv",
                                            dict(color="#9467bd", linewidth=1.2, alpha=0.75,
                                                 linestyle="-.", label="Brightness RGB d/dx (smooth)")),
    "brightness_rgb_d1_w5":      ColumnMeta("features", "derivatives", "deriv",
                                            dict(color="#c49c94", linewidth=0.9, alpha=0.60,
                                                 linestyle=":", label="Brightness RGB d/dx w5")),
    "brightness_rgb_d1_w7":      ColumnMeta("features", "derivatives", "deriv",
                                            dict(color="#f7b6d2", linewidth=0.9, alpha=0.60,
                                                 linestyle=":", label="Brightness RGB d/dx w7")),
    "brightness_rgb_d1_w9":      ColumnMeta("features", "derivatives", "deriv",
                                            dict(color="#dbdb8d", linewidth=0.9, alpha=0.60,
                                                 linestyle=":", label="Brightness RGB d/dx w9")),
    "brightness_rgb_d1_w11":     ColumnMeta("features", "derivatives", "deriv",
                                            dict(color="#9edae5", linewidth=0.9, alpha=0.60,
                                                 linestyle=":", label="Brightness RGB d/dx w11")),

    # ── Per-band derivatives (features_df) ────────────────────────────────
    "red_d1":          ColumnMeta("features", "derivatives", "deriv",
                                  dict(color="#ff9896", linewidth=1.0, alpha=0.65,
                                       linestyle="-.", label="Red d/dx")),
    "red_d1_smooth":   ColumnMeta("features", "derivatives", "deriv",
                                  dict(color="#d62728", linewidth=1.2, alpha=0.75,
                                       linestyle="-.", label="Red d/dx (smooth)")),
    "green_d1":        ColumnMeta("features", "derivatives", "deriv",
                                  dict(color="#98df8a", linewidth=1.0, alpha=0.65,
                                       linestyle="-.", label="Green d/dx")),
    "green_d1_smooth": ColumnMeta("features", "derivatives", "deriv",
                                  dict(color="#2ca02c", linewidth=1.2, alpha=0.75,
                                       linestyle="-.", label="Green d/dx (smooth)")),
    "blue_d1":         ColumnMeta("features", "derivatives", "deriv",
                                  dict(color="#aec7e8", linewidth=1.0, alpha=0.65,
                                       linestyle="-.", label="Blue d/dx")),
    "blue_d1_smooth":  ColumnMeta("features", "derivatives", "deriv",
                                  dict(color="#1f77b4", linewidth=1.2, alpha=0.75,
                                       linestyle="-.", label="Blue d/dx (smooth)")),

    # ── R/G ratio derivatives (features_df) ───────────────────────────────
    "rg_ratio_d1":        ColumnMeta("features", "derivatives", "deriv",
                                     dict(color="#ffbb78", linewidth=1.0, alpha=0.65,
                                          linestyle="-.", label="R/G ratio d/dx")),
    "rg_ratio_d1_smooth": ColumnMeta("features", "derivatives", "deriv",
                                     dict(color="#ff7f0e", linewidth=1.2, alpha=0.75,
                                          linestyle="-.", label="R/G ratio d/dx (smooth)")),

    # ── Window / statistical features (features_df) ───────────────────────
    # variability and local_range are DN-scale (rolling std / range of DN values).
    # slope and curvature are derivative-scale (DN/sample and DN/sample²).
    "variability": ColumnMeta("features", "window", "dn",
                              dict(color="#98df8a", linewidth=1.2, alpha=0.75,
                                   label="Variability (std)")),
    "local_range": ColumnMeta("features", "window", "dn",
                              dict(color="#c7c7c7", linewidth=1.0, alpha=0.65,
                                   label="Local range")),
    "slope":       ColumnMeta("features", "window", "deriv",
                              dict(color="#2ca02c", linewidth=1.2, alpha=0.75,
                                   linestyle="-.", label="Slope (window)")),
    "curvature":   ColumnMeta("features", "window", "deriv",
                              dict(color="#d62728", linewidth=1.0, alpha=0.65,
                                   linestyle=":", label="Curvature")),

    # ── Detection / categorical features (features_df) ────────────────────
    # spectral_angle is a continuous value (0–π rad); rendered as a line.
    # has_oscillations and has_rgb_foam_peak are boolean flags; excluded
    # from line plots — see PlotSpec note and TODO above.
    "spectral_angle":    ColumnMeta("features", "detection", "norm",
                                    dict(color="#7f7f7f", linewidth=1.2, alpha=0.75,
                                         label="Spectral angle")),
    "has_oscillations":  ColumnMeta("features", "detection", "binary", dict()),
    "has_rgb_foam_peak": ColumnMeta("features", "detection", "binary", dict()),

    # ── Transitions (stub — extend when interpret.py is integrated) ────────
    # Transition columns hold a single distance value per transect (scalar),
    # requiring axvline / axvspan rendering rather than per-distance line plots.
    # TODO: add column entries here once interpret.py schema is finalised.
}


# ── Column groups ──────────────────────────────────────────────────────────────

def _build_all_group() -> list[str]:
    """All non-stub, non-binary columns in registry order."""
    return [
        col for col, meta in COLUMN_REGISTRY.items()
        if meta.group != "transitions" and meta.scale != "binary"
    ]


COLUMN_GROUPS: dict[str, list[str]] = {
    "bands": ["red", "green", "blue", "nir"],

    "indices": [
        "ndwi", "ndvi", "nir_ratio", "brightness_rgb",
        "brightness", "blue_red_ratio", "red_green_ratio",
    ],

    "derivatives": [
        # NIR derivatives (NIR years only — NaN-filled otherwise)
        "nir_d1", "nir_d1_smooth",
        "nir_d1_w5", "nir_d1_w7", "nir_d1_w9", "nir_d1_w11",
        "nir_d2_w5", "nir_d2_w7", "nir_d2_w9",
        # Brightness derivatives (always present)
        "brightness_d1", "brightness_d1_smooth",
        "brightness_rgb_d1", "brightness_rgb_d1_smooth",
        # Multiscale brightness (non-NIR years only — NaN-filled otherwise)
        "brightness_rgb_d1_w5", "brightness_rgb_d1_w7",
        "brightness_rgb_d1_w9", "brightness_rgb_d1_w11",
        # Per-band derivatives
        "red_d1", "red_d1_smooth",
        "green_d1", "green_d1_smooth",
        "blue_d1", "blue_d1_smooth",
        # R/G ratio derivatives
        "rg_ratio_d1", "rg_ratio_d1_smooth",
    ],

    "window": ["variability", "slope", "curvature", "local_range"],

    "detection": ["spectral_angle", "has_oscillations", "has_rgb_foam_peak"],

    # Stub — populated once interpret.py integration is complete.
    "transitions": [],

    # Convenience alias: all non-stub, non-binary columns.
    "all": _build_all_group(),
}


# ── Private helpers ────────────────────────────────────────────────────────────

def _assign_axes(
    dn:    list[str],
    norm:  list[str],
    deriv: list[str],
) -> tuple[list[str], list[str]]:
    """
    Partition columns into primary and secondary y-axes based on scale family.

    Rules
    -----
    1 family present  → all columns on primary; secondary empty.
    2 families        → higher-priority family on primary, other on secondary.
                        Priority: dn > norm > deriv.
    3 families        → dn on primary; norm + deriv share secondary.
                        A WARNING is logged because norm [−1,1] and deriv
                        [DN/m] have incompatible units on the same axis.

    Binary columns must be stripped before calling this function.
    """
    # Ordered by priority so `present[0]` is always the highest-priority family.
    ordered = [(dn, "dn"), (norm, "norm"), (deriv, "deriv")]
    present = [(cols, name) for cols, name in ordered if cols]

    if len(present) == 0:
        return [], []

    if len(present) == 1:
        return present[0][0], []

    if len(present) == 2:
        return present[0][0], present[1][0]

    # All three families present.
    logger.warning(
        "Normalized [−1,1] and derivative [DN/m] columns share the secondary "
        "y-axis — two incompatible unit systems on one scale. "
        "Consider selecting only one of these groups to avoid visual confusion."
    )
    return dn, norm + deriv


def _axis_label(cols: list[str]) -> str:
    """Return a descriptive y-axis label derived from the scale families present."""
    if not cols:
        return ""
    scales = {COLUMN_REGISTRY[c].scale for c in cols if c in COLUMN_REGISTRY}
    if scales == {"dn"}:
        return "Spectral value (DN)"
    if scales == {"norm"}:
        return "Index value (normalized)"
    if scales == {"deriv"}:
        return "Derivative (DN/m)"
    # Mixed (3-family secondary, or unknown columns)
    return "Value (mixed units)"


def _is_all_nan(series: pd.Series) -> bool:
    """Return True if the series contains no finite, non-NaN values."""
    return not np.isfinite(series.values).any()


def _col_in_df(col: str, df: Optional[pd.DataFrame]) -> bool:
    """Return True if *col* is a non-all-NaN column in *df*."""
    if df is None or col not in df.columns:
        return False
    return not _is_all_nan(df[col])


# ── Public API ─────────────────────────────────────────────────────────────────

def expand_group_shorthands(requested: list[str]) -> list[str]:
    """
    Expand group shorthand names to individual column names.

    Unknown tokens (not a group key) are passed through unchanged so that
    they can be validated against the actual parquet columns downstream.

    Example
    -------
    >>> expand_group_shorthands(["bands", "ndwi", "nir_d1_smooth"])
    ['red', 'green', 'blue', 'nir', 'ndwi', 'nir_d1_smooth']
    """
    expanded: list[str] = []
    seen: set[str] = set()

    for token in requested:
        if token in COLUMN_GROUPS:
            for col in COLUMN_GROUPS[token]:
                if col not in seen:
                    expanded.append(col)
                    seen.add(col)
        else:
            if token not in seen:
                expanded.append(token)
                seen.add(token)

    return expanded


def resolve_plot_columns(
    requested:      list[str],
    profiles_df:    pd.DataFrame,
    features_df:    Optional[pd.DataFrame],
    transitions_df: Optional[pd.DataFrame] = None,  # stub
) -> PlotSpec:
    """
    Expand group shorthands, validate against actual parquet columns, and
    build a PlotSpec describing which columns go on which axes.

    Columns are excluded (with a WARNING) when:
    - They are not in COLUMN_REGISTRY (unknown name).
    - Their source parquet was not provided.
    - They are present in the parquet but entirely NaN (e.g. nir in an
      RGB-only year).

    Binary columns (has_oscillations, has_rgb_foam_peak) are excluded from
    both axis lists and a WARNING is logged — see PlotSpec for the future
    axvspan TODO.

    Args:
        requested:      Column names or group shorthands (e.g. ["bands", "ndwi"]).
        profiles_df:    Profiles parquet DataFrame (required).
        features_df:    Features parquet DataFrame, or None.
        transitions_df: Transitions parquet DataFrame, or None (stub).

    Returns:
        PlotSpec with resolved column assignments.
    """
    cols = expand_group_shorthands(requested)

    profile_cols:    list[str] = []
    feature_cols:    list[str] = []
    transition_cols: list[str] = []

    dn:     list[str] = []
    norm:   list[str] = []
    deriv:  list[str] = []
    binary: list[str] = []

    for col in cols:
        # ── Unknown column ─────────────────────────────────────────────
        if col not in COLUMN_REGISTRY:
            logger.warning(
                "Column %r not in COLUMN_REGISTRY — skipping. "
                "Run --list-features to see available columns.",
                col,
            )
            continue

        meta = COLUMN_REGISTRY[col]

        # ── Source availability ────────────────────────────────────────
        if meta.source == "profiles":
            if not _col_in_df(col, profiles_df):
                logger.warning(
                    "Column %r not found or all-NaN in profiles parquet — skipping.",
                    col,
                )
                continue
            profile_cols.append(col)

        elif meta.source == "features":
            if not _col_in_df(col, features_df):
                if features_df is None:
                    logger.warning(
                        "Column %r requires features parquet which was not provided — skipping.",
                        col,
                    )
                else:
                    logger.warning(
                        "Column %r not found or all-NaN in features parquet — skipping. "
                        "(This is expected for NIR-only columns in RGB years, and vice versa.)",
                        col,
                    )
                continue
            feature_cols.append(col)

        elif meta.source == "transitions":
            # Stub: transitions_df not yet produced by interpret.py.
            logger.warning(
                "Column %r is a transitions column — not yet supported "
                "(pending interpret.py integration). Skipping.",
                col,
            )
            continue

        # ── Scale family partition ─────────────────────────────────────
        if meta.scale == "dn":
            dn.append(col)
        elif meta.scale == "norm":
            norm.append(col)
        elif meta.scale == "deriv":
            deriv.append(col)
        elif meta.scale == "binary":
            binary.append(col)

    if binary:
        logger.warning(
            "Binary columns %s cannot be rendered as line plots and will be "
            "skipped. TODO: add axvspan shaded-region rendering.",
            binary,
        )

    primary_cols, secondary_cols = _assign_axes(dn, norm, deriv)

    return PlotSpec(
        profile_cols=profile_cols,
        feature_cols=feature_cols,
        transition_cols=transition_cols,
        primary_cols=primary_cols,
        secondary_cols=secondary_cols,
    )


def list_available_columns(
    profiles_df:    pd.DataFrame,
    features_df:    Optional[pd.DataFrame] = None,
    transitions_df: Optional[pd.DataFrame] = None,  # stub
) -> dict[str, list[str]]:
    """
    Return available columns grouped by semantic category.

    A column is included only if it is:
    - Present in COLUMN_REGISTRY.
    - Present in the correct source DataFrame.
    - Not entirely NaN in that DataFrame.

    Importable independently of plotting — safe to use from diagnostic
    scripts or run.py's --list-features handler.

    Args:
        profiles_df:    Profiles parquet DataFrame.
        features_df:    Features parquet DataFrame, or None.
        transitions_df: Transitions parquet DataFrame, or None (stub).

    Returns:
        Ordered dict mapping group name → list of available column names.
    """
    result: dict[str, list[str]] = {}

    for group, cols in COLUMN_GROUPS.items():
        if group == "all":
            continue  # derived; not listed separately

        available: list[str] = []
        for col in cols:
            if col not in COLUMN_REGISTRY:
                continue
            meta = COLUMN_REGISTRY[col]

            if meta.source == "profiles" and _col_in_df(col, profiles_df):
                available.append(col)
            elif meta.source == "features" and _col_in_df(col, features_df):
                available.append(col)
            elif meta.source == "transitions":
                pass  # stub: never available yet

        result[group] = available

    return result


def print_feature_table(grouped: dict[str, list[str]]) -> None:
    """
    Pretty-print the output of list_available_columns to stdout.

    Example output
    --------------
    Available columns by group
    ────────────────────────────────────────────────────────────────
      bands          (profiles)  : red  green  blue  nir
      indices        (features)  : ndwi  ndvi  brightness_rgb  brightness
      derivatives    (features)  : nir_d1_smooth  nir_d1_w5  nir_d1_w7  ...
      window         (features)  : variability  slope  curvature  local_range
      detection      (features)  : spectral_angle
      transitions    —           : (none — stub; pending interpret.py)

    Group shorthands accepted by --features:
      bands  indices  derivatives  window  detection  transitions  all
    """
    print()
    print("Available columns by group")
    print("─" * 66)

    for group, cols in grouped.items():
        # Determine the source label for this group (use first column's source).
        sources = {
            COLUMN_REGISTRY[c].source
            for c in cols
            if c in COLUMN_REGISTRY
        }
        source_label = ", ".join(sorted(sources)) if sources else "—"

        if not cols:
            value_str = "(none)"
            if group == "transitions":
                value_str = "(none — stub; pending interpret.py integration)"
        else:
            value_str = "  ".join(cols)

        print(f"  {group:<14} ({source_label:<10}): {value_str}")

    print()
    print("Group shorthands accepted by --features:")
    print("  " + "  ".join(COLUMN_GROUPS.keys()))
    print()


def sample_transects(
    all_ids:     list[int],
    *,
    random_n:    Optional[int] = None,
    sample_every: Optional[int] = None,
    seed:        Optional[int] = None,
) -> list[int]:
    """
    Unified transect sampler. Returns an ordered list of transect IDs.

    Priority: explicit random_n > sample_every stride > all IDs.

    Args:
        all_ids:      Sorted list of all transect_ids in the parquet.
        random_n:     Draw this many IDs without replacement (random sample).
        sample_every: Return every Nth ID from the sorted list.
        seed:         RNG seed for random_n (None → non-reproducible).

    Returns:
        Sorted list of selected transect IDs.

    Raises:
        ValueError: If random_n exceeds the number of available transects.
    """
    if random_n is not None:
        if random_n > len(all_ids):
            raise ValueError(
                f"random_n={random_n} exceeds the number of available "
                f"transects ({len(all_ids)})."
            )
        rng = random.Random(seed)
        selected = sorted(rng.sample(all_ids, random_n))
        logger.info(
            "Random sample: %d of %d transect(s) (seed=%s).",
            random_n, len(all_ids), seed,
        )
        return selected

    if sample_every is not None:
        selected = all_ids[::sample_every]
        logger.info(
            "Stride sample: every %dth transect → %d transect(s).",
            sample_every, len(selected),
        )
        return selected

    logger.info(
        "No transect filter — using all %d transect(s). This may be slow.",
        len(all_ids),
    )
    return all_ids