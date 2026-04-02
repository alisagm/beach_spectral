"""
Visualization module for spectral transect analysis.

X-axis convention
-----------------
All plot functions use ``distance`` (arc-length from the lowest-easting /
west end of the transect, as produced by ``sampler.sample_transect``) as the
x-axis.  West is always on the left, east always on the right.

Public API
----------
plot_spectral_single(transect_id, profiles_path, output_dir, features_path,
                     plot_spec)
    Convenience wrapper: loads parquet, filters one transect, renders PNG.
    Use for ad-hoc inspection.

plot_spectral_grid(transect_ids, profiles_df, output_dir, features_df,
                   plot_spec)
    Renders a multi-panel overview figure (ncols × nrows per page).
    Caller supplies pre-loaded DataFrames; no I/O inside this function.
    Use from run_plots() where the full parquet is already in memory.

Private helpers
---------------
_render_spectral_axes   — pure matplotlib; no I/O; called by both public fns
_plot_nir_derivative    — default-path secondary overlay (no PlotSpec)
_get_spectral_ylim      — robust y-limit calculation for DN columns
_get_column_ylim        — generalised y-limit for arbitrary column list
_plot_columns_on_axis   — render a list of columns onto a given Axes
_safe_percentile        — NaN/Inf-safe percentile
"""

import logging
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import PLOT_GRID_COLS, PLOT_GRID_ROWS

if TYPE_CHECKING:
    from .selector import PlotSpec

logger = logging.getLogger(__name__)

# ── Default band display config (used by the no-spec rendering path) ───────────

_BAND_STYLE = {
    "red":   dict(color="red",       linewidth=1.5, label="Red",   alpha=0.85),
    "green": dict(color="green",     linewidth=1.5, label="Green", alpha=0.85),
    "blue":  dict(color="steelblue", linewidth=1.5, label="Blue",  alpha=0.85),
    "nir":   dict(color="darkred",   linewidth=1.5, label="NIR",   alpha=0.85,
                  linestyle="--"),
}

_NIR_DERIV_COLOR  = "darkorange"
_NIR_DERIV_THRESH = -3.0   # same threshold used in TransitionDetector


# ── Private helpers ────────────────────────────────────────────────────────────

def _safe_percentile(values: np.ndarray, percentile: float,
                     default: float = 0.0) -> float:
    """Return *percentile* of *values*, ignoring NaN/Inf; fall back to *default*."""
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        logger.warning(
            "No valid values for percentile calculation; using default=%s", default
        )
        return default
    return float(np.percentile(valid, percentile))


def _get_spectral_ylim(profile_df: pd.DataFrame) -> tuple[float, float]:
    """
    Compute robust y-axis limits from the spectral band columns in *profile_df*.

    Uses 1st–99th percentile range with a small margin, falling back to
    (0, 255) if no valid values exist.  Only inspects red/green/blue/nir.
    """
    all_values: List[float] = []
    for band in ("red", "green", "blue", "nir"):
        if band in profile_df.columns:
            vals = profile_df[band].values
            all_values.extend(vals[np.isfinite(vals)])

    if not all_values:
        logger.warning("No valid spectral band values; using default y-limits (0, 255)")
        return (0.0, 255.0)

    arr = np.array(all_values)
    y_min = max(0.0, _safe_percentile(arr, 1, 0.0) - 20.0)
    y_max = _safe_percentile(arr, 99, 255.0) + 20.0

    if y_max <= y_min:
        y_max = y_min + 100.0

    return (y_min, y_max)


def _get_column_ylim(
    data: dict[str, pd.Series],
    cols: list[str],
) -> tuple[float, float]:
    """
    Compute robust y-axis limits for an arbitrary set of columns.

    Args:
        data: Mapping of column name → Series (already filtered to one transect).
        cols: Column names to include.

    Returns:
        (y_min, y_max) tuple safe for ax.set_ylim().
    """
    all_values: List[float] = []
    for col in cols:
        if col in data:
            vals = data[col].values
            all_values.extend(vals[np.isfinite(vals)])

    if not all_values:
        return (-1.0, 1.0)   # sensible fallback for normalized/derivative data

    arr = np.array(all_values)
    y_min = _safe_percentile(arr, 1, arr.min())
    y_max = _safe_percentile(arr, 99, arr.max())
    margin = max((y_max - y_min) * 0.1, 1e-6)

    return (y_min - margin, y_max + margin)


def _plot_columns_on_axis(
    ax:       plt.Axes,
    distance: pd.Series,
    data:     dict[str, pd.Series],
    cols:     list[str],
) -> None:
    """
    Plot a list of columns onto *ax* using styles from COLUMN_REGISTRY.

    Columns absent from *data* or entirely NaN are silently skipped (they
    were already warned about during resolve_plot_columns).

    Args:
        ax:       Matplotlib axes to draw on.
        distance: x-axis values (distance from west, metres).
        data:     Mapping of column name → per-transect Series.
        cols:     Ordered list of column names to plot.
    """
    # Import here to avoid a circular import at module load time.
    # selector imports nothing from plotting, so this is one-directional.
    from .selector import COLUMN_REGISTRY

    for col in cols:
        if col not in data:
            continue
        series = data[col]
        if series.isna().all():
            continue

        style = COLUMN_REGISTRY[col].style if col in COLUMN_REGISTRY else {}
        ax.plot(distance, series, **style)


def _plot_nir_derivative(ax: plt.Axes,
                         features_df: pd.DataFrame) -> plt.Axes:
    """
    Overlay the smoothed NIR first derivative on a secondary y-axis.

    This helper is used only by the default (no-PlotSpec) rendering path.
    When a PlotSpec is supplied, derivative columns are rendered uniformly
    by _plot_columns_on_axis.

    Args:
        ax:          Primary spectral axes.
        features_df: DataFrame containing ``distance`` and ``nir_d1_smooth``.

    Returns:
        The secondary axes (twinx), even if no valid data were found.
    """
    ax2 = ax.twinx()

    nir_d1 = features_df["nir_d1_smooth"]
    valid   = nir_d1[np.isfinite(nir_d1)]

    if len(valid) == 0:
        logger.warning("No valid NIR derivative values to plot")
        return ax2

    ax2.plot(
        features_df["distance"], nir_d1,
        color=_NIR_DERIV_COLOR, linewidth=1.5, alpha=0.7,
        label="NIR d/dx", linestyle="-.",
    )
    ax2.axhline(_NIR_DERIV_THRESH, color="red",  linestyle="--",
                linewidth=1.0, alpha=0.5, label="Boundary threshold")
    ax2.axhline(0,                 color="gray", linestyle=":",
                linewidth=0.5, alpha=0.5)

    abs_max = max(abs(valid.max()), abs(valid.min()))
    if abs_max == 0 or not np.isfinite(abs_max):
        abs_max = 10.0
    ax2.set_ylim(-abs_max * 1.2, abs_max * 1.2)

    ax2.set_ylabel("NIR derivative (DN/m)", fontsize=9, color=_NIR_DERIV_COLOR)
    ax2.tick_params(axis="y", labelcolor=_NIR_DERIV_COLOR, labelsize=8)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.85)

    return ax2


def _render_spectral_axes(
    ax:          plt.Axes,
    profile_df:  pd.DataFrame,
    features_df: Optional[pd.DataFrame] = None,
    *,
    title:     Optional[str] = None,
    plot_spec: Optional["PlotSpec"] = None,
) -> Optional[plt.Axes]:
    """
    Draw spectral profiles onto *ax*.  Pure matplotlib — no I/O.

    Default path (plot_spec=None)
    ------------------------------
    Plots whichever of red/green/blue/nir are non-null in *profile_df*.
    If *features_df* contains ``nir_d1_smooth``, a secondary derivative
    overlay is added via _plot_nir_derivative.

    Spec path (plot_spec supplied)
    -------------------------------
    Plots exactly the columns listed in plot_spec.primary_cols and
    plot_spec.secondary_cols, sourced from profile_df / features_df as
    indicated by each column's COLUMN_REGISTRY entry.  Y-axis labels are
    set dynamically from the scale families present.

    Args:
        ax:          Matplotlib axes to draw on.
        profile_df:  Rows for a single transect; must contain ``distance``.
        features_df: Optional features rows for the same transect.
        title:       Axes title string.
        plot_spec:   Pre-resolved PlotSpec from resolve_plot_columns(), or
                     None to use the default band + NIR-derivative rendering.

    Returns:
        Secondary axes (twinx) if one was created, else None.

    Note:
        Binary columns (has_oscillations, has_rgb_foam_peak) are excluded
        from PlotSpec.primary_cols / secondary_cols and therefore never
        reach this function.
        TODO: render binary columns as axvspan shaded regions once the
        rendering layer supports mixed continuous + discrete series.
    """

    # ── Shared setup ───────────────────────────────────────────────────────
    if profile_df.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=9)
        return None

    distance = profile_df["distance"]

    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xlabel("Distance from west (m)", fontsize=9)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)

    # ── Default path ───────────────────────────────────────────────────────
    if plot_spec is None:
        y_min, y_max = _get_spectral_ylim(profile_df)
        ax.set_ylim(y_min, y_max)

        for band, style in _BAND_STYLE.items():
            if band not in profile_df.columns:
                continue
            col = profile_df[band]
            if col.isna().all():
                continue
            ax.plot(distance, col, **style)

        ax.set_ylabel("Spectral value (DN)", fontsize=9)

        ax2 = None
        if (
            features_df is not None
            and not features_df.empty
            and "nir_d1_smooth" in features_df.columns
        ):
            ax2 = _plot_nir_derivative(ax, features_df)

        legend_loc = "upper left" if ax2 is not None else "upper right"
        ax.legend(loc=legend_loc, fontsize=8, framealpha=0.85)
        return ax2

    # ── Spec path ──────────────────────────────────────────────────────────
    # Build a flat column → Series lookup from whichever sources are available.
    data: dict[str, pd.Series] = {}

    for col in plot_spec.profile_cols:
        if col in profile_df.columns:
            data[col] = profile_df[col]

    if features_df is not None and not features_df.empty:
        for col in plot_spec.feature_cols:
            if col in features_df.columns:
                data[col] = features_df[col]

    if not data:
        ax.text(0.5, 0.5, "no columns available", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=9)
        return None

    # ── Primary axis ───────────────────────────────────────────────────────
    from .selector import COLUMN_REGISTRY, _axis_label

    ax2: Optional[plt.Axes] = None

    if plot_spec.primary_cols:
        _plot_columns_on_axis(ax, distance, data, plot_spec.primary_cols)

        primary_label = _axis_label(plot_spec.primary_cols)
        ax.set_ylabel(primary_label, fontsize=9)

        # Force a stable y-limit only for DN-scale primaries; let matplotlib
        # auto-scale for normalized / derivative primaries.
        from .selector import COLUMN_REGISTRY as _REG
        primary_is_dn = any(
            _REG.get(c) is not None and _REG[c].scale == "dn"
            for c in plot_spec.primary_cols
        )
        if primary_is_dn:
            # Re-use spectral ylim logic if bands are present, otherwise
            # derive from the actual column data.
            band_cols = [c for c in plot_spec.primary_cols
                         if c in ("red", "green", "blue", "nir")]
            if band_cols:
                y_min, y_max = _get_spectral_ylim(profile_df)
            else:
                y_min, y_max = _get_column_ylim(data, plot_spec.primary_cols)
            ax.set_ylim(y_min, y_max)

        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # ── Secondary axis ─────────────────────────────────────────────────────
    if plot_spec.secondary_cols:
        ax2 = ax.twinx()
        _plot_columns_on_axis(ax2, distance, data, plot_spec.secondary_cols)

        secondary_label = _axis_label(plot_spec.secondary_cols)
        ax2.set_ylabel(secondary_label, fontsize=9)
        ax2.tick_params(axis="y", labelsize=8)
        ax2.legend(loc="upper right", fontsize=8, framealpha=0.85)

    return ax2


# ── Public functions ───────────────────────────────────────────────────────────

def plot_spectral_single(
    transect_id:   int,
    profiles_path: Path,
    output_dir:    Path,
    features_path: Optional[Path] = None,
    figsize:       tuple = (12, 5),
    plot_spec:     Optional["PlotSpec"] = None,
) -> Optional[Path]:
    """
    Render and save a spectral profile PNG for one transect.

    Loads *profiles_path* (and optionally *features_path*) each call.
    Intended for interactive / ad-hoc use.  For batch rendering, supply
    pre-loaded DataFrames to :func:`_render_spectral_axes` directly, or
    use :func:`plot_spectral_grid`.

    Args:
        transect_id:   Integer transect identifier.
        profiles_path: Path to ``profiles_{year}.parquet``.
        output_dir:    Directory to write the PNG into.
        features_path: Optional path to ``features_{year}.parquet``.
        figsize:       Matplotlib figure size.
        plot_spec:     Pre-resolved PlotSpec, or None for default rendering.

    Returns:
        Path to the saved PNG, or None if the transect has no data.
    """
    profiles_df = pd.read_parquet(profiles_path)
    profile     = profiles_df[profiles_df["transect_id"] == transect_id]

    if profile.empty:
        logger.warning("Transect %s not found in %s", transect_id, profiles_path)
        return None

    features = None
    if features_path is not None and Path(features_path).exists():
        features_df = pd.read_parquet(features_path)
        features    = features_df[features_df["transect_id"] == transect_id]
        if features.empty:
            features = None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    _render_spectral_axes(ax, profile, features,
                          title=f"Transect {transect_id}",
                          plot_spec=plot_spec)

    output_path = output_dir / f"transect_{transect_id}_profile.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Saved %s", output_path)
    return output_path


def plot_spectral_grid(
    transect_ids: List[int],
    profiles_df:  pd.DataFrame,
    output_dir:   Path,
    features_df:  Optional[pd.DataFrame] = None,
    ncols:        int = PLOT_GRID_COLS,
    nrows:        int = PLOT_GRID_ROWS,
    plot_spec:    Optional["PlotSpec"] = None,
) -> List[Path]:
    """
    Render multi-panel overview figures for a list of transects.

    Paginates into figures of *ncols* × *nrows* panels.  Caller is
    responsible for loading parquets and filtering *transect_ids* to the
    desired sample before calling this function.

    Args:
        transect_ids: Ordered list of transect IDs to plot.
        profiles_df:  Full (or pre-filtered) profiles DataFrame.
        output_dir:   Directory to write ``overview_NNN.png`` files into.
        features_df:  Optional features DataFrame.
        ncols:        Grid columns per page (default from config).
        nrows:        Grid rows per page    (default from config).
        plot_spec:    Pre-resolved PlotSpec, or None for default rendering.

    Returns:
        List of paths to saved overview PNGs (one per page).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panels_per_page = ncols * nrows
    saved: List[Path] = []

    pages = [
        transect_ids[i : i + panels_per_page]
        for i in range(0, len(transect_ids), panels_per_page)
    ]

    for page_idx, page_ids in enumerate(pages, start=1):
        n_panels = len(page_ids)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 4.5, nrows * 3.0),
            squeeze=False,
        )
        fig.suptitle(
            f"Spectral profiles — page {page_idx} of {len(pages)}",
            fontsize=11, fontweight="bold", y=1.01,
        )

        for panel_idx, tid in enumerate(page_ids):
            row, col = divmod(panel_idx, ncols)
            ax = axes[row][col]

            profile = profiles_df[profiles_df["transect_id"] == tid]
            features = None
            if features_df is not None:
                feat_slice = features_df[features_df["transect_id"] == tid]
                features   = feat_slice if not feat_slice.empty else None

            _render_spectral_axes(ax, profile, features,
                                  title=f"T{tid}",
                                  plot_spec=plot_spec)

        # Hide unused panels on the last page.
        for panel_idx in range(n_panels, panels_per_page):
            row, col = divmod(panel_idx, ncols)
            axes[row][col].set_visible(False)

        fig.tight_layout()
        out_path = output_dir / f"overview_{page_idx:03d}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        logger.debug("Saved overview page %d → %s", page_idx, out_path)
        saved.append(out_path)

    return saved