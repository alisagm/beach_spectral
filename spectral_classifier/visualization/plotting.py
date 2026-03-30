"""
Visualization module for spectral transect analysis.

X-axis convention
-----------------
All plot functions use ``distance`` (arc-length from the lowest-easting /
west end of the transect, as produced by ``sampler.sample_transect``) as the
x-axis.  West is always on the left, east always on the right.

Public API
----------
plot_spectral_single(transect_id, profiles_path, output_dir, features_path)
    Convenience wrapper: loads parquet, filters one transect, renders PNG.
    Use for ad-hoc inspection.

plot_spectral_grid(transect_ids, profiles_df, output_dir, features_df)
    Renders a multi-panel overview figure (ncols × nrows per page).
    Caller supplies pre-loaded DataFrames; no I/O inside this function.
    Use from run_plots() where the full parquet is already in memory.

Private helpers
---------------
_render_spectral_axes   — pure matplotlib; no I/O; called by both public fns
_plot_nir_derivative    — secondary y-axis overlay; called by _render_spectral_axes
_get_spectral_ylim      — robust y-limit calculation
_safe_percentile        — NaN/Inf-safe percentile
"""

import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import PLOT_GRID_COLS, PLOT_GRID_ROWS

logger = logging.getLogger(__name__)

# ── Band display config ────────────────────────────────────────────────────────

_BAND_STYLE = {
    "red":   dict(color="red",     linewidth=1.5, label="Red",   alpha=0.85),
    "green": dict(color="green",   linewidth=1.5, label="Green", alpha=0.85),
    "blue":  dict(color="steelblue", linewidth=1.5, label="Blue",  alpha=0.85),
    "nir":   dict(color="darkred", linewidth=1.5, label="NIR",   alpha=0.85,
                  linestyle="--"),
}

_NIR_DERIV_COLOR  = "darkorange"
_NIR_DERIV_THRESH = -3.0          # same threshold used in TransitionDetector


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
    Compute robust y-axis limits from spectral band columns.

    Uses 1st–99th percentile range with a small margin, falling back to
    (0, 255) if no valid values exist.
    """
    all_values: List[float] = []
    for band in ("red", "green", "blue", "nir"):
        if band in profile_df.columns:
            vals = profile_df[band].values
            all_values.extend(vals[np.isfinite(vals)])

    if not all_values:
        logger.warning("No valid spectral values; using default y-limits (0, 255)")
        return (0.0, 255.0)

    arr = np.array(all_values)
    y_min = max(0.0, _safe_percentile(arr, 1, 0.0) - 20.0)
    y_max = _safe_percentile(arr, 99, 255.0) + 20.0

    if y_max <= y_min:
        y_max = y_min + 100.0

    return (y_min, y_max)


def _plot_nir_derivative(ax: plt.Axes,
                         features_df: pd.DataFrame) -> plt.Axes:
    """
    Overlay the smoothed NIR first derivative on a secondary y-axis.

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

    ax2.set_ylabel("NIR Derivative (units/m)", fontsize=10,
                   fontweight="bold", color=_NIR_DERIV_COLOR)
    ax2.tick_params(axis="y", labelcolor=_NIR_DERIV_COLOR)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.85)

    return ax2


def _render_spectral_axes(
    ax: plt.Axes,
    profile_df: pd.DataFrame,
    features_df: Optional[pd.DataFrame] = None,
    title: str = "",
) -> Optional[plt.Axes]:
    """
    Draw spectral band profiles onto *ax*.  Pure matplotlib — no I/O.

    Plots whichever of red/green/blue/nir are non-null in *profile_df*.
    If *features_df* is supplied and contains ``nir_d1_smooth``, a secondary
    derivative overlay is added via :func:`_plot_nir_derivative`.

    Args:
        ax:          Matplotlib axes to draw on.
        profile_df:  Rows for a single transect; must contain ``distance``.
        features_df: Optional features rows for the same transect.
        title:       Axes title string.

    Returns:
        Secondary axes if a derivative overlay was drawn, else None.
    """
    if profile_df.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=9)
        return None

    distance = profile_df["distance"]
    y_min, y_max = _get_spectral_ylim(profile_df)
    ax.set_ylim(y_min, y_max)

    for band, style in _BAND_STYLE.items():
        if band not in profile_df.columns:
            continue
        col = profile_df[band]
        if col.isna().all():
            continue
        ax.plot(distance, col, **style)

    ax2 = None
    if (
        features_df is not None
        and not features_df.empty
        and "nir_d1_smooth" in features_df.columns
    ):
        ax2 = _plot_nir_derivative(ax, features_df)

    ax.set_xlabel("Distance from west (m)", fontsize=9)
    ax.set_ylabel("Spectral value (DN)", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=8)

    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)

    legend_loc = "upper left" if ax2 is not None else "upper right"
    ax.legend(loc=legend_loc, fontsize=8, framealpha=0.85)

    return ax2


# ── Public functions ───────────────────────────────────────────────────────────

def plot_spectral_single(
    transect_id: int,
    profiles_path: Path,
    output_dir: Path,
    features_path: Optional[Path] = None,
    figsize: tuple = (12, 5),
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    _render_spectral_axes(ax, profile, features,
                          title=f"Transect {transect_id}")

    output_path = output_dir / f"transect_{transect_id}_profile.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Saved %s", output_path)
    return output_path


def plot_spectral_grid(
    transect_ids: List[int],
    profiles_df: pd.DataFrame,
    output_dir: Path,
    features_df: Optional[pd.DataFrame] = None,
    ncols: int = PLOT_GRID_COLS,
    nrows: int = PLOT_GRID_ROWS,
) -> List[Path]:
    """
    Render multi-panel overview figures for a list of transects.

    Paginates into figures of *ncols* × *nrows* panels.  Caller is
    responsible for loading parquets and filtering *transect_ids* to the
    desired sample before calling this function.

    Args:
        transect_ids: Ordered list of transect IDs to plot.
        profiles_df:  Full (or pre-filtered) profiles DataFrame; will be
                      filtered per transect_id inside this function.
        output_dir:   Directory to write ``overview_NNN.png`` files into.
        features_df:  Optional features DataFrame (same structure).
        ncols:        Grid columns per page  (default from config).
        nrows:        Grid rows per page     (default from config).

    Returns:
        List of paths to saved overview PNGs (one per page).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panels_per_page = ncols * nrows
    saved: List[Path] = []

    # Chunk transect_ids into pages
    pages = [
        transect_ids[i : i + panels_per_page]
        for i in range(0, len(transect_ids), panels_per_page)
    ]

    for page_idx, page_ids in enumerate(pages, start=1):
        n_panels = len(page_ids)
        # Always create a full ncols × nrows grid; blank unused panels
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

            profile  = profiles_df[profiles_df["transect_id"] == tid]
            features = None
            if features_df is not None:
                feat_slice = features_df[features_df["transect_id"] == tid]
                features   = feat_slice if not feat_slice.empty else None

            _render_spectral_axes(ax, profile, features,
                                  title=f"T{tid}")

        # Hide any unused axes in the last page
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