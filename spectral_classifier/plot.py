"""
Plotting entry point for the spectral transect pipeline.

Reads Parquet checkpoints written by compute.py and produces:
  - Per-transect spectral profile PNGs  →  plots/{year}/transects/
  - Multi-panel overview figures         →  plots/{year}/overview/

Usage (CLI via run.py / main.py)
---------------------------------
    from spectral_classifier.plot import run_plots

    run_plots(
        year          = 2004,
        profiles_path = Path("OUTPUT/profiles_2004.parquet"),
        output_dir    = Path("OUTPUT/plots"),
        features_path = Path("OUTPUT/features_2004.parquet"),   # optional
        transect_ids  = [100, 200, 300],   # explicit list  — or —
        sample_every  = 50,                # every Nth transect  — or —
        # neither → all transects (slow)
    )

Transect-selection priority
----------------------------
1. ``transect_ids``  — explicit list, used as-is.
2. ``sample_every``  — every Nth transect from the sorted full list.
3. Neither supplied  — all transects (may be thousands; use with care).
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from spectral_classifier.visualization.plotting import (
    plot_spectral_grid,
    plot_spectral_single,   # re-exported so callers need only import plot.py
)
from spectral_classifier.config import PLOT_GRID_COLS, PLOT_GRID_ROWS

logger = logging.getLogger(__name__)

__all__ = ["run_plots", "plot_spectral_single"]


def _resolve_transect_ids(
    all_ids: List[int],
    transect_ids: Optional[List[int]],
    sample_every: Optional[int],
) -> List[int]:
    """
    Apply transect-selection logic and return the final ordered list.

    Priority: explicit list > sample_every stride > all IDs.

    Args:
        all_ids:      Sorted list of every transect_id in the parquet.
        transect_ids: Caller-supplied explicit list, or None.
        sample_every: Keep every Nth transect, or None.

    Returns:
        List of transect IDs to plot.
    """
    if transect_ids is not None:
        missing = set(transect_ids) - set(all_ids)
        if missing:
            logger.warning(
                "%d requested transect_ids not found in profiles parquet: %s",
                len(missing), sorted(missing)[:10],
            )
        found = [tid for tid in transect_ids if tid in set(all_ids)]
        logger.info("Using %d explicitly requested transect(s).", len(found))
        return found

    if sample_every is not None:
        sampled = all_ids[::sample_every]
        logger.info(
            "Sampling every %dth transect → %d transect(s).",
            sample_every, len(sampled),
        )
        return sampled

    logger.info(
        "No transect filter supplied — plotting all %d transect(s). "
        "This may be slow.",
        len(all_ids),
    )
    return all_ids


def run_plots(
    year: int,
    profiles_path: Path,
    output_dir: Path,
    features_path: Optional[Path] = None,
    transitions_path: Optional[Path] = None,
    transect_ids: Optional[List[int]] = None,
    sample_every: Optional[int] = None,
    ncols: int = PLOT_GRID_COLS,
    nrows: int = PLOT_GRID_ROWS,
) -> None:
    """
    Load Parquet checkpoints and produce spectral profile plots for *year*.

    Parquets are loaded once; per-transect filtering happens in-memory.
    Produces two output types:

    * **Per-transect PNGs** — one file per selected transect, saved to
      ``{output_dir}/{year}/transects/transect_{id}_profile.png``.
    * **Overview grid** — multi-panel figure(s), one PNG per page, saved to
      ``{output_dir}/{year}/overview/overview_NNN.png``.

    Args:
        year:             Survey year (used for sub-directory naming only).
        profiles_path:    Path to ``profiles_{year}.parquet``.
        output_dir:       Root output directory.  Sub-directories are created
                          automatically.
        features_path:    Optional path to ``features_{year}.parquet``.
                          When supplied and the file exists, ``nir_d1_smooth``
                          is overlaid on each panel.
        transitions_path: Reserved for future use (interpret.py outputs).
                          Accepted but currently ignored.
        transect_ids:     Explicit list of transect IDs to plot.  Mutually
                          exclusive with *sample_every* (this wins if both
                          are supplied).
        sample_every:     Plot every Nth transect from the full sorted list.
        ncols:            Overview grid columns (default from config).
        nrows:            Overview grid rows    (default from config).
    """
    profiles_path = Path(profiles_path)
    output_dir    = Path(output_dir)

    if not profiles_path.exists():
        raise FileNotFoundError(
            f"Profiles parquet not found: {profiles_path}"
        )

    # ── Load parquets once ────────────────────────────────────────────────
    logger.info("Loading profiles from %s", profiles_path)
    profiles_df = pd.read_parquet(profiles_path)

    features_df: Optional[pd.DataFrame] = None
    if features_path is not None:
        features_path = Path(features_path)
        if features_path.exists():
            logger.info("Loading features from %s", features_path)
            features_df = pd.read_parquet(features_path)
        else:
            logger.warning(
                "features_path supplied but file not found: %s — "
                "derivative overlay will be skipped.",
                features_path,
            )

    if transitions_path is not None:
        logger.info(
            "transitions_path=%s received but transition overlays are not "
            "yet implemented (pending interpret.py integration).",
            transitions_path,
        )

    # ── Resolve which transects to plot ───────────────────────────────────
    all_ids = sorted(profiles_df["transect_id"].unique().tolist())
    selected_ids = _resolve_transect_ids(all_ids, transect_ids, sample_every)

    if not selected_ids:
        logger.warning("No transects selected — nothing to plot.")
        return

    # ── Output sub-directories ────────────────────────────────────────────
    transect_dir = output_dir / str(year) / "transects"
    overview_dir = output_dir / str(year) / "overview"
    transect_dir.mkdir(parents=True, exist_ok=True)
    overview_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-transect PNGs ─────────────────────────────────────────────────
    logger.info("Rendering %d per-transect PNG(s) → %s", len(selected_ids), transect_dir)
    failed = 0
    for tid in selected_ids:
        try:
            profile  = profiles_df[profiles_df["transect_id"] == tid]
            features = None
            if features_df is not None:
                feat_slice = features_df[features_df["transect_id"] == tid]
                features   = feat_slice if not feat_slice.empty else None

            # Import here to avoid circular-import risk at module level
            from spectral_classifier.visualization.plotting import (
                _render_spectral_axes,
            )
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 5))
            _render_spectral_axes(ax, profile, features,
                                  title=f"Transect {tid}  ({year})")
            out = transect_dir / f"transect_{tid}_profile.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)

        except Exception:
            logger.exception("Failed to plot transect %s — skipping.", tid)
            failed += 1

    if failed:
        logger.warning("%d transect(s) failed to render.", failed)

    # ── Overview grid ─────────────────────────────────────────────────────
    logger.info("Rendering overview grid → %s", overview_dir)
    overview_paths = plot_spectral_grid(
        transect_ids = selected_ids,
        profiles_df  = profiles_df,
        output_dir   = overview_dir,
        features_df  = features_df,
        ncols        = ncols,
        nrows        = nrows,
    )

    logger.info(
        "run_plots complete for %d: %d transect PNG(s), %d overview page(s).",
        year, len(selected_ids) - failed, len(overview_paths),
    )