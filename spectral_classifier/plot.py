"""
Plotting entry point for the spectral transect pipeline.

Reads Parquet checkpoints written by compute.py and produces:
  - Per-transect spectral profile PNGs  →  {output_dir}/transects/
  - Multi-panel overview figures         →  {output_dir}/overview/

Programmatic usage
------------------
    from spectral_classifier.plot import run_plots

    run_plots(
        year          = 2004,
        profiles_path = Path("OUTPUT/2004/profiles_2004.parquet"),
        output_dir    = Path("OUTPUT/2004/plots"),
        features_path = Path("OUTPUT/2004/features_2004.parquet"),
        random_n      = 20,                        # 20 randomly sampled transects
        features      = ["bands", "ndwi"],         # group shorthands or column names
    )

CLI usage
---------
    python -m spectral_classifier.plot --year 2004 --random-n 20 --features bands ndwi

    # List available columns for a year without plotting anything:
    python -m spectral_classifier.plot --year 2004 --list-features

Transect-selection priority
----------------------------
1. ``transect_ids``  — explicit list, used as-is.
2. ``random_n``      — N randomly sampled transects (use --seed for reproducibility).
3. ``sample_every``  — every Nth transect from the sorted full list.
4. None of the above — all transects (may be thousands; use with care).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from spectral_classifier.visualization.plotting import (
    plot_spectral_grid,
    plot_spectral_single,
    _render_spectral_axes,
)
from spectral_classifier.visualization.selector import (
    resolve_plot_columns,
    list_available_columns,
    print_feature_table,
    sample_transects,
    PlotSpec,
)
from spectral_classifier.config import PLOT_GRID_COLS, PLOT_GRID_ROWS
from spectral_classifier.utils import setup_logging

logger = logging.getLogger(__name__)

__all__ = ["run_plots", "plot_spectral_single"]


# ── Core pipeline function ─────────────────────────────────────────────────────

def run_plots(
    year:             int,
    profiles_path:    Path,
    output_dir:       Path,
    features_path:    Optional[Path] = None,
    transitions_path: Optional[Path] = None,
    transect_ids:     Optional[List[int]] = None,
    random_n:         Optional[int] = None,
    seed:             Optional[int] = None,
    sample_every:     Optional[int] = None,
    features:         Optional[List[str]] = None,
    ncols:            int = PLOT_GRID_COLS,
    nrows:            int = PLOT_GRID_ROWS,
) -> None:
    """
    Load Parquet checkpoints and produce spectral profile plots for *year*.

    Parquets are loaded once; per-transect filtering happens in-memory.
    Produces two output types:

    * **Per-transect PNGs** — one file per selected transect, saved to
      ``{output_dir}/transects/transect_{id}_profile.png``.
    * **Overview grid** — multi-panel figure(s), one PNG per page, saved to
      ``{output_dir}/overview/overview_NNN.png``.

    Args:
        year:             Survey year (used for sub-directory naming only).
        profiles_path:    Path to ``profiles_{year}.parquet``.
        output_dir:       Root output directory. Sub-directories are created
                          automatically.
        features_path:    Optional path to ``features_{year}.parquet``.
                          Required for any non-band feature columns.
        transitions_path: Optional path to ``transitions_{year}.parquet``
                          produced by interpret.py. When provided, shell line
                          positions are annotated on every plot panel.
        transect_ids:     Explicit list of transect IDs to plot.
                          Priority 1 — overrides all other selection args.
        random_n:         Plot this many randomly sampled transects.
                          Priority 2.
        seed:             RNG seed for random_n (None → non-reproducible).
        sample_every:     Plot every Nth transect from the sorted full list.
                          Priority 3.
        features:         Column names or group shorthands to plot
                          (e.g. ["bands", "ndwi", "nir_d1_smooth"]).
                          None → default rendering (bands + NIR derivative
                          overlay + shell line if transitions_path provided).
        ncols:            Overview grid columns (default from config).
        nrows:            Overview grid rows    (default from config).
    """
    profiles_path = Path(profiles_path)
    output_dir    = Path(output_dir)

    if not profiles_path.exists():
        raise FileNotFoundError(
            f"Profiles parquet not found: {profiles_path}"
        )

    # ── Load parquets ──────────────────────────────────────────────────────
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
                "feature columns will not be available.",
                features_path,
            )

    transitions_df: Optional[pd.DataFrame] = None
    if transitions_path is not None:
        transitions_path = Path(transitions_path)
        if transitions_path.exists():
            logger.info("Loading transitions from %s", transitions_path)
            transitions_df = pd.read_parquet(transitions_path)
            logger.info(
                "Loaded %d transition row(s) covering %d transect(s).",
                len(transitions_df),
                transitions_df["transect_id"].nunique(),
            )
        else:
            logger.warning(
                "transitions_path supplied but file not found: %s — "
                "shell line annotations will be skipped.",
                transitions_path,
            )

    # ── Resolve feature columns → PlotSpec ────────────────────────────────
    plot_spec: Optional[PlotSpec] = None
    if features is not None:
        plot_spec = resolve_plot_columns(
            requested=features,
            profiles_df=profiles_df,
            features_df=features_df,
            transitions_df=transitions_df,
        )
        logger.info(
            "PlotSpec resolved — primary: %s | secondary: %s",
            plot_spec.primary_cols,
            plot_spec.secondary_cols,
        )

    # ── Resolve which transects to plot ────────────────────────────────────
    all_ids = sorted(profiles_df["transect_id"].unique().tolist())

    if transect_ids is not None:
        # Explicit list — validate and filter.
        id_set = set(all_ids)
        missing = set(transect_ids) - id_set
        if missing:
            logger.warning(
                "%d requested transect_id(s) not found in profiles parquet: %s",
                len(missing), sorted(missing)[:10],
            )
        selected_ids = sorted(tid for tid in transect_ids if tid in id_set)
        logger.info("Using %d explicitly requested transect(s).", len(selected_ids))
    else:
        selected_ids = sample_transects(
            all_ids,
            random_n=random_n,
            sample_every=sample_every,
            seed=seed,
        )

    if not selected_ids:
        logger.warning("No transects selected — nothing to plot.")
        return

    # ── Output sub-directories ─────────────────────────────────────────────
    transect_dir = output_dir / "transects"
    overview_dir = output_dir / "overview"
    transect_dir.mkdir(parents=True, exist_ok=True)
    overview_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-transect PNGs ──────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    logger.info("Rendering %d per-transect PNG(s) → %s",
                len(selected_ids), transect_dir)
    failed = 0
    for tid in selected_ids:
        try:
            profile = profiles_df[profiles_df["transect_id"] == tid]

            feat_slice = None
            if features_df is not None:
                s = features_df[features_df["transect_id"] == tid]
                feat_slice = s if not s.empty else None

            trans_slice = None
            if transitions_df is not None:
                t = transitions_df[transitions_df["transect_id"] == tid]
                trans_slice = t if not t.empty else None

            fig, ax = plt.subplots(figsize=(12, 5))
            _render_spectral_axes(ax, profile, feat_slice, trans_slice,
                                  title=f"Transect {tid}  ({year})",
                                  plot_spec=plot_spec)
            out = transect_dir / f"transect_{tid}_profile.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)

        except Exception:
            logger.exception("Failed to plot transect %s — skipping.", tid)
            failed += 1

    if failed:
        logger.warning("%d transect(s) failed to render.", failed)

    # ── Overview grid ──────────────────────────────────────────────────────
    logger.info("Rendering overview grid → %s", overview_dir)
    overview_paths = plot_spectral_grid(
        transect_ids=selected_ids,
        profiles_df=profiles_df,
        output_dir=overview_dir,
        features_df=features_df,
        transitions_df=transitions_df,
        ncols=ncols,
        nrows=nrows,
        plot_spec=plot_spec,
    )

    logger.info(
        "run_plots complete for %d: %d transect PNG(s), %d overview page(s).",
        year, len(selected_ids) - failed, len(overview_paths),
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m spectral_classifier.plot",
        description=(
            "Render spectral profile plots from Parquet checkpoints.\n\n"
            "Transect-selection priority: --transect-ids > --random-n "
            "> --sample-every > all."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--output", "-o",
        type=Path, default=Path("OUTPUT"),
        help="Root output directory (default: OUTPUT).",
    )
    p.add_argument(
        "--year", "-y",
        type=str, required=True,
        help="Survey year to plot (e.g. 2012). Expects parquets at "
             "{output}/{year}/profiles_{year}.parquet etc.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    # ── Transect selection ─────────────────────────────────────────────────
    sel = p.add_argument_group(
        "transect selection",
        "Priority: --transect-ids > --random-n > --sample-every > all",
    )
    sel.add_argument(
        "--transect-ids",
        type=int, nargs="+", metavar="ID", default=None,
        help="Explicit list of transect IDs to plot.",
    )
    sel.add_argument(
        "--random-n",
        type=int, metavar="N", default=None,
        help="Plot N randomly sampled transects.",
    )
    sel.add_argument(
        "--seed",
        type=int, default=None,
        help="RNG seed for --random-n (omit for non-reproducible results).",
    )
    sel.add_argument(
        "--sample-every",
        type=int, metavar="N", default=None,
        help="Plot every Nth transect from the sorted list.",
    )

    # ── Feature selection ──────────────────────────────────────────────────
    feat = p.add_argument_group("feature selection")
    feat.add_argument(
        "--features",
        nargs="+", metavar="COL_OR_GROUP", default=None,
        help=(
            "Columns or group shorthands to plot. "
            "Groups: bands  indices  derivatives  window  detection  transitions  all. "
            "Individual column names are also accepted "
            "(e.g. --features bands ndwi nir_d1_smooth). "
            "Default: all available bands + NIR derivative overlay."
        ),
    )
    feat.add_argument(
        "--list-features",
        action="store_true",
        help=(
            "Print available columns for the given year and exit without "
            "rendering any plots. Loads parquets but performs no computation."
        ),
    )
    feat.add_argument(
        "--transitions-path",
        type=Path, default=None, metavar="PATH",
        help=(
            "Path to transitions_{year}.parquet produced by interpret.py. "
            "When omitted, the default location "
            "{output}/{year}/transitions_{year}.parquet is checked automatically. "
            "Shell line annotations are shown whenever this file is present."
        ),
    )

    # ── Grid layout ────────────────────────────────────────────────────────
    grid = p.add_argument_group("grid layout")
    grid.add_argument(
        "--plot-cols",
        type=int, default=PLOT_GRID_COLS, metavar="N",
        help=f"Overview grid columns (default: {PLOT_GRID_COLS}).",
    )
    grid.add_argument(
        "--plot-rows",
        type=int, default=PLOT_GRID_ROWS, metavar="N",
        help=f"Overview grid rows (default: {PLOT_GRID_ROWS}).",
    )

    return p.parse_args()


def _resolve_parquet_paths(
    output: Path, year: str
) -> tuple[Path, Optional[Path], Optional[Path]]:
    """
    Return (profiles_path, features_path, transitions_path) for a year directory.

    features_path and transitions_path are None when the files do not exist,
    so callers can safely skip loading without an extra existence check.
    """
    year_dir          = output / year
    profiles_path     = year_dir / f"profiles_{year}.parquet"
    features_path     = year_dir / f"features_{year}.parquet"
    transitions_path  = year_dir / f"transitions_{year}.parquet"
    return (
        profiles_path,
        features_path    if features_path.exists()    else None,
        transitions_path if transitions_path.exists() else None,
    )


if __name__ == "__main__":
    args = _parse_args()

    setup_logging(verbose=args.verbose)

    profiles_path, features_path, auto_transitions_path = _resolve_parquet_paths(
        args.output, args.year
    )

    if not profiles_path.exists():
        print(
            f"ERROR: profiles parquet not found: {profiles_path}\n"
            f"       Run 'python -m spectral_classifier.run --mode compute "
            f"--year {args.year}' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --transitions-path overrides auto-discovery; fall back to the standard
    # year-directory location when the flag is not supplied.
    transitions_path = args.transitions_path or auto_transitions_path
    if transitions_path is not None:
        print(f"  [plot] Transitions: {transitions_path}")
    else:
        print("  [plot] Transitions: not found — shell line annotations skipped.")

    # ── --list-features: print available columns and exit ──────────────────
    if args.list_features:
        profiles_df    = pd.read_parquet(profiles_path)
        features_df    = pd.read_parquet(features_path)    if features_path    else None
        transitions_df = pd.read_parquet(transitions_path) if transitions_path else None

        grouped = list_available_columns(profiles_df, features_df, transitions_df)
        print_feature_table(grouped)
        sys.exit(0)

    # ── Normal plot run ────────────────────────────────────────────────────
    plots_dir = args.output / args.year / "plots"

    run_plots(
        year             = int(args.year),
        profiles_path    = profiles_path,
        output_dir       = plots_dir,
        features_path    = features_path,
        transitions_path = transitions_path,
        transect_ids     = args.transect_ids,
        random_n         = args.random_n,
        seed             = args.seed,
        sample_every     = args.sample_every,
        features         = args.features,
        ncols            = args.plot_cols,
        nrows            = args.plot_rows,
    )