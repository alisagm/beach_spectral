#!/usr/bin/env python3
"""
Spectral Shoreline Detection System — Compute + Plot Steps

Samples spectral profiles along all transects, extracts feature statistics,
and optionally renders diagnostic plots.

Modes
-----
all      Run compute then plot (default).
compute  Profile sampling and feature extraction only.
plot     Plot from existing Parquet checkpoints (skips compute).

Usage examples
--------------
# Full pipeline, all years:
python run.py --imagery-root /path/to/imagery

# Compute only (skip plotting):
python run.py --imagery-root /path/to/imagery --mode compute

# (Re)plot a specific year from existing checkpoints:
python run.py --imagery-root /path/to/imagery --mode plot --year 2004

# Plot a sample of transects (every 50th) instead of all:
python run.py --imagery-root /path/to/imagery --mode plot --sample-every 50

# Plot a hand-picked set of transects:
python run.py --imagery-root /path/to/imagery --mode plot --transect-ids 100 500 1000

# Force recompute even if checkpoints already exist:
python run.py --imagery-root /path/to/imagery --mode compute --force

Output
------
For each year processed:
    OUTPUT/{year}/profiles_{year}.parquet    <- raw band samples + coordinates
    OUTPUT/{year}/features_{year}.parquet    <- derived feature arrays
    OUTPUT/{year}/plots/transects/*.png      <- per-transect spectral profiles
    OUTPUT/{year}/plots/overview/*.png       <- multi-panel overview grid(s)
    OUTPUT/{year}/compute.log               <- debug log
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from .compute import run_compute
from .interpret import run_interpret
from spectral_classifier.utils import (
    setup_logging,
    group_rasters_by_year,
    validate_output_directory,
    bundle_shell_lines_to_geojson,
    band_mode_label,
    band_mode_from_indices,
    load_band_config,
    resolve_band_indices,
)
from spectral_classifier.config import BAND_CONFIG_PATH, PLOT_GRID_COLS, PLOT_GRID_ROWS


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Spectral Shoreline Detection System — Compute + Plot Steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Common args ────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode", "-m",
        choices=["all", "compute", "interpret", "plot"],
        default="all",
        help=(
            "Pipeline step(s) to run: "
            "'all' = compute → interpret → plot (default), "
            "'compute' = profile sampling and feature extraction only, "
            "'interpret' = transition detection from existing feature Parquets, "
            "'plot' = render plots from existing Parquet checkpoints"
        ),
    )

    parser.add_argument(
        "--year", "-y",
        type=str,
        nargs="*",
        help="Specific year(s) to process (default: all years found in imagery root)",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("OUTPUT"),
        help="Root output directory (default: OUTPUT)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List years / imagery found without processing anything",
    )

    # ── Compute-specific args ──────────────────────────────────────────────
    compute_group = parser.add_argument_group("compute options")

    compute_group.add_argument(
        "--imagery-root", "-i",
        type=Path,
        help=(
            "Root directory containing imagery organised by year. "
            "Required for 'compute' and 'all' modes."
        ),
    )

    compute_group.add_argument(
        "--transects", "-t",
        type=Path,
        default=Path("INPUT/shorelineTransPais.json"),
        help="Path to transects GeoJSON file (default: INPUT/shorelineTransPais.json)",
    )

    compute_group.add_argument(
        "--band-config",
        type=Path,
        default=BAND_CONFIG_PATH,
        help=f"Path to band_config.json (default: {BAND_CONFIG_PATH})",
    )

    compute_group.add_argument(
        "--force",
        action="store_true",
        help="Re-run compute even if Parquet checkpoints already exist",
    )

    # ── Plot-specific args ─────────────────────────────────────────────────
    plot_group = parser.add_argument_group("plot options")

    plot_group.add_argument(
        "--sample-every",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Plot every Nth transect instead of all. "
            "Ignored if --transect-ids is supplied."
        ),
    )

    plot_group.add_argument(
        "--transect-ids",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help=(
            "Explicit list of transect IDs to plot "
            "(e.g. --transect-ids 100 500 1000). "
            "Takes priority over --sample-every."
        ),
    )

    plot_group.add_argument(
        "--plot-cols",
        type=int,
        default=PLOT_GRID_COLS,
        metavar="N",
        help=f"Overview grid columns (default: {PLOT_GRID_COLS})",
    )

    plot_group.add_argument(
        "--plot-rows",
        type=int,
        default=PLOT_GRID_ROWS,
        metavar="N",
        help=f"Overview grid rows (default: {PLOT_GRID_ROWS})",
    )

    args = parser.parse_args()

    # imagery-root is required unless mode is 'plot'
    if args.mode != "plot" and args.imagery_root is None:
        parser.error("--imagery-root is required for modes 'all' and 'compute'")

    return args


# ── Per-year step functions ────────────────────────────────────────────────────

def _compute_year(
    year: str,
    raster_paths: List[Path],
    transect_file: Path,
    output_root: Path,
    band_config_path: Path,
    verbose: bool,
    force: bool,
) -> bool:
    """
    Run the compute (sampling + feature extraction) step for one year.

    Skips if both Parquet checkpoints already exist and *force* is False.

    Returns:
        True if the step succeeded or was skipped; False on error.
    """
    year_output_dir = output_root / year
    profiles_path   = year_output_dir / f"profiles_{year}.parquet"
    features_path   = year_output_dir / f"features_{year}.parquet"

    _print_year_banner(year, len(raster_paths), year_output_dir)

    if not force and profiles_path.exists() and features_path.exists():
        print("  [compute] Skipping — checkpoints exist (use --force to overwrite)")
        return True

    try:
        profiles_path, features_path = run_compute(
            year=year,
            raster_paths=raster_paths,
            transect_geojson=transect_file,
            output_dir=year_output_dir,
            band_config_path=band_config_path,
            verbose=verbose,
        )
        print(f"  [compute] Profiles: {profiles_path}")
        print(f"  [compute] Features: {features_path}")
        return True

    except Exception as e:
        print(f"  [compute] ERROR: {e}")
        logging.exception("Compute failed for year %s", year)
        return False

def _interpret_year(
    year: str,
    output_root: Path,
    verbose: bool,
    force: bool,
) -> bool:
    """
    Run the interpret (transition detection) step for one year.
 
    Expects both Parquet checkpoints written by _compute_year to already
    exist.  Skips if transitions_{year}.parquet already exists and *force*
    is False.
 
    Returns:
        True if the step succeeded or was skipped; False on error.
    """
    year_output_dir  = output_root / year
    features_path    = year_output_dir / f"features_{year}.parquet"
    profiles_path    = year_output_dir / f"profiles_{year}.parquet"
    transitions_path = year_output_dir / f"transitions_{year}.parquet"
 
    if not force and transitions_path.exists():
        print("  [interpret] Skipping — checkpoint exists (use --force to overwrite)")
        return True
 
    # Guard: refuse to run if compute outputs are missing.
    missing = [p for p in (features_path, profiles_path) if not p.exists()]
    if missing:
        for p in missing:
            print(f"  [interpret] ERROR — missing input: {p}")
        print("  [interpret] Run compute step first.")
        return False
 
    try:
        transitions_path, shellline_path = run_interpret(
            year=year,
            features_path=features_path,
            profiles_path=profiles_path,
            output_dir=year_output_dir,
            verbose=verbose,
        )
        print(f"  [interpret] Transitions: {transitions_path}")
        if shellline_path:
            print(f"  [interpret] Shell line: {shellline_path}")
        else:
            print("  [interpret] Warning — no shell lines detected (empty GeoJSON)")
        return True
 
    except Exception as e:
        print(f"  [interpret] ERROR: {e}")
        logging.exception("Interpret failed for year %s", year)
        return False

def _plot_year(
    year: str,
    output_root: Path,
    sample_every: Optional[int],
    transect_ids: Optional[List[int]],
    ncols: int,
    nrows: int,
) -> bool:
    """
    Run the plot step for one year from existing Parquet checkpoints.

    Expects:
        {output_root}/{year}/profiles_{year}.parquet   (required)
        {output_root}/{year}/features_{year}.parquet   (optional; overlays NIR d/dx)

    Plots are written to:
        {output_root}/{year}/plots/transects/
        {output_root}/{year}/plots/overview/

    Returns:
        True if plotting succeeded; False on error.
    """
    year_output_dir = output_root / year
    profiles_path   = year_output_dir / f"profiles_{year}.parquet"
    features_path   = year_output_dir / f"features_{year}.parquet"

    if not profiles_path.exists():
        print(
            f"  [plot] ERROR: profiles parquet not found: {profiles_path}\n"
            f"         Run '--mode compute' first."
        )
        return False

    features_arg = features_path if features_path.exists() else None
    if features_arg is None:
        print(f"  [plot] Features parquet not found — NIR derivative overlay skipped.")

    plots_dir = year_output_dir / "plots"
    year_config = load_band_config(BAND_CONFIG_PATH, year)
    band_indices = resolve_band_indices(year_config)
    band_label = band_mode_label(band_mode_from_indices(band_indices))

    # Summarise what will be plotted
    if transect_ids is not None:
        print(f"  [plot] Transect selection: {len(transect_ids)} explicit ID(s)")
    elif sample_every is not None:
        print(f"  [plot] Transect selection: every {sample_every}th transect")
    else:
        print(f"  [plot] Transect selection: all transects (this may be slow)")

    try:
        from .plot import run_plots  # deferred: only imported when plot step runs
        run_plots(
            year=int(year),
            profiles_path=profiles_path,
            output_dir=plots_dir,
            features_path=features_arg,
            transect_ids=transect_ids,
            sample_every=sample_every,
            ncols=ncols,
            nrows=nrows,
            band_label=band_label,
        )
        print(f"  [plot] Plots written to: {plots_dir}")
        return True

    except Exception as e:
        print(f"  [plot] ERROR: {e}")
        logging.exception("Plot failed for year %s", year)
        return False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_year_banner(year: str, n_rasters: int, output_dir: Path) -> None:
    print(f"\n{'='*60}")
    print(f"Year: {year}  ({n_rasters} rasters)")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


def _resolve_years(args, years_data: dict) -> List[str]:
    """
    Filter the discovered year dict to the years requested by --year.

    Returns the sorted list of years to process, or exits on bad input.
    """
    if args.year:
        to_process = [y for y in args.year if y in years_data]
        missing    = [y for y in args.year if y not in years_data]
        if missing:
            print(f"\nWARNING: Requested years not found in imagery: {missing}")
        if not to_process:
            print(f"Available years: {sorted(years_data.keys())}")
            sys.exit(1)
        return to_process
    return sorted(years_data.keys())


def _discover_years(args) -> dict:
    """
    Scan imagery root and return {year: [raster_paths]}.

    In plot-only mode, instead synthesises the year list from existing
    Parquet checkpoints in the output directory so imagery is not needed.
    """
    if args.mode == "plot":
        # Infer available years from existing profiles parquets
        years_data = {}
        if args.output.exists():
            for year_dir in sorted(args.output.iterdir()):
                if not year_dir.is_dir() or not year_dir.name.isdigit():
                    continue
                p = year_dir / f"profiles_{year_dir.name}.parquet"
                if p.exists():
                    years_data[year_dir.name] = []   # no rasters needed for plot
        if not years_data:
            print(
                f"\nERROR: No profiles parquets found under {args.output}.\n"
                f"       Run '--mode compute' first."
            )
            sys.exit(1)
        return years_data

    # compute or all — scan imagery root
    print("\nScanning imagery...")
    years_data = group_rasters_by_year(args.imagery_root, recursive=True)
    if not years_data:
        print("\nERROR: No imagery found")
        sys.exit(1)
    print(f"Found {len(years_data)} years: {sorted(years_data.keys())}")
    return years_data


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    validate_output_directory(args.output)
    setup_logging(
        verbose=args.verbose,
        log_file=args.output / "compute_batch.log",
    )

    print("\n" + "=" * 60)
    print("SPECTRAL SHORELINE DETECTION")
    print(f"Mode: {args.mode.upper()}")
    print("=" * 60)
    if args.imagery_root:
        print(f"Imagery root: {args.imagery_root}")
    if args.mode != "plot":
        print(f"Transects:    {args.transects}")
        print(f"Band config:  {args.band_config}")
    print(f"Output:       {args.output}")

    # Validate compute-mode inputs
    if args.mode != "plot":
        for label, path in [
            ("Imagery root", args.imagery_root),
            ("Transects file", args.transects),
            ("band_config.json", args.band_config),
        ]:
            if not path.exists():
                print(f"\nERROR: {label} not found: {path}")
                sys.exit(1)

    # Discover years
    years_data     = _discover_years(args)
    years_to_process = _resolve_years(args, years_data)
    print(f"Years to process: {years_to_process}")

    # Dry run
    if args.dry_run:
        print("\n--- DRY RUN ---")
        for year in years_to_process:
            paths = years_data.get(year, [])
            print(f"\n{year}: {len(paths)} rasters")
            for p in paths[:5]:
                print(f"  {p.name}")
            if len(paths) > 5:
                print(f"  ... and {len(paths) - 5} more")
        sys.exit(0)

    # Process each year
    start_time = datetime.now()
    compute_results: dict[str, bool] = {}
    interpret_results: dict[str, bool] = {}
    plot_results:    dict[str, bool] = {}

    for year in years_to_process:
        raster_paths = years_data.get(year, [])

        if args.mode in ("compute", "all"):
            _print_year_banner(year, len(raster_paths), args.output / year)
            compute_results[year] = _compute_year(
                year=year,
                raster_paths=raster_paths,
                transect_file=args.transects,
                output_root=args.output,
                band_config_path=args.band_config,
                verbose=args.verbose,
                force=args.force,
            )
            if args.mode == "all" and not compute_results[year]:
                print(f"  [interpret] Skipping — compute step failed for {year}")
                print(f"  [plot]      Skipping — compute step failed for {year}")
                interpret_results[year] = False
                plot_results[year] = False
                continue
 
        if args.mode in ("interpret", "all"):
            if args.mode == "interpret":
                _print_year_banner(year, len(raster_paths), args.output / year)
            interpret_results[year] = _interpret_year(
                year=year,
                output_root=args.output,
                verbose=args.verbose,
                force=args.force,
            )
            if args.mode == "all" and not interpret_results[year]:
                print(f"  [plot] Skipping — interpret step failed for {year}")
                plot_results[year] = False
                continue
 
        if args.mode in ("plot", "all"):
            if args.mode == "plot":
                _print_year_banner(year, len(raster_paths), args.output / year)
            plot_results[year] = _plot_year(
                year=year,
                output_root=args.output,
                sample_every=args.sample_every,
                transect_ids=args.transect_ids,
                ncols=args.plot_cols,
                nrows=args.plot_rows,
            )
            
    # ── Multi-year shell line bundle ─────────────────────────────────────────
    # Only bundle when the interpret step ran (shell lines were produced)
    # and at least one year succeeded.
    succeeded_interpret = [y for y, ok in interpret_results.items() if ok]
    if len(succeeded_interpret) >= 1:
        bundle_path = bundle_shell_lines_to_geojson(
            output_root=args.output,
            years=succeeded_interpret,
        )
        if bundle_path:
            print(f"\nShell line bundle: {bundle_path}")
        else:
            print("\nWarning — shell line bundle skipped (no valid per-year GeoJSONs)")

    # Summary
    elapsed = datetime.now() - start_time
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed}")

    any_failure = False

    if compute_results:
        n_ok = sum(v for v in compute_results.values())
        print(f"Compute:  {n_ok}/{len(compute_results)} succeeded")
        failed = [y for y, v in compute_results.items() if not v]
        if failed:
            print(f"  Failed years: {failed}")
            any_failure = True

    if interpret_results:
        n_ok = sum(v for v in interpret_results.values())
        print(f"Interpret:  {n_ok}/{len(interpret_results)} succeeded")
        failed = [y for y, v in interpret_results.items() if not v]
        if failed:
            print(f"  Failed years: {failed}")
            any_failure = True
            
    if plot_results:
        n_ok = sum(v for v in plot_results.values())
        print(f"Plot:     {n_ok}/{len(plot_results)} succeeded")
        failed = [y for y, v in plot_results.items() if not v]
        if failed:
            print(f"  Failed years: {failed}")
            any_failure = True

    if any_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()