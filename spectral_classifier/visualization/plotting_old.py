"""
Visualization module for spectral transect analysis.

X-axis convention
-----------------
All plot functions use ``distance`` (arc-length from the lowest-easting /
west end of the transect, as produced by ``sampler.sample_transect``) as the
x-axis.  This guarantees that west is always on the left and east is always
on the right without any directional flag.
"""

import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from ..config import LANDCOVER_COLORS
from ..transition import TransitionDetector

logger = logging.getLogger(__name__)


def _safe_percentile(values: np.ndarray, percentile: float, default: float = 0.0) -> float:
    """
    Compute percentile with NaN/Inf handling.
    
    Args:
        values: Array of values (may contain NaN/Inf)
        percentile: Percentile to compute (0-100)
        default: Value to return if computation fails
        
    Returns:
        Percentile value, or default if all values are invalid
    """
    valid_values = values[np.isfinite(values)]
    
    if len(valid_values) == 0:
        logger.warning(f"No valid values for percentile calculation, using default={default}")
        return default
    
    return float(np.percentile(valid_values, percentile))


def _get_spectral_ylim(data: pd.DataFrame) -> tuple:
    """
    Calculate safe y-axis limits for spectral data.
    
    Args:
        data: DataFrame with red, green, blue, nir columns
        
    Returns:
        Tuple of (y_min, y_max) with safe defaults if data is invalid
    """
    all_values = []
    for band in ['red', 'green', 'blue', 'nir']:
        if band in data.columns:
            band_values = data[band].values
            all_values.extend(band_values[np.isfinite(band_values)])
    
    if not all_values:
        logger.warning("No valid spectral values found, using default y-limits")
        return (0, 255)  # Default for 8-bit imagery
    
    all_values = np.array(all_values)
    
    y_min = max(0, _safe_percentile(all_values, 1, 0) - 20)
    y_max = _safe_percentile(all_values, 99, 255) + 20
    
    if y_max <= y_min:
        y_max = y_min + 100
    
    return (y_min, y_max)


def plot_transect_analysis(
    result: Dict,
    output_dir: Path,
    show_transitions: bool = True,
    figsize: tuple = (14, 8)
) -> Path:
    """
    Create comprehensive visualization of transect analysis with NIR derivative overlay.

    The x-axis is ``distance`` (metres from the west / lowest-easting end of the
    transect), so west is always on the left.

    Args:
        result: Dictionary with transect analysis results
        output_dir: Directory to save plot
        show_transitions: Whether to mark transition zones
        figsize: Figure size in inches

    Returns:
        Path to saved figure
    """
    transect_id = result['transect_id']
    data = result['data']
    features = result.get('features', data)
    landcover = result['landcover']
    transitions = result.get('transitions', [])

    logger.debug(f"Plotting transect {transect_id} with NIR derivative overlay")

    if len(data) == 0:
        logger.warning(f"Transect {transect_id} has no data to plot")
        return None

    # Filter transitions to only show shore boundaries (shell line)
    if transitions:
        shore_transitions = [t for t in transitions
                            if TransitionDetector.is_shore_boundary(t)]
        logger.debug(
            f"Filtered {len(transitions)} transitions -> "
            f"{len(shore_transitions)} shore boundaries for plotting"
        )
        transitions = shore_transitions

    fig, ax = plt.subplots(figsize=figsize)

    y_min, y_max = _get_spectral_ylim(data)
    ax.set_ylim(y_min, y_max)

    _plot_classification_background(ax, data, landcover)
    _plot_spectral_bands(ax, data)

    ax2 = None
    if 'nir_d1_smooth' in features.columns:
        ax2 = _plot_nir_derivative(ax, features)

    if show_transitions and transitions:
        _plot_transitions_with_annotations(ax, transitions, data, ax2)

    ax.set_xlabel('Distance from West (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spectral Value (DN)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Spectral Profile with Transitions - Transect {transect_id}',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    output_path = Path(output_dir) / f'transect_{transect_id}_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.debug(f"Saved plot to {output_path}")

    return output_path


def _plot_classification_background(
    ax: plt.Axes,
    data: pd.DataFrame,
    landcover: pd.DataFrame
):
    """Plot color-coded background regions for landcover classes."""
    classes = landcover['predicted_class'].values
    distances = data['distance'].values

    if len(classes) == 0 or len(distances) == 0:
        return

    y_min, y_max = ax.get_ylim()

    current_class = classes[0]
    start_dist = distances[0]

    for i in range(1, len(classes)):
        if classes[i] != current_class or i == len(classes) - 1:
            end_dist = distances[i] if i < len(classes) - 1 else distances[-1]

            color = LANDCOVER_COLORS.get(current_class, '#808080')

            ax.axvspan(
                start_dist,
                end_dist,
                alpha=0.2,
                color=color,
                zorder=0
            )

            current_class = classes[i]
            start_dist = distances[i]


def _plot_spectral_bands(ax: plt.Axes, data: pd.DataFrame):
    """Plot spectral band profiles."""
    distance = data['distance']

    ax.plot(distance, data['red'],   color='red',     linewidth=1.5, label='Red',   alpha=0.8)
    ax.plot(distance, data['green'], color='green',   linewidth=1.5, label='Green', alpha=0.8)

    if 'blue' in data.columns and not data['blue'].isna().all():
        ax.plot(distance, data['blue'], color='blue', linewidth=1.5, label='Blue', alpha=0.8)

    if 'nir' in data.columns and not data['nir'].isna().all():
        ax.plot(distance, data['nir'], color='darkred', linewidth=1.5, label='NIR',
                alpha=0.8, linestyle='--')


def _plot_nir_derivative(
    ax: plt.Axes,
    features: pd.DataFrame
) -> plt.Axes:
    """
    Plot NIR derivative on secondary y-axis.

    Args:
        ax: Primary axes with spectral bands
        features: DataFrame with nir_d1_smooth column

    Returns:
        Secondary y-axis
    """
    ax2 = ax.twinx()

    distance = features['distance']
    nir_d1 = features['nir_d1_smooth']

    valid_deriv = nir_d1[np.isfinite(nir_d1)]
    if len(valid_deriv) == 0:
        logger.warning("No valid NIR derivative values to plot")
        return ax2

    ax2.plot(
        distance,
        nir_d1,
        color='darkorange',
        linewidth=1.5,
        alpha=0.7,
        label='NIR d/dx',
        linestyle='-.'
    )

    ax2.axhline(-3.0, color='red',  linestyle='--', linewidth=1,   alpha=0.5, label='Boundary threshold')
    ax2.axhline( 0,   color='gray', linestyle=':',  linewidth=0.5, alpha=0.5)

    ax2.set_ylabel('NIR Derivative (units/m)', fontsize=11, fontweight='bold', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    deriv_abs_max = max(abs(valid_deriv.max()), abs(valid_deriv.min()))
    if deriv_abs_max == 0 or not np.isfinite(deriv_abs_max):
        deriv_abs_max = 10
    ax2.set_ylim(-deriv_abs_max * 1.2, deriv_abs_max * 1.2)

    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)

    return ax2


def _plot_transitions_with_annotations(
    ax: plt.Axes,
    transitions: List[Dict],
    data: pd.DataFrame,
    ax2: plt.Axes = None
):
    """
    Mark transition zones with precise distance annotations.

    Args:
        ax: Primary axes
        transitions: List of transition dictionaries (shore boundaries only)
        data: DataFrame with spectral data
        ax2: Secondary axes (for derivative), optional
    """
    y_min, y_max = ax.get_ylim()

    for transition in transitions:
        dist = transition['distance']

        ax.axvline(dist, color='purple', linewidth=2.5, alpha=0.7, linestyle=':', zorder=10)

        ax.annotate(
            '',
            xy=(dist, y_max * 0.95),
            xytext=(dist, y_max * 1.02),
            arrowprops=dict(arrowstyle='->', color='purple', lw=2, alpha=0.8),
            zorder=11
        )

        ax.text(
            dist,
            y_max * 1.05,
            f'{dist:.1f}m\nShell Line',
            fontsize=9,
            ha='center',
            va='bottom',
            color='purple',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='purple', alpha=0.9),
            zorder=12
        )


def plot_summary_statistics(
    all_results: List[Dict],
    output_dir: Path
) -> Path:
    """
    Create summary statistics plot across all transects.

    Args:
        all_results: List of all transect results
        output_dir: Directory to save plot

    Returns:
        Path to saved figure
    """
    logger.debug("Creating summary statistics plot")

    transect_ids = []
    class_distributions = []
    num_transitions = []

    for result in all_results:
        transect_ids.append(result['transect_id'])
        classes = result['landcover']['predicted_class'].value_counts()
        class_distributions.append(classes)
        num_transitions.append(len(result.get('transitions', [])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(range(len(transect_ids)), num_transitions, color='steelblue')
    ax1.set_xlabel('Transect ID', fontsize=12)
    ax1.set_ylabel('Number of Transitions', fontsize=12)
    ax1.set_title('Transitions Detected per Transect', fontsize=14)
    ax1.set_xticks(range(len(transect_ids)))
    ax1.set_xticklabels(transect_ids, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    all_classes = pd.concat(class_distributions, axis=1).sum(axis=1)
    colors = [LANDCOVER_COLORS.get(cls, '#808080') for cls in all_classes.index]

    ax2.pie(
        all_classes.values,
        labels=all_classes.index,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90
    )
    ax2.set_title('Overall Landcover Distribution', fontsize=14)

    plt.tight_layout()

    output_path = Path(output_dir) / 'summary_statistics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.debug(f"Saved summary plot to {output_path}")

    return output_path


def plot_spectral_only(
    transect_id: str,
    data: pd.DataFrame,
    output_dir: Path,
    features: pd.DataFrame = None,
    figsize: tuple = (14, 6)
) -> Path:
    """
    Create clean spectral profile plot without classification overlay.
    Designed for manual annotation of training data.

    The x-axis is ``distance`` (metres from the west / lowest-easting end),
    so west is always on the left and east is always on the right.

    Args:
        transect_id: Transect identifier
        data: DataFrame with spectral values (distance, red, green, blue, nir)
        output_dir: Directory to save plot
        features: Optional DataFrame with computed features (including nir_d1_smooth)
        figsize: Figure size in inches

    Returns:
        Path to saved figure
    """
    logger.debug(f"Plotting clean spectral profile for transect {transect_id}")

    if len(data) == 0:
        logger.warning(f"Transect {transect_id} has no data to plot")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    distance = data['distance']

    ax.plot(distance, data['red'],   color='red',   linewidth=2, label='Red',   alpha=0.9)
    ax.plot(distance, data['green'], color='green', linewidth=2, label='Green', alpha=0.9)

    if 'blue' in data.columns and not data['blue'].isna().all():
        ax.plot(distance, data['blue'], color='blue', linewidth=2, label='Blue', alpha=0.9)

    if 'nir' in data.columns and not data['nir'].isna().all():
        ax.plot(distance, data['nir'], color='darkred', linewidth=2, label='NIR',
                alpha=0.9, linestyle='--')

    ax2 = None
    if features is not None and 'nir_d1_smooth' in features.columns:
        ax2 = _plot_nir_derivative(ax, features)

    ax.set_xlabel('Distance from West (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Spectral Value', fontsize=13, fontweight='bold')
    ax.set_title(
        f'Spectral Profile - Transect {transect_id}',
        fontsize=15,
        fontweight='bold',
        pad=15
    )

    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.5)
    ax.minorticks_on()

    ax.legend(loc='upper right' if ax2 is None else 'upper left', fontsize=11, framealpha=0.9)

    y_min, y_max = _get_spectral_ylim(data)
    ax.set_ylim(y_min, y_max)

    ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()

    output_path = Path(output_dir) / f'transect_{transect_id}_clean.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.debug(f"Saved clean plot to {output_path}")

    return output_path


def create_legend_patch(landcover_classes: List[str]) -> List:
    """
    Create legend patches for landcover classes.

    Args:
        landcover_classes: List of class names

    Returns:
        List of matplotlib patches
    """
    patches = []
    for cls in landcover_classes:
        color = LANDCOVER_COLORS.get(cls, '#808080')
        patch = mpatches.Patch(color=color, label=cls, alpha=0.5)
        patches.append(patch)

    return patches