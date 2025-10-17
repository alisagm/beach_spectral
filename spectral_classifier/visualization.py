"""
Visualization module for spectral transect analysis.
"""

import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from .config import LANDCOVER_COLORS

logger = logging.getLogger(__name__)


def plot_transect_analysis(
    result: Dict,
    output_dir: Path,
    show_transitions: bool = True,
    figsize: tuple = (14, 8)
) -> Path:
    """
    Create comprehensive visualization of transect analysis with NIR derivative overlay.

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
    features = result.get('features', data)  # Get features if available
    landcover = result['landcover']
    transitions = result.get('transitions', [])
    direction = result.get('direction', 'west_to_east')

    logger.debug(f"Plotting transect {transect_id} with NIR derivative overlay")

    # PHASE 7C: Filter transitions to only show shore boundaries (shell line)
    if transitions:
        from .transition import TransitionDetector
        shore_transitions = [t for t in transitions
                            if TransitionDetector.is_shore_boundary(t)]
        logger.debug(f"Filtered {len(transitions)} transitions -> {len(shore_transitions)} shore boundaries for plotting")
        transitions = shore_transitions

    fig, ax = plt.subplots(figsize=figsize)

    # Plot background classification regions
    _plot_classification_background(ax, data, landcover)

    # Plot spectral bands
    _plot_spectral_bands(ax, data)

    # Add NIR derivative on secondary y-axis
    if 'nir_d1_smooth' in features.columns:
        ax2 = _plot_nir_derivative(ax, features)
    else:
        ax2 = None

    # Mark transitions with precise distance annotations
    if show_transitions and transitions:
        _plot_transitions_with_annotations(ax, transitions, data, ax2)

    # Configure primary axes
    xlabel = 'Distance from West (m)'

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Spectral Value (DN)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Spectral Profile with Transitions - Transect {transect_id}',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Set reasonable y-limits for spectral values
    all_values = data[['red', 'green', 'blue', 'nir']].values.flatten()
    y_min = max(0, np.percentile(all_values, 1) - 20)
    y_max = np.percentile(all_values, 99) + 20
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / f'transect_{transect_id}_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plot to {output_path}")

    return output_path


def _plot_classification_background(
    ax: plt.Axes,
    data: pd.DataFrame,
    landcover: pd.DataFrame
):
    """Plot color-coded background regions for landcover classes."""
    classes = landcover['predicted_class'].values
    distances = data['distance'].values

    # Get y-axis limits for background spans
    y_min, y_max = ax.get_ylim()
    if y_min == 0 and y_max == 1:  # Not set yet
        all_values = data[['red', 'green', 'blue', 'nir']].values.flatten()
        y_min = max(0, np.percentile(all_values, 1) - 20)
        y_max = np.percentile(all_values, 99) + 20

    # Plot each class segment
    current_class = classes[0]
    start_dist = distances[0]

    for i in range(1, len(classes)):
        if classes[i] != current_class or i == len(classes) - 1:
            # End of current segment
            end_dist = distances[i] if i < len(classes) - 1 else distances[-1]

            color = LANDCOVER_COLORS.get(current_class, '#808080')

            ax.axvspan(
                start_dist,
                end_dist,
                alpha=0.2,
                color=color,
                zorder=0
            )

            # Update for next segment
            current_class = classes[i]
            start_dist = distances[i]


def _plot_spectral_bands(ax: plt.Axes, data: pd.DataFrame):
    """Plot spectral band profiles."""
    distance = data['distance']

    ax.plot(
        distance,
        data['red'],
        color='red',
        linewidth=1.5,
        label='Red',
        alpha=0.8
    )
    ax.plot(
        distance,
        data['green'],
        color='green',
        linewidth=1.5,
        label='Green',
        alpha=0.8
    )
    ax.plot(
        distance,
        data['blue'],
        color='blue',
        linewidth=1.5,
        label='Blue',
        alpha=0.8
    )
    ax.plot(
        distance,
        data['nir'],
        color='darkred',
        linewidth=1.5,
        label='NIR',
        alpha=0.8,
        linestyle='--'
    )


def _plot_centroid(ax: plt.Axes, data: pd.DataFrame):
    """Plot centroid line (mean of all bands)."""
    distance = data['distance']
    centroid = data[['red', 'green', 'blue', 'nir']].mean(axis=1)

    ax.plot(
        distance,
        centroid,
        color='black',
        linewidth=2,
        label='Centroid',
        alpha=0.6
    )


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

    # Plot NIR derivative
    ax2.plot(
        distance,
        nir_d1,
        color='darkorange',
        linewidth=1.5,
        alpha=0.7,
        label='NIR d/dx',
        linestyle='-.'
    )

    # Add threshold line
    ax2.axhline(
        -3.0,
        color='red',
        linestyle='--',
        linewidth=1,
        alpha=0.5,
        label='Boundary threshold'
    )
    ax2.axhline(
        0,
        color='gray',
        linestyle=':',
        linewidth=0.5,
        alpha=0.5
    )

    # Configure secondary axis
    ax2.set_ylabel('NIR Derivative (units/m)', fontsize=11, fontweight='bold', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    # Set y-limits for derivative
    deriv_max = max(abs(nir_d1.max()), abs(nir_d1.min()))
    ax2.set_ylim(-deriv_max * 1.2, deriv_max * 1.2)

    # Add legend for derivative
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

    PHASE 7C: Simplified to show only shell line boundaries with clean labels.

    Args:
        ax: Primary axes
        transitions: List of transition dictionaries (should be shore boundaries only)
        data: DataFrame with spectral data
        ax2: Secondary axes (for derivative), optional
    """
    y_min, y_max = ax.get_ylim()

    for i, transition in enumerate(transitions):
        dist = transition['distance']
        confidence = transition['confidence']

        # PHASE 7C: Use simplified "Shell Line" label for shore boundaries
        boundary_label = 'Shell Line'

        # Plot vertical line
        ax.axvline(
            dist,
            color='purple',
            linewidth=2.5,
            alpha=0.7,
            linestyle=':',
            zorder=10
        )

        # Add marker arrow at top
        ax.annotate(
            '',
            xy=(dist, y_max * 0.95),
            xytext=(dist, y_max * 1.02),
            arrowprops=dict(
                arrowstyle='->',
                color='purple',
                lw=2,
                alpha=0.8
            ),
            zorder=11
        )

        # Add distance annotation with simplified label
        annotation_text = f'{dist:.1f}m\n{boundary_label}'

        ax.text(
            dist,
            y_max * 1.05,
            annotation_text,
            fontsize=9,
            ha='center',
            va='bottom',
            color='purple',
            fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=0.4',
                facecolor='white',
                edgecolor='purple',
                alpha=0.9
            ),
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

    # Collect statistics
    transect_ids = []
    class_distributions = []
    num_transitions = []

    for result in all_results:
        transect_ids.append(result['transect_id'])

        # Class distribution
        classes = result['landcover']['predicted_class'].value_counts()
        class_distributions.append(classes)

        # Number of transitions
        num_transitions.append(len(result.get('transitions', [])))

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Transitions per transect
    ax1.bar(range(len(transect_ids)), num_transitions, color='steelblue')
    ax1.set_xlabel('Transect ID', fontsize=12)
    ax1.set_ylabel('Number of Transitions', fontsize=12)
    ax1.set_title('Transitions Detected per Transect', fontsize=14)
    ax1.set_xticks(range(len(transect_ids)))
    ax1.set_xticklabels(transect_ids, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Overall class distribution
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

    # Save
    output_path = Path(output_dir) / 'summary_statistics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved summary plot to {output_path}")

    return output_path


def plot_spectral_only(
    transect_id: str,
    data: pd.DataFrame,
    output_dir: Path,
    direction: str = 'west_to_east',
    features: pd.DataFrame = None,
    figsize: tuple = (14, 6)
) -> Path:
    """
    Create clean spectral profile plot without classification overlay.
    Designed for manual annotation of training data.

    Args:
        transect_id: Transect identifier
        data: DataFrame with spectral values (distance, red, green, blue, nir)
        output_dir: Directory to save plot
        direction: Transect direction ('west_to_east' or 'east_to_west')
        features: Optional DataFrame with computed features (including nir_d1_smooth)
        figsize: Figure size in inches

    Returns:
        Path to saved figure
    """
    logger.debug(f"Plotting clean spectral profile for transect {transect_id}")

    fig, ax = plt.subplots(figsize=figsize)

    distance = data['distance']

    # Plot spectral bands
    ax.plot(
        distance,
        data['red'],
        color='red',
        linewidth=2,
        label='Red',
        alpha=0.9
    )
    ax.plot(
        distance,
        data['green'],
        color='green',
        linewidth=2,
        label='Green',
        alpha=0.9
    )
    ax.plot(
        distance,
        data['blue'],
        color='blue',
        linewidth=2,
        label='Blue',
        alpha=0.9
    )
    ax.plot(
        distance,
        data['nir'],
        color='darkred',
        linewidth=2,
        label='NIR',
        alpha=0.9,
        linestyle='--'
    )

    # Add NIR derivative on secondary y-axis if available
    ax2 = None
    if features is not None and 'nir_d1_smooth' in features.columns:
        ax2 = _plot_nir_derivative(ax, features)

    # Configure axes
    # With corrected distance calculation, west is always at distance=0 for both directions
    xlabel = 'Distance from West (m)'

    ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
    ax.set_ylabel('Spectral Value', fontsize=13, fontweight='bold')
    ax.set_title(
        f'Spectral Profile - Transect {transect_id}',
        fontsize=15,
        fontweight='bold',
        pad=15
    )

    # Enhanced grid
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.5)
    ax.minorticks_on()

    # Legend
    ax.legend(loc='upper right' if ax2 is None else 'upper left', fontsize=11, framealpha=0.9)

    # Set y-limits with some padding
    all_values = data[['red', 'green', 'blue', 'nir']].values.flatten()
    y_min = max(0, np.percentile(all_values, 1) - 20)
    y_max = np.percentile(all_values, 99) + 20
    ax.set_ylim(y_min, y_max)

    # Increase tick label size for readability
    ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / f'transect_{transect_id}_clean.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved clean plot to {output_path}")

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
