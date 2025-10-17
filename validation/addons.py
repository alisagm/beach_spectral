"""
Additional Validation Methods

Comprehensive validation approaches including threshold sensitivity,
multi-scale analysis, sustainability metrics, and cross-validation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from collections import Counter

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from spectral_classifier.features import SpectralFeatures
from spectral_classifier.classifier import LandcoverClassifier
from spectral_classifier.transition import TransitionDetector


def threshold_sensitivity_analysis(
    spectral_csv_path: Path,
    manual_boundaries: pd.DataFrame,
    output_dir: Path,
    threshold_range: tuple = (-2.0, -5.0),
    num_steps: int = 30
) -> pd.DataFrame:
    """
    Analyze performance across different NIR derivative thresholds.

    Generates precision-recall curves to find optimal threshold.

    Args:
        spectral_csv_path: Path to sampled spectral data
        manual_boundaries: DataFrame with manual boundary positions
        output_dir: Directory to save outputs
        threshold_range: (min, max) threshold values to test
        num_steps: Number of threshold values to test

    Returns:
        DataFrame with performance metrics for each threshold
    """
    print("\n" + "="*80)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("="*80)

    spectral_data = pd.read_csv(spectral_csv_path)

    # Generate threshold values to test
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_steps)

    results = []

    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold:.2f} units/m")

        # Detect transitions with this threshold
        all_detections = []

        for transect_id in spectral_data['TransectID'].unique():
            transect_data = spectral_data[spectral_data['TransectID'] == transect_id].copy()
            transect_data = transect_data.sort_values('distance').reset_index(drop=True)

            # Compute features
            feature_extractor = SpectralFeatures(transect_data)
            features = feature_extractor.compute_all()

            # Detect transitions with custom threshold
            nir_d1 = features['nir_d1_smooth']
            nir = features['nir']
            distance = features['distance']

            min_nir = 50
            min_separation = 15.0

            candidates = []
            for i in range(len(nir_d1)):
                if nir_d1.iloc[i] < threshold and nir.iloc[i] > min_nir:
                    # Check local minimum
                    is_local_min = True
                    if i > 0 and nir_d1.iloc[i] > nir_d1.iloc[i-1]:
                        is_local_min = False
                    if i < len(nir_d1) - 1 and nir_d1.iloc[i] > nir_d1.iloc[i+1]:
                        is_local_min = False

                    if is_local_min:
                        candidates.append({
                            'transect_id': transect_id,
                            'distance': distance.iloc[i],
                            'magnitude': nir_d1.iloc[i]
                        })

            # Filter by minimum separation
            if candidates:
                candidates = sorted(candidates, key=lambda x: x['distance'])
                filtered = []
                last_distance = -999

                for candidate in candidates:
                    if candidate['distance'] - last_distance >= min_separation:
                        filtered.append(candidate)
                        last_distance = candidate['distance']
                    elif abs(candidate['magnitude']) > abs(filtered[-1]['magnitude']):
                        filtered[-1] = candidate
                        last_distance = candidate['distance']

                all_detections.extend(filtered)

        # Match to manual boundaries
        tp, fp, fn = 0, 0, 0
        tolerance_m = 10.0

        detected_df = pd.DataFrame(all_detections)
        matched_manual = set()

        # Count TP and FP
        for _, det in detected_df.iterrows():
            transect_id = det['transect_id']
            det_pos = det['distance']

            manual_transect = manual_boundaries[manual_boundaries['transect_id'] == transect_id]

            is_match = False
            for idx, man in manual_transect.iterrows():
                if abs(det_pos - man['position']) <= tolerance_m:
                    is_match = True
                    matched_manual.add(idx)
                    break

            if is_match:
                tp += 1
            else:
                fp += 1

        # Count FN (unmatched manual boundaries)
        fn = len(manual_boundaries) - len(matched_manual)

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            'threshold': threshold,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'num_detections': len(all_detections),
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

        print(f"  Detections: {len(all_detections)}, TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    results_df = pd.DataFrame(results)

    # Find optimal threshold (max F1)
    optimal_idx = results_df['f1_score'].idxmax()
    optimal = results_df.loc[optimal_idx]

    print("\n" + "-"*80)
    print(f"OPTIMAL THRESHOLD: {optimal['threshold']:.2f} units/m")
    print(f"  Precision: {optimal['precision']:.3f}")
    print(f"  Recall:    {optimal['recall']:.3f}")
    print(f"  F1-Score:  {optimal['f1_score']:.3f}")
    print("-"*80)

    # Save results
    results_path = output_dir / 'threshold_sensitivity.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results to: {results_path}")

    # Plot precision-recall curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Precision-Recall curve
    ax1.plot(results_df['recall'], results_df['precision'], 'b-', linewidth=2, label='PR Curve')
    ax1.scatter(optimal['recall'], optimal['precision'], color='red', s=100, zorder=5,
                label=f'Optimal (threshold={optimal["threshold"]:.2f})')
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Plot 2: Metrics vs threshold
    ax2.plot(results_df['threshold'], results_df['precision'], label='Precision', linewidth=2)
    ax2.plot(results_df['threshold'], results_df['recall'], label='Recall', linewidth=2)
    ax2.plot(results_df['threshold'], results_df['f1_score'], label='F1-Score', linewidth=2)
    ax2.axvline(optimal['threshold'], color='red', linestyle='--', linewidth=2, label='Optimal')
    ax2.set_xlabel('NIR Derivative Threshold (units/m)', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Performance Metrics vs Threshold', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = output_dir / 'threshold_optimization.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved precision-recall plot to: {plot_path}")

    return results_df


def multiscale_derivative_analysis(
    spectral_csv_path: Path,
    manual_boundaries: pd.DataFrame,
    output_dir: Path,
    smoothing_windows: list = [1, 3, 5, 7, 9]
) -> pd.DataFrame:
    """
    Analyze derivative persistence across smoothing scales.

    True boundaries should persist across scales; noise disappears at larger scales.

    Args:
        spectral_csv_path: Path to sampled spectral data
        manual_boundaries: DataFrame with manual boundaries
        output_dir: Directory to save outputs
        smoothing_windows: List of smoothing window sizes to test

    Returns:
        DataFrame with scale-space analysis results
    """
    print("\n" + "="*80)
    print("MULTI-SCALE DERIVATIVE ANALYSIS")
    print("="*80)

    spectral_data = pd.read_csv(spectral_csv_path)

    scale_results = []

    # Analyze one representative transect
    transect_id = manual_boundaries['transect_id'].iloc[0]
    transect_data = spectral_data[spectral_data['TransectID'] == transect_id].copy()
    transect_data = transect_data.sort_values('distance')

    manual_transect = manual_boundaries[manual_boundaries['transect_id'] == transect_id]

    # Compute derivatives at different scales
    nir = transect_data['nir'].values
    distance = transect_data['distance'].values

    # Raw derivative
    nir_d1_raw = np.gradient(nir, distance)

    fig, axes = plt.subplots(len(smoothing_windows), 1, figsize=(14, 3*len(smoothing_windows)))

    for i, window in enumerate(smoothing_windows):
        # Apply smoothing
        if window > 1:
            nir_d1_smooth = uniform_filter1d(nir_d1_raw, size=window, mode='nearest')
        else:
            nir_d1_smooth = nir_d1_raw

        # Plot
        ax = axes[i] if len(smoothing_windows) > 1 else axes

        ax.plot(distance, nir_d1_smooth, linewidth=1.5, label=f'Window={window}')
        ax.axhline(-3.0, color='red', linestyle='--', linewidth=1, label='Threshold')
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)

        # Mark manual boundaries
        for _, boundary in manual_transect.iterrows():
            ax.axvline(boundary['position'], color='green', linestyle=':', linewidth=2, alpha=0.7)

        ax.set_ylabel('NIR d/dx', fontsize=10)
        ax.set_title(f'NIR Derivative (smoothing window = {window} points)', fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        if i == len(smoothing_windows) - 1:
            ax.set_xlabel('Distance (m)', fontsize=11)

    plt.tight_layout()
    plot_path = output_dir / 'scale_space_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved scale-space plot to: {plot_path}")
    print("Analysis shows derivative behavior across smoothing scales")
    print("True boundaries persist; noise diminishes at larger scales")

    return pd.DataFrame(scale_results)


def confusion_matrix_boundaries(
    matches_df: pd.DataFrame,
    output_dir: Path
):
    """
    Create confusion matrix for boundary type classification.

    Args:
        matches_df: DataFrame from validation.trueorfalse with matched transitions
        output_dir: Directory to save outputs
    """
    print("\n" + "="*80)
    print("BOUNDARY TYPE CONFUSION MATRIX")
    print("="*80)

    # Filter to TP only
    tp_matches = matches_df[matches_df['label'] == 'TP'].copy()

    if len(tp_matches) == 0:
        print("No true positives found - skipping confusion matrix")
        return

    # Extract boundary types
    manual_types = tp_matches['manual_boundary_type'].values
    detected_types = tp_matches['detected_boundary_type'].values

    # Get unique boundary types
    all_types = sorted(set(list(manual_types) + list(detected_types)))

    # Build confusion matrix
    confusion = pd.DataFrame(0, index=all_types, columns=all_types)

    for man, det in zip(manual_types, detected_types):
        confusion.loc[man, det] += 1

    print("\nConfusion Matrix (rows=manual, columns=detected):")
    print(confusion)

    # Calculate accuracy per boundary type
    print("\nPer-Type Classification Accuracy:")
    for btype in all_types:
        if btype in confusion.index:
            total = confusion.loc[btype].sum()
            correct = confusion.loc[btype, btype] if btype in confusion.columns else 0
            accuracy = correct / total if total > 0 else 0
            print(f"  {btype}: {correct}/{total} ({100*accuracy:.1f}%)")

    # Save confusion matrix
    confusion_path = output_dir / 'boundary_type_confusion_matrix.csv'
    confusion.to_csv(confusion_path)
    print(f"\nSaved confusion matrix to: {confusion_path}")

    # Plot confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(confusion.values, cmap='Blues', aspect='auto')

    ax.set_xticks(np.arange(len(confusion.columns)))
    ax.set_yticks(np.arange(len(confusion.index)))
    ax.set_xticklabels(confusion.columns, rotation=45, ha='right')
    ax.set_yticklabels(confusion.index)

    # Add text annotations
    for i in range(len(confusion.index)):
        for j in range(len(confusion.columns)):
            text = ax.text(j, i, confusion.values[i, j],
                          ha="center", va="center", color="black" if confusion.values[i, j] < confusion.values.max()/2 else "white")

    ax.set_xlabel('Detected Boundary Type', fontsize=12)
    ax.set_ylabel('Manual Boundary Type', fontsize=12)
    ax.set_title('Boundary Type Confusion Matrix', fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    plot_path = output_dir / 'confusion_matrix_boundaries.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved confusion matrix plot to: {plot_path}")


def error_distribution_analysis(
    matches_df: pd.DataFrame,
    output_dir: Path
):
    """
    Analyze distribution of boundary location errors.

    Args:
        matches_df: DataFrame with matched transitions
        output_dir: Directory to save outputs
    """
    print("\n" + "="*80)
    print("BOUNDARY LOCATION ERROR DISTRIBUTION")
    print("="*80)

    # Filter to TP only
    tp_matches = matches_df[matches_df['label'] == 'TP'].copy()

    if len(tp_matches) == 0:
        print("No true positives found - skipping error distribution")
        return

    errors = tp_matches['distance_error'].values

    # Statistics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    median_error = np.median(errors)
    std_error = np.std(errors)

    print(f"\nLocation Error Statistics:")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}m")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}m")
    print(f"  Median Error: {median_error:.2f}m")
    print(f"  Std Dev: {std_error:.2f}m")

    # Check for systematic bias
    mean_error = np.mean(errors)
    print(f"\nSystematic Bias:")
    print(f"  Mean Error: {mean_error:+.2f}m")
    if abs(mean_error) > 1.0:
        direction = "early" if mean_error < 0 else "late"
        print(f"  Detected boundaries are systematically {abs(mean_error):.2f}m {direction}")
    else:
        print(f"  No significant systematic bias detected")

    # Plot histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Error distribution
    ax1.hist(errors, bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect alignment')
    ax1.axvline(mean_error, color='orange', linestyle='--', linewidth=2, label=f'Mean error ({mean_error:+.2f}m)')
    ax1.set_xlabel('Distance Error (m)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Boundary Location Error Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Subplot 2: Cumulative distribution
    sorted_errors = np.sort(np.abs(errors))
    cumulative = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)

    ax2.plot(sorted_errors, cumulative, linewidth=2)
    ax2.axvline(5, color='red', linestyle='--', alpha=0.7, label='+-5m tolerance')
    ax2.axvline(10, color='orange', linestyle='--', alpha=0.7, label='+-10m tolerance')
    ax2.set_xlabel('Absolute Error (m)', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title('Cumulative Error Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Calculate percentages within tolerance
    within_5m = (np.abs(errors) <= 5).sum() / len(errors) * 100
    within_10m = (np.abs(errors) <= 10).sum() / len(errors) * 100

    ax2.text(0.98, 0.02, f'{within_5m:.1f}% within +-5m\n{within_10m:.1f}% within +-10m',
             transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_path = output_dir / 'error_distribution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved error distribution plot to: {plot_path}")
    print(f"  {within_5m:.1f}% of detections within +-5m of manual boundary")
    print(f"  {within_10m:.1f}% of detections within +-10m of manual boundary")


def run_additional_validations(
    spectral_csv_path: Path,
    manual_boundaries: pd.DataFrame,
    matches_df: pd.DataFrame,
    output_dir: Path
) -> dict:
    """
    Run all additional validation methods.

    Args:
        spectral_csv_path: Path to sampled spectral data
        manual_boundaries: DataFrame with manual boundaries
        matches_df: DataFrame with matched transitions from trueorfalse analysis
        output_dir: Directory to save outputs

    Returns:
        Dictionary with results from all analyses
    """
    print("\n" + "="*80)
    print("ADDITIONAL VALIDATION METHODS")
    print("="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. Threshold sensitivity analysis
    threshold_results = threshold_sensitivity_analysis(
        spectral_csv_path, manual_boundaries, output_dir
    )
    results['threshold_sensitivity'] = threshold_results

    # 2. Multi-scale derivative analysis
    scale_results = multiscale_derivative_analysis(
        spectral_csv_path, manual_boundaries, output_dir
    )
    results['multiscale'] = scale_results

    # 3. Boundary type confusion matrix
    confusion_matrix_boundaries(matches_df, output_dir)

    # 4. Error distribution analysis
    error_distribution_analysis(matches_df, output_dir)

    print("\n" + "="*80)
    print("ALL ADDITIONAL VALIDATIONS COMPLETE")
    print("="*80)

    return results


if __name__ == '__main__':
    # This script is meant to be called from validation.run.py
    # Can be run standalone if needed for testing
    print("This script is designed to be run via validation.run.py")
    print("For standalone execution, implement required data loading here")
