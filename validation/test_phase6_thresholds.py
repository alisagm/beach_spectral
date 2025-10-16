"""
Phase 6C: Threshold Testing Script

Test multiple threshold configurations to find optimal balance between
precision and recall for shell line detection.

This script:
1. Tests 4 threshold sets (STRICT, MODERATE, RELAXED, EMPIRICAL)
2. Runs detection on all 10 transects in run05
3. Compares detected locations to manual labels
4. Computes precision, recall, F1-score for each threshold set
5. Generates comparison plots and recommendations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import copy

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectral_classifier.transition import TransitionDetector
from spectral_classifier.features import SpectralFeatures
from spectral_classifier.config import THRESHOLDS


def load_manual_shell_lines(seed):
    """Load manual shell line locations from classified data."""
    manual_file = f'feature_analysis/data/classified_manual_{seed}.csv'
    manual = pd.read_csv(manual_file)

    # Parse shell line locations from zone boundaries
    shell_lines = {}
    for transect_id in manual['transectID'].unique():
        transect_data = manual[manual['transectID'] == transect_id].sort_values('start[m]')

        # Find the transition from BEACH_DRY to BEACH_WET
        for i in range(len(transect_data) - 1):
            current_class = transect_data.iloc[i]['CLASS']
            next_class = transect_data.iloc[i + 1]['CLASS']

            if current_class == 'BEACH_DRY' and next_class == 'BEACH_WET':
                shell_line_location = transect_data.iloc[i]['end[m]']
                shell_lines[transect_id] = shell_line_location
                break

    return shell_lines


def create_threshold_config(name, nir_drop, brightness_before, nir_before):
    """Create a threshold configuration for testing."""
    config = copy.deepcopy(THRESHOLDS)

    # Update dry_wet boundary thresholds
    if 'boundary_thresholds' in config and 'dry_wet' in config['boundary_thresholds']:
        config['boundary_thresholds']['dry_wet'].update({
            'min_nir_drop_absolute': nir_drop,
            'brightness_before_min': brightness_before,
            'nir_before_min': nir_before
        })

    return config


def detect_shell_line(transect_data, thresholds):
    """
    Run detection on a single transect.

    Returns:
        dict with 'distance', 'confidence', or None if no detection
    """
    try:
        # Extract features
        feat_extractor = SpectralFeatures(transect_data)
        features = feat_extractor.compute_all()

        # Detect transitions
        detector = TransitionDetector(thresholds)
        transitions = detector.find_transitions(features, transect_data)

        # Find dry->wet transition (shell line)
        for trans in transitions:
            if trans.get('type') == 'dry_wet' or trans.get('boundary_type') == 'DRY->WET':
                return {
                    'distance': trans['distance'],
                    'confidence': trans.get('confidence', 0.0)
                }

        return None

    except Exception as e:
        print(f"Error in detection: {e}")
        return None


def validate_transect(transect_id, transect_data, manual_location, thresholds):
    """
    Run detector on a transect and compare to manual label.

    Returns:
        dict with result information
    """
    detected = detect_shell_line(transect_data, thresholds)

    if detected is None:
        return {
            'transect_id': transect_id,
            'manual_location': manual_location,
            'detected_location': None,
            'error_m': None,
            'confidence': None,
            'result': 'FALSE_NEGATIVE'
        }

    detected_location = detected['distance']
    error = abs(detected_location - manual_location)

    # Classification based on error threshold
    if error <= 5.0:
        result = 'CORRECT'
    elif error <= 10.0:
        result = 'LOCALIZATION_ERROR'
    else:
        result = 'FALSE_POSITIVE'

    return {
        'transect_id': transect_id,
        'manual_location': manual_location,
        'detected_location': detected_location,
        'error_m': error,
        'confidence': detected['confidence'],
        'result': result
    }


def test_threshold_set(name, nir_drop, brightness_before, nir_before, spectral_data, shell_lines):
    """Test a specific threshold configuration on all transects."""

    print(f"\n{'='*80}")
    print(f"TESTING: {name}")
    print(f"{'='*80}")
    print(f"Thresholds: NIR drop >= {nir_drop}, Brightness >= {brightness_before}, NIR before >= {nir_before}")

    # Create threshold configuration
    thresholds = create_threshold_config(name, nir_drop, brightness_before, nir_before)

    # Run detection on all transects
    results = []
    for transect_id, manual_loc in shell_lines.items():
        transect_data = spectral_data[spectral_data['TransectID'] == transect_id].copy()

        if len(transect_data) == 0:
            print(f"WARNING: No data for transect {transect_id}")
            continue

        result = validate_transect(transect_id, transect_data, manual_loc, thresholds)
        results.append(result)

        # Print result
        status_icon = {
            'CORRECT': 'OK',
            'FALSE_NEGATIVE': 'MISS',
            'FALSE_POSITIVE': 'FP',
            'LOCALIZATION_ERROR': 'NEAR'
        }.get(result['result'], '?')

        if result['detected_location'] is not None:
            print(f"  [{status_icon}] Transect {transect_id}: "
                  f"detected={result['detected_location']:.1f}m, "
                  f"manual={result['manual_location']:.1f}m, "
                  f"error={result['error_m']:.1f}m, "
                  f"conf={result['confidence']:.2f}")
        else:
            print(f"  [{status_icon}] Transect {transect_id}: "
                  f"NO DETECTION (manual={result['manual_location']:.1f}m)")

    # Compute metrics
    results_df = pd.DataFrame(results)

    correct = (results_df['result'] == 'CORRECT').sum()
    near = (results_df['result'] == 'LOCALIZATION_ERROR').sum()
    false_neg = (results_df['result'] == 'FALSE_NEGATIVE').sum()
    false_pos = (results_df['result'] == 'FALSE_POSITIVE').sum()
    total = len(results_df)

    # Precision: of all detections, how many were correct?
    detections = correct + false_pos + near
    precision = correct / detections if detections > 0 else 0

    # Recall: of all true shell lines, how many did we detect correctly?
    recall = correct / (correct + false_neg) if (correct + false_neg) > 0 else 0

    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Average error for successful detections
    detected_results = results_df[results_df['detected_location'].notna()]
    avg_error = detected_results['error_m'].mean() if len(detected_results) > 0 else None

    print(f"\n--- Metrics ---")
    print(f"Total: {total}, Correct: {correct}, Near: {near}, False Neg: {false_neg}, False Pos: {false_pos}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    if avg_error is not None:
        print(f"Avg Error (detections): {avg_error:.2f}m")

    return {
        'name': name,
        'thresholds': {
            'nir_drop': nir_drop,
            'brightness_before': brightness_before,
            'nir_before': nir_before
        },
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'correct': correct,
            'near': near,
            'false_neg': false_neg,
            'false_pos': false_pos,
            'total': total,
            'avg_error': avg_error
        },
        'results': results_df
    }


def generate_comparison_plots(threshold_results):
    """Generate comparison plots for all threshold sets."""

    print(f"\n{'='*80}")
    print("GENERATING COMPARISON PLOTS")
    print(f"{'='*80}")

    # Create output directory
    Path('validation/plots').mkdir(parents=True, exist_ok=True)

    # Plot 1: Precision-Recall comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Phase 6C: Threshold Comparison', fontsize=14, fontweight='bold')

    names = [r['name'] for r in threshold_results]
    precisions = [r['metrics']['precision'] for r in threshold_results]
    recalls = [r['metrics']['recall'] for r in threshold_results]
    f1s = [r['metrics']['f1'] for r in threshold_results]

    # Bar chart
    ax = axes[0]
    x = np.arange(len(names))
    width = 0.25

    ax.bar(x - width, precisions, width, label='Precision', color='steelblue')
    ax.bar(x, recalls, width, label='Recall', color='orange')
    ax.bar(x + width, f1s, width, label='F1 Score', color='green')

    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics by Threshold Set')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    # Precision-Recall curve
    ax = axes[1]
    colors = ['red', 'orange', 'green', 'blue']
    for i, result in enumerate(threshold_results):
        ax.scatter(result['metrics']['recall'], result['metrics']['precision'],
                  s=150, color=colors[i], label=result['name'], marker='o')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)

    # Add diagonal line (F1=0.5 contour)
    x_line = np.linspace(0.01, 1, 100)
    for f1_level in [0.5, 0.7, 0.9]:
        y_line = (f1_level * x_line) / (2 * x_line - f1_level)
        y_line = np.clip(y_line, 0, 1)
        ax.plot(x_line, y_line, 'k--', alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    output_path = 'validation/plots/phase6c_threshold_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"OK: Saved plot to {output_path}")

    plt.close()


def generate_recommendations(threshold_results):
    """Generate recommendations based on test results."""

    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")

    # Sort by F1 score
    sorted_results = sorted(threshold_results, key=lambda x: x['metrics']['f1'], reverse=True)

    print("\nRanking by F1 Score:")
    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. {result['name']}: F1={result['metrics']['f1']:.3f}, "
              f"Precision={result['metrics']['precision']:.3f}, "
              f"Recall={result['metrics']['recall']:.3f}")

    # Find best balanced option
    best = sorted_results[0]

    print(f"\nRECOMMENDED THRESHOLD SET: {best['name']}")
    print(f"  NIR drop: >= {best['thresholds']['nir_drop']}")
    print(f"  Brightness before: >= {best['thresholds']['brightness_before']}")
    print(f"  NIR before: >= {best['thresholds']['nir_before']}")
    print(f"\nExpected Performance:")
    print(f"  Precision: {best['metrics']['precision']:.1%}")
    print(f"  Recall: {best['metrics']['recall']:.1%}")
    print(f"  F1 Score: {best['metrics']['f1']:.3f}")
    print(f"  Average error: {best['metrics']['avg_error']:.2f}m (when detected)")

    # Check if we still need additional improvements
    if best['metrics']['f1'] < 0.85:
        print("\nWARNING: F1 score < 0.85. Consider:")
        print("  1. Adding VEG->DRY location-based discrimination")
        print("  2. Implementing fallback detection mode")
        print("  3. Reviewing feature engineering")
    else:
        print("\nOK: Performance meets target (F1 >= 0.85)")


def main():
    """Run threshold testing on validation dataset."""

    print("="*80)
    print("PHASE 6C: THRESHOLD TESTING")
    print("="*80)
    print("\nGoal: Find optimal threshold set for shell line detection")
    print("Dataset: training_output/seed_321197/run05 (10 transects)\n")

    # Load data
    print("Loading data...")
    spectral_data = pd.read_csv('training_output/seed_321197/run05/sampled_spectral_data.csv')
    shell_lines = load_manual_shell_lines(321197)

    print(f"Loaded {len(shell_lines)} manually classified transects")
    print(f"Shell line locations: {sorted(shell_lines.items())}")

    # Define threshold sets to test
    # Based on diagnostic analysis results
    threshold_sets = [
        {
            'name': 'STRICT',
            'nir_drop': 40,
            'brightness_before': 190,
            'nir_before': 160,
            'description': 'Current Phase 6A/6B (baseline)'
        },
        {
            'name': 'MODERATE',
            'nir_drop': 35,
            'brightness_before': 180,
            'nir_before': 145,
            'description': 'Moderately relaxed (10th percentile based)'
        },
        {
            'name': 'RELAXED',
            'nir_drop': 39,
            'brightness_before': 183,
            'nir_before': 139,
            'description': 'Relaxed (5th percentile)'
        },
        {
            'name': 'EMPIRICAL',
            'nir_drop': 43,
            'brightness_before': 172,
            'nir_before': 147,
            'description': 'Based on empirical median (85% threshold)'
        }
    ]

    # Test each threshold set
    results = []
    for ts in threshold_sets:
        result = test_threshold_set(
            ts['name'],
            ts['nir_drop'],
            ts['brightness_before'],
            ts['nir_before'],
            spectral_data,
            shell_lines
        )
        results.append(result)

    # Save detailed results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    output_dir = Path('validation/outputs_phase6c')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-transect results for each threshold set
    for result in results:
        output_file = output_dir / f"results_{result['name'].lower()}.csv"
        result['results'].to_csv(output_file, index=False)
        print(f"Saved {output_file}")

    # Save summary metrics
    summary_data = []
    for result in results:
        summary_data.append({
            'threshold_set': result['name'],
            'nir_drop': result['thresholds']['nir_drop'],
            'brightness_before': result['thresholds']['brightness_before'],
            'nir_before': result['thresholds']['nir_before'],
            **result['metrics']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / 'threshold_comparison_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved {summary_file}")

    # Generate plots
    generate_comparison_plots(results)

    # Generate recommendations
    generate_recommendations(results)

    print(f"\n{'='*80}")
    print("THRESHOLD TESTING COMPLETE")
    print(f"{'='*80}")
    print("\nOutput files:")
    print(f"  - validation/outputs_phase6c/results_*.csv")
    print(f"  - validation/outputs_phase6c/threshold_comparison_summary.csv")
    print(f"  - validation/plots/phase6c_threshold_comparison.png")
    print("\nNext steps:")
    print("  1. Review recommended threshold set")
    print("  2. Implement VEG->DRY discrimination (Step 3)")
    print("  3. Implement fallback detection mode (Step 4)")


if __name__ == '__main__':
    main()
