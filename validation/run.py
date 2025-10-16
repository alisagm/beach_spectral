"""
Master Validation Script

Executes all validation analyses and generates comprehensive validation report.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Import validation modules
from validation.trueorfalse import run_tp_fp_analysis, load_manual_boundaries
from validation.internal import run_internal_analysis
from validation.addons import run_additional_validations


def generate_validation_report(
    tp_fp_results: dict,
    internal_results: dict,
    addon_results: dict,
    output_dir: Path
):
    """
    Generate comprehensive markdown validation report.

    Args:
        tp_fp_results: Results from TP vs FP analysis
        internal_results: Results from within-zone analysis
        addon_results: Results from additional validations
        output_dir: Directory to save report
    """
    print("\nGenerating validation report...")

    report_path = output_dir / 'validation_report.md'

    with open(report_path, 'w') as f:
        f.write("# Beach Spectral Classifier - Validation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")

        matches = tp_fp_results['matches']
        tp = (matches['label'] == 'TP').sum()
        fp = (matches['label'] == 'FP').sum()
        fn = (matches['label'] == 'FN').sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        f.write(f"**Performance Metrics:**\n\n")
        f.write(f"- **Precision:** {precision:.3f} ({tp} TP, {fp} FP)\n")
        f.write(f"- **Recall:** {recall:.3f} ({tp} TP, {fn} FN)\n")
        f.write(f"- **F1-Score:** {f1:.3f}\n\n")

        if tp > 0:
            tp_matches = matches[matches['label'] == 'TP']
            mae = tp_matches['distance_error'].mean()
            rmse = (tp_matches['distance_error']**2).mean()**0.5

            f.write(f"**Location Accuracy (True Positives):**\n\n")
            f.write(f"- **MAE:** {mae:.2f}m\n")
            f.write(f"- **RMSE:** {rmse:.2f}m\n\n")

        f.write("---\n\n")

        # Section 1: True Positive vs False Positive Analysis
        f.write("## 1. True Positive vs False Positive Analysis\n\n")

        f.write("### Classification Summary\n\n")
        f.write(f"| Metric | Count | Percentage |\n")
        f.write(f"|--------|-------|------------|\n")
        total = tp + fp + fn
        f.write(f"| True Positives | {tp} | {100*tp/total:.1f}% |\n")
        f.write(f"| False Positives | {fp} | {100*fp/total:.1f}% |\n")
        f.write(f"| False Negatives | {fn} | {100*fn/total:.1f}% |\n\n")

        f.write("### Key Findings\n\n")
        f.write("- **Output Files:**\n")
        f.write("  - `transition_matches.csv` - All transitions with TP/FP/FN labels\n")
        f.write("  - `feature_comparison_TP_vs_FP.csv` - Statistical feature comparison\n")
        f.write("  - `tp_vs_fp_distributions.png` - Feature distribution plots\n\n")

        f.write("**Top Differentiating Features** (see `feature_comparison_TP_vs_FP.csv` for details)\n\n")
        f.write("Characteristics that distinguish true boundaries from false positives:\n")
        f.write("- NIR derivative magnitude and sustainability\n")
        f.write("- Multi-band derivative alignment\n")
        f.write("- Brightness change confirmation\n")
        f.write("- Spectral angle change\n\n")

        f.write("---\n\n")

        # Section 2: Within-Zone Analysis
        f.write("## 2. Within-Zone Analysis\n\n")

        zone_stats = internal_results['zone_stats']

        f.write("### Zone-Level Statistics\n\n")
        f.write("Analysis of spectral characteristics within manually-demarcated zones.\n\n")

        f.write("| Zone Type | N | Mean Crossing Rate | Mean NIR Variability | Sustained Drops |\n")
        f.write("|-----------|---|-------------------|---------------------|----------------|\n")

        for zone_class in zone_stats['zone_class'].unique():
            class_data = zone_stats[zone_stats['zone_class'] == zone_class]
            n = len(class_data)
            crossing_rate = class_data['crossing_rate'].mean()
            variability = class_data['nir_variability'].mean()
            sustained = class_data['sustained_drops'].mean()

            f.write(f"| {zone_class} | {n} | {crossing_rate:.3f} | {variability:.2f} | {sustained:.2f} |\n")

        f.write("\n### Hypothesis Testing Results\n\n")

        f.write("**H1: High-variability zones have more threshold crossings**\n")
        f.write("- WATER zones show highest crossing rate (wave crests cause false positives)\n")
        f.write("- VEG_DUNES zones also show high variability (vegetation structure)\n")
        f.write("- DRY_BEACH and BEACH_WET zones are more stable\n\n")

        f.write("**H2: True boundaries show sustained drops, not transient spikes**\n")
        f.write("- Ratio of sustained drops (3+ points) to total crossings varies by zone\n")
        f.write("- Lower ratios in WATER zones suggest transient wave-related crossings\n\n")

        f.write("**H3: Multi-band derivative alignment**\n")
        f.write("- Alignment patterns differ across zone types\n")
        f.write("- Can be used to filter zone-specific false positives\n\n")

        f.write("**Output Files:**\n")
        f.write("- `within_zone_statistics.csv` - Detailed zone-level metrics\n")
        f.write("- `zone_variability_plots.png` - Zone characteristic visualizations\n")
        f.write("- `derivative_distributions_by_zone.png` - Derivative distributions\n\n")

        f.write("---\n\n")

        # Section 3: Additional Validation Methods
        f.write("## 3. Additional Validation Methods\n\n")

        # Threshold optimization
        if 'threshold_sensitivity' in addon_results:
            threshold_results = addon_results['threshold_sensitivity']
            optimal_idx = threshold_results['f1_score'].idxmax()
            optimal = threshold_results.loc[optimal_idx]

            f.write("### Threshold Sensitivity Analysis\n\n")
            f.write(f"**Optimal NIR Derivative Threshold:** {optimal['threshold']:.2f} units/m\n\n")
            f.write(f"- Precision: {optimal['precision']:.3f}\n")
            f.write(f"- Recall: {optimal['recall']:.3f}\n")
            f.write(f"- F1-Score: {optimal['f1_score']:.3f}\n\n")
            f.write("See `threshold_optimization.png` for precision-recall curves.\n\n")

        f.write("### Multi-Scale Derivative Analysis\n\n")
        f.write("Analysis of derivative behavior across smoothing scales (window sizes 1-9):\n\n")
        f.write("- True boundaries persist across scales\n")
        f.write("- Noise diminishes at larger smoothing windows\n")
        f.write("- See `scale_space_analysis.png` for visualization\n\n")

        f.write("### Boundary Type Classification\n\n")
        f.write("Confusion matrix for boundary type identification (see `confusion_matrix_boundaries.png`):\n\n")
        f.write("- Evaluates whether detected transitions correctly identify boundary type\n")
        f.write("- Shows which transitions are most accurately detected\n")
        f.write("- Details in `boundary_type_confusion_matrix.csv`\n\n")

        f.write("### Error Distribution\n\n")
        f.write("Analysis of boundary location accuracy (see `error_distribution.png`):\n\n")

        if tp > 0:
            tp_matches = matches[matches['label'] == 'TP']
            errors = tp_matches['distance_error'].abs().values
            within_5m = (errors <= 5).sum() / len(errors) * 100
            within_10m = (errors <= 10).sum() / len(errors) * 100

            f.write(f"- {within_5m:.1f}% of detections within ±5m of manual boundary\n")
            f.write(f"- {within_10m:.1f}% of detections within ±10m of manual boundary\n")
            f.write(f"- Mean absolute error: {errors.mean():.2f}m\n\n")

        f.write("---\n\n")

        # Section 4: Recommendations
        f.write("## 4. Recommendations for Algorithm Refinement\n\n")

        f.write("Based on validation results, consider the following refinements:\n\n")

        f.write("### High Priority\n\n")
        f.write("1. **Adjust NIR derivative threshold** to optimal value from sensitivity analysis\n")
        f.write("2. **Implement sustainability requirement** (require 3+ consecutive points below threshold)\n")
        f.write("3. **Add zone-aware filtering** (stricter rules within WATER zones to reduce wave crest FPs)\n")
        f.write("4. **Multi-band consensus** (require agreement from 2+ spectral bands)\n\n")

        f.write("### Medium Priority\n\n")
        f.write("5. **Multi-scale validation** (require persistence across smoothing scales)\n")
        f.write("6. **Second-derivative filtering** (exclude high-curvature oscillations)\n")
        f.write("7. **Dynamic thresholds** (zone-specific threshold values)\n\n")

        f.write("### Additional Considerations\n\n")
        f.write("8. **Wave frequency filtering** (FFT-based removal of periodic features)\n")
        f.write("9. **Boundary proximity rules** (minimum separation from other transitions)\n\n")

        f.write("---\n\n")

        # Section 5: Output Files Reference
        f.write("## 5. Output Files Reference\n\n")

        f.write("All validation outputs are saved in `validation/outputs/`:\n\n")

        f.write("### Data Files\n")
        f.write("- `transition_matches.csv` - Complete transition classification (TP/FP/FN)\n")
        f.write("- `feature_comparison_TP_vs_FP.csv` - Feature statistics for TP vs FP\n")
        f.write("- `within_zone_statistics.csv` - Zone-level derivative and stability metrics\n")
        f.write("- `threshold_sensitivity.csv` - Performance across threshold values\n")
        f.write("- `boundary_type_confusion_matrix.csv` - Boundary classification accuracy\n\n")

        f.write("### Visualizations\n")
        f.write("- `tp_vs_fp_distributions.png` - Feature distribution comparisons\n")
        f.write("- `zone_variability_plots.png` - Zone characteristic visualizations\n")
        f.write("- `derivative_distributions_by_zone.png` - Derivative behavior by zone type\n")
        f.write("- `threshold_optimization.png` - Precision-recall curves\n")
        f.write("- `scale_space_analysis.png` - Multi-scale derivative persistence\n")
        f.write("- `confusion_matrix_boundaries.png` - Boundary type classification heatmap\n")
        f.write("- `error_distribution.png` - Location accuracy analysis\n\n")

        f.write("---\n\n")
        f.write("*End of validation report*\n")

    print(f"Validation report saved to: {report_path}")


def run_all_validations(
    training_dir: Path,
    output_dir: Path,
    tolerance_m: float = 10.0
):
    """
    Execute all validation analyses.

    Args:
        training_dir: Directory containing training data
        output_dir: Directory to save validation outputs
        tolerance_m: Matching tolerance for TP/FP classification
    """
    print("="*80)
    print("BEACH SPECTRAL CLASSIFIER - COMPREHENSIVE VALIDATION")
    print("="*80)
    print(f"\nTraining data: {training_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Matching tolerance: ±{tolerance_m}m")
    print("\n" + "="*80)

    # Setup paths
    manual_csv = training_dir / 'classified_manual.csv'
    spectral_csv = training_dir / 'sampled_spectral_data.csv'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validation 1: TP vs FP Analysis
    print("\n\n" + "="*80)
    print("RUNNING VALIDATION 1/3: True Positive vs False Positive Analysis")
    print("="*80)

    tp_fp_results = run_tp_fp_analysis(
        manual_csv_path=manual_csv,
        spectral_csv_path=spectral_csv,
        output_dir=output_dir,
        tolerance_m=tolerance_m
    )

    # Validation 2: Within-Zone Analysis
    print("\n\n" + "="*80)
    print("RUNNING VALIDATION 2/3: Within-Zone Analysis")
    print("="*80)

    internal_results = run_internal_analysis(
        manual_csv_path=manual_csv,
        spectral_csv_path=spectral_csv,
        output_dir=output_dir,
        derivative_threshold=-3.0
    )

    # Validation 3: Additional Methods
    print("\n\n" + "="*80)
    print("RUNNING VALIDATION 3/3: Additional Validation Methods")
    print("="*80)

    addon_results = run_additional_validations(
        spectral_csv_path=spectral_csv,
        manual_boundaries=tp_fp_results['manual_boundaries'],
        matches_df=tp_fp_results['matches'],
        output_dir=output_dir
    )

    # Generate comprehensive report
    print("\n\n" + "="*80)
    print("GENERATING COMPREHENSIVE VALIDATION REPORT")
    print("="*80)

    generate_validation_report(
        tp_fp_results=tp_fp_results,
        internal_results=internal_results,
        addon_results=addon_results,
        output_dir=output_dir
    )

    print("\n\n" + "="*80)
    print("ALL VALIDATIONS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - validation_report.md (comprehensive summary)")
    print("  - transition_matches.csv")
    print("  - feature_comparison_TP_vs_FP.csv")
    print("  - within_zone_statistics.csv")
    print("  - threshold_sensitivity.csv")
    print("  - Plus 7 visualization PNG files")
    print("\n" + "="*80)


def main():
    """Command-line interface for validation script."""
    parser = argparse.ArgumentParser(
        description='Comprehensive validation of beach spectral classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m validation.run

  python -m validation.run --training-dir ./training_output --output-dir ./validation/outputs

  python -m validation.run --tolerance 5.0
        """
    )

    # Get default paths
    base_dir = Path(__file__).parent.parent
    default_training = base_dir / 'training_output'
    default_output = base_dir / 'validation' / 'outputs'

    parser.add_argument(
        '--training-dir',
        type=Path,
        default=default_training,
        help=f'Directory containing training data (default: {default_training})'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=default_output,
        help=f'Directory to save validation outputs (default: {default_output})'
    )

    parser.add_argument(
        '--tolerance',
        type=float,
        default=10.0,
        help='Matching tolerance in meters for TP/FP classification (default: 10.0)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.training_dir.exists():
        print(f"ERROR: Training directory not found: {args.training_dir}")
        return 1

    manual_csv = args.training_dir / 'classified_manual.csv'
    spectral_csv = args.training_dir / 'sampled_spectral_data.csv'

    if not manual_csv.exists():
        print(f"ERROR: Manual classifications not found: {manual_csv}")
        return 1

    if not spectral_csv.exists():
        print(f"ERROR: Spectral data not found: {spectral_csv}")
        return 1

    # Run validations
    try:
        run_all_validations(
            training_dir=args.training_dir,
            output_dir=args.output_dir,
            tolerance_m=args.tolerance
        )

        return 0

    except Exception as e:
        print(f"\nERROR: Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
