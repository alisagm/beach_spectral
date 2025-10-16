"""
Phase 6C Diagnostic Analysis
Compare empirical vs. real data to understand why Phase 6A/6B thresholds are failing.

This script analyzes:
1. Distribution of key features in empirical shell lines (n=20 manual classifications)
2. Actual feature values in run05 data where detection failed
3. Whether thresholds are too strict for real-world variation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_empirical_data():
    """Analyze the empirical shell line data from manual classifications."""

    print("=" * 80)
    print("STEP 1: ANALYZING EMPIRICAL DATA")
    print("=" * 80)

    # Load empirical transition characteristics
    empirical = pd.read_csv('analysis/feature_analysis/outputs/statistics/transition_characteristics.csv')

    # Filter for shell lines only (DRY->WET transitions)
    shell_lines = empirical[empirical['transition_type'] == 'BEACH_DRY->BEACH_WET']

    print(f"\nTotal shell line samples: {len(shell_lines)}")
    print(f"Unique transects: {shell_lines['transect_id'].nunique()}")

    # Key features for Phase 6 thresholds
    features_of_interest = [
        'nir_drop_absolute',
        'brightness_before_mean',
        'nir_before_mean',
        'variance_ratio'
    ]

    print("\n" + "-" * 80)
    print("EMPIRICAL SHELL LINE STATISTICS (from n=20 manual classifications)")
    print("-" * 80)

    results = {}
    for feature in features_of_interest:
        if feature in shell_lines.columns:
            values = shell_lines[feature].dropna()
            percentiles = {
                'min': values.min(),
                '5th': values.quantile(0.05),
                '25th': values.quantile(0.25),
                'median': values.quantile(0.50),
                '75th': values.quantile(0.75),
                '95th': values.quantile(0.95),
                'max': values.max(),
                'mean': values.mean(),
                'std': values.std()
            }
            results[feature] = percentiles

            print(f"\n{feature}:")
            print(f"  Min:     {percentiles['min']:>8.2f}")
            print(f"  5th %:   {percentiles['5th']:>8.2f}")
            print(f"  25th %:  {percentiles['25th']:>8.2f}")
            print(f"  Median:  {percentiles['median']:>8.2f}")
            print(f"  75th %:  {percentiles['75th']:>8.2f}")
            print(f"  95th %:  {percentiles['95th']:>8.2f}")
            print(f"  Max:     {percentiles['max']:>8.2f}")
            print(f"  Mean:    {percentiles['mean']:>8.2f} ± {percentiles['std']:.2f}")

    # Compare to current Phase 6A/6B thresholds
    print("\n" + "=" * 80)
    print("COMPARISON TO CURRENT THRESHOLDS")
    print("=" * 80)

    current_thresholds = {
        'nir_drop_absolute': 40,
        'brightness_before_mean': 190,
        'nir_before_mean': 160
    }

    for feature, threshold in current_thresholds.items():
        if feature in results:
            percentiles = results[feature]
            # What % of empirical data meets this threshold?
            values = shell_lines[feature].dropna()
            pct_meeting = (values >= threshold).mean() * 100

            print(f"\n{feature}:")
            print(f"  Current threshold: {threshold}")
            print(f"  Empirical median: {percentiles['median']:.2f}")
            print(f"  % of empirical data meeting threshold: {pct_meeting:.1f}%")

            if pct_meeting < 90:
                print(f"  WARNING: Threshold rejects {100-pct_meeting:.1f}% of known shell lines!")
            else:
                print(f"  OK: Threshold appears reasonable")

    return shell_lines, results

def analyze_failed_detections():
    """Analyze the actual transect data where detection failed."""

    print("\n" + "=" * 80)
    print("STEP 2: ANALYZING FAILED DETECTIONS IN RUN05")
    print("=" * 80)

    # Load spectral data
    spectral = pd.read_csv('analysis/training_output/seed_321197/run05/sampled_spectral_data.csv')

    print(f"\nTotal transects in run05: {spectral['TransectID'].nunique()}")
    print(f"Total data points: {len(spectral)}")

    # Get unique transects
    transects = spectral['TransectID'].unique()
    print(f"\nTransect IDs: {sorted(transects)}")

    # Load manual classifications to know where shell lines actually are
    try:
        manual = pd.read_csv('analysis/feature_analysis/data/classified_manual_321197.csv')

        # Parse shell line locations from zone boundaries
        # Shell line is the boundary between BEACH_DRY and BEACH_WET
        shell_lines = {}
        for transect_id in manual['transectID'].unique():
            transect_data = manual[manual['transectID'] == transect_id].sort_values('start[m]')

            # Find the transition from BEACH_DRY to BEACH_WET
            for i in range(len(transect_data) - 1):
                current_class = transect_data.iloc[i]['CLASS']
                next_class = transect_data.iloc[i + 1]['CLASS']

                if current_class == 'BEACH_DRY' and next_class == 'BEACH_WET':
                    # Shell line is at the end of BEACH_DRY zone
                    shell_line_location = transect_data.iloc[i]['end[m]']
                    shell_lines[transect_id] = shell_line_location
                    break

        print(f"\nManual shell line locations (DRY->WET boundary):")
        for tid in sorted(shell_lines.keys()):
            if tid in transects:
                print(f"  Transect {tid}: {shell_lines[tid]:.1f}m")

        return spectral, shell_lines

    except FileNotFoundError:
        print("\nWARNING: Manual classification file not found")
        return spectral, None

def analyze_candidate_rejections():
    """
    Look at the debug output to understand WHY candidates were rejected.
    This requires parsing the log file from run05.
    """

    print("\n" + "=" * 80)
    print("STEP 3: ANALYZING REJECTION REASONS")
    print("=" * 80)

    # Check if debug log exists
    log_paths = [
        'analysis/training_output/seed_321197/run05/detector_debug.log',
        'analysis/training_output/seed_321197/run05/run.log'
    ]

    log_found = False
    for log_path in log_paths:
        if Path(log_path).exists():
            print(f"\nFound log file: {log_path}")
            log_found = True

            # Read and show relevant rejection lines
            with open(log_path, 'r') as f:
                lines = f.readlines()

            # Look for rejection reasons
            rejection_lines = [l for l in lines if 'rejected' in l.lower() or 'candidate at' in l.lower()]

            if rejection_lines:
                print(f"\nShowing rejection examples (first 20):")
                for line in rejection_lines[:20]:
                    print(f"  {line.strip()}")

            break

    if not log_found:
        print("\nWARNING: No debug log found. Run detector with debug logging enabled.")

def generate_diagnostic_plots(shell_lines_empirical, results):
    """Generate plots comparing empirical distributions to thresholds."""

    print("\n" + "=" * 80)
    print("STEP 4: GENERATING DIAGNOSTIC PLOTS")
    print("=" * 80)

    # Create output directory
    Path('validation/plots').mkdir(parents=True, exist_ok=True)

    # Plot distributions for key features
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Phase 6C: Empirical Shell Line Feature Distributions', fontsize=14, fontweight='bold')

    features_to_plot = [
        ('nir_drop_absolute', 40, 'NIR Drop (absolute)', 'NIR Drop'),
        ('brightness_before_mean', 190, 'Brightness Before', 'Brightness'),
        ('nir_before_mean', 160, 'NIR Before', 'NIR Value'),
        ('variance_ratio', 2.0, 'Variance Ratio', 'Ratio')
    ]

    for idx, (feature, threshold, title, xlabel) in enumerate(features_to_plot):
        ax = axes[idx // 2, idx % 2]

        if feature in shell_lines_empirical.columns:
            data = shell_lines_empirical[feature].dropna()

            # Histogram
            ax.hist(data, bins=15, alpha=0.7, color='steelblue', edgecolor='black')

            # Add threshold line
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                      label=f'Phase 6 threshold: {threshold}')

            # Add median line
            median_val = data.median()
            ax.axvline(median_val, color='green', linestyle='-', linewidth=2,
                      label=f'Empirical median: {median_val:.1f}')

            # Labels
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Count')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = 'validation/plots/phase6c_diagnostics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nOK: Saved plot to {output_path}")

    plt.close()

def generate_recommendations(results):
    """Generate threshold recommendations based on empirical data."""

    print("\n" + "=" * 80)
    print("STEP 5: THRESHOLD RECOMMENDATIONS")
    print("=" * 80)

    recommendations = []

    # Recommend thresholds at 5th percentile (captures 95% of empirical data)
    threshold_mapping = {
        'nir_drop_absolute': ('min_nir_drop_absolute', 40),
        'brightness_before_mean': ('brightness_before_min', 190),
        'nir_before_mean': ('nir_before_min', 160)
    }

    print("\nRECOMMENDED THRESHOLD SETS:\n")

    # STRICT (current)
    print("1. STRICT (current Phase 6A/6B):")
    print("   - NIR drop: >=40")
    print("   - Brightness before: >=190")
    print("   - NIR before: >=160")
    print("   - Risk: May reject valid shell lines\n")

    # MODERATE (based on 10th percentile)
    print("2. MODERATE (10th percentile):")
    for feature, (param_name, current) in threshold_mapping.items():
        if feature in results:
            percentile_10 = results[feature]['5th']  # Using 5th as conservative
            recommended = int(percentile_10 * 0.9)  # 10% below 5th percentile
            print(f"   - {param_name}: >={recommended} (was {current})")
    print("   - Expected coverage: ~90% of empirical data\n")

    # RELAXED (based on 5th percentile)
    print("3. RELAXED (5th percentile):")
    for feature, (param_name, current) in threshold_mapping.items():
        if feature in results:
            percentile_5 = int(results[feature]['5th'])
            print(f"   - {param_name}: >={percentile_5} (was {current})")
    print("   - Expected coverage: ~95% of empirical data\n")

    # EMPIRICAL MEDIAN
    print("4. EMPIRICAL MEDIAN (safest):")
    for feature, (param_name, current) in threshold_mapping.items():
        if feature in results:
            median = int(results[feature]['median'] * 0.85)  # 15% below median
            print(f"   - {param_name}: >={median} (was {current})")
    print("   - Expected coverage: >50% of empirical data")
    print("   - Risk: May allow more false positives\n")

    print("\nNEXT STEPS:")
    print("1. Implement test_phase6_thresholds.py to test all 4 threshold sets")
    print("2. Run on validation set and compute precision/recall/F1")
    print("3. Select optimal threshold set based on performance")
    print("4. Add VEG->DRY location-based discrimination")
    print("5. Implement fallback detection mode")

def main():
    """Run complete diagnostic analysis."""

    print("\n" + "=" * 80)
    print("PHASE 6C DIAGNOSTIC ANALYSIS")
    print("=" * 80)
    print("\nGoal: Understand why Phase 6A/6B detected 0 transitions on most transects")
    print("Approach: Compare empirical data distributions to current thresholds\n")

    # Step 1: Analyze empirical data
    shell_lines_empirical, results = analyze_empirical_data()

    # Step 2: Analyze failed detections
    spectral_data, manual_labels = analyze_failed_detections()

    # Step 3: Analyze rejection reasons
    analyze_candidate_rejections()

    # Step 4: Generate plots
    generate_diagnostic_plots(shell_lines_empirical, results)

    # Step 5: Generate recommendations
    generate_recommendations(results)

    print("\n" + "=" * 80)
    print("DIAGNOSTIC ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nOutput files:")
    print("  - validation/plots/phase6c_diagnostics.png")
    print("\nReview the analysis above and proceed to Step 2: Threshold Testing")

if __name__ == '__main__':
    main()
