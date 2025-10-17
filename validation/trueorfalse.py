"""
True Positive vs False Positive Analysis

Matches detected transitions to manual boundaries and analyzes characteristics
that differentiate true boundaries from false positives.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from spectral_classifier.features import SpectralFeatures
from spectral_classifier.classifier import LandcoverClassifier
from spectral_classifier.transition import TransitionDetector


def load_manual_boundaries(manual_csv_path: Path) -> pd.DataFrame:
    """
    Load manual boundary positions from classified_manual.csv.

    Returns:
        DataFrame with columns: transect_id, position, from_class, to_class, boundary_type
    """
    print("Loading manual boundaries...")

    # Read manual classifications
    manual_data = []
    with open(manual_csv_path, 'r') as f:
        headers = f.readline().strip().split(',')

        for line in f:
            parts = line.strip().split(',', 3)
            tid = int(parts[0])
            start = float(parts[1])
            end = float(parts[2])

            # Extract class name
            remaining = parts[3]
            if '[' in remaining:
                class_end = remaining.index('[') - 1
                class_name = remaining[:class_end].rstrip(',')
            elif ',' in remaining:
                class_name, _ = remaining.rsplit(',', 1)
            else:
                # No comma or brackets - use the whole string as class name
                class_name = remaining.strip()

            manual_data.append({
                'transectID': tid,
                'start[m]': start,
                'end[m]': end,
                'CLASS': class_name
            })

    manual = pd.DataFrame(manual_data)

    # Extract boundary positions (zone transitions)
    boundaries = []

    for tid in manual['transectID'].unique():
        transect_manual = manual[manual['transectID'] == tid].sort_values('start[m]').reset_index(drop=True)

        # Boundaries occur at zone transitions (end of each zone except last)
        for i in range(len(transect_manual) - 1):
            row = transect_manual.iloc[i]
            next_row = transect_manual.iloc[i + 1]

            boundary_pos = row['end[m]']
            from_class = row['CLASS']
            to_class = next_row['CLASS']

            boundaries.append({
                'transect_id': tid,
                'position': boundary_pos,
                'from_class': from_class,
                'to_class': to_class,
                'boundary_type': f"{from_class}_to_{to_class}"
            })

    boundaries_df = pd.DataFrame(boundaries)

    print(f"Loaded {len(boundaries_df)} manual boundaries from {len(manual['transectID'].unique())} transects")
    print("\nBoundary type distribution:")
    print(boundaries_df['boundary_type'].value_counts())

    return boundaries_df


def detect_transitions_on_training_data(spectral_csv_path: Path) -> pd.DataFrame:
    """
    Run transition detection on training transects.

    Returns:
        DataFrame with detected transitions
    """
    print("\nRunning transition detection on training data...")

    # Load spectral data
    spectral_data = pd.read_csv(spectral_csv_path)

    all_detections = []

    for transect_id in spectral_data['TransectID'].unique():
        # Get data for this transect
        transect_data = spectral_data[spectral_data['TransectID'] == transect_id].copy()
        transect_data = transect_data.sort_values('distance').reset_index(drop=True)

        # Rename columns to match expected format
        transect_data = transect_data.rename(columns={
            'TransectID': 'transect_id'
        })

        # Compute features
        feature_extractor = SpectralFeatures(transect_data)
        features = feature_extractor.compute_all()

        # Classify landcover
        classifier = LandcoverClassifier()
        landcover = classifier.classify(features)
        landcover = classifier.apply_spatial_smoothing(landcover)

        # Detect transitions
        detector = TransitionDetector()
        transitions = detector.find_transitions(features, landcover)

        # Add transect ID to each transition
        for t in transitions:
            t['transect_id'] = transect_id
            all_detections.append(t)

    detections_df = pd.DataFrame(all_detections)

    print(f"Detected {len(detections_df)} transitions across {len(spectral_data['TransectID'].unique())} transects")

    return detections_df


def match_transitions(detected: pd.DataFrame, manual: pd.DataFrame, tolerance_m: float = 10.0) -> pd.DataFrame:
    """
    Match detected transitions to manual boundaries.

    Args:
        detected: DataFrame with detected transitions
        manual: DataFrame with manual boundaries
        tolerance_m: Matching tolerance in meters (default: 10m)

    Returns:
        DataFrame with matched transitions labeled as TP/FP/FN
    """
    print(f"\nMatching transitions with +-{tolerance_m}m tolerance...")

    matched_detections = []

    for _, det in detected.iterrows():
        transect_id = det['transect_id']
        det_pos = det['distance']

        # Find manual boundaries for this transect
        manual_transect = manual[manual['transect_id'] == transect_id]

        # Check if within tolerance of any manual boundary
        matches = []
        for _, man in manual_transect.iterrows():
            distance_error = abs(det_pos - man['position'])
            if distance_error <= tolerance_m:
                matches.append({
                    'manual_position': man['position'],
                    'manual_boundary_type': man['boundary_type'],
                    'distance_error': distance_error
                })

        # Classify detection
        if matches:
            # True Positive - match to closest manual boundary
            best_match = min(matches, key=lambda x: x['distance_error'])
            label = 'TP'
            manual_pos = best_match['manual_position']
            manual_type = best_match['manual_boundary_type']
            error = best_match['distance_error']
        else:
            # False Positive - no manual boundary nearby
            label = 'FP'
            manual_pos = np.nan
            manual_type = 'None'
            error = np.nan

        matched_detections.append({
            'transect_id': transect_id,
            'detected_position': det_pos,
            'label': label,
            'manual_position': manual_pos,
            'manual_boundary_type': manual_type,
            'distance_error': error,
            'detected_boundary_type': det.get('boundary_type', 'Unknown'),
            'confidence': det.get('confidence', np.nan),
            'magnitude': det.get('magnitude', np.nan),
            'nir_value': det.get('nir_value', np.nan),
            'index': det.get('index', np.nan)
        })

    # Find False Negatives (manual boundaries with no detection)
    for _, man in manual.iterrows():
        transect_id = man['transect_id']
        man_pos = man['position']

        # Check if any detection matched this manual boundary
        detected_transect = detected[detected['transect_id'] == transect_id]

        has_match = False
        for _, det in detected_transect.iterrows():
            if abs(det['distance'] - man_pos) <= tolerance_m:
                has_match = True
                break

        if not has_match:
            # False Negative
            matched_detections.append({
                'transect_id': transect_id,
                'detected_position': np.nan,
                'label': 'FN',
                'manual_position': man_pos,
                'manual_boundary_type': man['boundary_type'],
                'distance_error': np.nan,
                'detected_boundary_type': 'None',
                'confidence': np.nan,
                'magnitude': np.nan,
                'nir_value': np.nan,
                'index': np.nan
            })

    matches_df = pd.DataFrame(matched_detections)

    # Print classification summary
    print("\nClassification Summary:")
    counts = Counter(matches_df['label'])
    total = sum(counts.values())

    tp = counts['TP']
    fp = counts['FP']
    fn = counts['FN']

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  True Positives (TP):  {tp:3d} ({100*tp/total:5.1f}%)")
    print(f"  False Positives (FP): {fp:3d} ({100*fp/total:5.1f}%)")
    print(f"  False Negatives (FN): {fn:3d} ({100*fn/total:5.1f}%)")
    print(f"\nPerformance Metrics:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")

    if tp > 0:
        mae = matches_df[matches_df['label'] == 'TP']['distance_error'].mean()
        rmse = np.sqrt((matches_df[matches_df['label'] == 'TP']['distance_error']**2).mean())
        print(f"\nLocation Accuracy (for TP only):")
        print(f"  Mean Absolute Error (MAE): {mae:.2f}m")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}m")

    return matches_df


def extract_features_at_transitions(spectral_csv_path: Path, matches_df: pd.DataFrame, window_m: float = 5.0) -> pd.DataFrame:
    """
    Extract detailed spectral/derivative features at each transition location.

    Args:
        spectral_csv_path: Path to sampled spectral data
        matches_df: DataFrame with matched transitions
        window_m: Window size for feature extraction (meters)

    Returns:
        DataFrame with features for each transition
    """
    print(f"\nExtracting features at transitions (+-{window_m}m window)...")

    spectral_data = pd.read_csv(spectral_csv_path)

    transition_features = []

    for _, row in matches_df.iterrows():
        if pd.isna(row['detected_position']):
            continue  # Skip FN (no detected position)

        transect_id = row['transect_id']
        position = row['detected_position']

        # Get spectral data for this transect
        transect_data = spectral_data[spectral_data['TransectID'] == transect_id].copy()
        transect_data = transect_data.sort_values('distance')

        # Extract window around transition
        window_data = transect_data[
            (transect_data['distance'] >= position - window_m) &
            (transect_data['distance'] <= position + window_m)
        ].copy()

        if len(window_data) == 0:
            continue

        # Compute features
        feature_extractor = SpectralFeatures(window_data)
        features = feature_extractor.compute_all()

        # Extract key statistics
        feature_dict = {
            'transect_id': transect_id,
            'position': position,
            'label': row['label'],
            'manual_boundary_type': row['manual_boundary_type'],

            # NIR derivative features
            'nir_d1_mean': features['nir_d1_smooth'].mean(),
            'nir_d1_min': features['nir_d1_smooth'].min(),
            'nir_d1_max': features['nir_d1_smooth'].max(),
            'nir_d1_std': features['nir_d1_smooth'].std(),

            # NIR absolute change
            'nir_drop': window_data['nir'].iloc[0] - window_data['nir'].iloc[-1],

            # Brightness features
            'brightness_d1_mean': features['brightness_d1'].mean(),
            'brightness_drop': features['brightness'].iloc[0] - features['brightness'].iloc[-1],

            # Multi-band derivatives (check alignment)
            'red_d1_mean': np.gradient(window_data['red'], window_data['distance']).mean(),
            'green_d1_mean': np.gradient(window_data['green'], window_data['distance']).mean(),
            'blue_d1_mean': np.gradient(window_data['blue'], window_data['distance']).mean(),

            # Spectral angle change
            'spectral_angle_mean': features['spectral_angle'].mean(),
            'spectral_angle_max': features['spectral_angle'].max(),

            # Variability within window
            'nir_variability': features['nir'].std(),
            'brightness_variability': features['brightness'].std(),
        }

        transition_features.append(feature_dict)

    features_df = pd.DataFrame(transition_features)

    print(f"Extracted features for {len(features_df)} transitions")

    return features_df


def compare_tp_vs_fp(features_df: pd.DataFrame, output_dir: Path):
    """
    Compare feature distributions between TP and FP detections.

    Args:
        features_df: DataFrame with extracted features
        output_dir: Directory to save outputs
    """
    print("\nComparing TP vs FP feature distributions...")

    if len(features_df) == 0:
        print("  No features extracted (no detections) - skipping comparison")
        return

    tp_features = features_df[features_df['label'] == 'TP']
    fp_features = features_df[features_df['label'] == 'FP']

    if len(fp_features) == 0:
        print("  No false positives found - skipping comparison")
        return

    # Select features to compare
    feature_cols = [
        'nir_d1_mean', 'nir_d1_min', 'nir_d1_std',
        'nir_drop', 'brightness_drop',
        'red_d1_mean', 'green_d1_mean', 'blue_d1_mean',
        'spectral_angle_max', 'nir_variability'
    ]

    comparison = []

    for feature in feature_cols:
        tp_mean = tp_features[feature].mean()
        tp_std = tp_features[feature].std()
        fp_mean = fp_features[feature].mean()
        fp_std = fp_features[feature].std()

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(tp_features)-1)*tp_std**2 + (len(fp_features)-1)*fp_std**2) /
                            (len(tp_features) + len(fp_features) - 2))
        effect_size = (tp_mean - fp_mean) / pooled_std if pooled_std > 0 else 0

        comparison.append({
            'feature': feature,
            'TP_mean': tp_mean,
            'TP_std': tp_std,
            'FP_mean': fp_mean,
            'FP_std': fp_std,
            'difference': tp_mean - fp_mean,
            'effect_size': abs(effect_size)
        })

    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('effect_size', ascending=False)

    # Save to CSV
    output_path = output_dir / 'feature_comparison_TP_vs_FP.csv'
    comparison_df.to_csv(output_path, index=False)
    print(f"\nSaved feature comparison to: {output_path}")

    # Print top differentiating features
    print("\nTop Differentiating Features (by effect size):")
    print(comparison_df[['feature', 'TP_mean', 'FP_mean', 'difference', 'effect_size']].head(5).to_string(index=False))

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    top_features = comparison_df.head(6)['feature'].tolist()

    for i, feature in enumerate(top_features):
        ax = axes[i]

        tp_vals = tp_features[feature].dropna()
        fp_vals = fp_features[feature].dropna()

        ax.hist(tp_vals, bins=15, alpha=0.6, label='TP', color='green', density=True)
        ax.hist(fp_vals, bins=15, alpha=0.6, label='FP', color='red', density=True)

        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{feature}\n(effect size={comparison_df[comparison_df["feature"]==feature]["effect_size"].values[0]:.2f})',
                     fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'tp_vs_fp_distributions.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved distribution plots to: {plot_path}")


def plot_spectral_profiles_with_boundaries(
    spectral_csv_path: Path,
    manual_boundaries: pd.DataFrame,
    detected_transitions: pd.DataFrame,
    matches_df: pd.DataFrame,
    output_dir: Path,
    max_transects: int = 10
):
    """
    Create spectral profile plots showing both manual and algorithmic boundaries.

    This provides visual comparison to easily assess boundary detection quality.

    Args:
        spectral_csv_path: Path to sampled spectral data
        manual_boundaries: DataFrame with manual boundary positions
        detected_transitions: DataFrame with detected transitions
        matches_df: DataFrame with TP/FP/FN labels
        output_dir: Directory to save plots
        max_transects: Maximum number of transects to plot (default: 10)
    """
    print(f"\nGenerating spectral profile plots with boundary comparison...")

    spectral_data = pd.read_csv(spectral_csv_path)
    transect_ids = sorted(spectral_data['TransectID'].unique())[:max_transects]

    # Create a multi-panel figure
    n_transects = len(transect_ids)
    n_cols = 2
    n_rows = (n_transects + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, transect_id in enumerate(transect_ids):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Get spectral data for this transect
        transect_data = spectral_data[spectral_data['TransectID'] == transect_id].copy()
        transect_data = transect_data.sort_values('distance')

        # Plot NIR profile (primary detection signal)
        ax.plot(transect_data['distance'], transect_data['nir'],
                'k-', linewidth=1.5, label='NIR', alpha=0.7)

        # Plot manual boundaries (ground truth)
        manual_transect = manual_boundaries[manual_boundaries['transect_id'] == transect_id]
        for _, man_row in manual_transect.iterrows():
            ax.axvline(man_row['position'], color='blue', linestyle='--',
                      linewidth=2, alpha=0.8, label='Manual (Ground Truth)')

        # Plot detected boundaries (algorithmic)
        detected_transect = detected_transitions[detected_transitions['transect_id'] == transect_id]
        for _, det_row in detected_transect.iterrows():
            # Get match info to determine TP/FP
            match_info = matches_df[
                (matches_df['transect_id'] == transect_id) &
                (matches_df['detected_position'] == det_row['distance'])
            ]

            if len(match_info) > 0:
                label_type = match_info.iloc[0]['label']
                if label_type == 'TP':
                    color = 'green'
                    linestyle = '-'
                    alpha = 0.8
                    marker_label = 'Detected (TP)'
                else:  # FP
                    color = 'red'
                    linestyle = ':'
                    alpha = 0.6
                    marker_label = 'Detected (FP)'
            else:
                color = 'orange'
                linestyle = '-'
                alpha = 0.7
                marker_label = 'Detected'

            ax.axvline(det_row['distance'], color=color, linestyle=linestyle,
                      linewidth=2, alpha=alpha, label=marker_label)

        # Mark false negatives (manual boundaries with no detection)
        fn_transect = matches_df[
            (matches_df['transect_id'] == transect_id) &
            (matches_df['label'] == 'FN')
        ]
        for _, fn_row in fn_transect.iterrows():
            # Add a marker on the NIR profile at the manual position
            manual_pos = fn_row['manual_position']
            # Find closest data point
            closest_idx = (transect_data['distance'] - manual_pos).abs().idxmin()
            nir_value = transect_data.loc[closest_idx, 'nir']
            ax.plot(manual_pos, nir_value, 'rx', markersize=15,
                   markeredgewidth=3, label='Missed (FN)')

        # Labels and formatting
        ax.set_xlabel('Distance along transect (m)', fontsize=10)
        ax.set_ylabel('NIR reflectance', fontsize=10)
        ax.set_title(f'Transect {transect_id}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=8)

    # Remove empty subplots if odd number of transects
    if n_transects % 2 == 1:
        fig.delaxes(axes[n_rows-1, n_cols-1])

    plt.tight_layout()
    plot_path = output_dir / 'spectral_profiles_boundary_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved spectral profile comparison plots to: {plot_path}")
    print(f"  Blue dashed lines: Manual boundaries (ground truth)")
    print(f"  Green solid lines: True Positive detections")
    print(f"  Red dotted lines: False Positive detections")
    print(f"  Red X markers: False Negative (missed boundaries)")


def run_tp_fp_analysis(
    manual_csv_path: Path,
    spectral_csv_path: Path,
    output_dir: Path,
    tolerance_m: float = 10.0,
    boundary_type: str = 'DRY_WET'
) -> dict:
    """
    Main function to run complete TP vs FP analysis.

    Args:
        manual_csv_path: Path to classified_manual.csv
        spectral_csv_path: Path to sampled_spectral_data.csv
        output_dir: Directory to save outputs
        tolerance_m: Matching tolerance in meters
        boundary_type: Boundary type to filter for (DRY_WET for shell lines)

    Returns:
        Dictionary with analysis results
    """
    print("="*80)
    print("TRUE POSITIVE vs FALSE POSITIVE ANALYSIS")
    print(f"Boundary Type Filter: {boundary_type}")
    print("="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load manual boundaries
    manual_boundaries = load_manual_boundaries(manual_csv_path)

    # Filter manual boundaries for specified type
    # Map validation boundary type names to manual CSV format
    manual_boundary_type_map = {
        'DRY_WET': 'BEACH_DRY_to_BEACH_WET',
        'VEG_DRY': 'VEGETATED_DUNES_to_BEACH_DRY',
        'WET_WATER': 'BEACH_WET_to_WATER'
    }

    if boundary_type in manual_boundary_type_map:
        target_boundary = manual_boundary_type_map[boundary_type]
        n_before = len(manual_boundaries)
        manual_boundaries = manual_boundaries[manual_boundaries['boundary_type'] == target_boundary].copy()
        n_after = len(manual_boundaries)
        print(f"\nFiltered manual boundaries: {n_before} total -> {n_after} {boundary_type} boundaries")

    # Step 2: Detect transitions on training data
    detected_transitions = detect_transitions_on_training_data(spectral_csv_path)

    # Filter detected transitions for specified boundary type
    # Map validation boundary type to detector format (uses different naming convention)
    detector_boundary_type_map = {
        'DRY_WET': ['DRY_BEACH->BEACH_WET', 'BEACH_WET->DRY_BEACH'],
        'VEG_DRY': ['VEG_DUNES->DRY_BEACH', 'DRY_BEACH->VEG_DUNES',
                    'VEGETATED_DUNES->BEACH_DRY', 'BEACH_DRY->VEGETATED_DUNES'],
        'WET_WATER': ['BEACH_WET->WATER', 'WATER->BEACH_WET']
    }

    if boundary_type and len(detected_transitions) > 0:
        n_before = len(detected_transitions)
        target_types = detector_boundary_type_map.get(boundary_type, [])
        detected_transitions = detected_transitions[detected_transitions['boundary_type'].isin(target_types)].copy()
        n_after = len(detected_transitions)
        print(f"Filtered detected transitions: {n_before} total -> {n_after} {boundary_type} detections")

    # Step 3: Match detected to manual
    matches = match_transitions(detected_transitions, manual_boundaries, tolerance_m)

    # Save matches
    matches_path = output_dir / 'transition_matches.csv'
    matches.to_csv(matches_path, index=False)
    print(f"\nSaved transition matches to: {matches_path}")

    # Step 4: Extract features at transitions
    features = extract_features_at_transitions(spectral_csv_path, matches, window_m=5.0)

    # Step 5: Compare TP vs FP
    compare_tp_vs_fp(features, output_dir)

    # Step 6: Generate spectral profile plots with boundary comparison
    plot_spectral_profiles_with_boundaries(
        spectral_csv_path=spectral_csv_path,
        manual_boundaries=manual_boundaries,
        detected_transitions=detected_transitions,
        matches_df=matches,
        output_dir=output_dir,
        max_transects=10
    )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    return {
        'manual_boundaries': manual_boundaries,
        'detected_transitions': detected_transitions,
        'matches': matches,
        'features': features
    }


if __name__ == '__main__':
    # Setup paths
    base_dir = Path(__file__).parent.parent
    training_dir = base_dir / 'training_output'
    output_dir = base_dir / 'validation' / 'outputs'

    manual_csv = training_dir / 'classified_manual.csv'
    spectral_csv = training_dir / 'sampled_spectral_data.csv'

    # Run analysis
    results = run_tp_fp_analysis(
        manual_csv_path=manual_csv,
        spectral_csv_path=spectral_csv,
        output_dir=output_dir,
        tolerance_m=10.0
    )
