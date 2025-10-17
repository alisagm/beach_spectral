"""
Feature Importance Analysis for Beach Spectral Classifier

Evaluates the diagnostic value of features (including R/G ratio from Phase 1)
for distinguishing between landcover classes.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from spectral_classifier.features import SpectralFeatures


def load_manual_classifications(manual_csv_path: Path) -> pd.DataFrame:
    """
    Load manual zone classifications.

    Returns:
        DataFrame with manual classifications
    """
    print("Loading manual classifications...")

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
                class_name = remaining.strip()

            manual_data.append({
                'transectID': tid,
                'start': start,
                'end': end,
                'CLASS': class_name
            })

    manual_df = pd.DataFrame(manual_data)

    print(f"Loaded {len(manual_df)} zones from {len(manual_df['transectID'].unique())} transects")
    print(f"\nClass distribution:")
    print(manual_df['CLASS'].value_counts())

    return manual_df


def extract_features_by_class(
    spectral_csv_path: Path,
    manual_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract all features for each point along with its manual classification.

    Returns:
        DataFrame with features and manual class labels
    """
    print("\nExtracting features for all points...")

    spectral_data = pd.read_csv(spectral_csv_path)

    all_features = []

    for tid in manual_df['transectID'].unique():
        # Get transect data
        transect_data = spectral_data[spectral_data['TransectID'] == tid].copy()
        transect_data = transect_data.sort_values('distance').reset_index(drop=True)

        # Compute features
        feature_extractor = SpectralFeatures(transect_data)
        features = feature_extractor.compute_all()

        # Add manual classifications
        transect_manual = manual_df[manual_df['transectID'] == tid]

        for idx, row in features.iterrows():
            distance = row['distance']

            # Find which zone this point belongs to
            zone = transect_manual[
                (transect_manual['start'] <= distance) &
                (transect_manual['end'] >= distance)
            ]

            if len(zone) > 0:
                manual_class = zone.iloc[0]['CLASS']

                feature_row = row.to_dict()
                feature_row['manual_class'] = manual_class
                feature_row['TransectID'] = tid
                all_features.append(feature_row)

    features_df = pd.DataFrame(all_features)

    print(f"Extracted features for {len(features_df)} points")
    print(f"\nPoints per class:")
    print(features_df['manual_class'].value_counts())

    return features_df


def compute_class_statistics(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean and std dev for each feature by class.

    Returns:
        DataFrame with feature statistics by class
    """
    print("\nComputing class-wise feature statistics...")

    # Select numeric features to analyze
    feature_cols = [
        'brightness', 'variability', 'ndvi', 'ndwi',
        'nir_ratio', 'blue_red_ratio',
        'red_green_ratio',  # PHASE 1 feature
        'nir', 'red', 'green', 'blue'
    ]

    # Filter to only features that exist
    feature_cols = [f for f in feature_cols if f in features_df.columns]

    stats = []

    for feature in feature_cols:
        for class_name in features_df['manual_class'].unique():
            class_data = features_df[features_df['manual_class'] == class_name][feature]

            stats.append({
                'feature': feature,
                'class': class_name,
                'mean': class_data.mean(),
                'std': class_data.std(),
                'min': class_data.min(),
                'max': class_data.max(),
                'n': len(class_data)
            })

    stats_df = pd.DataFrame(stats)

    return stats_df


def compute_feature_discriminability(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute how well each feature discriminates between classes.

    Uses ANOVA F-statistic and effect sizes between key class pairs.

    Returns:
        DataFrame with discriminability metrics sorted by importance
    """
    print("\nComputing feature discriminability...")

    feature_cols = [
        'brightness', 'variability', 'ndvi', 'ndwi',
        'nir_ratio', 'blue_red_ratio',
        'red_green_ratio',  # PHASE 1 feature
        'nir', 'red', 'green', 'blue'
    ]

    # Filter to only features that exist
    feature_cols = [f for f in feature_cols if f in features_df.columns]

    results = []

    for feature in feature_cols:
        # Get values by class
        class_values = {}
        for class_name in features_df['manual_class'].unique():
            class_values[class_name] = features_df[
                features_df['manual_class'] == class_name
            ][feature].dropna().values

        # ANOVA F-statistic (tests if means differ across groups)
        groups = list(class_values.values())
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            f_stat, p_value = f_oneway(*groups)
        else:
            f_stat, p_value = 0, 1.0

        # Effect size for key class pairs
        # DRY_BEACH vs BEACH_WET (shell-line)
        dry_wet_effect = compute_cohens_d(
            class_values.get('BEACH_DRY', np.array([])),
            class_values.get('BEACH_WET', np.array([]))
        )

        # BEACH_WET vs WATER (waterline)
        wet_water_effect = compute_cohens_d(
            class_values.get('BEACH_WET', np.array([])),
            class_values.get('WATER', np.array([]))
        )

        # VEG_DUNES vs BEACH_DRY (vegetation boundary)
        veg_dry_effect = compute_cohens_d(
            class_values.get('VEGETATED_DUNES', np.array([])),
            class_values.get('BEACH_DRY', np.array([]))
        )

        # Overall discriminability score (weighted average of effect sizes)
        overall_score = (abs(dry_wet_effect) + abs(wet_water_effect) + abs(veg_dry_effect)) / 3

        results.append({
            'feature': feature,
            'f_statistic': f_stat,
            'p_value': p_value,
            'dry_wet_effect': dry_wet_effect,
            'wet_water_effect': wet_water_effect,
            'veg_dry_effect': veg_dry_effect,
            'overall_discriminability': overall_score
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('overall_discriminability', ascending=False)

    return results_df


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Returns:
        Effect size (positive means group1 > group2)
    """
    if len(group1) == 0 or len(group2) == 0:
        return 0.0

    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def plot_feature_distributions(features_df: pd.DataFrame, output_dir: Path):
    """
    Create distribution plots for top discriminating features.

    Args:
        features_df: DataFrame with features and manual classes
        output_dir: Directory to save plots
    """
    print("\nGenerating feature distribution plots...")

    # Get top features
    discriminability = compute_feature_discriminability(features_df)
    top_features = discriminability.head(9)['feature'].tolist()

    # Create 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    classes = features_df['manual_class'].unique()
    colors = {'VEGETATED_DUNES': 'green', 'BEACH_DRY': 'orange',
              'BEACH_WET': 'brown', 'WATER': 'blue'}

    for i, feature in enumerate(top_features):
        ax = axes[i]

        for class_name in classes:
            class_data = features_df[features_df['manual_class'] == class_name][feature].dropna()

            ax.hist(class_data, bins=20, alpha=0.5, label=class_name,
                   color=colors.get(class_name, 'gray'), density=True)

        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{feature}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'feature_distributions_by_class.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved feature distribution plots to: {plot_path}")


def plot_feature_importance_ranking(discriminability_df: pd.DataFrame, output_dir: Path):
    """
    Create bar chart showing feature importance ranking.

    Args:
        discriminability_df: DataFrame with discriminability scores
        output_dir: Directory to save plots
    """
    print("Generating feature importance ranking plot...")

    fig, ax = plt.subplots(figsize=(12, 8))

    features = discriminability_df['feature'].values
    scores = discriminability_df['overall_discriminability'].values

    # Highlight R/G ratio
    colors = ['red' if f == 'red_green_ratio' else 'steelblue' for f in features]

    bars = ax.barh(features, scores, color=colors, alpha=0.7)

    ax.set_xlabel('Overall Discriminability Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance for Class Discrimination', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (feature, score) in enumerate(zip(features, scores)):
        ax.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=9)

    # Add legend for R/G ratio
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='R/G Ratio (Phase 1)'),
        Patch(facecolor='steelblue', alpha=0.7, label='Other features')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plot_path = output_dir / 'feature_importance_ranking.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved feature importance ranking to: {plot_path}")


def plot_pairwise_effect_sizes(discriminability_df: pd.DataFrame, output_dir: Path):
    """
    Create heatmap showing effect sizes for key boundary pairs.

    Args:
        discriminability_df: DataFrame with effect sizes
        output_dir: Directory to save plots
    """
    print("Generating pairwise effect size heatmap...")

    # Prepare data for heatmap
    features = discriminability_df['feature'].values
    boundaries = ['VEG->DRY', 'DRY->WET', 'WET->WATER']

    data = np.array([
        discriminability_df['veg_dry_effect'].values,
        discriminability_df['dry_wet_effect'].values,
        discriminability_df['wet_water_effect'].values
    ]).T

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 10))

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-2, vmax=2)

    # Set ticks
    ax.set_xticks(np.arange(len(boundaries)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(boundaries, fontsize=11)
    ax.set_yticklabels(features, fontsize=10)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cohen's d (Effect Size)", fontsize=11, fontweight='bold')

    # Add values in cells
    for i in range(len(features)):
        for j in range(len(boundaries)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title('Feature Effect Sizes for Key Boundaries', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plot_path = output_dir / 'feature_effect_sizes_heatmap.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved effect size heatmap to: {plot_path}")


def run_feature_importance_analysis(
    manual_csv_path: Path,
    spectral_csv_path: Path,
    output_dir: Path
) -> dict:
    """
    Main function to run feature importance analysis.

    Args:
        manual_csv_path: Path to classified_manual.csv
        spectral_csv_path: Path to sampled_spectral_data.csv
        output_dir: Directory to save outputs

    Returns:
        Dictionary with analysis results
    """
    print("="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load manual classifications
    manual_df = load_manual_classifications(manual_csv_path)

    # Step 2: Extract features with manual labels
    features_df = extract_features_by_class(spectral_csv_path, manual_df)

    # Step 3: Compute class-wise statistics
    stats_df = compute_class_statistics(features_df)

    # Save statistics
    stats_path = output_dir / 'feature_statistics_by_class.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"\nSaved feature statistics to: {stats_path}")

    # Step 4: Compute discriminability
    discriminability_df = compute_feature_discriminability(features_df)

    # Save discriminability
    discrim_path = output_dir / 'feature_discriminability.csv'
    discriminability_df.to_csv(discrim_path, index=False)
    print(f"Saved feature discriminability to: {discrim_path}")

    # Print top features
    print("\n" + "="*80)
    print("TOP DISCRIMINATING FEATURES")
    print("="*80)
    print(discriminability_df[['feature', 'overall_discriminability', 'dry_wet_effect']].head(10).to_string(index=False))

    # Step 5: Generate visualizations
    plot_feature_distributions(features_df, output_dir)
    plot_feature_importance_ranking(discriminability_df, output_dir)
    plot_pairwise_effect_sizes(discriminability_df, output_dir)

    # Print R/G ratio analysis
    print("\n" + "="*80)
    print("R/G RATIO ANALYSIS (Phase 1 Feature)")
    print("="*80)

    if 'red_green_ratio' in discriminability_df['feature'].values:
        rg_row = discriminability_df[discriminability_df['feature'] == 'red_green_ratio'].iloc[0]
        rg_rank = discriminability_df[discriminability_df['feature'] == 'red_green_ratio'].index[0] + 1

        print(f"Ranking: #{rg_rank} out of {len(discriminability_df)} features")
        print(f"Overall Discriminability Score: {rg_row['overall_discriminability']:.4f}")
        print(f"Effect Sizes:")
        print(f"  VEG->DRY:     {rg_row['veg_dry_effect']:.4f}")
        print(f"  DRY->WET:     {rg_row['dry_wet_effect']:.4f}")
        print(f"  WET->WATER:   {rg_row['wet_water_effect']:.4f}")
        print(f"F-statistic:  {rg_row['f_statistic']:.2f} (p={rg_row['p_value']:.4e})")

        # Recommendation
        if rg_rank <= 5:
            print("\n[+] CONCLUSION: R/G ratio is a HIGHLY DISCRIMINATIVE feature")
            print("  Recommendation: Consider incorporating into classification rules")
        elif rg_rank <= 10:
            print("\n[+] CONCLUSION: R/G ratio is a MODERATELY DISCRIMINATIVE feature")
            print("  Recommendation: Current use in boundary detection (Phase 3) is appropriate")
        else:
            print("\n[o] CONCLUSION: R/G ratio has LIMITED discriminative value")
            print("  Recommendation: Maintain for boundary validation; low priority for classification")
    else:
        print("WARNING: R/G ratio not found in features")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    return {
        'features': features_df,
        'statistics': stats_df,
        'discriminability': discriminability_df
    }


if __name__ == '__main__':
    # Setup paths
    base_dir = Path(__file__).parent.parent
    training_dir = base_dir / 'training_output' / 'seed_612823'
    output_dir = base_dir / 'validation' / 'outputs_phase5'

    manual_csv = training_dir / 'classified_manual.csv'
    spectral_csv = training_dir / 'sampled_spectral_data.csv'

    # Run analysis
    results = run_feature_importance_analysis(
        manual_csv_path=manual_csv,
        spectral_csv_path=spectral_csv,
        output_dir=output_dir
    )
