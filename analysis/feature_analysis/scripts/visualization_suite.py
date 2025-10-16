"""
Visualization Suite for Feature Analysis

Generates comprehensive visualizations of spectral characteristics and transition signatures.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True

# Paths
STATS_DIR = Path(__file__).parent.parent / 'outputs' / 'statistics'
VIZ_DIR = Path(__file__).parent.parent / 'outputs' / 'visualizations'
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme matching config.py
LANDCOVER_COLORS = {
    'VEGETATED_DUNES': '#228B22',
    'BEACH_DRY': '#F4A460',
    'BEACH_WET': '#CD853F',
    'WATER': '#1E90FF',
}

print('=' * 80)
print('VISUALIZATION SUITE')
print('=' * 80)
print(f'Loading data from: {STATS_DIR}')
print(f'Saving visualizations to: {VIZ_DIR}')
print()

# Load data
class_profiles = pd.read_csv(STATS_DIR / 'class_spectral_profiles.csv')
transitions = pd.read_csv(STATS_DIR / 'transition_characteristics.csv')

print(f'Loaded {len(class_profiles)} class feature records')
print(f'Loaded {len(transitions)} transition records')
print()

# =============================================================================
# 1. LANDCOVER CLASS PROFILES
# =============================================================================

print('Generating landcover class profile visualizations...')

# Box plots for key discriminative features
key_features = [
    ('brightness_mean', 'Brightness', 'units'),
    ('nir_mean', 'NIR', 'units'),
    ('nir_ratio_mean', 'NIR Ratio', 'ratio'),
    ('ndvi_mean', 'NDVI', 'index'),
    ('ndwi_mean', 'NDWI', 'index'),
    ('variability_mean', 'Variability', 'std'),
    ('rg_ratio_mean', 'R/G Ratio', 'ratio'),
]

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for idx, (feature_name, feature_label, unit) in enumerate(key_features):
    if idx >= len(axes):
        break

    ax = axes[idx]

    # Get data for this feature
    feature_data = class_profiles[class_profiles['feature'] == feature_name].copy()

    if len(feature_data) == 0:
        ax.text(0.5, 0.5, f'No data for {feature_label}',
                ha='center', va='center', transform=ax.transAxes)
        continue

    # Prepare data for box plot
    classes_order = ['VEGETATED_DUNES', 'BEACH_DRY', 'BEACH_WET', 'WATER']
    data_to_plot = []
    labels_to_plot = []
    colors_to_plot = []

    for cls in classes_order:
        cls_data = feature_data[feature_data['class'] == cls]
        if len(cls_data) > 0:
            # Use mean as the value
            data_to_plot.append(cls_data['mean'].values[0])
            labels_to_plot.append(cls.replace('VEGETATED_DUNES', 'VEG_DUNES'))
            colors_to_plot.append(LANDCOVER_COLORS.get(cls, '#808080'))

    # Create bar plot with error bars
    x_pos = np.arange(len(labels_to_plot))
    means = data_to_plot
    stds = []
    for cls in classes_order:
        cls_data = feature_data[feature_data['class'] == cls]
        if len(cls_data) > 0:
            stds.append(cls_data['std'].values[0])

    ax.bar(x_pos, means, yerr=stds, color=colors_to_plot, alpha=0.7, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_to_plot, rotation=45, ha='right')
    ax.set_ylabel(f'{feature_label} ({unit})')
    ax.set_title(feature_label)
    ax.grid(axis='y', alpha=0.3)

# Remove empty subplots
for idx in range(len(key_features), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
output_file = VIZ_DIR / 'class_profiles_key_features.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'  Saved: {output_file.name}')
plt.close()

# =============================================================================
# 2. FEATURE DISCRIMINABILITY COMPARISON
# =============================================================================

print('Generating feature discriminability comparison...')

# Group by feature and compute range across classes
feature_ranges = []
for feature_name in class_profiles['feature'].unique():
    feature_data = class_profiles[class_profiles['feature'] == feature_name]
    if len(feature_data) < 2:
        continue

    mean_range = feature_data['mean'].max() - feature_data['mean'].min()
    avg_std = feature_data['std'].mean()

    # Coefficient of variation across classes
    if avg_std > 0:
        cv = mean_range / avg_std
    else:
        cv = 0

    feature_ranges.append({
        'feature': feature_name,
        'range': mean_range,
        'avg_std': avg_std,
        'cv': cv
    })

feature_ranges_df = pd.DataFrame(feature_ranges)
feature_ranges_df = feature_ranges_df.sort_values('cv', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(len(feature_ranges_df))
ax.barh(y_pos, feature_ranges_df['cv'], color='steelblue', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([f.replace('_mean', '').replace('_', ' ').title()
                     for f in feature_ranges_df['feature']])
ax.set_xlabel('Discriminability (Range / Avg Std)')
ax.set_title('Top 15 Most Discriminative Features Across Classes')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
output_file = VIZ_DIR / 'feature_discriminability_ranking.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'  Saved: {output_file.name}')
plt.close()

# =============================================================================
# 3. TRANSITION SIGNATURES BY TYPE
# =============================================================================

print('Generating transition signature comparisons...')

# Key transition metrics
transition_metrics = [
    ('nir_drop_absolute', 'NIR Drop (units)'),
    ('brightness_drop_absolute', 'Brightness Drop (units)'),
    ('nir_d1_peak', 'NIR Derivative Peak (units/m)'),
    ('nir_sustained_drop_points', 'Sustained Drop (points)'),
    ('transition_width_nir', 'Transition Width (m)'),
    ('variability_ratio', 'Variability Ratio'),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

transition_types = ['VEGETATED_DUNES->BEACH_DRY',
                   'BEACH_DRY->BEACH_WET', 'BEACH_WET->WATER']
transition_colors = {
    'VEGETATED_DUNES->BEACH_DRY': '#90EE90',
    'BEACH_DRY->BEACH_WET': '#FF6347',
    'BEACH_WET->WATER': '#4169E1',
}

for idx, (metric, label) in enumerate(transition_metrics):
    if idx >= len(axes):
        break

    ax = axes[idx]

    data_to_plot = []
    labels_to_plot = []
    colors_to_plot = []

    for trans_type in transition_types:
        trans_data = transitions[transitions['transition_type'] == trans_type]
        if len(trans_data) > 0 and metric in trans_data.columns:
            values = trans_data[metric].dropna()
            if len(values) > 0:
                data_to_plot.append(values.values)
                # Shorten labels
                short_label = trans_type.replace('VEGETATED_DUNES', 'VEG')
                labels_to_plot.append(short_label)
                colors_to_plot.append(transition_colors.get(trans_type, '#808080'))

    if len(data_to_plot) > 0:
        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_to_plot):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(label)
        ax.set_title(label.split('(')[0].strip())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Adjust x-axis labels
        ax.set_xticklabels(labels_to_plot, rotation=45, ha='right', fontsize=8)

plt.tight_layout()
output_file = VIZ_DIR / 'transition_signatures_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'  Saved: {output_file.name}')
plt.close()

# =============================================================================
# 4. SHELL LINE DETAILED PROFILE
# =============================================================================

print('Generating shell line detailed profile...')

shell_line = transitions[transitions['transition_type'] == 'BEACH_DRY->BEACH_WET'].copy()

if len(shell_line) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # NIR characteristics
    ax = axes[0, 0]
    metrics = ['nir_before_mean', 'nir_after_mean']
    labels = ['Before\n(Dry)', 'After\n(Wet)']
    colors = ['#F4A460', '#CD853F']
    data = [shell_line['nir_before_mean'].values, shell_line['nir_after_mean'].values]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('NIR (units)')
    ax.set_title('NIR Before vs After Shell Line')
    ax.grid(axis='y', alpha=0.3)

    # NIR drop distribution
    ax = axes[0, 1]
    ax.hist(shell_line['nir_drop_absolute'].dropna(), bins=10, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(shell_line['nir_drop_absolute'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.set_xlabel('NIR Drop (units)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'NIR Drop Distribution\nMean: {shell_line["nir_drop_absolute"].mean():.1f} ± {shell_line["nir_drop_absolute"].std():.1f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # NIR derivative peak
    ax = axes[0, 2]
    ax.hist(shell_line['nir_d1_peak'].dropna(), bins=10, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(shell_line['nir_d1_peak'].mean(), color='darkblue', linestyle='--', linewidth=2, label='Mean')
    ax.set_xlabel('NIR Derivative Peak (units/m)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'NIR Derivative Peak\nMean: {shell_line["nir_d1_peak"].mean():.1f} ± {shell_line["nir_d1_peak"].std():.1f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Brightness before vs after
    ax = axes[1, 0]
    data = [shell_line['brightness_before_mean'].values, shell_line['brightness_after_mean'].values]
    bp = ax.boxplot(data, labels=['Before\n(Dry)', 'After\n(Wet)'], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Brightness (units)')
    ax.set_title('Brightness Before vs After Shell Line')
    ax.grid(axis='y', alpha=0.3)

    # Transition width
    ax = axes[1, 1]
    ax.hist(shell_line['transition_width_nir'].dropna(), bins=8, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(shell_line['transition_width_nir'].mean(), color='darkgreen', linestyle='--', linewidth=2, label='Mean')
    ax.set_xlabel('Transition Width (m)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Transition Width (80% change)\nMean: {shell_line["transition_width_nir"].mean():.1f} ± {shell_line["transition_width_nir"].std():.1f} m')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Sustained drop points
    ax = axes[1, 2]
    ax.hist(shell_line['nir_sustained_drop_points'].dropna(), bins=range(4, 17),
            color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(shell_line['nir_sustained_drop_points'].mean(), color='darkviolet',
               linestyle='--', linewidth=2, label='Mean')
    ax.set_xlabel('Sustained Drop (consecutive points)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Sustained Drop Duration\nMean: {shell_line["nir_sustained_drop_points"].mean():.1f} ± {shell_line["nir_sustained_drop_points"].std():.1f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Shell Line (BEACH_DRY -> BEACH_WET) Detailed Characteristics',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_file = VIZ_DIR / 'shell_line_detailed_profile.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'  Saved: {output_file.name}')
    plt.close()

# =============================================================================
# 5. CORRELATION HEATMAP - SHELL LINE FEATURES
# =============================================================================

print('Generating shell line feature correlation heatmap...')

if len(shell_line) > 0:
    # Select numeric columns for correlation
    numeric_cols = [
        'nir_drop_absolute', 'brightness_drop_absolute',
        'nir_d1_peak', 'brightness_d1_peak',
        'nir_ratio_change', 'rg_ratio_at_boundary',
        'transition_width_nir', 'band_synchrony_std',
        'nir_sustained_drop_points', 'variability_ratio'
    ]

    # Filter to available columns
    available_cols = [col for col in numeric_cols if col in shell_line.columns]

    if len(available_cols) > 3:
        corr_data = shell_line[available_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))

        # Manual heatmap creation without seaborn
        im = ax.imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation', rotation=270, labelpad=20)

        # Add text annotations
        for i in range(len(corr_data)):
            for j in range(len(corr_data.columns)):
                text = ax.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

        ax.set_xticks(np.arange(len(corr_data.columns)))
        ax.set_yticks(np.arange(len(corr_data.index)))
        ax.set_title('Shell Line Feature Correlations', fontsize=14, fontweight='bold')

        # Rotate labels for readability
        labels = [col.replace('_', ' ').title() for col in available_cols]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)

        plt.tight_layout()
        output_file = VIZ_DIR / 'shell_line_correlation_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f'  Saved: {output_file.name}')
        plt.close()

# =============================================================================
# 6. SCATTER PLOTS - KEY FEATURE RELATIONSHIPS
# =============================================================================

print('Generating feature relationship scatter plots...')

if len(shell_line) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # NIR drop vs derivative peak
    ax = axes[0, 0]
    ax.scatter(shell_line['nir_drop_absolute'], shell_line['nir_d1_peak'],
               color='steelblue', s=100, alpha=0.6, edgecolors='black')
    ax.set_xlabel('NIR Drop (units)')
    ax.set_ylabel('NIR Derivative Peak (units/m)')
    ax.set_title('NIR Drop vs Derivative Peak')
    ax.grid(alpha=0.3)

    # NIR drop vs transition width
    ax = axes[0, 1]
    ax.scatter(shell_line['nir_drop_absolute'], shell_line['transition_width_nir'],
               color='green', s=100, alpha=0.6, edgecolors='black')
    ax.set_xlabel('NIR Drop (units)')
    ax.set_ylabel('Transition Width (m)')
    ax.set_title('NIR Drop vs Transition Width')
    ax.grid(alpha=0.3)

    # NIR derivative vs sustained drop
    ax = axes[1, 0]
    ax.scatter(shell_line['nir_d1_peak'], shell_line['nir_sustained_drop_points'],
               color='coral', s=100, alpha=0.6, edgecolors='black')
    ax.set_xlabel('NIR Derivative Peak (units/m)')
    ax.set_ylabel('Sustained Drop (points)')
    ax.set_title('Derivative Peak vs Sustained Drop')
    ax.grid(alpha=0.3)

    # Brightness drop vs NIR drop
    ax = axes[1, 1]
    ax.scatter(shell_line['nir_drop_absolute'], shell_line['brightness_drop_absolute'],
               color='purple', s=100, alpha=0.6, edgecolors='black')
    ax.set_xlabel('NIR Drop (units)')
    ax.set_ylabel('Brightness Drop (units)')
    ax.set_title('NIR Drop vs Brightness Drop')
    ax.grid(alpha=0.3)

    plt.suptitle('Shell Line Feature Relationships', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_file = VIZ_DIR / 'shell_line_feature_relationships.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'  Saved: {output_file.name}')
    plt.close()

# =============================================================================
# 7. SUMMARY STATISTICS TABLE (as image)
# =============================================================================

print('Generating summary statistics table...')

if len(shell_line) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare summary data
    summary_data = []
    summary_data.append(['Metric', 'Mean ± Std', 'Min', 'Max', 'n'])
    summary_data.append(['=' * 40, '=' * 20, '=' * 10, '=' * 10, '=' * 5])

    metrics_for_table = [
        ('nir_drop_absolute', 'NIR Drop (units)'),
        ('nir_before_mean', 'NIR Before (units)'),
        ('nir_after_mean', 'NIR After (units)'),
        ('nir_d1_peak', 'NIR Deriv Peak (units/m)'),
        ('nir_sustained_drop_points', 'Sustained Drop (pts)'),
        ('brightness_drop_absolute', 'Brightness Drop (units)'),
        ('brightness_before_mean', 'Brightness Before'),
        ('brightness_after_mean', 'Brightness After'),
        ('transition_width_nir', 'Width (m)'),
        ('nir_ratio_change', 'NIR Ratio Change'),
        ('rg_ratio_at_boundary', 'R/G Ratio'),
        ('variability_ratio', 'Variability Ratio'),
        ('band_synchrony_std', 'Band Sync Std (m)'),
    ]

    for col, label in metrics_for_table:
        if col in shell_line.columns:
            values = shell_line[col].dropna()
            if len(values) > 0:
                mean_str = f'{values.mean():.2f} ± {values.std():.2f}'
                min_str = f'{values.min():.2f}'
                max_str = f'{values.max():.2f}'
                n_str = f'{len(values)}'
                summary_data.append([label, mean_str, min_str, max_str, n_str])

    table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                    colWidths=[0.4, 0.25, 0.12, 0.12, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style separator row
    for i in range(5):
        table[(1, i)].set_facecolor('#d0d0d0')

    plt.title('Shell Line Summary Statistics (BEACH_DRY -> BEACH_WET)',
              fontsize=14, fontweight='bold', pad=20)

    output_file = VIZ_DIR / 'shell_line_summary_table.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'  Saved: {output_file.name}')
    plt.close()

# =============================================================================
# 8. COMPARISON: ALL BOUNDARY TYPES
# =============================================================================

print('Generating boundary type comparison...')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Define metrics to compare
comparison_metrics = [
    ('nir_drop_absolute', 'NIR Drop (units)'),
    ('nir_d1_peak', 'NIR Derivative Peak (units/m)'),
    ('nir_sustained_drop_points', 'Sustained Drop (points)'),
    ('transition_width_nir', 'Transition Width (m)'),
]

for idx, (metric, label) in enumerate(comparison_metrics):
    ax = axes[idx // 2, idx % 2]

    data_by_type = []
    labels_by_type = []

    for trans_type in transition_types:
        trans_data = transitions[transitions['transition_type'] == trans_type]
        if len(trans_data) > 0 and metric in trans_data.columns:
            values = trans_data[metric].dropna()
            if len(values) > 0:
                data_by_type.append(values.values)
                short_label = trans_type.replace('VEGETATED_DUNES', 'VEG')
                labels_by_type.append(short_label)

    if len(data_by_type) > 0:
        bp = ax.boxplot(data_by_type, labels=labels_by_type, patch_artist=True)

        # Color code
        colors_map = ['#90EE90', '#FF6347', '#4169E1']
        for patch, color in zip(bp['boxes'], colors_map[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(label)
        ax.set_title(label.split('(')[0].strip())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(labels_by_type, rotation=45, ha='right', fontsize=9)

        # Highlight shell line
        shell_idx = [i for i, lbl in enumerate(labels_by_type) if 'BEACH_DRY->BEACH_WET' in lbl]
        if shell_idx:
            bp['boxes'][shell_idx[0]].set_linewidth(3)
            bp['boxes'][shell_idx[0]].set_edgecolor('red')

plt.suptitle('Boundary Type Comparison (Shell Line Highlighted in Red)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
output_file = VIZ_DIR / 'boundary_type_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'  Saved: {output_file.name}')
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================

print()
print('=' * 80)
print('VISUALIZATION SUITE COMPLETE')
print('=' * 80)
print(f'Generated visualizations saved to: {VIZ_DIR}')
print()
print('Files created:')
viz_files = sorted(VIZ_DIR.glob('*.png'))
for f in viz_files:
    print(f'  - {f.name}')
print()
print(f'Total: {len(viz_files)} visualization files')
