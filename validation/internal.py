"""
Within-Zone Analysis

Analyzes spectral and derivative characteristics within manually-demarcated zones
to understand internal variability and identify patterns that cause false positives.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.fft import fft, fftfreq

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from spectral_classifier.features import SpectralFeatures


def load_manual_zones(manual_csv_path: Path) -> pd.DataFrame:
    """
    Load manual zone classifications.

    Returns:
        DataFrame with zones
    """
    print("Loading manual zone classifications...")

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
                'CLASS': class_name,
                'span': end - start
            })

    zones_df = pd.DataFrame(manual_data)

    print(f"Loaded {len(zones_df)} zones from {len(zones_df['transectID'].unique())} transects")
    print("\nZone type distribution:")
    print(zones_df['CLASS'].value_counts())

    return zones_df


def analyze_zone_derivatives(
    spectral_csv_path: Path,
    zones_df: pd.DataFrame,
    derivative_threshold: float = -3.0
) -> pd.DataFrame:
    """
    Analyze derivative characteristics within each zone.

    Args:
        spectral_csv_path: Path to sampled spectral data
        zones_df: DataFrame with manual zones
        derivative_threshold: NIR derivative threshold for boundary detection

    Returns:
        DataFrame with zone-level statistics
    """
    print(f"\nAnalyzing derivative characteristics within zones...")
    print(f"Using derivative threshold: {derivative_threshold} units/m")

    spectral_data = pd.read_csv(spectral_csv_path)

    zone_stats = []

    for _, zone in zones_df.iterrows():
        transect_id = zone['transectID']
        start = zone['start[m]']
        end = zone['end[m]']
        zone_class = zone['CLASS']

        # Get spectral data for this zone
        transect_data = spectral_data[spectral_data['TransectID'] == transect_id].copy()
        zone_data = transect_data[
            (transect_data['distance'] >= start) &
            (transect_data['distance'] <= end)
        ].copy()

        if len(zone_data) < 3:
            continue  # Skip zones with insufficient data

        zone_data = zone_data.sort_values('distance').reset_index(drop=True)

        # Compute features
        feature_extractor = SpectralFeatures(zone_data)
        features = feature_extractor.compute_all()

        # NIR derivative statistics
        nir_d1 = features['nir_d1_smooth']

        # Count threshold crossings (potential false positives)
        threshold_crossings = (nir_d1 < derivative_threshold).sum()
        crossing_rate = threshold_crossings / len(nir_d1)

        # Sustained drops (3+ consecutive points below threshold)
        sustained_drops = 0
        in_drop = False
        drop_length = 0

        for val in nir_d1:
            if val < derivative_threshold:
                if in_drop:
                    drop_length += 1
                else:
                    in_drop = True
                    drop_length = 1
            else:
                if in_drop and drop_length >= 3:
                    sustained_drops += 1
                in_drop = False
                drop_length = 0

        if in_drop and drop_length >= 3:
            sustained_drops += 1

        # Spectral stability metrics
        nir = features['nir']
        brightness = features['brightness']

        # Coefficient of variation (CV = std/mean)
        nir_cv = nir.std() / nir.mean() if nir.mean() != 0 else 0
        brightness_cv = brightness.std() / brightness.mean() if brightness.mean() != 0 else 0

        # Linear trend analysis
        distances = features['distance'].values
        nir_slope, nir_intercept, nir_r, _, _ = linregress(distances, nir.values)
        brightness_slope, _, brightness_r, _, _ = linregress(distances, brightness.values)

        # Oscillation frequency (if water zone)
        dominant_frequency = np.nan
        if zone_class == 'WATER' and len(nir) > 10:
            # FFT analysis
            nir_detrended = nir - (nir_slope * distances + nir_intercept)
            fft_vals = fft(nir_detrended)
            freqs = fftfreq(len(nir_detrended), d=np.mean(np.diff(distances)))

            # Find dominant frequency (ignore DC component)
            power = np.abs(fft_vals[1:len(fft_vals)//2])
            if len(power) > 0:
                dominant_freq_idx = np.argmax(power)
                dominant_frequency = abs(freqs[dominant_freq_idx + 1])

        # Multi-band derivative alignment
        red_d1 = np.gradient(zone_data['red'], zone_data['distance']).mean()
        green_d1 = np.gradient(zone_data['green'], zone_data['distance']).mean()
        blue_d1 = np.gradient(zone_data['blue'], zone_data['distance']).mean()

        # Check if derivatives have same sign (aligned)
        derivatives = [nir_d1.mean(), red_d1, green_d1, blue_d1]
        all_negative = all(d < 0 for d in derivatives)
        all_positive = all(d > 0 for d in derivatives)
        derivative_alignment = 'aligned' if (all_negative or all_positive) else 'misaligned'

        zone_stats.append({
            'transect_id': transect_id,
            'zone_class': zone_class,
            'start': start,
            'end': end,
            'span': end - start,
            'num_points': len(zone_data),

            # Derivative statistics
            'nir_d1_mean': nir_d1.mean(),
            'nir_d1_std': nir_d1.std(),
            'nir_d1_min': nir_d1.min(),
            'nir_d1_max': nir_d1.max(),

            # Threshold crossing analysis
            'threshold_crossings': threshold_crossings,
            'crossing_rate': crossing_rate,
            'sustained_drops': sustained_drops,

            # Stability metrics
            'nir_cv': nir_cv,
            'brightness_cv': brightness_cv,
            'nir_variability': nir.std(),
            'brightness_variability': brightness.std(),

            # Trend analysis
            'nir_slope': nir_slope,
            'nir_r_squared': nir_r**2,
            'brightness_slope': brightness_slope,

            # Oscillation analysis
            'dominant_frequency': dominant_frequency,

            # Multi-band alignment
            'derivative_alignment': derivative_alignment,
            'red_d1_mean': red_d1,
            'green_d1_mean': green_d1,
            'blue_d1_mean': blue_d1,
        })

    stats_df = pd.DataFrame(zone_stats)

    print(f"Analyzed {len(stats_df)} zones")

    return stats_df


def test_hypotheses(stats_df: pd.DataFrame):
    """
    Test hypotheses about false positive patterns.

    Args:
        stats_df: DataFrame with zone statistics
    """
    print("\n" + "="*80)
    print("HYPOTHESIS TESTING")
    print("="*80)

    # H1: False positives occur in high-variability zones (especially WATER)
    print("\nH1: High-variability zones have more threshold crossings")
    print("-" * 60)

    for zone_class in stats_df['zone_class'].unique():
        class_data = stats_df[stats_df['zone_class'] == zone_class]

        print(f"\n{zone_class}:")
        print(f"  Mean threshold crossings: {class_data['threshold_crossings'].mean():.2f}")
        print(f"  Mean crossing rate: {class_data['crossing_rate'].mean():.3f}")
        print(f"  Mean sustained drops (3+ points): {class_data['sustained_drops'].mean():.2f}")
        print(f"  Mean NIR variability: {class_data['nir_variability'].mean():.2f}")
        print(f"  Mean NIR CV: {class_data['nir_cv'].mean():.3f}")

    # H2: True boundaries show sustained drops, not transient spikes
    print("\n\nH2: Sustained drops vs. transient threshold crossings")
    print("-" * 60)

    for zone_class in stats_df['zone_class'].unique():
        class_data = stats_df[stats_df['zone_class'] == zone_class]

        total_crossings = class_data['threshold_crossings'].sum()
        total_sustained = class_data['sustained_drops'].sum()

        sustained_ratio = total_sustained / total_crossings if total_crossings > 0 else 0

        print(f"\n{zone_class}:")
        print(f"  Total threshold crossings: {total_crossings}")
        print(f"  Total sustained drops: {total_sustained}")
        print(f"  Sustained ratio: {sustained_ratio:.3f}")

    # H3: Multi-band derivatives correlate at true boundaries
    print("\n\nH3: Multi-band derivative alignment")
    print("-" * 60)

    for zone_class in stats_df['zone_class'].unique():
        class_data = stats_df[stats_df['zone_class'] == zone_class]

        aligned_count = (class_data['derivative_alignment'] == 'aligned').sum()
        misaligned_count = (class_data['derivative_alignment'] == 'misaligned').sum()
        total = aligned_count + misaligned_count

        print(f"\n{zone_class}:")
        print(f"  Aligned:    {aligned_count}/{total} ({100*aligned_count/total:.1f}%)")
        print(f"  Misaligned: {misaligned_count}/{total} ({100*misaligned_count/total:.1f}%)")


def plot_zone_statistics(stats_df: pd.DataFrame, output_dir: Path):
    """
    Create visualizations of zone-level statistics.

    Args:
        stats_df: DataFrame with zone statistics
        output_dir: Directory to save plots
    """
    print("\nGenerating zone statistics plots...")

    # Plot 1: Threshold crossings by zone type
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: Threshold crossing rate
    ax = axes[0, 0]
    zone_types = stats_df['zone_class'].unique()
    crossing_rates = [stats_df[stats_df['zone_class'] == zt]['crossing_rate'].mean() for zt in zone_types]

    ax.bar(zone_types, crossing_rates, color=['green', 'orange', 'brown', 'blue'])
    ax.set_ylabel('Mean Crossing Rate', fontsize=11)
    ax.set_title('Threshold Crossing Rate by Zone Type', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(crossing_rates) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')

    # Subplot 2: NIR variability
    ax = axes[0, 1]
    variabilities = [stats_df[stats_df['zone_class'] == zt]['nir_variability'].mean() for zt in zone_types]

    ax.bar(zone_types, variabilities, color=['green', 'orange', 'brown', 'blue'])
    ax.set_ylabel('Mean NIR Std Dev', fontsize=11)
    ax.set_title('NIR Variability by Zone Type', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(variabilities) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')

    # Subplot 3: Sustained drops
    ax = axes[1, 0]
    sustained = [stats_df[stats_df['zone_class'] == zt]['sustained_drops'].mean() for zt in zone_types]

    ax.bar(zone_types, sustained, color=['green', 'orange', 'brown', 'blue'])
    ax.set_ylabel('Mean Sustained Drops', fontsize=11)
    ax.set_title('Sustained Derivative Drops (3+ points)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(sustained) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')

    # Subplot 4: Coefficient of variation
    ax = axes[1, 1]
    cvs = [stats_df[stats_df['zone_class'] == zt]['nir_cv'].mean() for zt in zone_types]

    ax.bar(zone_types, cvs, color=['green', 'orange', 'brown', 'blue'])
    ax.set_ylabel('Mean NIR CV', fontsize=11)
    ax.set_title('NIR Coefficient of Variation', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(cvs) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = output_dir / 'zone_variability_plots.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved zone variability plots to: {plot_path}")

    # Plot 2: Derivative distributions by zone
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, zone_type in enumerate(zone_types):
        if i >= 4:
            break

        ax = axes[i]
        zone_data = stats_df[stats_df['zone_class'] == zone_type]

        ax.hist(zone_data['nir_d1_mean'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(-3.0, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax.axvline(0, color='gray', linestyle=':', linewidth=1)

        ax.set_xlabel('NIR d/dx (units/m)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{zone_type} - NIR Derivative Distribution', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'derivative_distributions_by_zone.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved derivative distributions to: {plot_path}")


def run_internal_analysis(
    manual_csv_path: Path,
    spectral_csv_path: Path,
    output_dir: Path,
    derivative_threshold: float = -3.0
) -> dict:
    """
    Main function to run within-zone analysis.

    Args:
        manual_csv_path: Path to classified_manual.csv
        spectral_csv_path: Path to sampled_spectral_data.csv
        output_dir: Directory to save outputs
        derivative_threshold: NIR derivative threshold

    Returns:
        Dictionary with analysis results
    """
    print("="*80)
    print("WITHIN-ZONE ANALYSIS")
    print("="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load manual zones
    zones = load_manual_zones(manual_csv_path)

    # Step 2: Analyze zone derivatives and characteristics
    zone_stats = analyze_zone_derivatives(spectral_csv_path, zones, derivative_threshold)

    # Save statistics
    stats_path = output_dir / 'within_zone_statistics.csv'
    zone_stats.to_csv(stats_path, index=False)
    print(f"\nSaved zone statistics to: {stats_path}")

    # Step 3: Test hypotheses
    test_hypotheses(zone_stats)

    # Step 4: Create visualizations
    plot_zone_statistics(zone_stats, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    return {
        'zones': zones,
        'zone_stats': zone_stats
    }


if __name__ == '__main__':
    # Setup paths
    base_dir = Path(__file__).parent.parent
    training_dir = base_dir / 'training_output'
    output_dir = base_dir / 'validation' / 'outputs'

    manual_csv = training_dir / 'classified_manual.csv'
    spectral_csv = training_dir / 'sampled_spectral_data.csv'

    # Run analysis
    results = run_internal_analysis(
        manual_csv_path=manual_csv,
        spectral_csv_path=spectral_csv,
        output_dir=output_dir,
        derivative_threshold=-3.0
    )
