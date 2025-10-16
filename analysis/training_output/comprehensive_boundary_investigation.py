"""
Comprehensive Boundary Investigation Script

This script performs deep analysis to understand WHY boundaries are being missed,
particularly VEG_DUNES → BEACH_DRY boundaries (50% of false negatives).

Analysis Components:
1. Spectral profiles at all boundary types (detected vs missed)
2. Multi-scale smoothing experiment (windows 1, 3, 5, 7, 9, 11)
3. Signal-to-noise ratio by zone type
4. Alternative signal processing methods comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.ndimage import uniform_filter1d, gaussian_filter1d, median_filter
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("COMPREHENSIVE BOUNDARY INVESTIGATION")
print("=" * 80)

# ============================================================================
# PART 1: Load Data
# ============================================================================

print("\n[1/6] Loading data...")

# Load manual boundaries
manual_data = []
with open('classified_manual.csv', 'r') as f:
    headers = f.readline()
    for line in f:
        parts = line.strip().split(',', 3)
        tid = int(parts[0])
        start = float(parts[1])
        end = float(parts[2])
        remaining = parts[3]

        if '[' in remaining:
            class_end = remaining.index('[') - 1
            class_name = remaining[:class_end].rstrip(',')
        else:
            class_name, _ = remaining.rsplit(',', 1)

        manual_data.append({
            'transectID': tid,
            'start': start,
            'end': end,
            'CLASS': class_name
        })

manual = pd.DataFrame(manual_data)

# Extract boundary positions
boundaries = []
for tid in manual['transectID'].unique():
    transect_manual = manual[manual['transectID'] == tid].sort_values('start').reset_index(drop=True)

    for i in range(len(transect_manual) - 1):
        row = transect_manual.iloc[i]
        next_row = transect_manual.iloc[i + 1]

        boundaries.append({
            'transect_id': tid,
            'position': row['end'],
            'from_class': row['CLASS'],
            'to_class': next_row['CLASS'],
            'boundary_type': f"{row['CLASS']}_to_{next_row['CLASS']}"
        })

boundaries_df = pd.DataFrame(boundaries)

# Load spectral data
spectral_data = pd.read_csv('sampled_spectral_data.csv')

# Load validation results to identify detected vs missed
validation_matches = pd.read_csv('../validation/outputs/transition_matches.csv')

print(f"  Total manual boundaries: {len(boundaries_df)}")
print(f"  Boundary types: {boundaries_df['boundary_type'].nunique()}")
print(f"  Transects: {len(manual['transectID'].unique())}")

# ============================================================================
# PART 2: Identify Detected vs Missed Boundaries
# ============================================================================

print("\n[2/6] Identifying detected vs missed boundaries...")

# Mark each manual boundary as detected (TP) or missed (FN)
boundaries_df['detected'] = False
boundaries_df['detection_type'] = 'FN'

for _, boundary in boundaries_df.iterrows():
    tid = boundary['transect_id']
    pos = boundary['position']

    # Check if this boundary was detected (TP)
    tp_matches = validation_matches[
        (validation_matches['transect_id'] == tid) &
        (validation_matches['label'] == 'TP')
    ]

    for _, tp in tp_matches.iterrows():
        if abs(tp['detected_position'] - pos) < 10:  # 10m tolerance
            boundaries_df.loc[
                (boundaries_df['transect_id'] == tid) &
                (boundaries_df['position'] == pos),
                ['detected', 'detection_type']
            ] = [True, 'TP']
            break

detected_count = boundaries_df['detected'].sum()
missed_count = (~boundaries_df['detected']).sum()

print(f"  Detected (TP): {detected_count}")
print(f"  Missed (FN): {missed_count}")
print(f"\n  Missed boundaries by type:")
for btype in boundaries_df['boundary_type'].unique():
    total = (boundaries_df['boundary_type'] == btype).sum()
    missed = ((boundaries_df['boundary_type'] == btype) & (~boundaries_df['detected'])).sum()
    pct = 100 * missed / total if total > 0 else 0
    print(f"    {btype:40s}: {missed}/{total} ({pct:.0f}%)")

# ============================================================================
# PART 3: Multi-Scale Smoothing Analysis
# ============================================================================

print("\n[3/6] Multi-scale smoothing analysis...")

def apply_smoothing(data, window_size, method='uniform'):
    """Apply smoothing with specified window and method."""
    if method == 'uniform':
        return uniform_filter1d(data, size=window_size, mode='nearest')
    elif method == 'gaussian':
        sigma = window_size / 3.0  # std dev ~ 1/3 of window
        return gaussian_filter1d(data, sigma=sigma, mode='nearest')
    elif method == 'median':
        return median_filter(data, size=window_size, mode='nearest')
    else:
        return data

def get_profile_at_boundary(spectral_data, transect_id, boundary_pos, window=15):
    """Extract spectral profile ±window around boundary."""
    transect_data = spectral_data[spectral_data['TransectID'] == transect_id].copy()
    transect_data = transect_data.sort_values('distance').reset_index(drop=True)

    mask = (transect_data['distance'] >= boundary_pos - window) & \
           (transect_data['distance'] <= boundary_pos + window)

    return transect_data[mask].copy()

# Test different smoothing window sizes
smoothing_windows = [1, 3, 5, 7, 9, 11]
bands = ['red', 'green', 'blue', 'nir']

# Analyze each boundary type separately
output_dir = Path('boundary_investigation_outputs')
output_dir.mkdir(exist_ok=True)

boundary_analysis_results = []

for boundary_type in boundaries_df['boundary_type'].unique():
    print(f"\n  Analyzing: {boundary_type}")

    type_boundaries = boundaries_df[boundaries_df['boundary_type'] == boundary_type]
    detected_boundaries = type_boundaries[type_boundaries['detected']]
    missed_boundaries = type_boundaries[~type_boundaries['detected']]

    print(f"    Detected: {len(detected_boundaries)}, Missed: {len(missed_boundaries)}")

    # Collect profiles for detected vs missed
    detected_profiles = []
    missed_profiles = []

    for _, boundary in detected_boundaries.iterrows():
        profile = get_profile_at_boundary(
            spectral_data, boundary['transect_id'], boundary['position'], window=15
        )
        if len(profile) > 0:
            profile['dist_from_boundary'] = profile['distance'] - boundary['position']
            profile['detection_status'] = 'detected'
            detected_profiles.append(profile)

    for _, boundary in missed_boundaries.iterrows():
        profile = get_profile_at_boundary(
            spectral_data, boundary['transect_id'], boundary['position'], window=15
        )
        if len(profile) > 0:
            profile['dist_from_boundary'] = profile['distance'] - boundary['position']
            profile['detection_status'] = 'missed'
            missed_profiles.append(profile)

    if not detected_profiles and not missed_profiles:
        continue

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 14))

    # Top section: Detected boundaries at multiple scales
    if detected_profiles:
        detected_combined = pd.concat(detected_profiles, ignore_index=True)

        for scale_idx, window_size in enumerate(smoothing_windows):
            ax = plt.subplot(4, 6, scale_idx + 1)

            for band in bands:
                # Apply smoothing
                grouped = detected_combined.groupby('dist_from_boundary')[band].mean()
                smoothed = apply_smoothing(grouped.values, window_size, method='uniform')

                # Compute derivative
                derivative = np.gradient(smoothed, grouped.index.values)

                # Plot raw signal
                color = {'red': 'red', 'green': 'green', 'blue': 'blue', 'nir': 'darkred'}[band]
                ax.plot(grouped.index, smoothed, color=color, alpha=0.7, linewidth=1.5, label=band if scale_idx == 0 else '')

            ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
            ax.set_title(f'Window={window_size}' + (' (current)' if window_size == 3 else ''), fontsize=9)
            ax.grid(True, alpha=0.3)
            if scale_idx == 0:
                ax.legend(fontsize=7, loc='upper left')
            if scale_idx == 0:
                ax.set_ylabel('DETECTED\n Spectral Value', fontsize=9, fontweight='bold')

    # Second row: Detected - Derivatives at multiple scales
    if detected_profiles:
        for scale_idx, window_size in enumerate(smoothing_windows):
            ax = plt.subplot(4, 6, 6 + scale_idx + 1)

            for band in bands:
                grouped = detected_combined.groupby('dist_from_boundary')[band].mean()
                smoothed = apply_smoothing(grouped.values, window_size, method='uniform')
                distances = grouped.index.values
                derivative = np.gradient(smoothed, distances)

                color = {'red': 'red', 'green': 'green', 'blue': 'blue', 'nir': 'darkred'}[band]
                ax.plot(distances, derivative, color=color, alpha=0.7, linewidth=1.5)

            ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
            ax.axhline(0, color='gray', linestyle=':', linewidth=1)
            ax.grid(True, alpha=0.3)
            if scale_idx == 0:
                ax.set_ylabel('DETECTED\n Derivative', fontsize=9, fontweight='bold')

    # Third row: Missed boundaries at multiple scales
    if missed_profiles:
        missed_combined = pd.concat(missed_profiles, ignore_index=True)

        for scale_idx, window_size in enumerate(smoothing_windows):
            ax = plt.subplot(4, 6, 12 + scale_idx + 1)

            for band in bands:
                grouped = missed_combined.groupby('dist_from_boundary')[band].mean()
                smoothed = apply_smoothing(grouped.values, window_size, method='uniform')

                color = {'red': 'red', 'green': 'green', 'blue': 'blue', 'nir': 'darkred'}[band]
                ax.plot(grouped.index, smoothed, color=color, alpha=0.7, linewidth=1.5)

            ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
            ax.grid(True, alpha=0.3)
            if scale_idx == 0:
                ax.set_ylabel('MISSED\n Spectral Value', fontsize=9, fontweight='bold')

    # Fourth row: Missed - Derivatives at multiple scales
    if missed_profiles:
        for scale_idx, window_size in enumerate(smoothing_windows):
            ax = plt.subplot(4, 6, 18 + scale_idx + 1)

            for band in bands:
                grouped = missed_combined.groupby('dist_from_boundary')[band].mean()
                smoothed = apply_smoothing(grouped.values, window_size, method='uniform')
                distances = grouped.index.values
                derivative = np.gradient(smoothed, distances)

                color = {'red': 'red', 'green': 'green', 'blue': 'blue', 'nir': 'darkred'}[band]
                ax.plot(distances, derivative, color=color, alpha=0.7, linewidth=1.5)

            ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
            ax.axhline(0, color='gray', linestyle=':', linewidth=1)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Distance from boundary (m)', fontsize=8)
            if scale_idx == 0:
                ax.set_ylabel('MISSED\n Derivative', fontsize=9, fontweight='bold')

    plt.suptitle(f'{boundary_type}\nMulti-Scale Analysis: Detected vs Missed',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    safe_filename = boundary_type.replace('→', '_to_').replace(' ', '_')
    output_path = output_dir / f'multiscale_{safe_filename}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"      Saved: {output_path.name}")

    # Calculate derivative statistics for each scale
    scale_stats = []
    for window_size in smoothing_windows:
        if detected_profiles:
            detected_combined = pd.concat(detected_profiles, ignore_index=True)
            detected_nir = detected_combined.groupby('dist_from_boundary')['nir'].mean()
            detected_smoothed = apply_smoothing(detected_nir.values, window_size)
            detected_deriv = np.gradient(detected_smoothed, detected_nir.index.values)
            detected_max_deriv = np.abs(detected_deriv).max()
        else:
            detected_max_deriv = 0

        if missed_profiles:
            missed_combined = pd.concat(missed_profiles, ignore_index=True)
            missed_nir = missed_combined.groupby('dist_from_boundary')['nir'].mean()
            missed_smoothed = apply_smoothing(missed_nir.values, window_size)
            missed_deriv = np.gradient(missed_smoothed, missed_nir.index.values)
            missed_max_deriv = np.abs(missed_deriv).max()
        else:
            missed_max_deriv = 0

        scale_stats.append({
            'boundary_type': boundary_type,
            'window_size': window_size,
            'detected_max_deriv': detected_max_deriv,
            'missed_max_deriv': missed_max_deriv,
            'ratio': detected_max_deriv / missed_max_deriv if missed_max_deriv > 0 else np.inf
        })

    boundary_analysis_results.extend(scale_stats)

# Save scale statistics
scale_stats_df = pd.DataFrame(boundary_analysis_results)
scale_stats_df.to_csv(output_dir / 'multiscale_derivative_statistics.csv', index=False)

print(f"\n  Multi-scale analysis complete!")
print(f"  Results saved to: {output_dir}")

# ============================================================================
# PART 4: Signal-to-Noise Ratio Analysis by Zone
# ============================================================================

print("\n[4/6] Signal-to-noise ratio analysis by zone...")

zone_snr_results = []

for tid in manual['transectID'].unique():
    transect_spectral = spectral_data[spectral_data['TransectID'] == tid].sort_values('distance')
    transect_manual = manual[manual['transectID'] == tid]

    for _, zone in transect_manual.iterrows():
        # Get spectral data for this zone
        zone_data = transect_spectral[
            (transect_spectral['distance'] >= zone['start']) &
            (transect_spectral['distance'] <= zone['end'])
        ]

        if len(zone_data) < 5:
            continue

        for band in bands:
            values = zone_data[band].values

            # Signal = mean
            signal_mean = np.mean(values)

            # Noise = standard deviation
            noise_std = np.std(values)

            # SNR in dB
            snr_db = 20 * np.log10(signal_mean / noise_std) if noise_std > 0 else np.inf

            zone_snr_results.append({
                'transect_id': tid,
                'zone_class': zone['CLASS'],
                'band': band,
                'signal_mean': signal_mean,
                'noise_std': noise_std,
                'snr_db': snr_db
            })

zone_snr_df = pd.DataFrame(zone_snr_results)
zone_snr_df.to_csv(output_dir / 'zone_snr_analysis.csv', index=False)

# Print SNR summary
print("\n  SNR by zone type (dB):")
for zone_class in manual['CLASS'].unique():
    zone_snr = zone_snr_df[zone_snr_df['zone_class'] == zone_class]
    mean_snr = zone_snr.groupby('band')['snr_db'].mean()
    print(f"\n    {zone_class}:")
    for band in bands:
        if band in mean_snr.index:
            print(f"      {band:6s}: {mean_snr[band]:6.2f} dB")

# ============================================================================
# PART 5: Alternative Methods Comparison
# ============================================================================

print("\n[5/6] Testing alternative signal processing methods...")

# For VEG_DUNES boundaries specifically (highest miss rate)
veg_boundaries = boundaries_df[boundaries_df['from_class'] == 'VEGETATED_DUNE']

alternative_methods_results = []

for _, boundary in veg_boundaries.iterrows():
    profile = get_profile_at_boundary(
        spectral_data, boundary['transect_id'], boundary['position'], window=20
    )

    if len(profile) < 10:
        continue

    nir_signal = profile['nir'].values
    distances = profile['distance'].values

    # Method 1: Current (uniform smoothing window=3)
    nir_smooth_w3 = uniform_filter1d(nir_signal, size=3, mode='nearest')
    deriv_w3 = np.gradient(nir_smooth_w3, distances)
    max_deriv_w3 = np.abs(deriv_w3).max()

    # Method 2: Gaussian smoothing
    nir_gaussian = gaussian_filter1d(nir_signal, sigma=2.0, mode='nearest')
    deriv_gaussian = np.gradient(nir_gaussian, distances)
    max_deriv_gaussian = np.abs(deriv_gaussian).max()

    # Method 3: Savitzky-Golay filter (polynomial smoothing)
    if len(nir_signal) >= 5:
        nir_savgol = scipy_signal.savgol_filter(nir_signal, window_length=5, polyorder=2, mode='nearest')
        deriv_savgol = np.gradient(nir_savgol, distances)
        max_deriv_savgol = np.abs(deriv_savgol).max()
    else:
        max_deriv_savgol = 0

    # Method 4: Median filter (noise robust)
    nir_median = median_filter(nir_signal, size=5, mode='nearest')
    deriv_median = np.gradient(nir_median, distances)
    max_deriv_median = np.abs(deriv_median).max()

    alternative_methods_results.append({
        'transect_id': boundary['transect_id'],
        'boundary_position': boundary['position'],
        'detected': boundary['detected'],
        'uniform_w3_max_deriv': max_deriv_w3,
        'gaussian_max_deriv': max_deriv_gaussian,
        'savgol_max_deriv': max_deriv_savgol,
        'median_max_deriv': max_deriv_median
    })

alt_methods_df = pd.DataFrame(alternative_methods_results)
alt_methods_df.to_csv(output_dir / 'alternative_methods_comparison.csv', index=False)

# Compare methods for detected vs missed
if len(alt_methods_df) > 0:
    detected_alt = alt_methods_df[alt_methods_df['detected']]
    missed_alt = alt_methods_df[~alt_methods_df['detected']]

    print("\n  VEG_DUNES boundaries - Max derivative by method:")
    print(f"\n    DETECTED boundaries (n={len(detected_alt)}):")
    if len(detected_alt) > 0:
        print(f"      Uniform w=3:  {detected_alt['uniform_w3_max_deriv'].mean():.2f}")
        print(f"      Gaussian:     {detected_alt['gaussian_max_deriv'].mean():.2f}")
        print(f"      Savitzky-Golay: {detected_alt['savgol_max_deriv'].mean():.2f}")
        print(f"      Median:       {detected_alt['median_max_deriv'].mean():.2f}")

    print(f"\n    MISSED boundaries (n={len(missed_alt)}):")
    if len(missed_alt) > 0:
        print(f"      Uniform w=3:  {missed_alt['uniform_w3_max_deriv'].mean():.2f}")
        print(f"      Gaussian:     {missed_alt['gaussian_max_deriv'].mean():.2f}")
        print(f"      Savitzky-Golay: {missed_alt['savgol_max_deriv'].mean():.2f}")
        print(f"      Median:       {missed_alt['median_max_deriv'].mean():.2f}")

# ============================================================================
# PART 6: Generate Summary Report
# ============================================================================

print("\n[6/6] Generating summary report...")

report_path = output_dir / 'INVESTIGATION_SUMMARY.md'

with open(report_path, 'w') as f:
    f.write("# Comprehensive Boundary Investigation - Summary Report\n\n")
    f.write("## Executive Summary\n\n")

    f.write(f"**Total manual boundaries:** {len(boundaries_df)}\n\n")
    f.write(f"**Detected:** {detected_count} ({100*detected_count/len(boundaries_df):.1f}%)\n\n")
    f.write(f"**Missed:** {missed_count} ({100*missed_count/len(boundaries_df):.1f}%)\n\n")

    f.write("## Missed Boundaries by Type\n\n")
    f.write("| Boundary Type | Missed | Total | Miss Rate |\n")
    f.write("|---------------|--------|-------|----------|\n")
    for btype in boundaries_df['boundary_type'].unique():
        total = (boundaries_df['boundary_type'] == btype).sum()
        missed = ((boundaries_df['boundary_type'] == btype) & (~boundaries_df['detected'])).sum()
        pct = 100 * missed / total if total > 0 else 0
        f.write(f"| {btype} | {missed} | {total} | {pct:.0f}% |\n")

    f.write("\n## Key Findings\n\n")
    f.write("### 1. Multi-Scale Smoothing Analysis\n\n")
    f.write("- Generated multi-scale spectral profile plots for all boundary types\n")
    f.write("- Compared detected vs missed boundaries at windows: 1, 3, 5, 7, 9, 11\n")
    f.write("- **See plots in `boundary_investigation_outputs/multiscale_*.png`**\n\n")

    f.write("### 2. Signal-to-Noise Ratio by Zone\n\n")
    for zone_class in manual['CLASS'].unique():
        zone_snr = zone_snr_df[zone_snr_df['zone_class'] == zone_class]
        if len(zone_snr) > 0:
            mean_snr_nir = zone_snr[zone_snr['band'] == 'nir']['snr_db'].mean()
            f.write(f"- **{zone_class}:** NIR SNR = {mean_snr_nir:.1f} dB\n")

    f.write("\n### 3. Alternative Smoothing Methods (VEG_DUNES boundaries)\n\n")
    if len(alt_methods_df) > 0:
        detected_alt = alt_methods_df[alt_methods_df['detected']]
        missed_alt = alt_methods_df[~alt_methods_df['detected']]

        if len(detected_alt) > 0 and len(missed_alt) > 0:
            f.write("**Detected boundaries - Max derivative magnitude:**\n")
            f.write(f"- Uniform (w=3): {detected_alt['uniform_w3_max_deriv'].mean():.2f}\n")
            f.write(f"- Gaussian: {detected_alt['gaussian_max_deriv'].mean():.2f}\n")
            f.write(f"- Savitzky-Golay: {detected_alt['savgol_max_deriv'].mean():.2f}\n")
            f.write(f"- Median filter: {detected_alt['median_max_deriv'].mean():.2f}\n\n")

            f.write("**Missed boundaries - Max derivative magnitude:**\n")
            f.write(f"- Uniform (w=3): {missed_alt['uniform_w3_max_deriv'].mean():.2f}\n")
            f.write(f"- Gaussian: {missed_alt['gaussian_max_deriv'].mean():.2f}\n")
            f.write(f"- Savitzky-Golay: {missed_alt['savgol_max_deriv'].mean():.2f}\n")
            f.write(f"- Median filter: {missed_alt['median_max_deriv'].mean():.2f}\n\n")

    f.write("\n## Next Steps\n\n")
    f.write("**CRITICAL: Review the multi-scale plots to answer:**\n\n")
    f.write("1. Can you visually identify boundaries after smoothing?\n")
    f.write("2. Are missed boundaries gradual transitions or sharp but noisy?\n")
    f.write("3. What smoothing window makes boundaries most visible?\n")
    f.write("4. Do VEG_DUNES boundaries look fundamentally different?\n\n")

    f.write("**Based on these answers, we can determine:**\n")
    f.write("- Whether to adjust smoothing parameters\n")
    f.write("- Whether to use zone-adaptive detection rules\n")
    f.write("- Whether preprocessing (denoising) is needed\n")
    f.write("- Whether multi-scale detection would help\n")

print(f"\n{'=' * 80}")
print("INVESTIGATION COMPLETE!")
print(f"{'=' * 80}")
print(f"\nResults saved to: {output_dir.absolute()}")
print(f"\nGenerated files:")
print(f"  - INVESTIGATION_SUMMARY.md (read this first!)")
print(f"  - multiscale_*.png (REVIEW THESE - Can you see boundaries?)")
print(f"  - multiscale_derivative_statistics.csv")
print(f"  - zone_snr_analysis.csv")
print(f"  - alternative_methods_comparison.csv")
print(f"\n{'=' * 80}")
print("NEXT: Review the plots and answer the questions in the summary!")
print(f"{'=' * 80}\n")
