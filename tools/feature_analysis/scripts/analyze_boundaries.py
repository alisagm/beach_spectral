"""
Analyze spectral characteristics at manually-identified boundaries.

This script examines what happens spectrally at the transition zones to improve
automated boundary detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Load manual classifications
# Read with special handling for arrays in WAVE_CREST_POSITIONS column
import csv

manual_data = []
with open('classified_manual.csv', 'r') as f:
    # Read first line for headers
    headers = f.readline().strip().split(',')

    for line in f:
        # Split carefully to handle arrays
        # Format: transectID,start[m],end[m],CLASS,[array or NA]
        parts = line.strip().split(',', 3)  # Split first 3 commas
        tid = int(parts[0])
        start = float(parts[1])
        end = float(parts[2])

        # Remaining part has CLASS and WAVE_CREST_POSITIONS
        remaining = parts[3]
        # Find last comma before bracket or NA
        if '[' in remaining:
            class_end = remaining.index('[') - 1
            class_name = remaining[:class_end].rstrip(',')
            wave_crests = remaining[class_end+1:]
        else:
            # No bracket, simple split
            class_name, wave_crests = remaining.rsplit(',', 1)

        manual_data.append({
            'transectID': tid,
            'start[m]': start,
            'end[m]': end,
            'CLASS': class_name,
            'WAVE_CREST_POSITIONS': wave_crests
        })

manual = pd.DataFrame(manual_data)

# Load spectral data
spectral_data = pd.read_csv('sampled_spectral_data.csv')

# Extract boundary positions for each transect
boundaries = []

for tid in manual['transectID'].unique():
    transect_manual = manual[manual['transectID'] == tid].sort_values('start[m]').reset_index(drop=True)

    # Extract boundary positions (end of each zone except last)
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

print("=" * 80)
print("MANUAL BOUNDARY POSITIONS")
print("=" * 80)
print(f"\nTotal boundaries identified: {len(boundaries_df)}")
print("\nBoundary types:")
print(boundaries_df['boundary_type'].value_counts())

# Analyze spectral profiles at boundaries
def get_profile_at_boundary(spectral_data, transect_id, boundary_pos, window=10):
    """Extract spectral profile +-window around boundary."""
    transect_data = spectral_data[spectral_data['TransectID'] == transect_id].copy()
    transect_data = transect_data.sort_values('distance')

    # Find points within window
    mask = (transect_data['distance'] >= boundary_pos - window) & \
           (transect_data['distance'] <= boundary_pos + window)

    return transect_data[mask]

def calculate_derivatives(data, bands=['red', 'green', 'blue', 'nir']):
    """Calculate first and second derivatives of spectral bands."""
    derivatives = {}

    for band in bands:
        # First derivative (rate of change)
        derivatives[f'{band}_d1'] = np.gradient(data[band].values, data['distance'].values)
        # Second derivative (acceleration of change)
        derivatives[f'{band}_d2'] = np.gradient(derivatives[f'{band}_d1'], data['distance'].values)

    return pd.DataFrame(derivatives, index=data.index)

# Analyze each boundary type
print("\n" + "=" * 80)
print("SPECTRAL ANALYSIS AT BOUNDARIES (+-10m window)")
print("=" * 80)

boundary_stats = []

for boundary_type in boundaries_df['boundary_type'].unique():
    print(f"\n\n### {boundary_type} ###")

    type_boundaries = boundaries_df[boundaries_df['boundary_type'] == boundary_type]

    all_profiles = []
    all_derivatives = []

    for _, boundary in type_boundaries.iterrows():
        profile = get_profile_at_boundary(
            spectral_data,
            boundary['transect_id'],
            boundary['position'],
            window=10
        )

        if len(profile) > 0:
            derivs = calculate_derivatives(profile)
            profile = pd.concat([profile, derivs], axis=1)
            profile['dist_from_boundary'] = profile['distance'] - boundary['position']
            all_profiles.append(profile)

    if all_profiles:
        combined = pd.concat(all_profiles, ignore_index=True)

        # Split into before/after boundary
        before = combined[combined['dist_from_boundary'] < 0]
        after = combined[combined['dist_from_boundary'] >= 0]

        print(f"\nSample size: {len(type_boundaries)} boundaries, {len(combined)} points")

        print("\n--- BEFORE boundary (landward side) ---")
        bands = ['red', 'green', 'blue', 'nir']
        for band in bands:
            mean_val = before[band].mean()
            std_val = before[band].std()
            print(f"  {band:6s}: {mean_val:6.1f} +- {std_val:5.1f}")

        print("\n--- AFTER boundary (seaward side) ---")
        for band in bands:
            mean_val = after[band].mean()
            std_val = after[band].std()
            print(f"  {band:6s}: {mean_val:6.1f} +- {std_val:5.1f}")

        print("\n--- CHANGE across boundary ---")
        for band in bands:
            delta = after[band].mean() - before[band].mean()
            pct_change = (delta / before[band].mean()) * 100 if before[band].mean() != 0 else 0
            print(f"  {band:6s}: {delta:+7.1f} ({pct_change:+6.1f}%)")

        print("\n--- FIRST DERIVATIVE (rate of change) ---")
        for band in bands:
            d1_key = f'{band}_d1'
            mean_d1 = combined[d1_key].mean()
            max_d1 = combined[d1_key].abs().max()
            print(f"  {band:6s}: mean={mean_d1:+7.2f}, max_abs={max_d1:7.2f}")

        print("\n--- SECOND DERIVATIVE (acceleration) ---")
        for band in bands:
            d2_key = f'{band}_d2'
            mean_d2 = combined[d2_key].mean()
            max_d2 = combined[d2_key].abs().max()
            print(f"  {band:6s}: mean={mean_d2:+7.2f}, max_abs={max_d2:7.2f}")

        # Store for plotting
        boundary_stats.append({
            'type': boundary_type,
            'combined': combined,
            'before': before,
            'after': after
        })

# Create visualization with derivatives
print("\n" + "=" * 80)
print("GENERATING BOUNDARY PROFILE PLOTS...")
print("=" * 80)

for stat in boundary_stats:
    boundary_type = stat['type']
    combined = stat['combined']

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Raw spectral values
    ax = axes[0]
    for band, color in [('red', 'red'), ('green', 'green'), ('blue', 'blue'), ('nir', 'darkred')]:
        # Group by distance and average
        grouped = combined.groupby('dist_from_boundary')[band].agg(['mean', 'std'])
        ax.plot(grouped.index, grouped['mean'], color=color, label=band, linewidth=2)
        ax.fill_between(
            grouped.index,
            grouped['mean'] - grouped['std'],
            grouped['mean'] + grouped['std'],
            color=color,
            alpha=0.2
        )

    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Boundary')
    ax.set_xlabel('Distance from Boundary (m)', fontsize=11)
    ax.set_ylabel('Spectral Value', fontsize=11)
    ax.set_title(f'{boundary_type} - Raw Spectral Profile', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 2: First derivative
    ax = axes[1]
    for band, color in [('red', 'red'), ('green', 'green'), ('blue', 'blue'), ('nir', 'darkred')]:
        d1_key = f'{band}_d1'
        grouped = combined.groupby('dist_from_boundary')[d1_key].agg(['mean', 'std'])
        ax.plot(grouped.index, grouped['mean'], color=color, label=f'{band} d/dx', linewidth=2)
        ax.fill_between(
            grouped.index,
            grouped['mean'] - grouped['std'],
            grouped['mean'] + grouped['std'],
            color=color,
            alpha=0.2
        )

    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Boundary')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Distance from Boundary (m)', fontsize=11)
    ax.set_ylabel('First Derivative (rate of change)', fontsize=11)
    ax.set_title(f'{boundary_type} - First Derivative', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 3: Second derivative
    ax = axes[2]
    for band, color in [('red', 'red'), ('green', 'green'), ('blue', 'blue'), ('nir', 'darkred')]:
        d2_key = f'{band}_d2'
        grouped = combined.groupby('dist_from_boundary')[d2_key].agg(['mean', 'std'])
        ax.plot(grouped.index, grouped['mean'], color=color, label=f'{band} d²/dx²', linewidth=2)
        ax.fill_between(
            grouped.index,
            grouped['mean'] - grouped['std'],
            grouped['mean'] + grouped['std'],
            color=color,
            alpha=0.2
        )

    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Boundary')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Distance from Boundary (m)', fontsize=11)
    ax.set_ylabel('Second Derivative (acceleration)', fontsize=11)
    ax.set_title(f'{boundary_type} - Second Derivative', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    safe_filename = boundary_type.replace('->', '_to_').replace(' ', '_')
    output_path = f'boundary_analysis_{safe_filename}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
