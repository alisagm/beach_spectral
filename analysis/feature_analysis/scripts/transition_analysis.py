"""
Transition Zone Analysis

Analyzes spectral behavior in +-10m regions around landcover boundaries.
Focus on shell line (BEACH_DRY -> BEACH_WET) characteristics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Configuration
TRANSITION_WINDOW = 10.0  # meters - window around boundary to analyze
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs' / 'statistics'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load manual classification data
DATA_DIR = Path(__file__).parent.parent / 'data'
manual_321197 = pd.read_csv(DATA_DIR / 'classified_manual_321197.csv',
                             usecols=['transectID', 'start[m]', 'end[m]', 'CLASS'])
manual_612823 = pd.read_csv(DATA_DIR / 'classified_manual_612823.csv',
                             usecols=['transectID', 'start[m]', 'end[m]', 'CLASS'])

# Combine
manual_321197['seed'] = 321197
manual_612823['seed'] = 612823
manual = pd.concat([manual_321197, manual_612823], ignore_index=True)

print('=' * 80)
print('TRANSITION ZONE ANALYSIS')
print('=' * 80)
print(f'Analyzing +/-{TRANSITION_WINDOW}m regions around landcover boundaries')
print('Focus: BEACH_DRY -> BEACH_WET (shell line)')
print('=' * 80)
print()

def load_spectral_data(transect_id, seed):
    """Load spectral data for a given transect."""
    base_path = Path(__file__).parent.parent.parent / 'training_output'

    possible_paths = [
        base_path / f'seed_{seed}' / f'run01' / 'sampled_spectral_data.csv',
        base_path / f'seed_{seed}' / 'training_v01' / 'sampled_spectral_data.csv',
        base_path / f'seed_{seed}' / 'training_v02' / 'sampled_spectral_data.csv',
        base_path / f'seed_{seed}' / 'training_v04' / 'sampled_spectral_data.csv',
        base_path / f'seed_{seed}' / 'run02' / 'sampled_spectral_data.csv',
        base_path / f'seed_{seed}' / 'run05' / 'sampled_spectral_data.csv',
    ]

    for path in possible_paths:
        if path.exists():
            df = pd.read_csv(path)
            if transect_id in df['TransectID'].values:
                return df[df['TransectID'] == transect_id].copy()

    return None

def compute_derivatives(series, distance, window=5):
    """Compute smoothed 1st and 2nd derivatives."""
    if len(series) < window:
        return np.full(len(series), np.nan), np.full(len(series), np.nan)

    # Smooth with Savitzky-Golay
    window_len = min(window, len(series)//2*2+1)
    if window_len < 3:
        window_len = 3
    smoothed = savgol_filter(series, window_length=window_len, polyorder=2)

    # Compute derivatives
    d1 = np.gradient(smoothed, distance)
    d2 = np.gradient(d1, distance)

    return d1, d2

def analyze_transition(transect_data, boundary_location, before_class, after_class):
    """
    Analyze transition characteristics around a boundary.

    Args:
        transect_data: Full transect spectral data
        boundary_location: Distance along transect where boundary occurs
        before_class: Landcover class before boundary
        after_class: Landcover class after boundary

    Returns:
        Dictionary of transition characteristics
    """
    # Extract window around boundary
    window_start = boundary_location - TRANSITION_WINDOW
    window_end = boundary_location + TRANSITION_WINDOW

    window_data = transect_data[
        (transect_data['distance'] >= window_start) &
        (transect_data['distance'] <= window_end)
    ].copy()

    if len(window_data) < 5:
        return None

    # Sort by distance
    window_data = window_data.sort_values('distance')

    # Extract band values
    distance = window_data['distance'].values
    r = window_data['red'].values
    g = window_data['green'].values
    b = window_data['blue'].values
    nir = window_data['nir'].values

    # Computed features
    brightness = (r + g + b + nir) / 4.0
    ndvi = (nir - r) / (nir + r + 1e-10)
    ndwi = (g - nir) / (g + nir + 1e-10)
    nir_ratio = nir / (brightness + 1e-10)
    rg_ratio = r / (g + 1e-10)

    # Compute derivatives
    nir_d1, nir_d2 = compute_derivatives(nir, distance)
    bright_d1, bright_d2 = compute_derivatives(brightness, distance)
    rg_d1, _ = compute_derivatives(rg_ratio, distance)

    # Split into before/after regions
    before_mask = distance < boundary_location
    after_mask = distance >= boundary_location

    if not np.any(before_mask) or not np.any(after_mask):
        return None

    stats = {
        'boundary_location': boundary_location,
        'before_class': before_class,
        'after_class': after_class,
        'transition_type': f'{before_class}->{after_class}',
    }

    # === Pre/Post Value Differences ===
    stats['nir_before_mean'] = np.mean(nir[before_mask])
    stats['nir_after_mean'] = np.mean(nir[after_mask])
    stats['nir_drop_absolute'] = stats['nir_before_mean'] - stats['nir_after_mean']

    stats['brightness_before_mean'] = np.mean(brightness[before_mask])
    stats['brightness_after_mean'] = np.mean(brightness[after_mask])
    stats['brightness_drop_absolute'] = stats['brightness_before_mean'] - stats['brightness_after_mean']

    stats['nir_ratio_before_mean'] = np.mean(nir_ratio[before_mask])
    stats['nir_ratio_after_mean'] = np.mean(nir_ratio[after_mask])
    stats['nir_ratio_change'] = stats['nir_ratio_before_mean'] - stats['nir_ratio_after_mean']

    stats['rg_ratio_before_mean'] = np.mean(rg_ratio[before_mask])
    stats['rg_ratio_after_mean'] = np.mean(rg_ratio[after_mask])
    stats['rg_ratio_change'] = stats['rg_ratio_before_mean'] - stats['rg_ratio_after_mean']

    # === Transition Sharpness (derivative peaks) ===
    stats['nir_d1_peak'] = np.min(nir_d1)  # Most negative (steepest drop)
    stats['nir_d1_peak_location'] = distance[np.argmin(nir_d1)]

    stats['brightness_d1_peak'] = np.min(bright_d1)
    stats['brightness_d1_peak_location'] = distance[np.argmin(bright_d1)]

    # === Transition Width ===
    # Distance over which 80% of change occurs
    nir_change_total = abs(stats['nir_drop_absolute'])
    if nir_change_total > 5:  # Only if significant change
        nir_start_val = stats['nir_before_mean']
        threshold_20pct = nir_start_val - 0.2 * nir_change_total
        threshold_80pct = nir_start_val - 0.8 * nir_change_total

        # Find where these thresholds are crossed
        idx_20 = np.where(nir < threshold_20pct)[0]
        idx_80 = np.where(nir < threshold_80pct)[0]

        if len(idx_20) > 0 and len(idx_80) > 0:
            dist_20 = distance[idx_20[0]]
            dist_80 = distance[idx_80[0]]
            stats['transition_width_nir'] = dist_80 - dist_20
        else:
            stats['transition_width_nir'] = np.nan
    else:
        stats['transition_width_nir'] = np.nan

    # === Transition Asymmetry ===
    # Compare rate of approach from each side
    # Use derivative magnitude before vs after peak
    peak_idx = np.argmin(nir_d1)
    if peak_idx > 2 and peak_idx < len(nir_d1) - 2:
        approach_rate_before = np.mean(np.abs(nir_d1[peak_idx-3:peak_idx]))
        approach_rate_after = np.mean(np.abs(nir_d1[peak_idx+1:peak_idx+4]))
        if approach_rate_after > 0:
            stats['transition_asymmetry'] = approach_rate_before / approach_rate_after
        else:
            stats['transition_asymmetry'] = np.nan
    else:
        stats['transition_asymmetry'] = np.nan

    # === Multi-Band Synchrony ===
    # How well do all bands agree on transition location?
    r_d1, _ = compute_derivatives(r, distance)
    g_d1, _ = compute_derivatives(g, distance)
    b_d1, _ = compute_derivatives(b, distance)

    r_peak_loc = distance[np.argmin(r_d1)]
    g_peak_loc = distance[np.argmin(g_d1)]
    b_peak_loc = distance[np.argmin(b_d1)]
    nir_peak_loc = distance[np.argmin(nir_d1)]

    # Standard deviation of peak locations
    peak_locs = [r_peak_loc, g_peak_loc, b_peak_loc, nir_peak_loc]
    stats['band_synchrony_std'] = np.std(peak_locs)

    # === Variability (texture change) ===
    stats['brightness_std_before'] = np.std(brightness[before_mask])
    stats['brightness_std_after'] = np.std(brightness[after_mask])
    if stats['brightness_std_before'] > 0:
        stats['variability_ratio'] = stats['brightness_std_after'] / stats['brightness_std_before']
    else:
        stats['variability_ratio'] = np.nan

    # === Second Derivative (inflection analysis) ===
    stats['nir_d2_max'] = np.max(np.abs(nir_d2))
    stats['nir_d2_at_boundary'] = nir_d2[np.argmin(np.abs(distance - boundary_location))]

    # === Sustained Drop Analysis ===
    # Count consecutive points with strong negative derivative
    threshold = -3.0  # units/m
    consecutive = 0
    max_consecutive = 0
    for val in nir_d1:
        if val < threshold:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    stats['nir_sustained_drop_points'] = max_consecutive

    # === R/G Ratio Pattern (for shell line validation) ===
    stats['rg_ratio_at_boundary'] = rg_ratio[np.argmin(np.abs(distance - boundary_location))]
    stats['rg_ratio_d1_peak'] = np.min(rg_d1)  # Most negative

    return stats

# === Extract Boundaries from Manual Classifications ===
print('Extracting boundaries from manual classifications...')

boundaries = []

# Group by transect
for tid, group in manual.groupby('transectID'):
    # Sort by start position
    group = group.sort_values('start[m]')

    # Find boundaries (end of one section = start of next)
    for i in range(len(group) - 1):
        current_row = group.iloc[i]
        next_row = group.iloc[i + 1]

        boundary_loc = current_row['end[m]']
        before_class = current_row['CLASS']
        after_class = next_row['CLASS']
        seed = current_row['seed']

        boundaries.append({
            'transect_id': tid,
            'boundary_location': boundary_loc,
            'before_class': before_class,
            'after_class': after_class,
            'transition_type': f'{before_class}->{after_class}',
            'seed': seed
        })

boundaries_df = pd.DataFrame(boundaries)
print(f'  Found {len(boundaries_df)} boundaries')
print(f'  Transition types: {boundaries_df["transition_type"].value_counts().to_dict()}')
print()

# === Analyze Each Boundary ===
print('Analyzing transition characteristics...')

transition_results = []

for idx, boundary in boundaries_df.iterrows():
    tid = boundary['transect_id']
    loc = boundary['boundary_location']
    before_class = boundary['before_class']
    after_class = boundary['after_class']
    seed = boundary['seed']

    # Load spectral data
    spectral_data = load_spectral_data(tid, seed)
    if spectral_data is None:
        print(f'  Warning: No spectral data for transect {tid}')
        continue

    # Analyze transition
    trans_stats = analyze_transition(spectral_data, loc, before_class, after_class)
    if trans_stats is not None:
        trans_stats['transect_id'] = tid
        trans_stats['seed'] = seed
        transition_results.append(trans_stats)

print(f'  Analyzed {len(transition_results)} transitions')
print()

# === Save Detailed Results ===
if len(transition_results) > 0:
    results_df = pd.DataFrame(transition_results)
    output_file = OUTPUT_DIR / 'transition_characteristics.csv'
    results_df.to_csv(output_file, index=False)
    print(f'Saved: {output_file}')
    print()

    # === Aggregate Statistics by Transition Type ===
    print('=' * 80)
    print('TRANSITION SIGNATURES BY TYPE')
    print('=' * 80)

    for trans_type in results_df['transition_type'].unique():
        trans_data = results_df[results_df['transition_type'] == trans_type]

        print(f'\n{trans_type} (n={len(trans_data)})')
        print('-' * 80)

        key_metrics = [
            ('nir_drop_absolute', 'NIR Drop (abs)'),
            ('brightness_drop_absolute', 'Brightness Drop (abs)'),
            ('nir_d1_peak', 'NIR Derivative Peak'),
            ('transition_width_nir', 'Transition Width (m)'),
            ('nir_ratio_change', 'NIR Ratio Change'),
            ('rg_ratio_at_boundary', 'R/G Ratio at Boundary'),
            ('band_synchrony_std', 'Band Synchrony (std)'),
            ('variability_ratio', 'Variability Ratio'),
            ('nir_sustained_drop_points', 'Sustained Drop (pts)')
        ]

        for metric_name, metric_label in key_metrics:
            if metric_name in trans_data.columns:
                values = trans_data[metric_name].dropna()
                if len(values) > 0:
                    print(f'  {metric_label:30s}: {values.mean():7.2f} +- {values.std():6.2f}  '
                          f'range [{values.min():7.2f}, {values.max():7.2f}]')

    # === Focus on Shell Line ===
    print('\n' + '=' * 80)
    print('SHELL LINE (BEACH_DRY -> BEACH_WET) DETAILED SIGNATURE')
    print('=' * 80)

    shell_line = results_df[results_df['transition_type'] == 'BEACH_DRY->BEACH_WET']

    if len(shell_line) > 0:
        print(f'\nSample size: {len(shell_line)} shell line boundaries')
        print()

        print('NIR Characteristics:')
        print(f'  Absolute drop:         {shell_line["nir_drop_absolute"].mean():6.1f} +- {shell_line["nir_drop_absolute"].std():5.1f} units')
        print(f'  Before value:          {shell_line["nir_before_mean"].mean():6.1f} +- {shell_line["nir_before_mean"].std():5.1f} units')
        print(f'  After value:           {shell_line["nir_after_mean"].mean():6.1f} +- {shell_line["nir_after_mean"].std():5.1f} units')
        print(f'  Derivative peak:       {shell_line["nir_d1_peak"].mean():6.2f} +- {shell_line["nir_d1_peak"].std():5.2f} units/m')
        print(f'  Sustained drop:        {shell_line["nir_sustained_drop_points"].mean():6.1f} +- {shell_line["nir_sustained_drop_points"].std():5.1f} points')
        print()

        print('Brightness Characteristics:')
        print(f'  Absolute drop:         {shell_line["brightness_drop_absolute"].mean():6.1f} +- {shell_line["brightness_drop_absolute"].std():5.1f} units')
        print(f'  Before value:          {shell_line["brightness_before_mean"].mean():6.1f} +- {shell_line["brightness_before_mean"].std():5.1f} units')
        print(f'  After value:           {shell_line["brightness_after_mean"].mean():6.1f} +- {shell_line["brightness_after_mean"].std():5.1f} units')
        print()

        print('Transition Geometry:')
        print(f'  Width (80% change):    {shell_line["transition_width_nir"].mean():6.1f} +- {shell_line["transition_width_nir"].std():5.1f} m')
        print(f'  Asymmetry ratio:       {shell_line["transition_asymmetry"].mean():6.2f} +- {shell_line["transition_asymmetry"].std():5.2f}')
        print(f'  Band synchrony (std):  {shell_line["band_synchrony_std"].mean():6.2f} +- {shell_line["band_synchrony_std"].std():5.2f} m')
        print()

        print('Spectral Indices:')
        print(f'  NIR ratio change:      {shell_line["nir_ratio_change"].mean():6.3f} +- {shell_line["nir_ratio_change"].std():5.3f}')
        print(f'  R/G ratio at boundary: {shell_line["rg_ratio_at_boundary"].mean():6.3f} +- {shell_line["rg_ratio_at_boundary"].std():5.3f}')
        print(f'  Variability ratio:     {shell_line["variability_ratio"].mean():6.2f} +- {shell_line["variability_ratio"].std():5.2f}')

print('\n' + '=' * 80)
print('Analysis complete!')
print('=' * 80)
