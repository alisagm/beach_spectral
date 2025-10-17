"""
Comprehensive Spectral Profiling Analysis

Computes exhaustive spectral features for manually classified landcover sections.
Implements point-by-point and rolling-window analyses as described in IMPROVEMENT_PLAN.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Configuration
BOUNDARY_TOLERANCE = 5.0  # meters - exclude edges of each zone
ROLLING_WINDOWS = [3, 5, 7, 9, 11]  # meters - multiple scales
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs' / 'statistics'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load manual classification data from both seeds
DATA_DIR = Path(__file__).parent.parent / 'data'
manual_321197 = pd.read_csv(DATA_DIR / 'classified_manual_321197.csv',
                             usecols=['transectID', 'start[m]', 'end[m]', 'CLASS'])
manual_612823 = pd.read_csv(DATA_DIR / 'classified_manual_612823.csv',
                             usecols=['transectID', 'start[m]', 'end[m]', 'CLASS'])

# Combine both datasets
manual = pd.concat([manual_321197, manual_612823], ignore_index=True)

print('=' * 80)
print('COMPREHENSIVE SPECTRAL PROFILING ANALYSIS')
print('=' * 80)
print(f'Dataset: {len(manual)} landcover sections from {manual["transectID"].nunique()} transects')
print(f'Classes: {manual["CLASS"].unique()}')
print(f'Boundary tolerance: {BOUNDARY_TOLERANCE}m')
print(f'Rolling window sizes: {ROLLING_WINDOWS}')
print('=' * 80)
print()

def load_spectral_data(transect_id, seed):
    """Load spectral data for a given transect from appropriate seed folder."""
    # Try to find the spectral data file
    base_path = Path(__file__).parent.parent.parent / 'training_output'

    # Try different possible locations
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

def compute_basic_features(r, g, b, nir):
    """Compute basic spectral features from raw bands."""
    epsilon = 1e-10

    features = {}

    # Raw band values
    features['red'] = r
    features['green'] = g
    features['blue'] = b
    features['nir'] = nir

    # Brightness
    features['brightness'] = (r + g + b + nir) / 4.0

    # Spectral indices
    features['ndvi'] = (nir - r) / (nir + r + epsilon)
    features['ndwi'] = (g - nir) / (g + nir + epsilon)
    features['nir_ratio'] = nir / (features['brightness'] + epsilon)

    # Band ratios (all combinations)
    features['red_green_ratio'] = r / (g + epsilon)
    features['red_blue_ratio'] = r / (b + epsilon)
    features['red_nir_ratio'] = r / (nir + epsilon)
    features['green_blue_ratio'] = g / (b + epsilon)
    features['green_nir_ratio'] = g / (nir + epsilon)
    features['blue_red_ratio'] = b / (r + epsilon)
    features['blue_nir_ratio'] = b / (nir + epsilon)

    # Normalized differences (beyond NDVI/NDWI)
    features['nd_red_green'] = (r - g) / (r + g + epsilon)
    features['nd_red_blue'] = (r - b) / (r + b + epsilon)
    features['nd_green_blue'] = (g - b) / (g + b + epsilon)

    return features

def compute_derivatives(series, distance, window=3):
    """Compute 1st and 2nd derivatives with smoothing."""
    # Apply Savitzky-Golay filter for smoothing before differentiation
    if len(series) < window:
        return pd.Series([np.nan]*len(series)), pd.Series([np.nan]*len(series))

    # Smooth the series
    smoothed = savgol_filter(series, window_length=min(window, len(series)//2*2+1), polyorder=2)

    # Compute first derivative (gradient)
    d1 = np.gradient(smoothed, distance)

    # Compute second derivative (curvature)
    d2 = np.gradient(d1, distance)

    return pd.Series(d1, index=series.index), pd.Series(d2, index=series.index)

def compute_rolling_features(series, distance, window_size):
    """Compute rolling window statistics."""
    # Convert window size in meters to number of points
    # Assuming 1m sampling interval
    window_points = int(window_size)

    if len(series) < window_points:
        return {}

    features = {}
    features[f'rolling_mean_{window_size}m'] = series.rolling(window_points, center=True).mean()
    features[f'rolling_std_{window_size}m'] = series.rolling(window_points, center=True).std()
    features[f'rolling_min_{window_size}m'] = series.rolling(window_points, center=True).min()
    features[f'rolling_max_{window_size}m'] = series.rolling(window_points, center=True).max()
    features[f'rolling_range_{window_size}m'] = features[f'rolling_max_{window_size}m'] - features[f'rolling_min_{window_size}m']

    return features

def analyze_zone_section(zone_data):
    """Perform comprehensive analysis on a zone section."""
    if len(zone_data) == 0:
        return None

    r = zone_data['red'].values
    g = zone_data['green'].values
    b = zone_data['blue'].values
    nir = zone_data['nir'].values
    distance = zone_data['distance'].values

    stats = {}

    # === Basic Statistics ===
    for band_name, band_data in [('red', r), ('green', g), ('blue', b), ('nir', nir)]:
        stats[f'{band_name}_mean'] = np.mean(band_data)
        stats[f'{band_name}_std'] = np.std(band_data)
        stats[f'{band_name}_min'] = np.min(band_data)
        stats[f'{band_name}_max'] = np.max(band_data)
        stats[f'{band_name}_median'] = np.median(band_data)
        stats[f'{band_name}_q25'] = np.percentile(band_data, 25)
        stats[f'{band_name}_q75'] = np.percentile(band_data, 75)
        stats[f'{band_name}_iqr'] = stats[f'{band_name}_q75'] - stats[f'{band_name}_q25']
        stats[f'{band_name}_cv'] = stats[f'{band_name}_std'] / (stats[f'{band_name}_mean'] + 1e-10)
        stats[f'{band_name}_range'] = stats[f'{band_name}_max'] - stats[f'{band_name}_min']

        # Distribution shape
        if len(band_data) > 3:
            stats[f'{band_name}_skew'] = skew(band_data)
            stats[f'{band_name}_kurtosis'] = kurtosis(band_data)

    # === Derived Features ===
    brightness = (r + g + b + nir) / 4.0
    stats['brightness_mean'] = np.mean(brightness)
    stats['brightness_std'] = np.std(brightness)
    stats['brightness_cv'] = stats['brightness_std'] / (stats['brightness_mean'] + 1e-10)

    # Compute NDVI for each point
    ndvi = (nir - r) / (nir + r + 1e-10)
    stats['ndvi_mean'] = np.mean(ndvi)
    stats['ndvi_std'] = np.std(ndvi)
    stats['ndvi_min'] = np.min(ndvi)
    stats['ndvi_max'] = np.max(ndvi)

    # Compute NDWI
    ndwi = (g - nir) / (g + nir + 1e-10)
    stats['ndwi_mean'] = np.mean(ndwi)
    stats['ndwi_std'] = np.std(ndwi)

    # NIR ratio
    nir_ratio = nir / (brightness + 1e-10)
    stats['nir_ratio_mean'] = np.mean(nir_ratio)
    stats['nir_ratio_std'] = np.std(nir_ratio)

    # Band ratios
    stats['rg_ratio_mean'] = np.mean(r / (g + 1e-10))
    stats['rg_ratio_std'] = np.std(r / (g + 1e-10))
    stats['blue_red_ratio_mean'] = np.mean(b / (r + 1e-10))
    stats['blue_red_ratio_std'] = np.std(b / (r + 1e-10))

    # === Spatial Characteristics ===
    stats['n_points'] = len(zone_data)
    stats['length_m'] = distance[-1] - distance[0] if len(distance) > 1 else 0

    # Variability (local texture)
    if len(brightness) > 5:
        stats['variability_mean'] = np.std(brightness)
    else:
        stats['variability_mean'] = np.nan

    # === Derivative Features ===
    if len(zone_data) >= 5:
        # NIR derivatives
        nir_series = pd.Series(nir)
        dist_series = pd.Series(distance)
        nir_d1, nir_d2 = compute_derivatives(nir_series, dist_series, window=5)

        stats['nir_d1_mean'] = nir_d1.mean()
        stats['nir_d1_std'] = nir_d1.std()
        stats['nir_d1_min'] = nir_d1.min()
        stats['nir_d1_max'] = nir_d1.max()

        stats['nir_d2_mean'] = nir_d2.mean()
        stats['nir_d2_std'] = nir_d2.std()

        # Brightness derivatives
        brightness_series = pd.Series(brightness)
        bright_d1, bright_d2 = compute_derivatives(brightness_series, dist_series, window=5)

        stats['brightness_d1_mean'] = bright_d1.mean()
        stats['brightness_d1_std'] = bright_d1.std()
    else:
        for key in ['nir_d1_mean', 'nir_d1_std', 'nir_d1_min', 'nir_d1_max',
                    'nir_d2_mean', 'nir_d2_std', 'brightness_d1_mean', 'brightness_d1_std']:
            stats[key] = np.nan

    # === Cross-Band Correlations ===
    if len(zone_data) > 3:
        stats['corr_red_green'] = np.corrcoef(r, g)[0, 1]
        stats['corr_red_nir'] = np.corrcoef(r, nir)[0, 1]
        stats['corr_green_nir'] = np.corrcoef(g, nir)[0, 1]
        stats['corr_blue_nir'] = np.corrcoef(b, nir)[0, 1]
    else:
        for key in ['corr_red_green', 'corr_red_nir', 'corr_green_nir', 'corr_blue_nir']:
            stats[key] = np.nan

    return stats

# === Main Analysis Loop ===
print('Analyzing landcover sections...')

# Store results for each class (handle naming variations)
class_results = {
    'VEGETATED_DUNES': [],
    'BEACH_DRY': [],
    'BEACH_WET': [],
    'WATER': []
}

# Map transect ID to seed
transect_to_seed = {}
for _, row in manual_321197.iterrows():
    transect_to_seed[row['transectID']] = 321197
for _, row in manual_612823.iterrows():
    transect_to_seed[row['transectID']] = 612823

# Process each zone section
for idx, row in manual.iterrows():
    tid = row['transectID']
    zone_class = row['CLASS']
    start, end = row['start[m]'], row['end[m]']

    # Apply boundary tolerance
    start_buffered = start + BOUNDARY_TOLERANCE
    end_buffered = end - BOUNDARY_TOLERANCE

    if end_buffered <= start_buffered:
        continue  # Zone too small after buffering

    # Load spectral data
    seed = transect_to_seed.get(tid)
    if seed is None:
        continue

    spectral_data = load_spectral_data(tid, seed)
    if spectral_data is None:
        print(f'  Warning: No spectral data found for transect {tid}')
        continue

    # Extract zone section
    zone_data = spectral_data[
        (spectral_data['distance'] >= start_buffered) &
        (spectral_data['distance'] < end_buffered)
    ].copy()

    if len(zone_data) == 0:
        continue

    # Analyze this section
    section_stats = analyze_zone_section(zone_data)
    if section_stats is not None:
        section_stats['transect_id'] = tid
        section_stats['section_start'] = start
        section_stats['section_end'] = end
        section_stats['class'] = zone_class
        class_results[zone_class].append(section_stats)

print(f'  Processed {sum(len(v) for v in class_results.values())} sections')
print()

# === Aggregate Statistics by Class ===
print('Computing aggregate statistics by class...')

summary_stats = []

for landcover_class in ['VEGETATED_DUNES', 'BEACH_DRY', 'BEACH_WET', 'WATER']:
    sections = class_results[landcover_class]

    if len(sections) == 0:
        print(f'  {landcover_class}: No data')
        continue

    df_sections = pd.DataFrame(sections)

    print(f'  {landcover_class}: {len(sections)} sections, {df_sections["n_points"].sum():.0f} total points')

    # Compute mean and std for each feature
    for col in df_sections.columns:
        if col in ['transect_id', 'section_start', 'section_end', 'class']:
            continue

        if df_sections[col].dtype in [np.float64, np.int64]:
            summary_stats.append({
                'class': landcover_class,
                'feature': col,
                'mean': df_sections[col].mean(),
                'std': df_sections[col].std(),
                'min': df_sections[col].min(),
                'max': df_sections[col].max(),
                'median': df_sections[col].median(),
                'n_sections': len(sections)
            })

# Save summary statistics
summary_df = pd.DataFrame(summary_stats)
output_file = OUTPUT_DIR / 'class_spectral_profiles.csv'
summary_df.to_csv(output_file, index=False)
print(f'\nSaved: {output_file}')

# === Print Key Findings ===
print('\n' + '=' * 80)
print('KEY FINDINGS: LANDCOVER CLASS SIGNATURES')
print('=' * 80)

for landcover_class in ['VEGETATED_DUNES', 'BEACH_DRY', 'BEACH_WET', 'WATER']:
    class_stats = summary_df[summary_df['class'] == landcover_class]

    if len(class_stats) == 0:
        continue

    print(f'\n{landcover_class}')
    print('-' * 80)

    # Key discriminative features
    key_features = [
        ('brightness_mean', 'Brightness'),
        ('nir_mean', 'NIR'),
        ('nir_ratio_mean', 'NIR Ratio'),
        ('ndvi_mean', 'NDVI'),
        ('ndwi_mean', 'NDWI'),
        ('variability_mean', 'Variability'),
        ('rg_ratio_mean', 'R/G Ratio')
    ]

    for feat_name, feat_label in key_features:
        feat_data = class_stats[class_stats['feature'] == feat_name]
        if len(feat_data) > 0:
            row = feat_data.iloc[0]
            print(f'  {feat_label:20s}: {row["mean"]:7.2f} +- {row["std"]:6.2f}  '
                  f'range [{row["min"]:7.2f}, {row["max"]:7.2f}]')

print('\n' + '=' * 80)
print('Analysis complete!')
print('=' * 80)
