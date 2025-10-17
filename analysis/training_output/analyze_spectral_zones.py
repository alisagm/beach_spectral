"""Analyze spectral characteristics by manually classified zones."""
import pandas as pd
import numpy as np

# Configuration
BOUNDARY_TOLERANCE = 5.0  # meters - buffer around boundaries to avoid edge effects

# Load data
df = pd.read_csv('sampled_spectral_data.csv')

# Read manual classifications - need to handle the list column carefully
manual = pd.read_csv('classified_manual.csv',
                     usecols=['transectID', 'start[m]', 'end[m]', 'CLASS'])

print('=' * 70)
print('SPECTRAL CHARACTERISTICS BY LAND COVER CLASS')
print(f'(using {BOUNDARY_TOLERANCE}m tolerance buffer at zone boundaries)')
print('=' * 70)

# Collect statistics for each class
class_stats = {}

for _, row in manual.iterrows():
    tid = row['transectID']
    zone = row['CLASS']
    start, end = row['start[m]'], row['end[m]']

    # Apply tolerance buffer to avoid boundary uncertainty
    start_buffered = start + BOUNDARY_TOLERANCE
    end_buffered = end - BOUNDARY_TOLERANCE

    # Skip zones that are too small after buffering
    if end_buffered <= start_buffered:
        continue

    # Get spectral data for this zone
    tdata = df[df['TransectID'] == tid]
    zone_data = tdata[(tdata['distance'] >= start_buffered) & (tdata['distance'] < end_buffered)]

    if len(zone_data) == 0:
        continue

    # Initialize if needed
    if zone not in class_stats:
        class_stats[zone] = {
            'red': [], 'green': [], 'blue': [], 'nir': [],
            'brightness': [], 'ndvi': [], 'ndwi': [],
            'nir_ratio': [], 'blue_red_ratio': [], 'variability': []
        }

    # Calculate features for each point
    for _, point in zone_data.iterrows():
        r, g, b, n = point['red'], point['green'], point['blue'], point['nir']

        # Basic stats
        class_stats[zone]['red'].append(r)
        class_stats[zone]['green'].append(g)
        class_stats[zone]['blue'].append(b)
        class_stats[zone]['nir'].append(n)

        # Derived features
        brightness = (r + g + b + n) / 4.0
        class_stats[zone]['brightness'].append(brightness)

        ndvi = (n - r) / (n + r) if (n + r) > 0 else 0
        class_stats[zone]['ndvi'].append(ndvi)

        ndwi = (g - n) / (g + n) if (g + n) > 0 else 0
        class_stats[zone]['ndwi'].append(ndwi)

        nir_ratio = n / brightness if brightness > 0 else 0
        class_stats[zone]['nir_ratio'].append(nir_ratio)

        blue_red_ratio = b / r if r > 0 else 0
        class_stats[zone]['blue_red_ratio'].append(blue_red_ratio)

# Calculate variability for each zone segment
for _, row in manual.iterrows():
    tid = row['transectID']
    zone = row['CLASS']
    start, end = row['start[m]'], row['end[m]']

    # Apply tolerance buffer
    start_buffered = start + BOUNDARY_TOLERANCE
    end_buffered = end - BOUNDARY_TOLERANCE

    if end_buffered <= start_buffered:
        continue

    tdata = df[df['TransectID'] == tid]
    zone_data = tdata[(tdata['distance'] >= start_buffered) & (tdata['distance'] < end_buffered)]

    if len(zone_data) > 5:  # Need enough points for variability
        # Calculate brightness variability (std of brightness)
        brightness = (zone_data['red'] + zone_data['green'] +
                     zone_data['blue'] + zone_data['nir']) / 4.0
        variability = brightness.std()
        class_stats[zone]['variability'].append(variability)

# Print summary statistics for each class
for zone in ['VEGETATED_DUNE', 'BEACH_DRY', 'BEACH_WET', 'WATER']:
    if zone not in class_stats:
        continue

    print(f'\n{zone}')
    print('-' * 70)

    stats = class_stats[zone]
    n_points = len(stats['red'])
    n_segments = len([row for _, row in manual.iterrows() if row['CLASS'] == zone])

    print(f'  Sample size: {n_points} points across {n_segments} transect segments')
    print()

    # Band values
    print(f'  Raw bands:')
    print(f'    Red:   {np.mean(stats["red"]):6.1f} +- {np.std(stats["red"]):5.1f}  '
          f'range [{np.min(stats["red"]):3.0f}, {np.max(stats["red"]):3.0f}]')
    print(f'    Green: {np.mean(stats["green"]):6.1f} +- {np.std(stats["green"]):5.1f}  '
          f'range [{np.min(stats["green"]):3.0f}, {np.max(stats["green"]):3.0f}]')
    print(f'    Blue:  {np.mean(stats["blue"]):6.1f} +- {np.std(stats["blue"]):5.1f}  '
          f'range [{np.min(stats["blue"]):3.0f}, {np.max(stats["blue"]):3.0f}]')
    print(f'    NIR:   {np.mean(stats["nir"]):6.1f} +- {np.std(stats["nir"]):5.1f}  '
          f'range [{np.min(stats["nir"]):3.0f}, {np.max(stats["nir"]):3.0f}]')
    print()

    # Derived features
    print(f'  Derived features:')
    print(f'    Brightness:     {np.mean(stats["brightness"]):6.2f} +- {np.std(stats["brightness"]):5.2f}')
    print(f'    NDVI:           {np.mean(stats["ndvi"]):6.3f} +- {np.std(stats["ndvi"]):5.3f}  '
          f'range [{np.min(stats["ndvi"]):6.3f}, {np.max(stats["ndvi"]):6.3f}]')
    print(f'    NDWI:           {np.mean(stats["ndwi"]):6.3f} +- {np.std(stats["ndwi"]):5.3f}  '
          f'range [{np.min(stats["ndwi"]):6.3f}, {np.max(stats["ndwi"]):6.3f}]')
    print(f'    NIR ratio:      {np.mean(stats["nir_ratio"]):6.3f} +- {np.std(stats["nir_ratio"]):5.3f}')
    print(f'    Blue/Red ratio: {np.mean(stats["blue_red_ratio"]):6.3f} +- {np.std(stats["blue_red_ratio"]):5.3f}')

    if stats['variability']:
        print(f'    Variability:    {np.mean(stats["variability"]):6.2f} +- {np.std(stats["variability"]):5.2f}  '
              f'range [{np.min(stats["variability"]):5.2f}, {np.max(stats["variability"]):5.2f}]')

print('\n' + '=' * 70)
