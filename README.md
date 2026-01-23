# Spectral Shoreline Detection System

Automated detection of beach-water boundaries (shell lines) using spectral analysis of aerial and satellite imagery.

## Overview

This system processes multi-band imagery (4-band RGBN, 3-band CIR, or 3-band RGB) along cross-shore transects to identify the dry beach / wet beach transition (shell line). The primary output is a GeoJSON file containing the detected shoreline position for each transect.

### Key Features

- **Multi-band support**: Works with 4-band (RGBN), CIR [NIR,R,G], and RGB imagery
- **Automatic band detection**: Distinguishes CIR from RGB automatically
- **Year-level processing**: Batch processing with consistent band handling per year
- **Cascading fallback**: Always provides output, even for difficult imagery
- **Confidence scoring**: Each detection includes reliability assessment

## Quick Start

```bash
# Process all years in imagery directory
python run.py \
    --imagery-root ./PAIS_shorelines/imagery \
    --transects ./INPUT/shorelineTransPais.json \
    --output ./OUTPUT

# Process specific year
python run.py --year 2020 --imagery-root ./imagery --output ./OUTPUT

# List available years without processing
python run.py --imagery-root ./imagery --dry-run
```

## Output

For each year processed:

```
OUTPUT/{year}/
├── shellline_{year}.geojson    # PRIMARY OUTPUT - detected shoreline
├── summary_{year}.json         # Processing metadata and statistics
├── spectral_profile_{year}.png # Diagnostic plot for transect 1000
└── processing.log              # Debug log
```

## Package Structure

```
spectral_classifier/
├── __init__.py          # Package exports
├── config.py            # Configuration and thresholds
├── main.py              # Pipeline orchestration
├── run.py               # CLI entry point
│
├── spectral/            # Spectral analysis
│   ├── sampler.py       # Value extraction along transects
│   └── features.py      # Feature computation (REUSABLE)
│
├── transition/          # Boundary detection
│   ├── nir.py           # NIR-based detection
│   ├── rgb.py           # RGB fallback detection
│   └── shell_line.py    # TransitionDetector class
│
├── utils/               # Utilities
│   ├── data_io.py       # Raster/GeoJSON loading
│   ├── export.py        # CSV/JSON export
│   ├── batch.py         # Year-based batch processing
│   └── logging_config.py
│
├── visualization/       # Plotting
│   └── plotting.py      # Diagnostic visualizations
│
└── optional/            # Non-essential modules
    ├── landcover_classifier.py  # Point classification
    └── boundary_types.py        # Veg/waterline detection

tools/
├── validate_shorelines.py      # Validation vs manual shorelines
└── DEPRECATED_HEURISTICS.md    # Historical approaches
```

## Detection Approach

### Primary: NIR Derivative (4-band and CIR)

The NIR band shows strong contrast between dry sand (high reflectance) and wet sand/water (low reflectance). The shell line is detected as the point of maximum negative NIR derivative.

**Confidence factors:**
- Derivative magnitude
- NIR drop absolute value
- Multi-band consensus
- R/G ratio validation (~1.0 for sand)
- Expected zone location (40-120m from land)

### Fallback: RGB Brightness (RGB-only)

When NIR is unavailable, brightness derivative is used as a proxy. Confidence is capped at 0.70 due to reduced reliability.

### Selection Strategy

1. **Strict mode**: High-confidence candidates in expected zone
2. **Relaxed mode**: Re-run with lower thresholds
3. **Best-available**: Strongest derivative in valid range
4. **Last resort**: Expected zone median (guaranteed output)

## Reusable Components

The spectral profiling in `spectral/features.py` is designed for adaptation to other applications:

### Derivative Features
```python
# These methods can detect edges in ANY spectral band
_compute_nir_derivative()      # First derivative
_compute_nir_derivative_multiscale()  # Multi-scale
_compute_nir_derivative_second()      # Inflection points
```

### Multi-band Consensus
```python
# Verify multiple bands agree on a transition
check_multi_band_consensus(features, index, threshold)
```

### Statistical Features
```python
_compute_variability()   # Local texture
_compute_slope()         # Trend analysis
_compute_spectral_angle()  # Spectral similarity
```

## Configuration

Key thresholds in `config.py`:

```python
THRESHOLDS = {
    # NIR derivative threshold for shell line detection
    'boundary_thresholds': {
        'dry_wet': {
            'nir_threshold': -8.0,          # Primary trigger
            'min_nir_drop_absolute': 39.0,  # Context validation
            'expected_location_min': 40,    # Zone constraints
            'expected_location_max': 120,
        }
    },
    
    # RGB fallback (lower thresholds, lower confidence)
    'rgb_thresholds': {
        'brightness_derivative_threshold': -5.0,
        'max_confidence': 0.70,
    }
}
```

## Validation

Compare algorithmic results against manual digitization:

```bash
python tools/validate_shorelines.py
```

Outputs per-year statistics including mean offset, directional bias, and per-transect comparisons.

## Requirements

- Python 3.8+
- geopandas
- rasterio
- shapely
- pandas
- numpy
- matplotlib
- scipy

## References

- Training data: seed=321197, n=2602 points, 10 transects
- Phase 6 validation: 20-transect feature analysis
- Empirical thresholds calibrated to PAIS (Padre Island National Seashore) imagery

---

*Developed for coastal shoreline monitoring at Padre Island National Seashore*