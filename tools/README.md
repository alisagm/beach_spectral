# Development Tools

This directory contains development and data generation utilities for the beach spectral classifier project.

## Contents

### training_data.py

Generates training/test datasets by running the spectral classifier on randomly sampled transects with various configurations.

**Purpose:**
- Generate spectral data samples for analysis
- Test classifier behavior across different transects
- Produce logs for debugging and threshold tuning

**Usage:**

```bash
# Generate training data with 10 transects
python -m tools.training_data \
  --boundary-mode shore_only \
  --num-transects 10 \
  --seed 321197

# Run with custom threshold configuration
python -m tools.training_data \
  --boundary-mode shore_only \
  --num-transects 20 \
  --seed 612823 \
  --threshold-mode MODERATE
```

**Arguments:**
- `--boundary-mode`: Detection mode (shore_only, all, foam_enhanced)
- `--num-transects`: Number of transects to sample (default: 10)
- `--seed`: Random seed for reproducibility
- `--threshold-mode`: Threshold configuration (STRICT, MODERATE, RELAXED)

**Outputs:**
Results are saved to `analysis/training_output/`:
- `sampled_spectral_data.csv` - All spectral samples with classifications
- `training_data.log` - Detailed processing log
- `training_sample_info.json` - Sample metadata
- `seed_*/` - Seed-specific data and plots

## Related Directories

- `analysis/` - Analysis workspace with training outputs and feature analysis
- `validation/` - Validation suite for performance testing
- `tests/` - Unit tests and diagnostic scripts

## Notes

- These tools are for development only and are not part of the core `spectral_classifier` API
- Training data generation requires access to raster data and transect geometries
- See PROJECT_STATUS.md for current development phase and targets
