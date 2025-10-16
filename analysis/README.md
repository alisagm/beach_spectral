# Analysis Workspace

This directory contains all analysis outputs, feature investigations, and training data from the beach spectral classifier development process.

## Directory Structure

```
analysis/
├── feature_analysis/         # Spectral characteristic analysis
│   ├── data/                # Manually classified training data
│   ├── scripts/             # Analysis scripts
│   ├── outputs/             # Analysis results (CSV, plots)
│   ├── IMPROVEMENT_PLAN.md  # Detailed improvement methodology
│   └── README.md            # Feature analysis documentation
│
└── training_output/         # Training data generation outputs
    ├── seed_321197/        # First seed dataset
    ├── seed_612823/        # Second seed dataset
    ├── *.py                # Ad-hoc analysis scripts
    ├── sampled_spectral_data.csv
    ├── training_data.log
    ├── training_sample_info.json
    └── plots/              # Visualization outputs
```

## Purpose

This workspace consolidates all intermediate analysis, training data, and investigation results used during the development and tuning of the spectral classifier. These materials are separate from:
- **Core module** (`spectral_classifier/`) - Production code
- **Validation suite** (`validation/`) - Formal performance testing
- **Tests** (`tests/`) - Unit and integration tests

## Key Components

### Feature Analysis

Contains comprehensive spectral analysis of manually classified landcover sections:
- **Dataset**: 20 manually classified transects (76 sections, 60 boundaries)
- **Purpose**: Empirically derive optimal detection thresholds
- **Results**: Class signatures, boundary characteristics, detection rules

See `feature_analysis/README.md` for detailed findings.

### Training Output

Contains outputs from `tools/training_data.py` runs:
- Random transect samples with different seeds
- Spectral profiles and classifications
- Boundary detection results
- Diagnostic plots and logs

## Workflow

1. **Generate training data** using `tools/training_data.py`
2. **Analyze results** in this workspace
3. **Derive insights** from feature_analysis scripts
4. **Update thresholds** in `spectral_classifier/config.py`
5. **Validate improvements** using `validation/run.py`

## Related Documentation

- `PROJECT_STATUS.md` - Current development status
- `docs/IMPROVEMENT_HISTORY.md` - Development timeline
- `docs/PHASE_6_IMPLEMENTATION.md` - Current phase details
- `validation/outputs/` - Formal validation results

## Notes

- This workspace is for development only and is not part of the production deployment
- Analysis scripts may have dependencies on local data paths
- Outputs are version-controlled for reproducibility but may be large
- Use seed-specific folders to compare different parameter configurations
