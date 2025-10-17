# Beach Spectral Classifier - Package Structure

## Overview
This document describes the structure of the beach_spectral package and locations of key data files.

## Package Structure

```
beach_spectral/
├── spectral_classifier/        # Main classification module
│   ├── main.py                 # Primary classification pipeline
│   ├── classifier.py           # Core classification logic
│   ├── features.py             # Feature extraction
│   ├── transition.py           # Transition detection
│   ├── visualization.py        # Visualization utilities
│   ├── sampler.py              # Data sampling
│   ├── data_io.py              # Data I/O operations
│   ├── config.py               # Configuration management
│   ├── utils.py                # Utility functions
│   └── archive/                # Deprecated code
│
├── tools/                      # Development and analysis tools
│   ├── feature_analysis/       # Feature analysis module
│   │   ├── scripts/            # Analysis scripts
│   │   ├── data/               # Analysis data files
│   │   └── outputs/            # Analysis outputs
│   │       ├── reports/        # Analysis reports
│   │       ├── statistics/     # Statistical outputs
│   │       └── visualizations/ # Analysis plots
│   ├── validation/             # Validation pipeline
│   │   ├── run.py              # Main validation runner
│   │   ├── internal.py         # Internal validation logic
│   │   ├── feature_importance.py  # Feature importance analysis
│   │   ├── addons.py           # Additional validation utilities
│   │   └── trueorfalse.py      # Validation metrics
│   ├── tests/                  # Unit and integration tests
│   │   └── test_single_transect.py
│   └── generate_validation_samples.py  # Training data generation
│
├── data/                       # Data directory (runtime)
│   ├── inputs/                 # Manually classified transect profiles
│   │   └── *.csv              # Classified CSV files
│   └── output/                 # Testing and validation outputs
│       └── <seed>_run_<N>/    # Individual run outputs
│
├── ARCHIVE/                    # Historical outputs and documentation
│   ├── outputs/                # Archived testing/validation outputs
│   └── docs/                   # Archived documentation
│
└── .claude/                    # Claude Code configuration
    └── STRUCTURE.md            # This file

```

## External Data Locations

### Raster Dataset
**Location:** `C:\Users\alisa\Desktop\SIP\input_data\NAIP22`
**Description:** NAIP (National Agriculture Imagery Program) 2022 aerial imagery raster dataset

### Shoreline Transect GeoJSONs
**Location:** `C:\Users\alisa\Desktop\SIP\input_data\PAIS_measured`
**Description:** Padre Island National Seashore (PAIS) measured shoreline transects in GeoJSON format

### Manually Classified Transect Profiles
**Location:** `beach_spectral/data/inputs/`
**Structure:**
- `classified_manual_<seed>.csv` - Manually classified spectral profiles for specific random seeds
- Each CSV contains:
  - Transect ID
  - Distance along transect
  - Spectral values (RGB/NIR bands)
  - Manual classification labels
  - Zone boundaries

## Output Structure

### Testing and Validation Outputs
**Location:** `beach_spectral/data/output/<seed>_run_<number>/`
**Structure:**
- Each test/validation run creates a uniquely named subfolder
- Naming convention: `<random_seed>_run_<sequential_number>`
- Example: `321197_run_001/`

**Contents:**
- Classification results
- Comparison plots
- Validation metrics
- Boundary detection outputs
- Feature importance analysis

### Archived Outputs
**Location:** `beach_spectral/ARCHIVE/outputs/`
**Description:** Historical outputs from previous testing and validation runs

## Configuration Files

### Git Configuration
- `.gitignore` - Excludes:
  - `data/output/` - Runtime outputs
  - `ARCHIVE/` - Historical data
  - Python cache files
  - IDE files

### Project Documentation
- `PROJECT_STATUS.md` - Current project status (concise)
- `HISTORY.md` - Historical documentation with relative paths to archived files
- `README.md` - Package overview and usage

## Module Descriptions

### spectral_classifier/
The main classification module containing the core logic for:
- Loading and preprocessing spectral data
- Extracting features from spectral profiles
- Detecting transitions between beach zones
- Classifying beach features
- Visualizing results

### tools/
Consolidated directory containing all development, testing, and analysis utilities:

#### feature_analysis/
Detailed spectral profile characteristic analysis
- Spectral zone analysis scripts
- Boundary detection diagnostics
- Feature profiling and visualization
- Outputs organized by type (reports, statistics, visualizations)

#### validation/
Validation pipeline for testing classifier accuracy:
- Compares automated classifications against manual classifications
- Generates validation reports and metrics
- Produces comparison visualizations
- Feature importance analysis

#### tests/
Unit and integration tests for the package:
- Single transect classification tests
- Component-level tests

#### generate_validation_samples.py
Training data generation and validation sample creation

## Workflow

### Standard Testing/Validation Workflow
1. **Generate/Prepare Training Data:** `tools/generate_validation_samples.py`
   - Creates or loads manually classified data for a specific seed
   - Saves to `data/inputs/classified_manual_<seed>.csv`

2. **Run Validation:** `tools/validation/run.py`
   - Runs classifier on test transects
   - Compares results against manual classifications
   - Saves outputs to `data/output/<seed>_run_<number>/`

3. **Analyze Results:**
   - Review validation reports
   - Examine comparison plots
   - Iterate on classifier improvements

### Feature Analysis Workflow
1. Run analysis scripts from `tools/feature_analysis/scripts/`
2. Outputs saved to `tools/feature_analysis/outputs/`
3. Use results to inform classifier improvements

### Testing Workflow
1. Run unit tests from `tools/tests/`
2. Verify individual component functionality
3. Ensure code quality and correctness
