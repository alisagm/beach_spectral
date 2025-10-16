# Test Scripts

This directory contains test scripts, diagnostic tools, and single-transect testing utilities for the beach spectral classifier.

## Contents

### test_single_transect.py

Tests the classifier on a single specified transect for debugging and detailed inspection.

**Purpose:**
- Debug specific transects
- Inspect detailed classification steps
- Visualize intermediate processing stages

**Usage:**
```bash
python -m tests.test_single_transect --transect-id T001
```

### diagnostics_phase6c.py

Phase 6C diagnostics script for threshold tuning validation.

**Purpose:**
- Analyze Phase 6C threshold performance
- Compare STRICT vs MODERATE vs RELAXED configurations
- Generate diagnostic reports for threshold tuning

**Usage:**
```bash
python -m tests.diagnostics_phase6c
```

## Test Categories

### Unit Tests
- (Future) Individual component tests for features, classifier, transition detector

### Integration Tests
- Single transect tests (test_single_transect.py)
- Multi-transect validation (see `validation/`)

### Diagnostics
- Phase-specific performance analysis (diagnostics_phase6c.py)
- Threshold tuning investigations

## Related Testing Resources

### Validation Suite (`validation/`)
For comprehensive performance testing:
- `validation/run.py` - Full validation pipeline
- `validation/test_phase6_thresholds.py` - Threshold comparison tests
- `validation/internal.py` - Internal metrics computation

### Analysis Workspace (`analysis/`)
For exploratory data analysis:
- `analysis/training_output/` - Sample data generation
- `analysis/feature_analysis/` - Feature investigation

### Development Tools (`tools/`)
For data generation:
- `tools/training_data.py` - Training sample generation

## Running Tests

```bash
# Test single transect
python -m tests.test_single_transect --transect-id T001

# Run diagnostics
python -m tests.diagnostics_phase6c

# Run full validation suite (from root)
python validation/run.py
```

## Notes

- These are development/debugging tools, not automated CI tests
- For production testing, use the validation suite in `validation/`
- Test outputs may be written to local directories
- Some tests require access to raster data and transect geometries
