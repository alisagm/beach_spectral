# Beach Spectral Classifier - Project Status

**Last Updated:** 2025-10-17
**Status:** Phase 8 - Workspace Reorganization 🔄

---

## Overview

Automated shell line (DRY→WET boundary) detection using spectral (RGB/NIR) analysis along beach transects.

**Current Performance (Phase 7, 10 transects, ±5m tolerance):**
- F1-Score: 0.80
- Precision: 1.00 (zero false positives)
- Mean Absolute Error: 2.37m
- Detection Rate: 80%

---

## Project Structure

See `.claude/STRUCTURE.md` for complete package structure and data locations.

### Key Directories

```
beach_spectral/
├── spectral_classifier/        # Main classification module
├── tools/                      # Analysis and testing utilities
│   ├── feature_analysis/       # Feature analysis module
│   └── generate_validation_samples.py  # Validation sample generator
├── validation/                 # Validation pipeline
├── data/                       # Data directory
│   ├── inputs/                 # Manually classified transect profiles
│   └── output/                 # Testing and validation outputs
└── ARCHIVE/                    # Historical outputs and documentation
```

### External Data

- **Raster Dataset:** `C:\Users\alisa\Desktop\SIP\input_data\NAIP22`
- **Transect GeoJSONs:** `C:\Users\alisa\Desktop\SIP\input_data\PAIS_measured`
- **Manual Classifications:** `data/inputs/classified_manual_<seed>.csv`

---

## Quick Start

### Generate Validation Samples

```bash
python -m tools.generate_validation_samples \
  --rasters C:\Users\alisa\Desktop\SIP\input_data\NAIP22 \
  --transects C:\Users\alisa\Desktop\SIP\input_data\PAIS_measured \
  --output data/output/<seed>_run_001 \
  --seed 321197 \
  --num-samples 10
```

### Run Validation

```bash
python -m validation.run \
  --manual-csv data/inputs/classified_manual_321197.csv \
  --spectral-csv data/output/<seed>_run_001/sampled_spectral_data.csv \
  --output-dir data/output/<seed>_run_001/validation \
  --tolerance 5.0
```

---

## Current Implementation

### Phase 7: Guaranteed Shell Line Detection

**Implemented Features:**
1. **4-tier progressive fallback** - Ensures detection in all transects
   - STRICT (conf ≥ 0.60): Expected zone, high confidence
   - RELAXED (conf ≥ 0.50): Expected zone, lower thresholds
   - BEST-AVAILABLE (conf ≥ 0.40): Strongest signal anywhere
   - LAST-RESORT (conf ≥ 0.30): Default position

2. **Boundary-aware classification correction** - Enforces spatial consistency
   - Landward classes only before shell line
   - Seaward classes only after shell line

3. **Simplified visualization** - Shows only shell line boundaries

**Known Limitation:** Boundary type filtering can reject guaranteed detections when class labels are incorrect (~5-10% of transects).

### Phase 8: Workspace Reorganization (Current)

**Status:** In Progress

**Changes:**
- Consolidated scattered outputs into `ARCHIVE/`
- Moved `feature_analysis` to `tools/`
- Renamed `training_data.py` → `generate_validation_samples.py`
- Created standardized `data/` structure
- Established consistent output naming: `<seed>_run_<number>/`
- Centralized documentation

**Next Steps:**
- Update all import statements
- Configure scripts for new output paths
- Test complete workflow

---

## Key Files

### Core Module
- `spectral_classifier/main.py` - Classification pipeline
- `spectral_classifier/transition.py` - Boundary detection (Phase 7)
- `spectral_classifier/classifier.py` - Landcover classification
- `spectral_classifier/config.py` - Configuration and thresholds

### Tools & Validation
- `tools/generate_validation_samples.py` - Sample generator
- `tools/feature_analysis/` - Feature analysis module
- `validation/run.py` - Main validation script

### Documentation
- `.claude/STRUCTURE.md` - Package structure and paths
- `HISTORY.md` - Development history with links to archived docs
- `README.md` - User guide

---

## Performance Targets

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| F1-Score | ≥ 0.70 | 0.80 | ✅ |
| Precision | ≥ 0.70 | 1.00 | ✅ |
| Recall | ≥ 0.70 | 0.67 | ⚠️ (95%) |
| MAE | < 10m | 2.37m | ✅ |

---

## Recent Changes

**2025-10-17 - Phase 8 Workspace Reorganization** 🔄
- Restructured folders and consolidated outputs
- Created STRUCTURE.md and HISTORY.md
- Updated .gitignore for new structure

**2025-10-17 - Phase 7 Implementation Complete** ✅
- Guaranteed shell line detection with fallbacks
- Boundary-aware classification correction
- Simplified shell line visualization

**See `HISTORY.md` for complete development history and links to phase documentation.**

---

## Next Actions

1. Update import statements in all scripts for relocated modules
2. Configure scripts to use new `data/output/<seed>_run_<number>/` paths
3. Run full validation test with reorganized structure
4. Address Phase 7 boundary filtering limitation (future work)

---

*For detailed historical information, performance comparisons, and phase documentation, see `HISTORY.md` and `ARCHIVE/docs/`.*
