# Feature Analysis - Spectral Classifier Improvement Project

## Overview

This folder contains a comprehensive analysis of spectral characteristics for manually classified landcover sections, focused on improving the detection of the shell line (BEACH_DRY → BEACH_WET transition).

**Created**: 2025-10-15
**Dataset**: 20 manually classified transects (10 from seed 321197, 10 from seed 612823)
**Total Samples**: 76 landcover sections, 60 boundaries analyzed

---

## Folder Structure

```
feature_analysis/
├── IMPROVEMENT_PLAN.md              # Detailed improvement plan and methodology
├── README.md                        # This file - summary of results
├── data/                            # Source data
│   ├── classified_manual_321197.csv
│   └── classified_manual_612823.csv
├── scripts/                         # Analysis scripts
│   ├── spectral_profiling.py        # Landcover class characterization
│   ├── transition_analysis.py       # Boundary analysis
│   ├── run_full_analysis.py         # Master script
│   └── [copied from training_output/]
└── outputs/
    ├── statistics/                  # CSV results
    │   ├── class_spectral_profiles.csv
    │   └── transition_characteristics.csv
    ├── visualizations/              # Plots (to be generated)
    └── reports/                     # Summary documents (to be generated)
```

---

## Key Findings

### Landcover Class Signatures

#### BEACH_DRY (n=20 sections, 821 points)
- **Brightness**: 210 ± 5 (very bright, uniform)
- **NIR**: 188 ± 8 (high reflectance)
- **NIR Ratio**: 0.90 ± 0.02
- **Variability**: 3.7 ± 2.1 (very smooth/uniform)
- **R/G Ratio**: 1.01 ± 0.00 (red ≈ green)
- **NDVI**: -0.08 ± 0.02
- **NDWI**: 0.08 ± 0.02

**Distinguishing Features**: Extremely bright, high NIR, very low variability

#### BEACH_WET (n=16 sections, 227 points)
- **Brightness**: 156 ± 15 (moderate, more variable)
- **NIR**: 96 ± 17 (intermediate - water-saturated sand)
- **NIR Ratio**: 0.61 ± 0.06
- **Variability**: 11.3 ± 5.2 (moderate texture)
- **R/G Ratio**: 1.05 ± 0.03
- **NDVI**: -0.34 ± 0.07
- **NDWI**: 0.32 ± 0.06

**Distinguishing Features**: Intermediate NIR, elevated NDWI, moderate variability

#### WATER (n=20 sections, 3136 points)
- **Brightness**: 107 ± 14 (dark)
- **NIR**: 24 ± 7 (very low - water absorbs NIR)
- **NIR Ratio**: 0.20 ± 0.04
- **Variability**: 30.5 ± 6.8 (high - waves, foam)
- **R/G Ratio**: 0.82 ± 0.06 (blue-shifted)
- **NDVI**: -0.71 ± 0.05
- **NDWI**: 0.75 ± 0.05 (very high - water signal)

**Distinguishing Features**: Very low NIR, high NDWI, high variability, blue-green color

#### VEGETATED_DUNES (n=20 sections combined, 923 points)
- **Brightness**: 143 ± 19 (moderate, variable)
- **NIR**: 126 ± 15 (moderately high - vegetation)
- **NIR Ratio**: 0.88 ± 0.04
- **Variability**: 27 ± 10 (high - vegetation structure)
- **R/G Ratio**: 1.07 ± 0.04
- **NDVI**: -0.12 ± 0.03 (still negative - not healthy veg)
- **NDWI**: 0.10 ± 0.03

**Distinguishing Features**: Moderate brightness, high variability, higher R/G ratio

---

### Shell Line (BEACH_DRY → BEACH_WET) Signature

**Sample Size**: 20 boundaries analyzed

#### NIR Characteristics
- **Absolute drop**: 55.1 ± 13.6 units (highly consistent)
- **Before value**: 171.6 ± 18.2 units (dry beach level)
- **After value**: 116.4 ± 20.1 units (wet beach level)
- **Derivative peak**: -13.80 ± 4.00 units/m (sharp negative slope)
- **Sustained drop**: 9.1 ± 3.1 consecutive points (reliable signal)

#### Brightness Characteristics
- **Absolute drop**: 32.0 ± 9.2 units
- **Before value**: 201.8 ± 9.6 units (bright sand)
- **After value**: 169.7 ± 14.9 units (darker wet sand)

#### Transition Geometry
- **Transition width**: 3.9 ± 1.8 meters (80% change span)
- **Asymmetry ratio**: 0.88 ± 0.44 (fairly symmetric)
- **Band synchrony**: 1.85 ± 2.06 meters std (bands mostly agree)

#### Spectral Indices
- **NIR ratio change**: 0.171 ± 0.052 (large shift)
- **R/G ratio at boundary**: 1.025 ± 0.045 (R ≈ G, as expected)
- **Variability ratio**: 5.87 ± 5.53 (big increase - wet sand is textured)

---

## Algorithm Improvement Recommendations

Based on these empirical findings, the following improvements are recommended for spectral_classifier/transition.py:

### 1. Strengthen Shell Line Detection Criteria

**Current Issue**: False positives from within-zone variability and weak candidates

**Proposed Solution - Multi-Criteria Composite Score**:

```python
# Required conditions (all must pass):
1. NIR derivative < -8.0 units/m (currently -4.38, too permissive)
2. Absolute NIR drop > 40 units (empirical minimum from data)
3. Brightness before > 190 (confirm dry beach context)
4. Sustained drop ≥ 5 consecutive points (currently 3)

# Confidence bonuses:
+ 0.15 if NIR drop > 50 units (typical value)
+ 0.10 if derivative < -12 units/m (sharp transition)
+ 0.10 if brightness drop > 25 units
+ 0.05 if R/G ratio 0.98-1.08 (shell line signature)
+ 0.05 if variability increases >2x
```

### 2. Contextual Validation

- **Expected location**: Shell line typically occurs 40-120m from transect start
- **Uniqueness**: Only one major DRY→WET transition expected per transect
- **Zone confirmation**: Check that NIR and brightness values before boundary match DRY_BEACH profile (NIR ~180-190, brightness ~200-210, variability <10)

### 3. Relative vs Absolute Detection

Instead of requiring absolute threshold, **rank all candidates** in expected region and select best:

1. Find all NIR derivative minima in region [40-120m]
2. Compute composite confidence score for each
3. Select highest-scoring candidate (even if below absolute threshold)
4. Only fail if no candidates have confidence > 0.60

### 4. New Features to Implement

- **Brightness gradient**: d(brightness)/dx (secondary confirmation)
- **Variability change**: std_after / std_before (expect ratio > 2)
- **Multi-scale consensus**: Check 3m, 5m, 7m windows all show signal
- **Context features**: Mean NIR in previous 20m, next 20m

---

## Comparison: Shell Line vs Other Boundaries

| Characteristic | DRY→WET (Shell) | WET→WATER | VEG→DRY |
|----------------|-----------------|-----------|---------|
| NIR Drop | **55 ± 14** | 46 ± 25 | -4 ± 23 |
| Derivative Peak | **-13.8 ± 4.0** | -13.8 ± 7.4 | -25.4 ± 10.3 |
| Sustained Drop | **9.1 ± 3.1** | 7.1 ± 2.8 | 3.9 ± 1.1 |
| Transition Width | **3.9 ± 1.8 m** | 4.7 ± 4.0 m | 1.1 ± 1.9 m |
| R/G Ratio | **1.02 ± 0.04** | 1.02 ± 0.03 | 1.02 ± 0.02 |
| Variability Ratio | **5.87 ± 5.53** | 2.32 ± 3.49 | 0.69 ± 0.33 |

**Key Discriminators for Shell Line**:
1. **Consistent large NIR drop** (55 units, low variability)
2. **High sustained drop** (longest of all boundary types)
3. **Large variability increase** (5-6x, unique signature)
4. **Brightness context** (always starts from very bright DRY zone)

---

## Next Steps

### Immediate Actions
1. ✅ Run comprehensive spectral analysis → COMPLETE
2. ⏳ Create visualization suite (box plots, transition profiles, ROC curves)
3. ⏳ Generate final characterization report with dummy spectra
4. ⏳ Implement algorithm improvements in transition.py
5. ⏳ Validate on training data and measure performance gain

### Implementation Priority
1. **Phase 6A**: Tighten shell line thresholds based on empirical data
   - NIR derivative: -4.38 → -8.0 units/m
   - Absolute NIR drop: 15 → 40 units minimum
   - Sustained drop: 3 → 5 points

2. **Phase 6B**: Add contextual validation
   - Brightness check before boundary
   - Expected location prior
   - Uniqueness constraint

3. **Phase 6C**: Implement composite confidence scoring
   - Multi-criteria evaluation
   - Relative candidate ranking
   - Feature ensemble approach

---

## Performance Targets

### Baseline (Phase 5)
- Precision: TBD (to be measured)
- Recall: TBD (to be measured)
- F1-score: TBD (to be measured)

### Target (Phase 6)
- **Precision**: > 0.90 (few false positives)
- **Recall**: > 0.90 (few missed detections)
- **F1-score**: > 0.90 (balanced performance)
- **Median error**: < 3 meters (accurate localization)

### Stretch Goals
- F1-score > 0.95
- Zero false positives in uniform DRY_BEACH zones
- Robust performance across different environmental conditions

---

## Usage

### Running the Analysis

```bash
# Run full pipeline
cd feature_analysis/scripts
python run_full_analysis.py

# Run individual analyses
python spectral_profiling.py
python transition_analysis.py
```

### Outputs

**CSV Files** (outputs/statistics/):
- `class_spectral_profiles.csv`: Comprehensive statistics for each landcover class
- `transition_characteristics.csv`: Detailed metrics for each boundary

**Visualizations** (to be generated):
- Class profile box plots
- Transition signature waterfall plots
- ROC curves for boundary detection
- Feature importance rankings

---

## References

- Improvement plan: `IMPROVEMENT_PLAN.md`
- Current algorithm: `../spectral_classifier/transition.py`
- Configuration: `../spectral_classifier/config.py`
- Training data: `../training_output/seed_*/`
- Validation results: `../validation/outputs_phase5/`

---

**Last Updated**: 2025-10-15
**Status**: Analysis Complete → Algorithm Improvement In Progress
