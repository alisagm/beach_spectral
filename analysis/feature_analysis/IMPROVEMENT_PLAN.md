# Spectral Classifier Improvement Plan

**Date**: 2025-10-15
**Focus**: Shell line (BEACH_DRY → BEACH_WET transition) detection accuracy
**Approach**: Exhaustive spectral analysis → empirical characterization → algorithm refinement

---

## Executive Summary

The spectral classifier has undergone 5 phases of iterative refinement but still shows problematic outputs. This plan implements a systematic, data-driven approach to:

1. **Exhaustively analyze** spectral profiles of manually classified landcover sections
2. **Identify empirical signatures** for each landcover class and transition type
3. **Extract discriminative features** that distinguish shell line from false positives
4. **Refine detection algorithms** based on empirical findings
5. **Validate improvements** against ground truth data

**Key Insight**: Previous iterations focused on threshold tuning. This plan shifts to **feature discovery** and **signature-based detection**.

---

## Background: Current State

### Algorithm Status (Phase 5)

**Detection Methods** (spectral_classifier/transition.py):
- **VEG_DUNES→BEACH_DRY**: Inflection point detection (2nd derivative zero-crossing)
- **BEACH_WET→WATER**: RGB foam peak detection + NIR confirmation
- **BEACH_DRY→BEACH_WET** (shell line): NIR derivative magnitude + sustainability + multi-band consensus

**Validation Findings** (validation/outputs_phase5):
- NIR is strongest discriminator (F-statistic: 14,388)
- Brightness highly discriminative (F-stat: 1,748)
- NIR ratio also excellent (F-stat: 14,388)
- BEACH_DRY very distinct: brightness 207±9, variability 3±4, NIR ratio 0.83±0.07
- BEACH_WET intermediate: brightness 163±21, variability 8±5, NIR ratio 0.54±0.14

**Known Issues**:
- False positives from within-zone variability
- Missed detections when transition is gradual
- Multiple weak candidates vs single strong candidate
- Inconsistent performance across different transects

---

## Phase 1: Exhaustive Feature Analysis Pipeline

### Objective
Systematically analyze all spectral characteristics to identify the "signature" of each landcover class and transition.

### Data Sources
- `training_output/seed_321197/classified_manual_321197.csv` (10 transects, manually labeled)
- `training_output/seed_612823/classified_manual_612823.csv` (10 transects, manually labeled)
- Corresponding `sampled_spectral_data.csv` files with R/G/B/NIR values

### 1.1 Point-by-Point Spectral Analysis

For each landcover class (with 5m boundary tolerance):

**Raw Band Statistics**:
- Mean, std, min, max, median, quartiles
- Coefficient of variation (CV)
- Distribution shapes (skewness, kurtosis)

**Band Ratios** (all pairwise combinations):
- R/G, R/B, R/NIR
- G/B, G/NIR
- B/NIR
- Normalized differences (e.g., (R-G)/(R+G))

**Spectral Indices**:
- NDVI: (NIR-R)/(NIR+R)
- NDWI: (G-NIR)/(G+NIR)
- NDSI: Normalized difference combinations
- Brightness: (R+G+B+NIR)/4
- Variability: std(brightness) in local window
- Red/Green ratio (for shell line validation)
- Blue/Red ratio (for water detection)

**First Derivatives** (rate of change, units/meter):
- dR/dx, dG/dx, dB/dx, dNIR/dx
- d(brightness)/dx
- d(NDVI)/dx, d(NDWI)/dx
- d(nir_ratio)/dx

**Second Derivatives** (curvature):
- d²NIR/dx² (inflection detection)
- d²brightness/dx²
- Zero-crossing analysis

**Higher-Order Features**:
- Third derivatives (transition rate changes)
- Spectral angle (change in spectral vector direction)
- Cross-band correlation (how do bands co-vary?)

### 1.2 Rolling-Window Analysis

**Window Sizes**: 3m, 5m, 7m, 9m, 11m

For each window size, compute:
- Moving average (smoothed signal)
- Moving std dev (local variability)
- Local minima/maxima locations
- Trend classification (increasing/decreasing/stable)
- Autocorrelation (periodicity detection)
- Spectral angle between consecutive windows

**Multi-Scale Analysis**:
- Compare derivative magnitudes across scales
- Identify scale-invariant features
- Detect scale-dependent features (noise vs signal)

### 1.3 Transition Zone Analysis

**Region**: ±10m around manually labeled boundaries

**Measurements**:
- **Pre/post differences**: Mean values before vs after transition
- **Transition sharpness**: Peak derivative magnitude
- **Transition width**: Distance over which 80% of change occurs
- **Transition asymmetry**: Different rates of approach from each side
- **Multi-band synchrony**: Do all bands transition at same location?
- **Confidence indicators**: Sustainability, consensus, noise level

**Boundary-Specific Patterns**:
- VEG→DRY: Inflection point characteristics
- DRY→WET: Shell line signature (primary focus)
- WET→WATER: Foam/surf characteristics

---

## Phase 2: Statistical Characterization

### Objective
Extract quantitative discriminators for each class and transition type.

### 2.1 Class Separability Metrics

**Pairwise Comparisons**:
- Effect sizes (Cohen's d) for all class pairs
- Welch's t-test (unequal variances)
- Mann-Whitney U test (non-parametric)

**Binary Classification Performance**:
- ROC curves for each boundary type
- Optimal thresholds (Youden's index: max(sensitivity + specificity - 1))
- Area Under Curve (AUC) for feature ranking
- Confidence intervals (bootstrap)

**Multivariate Separability**:
- Linear Discriminant Analysis (LDA)
- Canonical variate analysis
- Mahalanobis distance between class centroids

### 2.2 Feature Importance Ranking

**Statistical Tests**:
- ANOVA F-statistics (already computed in phase5)
- Kruskal-Wallis H-test (non-parametric)
- Chi-square test for categorical features

**Machine Learning Methods**:
- Random Forest feature importances (Gini importance)
- Logistic Regression coefficients (L1 regularized)
- Mutual Information scores
- SHAP values (interpretable ML)

**Correlation Analysis**:
- Feature redundancy (correlation matrix)
- Principal Component Analysis (dimensionality reduction)
- Identify orthogonal feature combinations

### 2.3 Pattern Identification

**Decision Tree Rules**:
- Train interpretable decision tree
- Extract human-readable rules (e.g., "if NIR > 180 and brightness > 200 then DRY_BEACH")
- Maximum depth = 4 for interpretability

**Clustering Validation**:
- K-means clustering (k=4 for 4 classes)
- Check if clusters match manual labels
- Identify outliers and misclassified points

**Outlier Detection**:
- Isolation Forest
- Local Outlier Factor (LOF)
- Flag anomalous spectral profiles within each class

---

## Phase 3: Visualization Suite

### Objective
Make complex spectral patterns human-interpretable.

### 3.1 Per-Class Profiles

**Distribution Plots**:
- Box plots: All features by class (median, quartiles, outliers)
- Violin plots: Full distribution shapes (multimodal detection)
- Histograms: Frequency distributions with KDE overlay
- Q-Q plots: Check normality assumptions

**Multivariate Visualizations**:
- Heatmaps: Feature correlation matrices per class
- Radar charts: Multi-dimensional class signatures (6-8 key features)
- Parallel coordinates: High-dimensional patterns
- Scatter plot matrix: Pairwise feature relationships

**Color Coding**: Match LANDCOVER_COLORS from config.py for consistency

### 3.2 Transition Visualizations

**Spatial Profiles** (±20m around transitions):
- Waterfall plots: All 4 bands + derivatives stacked
- Line plots: Band evolution with shaded transition region
- Derivative profiles: 1st/2nd derivatives highlighting peaks
- Stacked area charts: Relative band contributions

**Comparative Analysis**:
- Overlay multiple transitions of same type
- Average profile ± confidence interval
- Identify consistent vs variable patterns
- Highlight discriminative regions

**Animated Transects**:
- Frame-by-frame spectral evolution
- Useful for presentations and debugging

### 3.3 Discriminability Visualizations

**Feature Importance**:
- Horizontal bar charts (sorted by importance)
- Grouped by feature type (raw bands, ratios, derivatives)
- Color-coded by statistical significance

**Classification Performance**:
- Confusion matrices for boundary types
- ROC curves with AUC values
- Precision-Recall curves
- F1-score vs threshold plots

**Separability Plots**:
- 2D scatter: Best pair of features for visual separation
- 3D scatter: Top 3 features with interactive rotation
- Decision boundaries from logistic regression

### 3.4 Dummy Spectra Generation

**Objective**: Create synthetic transects that match empirical statistics

**Method**:
1. Extract statistical parameters from real data:
   - Mean vectors (R, G, B, NIR per class)
   - Covariance matrices (preserve band correlations)
   - Derivative characteristics (transition sharpness)

2. Generate synthetic transect:
   - Start with random seed
   - Sample from multivariate normal distributions
   - Apply spatial smoothing (realistic autocorrelation)
   - Insert realistic transitions between zones

3. Validation:
   - Visual comparison to real transects
   - Statistical comparison (KS test, chi-square)
   - Expert review (does it "look right"?)

**Use Cases**:
- Algorithm testing without overfitting to specific transects
- Documentation and presentation
- Training data augmentation

---

## Phase 4: Algorithm Improvements

### Objective
Translate findings into actionable detector enhancements.

### 4.1 Shell Line Enhancement Strategy

**Current Hypothesis on Issues**:
1. **Gradual transitions**: Some shell lines are not sharp (derivative may be weak)
2. **Within-zone variability**: Dry beach can have small NIR variations that trigger false positives
3. **Multiple candidates**: Algorithm sees several weak transitions, unsure which is shell line
4. **Context ignorance**: Current detector doesn't use spatial context (expected location, neighboring zones)

**Proposed Solutions**:

#### A. Dual-Threshold Approach
- **Requirement 1**: NIR derivative magnitude (already implemented)
- **Requirement 2**: Absolute NIR drop (15+ units, already added in Phase 4)
- **Requirement 3**: NEW - Brightness context validation
  - Before transition: brightness > 195 (DRY_BEACH signature)
  - After transition: brightness < 175 (not dry beach anymore)

#### B. Relative Detection
Instead of absolute threshold, find **best candidate in expected region**:
- Expected shell line location: 40-120m from transect start (based on manual data)
- Rank all candidates in this region by composite score
- Select top-ranked candidate (even if below absolute threshold)

#### C. Composite Confidence Score
Replace binary pass/fail with graduated confidence:

```
confidence = base_score + bonuses - penalties

base_score (0.4-0.7):
  - NIR derivative magnitude (normalized)
  - Absolute NIR drop (normalized)

bonuses (0.0-0.3):
  + 0.10 if sustained (3+ consecutive points)
  + 0.10 if multi-band consensus (2+ bands)
  + 0.05 if perfect R/G ratio match (~1.0)
  + 0.05 if brightness context matches (DRY before, WET after)

penalties (0.0-0.3):
  - 0.10 if within-zone transition (same class before/after)
  - 0.15 if in high-variability region (std > 15)
  - 0.10 if multiple strong candidates nearby (ambiguous)
```

#### D. Contextual Validation
Use spatial and spectral context:
- **Brightness confirmation**: DRY zone should be very bright (>200) and uniform (std < 10)
- **Monotonic NIR trend**: NIR should generally decrease along transect
- **Transition uniqueness**: Only one major DRY→WET transition expected per transect
- **Geometric constraints**: Shell line unlikely in first 30m or last 50m

#### E. Ensemble Detection
Combine multiple weak signals:
- NIR derivative (primary)
- Brightness drop (secondary)
- NDWI increase (secondary)
- R/G ratio pattern (tertiary)
- Variability change (tertiary)

Vote or weighted average → higher confidence when multiple signals agree

### 4.2 Feature Engineering

**New Features to Implement**:

1. **Brightness gradient**: d(brightness)/dx
   - Sharp drop expected at shell line
   - Less sensitive to individual band noise

2. **Variability ratio**: std_after / std_before
   - DRY→WET should show increase in variability
   - Wet beach is more textured (water interaction)

3. **Multi-scale derivative consensus**:
   - Compute NIR derivative at 3m, 5m, 7m windows
   - Count how many scales show strong signal
   - True boundaries show signal across scales

4. **Spectral coherence index**:
   - Measure how well all bands agree on transition location
   - High coherence → high confidence
   - Low coherence → noisy/ambiguous region

5. **Contextual features**:
   - Distance from transect start (prior probability)
   - Mean NIR in previous 20m (expected DRY level)
   - Mean NIR in next 20m (expected WET level)

### 4.3 Decision Logic Refinement

**Hierarchical Detection Pipeline**:

```
Step 1: Candidate Identification (permissive)
  - Find all NIR derivative minima < -3.0 units/m
  - Require NIR > 50 (not in deep water)
  - Require distance in range [30m, 180m]
  → Result: List of 1-20 candidates per transect

Step 2: Candidate Scoring (composite)
  - Compute confidence score for each candidate
  - Use all available features (NIR, brightness, context, etc.)
  → Result: Ranked list with scores 0.0-1.0

Step 3: Contextual Filtering (intelligent)
  - Remove within-zone candidates (class doesn't change)
  - Remove high-variability false positives (std > 15)
  - Remove candidates in water zone (NIR < 60)
  → Result: Filtered list of plausible candidates

Step 4: Selection (choose best)
  - If any candidate has confidence > 0.85: select it
  - Else: select highest-confidence candidate in expected region [40-120m]
  - If no candidates remain: report "no shell line detected"
  → Result: Single boundary or none

Step 5: Refinement (optional)
  - Fine-tune location by finding exact derivative peak within ±3m
  - Adjust confidence based on neighborhood consistency
  → Result: Final boundary location + confidence
```

### 4.4 Adaptive Thresholds

**Environmental Adaptation**:
- Normalize for illumination (if metadata available)
- Account for tidal stage (if location data available)
- Adjust for sensor differences (if multiple image sources)

**Per-Transect Normalization** (if needed):
- Z-score normalization: (value - mean) / std
- Robust scaling: Use median and IQR instead of mean/std
- Only if cross-transect variability is high

**Bayesian Prior Probabilities**:
- Shell line more likely in 40-120m region (prior from manual data)
- Less likely near edges (edge effects)
- Incorporate prior into confidence calculation

---

## Phase 5: Implementation and Validation

### 5.1 Implementation Workflow

1. **Feature engineering** (Week 1):
   - Add new features to features.py
   - Compute on training data
   - Verify correctness (spot checks)

2. **Algorithm refinement** (Week 2):
   - Implement hierarchical detection pipeline in transition.py
   - Add composite confidence scoring
   - Add contextual validation

3. **Testing** (Week 2-3):
   - Run on training data (20 transects with manual labels)
   - Measure precision, recall, F1-score
   - Analyze errors (false positives, false negatives)

4. **Iteration** (Week 3):
   - Adjust weights in confidence formula
   - Tune thresholds based on ROC analysis
   - Repeat testing until metrics satisfactory

5. **Validation** (Week 4):
   - Run on held-out validation set (if available)
   - Compare Phase 6 vs Phase 5 performance
   - Document improvements

### 5.2 Success Metrics

**Primary Metric**: Shell line detection accuracy
- **Precision**: % of detected shell lines that are correct (±5m tolerance)
- **Recall**: % of true shell lines that are detected
- **F1-score**: Harmonic mean of precision and recall
- **Target**: F1 > 0.90 (currently unknown, baseline to be established)

**Secondary Metrics**:
- Median distance error (m) for true positives
- False positive rate (FPs per transect)
- Confidence calibration (does 0.8 confidence mean 80% accurate?)

**Qualitative Assessment**:
- Visual inspection of detected boundaries on sample transects
- Expert review (do results "make sense"?)
- Consistency across similar transects

### 5.3 Error Analysis

**False Positive Analysis**:
- Where do FPs occur? (within DRY zone, in water, at wave crests?)
- What features distinguish FPs from TPs?
- Can we add filters to eliminate specific FP types?

**False Negative Analysis**:
- Why were true shell lines missed? (too gradual, noisy, unusual spectra?)
- Are there alternative features that would detect them?
- Should we relax certain requirements?

**Confusion Matrix**:
- DRY→WET vs DRY→DRY (within-zone FP)
- DRY→WET vs WET→WATER (wrong boundary type)
- DRY→WET vs VEG→DRY (wrong location)

---

## Deliverables

### 1. Feature Analysis Folder Structure

```
feature_analysis/
├── IMPROVEMENT_PLAN.md (this document)
├── data/
│   ├── classified_manual_321197.csv
│   └── classified_manual_612823.csv
├── scripts/
│   ├── spectral_profiling.py        # Comprehensive feature computation
│   ├── transition_analysis.py       # Boundary-specific analysis
│   ├── visualization_suite.py       # Generate all plots
│   ├── statistical_tests.py         # Separability metrics, feature importance
│   ├── dummy_spectra_generator.py   # Synthetic transect generation
│   └── run_full_analysis.py         # Master script to run everything
├── outputs/
│   ├── statistics/
│   │   ├── class_statistics.csv
│   │   ├── transition_statistics.csv
│   │   ├── feature_importance.csv
│   │   └── separability_metrics.csv
│   ├── visualizations/
│   │   ├── class_profiles.png
│   │   ├── boundary_signatures.png
│   │   ├── feature_importance.png
│   │   ├── dummy_spectra_examples.png
│   │   └── roc_curves_by_boundary.png
│   └── reports/
│       └── SPECTRAL_CHARACTERIZATION_REPORT.md
```

### 2. Comprehensive Report

**SPECTRAL_CHARACTERIZATION_REPORT.md** will include:

#### Section 1: Landcover Class Characteristics
- Statistical profiles for each class (VEGETATED_DUNES, BEACH_DRY, BEACH_WET, WATER)
- Visual characteristics (brightness, color, texture)
- Empirical ranges for all features
- Discriminative features (what makes each class unique?)

#### Section 2: Transition Zone Signatures
- DRY→WET shell line signature (primary focus)
- WET→WATER waterline signature (secondary)
- VEG→DRY vegetation boundary signature (tertiary)
- Derivative profiles, transition widths, asymmetry

#### Section 3: Feature Importance Rankings
- Top 10 features for class discrimination
- Top 10 features for boundary detection
- Redundant features (high correlation, low added value)
- Recommended minimal feature set

#### Section 4: Dummy Spectra and Validation
- Synthetic transect examples
- Statistical validation (match real data distributions?)
- Visual validation (expert assessment)
- Use cases for testing and documentation

#### Section 5: Recommended Algorithm Improvements
- Specific changes to transition.py
- New features to add to features.py
- Updated thresholds for config.py
- Expected performance improvements

### 3. Updated Algorithm Implementation

**Modified Files**:
- `spectral_classifier/features.py`: New feature engineering
- `spectral_classifier/transition.py`: Enhanced shell line detector
- `spectral_classifier/config.py`: Updated thresholds and configuration

**New Capabilities**:
- Hierarchical detection pipeline
- Composite confidence scoring
- Contextual validation
- Ensemble approach

**Performance Metrics**:
- Baseline (Phase 5): TBD from validation run
- Target (Phase 6): F1 > 0.90 for shell line detection
- Stretch goal: F1 > 0.95

---

## Timeline

**Week 1: Analysis**
- Day 1-2: Set up infrastructure, run initial analyses
- Day 3-4: Comprehensive feature computation and statistics
- Day 5: Generate visualizations

**Week 2: Characterization**
- Day 1-2: Statistical testing and feature importance
- Day 3: Dummy spectra generation
- Day 4-5: Write characterization report

**Week 3: Implementation**
- Day 1-2: Feature engineering
- Day 3-4: Algorithm refinement
- Day 5: Initial testing

**Week 4: Validation and Iteration**
- Day 1-2: Comprehensive validation
- Day 3-4: Error analysis and refinement
- Day 5: Final documentation

---

## Success Criteria

### Minimum Viable Improvement
- **Precision** > 0.85 (few false positives)
- **Recall** > 0.85 (few missed detections)
- **Median error** < 3m (accurate localization)

### Target Performance
- **F1-score** > 0.90
- **Confidence calibration**: 90% of high-confidence (>0.8) detections are correct
- **Robustness**: Consistent performance across different transects

### Stretch Goals
- **F1-score** > 0.95
- **Zero false positives** in DRY_BEACH zones
- **Automated confidence thresholds** (adaptive)
- **Transferability**: Works on new datasets without retraining

---

## References

- Current implementation: `spectral_classifier/transition.py` (Phases 1-5)
- Training data: `training_output/seed_*/classified_manual_*.csv`
- Validation results: `validation/outputs_phase5/`
- Configuration: `spectral_classifier/config.py`

---

## Next Steps

1. ✅ Create folder structure
2. ⏳ Copy training data
3. ⏳ Implement spectral profiling script
4. ⏳ Run initial analysis
5. ⏳ Generate visualizations
6. ⏳ Write characterization report
7. ⏳ Implement algorithm improvements
8. ⏳ Validate and iterate

---

**Last Updated**: 2025-10-15
**Status**: Planning Complete → Implementation In Progress
