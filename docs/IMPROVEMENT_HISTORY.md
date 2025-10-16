# Beach Spectral Classifier - Improvement History

**Document Purpose:** Consolidated history of all algorithm improvements from initial development through Phase 6C

**Last Updated:** 2025-10-16

---

## Table of Contents

1. [Phase 1: Initial Validation Fixes](#phase-1-initial-validation-fixes)
2. [Phase 2: Boundary-Type-Specific Detection](#phase-2-boundary-type-specific-detection)
3. [Phase 3-4: Zone-Aware Filtering](#phase-3-4-zone-aware-filtering)
4. [Phase 5: Feature Analysis & Characterization](#phase-5-feature-analysis--characterization)
5. [Phase 6A/6B: Empirical Threshold Tightening](#phase-6ab-empirical-threshold-tightening)
6. [Phase 6C: Current Work - Threshold Tuning](#phase-6c-current-work---threshold-tuning)

---

## Phase 1: Initial Validation Fixes

**Date:** Early 2025
**Objective:** Reduce false positive rate in WATER zones
**Baseline Performance:** Precision 0.407, Recall 0.759, F1 0.530

### Problems Identified

1. **Water zone wave crests** - Highest crossing rate (0.629) causing transient false positives
2. **Vegetated dune structure** - High variability (31.44 NIR variability) causing false boundaries
3. **Transient spikes** - Not enough emphasis on sustained drops vs momentary crossings
4. **Single-band reliance** - Over-relying on NIR without multi-band confirmation

### Implemented Solutions

#### 1.1 Multi-Band Derivatives
**File:** `spectral_classifier/features.py`

Added derivative computation for all bands:
- `red_d1`, `green_d1`, `blue_d1`
- Smoothed versions: `red_d1_smooth`, `green_d1_smooth`, `blue_d1_smooth`
- Enables multi-band consensus validation

#### 1.2 Sustainability Requirement
**File:** `spectral_classifier/transition.py`

- Require 3+ consecutive points below threshold
- Filters transient crossings in WATER zones
- Based on observation: WATER zones have high total crossings but low sustained drops

#### 1.3 Multi-Band Consensus
**File:** `spectral_classifier/transition.py`

- Require agreement from 2+ spectral bands
- Adjusts confidence based on number of agreeing bands
- High precision improvement with minimal recall loss

#### 1.4 NIR Derivative Threshold Update
**File:** `spectral_classifier/config.py`

- Updated threshold: -3.0 → **-4.38** (optimal from sensitivity analysis)
- Added config parameters:
  - `nir_derivative_threshold`: -4.38
  - `derivative_sustainability_points`: 3
  - `multi_band_consensus_required`: 2

### Results
- **Precision:** 0.407 → 0.522 (+28%)
- **Recall:** 0.759 → 0.400 (-47% - too strict!)
- **F1-Score:** 0.530 → 0.453 (-15%)

**Conclusion:** Improved precision but recall dropped too much. Filters too strict.

---

## Phase 2: Boundary-Type-Specific Detection

**Date:** Early-Mid 2025
**Objective:** Address poor recall (40%) by using specialized detection per boundary type
**Problem:** Different boundaries have fundamentally different spectral signatures

### Key Insight

**Treating all boundaries the same is the fundamental problem.**

Different boundary types require different detection approaches:
- **VEG_DUNES → BEACH_DRY:** Inflection point (curvature/second derivative)
- **BEACH_WET → WATER:** RGB foam bump + NIR confirmation
- **BEACH_DRY → BEACH_WET:** NIR derivative drop (original method works!)

### Miss Rates Before Phase 2
- **VEG_DUNES→BEACH_DRY:** 80% missed (8/10)
- **BEACH_WET→WATER:** 70% missed (7/10)
- **BEACH_DRY→BEACH_WET:** 30% missed (3/10)

### Implemented Solutions

#### 2.1 Second Derivative (Curvature) Features
**File:** `spectral_classifier/features.py`

Added second derivatives for all bands:
```python
def _compute_nir_derivative_second(self, window_size=5):
    # Second derivative = curvature
    # Critical for VEG_DUNES boundaries (inflection points)
```

Features added: `nir_d2`, `red_d2`, `green_d2`, `blue_d2`
Multiple scales: `nir_d2_w5`, `nir_d2_w7`, `nir_d2_w9`

#### 2.2 RGB Foam Peak Detection
**File:** `spectral_classifier/features.py`

```python
def _detect_rgb_foam_peaks(self):
    # Detect RGB peaks indicating foam from breaking surf
    # Critical for BEACH_WET→WATER boundaries
```

Uses `scipy.signal.find_peaks` with prominence and width requirements.

#### 2.3 Multi-Scale Derivative Computation
**File:** `spectral_classifier/features.py`

Compute derivatives at multiple smoothing scales: windows 3, 5, 7, 9
- VEG boundaries: window 7-9 (coarse smoothing for vegetation noise)
- Surf zone: window 7-11 (coarse for wave patterns)
- Dry/wet: window 5-9 (medium scale)

#### 2.4 Specialized Boundary Detectors
**File:** `spectral_classifier/transition.py`

**Method 1: VEG_DUNES→BEACH_DRY**
```python
def _detect_vegetation_boundaries(self, features):
    # Inflection point detection using second derivative zero-crossings
    # Optimal smoothing: window 7-9
```

**Method 2: BEACH_WET→WATER**
```python
def _detect_surf_zone_boundaries(self, features):
    # RGB foam peak + NIR drop combination
    # Optimal smoothing: window 7-11
```

**Method 3: BEACH_DRY→BEACH_WET**
```python
def _detect_dry_wet_boundaries(self, features):
    # Derivative magnitude (original approach - works well!)
    # Optimal smoothing: window 5-9
    # Applies sustainability + multi-band consensus (Phase 1 filters)
```

#### 2.5 Main Detection Logic Update
```python
def find_transitions(self, features, landcover):
    all_transitions = []

    # Use specialized detectors
    veg_transitions = self._detect_vegetation_boundaries(features)
    surf_transitions = self._detect_surf_zone_boundaries(features)
    dw_transitions = self._detect_dry_wet_boundaries(features)

    all_transitions.extend([...])

    # Merge nearby detections (within 5m)
    # Filter and validate
```

#### 2.6 Convert Hard Filters to Confidence Modifiers
For `_detect_dry_wet_boundaries()` only:
- Base confidence: 0.50 + (magnitude - 3.0) / 15.0
- Bonuses:
  - +0.15 if sustained (3+ points)
  - +0.15 if 2+ bands agree
  - +0.25 if 3+ bands agree
- Minimum confidence threshold: 0.60 for final filtering

### Expected Results
- **Precision:** 0.522 → 0.55-0.65
- **Recall:** 0.400 → 0.70-0.80 (+30-40% improvement!)
- **F1-Score:** 0.453 → 0.60-0.70

---

## Phase 3-4: Zone-Aware Filtering

**Date:** Mid 2025
**Objective:** Reduce false positives by applying zone-specific rules
**Problem:** 52.5% of false positives occur in WATER zones

### Implemented Solutions

#### 3.1 Zone-Aware Filtering Logic
**File:** `spectral_classifier/transition.py`

```python
def _is_in_water_zone(self, index, landcover, look_window=5):
    # Check if index is within or near WATER zone

def _filter_transitions(self, transitions, features, landcover):
    # For transitions in WATER zones:
    # - Require higher confidence (0.75 instead of 0.60)
    # - OR require stronger sustainability (5+ points instead of 3)
    # - OR skip if not adjacent to BEACH_WET zone
```

#### 3.2 Zone-Specific Thresholds (Config)
**File:** `spectral_classifier/config.py`

```python
'zone_thresholds': {
    'WATER': -6.0,        # Stricter (more false positives)
    'BEACH_WET': -4.38,   # Optimal threshold
    'DRY_BEACH': -4.38,   # Standard threshold
    'VEG_DUNES': -5.0,    # Slightly stricter (high variability)
}
```

### Results
Successfully reduced WATER zone false positives while maintaining performance in beach zones.

---

## Phase 5: Feature Analysis & Characterization

**Date:** Mid-Late 2025
**Objective:** Systematic, data-driven analysis of spectral signatures
**Approach:** Exhaustive feature analysis → empirical characterization → algorithm refinement

### Data Sources
- 20 manually classified transects (10 per seed)
- `classified_manual_321197.csv`, `classified_manual_612823.csv`
- Corresponding spectral data with R/G/B/NIR values

### Analysis Conducted

#### 5.1 Point-by-Point Spectral Analysis
For each landcover class (with 5m boundary tolerance):
- Raw band statistics (mean, std, min, max, quartiles)
- Band ratios (all pairwise combinations)
- Spectral indices (NDVI, NDWI, brightness, variability)
- First derivatives (rate of change)
- Second derivatives (curvature)
- Higher-order features (spectral angle, cross-band correlation)

#### 5.2 Rolling-Window Analysis
Window sizes: 3m, 5m, 7m, 9m, 11m
- Moving averages, std dev, local minima/maxima
- Trend classification, autocorrelation
- Multi-scale analysis (scale-invariant vs scale-dependent features)

#### 5.3 Transition Zone Analysis
Region: ±10m around manually labeled boundaries
- Pre/post differences
- Transition sharpness, width, asymmetry
- Multi-band synchrony
- Confidence indicators

### Key Findings

**BEACH_DRY Signature:**
- Brightness: 207 ± 9 (very bright, uniform)
- Variability: 3 ± 4 (extremely low)
- NIR ratio: 0.83 ± 0.07
- R/G ratio: ~1.0 ± 0.04 (red ≈ green, neutral sand)

**BEACH_WET Signature:**
- Brightness: 163 ± 21 (darker, more variable)
- Variability: 8 ± 5 (moderate texture)
- NIR ratio: 0.54 ± 0.14
- R/G ratio: >1.0 (red > green)

**Shell Line (DRY→WET) Transition:**
- NIR drop: **55.1 ± 13.6 units** (most reliable)
- Derivative peak: **-13.8 ± 4.0 units/m** (sharp, sustained)
- Sustained drop: **9.1 ± 3.1 points** (longest of all boundary types)
- Brightness before: **201.8 ± 9.6 units** (very bright sand)
- NIR before: **171.6 ± 18.2 units** (dry beach level)
- Variability ratio: **5.87 ± 5.53** (smooth→textured, unique signature)

### Deliverables
- `feature_analysis/outputs/statistics/class_spectral_profiles.csv`
- `feature_analysis/outputs/statistics/transition_characteristics.csv`
- `feature_analysis/outputs/visualizations/` (class profiles, boundary signatures, etc.)
- `feature_analysis/README.md` (comprehensive analysis summary)

---

## Phase 6A/6B: Empirical Threshold Tightening

**Date:** Late 2025 (October)
**Objective:** Use Phase 5 empirical findings to improve shell line detection
**Focus:** BEACH_DRY → BEACH_WET transition accuracy

### Problems from Phase 5
1. **NIR derivative threshold too permissive** (-4.38 vs empirical -13.8 ± 4.0)
2. **No absolute NIR drop requirement** (empirical: 55 ± 14 units)
3. **No context validation** (brightness/NIR before boundary)
4. **No variability change check** (empirical: 5.9x increase)
5. **Binary pass/fail** instead of confidence scoring

### Phase 6A Implementation

#### 6A.1 Strengthen Detection Criteria
**File:** `spectral_classifier/transition.py` (lines 890-1020)

Updated `_detect_dry_wet_boundaries()`:

```python
# NEW THRESHOLDS (based on empirical data)
'dry_wet': {
    'nir_threshold': -8.0,              # TIGHTENED from -4.0
    'min_nir_drop_absolute': 40.0,      # NEW: Absolute drop requirement
    'brightness_before_min': 190,       # NEW: Context check
    'nir_before_min': 160,              # NEW: Context check
    'variability_ratio_min': 2.0,       # NEW: Smooth→rough check
    'expected_location_min': 40,        # NEW: Spatial prior
    'expected_location_max': 120,       # NEW: Spatial prior
}
```

Added validation checks:
1. **Absolute NIR drop:** nir_before - nir_at ≥ 40 units
2. **Brightness context:** brightness_before ≥ 190 (dry beach)
3. **NIR context:** nir_before ≥ 160 (dry beach level)
4. **Variability ratio:** var_after / var_before ≥ 2.0 (smooth→rough)
5. **Expected location:** 40m ≤ distance ≤ 120m (typical shell line zone)

#### 6A.2 Composite Confidence Scoring
Multi-criteria scoring with bonuses:

```python
base_confidence = 0.50 + (magnitude - 8.0) / 30.0  # [0.50, 0.70]

# Bonuses (up to +0.55 total):
if magnitude > 12.0: confidence += 0.10     # Strong derivative
if nir_drop_abs > 50: confidence += 0.08    # Large NIR drop
if brightness_drop > 25: confidence += 0.05 # Large brightness drop
if var_ratio > 4.0: confidence += 0.07      # High variability ratio
if is_sustained: confidence += 0.08         # Sustained drop
if num_bands >= 3: confidence += 0.08       # Multi-band consensus
if 0.98 <= rg_ratio <= 1.08: confidence += 0.04  # Shell line color
if in_expected_zone: confidence += 0.05     # Spatial prior

confidence = min(confidence, 0.95)  # Cap at 0.95
```

### Phase 6B Implementation

#### 6B.1 Relative Candidate Ranking
**File:** `spectral_classifier/transition.py` (lines 1137-1202)

```python
def _select_best_shell_line_candidate(self, candidates, features):
    """
    Select single best shell line from dry/wet boundary detections.

    Strategy:
    1. Filter to expected location range (40-120m)
    2. Rank by composite confidence score
    3. Select highest-scoring candidate
    4. Only accept if confidence > 0.60 (relaxed from filtering threshold)
    """
```

Ensures:
- **Uniqueness:** Only one shell line per transect
- **Relative ranking:** Best candidate selected even if weak
- **Environmental adaptability:** Handles variability across transects

#### 6B.2 Enhanced Logging
Added structured logging with rejection reasons:
```python
rejection_reasons = []
if nir_drop_abs < min_nir_drop_abs:
    rejection_reasons.append(f"NIR_drop={nir_drop_abs:.1f}<{min_nir_drop_abs}")
# ... other checks ...
logger.debug(f"Rejected at {distance:.1f}m: {', '.join(rejection_reasons)}")
```

### Phase 6A/6B Results

**Target Metrics:**
- Precision: ≥ 0.90
- Recall: ≥ 0.90
- F1-score: ≥ 0.90
- Median error: < 3m

**Actual Results:**
- **Problem:** Detection rate too low (0/10 or 2/10 transects)
- **Issue:** Thresholds too strict, rejecting valid shell lines
- **Root cause:** Empirical data (n=20) may not capture full environmental variation

---

## Phase 6C: Current Work - Threshold Tuning

**Date:** October 16, 2025 (ACTIVE)
**Status:** Under Development
**Objective:** Find optimal balance between precision and recall

### Problem Analysis

From `training_output/training_data.log` (seed 321197, 10 transects):
- **Detection rate:** 4/10 transects (40%)
- **Target:** ≥85% detection rate

**Example rejections (Transect 100):**
- 50m: `NIR_drop=6.6<40, var_ratio=1.25<2.0` - VEG→DRY (correctly rejected)
- 116m: `NIR_drop=19.8<40, brightness=185.6<190, NIR_before=140.8<160` - **Likely shell line but failed thresholds**

**Critical insight:** Phase 6A/6B thresholds (NIR ≥40, brightness ≥190, NIR before ≥160) are too strict for real data.

### Diagnostic Plan

#### 6C.1 Threshold Testing
Test multiple threshold sets:

| Set | NIR Drop | Brightness Before | NIR Before | Description |
|-----|----------|-------------------|------------|-------------|
| **STRICT** | ≥40 | ≥190 | ≥160 | Phase 6A/6B (current) |
| **MODERATE** | ≥30 | ≥180 | ≥150 | 25% relaxed |
| **RELAXED** | ≥20 | ≥170 | ≥140 | 50% relaxed |

For each set:
1. Run on test transects (10-20 transects)
2. Compare to manual labels
3. Compute precision, recall, F1
4. Select optimal configuration

#### 6C.2 VEG→DRY Discrimination
**Problem:** VEG→DRY boundaries also have sharp NIR drops, can be mistaken for shell lines

**Solution:** Location-based penalty
```python
# VEG zone typically 40-70m from transect start
# Shell line typically 90-120m

veg_zone_min, veg_zone_max = 40, 70
if veg_zone_min <= dist <= veg_zone_max:
    if nir_drop_abs < 50:  # Shell lines typically >50
        confidence -= 0.20  # Heavy penalty
```

#### 6C.3 Fallback Detection Mode
**Strategy:** If no candidates pass strict thresholds, use relaxed criteria

```python
def _select_best_shell_line_candidate(self, candidates, features):
    min_confidence_strict = 0.75
    min_confidence_fallback = 0.50

    # Try strict filtering first
    if best_candidate and confidence >= min_confidence_strict:
        return [best_candidate]

    # FALLBACK: Relax thresholds and try again
    relaxed_candidates = self._detect_with_relaxed_thresholds(features)
    # ... select best from relaxed pool
```

#### 6C.4 Validation Script
**File:** `validation/validate_phase6.py`

```python
def validate_transect(transect_id, spectral_data, manual_location):
    # Run detector
    # Compare to manual label
    # Return: CORRECT, LOCALIZATION_ERROR, FALSE_POSITIVE, FALSE_NEGATIVE

def main():
    # Run on all 20 manually classified transects
    # Compute metrics
    # Generate report
```

### Phase 6C Goals

**Performance Targets:**
- **Precision:** ≥ 0.85 (85% of detections correct within ±5m)
- **Recall:** ≥ 0.85 (detect 85% of true shell lines)
- **F1-Score:** ≥ 0.85
- **Median Error:** < 5m
- **Zero VEG→DRY false positives**

**Deliverables:**
1. Tuned thresholds in `config.py` with justification
2. Validation report: `validation/phase6c_results.csv`
3. Performance comparison (Phase 5 → 6A/6B → 6C)
4. Diagnostic plots

### Next Steps

1. **Threshold tuning** (In Progress)
   - Test MODERATE and RELAXED threshold sets
   - Compute precision-recall curves
   - Select optimal configuration

2. **Validation against manual labels**
   - Run on 20 manually classified transects
   - Analyze failure modes
   - Iterate if needed

3. **Fallback mode enhancement**
   - Implement graceful degradation
   - Test on difficult transects

4. **Documentation & reporting**
   - Document threshold selection rationale
   - Create performance comparison report

---

## Summary of Key Improvements

### What Works Well
1. **Boundary-type-specific detection** (Phase 2)
   - VEG boundaries: Inflection points
   - Surf zone: RGB foam peaks
   - Dry/wet: NIR derivative magnitude

2. **Multi-band consensus** (Phase 1)
   - Reduces false positives
   - Improves confidence calibration

3. **Zone-aware filtering** (Phase 3-4)
   - Targets WATER zone false positives
   - Maintains performance in beach zones

4. **Empirical characterization** (Phase 5)
   - Data-driven threshold selection
   - Quantified landcover signatures
   - Identified discriminative features

### Current Challenges
1. **Threshold brittleness** (Phase 6A/6B → 6C)
   - Strict thresholds → low recall
   - Relaxed thresholds → potential FP increase
   - Need optimal balance

2. **Environmental variation**
   - Empirical data (n=20) may not capture full range
   - Need adaptive/robust thresholds

3. **VEG→DRY confusion**
   - Similar NIR signatures to shell line
   - Requires spatial context to disambiguate

### Lessons Learned
1. **One size doesn't fit all:** Different boundaries need different detectors
2. **Hard filters are brittle:** Soft confidence modifiers work better
3. **Context matters:** Spatial and spectral context improves accuracy
4. **Relative ranking helps:** Best candidate selection for difficult cases
5. **Validation is critical:** Empirical data reveals gaps in assumptions

---

## References

### Code Files
- `spectral_classifier/transition.py` - Boundary detection logic
- `spectral_classifier/features.py` - Feature engineering
- `spectral_classifier/classifier.py` - Landcover classification
- `spectral_classifier/config.py` - Thresholds and parameters

### Data Files
- `feature_analysis/outputs/statistics/transition_characteristics.csv`
- `feature_analysis/data/classified_manual_*.csv`
- `training_output/training_data.log`

### Documentation
- `feature_analysis/README.md` - Analysis methodology
- `docs/PHASE_6_IMPLEMENTATION.md` - Current phase details
- `PROJECT_STATUS.md` - Current state summary

---

**Document Status:** Active
**Next Update:** After Phase 6C validation results
