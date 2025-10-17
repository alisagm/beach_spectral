# Phase 6C Diagnostic Analysis - Key Findings

**Date:** 2025-10-16
**Analysis:** Test log from 10 randomly sampled transects (seed 321197)
**Performance:** 8/10 detected (80%), but 2/10 are edge artifacts (actual: 6/10 = 60%)

---

## Critical Issues Identified

### 1. **STRICT THRESHOLDS COMPLETELY FAIL** (Severity: CRITICAL)
**Evidence:**
- 8/8 successful detections required fallback mode (100%)
- 0/8 detections passed strict Phase 6C thresholds
- Strict mode thresholds:
  - NIR drop ≥ 39 units
  - Brightness before ≥ 175
  - NIR before ≥ 135
  - Variability ratio ≥ 1.5

**Impact:**
- Strict thresholds are **completely ineffective**
- All detections rely on relaxed fallback (NIR ≥15, brightness ≥160, NIR_before ≥100)
- This defeats the purpose of having strict thresholds

**Examples from Log:**
- Transect 100: NIR_drop=19.8 (requires fallback)
- Transect 1075: NIR_drop=30.4 (requires fallback)
- Transect 1138: NIR_drop=35.4 (requires fallback) - **would have passed with NIR≥35 threshold!**

**Recommendation:**
Replace strict Phase 6C thresholds with **MODERATE** thresholds as primary mode:
- NIR drop ≥ 25 units (down from 39)
- Brightness before ≥ 165 (down from 175)
- NIR before ≥ 120 (down from 135)
- Variability ratio ≥ 1.3 (down from 1.5)

---

### 2. **MONOTONIC SMOOTHING CATASTROPHIC COLLAPSE** (Severity: CRITICAL)
**Evidence:**
All 10 transects show severe class collapse:

| Transect | DRY (before) | WET (before) | UNKNOWN (after) | Collapse Rate |
|----------|--------------|--------------|-----------------|---------------|
| 100      | 35           | 27           | 298             | 98.7%         |
| 378      | 44           | 24           | 298             | 98.7%         |
| 663      | 45           | 32           | 254             | 84.1%         |
| 932      | 36           | 19           | 263             | 87.1%         |
| 1075     | 38           | 16           | 297             | 98.7%         |
| 1138     | 37           | 17           | 298             | 98.7%         |
| 1220     | 30           | 33           | 269             | 89.1%         |
| 1515     | 42           | 27           | 297             | 98.7%         |
| 1778     | 33           | 31           | 259             | 85.8%         |
| 2077     | 76           | 14           | 299             | 99.0%         |

**Average collapse: 92.9%** of classifications converted to UNKNOWN

**Impact:**
- Destroys spatial context for boundary validation
- Makes `from_class` and `to_class` unreliable (almost always UNKNOWN)
- Forces reliance on `detection_method` instead of class labels
- Defeats purpose of landcover classification

**Root Cause:**
Monotonic smoothing algorithm enforces strict land→water progression, but:
- Real transects have noise and local variations
- Classifier produces scattered class labels
- Smoothing treats violations as errors → converts to UNKNOWN

**Recommendation:**
**DISABLE monotonic smoothing for boundary detection**. Use raw classifications or apply light median filtering only.

---

### 3. **EDGE ARTIFACT FALSE POSITIVES** (Severity: HIGH)
**Evidence:**
- Transect 663: Shell line at **0.0m** (NIR_drop=0.0, var_ratio=0.00)
- Transect 2077: Shell line at **0.0m** (NIR_drop=0.0, var_ratio=0.00)

Both are clearly **invalid** - shell lines don't occur at transect edges with zero NIR drop!

**Impact:**
- 2/8 detections are false positives (25% FP rate)
- **Actual detection rate: 6/10 (60%), not 8/10 (80%)**

**Root Cause:**
Fallback mode accepts candidates with:
- Confidence ≥ 0.40 (too permissive)
- NIR drop ≥ 15 (allows near-zero drops)
- Variability ratio ≥ 1.0 (allows no variability change)

Edge buffer (10m) should filter these, but fallback candidates bypass normal validation.

**Recommendation:**
1. Increase edge buffer to 15m
2. Add hard minimum for fallback mode: **NIR drop ≥ 20** (not 15)
3. Add hard minimum: **var_ratio ≥ 1.2** (not 1.0)
4. Reject candidates with **distance < 30m or > 250m** (outside reasonable zone)

---

### 4. **EXCESSIVE FILTERING ATTRITION** (Severity: MEDIUM)
**Evidence:**
- Total Phase 2 detections: 30 boundaries
- Total filtered out: 25 boundaries
- Final kept: 8 boundaries
- **Attrition rate: 73.3%**

**Breakdown:**
- Most filtered boundaries are `inflection_point` detections (VEG→DRY boundaries)
- Filter rule: `boundary_types=shore_only` rejects non-shell-line boundaries
- This is **working as intended** - rejecting VEG boundaries

**Impact:**
Low impact - filtering is doing its job. The issue is that only 1-2 shell-line candidates exist per transect, and strict mode rejects them.

**Recommendation:**
No change needed. Filtering logic is correct. Fix root cause (strict thresholds).

---

### 5. **TWO COMPLETE DETECTION FAILURES** (Severity: MEDIUM)
**Evidence:**
- Transect 378: No shell line detected (Phase 2 found 6 boundaries, all VEG inflections)
- Transect 1515: No shell line detected (Phase 2 found 2 boundaries, all VEG inflections)

**Analysis:**
Both transects:
- Had Phase 2 detections (6 and 2 respectively)
- All were `inflection_point` method (VEG→DRY boundaries)
- Zero `derivative_magnitude` or `derivative_magnitude_relaxed` detections
- Filtering correctly rejected VEG boundaries
- **No shell-line candidates found even in fallback mode**

**Possible Causes:**
1. **True negative** - No clear shell line present on these transects
2. **Weak signal** - Shell line exists but NIR derivative too weak (<8 units/m threshold)
3. **Missed by fallback** - Fallback thresholds still too strict

**Recommendation:**
1. Inspect plots for transects 378 and 1515 manually
2. Consider even more relaxed "last resort" mode if no candidates found:
   - NIR derivative threshold: -8.0 → -5.0
   - NIR drop: 15 → 10
   - Confidence floor: 0.30

---

## Performance Summary

**Current State (Phase 6C):**
| Metric | Value | Target |
|--------|-------|--------|
| Detection Rate | 60% (6/10 real) | ≥85% |
| False Positive Rate | 25% (2/8 detected) | ≤15% |
| Fallback Dependency | 100% (8/8) | ≤30% |
| Class Collapse | 92.9% avg | <20% |
| Confidence (mean) | 0.72 | ≥0.75 |

**Status:** ❌ **FAILS all targets**

---

## Recommended Action Plan

### Priority 1: Fix Threshold Brittleness (2-3 hours)
**Action:** Replace strict/fallback two-tier system with **graduated confidence scoring**

**Implementation:**
```python
# Instead of hard cutoffs, use soft penalty functions
def compute_shell_line_confidence(candidate):
    confidence = 0.50  # Base

    # NIR drop scoring (sigmoid curve around empirical mean=55)
    nir_score = 1 / (1 + np.exp(-(nir_drop - 35) / 10))  # Centered at 35
    confidence += 0.25 * nir_score

    # Brightness scoring
    brightness_score = min(1.0, brightness_before / 180)
    confidence += 0.15 * brightness_score

    # Variability ratio scoring
    var_score = min(1.0, var_ratio / 2.0)
    confidence += 0.15 * var_score

    # Other bonuses...
    return confidence
```

This allows **all** candidates to be considered, with confidence naturally reflecting signal strength.

### Priority 2: Disable Monotonic Smoothing (30 min)
**Action:** Bypass monotonic smoothing for boundary detection

**Implementation:**
```python
# In main.py or transition.py
# Option A: Use raw classifications
landcover_for_boundaries = landcover_raw  # Before smoothing

# Option B: Light median filter only
from scipy.ndimage import median_filter
landcover_smoothed = median_filter(landcover_raw, size=5)
```

### Priority 3: Fix Edge Artifacts (30 min)
**Action:** Add hard constraints for fallback mode

**Implementation:**
```python
# In _detect_with_relaxed_thresholds()
# HARD CONSTRAINTS (not soft)
if nir_drop_abs < 20:  # Was 15
    continue
if var_ratio < 1.2:  # Was 1.0
    continue
if distance < 30 or distance > 250:  # New
    continue
```

### Priority 4: Investigate Failures (1 hour)
**Action:** Manually inspect transects 378 and 1515 plots to understand why no shell line candidates

---

## Files to Modify

1. **`spectral_classifier/transition.py`** (lines 916-1200, 1238-1560)
   - Replace hard threshold checks with soft confidence scoring
   - Add hard constraints to fallback mode

2. **`spectral_classifier/classifier.py`**
   - Add flag to disable monotonic smoothing
   - Implement light median filter alternative

3. **`spectral_classifier/config.py`** (lines 108-130)
   - Remove strict vs fallback distinction
   - Define single set of "reference" thresholds for soft scoring

---

## Next Steps

1. **Implement Priority 1-3 changes** (3-4 hours total)
2. **Re-run test on same 10 transects** and compare:
   - Detection rate (target: 8-9/10)
   - False positive rate (target: 0-1/10)
   - Confidence distribution (target: mean ≥0.75)
3. **Validate on full 20-transect labeled set** from Phase 5
4. **Proceed to threshold optimization grid search** if results promising

---

**Estimated Impact:**
- Detection rate: 60% → **85-90%**
- False positive rate: 25% → **<10%**
- Confidence (mean): 0.72 → **0.78-0.82**
