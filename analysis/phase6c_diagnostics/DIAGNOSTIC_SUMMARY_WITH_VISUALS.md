# Phase 6C Shell-Line Detection - Diagnostic Analysis Summary

**Analysis Date:** 2025-10-16
**Dataset:** 10 randomly sampled transects (seed 321197)
**Current Performance:** 6/10 valid detections (60%), 2/10 edge artifacts (false positives)

---

## Executive Summary

The diagnostic analysis reveals **three critical failures** in the current Phase 6C implementation:

1. **Strict thresholds completely ineffective** - 100% of detections require fallback mode
2. **Monotonic smoothing catastrophic** - Collapses 92.9% of classifications to UNKNOWN
3. **Edge artifacts** - 25% false positive rate from 0.0m detections

**Overall Assessment:** Current system **FAILS all performance targets**

---

## Visualizations Generated

The diagnostic analysis produced the following visualizations (see `/analysis/phase6c_diagnostics/` directory):

### 1. **detection_overview.png**
Four-panel overview showing:
- **Detection Success Rate:** 8/10 detected (80.0%), but includes 2 edge artifacts
- **Detection Mode Distribution:** 100% fallback mode (strict mode completely failed)
- **Confidence Distribution:** Mean 0.72, range 0.62-0.79 (below 0.75 target)
- **NIR Drop vs Variability Scatter:** Shows most detections have NIR <40 and var_ratio <2.0

**Key Insight:** All detections cluster in "weak signal" region, confirming strict thresholds are too harsh.

### 2. **pipeline_progression.png**
Bar chart showing detection pipeline attrition per transect:
- **Phase 2 Detected:** 1-6 boundaries per transect (total 30)
- **Filtered Out:** Most boundaries (total 25) rejected by `shore_only` filter
- **Final Detection:** 0-2 per transect (total 8, including 2 invalid)

**Key Insight:** Filtering is working correctly - the issue is upstream (detection stage).

### 3. **detection_results.csv**
Tabular data for each transect:

| transect_id | detected | mode     | distance | confidence | nir_drop | var_ratio | phase2_boundaries | filtered |
|-------------|----------|----------|----------|------------|----------|-----------|-------------------|----------|
| 100         | True     | fallback | 116.0    | 0.62       | 19.8     | 1.47      | 3                 | 3        |
| 378         | False    | -        | -        | -          | -        | -         | 6                 | 6        |
| 663         | True     | fallback | 0.0      | 0.68       | 0.0      | 0.00      | 3                 | 3        |
| 932         | True     | fallback | 113.0    | 0.79       | 53.0     | 1.43      | 4                 | 3        |
| 1075        | True     | fallback | 102.0    | 0.76       | 30.4     | 2.28      | 2                 | 1        |
| 1138        | True     | fallback | 100.0    | 0.73       | 35.4     | 1.91      | 3                 | 1        |
| 1220        | True     | fallback | 249.0    | 0.74       | 56.2     | 1.25      | 3                 | 3        |
| 1515        | False    | -        | -        | -          | -        | -         | 2                 | 2        |
| 1778        | True     | fallback | 117.0    | 0.74       | 34.8     | 1.55      | 3                 | 2        |
| 2077        | True     | fallback | 0.0      | 0.68       | 0.0      | 0.00      | 1                 | 1        |

**Observations:**
- Transects 663 and 2077 have NIR_drop=0.0, var_ratio=0.00 → **FALSE POSITIVES**
- Transects 378 and 1515 had no shell-line candidates → **FALSE NEGATIVES**
- Valid detections (6/10) have NIR drops 19.8-56.2, highly variable

---

## Critical Issue #1: Strict Threshold Failure

### Problem
**Zero detections (0/8) passed strict Phase 6C thresholds.** 100% rely on fallback mode.

### Evidence
Strict thresholds from `config.py`:
```python
'dry_wet': {
    'min_nir_drop_absolute': 39.0,     # FAILS on 6/8 valid detections
    'brightness_before_min': 175,      # Moderately strict
    'nir_before_min': 135,             # Moderately strict
    'variability_ratio_min': 1.5,      # FAILS on 4/8 valid detections
}
```

**Actual data from detections:**
- NIR drop range: 19.8 - 56.2 (median: 33.1)
  - 4/6 valid detections have NIR < 39 → **Would be rejected by strict mode!**
- Variability ratio range: 1.25 - 2.28 (median: 1.51)
  - 3/6 valid detections have var_ratio < 1.5 → **Would be rejected by strict mode!**

### Root Cause
Phase 5 empirical data (n=20 transects) was **selection-biased toward strong signals**:
- Empirical mean NIR drop: 55.1 ± 13.6
- Real-world median: 33.1 (40% lower!)
- Phase 6A set threshold at 5th percentile (39), but this is still too strict

### Recommendation
**Replace hard thresholds with soft confidence scoring.** See Priority 1 in KEY_FINDINGS.md.

---

## Critical Issue #2: Monotonic Smoothing Collapse

### Problem
Monotonic smoothing converts **92.9%** of classifications to UNKNOWN (average across 10 transects).

### Evidence

**Example: Transect 100**
```
Before smoothing:
  WATER:      170 points (56.3%)
  DRY_BEACH:   35 points (11.6%)
  BEACH_WET:   27 points (8.9%)
  UNKNOWN:     65 points (21.5%)
  VEG_DUNES:    5 points (1.7%)

After smoothing:
  UNKNOWN:    298 points (98.7%)  ← Collapsed!
  VEG_DUNES:    1 point  (0.3%)
  DRY_BEACH:    1 point  (0.3%)
  BEACH_WET:    1 point  (0.3%)
  WATER:        1 point  (0.3%)
```

**This pattern repeats on ALL 10 transects.**

### Impact
1. **Destroys spatial context** for boundary validation
2. **from_class** and **to_class** fields become useless (always UNKNOWN)
3. Forces system to rely on `detection_method` instead of class labels
4. Defeats the entire purpose of landcover classification

### Root Cause
Monotonic smoothing enforces strict land→water monotonic progression (VEG → DRY → WET → WATER). When this progression is violated (due to classifier noise or real terrain variations), points are marked UNKNOWN.

**The algorithm is too strict** - real transects aren't perfectly monotonic.

### Recommendation
**Disable monotonic smoothing for boundary detection.** Use raw classifications or apply light median filtering only (window=5 points). See Priority 2 in KEY_FINDINGS.md.

---

## Critical Issue #3: Edge Artifact False Positives

### Problem
2/8 detections occur at **distance=0.0m** with **NIR_drop=0.0** and **var_ratio=0.00**. These are clearly invalid.

### Evidence
- **Transect 663:** Shell line at 0.0m (fallback mode, conf=0.68)
- **Transect 2077:** Shell line at 0.0m (fallback mode, conf=0.68)

Both have identical characteristics suggesting a systematic bug in fallback mode.

### Root Cause
Fallback mode (`_detect_with_relaxed_thresholds()`) has overly permissive thresholds:
```python
min_nir_drop_abs = 15.0   # Allows near-zero drops
var_ratio_min = 1.0       # Allows no variability change
confidence >= 0.40        # Very low acceptance threshold
```

Edge buffer (10m) should filter these, but appears to fail for fallback candidates.

### Recommendation
1. Increase edge buffer from 10m to **15m**
2. Add **hard minimums** for fallback mode:
   - NIR drop ≥ 20 (not 15)
   - Variability ratio ≥ 1.2 (not 1.0)
   - Distance ≥ 30m AND ≤ 250m
3. See Priority 3 in KEY_FINDINGS.md

---

## Performance Metrics

### Current State (Phase 6C)
| Metric                  | Value        | Target  | Status |
|-------------------------|--------------|---------|--------|
| **Detection Rate**      | 60% (6/10)   | ≥85%    | ❌ FAIL |
| **False Positive Rate** | 25% (2/8)    | ≤15%    | ❌ FAIL |
| **Precision**           | 0.75         | ≥0.85   | ❌ FAIL |
| **Recall**              | 0.60         | ≥0.85   | ❌ FAIL |
| **F1-Score**            | 0.67         | ≥0.85   | ❌ FAIL |
| **Fallback Dependency** | 100% (8/8)   | ≤30%    | ❌ FAIL |
| **Class Collapse**      | 92.9% avg    | <20%    | ❌ FAIL |
| **Confidence (mean)**   | 0.72         | ≥0.75   | ❌ FAIL |

### Breakdown by Transect
| Transect | Detection | Status          | Notes |
|----------|-----------|-----------------|-------|
| 100      | 116.0m    | ✓ Valid         | Weak signal (NIR=19.8) |
| 378      | None      | ❌ False Negative | No candidates found |
| 663      | 0.0m      | ❌ False Positive | Edge artifact |
| 932      | 113.0m    | ✓ Valid         | Strong signal (NIR=53.0) |
| 1075     | 102.0m    | ✓ Valid         | Moderate signal (NIR=30.4) |
| 1138     | 100.0m    | ✓ Valid         | Moderate signal (NIR=35.4) |
| 1220     | 249.0m    | ✓ Valid         | Strong signal (NIR=56.2) |
| 1515     | None      | ❌ False Negative | No candidates found |
| 1778     | 117.0m    | ✓ Valid         | Moderate signal (NIR=34.8) |
| 2077     | 0.0m      | ❌ False Positive | Edge artifact |

**True Performance:**
- True Positives: 6
- False Positives: 2 (edge artifacts)
- False Negatives: 2 (no detection)
- Precision: 6/8 = 0.75
- Recall: 6/10 = 0.60
- F1-Score: 0.67

---

## Recommended Action Plan (Prioritized)

### ✅ COMPLETED: Step 1 - Diagnostic Analysis
- [x] Parsed test log and extracted detection results
- [x] Generated visualizations (overview, pipeline, scatter plots)
- [x] Identified three critical failures
- [x] Documented findings in KEY_FINDINGS.md

**Time spent:** ~2 hours
**Outputs:**
- `detection_overview.png`
- `pipeline_progression.png`
- `detection_results.csv`
- `diagnostic_summary.txt`
- `KEY_FINDINGS.md`

---

### 🔄 NEXT: Step 2 - Implement Fixes (Priority 1-3)

**Priority 1: Soft Confidence Scoring** (2-3 hours)
- Replace hard threshold checks in `_detect_dry_wet_boundaries()`
- Implement sigmoid/linear scoring functions for each criterion
- Remove strict vs fallback distinction

**Priority 2: Disable Monotonic Smoothing** (30 minutes)
- Add flag to bypass smoothing in `classifier.py`
- Implement light median filter alternative (window=5)
- Test impact on boundary detection

**Priority 3: Fix Edge Artifacts** (30 minutes)
- Add hard constraints to `_detect_with_relaxed_thresholds()`
- Increase edge buffer
- Add distance bounds (30m - 250m)

**Estimated Total:** 3-4 hours

---

### ⏳ PENDING: Step 3 - Validation & Testing

After implementing fixes:
1. Re-run test on same 10 transects
2. Compare before/after metrics:
   - Detection rate (target: 8-9/10)
   - False positive rate (target: 0-1/10)
   - Confidence distribution (target: mean ≥0.75)
3. If successful, validate on full 20-transect labeled set
4. Generate comparison report

**Estimated Total:** 2 hours

---

## Files Created

This diagnostic analysis generated the following files in `/analysis/phase6c_diagnostics/`:

### Visualizations
- `detection_overview.png` - 4-panel performance overview
- `pipeline_progression.png` - Detection pipeline attrition

### Data Files
- `detection_results.csv` - Tabular results per transect
- `diagnostic_summary.txt` - Text summary report

### Documentation
- `KEY_FINDINGS.md` - Detailed issue analysis and recommendations
- `DIAGNOSTIC_SUMMARY_WITH_VISUALS.md` - This file (comprehensive report)

### Scripts
- `analyze_test_log.py` - Log parsing and visualization script
- `validate_current_performance.py` - Validation framework (template)

---

## Next Steps

1. **Review visualizations** in `/analysis/phase6c_diagnostics/` directory
2. **Read KEY_FINDINGS.md** for detailed technical analysis
3. **Proceed with implementing Priority 1-3 fixes** as outlined above
4. **Re-test and validate** improvements

**Estimated Time to Fix:** 3-4 hours of focused implementation + 2 hours testing = **5-6 hours total**

**Expected Outcome:**
- Detection rate: 60% → **85-90%**
- False positive rate: 25% → **<10%**
- F1-Score: 0.67 → **≥0.85**

---

**Analysis Complete!** 🎯

All diagnostic outputs are available in:
`C:\Users\alisa\Desktop\SIP\git\beach_spectral\analysis\phase6c_diagnostics\`
