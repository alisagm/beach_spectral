# Deprecated Detection Heuristics

This document records detection approaches that were tried and ultimately removed from the active codebase. Preserved for historical reference and to prevent re-implementing approaches that didn't work.

---

## 1. NDVI-Based Vegetation Detection

**Status:** Removed in Phase 8

**Original Approach:**
Used NDVI (Normalized Difference Vegetation Index) as a primary classifier for identifying vegetated dunes vs. bare sand.

```python
# Original thresholds
'ndvi_veg_min': 0.2,   # Vegetation identification
'ndvi_land_max': 0.3,  # Land vs water
```

**Why It Failed:**
Empirical analysis of PAIS imagery showed **all NDVI values were negative** across all landcover classes:
- Water: -0.70 ± 0.11
- Wet Beach: -0.39 ± 0.13
- Dry Beach: -0.08 ± 0.03
- Veg Dunes: -0.12 ± 0.05

The expected positive NDVI values for vegetation never materialized, likely due to:
1. Sparse, low-greenness coastal vegetation
2. Sand/vegetation mixing at transect scale
3. Possible sensor calibration or atmospheric effects

**Replacement:**
NIR ratio (`nir_ratio = nir / visible_mean`) proved far more reliable for distinguishing landcover classes. The empirical NIR ratio ranges showed clear separation:
- Water: 0.21 ± 0.10
- Wet Beach: 0.56 ± 0.13
- Dry Beach/Veg: 0.89 ± 0.04

---

## 2. Legacy Transition Detection Methods

**Status:** Removed in Phase 8

**Original Methods:**

### 2.1 NIR Drop Detection (`_detect_nir_drop`)
```python
# Legacy parameters
'nir_drop': 20,         # NIR decrease threshold
'nir_drop_window': 5,   # Detection window
```

Simple threshold-based NIR drop detection without derivative analysis.

**Why Removed:** Replaced by derivative-based detection which is more robust to varying absolute NIR values and provides magnitude information for confidence scoring.

### 2.2 Brightness Drop Detection (`_detect_brightness_drop`)
```python
'brightness_drop': 30,  # Brightness decrease threshold
```

Threshold-based brightness change detection.

**Why Removed:** Brightness varies significantly with sun angle, atmospheric conditions, and sensor differences. Derivative-based detection normalizes for these effects.

### 2.3 Spectral Angle Change (`_detect_spectral_angle_change`)
```python
'spectral_angle_threshold': 30,  # Degrees
```

Detected transitions based on spectral angle changes between consecutive points.

**Why Removed:** Spectral angle is too sensitive to noise and not specific to beach-water transitions. Many irrelevant features (shadows, debris) create similar spectral angle changes.

### 2.4 NDWI Transition Detection (`_detect_ndwi_transition`)
```python
'ndwi_transition': 0.2,  # NDWI change threshold
```

Used NDWI changes to identify water boundaries.

**Why Removed:** NDWI is useful for water confirmation but not reliable for shell line detection. The transition from dry to wet beach shows NDWI changes, but they overlap significantly with natural variability.

---

## 3. Phase-Specific Threshold Evolution

### Phase 2 Fixes (False Positive Reduction)

**Problem:** Initial thresholds produced too many false positives, particularly:
- Wave crests detected as shell lines
- Vegetation boundaries detected as shell lines

**Changes:**
```python
# Tightened thresholds
'second_deriv_threshold': 1.0,    # Increased from 0.5
'min_nir_change': 20,             # Increased from 10
'rgb_peak_prominence': 10.0,      # Increased from 5.0
'require_nir_drop': True,         # New requirement
'min_boundary_confidence': 0.75,  # Increased from 0.60
```

### Phase 6 Empirical Calibration

**Problem:** Phase 2 thresholds were too strict and rejected ~25% of valid shell lines.

**Analysis:** 20-transect feature analysis provided empirical distributions:
- NIR before: median=173, 5th percentile=139
- Brightness before: median=203, 5th percentile=183

**Changes:**
```python
# Relaxed to 5th percentile values (with safety margin)
'min_nir_drop_absolute': 39.0,    # Was 40
'brightness_before_min': 175,     # Was 190
'nir_before_min': 135,            # Was 160
'variability_ratio_min': 1.5,     # Was 2.0
```

---

## 4. Attempted Classification Schemes

### 4.1 ALL_LAND Class

**Status:** Removed

**Original Purpose:** Classify areas beyond dunes/beach as "ALL_LAND" to distinguish from beach zones.

**Why Removed:** 
- Transects don't typically extend far enough inland
- No operational need for this distinction
- Added complexity without improving shell line detection

### 4.2 WAVE_CRESTS Sub-Classification

**Status:** Kept but de-emphasized

**Original Purpose:** Distinguish wave crests from open water for more precise waterline detection.

**Current Status:** Still computed but not used for shell line detection. May be useful for surf zone analysis in future work.

---

## 5. Band Detection Heuristics

### 5.1 Variance Ratio Method

**Status:** In use but with caveats

**Method:** CIR vs RGB detection based on Band1/Band2 standard deviation ratio.
- CIR: NIR (band1) has higher variance → ratio > 1.05
- RGB: Red (band1) has similar variance → ratio ≤ 1.05

**Empirical Values:**
- CIR: ~1.14 ratio
- RGB: ~0.97 ratio

**Limitations Found:**
- Unreliable when imagery has low vegetation variability
- Tile boundary effects can flip detection between tiles
- Year-level resolution (majority voting) more reliable than per-tile detection

**Current Approach:**
1. Check filename for `_cir_` indicator (most reliable)
2. Use variance heuristic as fallback
3. Apply year-level majority voting for consistency

---

## 6. Lessons Learned

### What Works:
1. **NIR derivatives** - Robust, interpretable, magnitude provides confidence
2. **Multi-band consensus** - Increases reliability, reduces false positives
3. **R/G ratio validation** - Shell line has characteristic R/G ≈ 1.0
4. **Expected zone constraints** - Prior knowledge improves accuracy
5. **Cascading fallback strategy** - Ensures output even for difficult imagery

### What Doesn't Work:
1. **NDVI for coastal environments** - Values don't match expectations
2. **Simple threshold detection** - Too sensitive to image variations
3. **Per-tile band detection** - Inconsistent across tile boundaries
4. **Overly strict thresholds** - Rejects valid detections

### Design Principles:
1. **Derivative-based > Threshold-based** - More robust to absolute value variations
2. **Year-level consistency > Tile-level accuracy** - Prevents transect discontinuities
3. **Guaranteed output > Perfect accuracy** - Always provide a shell line estimate
4. **Confidence scoring > Binary detection** - Allows downstream filtering

---

## References

- Original training data: seed=321197, n=2602 points, 10 transects
- Phase 6 validation: 20-transect feature analysis
- Threshold optimization: Grid search on labeled validation set

---

*Last updated: January 2026*
*Prior to refactoring for handoff*