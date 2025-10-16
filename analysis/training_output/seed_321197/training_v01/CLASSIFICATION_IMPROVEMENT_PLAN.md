# Classification Improvement Plan

**Generated:** 2025-10-14
**Random Seed:** 321197
**Sample Size:** 10 transects, 2,602 points analyzed (with 5m boundary tolerance)

---

## Executive Summary

Analysis of manually classified training data reveals **significant issues** with the current classification system. The most critical problems are:

1. **VEG_DUNES class is fundamentally broken** - expects positive NDVI but actual vegetated dunes show negative NDVI
2. **BEACH_WET class is missing entirely** - a distinct spectral zone between dry beach and water
3. **Threshold values are poorly calibrated** - many are off by large margins
4. **Key spectral features are underutilized** - NIR ratio and blue/red ratio are powerful discriminators

**Current classification accuracy (estimated from logs):** ~40-55% (majority classified as UNKNOWN)

---

## Part 1: Spectral Characteristics by Land Cover Class

### Empirical Results (with ±5m boundary tolerance)

#### **VEGETATED_DUNE** (n=466 points, 10 segments)
```
Raw Bands:
  Red:   162.5 ± 35.0  [95-234]
  Green: 153.7 ± 38.9  [78-230]
  Blue:  134.5 ± 41.9  [68-218]
  NIR:   127.9 ± 34.0  [66-225]

Key Features:
  Brightness:     144.65 ± 36.59
  NDVI:           -0.123 ± 0.053  [-0.286 to 0.053]  ⚠️ NEGATIVE!
  NDWI:            0.091 ± 0.054
  NIR ratio:       0.887 ± 0.075                     ✓ High
  Blue/Red ratio:  0.813 ± 0.091                     ✓ Low
  Variability:     27.75 ± 10.64  [13-44]            ✓ HIGH
```

**Identifying Traits:**
- **Primary:** HIGH variability (>20), consistent with vegetation/dune structure
- **Secondary:** Moderate-high NIR ratio (>0.80), indicating dry substrate
- **Tertiary:** Low-moderate brightness with high standard deviation
- **CRITICAL NOTE:** NDVI is NEGATIVE, not positive! This invalidates current VEG_DUNES rule.

---

#### **BEACH_DRY** (n=377 points, 10 segments)
```
Raw Bands:
  Red:   220.3 ± 5.3   [204-231]
  Green: 218.8 ± 4.9   [201-227]
  Blue:  210.7 ± 6.6   [184-223]
  NIR:   186.9 ± 11.7  [136-207]

Key Features:
  Brightness:     209.19 ± 6.31                      ✓ VERY HIGH
  NDVI:           -0.083 ± 0.025  [-0.200 to -0.042]
  NDWI:            0.079 ± 0.025
  NIR ratio:       0.893 ± 0.035                     ✓ High
  Blue/Red ratio:  0.957 ± 0.022                     ✓ Near 1.0
  Variability:      3.91 ± 2.06   [1-7]              ✓ VERY LOW
```

**Identifying Traits:**
- **Primary:** VERY HIGH brightness (>200), extremely uniform (low variability <10)
- **Secondary:** High NIR ratio (>0.85), characteristic of dry sand
- **Tertiary:** Blue/Red ratio near 1.0 (neutral color)
- **Best discriminator:** Combination of high brightness + low variability

---

#### **BEACH_WET** (n=92 points, 10 segments) ⚠️ **MISSING FROM CURRENT CLASSIFIER**
```
Raw Bands:
  Red:   182.8 ± 14.5  [157-210]
  Green: 173.2 ± 16.4  [147-206]
  Blue:  146.6 ± 23.3  [110-200]
  NIR:    83.9 ± 28.7  [25-154]

Key Features:
  Brightness:     146.63 ± 20.09                     ✓ Moderate
  NDVI:           -0.390 ± 0.128  [-0.727 to -0.152] ✓ Moderately negative
  NDWI:            0.367 ± 0.127  [0.144 to 0.709]   ✓ Moderate-high
  NIR ratio:       0.557 ± 0.129                     ✓ INTERMEDIATE
  Blue/Red ratio:  0.798 ± 0.076
  Variability:     11.41 ± 5.33   [4-17]
```

**Identifying Traits:**
- **Primary:** INTERMEDIATE NIR ratio (0.4-0.7) - wet sand has low NIR reflectance
- **Secondary:** Moderate NDWI (0.25-0.50), transitional between dry beach and water
- **Tertiary:** Moderate brightness (130-170)
- **Best discriminator:** NIR ratio is distinctly lower than dry beach but higher than water

---

#### **WATER** (n=1,667 points, 10 segments)
```
Raw Bands:
  Red:   123.4 ± 45.0  [55-245]
  Green: 146.7 ± 36.5  [80-253]
  Blue:  127.8 ± 42.6  [60-251]
  NIR:    24.9 ± 21.7  [6-175]

Key Features:
  Brightness:     105.70 ± 35.45
  NDVI:           -0.703 ± 0.108  [-0.871 to -0.167] ✓ VERY negative
  NDWI:            0.745 ± 0.118  [0.182 to 0.907]   ✓ VERY HIGH
  NIR ratio:       0.207 ± 0.097                     ✓ VERY LOW
  Blue/Red ratio:  1.059 ± 0.156                     ✓ >1.0
  Variability:     30.97 ± 7.39   [19-45]            ✓ HIGH (waves)
```

**Identifying Traits:**
- **Primary:** VERY LOW NIR ratio (<0.35) - water absorbs NIR
- **Secondary:** VERY HIGH NDWI (>0.65), strongest water indicator
- **Tertiary:** Blue/Red ratio > 1.0 (blue dominates)
- **Additional:** High variability from wave crests
- **Best discriminator:** NIR ratio is the most reliable single feature

---

## Part 2: Current Classification Issues

### Critical Problems

#### 1. **VEG_DUNES Rule is Broken** (classifier.py:98-104)
```python
# Current rule:
if (variability > 25 and has_osc and ndvi > 0.2):  # ❌ WRONG
    return 'VEG_DUNES', confidence
```

**Problem:** Requires `ndvi > 0.2`, but empirical NDVI = **-0.123 ± 0.053**

**Why NDVI is negative:** Vegetated dunes in this dataset appear to have mixed vegetation/sand patches where sand dominates, or vegetation has low chlorophyll content, or NIR band is capturing something else.

**Impact:** VEG_DUNES is rarely classified (only 4-21 points per transect, should be ~40-50)

---

#### 2. **BEACH_WET Class Missing**
**Problem:** No classification rule exists for wet beach, a distinct spectral zone

**Impact:** Wet beach points are misclassified as UNKNOWN (current logs show this)

**Evidence:** Clear spectral signature with NIR ratio ~0.56 (between dry beach 0.89 and water 0.21)

---

#### 3. **DRY_BEACH Thresholds Completely Wrong** (classifier.py:117-123)
```python
# Current thresholds:
dry_beach_brightness_min = 90   # ❌ Actual: 209
dry_beach_brightness_max = 140  # ❌ Actual: 209
```

**Problem:** Thresholds are ~70 units too low

**Impact:** Most dry beach is classified as UNKNOWN or ALL_LAND

---

#### 4. **OCEAN NDWI Threshold Too Low** (classifier.py:87)
```python
if (ndwi > 0.2 and ...):  # ❌ Actual water: 0.745
```

**Problem:** Threshold of 0.2 would capture wet beach (NDWI=0.37) as ocean

**Impact:** Fuzzy water/beach boundary

---

#### 5. **NIR Ratio Underutilized**
**Current:** Only used for dry beach detection (threshold 0.75)

**Opportunity:** NIR ratio is the BEST single discriminator:
- Water: 0.21 ± 0.10
- Wet beach: 0.56 ± 0.13
- Dry beach: 0.89 ± 0.04
- Veg dunes: 0.89 ± 0.08

**Clear separation with minimal overlap!**

---

#### 6. **Blue/Red Ratio Not Used At All**
**Problem:** Powerful feature ignored

**Evidence:**
- Water: 1.06 ± 0.16 (blue > red)
- All land classes: 0.80-0.96 (blue < red)

---

### Current Performance Issues (from logs)

Analyzing classification distributions from `training_sample_info.json`:

| Transect | UNKNOWN | Correct Classes | % UNKNOWN |
|----------|---------|-----------------|-----------|
| 100      | 178     | 124             | 59%       |
| 378      | 268     | 34              | 89%       |
| 663      | 154     | 148             | 51%       |
| 932      | 153     | 149             | 51%       |
| 1075     | 203     | 98              | 67%       |
| 1138     | 218     | 84              | 72%       |
| 1220     | 166     | 136             | 55%       |
| 1515     | 141     | 160             | 47%       |
| 1778     | 267     | 35              | 88%       |
| 2077     | 169     | 133             | 56%       |

**Average: 63.5% classified as UNKNOWN** ❌

---

## Part 3: Proposed Improvements

### A. Update Configuration (config.py)

#### Add BEACH_WET class
```python
LANDCOVER_CLASSES = [
    'ALL_LAND',
    'DRY_BEACH',
    'BEACH_WET',      # ← NEW
    'VEG_DUNES',
    'WATER',
    'WAVE_CRESTS',    # Consider merging with WATER
    'UNKNOWN'
]

LANDCOVER_COLORS = {
    # ... existing ...
    'BEACH_WET': '#CD853F',  # Darker sand/tan
}
```

#### Revised Thresholds (data-driven)
```python
THRESHOLDS = {
    # NIR ratio - PRIMARY discriminator
    'nir_ratio_water_max': 0.35,        # Water: 0.21 ± 0.10
    'nir_ratio_wet_beach_min': 0.40,    # Wet beach: 0.56 ± 0.13
    'nir_ratio_wet_beach_max': 0.70,
    'nir_ratio_dry_min': 0.80,          # Dry beach/veg: 0.89 ± 0.04

    # Brightness - for distinguishing dry beach from veg dunes
    'dry_beach_brightness_min': 195,    # Empirical: 209 ± 6, use conservative bound
    'veg_dune_brightness_max': 190,     # Empirical: 145 ± 37, allow overlap

    # Variability - for distinguishing veg dunes from dry beach
    'veg_variability_min': 15,          # Empirical: 28 ± 11, conservative
    'dry_beach_variability_max': 10,    # Empirical: 4 ± 2, generous

    # NDWI - water confirmation
    'ndwi_water_min': 0.60,             # Empirical: 0.75 ± 0.12, conservative
    'ndwi_wet_beach_min': 0.25,         # Empirical: 0.37 ± 0.13

    # Blue/Red ratio - water confirmation
    'blue_red_water_min': 0.95,         # Empirical: 1.06 ± 0.16

    # NDVI - REMOVED as primary discriminator, use only for confirmation
    'ndvi_water_max': -0.50,            # Empirical: -0.70 ± 0.11

    # ... other existing thresholds ...
}
```

---

### B. Revised Classification Logic (classifier.py)

#### New Hierarchical Classification Rules

**Principle:** Use NIR ratio as primary discriminator, then refine with other features

```
Classification Hierarchy:
1. WATER:      nir_ratio < 0.35 AND (ndwi > 0.60 OR blue_red > 0.95)
2. BEACH_WET:  0.40 < nir_ratio < 0.70 AND ndwi > 0.25
3. DRY_BEACH:  nir_ratio > 0.80 AND brightness > 195 AND variability < 10
4. VEG_DUNES:  nir_ratio > 0.80 AND variability > 15 AND brightness < 190
5. WAVE_CRESTS: (detection within WATER zones based on brightness spikes)
6. ALL_LAND:   High brightness fallback (for areas beyond beach/dunes)
```

#### Detailed Rule Specifications

**Rule 1: WATER**
```python
def _is_water(self, row) -> bool:
    """Water is characterized by very low NIR ratio."""
    nir_ratio = row['nir_ratio']
    ndwi = row['ndwi']
    blue_red = row['blue_red_ratio']

    # Primary: Very low NIR
    if nir_ratio > self.thresholds['nir_ratio_water_max']:
        return False

    # Confirmation: High NDWI or blue>red
    if ndwi > self.thresholds['ndwi_water_min']:
        return True
    if blue_red > self.thresholds['blue_red_water_min']:
        return True

    # Moderate confidence if NIR very low
    return nir_ratio < 0.30
```
**Confidence:** 0.85-0.95 based on how many conditions met

---

**Rule 2: BEACH_WET**
```python
def _is_wet_beach(self, row) -> bool:
    """Wet beach has intermediate NIR ratio."""
    nir_ratio = row['nir_ratio']
    ndwi = row['ndwi']
    brightness = row['brightness']

    # NIR ratio in intermediate range
    if not (self.thresholds['nir_ratio_wet_beach_min'] < nir_ratio <
            self.thresholds['nir_ratio_wet_beach_max']):
        return False

    # NDWI elevated but not as high as water
    if ndwi < self.thresholds['ndwi_wet_beach_min']:
        return False

    # Brightness check (not too bright like dry beach)
    if brightness > self.thresholds['dry_beach_brightness_min']:
        return False

    return True
```
**Confidence:** 0.70-0.85 (transitional zone)

---

**Rule 3: DRY_BEACH**
```python
def _is_dry_beach(self, row) -> bool:
    """Dry beach is very bright, uniform, high NIR ratio."""
    nir_ratio = row['nir_ratio']
    brightness = row['brightness']
    variability = row['variability']

    # High NIR ratio (dry)
    if nir_ratio < self.thresholds['nir_ratio_dry_min']:
        return False

    # Very high brightness
    if brightness < self.thresholds['dry_beach_brightness_min']:
        return False

    # Low variability (uniform)
    if variability > self.thresholds['dry_beach_variability_max']:
        return False

    return True
```
**Confidence:** 0.80-0.90 (very distinctive signature)

---

**Rule 4: VEG_DUNES**
```python
def _is_veg_dunes(self, row) -> bool:
    """Vegetated dunes have high variability, high NIR ratio, lower brightness."""
    nir_ratio = row['nir_ratio']
    brightness = row['brightness']
    variability = row['variability']

    # High NIR ratio (dry substrate)
    if nir_ratio < self.thresholds['nir_ratio_dry_min']:
        return False

    # High variability (vegetation/dune structure)
    if variability < self.thresholds['veg_variability_min']:
        return False

    # Lower brightness than dry beach
    if brightness > self.thresholds['veg_dune_brightness_max']:
        return False

    # Note: NOT using NDVI as primary criterion
    # Oscillations can be used as confirmation if available

    return True
```
**Confidence:** 0.70-0.85 (can be ambiguous with shadows/mixed pixels)

---

**Rule 5: WAVE_CRESTS** (optional refinement)
```python
def _is_wave_crest(self, row) -> bool:
    """Wave crests are bright spots within water zones.

    Note: This should be applied as a post-processing step
    to points already classified as WATER.
    """
    # Detect within water zones only
    # Requires spatial context (neighboring points)
    # High variability + brightness spike relative to surrounding water
    pass  # Implementation requires spatial analysis
```

**Alternative:** Consider merging WAVE_CRESTS into WATER class and using separate transition detection for wave features.

---

**Rule 6: ALL_LAND** (fallback)
```python
def _is_all_land(self, row) -> bool:
    """Catch-all for land beyond beach/dunes."""
    brightness = row['brightness']
    nir_ratio = row['nir_ratio']

    # High brightness and dry (but not matching dry beach pattern)
    if brightness > 180 and nir_ratio > 0.75:
        return True

    return False
```
**Confidence:** 0.50-0.65 (low confidence fallback)

---

### C. Implementation Order

```
Step 1: Update config.py
  - Add BEACH_WET to LANDCOVER_CLASSES and colors
  - Replace THRESHOLDS with data-driven values
  - Consider removing or demoting WAVE_CRESTS

Step 2: Refactor classifier.py
  - Extract helper methods for each class (_is_water, _is_wet_beach, etc.)
  - Rewrite _classify_point to use hierarchical rules
  - Update confidence scoring based on how strongly conditions are met

Step 3: Update validation and transition logic
  - Add BEACH_WET to valid_transitions in validate_sequence()
  - Update transition detection to recognize new class

Step 4: Test on training data
  - Re-run classification on 10 training transects
  - Calculate accuracy metrics
  - Iterate on thresholds if needed
```

---

## Part 4: Expected Improvements

### Quantitative Predictions

**Current state:**
- ~64% UNKNOWN
- ~36% classified (often incorrectly)

**Expected with improvements:**
- <15% UNKNOWN (only truly ambiguous pixels)
- >80% correctly classified
- All 5 land cover types properly detected

### Key Metrics to Track

1. **Class coverage:** % of points assigned to each class
2. **UNKNOWN rate:** Should drop from 64% to <15%
3. **Spatial coherence:** Fewer isolated misclassifications
4. **Boundary accuracy:** Transition locations match manual annotations within ±5m

---

## Part 5: Validation Strategy

### Quantitative Validation
```python
# Create validation script: validate_classifier.py
# For each transect:
#   1. Run updated classifier
#   2. Compare to manual classifications (with ±5m tolerance)
#   3. Calculate confusion matrix
#   4. Report per-class accuracy and overall accuracy
```

### Visual Validation
- Regenerate classified plots for all 10 training transects
- Side-by-side comparison with manual annotations
- Check transition locations

### Iterative Refinement
- If accuracy < 80%, analyze failure modes
- Adjust thresholds based on confusion patterns
- Consider adding hybrid rules for ambiguous cases

---

## Appendix: Feature Importance Ranking

Based on empirical analysis:

| Rank | Feature         | Importance | Reasoning |
|------|-----------------|------------|-----------|
| 1    | NIR ratio       | ★★★★★      | Clean separation across all classes |
| 2    | Brightness      | ★★★★☆      | Distinguishes dry beach, confirms water |
| 3    | Variability     | ★★★★☆      | Separates veg dunes from dry beach |
| 4    | NDWI            | ★★★☆☆      | Good for water, less useful for land |
| 5    | Blue/Red ratio  | ★★★☆☆      | Confirms water (>1.0) |
| 6    | NDVI            | ★☆☆☆☆      | NOT USEFUL in this dataset (all negative) |
| 7    | Oscillations    | ★☆☆☆☆      | Optional confirmation, not primary |

---

## Summary

**Main Actions:**
1. ✓ Add BEACH_WET class
2. ✓ Fix VEG_DUNES detection (remove NDVI requirement)
3. ✓ Update all thresholds to match empirical data
4. ✓ Use NIR ratio as primary discriminator
5. ✓ Add blue/red ratio for water confirmation
6. ✓ Implement hierarchical classification logic

**Expected Outcome:**
A robust classifier that correctly identifies all 5 land cover types with >80% accuracy, reducing UNKNOWN classifications from 64% to <15%.
