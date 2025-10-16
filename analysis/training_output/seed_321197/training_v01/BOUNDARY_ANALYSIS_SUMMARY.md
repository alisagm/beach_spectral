# Boundary Analysis Summary & Implementation Plan

**Generated:** 2025-10-14
**Analysis:** 30 boundaries across 10 transects

---

## Key Findings

### 1. VEG_DUNES → DRY_BEACH (WEAK signal)
- **RGB:** +10-15 units (+5-8%)
- **NIR:** -8 units (-5%)
- **First derivative max:** ~40 units/m
- **Characteristics:** Gradual brightness increase, weak NIR signal

### 2. DRY_BEACH → BEACH_WET (STRONGEST signal) ⭐
- **RGB:** -20 to -35 units (-10 to -17%)
- **NIR DROP:** -59 units (-35%) **← KEY DISCRIMINATOR**
- **NIR d/dx:** mean=-4.73, max=26.5 units/m
- **Characteristics:** Sharp NIR decline, consistent across all transects

### 3. BEACH_WET → WATER (STRONG signal) ⭐
- **RGB:** -7 to -15 units (-4 to -8%)
- **NIR DROP:** -44 units (-50%) **← LARGEST RELATIVE DROP**
- **NIR d/dx:** mean=-3.20, max=52 units/m
- **Second derivative:** POSITIVE (deceleration)
- **Characteristics:** Very sharp NIR drop, brightness decreases

---

## Spectral Derivative Patterns

| Boundary Type | NIR d/dx (mean) | NIR d/dx (max) | RGB d/dx (max) | Key Pattern |
|---------------|-----------------|----------------|----------------|-------------|
| VEG→DRY       | +0.88           | 45.5           | 33-41          | Weak, noisy |
| **DRY→WET**   | **-4.73**       | **26.5**       | 11-14.5        | **Sharp, consistent** |
| **WET→WATER** | **-3.20**       | **52.0**       | 35-51          | **Very sharp** |

---

## Recommended Transition Detection Strategy

### Current Issues
1. ✗ Too many false positives (detecting noise/wave crests)
2. ✗ Not focused on major zone boundaries
3. ✗ Uses spectral indices instead of raw derivatives

### Proposed Solution

#### PRIMARY DETECTOR: NIR First Derivative
```python
# Detect sustained NIR drops
nir_d1 = np.gradient(nir_values, distance_values)

# Smooth derivatives to reduce noise
nir_d1_smooth = uniform_filter1d(nir_d1, size=3)

# Detect significant negative slopes
is_boundary = (nir_d1_smooth < -3.0) & (nir_values > 50)  # Avoid noise in low-NIR water

# Require sustained drop over 3-5m
sustained_drop = convolve with window of 3-5 points
```

#### SECONDARY CONFIRMATIONS
1. **Brightness decrease:** Δbrightness < -10 units
2. **NIR absolute drop:** ΔNIR < -30 units over 5m window
3. **Class transition:** Crossing from classified zone to different zone

#### FILTERING RULES
1. **Minimum separation:** 15m between transitions (can't have 2 boundaries closer than this)
2. **Within-class noise:** Ignore fluctuations within same classified zone
3. **Wave crests:** Oscillations within WATER zone, NOT boundaries
4. **Edge effects:** Ignore derivatives within 5m of transect ends

---

## Visualization Updates

### Current Plots
- Spectral bands (R, G, B, NIR)
- Classified background zones
- Transition markers with confidence

### Recommended Additions
**Option A: Add derivative subplot** (4 panels)
1. Raw spectral bands
2. Classified zones
3. **NIR first derivative** (highlight boundary signals)
4. Transitions marked

**Option B: Overlay derivatives on main plot**
- Use secondary y-axis for d(NIR)/dx
- Color-code: red=negative slope, green=positive slope
- Threshold line at -3.0

**Option C: Multi-band derivatives** (user requested)
- Plot d/dx for RGB + NIR
- Show relative magnitudes
- Highlight boundary locations

---

## Implementation Steps

### Step 1: Add NIR Derivative Features
```python
# In features.py, add to compute_all():
self.data['nir_d1'] = self._compute_nir_derivative()
self.data['nir_d2'] = self._compute_nir_second_derivative()
self.data['rgb_d1'] = self._compute_rgb_derivative()
```

### Step 2: Rewrite Transition Detector
```python
# In transition.py, replace _detect_spectral_transitions():
def _detect_nir_derivative_boundaries(self, features):
    """Detect boundaries using NIR first derivative."""
    # Calculate smoothed derivative
    # Find peaks in -d(NIR)/dx (negative slopes)
    # Filter by magnitude and duration
    # Return only major boundaries
```

### Step 3: Update Visualization
```python
# In visualization.py, add derivative panel:
def plot_with_derivatives(...):
    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[2, 1])
    # ax1: Spectral bands + classifications
    # ax2: NIR derivative with boundary threshold
```

### Step 4: Test on Training Data
- Re-run classification with new transition detection
- Compare detected boundaries to manual annotations
- Calculate boundary location accuracy (±5m tolerance)
- Iterate on thresholds if needed

---

## Expected Improvements

**Before:**
- 10-25 transitions per transect (many false positives)
- Detects wave crests as boundaries
- Inconsistent boundary detection

**After:**
- 3-4 transitions per transect (matches manual annotations)
- Ignores wave crests (oscillations within zones)
- Consistent, reproducible boundary detection
- Accurate to ±5m of manual annotations

---

## Next Actions

1. **Review this summary** - Confirm approach aligns with goals
2. **Choose visualization option** (A, B, or C above)
3. **Implement features + transition detector**
4. **Update visualization**
5. **Test and validate**

**Estimated implementation time:** 30-45 minutes

---

## References

- Manual classifications: `classified_manual.csv`
- Boundary analysis script: `analyze_boundaries.py`
- Derivative plots: `boundary_analysis_*.png` (3 files)
- Spectral data: `sampled_spectral_data.csv`
