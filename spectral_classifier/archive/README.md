# Deprecated Features Archive

This directory contains features that were removed from the main codebase
but preserved for historical reference.

## Contents

### deprecated_smoothing.py

**Removed**: Phase 8 (2025-10-17)

**Reason**: Monotonic smoothing caused catastrophic class collapse in Phase 6D testing, with 93% of points being reclassified as UNKNOWN in some transects.

**Details**: See docs/IMPROVEMENT_HISTORY.md - Phase 6D section, and validation/outputs/phase6d/ for diagnostic outputs.

**Contains**:
- `apply_monotonic_smoothing()` - Main entry point
- `_build_segments()` - Segment builder helper
- `_build_monotonic_from_zones()` - Zone-based monotonic sequence builder
- `_remove_short_segments()` - Short segment merger
- `_enforce_monotonic_sequence()` - Monotonic sequence enforcer

**DO NOT USE THESE METHODS** - They are disabled and unsupported.

## Other Removed Features

### Legacy Detection Methods (Phase 1)

**Removed**: Phase 8 (2025-10-17)

**Location**: Previously in `transition.py` lines 277-520

**Methods removed**:
- `_detect_nir_derivative_boundaries()` - NIR derivative-based detection
- `_detect_nir_drop()` - Sharp NIR decrease detection
- `_detect_ndwi_transition()` - NDWI transition detection
- `_detect_brightness_drop()` - Brightness decrease detection
- `_detect_spectral_angle_change()` - Spectral angle change detection

**Reason**: Superseded by Phase 2+ boundary-type-specific methods:
- `_detect_vegetation_boundaries()` (inflection detection)
- `_detect_surf_zone_boundaries()` (RGB foam detection)
- `_detect_dry_wet_boundaries()` (derivative magnitude)

### Deprecated Config Parameters

**Removed**: Phase 8 (2025-10-17)

**Location**: Previously in `config.py`

**Parameters removed**:
- `ndvi_veg_min`: 0.2 - Vegetation NDVI threshold (unreliable, all negative values)
- `ndvi_land_max`: 0.3 - Land NDVI threshold (unreliable, all negative values)
- `nir_drop`: 20 - NIR decrease threshold for legacy detection
- `nir_drop_window`: 5 - Window for legacy NIR drop detection
- `brightness_drop`: 30 - Brightness decrease for legacy detection
- `spectral_angle_threshold`: 30 - Degrees for legacy spectral angle detection
- `ndwi_transition`: 0.2 - NDWI threshold for legacy water detection

**Reason**: No longer used by Phase 2+ boundary-type-specific detection methods.

## Accessing Historical Code

For removed features not archived in this directory, see git history before Phase 8:

```bash
# View legacy detection methods
git show phase7:spectral_classifier/transition.py

# View deprecated config parameters
git show phase7:spectral_classifier/config.py

# View full diff of Phase 8 changes
git diff phase7 phase8-cleanup-safe
```

## Phase 8 Cleanup Summary

Total lines removed from active codebase:
- Legacy detection methods: ~235 lines
- Monotonic smoothing: ~320 lines (moved to archive)
- Deprecated config parameters: ~7 lines
- **Total**: ~565 lines removed

This represents a ~15% reduction in codebase size while maintaining all active functionality.

---

**Last Updated**: 2025-10-17
**Phase**: 8 (Code Streamlining)
