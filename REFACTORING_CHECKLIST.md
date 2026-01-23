# Refactoring Checklist

## Files Created (NEW)

These files are complete and ready to use:

| File | Status | Notes |
|------|--------|-------|
| `README.md` | ✅ Done | Package documentation |
| `run.py` | ✅ Done | New CLI entry point |
| `utils/__init__.py` | ✅ Done | Package exports |
| `utils/logging_config.py` | ✅ Done | Extracted from utils.py |
| `utils/export.py` | ✅ Done | Extracted from utils.py |
| `utils/batch.py` | ✅ Done | Extracted from batch_by_year.py |
| `utils/helpers.py` | ✅ Done | Small utilities |
| `transition/__init__.py` | ✅ Done | Package exports |
| `transition/nir.py` | ✅ Done | NIR detection methods |
| `transition/rgb.py` | ✅ Done | RGB fallback methods |
| `transition/shell_line.py` | ✅ Done | TransitionDetector class |
| `spectral/__init__.py` | ✅ Done | Package exports |
| `visualization/__init__.py` | ✅ Done | Package exports |
| `optional/__init__.py` | ✅ Done | Package exports |
| `optional/boundary_types.py` | ✅ Done | Optional veg/waterline detection |
| `tools/DEPRECATED_HEURISTICS.md` | ✅ Done | Historical approaches |

---

## Files to MOVE (existing, minimal changes)

These files should be copied from the original project with import updates:

### 1. `config.py` → `spectral_classifier/config.py`
**Changes needed:** None (already clean)

### 2. `data_io.py` → `utils/data_io.py`
**Changes needed:**
```python
# Change:
from .config import BAND_DETECTION
# To:
from ..config import BAND_DETECTION
```

### 3. `sampler.py` → `spectral/sampler.py`
**Changes needed:**
```python
# Change:
from .config import SAMPLING_INTERVAL
from .data_io import RasterIndex, BAND_CONFIG_4BAND, BAND_CONFIG_CIR, BAND_CONFIG_RGB
# To:
from ..config import SAMPLING_INTERVAL
from ..utils.data_io import RasterIndex, BAND_CONFIG_4BAND, BAND_CONFIG_CIR, BAND_CONFIG_RGB
```

### 4. `features.py` → `spectral/features.py`
**Changes needed:**
```python
# Change:
from .config import EPSILON, THRESHOLDS
from .data_io import detect_band_mode_from_dataframe
# To:
from ..config import EPSILON, THRESHOLDS
from ..utils.data_io import detect_band_mode_from_dataframe
```
**Also:** Add `# REUSABLE` markers to derivative methods

### 5. `visualization.py` → `visualization/plotting.py`
**Changes needed:**
```python
# Change:
from .config import LANDCOVER_COLORS
from .transition import TransitionDetector
# To:
from ..config import LANDCOVER_COLORS
from ..transition import TransitionDetector
```

### 6. `footprint_clip.py` → `utils/footprint_clip.py`
**Changes needed:** Minimal - check imports

### 7. `classifier.py` → `optional/landcover_classifier.py`
**Changes needed:**
```python
# Change:
from .config import THRESHOLDS, LANDCOVER_CLASSES
from .data_io import detect_band_mode_from_dataframe
# To:
from ..config import THRESHOLDS, LANDCOVER_CLASSES
from ..utils.data_io import detect_band_mode_from_dataframe
```

### 8. `validate_shorelines.py` → `tools/validate_shorelines.py`
**Changes needed:**
- Update imports to use package
- Fix MultiLineString handling (known bug)

---

## Files to UPDATE (existing, significant changes)

### 9. `main.py` → `spectral_classifier/main.py`
**Changes needed:**
```python
# Update all imports to new structure:
from .config import ...
from .utils import ...
from .spectral import ...
from .transition import TransitionDetector
from .visualization import plot_transect_analysis
```

### 10. `__init__.py` → `spectral_classifier/__init__.py`
**Changes needed:** Update to reflect new structure
```python
from .main import analyze_all_transects, process_single_transect
from .utils import (
    build_raster_index,
    load_transects,
    # ... etc
)
from .spectral import SpectralFeatures, sample_transect
from .transition import TransitionDetector
from .visualization import plot_transect_analysis
from .config import THRESHOLDS, LANDCOVER_CLASSES, LANDCOVER_COLORS
```

---

## Directory Structure After Refactoring

```
spectral_classifier/
├── __init__.py         # UPDATE: new exports
├── config.py           # MOVE: as-is
├── main.py             # UPDATE: new imports
├── run.py              # NEW
├── README.md           # NEW
│
├── spectral/
│   ├── __init__.py     # NEW
│   ├── sampler.py      # MOVE: update imports
│   └── features.py     # MOVE: update imports, add markers
│
├── transition/
│   ├── __init__.py     # NEW
│   ├── nir.py          # NEW (extracted from transition.py)
│   ├── rgb.py          # NEW (extracted from transition.py)
│   └── shell_line.py   # NEW (extracted from transition.py)
│
├── utils/
│   ├── __init__.py     # NEW
│   ├── data_io.py      # MOVE: update imports
│   ├── footprint_clip.py # MOVE: check imports
│   ├── export.py       # NEW (extracted from utils.py)
│   ├── batch.py        # NEW (extracted from batch_by_year.py)
│   ├── helpers.py      # NEW (extracted from utils.py)
│   └── logging_config.py # NEW (extracted from utils.py)
│
├── visualization/
│   ├── __init__.py     # NEW
│   └── plotting.py     # MOVE: update imports (was visualization.py)
│
└── optional/
    ├── __init__.py     # NEW
    ├── landcover_classifier.py  # MOVE: update imports (was classifier.py)
    └── boundary_types.py        # NEW

tools/
├── validate_shorelines.py  # MOVE: update imports
├── DEPRECATED_HEURISTICS.md # NEW
└── batch_by_year.py        # KEEP for reference, replaced by run.py
```

---

## Files to DELETE (after refactoring)

| File | Reason |
|------|--------|
| `transition.py` | Split into transition/*.py |
| `utils.py` | Split into utils/*.py |
| `batch_by_year.py` | Replaced by run.py + utils/batch.py |

---

## Testing Checklist

After refactoring, verify:

1. [ ] `python -c "from spectral_classifier import TransitionDetector"` works
2. [ ] `python run.py --dry-run --imagery-root ./imagery` works
3. [ ] Process a single year and check GeoJSON output
4. [ ] Compare output to pre-refactor results (should be identical)

---

## Notes for Alisa

1. **Import updates** are the main manual work needed. The new files handle the split functionality.

2. **The transition module** now has clear separation:
   - `nir.py`: Functions your supervisor might adapt for forest edges
   - `rgb.py`: Fallback only, lower confidence
   - `shell_line.py`: Orchestration and selection logic

3. **DEPRECATED_HEURISTICS.md** documents why certain approaches were abandoned - this is important institutional knowledge for whoever picks up the project.

4. **The run.py** CLI replaces `batch_by_year.py` as the main entry point. It's cleaner and has proper argument parsing.

5. **validate_shorelines.py** still needs the MultiLineString fix - I noted this but didn't implement the fix since it's a separate bug.