# Spectral Transect Classification System

A Python-based system to automatically classify spectral profiles from cross-shore transects and identify transition zones (beach-water interfaces). The system analyzes multi-band spectral data (Red, Green, Blue, NIR) along transects extracted from GeoTIFF rasters and GeoJSON geometries.

## Features

- **Spatial indexing and lazy loading** of rasters for efficient memory usage
- **Automated CRS handling** with reprojection when needed
- **Direction detection**: Automatically detects transect orientation and plots consistently west-referenced
- **Feature extraction**: Spectral indices (NDVI, NDWI), statistical features, shape features
- **Rule-based classification**: ALL_LAND, OCEAN, DRY_BEACH, VEG_DUNES, WAVE_CRESTS
- **Transition zone detection**: Identifies beach-water interfaces with confidence scores
- **Clean logging**: Console shows only warnings/errors, detailed logs saved to file
- **Batch processing**: Processes all transects, visualizes 5 representative ones by default

## Installation

### Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

### Setup

```bash
# Clone or download the repository
cd beach_spectral

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Input Data Format

### 1. GeoTIFF Rasters
- **Band order**: Red (1), Green (2), Blue (3), NIR (4)
- All rasters must share the same CRS
- Can span multiple tiles

### 2. GeoJSON Transects
- Format: GeoJSON FeatureCollection
- Geometry: LineString (one per transect)
- Required property: `TransectID` (unique identifier)
- CRS: Any valid CRS (will be reprojected to match rasters if needed)

## Usage

### Command Line Interface

```bash
python -m spectral_classifier.main \
  --rasters /path/to/rasters \
  --transects /path/to/transects.geojson \
  --output /path/to/output \
  --num-visualize 5 \
  --verbose
```

**Arguments:**
- `--rasters`: Directory containing GeoTIFF files
- `--transects`: Path to GeoJSON file with transect LineStrings
- `--output`: Directory to save output files
- `--num-visualize`: Number of transects to visualize (default: 5)
- `--verbose`: Enable verbose debug logging

### Python API

```python
from spectral_classifier import analyze_all_transects

results = analyze_all_transects(
    raster_dir='path/to/rasters',
    transect_geojson='path/to/transects.geojson',
    output_dir='path/to/output',
    num_visualize=5,
    verbose=True
)
```

## Output Files

The system generates the following outputs in the specified output directory:

1. **`transect_analysis_YYYYMMDD_HHMMSS.csv`**
   - Columns: TransectID, distance, red, green, blue, nir, predicted_class, transition_flag, confidence
   - All sample points for all transects

2. **`summary_YYYYMMDD_HHMMSS.json`**
   - Summary statistics for all transects
   - Class distributions
   - Transition locations and counts
   - Processing metadata

3. **`transect_<ID>_analysis.png`** (for 5 representative transects)
   - 4-band spectral profiles (Red, Green, Blue, NIR)
   - Color-coded classification background
   - Transition zone markers with confidence scores
   - Direction-aware x-axis labels ("Distance from West" or "Distance to West")

4. **`processing.log`**
   - INFO-level messages and above (or DEBUG if --verbose flag used)
   - Console displays only WARNING and ERROR messages for clean output

## Configuration

Adjust classification thresholds and parameters in `spectral_classifier/config.py`:

```python
THRESHOLDS = {
    'land_brightness_min': 140,
    'ocean_brightness_max': 120,
    'ndwi_water_min': 0.2,
    'nir_drop': 20,
    # ... and more
}
```

## Module Structure

```
spectral_classifier/
├── __init__.py          # Package initialization
├── config.py            # Configuration parameters
├── data_io.py           # Raster & GeoJSON loading, CRS handling
├── sampler.py           # Spectral value extraction
├── features.py          # Feature extraction (SpectralFeatures)
├── classifier.py        # Landcover classification (LandcoverClassifier)
├── transition.py        # Transition detection (TransitionDetector)
├── visualization.py     # Plotting functions
├── utils.py             # Helper functions
└── main.py              # Pipeline orchestration
```

## Processing Pipeline

1. **Build raster spatial index** - Create bounding box index without loading pixel data
2. **Load transects** - Read GeoJSON and validate TransectID field
3. **CRS validation** - Reproject transects if CRS mismatch detected
4. **Direction detection** - Detect transect orientation from first transect (west-to-east or east-to-west)
5. **Sequential processing** - For each transect:
   - Find overlapping rasters (lazy load only needed tiles)
   - Sample spectral values every 1m (bilinear interpolation)
   - Adjust distances for consistent west-referenced plotting
   - Compute spectral features (NDVI, NDWI, brightness, variability, etc.)
   - Classify landcover using rule-based algorithm
   - Apply spatial smoothing (median filter)
   - Detect transition zones with confidence scores
6. **Export results** - Save CSV and JSON outputs
7. **Generate visualizations** - Plot 5 representative transects with direction-aware axes

## Landcover Classes

- **ALL_LAND**: High brightness, low variability, low NDVI
- **OCEAN**: High NDWI, blue >= red, low brightness
- **DRY_BEACH**: Moderate brightness, high NIR ratio
- **VEG_DUNES**: High variability, oscillations, positive NDVI
- **WAVE_CRESTS**: High variability, oscillations, low NDVI
- **UNKNOWN**: No rules matched

## Transition Detection

Transitions are detected using multiple criteria:
1. Sharp NIR drop (>20 units over 3-5 points)
2. NDWI increase (negative to positive)
3. Brightness decrease (>30 units)
4. Spectral angle change (>30 degrees)
5. Class change in smoothed classification

Each transition includes a confidence score (0-1) based on magnitude and context validation.

## Direction Detection & Plotting

The system automatically detects transect orientation from the **first transect** by comparing eastings:

- **West-to-East transects**: Start point has lower easting than end point
  - Distance plotted from 0m (west) to max (east)
  - X-axis label: "Distance from West (m)"

- **East-to-West transects**: Start point has higher easting than end point
  - Distance plotted from -max (east) to 0m (west)
  - X-axis label: "Distance to West (m)"

All transects are plotted consistently with **west as the reference point (0m)**, regardless of their digitized direction.

## Performance Optimization

- **Lazy loading**: Rasters loaded on-demand per transect
- **Spatial indexing**: Fast intersection queries
- **Memory efficiency**: O(single raster) instead of O(all rasters)
- **Selective visualization**: Process all, visualize subset
- **Clean console output**: Only warnings/errors shown in terminal, full logs in file
- **Vectorized operations**: NumPy/pandas for efficient feature computation

## Troubleshooting

### CRS Mismatch Error
```
ValueError: CRS mismatch: raster has EPSG:32610, expected EPSG:32611
```
**Solution**: Ensure all rasters share the same CRS. Reproject rasters if needed.

### No Overlapping Rasters
```
ValueError: No rasters overlap with transect geometry
```
**Solution**: Check that transect GeoJSON and rasters cover the same geographic area. Verify CRS compatibility.

### Missing TransectID
```
ValueError: GeoJSON must contain 'TransectID' field
```
**Solution**: Ensure your GeoJSON FeatureCollection has a `TransectID` property for each feature.

## References

See `INPUT/beach_spectral_spec.md` for detailed implementation specifications.

## License

MIT License

## Contact

For questions or issues, please contact the development team.
