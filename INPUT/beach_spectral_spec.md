# Spectral Transect Classification System - Implementation Guide

## Project Overview

Build a Python-based system to automatically classify spectral profiles from cross-shore transects and identify transition zones (beach-water interfaces). The system analyzes multi-band spectral data (Red, Green, Blue, NIR) along transects from land to sea.

## Input Data Format

### Primary Inputs

1. **4-Band GeoTIFF Rasters**
   - Band order: Red (1), Green (2), Blue (3), NIR (4)
   - Assumed to share the same CRS across all rasters
   - May span multiple tiles covering the study area
   - Spatial reference: Any projected CRS (e.g., UTM)

2. **GeoJSON Transect File**
   - Format: GeoJSON FeatureCollection
   - Geometry: LineString (one per transect)
   - Required property: `TransectID` (integer/string identifier)
   - Direction: Assumed consistent (west to east)
   - CRS: Any valid CRS (will be reprojected to match rasters if needed)

### Data Loading Strategy

**Spatial Indexing + Lazy Loading:**
- Build raster spatial index (bounding boxes) without loading pixel data
- Process transects sequentially
- For each transect, load only rasters that intersect its geometry
- Unload rasters after processing to minimize memory usage
- Memory footprint: O(single raster) instead of O(all rasters)

### Sampling Strategy

**Fixed-Interval Sampling:**
- Sample spectral values every 1 meter along each transect
- Use bilinear interpolation for sub-pixel accuracy
- Extract [Red, Green, Blue, NIR] values at each sample point
- Record distance from transect start (west end)
- Output format: DataFrame with columns `[TransectID, distance, red, green, blue, nir]`

## Core Requirements

### 1. Feature Extraction Module

Create a `SpectralFeatures` class that calculates:

**Spectral Indices:**
- `NDVI = (NIR - Red) / (NIR + Red)` - vegetation index
- `NDWI = (Green - NIR) / (Green + NIR)` - water index  
- `Brightness = mean(Red, Green, Blue, NIR)` - overall reflectance
- `NIR_ratio = NIR / mean(Red, Green, Blue)` - NIR prominence
- `Blue_red_ratio = Blue / Red` - water indicator

**Statistical Features (sliding window, size=5-10 points):**
- `variability = std(band_values in window)`
- `slope = linear regression slope of band in window`
- `curvature = second derivative approximation`
- `local_range = max - min in window`

**Shape Features:**
- Spectral angle between consecutive points
- Correlation between bands
- Presence of oscillations (peak detection)

### 2. Classification Module

Implement a `LandcoverClassifier` with rule-based classification:

```python
def classify_point(features):
    """
    Returns: 'ALL_LAND', 'OCEAN', 'DRY_BEACH', 'VEG_DUNES', 'WAVE_CRESTS'
    
    Rules:
    - ALL_LAND: Brightness > 140 AND variability < 15 AND NDVI < 0.3
    - OCEAN: NDWI > 0.2 AND Blue >= Red AND Brightness < 120
    - DRY_BEACH: 90 < Brightness < 140 AND NIR_ratio > 0.75 AND variability < 20
    - VEG_DUNES: variability > 25 AND has_oscillations AND NDVI > 0.2
    - WAVE_CRESTS: Similar to VEG_DUNES but NDVI < 0.2
    """
```

### 3. Transition Zone Detection Module

Implement a `TransitionDetector` class that identifies beach-water interfaces:

**Detection Criteria:**
1. **Sharp NIR drop:** Detect where NIR decreases by >20 units over 3-5 consecutive points
2. **NDWI increase:** Transition from NDWI < 0 to NDWI > 0.2
3. **Brightness decrease:** Drop of >30 units over short distance
4. **Spectral shape change:** Calculate angle between spectral vectors, flag changes >30°
5. **Context validation:** Verify neighboring points support beach→water sequence

**Output:** Return list of transition points with confidence scores

### 4. Spatial Context Analysis

Implement smoothing and context validation:
- Apply median filter (window=3) to reduce noise
- Check for valid landcover sequences (LAND → BEACH → OCEAN)
- Flag isolated misclassifications
- Refine boundaries using neighboring point consensus

## Implementation Structure

```
spectral_classifier/
├── __init__.py
├── config.py           # Threshold parameters, sampling config
├── data_io.py          # Raster & GeoJSON loading, CRS handling, spatial indexing
├── sampler.py          # Extract spectral values along transects with lazy loading
├── features.py         # SpectralFeatures class
├── classifier.py       # LandcoverClassifier class
├── transition.py       # TransitionDetector class
├── visualization.py    # Plotting functions
├── utils.py            # Helper functions
└── main.py             # Main pipeline orchestration
```

## Main Pipeline Logic

```python
def analyze_all_transects(raster_dir, transect_geojson, output_dir):
    # 1. Build raster spatial index
    raster_index = build_raster_index(raster_dir)

    # 2. Load transects and handle CRS
    transects = load_transects(transect_geojson)
    transects = reproject_if_needed(transects, raster_index.crs)
    transects = transects.sort_values('TransectID')

    # 3. Process each transect sequentially
    all_results = []
    for transect in transects.itertuples():
        # 3a. Find intersecting rasters
        overlapping_rasters = find_overlapping_rasters(transect.geometry, raster_index)

        # 3b. Sample spectral values (rasters loaded on-demand)
        spectral_data = sample_transect(transect, overlapping_rasters)

        # 3c. Extract features
        features = SpectralFeatures(spectral_data).compute_all()

        # 3d. Classify landcover
        classifier = LandcoverClassifier()
        landcover = classifier.classify(features)

        # 3e. Apply spatial smoothing
        landcover_smooth = apply_spatial_filter(landcover, window=3)

        # 3f. Detect transitions
        detector = TransitionDetector()
        transitions = detector.find_transitions(features, landcover_smooth)

        all_results.append({
            'transect_id': transect.TransectID,
            'data': spectral_data,
            'landcover': landcover_smooth,
            'transitions': transitions,
            'features': features
        })

    # 4. Export all results
    export_results(all_results, output_dir)

    # 5. Select 5 representative transects for visualization
    n_transects = len(transects)
    vis_indices = np.linspace(0, n_transects-1, min(5, n_transects), dtype=int)
    selected_transects = [all_results[i] for i in vis_indices]

    # 6. Generate visualizations for selected subset
    for result in selected_transects:
        plot_transect_analysis(result, output_dir)

    return all_results
```

## Visualization Requirements

Create plotting functions for individual transect analysis:

1. **Spectral profile plot:** Show all 4 bands (Red, Green, Blue, NIR) with distance on x-axis
2. **Classification overlay:** Color-code background regions by landcover type
3. **Transition markers:** Highlight detected transition zones with vertical lines and confidence scores
4. **Direction-aware plotting:** X-axis adapts based on transect direction detected from first transect

**Default Behavior:**
- Process ALL transects but visualize only 5 representative ones
- Select transects using: `np.linspace(0, n_transects-1, 5, dtype=int)` based on TransectID ordering
- This provides equidistant coverage across the alongshore range
- X-axis labeled "Distance from West (m)" or "Distance to West (m)" based on direction
- No summary statistics plots generated (only individual transect plots)

## Testing & Validation

1. **Unit tests:** Test each feature calculation with known inputs
2. **Threshold tuning:** Provide config file to adjust classification thresholds
3. **Validation metrics:** Calculate accuracy against manual annotations if provided
4. **Edge cases:** Handle missing data, extreme values, very short/long transects

## Configuration Parameters

Create a `config.py` with tunable parameters:

```python
THRESHOLDS = {
    'land_brightness': 140,
    'ocean_brightness': 120,
    'land_variability': 15,
    'veg_variability': 25,
    'ndwi_water': 0.2,
    'nir_drop': 20,
    'brightness_drop': 30,
    'window_size': 7,
}
```

## Expected Outputs

1. **CSV file:** `transect_analysis_YYYYMMDD_HHMMSS.csv`
   - Columns: TransectID, distance, red, green, blue, nir, predicted_class, transition_flag, confidence
   - Contains all sample points for all processed transects

2. **JSON summary:** `summary_YYYYMMDD_HHMMSS.json`
   - Metadata (timestamp, number of transects, processing info)
   - Per-transect summaries (class distribution, transitions, length)
   - Overall statistics across all transects

3. **Individual transect plots:** `transect_<ID>_analysis.png` (for 5 representative transects)
   - 4-band spectral profiles (R, G, B, NIR)
   - Color-coded classification background
   - Transition zone markers with confidence scores
   - Direction-aware x-axis labels

4. **Processing log:** `processing.log`
   - INFO-level messages and above saved to file
   - WARNING and ERROR messages displayed to console

## Dependencies

- numpy - numerical operations
- pandas - data handling
- matplotlib - visualization
- scipy - signal processing (find_peaks, filters)
- rasterio - GeoTIFF reading and raster operations
- geopandas - GeoJSON loading, CRS transformations, spatial operations
- shapely - geometry operations, spatial predicates
- pyproj - CRS handling and reprojection
- scikit-learn - optional for ML enhancement

## Notes for Implementation

- Handle division by zero in index calculations (use small epsilon)
- Normalize features if using ML approach later
- Consider vectorizing operations for speed
- Validate that distance array is monotonically increasing

**Logging:**
- Console (terminal): Display only WARNING and ERROR messages
- Log file: Save INFO-level messages and above (or DEBUG if verbose flag enabled)
- Provides clean console output while maintaining detailed file logs

**CRS Handling:**
- Assume all rasters share the same CRS (validate and error if mismatch detected)
- Compare transect GeoJSON CRS with raster CRS
- Reproject transects to match raster CRS if needed
- Use projected CRS for accurate distance calculations (e.g., UTM)

**Transect Processing:**
- Detect direction from first transect by comparing eastings of start/end points
- Apply consistent direction to all transects for plotting:
  - West-to-east: distances from 0 to max (westmost = 0m)
  - East-to-west: distances from -max to 0 (westmost = 0m)
- Do not assume land vs sea orientation
- Transition detection identifies all sharp spectral changes
- Sort transects by TransectID for consistent ordering

## Extension Opportunities

- Train Random Forest on manually annotated data for improved accuracy
- Add anomaly detection (Isolation Forest) for unusual spectral signatures
- Support multiple transects in batch processing
- Export results in GIS-compatible formats
- Interactive visualization for threshold tuning