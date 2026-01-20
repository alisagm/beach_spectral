"""
Configuration parameters for spectral transect classification system.

Supports 3-band (RGB or CIR) and 4-band (RGBN) imagery with mode-specific
thresholds for shell line detection.
"""

# Sampling configuration
SAMPLING_INTERVAL = 1.0  # meters - distance between sample points along transect

# Visualization configuration
NUM_TRANSECTS_TO_VISUALIZE = 5  # Number of representative transects to plot

# Boundary detection output configuration
# Controls which boundary types are returned by main processing pipeline
# Options:
#   'shore_only': BEACH_DRY->BEACH_WET only (swash line/shell line)
#   'waterline': Shore + BEACH_WET->WATER (shore + waterline boundaries)
#   'all': All boundaries including VEG_DUNES->BEACH_DRY
DEFAULT_BOUNDARY_TYPES = 'shore_only'

# Edge buffer for transition detection (meters)
# Transitions within this distance from transect start/end are rejected
EDGE_BUFFER_M = 10.0  # Increased from 5.0 to eliminate edge effect false positives

# Band indices (0-indexed for numpy array access)
BAND_INDICES = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'nir': 3
}

# ============================================================================
# BAND DETECTION CONFIGURATION
# ============================================================================
# Heuristics for distinguishing CIR [NIR,R,G] from RGB [R,G,B] in 3-band imagery
BAND_DETECTION = {
    # Variance ratio threshold: Band1_StdDev / Band2_StdDev
    # CIR: NIR (band1) has higher variance than Red (band2) → ratio > threshold
    # RGB: Red (band1) has similar/lower variance than Green (band2) → ratio ≤ threshold
    # Empirical values from PAIS imagery: CIR ≈ 1.14, RGB ≈ 0.97
    'cir_variance_ratio_threshold': 1.05,
    
    # Minimum valid samples needed for reliable detection
    'min_valid_samples': 100,
    
    # Sample size for band detection (random pixels)
    'sample_size': 10000,
}

# Classification thresholds
# Updated based on training data analysis (seed 321197, n=2602 points, 10 transects)
# See: training_output/CLASSIFICATION_IMPROVEMENT_PLAN.md
THRESHOLDS = {
    # NIR ratio - PRIMARY discriminator (most reliable feature)
    # Empirical ranges: Water=0.21+-0.10, WetBeach=0.56+-0.13, DryBeach/Veg=0.89+-0.04
    'nir_ratio_water_max': 0.35,        # Water has very low NIR absorption
    'nir_ratio_wet_beach_min': 0.40,    # Wet beach intermediate (water-saturated sand)
    'nir_ratio_wet_beach_max': 0.70,
    'nir_ratio_dry_min': 0.80,          # Dry beach and veg dunes both high

    # Brightness - for distinguishing dry beach from veg dunes
    # Empirical: DryBeach=209+-6, VegDunes=145+-37, WetBeach=147+-20, Water=106+-35
    'dry_beach_brightness_min': 195,    # Very bright, uniform sand
    'veg_dune_brightness_max': 190,     # Lower than dry beach, more variable
    'wet_beach_brightness_max': 175,    # Moderate brightness

    # Variability - distinguishes veg dunes (rough) from dry beach (smooth)
    # Empirical: VegDunes=28+-11, DryBeach=4+-2, WetBeach=11+-5, Water=31+-7
    'veg_variability_min': 15,          # High variability from vegetation/dune structure
    'dry_beach_variability_max': 10,    # Very low variability (smooth, uniform)

    # NDWI - water and wet beach detection (secondary confirmation)
    # Empirical: Water=0.75+-0.12, WetBeach=0.37+-0.13, DryBeach=0.08+-0.03
    'ndwi_water_min': 0.60,             # Strong water signal
    'ndwi_wet_beach_min': 0.25,         # Elevated but not as high as water

    # Blue/Red ratio - water confirmation (blue > red for water)
    # Empirical: Water=1.06+-0.16, Land classes=0.80-0.96
    'blue_red_water_min': 0.95,         # Water tends toward blue

    # NDVI - DEPRECATED as primary classifier (empirical values all negative!)
    # Kept for backwards compatibility and optional confirmation
    # Empirical: Water=-0.70+-0.11, WetBeach=-0.39+-0.13, DryBeach=-0.08+-0.03, VegDunes=-0.12+-0.05
    'ndvi_water_max': -0.50,            # Very negative for water

    # Legacy thresholds (deprecated, kept for compatibility)
    'land_brightness_min': 140,
    'ocean_brightness_max': 120,
    'land_variability_max': 15,
    'nir_ratio_beach_min': 0.75,       # Replaced by nir_ratio_dry_min

    # Transition detection - derivative-based (optimized from validation)
    'nir_derivative_threshold': -4.38,  # Optimal NIR derivative threshold (units/m) from validation
    'derivative_sustainability_points': 3,  # Minimum consecutive points below threshold for sustained drop
    'multi_band_consensus_required': 2,  # Minimum spectral bands that must agree (2-4)

    # Derivative threshold scaling factors (relative to NIR threshold)
    # Used to compute thresholds for other bands: threshold_band = nir_threshold * scaling_factor
    'derivative_scaling': {
        'red': 0.85,      # Red band slightly less sensitive than NIR
        'green': 0.80,    # Green band moderately less sensitive
        'blue': 0.75,     # Blue band less sensitive to beach transitions
    },

    # PHASE 2D: Boundary-type-specific thresholds and configuration
    # Different boundary types require different detection approaches
    # PHASE 2 FIX: Tightened thresholds to reduce false positives
    # PHASE 6: Enhanced with empirical data from 20-transect feature analysis
    'boundary_thresholds': {
        # VEG_DUNES->BEACH_DRY: Inflection point detection
        'veg_boundaries': {
            'second_deriv_threshold': 1.0,    # PHASE 2 FIX: Increased from 0.5 (only significant inflections)
            'trend_change_window': 10,         # Window (points) for trend analysis around inflection
            'min_nir_change': 20,              # PHASE 2 FIX: Increased from 10 (require larger NIR difference)
        },
        # BEACH_WET->WATER: RGB foam peak detection
        'surf_zone': {
            'rgb_peak_prominence': 10.0,       # PHASE 2 FIX: Increased from 5.0 (only strong foam peaks)
            'nir_threshold': -2.0,             # NIR drop requirement for confirmation
            'require_nir_drop': True,          # PHASE 2 FIX: Require NIR drop (no more foam-only detection)
        },
        # BEACH_DRY->BEACH_WET: Derivative magnitude (existing approach)
        # PHASE 6C: Relaxed thresholds based on Phase 6C diagnostic analysis
        # Previous Phase 6A/6B thresholds were too strict and rejected 25% of known shell lines
        'dry_wet': {
            'nir_threshold': -8.0,             # PHASE 6: Tightened from -4.0 (empirical: -13.8+-4.0)
            'require_sustainability': True,    # Check for sustained drop (confidence boost)
            'require_consensus': True,         # Check multi-band consensus (confidence boost)
            # PHASE 3: R/G ratio validation for shell-line detection
            'use_rg_ratio': True,              # Enable R/G ratio pattern validation
            'rg_ratio_threshold': 0.98,        # Expected R/G ratio for dry->wet (R ~ G)
            'rg_ratio_tolerance': 0.15,        # Acceptable variance from threshold (+-0.15)
            # PHASE 6C: Relaxed thresholds based on empirical 5th percentile
            # Diagnostic analysis showed strict thresholds rejected valid shell lines:
            # - nir_before_min=160 rejected 25% of shell lines (empirical median: 173, 5th%: 139)
            # - brightness_before_min=190 rejected 15% (empirical median: 203, 5th%: 183)
            # - min_nir_drop_absolute=40 was acceptable (90% coverage)
            'min_nir_drop_absolute': 39.0,     # PHASE 6C: Relaxed to 5th percentile (was 40)
            'brightness_before_min': 175,      # PHASE 6C: Relaxed from 190 (5th%=183, using 175 for safety)
            'nir_before_min': 135,             # PHASE 6C: Relaxed from 160 (5th%=139, using 135 for safety)
            'variability_ratio_min': 1.5,      # PHASE 6C: Relaxed from 2.0 (some shell lines have lower ratios)
            'expected_location_min': 40,       # PHASE 6: Expected zone (meters from start)
            'expected_location_max': 120,      # PHASE 6: Expected zone (meters from end)
            # PHASE 6C: VEG->DRY discrimination parameters
            'expected_location_veg_dry_min': 40,  # VEG->DRY typically occurs here
            'expected_location_veg_dry_max': 70,  # Penalize candidates in this zone
        },
    },

    # PHASE 2 FIX: Minimum confidence for final boundary acceptance
    'min_boundary_confidence': 0.75,  # Increased from 0.60 to reduce false positives

    # ========================================================================
    # RGB-ONLY MODE THRESHOLDS
    # ========================================================================
    # Used when NIR band is not available (3-band RGB imagery)
    # Brightness-based detection is less reliable than NIR-based detection
    'rgb_thresholds': {
        'brightness_derivative_threshold': -5.0,   # Less strict than NIR -8.0
        'min_brightness_drop_absolute': 25.0,      # Less than NIR 39.0
        'brightness_before_min': 150,              # Less strict than NIR 175
        'variability_ratio_min': 1.3,              # Less strict than NIR 1.5
        'max_confidence': 0.70,                    # Cap for RGB-only detections
        'min_acceptance_confidence': 0.45,         # Lower than NIR 0.60
    },

    # ========================================================================
    # CIR MODE THRESHOLDS
    # ========================================================================
    # Used for 3-band CIR [NIR, Red, Green] imagery
    # Still has NIR so uses full NIR-based detection, but missing Blue band
    'cir_thresholds': {
        'max_confidence': 0.90,                    # Slightly lower than 4-band (0.95) due to missing Blue
        'min_acceptance_confidence': 0.55,         # Same as 4-band
    },

    # PHASE 4: False positive filtering enhancements
    'min_nir_variability': 15.0,  # Minimum NIR std dev for valid within-zone transitions

    # Multi-scale smoothing windows for different boundary types
    'derivative_windows': {
        'veg_boundaries': 7,      # VEG_DUNES->BEACH_DRY (coarse, smooth vegetation noise)
        'surf_zone': 9,            # BEACH_WET->WATER (coarse, for foam detection)
        'dry_wet': 5,              # BEACH_DRY->BEACH_WET (medium, preserve sharpness)
    },

    # ========================================================================
    # DEPRECATED PARAMETERS (Removed in Phase 8)
    # ========================================================================
    # The following legacy transition detection parameters were removed as they
    # are no longer used by the Phase 2+ boundary-type-specific detection methods:
    # - 'nir_drop': 20 (NIR decrease threshold for legacy _detect_nir_drop)
    # - 'nir_drop_window': 5 (window for legacy NIR drop detection)
    # - 'brightness_drop': 30 (brightness decrease for legacy _detect_brightness_drop)
    # - 'spectral_angle_threshold': 30 (degrees, for legacy _detect_spectral_angle_change)
    # - 'ndwi_transition': 0.2 (NDWI threshold for legacy _detect_ndwi_transition)
    # - 'ndvi_veg_min': 0.2 (unreliable, NDVI values all negative in dataset)
    # - 'ndvi_land_max': 0.3 (unreliable, NDVI values all negative in dataset)
    #
    # Historical values available in git history before Phase 8.
    # ========================================================================

    # Feature calculation windows
    'window_size': 7,            # Points for statistical features
    'smoothing_window': 3,       # Points for spatial filtering

    # Sequence constraints for monotonic smoothing
    # Note: Relaxed for transition-based approach (uses NIR derivatives for boundaries)
    # PHASE 6D: Disable monotonic smoothing by default due to 93% UNKNOWN collapse
    'enable_monotonic_smoothing': False,  # PHASE 6D: Disabled due to catastrophic class collapse
    'min_category_span_m': 2.5,  # Minimum distance span for each category (meters)
}

# Small epsilon to prevent division by zero
EPSILON = 1e-10

# Landcover class labels
# Expected sequence from land to water: VEG_DUNES -> DRY_BEACH -> BEACH_WET -> WATER
LANDCOVER_CLASSES = [
    'VEG_DUNES',       # Vegetated dunes (optional, may not be present)
    'DRY_BEACH',       # Dry sand
    'BEACH_WET',       # Wet sand (transitional zone)
    'WATER',           # Open water
    'WAVE_CRESTS',     # Wave crests (optional refinement of water)
    'UNKNOWN'          # Unclassified (can appear anywhere)
]

# Landcover colors for visualization (RGB)
LANDCOVER_COLORS = {
    'VEG_DUNES': '#228B22',     # Green
    'DRY_BEACH': '#F4A460',     # Sandy brown
    'BEACH_WET': '#CD853F',     # Peru/darker tan - wet sand
    'WATER': '#1E90FF',         # Blue
    'WAVE_CRESTS': '#00CED1',   # Turquoise
    'UNKNOWN': '#808080'        # Gray
}

# Output configuration
OUTPUT_CSV_COLUMNS = [
    'TransectID',
    'distance',
    'red',
    'green',
    'blue',
    'nir',
    'predicted_class',
    'transition_flag',
    'confidence'
]

# Logging configuration
VERBOSE = True
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'