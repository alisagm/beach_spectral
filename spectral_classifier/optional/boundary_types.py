"""
Optional boundary type detection module.

This module contains detection methods for NON-shell-line boundaries:
- VEG_DUNES -> BEACH_DRY (vegetation boundary)
- BEACH_WET -> WATER (surf zone / waterline)

These boundaries are NOT required for the primary output (shell line GeoJSON)
but may be useful for:
- Comprehensive beach zone analysis
- Vegetation monitoring applications
- Surf zone studies

Usage:
    These detectors are accessed through TransitionDetector.find_transitions()
    by setting include_optional_boundaries=True, or can be called directly
    for custom workflows.
    
Note:
    The detection methods themselves are implemented in transition/nir.py
    as they require NIR data. This module serves as documentation and
    provides convenience wrappers.
"""

from typing import List, Dict

import pandas as pd

from ..transition.nir import (
    detect_vegetation_boundaries as _detect_veg_nir,
    detect_surf_zone_boundaries as _detect_surf_nir,
)
from ..utils.data_io import BAND_CONFIG_4BAND, BAND_CONFIG_CIR, BAND_CONFIG_RGB
from ..config import THRESHOLDS


def detect_all_boundaries(
    features: pd.DataFrame,
    landcover: pd.DataFrame = None,
    band_mode: str = None,
    thresholds: Dict = None
) -> Dict[str, List[Dict]]:
    """
    Detect all boundary types: vegetation, shell line, and waterline.
    
    This is a convenience function for getting a complete set of boundaries.
    For shell-line-only detection, use TransitionDetector directly.
    
    Args:
        features: DataFrame with spectral features
        landcover: Optional DataFrame with classified landcover
        band_mode: Band configuration ('4band', 'cir', 'rgb')
        thresholds: Optional custom thresholds
        
    Returns:
        Dictionary with keys:
            - 'vegetation': List of VEG_DUNES->BEACH_DRY transitions
            - 'shell_line': List of BEACH_DRY->BEACH_WET transitions
            - 'waterline': List of BEACH_WET->WATER transitions
    """
    if thresholds is None:
        thresholds = THRESHOLDS
        
    if band_mode is None:
        # Auto-detect from features
        from ..utils.data_io import detect_band_mode_from_dataframe
        band_mode = detect_band_mode_from_dataframe(features)
    
    has_nir = band_mode in [BAND_CONFIG_4BAND, BAND_CONFIG_CIR]
    
    results = {
        'vegetation': [],
        'shell_line': [],
        'waterline': []
    }
    
    if has_nir:
        # Vegetation boundaries
        results['vegetation'] = _detect_veg_nir(
            features, landcover, band_mode, thresholds
        )
        
        # Waterline boundaries
        results['waterline'] = _detect_surf_nir(
            features, landcover, band_mode, thresholds
        )
    else:
        # RGB-only: limited capability
        # Surf detection still works (uses RGB peaks)
        results['waterline'] = _detect_surf_nir(
            features, landcover, band_mode, thresholds
        )
    
    # Shell line detection is handled by TransitionDetector
    # which uses appropriate method based on band_mode
    from ..transition import TransitionDetector
    detector = TransitionDetector(thresholds)
    
    all_transitions = detector.find_transitions(
        features, landcover, include_optional_boundaries=False
    )
    results['shell_line'] = [
        t for t in all_transitions 
        if TransitionDetector.is_shore_boundary(t)
    ]
    
    return results


# ============================================================================
# BOUNDARY TYPE DESCRIPTIONS
# ============================================================================

BOUNDARY_TYPES = {
    'veg_inflection': {
        'name': 'Vegetation Boundary',
        'transition': 'VEG_DUNES -> BEACH_DRY',
        'detection_method': 'Second derivative inflection point',
        'requires_nir': True,
        'reliability': 'Medium - dependent on vegetation presence',
        'notes': 'May not be present if transect starts on bare sand'
    },
    'dry_wet_derivative': {
        'name': 'Shell Line',
        'transition': 'BEACH_DRY -> BEACH_WET',
        'detection_method': 'NIR/brightness derivative magnitude',
        'requires_nir': False,  # RGB fallback available
        'reliability': 'High with NIR, Medium with RGB-only',
        'notes': 'Primary output - always detected'
    },
    'surf_foam': {
        'name': 'Waterline',
        'transition': 'BEACH_WET -> WATER',
        'detection_method': 'RGB foam peak detection',
        'requires_nir': False,  # Works with RGB
        'reliability': 'Medium - depends on wave conditions',
        'notes': 'May be absent in calm conditions'
    }
}


def get_boundary_description(boundary_type: str) -> Dict:
    """
    Get description and metadata for a boundary type.
    
    Args:
        boundary_type: Boundary type string from transition dict
        
    Returns:
        Dictionary with boundary metadata
    """
    return BOUNDARY_TYPES.get(boundary_type, {
        'name': 'Unknown',
        'transition': 'Unknown',
        'detection_method': 'Unknown',
        'requires_nir': False,
        'reliability': 'Unknown',
        'notes': ''
    })