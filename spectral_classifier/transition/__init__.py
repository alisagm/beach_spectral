"""
Transition detection module for beach-water interface identification.

This module provides spectral-based boundary detection focused on
identifying the shell line (BEACH_DRY -> BEACH_WET transition).

Primary class:
    TransitionDetector - Main orchestration class for boundary detection

Submodules:
    - nir: NIR-based detection methods (4-band and CIR imagery)
    - rgb: RGB fallback detection methods (degraded mode)
    - shell_line: Selection, filtering, and TransitionDetector class

Example usage:
    from spectral_classifier.transition import TransitionDetector
    
    detector = TransitionDetector()
    transitions = detector.find_transitions(features, landcover)
    
    # Filter to shell line only
    shell_lines = TransitionDetector.filter_by_boundary_type(
        transitions, 'shore_only'
    )
"""

from .shell_line import TransitionDetector

from .pelt import detect_shell_line_pelt

# Expose detection methods for advanced users / other applications
from .nir import (
    detect_dry_wet_boundaries,
    detect_vegetation_boundaries,
    detect_surf_zone_boundaries,
    check_sustainability,
    check_multi_band_consensus,
    validate_rg_pattern_for_boundary_type,
)
from .rgb import (
    detect_dry_wet_boundaries_rgb,
    check_multi_band_consensus_rgb,
)

__all__ = [
    # Main class
    'TransitionDetector',

    # PELT mode
    'detect_shell_line_pelt',
    
    # NIR detection
    'detect_dry_wet_boundaries',
    'detect_vegetation_boundaries',
    'detect_surf_zone_boundaries',
    'check_sustainability',
    'check_multi_band_consensus',
    'validate_rg_pattern_for_boundary_type',
    
    # RGB detection  
    'detect_dry_wet_boundaries_rgb',
    'check_multi_band_consensus_rgb',
]