"""
Quick test - run detection on a single transect with current (relaxed) thresholds.
"""

import pandas as pd
import sys
import logging
from pathlib import Path

# Setup logging to see debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(name)s - %(message)s'
)

from spectral_classifier.transition import TransitionDetector
from spectral_classifier.features import SpectralFeatures
from spectral_classifier.config import THRESHOLDS

# Load transect 100 data
spectral_data = pd.read_csv('analysis/training_output/seed_321197/run05/sampled_spectral_data.csv')
transect_100 = spectral_data[spectral_data['TransectID'] == 100].copy()

print("="*80)
print("TEST: Single Transect Detection (Transect 100)")
print("="*80)
print(f"\nData points: {len(transect_100)}")
print(f"Distance range: {transect_100['distance'].min():.1f}m to {transect_100['distance'].max():.1f}m")
print(f"\nManual shell line location: 100.0m")

# Check current thresholds
print(f"\nCurrent Phase 6C thresholds:")
dry_wet_config = THRESHOLDS.get('boundary_thresholds', {}).get('dry_wet', {})
print(f"  min_nir_drop_absolute: {dry_wet_config.get('min_nir_drop_absolute', 'N/A')}")
print(f"  brightness_before_min: {dry_wet_config.get('brightness_before_min', 'N/A')}")
print(f"  nir_before_min: {dry_wet_config.get('nir_before_min', 'N/A')}")
print(f"  variability_ratio_min: {dry_wet_config.get('variability_ratio_min', 'N/A')}")

# Extract features
print(f"\nExtracting features...")
feat_extractor = SpectralFeatures(transect_100)
features = feat_extractor.compute_all()

print(f"Features extracted: {list(features.columns)}")

# Run detection
print(f"\nRunning transition detection...")
detector = TransitionDetector(THRESHOLDS)
transitions = detector.find_transitions(features, transect_100)

print(f"\n" + "="*80)
print(f"DETECTION RESULTS")
print(f"="*80)
print(f"Total transitions detected: {len(transitions)}")

if transitions:
    for trans in transitions:
        print(f"\n  Location: {trans['distance']:.1f}m")
        print(f"  Type: {trans.get('type', 'unknown')}")
        print(f"  Confidence: {trans.get('confidence', 0):.2f}")
        print(f"  NIR drop: {trans.get('nir_drop_abs', 'N/A')}")
        print(f"  Detection method: {trans.get('detection_method', 'unknown')}")
else:
    print("\nNO TRANSITIONS DETECTED!")
    print("\nThis means:")
    print("  1. Either no candidates were found in _detect_dry_wet_boundaries")
    print("  2. Or candidates were rejected by _filter_transitions")
    print("  3. Or candidates were rejected by _select_best_shell_line_candidate")
    print("\nCheck the DEBUG log output above for rejection reasons.")

