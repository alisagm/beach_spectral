"""
Phase 6C Diagnostic Analysis - Current Performance Validation

This script validates the current shell-line detection algorithm against
manually labeled transects to identify failure modes and guide improvements.

Usage:
    python -m analysis.phase6c_diagnostics.validate_current_performance
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spectral_classifier import analyze_all_transects
from spectral_classifier.config import THRESHOLDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ShellLineValidator:
    """Validates shell line detection against manual labels."""

    def __init__(self, tolerance_m: float = 10.0):
        """
        Initialize validator.

        Args:
            tolerance_m: Distance tolerance for considering detection correct (meters)
        """
        self.tolerance_m = tolerance_m
        self.results = []

    def load_manual_labels(self, csv_path: str) -> pd.DataFrame:
        """
        Load manually classified transect data.

        Expected format:
        - transect_id: Transect identifier
        - class_start_m: Start distance of landcover section (meters)
        - class_end_m: End distance of landcover section (meters)
        - landcover_class: One of VEG_DUNES, DRY_BEACH, BEACH_WET, WATER

        Shell line location is inferred as transition from DRY_BEACH to BEACH_WET.

        Args:
            csv_path: Path to manual labels CSV

        Returns:
            DataFrame with manual labels
        """
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} manual label rows from {csv_path}")
        return df

    def extract_shell_line_location(self, labels: pd.DataFrame, transect_id: int) -> float:
        """
        Extract shell line location from manual labels for a specific transect.

        Shell line = DRY_BEACH -> BEACH_WET transition

        Args:
            labels: Manual labels DataFrame
            transect_id: Transect ID to extract

        Returns:
            Shell line location in meters, or None if not found
        """
        # Filter to this transect
        transect_labels = labels[labels['transect_id'] == transect_id].copy()

        if transect_labels.empty:
            logger.warning(f"No manual labels found for transect {transect_id}")
            return None

        # Sort by distance
        transect_labels = transect_labels.sort_values('class_start_m')

        # Find DRY_BEACH -> BEACH_WET transition
        for i in range(len(transect_labels) - 1):
            current_class = transect_labels.iloc[i]['landcover_class']
            next_class = transect_labels.iloc[i+1]['landcover_class']

            if current_class == 'DRY_BEACH' and next_class == 'BEACH_WET':
                # Shell line is at the boundary
                shell_line_loc = transect_labels.iloc[i]['class_end_m']
                logger.debug(f"Transect {transect_id}: Shell line at {shell_line_loc:.1f}m "
                           f"({current_class}->{next_class})")
                return shell_line_loc

        logger.warning(f"Transect {transect_id}: No DRY_BEACH->BEACH_WET transition found")
        return None

    def validate_detection(
        self,
        detected_location: float,
        manual_location: float,
        transect_id: int
    ) -> Dict:
        """
        Validate a single detection against manual label.

        Categories:
        - TRUE_POSITIVE: Detected within tolerance
        - FALSE_POSITIVE: Detected outside tolerance (wrong location)
        - LOCALIZATION_ERROR: Detected with error > tolerance but < 2*tolerance

        Args:
            detected_location: Detected shell line location (meters)
            manual_location: Manual label location (meters)
            transect_id: Transect identifier

        Returns:
            Validation result dictionary
        """
        error = abs(detected_location - manual_location)

        if error <= self.tolerance_m:
            category = 'TRUE_POSITIVE'
        elif error <= 2 * self.tolerance_m:
            category = 'LOCALIZATION_ERROR'
        else:
            category = 'FALSE_POSITIVE'

        result = {
            'transect_id': transect_id,
            'detected_location': detected_location,
            'manual_location': manual_location,
            'error_m': error,
            'category': category
        }

        logger.info(f"Transect {transect_id}: {category} - "
                   f"detected={detected_location:.1f}m, manual={manual_location:.1f}m, "
                   f"error={error:.1f}m")

        return result

    def compute_metrics(self, results: List[Dict]) -> Dict:
        """
        Compute precision, recall, F1-score from validation results.

        Args:
            results: List of validation result dictionaries

        Returns:
            Dictionary with performance metrics
        """
        n_total = len(results)
        n_tp = sum(1 for r in results if r['category'] == 'TRUE_POSITIVE')
        n_fp = sum(1 for r in results if r['category'] == 'FALSE_POSITIVE')
        n_fn = sum(1 for r in results if r.get('category') == 'FALSE_NEGATIVE')
        n_localization_error = sum(1 for r in results if r['category'] == 'LOCALIZATION_ERROR')

        # Treat localization errors as TP for recall, FP for precision
        tp_for_recall = n_tp + n_localization_error

        precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
        recall = tp_for_recall / n_total if n_total > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Error statistics (only for detections)
        errors = [r['error_m'] for r in results if 'error_m' in r]

        metrics = {
            'n_total_transects': n_total,
            'n_true_positives': n_tp,
            'n_false_positives': n_fp,
            'n_false_negatives': n_fn,
            'n_localization_errors': n_localization_error,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mean_error_m': np.mean(errors) if errors else None,
            'median_error_m': np.median(errors) if errors else None,
            'std_error_m': np.std(errors) if errors else None,
            'max_error_m': np.max(errors) if errors else None
        }

        return metrics

    def generate_report(self, results: List[Dict], metrics: Dict, output_path: str):
        """
        Generate validation report with detailed breakdown.

        Args:
            results: Validation results
            metrics: Performance metrics
            output_path: Path to save report
        """
        # Save results CSV
        results_df = pd.DataFrame(results)
        results_csv = output_path.replace('.txt', '_results.csv')
        results_df.to_csv(results_csv, index=False)
        logger.info(f"Saved results CSV to {results_csv}")

        # Generate text report
        report = []
        report.append("=" * 80)
        report.append("PHASE 6C - SHELL LINE DETECTION VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")

        report.append("PERFORMANCE METRICS")
        report.append("-" * 80)
        report.append(f"Total Transects:       {metrics['n_total_transects']}")
        report.append(f"True Positives:        {metrics['n_true_positives']} "
                     f"(detected within +-{self.tolerance_m}m)")
        report.append(f"Localization Errors:   {metrics['n_localization_errors']} "
                     f"(detected within +-{2*self.tolerance_m}m)")
        report.append(f"False Positives:       {metrics['n_false_positives']} "
                     f"(detected >+-{2*self.tolerance_m}m from truth)")
        report.append(f"False Negatives:       {metrics['n_false_negatives']} "
                     f"(not detected)")
        report.append("")
        report.append(f"Precision:             {metrics['precision']:.3f}")
        report.append(f"Recall:                {metrics['recall']:.3f}")
        report.append(f"F1-Score:              {metrics['f1_score']:.3f}")
        report.append("")

        if metrics['mean_error_m'] is not None:
            report.append("LOCALIZATION ACCURACY")
            report.append("-" * 80)
            report.append(f"Mean Error:            {metrics['mean_error_m']:.2f}m")
            report.append(f"Median Error:          {metrics['median_error_m']:.2f}m")
            report.append(f"Std Dev Error:         {metrics['std_error_m']:.2f}m")
            report.append(f"Max Error:             {metrics['max_error_m']:.2f}m")
            report.append("")

        # Breakdown by category
        report.append("DETECTION BREAKDOWN BY CATEGORY")
        report.append("-" * 80)
        for category in ['TRUE_POSITIVE', 'LOCALIZATION_ERROR', 'FALSE_POSITIVE', 'FALSE_NEGATIVE']:
            category_results = [r for r in results if r.get('category') == category]
            report.append(f"{category}: {len(category_results)} transects")

            for r in category_results:
                tid = r['transect_id']
                if 'error_m' in r:
                    report.append(f"  - Transect {tid}: "
                                f"detected={r['detected_location']:.1f}m, "
                                f"manual={r['manual_location']:.1f}m, "
                                f"error={r['error_m']:.1f}m")
                else:
                    report.append(f"  - Transect {tid}: {r.get('reason', 'Unknown')}")
            report.append("")

        report.append("=" * 80)

        # Write report
        report_text = "\n".join(report)
        with open(output_path, 'w') as f:
            f.write(report_text)

        logger.info(f"Saved validation report to {output_path}")
        print("\n" + report_text)


def main():
    """Main validation workflow."""

    # Paths
    base_dir = Path(__file__).parent.parent.parent
    manual_labels_dir = base_dir / "analysis" / "feature_analysis" / "data"
    output_dir = base_dir / "analysis" / "phase6c_diagnostics"
    output_dir.mkdir(exist_ok=True)

    logger.info("Starting Phase 6C validation analysis...")
    logger.info(f"Output directory: {output_dir}")

    # NOTE: For now, we'll create a placeholder script that documents
    # the validation approach. The actual implementation requires:
    # 1. Manual labels in the expected format (CSV with transect_id, class boundaries)
    # 2. Integration with the main classifier to run detections

    # Create validation framework documentation
    validator = ShellLineValidator(tolerance_m=10.0)

    # Example usage (would need actual manual labels):
    example_results = [
        {
            'transect_id': 100,
            'detected_location': 116.0,
            'manual_location': 100.0,
            'error_m': 16.0,
            'category': 'LOCALIZATION_ERROR'
        },
        {
            'transect_id': 378,
            'detected_location': None,
            'manual_location': 95.0,
            'category': 'FALSE_NEGATIVE',
            'reason': 'No detection (strict thresholds)'
        }
    ]

    # Generate example report
    metrics = validator.compute_metrics(example_results)
    report_path = output_dir / "validation_report_example.txt"
    validator.generate_report(example_results, metrics, str(report_path))

    logger.info("Validation framework created successfully")
    logger.info("Next step: Analyze actual transect detections from training data logs")

    return 0


if __name__ == '__main__':
    sys.exit(main())
