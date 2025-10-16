"""
Utility functions for spectral transect classification system.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = True, log_file: Path = None):
    """
    Configure logging for the application.

    Console (terminal): Shows only WARNING and ERROR
    File: Shows INFO and above (or DEBUG if verbose=True)

    Args:
        verbose: Enable debug-level logging in file
        log_file: Optional path to log file
    """
    # Set file logging level
    file_level = logging.DEBUG if verbose else logging.INFO

    # Get root logger and clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set to lowest level
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler - only WARNING and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler - INFO/DEBUG and above
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def export_results_to_csv(
    all_results: List[Dict],
    output_dir: Path
) -> Path:
    """
    Export all transect results to CSV file.

    Args:
        all_results: List of transect analysis results
        output_dir: Directory to save CSV

    Returns:
        Path to saved CSV file
    """
    logger.debug("Exporting results to CSV")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = []

    for result in all_results:
        transect_id = result['transect_id']
        data = result['data']
        landcover = result['landcover']

        # Merge data and classification
        merged = data.copy()
        merged['predicted_class'] = landcover['predicted_class'].values
        merged['confidence'] = landcover['confidence'].values

        # Add transition flags
        if 'transition_flag' in landcover.columns:
            merged['transition_flag'] = landcover['transition_flag'].values
        else:
            merged['transition_flag'] = False

        all_data.append(merged)

    # Concatenate all transects
    combined = pd.concat(all_data, ignore_index=True)

    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'transect_analysis_{timestamp}.csv'

    combined.to_csv(output_path, index=False)

    logger.info(f"Exported {len(combined)} points to {output_path}")

    return output_path


def export_summary_json(
    all_results: List[Dict],
    output_dir: Path,
    processing_metadata: Dict = None
) -> Path:
    """
    Export summary statistics as JSON.

    Args:
        all_results: List of transect analysis results
        output_dir: Directory to save JSON
        processing_metadata: Optional metadata about processing

    Returns:
        Path to saved JSON file
    """
    logger.debug("Exporting summary to JSON")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_transects': len(all_results),
            'processing_info': processing_metadata or {}
        },
        'transects': []
    }

    for result in all_results:
        transect_summary = {
            'transect_id': str(result['transect_id']),
            'num_points': len(result['data']),
            'length_m': float(result['data']['distance'].max()),
            'class_distribution': result['landcover']['predicted_class'].value_counts().to_dict(),
            'transitions': [
                {
                    'distance': float(t['distance']),
                    'type': t['type'],
                    'confidence': float(t['confidence'])
                }
                for t in result.get('transitions', [])
            ],
            'num_transitions': len(result.get('transitions', []))
        }

        summary['transects'].append(transect_summary)

    # Overall statistics
    all_classes = pd.concat([
        r['landcover']['predicted_class']
        for r in all_results
    ])
    summary['overall_class_distribution'] = all_classes.value_counts().to_dict()

    total_transitions = sum(len(r.get('transitions', [])) for r in all_results)
    summary['total_transitions'] = total_transitions

    # Save to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'summary_{timestamp}.json'

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Exported summary to {output_path}")

    return output_path


def select_representative_transects(
    all_results: List[Dict],
    num_to_select: int = 5
) -> List[Dict]:
    """
    Select representative transects for visualization.

    Args:
        all_results: List of all transect results
        num_to_select: Number of transects to select

    Returns:
        List of selected transect results
    """
    n_transects = len(all_results)

    if n_transects <= num_to_select:
        logger.info(f"Selecting all {n_transects} transects for visualization")
        return all_results

    # Select evenly-spaced indices
    indices = np.linspace(0, n_transects - 1, num_to_select, dtype=int)

    selected = [all_results[i] for i in indices]

    logger.info(
        f"Selected {len(selected)} representative transects: "
        f"IDs = {[r['transect_id'] for r in selected]}"
    )

    return selected


def detect_transect_direction(transect_geometry) -> str:
    """
    Detect if transect runs west-to-east or east-to-west.

    Args:
        transect_geometry: Shapely LineString geometry

    Returns:
        'west_to_east' if start point is westmost, 'east_to_west' otherwise
    """
    coords = list(transect_geometry.coords)
    start_x = coords[0][0]  # Easting of start point
    end_x = coords[-1][0]   # Easting of end point

    if start_x < end_x:
        return 'west_to_east'
    else:
        return 'east_to_west'


def validate_output_directory(output_dir: Path) -> Path:
    """
    Validate and create output directory if needed.

    Args:
        output_dir: Path to output directory

    Returns:
        Validated Path object

    Raises:
        ValueError: If path exists and is not a directory
    """
    output_dir = Path(output_dir)

    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"{output_dir} exists but is not a directory")

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def calculate_processing_stats(all_results: List[Dict]) -> Dict:
    """
    Calculate processing statistics across all transects.

    Args:
        all_results: List of transect analysis results

    Returns:
        Dictionary with processing statistics
    """
    total_points = sum(len(r['data']) for r in all_results)
    total_length = sum(r['data']['distance'].max() for r in all_results)
    total_transitions = sum(len(r.get('transitions', [])) for r in all_results)

    # Confidence statistics
    all_confidences = []
    for r in all_results:
        if 'confidence' in r['landcover'].columns:
            all_confidences.extend(r['landcover']['confidence'].values)

    stats = {
        'num_transects': len(all_results),
        'total_sample_points': total_points,
        'total_length_m': total_length,
        'avg_points_per_transect': total_points / len(all_results),
        'avg_length_per_transect_m': total_length / len(all_results),
        'total_transitions': total_transitions,
        'avg_transitions_per_transect': total_transitions / len(all_results),
        'classification_confidence': {
            'mean': np.mean(all_confidences),
            'std': np.std(all_confidences),
            'min': np.min(all_confidences),
            'max': np.max(all_confidences)
        } if all_confidences else None
    }

    return stats


def print_processing_summary(stats: Dict):
    """
    Print formatted processing summary to console.

    Args:
        stats: Processing statistics dictionary
    """
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Number of transects processed: {stats['num_transects']}")
    print(f"Total sample points: {stats['total_sample_points']}")
    print(f"Total transect length: {stats['total_length_m']:.1f} m")
    print(f"Average points per transect: {stats['avg_points_per_transect']:.1f}")
    print(f"Average transect length: {stats['avg_length_per_transect_m']:.1f} m")
    print(f"\nTotal transitions detected: {stats['total_transitions']}")
    print(f"Average transitions per transect: {stats['avg_transitions_per_transect']:.2f}")

    if stats['classification_confidence']:
        conf = stats['classification_confidence']
        print(f"\nClassification confidence:")
        print(f"  Mean: {conf['mean']:.3f}")
        print(f"  Std:  {conf['std']:.3f}")
        print(f"  Range: [{conf['min']:.3f}, {conf['max']:.3f}]")

    print("=" * 60 + "\n")
