"""
Batch processing utilities for year-based imagery processing.

Provides helper functions for:
- Extracting dates/years from file paths
- Classifying rasters by band configuration
- Grouping imagery by year
- Resolving band configuration at the year level
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import rasterio

logger = logging.getLogger(__name__)


# ============================================================================
# DATE/YEAR EXTRACTION
# ============================================================================

def extract_year_from_path(path: Path) -> str:
    """
    Extract year from path like 'imagery/2016/20160122/file.tif'.
    
    Handles multiple date formats in directory names:
        - 4 digits: YYYY (e.g., '2016' → '2016')
        - 6 digits: YYYYMM (e.g., '200605' → '2006')
        - 8 digits: YYYYMMDD (e.g., '20160122' → '2016')
    
    Args:
        path: Path to raster file
        
    Returns:
        Year string (4 digits) or "unknown" if not found
    """
    for part in path.parts:
        # Check for 4-digit year (YYYY)
        if re.match(r'^\d{4}$', part):
            return part
        # Check for 6-digit date (YYYYMM)
        if re.match(r'^\d{6}$', part):
            return part[:4]
        # Check for 8-digit date (YYYYMMDD)
        if re.match(r'^\d{8}$', part):
            return part[:4]
    return "unknown"


def extract_capture_date_from_path(path: Path) -> str:
    """
    Extract capture date from path with best available precision.
    
    Returns:
        - 'YYYYMM' if month info available (from YYYYMM or YYYYMMDD folders)
        - 'YYYY' if only year available
        - 'unknown' if no date found
    
    Examples:
        'imagery/2016/20160122/file.tif' → '201601'
        'imagery/200605/file.tif' → '200605'
        'imagery/2015/file.tif' → '2015'
    """
    best_date = None
    best_precision = 0  # 4=year, 6=month, 8=day
    
    for part in path.parts:
        # 8-digit: YYYYMMDD → extract YYYYMM
        if re.match(r'^\d{8}$', part):
            candidate = part[:6]  # YYYYMM
            if best_precision < 8:
                best_date = candidate
                best_precision = 8
        # 6-digit: YYYYMM
        elif re.match(r'^\d{6}$', part):
            if best_precision < 6:
                best_date = part
                best_precision = 6
        # 4-digit: YYYY
        elif re.match(r'^\d{4}$', part):
            if best_precision < 4:
                best_date = part
                best_precision = 4
    
    return best_date or "unknown"


# ============================================================================
# RASTER CLASSIFICATION
# ============================================================================

def get_band_count_safe(path: Path) -> Tuple[int, Optional[str]]:
    """
    Safely get band count from a raster file.
    
    Args:
        path: Path to raster file
        
    Returns:
        Tuple of (band_count, error_message)
        If successful: (band_count, None)
        If failed: (0, error_message)
    """
    try:
        with rasterio.open(path) as src:
            return src.count, None
    except Exception as e:
        return 0, str(e)


def classify_raster(path: Path) -> Dict:
    """
    Classify a raster by band count and infer band mode.
    
    For 3-band imagery, runs variance-based detection to distinguish CIR from RGB.
    This classification is used for year-level majority voting.
    
    Args:
        path: Path to raster file
        
    Returns:
        Dict with keys:
            - 'path': Original path
            - 'band_count': Number of bands (0 if error)
            - 'band_mode': '4band', 'cir', 'rgb', or None
            - 'year': Extracted year
            - 'error': Error message or None
    """
    # Import here to avoid circular imports
    from ..utils.data_io import detect_band_configuration
    
    band_count, error = get_band_count_safe(path)
    
    if error:
        return {
            'path': path,
            'band_count': 0,
            'band_mode': None,
            'year': extract_year_from_path(path),
            'error': error
        }
    
    # Determine band mode
    if band_count >= 4:
        band_mode = '4band'
    elif band_count == 3:
        try:
            band_mode = detect_band_configuration(path)
        except Exception as e:
            logger.warning(f"Could not detect band config for {path.name}: {e}")
            band_mode = 'rgb'  # Default fallback
    else:
        band_mode = None
        error = f"Insufficient bands ({band_count})"
    
    return {
        'path': path,
        'band_count': band_count,
        'band_mode': band_mode,
        'year': extract_year_from_path(path),
        'error': error
    }


# ============================================================================
# YEAR-LEVEL GROUPING AND RESOLUTION
# ============================================================================

def group_rasters_by_year(
    raster_dir: Path,
    extensions: List[str] = None,
    recursive: bool = True
) -> Dict[str, List[Path]]:
    """
    Group raster files by year based on directory structure.
    
    Args:
        raster_dir: Root directory containing imagery
        extensions: List of extensions to include (default: ['.tif', '.tiff', '.jp2'])
        recursive: Search subdirectories recursively
        
    Returns:
        Dictionary mapping year strings to lists of raster paths
    """
    if extensions is None:
        extensions = ['.tif', '.tiff', '.jp2']
    
    raster_dir = Path(raster_dir)
    years = defaultdict(list)
    
    # Find all rasters
    for ext in extensions:
        pattern = f'**/*{ext}' if recursive else f'*{ext}'
        for path in raster_dir.glob(pattern):
            year = extract_year_from_path(path)
            years[year].append(path)
    
    # Sort paths within each year
    for year in years:
        years[year].sort()
    
    logger.info(f"Found imagery for {len(years)} years: {sorted(years.keys())}")
    
    return dict(years)


def resolve_year_band_config(
    raster_paths: List[Path],
    min_bands: int = 3
) -> Tuple[str, List[Path]]:
    """
    Resolve band configuration for a year's imagery using majority voting.
    
    When a year has mixed band configurations (e.g., some CIR, some RGB),
    this function determines the dominant configuration and filters to
    only include rasters matching that configuration.
    
    Priority: 4band > cir > rgb (when counts are equal)
    
    Args:
        raster_paths: List of raster paths for a single year
        min_bands: Minimum bands required (default: 3)
        
    Returns:
        Tuple of (resolved_band_mode, filtered_raster_paths)
    """
    classifications = []
    
    for path in raster_paths:
        info = classify_raster(path)
        if info['error'] is None and info['band_count'] >= min_bands:
            classifications.append(info)
    
    if not classifications:
        logger.warning("No valid rasters found for year")
        return 'rgb', []
    
    # Count by band mode
    mode_counts = defaultdict(int)
    mode_paths = defaultdict(list)
    
    for info in classifications:
        mode = info['band_mode']
        mode_counts[mode] += 1
        mode_paths[mode].append(info['path'])
    
    # Resolve: prefer 4band > cir > rgb when counts are equal
    priority = {'4band': 3, 'cir': 2, 'rgb': 1}
    
    best_mode = max(
        mode_counts.keys(),
        key=lambda m: (mode_counts[m], priority.get(m, 0))
    )
    
    logger.info(
        f"Resolved band config: {best_mode} "
        f"(counts: {dict(mode_counts)})"
    )
    
    return best_mode, mode_paths[best_mode]


def detect_cir_from_filename(path: Path) -> bool:
    """
    Detect if a file is CIR based on filename patterns.
    
    This is a heuristic fallback when statistical detection is unreliable.
    Common CIR indicators in filenames: '_cir_', '_CIR_', 'cir.', 'CIR.'
    
    Args:
        path: Path to raster file
        
    Returns:
        True if filename suggests CIR imagery
    """
    name_lower = path.name.lower()
    return '_cir' in name_lower or 'cir.' in name_lower or 'cir_' in name_lower