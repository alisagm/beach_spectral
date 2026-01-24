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
        - 4 digits: YYYY (e.g., '2016' â†’ '2016')
        - 6 digits: YYYYMM (e.g., '200605' â†’ '2006')
        - 8 digits: YYYYMMDD (e.g., '20160122' â†’ '2016')
    
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
        'imagery/2016/20160122/file.tif' â†’ '201601'
        'imagery/200605/file.tif' â†’ '200605'
        'imagery/2015/file.tif' â†’ '2015'
    """
    best_date = None
    best_precision = 0  # 4=year, 6=month, 8=day
    
    for part in path.parts:
        # 8-digit: YYYYMMDD â†’ extract YYYYMM
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
    Resolve band configuration for a year's imagery using priority-based selection.
    
    All rasters in a year typically share the same source, so we use the BEST
    available detection method rather than filtering by detected mode.
    
    Priority logic:
    1. If ANY raster is 4-band → use '4band' (NIR method) for entire year
    2. If all 3-band and ANY detected as CIR → use 'cir' for entire year
    3. Only use 'rgb' if ALL rasters are detected as RGB
    
    This favors NIR-based detection methods which are more reliable for
    coastal shoreline detection. Statistical CIR/RGB detection can misclassify
    in low-vegetation coastal environments.
    
    Args:
        raster_paths: List of raster paths for a single year
        min_bands: Minimum bands required (default: 3)
        
    Returns:
        Tuple of (resolved_band_mode, all_valid_raster_paths)
        Note: Returns ALL valid paths, not filtered by mode
    """
    classifications = []
    valid_paths = []
    
    for path in raster_paths:
        info = classify_raster(path)
        if info['error'] is None and info['band_count'] >= min_bands:
            classifications.append(info)
            valid_paths.append(path)
    
    if not classifications:
        logger.warning("No valid rasters found for year")
        return 'rgb', []
    
    # Count by band mode for logging
    mode_counts = defaultdict(int)
    for info in classifications:
        mode_counts[info['band_mode']] += 1
    
    logger.debug(f"Band mode detection counts: {dict(mode_counts)}")
    
    # Priority-based resolution (not majority voting)
    # 1. If ANY is 4-band, use 4band
    if mode_counts.get('4band', 0) > 0:
        resolved_mode = '4band'
        logger.info(
            f"Resolved band config: {resolved_mode} "
            f"({mode_counts['4band']}/{len(classifications)} are 4-band)"
        )
    # 2. If ANY is CIR (and none are 4-band), use CIR
    elif mode_counts.get('cir', 0) > 0:
        resolved_mode = 'cir'
        logger.info(
            f"Resolved band config: {resolved_mode} "
            f"({mode_counts['cir']}/{len(classifications)} detected as CIR, "
            f"favoring NIR-based detection)"
        )
    # 3. Only use RGB if ALL are RGB
    else:
        resolved_mode = 'rgb'
        logger.info(
            f"Resolved band config: {resolved_mode} "
            f"(all {len(classifications)} rasters are RGB-only)"
        )
    
    # Return ALL valid paths (not filtered) since all share same source
    return resolved_mode, valid_paths


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