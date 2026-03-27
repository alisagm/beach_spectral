"""
Batch processing utilities for year-based imagery processing.

Provides helper functions for:
- Extracting dates/years from file paths
- Grouping imagery by year
"""

import re
import logging
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

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
