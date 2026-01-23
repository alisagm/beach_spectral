"""
Miscellaneous helper functions for spectral transect analysis.

Contains small utility functions that don't fit elsewhere.
"""

import logging

logger = logging.getLogger(__name__)


def detect_transect_direction(transect_geometry) -> str:
    """
    Detect if transect runs west-to-east or east-to-west.
    
    For PAIS shorelines, transects typically run perpendicular to shore.
    This function determines the orientation based on the start/end
    coordinates to ensure consistent plotting direction.

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