"""
Visualization module for spectral transect analysis.

Provides plotting functions for:
- Spectral profiles with landcover classification
- Transition zone annotations
- Diagnostic plots for validation
"""

from .plotting import (
    plot_transect_analysis,
    # Additional functions will be exposed as plotting.py is refactored
)

__all__ = [
    'plot_transect_analysis',
]