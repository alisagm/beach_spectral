"""
Visualization module for spectral transect analysis.

Provides plotting functions for:
- Spectral profiles with landcover classification
- Transition zone annotations
- Diagnostic plots for validation
"""

from .plotting_old import (
    plot_transect_analysis,
    # Additional functions will be exposed as plotting.py is refactored
)
from .plotting import (
    plot_spectral_single,
    plot_spectral_grid
)
__all__ = [
    'plot_transect_analysis',
    'plot_spectral_single',
    'plot_spectral_grid'
]