"""
UI package for Qt6 components, panels, and styling.

This package provides the complete user interface layer including:
- Main application window and navigation
- Reusable UI components (file selectors, inputs, viewers)
- Specialized panels for filtering, alignment, and scoring
- Consistent styling and theming across all components

The UI follows the Model-View-Controller pattern with Qt signals/slots
for communication between components and the underlying services.

Architecture:
    components/: Reusable UI widgets and controls
    panels/: Specialized functional panels for different modes
    styles/: Consistent theming and visual styling
    MainWindow: Top-level application window and layout coordinator
"""

from .main_window import MainWindow
from .components import *
from .panels import *
from . import styles

__all__ = [
    'MainWindow',
    # Components
    'FileSelector',
    'SliderInput', 
    'HistogramView',
    # Panels
    'ModeSwitcher',
    'AlignmentControlsPanel',
    'AlignmentCanvasPanel',
    'AlignmentInfoPanel',
    'AlignmentPanel',
    'FilterPanel',
    'ScorePanel',
    # Styles
    'styles'
]
