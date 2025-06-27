"""UI package for Qt6 components and views."""

from .main_window import MainWindow
from .components import *
from .panels import *

__all__ = [
    'MainWindow',
    # Components
    'FileSelector',
    'SliderInput', 
    'HistogramView',
    # Panels
    'ModeSwitcher',
    'AlignmentControls',
    'AlignmentCanvas',
    'AlignmentInfo',
    'AlignmentPanel',
    'FilterPanel',
    'ScorePanel'
]
