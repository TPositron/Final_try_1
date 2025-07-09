"""
UI Panels Package - User Interface Panel Components

This package contains all UI panel components for the image analysis application,
providing modular and reusable interface elements.

Main Components:
- ModeSwitcher: Mode switching panel
- AlignmentControlsPanel: Alignment control interface
- AlignmentCanvasPanel: Alignment canvas display
- AlignmentInfoPanel: Alignment information display
- AlignmentPanel: Main alignment panel
- FilterPanel: Image filtering panel
- ScorePanel: Scoring results panel

Dependencies:
- Individual panel modules within this package
- Used by: UI main window and panel management system
- Coordinates with: View management and workflow components

Features:
- Modular panel architecture
- Reusable UI components
- Consistent interface design
- Integration with main application workflow
"""

from .mode_switcher import ModeSwitcher
from .alignment_controls import AlignmentControlsPanel
from .alignment_canvas import AlignmentCanvasPanel
from .alignment_info import AlignmentInfoPanel
from .alignment_panel import AlignmentPanel
from .filter_panel import FilterPanel
from .score_panel import ScorePanel

__all__ = [
    'ModeSwitcher',
    'AlignmentControlsPanel', 
    'AlignmentCanvasPanel',
    'AlignmentInfoPanel',
    'AlignmentPanel',
    'FilterPanel',
    'ScorePanel'
]

__all__ = [
    'ModeSwitcher',
    'AlignmentControlsPanel',
    'AlignmentCanvasPanel', 
    'AlignmentInfoPanel',
    'AlignmentPanel',
    'FilterPanel',
    'ScorePanel'
]
