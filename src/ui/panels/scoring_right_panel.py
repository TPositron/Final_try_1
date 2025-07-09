"""
Scoring Right Panel - Image Display and Comparison Interface

This module provides the right panel for the scoring view, featuring dual image
displays for GDS and SEM images with comparison tools and visual analysis.

Main Class:
- ScoringRightPanel: Right panel for scoring view with image displays

Key Methods:
- init_panel(): Initializes panel layout with image viewers and controls
- update_gds_image(): Updates GDS image display with new data
- update_sem_image(): Updates SEM image display with new data
- clear_images(): Clears both image displays

Signals Emitted:
- comparison_toggled(bool): Image comparison mode toggled

Dependencies:
- Uses: PySide6.QtWidgets, PySide6.QtCore (Qt framework)
- Uses: ui/base_panels.BaseViewPanel (base panel functionality)
- Uses: ui/components/image_viewer.ImageViewer (image display)
- Uses: ui/view_manager.ViewMode (view mode management)
- Called by: UI scoring workflow and main window
- Coordinates with: Scoring operations and image analysis

Features:
- Dual image viewer setup for GDS and SEM images
- Grouped image displays with clear labeling
- Toggle comparison mode for side-by-side analysis
- Integrated image viewer controls and functionality
- Consistent styling with application theme
- Signal-based communication for comparison operations
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                              QGroupBox, QPushButton, QLabel)
from PySide6.QtCore import Signal
from src.ui.base_panels import BaseViewPanel
from src.ui.components.image_viewer import ImageViewer
from src.ui.view_manager import ViewMode


class ScoringRightPanel(BaseViewPanel):
    """Right panel for scoring view with image displays."""
    
    # Signals
    comparison_toggled = Signal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_panel()
        
    def init_panel(self):
        """Initialize the panel layout and widgets."""
        layout = QVBoxLayout(self)
        
        # GDS image group
        gds_group = QGroupBox("GDS Image")
        gds_layout = QVBoxLayout(gds_group)
        self.gds_viewer = ImageViewer()
        gds_layout.addWidget(self.gds_viewer)
        layout.addWidget(gds_group)
        
        # SEM image group
        sem_group = QGroupBox("SEM Image")
        sem_layout = QVBoxLayout(sem_group)
        self.sem_viewer = ImageViewer()
        sem_layout.addWidget(self.sem_viewer)
        layout.addWidget(sem_group)
        
        # Comparison controls
        controls_layout = QHBoxLayout()
        self.toggle_comparison_btn = QPushButton("Toggle Comparison")
        self.toggle_comparison_btn.setCheckable(True)
        controls_layout.addWidget(self.toggle_comparison_btn)
        layout.addLayout(controls_layout)
        
        # Connect signals
        self.toggle_comparison_btn.toggled.connect(self.comparison_toggled)
        
    def update_gds_image(self, image_data):
        """Update the GDS image display."""
        self.gds_viewer.set_image(image_data)
        
    def update_sem_image(self, image_data):
        """Update the SEM image display."""
        self.sem_viewer.set_image(image_data)
        
    def clear_images(self):
        """Clear both image displays."""
        self.gds_viewer.clear()
        self.sem_viewer.clear() 