"""
Scoring Right Panel with GDS and SEM image displays.

This panel provides:
- GDS image display
- SEM image display
- Image comparison tools
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