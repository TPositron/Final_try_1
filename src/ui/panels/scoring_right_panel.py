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
from PySide6.QtCore import Signal, Qt
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
        """Initialize the panel layout and widgets - no file information, processing status, histogram, or statistics."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        
        # GDS Structure Selection with vertical slider (same as alignment and filtering)
        structure_group = QGroupBox("GDS Structure Selection")
        structure_group.setStyleSheet(self._get_group_style())
        structure_layout = QVBoxLayout(structure_group)
        
        # Create resizable splitter for structure list
        from PySide6.QtWidgets import QScrollArea, QListWidget, QSplitter
        structure_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Structure list
        self.structure_scroll = QScrollArea()
        self.structure_list = QListWidget()
        self.structure_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #404040;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
            QListWidget::item:hover {
                background-color: #3e3e42;
            }
        """)
        self.structure_scroll.setWidget(self.structure_list)
        self.structure_scroll.setWidgetResizable(True)
        
        structure_splitter.addWidget(self.structure_scroll)
        
        # Add stretch for resizing
        stretch_structure = QWidget()
        structure_splitter.addWidget(stretch_structure)
        structure_splitter.setSizes([200, 50])
        
        structure_layout.addWidget(structure_splitter)
        layout.addWidget(structure_group)
        
        layout.addStretch()
    
    def _get_group_style(self):
        return """
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                color: #ffffff;
                background-color: #1e1e1e;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """
        
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