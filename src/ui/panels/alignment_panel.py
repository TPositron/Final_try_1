"""
Alignment Panel
Composite panel combining alignment controls, canvas, and info displays.
"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSplitter
from PySide6.QtCore import Qt, Signal
from .alignment_controls import AlignmentControlsPanel
from .alignment_canvas import AlignmentCanvasPanel
from .alignment_info import AlignmentInfoPanel
from typing import Dict, Any
import numpy as np


class AlignmentPanel(QWidget):
    """Main alignment panel combining all alignment components."""
    
    # Signals
    transform_applied = Signal(dict)  # Emitted when transform is applied
    alignment_completed = Signal(dict)  # Emitted when alignment is completed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """Initialize the UI components."""
        layout = QHBoxLayout(self)
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Controls and info
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Alignment controls
        self.controls_panel = AlignmentControlsPanel()
        left_layout.addWidget(self.controls_panel)
        
        # Alignment info
        self.info_panel = AlignmentInfoPanel()
        left_layout.addWidget(self.info_panel)
        
        # Right side: Canvas
        self.canvas_panel = AlignmentCanvasPanel()
        
        # Add to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(self.canvas_panel)
        
        # Set splitter proportions (30% left, 70% right)
        splitter.setSizes([300, 700])
        
        layout.addWidget(splitter)
        
    def connect_signals(self):
        """Connect internal signals."""
        # Connect controls to canvas
        self.controls_panel.transform_changed.connect(self._on_transform_changed)
        self.controls_panel.reset_transform.connect(self._on_reset_transform)
        
        # Connect canvas signals
        canvas = self.canvas_panel.get_canvas()
        canvas.mouse_clicked.connect(self._on_canvas_clicked)
        
    def _on_transform_changed(self, transform: Dict[str, Any]):
        """Handle transform changes from controls."""
        # Update canvas
        canvas = self.canvas_panel.get_canvas()
        canvas.update_gds_transform(transform)
        
        # Update info panel
        self.info_panel.update_transform_info(transform)
        
        # Emit signal
        self.transform_applied.emit(transform)
        
    def _on_reset_transform(self):
        """Handle transform reset."""
        # Update info panel
        default_transform = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0,
            'transparency': 0.5
        }
        self.info_panel.update_transform_info(default_transform)
        self.info_panel.update_status("Reset", "blue")
        
    def _on_canvas_clicked(self, x: float, y: float):
        """Handle canvas click events."""
        # For future implementation: click-to-align functionality
        pass
        
    def load_sem_image(self, image_array: np.ndarray):
        """Load SEM image into the alignment canvas."""
        canvas = self.canvas_panel.get_canvas()
        canvas.load_sem_image(image_array)
        
        # Update status
        self.info_panel.update_image_info(True, canvas.gds_item is not None)
        self.info_panel.update_status("SEM loaded", "green")
        
    def load_gds_image(self, image_array: np.ndarray):
        """Load GDS overlay image into the alignment canvas."""
        canvas = self.canvas_panel.get_canvas()
        canvas.load_gds_image(image_array)
        
        # Update status
        self.info_panel.update_image_info(canvas.sem_item is not None, True)
        self.info_panel.update_status("GDS loaded", "green")
        
    def set_transform(self, transform: Dict[str, Any]):
        """Set the current transform values."""
        self.controls_panel.set_transform(transform)
        
    def get_current_transform(self) -> Dict[str, Any]:
        """Get the current transform values."""
        return self.controls_panel.get_current_transform()
        
    def fit_canvas_to_view(self):
        """Fit the canvas content to the view."""
        canvas = self.canvas_panel.get_canvas()
        canvas.fit_in_view()
        
    def reset_canvas_zoom(self):
        """Reset canvas zoom to actual size."""
        canvas = self.canvas_panel.get_canvas()
        canvas.zoom_to_actual_size()
        
    def update_alignment_metrics(self, correlation: float = None, overlap: float = None):
        """Update alignment quality metrics."""
        self.info_panel.update_metrics(correlation, overlap)
