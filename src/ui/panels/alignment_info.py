"""
Alignment Info Panel
Displays current transform parameters and alignment status.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QFrame
from PySide6.QtCore import Qt
from typing import Dict, Any, Optional


class AlignmentInfoPanel(QWidget):
    """Panel for displaying current alignment information."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        
        # Transform information group
        transform_group = QGroupBox("Current Transform")
        transform_layout = QVBoxLayout(transform_group)
        
        # Translation info
        self.translate_x_label = QLabel("X Translation: 0.0 px")
        self.translate_y_label = QLabel("Y Translation: 0.0 px")
        transform_layout.addWidget(self.translate_x_label)
        transform_layout.addWidget(self.translate_y_label)
        
        # Add separator - FIXED: Use proper enum namespace
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        transform_layout.addWidget(separator1)
        
        # Rotation info
        self.rotation_label = QLabel("Rotation: 0.0°")
        transform_layout.addWidget(self.rotation_label)
        
        # Add separator - FIXED: Use proper enum namespace
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        transform_layout.addWidget(separator2)
        
        # Scale info
        self.scale_label = QLabel("Scale: 1.00x")
        transform_layout.addWidget(self.scale_label)
        
        # Add separator - FIXED: Use proper enum namespace
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.Shape.HLine)
        transform_layout.addWidget(separator3)
        
        # Transparency info
        self.transparency_label = QLabel("Transparency: 50%")
        transform_layout.addWidget(self.transparency_label)
        
        layout.addWidget(transform_group)
        
        # Alignment status group
        status_group = QGroupBox("Alignment Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        status_layout.addWidget(self.status_label)
        
        self.image_info_label = QLabel("SEM: Not loaded\nGDS: Not loaded")
        status_layout.addWidget(self.image_info_label)
        
        layout.addWidget(status_group)
        
        # Quality metrics group (for future use)
        metrics_group = QGroupBox("Alignment Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        self.correlation_label = QLabel("Correlation: N/A")
        self.overlap_label = QLabel("Overlap: N/A")
        metrics_layout.addWidget(self.correlation_label)
        metrics_layout.addWidget(self.overlap_label)
        
        layout.addWidget(metrics_group)
        
        layout.addStretch()
        
    def update_transform_info(self, transform: Dict[str, Any]):
        """Update the transform information display."""
        translate_x = transform.get('translate_x', 0.0)
        translate_y = transform.get('translate_y', 0.0)
        rotation = transform.get('rotation', 0.0)
        scale = transform.get('scale', 1.0)
        transparency = transform.get('transparency', 0.5)
        
        self.translate_x_label.setText(f"X Translation: {translate_x:.1f} px")
        self.translate_y_label.setText(f"Y Translation: {translate_y:.1f} px")
        self.rotation_label.setText(f"Rotation: {rotation:.1f}°")
        self.scale_label.setText(f"Scale: {scale:.2f}x")
        self.transparency_label.setText(f"Transparency: {int(transparency * 100)}%")
        
    def update_status(self, status: str, color: str = "green"):
        """Update the alignment status."""
        self.status_label.setText(f"Status: {status}")
        self.status_label.setStyleSheet(f"QLabel {{ color: {color}; font-weight: bold; }}")
        
    def update_image_info(self, sem_loaded: bool, gds_loaded: bool):
        """Update the image loading status."""
        sem_status = "Loaded" if sem_loaded else "Not loaded"
        gds_status = "Loaded" if gds_loaded else "Not loaded"
        self.image_info_label.setText(f"SEM: {sem_status}\nGDS: {gds_status}")
        
    def update_metrics(self, correlation: Optional[float] = None, overlap: Optional[float] = None):
        """Update alignment quality metrics."""
        # FIXED: Use proper Optional type annotations instead of float = None
        corr_text = f"{correlation:.3f}" if correlation is not None else "N/A"
        overlap_text = f"{overlap:.3f}" if overlap is not None else "N/A"
        
        self.correlation_label.setText(f"Correlation: {corr_text}")
        self.overlap_label.setText(f"Overlap: {overlap_text}")
