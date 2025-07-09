"""
Alignment Controls Panel - Manual Transformation Controls

This module provides manual alignment controls for translate, rotate, scale, and
transparency adjustments with real-time parameter updates.

Main Class:
- AlignmentControlsPanel: Panel with manual alignment controls

Key Methods:
- setup_ui(): Initializes UI components and layout
- set_transform(): Sets current transform values
- reset_transforms(): Resets all transforms to defaults
- get_current_transform(): Gets current transform parameters

Private Methods:
- _on_translate_x_changed(): Handles X translation changes
- _on_translate_y_changed(): Handles Y translation changes
- _on_rotation_changed(): Handles rotation changes
- _on_scale_changed(): Handles scale changes
- _on_transparency_changed(): Handles transparency changes
- _on_reset_clicked(): Handles reset button click

Signals Emitted:
- transform_changed(dict): Transform parameters changed
- reset_transform(): Reset button clicked

Dependencies:
- Uses: PySide6.QtWidgets, PySide6.QtCore (Qt framework)
- Uses: typing (type hints)
- Called by: UI alignment components
- Coordinates with: Alignment canvas and transformation workflow

Features:
- Translation controls (X/Y with spinboxes)
- Rotation control with degree precision
- Scale control with decimal precision
- Transparency slider for overlay visibility
- Real-time parameter updates and validation
- Reset functionality for all parameters
- Consistent UI styling and layout
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                               QPushButton, QSlider, QSpinBox, QLabel, QDoubleSpinBox)
from PySide6.QtCore import Qt, Signal
from typing import Dict, Any


class AlignmentControlsPanel(QWidget):
    """Panel containing manual alignment controls."""
    
    # Signals
    transform_changed = Signal(dict)  # Emitted when any transform parameter changes
    reset_transform = Signal()  # Emitted when reset button is clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.current_transform = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0,
            'transparency': 0.5
        }
        
    def setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        
        # Translation controls
        translation_group = QGroupBox("Translation")
        translation_layout = QVBoxLayout(translation_group)
        
        # X translation
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.x_spinbox = QDoubleSpinBox()
        self.x_spinbox.setRange(-500, 500)
        self.x_spinbox.setSingleStep(1.0)
        self.x_spinbox.valueChanged.connect(self._on_translate_x_changed)
        x_layout.addWidget(self.x_spinbox)
        translation_layout.addLayout(x_layout)
        
        # Y translation
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.y_spinbox = QDoubleSpinBox()
        self.y_spinbox.setRange(-500, 500)
        self.y_spinbox.setSingleStep(1.0)
        self.y_spinbox.valueChanged.connect(self._on_translate_y_changed)
        y_layout.addWidget(self.y_spinbox)
        translation_layout.addLayout(y_layout)
        
        layout.addWidget(translation_group)
        
        # Rotation controls
        rotation_group = QGroupBox("Rotation")
        rotation_layout = QVBoxLayout(rotation_group)
        
        rotation_control_layout = QHBoxLayout()
        rotation_control_layout.addWidget(QLabel("Angle:"))
        self.rotation_spinbox = QDoubleSpinBox()
        self.rotation_spinbox.setRange(-180, 180)
        self.rotation_spinbox.setSingleStep(1.0)
        self.rotation_spinbox.setSuffix("°")
        self.rotation_spinbox.valueChanged.connect(self._on_rotation_changed)
        rotation_control_layout.addWidget(self.rotation_spinbox)
        rotation_layout.addLayout(rotation_control_layout)
        
        layout.addWidget(rotation_group)
        
        # Scale controls
        scale_group = QGroupBox("Scale")
        scale_layout = QVBoxLayout(scale_group)
        
        scale_control_layout = QHBoxLayout()
        scale_control_layout.addWidget(QLabel("Scale:"))
        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setRange(0.1, 5.0)
        self.scale_spinbox.setSingleStep(0.1)
        self.scale_spinbox.setValue(1.0)
        self.scale_spinbox.valueChanged.connect(self._on_scale_changed)
        scale_control_layout.addWidget(self.scale_spinbox)
        scale_layout.addLayout(scale_control_layout)
        
        layout.addWidget(scale_group)
        
        # Transparency controls
        transparency_group = QGroupBox("Transparency")
        transparency_layout = QVBoxLayout(transparency_group)
        
        # FIXED: Use proper enum namespace for Qt.Orientation.Horizontal
        self.transparency_slider = QSlider(Qt.Orientation.Horizontal)
        self.transparency_slider.setRange(0, 100)
        self.transparency_slider.setValue(50)
        self.transparency_slider.valueChanged.connect(self._on_transparency_changed)
        transparency_layout.addWidget(self.transparency_slider)
        
        transparency_label_layout = QHBoxLayout()
        transparency_label_layout.addWidget(QLabel("Opaque"))
        transparency_label_layout.addStretch()
        transparency_label_layout.addWidget(QLabel("Transparent"))
        transparency_layout.addLayout(transparency_label_layout)
        
        layout.addWidget(transparency_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.reset_button = QPushButton("Reset All")
        self.reset_button.clicked.connect(self._on_reset_clicked)
        button_layout.addWidget(self.reset_button)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
    def _on_translate_x_changed(self, value: float):
        """Handle X translation change."""
        self.current_transform['translate_x'] = value
        self.transform_changed.emit(self.current_transform.copy())
        
    def _on_translate_y_changed(self, value: float):
        """Handle Y translation change."""
        self.current_transform['translate_y'] = value
        self.transform_changed.emit(self.current_transform.copy())
        
    def _on_rotation_changed(self, value: float):
        """Handle rotation change."""
        self.current_transform['rotation'] = value
        self.transform_changed.emit(self.current_transform.copy())
        
    def _on_scale_changed(self, value: float):
        """Handle scale change."""
        self.current_transform['scale'] = value
        self.transform_changed.emit(self.current_transform.copy())
        
    def _on_transparency_changed(self, value: int):
        """Handle transparency change."""
        self.current_transform['transparency'] = value / 100.0
        self.transform_changed.emit(self.current_transform.copy())
        
    def _on_reset_clicked(self):
        """Handle reset button click."""
        self.reset_transforms()
        self.reset_transform.emit()
        
    def set_transform(self, transform: Dict[str, float]):
        """Set the current transform values."""
        self.current_transform.update(transform)
        
        # Update UI controls
        self.x_spinbox.setValue(transform.get('translate_x', 0.0))
        self.y_spinbox.setValue(transform.get('translate_y', 0.0))
        self.rotation_spinbox.setValue(transform.get('rotation', 0.0))
        self.scale_spinbox.setValue(transform.get('scale', 1.0))
        self.transparency_slider.setValue(int(transform.get('transparency', 0.5) * 100))
        
    def reset_transforms(self):
        """Reset all transform values to defaults."""
        default_transform = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0,
            'transparency': 0.5
        }
        self.set_transform(default_transform)
        
    def get_current_transform(self) -> Dict[str, float]:
        """Get the current transform values."""
        return self.current_transform.copy()