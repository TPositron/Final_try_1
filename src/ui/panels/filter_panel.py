"""
Filter Panel
Filter selection, previews, histogram, and kernel view for image processing.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                               QComboBox, QPushButton, QListWidget, QSplitter,
                               QLabel, QSlider, QSpinBox, QDoubleSpinBox)
from PySide6.QtCore import Qt, Signal
from ..components.histogram_view import HistogramView
from typing import Dict, Any, List
import numpy as np


class FilterPanel(QWidget):
    """Panel for filter selection, preview, and application."""
    
    # Signals
    filter_applied = Signal(str, dict)  # filter_name, parameters
    filter_previewed = Signal(str, dict)  # filter_name, parameters
    filter_reset = Signal()  # Reset to original image
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_filter = None
        self.current_parameters = {}
        self.available_filters = []
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the UI components."""
        layout = QHBoxLayout(self)
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Filter controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # Filter selection group
        filter_group = QGroupBox("Filter Selection")
        filter_layout = QVBoxLayout(filter_group)
        
        self.filter_combo = QComboBox()
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(QLabel("Filter Type:"))
        filter_layout.addWidget(self.filter_combo)
        
        controls_layout.addWidget(filter_group)
        
        # Filter parameters group
        self.params_group = QGroupBox("Parameters")
        self.params_layout = QVBoxLayout(self.params_group)
        controls_layout.addWidget(self.params_group)
        
        # Filter history group
        history_group = QGroupBox("Applied Filters")
        history_layout = QVBoxLayout(history_group)
        
        self.history_list = QListWidget()
        history_layout.addWidget(self.history_list)
        
        controls_layout.addWidget(history_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self._on_preview_clicked)
        button_layout.addWidget(self.preview_button)
        
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._on_apply_clicked)
        button_layout.addWidget(self.apply_button)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self._on_reset_clicked)
        button_layout.addWidget(self.reset_button)
        
        controls_layout.addLayout(button_layout)
        controls_layout.addStretch()
        
        # Right side: Histogram and kernel view
        self.histogram_view = HistogramView()
        
        # Add to splitter
        splitter.addWidget(controls_widget)
        splitter.addWidget(self.histogram_view)
        
        # Set splitter proportions (60% left, 40% right)
        splitter.setSizes([600, 400])
        
        layout.addWidget(splitter)
        
    def set_available_filters(self, filters: List[str]):
        """Set the list of available filters."""
        self.available_filters = filters
        self.filter_combo.clear()
        self.filter_combo.addItems(filters)
        
    def _on_filter_changed(self, filter_name: str):
        """Handle filter selection change."""
        self.current_filter = filter_name
        self._update_parameter_controls(filter_name)
        
    def _update_parameter_controls(self, filter_name: str):
        """Update parameter controls based on selected filter."""
        # Clear existing parameter controls
        for i in reversed(range(self.params_layout.count())):
            self.params_layout.itemAt(i).widget().setParent(None)
            
        # Add parameter controls based on filter type
        # This would be expanded with actual filter parameter definitions
        if filter_name == "Gaussian Blur":
            self._add_parameter_control("sigma", "Sigma", "double", 1.0, 0.1, 10.0)
        elif filter_name == "Canny":
            self._add_parameter_control("low_threshold", "Low Threshold", "int", 50, 0, 255)
            self._add_parameter_control("high_threshold", "High Threshold", "int", 150, 0, 255)
        elif filter_name == "Laplacian":
            self._add_parameter_control("ksize", "Kernel Size", "int", 3, 1, 15, step=2)
        elif filter_name == "Gabor":
            self._add_parameter_control("frequency", "Frequency", "double", 0.1, 0.01, 1.0)
            self._add_parameter_control("theta", "Theta", "double", 0.0, 0.0, 180.0)
            
    def _add_parameter_control(self, param_name: str, label: str, param_type: str, 
                             default_value, min_val, max_val, step=None):
        """Add a parameter control widget."""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(f"{label}:"))
        
        if param_type == "int":
            control = QSpinBox()
            control.setRange(int(min_val), int(max_val))
            if step:
                control.setSingleStep(step)
            control.setValue(int(default_value))
        elif param_type == "double":
            control = QDoubleSpinBox()
            control.setRange(float(min_val), float(max_val))
            control.setValue(float(default_value))
            control.setSingleStep(0.1)
        
        control.setObjectName(param_name)
        layout.addWidget(control)
        
        self.params_layout.addLayout(layout)
        
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        parameters = {}
        for i in range(self.params_layout.count()):
            layout = self.params_layout.itemAt(i)
            if hasattr(layout, 'itemAt') and layout.itemAt(1):
                widget = layout.itemAt(1).widget()
                if hasattr(widget, 'objectName') and widget.objectName():
                    param_name = widget.objectName()
                    if isinstance(widget, QSpinBox):
                        parameters[param_name] = widget.value()
                    elif isinstance(widget, QDoubleSpinBox):
                        parameters[param_name] = widget.value()
        return parameters
        
    def _on_preview_clicked(self):
        """Handle preview button click."""
        if self.current_filter:
            parameters = self._get_current_parameters()
            self.filter_previewed.emit(self.current_filter, parameters)
            
    def _on_apply_clicked(self):
        """Handle apply button click."""
        if self.current_filter:
            parameters = self._get_current_parameters()
            self.filter_applied.emit(self.current_filter, parameters)
            
            # Add to history
            filter_text = f"{self.current_filter}({', '.join([f'{k}={v}' for k, v in parameters.items()])})"
            self.history_list.addItem(filter_text)
            
    def _on_reset_clicked(self):
        """Handle reset button click."""
        self.filter_reset.emit()
        self.history_list.clear()
        
    def update_histogram(self, image_data: np.ndarray):
        """Update the histogram display."""
        self.histogram_view.update_histogram(image_data)
        
    def update_kernel_view(self, kernel: np.ndarray, title: str = "Filter Kernel"):
        """Update the kernel visualization."""
        self.histogram_view.update_kernel_view(kernel, title)
