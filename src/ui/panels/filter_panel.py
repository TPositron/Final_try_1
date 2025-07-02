"""
Filter Panel
Filter selection, previews, histogram, and kernel view for image processing.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                               QComboBox, QPushButton, QListWidget, QSplitter,
                               QLabel, QSlider, QSpinBox, QDoubleSpinBox, QInputDialog, QProgressBar)
from PySide6.QtCore import Qt, Signal
from ..components.histogram_view import HistogramView
from typing import Dict, Any, List
import numpy as np
import json
import os
import logging

logger = logging.getLogger(__name__)


class FilterPanel(QWidget):
    """Panel for filter selection, preview, and application."""
    
    # Signals
    filter_applied = Signal(str, dict)  # filter_name, parameters
    filter_previewed = Signal(str, dict)  # filter_name, parameters
    filter_reset = Signal()  # Reset to original image
    preset_saved = Signal(str, dict)  # preset_name, preset_data
    preset_loaded = Signal(str, dict)  # preset_name, preset_data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_filter = None
        self.current_parameters = {}
        self.available_filters = []
        self.presets = {}  # Store filter presets
        self.setup_ui()
        self._load_presets()
        
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
        
        # Preset management
        preset_layout = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("-- Select Preset --")
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
        preset_layout.addWidget(QLabel("Presets:"))
        preset_layout.addWidget(self.preset_combo)
        
        self.save_preset_button = QPushButton("Save")
        self.save_preset_button.clicked.connect(self._on_save_preset)
        preset_layout.addWidget(self.save_preset_button)
        
        filter_layout.addLayout(preset_layout)
        
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

        # Progress and status widgets
        self.status_label = QLabel("Ready.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.status_label)
        controls_layout.addWidget(self.progress_bar)

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
        # Emit preview signal when filter changes
        if self.current_filter:
            parameters = self._get_current_parameters()
            self.filter_previewed.emit(self.current_filter, parameters)
        
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
            # Connect to real-time preview
            control.valueChanged.connect(self._on_parameter_changed)
        elif param_type == "double":
            control = QDoubleSpinBox()
            control.setRange(float(min_val), float(max_val))
            control.setValue(float(default_value))
            control.setSingleStep(0.1)
            # Connect to real-time preview
            control.valueChanged.connect(self._on_parameter_changed)
        
        control.setObjectName(param_name)
        layout.addWidget(control)
        
        self.params_layout.addLayout(layout)
        
    def _on_parameter_changed(self, value):
        """Handle parameter value changes for real-time preview."""
        if self.current_filter:
            parameters = self._get_current_parameters()
            self.filter_previewed.emit(self.current_filter, parameters)
        
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
        
    def _on_parameter_changed(self):
        """Handle parameter value changes for real-time preview."""
        if self.current_filter:
            parameters = self._get_current_parameters()
            self.filter_previewed.emit(self.current_filter, parameters)
            
    def _on_preset_selected(self, preset_name: str):
        """Handle preset selection."""
        if preset_name in self.presets:
            preset = self.presets[preset_name]
            filter_name = preset.get('filter_name')
            parameters = preset.get('parameters', {})
            
            # Set filter and parameters
            if filter_name in self.available_filters:
                self.filter_combo.setCurrentText(filter_name)
                self._set_current_parameters(parameters)
                self.preset_loaded.emit(preset_name, preset)
                
    def _on_save_preset(self):
        """Save current filter and parameters as a preset."""
        if not self.current_filter:
            return
            
        name, ok = QInputDialog.getText(self, 'Save Preset', 'Enter preset name:')
        if ok and name:
            preset = {
                'filter_name': self.current_filter,
                'parameters': self._get_current_parameters()
            }
            self.presets[name] = preset
            self.preset_combo.addItem(name)
            self._save_presets()
            self.preset_saved.emit(name, preset)
            
    def _set_current_parameters(self, parameters: Dict[str, Any]):
        """Set parameter values in the UI controls."""
        for i in range(self.params_layout.count()):
            layout = self.params_layout.itemAt(i)
            if hasattr(layout, 'itemAt') and layout.itemAt(1):
                widget = layout.itemAt(1).widget()
                if hasattr(widget, 'objectName') and widget.objectName():
                    param_name = widget.objectName()
                    if param_name in parameters:
                        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                            widget.setValue(parameters[param_name])
                            
    def _load_presets(self):
        """Load presets from config file."""
        try:
            preset_file = os.path.join('config', 'filter_presets.json')
            if os.path.exists(preset_file):
                with open(preset_file, 'r') as f:
                    self.presets = json.load(f)
                    for name in self.presets.keys():
                        self.preset_combo.addItem(name)
        except Exception as e:
            print(f"Error loading presets: {e}")
            self.presets = {}
            
    def _save_presets(self):
        """Save presets to config file."""
        try:
            os.makedirs('config', exist_ok=True)
            preset_file = os.path.join('config', 'filter_presets.json')
            with open(preset_file, 'w') as f:
                json.dump(self.presets, f, indent=2)
        except Exception as e:
            print(f"Error saving presets: {e}")
            
    def reset(self):
        """Reset panel to default state."""
        self.current_filter = None
        self.current_parameters = {}
        self.filter_combo.setCurrentIndex(0)
        self.preset_combo.setCurrentIndex(0)
        # Clear parameter controls if any exist
        for control in getattr(self, '_parameter_controls', []):
            control.setParent(None)
        self._parameter_controls = []
        
    def get_current_state(self) -> Dict[str, Any]:
        """Get current panel state for session saving."""
        return {
            'current_filter': self.current_filter,
            'current_parameters': self.current_parameters.copy(),
            'presets': self.presets.copy()
        }
        
    def set_state(self, state: Dict[str, Any]):
        """Set panel state from session loading."""
        if 'current_filter' in state and state['current_filter']:
            # Find and set the filter in combo box
            index = self.filter_combo.findText(state['current_filter'])
            if index >= 0:
                self.filter_combo.setCurrentIndex(index)
                
        if 'current_parameters' in state:
            self.current_parameters = state['current_parameters'].copy()
            self._set_current_parameters(self.current_parameters)
            
        if 'presets' in state:
            self.presets.update(state['presets'])
            
    def load_image(self, image_array: np.ndarray):
        """Load image for filtering."""
        # Update histogram with new image
        self.update_histogram(image_array)
        
    def update_preview(self, preview_image: np.ndarray):
        """Update filter preview display."""
        # This would update a preview widget if one exists
        # For now, just update histogram
        if hasattr(self, 'histogram_view'):
            self.update_histogram(preview_image)
            
    def show_progress(self, progress_info):
        """Show pipeline progress information in the filter panel."""
        stage = progress_info.get('stage', '')
        status = progress_info.get('status', '')
        progress = progress_info.get('progress', 0)

        # Update the status label if it exists
        if hasattr(self, 'status_label'):
            self.status_label.setText(f"Pipeline: {stage.title()} - {status}")

        # Update any progress indicators
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(progress)
            self.progress_bar.setVisible(True if progress < 100 else False)

        # Log the progress for debugging
        logger.debug(f"FilterPanel progress: {stage} - {status} ({progress}%)")
        
    def get_current_config(self):
        """Get current filter configuration for pipeline processing."""
        config = {
            'enabled_filters': [],
            'filter_parameters': {},
            'preset_name': getattr(self, 'current_preset_name', None)
        }
        
        # Collect enabled filters and their parameters
        for filter_name, checkbox in self.filter_checkboxes.items():
            if checkbox.isChecked():
                config['enabled_filters'].append(filter_name)
                
                # Get parameters for this filter
                if filter_name in self.filter_controls:
                    params = {}
                    for param_name, control in self.filter_controls[filter_name].items():
                        if hasattr(control, 'value'):
                            params[param_name] = control.value()
                        elif hasattr(control, 'text'):
                            params[param_name] = control.text()
                    config['filter_parameters'][filter_name] = params
        
        return config
