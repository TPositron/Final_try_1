"""
Filter Panel - Comprehensive Image Processing and Filtering Interface

This module provides a comprehensive filtering panel with dynamic filter loading,
parameter controls, histogram visualization, and preset management.

Main Class:
- FilterPanel: Main filtering interface with controls and visualization

Key Methods:
- setup_ui(): Initializes UI with filter controls and histogram display
- set_available_filters(): Sets list of available filters
- _on_filter_changed(): Handles filter selection changes
- _update_parameter_controls(): Updates parameter controls for selected filter
- _on_apply_clicked(): Applies selected filter with current parameters
- _on_preview_clicked(): Generates filter preview
- _on_reset_clicked(): Resets all filters to original image
- update_histogram(): Updates histogram display with image data
- _load_dynamic_filters(): Loads filters from configuration files
- _on_save_clicked(): Saves current filtered image

Signals Emitted:
- filter_applied(str, dict): Filter applied with name and parameters
- filter_previewed(str, dict): Filter previewed with parameters
- filter_reset(): All filters reset to original
- preset_saved(str, dict): Filter preset saved
- preset_loaded(str, dict): Filter preset loaded
- save_image_requested(): Image save requested

Dependencies:
- Uses: PySide6.QtWidgets, PySide6.QtCore (Qt framework)
- Uses: numpy (array processing)
- Uses: json, os, logging (utilities)
- Uses: typing (type hints)
- Uses: ui/components (HistogramView, DynamicParameterPanel)
- Called by: UI main window and filtering workflow
- Coordinates with: Image processing services and histogram displays

Features:
- Dynamic filter loading from configuration files
- Real-time parameter adjustment with preview capability
- Filter history tracking and management
- Preset system for saving and loading filter configurations
- Integrated histogram display for image analysis
- Progress tracking and status updates
- Export capabilities for filtered images
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                               QComboBox, QPushButton, QListWidget, QSplitter,
                               QLabel, QSlider, QSpinBox, QDoubleSpinBox, QInputDialog, QProgressBar)
from PySide6.QtCore import Qt, Signal
from ..components.histogram_view import HistogramView
from ..components.dynamic_parameter_widgets import DynamicParameterPanel
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
    save_image_requested = Signal()  # Request to save current filtered image
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_filter = None
        self.current_parameters = {}
        self.available_filters = []
        self.presets = {}  # Store filter presets
        self.filter_folder_path = "config/filter_presets"  # Dynamic filter loading path
        self.setup_ui()
        self._load_dynamic_filters()
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
        
        # Dynamic parameter panel (replaces old static parameter controls)
        self.dynamic_params = DynamicParameterPanel()
        controls_layout.addWidget(self.dynamic_params)
        
        # Connect dynamic parameter signals
        self.dynamic_params.parameters_changed.connect(self._on_parameters_changed)
        self.dynamic_params.parameters_preview.connect(self._on_parameters_preview)
        
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
        
        # Save button (separate row)
        save_button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save SEM Image")
        self.save_button.clicked.connect(self._on_save_clicked)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        save_button_layout.addWidget(self.save_button)
        controls_layout.addLayout(save_button_layout)
        
        controls_layout.addStretch()

        # Progress and status widgets
        self.status_label = QLabel("Ready.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.status_label)
        controls_layout.addWidget(self.progress_bar)

        # Right side: Create structured right panel with histogram at top
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)
        
        # Histogram at the top of right panel with increased size
        self.histogram_view = HistogramView()
        self.histogram_view.setMinimumHeight(300)  # Increased from default
        self.histogram_view.setMaximumHeight(400)  # Set reasonable maximum
        right_layout.addWidget(self.histogram_view)
        
        # Add stretch to push histogram to top and allow space for future components
        right_layout.addStretch()
        
        # Add to splitter
        splitter.addWidget(controls_widget)
        splitter.addWidget(right_panel)
        
        # Adjust splitter proportions to give more space to right panel for larger histogram
        splitter.setSizes([250, 750])
        
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
        """Update parameter controls based on selected filter using dynamic parameter system."""
        # Use the dynamic parameter panel to update controls
        if hasattr(self, 'dynamic_params'):
            self.dynamic_params.update_for_filter(filter_name)
        
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Get current parameter values from dynamic parameter panel."""
        if hasattr(self, 'dynamic_params'):
            return self.dynamic_params.get_current_parameters()
        return {}
        
    def _on_parameters_changed(self, parameters: Dict[str, Any]):
        """Handle parameter value changes from dynamic parameter panel."""
        self.current_parameters = parameters
        if self.current_filter:
            self.filter_previewed.emit(self.current_filter, parameters)
            
    def _on_parameters_preview(self, parameters: Dict[str, Any]):
        """Handle preview request from dynamic parameter panel."""
        self.current_parameters = parameters
        if self.current_filter:
            self.filter_previewed.emit(self.current_filter, parameters)
            
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
        if hasattr(self, 'dynamic_params'):
            self.dynamic_params.reset()
        
    def update_histogram(self, image_data: np.ndarray):
        """Update the histogram display."""
        self.histogram_view.update_histogram(image_data)
        
    def update_kernel_view(self, kernel: np.ndarray, title: str = "Filter Kernel"):
        """Update the kernel visualization."""
        self.histogram_view.update_kernel_view(kernel, title)
            
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
        """Set parameter values in the dynamic parameter panel."""
        if hasattr(self, 'dynamic_params'):
            self.dynamic_params.set_parameters(parameters)
        self.current_parameters = parameters
                            
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
        # Reset dynamic parameter panel
        if hasattr(self, 'dynamic_params'):
            self.dynamic_params.reset()
        
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
        
        # Include current filter if selected
        if self.current_filter:
            config['enabled_filters'].append(self.current_filter)
            config['filter_parameters'][self.current_filter] = self._get_current_parameters()
        
        return config

    def _on_save_clicked(self):
        """Handle save button click."""
        try:
            # Emit signal to request saving of current filtered image
            self.save_image_requested.emit()
            
            # Update status
            if hasattr(self, 'status_label'):
                self.status_label.setText("Saving current filtered image...")
            
            logger.info("Save image requested")
            
        except Exception as e:
            logger.error(f"Error in save clicked handler: {e}")
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"Save error: {str(e)}")

    def _load_dynamic_filters(self):
        """Load filters dynamically from filter folder for Step 5 implementation."""
        try:
            filter_folder = os.path.join(os.getcwd(), self.filter_folder_path)
            if not os.path.exists(filter_folder):
                logger.warning(f"Filter folder not found: {filter_folder}")
                return
            
            dynamic_filters = []
            filter_configs = {}
            
            # Scan filter folder for JSON files
            for filename in os.listdir(filter_folder):
                if filename.endswith('.json'):
                    file_path = os.path.join(filter_folder, filename)
                    try:
                        with open(file_path, 'r') as f:
                            filter_config = json.load(f)
                        
                        # Extract filter name and configuration
                        filter_name = filter_config.get('name', filename.replace('.json', ''))
                        dynamic_filters.append(filter_name)
                        filter_configs[filter_name] = filter_config
                        
                        logger.info(f"Loaded dynamic filter: {filter_name}")
                        
                    except Exception as e:
                        logger.error(f"Error loading filter {filename}: {e}")
            
            # Update available filters with dynamically loaded ones
            self.available_filters.extend(dynamic_filters)
            self.filter_configs = filter_configs
            
            # Auto-generate UI for the dynamic filters
            self._auto_generate_filter_ui()
            
            logger.info(f"Loaded {len(dynamic_filters)} dynamic filters from {filter_folder}")
            
        except Exception as e:
            logger.error(f"Error loading dynamic filters: {e}")
    
    def _auto_generate_filter_ui(self):
        """Auto-generate UI based on available filters for Step 5."""
        try:
            # Update filter dropdown with all available filters
            if hasattr(self, 'filter_combo'):
                current_items = [self.filter_combo.itemText(i) for i in range(self.filter_combo.count())]
                
                # Add new filters that aren't already in the combo
                for filter_name in self.available_filters:
                    if filter_name not in current_items:
                        self.filter_combo.addItem(filter_name)
            
            # Update dynamic parameter panel to handle new filters
            if hasattr(self, 'dynamic_params'):
                # Store filter configurations for parameter generation
                if hasattr(self, 'filter_configs'):
                    self.dynamic_params.set_filter_configs(self.filter_configs)
            
            logger.info("Auto-generated UI for dynamic filters")
            
        except Exception as e:
            logger.error(f"Error auto-generating filter UI: {e}")
