"""
Sequential Filtering Panels - 4-Stage Image Processing Workflow

This module implements a comprehensive 4-stage sequential image processing workflow
for SEM image enhancement. Each stage builds upon the previous result, providing
a structured approach to image filtering and analysis.

Main Classes:
- SequentialFilterConfigManager: Manages filter configurations for all stages
- StageParameterWidget: Parameter control widget for individual filter parameters
- StageControlPanel: Control panel for each processing stage
- SequentialFilteringLeftPanel: Left panel with stage controls and filter selection
- SequentialFilteringRightPanel: Right panel with progress tracking and statistics

Key Methods (SequentialFilterConfigManager):
- _initialize_stages(): Sets up the 4 processing stages
- _load_sequential_filters(): Loads all filters organized by stage
- get_filters_for_stage(): Gets filters available for a specific stage
- get_stage_config(): Gets configuration for a processing stage

Key Methods (StageControlPanel):
- setup_ui(): Creates stage-specific UI with filter selection and parameters
- _on_filter_selected(): Handles filter selection and parameter setup
- get_current_parameters(): Gets current parameter values for the stage
- reset_stage(): Resets stage to initial state

Key Methods (SequentialFilteringLeftPanel):
- init_panel(): Initializes panel with all 4 stage controls
- _on_stage_preview(): Handles stage preview requests
- _on_stage_apply(): Handles stage application requests
- reset_all_stages(): Resets all stages to initial state

Key Methods (SequentialFilteringRightPanel):
- update_stage_progress(): Updates progress indicators for stages
- update_histogram(): Updates histogram display for current stage
- set_processing_status(): Sets overall processing status message

Signals Emitted:
- stage_preview_requested(int, str, dict): Stage preview with parameters
- stage_apply_requested(int, str, dict): Stage application with parameters
- stage_reset_requested(int): Stage reset requested
- stage_save_requested(int): Stage save requested
- reset_all_requested(): Reset all stages requested

Dependencies:
- Uses: PySide6.QtWidgets, PySide6.QtCore, PySide6.QtGui (Qt framework)
- Uses: numpy, cv2 (image processing)
- Uses: json, logging, os (system operations)
- Uses: typing, dataclasses, enum (type definitions)
- Uses: ui/view_manager.ViewMode, ui/components/histogram_view.HistogramView
- Called by: Main filtering workflow and UI management
- Coordinates with: Image processing services and filter operations

Processing Stages:
1. Contrast Enhancement: CLAHE, gamma correction, histogram equalization
2. Blur & Noise Reduction: Gaussian blur, median filter, bilateral filter, NLM denoising
3. Binarization: Simple threshold, adaptive threshold, Otsu threshold
4. Edge Detection: Canny, Laplacian, Sobel edge detection

Features:
- Sequential 4-stage processing workflow with stage dependencies
- Comprehensive filter library with pre-configured parameters
- Real-time parameter adjustment with immediate feedback
- Progress tracking with visual indicators for each stage
- Histogram analysis and statistics for each processing stage
- Stage-specific controls with preview and apply functionality
- Dark theme styling with color-coded stage indicators
- Parameter validation and error handling
- Save and reset functionality for individual stages
- Scrollable interface for compact display of all stages
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, 
    QComboBox, QLabel, QSplitter, QScrollArea, QListWidget, QFrame, 
    QProgressBar, QTextEdit, QCheckBox, QListWidgetItem, QSpinBox,
    QDoubleSpinBox, QSlider, QGridLayout, QFileDialog, QMessageBox,
    QTabWidget, QLineEdit, QStackedWidget
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QPixmap, QPainter, QImage, QValidator, QDoubleValidator, QIntValidator
import numpy as np
import cv2
import json
import logging
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Import your existing components
from src.ui.view_manager import ViewMode
from src.ui.components.histogram_view import HistogramView

logger = logging.getLogger(__name__)

class ProcessingStage(Enum):
    """Sequential processing stages."""
    CONTRAST_ENHANCEMENT = 0
    BLUR_NOISE_REDUCTION = 1
    BINARIZATION = 2
    EDGE_DETECTION = 3

@dataclass
class StageConfig:
    """Configuration for each processing stage."""
    name: str
    display_name: str
    description: str
    filters: List[str]
    icon: str
    color: str

@dataclass
class FilterParameter:
    """Simplified filter parameter definition."""
    name: str
    param_type: str  # 'int', 'float', 'choice'
    default_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: float = 1.0
    choices: Optional[List[str]] = None
    description: str = ""

@dataclass
class FilterConfig:
    """Filter configuration."""
    name: str
    display_name: str
    description: str
    stage: ProcessingStage
    parameters: List[FilterParameter]

class SequentialFilterConfigManager:
    """Manages sequential filter configurations for 4-stage workflow."""
    
    def __init__(self):
        self.stages: Dict[ProcessingStage, StageConfig] = {}
        self.filters: Dict[str, FilterConfig] = {}
        self._initialize_stages()
        self._load_sequential_filters()
    
    def _initialize_stages(self):
        """Initialize the 4 processing stages."""
        self.stages = {
            ProcessingStage.CONTRAST_ENHANCEMENT: StageConfig(
                name="contrast_enhancement",
                display_name="1. Contrast Enhancement",
                description="Improve image contrast and brightness",
                filters=["clahe", "gamma_correction", "histogram_equalization"],
                icon="📈",
                color="#ff6b35"
            ),
            ProcessingStage.BLUR_NOISE_REDUCTION: StageConfig(
                name="blur_noise_reduction", 
                display_name="2. Blur & Noise Reduction",
                description="Reduce noise and smooth the image",
                filters=["gaussian_blur", "median_filter", "bilateral_filter", "nlm_denoising"],
                icon="🔵",
                color="#4ecdc4"
            ),
            ProcessingStage.BINARIZATION: StageConfig(
                name="binarization",
                display_name="3. Binarization", 
                description="Convert to binary image",
                filters=["threshold", "adaptive_threshold", "otsu_threshold"],
                icon="⚫",
                color="#45b7d1"
            ),
            ProcessingStage.EDGE_DETECTION: StageConfig(
                name="edge_detection",
                display_name="4. Edge Detection",
                description="Detect edges and boundaries",
                filters=["canny", "laplacian", "sobel"],
                icon="📐",
                color="#96ceb4"
            )
        }
    
    def _load_sequential_filters(self):
        """Load filters organized by sequential stages."""
        
        # STAGE 1: CONTRAST ENHANCEMENT
        self.add_filter(FilterConfig(
            name="clahe",
            display_name="CLAHE",
            description="Contrast Limited Adaptive Histogram Equalization",
            stage=ProcessingStage.CONTRAST_ENHANCEMENT,
            parameters=[
                FilterParameter("clip_limit", "float", 2.0, 0.1, 10.0, 0.1, description="Contrast limiting threshold"),
                FilterParameter("tile_grid_size", "int", 8, 1, 16, 1, description="Tile grid size")
            ]
        ))
        
        self.add_filter(FilterConfig(
            name="gamma_correction",
            display_name="Gamma Correction",
            description="Adjust gamma for brightness correction",
            stage=ProcessingStage.CONTRAST_ENHANCEMENT,
            parameters=[
                FilterParameter("gamma", "float", 1.0, 0.1, 3.0, 0.1, description="Gamma value (< 1 = brighter, > 1 = darker)")
            ]
        ))
        
        self.add_filter(FilterConfig(
            name="histogram_equalization",
            display_name="Histogram Equalization",
            description="Standard histogram equalization",
            stage=ProcessingStage.CONTRAST_ENHANCEMENT,
            parameters=[]
        ))
        
        # STAGE 2: BLUR & NOISE REDUCTION
        self.add_filter(FilterConfig(
            name="gaussian_blur",
            display_name="Gaussian Blur",
            description="Gaussian smoothing filter",
            stage=ProcessingStage.BLUR_NOISE_REDUCTION,
            parameters=[
                FilterParameter("kernel_size", "int", 5, 3, 31, 2, description="Kernel size (must be odd)"),
                FilterParameter("sigma", "float", 1.0, 0.1, 10.0, 0.1, description="Standard deviation")
            ]
        ))
        
        self.add_filter(FilterConfig(
            name="median_filter",
            display_name="Median Filter",
            description="Median filtering for noise reduction",
            stage=ProcessingStage.BLUR_NOISE_REDUCTION,
            parameters=[
                FilterParameter("kernel_size", "int", 5, 3, 15, 2, description="Kernel size (must be odd)")
            ]
        ))
        
        self.add_filter(FilterConfig(
            name="bilateral_filter",
            display_name="Bilateral Filter",
            description="Edge-preserving smoothing",
            stage=ProcessingStage.BLUR_NOISE_REDUCTION,
            parameters=[
                FilterParameter("d", "int", 9, 5, 15, 2, description="Neighborhood diameter"),
                FilterParameter("sigma_color", "float", 75.0, 10.0, 150.0, 5.0, description="Color sigma"),
                FilterParameter("sigma_space", "float", 75.0, 10.0, 150.0, 5.0, description="Space sigma")
            ]
        ))
        
        self.add_filter(FilterConfig(
            name="nlm_denoising",
            display_name="Non-Local Means Denoising",
            description="Advanced noise reduction",
            stage=ProcessingStage.BLUR_NOISE_REDUCTION,
            parameters=[
                FilterParameter("h", "float", 10.0, 1.0, 30.0, 1.0, description="Filtering strength"),
                FilterParameter("template_window_size", "int", 7, 3, 15, 2, description="Template patch size"),
                FilterParameter("search_window_size", "int", 21, 7, 35, 2, description="Search window size")
            ]
        ))
        
        # STAGE 3: BINARIZATION
        self.add_filter(FilterConfig(
            name="threshold",
            display_name="Simple Threshold",
            description="Binary thresholding",
            stage=ProcessingStage.BINARIZATION,
            parameters=[
                FilterParameter("threshold_value", "int", 127, 0, 255, 1, description="Threshold value"),
                FilterParameter("max_value", "int", 255, 1, 255, 1, description="Maximum value")
            ]
        ))
        
        self.add_filter(FilterConfig(
            name="adaptive_threshold",
            display_name="Adaptive Threshold",
            description="Adaptive binary thresholding",
            stage=ProcessingStage.BINARIZATION,
            parameters=[
                FilterParameter("max_value", "int", 255, 1, 255, 1, description="Maximum value"),
                FilterParameter("block_size", "int", 11, 3, 31, 2, description="Block size (must be odd)"),
                FilterParameter("c", "float", 2.0, -10.0, 10.0, 0.5, description="Constant subtracted from mean")
            ]
        ))
        
        self.add_filter(FilterConfig(
            name="otsu_threshold",
            display_name="Otsu Threshold",
            description="Automatic Otsu thresholding",
            stage=ProcessingStage.BINARIZATION,
            parameters=[]
        ))
        
        # STAGE 4: EDGE DETECTION
        self.add_filter(FilterConfig(
            name="canny",
            display_name="Canny Edge Detection",
            description="Canny edge detector",
            stage=ProcessingStage.EDGE_DETECTION,
            parameters=[
                FilterParameter("low_threshold", "int", 50, 1, 255, 1, description="Lower threshold"),
                FilterParameter("high_threshold", "int", 150, 1, 255, 1, description="Upper threshold"),
                FilterParameter("aperture_size", "int", 3, 3, 7, 2, description="Sobel kernel size")
            ]
        ))
        
        self.add_filter(FilterConfig(
            name="laplacian",
            display_name="Laplacian Edge Detection",
            description="Laplacian edge detector",
            stage=ProcessingStage.EDGE_DETECTION,
            parameters=[
                FilterParameter("ksize", "int", 3, 1, 31, 2, description="Kernel size"),
                FilterParameter("scale", "float", 1.0, 0.1, 10.0, 0.1, description="Scale factor"),
                FilterParameter("delta", "float", 0.0, -50.0, 50.0, 1.0, description="Delta added to result")
            ]
        ))
        
        self.add_filter(FilterConfig(
            name="sobel",
            display_name="Sobel Edge Detection",
            description="Sobel edge detector",
            stage=ProcessingStage.EDGE_DETECTION,
            parameters=[
                FilterParameter("dx", "int", 1, 0, 2, 1, description="X derivative order"),
                FilterParameter("dy", "int", 1, 0, 2, 1, description="Y derivative order"),
                FilterParameter("ksize", "int", 3, 1, 31, 2, description="Kernel size")
            ]
        ))
    
    def add_filter(self, filter_config: FilterConfig):
        """Add a filter configuration."""
        self.filters[filter_config.name] = filter_config
    
    def get_filter(self, name: str) -> Optional[FilterConfig]:
        """Get filter configuration by name."""
        return self.filters.get(name)
    
    def get_filters_for_stage(self, stage: ProcessingStage) -> List[FilterConfig]:
        """Get filters for a specific stage."""
        return [f for f in self.filters.values() if f.stage == stage]
    
    def get_stage_config(self, stage: ProcessingStage) -> StageConfig:
        """Get stage configuration."""
        return self.stages[stage]

class StageParameterWidget(QWidget):
    """Parameter widget for sequential processing stages."""
    
    value_changed = Signal(str, object)
    
    def __init__(self, parameter: FilterParameter, parent=None):
        super().__init__(parent)
        self.parameter = parameter
        self.current_value = parameter.default_value
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        
        # Parameter label with improved styling
        label = QLabel(f"{self.parameter.description}:")
        label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-weight: bold;
                font-size: 10px;
                padding: 2px;
            }
        """)
        layout.addWidget(label)
        
        # Control layout
        control_layout = QHBoxLayout()
        control_layout.setSpacing(5)
        
        # Create appropriate control
        if self.parameter.param_type == "choice":
            self._setup_choice_control(control_layout)
        elif self.parameter.param_type == "int":
            self._setup_int_control(control_layout)
        elif self.parameter.param_type == "float":
            self._setup_float_control(control_layout)
        
        layout.addLayout(control_layout)
    
    def _setup_choice_control(self, layout):
        """Setup choice parameter with combo box."""
        self.combo = QComboBox()
        self.combo.addItems(self.parameter.choices or [])
        if self.parameter.default_value in (self.parameter.choices or []):
            self.combo.setCurrentText(str(self.parameter.default_value))
        self.combo.currentTextChanged.connect(self._on_choice_changed)
        self.combo.setStyleSheet(self._get_combo_style())
        layout.addWidget(self.combo)
    
    def _setup_int_control(self, layout):
        """Setup integer parameter with spinbox."""
        self.spinbox = QSpinBox()
        min_val = int(self.parameter.min_value or 0)
        max_val = int(self.parameter.max_value or 100)
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setValue(int(self.parameter.default_value))
        self.spinbox.setSingleStep(int(self.parameter.step))
        self.spinbox.valueChanged.connect(self._on_int_changed)
        self.spinbox.setStyleSheet(self._get_spinbox_style())
        self.spinbox.setMinimumWidth(60)
        layout.addWidget(self.spinbox)
    
    def _setup_float_control(self, layout):
        """Setup float parameter with double spinbox."""
        self.double_spinbox = QDoubleSpinBox()
        min_val = float(self.parameter.min_value or 0.0)
        max_val = float(self.parameter.max_value or 10.0)
        self.double_spinbox.setRange(min_val, max_val)
        self.double_spinbox.setValue(float(self.parameter.default_value))
        self.double_spinbox.setSingleStep(float(self.parameter.step))
        self.double_spinbox.setDecimals(2)
        self.double_spinbox.valueChanged.connect(self._on_float_changed)
        self.double_spinbox.setStyleSheet(self._get_spinbox_style())
        self.double_spinbox.setMinimumWidth(60)
        layout.addWidget(self.double_spinbox)
    
    def _on_choice_changed(self, text: str):
        self.current_value = text
        self.value_changed.emit(self.parameter.name, self.current_value)
    
    def _on_int_changed(self, value: int):
        self.current_value = value
        self.value_changed.emit(self.parameter.name, self.current_value)
    
    def _on_float_changed(self, value: float):
        self.current_value = value
        self.value_changed.emit(self.parameter.name, self.current_value)
    
    def get_value(self):
        return self.current_value
    
    def set_value(self, value):
        if self.parameter.param_type == "int" and hasattr(self, 'spinbox'):
            self.spinbox.setValue(int(value))
        elif self.parameter.param_type == "float" and hasattr(self, 'double_spinbox'):
            self.double_spinbox.setValue(float(value))
        elif self.parameter.param_type == "choice" and hasattr(self, 'combo'):
            self.combo.setCurrentText(str(value))
        self.current_value = value
    
    def _get_combo_style(self):
        return """
            QComboBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px;
                min-height: 12px;
                font-size: 10px;
            }
        """
    
    def _get_spinbox_style(self):
        return """
            QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px;
                min-height: 12px;
                font-size: 10px;
            }
        """

class StageControlPanel(QWidget):
    """Control panel for each sequential processing stage."""
    
    # Signals for stage operations
    preview_requested = Signal(int)  # stage_index
    apply_requested = Signal(int)    # stage_index
    reset_requested = Signal(int)    # stage_index
    save_requested = Signal(int)     # stage_index
    
    def __init__(self, stage: ProcessingStage, stage_config: StageConfig, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.stage_config = stage_config
        self.config_manager = SequentialFilterConfigManager()
        self.current_filter = None
        self.parameter_widgets = {}
        self.is_applied = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the stage control panel UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # Stage header with icon and status
        header_layout = QHBoxLayout()
        
        # Stage icon and title
        header_label = QLabel(f"{self.stage_config.icon} {self.stage_config.display_name}")
        header_label.setStyleSheet(f"""
            QLabel {{
                color: {self.stage_config.color};
                font-weight: bold;
                font-size: 12px;
                padding: 4px;
            }}
        """)
        
        # Status indicator
        self.status_indicator = QLabel("○")
        self.status_indicator.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_indicator)
        
        main_layout.addLayout(header_layout)
        
        # Description
        desc_label = QLabel(self.stage_config.description)
        desc_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                font-style: italic;
                font-size: 10px;
                padding: 4px;
            }
        """)
        desc_label.setWordWrap(True)
        main_layout.addWidget(desc_label)
        
        # Filter selection
        filter_layout = QVBoxLayout()
        
        filter_label = QLabel("Filter:")
        filter_label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 10px;")
        filter_layout.addWidget(filter_label)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("-- Select Filter --", None)
        
        # Add filters for this stage
        filters = self.config_manager.get_filters_for_stage(self.stage)
        for filter_config in filters:
            self.filter_combo.addItem(filter_config.display_name, filter_config.name)
        
        self.filter_combo.currentTextChanged.connect(self._on_filter_selected)
        self.filter_combo.setStyleSheet(self._get_combo_style())
        filter_layout.addWidget(self.filter_combo)
        
        main_layout.addLayout(filter_layout)
        
        # Parameters area (initially hidden)
        self.param_widget = QWidget()
        self.param_layout = QVBoxLayout(self.param_widget)
        self.param_layout.setContentsMargins(0, 0, 0, 0)
        self.param_widget.hide()
        
        main_layout.addWidget(self.param_widget)
        
        # Stage controls
        controls_layout = QGridLayout()
        
        # Row 1: Preview and Apply
        self.preview_btn = QPushButton("👁️ Preview")
        self.preview_btn.setStyleSheet(self._get_button_style("#0078d4", "#106ebe"))
        self.preview_btn.clicked.connect(lambda: self.preview_requested.emit(self.stage.value))
        self.preview_btn.setEnabled(False)
        
        self.apply_btn = QPushButton("✓ Apply")
        self.apply_btn.setStyleSheet(self._get_button_style("#28a745", "#218838"))
        self.apply_btn.clicked.connect(lambda: self._on_apply_clicked())
        self.apply_btn.setEnabled(False)
        
        controls_layout.addWidget(self.preview_btn, 0, 0)
        controls_layout.addWidget(self.apply_btn, 0, 1)
        
        # Row 2: Reset and Save
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.setStyleSheet(self._get_button_style("#dc3545", "#c82333"))
        self.reset_btn.clicked.connect(lambda: self.reset_requested.emit(self.stage.value))
        
        self.save_btn = QPushButton("💾 Save")
        self.save_btn.setStyleSheet(self._get_button_style("#17a2b8", "#138496"))
        self.save_btn.clicked.connect(lambda: self.save_requested.emit(self.stage.value))
        
        controls_layout.addWidget(self.reset_btn, 1, 0)
        controls_layout.addWidget(self.save_btn, 1, 1)
        
        main_layout.addLayout(controls_layout)
        
        # Stage status
        self.stage_status = QLabel("Ready")
        self.stage_status.setStyleSheet("""
            QLabel {
                color: #ffffff;
                background-color: #2a2a2a;
                padding: 4px;
                border-radius: 3px;
                font-size: 9px;
                border: 1px solid #555555;
            }
        """)
        self.stage_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.stage_status)
        
        # Add stretch at the bottom
        main_layout.addStretch()
    
    def _on_filter_selected(self):
        """Handle filter selection."""
        current_data = self.filter_combo.currentData()
        
        if current_data is None:
            self.current_filter = None
            self._update_ui_state()
            return
        
        # Ensure current_data is a string before assigning
        if not isinstance(current_data, str):
            self.current_filter = None
            self._update_ui_state()
            return
            
        self.current_filter = current_data
        
        # Clear existing parameter widgets
        for widget in self.parameter_widgets.values():
            widget.setParent(None)
        self.parameter_widgets.clear()
        
        if self.current_filter and isinstance(self.current_filter, str):
            filter_config = self.config_manager.get_filter(self.current_filter)
            
            if filter_config and filter_config.parameters:
                self.param_widget.show()
                for param in filter_config.parameters:
                    widget = StageParameterWidget(param)
                    widget.value_changed.connect(self._on_parameter_changed)
                    self.parameter_widgets[param.name] = widget
                    self.param_layout.addWidget(widget)
            else:
                self.param_widget.hide()
        else:
            self.param_widget.hide()
        
        self._update_ui_state()
    
    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter changes."""
        pass  # Parameters are automatically updated in widgets
    
    def _on_apply_clicked(self):
        """Handle apply button click."""
        self.apply_requested.emit(self.stage.value)
        self.is_applied = True
        self._update_status_indicator()
    
    def _update_ui_state(self):
        """Update UI state based on current selection."""
        has_filter = self.current_filter is not None and isinstance(self.current_filter, str)
        self.preview_btn.setEnabled(has_filter)
        self.apply_btn.setEnabled(has_filter)
        
        if has_filter and self.current_filter:
            filter_config = self.config_manager.get_filter(self.current_filter)
            if filter_config:
                self.stage_status.setText(f"Ready: {filter_config.display_name}")
            else:
                self.stage_status.setText(f"Ready: {self.current_filter}")
        else:
            self.stage_status.setText("Select a filter")
    
    def _update_status_indicator(self):
        """Update the status indicator based on stage state."""
        if self.is_applied:
            self.status_indicator.setText("●")
            self.status_indicator.setStyleSheet(f"""
                QLabel {{
                    color: {self.stage_config.color};
                    font-size: 14px;
                    font-weight: bold;
                }}
            """)
        else:
            self.status_indicator.setText("○")
            self.status_indicator.setStyleSheet("""
                QLabel {
                    color: #666666;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
    
    def get_current_parameters(self) -> dict:
        """Get current parameter values."""
        parameters = {}
        for param_name, widget in self.parameter_widgets.items():
            value = widget.get_value()
            
            # Special handling for certain parameters
            if param_name == "tile_grid_size":
                parameters["tile_grid_x"] = value
                parameters["tile_grid_y"] = value
            elif param_name == "sigma":
                parameters["sigma_x"] = value
                parameters["sigma_y"] = value
            elif param_name == "kernel_size":
                # Ensure odd values for OpenCV filters
                if value % 2 == 0:
                    value += 1
                parameters[param_name] = value
            else:
                parameters[param_name] = value
        
        return parameters
    
    def reset_stage(self):
        """Reset this stage."""
        self.is_applied = False
        self.filter_combo.setCurrentIndex(0)
        self._update_status_indicator()
        self.stage_status.setText("Reset")
    
    def set_processing_status(self, message: str):
        """Set processing status message."""
        self.stage_status.setText(message)
    
    def _get_combo_style(self):
        return """
            QComboBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px;
                min-height: 16px;
                font-size: 10px;
            }
        """
    
    def _get_button_style(self, bg_color: str, hover_color: str):
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                padding: 6px 10px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 9px;
                min-height: 16px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:disabled {{
                background-color: #555555;
                color: #999999;
            }}
        """

class SequentialFilteringLeftPanel(QWidget):
    """
    Phase 3: Sequential Filtering Left Panel
    4-Step Sequential Workflow: Contrast Enhancement → Blur & Noise Reduction → Binarization → Edge Detection
    """
    
    # Signals for sequential processing
    stage_preview_requested = Signal(int, str, dict)  # stage, filter_name, parameters
    stage_apply_requested = Signal(int, str, dict)    # stage, filter_name, parameters
    stage_reset_requested = Signal(int)               # stage
    stage_save_requested = Signal(int)                # stage
    reset_all_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.config_manager = SequentialFilterConfigManager()
        self.stage_panels = {}
        self.current_stage = ProcessingStage.CONTRAST_ENHANCEMENT
        
        self.init_panel()
    
    def init_panel(self):
        """Initialize the sequential filtering panel."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)
        
        # Panel header
        header_layout = QHBoxLayout()
        
        title_label = QLabel("🔄 Sequential Filtering")
        title_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-weight: bold;
                font-size: 14px;
                padding: 4px;
            }
        """)
        
        # Global reset button
        self.global_reset_btn = QPushButton("Reset All Stages")
        self.global_reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        self.global_reset_btn.clicked.connect(self.reset_all_requested)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.global_reset_btn)
        
        main_layout.addLayout(header_layout)
        
        # Description
        desc_label = QLabel(
            "Process your image through 4 sequential stages. Each stage builds on the previous result."
        )
        desc_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                font-style: italic;
                font-size: 10px;
                padding: 8px;
                background-color: #2a2a2a;
                border-radius: 4px;
                border: 1px solid #444444;
            }
        """)
        desc_label.setWordWrap(True)
        main_layout.addWidget(desc_label)
        
        # Scroll area for stage panels
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #2b2b2b;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #555555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666666;
            }
        """)
        
        # Container for stage panels
        stages_container = QWidget()
        stages_layout = QVBoxLayout(stages_container)
        stages_layout.setSpacing(10)
        
        # Create stage panels in order
        for stage in ProcessingStage:
            stage_config = self.config_manager.get_stage_config(stage)
            panel = StageControlPanel(stage, stage_config)
            
            # Connect stage signals
            panel.preview_requested.connect(self._on_stage_preview)
            panel.apply_requested.connect(self._on_stage_apply)
            panel.reset_requested.connect(self._on_stage_reset)
            panel.save_requested.connect(self._on_stage_save)
            
            # Create stage group box
            stage_group = QGroupBox()
            stage_group.setStyleSheet(f"""
                QGroupBox {{
                    border: 2px solid {stage_config.color};
                    border-radius: 8px;
                    margin-top: 5px;
                    padding-top: 5px;
                    background-color: #2b2b2b;
                }}
            """)
            
            stage_group_layout = QVBoxLayout(stage_group)
            stage_group_layout.addWidget(panel)
            
            self.stage_panels[stage] = panel
            stages_layout.addWidget(stage_group)
        
        # Add stretch at the end
        stages_layout.addStretch()
        
        scroll_area.setWidget(stages_container)
        main_layout.addWidget(scroll_area)
        
        print("✓ Sequential Filtering Left Panel initialized")
    
    def _on_stage_preview(self, stage_index: int):
        """Handle stage preview request."""
        stage = ProcessingStage(stage_index)
        panel = self.stage_panels[stage]
        
        if panel.current_filter:
            parameters = panel.get_current_parameters()
            panel.set_processing_status("Previewing...")
            self.stage_preview_requested.emit(stage_index, panel.current_filter, parameters)
    
    def _on_stage_apply(self, stage_index: int):
        """Handle stage apply request."""
        stage = ProcessingStage(stage_index)
        panel = self.stage_panels[stage]
        
        if panel.current_filter:
            parameters = panel.get_current_parameters()
            panel.set_processing_status("Applying...")
            self.stage_apply_requested.emit(stage_index, panel.current_filter, parameters)
    
    def _on_stage_reset(self, stage_index: int):
        """Handle stage reset request."""
        stage = ProcessingStage(stage_index)
        panel = self.stage_panels[stage]
        panel.reset_stage()
        self.stage_reset_requested.emit(stage_index)
    
    def _on_stage_save(self, stage_index: int):
        """Handle stage save request."""
        self.stage_save_requested.emit(stage_index)
    
    def reset_all_stages(self):
        """Reset all stages."""
        for panel in self.stage_panels.values():
            panel.reset_stage()
    
    def set_stage_completed(self, stage_index: int, success: bool):
        """Mark a stage as completed or failed."""
        stage = ProcessingStage(stage_index)
        panel = self.stage_panels[stage]
        
        if success:
            panel.set_processing_status("✓ Applied")
            panel.is_applied = True
        else:
            panel.set_processing_status("❌ Failed")
            panel.is_applied = False
        
        panel._update_status_indicator()

class SequentialFilteringRightPanel(QWidget):
    """
    Phase 3: Sequential Filtering Right Panel
    Enhanced to show progression through sequential stages
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_stage_results = {}  # Store intermediate results
        self.init_panel()
    
    def init_panel(self):
        """Initialize the sequential filtering right panel."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)
        
        # Processing Progress Section
        progress_group = QGroupBox("🔄 Processing Progress")
        progress_group.setStyleSheet(self._get_group_style())
        progress_layout = QVBoxLayout(progress_group)
        
        # Stage progress indicators
        self.stage_progress = {}
        stages_layout = QVBoxLayout()
        
        for stage in ProcessingStage:
            stage_config = SequentialFilterConfigManager().get_stage_config(stage)
            
            stage_layout = QHBoxLayout()
            
            # Stage indicator
            indicator = QLabel("○")
            indicator.setStyleSheet(f"""
                QLabel {{
                    color: #666666;
                    font-size: 12px;
                    font-weight: bold;
                    min-width: 16px;
                }}
            """)
            
            # Stage label
            label = QLabel(f"{stage_config.icon} {stage_config.display_name}")
            label.setStyleSheet(f"""
                QLabel {{
                    color: {stage_config.color};
                    font-size: 10px;
                    font-weight: bold;
                }}
            """)
            
            # Status label
            status = QLabel("Pending")
            status.setStyleSheet("""
                QLabel {
                    color: #cccccc;
                    font-size: 9px;
                    font-style: italic;
                }
            """)
            
            stage_layout.addWidget(indicator)
            stage_layout.addWidget(label)
            stage_layout.addStretch()
            stage_layout.addWidget(status)
            
            self.stage_progress[stage] = {
                'indicator': indicator,
                'label': label,
                'status': status
            }
            
            stages_layout.addLayout(stage_layout)
        
        progress_layout.addLayout(stages_layout)
        main_layout.addWidget(progress_group)
        
        # Current Stage Histogram
        histogram_group = QGroupBox("📊 Current Stage Histogram")
        histogram_group.setStyleSheet(self._get_group_style())
        histogram_layout = QVBoxLayout(histogram_group)
        
        self.histogram_view = HistogramView()
        histogram_layout.addWidget(self.histogram_view)
        
        main_layout.addWidget(histogram_group)
        
        # Stage Statistics
        stats_group = QGroupBox("📈 Stage Statistics")
        stats_group.setStyleSheet(self._get_group_style())
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(120)
        self.stats_text.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9px;
                padding: 4px;
            }
        """)
        stats_layout.addWidget(self.stats_text)
        
        main_layout.addWidget(stats_group)
        
        # Processing Status
        status_group = QGroupBox("⚡ Processing Status")
        status_group.setStyleSheet(self._get_group_style())
        status_layout = QVBoxLayout(status_group)
        
        self.processing_status = QLabel("Ready for sequential processing")
        self.processing_status.setStyleSheet("""
            QLabel {
                color: #ffffff;
                background-color: #2a2a2a;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                border: 2px solid #555555;
                font-size: 10px;
            }
        """)
        self.processing_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        status_layout.addWidget(self.processing_status)
        main_layout.addWidget(status_group)
        
        # Add stretch
        main_layout.addStretch()
        
        print("✓ Sequential Filtering Right Panel initialized")
    
    def update_stage_progress(self, stage_index: int, status: str, success: bool = True):
        """Update progress for a specific stage."""
        stage = ProcessingStage(stage_index)
        progress_info = self.stage_progress[stage]
        stage_config = SequentialFilterConfigManager().get_stage_config(stage)
        
        # Update indicator
        if success:
            progress_info['indicator'].setText("●")
            progress_info['indicator'].setStyleSheet(f"""
                QLabel {{
                    color: {stage_config.color};
                    font-size: 12px;
                    font-weight: bold;
                    min-width: 16px;
                }}
            """)
        else:
            progress_info['indicator'].setText("✗")
            progress_info['indicator'].setStyleSheet("""
                QLabel {
                    color: #ff0000;
                    font-size: 12px;
                    font-weight: bold;
                    min-width: 16px;
                }
            """)
        
        # Update status
        progress_info['status'].setText(status)
        if success:
            progress_info['status'].setStyleSheet("""
                QLabel {
                    color: #00ff00;
                    font-size: 9px;
                    font-style: italic;
                }
            """)
        else:
            progress_info['status'].setStyleSheet("""
                QLabel {
                    color: #ff0000;
                    font-size: 9px;
                    font-style: italic;
                }
            """)
    
    def update_histogram(self, image_data: np.ndarray, stage_index: Optional[int] = None):
        """Update histogram display for current stage."""
        if image_data is not None:
            self.histogram_view.update_histogram(image_data)
            self._update_statistics(image_data, stage_index)
    
    def _update_statistics(self, image_data: np.ndarray, stage_index: Optional[int] = None):
        """Update statistics for current stage."""
        try:
            stage_name = "Original"
            if stage_index is not None:
                stage = ProcessingStage(stage_index)
                stage_config = SequentialFilterConfigManager().get_stage_config(stage)
                stage_name = stage_config.display_name
            
            stats = {
                'Stage': stage_name,
                'Shape': f"{image_data.shape}",
                'Data type': str(image_data.dtype),
                'Min value': f"{np.min(image_data):.2f}",
                'Max value': f"{np.max(image_data):.2f}",
                'Mean': f"{np.mean(image_data):.2f}",
                'Std dev': f"{np.std(image_data):.2f}",
                'Unique values': f"{len(np.unique(image_data))}"
            }
            
            stats_text = f"📊 STAGE STATISTICS\n"
            stats_text += "═" * 25 + "\n"
            
            for key, value in stats.items():
                stats_text += f"▶ {key:<12}: {value}\n"
            
            stats_text += "═" * 25 + "\n"
            stats_text += f"✓ Updated: {self._get_timestamp()}"
            
            self.stats_text.setText(stats_text)
            
        except Exception as e:
            self.stats_text.setText(f"❌ Error calculating stats:\n{e}")
    
    def set_processing_status(self, message: str):
        """Set overall processing status."""
        self.processing_status.setText(message)
    
    def reset_all_progress(self):
        """Reset all stage progress indicators."""
        for stage, progress_info in self.stage_progress.items():
            progress_info['indicator'].setText("○")
            progress_info['indicator'].setStyleSheet("""
                QLabel {
                    color: #666666;
                    font-size: 12px;
                    font-weight: bold;
                    min-width: 16px;
                }
            """)
            progress_info['status'].setText("Pending")
            progress_info['status'].setStyleSheet("""
                QLabel {
                    color: #cccccc;
                    font-size: 9px;
                    font-style: italic;
                }
            """)
    
    def _get_timestamp(self):
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def _get_group_style(self):
        return """
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                color: #ffffff;
                background-color: #2b2b2b;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            """