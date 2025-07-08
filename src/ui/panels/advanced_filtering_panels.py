"""
Phase 2: Unified Layout Advanced Filtering Panels
Improvements: Merged action buttons, streamlined UI, unified control panel
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, 
    QComboBox, QLabel, QSplitter, QScrollArea, QListWidget, QFrame, 
    QProgressBar, QTextEdit, QCheckBox, QListWidgetItem, QSpinBox,
    QDoubleSpinBox, QSlider, QGridLayout, QFileDialog, QMessageBox,
    QTabWidget, QLineEdit
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

# Import your existing components
from src.ui.view_manager import ViewMode
from src.ui.components.histogram_view import HistogramView

logger = logging.getLogger(__name__)

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
    category: str
    parameters: List[FilterParameter]

class AdvancedFilterConfigManager:
    """FIXED: Filter configuration manager that matches ImageProcessingService exactly."""
    
    def __init__(self):
        self.filters: Dict[str, FilterConfig] = {}
        self._load_working_filters()
    
    def _load_working_filters(self):
        """Load all available filters from the filters folder."""
        
        # CLAHE
        self.add_filter(FilterConfig(
            name="clahe",
            display_name="CLAHE",
            description="Contrast Limited Adaptive Histogram Equalization",
            category="Enhancement",
            parameters=[
                FilterParameter("clip_limit", "float", 2.0, 0.1, 10.0, 0.1, description="Threshold for contrast limiting"),
                FilterParameter("tile_grid_x", "int", 8, 1, 32, 1, description="Tile grid X size"),
                FilterParameter("tile_grid_y", "int", 8, 1, 32, 1, description="Tile grid Y size")
            ]
        ))
        
        # Canny Edge Detection
        self.add_filter(FilterConfig(
            name="canny",
            display_name="Canny Edge Detection",
            description="Detects edges using Canny edge detector",
            category="Edge Detection",
            parameters=[
                FilterParameter("low_threshold", "int", 50, 1, 255, 1, description="Lower threshold"),
                FilterParameter("high_threshold", "int", 150, 1, 255, 1, description="Upper threshold")
            ]
        ))
        
        # Edge Detection (alias)
        self.add_filter(FilterConfig(
            name="edge_detection",
            display_name="Edge Detection",
            description="Detects edges using Canny edge detector",
            category="Edge Detection",
            parameters=[
                FilterParameter("low_threshold", "int", 50, 1, 255, 1, description="Lower threshold"),
                FilterParameter("high_threshold", "int", 150, 1, 255, 1, description="Upper threshold")
            ]
        ))
        
        # Difference of Gaussians
        self.add_filter(FilterConfig(
            name="dog",
            display_name="Difference of Gaussians",
            description="Edge detection using difference of Gaussian filters",
            category="Edge Detection",
            parameters=[
                FilterParameter("sigma1", "float", 1.0, 0.1, 10.0, 0.1, description="First Gaussian sigma"),
                FilterParameter("sigma2", "float", 2.0, 0.1, 10.0, 0.1, description="Second Gaussian sigma")
            ]
        ))
        
        # Threshold
        self.add_filter(FilterConfig(
            name="threshold",
            display_name="Binary Threshold",
            description="Apply binary threshold to image",
            category="Segmentation",
            parameters=[
                FilterParameter("threshold", "int", 127, 0, 255, 1, description="Threshold value"),
                FilterParameter("max_value", "int", 255, 0, 255, 1, description="Maximum value")
            ]
        ))
        
        # Non-Local Means Denoising
        self.add_filter(FilterConfig(
            name="nlmd",
            display_name="Non-Local Means Denoising",
            description="Advanced denoising using non-local means",
            category="Noise Reduction",
            parameters=[
                FilterParameter("h", "int", 10, 1, 30, 1, description="Filter strength"),
                FilterParameter("template_window_size", "int", 7, 3, 21, 2, description="Template window size"),
                FilterParameter("search_window_size", "int", 21, 5, 31, 2, description="Search window size")
            ]
        ))
        
        # Gabor Filter
        self.add_filter(FilterConfig(
            name="gabor",
            display_name="Gabor Filter",
            description="Texture analysis using Gabor filter",
            category="Feature Detection",
            parameters=[
                FilterParameter("frequency", "float", 0.5, 0.01, 2.0, 0.01, description="Frequency of the filter"),
                FilterParameter("theta", "float", 0.0, 0.0, 180.0, 1.0, description="Orientation angle"),
                FilterParameter("bandwidth", "float", 1.0, 0.1, 5.0, 0.1, description="Bandwidth of the filter")
            ]
        ))
        
        # Laplacian
        self.add_filter(FilterConfig(
            name="laplacian",
            display_name="Laplacian",
            description="Edge detection using Laplacian operator",
            category="Edge Detection",
            parameters=[
                FilterParameter("ksize", "int", 3, 1, 31, 2, description="Kernel size")
            ]
        ))
        
        # Top Hat
        self.add_filter(FilterConfig(
            name="top_hat",
            display_name="Top Hat",
            description="Morphological top hat operation",
            category="Morphological",
            parameters=[
                FilterParameter("kernel_size", "int", 5, 3, 15, 2, description="Kernel size")
            ]
        ))
        
        # Total Variation
        self.add_filter(FilterConfig(
            name="total_variation",
            display_name="Total Variation",
            description="Total variation denoising",
            category="Noise Reduction",
            parameters=[
                FilterParameter("weight", "float", 1.0, 0.01, 5.0, 0.01, description="Regularization weight")
            ]
        ))
        
        # Wavelet
        self.add_filter(FilterConfig(
            name="wavelet",
            display_name="Wavelet",
            description="Wavelet-based filtering",
            category="Frequency Domain",
            parameters=[
                FilterParameter("wavelet", "choice", "haar", choices=["haar", "db2", "db4", "bior2.2"], description="Wavelet type"),
                FilterParameter("level", "int", 1, 1, 4, 1, description="Decomposition level")
            ]
        ))
        
        # FFT High Pass
        self.add_filter(FilterConfig(
            name="fft_highpass",
            display_name="FFT High Pass",
            description="High-pass filter in frequency domain",
            category="Frequency Domain",
            parameters=[
                FilterParameter("cutoff_frequency", "float", 0.1, 0.01, 1.0, 0.01, description="Cutoff frequency"),
                FilterParameter("sample_rate", "float", 1.0, 0.1, 10.0, 0.1, description="Sample rate")
            ]
        ))
    
    def add_filter(self, filter_config: FilterConfig):
        """Add a filter configuration."""
        self.filters[filter_config.name] = filter_config
    
    def get_filter(self, name: str) -> Optional[FilterConfig]:
        """Get filter configuration by name."""
        return self.filters.get(name)
    
    def get_filter_names(self) -> List[str]:
        """Get list of available filter names."""
        return list(self.filters.keys())
    
    def get_filters_by_category(self) -> Dict[str, List[FilterConfig]]:
        """Get filters grouped by category."""
        categories = {}
        for filter_config in self.filters.values():
            category = filter_config.category
            if category not in categories:
                categories[category] = []
            categories[category].append(filter_config)
        return categories

class SimpleParameterWidget(QWidget):
    """Simplified parameter widget with SINGLE input control."""
    
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
        
        # Parameter label
        label = QLabel(f"{self.parameter.description}:")
        label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 11px;")
        layout.addWidget(label)
        
        # Single control layout
        control_layout = QHBoxLayout()
        control_layout.setSpacing(5)
        
        # Choose ONE control type based on parameter type
        if self.parameter.param_type == "choice":
            self._setup_choice_control(control_layout)
        elif self.parameter.param_type == "int":
            self._setup_int_control(control_layout)
        elif self.parameter.param_type == "float":
            self._setup_float_control(control_layout)
        
        layout.addLayout(control_layout)
    
    def _setup_choice_control(self, layout):
        """Setup choice parameter with combo box only."""
        self.combo = QComboBox()
        self.combo.addItems(self.parameter.choices or [])
        if self.parameter.default_value in (self.parameter.choices or []):
            self.combo.setCurrentText(str(self.parameter.default_value))
        self.combo.currentTextChanged.connect(self._on_choice_changed)
        self.combo.setStyleSheet(self._get_combo_style())
        layout.addWidget(self.combo)
    
    def _setup_int_control(self, layout):
        """Setup integer parameter with spinbox only."""
        self.spinbox = QSpinBox()
        min_val = int(self.parameter.min_value or 0)
        max_val = int(self.parameter.max_value or 100)
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setValue(int(self.parameter.default_value))
        self.spinbox.setSingleStep(int(self.parameter.step))
        self.spinbox.valueChanged.connect(self._on_int_changed)
        self.spinbox.setStyleSheet(self._get_spinbox_style())
        self.spinbox.setMinimumWidth(80)
        layout.addWidget(self.spinbox)
    
    def _setup_float_control(self, layout):
        """Setup float parameter with double spinbox only."""
        self.double_spinbox = QDoubleSpinBox()
        min_val = float(self.parameter.min_value or 0.0)
        max_val = float(self.parameter.max_value or 10.0)
        self.double_spinbox.setRange(min_val, max_val)
        self.double_spinbox.setValue(float(self.parameter.default_value))
        self.double_spinbox.setSingleStep(float(self.parameter.step))
        self.double_spinbox.setDecimals(2)
        self.double_spinbox.valueChanged.connect(self._on_float_changed)
        self.double_spinbox.setStyleSheet(self._get_spinbox_style())
        self.double_spinbox.setMinimumWidth(80)
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
                border: 2px solid #555555;
                border-radius: 4px;
                padding: 4px;
                min-height: 16px;
            }
        """
    
    def _get_spinbox_style(self):
        return """
            QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 2px solid #555555;
                border-radius: 4px;
                padding: 4px;
                min-height: 16px;
            }
        """


class UnifiedActionPanel(QWidget):
    """PHASE 2: Unified action panel combining Quick Actions + Main Actions."""
    
    # Unified signals
    preview_requested = Signal()
    apply_requested = Signal()
    reset_requested = Signal()
    save_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup unified action panel with all controls in one place."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # UNIFIED ACTIONS - All buttons in one organized layout
        # Row 1: Primary actions (Preview & Apply)
        primary_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("🔍 Preview")
        self.preview_btn.setStyleSheet(self._get_button_style("#0078d4", "#106ebe"))
        self.preview_btn.clicked.connect(self.preview_requested)
        self.preview_btn.setEnabled(False)
        
        self.apply_btn = QPushButton("✓ Apply")
        self.apply_btn.setStyleSheet(self._get_button_style("#28a745", "#218838"))
        self.apply_btn.clicked.connect(self.apply_requested)
        self.apply_btn.setEnabled(False)
        
        primary_layout.addWidget(self.preview_btn)
        primary_layout.addWidget(self.apply_btn)
        layout.addLayout(primary_layout)
        
        # Row 2: Secondary actions (Reset & Save)
        secondary_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("🔄 Reset All")
        self.reset_btn.setStyleSheet(self._get_button_style("#dc3545", "#c82333"))
        self.reset_btn.clicked.connect(self.reset_requested)
        
        self.save_btn = QPushButton("💾 Save Image")
        self.save_btn.setStyleSheet(self._get_button_style("#17a2b8", "#138496"))
        self.save_btn.clicked.connect(self.save_requested)
        
        secondary_layout.addWidget(self.reset_btn)
        secondary_layout.addWidget(self.save_btn)
        layout.addLayout(secondary_layout)
        
        # Status indicator
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                background-color: #2a2a2a;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
                border: 1px solid #555555;
            }
        """)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
    
    def set_filter_selected(self, has_filter: bool):
        """Enable/disable primary buttons based on filter selection."""
        self.preview_btn.setEnabled(has_filter)
        self.apply_btn.setEnabled(has_filter)
    
    def set_status(self, message: str, color: str = "#ffffff"):
        """Set status message with optional color."""
        self.status_label.setText(message)
        self.status_label.setStyleSheet(self.status_label.styleSheet().replace("color: #ffffff;", f"color: {color};"))
    
    def _get_button_style(self, bg_color: str, hover_color: str):
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                padding: 10px 16px;
                border-radius: 6px;
                font-weight: bold;
                min-height: 20px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:disabled {{
                background-color: #555555;
                color: #999999;
            }}
        """


class AdvancedFilteringLeftPanel(QWidget):
    """Advanced filtering left panel with unified layout and merged actions."""
    
    # Signals that match your ImageProcessingService expectations
    filter_applied = Signal(str, dict)
    filter_previewed = Signal(str, dict)
    filter_reset = Signal()
    save_image_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize attributes
        self.config_manager = AdvancedFilterConfigManager()
        self.current_filter = None
        self.parameter_widgets = {}
        
        # Initialize UI
        self.init_panel()
    
    def init_panel(self):
        """Initialize the Phase 2 unified filtering panel UI."""
        # Create scrollable area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create content widget
        content_widget = QWidget()
        main_layout = QVBoxLayout(content_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)
        
        # Create tab widget for better organization
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(self._get_tab_style())
        
        # Set content widget to scroll area
        scroll_area.setWidget(content_widget)
        
        # Create layout for this panel
        panel_layout = QVBoxLayout(self)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll_area)
        
        # Tab 1: Filter Controls
        self.controls_tab = QWidget()
        self._setup_controls_tab()
        self.tab_widget.addTab(self.controls_tab, "Filter Controls")
        
        # Tab 2: History & Management
        self.history_tab = QWidget()
        self._setup_history_tab()
        self.tab_widget.addTab(self.history_tab, "History & Actions")
        
        main_layout.addWidget(self.tab_widget)
        
        print("✓ Advanced Filtering Left Panel initialized")
    
    def _setup_controls_tab(self):
        """Setup the filter controls tab with unified actions."""
        layout = QVBoxLayout(self.controls_tab)
        layout.setSpacing(12)
        
        # 1. Filter Selection Group
        filter_group = QGroupBox("Filter Selection")
        filter_group.setStyleSheet(self._get_group_style())
        filter_layout = QVBoxLayout(filter_group)
        
        # Filter combo
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("-- Select Filter --", None)
        
        # Add all available filters
        for filter_name in self.config_manager.get_filter_names():
            filter_config = self.config_manager.get_filter(filter_name)
            if filter_config:
                self.filter_combo.addItem(filter_config.display_name, filter_config.name)
        
        self.filter_combo.currentTextChanged.connect(self._on_filter_selected)
        self.filter_combo.setStyleSheet(self._get_combo_style())
        
        filter_layout.addWidget(QLabel("Filter Type:"))
        filter_layout.addWidget(self.filter_combo)
        
        # Filter description with better styling
        self.filter_description = QLabel("Select a filter to see description")
        self.filter_description.setWordWrap(True)
        self.filter_description.setStyleSheet("""
            QLabel {
                color: #cccccc;
                font-style: italic;
                padding: 8px;
                background-color: #2a2a2a;
                border-radius: 4px;
                border: 1px solid #444444;
                min-height: 40px;
            }
        """)
        filter_layout.addWidget(self.filter_description)
        
        layout.addWidget(filter_group)
        
        # 2. Parameters Group (initially hidden)
        self.param_group = QGroupBox("Filter Parameters")
        self.param_group.setStyleSheet(self._get_group_style())
        self.param_layout = QVBoxLayout(self.param_group)
        self.param_group.hide()
        
        layout.addWidget(self.param_group)
        
        # 3. UNIFIED ACTIONS PANEL - This appears in BOTH tabs
        self.unified_actions = UnifiedActionPanel()
        self.unified_actions.preview_requested.connect(self._preview_filter)
        self.unified_actions.apply_requested.connect(self._apply_filter)
        self.unified_actions.reset_requested.connect(self._reset_filters)
        self.unified_actions.save_requested.connect(self._save_image)
        
        # Add to controls tab
        actions_group = QGroupBox("Filter Actions")
        actions_group.setStyleSheet(self._get_group_style())
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.addWidget(self.unified_actions)
        
        layout.addWidget(actions_group)
        layout.addStretch()
    
    def _setup_history_tab(self):
        """Setup the history tab with the SAME unified actions."""
        layout = QVBoxLayout(self.history_tab)
        layout.setSpacing(12)
        
        # Filter History
        history_group = QGroupBox("Filter History")
        history_group.setStyleSheet(self._get_group_style())
        history_layout = QVBoxLayout(history_group)
        
        # History list with improved styling
        self.history_list = QListWidget()
        self.history_list.setStyleSheet("""
            QListWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                font-size: 10px;
                alternate-background-color: #333333;
            }
            QListWidget::item {
                padding: 6px;
                border-bottom: 1px solid #404040;
                border-radius: 2px;
                margin: 1px;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #4a4a4a;
            }
        """)
        self.history_list.itemDoubleClicked.connect(self._reapply_from_history)
        
        # History controls
        history_controls_layout = QHBoxLayout()
        
        clear_history_btn = QPushButton("Clear History")
        clear_history_btn.setStyleSheet(self._get_small_button_style("#6c757d", "#5a6268"))
        clear_history_btn.clicked.connect(self.history_list.clear)
        
        export_history_btn = QPushButton("Export History")
        export_history_btn.setStyleSheet(self._get_small_button_style("#17a2b8", "#138496"))
        export_history_btn.clicked.connect(self._export_history)
        
        history_controls_layout.addWidget(clear_history_btn)
        history_controls_layout.addWidget(export_history_btn)
        
        history_layout.addWidget(self.history_list)
        history_layout.addLayout(history_controls_layout)
        
        layout.addWidget(history_group)
        
        # SAME UNIFIED ACTIONS PANEL in history tab
        actions_group_history = QGroupBox("Filter Actions")
        actions_group_history.setStyleSheet(self._get_group_style())
        actions_layout_history = QVBoxLayout(actions_group_history)
        
        # Create second instance of unified actions (both tabs have the same controls)
        self.unified_actions_history = UnifiedActionPanel()
        self.unified_actions_history.preview_requested.connect(self._preview_filter)
        self.unified_actions_history.apply_requested.connect(self._apply_filter)
        self.unified_actions_history.reset_requested.connect(self._reset_filters)
        self.unified_actions_history.save_requested.connect(self._save_image)
        
        actions_layout_history.addWidget(self.unified_actions_history)
        layout.addWidget(actions_group_history)
        
        layout.addStretch()
    
    def _on_filter_selected(self):
        """Handle filter selection with improved type safety."""
        current_data = self.filter_combo.currentData()
        
        # Skip category separators and handle None values safely
        if current_data is None:
            self.current_filter = None
            self._update_ui_state()
            return
        
        # Ensure current_data is a string
        if not isinstance(current_data, str):
            self.current_filter = None
            self._update_ui_state()
            return
        
        self.current_filter = current_data
        
        # Clear existing parameter widgets
        for widget in self.parameter_widgets.values():
            widget.setParent(None)
        self.parameter_widgets.clear()
        
        # FIXED: Only proceed if we have a valid filter string
        if self.current_filter and isinstance(self.current_filter, str):
            filter_config = self.config_manager.get_filter(self.current_filter)
            
            # FIXED: Check if filter_config is not None before accessing attributes
            if filter_config is not None:
                # Update description with category info
                desc_text = f"[{filter_config.category}] {filter_config.description}"
                self.filter_description.setText(desc_text)
                
                # Create parameter widgets
                if filter_config.parameters:
                    self.param_group.show()
                    for param in filter_config.parameters:
                        widget = SimpleParameterWidget(param)
                        widget.value_changed.connect(self._on_parameter_changed)
                        self.parameter_widgets[param.name] = widget
                        self.param_layout.addWidget(widget)
                else:
                    self.param_group.hide()
            else:
                # Handle case where filter_config is None
                self.filter_description.setText("Unknown filter selected")
                self.param_group.hide()
        else:
            # Handle case where current_filter is None or not a string
            self.filter_description.setText("Select a filter to see description")
            self.param_group.hide()
        
        self._update_ui_state()
    
    def _update_ui_state(self):
        """Update UI state based on current filter selection."""
        has_filter = self.current_filter is not None and isinstance(self.current_filter, str)
        
        # Update both unified action panels
        self.unified_actions.set_filter_selected(has_filter)
        if hasattr(self, 'unified_actions_history'):
            self.unified_actions_history.set_filter_selected(has_filter)
        
        # Update status
        if has_filter and self.current_filter:
            filter_config = self.config_manager.get_filter(self.current_filter)
            if filter_config is not None:
                self.unified_actions.set_status(f"Ready: {filter_config.display_name}", "#00ff00")
                if hasattr(self, 'unified_actions_history'):
                    self.unified_actions_history.set_status(f"Ready: {filter_config.display_name}", "#00ff00")
            else:
                # Fallback if filter_config is None
                self.unified_actions.set_status(f"Ready: {self.current_filter}", "#00ff00")
                if hasattr(self, 'unified_actions_history'):
                    self.unified_actions_history.set_status(f"Ready: {self.current_filter}", "#00ff00")
        else:
            self.unified_actions.set_status("Select a filter", "#cccccc")
            if hasattr(self, 'unified_actions_history'):
                self.unified_actions_history.set_status("Select a filter", "#cccccc")

    
    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter changes."""
        print(f"Parameter {param_name} changed to {value}")
    
    def _preview_filter(self):
        """Preview current filter with current parameters."""
        if self.current_filter:
            parameters = self._get_current_parameters()
            print(f"Previewing filter: {self.current_filter} with params: {parameters}")
            self.unified_actions.set_status("Previewing...", "#ffa500")
            if hasattr(self, 'unified_actions_history'):
                self.unified_actions_history.set_status("Previewing...", "#ffa500")
            self.filter_previewed.emit(self.current_filter, parameters)
    
    def _apply_filter(self):
        """Apply current filter with current parameters."""
        if self.current_filter and isinstance(self.current_filter, str):
            parameters = self._get_current_parameters()
            print(f"Applying filter: {self.current_filter} with params: {parameters}")
            self.unified_actions.set_status("Applying...", "#ffa500")
            if hasattr(self, 'unified_actions_history'):
                self.unified_actions_history.set_status("Applying...", "#ffa500")
            self.filter_applied.emit(self.current_filter, parameters)
            
            # Add to history with better formatting
            filter_config = self.config_manager.get_filter(self.current_filter)
            
            # FIXED: Check if filter_config is not None before accessing attributes
            if filter_config is not None:
                display_name = filter_config.display_name
                category = filter_config.category
                param_str = ", ".join([f"{k}={v}" for k, v in parameters.items()])
                history_text = f"[{category}] {display_name}"
                if param_str:
                    history_text += f"\n    Parameters: {param_str}"
            else:
                # Fallback if filter_config is None
                param_str = ", ".join([f"{k}={v}" for k, v in parameters.items()])
                history_text = f"[Unknown] {self.current_filter}"
                if param_str:
                    history_text += f"\n    Parameters: {param_str}"
            
            item = QListWidgetItem(history_text)
            item.setData(Qt.ItemDataRole.UserRole, {"filter": self.current_filter, "params": parameters})
            self.history_list.addItem(item)
            self.history_list.scrollToBottom()
    
    def _reset_filters(self):
        """Reset all filters."""
        print("Resetting all filters")
        self.unified_actions.set_status("Resetting...", "#ff6600")
        if hasattr(self, 'unified_actions_history'):
            self.unified_actions_history.set_status("Resetting...", "#ff6600")
        self.filter_reset.emit()
    
    def _save_image(self):
        """Save current filtered image."""
        print("Saving filtered image")
        self.unified_actions.set_status("Saving...", "#00bfff")
        if hasattr(self, 'unified_actions_history'):
            self.unified_actions_history.set_status("Saving...", "#00bfff")
        self.save_image_requested.emit()
    
    def _reapply_from_history(self, item):
        """Reapply filter from history."""
        data = item.data(Qt.ItemDataRole.UserRole)
        if data:
            filter_name = data["filter"]
            parameters = data["params"]
            
            # Set filter in combo
            for i in range(self.filter_combo.count()):
                if self.filter_combo.itemData(i) == filter_name:
                    self.filter_combo.setCurrentIndex(i)
                    break
            
            # Set parameters
            for param_name, value in parameters.items():
                if param_name in self.parameter_widgets:
                    self.parameter_widgets[param_name].set_value(value)
            
            # Apply filter
            self.filter_applied.emit(filter_name, parameters)
    
    def _export_history(self):
        """Export filter history to file."""
        try:
            history_data = []
            for i in range(self.history_list.count()):
                item = self.history_list.item(i)
                data = item.data(Qt.ItemDataRole.UserRole)
                if data:
                    history_data.append(data)
            
            if history_data:
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Export Filter History", "filter_history.json", "JSON files (*.json)"
                )
                if file_path:
                    with open(file_path, 'w') as f:
                        json.dump(history_data, f, indent=2)
                    print(f"Filter history exported to {file_path}")
        except Exception as e:
            print(f"Error exporting history: {e}")
    
    def _get_current_parameters(self) -> dict:
        """FIXED: Get current parameter values matching ImageProcessingService expectations."""
        parameters = {}
        for param_name, widget in self.parameter_widgets.items():
            value = widget.get_value()
            
            # FIXED: Handle parameter mapping for ImageProcessingService
            if param_name == "tile_grid_size":
                # ImageProcessingService expects both tile_grid_x and tile_grid_y
                parameters["tile_grid_x"] = value
                parameters["tile_grid_y"] = value
                # Also include the original parameter for compatibility
                parameters["tile_grid_size"] = value
            elif param_name == "sigma":
                # ImageProcessingService expects both sigma_x and sigma_y
                parameters["sigma_x"] = value
                parameters["sigma_y"] = value
                # Also include the original parameter
                parameters["sigma"] = value
            elif param_name == "kernel_size":
                # Ensure odd values for OpenCV filters
                if value % 2 == 0:
                    value += 1
                parameters[param_name] = value
            else:
                # Direct parameter mapping
                parameters[param_name] = value
        
        return parameters
    
    # Styling methods
    def _get_tab_style(self):
        return """
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                padding: 8px 16px;
                margin-right: 2px;
                color: #ffffff;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #2b2b2b;
                border-bottom-color: #2b2b2b;
                color: #ffffff;
            }
            QTabBar::tab:hover {
                background-color: #4a4a4a;
                color: #ffffff;
            }
        """
    
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
    
    def _get_combo_style(self):
        return """
            QComboBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 2px solid #555555;
                border-radius: 4px;
                padding: 6px;
                min-height: 20px;
                font-size: 11px;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #555555;
                border-radius: 2px;
            }
            QComboBox::down-arrow {
                width: 12px;
                height: 12px;
            }
        """
    
    def _get_small_button_style(self, bg_color: str, hover_color: str):
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """


class AdvancedFilteringRightPanel(QWidget):
    """Advanced filtering right panel with improved layout and information display."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view_mode = ViewMode.FILTERING
        self.init_panel()
    
    def init_panel(self):
        """Initialize the Phase 2 right panel UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)
        
        # 1. Histogram Section with improved layout
        histogram_group = QGroupBox("📊 Image Histogram")
        histogram_group.setStyleSheet(self._get_group_style())
        histogram_layout = QVBoxLayout(histogram_group)
        
        self.histogram_view = HistogramView()
        histogram_layout.addWidget(self.histogram_view)
        
        main_layout.addWidget(histogram_group)
        
        # 2. Image Statistics with improved formatting
        stats_group = QGroupBox("📈 Image Statistics")
        stats_group.setStyleSheet(self._get_group_style())
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(140)
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
        
        # 3. Processing Status with visual indicators
        status_group = QGroupBox("⚡ Processing Status")
        status_group.setStyleSheet(self._get_group_style())
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                background-color: #2a2a2a;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                border: 2px solid #555555;
                font-size: 11px;
            }
        """)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Processing indicator
        self.processing_indicator = QLabel("●")
        self.processing_indicator.setStyleSheet("""
            QLabel {
                color: #00ff00;
                font-size: 16px;
                font-weight: bold;
                background: transparent;
                border: none;
                padding: 0px;
            }
        """)
        self.processing_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Status layout
        status_info_layout = QHBoxLayout()
        status_info_layout.addWidget(self.processing_indicator)
        status_info_layout.addWidget(self.status_label, 1)
        
        status_layout.addLayout(status_info_layout)
        main_layout.addWidget(status_group)
        
        # Add stretch
        main_layout.addStretch()
        
        print("✓ Advanced Filtering Right Panel initialized")
    
    def update_histogram(self, image_data: np.ndarray):
        """Update histogram display."""
        if image_data is not None:
            self.histogram_view.update_histogram(image_data)
            self._update_statistics(image_data)
    
    def _update_statistics(self, image_data: np.ndarray):
        """Update image statistics with improved formatting."""
        try:
            stats = {
                'Shape': f"{image_data.shape}",
                'Data type': str(image_data.dtype),
                'Min value': f"{np.min(image_data):.2f}",
                'Max value': f"{np.max(image_data):.2f}",
                'Mean': f"{np.mean(image_data):.2f}",
                'Std dev': f"{np.std(image_data):.2f}",
                'Range': f"{np.max(image_data) - np.min(image_data):.2f}",
                'Non-zero pixels': f"{np.count_nonzero(image_data):,}"
            }
            
            stats_text = "📊 IMAGE STATISTICS\n"
            stats_text += "═" * 25 + "\n"
            
            for key, value in stats.items():
                stats_text += f"▶ {key:<14}: {value}\n"
            
            stats_text += "═" * 25 + "\n"
            stats_text += f"✓ Updated: {self._get_timestamp()}"
            
            self.stats_text.setText(stats_text)
            
        except Exception as e:
            self.stats_text.setText(f"❌ Error calculating stats:\n{e}")
    
    def show_status(self, message: str, status_type: str = "info"):
        """Show status message with visual indicator."""
        self.status_label.setText(message)
        
        # Update indicator color based on status type
        colors = {
            "info": "#ffffff",
            "processing": "#ffa500",
            "success": "#00ff00",
            "error": "#ff0000",
            "warning": "#ffff00"
        }
        
        color = colors.get(status_type, "#ffffff")
        self.processing_indicator.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 16px;
                font-weight: bold;
                background: transparent;
                border: none;
                padding: 0px;
            }}
        """)
    
    def clear_displays(self):
        """Clear all displays."""
        if hasattr(self.histogram_view, 'clear_displays'):
            self.histogram_view.clear_displays()
        self.stats_text.clear()
        self.show_status("Ready")
    
    def _get_timestamp(self):
        """Get current timestamp for statistics."""
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
    
    def update_kernel(self, kernel):
        """Update kernel visualization (optional method for compatibility)."""
        try:
            # This method is called by some legacy code but not needed for Phase 2
            # Just log and ignore for now
            if kernel is not None:
                print(f"Kernel update requested (ignored): {kernel.shape if hasattr(kernel, 'shape') else type(kernel)}")
            else:
                print("Kernel cleared (ignored)")
        except Exception as e:
            print(f"Error in update_kernel: {e}")
