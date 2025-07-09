"""
Automatic Filter Panel - Categorized Automatic Filtering Interface

This module provides an automatic filtering panel with categorized filter selection
for contrast enhancement, denoising, binarisation, and edge detection operations.

Main Class:
- AutomaticFilterPanel: Panel for automatic filtering with categorized filter selection

Key Methods:
- setup_ui(): Initializes UI with filter categories and control buttons
- _setup_contrast_section(): Creates contrast enhancement filter options
- _setup_denoising_section(): Creates noise reduction filter options
- _setup_binarisation_section(): Creates binarisation filter options
- _setup_edge_detection_section(): Creates edge detection filter options
- _setup_control_buttons(): Creates action buttons for filter execution
- _setup_progress_section(): Creates progress tracking interface
- _on_contrast_changed(): Handles contrast filter selection changes
- _on_denoising_changed(): Handles denoising filter selection changes
- _on_binarisation_changed(): Handles binarisation filter selection changes
- _on_edge_detection_changed(): Handles edge detection filter selection changes
- _on_run_automatic_filtering(): Handles automatic filtering execution
- _on_reset_selections(): Resets all filter selections
- get_selected_filters(): Returns currently selected filters

Signals Emitted:
- automatic_filtering_requested(dict): Automatic filtering requested with selections

Dependencies:
- Uses: PySide6.QtWidgets, PySide6.QtCore (Qt framework)
- Uses: typing (type hints for Dict, Any, List)
- Called by: UI filtering components and workflow
- Coordinates with: Image processing services and filter operations

Filter Categories:
- Contrast Enhancement: CLAHE with various intensity levels
- Noise Reduction: Non-Local Means and Total Variation denoising
- Binarisation: Automatic and manual threshold methods
- Edge Detection: Canny and Laplacian edge detection

Features:
- Categorized filter selection with dropdown menus
- Pre-configured filter parameters for different intensity levels
- Progress tracking and status display
- Reset functionality for clearing selections
- Dark theme styling with consistent UI design
- Placeholder functionality for future automatic processing
- Multi-category filter combination support
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                               QComboBox, QPushButton, QLabel, QProgressBar)
from PySide6.QtCore import Qt, Signal
from typing import Dict, Any, List


class AutomaticFilterPanel(QWidget):
    """Panel for automatic filtering with categorized filter selection."""
    
    # Signals
    automatic_filtering_requested = Signal(dict)  # filter_selections
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.filter_selections = {
            'contrast': None,
            'denoising': None,
            'binarisation': None,
            'edge_detection': None
        }
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Instructions
        instructions = QLabel("Select automatic filtering options for each category:")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("""
            QLabel {
                background-color: #3c3c3c;
                color: #ffffff;
                padding: 10px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        layout.addWidget(instructions)
        
        # Filter Categories
        self._setup_contrast_section(layout)
        self._setup_denoising_section(layout)
        self._setup_binarisation_section(layout)
        self._setup_edge_detection_section(layout)
        
        # Control Buttons
        self._setup_control_buttons(layout)
        
        # Progress Indicator
        self._setup_progress_section(layout)
        
        layout.addStretch()
        
    def _setup_contrast_section(self, parent_layout):
        """Setup contrast filters section."""
        contrast_group = QGroupBox("Contrast Enhancement")
        contrast_group.setStyleSheet(self._get_group_style())
        contrast_layout = QVBoxLayout(contrast_group)
        
        # Contrast dropdown
        self.contrast_combo = QComboBox()
        self.contrast_combo.addItem("-- Select Contrast Filter --", None)
        self.contrast_combo.addItem("CLAHE Light Enhancement", {
            'filter': 'clahe',
            'clip_limit': 1.5,
            'tile_grid_size': (8, 8)
        })
        self.contrast_combo.addItem("CLAHE Medium Enhancement", {
            'filter': 'clahe',
            'clip_limit': 2.5,
            'tile_grid_size': (8, 8)
        })
        self.contrast_combo.addItem("CLAHE Strong Enhancement", {
            'filter': 'clahe',
            'clip_limit': 4.0,
            'tile_grid_size': (6, 6)
        })
        
        self.contrast_combo.currentIndexChanged.connect(self._on_contrast_changed)
        contrast_layout.addWidget(self.contrast_combo)
        
        parent_layout.addWidget(contrast_group)
        
    def _setup_denoising_section(self, parent_layout):
        """Setup denoising filters section."""
        denoising_group = QGroupBox("Noise Reduction")
        denoising_group.setStyleSheet(self._get_group_style())
        denoising_layout = QVBoxLayout(denoising_group)
        
        # Denoising dropdown
        self.denoising_combo = QComboBox()
        self.denoising_combo.addItem("-- Select Denoising Filter --", None)
        self.denoising_combo.addItem("Non-Local Means Light", {
            'filter': 'nlmd',
            'h': 5,
            'template_window_size': 7,
            'search_window_size': 21
        })
        self.denoising_combo.addItem("Non-Local Means Medium", {
            'filter': 'nlmd',
            'h': 10,
            'template_window_size': 7,
            'search_window_size': 21
        })
        self.denoising_combo.addItem("Non-Local Means Strong", {
            'filter': 'nlmd',
            'h': 15,
            'template_window_size': 9,
            'search_window_size': 25
        })
        self.denoising_combo.addItem("Total Variation Light", {
            'filter': 'total_variation',
            'weight': 0.1
        })
        self.denoising_combo.addItem("Total Variation Medium", {
            'filter': 'total_variation',
            'weight': 0.2
        })
        self.denoising_combo.addItem("Total Variation Strong", {
            'filter': 'total_variation',
            'weight': 0.3
        })
        
        self.denoising_combo.currentIndexChanged.connect(self._on_denoising_changed)
        denoising_layout.addWidget(self.denoising_combo)
        
        parent_layout.addWidget(denoising_group)
        
    def _setup_binarisation_section(self, parent_layout):
        """Setup binarisation filters section."""
        binarisation_group = QGroupBox("Binarisation")
        binarisation_group.setStyleSheet(self._get_group_style())
        binarisation_layout = QVBoxLayout(binarisation_group)
        
        # Binarisation dropdown
        self.binarisation_combo = QComboBox()
        self.binarisation_combo.addItem("-- Select Binarisation Filter --", None)
        self.binarisation_combo.addItem("Auto Threshold", {
            'filter': 'threshold',
            'method': 'otsu'
        })
        self.binarisation_combo.addItem("Low Threshold (85)", {
            'filter': 'threshold',
            'threshold': 85,
            'max_value': 255
        })
        self.binarisation_combo.addItem("Medium Threshold (127)", {
            'filter': 'threshold',
            'threshold': 127,
            'max_value': 255
        })
        self.binarisation_combo.addItem("High Threshold (170)", {
            'filter': 'threshold',
            'threshold': 170,
            'max_value': 255
        })
        
        self.binarisation_combo.currentIndexChanged.connect(self._on_binarisation_changed)
        binarisation_layout.addWidget(self.binarisation_combo)
        
        parent_layout.addWidget(binarisation_group)
        
    def _setup_edge_detection_section(self, parent_layout):
        """Setup edge detection filters section."""
        edge_group = QGroupBox("Edge Detection")
        edge_group.setStyleSheet(self._get_group_style())
        edge_layout = QVBoxLayout(edge_group)
        
        # Edge detection dropdown
        self.edge_combo = QComboBox()
        self.edge_combo.addItem("-- Select Edge Detection Filter --", None)
        self.edge_combo.addItem("Canny Soft Edges", {
            'filter': 'canny',
            'low_threshold': 30,
            'high_threshold': 100
        })
        self.edge_combo.addItem("Canny Normal Edges", {
            'filter': 'canny',
            'low_threshold': 50,
            'high_threshold': 150
        })
        self.edge_combo.addItem("Canny Sharp Edges", {
            'filter': 'canny',
            'low_threshold': 80,
            'high_threshold': 200
        })
        self.edge_combo.addItem("Laplacian Soft", {
            'filter': 'laplacian',
            'kernel_size': 3
        })
        self.edge_combo.addItem("Laplacian Strong", {
            'filter': 'laplacian',
            'kernel_size': 5
        })
        
        self.edge_combo.currentIndexChanged.connect(self._on_edge_detection_changed)
        edge_layout.addWidget(self.edge_combo)
        
        parent_layout.addWidget(edge_group)
        
    def _setup_control_buttons(self, parent_layout):
        """Setup control buttons."""
        button_group = QGroupBox("Actions")
        button_group.setStyleSheet(self._get_group_style())
        button_layout = QVBoxLayout(button_group)
        
        # Run Automatic Filtering button (disabled/placeholder)
        self.run_button = QPushButton("Run Automatic Filtering")
        self.run_button.setEnabled(False)  # Disabled as per requirements
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #7f8c8d;
                color: #bdc3c7;
                border: none;
                padding: 12px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
                min-height: 35px;
            }
            QPushButton:enabled {
                background-color: #27ae60;
                color: white;
            }
            QPushButton:hover:enabled {
                background-color: #229954;
            }
            QPushButton:pressed:enabled {
                background-color: #1e8449;
            }
        """)
        self.run_button.clicked.connect(self._on_run_automatic_filtering)
        button_layout.addWidget(self.run_button)
        
        # Reset selections button
        self.reset_button = QPushButton("Reset Selections")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
            QPushButton:pressed {
                background-color: #d35400;
            }
        """)
        self.reset_button.clicked.connect(self._on_reset_selections)
        button_layout.addWidget(self.reset_button)
        
        parent_layout.addWidget(button_group)
        
    def _setup_progress_section(self, parent_layout):
        """Setup progress indicator area."""
        progress_group = QGroupBox("Processing Status")
        progress_group.setStyleSheet(self._get_group_style())
        progress_layout = QVBoxLayout(progress_group)
        
        # Status label
        self.status_label = QLabel("Ready - Select filters and run automatic processing")
        self.status_label.setStyleSheet("color: #ffffff; font-size: 11px;")
        progress_layout.addWidget(self.status_label)
        
        # Progress bar (for future use)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)  # Hidden by default
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                text-align: center;
                color: white;
                background-color: #2b2b2b;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        parent_layout.addWidget(progress_group)
        
    def _get_group_style(self):
        """Get consistent group box styling."""
        return """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 5px;
                color: #ffffff;
                background-color: #2b2b2b;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QComboBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
                font-size: 11px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #ffffff;
            }
        """
        
    def _on_contrast_changed(self, index):
        """Handle contrast filter selection change."""
        data = self.contrast_combo.currentData()
        self.filter_selections['contrast'] = data
        self._update_status()
        
    def _on_denoising_changed(self, index):
        """Handle denoising filter selection change."""
        data = self.denoising_combo.currentData()
        self.filter_selections['denoising'] = data
        self._update_status()
        
    def _on_binarisation_changed(self, index):
        """Handle binarisation filter selection change."""
        data = self.binarisation_combo.currentData()
        self.filter_selections['binarisation'] = data
        self._update_status()
        
    def _on_edge_detection_changed(self, index):
        """Handle edge detection filter selection change."""
        data = self.edge_combo.currentData()
        self.filter_selections['edge_detection'] = data
        self._update_status()
        
    def _update_status(self):
        """Update status display based on current selections."""
        selected_count = sum(1 for selection in self.filter_selections.values() if selection is not None)
        
        if selected_count == 0:
            self.status_label.setText("Ready - Select filters and run automatic processing")
        elif selected_count == 1:
            self.status_label.setText(f"1 filter selected - Add more or run processing")
        else:
            self.status_label.setText(f"{selected_count} filters selected - Ready for automatic processing")
            
        # Note: Run button remains disabled as per Step 11 requirements (placeholder functionality)
        
    def _on_run_automatic_filtering(self):
        """Handle run automatic filtering button click (placeholder)."""
        # Placeholder functionality - actual implementation in future steps
        selected_filters = {k: v for k, v in self.filter_selections.items() if v is not None}
        
        if selected_filters:
            self.status_label.setText("Processing... (Placeholder - functionality to be implemented)")
            self.automatic_filtering_requested.emit(selected_filters)
        else:
            self.status_label.setText("No filters selected")
            
    def _on_reset_selections(self):
        """Reset all filter selections."""
        self.contrast_combo.setCurrentIndex(0)
        self.denoising_combo.setCurrentIndex(0)
        self.binarisation_combo.setCurrentIndex(0)
        self.edge_combo.setCurrentIndex(0)
        
        self.filter_selections = {
            'contrast': None,
            'denoising': None,
            'binarisation': None,
            'edge_detection': None
        }
        
        self.status_label.setText("Selections reset - Ready to select filters")
        
    def get_selected_filters(self) -> Dict[str, Any]:
        """Get current filter selections."""
        return {k: v for k, v in self.filter_selections.items() if v is not None}
