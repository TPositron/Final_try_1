"""
Unified Main Window - Image Analysis Tool
Combines functionality from both main_window.py and modular_main_window_clean.py

This is the primary main window that provides:
1. Core SEM/GDS alignment functionality
2. Advanced filtering capabilities  
3. Sequential filtering workflow
4. Comprehensive scoring system
5. Hybrid alignment with 3-point selection

Use this file as the main entry point for the application.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QMenuBar, QMenu, QFileDialog, QMessageBox, QDockWidget,
                               QApplication, QStatusBar, QSplitter, QPushButton, QComboBox, QLabel,
                               QToolBar, QButtonGroup, QGroupBox, QTextEdit, QTabWidget, QStackedWidget)
from PySide6.QtCore import Qt, QTimer, Signal, QPoint
from pathlib import Path
from PySide6.QtGui import QAction, QIcon, QKeySequence

# Import UI components and services
from src.ui.components.image_viewer import ImageViewer
from src.ui.components.file_selector import FileSelector
from src.ui.components.histogram_view import HistogramView
from src.ui.view_manager import ViewManager, ViewMode
from src.ui.base_panels import ViewPanelManager
from src.services.simple_file_service import FileService
from src.services.simple_image_processing_service import ImageProcessingService
from src.services.transformations.transform_service import TransformService

# Import modular managers
from src.ui.managers.file_operations_manager import FileOperationsManager
from src.ui.managers.gds_operations_manager import GDSOperationsManager
from src.ui.managers.image_processing_manager import ImageProcessingManager
from src.ui.managers.alignment_operations_manager import AlignmentOperationsManager
from src.ui.managers.scoring_operations_manager import ScoringOperationsManager

# Import panels
from src.ui.panels.mode_switcher import ModeSwitcher
from src.ui.panels.alignment_panel import AlignmentPanel
from src.ui.panels.filter_panel import FilterPanel
from src.ui.panels.score_panel import ScorePanel
from src.ui.panels.alignment_left_panel import ManualAlignmentTab, ThreePointAlignmentTab

# Import filtering panels
from src.ui.panels.advanced_filtering_panels import (
    AdvancedFilteringLeftPanel, 
    AdvancedFilteringRightPanel
)
from src.ui.panels.sequential_filtering_panels import (
    SequentialFilteringLeftPanel,
    SequentialFilteringRightPanel,
    ProcessingStage
)

# Import core models
from src.core.models.structure_definitions import get_default_structures

import logging
logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Unified Main Window combining all functionality.
    
    Features:
    - Core SEM/GDS alignment with manual and hybrid modes
    - Advanced filtering with real-time preview
    - Sequential filtering workflow
    - Comprehensive scoring system
    - Robust error handling
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Analysis Tool - Unified")
        self.setMinimumSize(1400, 900)
        
        # Initialize core services
        self.file_service = FileService(self)
        self.image_processing_service = ImageProcessingService()
        self.transform_service = TransformService(self)
        
        # Store original image for filter reset
        self.original_sem_image = None
        
        # Initialize application state
        self.current_mode = "Manual"
        self.current_sem_data = None
        self.current_gds_data = None
        self.current_sem_path = None
        self.current_gds_path = None
        self.current_structure_id = 1
        self.current_aligned_gds_model = None
        self.current_sem_image = None
        self.current_sem_image_obj = None
        self.current_gds_overlay = None
        self.current_alignment_result = None
        self.current_scoring_results = {}
        self.current_scoring_method = "SSIM"
        
        # Sequential processing state
        self.sequential_images = {}
        self.current_processing_stage = None
        
        # Hybrid alignment state
        self.sem_points = []
        self.gds_points = []
        
        # Initialize managers
        self._initialize_managers()
        
        # Setup UI
        self._setup_ui()
        
        # Connect signals
        self._connect_signals()
        
        # Setup error handling
        self.setup_error_handling()
        
        logger.info("Unified main window initialized successfully")
    
    def _on_manual_alignment_changed(self, params):
        """Handle manual alignment parameter changes."""
        try:
            # Get original GDS overlay
            if not hasattr(self, 'gds_operations_manager') or self.gds_operations_manager.current_gds_overlay is None:
                return
            
            original_overlay = self.gds_operations_manager.current_gds_overlay
            
            # Apply transformations for preview (move and zoom only)
            transformed_overlay = self._apply_coordinate_transformations(original_overlay, params)
            
            # Update image viewer with transformed overlay
            if hasattr(self, 'image_viewer'):
                self.image_viewer.set_gds_overlay(transformed_overlay)
                self.image_viewer.set_overlay_visible(True)
                
                # Update transparency if specified
                if 'transparency' in params:
                    alpha = (100 - params['transparency']) / 100.0  # Convert to alpha (0-1)
                    self.image_viewer.set_overlay_alpha(alpha)
            
            # Store current transformed overlay
            self.current_gds_overlay = transformed_overlay
            
            # Enable save button when parameters change
            if hasattr(self, 'generate_aligned_gds_btn'):
                self.generate_aligned_gds_btn.setEnabled(True)
            
            logger.debug(f"Manual alignment parameters updated: {params}")
            
        except Exception as e:
            logger.error(f"Error handling manual alignment change: {e}")
    
    def _get_current_alignment_parameters(self):
        """Get current alignment parameters from manual alignment controls."""
        try:
            if hasattr(self, 'manual_alignment_controls'):
                return self.manual_alignment_controls.get_parameters()
            return {}
        except Exception as e:
            logger.error(f"Error getting alignment parameters: {e}")
            return {}
    
    def _apply_coordinate_transformations(self, overlay, params):
        """Apply coordinate transformations with proper center-based zoom."""
        try:
            import cv2
            import numpy as np
            
            transformed = overlay.copy()
            height, width = transformed.shape[:2]
            
            # Get transformation parameters
            x_offset = params.get('x_offset', 0)
            y_offset = params.get('y_offset', 0)
            scale = params.get('scale', 1.0)
            rotation = params.get('rotation', 0)
            
            # Calculate center point for scaling and rotation
            center_x = width / 2.0
            center_y = height / 2.0
            
            # Create transformation matrix for combined operations
            # Use getRotationMatrix2D for proper center-based scaling and rotation
            transform_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation, scale)
            
            # Add translation to the transformation matrix
            transform_matrix[0, 2] += x_offset
            transform_matrix[1, 2] += y_offset
            
            # Apply the combined transformation
            transformed = cv2.warpAffine(transformed, transform_matrix, (width, height),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error applying coordinate transformations: {e}")
            return overlay
    
    def _calculate_transformed_bounds(self, params, structure_id):
        """Calculate new GDS bounds based on alignment parameters."""
        try:
            # Get original bounds for the structure
            gds_service = self.gds_operations_manager.new_gds_service
            structures_info = gds_service.get_all_structures_info()
            
            if structure_id not in structures_info:
                raise ValueError(f"Structure {structure_id} not found")
            
            original_bounds = structures_info[structure_id]['bounds']
            
            # Calculate pixel to coordinate ratio
            bounds_width = original_bounds[2] - original_bounds[0]
            bounds_height = original_bounds[3] - original_bounds[1]
            
            pixel_to_coord_x = bounds_width / 1024
            pixel_to_coord_y = bounds_height / 666
            
            # Get transformation parameters
            x_offset_pixels = params.get('x_offset', 0)
            y_offset_pixels = params.get('y_offset', 0)
            scale = params.get('scale', 1.0)
            rotation = params.get('rotation', 0)
            
            # Convert pixel offsets to coordinate offsets
            x_offset_coord = x_offset_pixels * pixel_to_coord_x
            y_offset_coord = -y_offset_pixels * pixel_to_coord_y  # Flip Y for GDS coordinates
            
            # Calculate center of original bounds
            center_x = (original_bounds[0] + original_bounds[2]) / 2
            center_y = (original_bounds[1] + original_bounds[3]) / 2
            
            # Apply transformations in order: move, scale, rotation
            # 1. Move center
            new_center_x = center_x + x_offset_coord
            new_center_y = center_y + y_offset_coord
            
            # 2. Scale from new center
            scaled_width = bounds_width / scale  # Inverse scale for bounds
            scaled_height = bounds_height / scale
            
            # 3. Calculate new bounds
            new_bounds = [
                new_center_x - scaled_width/2,   # min_x
                new_center_y - scaled_height/2,  # min_y
                new_center_x + scaled_width/2,   # max_x
                new_center_y + scaled_height/2   # max_y
            ]
            
            print(f"DEBUG: Original bounds: {original_bounds}")
            print(f"DEBUG: Params: x={x_offset_pixels}, y={y_offset_pixels}, scale={scale}, rot={rotation}")
            print(f"DEBUG: New bounds: {new_bounds}")
            
            return new_bounds, rotation
            
        except Exception as e:
            logger.error(f"Error calculating transformed bounds: {e}")
            import traceback
            traceback.print_exc()
            return structures_info[structure_id]['bounds'], 0 if structure_id in structures_info else ([0, 0, 1000, 1000], 0)
    
    def _reset_save_button(self):
        """Reset the save button to original state."""
        if hasattr(self, 'generate_aligned_gds_btn'):
            self.generate_aligned_gds_btn.setText("Save Aligned GDS")
            self.generate_aligned_gds_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    padding: 8px;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                    color: #666666;
                }
            """)
    
    def _auto_load_default_gds(self):
        """Auto-load the default GDS file and select Structure 1."""
        try:
            print("Auto-loading default GDS file...")
            default_gds_path = Path("Data/GDS/Institute_Project_GDS1.gds")
            
            if default_gds_path.exists():
                # Load GDS file
                self.gds_operations_manager.load_gds_file_from_path(str(default_gds_path))
                
                # Auto-select Structure 1 after a short delay
                QTimer.singleShot(500, lambda: self.gds_operations_manager.select_structure_by_id(1))
                
                print(f"Auto-loaded GDS file: {default_gds_path}")
            else:
                print(f"Default GDS file not found: {default_gds_path}")
                
        except Exception as e:
            print(f"Error auto-loading default GDS: {e}")
    
    def _initialize_managers(self):
        """Initialize all manager modules."""
        try:
            self.file_operations_manager = FileOperationsManager(self)
            self.gds_operations_manager = GDSOperationsManager(self)
            self.image_processing_manager = ImageProcessingManager(self)
            self.alignment_operations_manager = AlignmentOperationsManager(self)
            self.scoring_operations_manager = ScoringOperationsManager(self)
            
            logger.info("All manager modules initialized")
            
        except Exception as e:
            logger.error(f"Error initializing managers: {e}")
            raise
    
    def _setup_ui(self):
        """Setup the unified UI layout."""
        try:
            # Central widget and layout
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QHBoxLayout(central_widget)
            
            # Main splitter
            self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
            main_layout.addWidget(self.main_splitter)
            
            # Left panel with tabs
            self._setup_left_panel()
            
            # Central image viewer
            self.image_viewer = ImageViewer()
            self.image_viewer.point_selected.connect(self._on_point_selected)
            self.main_splitter.addWidget(self.image_viewer)
            
            # Right panel
            self._setup_right_panel()
            
            # Set splitter sizes
            self.main_splitter.setSizes([400, 1000, 350])
            
            # Setup menus and status bar
            self._setup_menus()
            self._setup_status_bar()
            
            logger.info("UI setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up UI: {e}")
            raise
    
    def _setup_left_panel(self):
        """Setup left panel with all tabs."""
        self.left_panel_container = QWidget()
        self.left_panel_layout = QVBoxLayout(self.left_panel_container)
        self.left_panel_layout.setContentsMargins(0, 0, 0, 0)
        
        # Main tab widget
        self.main_tab_widget = QTabWidget()
        self.main_tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444444;
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                border: 1px solid #444444;
                padding: 8px 12px;
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
        """)
        
        # Create tabs
        self.alignment_tab = QWidget()
        self.advanced_filtering_tab = QWidget()
        self.sequential_filtering_tab = QWidget()
        self.scoring_tab = QWidget()
        
        # Setup tab contents
        self._setup_alignment_tab()
        self._setup_advanced_filtering_tab()
        self._setup_sequential_filtering_tab()
        self._setup_scoring_tab()
        
        # Add tabs
        self.main_tab_widget.addTab(self.alignment_tab, "Alignment")
        self.main_tab_widget.addTab(self.advanced_filtering_tab, "Advanced Filtering")
        self.main_tab_widget.addTab(self.sequential_filtering_tab, "Sequential Filtering")
        self.main_tab_widget.addTab(self.scoring_tab, "Scoring")
        
        # Connect tab changes
        self.main_tab_widget.currentChanged.connect(self._on_tab_changed)
        
        self.left_panel_layout.addWidget(self.main_tab_widget)
        self.main_splitter.addWidget(self.left_panel_container)
    
    def _setup_right_panel(self):
        """Setup right panel with file selector and histogram."""
        self.right_panel_container = QWidget()
        self.right_panel_layout = QVBoxLayout(self.right_panel_container)
        
        # File selector
        self.file_selector = FileSelector()
        self.file_selector.sem_file_selected.connect(self.load_sem_file)
        self.file_selector.gds_file_loaded.connect(self.gds_operations_manager.load_gds_file_from_path)
        self.file_selector.gds_structure_selected.connect(self._handle_structure_selection)
        self.right_panel_layout.addWidget(self.file_selector)
        
        # Histogram view
        self.histogram_view = HistogramView()
        self.histogram_view.setMaximumHeight(200)
        self.histogram_view.setMinimumHeight(150)
        self.right_panel_layout.addWidget(self.histogram_view)
        
        # View-specific content area
        self.view_specific_widget = QWidget()
        self.view_specific_layout = QVBoxLayout(self.view_specific_widget)
        self.right_panel_layout.addWidget(self.view_specific_widget)
        
        self.main_splitter.addWidget(self.right_panel_container)
        
        # Initialize file scanning
        self.file_selector.scan_directories()
        
        # Auto-load default GDS file after a short delay
        QTimer.singleShot(1000, self._auto_load_default_gds)
    
    def _setup_alignment_tab(self):
        """Setup alignment tab with manual and hybrid modes."""
        layout = QVBoxLayout(self.alignment_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Sub-tabs for alignment modes
        self.alignment_sub_tabs = QTabWidget()
        
        # Manual alignment
        self.manual_alignment_tab = QWidget()
        manual_layout = QVBoxLayout(self.manual_alignment_tab)
        self.manual_alignment_controls = ManualAlignmentTab()
        self.manual_alignment_controls.alignment_changed.connect(self._on_manual_alignment_changed)
        manual_layout.addWidget(self.manual_alignment_controls)
        self.alignment_sub_tabs.addTab(self.manual_alignment_tab, "Manual")
        
        # Hybrid alignment
        self.hybrid_alignment_tab = QWidget()
        self._setup_hybrid_alignment_content()
        self.alignment_sub_tabs.addTab(self.hybrid_alignment_tab, "Hybrid")
        
        layout.addWidget(self.alignment_sub_tabs)
        
        # Action buttons
        self._setup_alignment_action_buttons(layout)
        
        # Connect sub-tab changes
        self.alignment_sub_tabs.currentChanged.connect(self._on_alignment_subtab_changed)
        
        # Connect manual alignment reset signal
        if hasattr(self.manual_alignment_controls, 'reset_requested'):
            self.manual_alignment_controls.reset_requested.connect(self._on_reset_transformation_clicked)
    
    def _setup_hybrid_alignment_content(self):
        """Setup hybrid alignment with 3-point selection."""
        layout = QVBoxLayout(self.hybrid_alignment_tab)
        
        # Instructions
        instructions = QLabel("Select 3 corresponding points on SEM and GDS images for alignment.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("""
            QLabel {
                background-color: #3c3c3c;
                color: #ffffff;
                padding: 8px;
                border-radius: 4px;
                font-size: 12px;
            }
        """)
        layout.addWidget(instructions)
        
        # Point status
        self.points_status_label = QLabel("SEM Points: 0/3  |  GDS Points: 0/3")
        self.points_status_label.setStyleSheet("color: #ffffff; font-size: 12px;")
        layout.addWidget(self.points_status_label)
        
        # Ready status
        self.ready_status_label = QLabel("Status: Not Ready")
        self.ready_status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
        layout.addWidget(self.ready_status_label)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.clear_points_btn = QPushButton("Clear All Points")
        self.clear_points_btn.clicked.connect(self._clear_all_points)
        buttons_layout.addWidget(self.clear_points_btn)
        
        self.calculate_alignment_btn = QPushButton("Calculate Alignment")
        self.calculate_alignment_btn.setEnabled(False)
        self.calculate_alignment_btn.clicked.connect(self._calculate_alignment)
        buttons_layout.addWidget(self.calculate_alignment_btn)
        
        layout.addLayout(buttons_layout)
        layout.addStretch()
    
    def _setup_advanced_filtering_tab(self):
        """Setup advanced filtering tab."""
        layout = QHBoxLayout(self.advanced_filtering_tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Advanced filtering controls
        self.advanced_filtering_left_panel = AdvancedFilteringLeftPanel()
        self.advanced_filtering_left_panel.filter_applied.connect(self._on_advanced_filter_applied)
        self.advanced_filtering_left_panel.filter_previewed.connect(self._on_advanced_filter_previewed)
        self.advanced_filtering_left_panel.filter_reset.connect(self._on_advanced_filter_reset)
        self.advanced_filtering_left_panel.save_image_requested.connect(self._save_filtered_image)
        splitter.addWidget(self.advanced_filtering_left_panel)
        
        # Right panel: Info display
        self.advanced_filtering_right_panel = AdvancedFilteringRightPanel()
        splitter.addWidget(self.advanced_filtering_right_panel)
        
        splitter.setSizes([450, 300])
        layout.addWidget(splitter)
    
    def _setup_sequential_filtering_tab(self):
        """Setup sequential filtering tab."""
        layout = QHBoxLayout(self.sequential_filtering_tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Sequential controls
        self.sequential_filtering_left_panel = SequentialFilteringLeftPanel()
        self.sequential_filtering_left_panel.stage_preview_requested.connect(self._on_stage_preview_requested)
        self.sequential_filtering_left_panel.stage_apply_requested.connect(self._on_stage_apply_requested)
        self.sequential_filtering_left_panel.stage_reset_requested.connect(self._on_stage_reset_requested)
        splitter.addWidget(self.sequential_filtering_left_panel)
        
        # Right panel: Progress display
        self.sequential_filtering_right_panel = SequentialFilteringRightPanel()
        splitter.addWidget(self.sequential_filtering_right_panel)
        
        splitter.setSizes([450, 300])
        layout.addWidget(splitter)
    
    def _setup_scoring_tab(self):
        """Setup scoring tab."""
        layout = QVBoxLayout(self.scoring_tab)
        
        from src.ui.panels.scoring_tab_panel import ScoringTabPanel
        self.scoring_tab_panel = ScoringTabPanel()
        self.scoring_tab_panel.scoring_method_changed.connect(self._on_scoring_method_changed)
        self.scoring_tab_panel.calculate_scores_requested.connect(self._on_calculate_scores_requested)
        
        layout.addWidget(self.scoring_tab_panel)
    
    def _setup_alignment_action_buttons(self, parent_layout):
        """Setup action buttons for alignment tab."""
        parent_layout.addSpacing(20)
        
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)
        
        # First row
        row1_layout = QHBoxLayout()
        
        self.auto_align_btn = QPushButton("Automatic Alignment")
        self.auto_align_btn.clicked.connect(self._on_auto_align_clicked)
        row1_layout.addWidget(self.auto_align_btn)
        
        self.reset_transformation_btn = QPushButton("Reset Transformation")
        self.reset_transformation_btn.clicked.connect(self._on_reset_transformation_clicked)
        row1_layout.addWidget(self.reset_transformation_btn)
        
        action_layout.addLayout(row1_layout)
        
        # Second row - Save Aligned GDS (prominent button)
        self.generate_aligned_gds_btn = QPushButton("Save Aligned GDS")
        self.generate_aligned_gds_btn.setEnabled(False)
        self.generate_aligned_gds_btn.clicked.connect(self._on_generate_aligned_gds_clicked)
        self.generate_aligned_gds_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        action_layout.addWidget(self.generate_aligned_gds_btn)
        
        parent_layout.addWidget(action_group)
    
    def _setup_menus(self):
        """Setup application menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_sem_action = QAction("&Open SEM Image...", self)
        open_sem_action.setShortcut(QKeySequence.Open)
        open_sem_action.triggered.connect(self.open_sem_file)
        file_menu.addAction(open_sem_action)
        
        open_gds_action = QAction("Open &GDS File...", self)
        open_gds_action.setShortcut(QKeySequence("Ctrl+G"))
        open_gds_action.triggered.connect(self.open_gds_file)
        file_menu.addAction(open_gds_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("&Save Results...", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        fit_action = QAction("&Fit to Window", self)
        fit_action.setShortcut(QKeySequence("Ctrl+0"))
        fit_action.triggered.connect(self.fit_to_window)
        view_menu.addAction(fit_action)
        
        actual_size_action = QAction("&Actual Size", self)
        actual_size_action.setShortcut(QKeySequence("Ctrl+1"))
        actual_size_action.triggered.connect(self.actual_size)
        view_menu.addAction(actual_size_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        auto_align_action = QAction("&Auto Alignment", self)
        auto_align_action.setShortcut(QKeySequence("Ctrl+A"))
        auto_align_action.triggered.connect(self.run_auto_alignment)
        tools_menu.addAction(auto_align_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def _setup_status_bar(self):
        """Setup status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Unified Interface")
    
    def _connect_signals(self):
        """Connect all signals between components."""
        try:
            # File operations signals
            self.file_operations_manager.sem_image_loaded.connect(self.on_sem_image_loaded)
            self.file_operations_manager.file_operation_error.connect(self.on_file_error)
            
            # GDS operations signals
            self.gds_operations_manager.gds_file_loaded.connect(self.on_gds_file_loaded)
            self.gds_operations_manager.structure_selected.connect(self.on_structure_selected)
            self.gds_operations_manager.gds_operation_error.connect(self.on_gds_error)
            
            # Image processing signals
            self.image_processing_manager.filter_applied.connect(self.on_filter_applied)
            self.image_processing_manager.filters_reset.connect(self.on_filters_reset)
            
            # Alignment operations signals
            self.alignment_operations_manager.alignment_completed.connect(self.on_alignment_completed)
            self.alignment_operations_manager.alignment_reset.connect(self.on_alignment_reset)
            
            # Scoring operations signals
            self.scoring_operations_manager.scores_calculated.connect(self.on_scores_calculated)
            
            # File service signals
            self.file_service.loading_progress.connect(self.update_loading_progress)
            self.file_service.file_loaded.connect(self.on_file_loaded)
            self.file_service.loading_error.connect(self.on_loading_error)
            
            # Transform service signals
            self.transform_service.transform_applied.connect(self.update_gds_display)
            
            logger.info("All signals connected successfully")
            
        except Exception as e:
            logger.error(f"Error connecting signals: {e}")
    
    # File operations
    def load_sem_file(self, filepath: str):
        """Load SEM file."""
        try:
            from pathlib import Path
            result = self.file_service.load_sem_file(Path(filepath))
            if result:
                self.current_sem_data = result
                self.current_sem_path = filepath
                self.current_sem_image_obj = result
                
                if 'cropped_array' in result:
                    self.current_sem_image = result['cropped_array']
                    # Store original for filter reset
                    self.original_sem_image = self.current_sem_image.copy()
                    
                    # Set image in image processing service
                    if hasattr(self, 'image_processing_service'):
                        self.image_processing_service.set_image(self.current_sem_image)
                    
                    self.image_viewer.set_sem_image(self.current_sem_image)
                    self.histogram_view.update_histogram(self.current_sem_image)
                
                logger.info(f"SEM file loaded: {filepath}")
                return result
            else:
                raise ValueError(f"Failed to load SEM file: {filepath}")
                
        except Exception as e:
            logger.error(f"Error loading SEM file: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load SEM file: {e}")
    
    def load_gds_file(self, filepath: str):
        """Load GDS file with default structure."""
        self.gds_operations_manager.load_gds_file_from_path(filepath)
    
    def _handle_structure_selection(self, gds_file_path: str, structure_id: int):
        """Handle structure selection from FileSelector."""
        print(f"Structure selection: {structure_id} from {gds_file_path}")
        self.gds_operations_manager.select_structure_by_id(structure_id)
    
    def load_gds_file_with_structure(self, filepath: str, structure_id: int):
        """Load GDS file with specific structure."""
        try:
            from pathlib import Path
            result = self.file_service.load_gds_file(Path(filepath), structure_id=structure_id)
            if result:
                self.current_gds_data = result
                self.current_gds_path = filepath
                self.current_structure_id = structure_id
                
                if 'extracted_structure' in result:
                    binary_image = result['extracted_structure']['binary_image']
                    # Ensure binary image is properly formatted
                    if binary_image is not None:
                        # Convert to uint8 if needed
                        if binary_image.dtype != np.uint8:
                            binary_image = (binary_image * 255).astype(np.uint8)
                        
                        self.current_gds_overlay = binary_image
                        self.image_viewer.set_gds_overlay(binary_image)
                        
                        # Force overlay to be visible
                        self.image_viewer.set_overlay_visible(True)
                        
                        print(f"GDS overlay set: shape={binary_image.shape}, dtype={binary_image.dtype}, max={binary_image.max()}")
                
                logger.info(f"GDS file loaded: {filepath}, structure: {structure_id}")
                return result
            else:
                raise ValueError(f"Failed to load GDS structure {structure_id}: {filepath}")
                
        except Exception as e:
            logger.error(f"Error loading GDS file: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load GDS file: {e}")
    
    def open_sem_file(self):
        """Open SEM file dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open SEM Image", "", 
            "Image Files (*.tif *.tiff *.png *.jpg);;All Files (*)"
        )
        if filepath:
            self.load_sem_file(filepath)
    
    def open_gds_file(self):
        """Open GDS file dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open GDS File", "", 
            "GDS Files (*.gds);;All Files (*)"
        )
        if filepath:
            self.load_gds_file(filepath)
    
    def save_results(self):
        """Save current results."""
        try:
            if hasattr(self, 'file_operations_manager'):
                self.file_operations_manager.save_results()
            else:
                QMessageBox.information(self, "Info", "No results to save")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save results: {e}")
    
    # View operations
    def fit_to_window(self):
        """Fit content to window."""
        if hasattr(self.image_viewer, 'fit_to_window'):
            self.image_viewer.fit_to_window()
    
    def actual_size(self):
        """Show content at actual size."""
        if hasattr(self.image_viewer, 'reset_view'):
            self.image_viewer.reset_view()
    
    def run_auto_alignment(self):
        """Run automatic alignment."""
        try:
            if hasattr(self, 'alignment_operations_manager'):
                self.alignment_operations_manager.auto_align()
            else:
                QMessageBox.information(self, "Info", "Auto alignment not available")
        except Exception as e:
            logger.error(f"Error in auto alignment: {e}")
            QMessageBox.critical(self, "Error", f"Auto alignment failed: {e}")
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Image Analysis Tool",
            "Unified Image Analysis Tool\n\n"
            "A comprehensive tool for aligning SEM images with GDS layouts\n"
            "and analyzing their correspondence.\n\n"
            "Features:\n"
            "• Manual and hybrid alignment\n"
            "• Advanced filtering\n"
            "• Sequential filtering workflow\n"
            "• Comprehensive scoring\n\n"
            "Version 2.0"
        )
    
    # Signal handlers
    def on_sem_image_loaded(self, file_path, image_data):
        """Handle SEM image loaded."""
        logger.info(f"SEM image loaded: {file_path}")
        if 'cropped_array' in image_data:
            self.current_sem_image = image_data['cropped_array']
            self.original_sem_image = self.current_sem_image.copy()
            
            # Set image in image processing service
            if hasattr(self, 'image_processing_service'):
                self.image_processing_service.set_image(self.current_sem_image)
            
            self.histogram_view.update_histogram(self.current_sem_image)
    
    def on_gds_file_loaded(self, file_path):
        """Handle GDS file loaded."""
        logger.info(f"GDS file loaded: {file_path}")
        self.status_bar.showMessage(f"GDS file loaded: {Path(file_path).name}")
    
    def on_structure_selected(self, structure_name, overlay):
        """Handle structure selected."""
        logger.info(f"Structure selected: {structure_name}")
        if overlay is not None:
            # Ensure overlay is properly formatted
            if overlay.dtype != np.uint8:
                overlay = (overlay * 255).astype(np.uint8)
            
            self.current_gds_overlay = overlay
            self.image_viewer.set_gds_overlay(overlay)
            self.image_viewer.set_overlay_visible(True)
            
            self.status_bar.showMessage(f"Structure loaded: {structure_name}")
            print(f"Structure overlay set: shape={overlay.shape}, dtype={overlay.dtype}, max={overlay.max()}")
        else:
            print("Warning: Received None overlay for structure")
    
    def on_filter_applied(self, filter_name, parameters):
        """Handle filter applied."""
        logger.info(f"Filter applied: {filter_name}")
        if hasattr(self, 'current_sem_image') and self.current_sem_image is not None:
            self.histogram_view.update_histogram(self.current_sem_image)
    
    def on_filters_reset(self):
        """Handle filters reset."""
        logger.info("Filters reset")
        if hasattr(self, 'current_sem_image') and self.current_sem_image is not None:
            self.histogram_view.update_histogram(self.current_sem_image)
    
    def on_alignment_completed(self, alignment_result):
        """Handle alignment completed."""
        logger.info("Alignment completed")
        self.current_alignment_result = alignment_result
        if hasattr(self, 'generate_aligned_gds_btn'):
            self.generate_aligned_gds_btn.setEnabled(True)
    
    def on_alignment_reset(self):
        """Handle alignment reset."""
        logger.info("Alignment reset")
        self.current_alignment_result = None
        if hasattr(self, 'generate_aligned_gds_btn'):
            self.generate_aligned_gds_btn.setEnabled(False)
    
    def on_scores_calculated(self, scores):
        """Handle scores calculated."""
        logger.info(f"Scores calculated: {scores}")
        self.current_scoring_results = scores
        if hasattr(self, 'scoring_tab_panel'):
            self.scoring_tab_panel.display_results(scores)
    
    def on_file_error(self, operation, error_message):
        """Handle file operation error."""
        logger.error(f"File operation error ({operation}): {error_message}")
        QMessageBox.critical(self, "File Error", f"{operation} failed: {error_message}")
    
    def on_gds_error(self, operation, error_message):
        """Handle GDS operation error."""
        logger.error(f"GDS operation error ({operation}): {error_message}")
        self.status_bar.showMessage(f"GDS error: {operation} failed")
        QMessageBox.critical(self, "GDS Error", f"{operation} failed: {error_message}")
    
    def update_loading_progress(self, message: str):
        """Update loading progress."""
        self.status_bar.showMessage(message)
    
    def on_file_loaded(self, file_type: str, file_path: str):
        """Handle file loaded."""
        self.status_bar.showMessage(f"{file_type} file loaded: {file_path}")
    
    def on_loading_error(self, error_message: str):
        """Handle loading error."""
        self.status_bar.showMessage("Loading failed")
        QMessageBox.critical(self, "Loading Error", error_message)
    
    def update_gds_display(self, transform_data: dict):
        """Update GDS display with transform data."""
        try:
            if hasattr(self.image_viewer, 'update_gds_transform'):
                self.image_viewer.update_gds_transform(transform_data)
        except Exception as e:
            logger.error(f"Error updating GDS display: {e}")
    
    # Tab and mode switching
    def _on_tab_changed(self, index):
        """Handle tab change."""
        try:
            tab_names = ["alignment", "advanced_filtering", "sequential_filtering", "scoring"]
            if 0 <= index < len(tab_names):
                tab_name = tab_names[index]
                logger.info(f"Switched to {tab_name} tab")
                
                # Update display based on tab
                if index == 0:  # Alignment
                    self._switch_to_alignment_display()
                elif index == 1:  # Advanced filtering
                    self._switch_to_advanced_filtering_display()
                elif index == 2:  # Sequential filtering
                    self._switch_to_sequential_filtering_display()
                elif index == 3:  # Scoring
                    self._switch_to_scoring_display()
                    
        except Exception as e:
            logger.error(f"Error handling tab change: {e}")
    
    def _switch_to_alignment_display(self):
        """Switch to alignment display mode."""
        if self.current_sem_image is not None:
            self.image_viewer.set_sem_image(self.current_sem_image)
        if self.current_gds_overlay is not None:
            self.image_viewer.set_gds_overlay(self.current_gds_overlay)
            self.image_viewer.set_overlay_visible(True)
            print(f"Alignment display: GDS overlay visible={self.image_viewer.get_overlay_visible()}")
    
    def _switch_to_advanced_filtering_display(self):
        """Switch to advanced filtering display mode."""
        if self.current_sem_image is not None:
            self.image_viewer.set_sem_image(self.current_sem_image)
        # Hide GDS overlay in filtering mode
        if hasattr(self.image_viewer, 'set_overlay_visible'):
            self.image_viewer.set_overlay_visible(False)
    
    def _switch_to_sequential_filtering_display(self):
        """Switch to sequential filtering display mode."""
        if self.sequential_images:
            # Show latest sequential result
            latest_stage = max(self.sequential_images.keys())
            latest_image = self.sequential_images[latest_stage]
            self.image_viewer.set_sem_image(latest_image)
        elif self.current_sem_image is not None:
            self.image_viewer.set_sem_image(self.current_sem_image)
    
    def _switch_to_scoring_display(self):
        """Switch to scoring display mode."""
        if self.current_sem_image is not None:
            self.image_viewer.set_sem_image(self.current_sem_image)
        if self.current_gds_overlay is not None:
            self.image_viewer.set_gds_overlay(self.current_gds_overlay)
            self.image_viewer.set_overlay_visible(True)
            print(f"Scoring display: GDS overlay visible={self.image_viewer.get_overlay_visible()}")
    
    # Hybrid alignment methods
    def _on_alignment_subtab_changed(self, index):
        """Handle alignment sub-tab change."""
        if index == 1:  # Hybrid tab
            self.image_viewer.set_point_selection_mode(True, "both")
        else:  # Manual tab
            self.image_viewer.set_point_selection_mode(False)
    
    def _on_point_selected(self, x, y, point_type):
        """Handle point selection."""
        try:
            if point_type == "sem":
                if len(self.sem_points) < 3:
                    self.sem_points.append((x, y))
            elif point_type == "gds":
                if len(self.gds_points) < 3:
                    self.gds_points.append((x, y))
            
            self._update_hybrid_status()
            
        except Exception as e:
            logger.error(f"Error handling point selection: {e}")
    
    def _update_hybrid_status(self):
        """Update hybrid alignment status."""
        try:
            sem_count = len(self.sem_points)
            gds_count = len(self.gds_points)
            
            self.points_status_label.setText(f"SEM Points: {sem_count}/3  |  GDS Points: {gds_count}/3")
            
            if sem_count == 3 and gds_count == 3:
                self.ready_status_label.setText("Status: Ready for Alignment")
                self.ready_status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
                self.calculate_alignment_btn.setEnabled(True)
            else:
                self.ready_status_label.setText("Status: Not Ready")
                self.ready_status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
                self.calculate_alignment_btn.setEnabled(False)
                
        except Exception as e:
            logger.error(f"Error updating hybrid status: {e}")
    
    def _clear_all_points(self):
        """Clear all selected points."""
        self.sem_points = []
        self.gds_points = []
        self.image_viewer.clear_points("both")
        self._update_hybrid_status()
    
    def _calculate_alignment(self):
        """Calculate alignment from selected points."""
        try:
            if len(self.sem_points) == 3 and len(self.gds_points) == 3:
                self.alignment_operations_manager.manual_align_3_point(self.sem_points, self.gds_points)
                logger.info("3-point alignment calculation completed")
            else:
                QMessageBox.warning(self, "Warning", "Need exactly 3 points on both SEM and GDS images")
                
        except Exception as e:
            logger.error(f"Error calculating alignment: {e}")
            QMessageBox.critical(self, "Error", f"Alignment calculation failed: {e}")
    
    # Action button handlers
    def _on_auto_align_clicked(self):
        """Handle auto align button click."""
        self.run_auto_alignment()
    
    def _on_reset_transformation_clicked(self):
        """Handle reset transformation button click."""
        try:
            # Reset manual alignment controls
            if hasattr(self, 'manual_alignment_controls'):
                self.manual_alignment_controls.reset_parameters()
            
            # Reset image viewer overlay to original
            if hasattr(self, 'image_viewer') and hasattr(self, 'gds_operations_manager'):
                if self.gds_operations_manager.current_gds_overlay is not None:
                    original_overlay = self.gds_operations_manager.current_gds_overlay
                    self.current_gds_overlay = original_overlay
                    self.image_viewer.set_gds_overlay(original_overlay)
                    self.image_viewer.set_overlay_visible(True)
                    print("Reset: Restored original GDS overlay")
                else:
                    self.image_viewer.set_gds_overlay(None)
                    self.current_gds_overlay = None
                    print("Reset: No original overlay to restore")
            
            # Reset hybrid alignment points
            self._clear_all_points()
            
            self.status_bar.showMessage("Transformation reset")
            logger.info("Transformation reset completed")
            
        except Exception as e:
            logger.error(f"Error resetting transformation: {e}")
            QMessageBox.critical(self, "Reset Error", f"Failed to reset transformation: {e}")
    
    def _on_generate_aligned_gds_clicked(self):
        """Handle save aligned GDS button click - generate GDS with coordinate transformation."""
        try:
            from pathlib import Path
            from datetime import datetime
            import cv2
            
            # Get current alignment parameters
            params = self._get_current_alignment_parameters()
            if not params:
                QMessageBox.warning(self, "Save Error", "No alignment parameters available")
                return
            
            # Check if we have GDS service
            if not hasattr(self, 'gds_operations_manager') or not self.gds_operations_manager.new_gds_service:
                QMessageBox.warning(self, "Save Error", "No GDS service available")
                return
            
            # Create output directory
            output_dir = Path("Results/Aligned/manual")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get current structure ID from GDS operations manager
            if hasattr(self, 'gds_operations_manager') and self.gds_operations_manager.current_structure_name:
                # Extract structure number from name like "Structure 2"
                structure_name = self.gds_operations_manager.current_structure_name
                if structure_name.startswith("Structure "):
                    structure_id = int(structure_name.replace("Structure ", ""))
                else:
                    structure_id = 1
            else:
                structure_id = getattr(self, 'current_structure_id', 1)
            
            print(f"DEBUG: Using structure ID: {structure_id}")
            
            # Calculate new GDS bounds based on alignment parameters
            new_bounds, rotation_angle = self._calculate_transformed_bounds(params, structure_id)
            print(f"DEBUG: Calculated bounds: {new_bounds}, rotation: {rotation_angle}")
            
            # Generate new GDS image with transformed coordinates
            gds_service = self.gds_operations_manager.new_gds_service
            transformed_gds_image = gds_service.generate_structure_display(
                structure_id, (1024, 666), custom_bounds=new_bounds
            )
            print(f"DEBUG: Generated image shape: {transformed_gds_image.shape if transformed_gds_image is not None else 'None'}")
            print(f"DEBUG: Structure ID used: {structure_id}")
            
            if transformed_gds_image is None:
                QMessageBox.warning(self, "Save Error", "Failed to generate transformed GDS image")
                return
            
            # Apply rotation to the final image if needed
            final_image = transformed_gds_image
            if abs(rotation_angle) > 0.1:  # Only rotate if significant rotation
                height, width = final_image.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                final_image = cv2.warpAffine(final_image, rotation_matrix, (width, height),
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Save as PNG image
            png_filename = f"aligned_gds_{timestamp}.png"
            png_path = output_dir / png_filename
            cv2.imwrite(str(png_path), final_image)
            
            # Save transformation parameters
            params_filename = f"alignment_params_{timestamp}.txt"
            params_path = output_dir / params_filename
            with open(params_path, 'w') as f:
                f.write(f"Alignment Parameters - {timestamp}\n")
                f.write(f"Translation X: {params.get('x_offset', 0)} pixels\n")
                f.write(f"Translation Y: {params.get('y_offset', 0)} pixels\n")
                f.write(f"Rotation: {params.get('rotation', 0)} degrees\n")
                f.write(f"Scale: {params.get('scale', 1.0)}\n")
                f.write(f"Method: Manual alignment (coordinate-based)\n")
                f.write(f"New bounds: {new_bounds}\n")
                f.write(f"Applied rotation: {rotation_angle} degrees\n")
            
            self.status_bar.showMessage(f"Aligned GDS saved to: {png_path}")
            QMessageBox.information(self, "Save Successful", 
                                  f"Aligned GDS saved to:\n{png_path}\n\nParameters saved to:\n{params_path}")
            
            # Update button to show success
            self.generate_aligned_gds_btn.setText("Saved!")
            self.generate_aligned_gds_btn.setStyleSheet("""
                QPushButton {
                    background-color: #27AE60;
                    color: white;
                    font-weight: bold;
                    padding: 8px;
                    border: none;
                    border-radius: 4px;
                }
            """)
            
            # Reset button after 2 seconds
            QTimer.singleShot(2000, self._reset_save_button)
            
            logger.info(f"Aligned GDS saved: {png_path}")
            
        except Exception as e:
            logger.error(f"Error generating aligned GDS: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Save Error", f"Failed to save aligned GDS: {e}")
    
    # Filtering signal handlers
    def _on_advanced_filter_applied(self, filter_name: str, parameters: dict):
        """Handle advanced filter applied."""
        try:
            if not hasattr(self, 'current_sem_image') or self.current_sem_image is None:
                print("❌ ERROR: No SEM image loaded")
                return
            
            # Use original image as base for applying filters
            base_image = getattr(self, 'original_sem_image', self.current_sem_image)
            filtered_image = self._apply_filter_directly(filter_name, parameters, base_image)
            if filtered_image is not None:
                self.current_sem_image = filtered_image
                self.image_viewer.set_sem_image(filtered_image)
                self.histogram_view.update_histogram(filtered_image)
                
                if hasattr(self, 'advanced_filtering_right_panel'):
                    self.advanced_filtering_right_panel.update_histogram(filtered_image)
                    self.advanced_filtering_right_panel.show_status(f"✓ Applied {filter_name}", "success")
                
                print(f"✓ Filter applied: {filter_name}")
            else:
                print(f"❌ Filter failed: {filter_name}")
                    
        except Exception as e:
            print(f"Error in filter application: {e}")
    
    def _on_advanced_filter_previewed(self, filter_name: str, parameters: dict):
        """Handle advanced filter preview."""
        try:
            if not hasattr(self, 'current_sem_image') or self.current_sem_image is None:
                print("❌ ERROR: No SEM image loaded for preview")
                return
            
            # Always use original image for preview
            base_image = getattr(self, 'original_sem_image', self.current_sem_image)
            preview_result = self._apply_filter_directly(filter_name, parameters, base_image)
            
            if preview_result is not None:
                self.image_viewer.set_sem_image(preview_result)
                self.histogram_view.update_histogram(preview_result)
                
                if hasattr(self, 'advanced_filtering_right_panel'):
                    self.advanced_filtering_right_panel.update_histogram(preview_result)
                    self.advanced_filtering_right_panel.show_status(f"👁️ Preview: {filter_name}", "info")
                
                print(f"✓ Filter preview: {filter_name}")
            else:
                print(f"❌ Preview failed: {filter_name}")
                    
        except Exception as e:
            print(f"Error previewing advanced filter: {e}")
    
    def _on_advanced_filter_reset(self):
        """Handle advanced filter reset."""
        try:
            if hasattr(self, 'original_sem_image') and self.original_sem_image is not None:
                self.current_sem_image = self.original_sem_image.copy()
                self.image_viewer.set_sem_image(self.current_sem_image)
                self.histogram_view.update_histogram(self.current_sem_image)
                
                if hasattr(self, 'advanced_filtering_right_panel'):
                    self.advanced_filtering_right_panel.update_histogram(self.current_sem_image)
                    self.advanced_filtering_right_panel.show_status("✓ Filters reset", "success")
                
                print("✓ Reset to original image")
            else:
                print("❌ No original image available")
                    
        except Exception as e:
            print(f"Error resetting filters: {e}")
    
    def _on_stage_preview_requested(self, stage_index: int, filter_name: str, parameters: dict):
        """Handle sequential stage preview."""
        try:
            input_image = self._get_stage_input_image(stage_index)
            if input_image is not None:
                preview_result = self._apply_filter_directly(filter_name, parameters, input_image)
                if preview_result is not None:
                    self.image_viewer.set_sem_image(preview_result)
        except Exception as e:
            logger.error(f"Error in stage preview: {e}")
    
    def _on_stage_apply_requested(self, stage_index: int, filter_name: str, parameters: dict):
        """Handle sequential stage apply."""
        try:
            input_image = self._get_stage_input_image(stage_index)
            if input_image is not None:
                result_image = self._apply_filter_directly(filter_name, parameters, input_image)
                if result_image is not None:
                    self.sequential_images[stage_index] = result_image.copy()
                    self.image_viewer.set_sem_image(result_image)
                    self.current_sem_image = result_image
        except Exception as e:
            logger.error(f"Error in stage apply: {e}")
    
    def _on_stage_reset_requested(self, stage_index: int):
        """Handle sequential stage reset."""
        try:
            # Remove this stage and subsequent stages
            stages_to_remove = [i for i in self.sequential_images.keys() if i >= stage_index]
            for stage_idx in stages_to_remove:
                if stage_idx in self.sequential_images:
                    del self.sequential_images[stage_idx]
            
            # Revert to input image for the reset stage
            input_image = self._get_stage_input_image(stage_index)
            if input_image is not None:
                self.image_viewer.set_sem_image(input_image)
                self.current_sem_image = input_image
                
        except Exception as e:
            logger.error(f"Error in stage reset: {e}")
    
    def _get_stage_input_image(self, stage_index: int):
        """Get input image for a sequential stage."""
        try:
            if stage_index == 0:
                # First stage uses original image
                if hasattr(self, 'current_sem_image_obj') and self.current_sem_image_obj:
                    return getattr(self.current_sem_image_obj, 'cropped_array', self.current_sem_image)
                return self.current_sem_image
            else:
                # Subsequent stages use previous stage result
                previous_stage = stage_index - 1
                return self.sequential_images.get(previous_stage)
        except Exception as e:
            logger.error(f"Error getting stage input image: {e}")
            return None
    
    def _apply_filter_directly(self, filter_name: str, parameters: dict, input_image):
        """Apply filter directly using OpenCV."""
        try:
            if input_image is None:
                return None
            
            image = input_image.copy()
            
            # Comprehensive filter implementations
            if filter_name == "gaussian_blur":
                kernel_size = parameters.get('kernel_size', 5)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                sigma = parameters.get('sigma', 0)
                return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            
            elif filter_name == "median_filter":
                kernel_size = parameters.get('kernel_size', 5)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                return cv2.medianBlur(image, kernel_size)
            
            elif filter_name == "bilateral_filter":
                d = parameters.get('d', 9)
                sigma_color = parameters.get('sigma_color', 75)
                sigma_space = parameters.get('sigma_space', 75)
                return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            
            elif filter_name == "clahe":
                clip_limit = parameters.get('clip_limit', 2.0)
                tile_grid_size = parameters.get('tile_grid_size', 8)
                tile_grid_x = parameters.get('tile_grid_x', tile_grid_size)
                tile_grid_y = parameters.get('tile_grid_y', tile_grid_size)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(int(tile_grid_x), int(tile_grid_y)))
                if len(image.shape) == 3:
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    lab[:,:,0] = clahe.apply(lab[:,:,0])
                    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                else:
                    return clahe.apply(image)
            
            elif filter_name == "canny" or filter_name == "edge_detection":
                low_threshold = parameters.get('low_threshold', 50)
                high_threshold = parameters.get('high_threshold', 150)
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                return cv2.Canny(gray.astype(np.uint8), int(low_threshold), int(high_threshold))
            
            elif filter_name == "threshold":
                threshold_value = parameters.get('threshold', 127)
                max_value = parameters.get('max_value', 255)
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                _, result = cv2.threshold(gray.astype(np.uint8), int(threshold_value), int(max_value), cv2.THRESH_BINARY)
                return result
            
            elif filter_name == "unsharp_mask":
                sigma = parameters.get('sigma', 1.0)
                strength = parameters.get('strength', 1.5)
                blurred = cv2.GaussianBlur(image, (0, 0), sigma)
                return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
            
            elif filter_name == "edge_enhancement":
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                return cv2.filter2D(image, -1, kernel)
            
            elif filter_name == "histogram_equalization":
                if len(image.shape) == 3:
                    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                else:
                    return cv2.equalizeHist(image)
            
            elif filter_name == "noise_reduction":
                h = parameters.get('h', 10)
                template_window_size = parameters.get('template_window_size', 7)
                search_window_size = parameters.get('search_window_size', 21)
                if len(image.shape) == 3:
                    return cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window_size, search_window_size)
                else:
                    return cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)
            
            # Morphological operations
            elif filter_name == "morphological_opening":
                kernel_size = parameters.get('kernel_size', 5)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            
            elif filter_name == "morphological_closing":
                kernel_size = parameters.get('kernel_size', 5)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            
            # Brightness/Contrast
            elif filter_name == "brightness_contrast":
                alpha = parameters.get('contrast', 1.0)  # Contrast control (1.0-3.0)
                beta = parameters.get('brightness', 0)   # Brightness control (0-100)
                return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            elif filter_name == "dog":
                sigma1 = parameters.get('sigma1', 1.0)
                sigma2 = parameters.get('sigma2', 2.0)
                blur1 = cv2.GaussianBlur(image, (0, 0), sigma1)
                blur2 = cv2.GaussianBlur(image, (0, 0), sigma2)
                return cv2.normalize(blur1 - blur2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            elif filter_name == "gabor":
                frequency = parameters.get('frequency', 0.5)
                theta = parameters.get('theta', 0.0) * np.pi / 180.0
                bandwidth = parameters.get('bandwidth', 1.0)
                kernel = cv2.getGaborKernel((21, 21), bandwidth, theta, 2*np.pi/frequency, 0.5, 0, ktype=cv2.CV_32F)
                return cv2.filter2D(image, cv2.CV_8UC3, kernel)
            
            elif filter_name == "laplacian":
                ksize = parameters.get('ksize', 3)
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_16S, ksize=int(ksize))
                return cv2.convertScaleAbs(laplacian)
            
            elif filter_name == "nlmd":
                h = parameters.get('h', 10)
                template_window_size = parameters.get('template_window_size', 7)
                search_window_size = parameters.get('search_window_size', 21)
                if len(image.shape) == 3:
                    return cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window_size, search_window_size)
                else:
                    return cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)
            
            elif filter_name == "top_hat":
                kernel_size = parameters.get('kernel_size', 5)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(kernel_size), int(kernel_size)))
                return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            
            elif filter_name == "total_variation":
                weight = parameters.get('weight', 1.0)
                try:
                    from skimage.restoration import denoise_tv_chambolle
                    if len(image.shape) == 3:
                        denoised = denoise_tv_chambolle(image, weight=weight, channel_axis=-1)
                    else:
                        denoised = denoise_tv_chambolle(image, weight=weight)
                    return (denoised * 255).astype(np.uint8)
                except ImportError:
                    return cv2.bilateralFilter(image, 9, 75, 75)
            
            elif filter_name == "wavelet":
                wavelet_type = parameters.get('wavelet', 'haar')
                level = parameters.get('level', 1)
                try:
                    import pywt
                    if len(image.shape) == 3:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = image
                    coeffs = pywt.wavedec2(gray.astype(np.float32), wavelet_type, level=int(level))
                    coeffs_thresh = list(coeffs)
                    coeffs_thresh[0] = pywt.threshold(coeffs[0], 0.1, mode='soft')
                    reconstructed = pywt.waverec2(coeffs_thresh, wavelet_type)
                    return cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                except ImportError:
                    return cv2.Laplacian(image, cv2.CV_8U, ksize=3)
            
            elif filter_name == "fft_highpass":
                cutoff_frequency = parameters.get('cutoff_frequency', 0.1)
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                f_transform = np.fft.fft2(gray)
                f_shift = np.fft.fftshift(f_transform)
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                mask = np.ones((rows, cols), np.uint8)
                r = int(cutoff_frequency * min(rows, cols))
                cv2.circle(mask, (ccol, crow), r, 0, -1)
                f_shift = f_shift * mask
                f_ishift = np.fft.ifftshift(f_shift)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            else:
                print(f"Unknown filter: {filter_name}")
                return image
            
        except Exception as e:
            logger.error(f"Error applying filter {filter_name}: {e}")
            return None
    
    # Scoring signal handlers
    def _on_scoring_method_changed(self, method_name):
        """Handle scoring method change."""
        self.current_scoring_method = method_name
        if hasattr(self, 'scoring_operations_manager'):
            self.scoring_operations_manager.current_scoring_method = method_name
    
    def _on_calculate_scores_requested(self, method_name):
        """Handle calculate scores request."""
        try:
            if hasattr(self, 'scoring_operations_manager'):
                self.scoring_operations_manager.current_scoring_method = method_name
                self.scoring_operations_manager.calculate_scores()
        except Exception as e:
            logger.error(f"Error calculating scores: {e}")
    
    def _save_filtered_image(self):
        """Save current filtered image."""
        try:
            if self.current_sem_image is not None:
                from pathlib import Path
                from datetime import datetime
                
                output_dir = Path("Results/Filtered")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"filtered_image_{timestamp}.png"
                filepath = output_dir / filename
                
                cv2.imwrite(str(filepath), self.current_sem_image)
                
                self.status_bar.showMessage(f"Image saved: {filepath}")
                print(f"Filtered image saved: {filepath}")
            else:
                print("No image to save")
        except Exception as e:
            print(f"Error saving image: {e}")
    
    # Error handling
    def setup_error_handling(self):
        """Setup comprehensive error handling."""
        try:
            import sys
            sys.excepthook = self.handle_unhandled_exception
            logger.info("Error handling setup complete")
        except Exception as e:
            logger.error(f"Failed to setup error handling: {e}")
    
    def handle_unhandled_exception(self, exc_type, exc_value, exc_traceback):
        """Handle unhandled exceptions."""
        try:
            import traceback
            error_msg = f"Unhandled exception: {exc_type.__name__}: {exc_value}"
            logger.critical(error_msg)
            logger.critical("Traceback:\n" + "".join(traceback.format_tb(exc_traceback)))
            
            if exc_type != KeyboardInterrupt:
                QMessageBox.critical(self, "Application Error", 
                                   f"An unexpected error occurred:\n\n{exc_value}")
        except Exception as e:
            print(f"Critical error in error handler: {e}")
    
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                logger.info("Application closing")
                event.accept()
            else:
                event.ignore()
                
        except Exception as e:
            logger.error(f"Error during close: {e}")
            event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Image Analysis Tool - Unified")
    app.setOrganizationName("Image Analysis")
    
    window = MainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())