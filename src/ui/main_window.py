"""
Main Window - Unified Image Analysis Application

This is the primary main window that combines all functionality for SEM/GDS
image analysis, alignment, filtering, and scoring operations.

Main Class:
- MainWindow: Primary application window with tabbed interface

Key Features:
- Core SEM/GDS alignment functionality
- Advanced filtering capabilities
- Sequential filtering workflow
- Comprehensive scoring system
- Hybrid alignment with 3-point selection
- File operations and management
- Real-time image processing and preview

Main Methods:
- load_sem_file(): Loads SEM image files
- load_gds_file(): Loads GDS files with structure selection
- run_auto_alignment(): Executes automatic alignment
- save_results(): Saves analysis results
- Various signal handlers for UI interactions

Dependencies:
- Uses: PySide6.QtWidgets, PySide6.QtCore (Qt framework)
- Uses: cv2, numpy (image processing)
- Uses: services/* (core processing services)
- Uses: ui/components/* (UI components)
- Uses: ui/managers/* (operation managers)
- Uses: ui/panels/* (UI panels)

UI Structure:
- Left Panel: Tabbed interface (Alignment, Filtering, Scoring)
- Center: Image viewer with overlay capabilities
- Right Panel: File selector and histogram view
- Menu Bar: File, View, Tools, Help menus
- Status Bar: Operation status and progress

Entry Point:
- main(): Application entry point function
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QMenuBar, QMenu, QFileDialog, QMessageBox, QDockWidget,
                               QApplication, QStatusBar, QSplitter, QPushButton, QComboBox, QLabel,
                               QToolBar, QButtonGroup, QGroupBox, QTextEdit, QTabWidget, QStackedWidget,
                               QDialog, QGridLayout, QColorDialog)
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
from src.services.unified_transformation_service import UnifiedTransformationService

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

# UPDATED IMPORTS: Use the new simplified GDS files
from src.core.gds_display_generator import (
    get_structure_info, 
    generate_gds_display, 
    get_all_structures_info,
    list_available_structures,
    get_structure_definitions
)
from src.core.gds_aligned_generator import (
    generate_aligned_gds, 
    generate_transformed_gds,
    calculate_new_bounds,
    extract_and_render_gds
)

# Import core models (update if structure_definitions was using old files)
try:
    from src.core.models.structure_definitions import get_default_structures
except ImportError:
    # Fallback to gds_display_generator if structure_definitions doesn't exist
    from src.core.gds_display_generator import get_structure_definitions as get_default_structures

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
        self.transform_service = UnifiedTransformationService()
        
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
        
        # Import theme manager
        from src.ui.styles.theme import theme_manager
        self.theme_manager = theme_manager
        
        # Load and apply settings
        self._load_and_apply_settings()
        
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
        self._setup_error_handling()
        
        logger.info("Unified main window initialized successfully")
    
    def _load_and_apply_settings(self):
        """Load application settings and apply them."""
        try:
            from src.ui.settings_dialog import load_app_settings
            settings = load_app_settings()
            
            # Apply GDS color settings to image viewer
            if hasattr(self, 'image_viewer'):
                bg_color = settings.get('gds_bg_color', [0, 0, 0])
                bg_alpha = settings.get('gds_bg_alpha', 0)
                struct_color = settings.get('gds_struct_color', [0, 0, 0])
                struct_alpha = settings.get('gds_struct_alpha', 255)
                
                background_rgba = (bg_color[0], bg_color[1], bg_color[2], bg_alpha)
                structure_rgba = (struct_color[0], struct_color[1], struct_color[2], struct_alpha)
                
                self.image_viewer.set_gds_colors(background_rgba, structure_rgba)
                
                logger.info(f"Applied GDS colors - Background: {background_rgba}, Structure: {structure_rgba}")
            
            # Apply UI theme settings
            self._apply_ui_theme(settings)
            
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
    
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
        """Calculate new GDS bounds based on alignment parameters using the new simplified approach."""
        try:
            # Get structure info using the new simplified function
            struct_info = get_structure_info(structure_id)
            if not struct_info:
                raise ValueError(f"Structure {structure_id} not found")
            
            original_bounds = struct_info['bounds']
            
            # Get transformation parameters
            x_offset_pixels = params.get('x_offset', 0)
            y_offset_pixels = params.get('y_offset', 0)
            scale = params.get('scale', 1.0)
            rotation = params.get('rotation', 0)
            
            # Use the new bounds calculation function
            zoom_percentage = scale * 100  # Convert scale to percentage
            new_bounds = calculate_new_bounds(
                original_bounds, 
                zoom_percentage, 
                x_offset_pixels, 
                y_offset_pixels, 
                struct_info
            )
            
            print(f"DEBUG: Original bounds: {original_bounds}")
            print(f"DEBUG: Params: x={x_offset_pixels}, y={y_offset_pixels}, scale={scale}, rot={rotation}")
            print(f"DEBUG: New bounds: {new_bounds}")
            
            return new_bounds, rotation
            
        except Exception as e:
            logger.error(f"Error calculating transformed bounds: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to original bounds
            try:
                struct_info = get_structure_info(structure_id)
                return struct_info['bounds'] if struct_info else ([0, 0, 1000, 1000], 0), 0
            except:
                return ([0, 0, 1000, 1000], 0)
    
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
            
            # Apply settings after image viewer is created
            self._load_and_apply_settings()
            
            # Right panel
            self._setup_right_panel()
            
            # Set splitter sizes - smaller left panel, larger range
            self.main_splitter.setSizes([200, 1000, 350])
            self.main_splitter.setCollapsible(0, True)
            self.main_splitter.setCollapsible(2, True)
            # Allow much smaller minimum sizes
            self.left_panel_container.setMinimumWidth(50)
            self.right_panel_container.setMinimumWidth(50)
            
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
        
        # Create vertical splitter for resizable sections (filtering modes only)
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_panel_layout.addWidget(right_splitter)
        
        # Histogram section (filtering modes only)
        self.histogram_group = QGroupBox("📊 Image Histogram")
        self.histogram_group.setStyleSheet(self._get_group_style())
        histogram_layout = QVBoxLayout(self.histogram_group)
        
        self.histogram_view = HistogramView()
        histogram_layout.addWidget(self.histogram_view)
        right_splitter.addWidget(self.histogram_group)
        
        # Statistics section (filtering modes only)
        self.stats_group = QGroupBox("📈 Image Statistics")
        self.stats_group.setStyleSheet(self._get_group_style())
        stats_layout = QVBoxLayout(self.stats_group)
        
        self.stats_text = QTextEdit()
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
        right_splitter.addWidget(self.stats_group)
        
        # Processing status section (filtering modes only)
        self.status_group = QGroupBox("⚡ Processing Status")
        self.status_group.setStyleSheet(self._get_group_style())
        status_layout = QVBoxLayout(self.status_group)
        
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
        status_layout.addWidget(self.status_label)
        right_splitter.addWidget(self.status_group)
        
        # Hide these components by default (only show in filtering modes)
        self.histogram_group.hide()
        self.stats_group.hide()
        self.status_group.hide()
        
        # Sequential progress section (for sequential filtering mode)
        progress_group = QGroupBox("🔄 Processing Progress")
        progress_group.setStyleSheet(self._get_group_style())
        progress_layout = QVBoxLayout(progress_group)
        
        self.stage_progress = {}
        stages_layout = QVBoxLayout()
        
        from src.ui.panels.sequential_filtering_panels import ProcessingStage, SequentialFilterConfigManager
        for stage in ProcessingStage:
            stage_config = SequentialFilterConfigManager().get_stage_config(stage)
            
            stage_layout = QHBoxLayout()
            
            indicator = QLabel("○")
            indicator.setStyleSheet("color: #666666; font-size: 12px; font-weight: bold; min-width: 16px;")
            
            label = QLabel(f"{stage_config.icon} {stage_config.display_name}")
            label.setStyleSheet(f"color: {stage_config.color}; font-size: 10px; font-weight: bold;")
            
            status = QLabel("Pending")
            status.setStyleSheet("color: #cccccc; font-size: 9px; font-style: italic;")
            
            stage_layout.addWidget(indicator)
            stage_layout.addWidget(label)
            stage_layout.addStretch()
            stage_layout.addWidget(status)
            
            self.stage_progress[stage] = {'indicator': indicator, 'label': label, 'status': status}
            stages_layout.addLayout(stage_layout)
        
        progress_layout.addLayout(stages_layout)
        right_splitter.addWidget(progress_group)
        self.progress_group = progress_group
        progress_group.hide()  # Hidden by default
        
        # Set initial splitter sizes
        right_splitter.setSizes([200, 140, 80, 120])
        self.right_splitter = right_splitter
        
        self.main_splitter.addWidget(self.right_panel_container)
        
        # Initialize file scanning
        self.file_selector.scan_directories()
        
        # Initialize histogram and stats
        self._initialize_right_panel_displays()
        
        # File scanning completed
    
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
        
        # This will be added to splitter instead
        
        # Create vertical splitter for alignment sections
        alignment_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Add sub-tabs to splitter
        alignment_splitter.addWidget(self.alignment_sub_tabs)
        
        # Action buttons in separate section
        actions_widget = QWidget()
        actions_layout = QVBoxLayout(actions_widget)
        self._setup_alignment_action_buttons(actions_layout)
        alignment_splitter.addWidget(actions_widget)
        
        # Set initial sizes and add to main layout
        alignment_splitter.setSizes([300, 150])
        layout.addWidget(alignment_splitter)
        
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
        instructions.setStyleSheet("padding: 8px; border-radius: 4px; font-size: 12px;")
        layout.addWidget(instructions)
        
        # Point status
        self.points_status_label = QLabel("SEM Points: 0/3  |  GDS Points: 0/3")
        self.points_status_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.points_status_label)
        
        # Ready status
        self.ready_status_label = QLabel("Status: Not Ready")
        self.ready_status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.ready_status_label)
        
        # Point selection mode buttons
        mode_layout = QHBoxLayout()
        
        self.select_sem_btn = QPushButton("Select SEM Points")
        self.select_sem_btn.setCheckable(True)
        self.select_sem_btn.setChecked(True)
        self.select_sem_btn.clicked.connect(self._set_sem_mode)

        mode_layout.addWidget(self.select_sem_btn)
        
        self.select_gds_btn = QPushButton("Select GDS Points")
        self.select_gds_btn.setCheckable(True)
        self.select_gds_btn.clicked.connect(self._set_gds_mode)

        mode_layout.addWidget(self.select_gds_btn)
        
        self.current_point_mode = "sem"  # "sem" or "gds"
        layout.addLayout(mode_layout)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.clear_sem_btn = QPushButton("Clear SEM")
        self.clear_sem_btn.clicked.connect(self._clear_sem_points)
        buttons_layout.addWidget(self.clear_sem_btn)
        
        self.clear_gds_btn = QPushButton("Clear GDS")
        self.clear_gds_btn.clicked.connect(self._clear_gds_points)
        buttons_layout.addWidget(self.clear_gds_btn)
        
        self.clear_points_btn = QPushButton("Clear All")
        self.clear_points_btn.clicked.connect(self._clear_all_points)
        buttons_layout.addWidget(self.clear_points_btn)
        
        layout.addLayout(buttons_layout)
        
        # Calculate button
        calc_layout = QHBoxLayout()
        self.calculate_alignment_btn = QPushButton("Calculate Alignment")
        self.calculate_alignment_btn.setEnabled(False)
        self.calculate_alignment_btn.clicked.connect(self._calculate_alignment)
        calc_layout.addWidget(self.calculate_alignment_btn)
        
        layout.addLayout(calc_layout)
        
        # Show GDS toggle button
        gds_layout = QHBoxLayout()
        self.show_gds_hybrid_btn = QPushButton("Show GDS")
        self.show_gds_hybrid_btn.setCheckable(True)
        self.show_gds_hybrid_btn.clicked.connect(self._on_show_gds_hybrid_clicked)
        gds_layout.addWidget(self.show_gds_hybrid_btn)
        
        layout.addLayout(gds_layout)
        layout.addStretch()
    
    def _setup_advanced_filtering_tab(self):
        """Setup advanced filtering tab."""
        layout = QHBoxLayout(self.advanced_filtering_tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Advanced filtering controls only
        self.advanced_filtering_left_panel = AdvancedFilteringLeftPanel()
        self.advanced_filtering_left_panel.filter_applied.connect(self._on_advanced_filter_applied)
        self.advanced_filtering_left_panel.filter_previewed.connect(self._on_advanced_filter_previewed)
        self.advanced_filtering_left_panel.filter_reset.connect(self._on_advanced_filter_reset)
        self.advanced_filtering_left_panel.save_image_requested.connect(self._save_filtered_image)
        layout.addWidget(self.advanced_filtering_left_panel)
    
    def _setup_sequential_filtering_tab(self):
        """Setup sequential filtering tab."""
        layout = QHBoxLayout(self.sequential_filtering_tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Sequential filtering controls only
        self.sequential_filtering_left_panel = SequentialFilteringLeftPanel()
        self.sequential_filtering_left_panel.stage_preview_requested.connect(self._on_stage_preview_requested)
        self.sequential_filtering_left_panel.stage_apply_requested.connect(self._on_stage_apply_requested)
        self.sequential_filtering_left_panel.stage_reset_requested.connect(self._on_stage_reset_requested)
        layout.addWidget(self.sequential_filtering_left_panel)
    
    def _setup_scoring_tab(self):
        """Setup scoring tab."""
        layout = QVBoxLayout(self.scoring_tab)
        
        # Create vertical splitter for scoring sections
        scoring_splitter = QSplitter(Qt.Orientation.Vertical)
        
        from src.ui.panels.scoring_tab_panel import ScoringTabPanel
        self.scoring_tab_panel = ScoringTabPanel()
        self.scoring_tab_panel.scoring_method_changed.connect(self._on_scoring_method_changed)
        self.scoring_tab_panel.calculate_scores_requested.connect(self._on_calculate_scores_requested)
        
        scoring_splitter.addWidget(self.scoring_tab_panel)
        scoring_splitter.setSizes([400])
        layout.addWidget(scoring_splitter)
    
    def _setup_alignment_action_buttons(self, parent_layout):
        """Setup action buttons for alignment tab."""
        
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
        parent_layout.addStretch()
    
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
        
        tools_menu.addSeparator()
        
        settings_action = QAction("&Settings...", self)
        settings_action.setShortcut(QKeySequence("Ctrl+,"))
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)
        

        
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
            
            # Transform service signals (skip if method doesn't exist)
            if hasattr(self.transform_service, 'transform_applied'):
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
                    # Set both current and original to the loaded image
                    self.current_sem_image = result['cropped_array'].copy()
                    self.original_sem_image = result['cropped_array'].copy()
                    
                    # Set image in image processing service
                    if hasattr(self, 'image_processing_service'):
                        self.image_processing_service.set_image(self.current_sem_image)
                    
                    self.image_viewer.set_sem_image(self.current_sem_image)
                    
                    print(f"✓ SEM image loaded - Original and current reference set")
                    # Update reference status when new image is loaded
                    if hasattr(self, 'advanced_filtering_right_panel') and hasattr(self.advanced_filtering_right_panel, 'update_reference_status'):
                        self.advanced_filtering_right_panel.update_reference_status(is_original=True)
                
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
    
    def show_settings(self):
        """Show settings dialog."""
        try:
            from src.ui.settings_dialog import SettingsDialog, load_app_settings, save_app_settings
            
            current_settings = load_app_settings()
            dialog = SettingsDialog(self, current_settings)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                new_settings = dialog.get_settings()
                save_app_settings(new_settings)
                
                # Apply new settings immediately
                if hasattr(self, 'image_viewer'):
                    bg_color = new_settings.get('gds_bg_color', [0, 0, 0])
                    bg_alpha = new_settings.get('gds_bg_alpha', 0)
                    struct_color = new_settings.get('gds_struct_color', [0, 0, 0])
                    struct_alpha = new_settings.get('gds_struct_alpha', 255)
                    
                    background_rgba = (bg_color[0], bg_color[1], bg_color[2], bg_alpha)
                    structure_rgba = (struct_color[0], struct_color[1], struct_color[2], struct_alpha)
                    
                    self.image_viewer.set_gds_colors(background_rgba, structure_rgba)
                    
                    logger.info(f"Updated GDS colors - Background: {background_rgba}, Structure: {structure_rgba}")
                
                # Apply UI theme settings
                self._apply_ui_theme(new_settings)
                
                QMessageBox.information(self, "Settings", "Settings saved successfully!")
                
        except Exception as e:
            logger.error(f"Error showing settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to show settings: {e}")
    
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
            # Set both current and original to the loaded image
            self.current_sem_image = image_data['cropped_array'].copy()
            self.original_sem_image = image_data['cropped_array'].copy()
            
            # Set image in image processing service
            if hasattr(self, 'image_processing_service'):
                self.image_processing_service.set_image(self.current_sem_image)
            
            print(f"✓ SEM image loaded via signal - Original and current reference set")
    
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
            
            # Process GDS overlay to black and white for scoring mode
            if self.main_tab_widget.currentIndex() == 3:  # Scoring tab
                processed_overlay = self._process_gds_for_scoring(overlay)
                self.current_gds_overlay = processed_overlay
                self.image_viewer.set_gds_overlay(processed_overlay)
            else:
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
            # Update histogram and statistics in main right panel
            self._update_right_panel_displays(self.current_sem_image)
            self._update_status(f"Applied: {filter_name}")
    
    def on_filters_reset(self):
        """Handle filters reset."""
        logger.info("Filters reset")
        if hasattr(self, 'current_sem_image') and self.current_sem_image is not None:
            # Update histogram and statistics in main right panel
            self._update_right_panel_displays(self.current_sem_image)
            self._update_status("Filters Reset")
    
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
        
        # Hide histogram, statistics, and processing status for alignment
        if hasattr(self, 'histogram_group'):
            self.histogram_group.hide()
        if hasattr(self, 'stats_group'):
            self.stats_group.hide()
        if hasattr(self, 'status_group'):
            self.status_group.hide()
        if hasattr(self, 'progress_group'):
            self.progress_group.hide()
    
    def _switch_to_advanced_filtering_display(self):
        """Switch to advanced filtering display mode."""
        if self.current_sem_image is not None:
            self.image_viewer.set_sem_image(self.current_sem_image)
            # Update histogram and statistics in main right panel
            self._update_right_panel_displays(self.current_sem_image)
            self._update_status("Advanced Filtering Mode")
            print(f"✓ Switched to filtering mode - showing current reference image")
        # Hide GDS overlay in filtering mode
        if hasattr(self.image_viewer, 'set_overlay_visible'):
            self.image_viewer.set_overlay_visible(False)
        
        # Show histogram, statistics, and processing status for advanced filtering
        if hasattr(self, 'histogram_group'):
            self.histogram_group.show()
        if hasattr(self, 'stats_group'):
            self.stats_group.show()
        if hasattr(self, 'status_group'):
            self.status_group.show()
        # Hide progress section for advanced mode
        if hasattr(self, 'progress_group'):
            self.progress_group.hide()
    
    def _switch_to_sequential_filtering_display(self):
        """Switch to sequential filtering display mode."""
        if self.sequential_images:
            # Show latest sequential result
            latest_stage = max(self.sequential_images.keys())
            latest_image = self.sequential_images[latest_stage]
            self.image_viewer.set_sem_image(latest_image)
            self._update_right_panel_displays(latest_image)
        elif self.current_sem_image is not None:
            self.image_viewer.set_sem_image(self.current_sem_image)
            self._update_right_panel_displays(self.current_sem_image)
        self._update_status("Sequential Filtering Mode")
        
        # Show histogram, statistics, and processing status for sequential filtering
        if hasattr(self, 'histogram_group'):
            self.histogram_group.show()
        if hasattr(self, 'stats_group'):
            self.stats_group.show()
        if hasattr(self, 'status_group'):
            self.status_group.show()
        # Show progress section for sequential mode
        if hasattr(self, 'progress_group'):
            self.progress_group.show()
    
    def _switch_to_scoring_display(self):
        """Switch to scoring display mode."""
        if self.current_sem_image is not None:
            self.image_viewer.set_sem_image(self.current_sem_image)
        if self.current_gds_overlay is not None:
            # For scoring, process GDS to have white background and black structures
            processed_gds = self._process_gds_for_scoring(self.current_gds_overlay)
            self.image_viewer.set_gds_overlay(processed_gds)
            self.image_viewer.set_overlay_visible(True)
            print(f"Scoring display: GDS overlay visible={self.image_viewer.get_overlay_visible()}")
        
        # Hide histogram, statistics, and processing status for scoring
        if hasattr(self, 'histogram_group'):
            self.histogram_group.hide()
        if hasattr(self, 'stats_group'):
            self.stats_group.hide()
        if hasattr(self, 'status_group'):
            self.status_group.hide()
        if hasattr(self, 'progress_group'):
            self.progress_group.hide()
    
    # Hybrid alignment methods
    def _on_alignment_subtab_changed(self, index):
        """Handle alignment sub-tab change."""
        if index == 1:  # Hybrid tab
            self.image_viewer.set_point_selection_mode(True, self.current_point_mode)
            # Update button states to match current mode
            self._update_point_mode_buttons()
        else:  # Manual tab
            self.image_viewer.set_point_selection_mode(False)
    
    def _update_point_mode_buttons(self):
        """Update point mode button states."""
        if self.current_point_mode == "sem":
            self.select_sem_btn.setChecked(True)
            self.select_gds_btn.setChecked(False)
        else:
            self.select_sem_btn.setChecked(False)
            self.select_gds_btn.setChecked(True)
    
    def _set_sem_mode(self):
        """Set SEM point selection mode."""
        self.current_point_mode = "sem"
        self.select_sem_btn.setChecked(True)
        self.select_gds_btn.setChecked(False)
        # Update image viewer mode
        self.image_viewer.set_point_selection_mode(True, "sem")
    
    def _set_gds_mode(self):
        """Set GDS point selection mode."""
        self.current_point_mode = "gds"
        self.select_sem_btn.setChecked(False)
        self.select_gds_btn.setChecked(True)
        # Update image viewer mode
        self.image_viewer.set_point_selection_mode(True, "gds")
    
    def _clear_sem_points(self):
        """Clear only SEM points."""
        self.sem_points = []
        self.image_viewer.clear_points("sem")
        self._update_hybrid_status()
    
    def _clear_gds_points(self):
        """Clear only GDS points."""
        self.gds_points = []
        self.image_viewer.clear_points("gds")
        self._update_hybrid_status()
    
    def _on_point_selected(self, x, y, point_type):
        """Handle point selection based on current mode."""
        try:
            # Handle point removal (negative coordinates indicate removal)
            if x == -1 and y == -1:
                self._update_hybrid_status()
                return
            
            # Use current mode instead of point_type
            if self.current_point_mode == "sem":
                if len(self.sem_points) < 3:
                    self.sem_points.append((int(x), int(y)))
                    print(f"Added SEM point {len(self.sem_points)}: ({x}, {y})")
            elif self.current_point_mode == "gds":
                if len(self.gds_points) < 3:
                    self.gds_points.append((int(x), int(y)))
                    print(f"Added GDS point {len(self.gds_points)}: ({x}, {y})")
            
            # Update points in image viewer
            self.image_viewer.set_points(self.sem_points, "sem")
            self.image_viewer.set_points(self.gds_points, "gds")
            
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
                self.ready_status_label.setStyleSheet("font-weight: bold;")
                self.calculate_alignment_btn.setEnabled(True)
            else:
                self.ready_status_label.setText("Status: Not Ready")
                self.ready_status_label.setStyleSheet("font-weight: bold;")
                self.calculate_alignment_btn.setEnabled(False)
                
        except Exception as e:
            logger.error(f"Error updating hybrid status: {e}")
    
    def _clear_all_points(self):
        """Clear all selected points."""
        self._clear_sem_points()
        self._clear_gds_points()
        
        # Update image viewer
        self.image_viewer.set_points([], "sem")
        self.image_viewer.set_points([], "gds")
    
    def _calculate_alignment(self):
        """Calculate alignment from selected points."""
        try:
            if len(self.sem_points) == 3 and len(self.gds_points) == 3:
                print(f"Calculating alignment with:")
                print(f"  SEM points: {self.sem_points}")
                print(f"  GDS points: {self.gds_points}")
                
                # Calculate transformation parameters from 3 points
                from src.utils.three_point_alignment import calculate_transformation_parameters, apply_transformation_to_manual_alignment
                
                transform_params = calculate_transformation_parameters(self.sem_points, self.gds_points)
                logger.info(f"Calculated transformation: {transform_params}")
                
                # Apply to manual alignment system
                apply_transformation_to_manual_alignment(self, transform_params)
                
                # Enable save button
                if hasattr(self, 'generate_aligned_gds_btn'):
                    self.generate_aligned_gds_btn.setEnabled(True)
                
                logger.info("3-point alignment calculation completed")
            else:
                QMessageBox.warning(self, "Warning", "Need exactly 3 points on both SEM and GDS images")
                
        except Exception as e:
            logger.error(f"Error calculating alignment: {e}")
            QMessageBox.critical(self, "Error", f"Alignment calculation failed: {e}")
    
    def _on_show_gds_hybrid_clicked(self):
        """Toggle GDS overlay visibility in hybrid mode."""
        if hasattr(self, 'image_viewer'):
            if self.show_gds_hybrid_btn.isChecked():
                self.image_viewer.set_overlay_visible(True)
                self.image_viewer.set_overlay_alpha(0.7)  # 30% transparency
                self.show_gds_hybrid_btn.setText("Hide GDS")
            else:
                self.image_viewer.set_overlay_visible(False)
                self.show_gds_hybrid_btn.setText("Show GDS")
    
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
        """Handle save aligned GDS button click - generate GDS with coordinate transformation using the new simplified functions."""
        try:
            from pathlib import Path
            from datetime import datetime
            import cv2
            
            # Get current alignment parameters
            params = self._get_current_alignment_parameters()
            if not params:
                QMessageBox.warning(self, "Save Error", "No alignment parameters available")
                return
            
            # Create output directory
            output_dir = Path("Results/Aligned/manual")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with SEM and GDS names
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get SEM image name
            sem_name = "sem_image"
            if hasattr(self, 'current_sem_path') and self.current_sem_path:
                sem_name = Path(self.current_sem_path).stem
            
            # Get GDS structure name
            gds_name = "gds_structure"
            if hasattr(self, 'gds_operations_manager') and self.gds_operations_manager.current_structure_name:
                structure_name = self.gds_operations_manager.current_structure_name
                if structure_name.startswith("Structure "):
                    structure_id = int(structure_name.replace("Structure ", ""))
                    gds_name = f"structure_{structure_id}"
                else:
                    gds_name = structure_name.lower().replace(" ", "_")
                    structure_id = 1
            else:
                structure_id = getattr(self, 'current_structure_id', 1)
                gds_name = f"structure_{structure_id}"
            
            print(f"DEBUG: Using structure ID: {structure_id}")
            
            # Calculate new GDS bounds based on alignment parameters
            new_bounds, rotation_angle = self._calculate_transformed_bounds(params, structure_id)
            print(f"DEBUG: Calculated bounds: {new_bounds}, rotation: {rotation_angle}")
            
            # Use the new simplified function to generate aligned GDS
            transform_params = {
                'rotation': rotation_angle,
                'zoom': params.get('scale', 1.0) * 100,  # Convert to percentage
                'move_x': params.get('x_offset', 0),
                'move_y': params.get('y_offset', 0)
            }
            
            # Generate transformed GDS image using the new function
            transformed_gds_image = generate_transformed_gds(
                structure_id, 
                transform_params['rotation'],
                transform_params['zoom'],
                transform_params['move_x'],
                transform_params['move_y'],
                (1024, 666)
            )
            
            print(f"DEBUG: Generated image shape: {transformed_gds_image.shape if transformed_gds_image is not None else 'None'}")
            print(f"DEBUG: Structure ID used: {structure_id}")
            
            if transformed_gds_image is None:
                QMessageBox.warning(self, "Save Error", "Failed to generate transformed GDS image")
                return
            
            # Convert to black structures on white background for saving
            if len(transformed_gds_image.shape) == 3:
                gray = cv2.cvtColor(transformed_gds_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = transformed_gds_image.copy()
            
            # Create RGB image with black structures and white background
            rgb_image = np.full((gray.shape[0], gray.shape[1], 3), 255, dtype=np.uint8)  # White background
            structure_mask = gray < 127  # Black pixels are structures
            rgb_image[structure_mask] = [0, 0, 0]  # Black structures
            
            # Save as PNG with proper naming: sem_name_aligned_gds_name.png
            png_filename = f"{sem_name}_aligned_{gds_name}.png"
            png_path = output_dir / png_filename
            cv2.imwrite(str(png_path), rgb_image)
            
            # Display the aligned GDS in the UI with black structures on transparent background
            rgba_display = np.zeros((gray.shape[0], gray.shape[1], 4), dtype=np.uint8)
            rgba_display[structure_mask] = [0, 0, 0, 255]  # Black opaque structures
            # Background remains transparent (alpha = 0)
            
            self.current_gds_overlay = rgba_display
            self.image_viewer.set_gds_overlay(rgba_display)
            self.image_viewer.set_overlay_visible(True)
            
            # Save transformation parameters
            params_filename = f"{sem_name}_aligned_{gds_name}_params.txt"
            params_path = output_dir / params_filename
            with open(params_path, 'w') as f:
                f.write(f"Alignment Parameters - {timestamp}\n")
                f.write(f"SEM Image: {sem_name}\n")
                f.write(f"GDS Structure: {gds_name}\n")
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
        """Handle advanced filter applied - makes filtered result the new reference."""
        try:
            if not hasattr(self, 'current_sem_image') or self.current_sem_image is None:
                print("❌ ERROR: No SEM image loaded")
                return
            
            # Apply filter to current image (which becomes the new reference)
            filtered_image = self._apply_filter_directly(filter_name, parameters, self.current_sem_image)
            if filtered_image is not None:
                # Update current image to be the new reference for subsequent filters
                self.current_sem_image = filtered_image.copy()
                self.image_viewer.set_sem_image(filtered_image)
                
                # Update main right panel displays
                self._update_right_panel_displays(filtered_image)
                self._update_status(f"✓ Applied {filter_name}")
                
                print(f"✓ Filter applied: {filter_name} - Now using as reference for next filter")
            else:
                print(f"❌ Filter failed: {filter_name}")
                    
        except Exception as e:
            print(f"Error in filter application: {e}")
    
    def _on_advanced_filter_previewed(self, filter_name: str, parameters: dict):
        """Handle advanced filter preview - shows preview without changing reference."""
        try:
            if not hasattr(self, 'current_sem_image') or self.current_sem_image is None:
                print("❌ ERROR: No SEM image loaded for preview")
                return
            
            # Use current image (which may be a previously applied filter result) for preview
            preview_result = self._apply_filter_directly(filter_name, parameters, self.current_sem_image)
            
            if preview_result is not None:
                # Only update display, don't change the reference image
                self.image_viewer.set_sem_image(preview_result)
                
                # Update main right panel displays
                self._update_right_panel_displays(preview_result)
                self._update_status(f"👁️ Preview: {filter_name}")
                
                print(f"✓ Filter preview: {filter_name} (applied to current reference)")
            else:
                print(f"❌ Preview failed: {filter_name}")
                    
        except Exception as e:
            print(f"Error previewing advanced filter: {e}")
    
    def _on_advanced_filter_reset(self):
        """Handle advanced filter reset - restores original image as reference."""
        try:
            if hasattr(self, 'original_sem_image') and self.original_sem_image is not None:
                # Reset current image back to original, making it the new reference
                self.current_sem_image = self.original_sem_image.copy()
                self.image_viewer.set_sem_image(self.current_sem_image)
                
                # Update main right panel displays
                self._update_right_panel_displays(self.current_sem_image)
                self._update_status("✓ Filters reset to original")
                
                print("✓ Reset to original image - Original is now the reference")
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
                    # Update main right panel displays
                    self._update_right_panel_displays(preview_result)
                    self._update_status(f"👁️ Preview Stage {stage_index}: {filter_name}")
        except Exception as e:
            logger.error(f"Error in stage preview: {e}")
    
    def _on_stage_apply_requested(self, stage_index: int, filter_name: str, parameters: dict):
        """Handle sequential stage apply - result becomes new reference for next stage."""
        try:
            input_image = self._get_stage_input_image(stage_index)
            if input_image is not None:
                result_image = self._apply_filter_directly(filter_name, parameters, input_image)
                if result_image is not None:
                    # Store result for this stage
                    self.sequential_images[stage_index] = result_image.copy()
                    # Update display and current reference
                    self.image_viewer.set_sem_image(result_image)
                    self.current_sem_image = result_image.copy()
                    # Update main right panel displays
                    self._update_right_panel_displays(result_image)
                    self._update_status(f"✓ Stage {stage_index} applied: {filter_name}")
                    # Update stage completion status
                    if hasattr(self, 'sequential_filtering_left_panel'):
                        self.sequential_filtering_left_panel.set_stage_completed(stage_index, True)
                    print(f"✓ Stage {stage_index} applied: {filter_name} - Result is reference for next stage")
        except Exception as e:
            logger.error(f"Error in stage apply: {e}")
            if hasattr(self, 'sequential_filtering_left_panel'):
                self.sequential_filtering_left_panel.set_stage_completed(stage_index, False)
    
    def _on_stage_reset_requested(self, stage_index: int):
        """Handle sequential stage reset - reverts to previous stage result as reference."""
        try:
            # Remove this stage and subsequent stages
            stages_to_remove = [i for i in self.sequential_images.keys() if i >= stage_index]
            for stage_idx in stages_to_remove:
                if stage_idx in self.sequential_images:
                    del self.sequential_images[stage_idx]
            
            # Revert to input image for the reset stage (becomes new reference)
            input_image = self._get_stage_input_image(stage_index)
            if input_image is not None:
                self.image_viewer.set_sem_image(input_image)
                self.current_sem_image = input_image.copy()
                # Update main right panel displays
                self._update_right_panel_displays(input_image)
                self._update_status(f"✓ Stage {stage_index} reset")
                print(f"✓ Stage {stage_index} reset - Previous result is now reference")
                
        except Exception as e:
            logger.error(f"Error in stage reset: {e}")
    
    def _get_stage_input_image(self, stage_index: int):
        """Get input image for a sequential stage - uses previous stage result as reference."""
        try:
            if stage_index == 0:
                # First stage uses original image as reference
                if hasattr(self, 'original_sem_image') and self.original_sem_image is not None:
                    return self.original_sem_image
                elif hasattr(self, 'current_sem_image_obj') and self.current_sem_image_obj:
                    return getattr(self.current_sem_image_obj, 'cropped_array', self.current_sem_image)
                return self.current_sem_image
            else:
                # Subsequent stages use previous stage result as reference
                previous_stage = stage_index - 1
                previous_result = self.sequential_images.get(previous_stage)
                if previous_result is not None:
                    return previous_result
                else:
                    # If previous stage not found, fall back to stage 0 input
                    return self._get_stage_input_image(0)
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
                sigma = parameters.get('sigma', 1.0)
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                # Apply Gaussian blur before Canny if sigma > 0
                if sigma > 0:
                    gray = cv2.GaussianBlur(gray, (0, 0), sigma)
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
        """Save current filtered image to appropriate SEM_Filters subfolder."""
        try:
            if self.current_sem_image is not None:
                from pathlib import Path
                from datetime import datetime
                
                # Determine which tab is active
                current_tab = self.main_tab_widget.currentIndex()
                if current_tab == 1:  # Advanced filtering
                    output_dir = Path("Results/SEM_Filters/manual")
                elif current_tab == 2:  # Sequential filtering
                    output_dir = Path("Results/SEM_Filters/auto")
                else:
                    output_dir = Path("Results/SEM_Filters/manual")  # Default
                
                output_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"filtered_image_{timestamp}.png"
                filepath = output_dir / filename
                
                cv2.imwrite(str(filepath), self.current_sem_image)
                
                self.status_bar.showMessage(f"Image saved: {filepath}")
                print(f"✓ Filtered image saved: {filepath}")
            else:
                print("❌ No image to save")
        except Exception as e:
            print(f"Error saving image: {e}")
    
    def _process_gds_for_scoring(self, image):
        """Convert GDS image to white background, black structures for scoring."""
        try:
            import numpy as np
            import cv2
            
            if len(image.shape) == 4:  # RGBA
                # Extract alpha channel to identify structures
                alpha = image[:, :, 3]
                # Create binary image: white background, black structures
                binary_image = np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)
                binary_image[alpha > 0] = 0  # Black where there are structures
                return cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Ensure white background, black structures
            # If image has black background, invert it
            if np.mean(gray) < 127:
                gray = cv2.bitwise_not(gray)
            
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print(f"Error processing GDS for scoring: {e}")
            return image
    
    def _refresh_gds_overlay(self):
        """Refresh GDS overlay with current colors."""
        if hasattr(self, 'gds_operations_manager') and self.gds_operations_manager.current_gds_overlay is not None:
            original_overlay = self.gds_operations_manager.current_gds_overlay
            processed_overlay = self.theme_manager.process_gds_overlay(original_overlay)
            self.current_gds_overlay = processed_overlay
            self.image_viewer.set_gds_overlay(processed_overlay)
    

    
    def _apply_ui_theme(self, settings):
        """Apply UI theme settings to all components."""
        try:
            panel_color = settings.get('ui_panel_color', [43, 43, 43])
            button_color = settings.get('ui_button_color', [70, 130, 180])
            text_color = settings.get('ui_text_color', [255, 255, 255])
            
            panel_rgb = f"rgb({panel_color[0]}, {panel_color[1]}, {panel_color[2]})"
            button_rgb = f"rgb({button_color[0]}, {button_color[1]}, {button_color[2]})"
            text_rgb = f"rgb({text_color[0]}, {text_color[1]}, {text_color[2]})"
            
            # Create comprehensive stylesheet
            stylesheet = f"""
            * {{
                background-color: {panel_rgb};
                color: {text_rgb};
            }}
            QMainWindow, QWidget, QDialog, QFrame, QScrollArea {{
                background-color: {panel_rgb};
                color: {text_rgb};
            }}
            QGroupBox {{
                background-color: {panel_rgb};
                color: {text_rgb};
                border: 1px solid #555;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                color: {text_rgb};
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
            QPushButton {{
                background-color: {button_rgb};
                color: white;
                border: 1px solid #555;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: rgb({min(255, button_color[0]+20)}, {min(255, button_color[1]+20)}, {min(255, button_color[2]+20)});
            }}
            QPushButton:pressed {{
                background-color: rgb({max(0, button_color[0]-20)}, {max(0, button_color[1]-20)}, {max(0, button_color[2]-20)});
            }}
            QPushButton:checked {{
                background-color: rgb({max(0, button_color[0]-10)}, {max(0, button_color[1]-10)}, {max(0, button_color[2]-10)});
            }}
            QLabel {{
                color: {text_rgb};
                background-color: transparent;
            }}
            QTabWidget, QTabWidget::pane {{
                border: 1px solid #555;
                background-color: {panel_rgb};
            }}
            QTabBar::tab {{
                background-color: rgb({max(0, panel_color[0]-10)}, {max(0, panel_color[1]-10)}, {max(0, panel_color[2]-10)});
                color: {text_rgb};
                border: 1px solid #555;
                padding: 8px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {panel_rgb};
                border-bottom-color: {panel_rgb};
            }}
            QComboBox {{
                background-color: {button_rgb};
                color: white;
                border: 1px solid #555;
                padding: 4px;
                border-radius: 3px;
            }}
            QComboBox::drop-down {{
                background-color: {button_rgb};
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {panel_rgb};
                color: {text_rgb};
                border: 1px solid #555;
            }}
            QSpinBox, QDoubleSpinBox, QLineEdit {{
                background-color: rgb({min(255, panel_color[0]+20)}, {min(255, panel_color[1]+20)}, {min(255, panel_color[2]+20)});
                color: {text_rgb};
                border: 1px solid #555;
                padding: 4px;
                border-radius: 3px;
            }}
            QSlider::groove:horizontal {{
                background-color: rgb({max(0, panel_color[0]-20)}, {max(0, panel_color[1]-20)}, {max(0, panel_color[2]-20)});
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background-color: {button_rgb};
                border: 1px solid #555;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QCheckBox {{
                color: {text_rgb};
                background-color: transparent;
            }}
            QCheckBox::indicator {{
                background-color: rgb({min(255, panel_color[0]+20)}, {min(255, panel_color[1]+20)}, {min(255, panel_color[2]+20)});
                border: 1px solid #555;
                border-radius: 2px;
            }}
            QCheckBox::indicator:checked {{
                background-color: {button_rgb};
            }}
            QTextEdit, QPlainTextEdit {{
                background-color: rgb({min(255, panel_color[0]+10)}, {min(255, panel_color[1]+10)}, {min(255, panel_color[2]+10)});
                color: {text_rgb};
                border: 1px solid #555;
                border-radius: 3px;
            }}
            QScrollBar:vertical {{
                background-color: rgb({max(0, panel_color[0]-10)}, {max(0, panel_color[1]-10)}, {max(0, panel_color[2]-10)});
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {button_rgb};
                border-radius: 6px;
                min-height: 20px;
            }}
            QSplitter::handle {{
                background-color: rgb({max(0, panel_color[0]-20)}, {max(0, panel_color[1]-20)}, {max(0, panel_color[2]-20)});
            }}
            """
            
            self.setStyleSheet(stylesheet)
            
            # Apply to all child widgets recursively
            for widget in self.findChildren(QWidget):
                widget.setStyleSheet("")
                widget.update()
            logger.info(f"Applied UI theme - Panel: {panel_rgb}, Button: {button_rgb}, Text: {text_rgb}")
            
        except Exception as e:
            logger.error(f"Error applying UI theme: {e}")
    
    def _get_group_style(self):
        """Get group box style for right panel."""
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
    
    def _initialize_right_panel_displays(self):
        """Initialize right panel histogram and statistics."""
        try:
            if hasattr(self, 'current_sem_image') and self.current_sem_image is not None:
                self._update_right_panel_displays(self.current_sem_image)
        except Exception as e:
            logger.error(f"Error initializing right panel displays: {e}")
    
    def _update_right_panel_displays(self, image_data):
        """Update histogram and statistics in right panel."""
        try:
            if hasattr(self, 'histogram_view') and image_data is not None:
                self.histogram_view.update_histogram(image_data)
                self._update_statistics(image_data)
        except Exception as e:
            logger.error(f"Error updating right panel displays: {e}")
    
    def _update_statistics(self, image_data):
        """Update image statistics in right panel."""
        try:
            if not hasattr(self, 'stats_text'):
                return
                
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
            from datetime import datetime
            stats_text += f"✓ Updated: {datetime.now().strftime('%H:%M:%S')}"
            
            self.stats_text.setText(stats_text)
            
        except Exception as e:
            if hasattr(self, 'stats_text'):
                self.stats_text.setText(f"❌ Error calculating stats:\n{e}")
    
    def _update_status(self, message: str):
        """Update processing status in right panel."""
        try:
            if hasattr(self, 'status_label'):
                self.status_label.setText(message)
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
    # Error handling
    def _setup_error_handling(self):
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


class ColorPaletteDialog(QDialog):
    """Dialog for selecting colors from a 16-color palette."""
    
    def __init__(self, title, current_color, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.selected_color = current_color
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Color palette (16 colors)
        colors_group = QGroupBox("Select Color")
        colors_layout = QGridLayout(colors_group)
        
        # 16 color palette
        self.colors = [
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
            (255, 128, 0),   # Orange
            (128, 0, 255),   # Purple
            (255, 192, 203), # Pink
            (0, 128, 0),     # Dark Green
            (128, 128, 128), # Gray
            (255, 255, 255), # White
            (0, 0, 0),       # Black
            (139, 69, 19),   # Brown
            (255, 20, 147),  # Deep Pink
            (70, 130, 180)   # Steel Blue
        ]
        
        for i, color in enumerate(self.colors):
            btn = QPushButton()
            btn.setFixedSize(40, 40)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgb({color[0]}, {color[1]}, {color[2]});
                    border: 2px solid #333;
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    border: 3px solid #fff;
                }}
            """)
            btn.clicked.connect(lambda checked, c=color: self.set_color(c))
            colors_layout.addWidget(btn, i // 4, i % 4)
        
        layout.addWidget(colors_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        buttons_layout.addWidget(ok_btn)
        buttons_layout.addWidget(cancel_btn)
        layout.addLayout(buttons_layout)
    
    def set_color(self, color):
        self.selected_color = color
    
    def get_selected_color(self):
        return self.selected_color


class BackgroundThemeDialog(QDialog):
    """Dialog for selecting background theme."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Background Theme")
        self.setModal(True)
        self.selected_theme = "dark"
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        theme_group = QGroupBox("Select Background Theme")
        theme_layout = QVBoxLayout(theme_group)
        
        # Theme options
        self.light_btn = QPushButton("Light")
        self.light_btn.setFixedHeight(50)
        self.light_btn.setStyleSheet("background-color: #f0f0f0; color: black; font-size: 14px;")
        self.light_btn.clicked.connect(lambda: self.set_theme("light"))
        theme_layout.addWidget(self.light_btn)
        
        self.dark_btn = QPushButton("Dark (Current)")
        self.dark_btn.setFixedHeight(50)
        self.dark_btn.setStyleSheet("background-color: #2b2b2b; color: white; font-size: 14px;")
        self.dark_btn.clicked.connect(lambda: self.set_theme("dark"))
        theme_layout.addWidget(self.dark_btn)
        
        self.ultra_dark_btn = QPushButton("Ultra Dark")
        self.ultra_dark_btn.setFixedHeight(50)
        self.ultra_dark_btn.setStyleSheet("background-color: #000000; color: white; font-size: 14px;")
        self.ultra_dark_btn.clicked.connect(lambda: self.set_theme("ultra_dark"))
        theme_layout.addWidget(self.ultra_dark_btn)
        
        layout.addWidget(theme_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        buttons_layout.addWidget(ok_btn)
        buttons_layout.addWidget(cancel_btn)
        layout.addLayout(buttons_layout)
    
    def set_theme(self, theme):
        self.selected_theme = theme
    
    def get_selected_theme(self):
        return self.selected_theme


class TextColorDialog(QDialog):
    """Dialog for selecting text color."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Text Color")
        self.setModal(True)
        self.selected_color = "white"
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        color_group = QGroupBox("Select Text Color")
        color_layout = QVBoxLayout(color_group)
        
        # Color options
        self.white_btn = QPushButton("White")
        self.white_btn.setFixedHeight(50)
        self.white_btn.setStyleSheet("background-color: white; color: black; font-size: 14px;")
        self.white_btn.clicked.connect(lambda: self.set_color("white"))
        color_layout.addWidget(self.white_btn)
        
        self.black_btn = QPushButton("Black")
        self.black_btn.setFixedHeight(50)
        self.black_btn.setStyleSheet("background-color: black; color: white; font-size: 14px;")
        self.black_btn.clicked.connect(lambda: self.set_color("black"))
        color_layout.addWidget(self.black_btn)
        
        layout.addWidget(color_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        buttons_layout.addWidget(ok_btn)
        buttons_layout.addWidget(cancel_btn)
        layout.addLayout(buttons_layout)
    
    def set_color(self, color):
        self.selected_color = color
    
    def get_selected_color(self):
        return self.selected_color


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Image Analysis Tool - Unified")
    app.setOrganizationName("Image Analysis")
    
    # Create necessary directories
    from pathlib import Path
    Path("Results/Scoring").mkdir(parents=True, exist_ok=True)
    Path("Results/Aligned/manual").mkdir(parents=True, exist_ok=True)
    Path("Results/SEM_Filters/manual").mkdir(parents=True, exist_ok=True)
    Path("Results/SEM_Filters/auto").mkdir(parents=True, exist_ok=True)
    
    window = MainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())