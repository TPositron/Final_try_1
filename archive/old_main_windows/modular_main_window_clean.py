"""
Core Main Window - Combined Version with Both Filtering Types
File: src/main_window.py

Clean, focused main window that coordinates the different manager modules.
Now includes BOTH Advanced Filtering AND Phase 3 Sequential Filtering as separate tabs.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QMenuBar, QMenu, QFileDialog, QMessageBox, QDockWidget,
                               QApplication, QStatusBar, QSplitter, QPushButton, QComboBox, QLabel,
                               QToolBar, QButtonGroup, QGroupBox, QTextEdit, QTabWidget)
from PySide6.QtCore import Qt, QTimer, Signal, QPoint
from PySide6.QtGui import QAction, QIcon

# Import UI components and services
from src.ui.components.image_viewer import ImageViewer
from src.ui.components.file_selector import FileSelector
from src.ui.components.histogram_view import HistogramView
from src.ui.view_manager import ViewManager, ViewMode
from src.ui.base_panels import ViewPanelManager
from src.services.simple_file_service import FileService
from src.services.simple_image_processing_service import ImageProcessingService

# Import our modular managers
from src.ui.managers.file_operations_manager import FileOperationsManager
from src.ui.managers.gds_operations_manager import GDSOperationsManager
from src.ui.managers.image_processing_manager import ImageProcessingManager
from src.ui.managers.alignment_operations_manager import AlignmentOperationsManager
from src.ui.managers.scoring_operations_manager import ScoringOperationsManager

# Import alignment components
from src.ui.panels.alignment_left_panel import ManualAlignmentTab, ThreePointAlignmentTab

# IMPORT BOTH FILTERING PANEL TYPES
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

# Configuration
DEFAULT_GDS_FILE = "Institute_Project_GDS1.gds"


class MainWindow(QMainWindow):
    """
    Main Window for the Image Analysis Tool - Combined Version.
    
    This version includes BOTH Advanced Filtering AND Phase 3 Sequential Filtering
    as separate tabs for maximum flexibility.
    """
    
    def __init__(self):
        """Initialize the main window and all manager modules."""
        print("MainWindow constructor called")
        super().__init__()
        self.setWindowTitle("Image Analysis - SEM/GDS Alignment Tool (Combined)")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize core services
        self.file_service = FileService()
        self.image_processing_service = ImageProcessingService()
        
        # Initialize core application state
        self.current_sem_image = None
        self.current_sem_image_obj = None
        self.current_sem_path = None
        self.current_gds_overlay = None
        self.current_alignment_result = None
        self.current_scoring_results = {}
        self.current_scoring_method = "SSIM"
        
        # Phase 3: Initialize sequential processing state
        self.sequential_images = {}  # Store intermediate results
        self.current_processing_stage = None
        
        # Initialize manager modules
        self._initialize_managers()
        
        # Setup UI
        self._setup_ui()
        
        # Connect signals between managers
        self._connect_manager_signals()
        
        # Initialize view system
        self._initialize_view_system()
        
        print("✓ MainWindow initialization completed")
    
    def _initialize_managers(self):
        """Initialize all manager modules."""
        try:
            print("Initializing manager modules...")
            
            # Initialize managers
            self.file_operations_manager = FileOperationsManager(self)
            self.gds_operations_manager = GDSOperationsManager(self)
            self.image_processing_manager = ImageProcessingManager(self)
            self.alignment_operations_manager = AlignmentOperationsManager(self)
            self.scoring_operations_manager = ScoringOperationsManager(self)
            
            print("✓ All manager modules initialized")
            
        except Exception as e:
            print(f"Error initializing managers: {e}")
            raise
    
    def _setup_ui(self):
        """Setup the main UI layout and components."""
        try:
            print("Setting up UI...")
            
            # Create central widget and layout
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QHBoxLayout(central_widget)
            
            # Create main splitter
            self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
            main_layout.addWidget(self.main_splitter)
            
            # Setup left panel container with tabs
            self.left_panel_container = QWidget()
            self.left_panel_layout = QVBoxLayout(self.left_panel_container)
            self.left_panel_layout.setContentsMargins(0, 0, 0, 0)
            
            # Create main tab widget for ALL tabs
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
                QWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
            """)
            
            # Create ALL tab content areas
            self.alignment_tab = QWidget()
            self.advanced_filtering_tab = QWidget()  # Original Advanced Filtering
            self.sequential_filtering_tab = QWidget()  # New Sequential Filtering
            self.scoring_tab = QWidget()
            
            # Setup all tab contents
            self._setup_alignment_tab_content()
            self._setup_advanced_filtering_tab_content()  # Original
            self._setup_sequential_filtering_tab_content()  # New
            self._setup_scoring_tab_content()
            
            # Add ALL tabs
            self.main_tab_widget.addTab(self.alignment_tab, "Alignment")
            self.main_tab_widget.addTab(self.advanced_filtering_tab, "Advanced Filtering") 
            self.main_tab_widget.addTab(self.sequential_filtering_tab, "Sequential Filtering")
            self.main_tab_widget.addTab(self.scoring_tab, "Scoring")
            
            # Add tab widget to left panel
            self.left_panel_layout.addWidget(self.main_tab_widget)
            
            # Connect tab changes to view switching
            self.main_tab_widget.currentChanged.connect(self._on_tab_changed)
            
            self.main_splitter.addWidget(self.left_panel_container)
            
            # Setup central image viewer
            self.image_viewer = ImageViewer()
            
            # Connect point selection signal for hybrid alignment
            self.image_viewer.point_selected.connect(self._on_point_selected)
            
            self.main_splitter.addWidget(self.image_viewer)
            
            # Setup right panel container
            self.right_panel_container = QWidget()
            self.right_panel_layout = QVBoxLayout(self.right_panel_container)
            
            # Setup file selection panel
            self._setup_file_selection_panel()
            
            # Add histogram display
            self._setup_histogram_panel()
            
            # Common controls
            self._setup_common_controls()
            
            # Add view-specific content area
            self.view_specific_widget = QWidget()
            self.view_specific_layout = QVBoxLayout(self.view_specific_widget)
            self.right_panel_layout.addWidget(self.view_specific_widget)
            
            self.main_splitter.addWidget(self.right_panel_container)
            
            # Set initial splitter sizes
            self.main_splitter.setSizes([400, 1000, 350]) 
            
            # Setup menus and status bar
            self._setup_menu()
            self._setup_status_bar()
            
            print("✓ UI setup completed")
            
        except Exception as e:
            print(f"Error setting up UI: {e}")
            raise
    
    def _setup_file_selection_panel(self):
        """Setup file selection panel using FileSelector component."""
        try:
            # Create FileSelector component
            self.file_selector = FileSelector()
            
            # Connect FileSelector signals
            self.file_selector.sem_file_selected.connect(self.load_sem_image_from_path)
            self.file_selector.gds_file_loaded.connect(self.load_gds_file_from_path)
            self.file_selector.gds_structure_selected.connect(self.handle_structure_selection)
            
            # Add FileSelector to right panel
            self.right_panel_layout.addWidget(self.file_selector)
            
            # Initialize file scanning
            self.file_selector.scan_directories()
            
            print("✓ File selection panel setup completed")
            
        except Exception as e:
            print(f"Error setting up file selection panel: {e}")
    
    def load_sem_image_from_path(self, file_path: str):
        """Load SEM image from file path."""
        try:
            self.file_operations_manager.load_sem_image_from_path(file_path)
            print(f"SEM image loaded: {Path(file_path).name}")
        except Exception as e:
            print(f"Error loading SEM image: {e}")
    
    def load_gds_file_from_path(self, file_path: str):
        """Load GDS file from file path."""
        try:
            self.gds_operations_manager.load_gds_file_from_path(file_path)
            print(f"GDS file loaded: {Path(file_path).name}")
        except Exception as e:
            print(f"Error loading GDS file: {e}")
    
    def handle_structure_selection(self, gds_file_path: str, structure_id: int):
        """Handle structure selection from FileSelector."""
        try:
            self.gds_operations_manager.select_structure_by_id(structure_id)
            print(f"Structure {structure_id} selected from {Path(gds_file_path).name}")
        except Exception as e:
            print(f"Error handling structure selection: {e}")
    
    def _setup_histogram_panel(self):
        """Setup histogram display panel."""
        try:
            # Create histogram view component
            self.histogram_view = HistogramView()
            
            # Set appropriate size constraints
            self.histogram_view.setMaximumHeight(200)
            self.histogram_view.setMinimumHeight(150)
            
            # Add histogram view to right panel
            self.right_panel_layout.addWidget(self.histogram_view)
            
            print("✓ Histogram panel setup completed")
            
        except Exception as e:
            print(f"Error setting up histogram panel: {e}")
    
    def _setup_common_controls(self):
        """Setup common controls in the right panel."""
        try:
            # Additional common controls can be added here
            pass
            
        except Exception as e:
            print(f"Error setting up common controls: {e}")
    
    def _setup_menu(self):
        """Setup the main menu bar."""
        try:
            menubar = self.menuBar()
            
            # File menu
            file_menu = menubar.addMenu("File")
            
            load_sem_action = QAction("Load SEM Image", self)
            load_sem_action.triggered.connect(self.load_sem_image)
            file_menu.addAction(load_sem_action)
            
            load_gds_action = QAction("Load GDS File", self)
            load_gds_action.triggered.connect(self.load_gds_file)
            file_menu.addAction(load_gds_action)
            
            file_menu.addSeparator()
            
            save_results_action = QAction("Save Results", self)
            save_results_action.triggered.connect(self.save_results)
            file_menu.addAction(save_results_action)
            
            # Export actions for both filtering types
            export_workflow_action = QAction("Export Sequential Workflow", self)
            export_workflow_action.triggered.connect(self.export_sequential_workflow)
            file_menu.addAction(export_workflow_action)
            
            file_menu.addSeparator()
            
            exit_action = QAction("Exit", self)
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)
            
            # View menu
            view_menu = menubar.addMenu("View")
            
            reset_view_action = QAction("Reset View", self)
            reset_view_action.triggered.connect(self.image_viewer.reset_view)
            view_menu.addAction(reset_view_action)
            
            # Tools menu
            tools_menu = menubar.addMenu("Tools")
            
            auto_align_action = QAction("Auto Align", self)
            auto_align_action.triggered.connect(self.auto_align)
            tools_menu.addAction(auto_align_action)
            
            reset_alignment_action = QAction("Reset Alignment", self)
            reset_alignment_action.triggered.connect(self.reset_alignment)
            tools_menu.addAction(reset_alignment_action)
            
            calculate_scores_action = QAction("Calculate Scores", self)
            calculate_scores_action.triggered.connect(self.calculate_scores)
            tools_menu.addAction(calculate_scores_action)
            
        except Exception as e:
            print(f"Error setting up menu: {e}")
    
    def _setup_status_bar(self):
        """Setup the status bar."""
        try:
            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)
            self.status_bar.showMessage("Ready - Combined Filtering")
            
        except Exception as e:
            print(f"Error setting up status bar: {e}")
    
    def _connect_manager_signals(self):
        """Connect signals between different managers."""
        try:
            print("Connecting manager signals...")
            
            # File operations signals
            self.file_operations_manager.sem_image_loaded.connect(self.on_sem_image_loaded)
            self.file_operations_manager.file_operation_error.connect(self.on_file_error)
            
            # GDS operations signals
            self.gds_operations_manager.gds_file_loaded.connect(self.on_gds_file_loaded)
            self.gds_operations_manager.structure_selected.connect(self.on_structure_selected)
            self.gds_operations_manager.structure_combo_populated.connect(self.on_structure_combo_populated)
            
            # Image processing signals
            self.image_processing_manager.filter_applied.connect(self.on_filter_applied)
            self.image_processing_manager.filters_reset.connect(self.on_filters_reset)
            
            # Alignment operations signals
            self.alignment_operations_manager.alignment_completed.connect(self.on_alignment_completed)
            self.alignment_operations_manager.alignment_reset.connect(self.on_alignment_reset)
            
            # Scoring operations signals
            self.scoring_operations_manager.scores_calculated.connect(self.on_scores_calculated)
            
            # Auto-load default GDS file
            QTimer.singleShot(500, self.gds_operations_manager._auto_load_default_gds)
            
            print("✓ Manager signals connected")
            
        except Exception as e:
            print(f"Error connecting manager signals: {e}")
    
    def _initialize_view_system(self):
        """Initialize the view system and panels."""
        try:
            # Initialize view manager
            self.view_manager = ViewManager(self)
            
            # Initialize panel manager
            self.panel_manager = ViewPanelManager(self)
            
            # Connect view manager signals
            self.view_manager.view_changed.connect(self._on_view_changed)
            self.view_manager.view_data_updated.connect(self._on_view_data_updated)
            
            # Populate structure combo and auto-load default GDS file
            self.gds_operations_manager.populate_structure_combo()
            self.gds_operations_manager._auto_load_default_gds()
            
            # Initialize view manager with current state
            self._sync_view_manager_data()
            
            # Update panel availability
            self._update_panel_availability()
            
        except Exception as e:
            print(f"Error initializing view system: {e}")
    
    def _on_view_changed(self, old_view, new_view):
        """Handle view manager view change signal."""
        try:
            print(f"View changed from {old_view} to {new_view}")
        except Exception as e:
            print(f"Error handling view change: {e}")
    
    def _on_view_data_updated(self, view_mode, data):
        """Handle view manager data update signal."""
        try:
            print(f"View data updated for {view_mode}: {list(data.keys())}")
        except Exception as e:
            print(f"Error handling view data update: {e}")
    
    def _setup_alignment_tab_content(self):
        """Setup the alignment tab content with Manual and Hybrid sub-tabs."""
        try:
            # Create layout for alignment tab
            alignment_layout = QVBoxLayout(self.alignment_tab)
            alignment_layout.setContentsMargins(10, 10, 10, 10)
            
            # Create sub-tab widget for alignment modes
            self.alignment_sub_tabs = QTabWidget()
            self.alignment_sub_tabs.setStyleSheet("""
                QTabWidget::pane {
                    border: 1px solid #555555;
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QTabBar::tab {
                    background-color: #3c3c3c;
                    border: 1px solid #555555;
                    padding: 6px 10px;
                    margin-right: 1px;
                    color: #ffffff;
                    border-top-left-radius: 3px;
                    border-top-right-radius: 3px;
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
            
            # Create Manual alignment tab
            self.manual_alignment_tab = QWidget()
            self._setup_manual_alignment_content()
            self.alignment_sub_tabs.addTab(self.manual_alignment_tab, "Manual")
            
            # Create Hybrid alignment tab
            self.hybrid_alignment_tab = QWidget()
            self._setup_hybrid_alignment_content()
            self.alignment_sub_tabs.addTab(self.hybrid_alignment_tab, "Hybrid")
            
            # Add sub-tabs to main alignment tab
            alignment_layout.addWidget(self.alignment_sub_tabs)
            
            # Connect alignment sub-tab changes
            self.alignment_sub_tabs.currentChanged.connect(self._on_alignment_subtab_changed)
            
            # Add action buttons
            self._setup_alignment_action_buttons(alignment_layout)
            
            print("✓ Alignment tab content setup completed")
            
        except Exception as e:
            print(f"Error setting up alignment tab content: {e}")

    def _setup_manual_alignment_content(self):
        """Setup the manual alignment tab content."""
        try:
            # Create layout for manual alignment tab
            manual_layout = QVBoxLayout(self.manual_alignment_tab)
            manual_layout.setContentsMargins(5, 5, 5, 5)
            
            # Create and add the ManualAlignmentTab component
            self.manual_alignment_controls = ManualAlignmentTab()
            
            # Connect signals
            self.manual_alignment_controls.alignment_changed.connect(
                self.alignment_operations_manager.apply_manual_transformation
            )
            self.manual_alignment_controls.reset_requested.connect(
                self.alignment_operations_manager.reset_alignment
            )
            
            # Add to layout
            manual_layout.addWidget(self.manual_alignment_controls)
            
            print("✓ Manual alignment tab content setup completed")
            
        except Exception as e:
            print(f"Error setting up manual alignment content: {e}")

    def _setup_hybrid_alignment_content(self):
        """Setup the hybrid alignment tab content with SEM/GDS sub-tabs."""
        try:
            # Create layout for hybrid alignment tab
            hybrid_layout = QVBoxLayout(self.hybrid_alignment_tab)
            hybrid_layout.setContentsMargins(5, 5, 5, 5)
            
            # Create sub-tab widget for SEM/GDS point selection
            self.hybrid_sub_tabs = QTabWidget()
            self.hybrid_sub_tabs.setStyleSheet("""
                QTabWidget::pane {
                    border: 1px solid #666666;
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QTabBar::tab {
                    background-color: #444444;
                    border: 1px solid #666666;
                    padding: 4px 8px;
                    margin-right: 1px;
                    color: #ffffff;
                    border-top-left-radius: 2px;
                    border-top-right-radius: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #2b2b2b;
                    border-bottom-color: #2b2b2b;
                    color: #ffffff;
                }
                QTabBar::tab:hover {
                    background-color: #555555;
                    color: #ffffff;
                }
            """)
            
            # Create SEM Image sub-tab
            self.sem_image_tab = QWidget()
            self._setup_sem_image_subtab()
            self.hybrid_sub_tabs.addTab(self.sem_image_tab, "SEM Image")
            
            # Create GDS sub-tab  
            self.gds_subtab = QWidget()
            self._setup_gds_subtab()
            self.hybrid_sub_tabs.addTab(self.gds_subtab, "GDS")
            
            # Add sub-tabs to hybrid alignment tab
            hybrid_layout.addWidget(self.hybrid_sub_tabs)
            
            # Connect hybrid sub-tab changes
            self.hybrid_sub_tabs.currentChanged.connect(self._on_hybrid_subtab_changed)
            
            # Add status display
            self._setup_hybrid_status_display(hybrid_layout)
            
            # Add control buttons
            self._setup_hybrid_control_buttons(hybrid_layout)
            
            print("✓ Hybrid alignment tab content setup completed")
            
        except Exception as e:
            print(f"Error setting up hybrid alignment content: {e}")

    def _setup_sem_image_subtab(self):
        """Setup the SEM Image sub-tab for point selection."""
        try:
            layout = QVBoxLayout(self.sem_image_tab)
            layout.setContentsMargins(5, 5, 5, 5)
            
            # Instructions
            instructions = QLabel("Click to select up to 3 points on the SEM image for alignment.")
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
            
            # Point list display
            self.sem_points_label = QLabel("Selected Points: 0/3")
            self.sem_points_label.setStyleSheet("font-weight: bold; color: #ffffff;")
            layout.addWidget(self.sem_points_label)
            
            # Point coordinates display
            self.sem_coordinates_label = QLabel("Coordinates: None")
            self.sem_coordinates_label.setStyleSheet("color: #cccccc; font-size: 10px;")
            layout.addWidget(self.sem_coordinates_label)
            
            # Clear points button for SEM
            self.clear_sem_points_btn = QPushButton("Clear SEM Points")
            self.clear_sem_points_btn.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
                QPushButton:pressed {
                    background-color: #a93226;
                }
            """)
            layout.addWidget(self.clear_sem_points_btn)
            
            layout.addStretch()
            
        except Exception as e:
            print(f"Error setting up SEM image sub-tab: {e}")

    def _setup_gds_subtab(self):
        """Setup the GDS sub-tab for point selection."""
        try:
            layout = QVBoxLayout(self.gds_subtab)
            layout.setContentsMargins(5, 5, 5, 5)
            
            # Instructions
            instructions = QLabel("Click to select up to 3 corresponding points on the GDS overlay.")
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
            
            # Point list display
            self.gds_points_label = QLabel("Selected Points: 0/3")
            self.gds_points_label.setStyleSheet("font-weight: bold; color: #ffffff;")
            layout.addWidget(self.gds_points_label)
            
            # Point coordinates display
            self.gds_coordinates_label = QLabel("Coordinates: None")
            self.gds_coordinates_label.setStyleSheet("color: #cccccc; font-size: 10px;")
            layout.addWidget(self.gds_coordinates_label)
            
            # Clear points button for GDS
            self.clear_gds_points_btn = QPushButton("Clear GDS Points")
            self.clear_gds_points_btn.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
                QPushButton:pressed {
                    background-color: #a93226;
                }
            """)
            layout.addWidget(self.clear_gds_points_btn)
            
            layout.addStretch()
            
        except Exception as e:
            print(f"Error setting up GDS sub-tab: {e}")

    def _setup_hybrid_status_display(self, parent_layout):
        """Setup status display for hybrid alignment."""
        try:
            # Status group
            status_group = QGroupBox("Alignment Status")
            status_group.setStyleSheet("""
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
            """)
            
            status_layout = QVBoxLayout(status_group)
            
            # Point count status
            self.points_status_label = QLabel("SEM Points: 0/3  |  GDS Points: 0/3")
            self.points_status_label.setStyleSheet("color: #ffffff; font-size: 12px;")
            status_layout.addWidget(self.points_status_label)
            
            # Ready status
            self.ready_status_label = QLabel("Status: Not Ready")
            self.ready_status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
            status_layout.addWidget(self.ready_status_label)
            
            parent_layout.addWidget(status_group)
            
        except Exception as e:
            print(f"Error setting up hybrid status display: {e}")

    def _setup_hybrid_control_buttons(self, parent_layout):
        """Setup control buttons for hybrid alignment."""
        try:
            buttons_layout = QVBoxLayout()
            
            # Calculate Alignment button
            self.calculate_alignment_btn = QPushButton("Calculate Alignment")
            self.calculate_alignment_btn.setEnabled(False)
            self.calculate_alignment_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:hover:enabled {
                    background-color: #2980b9;
                }
                QPushButton:pressed:enabled {
                    background-color: #21618c;
                }
                QPushButton:disabled {
                    background-color: #7f8c8d;
                    color: #bdc3c7;
                }
            """)
            buttons_layout.addWidget(self.calculate_alignment_btn)
            
            buttons_layout.addSpacing(10)
            parent_layout.addLayout(buttons_layout)
            
            # Initialize point tracking
            self.sem_points = []
            self.gds_points = []
            
            # Connect buttons
            self.clear_sem_points_btn.clicked.connect(self._clear_sem_points)
            self.clear_gds_points_btn.clicked.connect(self._clear_gds_points)
            self.calculate_alignment_btn.clicked.connect(self._calculate_alignment)
            
        except Exception as e:
            print(f"Error setting up hybrid control buttons: {e}")

    def _clear_sem_points(self):
        """Clear SEM points."""
        try:
            self.sem_points = []
            self.image_viewer.clear_points("sem")
            self._update_hybrid_status()
            print("SEM points cleared")
        except Exception as e:
            print(f"Error clearing SEM points: {e}")

    def _clear_gds_points(self):
        """Clear GDS points."""
        try:
            self.gds_points = []
            self.image_viewer.clear_points("gds")
            self._update_hybrid_status()
            print("GDS points cleared")
        except Exception as e:
            print(f"Error clearing GDS points: {e}")

    def _calculate_alignment(self):
        """Calculate alignment from selected points."""
        try:
            if len(self.sem_points) != 3 or len(self.gds_points) != 3:
                print("Error: Need exactly 3 points on both SEM and GDS images")
                return
            
            # Validate points
            is_valid, validation_message = self._validate_points()
            if not is_valid:
                print(f"Point validation failed: {validation_message}")
                return
            
            print(f"Calculating alignment from {len(self.sem_points)} SEM points and {len(self.gds_points)} GDS points")
            
            # Use alignment operations manager for 3-point alignment
            if hasattr(self, 'alignment_operations_manager'):
                self.alignment_operations_manager.manual_align_3_point(self.sem_points, self.gds_points)
                print("✓ 3-point alignment calculation completed")
                
                if hasattr(self, 'status_bar'):
                    self.status_bar.showMessage("3-point alignment calculation completed")
            else:
                print("Error: Alignment operations manager not available")
                
        except Exception as e:
            print(f"Error calculating alignment: {e}")

    def _update_hybrid_status(self):
        """Update hybrid alignment status display."""
        try:
            sem_count = len(self.sem_points)
            gds_count = len(self.gds_points)
            
            # Update point count display
            self.points_status_label.setText(f"SEM Points: {sem_count}/3  |  GDS Points: {gds_count}/3")
            self.sem_points_label.setText(f"Selected Points: {sem_count}/3")
            self.gds_points_label.setText(f"Selected Points: {gds_count}/3")
            
            # Update coordinates display
            if sem_count > 0:
                coords_str = ", ".join([f"({int(p[0])}, {int(p[1])})" for p in self.sem_points])
                self.sem_coordinates_label.setText(f"Coordinates: {coords_str}")
            else:
                self.sem_coordinates_label.setText("Coordinates: None")
                
            if gds_count > 0:
                coords_str = ", ".join([f"({int(p[0])}, {int(p[1])})" for p in self.gds_points])
                self.gds_coordinates_label.setText(f"Coordinates: {coords_str}")
            else:
                self.gds_coordinates_label.setText("Coordinates: None")
            
            # Update ready status
            if sem_count == 3 and gds_count == 3:
                self.ready_status_label.setText("Status: Ready for Alignment")
                self.ready_status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
                self.calculate_alignment_btn.setEnabled(True)
            else:
                self.ready_status_label.setText("Status: Not Ready")
                self.ready_status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
                self.calculate_alignment_btn.setEnabled(False)
                
        except Exception as e:
            print(f"Error updating hybrid status: {e}")

    def _setup_alignment_action_buttons(self, parent_layout):
        """Setup action buttons at the bottom of alignment tab."""
        try:
            # Add spacing before buttons
            parent_layout.addSpacing(20)
            
            # Create action buttons group
            action_group = QGroupBox("Actions")
            action_group.setStyleSheet("""
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
            """)
            
            action_layout = QVBoxLayout(action_group)
            
            # First row: Automatic Alignment and Reset Transformation
            row1_layout = QHBoxLayout()
            
            self.auto_align_btn = QPushButton("Automatic Alignment")
            self.auto_align_btn.setStyleSheet("""
                QPushButton {
                    background-color: #333333;
                    color: white;
                    border: none;
                    padding: 4px 10px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 10px;
                    min-height: 18px;
                }
                QPushButton:hover {
                    background-color: #555555;
                }
                QPushButton:pressed {
                    background-color: #444444;
                }
            """)
            
            self.reset_transformation_btn = QPushButton("Reset Transformation")
            self.reset_transformation_btn.setStyleSheet("""
                QPushButton {
                    background-color: #333333;
                    color: white;
                    border: none;
                    padding: 4px 10px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 10px;
                    min-height: 18px;
                }
                QPushButton:hover {
                    background-color: #555555;
                }
                QPushButton:pressed {
                    background-color: #444444;
                }
            """)
            
            row1_layout.addWidget(self.auto_align_btn)
            row1_layout.addWidget(self.reset_transformation_btn)
            action_layout.addLayout(row1_layout)
            
            # Second row: Generate Aligned GDS
            self.generate_aligned_gds_btn = QPushButton("Generate Aligned GDS")
            self.generate_aligned_gds_btn.setStyleSheet("""
                QPushButton {
                    background-color: #333333;
                    color: white;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 11px;
                    min-height: 22px;
                }
                QPushButton:hover {
                    background-color: #555555;
                }
                QPushButton:pressed {
                    background-color: #444444;
                }
                QPushButton:disabled {
                    background-color: #7f8c8d;
                    color: #bdc3c7;
                }
            """)
            self.generate_aligned_gds_btn.setEnabled(False)  # Initially disabled
            
            action_layout.addWidget(self.generate_aligned_gds_btn)
            
            # Connect button signals
            self.auto_align_btn.clicked.connect(self._on_auto_align_clicked)
            self.reset_transformation_btn.clicked.connect(self._on_reset_transformation_clicked)
            self.generate_aligned_gds_btn.clicked.connect(self._on_generate_aligned_gds_clicked)
            
            # Add action group to parent layout
            parent_layout.addWidget(action_group)
            
            print("✓ Alignment action buttons setup completed")
            
        except Exception as e:
            print(f"Error setting up alignment action buttons: {e}")
    
    # ADVANCED FILTERING TAB (Original from first file)
    def _setup_advanced_filtering_tab_content(self):
        """Setup the advanced filtering tab content with unified panels."""
        try:
            # Create layout for advanced filtering tab
            filtering_layout = QHBoxLayout(self.advanced_filtering_tab)
            filtering_layout.setContentsMargins(5, 5, 5, 5)
            
            # Create splitter for left and right panels
            filtering_splitter = QSplitter(Qt.Orientation.Horizontal)
            
            # LEFT PANEL: Advanced Filtering Controls
            self.advanced_filtering_left_panel = AdvancedFilteringLeftPanel()
            
            # Connect advanced filtering signals
            self.advanced_filtering_left_panel.filter_applied.connect(self._on_advanced_filter_applied)
            self.advanced_filtering_left_panel.filter_previewed.connect(self._on_advanced_filter_previewed)
            self.advanced_filtering_left_panel.filter_reset.connect(self._on_advanced_filter_reset)
            self.advanced_filtering_left_panel.save_image_requested.connect(self._on_advanced_save_image_requested)
            
            # Add left panel to splitter
            filtering_splitter.addWidget(self.advanced_filtering_left_panel)
            
            # RIGHT PANEL: Advanced Info Display
            self.advanced_filtering_right_panel = AdvancedFilteringRightPanel()
            
            # Add right panel to splitter
            filtering_splitter.addWidget(self.advanced_filtering_right_panel)
            
            # Set splitter sizes
            filtering_splitter.setSizes([450, 300])
            
            # Add splitter to main filtering layout
            filtering_layout.addWidget(filtering_splitter)
            
            # Store references for compatibility
            self.filter_panel = self.advanced_filtering_left_panel
            
            print("✓ Advanced Filtering panels setup completed")
            
        except Exception as e:
            print(f"Error setting up advanced filtering panels: {e}")

    # SEQUENTIAL FILTERING TAB (From second file)
    def _setup_sequential_filtering_tab_content(self):
        """Setup the sequential filtering tab content with Phase 3 Sequential Workflow."""
        try:
            # Create layout for sequential filtering tab
            filtering_layout = QHBoxLayout(self.sequential_filtering_tab)
            filtering_layout.setContentsMargins(5, 5, 5, 5)
            
            # Create splitter for left and right panels
            filtering_splitter = QSplitter(Qt.Orientation.Horizontal)
            
            # LEFT PANEL: Sequential Filtering Controls (Phase 3)
            self.sequential_filtering_left_panel = SequentialFilteringLeftPanel()
            
            # Connect sequential filtering signals
            self.sequential_filtering_left_panel.stage_preview_requested.connect(self._on_stage_preview_requested)
            self.sequential_filtering_left_panel.stage_apply_requested.connect(self._on_stage_apply_requested)
            self.sequential_filtering_left_panel.stage_reset_requested.connect(self._on_stage_reset_requested)
            self.sequential_filtering_left_panel.stage_save_requested.connect(self._on_stage_save_requested)
            self.sequential_filtering_left_panel.reset_all_requested.connect(self._on_reset_all_stages_requested)
            
            # Add left panel to splitter
            filtering_splitter.addWidget(self.sequential_filtering_left_panel)
            
            # RIGHT PANEL: Sequential Progress Display (Phase 3)
            self.sequential_filtering_right_panel = SequentialFilteringRightPanel()
            
            # Add right panel to splitter
            filtering_splitter.addWidget(self.sequential_filtering_right_panel)
            
            # Set splitter sizes
            filtering_splitter.setSizes([450, 300])
            
            # Add splitter to main filtering layout
            filtering_layout.addWidget(filtering_splitter)
            
            print("✓ Phase 3 Sequential Filtering panels setup completed")
            
        except Exception as e:
            print(f"Error setting up Phase 3 sequential filtering panels: {e}")

    def _setup_scoring_tab_content(self):
        """Setup the scoring tab content."""
        try:
            # Create layout for scoring tab
            scoring_layout = QVBoxLayout(self.scoring_tab)
            scoring_layout.setContentsMargins(0, 0, 0, 0)
            
            # Import and create the ScoringTabPanel component
            from src.ui.panels.scoring_tab_panel import ScoringTabPanel
            
            # Create scoring tab panel
            self.scoring_tab_panel = ScoringTabPanel()
            
            # Connect scoring tab panel signals
            self.scoring_tab_panel.scoring_method_changed.connect(self._on_scoring_method_changed)
            self.scoring_tab_panel.calculate_scores_requested.connect(self._on_calculate_scores_requested)
            
            # Add the scoring tab panel to the scoring tab
            scoring_layout.addWidget(self.scoring_tab_panel)
            
            print("✓ Scoring tab content setup completed")
            
        except Exception as e:
            print(f"Error setting up scoring tab content: {e}")

    def _on_scoring_method_changed(self, method_name):
        """Handle scoring method change from scoring tab."""
        try:
            print(f"Scoring method changed to: {method_name}")
            
            # Update current scoring method
            self.current_scoring_method = method_name
            
            # Update scoring operations manager if available
            if hasattr(self, 'scoring_operations_manager'):
                self.scoring_operations_manager.current_scoring_method = method_name
            
            # Update status
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f"Selected scoring method: {method_name}")
            
        except Exception as e:
            print(f"Error handling scoring method change: {e}")

    def _on_calculate_scores_requested(self, method_name):
        """Handle calculate scores request from scoring tab."""
        try:
            print(f"Calculate scores requested for method: {method_name}")
            
            # Use existing scoring operations manager to calculate scores
            if hasattr(self, 'scoring_operations_manager'):
                self.scoring_operations_manager.current_scoring_method = method_name
                self.scoring_operations_manager.calculate_scores()
            else:
                print("Warning: Scoring operations manager not available")
                
                if hasattr(self, 'calculate_scores'):
                    self.calculate_scores()
                else:
                    print("Error: No scoring calculation method available")
                    
                    if hasattr(self, 'status_bar'):
                        self.status_bar.showMessage("Error: No scoring system available")
            
        except Exception as e:
            print(f"Error handling calculate scores request: {e}")
            
            # Re-enable the calculate button on error
            if hasattr(self, 'scoring_tab_panel'):
                self.scoring_tab_panel.calculate_button.setEnabled(True)
                self.scoring_tab_panel.status_label.setText("Error occurred during calculation")

    # ADVANCED FILTERING SIGNAL HANDLERS (from first file)
    def _on_advanced_filter_applied(self, filter_name: str, parameters: dict):
        """Enhanced handler with better debugging for advanced filtering."""
        try:
            print(f"🔧 FILTER APPLICATION DEBUG:")
            print(f"   Filter Name: '{filter_name}'")
            print(f"   Parameters: {parameters}")
            
            # Check if image processing manager exists
            if not hasattr(self, 'image_processing_manager'):
                print(f"❌ ERROR: image_processing_manager not found")
                return
                
            # Check if service exists
            if not hasattr(self.image_processing_manager, 'image_processing_service'):
                print(f"❌ ERROR: image_processing_service not found")
                return
                
            # Check available filters in service
            service = self.image_processing_manager.image_processing_service
            available_filters = service.get_available_filters()
            print(f"   Available filters in service: {available_filters}")
            
            # Check if the filter exists
            if filter_name not in available_filters:
                print(f"❌ ERROR: Filter '{filter_name}' not found in service!")
                print(f"   Did you mean one of: {available_filters}")
                return
                
            # Show status in right panel with status type
            if hasattr(self, 'advanced_filtering_right_panel'):
                self.advanced_filtering_right_panel.show_status(f"Applying {filter_name}...", "processing")
            
            # Try the primary method
            try:
                print(f"   Trying: image_processing_manager.on_filter_applied()")
                self.image_processing_manager.on_filter_applied(filter_name, parameters)
                print(f"✅ SUCCESS: Filter applied via on_filter_applied()")
            except Exception as e:
                print(f"❌ ERROR in on_filter_applied(): {e}")
                
                # Try alternative method
                try:
                    print(f"   Trying: image_processing_service.apply_filter()")
                    result = service.apply_filter(filter_name, parameters)
                    if result is not None:
                        # Update display manually
                        self.current_sem_image = result
                        self.image_viewer.set_sem_image(result)
                        print(f"✅ SUCCESS: Filter applied via direct service call")
                    else:
                        print(f"❌ ERROR: Service returned None")
                except Exception as e2:
                    print(f"❌ ERROR in direct service call: {e2}")
                    raise e2
            
            # Success feedback
            if hasattr(self, 'advanced_filtering_right_panel'):
                self.advanced_filtering_right_panel.show_status(f"✓ Applied {filter_name}", "success")
            
            # Update status bar
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f"Applied advanced filter: {filter_name}", 3000)
                    
        except Exception as e:
            print(f"💥 FATAL ERROR in filter application: {e}")
            import traceback
            traceback.print_exc()
            
            if hasattr(self, 'advanced_filtering_right_panel'):
                self.advanced_filtering_right_panel.show_status(f"❌ Error: {e}", "error")
    def _on_advanced_filter_previewed(self, filter_name: str, parameters: dict):
        """Enhanced preview handler for advanced filtering."""
        try:
            print(f"Previewing advanced filter: {filter_name} with params: {parameters}")
            
            if hasattr(self, 'advanced_filtering_right_panel'):
                self.advanced_filtering_right_panel.show_status(f"👁️ Previewing {filter_name}...", "processing")
            
            if hasattr(self, 'image_processing_manager'):
                if hasattr(self.image_processing_manager, 'preview_filter'):
                    self.image_processing_manager.preview_filter(filter_name, parameters)
                elif hasattr(self.image_processing_manager, 'on_filter_preview'):
                    # Use alternative method
                    self.image_processing_manager.on_filter_preview(filter_name, parameters)
                else:
                    print("ERROR: No suitable preview method found")
                    # Try preview using direct application
                    if hasattr(self, 'current_sem_image') and self.current_sem_image is not None:
                        preview_result = self._apply_filter_directly(filter_name, parameters, self.current_sem_image)
                        if preview_result is not None:
                            self.image_viewer.set_sem_image(preview_result)
                    
            if hasattr(self, 'advanced_filtering_right_panel'):
                self.advanced_filtering_right_panel.show_status(f"👁️ Preview: {filter_name}", "info")
                    
        except Exception as e:
            print(f"Error previewing advanced filter: {e}")

    def _on_advanced_filter_reset(self):
        """Enhanced reset handler for advanced filtering."""
        try:
            print("Resetting advanced filters")
            
            if hasattr(self, 'advanced_filtering_right_panel'):
                self.advanced_filtering_right_panel.show_status("🔄 Resetting filters...", "processing")
            
            if hasattr(self, 'image_processing_manager'):
                if hasattr(self.image_processing_manager, 'reset_filters'):
                    self.image_processing_manager.reset_filters()
                elif hasattr(self.image_processing_manager, 'on_reset_filters'):
                    # Use alternative method
                    self.image_processing_manager.on_reset_filters()
                else:
                    print("ERROR: No suitable reset method found")
                    # Reset manually by restoring original image
                    if hasattr(self, 'current_sem_image_obj') and self.current_sem_image_obj is not None:
                        original_image = getattr(self.current_sem_image_obj, 'cropped_array', None)
                        if original_image is not None:
                            self.image_viewer.set_sem_image(original_image)
                            self.current_sem_image = original_image
                    
            if hasattr(self, 'advanced_filtering_right_panel'):
                self.advanced_filtering_right_panel.show_status("✓ Filters reset", "success")
                    
        except Exception as e:
            print(f"Error resetting advanced filters: {e}")

    def _apply_filter_fallback(self, filter_name: str, parameters: dict):
        """Fallback method to apply filters when the manager methods don't exist."""
        try:
            if self.current_sem_image is None:
                print("No image available for filtering")
                return
                
            # Apply filter directly
            filtered_image = self._apply_filter_directly(filter_name, parameters, self.current_sem_image)
            
            if filtered_image is not None:
                # Update the display
                self.image_viewer.set_sem_image(filtered_image)
                self.current_sem_image = filtered_image
                
                # Update histogram
                if hasattr(self, 'histogram_view'):
                    self.histogram_view.update_histogram(filtered_image)
                
                if hasattr(self, 'advanced_filtering_right_panel'):
                    self.advanced_filtering_right_panel.update_histogram(filtered_image)
                
                print(f"✓ Filter applied successfully using fallback: {filter_name}")
            else:
                print(f"❌ Filter application failed: {filter_name}")
                
        except Exception as e:
            print(f"Error in filter fallback: {e}")

    def _on_advanced_save_image_requested(self):
        """Handle save image request from advanced filtering panel."""
        try:
            print("Save image requested from advanced filtering panel")
            
            # Show status in right panel
            if hasattr(self, 'advanced_filtering_right_panel'):
                self.advanced_filtering_right_panel.show_status("💾 Saving image...", "processing")
            
            # Use file operations manager to save image
            if hasattr(self, 'file_operations_manager'):
                # Try to save the current filtered image
                if hasattr(self, 'image_processing_service') and self.image_processing_service.current_image is not None:
                    # Save the filtered image
                    self.file_operations_manager.save_filtered_image(self.image_processing_service.current_image)
                elif hasattr(self, 'current_sem_image') and self.current_sem_image is not None:
                    # Save the current SEM image if no filtered version
                    self.file_operations_manager.save_filtered_image(self.current_sem_image)
                else:
                    print("No image available to save")
                    if hasattr(self, 'advanced_filtering_right_panel'):
                        self.advanced_filtering_right_panel.show_status("❌ No image to save", "error")
                    return
                
                # Success feedback
                if hasattr(self, 'advanced_filtering_right_panel'):
                    self.advanced_filtering_right_panel.show_status("✓ Image saved", "success")
                    
            else:
                print("Warning: File operations manager not available")
                if hasattr(self, 'advanced_filtering_right_panel'):
                    self.advanced_filtering_right_panel.show_status("❌ Save failed", "error")
            
            # Update status bar
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage("Advanced filtered image saved successfully", 3000)
                
        except Exception as e:
            print(f"Error saving advanced filtered image: {e}")
            if hasattr(self, 'advanced_filtering_right_panel'):
                self.advanced_filtering_right_panel.show_status(f"❌ Save error: {e}", "error")

    # SEQUENTIAL FILTERING SIGNAL HANDLERS (from second file with fixes)
    def _prepare_histogram_image(self, image_data) -> Optional[np.ndarray]:
        """
        Prepare image data for histogram processing.
        
        Args:
            image_data: Input image data (can be various types)
            
        Returns:
            Properly typed numpy array or None if conversion fails
        """
        try:
            if image_data is None:
                return None
                
            # Convert to numpy array with proper dtype
            if isinstance(image_data, np.ndarray):
                # If already numpy array, ensure correct dtype
                if image_data.dtype != np.uint8:
                    # Convert to uint8 if needed
                    if image_data.dtype in [np.float32, np.float64]:
                        # Assume float values are in range [0, 1] or [0, 255]
                        if image_data.max() <= 1.0:
                            image_data = (image_data * 255).astype(np.uint8)
                        else:
                            image_data = np.clip(image_data, 0, 255).astype(np.uint8)
                    else:
                        image_data = image_data.astype(np.uint8)
                return image_data
            else:
                # Convert other types to numpy array
                array_data = np.asarray(image_data, dtype=np.uint8)
                return array_data
                
        except Exception as e:
            print(f"Error preparing histogram image: {e}")
            return None

    def _safe_set_image(self, image_data) -> bool:
        """
        Safely set image in image processing service with proper None checking.
        
        Args:
            image_data: Image data to set
            
        Returns:
            True if image was set successfully, False otherwise
        """
        try:
            if image_data is None:
                print("Warning: Cannot set None image")
                return False
                
            # Ensure it's a proper numpy array
            processed_image = self._ensure_numpy_array(image_data)
            if processed_image is None:
                print("Warning: Failed to convert image data to numpy array")
                return False
                
            # Set the image
            if hasattr(self, 'image_processing_manager'):
                self.image_processing_manager.image_processing_service.set_image(processed_image)
                return True
            else:
                print("Warning: Image processing manager not available")
                return False
                
        except Exception as e:
            print(f"Error in safe_set_image: {e}")
            return False

    def _on_stage_preview_requested(self, stage_index: int, filter_name: str, parameters: dict):
        """Handle stage preview request in sequential workflow - FIXED VERSION."""
        try:
            print(f"Stage {stage_index} preview requested: {filter_name} with params: {parameters}")
            
            # Get the input image for this stage
            input_image = self._get_stage_input_image(stage_index)
            if input_image is None:
                print(f"No input image available for stage {stage_index}")
                self.sequential_filtering_right_panel.set_processing_status("❌ No input image")
                return
            
            # Update right panel status
            self.sequential_filtering_right_panel.set_processing_status(f"Previewing Stage {stage_index + 1}...")
            
            # Store current state
            original_image = None
            if hasattr(self, 'image_processing_manager'):
                original_image = self.image_processing_manager.image_processing_service.current_image
            
            # Use safe image setting
            if self._safe_set_image(input_image):
                # Apply filter using the service directly (for preview)
                preview_result = self._apply_filter_directly(filter_name, parameters, input_image)
                
                if preview_result is not None:
                    # Update image viewer with preview
                    self.image_viewer.set_sem_image(preview_result)
                    
                    # Update histogram safely
                    histogram_image = self._prepare_histogram_image(preview_result)
                    if histogram_image is not None:
                        self.sequential_filtering_right_panel.update_histogram(histogram_image, stage_index)
                
                    # Update status
                    self.sequential_filtering_right_panel.set_processing_status(f"👁️ Previewing Stage {stage_index + 1}")
                    
                    print(f"✓ Stage {stage_index} preview completed")
                else:
                    print(f"❌ Stage {stage_index} preview failed")
                    self.sequential_filtering_right_panel.set_processing_status("❌ Preview failed")
                
                # Restore original current image if it existed
                if original_image is not None:
                    self._safe_set_image(original_image)
            else:
                print("Failed to set input image for preview")
                self.sequential_filtering_right_panel.set_processing_status("❌ Failed to set input image")
            
        except Exception as e:
            print(f"Error in stage preview: {e}")
            self.sequential_filtering_right_panel.set_processing_status("❌ Preview error")

    def _on_stage_apply_requested(self, stage_index: int, filter_name: str, parameters: dict):
        """Handle stage apply request in sequential workflow - FIXED VERSION."""
        try:
            print(f"Stage {stage_index} apply requested: {filter_name} with params: {parameters}")
            
            # Get the input image for this stage
            input_image = self._get_stage_input_image(stage_index)
            if input_image is None:
                print(f"No input image available for stage {stage_index}")
                self.sequential_filtering_right_panel.set_processing_status("❌ No input image")
                return
            
            # Update right panel status
            self.sequential_filtering_right_panel.set_processing_status(f"Applying Stage {stage_index + 1}...")
            
            # Apply filter using direct method
            if hasattr(self, 'image_processing_manager'):
                # Apply the filter directly to the input image
                result_image = self._apply_filter_directly(filter_name, parameters, input_image)
                
                if result_image is not None:
                    # Store the result for this stage
                    self.sequential_images[stage_index] = result_image.copy()
                    
                    # Update image viewer with result
                    self.image_viewer.set_sem_image(result_image)
                    
                    # FIX: Ensure we pass a valid numpy array to update_histogram
                    histogram_image = self._prepare_histogram_image(result_image)
                    if histogram_image is not None:
                        self.sequential_filtering_right_panel.update_histogram(histogram_image, stage_index)
                    
                    # Update progress in right panel
                    self.sequential_filtering_right_panel.update_stage_progress(stage_index, "✓ Applied", True)
                    
                    # Mark stage as completed in left panel
                    self.sequential_filtering_left_panel.set_stage_completed(stage_index, True)
                    
                    # Update status
                    stage_name = ProcessingStage(stage_index).name.replace('_', ' ').title()
                    self.sequential_filtering_right_panel.set_processing_status(f"✓ Stage {stage_index + 1} ({stage_name}) Applied")
                    
                    # Update main window's current image reference
                    self.current_sem_image = result_image
                    
                    print(f"✓ Stage {stage_index} applied successfully")
                    
                    # Update status bar
                    if hasattr(self, 'status_bar'):
                        self.status_bar.showMessage(f"Sequential Stage {stage_index + 1} applied: {filter_name}", 3000)
                    
                else:
                    print(f"❌ Stage {stage_index} application failed")
                    self.sequential_filtering_right_panel.update_stage_progress(stage_index, "❌ Failed", False)
                    self.sequential_filtering_left_panel.set_stage_completed(stage_index, False)
                    self.sequential_filtering_right_panel.set_processing_status("❌ Application failed")
            else:
                print("ImageProcessingManager not available")
                self.sequential_filtering_right_panel.set_processing_status("❌ Manager not available")
            
        except Exception as e:
            print(f"Error in stage apply: {e}")
            self.sequential_filtering_right_panel.update_stage_progress(stage_index, "❌ Error", False)
            self.sequential_filtering_left_panel.set_stage_completed(stage_index, False)
            self.sequential_filtering_right_panel.set_processing_status("❌ Application error")

    def _on_stage_reset_requested(self, stage_index: int):
        """Handle stage reset request in sequential workflow."""
        try:
            print(f"Stage {stage_index} reset requested")
            
            # Remove this stage and all subsequent stages from stored results
            stages_to_remove = [i for i in self.sequential_images.keys() if i >= stage_index]
            for stage_idx in stages_to_remove:
                if stage_idx in self.sequential_images:
                    del self.sequential_images[stage_idx]
            
            # Reset progress for this stage and subsequent stages
            for i in range(stage_index, len(ProcessingStage)):
                self.sequential_filtering_right_panel.update_stage_progress(i, "Pending", True)
                # Reset the indicator to empty circle
                stage = ProcessingStage(i)
                progress_info = self.sequential_filtering_right_panel.stage_progress[stage]
                progress_info['indicator'].setText("○")
                progress_info['indicator'].setStyleSheet("""
                    QLabel {
                        color: #666666;
                        font-size: 12px;
                        font-weight: bold;
                        min-width: 16px;
                    }
                """)
            
            # Revert to the input image for the reset stage
            input_image = self._get_stage_input_image(stage_index)
            if input_image is not None:
                self.image_viewer.set_sem_image(input_image)
                self.current_sem_image = input_image
                
                # Update histogram
                histogram_image = self._prepare_histogram_image(input_image)
                if histogram_image is not None:
                    self.sequential_filtering_right_panel.update_histogram(histogram_image, stage_index - 1 if stage_index > 0 else None)
            
            # Update status
            self.sequential_filtering_right_panel.set_processing_status(f"Stage {stage_index + 1} reset")
            
            print(f"✓ Stage {stage_index} reset completed")
            
        except Exception as e:
            print(f"Error in stage reset: {e}")

    def _on_stage_save_requested(self, stage_index: int):
        """Handle stage save request in sequential workflow."""
        try:
            print(f"Stage {stage_index} save requested")
            
            # Get the result image for this stage
            if stage_index in self.sequential_images:
                stage_image = self.sequential_images[stage_index]
                
                # Create save directory
                from pathlib import Path
                from datetime import datetime
                import cv2
                
                save_dir = Path("Results/SEM_Filters/sequential")
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                stage_name = ProcessingStage(stage_index).name.lower()
                filename = f"stage_{stage_index + 1}_{stage_name}_{timestamp}.png"
                save_path = save_dir / filename
                
                # Save the image
                success = cv2.imwrite(str(save_path), stage_image)
                
                if success:
                    self.sequential_filtering_right_panel.set_processing_status(f"💾 Stage {stage_index + 1} saved")
                    print(f"✓ Stage {stage_index} saved to: {save_path}")
                    
                    if hasattr(self, 'status_bar'):
                        self.status_bar.showMessage(f"Sequential Stage {stage_index + 1} saved: {filename}", 3000)
                else:
                    print(f"❌ Failed to save stage {stage_index}")
                    self.sequential_filtering_right_panel.set_processing_status("❌ Save failed")
            else:
                print(f"No result available for stage {stage_index}")
                self.sequential_filtering_right_panel.set_processing_status("❌ No result to save")
            
        except Exception as e:
            print(f"Error saving stage {stage_index}: {e}")
            self.sequential_filtering_right_panel.set_processing_status("❌ Save error")

    def _on_reset_all_stages_requested(self):
        """Handle reset all stages request."""
        try:
            print("Reset all stages requested")
            
            # Clear all sequential results
            self.sequential_images.clear()
            
            # Reset all panels
            self.sequential_filtering_left_panel.reset_all_stages()
            self.sequential_filtering_right_panel.reset_all_progress()
            
            # Revert to original image
            if hasattr(self, 'current_sem_image_obj') and self.current_sem_image_obj is not None:
                original_image = getattr(self.current_sem_image_obj, 'cropped_array', self.current_sem_image)
                if original_image is not None:
                    self.image_viewer.set_sem_image(original_image)
                    self.current_sem_image = original_image
                    
                    # Reset histogram to original
                    histogram_image = self._prepare_histogram_image(original_image)
                    if histogram_image is not None:
                        self.sequential_filtering_right_panel.update_histogram(histogram_image)
            
            # Update status
            self.sequential_filtering_right_panel.set_processing_status("All stages reset")
            
            print("✓ All stages reset completed")
            
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage("All sequential stages reset", 3000)
            
        except Exception as e:
            print(f"Error resetting all stages: {e}")

    def _get_stage_input_image(self, stage_index: int) -> Optional[np.ndarray]:
        """
        Get the input image for a specific stage in the sequential workflow.
        
        Args:
            stage_index: The stage index
            
        Returns:
            Input image as numpy array or None if not available
        """
        try:
            if stage_index == 0:
                # Stage 0 (Contrast Enhancement) uses the original image
                if hasattr(self, 'current_sem_image_obj') and self.current_sem_image_obj is not None:
                    original_image = getattr(self.current_sem_image_obj, 'cropped_array', None)
                    return self._ensure_numpy_array(original_image)
                elif hasattr(self, 'current_sem_image') and self.current_sem_image is not None:
                    return self._ensure_numpy_array(self.current_sem_image)
                else:
                    return None
            else:
                # Subsequent stages use the result from the previous stage
                previous_stage = stage_index - 1
                if previous_stage in self.sequential_images:
                    return self._ensure_numpy_array(self.sequential_images[previous_stage])
                else:
                    # If previous stage hasn't been applied, return None
                    print(f"Warning: Previous stage {previous_stage} not applied yet")
                    return None
        
        except Exception as e:
            print(f"Error getting stage input image: {e}")
            return None
    
    def export_sequential_workflow(self):
        """Export the complete sequential workflow results."""
        try:
            if not self.sequential_images:
                QMessageBox.warning(self, "No Sequential Results", "No sequential processing results to export.")
                return
            
            # Get save directory
            save_dir = QFileDialog.getExistingDirectory(
                self, "Export Sequential Workflow Results", "Results/SEM_Filters/sequential"
            )
            
            if save_dir:
                from pathlib import Path
                from datetime import datetime
                import cv2
                import json
                
                save_path = Path(save_dir)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create workflow summary
                workflow_summary = {
                    "timestamp": timestamp,
                    "original_image": str(self.current_sem_path) if hasattr(self, 'current_sem_path') else "Unknown",
                    "stages_completed": len(self.sequential_images),
                    "stages": {}
                }
                
                # Save each stage result
                for stage_index, stage_image in self.sequential_images.items():
                    stage_name = ProcessingStage(stage_index).name.lower()
                    
                    # Save image
                    image_filename = f"stage_{stage_index + 1}_{stage_name}_{timestamp}.png"
                    image_path = save_path / image_filename
                    cv2.imwrite(str(image_path), stage_image)
                    
                    # Add to summary
                    workflow_summary["stages"][stage_index] = {
                        "stage_name": stage_name,
                        "image_file": image_filename,
                        "shape": list(stage_image.shape)
                    }
                
                # Save workflow summary
                summary_path = save_path / f"sequential_workflow_summary_{timestamp}.json"
                with open(summary_path, 'w') as f:
                    json.dump(workflow_summary, f, indent=2)
                
                QMessageBox.information(
                    self, "Export Complete", 
                    f"Sequential workflow exported to:\n{save_path}\n\n"
                    f"Stages exported: {len(self.sequential_images)}\n"
                    f"Summary file: {summary_path.name}"
                )
                
                print(f"✓ Sequential workflow exported to: {save_path}")
            
        except Exception as e:
            print(f"Error exporting sequential workflow: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export sequential workflow:\n{str(e)}")

    # Delegate methods to managers (for compatibility)
    
    def load_sem_image(self):
        """Load a SEM image file."""
        self.file_operations_manager.load_sem_image()
    
    def load_gds_file(self):
        """Load a GDS file."""
        self.gds_operations_manager.load_gds_file()
    
    def save_results(self):
        """Save current results."""
        self.file_operations_manager.save_results()
    
    def auto_align(self):
        """Perform automatic alignment."""
        self.alignment_operations_manager.auto_align()
    
    def reset_alignment(self):
        """Reset alignment."""
        self.alignment_operations_manager.reset_alignment()
    
    def calculate_scores(self):
        """Calculate alignment scores."""
        self.scoring_operations_manager.calculate_scores()
    
    # Signal handler methods
        
    def on_sem_image_loaded(self, file_path, image_data):
        """Handle SEM image loaded with both filtering systems - FIXED VERSION.""" 
        try:
            print(f"SEM image loaded: {file_path}")
            
            if 'cropped_array' in image_data:
                # Use safe image setting
                cropped_array = image_data['cropped_array']
                if self._safe_set_image(cropped_array):
                    # Update main histogram view
                    if hasattr(self, 'histogram_view'):
                        self.histogram_view.update_histogram(cropped_array)
                    
                    # Update advanced filtering right panel
                    if hasattr(self, 'advanced_filtering_right_panel'):
                        self.advanced_filtering_right_panel.update_histogram(cropped_array)
                        self.advanced_filtering_right_panel.show_status("✓ Image loaded", "success")
                    
                    # Update sequential filtering right panel with proper type handling
                    if hasattr(self, 'sequential_filtering_right_panel'):
                        histogram_image = self._prepare_histogram_image(cropped_array)
                        if histogram_image is not None:
                            self.sequential_filtering_right_panel.update_histogram(histogram_image)
                        self.sequential_filtering_right_panel.set_processing_status("✓ Image loaded - Ready for sequential processing")
                        self.sequential_filtering_right_panel.reset_all_progress()
                    
                    # Clear any existing sequential results
                    self.sequential_images.clear()
                else:
                    print("Warning: Failed to set SEM image")
            
            self._update_panel_availability()
            
        except Exception as e:
            print(f"Error handling SEM image loaded: {e}")

    def on_gds_file_loaded(self, file_path):
        """Handle GDS file loaded signal."""
        try:
            print(f"GDS file loaded: {file_path}")
            self._update_panel_availability()
            
        except Exception as e:
            print(f"Error handling GDS file loaded: {e}")
    
    def on_structure_selected(self, structure_name, overlay):
        """Handle structure selected signal."""
        try:
            print(f"Structure selected: {structure_name}")
            # Update current overlay reference for compatibility
            self.current_gds_overlay = overlay
            
            # Display GDS in alignment tab
            self._display_gds_in_alignment_tab(overlay)
            
            # Update panel availability
            self._update_panel_availability()
            
            # Update alignment display for compatibility
            self.update_alignment_display()
            
        except Exception as e:
            print(f"Error handling structure selected: {e}")
            
    def _display_gds_in_alignment_tab(self, overlay):
        """Display GDS overlay in the alignment tab."""
        try:
            # Update image viewer if available
            if hasattr(self, 'image_viewer'):
                self.image_viewer.set_gds_overlay(overlay)
                print("GDS overlay set in image viewer")
            
            # Show feedback message
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f"GDS structure displayed in alignment tab", 3000)
                
        except Exception as e:
            print(f"Error displaying GDS in alignment tab: {e}")
    
    def on_structure_combo_populated(self, count):
        """Handle structure combo populated signal."""
        try:
            print(f"Structure combo populated with {count} structures")
            
        except Exception as e:
            print(f"Error handling structure combo populated: {e}")
    
    def on_filter_applied(self, filter_name, parameters):
        """Handle filter applied signal."""
        try:
            print(f"Filter applied: {filter_name}")
            
            # Update histogram with filtered image
            if hasattr(self, 'histogram_view') and hasattr(self, 'current_sem_image'):
                if self.current_sem_image is not None:
                    self.histogram_view.update_histogram(self.current_sem_image)
            
            # Update advanced filtering right panel
            if hasattr(self, 'advanced_filtering_right_panel') and hasattr(self, 'current_sem_image'):
                if self.current_sem_image is not None:
                    self.advanced_filtering_right_panel.update_histogram(self.current_sem_image)
                    self.advanced_filtering_right_panel.show_status(f"Applied: {filter_name}")
            
            # Update alignment display if needed
            if self.current_gds_overlay is not None:
                self.update_alignment_display()
                
        except Exception as e:
            print(f"Error handling filter applied: {e}")
    
    def on_filters_reset(self):
        """Handle filters reset signal."""
        try:
            print("Filters reset")
            
            # Update histogram with original image
            if hasattr(self, 'histogram_view') and hasattr(self, 'current_sem_image'):
                if self.current_sem_image is not None:
                    self.histogram_view.update_histogram(self.current_sem_image)
            
            # Update advanced filtering right panel
            if hasattr(self, 'advanced_filtering_right_panel'):
                if hasattr(self, 'current_sem_image') and self.current_sem_image is not None:
                    self.advanced_filtering_right_panel.update_histogram(self.current_sem_image)
                self.advanced_filtering_right_panel.show_status("Filters reset")
                if hasattr(self.advanced_filtering_right_panel, 'update_kernel'):
                    self.advanced_filtering_right_panel.update_kernel(None)
            
            # Update alignment display if needed
            if self.current_gds_overlay is not None:
                self.update_alignment_display()
                
        except Exception as e:
            print(f"Error handling filters reset: {e}")
    
    def on_alignment_completed(self, alignment_result):
        """Handle alignment completed signal."""
        try:
            print("Alignment completed")
            self.current_alignment_result = alignment_result
            self._update_panel_availability()
            
            # Enable Generate Aligned GDS button when alignment is completed
            if hasattr(self, 'generate_aligned_gds_btn'):
                self.generate_aligned_gds_btn.setEnabled(True)
            
        except Exception as e:
            print(f"Error handling alignment completed: {e}")
    
    def on_alignment_reset(self):
        """Handle alignment reset signal."""
        try:
            print("Alignment reset")
            self.current_alignment_result = None
            self._update_panel_availability()
            
            # Disable Generate Aligned GDS button when alignment is reset
            if hasattr(self, 'generate_aligned_gds_btn'):
                self.generate_aligned_gds_btn.setEnabled(False)
            
        except Exception as e:
            print(f"Error handling alignment reset: {e}")
    
    def on_scores_calculated(self, scores):
        """Handle scores calculated signal."""
        try:
            print(f"Scores calculated: {scores}")
            self.current_scoring_results = scores
            
            # Update the scoring tab panel with the results
            if hasattr(self, 'scoring_tab_panel'):
                self.scoring_tab_panel.display_results(scores)
            
        except Exception as e:
            print(f"Error handling scores calculated: {e}")
    
    def on_file_error(self, operation, error_message):
        """Handle file operation error signal."""
        print(f"File operation error ({operation}): {error_message}")
    
    # Utility methods
    
    def switch_view(self, view_mode: ViewMode):
        """Switch to a different view mode."""
        try:
            if hasattr(self, 'view_manager'):
                self.view_manager.switch_to_view(view_mode)
            
            # Update tab selection - NOW HANDLES 4 TABS
            if hasattr(self, 'main_tab_widget'):
                tab_index = {
                    ViewMode.ALIGNMENT: 0,
                    ViewMode.FILTERING: 1,  # Advanced Filtering
                    ViewMode.SCORING: 3      # Scoring is now tab 3 (Sequential is tab 2)
                }.get(view_mode, 0)
                self.main_tab_widget.setCurrentIndex(tab_index)
            
            self._update_panel_availability()
            
        except Exception as e:
            print(f"Error switching view: {e}")
    
    def update_alignment_display(self):
        """Update the alignment display."""
        try:
            if hasattr(self, 'image_viewer') and self.current_gds_overlay is not None:
                self.image_viewer.set_gds_overlay(self.current_gds_overlay)
                
        except Exception as e:
            print(f"Error updating alignment display: {e}")
    
    def _update_panel_availability(self):
        """Update panel availability based on current application state."""
        try:
            # Update tab availability - NOW HANDLES 4 TABS
            if hasattr(self, 'main_tab_widget'):
                self.main_tab_widget.setTabEnabled(0, True)  # Alignment
                self.main_tab_widget.setTabEnabled(1, True)  # Advanced Filtering  
                self.main_tab_widget.setTabEnabled(2, True)  # Sequential Filtering
                self.main_tab_widget.setTabEnabled(3, True)  # Scoring
            
            # Update panel manager if available
            if hasattr(self, 'panel_manager'):
                self.panel_manager.update_panel_availability()
            
        except Exception as e:
            print(f"Error updating panel availability: {e}")
    
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            print("Closing application...")
            
            # Cleanup managers if needed
            if hasattr(self, 'file_operations_manager'):
                self.file_operations_manager.cleanup_temp_files()
            
            event.accept()
            print("✓ Application closed successfully")
            
        except Exception as e:
            print(f"Error during application close: {e}")
            event.accept()
    
    def _on_tab_changed(self, index):
        """Handle tab change to switch view mode - NOW HANDLES 4 TABS."""
        try:
            # Map tab indices to view modes and display methods
            tab_mapping = {
                0: ('alignment', ViewMode.ALIGNMENT, self._switch_to_alignment_display),
                1: ('advanced_filtering', None, self._switch_to_advanced_filtering_display),
                2: ('sequential_filtering', None, self._switch_to_sequential_filtering_display),
                3: ('scoring', ViewMode.SCORING, self._switch_to_scoring_display)
            }
            
            if index in tab_mapping:
                tab_name, view_mode, display_method = tab_mapping[index]
                
                # Switch view mode if applicable
                if view_mode and hasattr(self, 'view_manager'):
                    self.view_manager.switch_to_view(view_mode)
                
                # Switch display
                if display_method:
                    display_method()
                    
                self._update_panel_availability()
                
                print(f"✓ Switched to {tab_name} tab")
                
        except Exception as e:
            print(f"Error handling tab change: {e}")
    
    def _switch_to_advanced_filtering_display(self):
        """Switch image viewer to advanced filtering display mode (SEM only)."""
        try:
            if hasattr(self, 'image_viewer'):
                # Hide GDS overlay in filtering mode
                if hasattr(self.image_viewer, 'set_overlay_visible'):
                    self.image_viewer.set_overlay_visible(False)
                
                # Show current SEM image
                if self.current_sem_image is not None:
                    self.image_viewer.set_sem_image(self.current_sem_image)
                
                # Update histogram with current image
                if hasattr(self, 'histogram_view') and self.current_sem_image is not None:
                    self.histogram_view.update_histogram(self.current_sem_image)
                
                # Update advanced filtering right panel
                if hasattr(self, 'advanced_filtering_right_panel') and self.current_sem_image is not None:
                    self.advanced_filtering_right_panel.update_histogram(self.current_sem_image)
                
                print("✓ Switched to advanced filtering display mode")
                
        except Exception as e:
            print(f"Error switching to advanced filtering display: {e}")

    def _switch_to_sequential_filtering_display(self):
        """Switch image viewer to sequential filtering display mode."""
        try:
            if hasattr(self, 'image_viewer'):
                # Hide GDS overlay in filtering mode
                if hasattr(self.image_viewer, 'set_overlay_visible'):
                    self.image_viewer.set_overlay_visible(False)
                
                # Show the original image or latest sequential result
                if self.sequential_images:
                    # Show the result from the highest completed stage
                    latest_stage = max(self.sequential_images.keys())
                    latest_image = self.sequential_images[latest_stage]
                    self.image_viewer.set_sem_image(latest_image)
                    self.current_sem_image = latest_image
                    
                    # Update right panel with proper type handling
                    histogram_image = self._prepare_histogram_image(latest_image)
                    if histogram_image is not None:
                        self.sequential_filtering_right_panel.update_histogram(histogram_image, latest_stage)
                else:
                    # Show original SEM image
                    if hasattr(self, 'current_sem_image_obj') and self.current_sem_image_obj is not None:
                        original_image = getattr(self.current_sem_image_obj, 'cropped_array', self.current_sem_image)
                        if original_image is not None:
                            self.image_viewer.set_sem_image(original_image)
                            self.current_sem_image = original_image
                            
                            # Update right panel with proper type handling
                            histogram_image = self._prepare_histogram_image(original_image)
                            if histogram_image is not None:
                                self.sequential_filtering_right_panel.update_histogram(histogram_image)
                    elif self.current_sem_image is not None:
                        self.image_viewer.set_sem_image(self.current_sem_image)
                        
                        # Update right panel with proper type handling
                        histogram_image = self._prepare_histogram_image(self.current_sem_image)
                        if histogram_image is not None:
                            self.sequential_filtering_right_panel.update_histogram(histogram_image)
                
                print("✓ Switched to sequential filtering display mode")
                
        except Exception as e:
            print(f"Error switching to sequential filtering display: {e}")
    
    def _switch_to_alignment_display(self):
        """Switch image viewer to alignment display mode (SEM + GDS)."""
        try:
            if hasattr(self, 'image_viewer'):
                # Ensure we're showing the original SEM image (not filtered) for alignment
                if hasattr(self, 'current_sem_image_obj') and self.current_sem_image_obj is not None:
                    original_image = getattr(self.current_sem_image_obj, 'cropped_array', self.current_sem_image)
                    if original_image is not None:
                        self.image_viewer.set_sem_image(original_image)
                elif self.current_sem_image is not None:
                    self.image_viewer.set_sem_image(self.current_sem_image)
                
                # Show GDS overlay in alignment mode
                if self.current_gds_overlay is not None:
                    self.image_viewer.set_gds_overlay(self.current_gds_overlay)
                    if hasattr(self.image_viewer, 'set_overlay_visible'):
                        self.image_viewer.set_overlay_visible(True)
                
                # Apply any existing alignment transformations
                if self.current_alignment_result is not None:
                    self.image_viewer.set_alignment_result(self.current_alignment_result)
                
                print("✓ Switched to alignment display mode")
                
        except Exception as e:
            print(f"Error switching to alignment display: {e}")
    
    def _switch_to_scoring_display(self):
        """Switch image viewer to scoring display mode (comparison results)."""
        try:
            if hasattr(self, 'image_viewer'):
                # Set SEM image (original unfiltered for accurate scoring)
                if hasattr(self, 'current_sem_image_obj') and self.current_sem_image_obj is not None:
                    original_image = getattr(self.current_sem_image_obj, 'cropped_array', self.current_sem_image)
                    if original_image is not None:
                        self.image_viewer.set_sem_image(original_image)
                elif self.current_sem_image is not None:
                    self.image_viewer.set_sem_image(self.current_sem_image)
                
                # Show aligned GDS overlay if available
                if self.current_gds_overlay is not None:
                    self.image_viewer.set_gds_overlay(self.current_gds_overlay)
                    if hasattr(self.image_viewer, 'set_overlay_visible'):
                        self.image_viewer.set_overlay_visible(True)
                    
                    # Apply alignment if available
                    if self.current_alignment_result is not None:
                        self.image_viewer.set_alignment_result(self.current_alignment_result)
                
                # Update scoring display with current results if available
                if hasattr(self, 'scoring_tab_panel') and self.current_scoring_results:
                    self.scoring_tab_panel.display_results(self.current_scoring_results)
                
                print("✓ Switched to scoring display mode")
                
        except Exception as e:
            print(f"Error switching to scoring display: {e}")
    
    def _on_hybrid_subtab_changed(self, index):
        """Handle hybrid sub-tab changes to set point selection mode."""
        try:
            if not hasattr(self, 'alignment_sub_tabs') or not hasattr(self, 'hybrid_sub_tabs'):
                return
            
            # Only enable point selection if we're on the Hybrid alignment tab
            if self.alignment_sub_tabs.currentIndex() == 1:  # Hybrid tab
                if index == 0:  # SEM Image sub-tab
                    self.image_viewer.set_point_selection_mode(True, "sem")
                elif index == 1:  # GDS sub-tab
                    self.image_viewer.set_point_selection_mode(True, "gds")
            else:
                self.image_viewer.set_point_selection_mode(False)
                
        except Exception as e:
            print(f"Error handling hybrid sub-tab change: {e}")
    
    def _on_alignment_subtab_changed(self, index):
        """Handle alignment sub-tab changes to manage point selection mode."""
        try:
            if index == 1:  # Hybrid tab
                # Enable point selection based on current hybrid sub-tab
                if hasattr(self, 'hybrid_sub_tabs'):
                    current_hybrid_tab = self.hybrid_sub_tabs.currentIndex()
                    if current_hybrid_tab == 0:  # SEM Image sub-tab
                        self.image_viewer.set_point_selection_mode(True, "sem")
                    elif current_hybrid_tab == 1:  # GDS sub-tab
                        self.image_viewer.set_point_selection_mode(True, "gds")
            else:
                # Disable point selection for Manual tab
                self.image_viewer.set_point_selection_mode(False)
                
        except Exception as e:
            print(f"Error handling alignment sub-tab change: {e}")
    
    def _on_point_selected(self, x, y, point_type):
        """Handle point selection from image viewer."""
        try:
            if x == -1 and y == -1:
                print(f"{point_type.upper()} point removed")
            else:
                print(f"{point_type.upper()} point added at ({x}, {y})")
            
            # Update point tracking
            if point_type == "sem":
                self.sem_points = self.image_viewer.get_points("sem")
            elif point_type == "gds":
                self.gds_points = self.image_viewer.get_points("gds")
            
            # Update status display
            self._update_hybrid_status()
            
        except Exception as e:
            print(f"Error handling point selection: {e}")
    
    def _validate_points(self):
        """Validate that points are within image bounds and properly formatted."""
        try:
            max_width = 1024
            max_height = 666
            
            # Check SEM points
            for i, (x, y) in enumerate(self.sem_points):
                if not (0 <= x < max_width and 0 <= y < max_height):
                    return False, f"SEM point {i+1} is out of bounds: ({x}, {y})"
            
            # Check GDS points  
            for i, (x, y) in enumerate(self.gds_points):
                if not (0 <= x < max_width and 0 <= y < max_height):
                    return False, f"GDS point {i+1} is out of bounds: ({x}, {y})"
            
            # Check for duplicate points
            if len(set(self.sem_points)) != len(self.sem_points):
                return False, "Duplicate SEM points detected"
            
            if len(set(self.gds_points)) != len(self.gds_points):
                return False, "Duplicate GDS points detected"
            
            return True, "Points are valid"
            
        except Exception as e:
            return False, f"Point validation error: {e}"
    
    def show_message(self, title: str, message: str):
        """Show information message to user."""
        try:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            msg_box.exec()
        except Exception as e:
            print(f"Error showing message: {e}")
    
    def show_error(self, title: str, message: str):
        """Show error message to user."""
        try:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            msg_box.exec()
        except Exception as e:
            print(f"Error showing error message: {e}")

    # Action button handlers
    
    def _on_auto_align_clicked(self):
        """Handle automatic alignment button click."""
        try:
            print("Automatic alignment button clicked")
            if hasattr(self, 'alignment_operations_manager'):
                self.alignment_operations_manager.auto_align()
            else:
                print("Alignment operations manager not available")
                
        except Exception as e:
            print(f"Error in automatic alignment: {e}")
    
    def _on_reset_transformation_clicked(self):
        """Handle reset transformation button click."""
        try:
            print("Reset transformation button clicked")
            if hasattr(self, 'alignment_operations_manager'):
                self.alignment_operations_manager.reset_alignment()
            else:
                print("Alignment operations manager not available")
                
        except Exception as e:
            print(f"Error in reset transformation: {e}")
    
    def _on_generate_aligned_gds_clicked(self):
        """Handle generate aligned GDS button click."""
        try:
            print("Generate aligned GDS button clicked")
            if hasattr(self, 'alignment_operations_manager'):
                self.alignment_operations_manager.generate_aligned_gds()
            else:
                print("Alignment operations manager not available")
                
        except Exception as e:
            print(f"Error in generate aligned GDS: {e}")
    
    def _sync_view_manager_data(self):
        """Synchronize view manager data with current main window state."""
        try:
            # Sync SEM image and GDS overlay for all views
            for view_mode in [ViewMode.ALIGNMENT, ViewMode.FILTERING, ViewMode.SCORING]:
                view_data = self.view_manager.get_view_data(view_mode)
                
                # Update common data
                view_data['sem_image'] = self.current_sem_image
                view_data['gds_overlay'] = self.current_gds_overlay
                
                # Update view-specific data
                if view_mode == ViewMode.ALIGNMENT:
                    view_data['alignment_result'] = self.current_alignment_result
                elif view_mode == ViewMode.SCORING:
                    view_data['scoring_method'] = self.current_scoring_method
                    view_data['scores'] = self.current_scoring_results
                
                self.view_manager.update_view_data(view_mode, view_data)
            
        except Exception as e:
            print(f"Error syncing view manager data: {e}")
    
    def _ensure_numpy_array(self, image_data) -> Optional[np.ndarray]:
        """
        Ensure the image data is properly typed as a numpy array.
        
        Args:
            image_data: Input image data
            
        Returns:
            Numpy array or None if conversion fails
        """
        try:
            if image_data is None:
                return None
                
            # Convert to numpy array if it's not already
            if isinstance(image_data, np.ndarray):
                return image_data.astype(np.uint8) if image_data.dtype != np.uint8 else image_data
            else:
                return np.asarray(image_data, dtype=np.uint8)
                
        except Exception as e:
            print(f"Error ensuring numpy array: {e}")
            return None

    def _apply_filter_directly(self, filter_name: str, parameters: dict, input_image) -> Optional[np.ndarray]:
        """Apply filter directly to an image using OpenCV operations - FIXED VERSION."""
        try:
            # Ensure input is a proper numpy array
            image = self._ensure_numpy_array(input_image)
            if image is None:
                return None
                
            # Make a copy to avoid modifying the original
            image = image.copy()
            
            # Apply different filters based on filter_name
            if filter_name == "clahe":
                # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clip_limit = parameters.get('clip_limit', 2.0)
                tile_grid_x = parameters.get('tile_grid_x', 8)
                tile_grid_y = parameters.get('tile_grid_y', 8)
                
                if len(image.shape) == 3:
                    # Convert to LAB color space for better results
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_x, tile_grid_y))
                    lab[:,:,0] = clahe.apply(lab[:,:,0])
                    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                else:
                    # Grayscale image
                    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_x, tile_grid_y))
                    result = clahe.apply(image)
                
                return result.astype(np.uint8)
                
            elif filter_name == "gamma_correction":
                # Gamma correction
                gamma = parameters.get('gamma', 1.0)
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
                result = cv2.LUT(image, table)
                return result.astype(np.uint8)
                
            elif filter_name == "histogram_equalization":
                # Histogram equalization
                if len(image.shape) == 3:
                    # Convert to YUV, equalize Y channel, convert back
                    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                    result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                else:
                    result = cv2.equalizeHist(image)
                return result.astype(np.uint8)
                
            elif filter_name == "gaussian_blur":
                # Gaussian blur
                kernel_size = parameters.get('kernel_size', 5)
                sigma = parameters.get('sigma', 1.0)
                if kernel_size % 2 == 0:
                    kernel_size += 1  # Ensure odd kernel size
                result = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
                return result.astype(np.uint8)
                
            elif filter_name == "median_filter":
                # Median filter
                kernel_size = parameters.get('kernel_size', 5)
                if kernel_size % 2 == 0:
                    kernel_size += 1  # Ensure odd kernel size
                result = cv2.medianBlur(image, kernel_size)
                return result.astype(np.uint8)
                
            elif filter_name == "bilateral_filter":
                # Bilateral filter
                d = parameters.get('d', 9)
                sigma_color = parameters.get('sigma_color', 75)
                sigma_space = parameters.get('sigma_space', 75)
                result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
                return result.astype(np.uint8)
                
            elif filter_name == "nlm_denoising":
                # Non-local means denoising
                h = parameters.get('h', 10)
                template_window_size = parameters.get('template_window_size', 7)
                search_window_size = parameters.get('search_window_size', 21)
                
                if len(image.shape) == 3:
                    result = cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window_size, search_window_size)
                else:
                    result = cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)
                return result.astype(np.uint8)
                
            elif filter_name == "threshold":
                # Simple threshold
                threshold_value = parameters.get('threshold_value', 127)
                max_value = parameters.get('max_value', 255)
                
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                _, result = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY)
                return result.astype(np.uint8)
                
            elif filter_name == "adaptive_threshold":
                # Adaptive threshold
                max_value = parameters.get('max_value', 255)
                block_size = parameters.get('block_size', 11)
                c = parameters.get('c', 2)
                
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                result = cv2.adaptiveThreshold(gray, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
                return result.astype(np.uint8)
                
            elif filter_name == "otsu_threshold":
                # Otsu threshold
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return result.astype(np.uint8)
                
            elif filter_name == "canny":
                # Canny edge detection
                low_threshold = parameters.get('low_threshold', 50)
                high_threshold = parameters.get('high_threshold', 150)
                aperture_size = parameters.get('aperture_size', 3)
                
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                result = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=aperture_size)
                return result.astype(np.uint8)
                
            elif filter_name == "laplacian":
                # Laplacian edge detection
                ksize = parameters.get('ksize', 3)
                scale = parameters.get('scale', 1)
                delta = parameters.get('delta', 0)
                
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                result = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize, scale=scale, delta=delta)
                result = np.absolute(result)
                return result.astype(np.uint8)
                
            elif filter_name == "sobel":
                # Sobel edge detection
                dx = parameters.get('dx', 1)
                dy = parameters.get('dy', 1)
                ksize = parameters.get('ksize', 3)
                
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                grad_x = cv2.Sobel(gray, cv2.CV_64F, dx, 0, ksize=ksize)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, dy, ksize=ksize)
                result = np.sqrt(grad_x**2 + grad_y**2)
                return result.astype(np.uint8)
                
            elif filter_name == "sharpen":
                # Sharpen filter
                amount = parameters.get('amount', 1.5)
                
                # Create sharpening kernel
                kernel = np.array([[-1,-1,-1],
                                [-1, 9,-1],
                                [-1,-1,-1]]) * amount
                kernel[1,1] = kernel[1,1] - amount + 1
                
                result = cv2.filter2D(image, -1, kernel)
                result = np.clip(result, 0, 255).astype(np.uint8)
                return result
                
            else:
                print(f"Unknown filter: {filter_name}")
                return None
                
        except Exception as e:
            print(f"Error applying filter {filter_name}: {e}")
            return None


def main():
    """Main entry point for the application."""
    try:
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName("SEM/GDS Image Analysis Tool - Combined")
        app.setApplicationVersion("3.1.0")
        app.setOrganizationName("Research Lab")
        
        # Create and show main window
        main_window = MainWindow()
        main_window.show()
        
        # Start event loop
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()