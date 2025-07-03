"""
Core Main Window - Modular Version
Clean, focused main window that coordinates the different manager modules.
This replaces the monolithic main_window_v2.py with a maintainable, modular structure.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
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

# Import alignment components for Step 3
from src.ui.panels.alignment_left_panel import ManualAlignmentTab, ThreePointAlignmentTab

# Import core models
from src.core.models.structure_definitions import get_default_structures

# Configuration
DEFAULT_GDS_FILE = "Institute_Project_GDS1.gds"


class MainWindow(QMainWindow):
    """
    Main Window for the Image Analysis Tool - Modular Version.
    
    This is a clean, focused main window that coordinates different manager modules
    rather than containing all functionality in one monolithic class.
    """
    
    def __init__(self):
        """Initialize the main window and all manager modules."""
        print("MainWindow constructor called")
        super().__init__()
        self.setWindowTitle("Image Analysis - SEM/GDS Alignment Tool")
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
            self.main_splitter = QSplitter(Qt.Horizontal)
            main_layout.addWidget(self.main_splitter)
            
            # Setup left panel container with tabs
            self.left_panel_container = QWidget()
            self.left_panel_container.setMaximumWidth(220)  # Make it a bit wider
            self.left_panel_container.setMinimumWidth(180)  # Ensure minimum usability
            self.left_panel_layout = QVBoxLayout(self.left_panel_container)
            self.left_panel_layout.setContentsMargins(0, 0, 0, 0)
            
            # Create main tab widget for Alignment, Filtering, Scoring
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
            
            # Create tab content areas
            self.alignment_tab = QWidget()
            self.filtering_tab = QWidget()
            self.scoring_tab = QWidget()
            
            # Setup alignment tab content with existing ManualAlignmentTab
            self._setup_alignment_tab_content()
            
            # Setup filtering tab content
            self._setup_filtering_tab_content()
            
            # Setup scoring tab content
            self._setup_scoring_tab_content()
            
            # Add tabs
            self.main_tab_widget.addTab(self.alignment_tab, "Alignment")
            self.main_tab_widget.addTab(self.filtering_tab, "Filtering") 
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
            
            # Setup right panel container with file selection moved to top
            self.right_panel_container = QWidget()
            self.right_panel_layout = QVBoxLayout(self.right_panel_container)
            
            # Move file selection to top-right panel
            self._setup_file_selection_panel()
            
            # Add histogram display below file selection
            self._setup_histogram_panel()
            
            # Common controls below file selection
            self._setup_common_controls()
            
            # Add view-specific content area
            self.view_specific_widget = QWidget()
            self.view_specific_layout = QVBoxLayout(self.view_specific_widget)
            self.right_panel_layout.addWidget(self.view_specific_widget)
            
            self.main_splitter.addWidget(self.right_panel_container)
            
            # Set initial splitter sizes - expand canvas height
            self.main_splitter.setSizes([220, 1320, 250])  # Adjust for slightly wider left panel
            
            # Setup menus (toolbar removed - using tabs instead)
            self._setup_menu()
            self._setup_status_bar()
            
            print("✓ UI setup completed")
            
        except Exception as e:
            print(f"Error setting up UI: {e}")
            raise
    
    def _setup_file_selection_panel(self):
        """Setup enhanced file selection panel using FileSelector component."""
        try:
            # Create FileSelector component
            self.file_selector = FileSelector()
            
            # Connect FileSelector signals to existing methods
            self.file_selector.sem_file_selected.connect(self.load_sem_image_from_path)
            self.file_selector.gds_file_loaded.connect(self.load_gds_file_from_path)
            self.file_selector.gds_structure_selected.connect(self.handle_structure_selection)
            
            # Add FileSelector to right panel
            self.right_panel_layout.addWidget(self.file_selector)
            
            # Initialize file scanning
            self.file_selector.scan_directories()
            
            print("✓ File selection panel setup with FileSelector component")
            
        except Exception as e:
            print(f"Error setting up file selection panel: {e}")
    
    def load_sem_image_from_path(self, file_path: str):
        """Load SEM image from file path (adapter for FileSelector signal)."""
        try:
            self.file_operations_manager.load_sem_image_from_path(file_path)
            print(f"SEM image loaded: {Path(file_path).name}")
        except Exception as e:
            print(f"Error loading SEM image: {e}")
    
    def load_gds_file_from_path(self, file_path: str):
        """Load GDS file from file path (adapter for FileSelector signal)."""
        try:
            self.gds_operations_manager.load_gds_file_from_path(file_path)
            print(f"GDS file loaded: {Path(file_path).name}")
        except Exception as e:
            print(f"Error loading GDS file: {e}")
    
    def handle_structure_selection(self, gds_file_path: str, structure_id: int):
        """Handle structure selection from FileSelector."""
        try:
            # Use the GDS operations manager to handle structure selection
            self.gds_operations_manager.select_structure_by_id(structure_id)
            print(f"Structure {structure_id} selected from {Path(gds_file_path).name}")
        except Exception as e:
            print(f"Error handling structure selection: {e}")
    
    def _setup_histogram_panel(self):
        """Setup histogram display panel in the right panel."""
        try:
            # Create histogram view component
            self.histogram_view = HistogramView()
            
            # Set appropriate size constraints for right panel
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
            # Note: File selection and structure selection are now handled by FileSelector component
            # This method is kept for any additional common controls that might be needed
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
            self.status_bar.showMessage("Ready")
            
        except Exception as e:
            print(f"Error setting up status bar: {e}")
    
    def _connect_manager_signals(self):
        """Connect signals between different managers."""
        try:
            print("Connecting manager signals...")
            
            # Note: Structure combo signal is now handled by FileSelector component
            # No need to connect old structure_combo since it's replaced by FileSelector
            
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
            
            # Ensure default GDS file is auto-loaded after signals are connected
            from PySide6.QtCore import QTimer
            QTimer.singleShot(500, self.gds_operations_manager._auto_load_default_gds)
            
            print("✓ Manager signals connected")
            
        except Exception as e:
            print(f"Error connecting manager signals: {e}")
    
    def _initialize_view_system(self):
        """Initialize the view system and panels."""
        try:
            # Initialize view manager
            self.view_manager = ViewManager(self)
            
            # Initialize panel manager for view-specific panels
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
            # Additional view change handling can be added here
        except Exception as e:
            print(f"Error handling view change: {e}")
    
    def _on_view_data_updated(self, view_mode, data):
        """Handle view manager data update signal."""
        try:
            print(f"View data updated for {view_mode}: {list(data.keys())}")
            # Additional data update handling can be added here
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
            
            # Connect alignment sub-tab changes to manage point selection mode
            self.alignment_sub_tabs.currentChanged.connect(self._on_alignment_subtab_changed)
            
            # Add action buttons at the bottom of alignment tab
            self._setup_alignment_action_buttons(alignment_layout)
            
            print("✓ Alignment tab content setup with ManualAlignmentTab component")
            
        except Exception as e:
            print(f"Error setting up alignment tab content: {e}")
    
    def _setup_manual_alignment_content(self):
        """Setup the manual alignment tab content."""
        try:
            # Create layout for manual alignment tab
            manual_layout = QVBoxLayout(self.manual_alignment_tab)
            manual_layout.setContentsMargins(5, 5, 5, 5)
            
            # Create and add the existing ManualAlignmentTab component
            self.manual_alignment_controls = ManualAlignmentTab()
            
            # Connect the manual alignment signals to the alignment operations manager
            self.manual_alignment_controls.alignment_changed.connect(
                self.alignment_operations_manager.apply_manual_transformation
            )
            self.manual_alignment_controls.reset_requested.connect(
                self.alignment_operations_manager.reset_alignment
            )
            
            # Add manual alignment controls to the tab
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
            
            # Connect hybrid sub-tab changes to manage point selection mode
            self.hybrid_sub_tabs.currentChanged.connect(self._on_hybrid_subtab_changed)
            
            # Add status display for selected points
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
            
            # Add spacing
            buttons_layout.addSpacing(10)
            
            parent_layout.addLayout(buttons_layout)
            
            # Initialize point tracking
            self.sem_points = []
            self.gds_points = []
            
            # Connect buttons (placeholder - actual functionality in Step 6)
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
            print(f"SEM points: {self.sem_points}")
            print(f"GDS points: {self.gds_points}")
            
            # Use the existing alignment operations manager for 3-point alignment
            if hasattr(self, 'alignment_operations_manager'):
                self.alignment_operations_manager.manual_align_3_point(self.sem_points, self.gds_points)
                print("✓ 3-point alignment calculation completed")
                
                # Update status
                if hasattr(self, 'status_bar'):
                    self.status_bar.showMessage("3-point alignment calculation completed")
            else:
                print("Error: Alignment operations manager not available")
                
        except Exception as e:
            print(f"Error calculating alignment: {e}")
            import traceback
            traceback.print_exc()
    
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
            # Add some spacing before buttons
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
                QPushButton:disabled {
                    background-color: #7f8c8d;
                    color: #bdc3c7;
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
            
            # Second row: Generate Aligned GDS (prominent button)
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

    def _setup_filtering_tab_content(self):
        """Setup the filtering tab content with Manual Filtering sub-tab."""
        try:
            # Create layout for filtering tab
            filtering_layout = QVBoxLayout(self.filtering_tab)
            filtering_layout.setContentsMargins(10, 10, 10, 10)
            
            # Create sub-tab widget for filtering modes
            self.filtering_sub_tabs = QTabWidget()
            self.filtering_sub_tabs.setStyleSheet("""
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
            
            # Create Manual Filtering tab
            self.manual_filtering_tab = QWidget()
            self._setup_manual_filtering_content()
            self.filtering_sub_tabs.addTab(self.manual_filtering_tab, "Manual Filtering")
            
            # Create Automatic Filtering tab
            self.automatic_filtering_tab = QWidget()
            self._setup_automatic_filtering_content()
            self.filtering_sub_tabs.addTab(self.automatic_filtering_tab, "Automatic Filtering")
            
            # Add sub-tabs to main filtering tab
            filtering_layout.addWidget(self.filtering_sub_tabs)
            
            print("✓ Filtering tab content setup completed")
            
        except Exception as e:
            print(f"Error setting up filtering tab content: {e}")
            import traceback
            traceback.print_exc()
    
    def _setup_manual_filtering_content(self):
        """Setup the manual filtering tab content."""
        try:
            # Create layout for manual filtering tab
            manual_layout = QVBoxLayout(self.manual_filtering_tab)
            manual_layout.setContentsMargins(5, 5, 5, 5)
            
            # Import and create the existing FilterPanel component
            from src.ui.panels.filter_panel import FilterPanel
            
            # Create filter panel
            self.filter_panel = FilterPanel()
            
            # Connect filter panel signals to image processing manager
            self.filter_panel.filter_applied.connect(self.image_processing_manager.on_filter_applied)
            self.filter_panel.filter_previewed.connect(self.image_processing_manager.on_filter_preview)
            self.filter_panel.filter_reset.connect(self.image_processing_manager.on_reset_filters)
            self.filter_panel.save_image_requested.connect(self.image_processing_manager.save_current_image)
            
            # Initialize filter panel with available filters
            try:
                available_filters = self.image_processing_service.get_available_filters()
                self.filter_panel.set_available_filters(available_filters)
            except Exception as filter_error:
                print(f"Error initializing filters: {filter_error}")
                # Set some basic filters as fallback
                fallback_filters = ["gaussian_blur", "threshold", "edge_detection"]
                self.filter_panel.set_available_filters(fallback_filters)
            
            # Add the filter panel to the manual filtering tab
            manual_layout.addWidget(self.filter_panel)
            
            print("✓ Manual filtering content setup with FilterPanel component")
            
        except Exception as e:
            print(f"Error setting up manual filtering content: {e}")
            import traceback
            traceback.print_exc()
    
    def _setup_automatic_filtering_content(self):
        """Setup the automatic filtering tab content."""
        try:
            # Create layout for automatic filtering tab
            automatic_layout = QVBoxLayout(self.automatic_filtering_tab)
            automatic_layout.setContentsMargins(5, 5, 5, 5)
            
            # Import and create the AutomaticFilterPanel component
            from src.ui.panels.automatic_filter_panel import AutomaticFilterPanel
            
            # Create automatic filter panel
            self.automatic_filter_panel = AutomaticFilterPanel()
            
            # Connect automatic filter panel signals (placeholder for future functionality)
            self.automatic_filter_panel.automatic_filtering_requested.connect(self._on_automatic_filtering_requested)
            
            # Add the automatic filter panel to the automatic filtering tab
            automatic_layout.addWidget(self.automatic_filter_panel)
            
            print("✓ Automatic filtering content setup with AutomaticFilterPanel component")
            
        except Exception as e:
            print(f"Error setting up automatic filtering content: {e}")
            import traceback
            traceback.print_exc()
    
    def _setup_scoring_tab_content(self):
        """Setup the scoring tab content with clean white background design."""
        try:
            # Create layout for scoring tab
            scoring_layout = QVBoxLayout(self.scoring_tab)
            scoring_layout.setContentsMargins(0, 0, 0, 0)  # No margins for full white background
            
            # Import and create the ScoringTabPanel component
            from src.ui.panels.scoring_tab_panel import ScoringTabPanel
            
            # Create scoring tab panel
            self.scoring_tab_panel = ScoringTabPanel()
            
            # Connect scoring tab panel signals to scoring operations manager
            self.scoring_tab_panel.scoring_method_changed.connect(self._on_scoring_method_changed)
            self.scoring_tab_panel.calculate_scores_requested.connect(self._on_calculate_scores_requested)
            
            # Add the scoring tab panel to the scoring tab
            scoring_layout.addWidget(self.scoring_tab_panel)
            
            print("✓ Scoring tab content setup with ScoringTabPanel component")
            
        except Exception as e:
            print(f"Error setting up scoring tab content: {e}")
            import traceback
            traceback.print_exc()

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
                # Set the method and calculate scores
                self.scoring_operations_manager.current_scoring_method = method_name
                self.scoring_operations_manager.calculate_scores()
            else:
                print("Warning: Scoring operations manager not available")
                
                # Fallback: use scoring calculator directly if available
                if hasattr(self, 'calculate_scores'):
                    self.calculate_scores()
                else:
                    print("Error: No scoring calculation method available")
                    
                    # Update status to show error
                    if hasattr(self, 'status_bar'):
                        self.status_bar.showMessage("Error: No scoring system available")
            
        except Exception as e:
            print(f"Error handling calculate scores request: {e}")
            
            # Re-enable the calculate button on error
            if hasattr(self, 'scoring_tab_panel'):
                self.scoring_tab_panel.calculate_button.setEnabled(True)
                self.scoring_tab_panel.status_label.setText("Error occurred during calculation")

    def _on_automatic_filtering_requested(self, filter_selections):
        """Handle automatic filtering request (placeholder functionality)."""
        try:
            print("Automatic filtering requested with selections:")
            for category, filter_data in filter_selections.items():
                if filter_data:
                    print(f"  {category}: {filter_data['filter']} with parameters {filter_data}")
            
            # Update status
            if hasattr(self, 'status_bar'):
                filter_count = len(filter_selections)
                self.status_bar.showMessage(f"Automatic filtering requested with {filter_count} filters (placeholder)")
            
            print("Note: Automatic filtering functionality to be implemented in future steps")
            
        except Exception as e:
            print(f"Error handling automatic filtering request: {e}")

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
        """Handle SEM image loaded signal."""
        try:
            print(f"SEM image loaded: {file_path}")
            
            # Set up image processing with the new image
            if 'cropped_array' in image_data:
                self.image_processing_manager.image_processing_service.set_image(
                    image_data['cropped_array']
                )
                
                # Update histogram display
                if hasattr(self, 'histogram_view'):
                    self.histogram_view.update_histogram(image_data['cropped_array'])
                
                # Update histogram if filter panel exists
                if hasattr(self, 'filter_panel') and hasattr(self.filter_panel, 'update_histogram'):
                    self.filter_panel.update_histogram(image_data['cropped_array'])
            
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
            # Check if we're currently on the alignment tab
            current_tab = self.main_tab_widget.currentIndex()
            is_alignment_tab = (current_tab == 0)  # Alignment is typically the first tab
            
            # Display in alignment tabs
            if hasattr(self, 'manual_alignment_controls'):
                # Display in manual alignment tab
                if hasattr(self.manual_alignment_controls, '_display_gds_image'):
                    self.manual_alignment_controls._display_gds_image(overlay)
                    print("GDS displayed in manual alignment tab")
                    
            # Convert overlay to grayscale for binary display methods
            if len(overlay.shape) == 3:
                gds_binary = np.any(overlay > 0, axis=2).astype(np.uint8) * 255
            else:
                gds_binary = overlay
                
            # Update image viewer if available
            if hasattr(self, 'image_viewer'):
                self.image_viewer.set_gds_overlay(overlay)
                print("GDS overlay set in image viewer")
            
            # Show feedback message
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f"GDS structure displayed in alignment tab", 3000)
                
        except Exception as e:
            print(f"Error displaying GDS in alignment tab: {e}")
            import traceback
            traceback.print_exc()
    
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
            
            # Update alignment display if needed
            if self.current_gds_overlay is not None:
                self.update_alignment_display()
                
        except Exception as e:
            print(f"Error handling filters reset: {e}")
    
    def on_alignment_completed(self, alignment_result):
        """Handle alignment completed signal."""
        try:
            print("Alignment completed")
            # Update current alignment result for compatibility
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
            # Update current scoring results for compatibility
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
            
            # Update tab selection
            if hasattr(self, 'main_tab_widget'):
                tab_index = {
                    ViewMode.ALIGNMENT: 0,
                    ViewMode.FILTERING: 1, 
                    ViewMode.SCORING: 2
                }.get(view_mode, 0)
                self.main_tab_widget.setCurrentIndex(tab_index)
            
            self._update_panel_availability()
            
        except Exception as e:
            print(f"Error switching view: {e}")
    
    def update_alignment_display(self):
        """Update the alignment display."""
        try:
            # This method provides compatibility with existing panel code
            if hasattr(self, 'image_viewer') and self.current_gds_overlay is not None:
                self.image_viewer.set_gds_overlay(self.current_gds_overlay)
                
        except Exception as e:
            print(f"Error updating alignment display: {e}")
    
    def _update_panel_availability(self):
        """Update panel availability based on current application state."""
        try:
            # Update tab availability
            if hasattr(self, 'main_tab_widget'):
                # All tabs are always available - let the user switch between them
                # The individual tab content will handle cases where data is not available
                self.main_tab_widget.setTabEnabled(0, True)  # Alignment
                self.main_tab_widget.setTabEnabled(1, True)  # Filtering  
                self.main_tab_widget.setTabEnabled(2, True)  # Scoring
            
            # Update panel manager if available
            if hasattr(self, 'panel_manager'):
                self.panel_manager.update_panel_availability()
            
        except Exception as e:
            print(f"Error updating panel availability: {e}")
    
    def _update_score_overlays(self):
        """Update score overlays (compatibility method)."""
        try:
            # This method provides compatibility with existing scoring code
            pass
            
        except Exception as e:
            print(f"Error updating score overlays: {e}")
    
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            print("Closing application...")
            
            # Cleanup managers if needed
            if hasattr(self, 'file_operations_manager'):
                self.file_operations_manager.cleanup_temp_files()
            
            # Accept the close event
            event.accept()
            
            print("✓ Application closed successfully")
            
        except Exception as e:
            print(f"Error during application close: {e}")
            event.accept()  # Close anyway
    
    def _on_tab_changed(self, index):
        """Handle tab change to switch view mode."""
        try:
            # Save current tab state before switching
            self._save_current_tab_state()
            
            view_modes = [ViewMode.ALIGNMENT, ViewMode.FILTERING, ViewMode.SCORING]
            if 0 <= index < len(view_modes):
                view_mode = view_modes[index]
                if hasattr(self, 'view_manager'):
                    self.view_manager.switch_to_view(view_mode)
                
                # Restore state for the new tab
                self._restore_tab_state(view_mode)
                
                # Handle display based on view mode
                if view_mode == ViewMode.FILTERING:
                    # Show only SEM image in filtering mode
                    self._switch_to_filtering_display()
                elif view_mode == ViewMode.ALIGNMENT:
                    # Show SEM + GDS overlay in alignment mode
                    self._switch_to_alignment_display()
                elif view_mode == ViewMode.SCORING:
                    # Show comparison results in scoring mode
                    self._switch_to_scoring_display()
                    
                self._update_panel_availability()
                
                # Sync view manager data after switching
                self._sync_view_manager_data()
                
        except Exception as e:
            print(f"Error handling tab change: {e}")
    
    def _switch_to_filtering_display(self):
        """Switch image viewer to filtering display mode (SEM only)."""
        try:
            if hasattr(self, 'image_viewer'):
                # Hide GDS overlay in filtering mode
                if hasattr(self.image_viewer, 'set_overlay_visible'):
                    self.image_viewer.set_overlay_visible(False)
                
                # Get the current filtered image from view manager data
                view_data = self.view_manager.get_view_data(ViewMode.FILTERING)
                filtered_image = view_data.get('filtered_image')
                
                # Show filtered image if available, otherwise show original SEM
                if filtered_image is not None:
                    self.image_viewer.set_sem_image(filtered_image)
                    # Update current reference to filtered image
                    self.current_sem_image = filtered_image
                elif self.current_sem_image is not None:
                    self.image_viewer.set_sem_image(self.current_sem_image)
                
                # Update histogram with current image
                if hasattr(self, 'histogram_view') and self.current_sem_image is not None:
                    self.histogram_view.update_histogram(self.current_sem_image)
                
                print("✓ Switched to filtering display mode")
                
        except Exception as e:
            print(f"Error switching to filtering display: {e}")
    
    def _switch_to_alignment_display(self):
        """Switch image viewer to alignment display mode (SEM + GDS)."""
        try:
            if hasattr(self, 'image_viewer'):
                # Ensure we're showing the original SEM image (not filtered) for alignment
                if hasattr(self, 'current_sem_image_obj') and self.current_sem_image_obj is not None:
                    # Get original unfiltered image
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
                # For scoring, we want to show the aligned result
                # This includes SEM image with aligned GDS overlay
                
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
                # Point was removed
                print(f"{point_type.upper()} point removed")
            else:
                # Point was added
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
            # For testing purposes, if no SEM image is loaded, use default bounds
            # In real usage, this would require a loaded SEM image
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
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            msg_box.exec()
        except Exception as e:
            print(f"Error showing message: {e}")
    
    def show_error(self, title: str, message: str):
        """Show error message to user."""
        try:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
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
    
    def _save_current_tab_state(self):
        """Save the current tab's state before switching to another tab."""
        try:
            if not hasattr(self, 'view_manager') or not hasattr(self, 'main_tab_widget'):
                return
            
            current_index = self.main_tab_widget.currentIndex()
            view_modes = [ViewMode.ALIGNMENT, ViewMode.FILTERING, ViewMode.SCORING]
            
            if 0 <= current_index < len(view_modes):
                current_view = view_modes[current_index]
                
                # Save state based on current view
                if current_view == ViewMode.ALIGNMENT:
                    self._save_alignment_state()
                elif current_view == ViewMode.FILTERING:
                    self._save_filtering_state()
                elif current_view == ViewMode.SCORING:
                    self._save_scoring_state()
                
                # Save common canvas state
                self._save_canvas_state(current_view)
                
        except Exception as e:
            print(f"Error saving tab state: {e}")
    
    def _save_alignment_state(self):
        """Save alignment tab specific state."""
        try:
            alignment_data = {}
            
            # Save manual alignment settings if available
            if hasattr(self, 'manual_alignment_controls'):
                # Get current transformation values from manual controls
                # This will be implemented based on the actual manual alignment component
                pass
            
            # Save hybrid alignment points
            if hasattr(self, 'sem_points') and hasattr(self, 'gds_points'):
                alignment_data['selected_points'] = {
                    'sem': self.sem_points.copy(),
                    'gds': self.gds_points.copy()
                }
            
            # Save current alignment result
            if self.current_alignment_result is not None:
                alignment_data['alignment_result'] = self.current_alignment_result.copy()
            else:
                alignment_data['alignment_result'] = None
            
            # Save which alignment mode is active (manual vs hybrid)
            if hasattr(self, 'alignment_sub_tabs'):
                alignment_data['mode'] = 'hybrid' if self.alignment_sub_tabs.currentIndex() == 1 else 'manual'
                if alignment_data['mode'] == 'hybrid' and hasattr(self, 'hybrid_sub_tabs'):
                    alignment_data['hybrid_sub_mode'] = self.hybrid_sub_tabs.currentIndex()
            
            # Update view manager
            self.view_manager.update_view_data(ViewMode.ALIGNMENT, alignment_data)
            
        except Exception as e:
            print(f"Error saving alignment state: {e}")
    
    def _save_filtering_state(self):
        """Save filtering tab specific state."""
        try:
            filtering_data = {}
            
            # Save filter history from image processing service
            if hasattr(self, 'image_processing_service') and self.image_processing_service is not None:
                if hasattr(self.image_processing_service, 'get_filter_history'):
                    filtering_data['filter_history'] = self.image_processing_service.get_filter_history()
                
                # Save current filtered image if different from original
                if hasattr(self.image_processing_service, 'current_image') and self.image_processing_service.current_image is not None:
                    filtering_data['filtered_image'] = self.image_processing_service.current_image
            
            # Save active filter selection if available
            if hasattr(self, 'filter_panel'):
                # Get current filter selection from filter panel
                # This will depend on the actual filter panel implementation
                pass
            
            # Save which filtering mode is active (manual vs automatic)
            if hasattr(self, 'filtering_sub_tabs'):
                filtering_data['filter_mode'] = 'automatic' if self.filtering_sub_tabs.currentIndex() == 1 else 'manual'
            
            # Update view manager
            self.view_manager.update_view_data(ViewMode.FILTERING, filtering_data)
            
        except Exception as e:
            print(f"Error saving filtering state: {e}")
    
    def _save_scoring_state(self):
        """Save scoring tab specific state."""
        try:
            scoring_data = {}
            
            # Save current scoring method and results
            scoring_data['scoring_method'] = self.current_scoring_method
            if self.current_scoring_results:
                scoring_data['scores'] = self.current_scoring_results.copy()
            
            # Save scoring tab panel state if available
            if hasattr(self, 'scoring_tab_panel'):
                scoring_data['selected_method'] = self.scoring_tab_panel.get_selected_method()
            
            # Update view manager
            self.view_manager.update_view_data(ViewMode.SCORING, scoring_data)
            
        except Exception as e:
            print(f"Error saving scoring state: {e}")
    
    def _save_canvas_state(self, view_mode):
        """Save canvas state for the current view mode."""
        try:
            if not hasattr(self, 'image_viewer'):
                return
            
            canvas_data = {}
            
            # Save zoom and pan state
            if hasattr(self.image_viewer, '_zoom_factor'):
                canvas_data['zoom_factor'] = self.image_viewer._zoom_factor
            if hasattr(self.image_viewer, '_pan_offset'):
                canvas_data['pan_offset'] = (self.image_viewer._pan_offset.x(), self.image_viewer._pan_offset.y())
            
            # Save overlay visibility and transparency
            if hasattr(self.image_viewer, '_overlay_visible'):
                canvas_data['overlay_visible'] = self.image_viewer._overlay_visible
            if hasattr(self.image_viewer, '_overlay_alpha'):
                canvas_data['overlay_alpha'] = self.image_viewer._overlay_alpha
            
            # Update view manager with canvas data
            current_data = self.view_manager.get_view_data(view_mode)
            current_data['canvas_state'] = canvas_data
            self.view_manager.update_view_data(view_mode, current_data)
            
        except Exception as e:
            print(f"Error saving canvas state: {e}")

    def _restore_tab_state(self, view_mode):
        """Restore saved state when switching to a tab."""
        try:
            # Get saved data for this view mode
            view_data = self.view_manager.get_view_data(view_mode)
            
            # Restore state based on view mode
            if view_mode == ViewMode.ALIGNMENT:
                self._restore_alignment_state(view_data)
            elif view_mode == ViewMode.FILTERING:
                self._restore_filtering_state(view_data)
            elif view_mode == ViewMode.SCORING:
                self._restore_scoring_state(view_data)
            
            # Restore canvas state
            self._restore_canvas_state(view_data)
            
        except Exception as e:
            print(f"Error restoring tab state: {e}")
    
    def _restore_alignment_state(self, view_data):
        """Restore alignment tab specific state."""
        try:
            # Restore selected points
            if 'selected_points' in view_data:
                points_data = view_data['selected_points']
                if 'sem' in points_data:
                    self.sem_points = points_data['sem'].copy()
                if 'gds' in points_data:
                    self.gds_points = points_data['gds'].copy()
                
                # Update UI display
                if hasattr(self, '_update_hybrid_status'):
                    self._update_hybrid_status()
            
            # Restore alignment result
            if 'alignment_result' in view_data and view_data['alignment_result'] is not None:
                self.current_alignment_result = view_data['alignment_result'].copy()
                
                # Apply alignment to image viewer
                if hasattr(self, 'image_viewer') and self.current_alignment_result:
                    self.image_viewer.set_alignment_result(self.current_alignment_result)
            
            # Restore alignment mode (manual vs hybrid)
            if 'mode' in view_data and hasattr(self, 'alignment_sub_tabs'):
                mode_index = 1 if view_data['mode'] == 'hybrid' else 0
                self.alignment_sub_tabs.setCurrentIndex(mode_index)
                
                # Restore hybrid sub-mode
                if view_data['mode'] == 'hybrid' and 'hybrid_sub_mode' in view_data:
                    if hasattr(self, 'hybrid_sub_tabs'):
                        self.hybrid_sub_tabs.setCurrentIndex(view_data['hybrid_sub_mode'])
            
        except Exception as e:
            print(f"Error restoring alignment state: {e}")
    
    def _restore_filtering_state(self, view_data):
        """Restore filtering tab specific state."""
        try:
            # Restore filtered image if available
            if 'filtered_image' in view_data and view_data['filtered_image'] is not None:
                # Apply the filtered image to the image viewer
                if hasattr(self, 'image_viewer'):
                    self.image_viewer.set_sem_image(view_data['filtered_image'])
                
                # Update current SEM image reference to filtered version
                self.current_sem_image = view_data['filtered_image']
                
                # Update histogram
                if hasattr(self, 'histogram_view'):
                    self.histogram_view.update_histogram(view_data['filtered_image'])
            
            # Restore filter mode (manual vs automatic)
            if 'filter_mode' in view_data and hasattr(self, 'filtering_sub_tabs'):
                mode_index = 1 if view_data['filter_mode'] == 'automatic' else 0
                self.filtering_sub_tabs.setCurrentIndex(mode_index)
            
        except Exception as e:
            print(f"Error restoring filtering state: {e}")
    
    def _restore_scoring_state(self, view_data):
        """Restore scoring tab specific state."""
        try:
            # Restore scoring method
            if 'scoring_method' in view_data:
                self.current_scoring_method = view_data['scoring_method']
                
                # Update scoring tab panel if available
                if hasattr(self, 'scoring_tab_panel'):
                    # Set the radio button selection to match saved method
                    method_key = view_data.get('selected_method', view_data['scoring_method'])
                    if method_key in self.scoring_tab_panel.method_radios:
                        self.scoring_tab_panel.method_radios[method_key].setChecked(True)
                        self.scoring_tab_panel.current_method = method_key
            
            # Restore scoring results
            if 'scores' in view_data:
                self.current_scoring_results = view_data['scores'].copy()
                
                # Display results in scoring tab panel
                if hasattr(self, 'scoring_tab_panel'):
                    self.scoring_tab_panel.display_results(self.current_scoring_results)
            
        except Exception as e:
            print(f"Error restoring scoring state: {e}")
    
    def _restore_canvas_state(self, view_data):
        """Restore canvas state for the view mode."""
        try:
            if not hasattr(self, 'image_viewer') or 'canvas_state' not in view_data:
                return
            
            canvas_data = view_data['canvas_state']
            
            # Restore zoom and pan
            if 'zoom_factor' in canvas_data:
                self.image_viewer._zoom_factor = canvas_data['zoom_factor']
            if 'pan_offset' in canvas_data:
                x, y = canvas_data['pan_offset']
                self.image_viewer._pan_offset = QPoint(x, y)
            
            # Restore overlay settings
            if 'overlay_visible' in canvas_data:
                self.image_viewer._overlay_visible = canvas_data['overlay_visible']
            if 'overlay_alpha' in canvas_data:
                self.image_viewer._overlay_alpha = canvas_data['overlay_alpha']
            
            # Trigger canvas update
            self.image_viewer.update()
            
        except Exception as e:
            print(f"Error restoring canvas state: {e}")

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

    # ...existing code...
