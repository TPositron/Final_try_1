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
                               QToolBar, QButtonGroup, QGroupBox, QTextEdit)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QIcon

# Import UI components and services
from src.ui.components.image_viewer import ImageViewer
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
            
            # Setup left panel container
            self.left_panel_container = QWidget()
            self.left_panel_layout = QVBoxLayout(self.left_panel_container)
            self.left_panel_layout.setContentsMargins(0, 0, 0, 0)
            self.main_splitter.addWidget(self.left_panel_container)
            
            # Setup central image viewer
            self.image_viewer = ImageViewer()
            self.main_splitter.addWidget(self.image_viewer)
            
            # Setup right panel container
            self.right_panel_container = QWidget()
            self.right_panel_layout = QVBoxLayout(self.right_panel_container)
            
            # Common controls at the top of right panel
            self._setup_common_controls()
            
            # Add view-specific content area
            self.view_specific_widget = QWidget()
            self.view_specific_layout = QVBoxLayout(self.view_specific_widget)
            self.right_panel_layout.addWidget(self.view_specific_widget)
            
            self.main_splitter.addWidget(self.right_panel_container)
            
            # Set initial splitter sizes
            self.main_splitter.setSizes([280, 950, 250])
            
            # Setup menus and toolbars
            self._setup_menu()
            self._setup_toolbar()
            self._setup_status_bar()
            
            print("✓ UI setup completed")
            
        except Exception as e:
            print(f"Error setting up UI: {e}")
            raise
    
    def _setup_common_controls(self):
        """Setup common controls in the right panel."""
        try:
            # Create common controls group
            common_group = QGroupBox("Structure Selection")
            common_layout = QVBoxLayout(common_group)
            
            # Structure selection combo
            structure_label = QLabel("GDS Structure:")
            self.structure_combo = QComboBox()
            self.structure_combo.setEnabled(False)  # Initially disabled
            
            common_layout.addWidget(structure_label)
            common_layout.addWidget(self.structure_combo)
            
            # Add to right panel layout
            self.right_panel_layout.addWidget(common_group)
            
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
    
    def _setup_toolbar(self):
        """Setup the view selection toolbar."""
        try:
            # Create toolbar
            self.view_toolbar = QToolBar("View Selection")
            self.view_toolbar.setObjectName("ViewToolbar")
            self.view_toolbar.setMovable(False)
            self.addToolBar(Qt.TopToolBarArea, self.view_toolbar)
            
            # Create button group for exclusive selection
            self.view_button_group = QButtonGroup(self)
            self.view_button_group.setExclusive(True)
            
            # Define view buttons
            view_configs = [
                (ViewMode.ALIGNMENT, "Alignment", "Align SEM and GDS images"),
                (ViewMode.FILTERING, "Filtering", "Apply image filters"),
                (ViewMode.SCORING, "Scoring", "Calculate alignment scores")
            ]
            
            # Create buttons
            self.view_buttons = {}
            for view_mode, label, tooltip in view_configs:
                button = QPushButton(label)
                button.setCheckable(True)
                button.setToolTip(tooltip)
                button.clicked.connect(lambda checked, vm=view_mode: self.switch_view(vm))
                
                # Add to toolbar and button group
                self.view_toolbar.addWidget(button)
                self.view_button_group.addButton(button)
                self.view_buttons[view_mode] = button
            
            # Set initial view
            self.view_buttons[ViewMode.ALIGNMENT].setChecked(True)
            
        except Exception as e:
            print(f"Error setting up toolbar: {e}")
    
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
            
            # Connect structure combo signal after managers are initialized
            self.structure_combo.currentTextChanged.connect(
                self.gds_operations_manager.on_structure_selected
            )
            
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
            
            # Populate structure combo and auto-load GDS
            self.gds_operations_manager.populate_structure_combo()
            
            # Update panel availability
            self._update_panel_availability()
            
        except Exception as e:
            print(f"Error initializing view system: {e}")
    
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
                self.image_processing_manager.image_processing_service.set_original_image(
                    image_data['cropped_array']
                )
            
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
            self._update_panel_availability()
            
        except Exception as e:
            print(f"Error handling structure selected: {e}")
    
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
            # Update alignment display if needed
            if self.current_gds_overlay is not None:
                self.update_alignment_display()
                
        except Exception as e:
            print(f"Error handling filter applied: {e}")
    
    def on_filters_reset(self):
        """Handle filters reset signal."""
        try:
            print("Filters reset")
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
            
        except Exception as e:
            print(f"Error handling alignment completed: {e}")
    
    def on_alignment_reset(self):
        """Handle alignment reset signal."""
        try:
            print("Alignment reset")
            self.current_alignment_result = None
            self._update_panel_availability()
            
        except Exception as e:
            print(f"Error handling alignment reset: {e}")
    
    def on_scores_calculated(self, scores):
        """Handle scores calculated signal."""
        try:
            print(f"Scores calculated: {scores}")
            # Update current scoring results for compatibility
            self.current_scoring_results = scores
            
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
            
            # Update button states
            for vm, button in self.view_buttons.items():
                button.setChecked(vm == view_mode)
            
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
            # Update view button availability
            if hasattr(self, 'view_buttons'):
                # Alignment view is always available
                self.view_buttons[ViewMode.ALIGNMENT].setEnabled(True)
                
                # Filtering view requires SEM image
                has_sem = self.current_sem_image is not None
                self.view_buttons[ViewMode.FILTERING].setEnabled(has_sem)
                
                # Scoring view requires both SEM image and GDS structure
                has_gds = self.gds_operations_manager.is_structure_selected()
                self.view_buttons[ViewMode.SCORING].setEnabled(has_sem and has_gds)
            
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
