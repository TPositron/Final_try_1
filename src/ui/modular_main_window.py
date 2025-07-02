"""
Main Window - Core Module
Handles the core window functionality and coordinates other modules.
"""

import sys
from pathlib import Path
from PySide6.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QComboBox, QLabel, QStatusBar, QToolBar
from PySide6.QtCore import Qt

# Import our modular components
from .ui_setup import UISetup
from .file_operations import FileOperations  
from .gds_operations import GDSOperations
from .image_processing import ImageProcessing
from .alignment_operations import AlignmentOperations
from .scoring_operations import ScoringOperations
from .signal_handlers import SignalHandlers
from .view_controller import ViewController

# Import services
from src.services.simple_file_service import FileService
from src.services.simple_image_processing_service import ImageProcessingService
from src.services.simple_alignment_service import AlignmentService
from src.services.simple_scoring_service import ScoringService
from src.services.new_gds_service import NewGDSService
from src.services.transformation_service import TransformationService

# Import UI components
from src.ui.components.image_viewer import ImageViewer
from src.ui.view_manager import ViewManager, ViewMode
from src.ui.base_panels import ViewPanelManager

# Import core models
from src.core.models.structure_definitions import get_default_structures

# Configuration
DEFAULT_GDS_FILE = "Institute_Project_GDS1.gds"


class MainWindow(QMainWindow):
    """
    Main Window for the Image Analysis Tool.
    
    This class coordinates all the different modules and provides
    the main interface for the application.
    """
    
    def __init__(self):
        """Initialize the main window and all its components."""
        print("MainWindow constructor called")
        super().__init__()
        
        # Initialize core application data
        self.current_sem_image = None
        self.current_sem_image_obj = None
        self.current_sem_path = None
        
        # Initialize services
        self.file_service = FileService()
        
        # Initialize UI setup module
        self.ui_setup = UISetup(self)
        
        # Initialize operation modules
        self.file_operations = FileOperations(self)
        self.gds_operations = GDSOperations(self)
        self.image_processing = ImageProcessing(self)
        self.alignment_operations = AlignmentOperations(self)
        self.scoring_operations = ScoringOperations(self)
        
        # Initialize view controller
        self.view_controller = ViewController(self)
        
        # Initialize signal handlers
        self.signal_handlers = SignalHandlers(self)
        
        # Setup the main window
        self._setup_main_window()
        
        # Connect all signals
        self.signal_handlers.connect_all_signals()
        
        print("✓ MainWindow initialization completed")
    
    def _setup_main_window(self):
        """Setup the main window UI and components."""
        try:
            # Set window properties
            self.setWindowTitle("Image Analysis - SEM/GDS Alignment Tool")
            self.setGeometry(100, 100, 1400, 900)
            
            # Setup UI components
            self.ui_setup.setup_ui()
            self.ui_setup.setup_menu()
            self.ui_setup.setup_status_bar()
            
            # Initialize view system
            self.view_controller.initialize_view_system()
            
            # Initialize panel manager for view-specific panels
            self.panel_manager = ViewPanelManager(self)
            
            # Populate structure combo and auto-load GDS
            self.gds_operations.populate_structure_combo()
            
            print("✓ Main window setup completed")
            
        except Exception as e:
            print(f"Error setting up main window: {e}")
            raise
    
    def _update_panel_availability(self):
        """Update panel availability based on current application state."""
        try:
            # Update view availability
            self.view_controller.update_view_availability()
            
            # Update panel manager
            if hasattr(self, 'panel_manager'):
                self.panel_manager.update_panel_availability()
            
        except Exception as e:
            print(f"Error updating panel availability: {e}")
    
    # Delegate methods to operation modules
    
    def load_sem_image(self):
        """Load a SEM image file."""
        self.file_operations.load_sem_image()
    
    def load_gds_file(self):
        """Load a GDS file."""
        self.gds_operations.load_gds_file()
    
    def auto_align(self):
        """Perform automatic alignment."""
        self.alignment_operations.auto_align()
    
    def calculate_scores(self):
        """Calculate alignment scores."""
        self.scoring_operations.calculate_scores()
    
    def reset_alignment(self):
        """Reset alignment."""
        self.alignment_operations.reset_alignment()
    
    def save_results(self):
        """Save current results."""
        self.file_operations.save_results()
    
    # Utility methods for compatibility
    
    def on_file_loaded(self, file_type, file_path):
        """Handle file loaded event (compatibility method)."""
        # This is now handled by signal handlers
        pass
    
    def on_structure_selected(self, structure_name):
        """Handle structure selection (compatibility method)."""
        # This is now handled by GDS operations
        self.gds_operations.on_structure_selected(structure_name)
    
    def on_filter_applied(self, filter_name, parameters):
        """Handle filter applied event (compatibility method)."""
        # This is now handled by image processing
        self.image_processing.on_filter_applied(filter_name, parameters)
    
    def on_filter_preview(self, filter_name, parameters):
        """Handle filter preview event (compatibility method)."""
        # This is now handled by image processing
        self.image_processing.on_filter_preview(filter_name, parameters)
    
    def on_reset_filters(self):
        """Handle reset filters event (compatibility method)."""
        # This is now handled by image processing
        self.image_processing.on_reset_filters()
    
    def switch_view(self, view_mode: ViewMode):
        """Switch view mode (compatibility method)."""
        # This is now handled by view controller
        self.view_controller.switch_view(view_mode)
    
    def update_alignment_display(self):
        """Update alignment display (compatibility method)."""
        # This is now handled by alignment operations
        self.alignment_operations.update_alignment_display()
    
    # Properties for accessing UI components
    
    @property
    def structure_combo(self):
        """Get the structure combo box."""
        return getattr(self.ui_setup, 'structure_combo', None)
    
    @property
    def status_bar(self):
        """Get the status bar."""
        return getattr(self.ui_setup, 'status_bar', None)
    
    @property
    def image_viewer(self):
        """Get the image viewer."""
        return getattr(self.ui_setup, 'image_viewer', None)
    
    @property
    def view_toolbar(self):
        """Get the view toolbar."""
        return getattr(self.ui_setup, 'view_toolbar', None)
    
    @property
    def left_panel_layout(self):
        """Get the left panel layout."""
        return getattr(self.ui_setup, 'left_panel_layout', None)
    
    @property
    def view_specific_layout(self):
        """Get the view specific layout."""
        return getattr(self.ui_setup, 'view_specific_layout', None)
    
    def get_application_state(self):
        """Get the current application state."""
        return {
            'has_sem_image': self.current_sem_image is not None,
            'has_gds_file': self.gds_operations.is_gds_loaded(),
            'has_structure': self.gds_operations.is_structure_selected(),
            'has_alignment': self.alignment_operations.is_aligned(),
            'has_scores': self.scoring_operations.has_scores(),
            'current_view': str(self.view_controller.get_current_view()),
            'applied_filters': self.image_processing.get_applied_filters()
        }
    
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            print("Closing application...")
            
            # Cleanup view system
            self.view_controller.cleanup_view_system()
            
            # Disconnect signals
            self.signal_handlers.disconnect_all_signals()
            
            # Accept the close event
            event.accept()
            
            print("✓ Application closed successfully")
            
        except Exception as e:
            print(f"Error during application close: {e}")
            event.accept()  # Close anyway
        print("MainWindow constructor called")
        super().__init__()
        
        # Basic window setup
        self.setWindowTitle("Image Analysis - SEM/GDS Alignment Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize services
        self._initialize_services()
        
        # Initialize UI components
        self._initialize_ui_components()
        
        # Initialize data storage
        self._initialize_data_storage()
        
        # Initialize modular components
        self._initialize_modules()
        
        # Setup the complete UI
        self._setup_complete_ui()
        
        print("MainWindow initialization complete")
    
    def _initialize_services(self):
        """Initialize all service objects."""
        self.file_service = FileService()
        self.file_manager = self.file_service  # Alias for compatibility
        self.image_processing_service = ImageProcessingService()
        self.alignment_service = AlignmentService()
        self.scoring_service = ScoringService()
        self.new_gds_service = NewGDSService()
        self.transformation_service = TransformationService()
        
        # GDS loader will be initialized when needed
        self.simple_gds_loader = None
        
        print("Services initialized")
    
    def _initialize_ui_components(self):
        """Initialize core UI components."""
        # Main image viewer
        self.image_viewer = ImageViewer()
        
        # View management system
        self.view_manager = ViewManager(self)
        self.panel_manager = ViewPanelManager(self)
        
        # Structure management
        self.structure_manager = get_default_structures()
        
        print("UI components initialized")
    
    def _initialize_data_storage(self):
        """Initialize all data storage variables."""
        # SEM image data
        self.current_sem_image = None
        self.current_sem_image_obj = None
        self.current_sem_path = None
        
        # GDS data
        self.current_structures = {}
        self.current_structure_name = None
        self.current_gds_filename = None
        self.current_gds_filepath = None
        self.current_gds_overlay = None
        self.current_gds_model = None
        
        # Processing results
        self.current_alignment_result = None
        self.current_transformation = None
        self.current_scoring_method = "SSIM"
        self.current_scoring_results = {}
        
        # UI state
        self._processing_structure_selection = False
        self.overlay_renderer = None
        
        print("Data storage initialized")
    
    def _initialize_modules(self):
        """Initialize all modular components."""
        # Pass self to each module so they can access main window resources
        self.ui_setup = UISetup(self)
        self.file_operations = FileOperations(self)
        self.gds_operations = GDSOperations(self)
        self.image_processing = ImageProcessing(self)
        self.alignment_operations = AlignmentOperations(self)
        self.scoring_operations = ScoringOperations(self)
        self.view_controller = ViewController(self)
        
        # Signal handlers should be initialized last as it connects everything
        self.signal_handlers = SignalHandlers(self)
        
        print("Modules initialized")
    
    def _setup_complete_ui(self):
        """Setup the complete UI using the modular components."""
        # Setup UI structure
        self.ui_setup.setup_ui()
        self.ui_setup.setup_menu()
        self.ui_setup.setup_view_toolbar()
        self.ui_setup.setup_status_bar()
        
        # Connect all signals
        self.signal_handlers.connect_all_signals()
        
        # Initialize view system
        self.view_controller.initialize_view_system()
        
        # Auto-load default GDS if available
        self.file_operations.auto_load_default_gds()
        
        print("Complete UI setup finished")
    
    # Delegate methods to appropriate modules
    def load_sem_image(self):
        """Load SEM image (delegated to file_operations)."""
        return self.file_operations.load_sem_image()
    
    def load_gds_file(self):
        """Load GDS file (delegated to file_operations)."""
        return self.file_operations.load_gds_file()
    
    def on_structure_selected(self, structure_name):
        """Handle structure selection (delegated to gds_operations)."""
        return self.gds_operations.on_structure_selected(structure_name)
    
    def calculate_scores(self):
        """Calculate alignment scores (delegated to scoring_operations)."""
        return self.scoring_operations.calculate_scores()
    
    def save_results(self):
        """Save results (delegated to file_operations)."""
        return self.file_operations.save_results()
    
    def switch_view(self, view_mode: ViewMode):
        """Switch view (delegated to view_controller)."""
        return self.view_controller.switch_view(view_mode)
    
    # Utility methods that modules might need
    def validate_required_data(self, sem_required=False, gds_required=False, alignment_required=False):
        """Validate that required data is available."""
        from PySide6.QtWidgets import QMessageBox
        
        if sem_required and self.current_sem_image is None:
            QMessageBox.warning(self, "Warning", "SEM image is required for this operation")
            return False
        if gds_required and self.current_gds_overlay is None:
            QMessageBox.warning(self, "Warning", "GDS overlay is required for this operation")
            return False
        if alignment_required and self.current_alignment_result is None:
            QMessageBox.warning(self, "Warning", "Alignment result is required for this operation")
            return False
        return True
    
    def handle_service_error(self, operation_name: str, error: Exception):
        """Handle service errors in a consistent way."""
        from PySide6.QtWidgets import QMessageBox
        
        error_msg = f"{operation_name} failed: {str(error)}"
        print(f"Error in {operation_name}: {error}")
        QMessageBox.critical(self, "Error", error_msg)
        
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(f"Error: {operation_name} failed")


# Convenience function for creating the main window
def create_main_window():
    """Create and return a new MainWindow instance."""
    return MainWindow()
