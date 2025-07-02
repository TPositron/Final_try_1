"""
Core Main Window Module
Coordinates all the modular components and provides the main application interface.
"""

import sys
from pathlib import Path
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QComboBox, QLabel, QStatusBar, QToolBar
from PySide6.QtCore import Qt

# Import our modular components
from .ui_setup import UISetup
from .file_handler import FileHandler
from .gds_manager import GDSManager
from .image_processor import ImageProcessor
from .alignment_controller import AlignmentController
from .scoring_calculator import ScoringCalculator

# Import UI components
from src.ui.components.image_viewer import ImageViewer
from src.ui.view_manager import ViewManager, ViewMode
from src.ui.base_panels import ViewPanelManager

# Import core models
from src.core.models.structure_definitions import get_default_structures

# Configuration
DEFAULT_GDS_FILE = "Institute_Project_GDS1.gds"


class CoreMainWindow(QMainWindow):
    """
    Core Main Window for the Image Analysis Tool.
    
    This class coordinates all the modular components and provides
    the main interface for the application.
    """
    
    def __init__(self):
        """Initialize the core main window and all its components."""
        print("CoreMainWindow constructor called")
        super().__init__()
        
        # Initialize core application data
        self.current_sem_image = None
        self.current_sem_image_obj = None
        self.current_sem_path = None
        
        # Initialize all modular components
        self._initialize_modules()
        
        # Setup the main window
        self._setup_main_window()
        
        # Connect all module signals
        self._connect_module_signals()
        
        # Initialize the application
        self._initialize_application()
        
        print("✓ CoreMainWindow initialization completed")
    
    def _initialize_modules(self):
        """Initialize all modular components."""
        try:
            print("Initializing modular components...")
            
            # Initialize UI setup module
            self.ui_setup = UISetup(self)
            
            # Initialize operation modules
            self.file_handler = FileHandler(self)
            self.gds_manager = GDSManager(self)
            self.image_processor = ImageProcessor(self)
            self.alignment_controller = AlignmentController(self)
            self.scoring_calculator = ScoringCalculator(self)
            
            # Initialize view manager and panel manager (will be created during UI setup)
            self.view_manager = None
            self.panel_manager = None
            
            print("✓ All modules initialized")
            
        except Exception as e:
            print(f"Error initializing modules: {e}")
            raise
    
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
            
            # Initialize view manager and panel manager
            self.view_manager = ViewManager(self)
            self.panel_manager = ViewPanelManager(self)
            
            print("✓ Main window setup completed")
            
        except Exception as e:
            print(f"Error setting up main window: {e}")
            raise
    
    def _connect_module_signals(self):
        """Connect signals between modules."""
        try:
            print("Connecting module signals...")
            
            # File handler signals
            self.file_handler.sem_image_loaded.connect(self.on_sem_image_loaded)
            self.file_handler.gds_file_loaded.connect(self.on_gds_file_loaded)
            self.file_handler.results_saved.connect(self.on_results_saved)
            
            # GDS manager signals
            self.gds_manager.gds_file_loaded.connect(self.on_gds_file_loaded)
            self.gds_manager.structure_selected.connect(self.on_structure_selected)
            self.gds_manager.structure_combo_updated.connect(self.on_structure_combo_updated)
            
            # Image processor signals
            self.image_processor.filter_applied.connect(self.on_filter_applied)
            self.image_processor.filters_reset.connect(self.on_filters_reset)
            self.image_processor.image_processed.connect(self.on_image_processed)
            
            # Alignment controller signals
            self.alignment_controller.alignment_completed.connect(self.on_alignment_completed)
            self.alignment_controller.alignment_reset.connect(self.on_alignment_reset)
            self.alignment_controller.transformation_applied.connect(self.on_transformation_applied)
            
            # Scoring calculator signals
            self.scoring_calculator.scores_calculated.connect(self.on_scores_calculated)
            self.scoring_calculator.batch_scoring_completed.connect(self.on_batch_scoring_completed)
            
            # UI component signals
            if hasattr(self.ui_setup, 'structure_combo'):
                self.ui_setup.structure_combo.currentTextChanged.connect(
                    self.gds_manager.on_structure_selected
                )
            
            print("✓ All module signals connected")
            
        except Exception as e:
            print(f"Error connecting module signals: {e}")
    
    def _initialize_application(self):
        """Initialize the application with default settings."""
        try:
            print("Initializing application...")
            
            # Populate structure combo and auto-load GDS
            self.gds_manager.populate_structure_combo()
            self.gds_manager.auto_load_default_gds()
            
            # Set initial view mode
            self._switch_to_view(ViewMode.ALIGNMENT)
            
            # Update panel availability
            self._update_panel_availability()
            
            print("✓ Application initialization completed")
            
        except Exception as e:
            print(f"Error during application initialization: {e}")
    
    # Signal Handler Methods
    
    def on_sem_image_loaded(self, sem_image_obj, file_path):
        """Handle SEM image loaded signal."""
        try:
            print(f"SEM image loaded: {Path(file_path).name}")
            
            # Update panel availability
            self._update_panel_availability()
            
            # Switch to best available view
            self._switch_to_best_available_view()
            
        except Exception as e:
            print(f"Error handling SEM image loaded: {e}")
    
    def on_gds_file_loaded(self, file_path):
        """Handle GDS file loaded signal."""
        try:
            print(f"GDS file loaded: {Path(file_path).name}")
            
            # Update panel availability
            self._update_panel_availability()
            
        except Exception as e:
            print(f"Error handling GDS file loaded: {e}")
    
    def on_structure_selected(self, structure_name, overlay):
        """Handle structure selected signal."""
        try:
            print(f"Structure selected: {structure_name}")
            
            # Update panel availability
            self._update_panel_availability()
            
            # Switch to best available view
            self._switch_to_best_available_view()
            
        except Exception as e:
            print(f"Error handling structure selected: {e}")
    
    def on_structure_combo_updated(self, structure_names):
        """Handle structure combo updated signal."""
        try:
            print(f"Structure combo updated with {len(structure_names)} structures")
            
        except Exception as e:
            print(f"Error handling structure combo updated: {e}")
    
    def on_filter_applied(self, filter_name, parameters):
        """Handle filter applied signal."""
        try:
            print(f"Filter applied: {filter_name}")
            
            # Update alignment and scoring if they exist
            if self.alignment_controller.is_aligned():
                # Re-calculate alignment with new filtered image
                self.alignment_controller.update_alignment_display()
            
        except Exception as e:
            print(f"Error handling filter applied: {e}")
    
    def on_filters_reset(self):
        """Handle filters reset signal."""
        try:
            print("Filters reset")
            
            # Update dependent modules
            if self.alignment_controller.is_aligned():
                self.alignment_controller.update_alignment_display()
            
        except Exception as e:
            print(f"Error handling filters reset: {e}")
    
    def on_image_processed(self, processed_image):
        """Handle image processed signal."""
        try:
            print("Image processed")
            
            # Update current image reference
            self.current_sem_image = processed_image
            
        except Exception as e:
            print(f"Error handling image processed: {e}")
    
    def on_alignment_completed(self, alignment_result):
        """Handle alignment completed signal."""
        try:
            method = alignment_result.get('method', 'unknown')
            score = alignment_result.get('score', 'N/A')
            print(f"Alignment completed: {method} (score: {score})")
            
            # Update panel availability
            self._update_panel_availability()
            
            # Auto-calculate scores if in scoring view
            current_view = self._get_current_view()
            if current_view == ViewMode.SCORING:
                self.scoring_calculator.calculate_scores()
            
        except Exception as e:
            print(f"Error handling alignment completed: {e}")
    
    def on_alignment_reset(self):
        """Handle alignment reset signal."""
        try:
            print("Alignment reset")
            
            # Clear scores since alignment changed
            self.scoring_calculator.clear_scores()
            
            # Update panel availability
            self._update_panel_availability()
            
        except Exception as e:
            print(f"Error handling alignment reset: {e}")
    
    def on_transformation_applied(self, transformation):
        """Handle transformation applied signal."""
        try:
            print("Transformation applied")
            
            # Clear scores since transformation changed
            self.scoring_calculator.clear_scores()
            
        except Exception as e:
            print(f"Error handling transformation applied: {e}")
    
    def on_scores_calculated(self, scores):
        """Handle scores calculated signal."""
        try:
            print(f"Scores calculated: {list(scores.keys())}")
            
            # Update UI panels that show scoring results
            self._update_scoring_display(scores)
            
        except Exception as e:
            print(f"Error handling scores calculated: {e}")
    
    def on_batch_scoring_completed(self, batch_results):
        """Handle batch scoring completed signal."""
        try:
            successful = sum(1 for result in batch_results if result.get('success', False))
            total = len(batch_results)
            print(f"Batch scoring completed: {successful}/{total} successful")
            
        except Exception as e:
            print(f"Error handling batch scoring completed: {e}")
    
    def on_results_saved(self, file_path):
        """Handle results saved signal."""
        try:
            print(f"Results saved to: {file_path}")
            
        except Exception as e:
            print(f"Error handling results saved: {e}")
    
    # Public Interface Methods (for compatibility with existing code)
    
    def load_sem_image(self):
        """Load a SEM image file."""
        return self.file_handler.load_sem_image()
    
    def load_gds_file(self):
        """Load a GDS file."""
        return self.file_handler.load_gds_file()
    
    def auto_align(self):
        """Perform automatic alignment."""
        return self.alignment_controller.auto_align()
    
    def calculate_scores(self):
        """Calculate alignment scores."""
        return self.scoring_calculator.calculate_scores()
    
    def reset_alignment(self):
        """Reset alignment."""
        return self.alignment_controller.reset_alignment()
    
    def save_results(self):
        """Save current results."""
        return self.file_handler.save_results()
    
    def on_filter_applied_compat(self, filter_name, parameters):
        """Handle filter applied event (compatibility method)."""
        return self.image_processor.on_filter_applied(filter_name, parameters)
    
    def on_filter_preview(self, filter_name, parameters):
        """Handle filter preview event (compatibility method)."""
        return self.image_processor.on_filter_preview(filter_name, parameters)
    
    def on_reset_filters(self):
        """Handle reset filters event (compatibility method)."""
        return self.image_processor.on_reset_filters()
    
    def switch_view(self, view_mode: ViewMode):
        """Switch view mode."""
        self._switch_to_view(view_mode)
    
    def update_alignment_display(self):
        """Update alignment display."""
        return self.alignment_controller.update_alignment_display()
    
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
    
    # Helper Methods
    
    def _update_panel_availability(self):
        """Update panel availability based on current application state."""
        try:
            # Update panel manager if available
            if self.panel_manager:
                self.panel_manager.update_panel_availability()
            
        except Exception as e:
            print(f"Error updating panel availability: {e}")
    
    def _switch_to_view(self, view_mode: ViewMode):
        """Switch to a specific view mode."""
        try:
            if self.view_manager:
                self.view_manager.switch_view(view_mode)
            
            # Update panel manager
            if self.panel_manager:
                self.panel_manager.switch_to_view(view_mode)
            
        except Exception as e:
            print(f"Error switching to view {view_mode}: {e}")
    
    def _switch_to_best_available_view(self):
        """Switch to the best available view based on current state."""
        try:
            # Priority: Scoring (if aligned), Filtering (if SEM loaded), Alignment
            if (self.current_sem_image is not None and 
                self.gds_manager.is_structure_selected() and
                self.alignment_controller.is_aligned()):
                self._switch_to_view(ViewMode.SCORING)
            elif self.current_sem_image is not None:
                self._switch_to_view(ViewMode.FILTERING)
            else:
                self._switch_to_view(ViewMode.ALIGNMENT)
                
        except Exception as e:
            print(f"Error switching to best available view: {e}")
    
    def _get_current_view(self):
        """Get the current view mode."""
        if self.view_manager:
            return self.view_manager.current_view
        return ViewMode.ALIGNMENT
    
    def _update_scoring_display(self, scores):
        """Update the scoring display in UI panels."""
        try:
            # Update scoring panel if available
            if self.panel_manager:
                scoring_panel = self.panel_manager.left_panels.get(ViewMode.SCORING)
                if scoring_panel and hasattr(scoring_panel, 'update_scores'):
                    scoring_panel.update_scores(scores)
            
        except Exception as e:
            print(f"Error updating scoring display: {e}")
    
    def get_application_state(self):
        """Get the current application state."""
        return {
            'has_sem_image': self.current_sem_image is not None,
            'has_gds_file': self.gds_manager.is_gds_loaded(),
            'has_structure': self.gds_manager.is_structure_selected(),
            'has_alignment': self.alignment_controller.is_aligned(),
            'has_scores': self.scoring_calculator.has_scores(),
            'current_view': str(self._get_current_view()),
            'applied_filters': self.image_processor.get_applied_filters(),
            'module_status': {
                'file_handler': 'active',
                'gds_manager': 'active',
                'image_processor': 'active',
                'alignment_controller': 'active',
                'scoring_calculator': 'active'
            }
        }
    
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            print("Closing application...")
            
            # Cleanup modules
            self.file_handler.clear_file_data()
            self.image_processor.clear_processing_data()
            self.alignment_controller.clear_alignment_data()
            self.scoring_calculator.clear_scores()
            
            # Accept the close event
            event.accept()
            
            print("✓ Application closed successfully")
            
        except Exception as e:
            print(f"Error during application close: {e}")
            event.accept()  # Close anyway
