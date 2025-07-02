"""
Main Window
Assembles menus, panels, and layouts for the Image Analysis application.
"""

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QMenuBar, QStatusBar, QStackedWidget, QFileDialog,
                               QMessageBox, QApplication, QProgressBar)
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QAction, QKeySequence
from .panels.mode_switcher import ModeSwitcher
from .panels.alignment_panel import AlignmentPanel
from .panels.filter_panel import FilterPanel
from .panels.score_panel import ScorePanel
from .components.file_selector import FileSelector
from ..services.simple_file_service import FileService
import sys
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.current_mode = "Manual"
        
        # Setup error handling first
        self.setup_error_handling()
        
        # Initialize file service
        self.file_service = FileService(self)
        
        # Current loaded data
        self.current_sem_data = None
        self.current_gds_data = None
        self.current_sem_path = None
        self.current_gds_path = None
        self.current_structure_id = 1  # Default structure
        
        self.setup_ui()
        self.setup_menus()
        self.setup_statusbar()
        self.connect_signals()
        self.connect_file_service_signals()
        self.setup_alignment_parameter_flow()
        self.setup_error_handling()
        
    def setup_ui(self):
        """Initialize the main UI components."""
        self.setWindowTitle("Image Analysis Tool")
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Top section: File selector and mode switcher
        top_layout = QHBoxLayout()
        
        # File selector
        self.file_selector = FileSelector()
        top_layout.addWidget(self.file_selector)
        
        # Mode switcher
        self.mode_switcher = ModeSwitcher()
        top_layout.addWidget(self.mode_switcher)
        
        layout.addLayout(top_layout)
        
        # Main content area with stacked widget for different modes
        self.stacked_widget = QStackedWidget()
        
        # Create panels for each mode
        self.alignment_panel = AlignmentPanel()
        self.filter_panel = FilterPanel()
        self.score_panel = ScorePanel()
        
        # Add panels to stacked widget
        self.stacked_widget.addWidget(self.alignment_panel)  # Index 0: Manual/Auto mode
        self.stacked_widget.addWidget(self.filter_panel)     # Index 1: Filter mode  
        self.stacked_widget.addWidget(self.score_panel)      # Index 2: Score mode
        
        layout.addWidget(self.stacked_widget)
        
    def setup_menus(self):
        """Setup application menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        # Open SEM action
        open_sem_action = QAction("&Open SEM Image...", self)
        open_sem_action.setShortcut(QKeySequence.Open)
        open_sem_action.triggered.connect(self.open_sem_file)
        file_menu.addAction(open_sem_action)
        
        # Open GDS action
        open_gds_action = QAction("Open &GDS File...", self)
        open_gds_action.setShortcut(QKeySequence("Ctrl+G"))
        open_gds_action.triggered.connect(self.open_gds_file)
        file_menu.addAction(open_gds_action)
        
        file_menu.addSeparator()
        
        # Save results action
        save_action = QAction("&Save Results...", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        # Fit to window action
        fit_action = QAction("&Fit to Window", self)
        fit_action.setShortcut(QKeySequence("Ctrl+0"))
        fit_action.triggered.connect(self.fit_to_window)
        view_menu.addAction(fit_action)
        
        # Actual size action
        actual_size_action = QAction("&Actual Size", self)
        actual_size_action.setShortcut(QKeySequence("Ctrl+1"))
        actual_size_action.triggered.connect(self.actual_size)
        view_menu.addAction(actual_size_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        # Auto alignment action
        auto_align_action = QAction("&Auto Alignment", self)
        auto_align_action.setShortcut(QKeySequence("Ctrl+A"))
        auto_align_action.triggered.connect(self.run_auto_alignment)
        tools_menu.addAction(auto_align_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        # About action
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_statusbar(self):
        """Setup status bar with progress indication."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add progress bar for file operations
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        self.status_bar.showMessage("Ready")
        
    def connect_signals(self):
        """Connect UI component signals."""
        # Mode switcher
        self.mode_switcher.mode_changed.connect(self.change_mode)
        
        # File selector
        self.file_selector.sem_file_selected.connect(self.load_sem_file)
        self.file_selector.gds_file_loaded.connect(self.load_gds_file)                    # Default load
        self.file_selector.gds_structure_selected.connect(self.load_gds_file_with_structure)  # Specific structure
        
    def connect_file_service_signals(self):
        """Connect file service signals for progress and status updates."""
        # File loading progress
        self.file_service.loading_progress.connect(self.update_loading_progress)
        self.file_service.file_loaded.connect(self.on_file_loaded)
        self.file_service.loading_error.connect(self.on_loading_error)
        self.file_service.file_saved.connect(self.on_file_saved)
        self.file_service.error_occurred.connect(self.on_error_occurred)
        
    def update_loading_progress(self, message: str):
        """Update loading progress in status bar."""
        self.status_bar.showMessage(message)
        self.progress_bar.setVisible(True)
        
    def on_file_loaded(self, file_type: str, file_path: str):
        """Handle successful file loading."""
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"{file_type} file loaded: {file_path}")
        
    def on_loading_error(self, error_message: str):
        """Handle file loading errors."""
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Loading failed")
        QMessageBox.critical(self, "Loading Error", error_message)
        
    def on_file_saved(self, file_type: str, file_path: str):
        """Handle successful file saving."""
        self.status_bar.showMessage(f"{file_type} saved: {file_path}")
        
    def on_error_occurred(self, error_message: str):
        """Handle general errors."""
        QMessageBox.warning(self, "Error", error_message)
        
    def connect_file_service_signals(self):
        """Connect file service signals for progress and status updates."""
        # File loading progress
        self.file_service.loading_progress.connect(self.update_loading_progress)
        self.file_service.file_loaded.connect(self.on_file_loaded)
        self.file_service.loading_error.connect(self.on_loading_error)
        self.file_service.file_saved.connect(self.on_file_saved)
        self.file_service.error_occurred.connect(self.on_error_occurred)
        
    def update_loading_progress(self, message: str):
        """Update loading progress in status bar."""
        self.status_bar.showMessage(message)
        self.progress_bar.setVisible(True)
        
    def on_file_loaded(self, file_type: str, file_path: str):
        """Handle successful file loading."""
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"{file_type} file loaded: {file_path}")
        
    def on_loading_error(self, error_message: str):
        """Handle file loading errors."""
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Loading failed")
        QMessageBox.critical(self, "Loading Error", error_message)
        
    def on_file_saved(self, file_type: str, file_path: str):
        """Handle successful file saving."""
        self.status_bar.showMessage(f"{file_type} saved: {file_path}")
        
    def on_error_occurred(self, error_message: str):
        """Handle general errors."""
        QMessageBox.warning(self, "Error", error_message)

    def change_mode(self, mode: str):
        """Change the current application mode."""
        self.current_mode = mode
        
        # Switch to appropriate panel
        if mode in ["Manual", "Auto"]:
            self.stacked_widget.setCurrentIndex(0)  # Alignment panel
        elif mode == "Filter":
            self.stacked_widget.setCurrentIndex(1)  # Filter panel
        elif mode == "Score":
            self.stacked_widget.setCurrentIndex(2)  # Score panel
            
        self.status_bar.showMessage(f"Mode: {mode}")
        
    def load_sem_file(self, filepath: str):
        """Load SEM file using file service with error handling."""
        def _load_sem():
            from pathlib import Path
            result = self.file_service.load_sem_file(Path(filepath))
            if result:
                self.current_sem_data = result
                self.current_sem_path = filepath
                logger.info(f"SEM file loaded successfully: {filepath}")
                
                # Update alignment panel with SEM data
                if hasattr(self, 'alignment_panel'):
                    self.alignment_panel.set_initial_sem_image(result['cropped_array'])
                return result
            else:
                raise ValueError(f"Failed to load SEM file: {filepath}")
        
        return self.safe_operation(_load_sem, "SEM Loading")
            
    def load_gds_file(self, filepath: str):
        """Load GDS file with default structure 1."""
        print(f"Loading GDS file with default Structure 1: {filepath}")
        self.load_gds_file_with_structure(filepath, structure_id=1)

    def load_gds_file_with_structure(self, filepath: str, structure_id: int):
        """Load ONLY the specified structure region from GDS file."""
        print(f"\n=== LOADING STRUCTURE {structure_id} ===")
        print(f"File: {filepath}")
        
        def _load_gds():
            from pathlib import Path
            result = self.file_service.load_gds_file(Path(filepath), structure_id=structure_id)
            if result:
                self.current_gds_data = result
                self.current_gds_path = filepath
                self.current_structure_id = structure_id
                
                print(f"Structure Name: {result.get('structure_name', 'Unknown')}")
                print(f"Structure Bounds: {result['extracted_structure']['frame_data']['bounds']}")
                print(f"Polygon Count: {result['extracted_structure']['frame_data']['polygon_count']}")
                print(f"Binary Image Shape: {result['extracted_structure']['binary_image'].shape}")
                
                import numpy as np
                print(f"Non-zero pixels: {np.count_nonzero(result['extracted_structure']['binary_image'])}")
                
                logger.info(f"GDS file loaded successfully: {filepath}, structure: {structure_id}")
                
                # Update alignment panel with initial GDS data
                if hasattr(self, 'alignment_panel') and 'extracted_structure' in result:
                    binary_image = result['extracted_structure']['binary_image']
                    self.alignment_panel.set_initial_gds_image(binary_image)
                    
                print(f"Structure {structure_id} image displayed successfully")
                print("================================\n")
                    
                return result
            else:
                raise ValueError(f"Failed to load GDS structure {structure_id}: {filepath}")
        
        return self.safe_operation(_load_gds, "GDS Structure Loading")
        """Create aligned GDS image using current loaded data with error handling."""
        def _create_aligned():
            if not self.current_gds_data:
                raise ValueError("No GDS file loaded")
                
            # Add alignment type to parameters
            alignment_params['alignment_type'] = self.current_mode.lower()
            
            # Create aligned GDS image
            result = self.file_service.complete_aligned_gds_workflow(
                structure_id=self.current_structure_id,
                alignment_params=alignment_params,
                sem_image_path=self.current_sem_path,
                save_results=True
            )
            
            if result and result.get('success', False):
                logger.info("Aligned GDS image created and saved successfully")
                
                # Update alignment panel display
                if hasattr(self, 'alignment_panel') and 'aligned_data' in result:
                    aligned_image = result['aligned_data']['aligned_binary_image']
                    self.alignment_panel.set_aligned_gds_image(aligned_image)
                
                return result
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                raise ValueError(error_msg)
        
        return self.safe_operation(_create_aligned, "Aligned GDS Creation")
            
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
        """Save current alignment results."""
        try:
            if not self.current_gds_data or not self.current_sem_data:
                QMessageBox.warning(self, "Warning", "Both SEM and GDS files must be loaded to save results")
                return
                
            # Get current alignment parameters
            current_params = self.get_current_alignment_parameters()
            current_params['alignment_type'] = self.current_mode.lower()
            
            # Create and save aligned GDS image
            result = self.create_aligned_gds_image(current_params)
            
            if result and result.get('success', False):
                saved_files = result.get('saved_files', {})
                if saved_files:
                    file_list = '\n'.join([f"{k}: {v.name}" for k, v in saved_files.items()])
                    QMessageBox.information(self, "Success", 
                                          f"Results saved successfully!\n\nFiles saved:\n{file_list}")
                else:
                    QMessageBox.information(self, "Success", "Results saved successfully!")
            else:
                QMessageBox.critical(self, "Error", "Failed to save results")
                
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save results: {e}")
            
    def fit_to_window(self):
        """Fit content to window."""
        if self.current_mode in ["Manual", "Auto"]:
            self.alignment_panel.fit_canvas_to_view()
            
    def actual_size(self):
        """Show content at actual size."""
        if self.current_mode in ["Manual", "Auto"]:
            self.alignment_panel.reset_canvas_zoom()
            
    def run_auto_alignment(self):
        """Run automatic alignment using auto alignment service with error handling."""
        def _run_auto_alignment():
            if not self.current_sem_data or not self.current_gds_data:
                raise ValueError("Both SEM and GDS files must be loaded for auto alignment")
                
            # Import auto alignment service
            from ..services.transformations.auto_alignment_service import AutoAlignmentService
            
            self.status_bar.showMessage("Running auto alignment...")
            
            # Get image arrays
            sem_image = self.current_sem_data['cropped_array']
            gds_image = self.current_gds_data['extracted_structure']['binary_image']
            
            # Create auto alignment service
            auto_service = AutoAlignmentService()
            
            # Run auto alignment
            result = auto_service.auto_align_images(sem_image, gds_image)
            
            if result.get('success', False):
                # Extract transformation parameters
                transformation_matrix = result.get('transformation_matrix')
                if transformation_matrix is not None:
                    # Convert matrix to alignment parameters
                    alignment_params = self.extract_alignment_params_from_matrix(transformation_matrix)
                    alignment_params['alignment_type'] = 'auto'
                    
                    # Create aligned GDS image
                    aligned_result = self.create_aligned_gds_image(alignment_params)
                    
                    if aligned_result:
                        quality_score = result.get('quality_score', 0.0)
                        self.status_bar.showMessage(f"Auto alignment completed! Quality: {quality_score:.3f}")
                        
                        QMessageBox.information(self, "Success", 
                                              f"Auto alignment completed!\nQuality score: {quality_score:.3f}")
                        return aligned_result
                    else:
                        raise ValueError("Failed to create aligned GDS image")
                else:
                    raise ValueError("Failed to extract transformation parameters from alignment result")
            else:
                error_msg = result.get('error', 'Unknown error')
                raise ValueError(error_msg)
        
        return self.safe_operation(_run_auto_alignment, "Auto Alignment")
            
    def extract_alignment_params_from_matrix(self, matrix):
        """Extract alignment parameters from transformation matrix."""
        try:
            import numpy as np
            
            # Extract translation
            translation = (float(matrix[0, 2]), float(matrix[1, 2]))
            
            # Extract scale and rotation
            a, b = matrix[0, 0], matrix[0, 1]
            c, d = matrix[1, 0], matrix[1, 1]
            
            # Calculate scale
            scale_x = np.sqrt(a*a + b*b)
            scale_y = np.sqrt(c*c + d*d)
            zoom = (scale_x + scale_y) / 2
            
            # Calculate rotation in degrees
            rotation = np.arctan2(b, a) * 180 / np.pi
            
            return {
                'translation': translation,
                'zoom': zoom,
                'rotation': rotation
            }
            
        except Exception as e:
            logger.error(f"Error extracting alignment parameters: {e}")
            return {'translation': (0, 0), 'zoom': 1.0, 'rotation': 0.0}
            
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Image Analysis Tool",
            "Image Analysis Tool\n\n"
            "A tool for aligning SEM images with GDS layouts\n"
            "and analyzing their correspondence.\n\n"
            "Version 1.0"
        )
        
    def closeEvent(self, event):
        """Handle application close event."""
        reply = QMessageBox.question(
            self, "Confirm Exit",
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def setup_alignment_parameter_flow(self):
        """Setup proper alignment parameter flow between UI components."""
        try:
            # Connect alignment panel signals if available
            if hasattr(self, 'alignment_panel') and self.alignment_panel:
                # Connect new transform signals
                if hasattr(self.alignment_panel, 'transform_changed'):
                    self.alignment_panel.transform_changed.connect(self.on_transform_changed)
                if hasattr(self.alignment_panel, 'alignment_applied'):
                    self.alignment_panel.alignment_applied.connect(self.on_alignment_applied)
                if hasattr(self.alignment_panel, 'reset_requested'):
                    self.alignment_panel.reset_requested.connect(self.on_alignment_reset)
                
                # Connect file selection signals
                if hasattr(self.alignment_panel, 'sem_file_selected'):
                    self.alignment_panel.sem_file_selected.connect(self.load_sem_file)
                if hasattr(self.alignment_panel, 'gds_file_loaded'):
                    self.alignment_panel.gds_file_loaded.connect(self.load_gds_file)
                if hasattr(self.alignment_panel, 'gds_structure_selected'):
                    self.alignment_panel.gds_structure_selected.connect(self.load_gds_file_with_structure)
                
                logger.info("Alignment parameter flow setup complete")
            else:
                logger.warning("Alignment panel not available for parameter flow setup")
                
        except Exception as e:
            logger.error(f"Failed to setup alignment parameter flow: {e}")

    def on_transform_changed(self, transform_params: dict):
        """Handle transform parameter changes from alignment panel."""
        try:
            # Update real-time display with new transform parameters
            self.update_aligned_gds_real_time(transform_params)
            logger.debug(f"Transform parameters updated: {transform_params}")
        except Exception as e:
            logger.error(f"Error updating transform parameters: {e}")

    def on_alignment_applied(self, transform_params: dict):
        """Handle alignment application from alignment panel."""
        try:
            # Apply final alignment with current parameters
            self.apply_final_alignment(transform_params)
            self.status_bar.showMessage("Alignment applied successfully")
            logger.info(f"Alignment applied with parameters: {transform_params}")
        except Exception as e:
            logger.error(f"Error applying alignment: {e}")
            self.status_bar.showMessage("Error applying alignment")

    def on_alignment_reset(self):
        """Handle alignment reset from alignment panel."""
        try:
            # Reset alignment to default state
            self.reset_alignment_state()
            self.status_bar.showMessage("Alignment reset")
            logger.info("Alignment reset requested")
        except Exception as e:
            logger.error(f"Error resetting alignment: {e}")
            self.status_bar.showMessage("Error resetting alignment")

    def on_translation_x_changed(self, value: float):
        """Handle translation X parameter change."""
        try:
            current_params = self.get_current_alignment_parameters()
            current_params['translation'] = (value, current_params['translation'][1])
            self.update_aligned_gds_real_time(current_params)
            logger.debug(f"Translation X updated to: {value}")
        except Exception as e:
            logger.error(f"Error updating translation X: {e}")

    def on_translation_y_changed(self, value: float):
        """Handle translation Y parameter change."""
        try:
            current_params = self.get_current_alignment_parameters()
            current_params['translation'] = (current_params['translation'][0], value)
            self.update_aligned_gds_real_time(current_params)
            logger.debug(f"Translation Y updated to: {value}")
        except Exception as e:
            logger.error(f"Error updating translation Y: {e}")

    def on_rotation_changed(self, value: float):
        """Handle rotation parameter change."""
        try:
            current_params = self.get_current_alignment_parameters()
            current_params['rotation'] = value
            self.update_aligned_gds_real_time(current_params)
            logger.debug(f"Rotation updated to: {value}")
        except Exception as e:
            logger.error(f"Error updating rotation: {e}")

    def on_scale_changed(self, value: float):
        """Handle scale parameter change."""
        try:
            current_params = self.get_current_alignment_parameters()
            current_params['zoom'] = value
            self.update_aligned_gds_real_time(current_params)
            logger.debug(f"Scale updated to: {value}")
        except Exception as e:
            logger.error(f"Error updating scale: {e}")

    def on_reset_alignment_clicked(self):
        """Handle alignment reset."""
        try:
            # Reset to default parameters
            default_params = {
                'translation': (0.0, 0.0),
                'zoom': 1.0,
                'rotation': 0.0,
                'alignment_type': 'manual'
            }
            
            # Update alignment panel controls if available
            if hasattr(self, 'alignment_panel'):
                self.alignment_panel.reset_controls()
            
            # Update aligned GDS display
            self.update_aligned_gds_real_time(default_params)
            
            logger.info("Alignment parameters reset to default")
            self.status_bar.showMessage("Alignment reset to default values")
            
        except Exception as e:
            logger.error(f"Error resetting alignment: {e}")

    def get_current_alignment_parameters(self) -> Dict[str, Any]:
        """Get current alignment parameters."""
        try:
            # Try to get from alignment panel if available
            if hasattr(self, 'alignment_panel') and hasattr(self.alignment_panel, 'get_current_parameters'):
                return self.alignment_panel.get_current_parameters()
            else:
                # Return default parameters
                return {
                    'translation': (0.0, 0.0),
                    'zoom': 1.0,
                    'rotation': 0.0,
                    'alignment_type': 'manual'
                }
        except Exception as e:
            logger.error(f"Error getting current alignment parameters: {e}")
            return {}

    def set_alignment_parameters(self, params: Dict[str, Any]):
        """Set alignment parameters in UI controls."""
        try:
            if not self.validate_alignment_parameters(params):
                logger.error("Invalid alignment parameters provided")
                return False
            
            # Update alignment panel controls if available
            if hasattr(self, 'alignment_panel') and hasattr(self.alignment_panel, 'set_parameters'):
                self.alignment_panel.set_parameters(params)
            
            # Update aligned GDS display
            self.update_aligned_gds_real_time(params)
            
            logger.info(f"Alignment parameters set: {params}")
            return True
                
        except Exception as e:
            logger.error(f"Error setting alignment parameters: {e}")
            return False

    def validate_alignment_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate alignment parameters before processing."""
        try:
            required_keys = ['translation', 'zoom', 'rotation']
            
            # Check all required keys exist
            for key in required_keys:
                if key not in params:
                    logger.error(f"Missing required parameter: {key}")
                    return False
            
            # Validate parameter ranges
            translation = params['translation']
            zoom = params['zoom']
            rotation = params['rotation']
            
            # Check reasonable ranges
            if abs(translation[0]) > 2000 or abs(translation[1]) > 2000:
                logger.error(f"Translation values too large: {translation}")
                return False
            
            if zoom < 0.1 or zoom > 10.0:
                logger.error(f"Zoom value out of range: {zoom}")
                return False
            
            if abs(rotation) > 360:
                logger.error(f"Rotation value out of range: {rotation}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating alignment parameters: {e}")
            return False

    def update_aligned_gds_real_time(self, params: Dict[str, Any]):
        """Update aligned GDS display in real-time during parameter changes."""
        try:
            if not self.current_gds_data:
                return
                
            # Create aligned image with current parameters
            result = self.create_aligned_gds_image(params)
            
            if result and 'aligned_data' in result:
                # Update display in alignment panel
                aligned_image = result['aligned_data']['aligned_binary_image']
                if hasattr(self, 'alignment_panel'):
                    self.alignment_panel.set_aligned_gds_image(aligned_image)
                
        except Exception as e:
            logger.error(f"Error in real-time aligned GDS update: {e}")

    def setup_error_handling(self):
        """Setup comprehensive error handling for the application."""
        try:
            # Setup global exception handler
            import sys
            sys.excepthook = self.handle_unhandled_exception
            
            # Setup error logging
            self.setup_error_logging()
            
            logger.info("Error handling setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup error handling: {e}")

    def setup_error_logging(self):
        """Setup error logging to file and console."""
        try:
            import logging
            from pathlib import Path
            
            # Create logs directory
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # Setup file handler for errors
            error_log_path = logs_dir / "error.log"
            file_handler = logging.FileHandler(error_log_path)
            file_handler.setLevel(logging.ERROR)
            
            # Setup formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
            
            logger.info(f"Error logging setup to: {error_log_path}")
            
        except Exception as e:
            logger.error(f"Failed to setup error logging: {e}")

    def handle_unhandled_exception(self, exc_type, exc_value, exc_traceback):
        """Handle unhandled exceptions globally."""
        try:
            import traceback
            
            # Log the exception
            error_msg = f"Unhandled exception: {exc_type.__name__}: {exc_value}"
            logger.critical(error_msg)
            logger.critical("Traceback:\n" + "".join(traceback.format_tb(exc_traceback)))
            
            # Show user-friendly error dialog
            if exc_type != KeyboardInterrupt:
                self.show_critical_error("Application Error", 
                                       f"An unexpected error occurred:\n\n{exc_value}\n\n" +
                                       "Please check the error log for details.")
            
        except Exception as e:
            # Fallback error handling
            print(f"Critical error in error handler: {e}")

    def show_critical_error(self, title: str, message: str):
        """Show critical error dialog to user."""
        try:
            QMessageBox.critical(self, title, message)
        except Exception as e:
            logger.error(f"Failed to show critical error dialog: {e}")
            print(f"Critical Error - {title}: {message}")

    def handle_file_loading_error(self, error: Exception, file_path: str, file_type: str):
        """Handle file loading errors with specific guidance."""
        try:
            error_msg = str(error)
            
            # Provide specific error messages based on error type
            if "FileNotFoundError" in str(type(error)):
                user_msg = f"File not found: {file_path}\n\nPlease check that the file exists and try again."
            elif "PermissionError" in str(type(error)):
                user_msg = f"Permission denied accessing: {file_path}\n\nPlease check file permissions and close any programs using this file."
            elif "ValueError" in str(type(error)) and "format" in error_msg.lower():
                user_msg = f"Unsupported file format: {file_path}\n\nSupported formats for {file_type}:\n" + \
                          "- SEM: .tif, .tiff, .png\n- GDS: .gds, .gdsii"
            elif "empty" in error_msg.lower():
                user_msg = f"Empty or corrupted file: {file_path}\n\nPlease check the file and try again."
            elif "gdstk" in error_msg.lower():
                user_msg = f"GDS file reading error: {file_path}\n\nThe GDS file may be corrupted or use an unsupported format."
            else:
                user_msg = f"Error loading {file_type} file: {file_path}\n\nError: {error_msg}"
            
            # Log detailed error
            logger.error(f"File loading error - {file_type}: {file_path} - {error}")
            
            # Show user-friendly message
            QMessageBox.critical(self, f"{file_type} Loading Error", user_msg)
            
        except Exception as e:
            logger.error(f"Error in file loading error handler: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load {file_type} file: {file_path}")

    def handle_alignment_error(self, error: Exception, alignment_type: str):
        """Handle alignment errors with specific guidance."""
        try:
            error_msg = str(error)
            
            # Provide specific error messages based on error type
            if "No features detected" in error_msg:
                user_msg = f"Auto alignment failed: No features detected in images.\n\n" + \
                          "Suggestions:\n" + \
                          "- Check image quality and contrast\n" + \
                          "- Try manual alignment instead\n" + \
                          "- Verify images are properly loaded"
            elif "Not enough matches" in error_msg:
                user_msg = f"Auto alignment failed: Not enough matching features found.\n\n" + \
                          "Suggestions:\n" + \
                          "- Images may be too different\n" + \
                          "- Try adjusting feature detection settings\n" + \
                          "- Use hybrid alignment with manual points"
            elif "transformation matrix" in error_msg.lower():
                user_msg = f"Alignment failed: Could not calculate transformation.\n\n" + \
                          "Suggestions:\n" + \
                          "- Try different alignment method\n" + \
                          "- Check if images are properly aligned\n" + \
                          "- Use manual alignment"
            elif "Missing" in error_msg and "parameter" in error_msg:
                user_msg = f"Alignment parameter error: {error_msg}\n\n" + \
                          "Please reset alignment and try again."
            else:
                user_msg = f"{alignment_type} alignment failed: {error_msg}\n\n" + \
                          "Please try a different alignment method or check your images."
            
            # Log detailed error
            logger.error(f"Alignment error - {alignment_type}: {error}")
            
            # Show user-friendly message
            QMessageBox.critical(self, f"{alignment_type} Alignment Error", user_msg)
            
        except Exception as e:
            logger.error(f"Error in alignment error handler: {e}")
            QMessageBox.critical(self, "Alignment Error", f"{alignment_type} alignment failed")

    def handle_gds_processing_error(self, error: Exception, operation: str):
        """Handle GDS processing errors with specific guidance."""
        try:
            error_msg = str(error)
            
            # Provide specific error messages based on error type
            if "structure" in error_msg.lower() and "not found" in error_msg.lower():
                user_msg = f"GDS structure error: {error_msg}\n\n" + \
                          "Suggestions:\n" + \
                          "- Check if the selected structure exists in the GDS file\n" + \
                          "- Try a different structure (1-5)\n" + \
                          "- Verify GDS file contains the expected structures"
            elif "bounds" in error_msg.lower() or "coordinates" in error_msg.lower():
                user_msg = f"GDS coordinate error: {error_msg}\n\n" + \
                          "The GDS file may have unexpected coordinate ranges.\n" + \
                          "Please check the GDS file structure."
            elif "polygon" in error_msg.lower():
                user_msg = f"GDS polygon error: {error_msg}\n\n" + \
                          "The GDS file may contain invalid polygon data.\n" + \
                          "Please check the GDS file integrity."
            elif "transformation" in error_msg.lower():
                user_msg = f"GDS transformation error: {error_msg}\n\n" + \
                          "Alignment parameters may be out of range.\n" + \
                          "Try resetting alignment or using smaller parameter values."
            else:
                user_msg = f"GDS processing error during {operation}: {error_msg}\n\n" + \
                          "Please check the GDS file and try again."
            
            # Log detailed error
            logger.error(f"GDS processing error - {operation}: {error}")
            
            # Show user-friendly message
            QMessageBox.critical(self, "GDS Processing Error", user_msg)
            
        except Exception as e:
            logger.error(f"Error in GDS processing error handler: {e}")
            QMessageBox.critical(self, "GDS Error", f"GDS processing failed during {operation}")

    def handle_save_error(self, error: Exception, file_path: str, file_type: str):
        """Handle file saving errors with specific guidance."""
        try:
            error_msg = str(error)
            
            # Provide specific error messages based on error type
            if "PermissionError" in str(type(error)):
                user_msg = f"Permission denied saving to: {file_path}\n\n" + \
                          "Suggestions:\n" + \
                          "- Check folder permissions\n" + \
                          "- Close any programs using the file\n" + \
                          "- Try saving to a different location"
            elif "FileNotFoundError" in str(type(error)):
                user_msg = f"Directory not found: {file_path}\n\n" + \
                          "The output directory may not exist or be accessible."
            elif "disk" in error_msg.lower() or "space" in error_msg.lower():
                user_msg = f"Insufficient disk space to save: {file_path}\n\n" + \
                          "Please free up disk space and try again."
            elif "readonly" in error_msg.lower() or "read-only" in error_msg.lower():
                user_msg = f"Cannot write to read-only location: {file_path}\n\n" + \
                          "Please choose a different save location."
            else:
                user_msg = f"Error saving {file_type} file: {file_path}\n\n" + \
                          f"Error: {error_msg}"
            
            # Log detailed error
            logger.error(f"Save error - {file_type}: {file_path} - {error}")
            
            # Show user-friendly message
            QMessageBox.critical(self, f"Save Error", user_msg)
            
        except Exception as e:
            logger.error(f"Error in save error handler: {e}")
            QMessageBox.critical(self, "Save Error", f"Failed to save {file_type} file")

    def validate_application_state(self) -> bool:
        """Validate application state before operations."""
        try:
            # Check if required services are available
            if not hasattr(self, 'file_service') or not self.file_service:
                self.show_critical_error("Service Error", "File service not available. Please restart the application.")
                return False
            
            # Check if required directories exist
            from pathlib import Path
            data_dir = Path("Data")
            if not data_dir.exists():
                self.show_critical_error("Directory Error", 
                                       f"Data directory not found: {data_dir.absolute()}\n\n" +
                                       "Please ensure the application is run from the correct directory.")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating application state: {e}")
            return False

    def safe_operation(self, operation_func, operation_name: str, *args, **kwargs):
        """Safely execute an operation with error handling."""
        try:
            # Validate application state first
            if not self.validate_application_state():
                return None
            
            # Execute operation
            result = operation_func(*args, **kwargs)
            return result
            
        except FileNotFoundError as e:
            self.handle_file_loading_error(e, str(args[0]) if args else "unknown", operation_name)
            return None
        except PermissionError as e:
            self.handle_save_error(e, str(args[0]) if args else "unknown", operation_name)
            return None
        except ValueError as e:
            if "alignment" in operation_name.lower():
                self.handle_alignment_error(e, operation_name)
            elif "gds" in operation_name.lower():
                self.handle_gds_processing_error(e, operation_name)
            else:
                QMessageBox.critical(self, f"{operation_name} Error", f"Invalid data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in {operation_name}: {e}")
            QMessageBox.critical(self, f"{operation_name} Error", 
                               f"An unexpected error occurred during {operation_name}:\n\n{e}")
            return None

    def update_aligned_gds_real_time(self, transform_params: dict):
        """Update aligned GDS display in real-time."""
        try:
            # This would update the GDS overlay with new transform parameters
            # Implementation depends on the specific alignment service
            logger.debug(f"Real-time GDS update with params: {transform_params}")
        except Exception as e:
            logger.error(f"Error in real-time GDS update: {e}")
    
    def apply_final_alignment(self, transform_params: dict):
        """Apply final alignment transformation."""
        try:
            # This would apply the final alignment transformation
            # Implementation depends on the specific alignment service
            logger.info(f"Final alignment applied with params: {transform_params}")
        except Exception as e:
            logger.error(f"Error applying final alignment: {e}")
    
    def reset_alignment_state(self):
        """Reset alignment to default state."""
        try:
            # Reset any alignment-related state
            if hasattr(self, 'alignment_panel'):
                # The alignment panel handles its own reset
                pass
            logger.info("Alignment state reset")
        except Exception as e:
            logger.error(f"Error resetting alignment state: {e}")

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Image Analysis Tool")
    app.setOrganizationName("Image Analysis")
    
    window = MainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
