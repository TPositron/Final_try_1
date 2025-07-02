"""
File Operations Module
Handles all file loading, saving, and management operations.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtCore import QTimer

# Configuration
DEFAULT_GDS_FILE = "Institute_Project_GDS1.gds"


class FileOperations:
    """
    Handles all file operations for the main window.
    """
    
    def __init__(self, main_window):
        """Initialize file operations with reference to main window."""
        self.main_window = main_window
    
    def load_sem_image(self):
        """Load SEM image, create cropped version, save to cut folder, and display."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Load SEM Image", "Data/SEM",
            "Image Files (*.tif *.tiff *.png *.jpg *.jpeg)")

        if file_path:
            try:
                # Use the file service to load and crop SEM image
                sem_data = self.main_window.file_service.load_sem_file(Path(file_path))
                
                if sem_data and 'cropped_array' in sem_data:
                    # Get the already cropped image array
                    cropped_image = sem_data['cropped_array']
                    
                    # Store the image data and path
                    self.main_window.current_sem_image = cropped_image
                    self.main_window.current_sem_image_obj = sem_data['sem_image']  # SemImage object
                    self.main_window.current_sem_path = file_path  # Store the original SEM file path
                    
                    # Create cut folder if it doesn't exist
                    cut_folder = Path("Results") / "cut"
                    cut_folder.mkdir(parents=True, exist_ok=True)
                    
                    # Save the cropped image to the cut folder
                    original_filename = Path(file_path).name
                    base_name = Path(file_path).stem
                    cut_filename = f"{base_name}_cropped_1024x666.png"
                    cut_path = cut_folder / cut_filename
                    
                    # Convert numpy array to PIL Image and save
                    if len(cropped_image.shape) == 2:
                        # Grayscale image
                        pil_image = Image.fromarray(cropped_image.astype(np.uint8))
                    else:
                        # Color image
                        pil_image = Image.fromarray(cropped_image.astype(np.uint8))
                    
                    pil_image.save(cut_path)
                    print(f"Cropped image saved to: {cut_path}")
                    
                    # Display in image viewer immediately
                    self.main_window.image_viewer.set_sem_image(self.main_window.current_sem_image)
                    
                    # Update status with size info
                    h, w = cropped_image.shape[:2]
                    self.main_window.status_bar.showMessage(f"Loaded and cropped SEM image: {original_filename} ({w}x{h}) - Saved to cut folder")
                    
                    print(f"SEM image loaded: {original_filename}, cropped to: {w}x{h}, saved as: {cut_filename}")
                    
                    # Update panel availability through view controller
                    self.main_window.view_controller.update_panel_availability()
                else:
                    self.main_window.status_bar.showMessage(f"Failed to load SEM image data")
                
            except Exception as e:
                QMessageBox.critical(self.main_window, "Error", f"Failed to load SEM image: {str(e)}")
                print(f"Debug - SEM loading error: {e}")
    
    def load_gds_file(self):
        """Enhanced GDS file loading with new services."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Load GDS File", str(self.main_window.file_manager.get_gds_dir()),
            "GDS Files (*.gds *.gds2)")
        
        if file_path:
            try:
                gds_filename = Path(file_path).name
                
                # Use new GDS service for loading
                success = self.main_window.new_gds_service.load_gds_file(file_path)
                
                if success:
                    # Enable structure selection
                    self.main_window.structure_combo.setEnabled(True)
                    
                    # Store current GDS filename for structure loading
                    self.main_window.current_gds_filename = gds_filename
                    self.main_window.current_gds_filepath = str(file_path)
                    
                    # Populate structure combo with available structures
                    self.populate_structure_combo()
                    
                    self.main_window.status_bar.showMessage(f"Loaded GDS file: {gds_filename}")
                    print(f"✓ GDS file loaded successfully: {gds_filename}")
                else:
                    raise RuntimeError(f"Failed to load GDS file: {gds_filename}")
                
            except Exception as e:
                QMessageBox.critical(self.main_window, "Error", f"Failed to load GDS file: {str(e)}")
                print(f"Debug - GDS loading error: {e}")
                
                # Disable structure selection on error
                self.main_window.structure_combo.setEnabled(False)
                self.main_window.current_gds_filename = None
                self.main_window.current_gds_filepath = None
    
    def populate_structure_combo(self):
        """Populate the structure selection combo box with predefined structures."""
        self.main_window.structure_combo.clear()
        self.main_window.structure_combo.addItem("Select Structure...", "")
        
        # Add structures using new GDS service
        structures_info = self.main_window.new_gds_service.get_all_structures_info()
        for structure_num, info in structures_info.items():
            display_name = f"Structure {structure_num} - {info['name']}"
            # Store the structure number format for compatibility
            self.main_window.structure_combo.addItem(display_name, f"Structure {structure_num}")
        
        print(f"Populated structure combo with {len(structures_info)} structures")
    
    def auto_load_default_gds(self):
        """Auto-load the default GDS file if it exists - simplified approach."""
        try:
            default_gds_path = self.main_window.file_service.gds_dir / DEFAULT_GDS_FILE
            if default_gds_path.exists():
                print(f"Auto-loading default GDS file: {default_gds_path}")
                
                # Use simple GDS loader
                from src.core.simple_gds_loader import SimpleGDSLoader
                self.main_window.simple_gds_loader = SimpleGDSLoader(str(default_gds_path))
                
                self.main_window.current_gds_filename = DEFAULT_GDS_FILE
                self.main_window.current_gds_filepath = str(default_gds_path)
                
                # Populate structure combo
                self.populate_structure_combo()
                
                print(f"Simple GDS loader created successfully")
                return True
            else:
                print(f"Default GDS file not found: {default_gds_path}")
                return False
        except Exception as e:
            print(f"Error auto-loading GDS file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_results(self):
        """Save alignment and scoring results."""
        if not self.main_window.validate_required_data(alignment_required=True):
            return
            
        try:
            timestamp = QTimer().remainingTime()  
            result_dir = self.main_window.file_manager.create_result_subdir(f"alignment_result_{timestamp}")
            
            saved_files = self.main_window.alignment_service.save_alignment_result(
                self.main_window.current_alignment_result, result_dir)
            
            # Save scores through scoring operations
            if self.main_window.current_scoring_results:
                self.main_window.file_manager.save_scores(
                    self.main_window.current_scoring_results, 
                    "alignment_scores", 
                    subdir=result_dir.name)
            
            self.main_window.status_bar.showMessage(f"Results saved to: {result_dir}")
            QMessageBox.information(self.main_window, "Success", f"Results saved to:\\n{result_dir}")
            
        except Exception as e:
            self.main_window.handle_service_error("Results saving", e)
    
    def save_aligned_gds(self):
        """Save the aligned GDS structure."""
        if not self.main_window.validate_required_data(gds_required=True, alignment_required=True):
            return
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self.main_window, "Save Aligned GDS", "Results/Aligned",
                "GDS Files (*.gds)")
            
            if not file_path:
                return
            
            # Use GDS operations to save the aligned structure
            success = self.main_window.gds_operations.save_aligned_structure(file_path)
            
            if success:
                self.main_window.status_bar.showMessage(f"Aligned GDS saved to: {file_path}")
                QMessageBox.information(self.main_window, "Save Successful", f"Aligned GDS saved to:\\n{file_path}")
                
        except Exception as e:
            self.main_window.handle_service_error("Aligned GDS saving", e)
