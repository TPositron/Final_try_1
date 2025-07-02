"""
File Operations Module
Handles all file operations including loading SEM images, GDS files, and saving results.
Extracted from main_window_v2.py to create a focused, maintainable module.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtCore import QObject, Signal
from PIL import Image

from src.services.simple_file_service import FileService


class FileOperationsManager(QObject):
    """Manages all file operations for the application."""
    
    # Signals for file operations
    sem_image_loaded = Signal(str, object)  # file_path, image_data
    gds_file_loaded = Signal(str)  # file_path
    results_saved = Signal(str, str)  # file_path, file_type
    file_operation_error = Signal(str, str)  # operation, error_message
    
    def __init__(self, main_window):
        """Initialize with reference to main window."""
        super().__init__()
        self.main_window = main_window
        self.file_service = FileService()
        
        # Current file state
        self.current_sem_path = None
        self.current_gds_path = None
        
    def load_sem_image(self):
        """Load SEM image, create cropped version, save to cut folder, and display."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Load SEM Image", "Data/SEM",
            "Image Files (*.tif *.tiff *.png *.jpg *.jpeg)")

        if file_path:
            try:
                # Use the file service to load and crop SEM image
                sem_data = self.file_service.load_sem_file(Path(file_path))
                
                if sem_data and 'cropped_array' in sem_data:
                    # Get the already cropped image array
                    cropped_image = sem_data['cropped_array']
                    
                    # Store the image data and path
                    self.main_window.current_sem_image = cropped_image
                    self.main_window.current_sem_image_obj = sem_data['sem_image']  # SemImage object
                    self.current_sem_path = file_path  # Store the original SEM file path
                    
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
                    if hasattr(self.main_window, 'image_viewer'):
                        self.main_window.image_viewer.set_sem_image(self.main_window.current_sem_image)
                    
                    # Update status with size info
                    h, w = cropped_image.shape[:2]
                    if hasattr(self.main_window, 'status_bar'):
                        self.main_window.status_bar.showMessage(f"Loaded and cropped SEM image: {original_filename} ({w}x{h}) - Saved to cut folder")
                    
                    print(f"SEM image loaded: {original_filename}, cropped to: {w}x{h}, saved as: {cut_filename}")
                    
                    # Emit signal
                    self.sem_image_loaded.emit(file_path, sem_data)
                    
                    # Update panel availability
                    if hasattr(self.main_window, '_update_panel_availability'):
                        self.main_window._update_panel_availability()
                else:
                    if hasattr(self.main_window, 'status_bar'):
                        self.main_window.status_bar.showMessage(f"Failed to load SEM image data")
                    self.file_operation_error.emit("load_sem", "Failed to load SEM image data")
                
            except Exception as e:
                QMessageBox.critical(self.main_window, "Error", f"Failed to load SEM image: {str(e)}")
                print(f"Debug - SEM loading error: {e}")
                self.file_operation_error.emit("load_sem", str(e))
    
    def save_results(self):
        """Save current results to file."""
        try:
            # Validate that we have data to save
            if not self._validate_save_prerequisites():
                return
            
            # Get save location
            save_path, _ = QFileDialog.getSaveFileName(
                self.main_window,
                "Save Results",
                "Results/",
                "PNG Files (*.png);;All Files (*)"
            )
            
            if save_path:
                # Determine what to save based on current application state
                self._save_current_state(save_path)
                
        except Exception as e:
            error_msg = f"Failed to save results: {str(e)}"
            print(f"Save error: {error_msg}")
            QMessageBox.critical(self.main_window, "Save Error", error_msg)
            self.file_operation_error.emit("save_results", str(e))
    
    def _save_current_state(self, save_path):
        """Save the current application state to file."""
        try:
            # Get the current image from the image viewer
            if hasattr(self.main_window, 'image_viewer'):
                # Try to get the composite image (SEM + overlay)
                composite_image = self._create_composite_image()
                
                if composite_image is not None:
                    # Save composite image
                    cv2.imwrite(save_path, composite_image)
                    
                    # Update status
                    if hasattr(self.main_window, 'status_bar'):
                        self.main_window.status_bar.showMessage(f"Results saved to: {save_path}")
                    
                    # Emit signal
                    self.results_saved.emit(save_path, "composite_image")
                    
                    print(f"Results saved to: {save_path}")
                else:
                    raise RuntimeError("No composite image available to save")
            else:
                raise RuntimeError("No image viewer available")
                
        except Exception as e:
            raise RuntimeError(f"Failed to save current state: {str(e)}")
    
    def _create_composite_image(self):
        """Create a composite image of SEM + GDS overlay."""
        try:
            if self.main_window.current_sem_image is None:
                return None
            
            # Start with the SEM image
            composite = self.main_window.current_sem_image.copy()
            
            # If there's a GDS overlay, add it
            if hasattr(self.main_window, 'gds_operations'):
                gds_overlay = getattr(self.main_window.gds_operations, 'current_gds_overlay', None)
                if gds_overlay is not None:
                    # Blend the overlay with the SEM image
                    composite = self._blend_images(composite, gds_overlay)
            
            return composite
            
        except Exception as e:
            print(f"Error creating composite image: {e}")
            return None
    
    def _blend_images(self, background, overlay, alpha=0.5):
        """Blend two images together."""
        try:
            # Ensure both images have the same shape
            if background.shape != overlay.shape:
                if len(overlay.shape) == 3 and len(background.shape) == 2:
                    # Convert grayscale background to color
                    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
                elif len(background.shape) == 3 and len(overlay.shape) == 2:
                    # Convert grayscale overlay to color
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
            
            # Blend the images
            blended = cv2.addWeighted(background, 1-alpha, overlay, alpha, 0)
            return blended
            
        except Exception as e:
            print(f"Error blending images: {e}")
            return background
    
    def _validate_save_prerequisites(self):
        """Validate that we have data to save."""
        if self.main_window.current_sem_image is None:
            QMessageBox.warning(
                self.main_window,
                "No Data to Save",
                "Please load a SEM image first."
            )
            return False
        
        return True
    
    def export_alignment_data(self, file_path):
        """Export alignment data to a file."""
        try:
            if not hasattr(self.main_window, 'alignment_operations'):
                raise RuntimeError("No alignment operations available")
            
            alignment_info = self.main_window.alignment_operations.get_alignment_info()
            
            if alignment_info is None:
                raise RuntimeError("No alignment data available")
            
            # Export alignment data as JSON
            import json
            with open(file_path, 'w') as f:
                json.dump(alignment_info, f, indent=2, default=str)
            
            print(f"Alignment data exported to: {file_path}")
            self.results_saved.emit(file_path, "alignment_data")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to export alignment data: {str(e)}"
            print(error_msg)
            self.file_operation_error.emit("export_alignment", str(e))
            return False
    
    def export_scoring_results(self, file_path):
        """Export scoring results to a file."""
        try:
            if not hasattr(self.main_window, 'scoring_operations'):
                raise RuntimeError("No scoring operations available")
            
            if not self.main_window.scoring_operations.has_scores():
                raise RuntimeError("No scoring results available")
            
            # Export scoring results
            success = self.main_window.scoring_operations.export_scores(file_path)
            
            if success:
                self.results_saved.emit(file_path, "scoring_results")
                return True
            else:
                return False
                
        except Exception as e:
            error_msg = f"Failed to export scoring results: {str(e)}"
            print(error_msg)
            self.file_operation_error.emit("export_scoring", str(e))
            return False
    
    def get_recent_files(self):
        """Get list of recently used files."""
        # This could be implemented to track recent files
        return {
            'sem_files': [],
            'gds_files': []
        }
    
    def get_file_info(self, file_path):
        """Get information about a file."""
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            stat = path.stat()
            return {
                'name': path.name,
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'exists': True
            }
            
        except Exception as e:
            print(f"Error getting file info: {e}")
            return None
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            # Clean up any temporary files created during processing
            temp_dir = Path("temp")
            if temp_dir.exists():
                for temp_file in temp_dir.glob("*"):
                    try:
                        temp_file.unlink()
                    except Exception as e:
                        print(f"Could not delete temp file {temp_file}: {e}")
            
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")
