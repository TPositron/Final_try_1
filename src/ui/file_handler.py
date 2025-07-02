"""
File Handler Module
Handles all file operations including SEM image loading, GDS file loading, and result saving.
"""

import os
from pathlib import Path
from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtCore import QObject, Signal
import cv2
import numpy as np

from src.services.simple_file_service import FileService
from src.core.models import SemImage


class FileHandler(QObject):
    """Handles all file operations for the Image Analysis application."""
    
    # Signals
    sem_image_loaded = Signal(object, str)  # SemImage object, file path
    gds_file_loaded = Signal(str)  # file path
    results_saved = Signal(str)  # save path
    
    def __init__(self, main_window):
        """Initialize file handler with reference to main window."""
        super().__init__()
        self.main_window = main_window
        self.file_service = FileService()
        
        # Current file states
        self.current_sem_path = None
        self.current_gds_path = None
        
    def load_sem_image(self):
        """Load a SEM image file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self.main_window,
                "Load SEM Image",
                str(self.file_service.get_sem_dir()),
                "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)"
            )
            
            if file_path:
                print(f"Loading SEM image: {file_path}")
                
                # Load image using file service
                sem_image_obj = self.file_service.load_sem_image(file_path)
                
                if sem_image_obj is not None:
                    # Store the file path
                    self.current_sem_path = file_path
                    
                    # Get the image array
                    sem_image = sem_image_obj.get_image_array()
                    
                    # Validate image
                    if sem_image is not None and sem_image.size > 0:
                        # Store in main window
                        self.main_window.current_sem_image = sem_image
                        self.main_window.current_sem_image_obj = sem_image_obj
                        self.main_window.current_sem_path = file_path
                        
                        # Update image viewer
                        if hasattr(self.main_window, 'image_viewer'):
                            self.main_window.image_viewer.set_sem_image(sem_image)
                        
                        # Initialize image processing with original image
                        if hasattr(self.main_window, 'image_processing'):
                            self.main_window.image_processing.set_original_image(sem_image)
                        
                        # Update status
                        filename = Path(file_path).name
                        if hasattr(self.main_window, 'status_bar'):
                            self.main_window.status_bar.showMessage(f"Loaded SEM image: {filename}")
                        
                        # Emit signal
                        self.sem_image_loaded.emit(sem_image_obj, file_path)
                        
                        print(f"✓ SEM image loaded successfully: {filename}")
                        print(f"Image shape: {sem_image.shape}, dtype: {sem_image.dtype}")
                        
                        return True
                    else:
                        raise ValueError("Loaded image is empty or invalid")
                else:
                    raise ValueError("Failed to load SEM image data")
                    
        except Exception as e:
            error_msg = f"Failed to load SEM image: {str(e)}"
            print(f"SEM loading error: {error_msg}")
            QMessageBox.critical(self.main_window, "Error", error_msg)
            return False
    
    def load_gds_file(self):
        """Load a GDS file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self.main_window,
                "Load GDS File",
                str(self.file_service.get_gds_dir()),
                "GDS Files (*.gds *.gds2)"
            )
            
            if file_path:
                print(f"Loading GDS file: {file_path}")
                
                # Store the file path
                self.current_gds_path = file_path
                
                # Delegate to GDS manager if available
                if hasattr(self.main_window, 'gds_manager'):
                    success = self.main_window.gds_manager.load_gds_file(file_path)
                    if success:
                        # Emit signal
                        self.gds_file_loaded.emit(file_path)
                        return True
                    else:
                        raise RuntimeError("GDS manager failed to load file")
                else:
                    # Fallback: just store the path
                    filename = Path(file_path).name
                    if hasattr(self.main_window, 'status_bar'):
                        self.main_window.status_bar.showMessage(f"GDS file selected: {filename}")
                    
                    self.gds_file_loaded.emit(file_path)
                    return True
                    
        except Exception as e:
            error_msg = f"Failed to load GDS file: {str(e)}"
            print(f"GDS loading error: {error_msg}")
            QMessageBox.critical(self.main_window, "Error", error_msg)
            return False
    
    def save_results(self):
        """Save current analysis results."""
        try:
            # Check if there are results to save
            if not self._has_results_to_save():
                QMessageBox.information(
                    self.main_window,
                    "No Results",
                    "No analysis results available to save."
                )
                return
            
            # Get save location
            file_path, _ = QFileDialog.getSaveFileName(
                self.main_window,
                "Save Results",
                str(self.file_service.get_results_dir()),
                "JSON Files (*.json);;All Files (*)"
            )
            
            if file_path:
                print(f"Saving results to: {file_path}")
                
                # Collect results data
                results_data = self._collect_results_data()
                
                # Save using file service
                success = self.file_service.save_analysis_results(file_path, results_data)
                
                if success:
                    if hasattr(self.main_window, 'status_bar'):
                        self.main_window.status_bar.showMessage(f"Results saved to: {Path(file_path).name}")
                    
                    # Emit signal
                    self.results_saved.emit(file_path)
                    
                    print(f"✓ Results saved successfully to: {file_path}")
                    return True
                else:
                    raise RuntimeError("File service failed to save results")
                    
        except Exception as e:
            error_msg = f"Failed to save results: {str(e)}"
            print(f"Save error: {error_msg}")
            QMessageBox.critical(self.main_window, "Error", error_msg)
            return False
    
    def save_current_image(self, image_type="processed"):
        """Save the current processed image."""
        try:
            if self.main_window.current_sem_image is None:
                QMessageBox.warning(
                    self.main_window,
                    "No Image",
                    "No image available to save."
                )
                return False
            
            # Get save location
            file_path, _ = QFileDialog.getSaveFileName(
                self.main_window,
                f"Save {image_type.title()} Image",
                str(self.file_service.get_results_dir()),
                "PNG Files (*.png);;TIFF Files (*.tif);;All Files (*)"
            )
            
            if file_path:
                # Get the image to save
                image_to_save = self.main_window.current_sem_image
                
                # Convert to appropriate format
                if image_to_save.dtype == np.float64 or image_to_save.dtype == np.float32:
                    # Normalize to 0-255 for saving
                    normalized = cv2.normalize(image_to_save, None, 0, 255, cv2.NORM_MINMAX)
                    image_to_save = normalized.astype(np.uint8)
                
                # Save the image
                success = cv2.imwrite(file_path, image_to_save)
                
                if success:
                    if hasattr(self.main_window, 'status_bar'):
                        self.main_window.status_bar.showMessage(f"Image saved to: {Path(file_path).name}")
                    
                    print(f"✓ {image_type.title()} image saved to: {file_path}")
                    return True
                else:
                    raise RuntimeError("cv2.imwrite failed")
                    
        except Exception as e:
            error_msg = f"Failed to save {image_type} image: {str(e)}"
            print(f"Image save error: {error_msg}")
            QMessageBox.critical(self.main_window, "Error", error_msg)
            return False
    
    def export_alignment_overlay(self):
        """Export the current alignment overlay as an image."""
        try:
            # Check if alignment result exists
            alignment_result = getattr(self.main_window, 'current_alignment_result', None)
            if alignment_result is None:
                QMessageBox.warning(
                    self.main_window,
                    "No Alignment",
                    "No alignment result available to export."
                )
                return False
            
            # Get save location
            file_path, _ = QFileDialog.getSaveFileName(
                self.main_window,
                "Export Alignment Overlay",
                str(self.file_service.get_results_dir()),
                "PNG Files (*.png);;All Files (*)"
            )
            
            if file_path:
                # Get the aligned overlay
                if 'transformed_gds' in alignment_result:
                    overlay = alignment_result['transformed_gds']
                    
                    # Convert to uint8 if needed
                    if overlay.dtype != np.uint8:
                        overlay = (overlay * 255).astype(np.uint8)
                    
                    # Save the overlay
                    success = cv2.imwrite(file_path, overlay)
                    
                    if success:
                        if hasattr(self.main_window, 'status_bar'):
                            self.main_window.status_bar.showMessage(f"Overlay exported to: {Path(file_path).name}")
                        
                        print(f"✓ Alignment overlay exported to: {file_path}")
                        return True
                    else:
                        raise RuntimeError("Failed to save overlay image")
                else:
                    raise ValueError("No transformed GDS data in alignment result")
                    
        except Exception as e:
            error_msg = f"Failed to export alignment overlay: {str(e)}"
            print(f"Export error: {error_msg}")
            QMessageBox.critical(self.main_window, "Error", error_msg)
            return False
    
    def _has_results_to_save(self):
        """Check if there are analysis results available to save."""
        # Check for scoring results
        if hasattr(self.main_window, 'scoring_calculator'):
            if self.main_window.scoring_calculator.has_scores():
                return True
        
        # Check for alignment results
        if hasattr(self.main_window, 'current_alignment_result'):
            if self.main_window.current_alignment_result is not None:
                return True
        
        # Check for processed image
        if self.main_window.current_sem_image is not None:
            return True
        
        return False
    
    def _collect_results_data(self):
        """Collect all available results data for saving."""
        results_data = {
            'timestamp': str(np.datetime64('now')),
            'files': {
                'sem_image': self.current_sem_path,
                'gds_file': self.current_gds_path
            }
        }
        
        # Add SEM image info
        if self.main_window.current_sem_image is not None:
            img = self.main_window.current_sem_image
            results_data['sem_image_info'] = {
                'shape': img.shape,
                'dtype': str(img.dtype),
                'min_value': float(np.min(img)),
                'max_value': float(np.max(img)),
                'mean_value': float(np.mean(img))
            }
        
        # Add alignment results
        if hasattr(self.main_window, 'current_alignment_result') and self.main_window.current_alignment_result:
            results_data['alignment'] = {
                'method': self.main_window.current_alignment_result.get('method', 'unknown'),
                'score': self.main_window.current_alignment_result.get('score', 0.0)
            }
        
        # Add scoring results
        if hasattr(self.main_window, 'scoring_calculator'):
            scores = self.main_window.scoring_calculator.get_current_scores()
            if scores:
                results_data['scores'] = scores
        
        # Add applied filters
        if hasattr(self.main_window, 'image_processor'):
            filters = self.main_window.image_processor.get_applied_filters()
            if filters:
                results_data['applied_filters'] = filters
        
        return results_data
    
    def get_current_file_info(self):
        """Get information about currently loaded files."""
        return {
            'sem_image_path': self.current_sem_path,
            'gds_file_path': self.current_gds_path,
            'has_sem_image': self.main_window.current_sem_image is not None,
            'has_gds_file': self.current_gds_path is not None
        }
    
    def clear_file_data(self):
        """Clear all loaded file data."""
        self.current_sem_path = None
        self.current_gds_path = None
        self.main_window.current_sem_image = None
        self.main_window.current_sem_image_obj = None
        self.main_window.current_sem_path = None
        
        print("File data cleared")
