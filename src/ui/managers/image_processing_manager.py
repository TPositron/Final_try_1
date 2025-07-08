
"""
Image Processing Manager
Handles all image processing operations, filtering, and image-related functionality.
Extracted from main_window_v2.py to create a focused, maintainable module.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QObject, Signal

from src.services.simple_image_processing_service import ImageProcessingService


class ImageProcessingManager(QObject):
    """Manages all image processing operations for the application."""
    
    # Signals for image processing operations
    filter_applied = Signal(str, dict)  # filter_name, parameters
    filter_previewed = Signal(str, dict, object)  # filter_name, parameters, preview_image
    filters_reset = Signal()
    image_processing_error = Signal(str, str)  # operation, error_message
    
    def __init__(self, main_window):
        """Initialize with reference to main window."""
        super().__init__()
        self.original_image: Optional[np.ndarray] = None
        self.current_image: Optional[np.ndarray] = None
        self.filter_history: list = []

        self.main_window = main_window
        self.image_processing_service = ImageProcessingService()
        
        # Image processing state
        self.applied_filters = []  # Track applied filters
        
    def set_image(self, image: np.ndarray):
        """Set the original image for processing."""
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.filter_history = []

    def on_filter_applied(self, filter_name, parameters):
        """Handle filter application from the filter panel."""
        if not self._validate_required_data(sem_required=True):
            return
            
        try:
            # Apply filter using image processing service
            self.image_processing_service.apply_filter(filter_name, parameters)
            filtered_image = self.image_processing_service.get_current_image()
            
            # Update the image viewer
            if hasattr(self.main_window, 'image_viewer'):
                self.main_window.image_viewer.set_sem_image(filtered_image)
            
            # Update the main window's current image reference
            self.main_window.current_sem_image = filtered_image
            
            # Update histogram
            if hasattr(self.main_window, 'histogram_view'):
                self.main_window.histogram_view.update_histogram(filtered_image)
            
            # Update advanced filtering right panel histogram
            if hasattr(self.main_window, 'advanced_filtering_right_panel'):
                self.main_window.advanced_filtering_right_panel.update_histogram(filtered_image)
            
            # Track applied filter
            self.applied_filters.append({
                'name': filter_name,
                'parameters': parameters
            })
            
            # Update status
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage(f"Applied filter: {filter_name}")
            
            # Update alignment display if overlay exists
            if hasattr(self.main_window, 'current_gds_overlay') and self.main_window.current_gds_overlay is not None:
                if hasattr(self.main_window, 'update_alignment_display'):
                    self.main_window.update_alignment_display()
            
            # Emit signal
            self.filter_applied.emit(filter_name, parameters)
            
            print(f"✓ Filter applied: {filter_name}")
                
        except Exception as e:
            self._handle_service_error("Filter Application", e)
            
    def on_filter_preview(self, filter_name, parameters):
        """Handle filter preview from the filter panel."""
        if not self._validate_required_data(sem_required=True, show_warning=False):
            return
            
        try:
            # Generate preview using image processing service
            preview_array = self.image_processing_service.preview_filter(filter_name, parameters)
            
            if preview_array is not None:
                # Update the image viewer with preview
                if hasattr(self.main_window, 'image_viewer'):
                    self.main_window.image_viewer.set_sem_image(preview_array)
                
                # Update histogram with preview
                if hasattr(self.main_window, 'histogram_view'):
                    self.main_window.histogram_view.update_histogram(preview_array)
                
                # Update advanced filtering right panel histogram
                if hasattr(self.main_window, 'advanced_filtering_right_panel'):
                    self.main_window.advanced_filtering_right_panel.update_histogram(preview_array)
                
                # Emit signal
                self.filter_previewed.emit(filter_name, parameters, preview_array)
                
                print(f"✓ Filter preview generated: {filter_name}")
            else:
                print(f"❌ Filter preview failed: {filter_name}")
            
        except Exception as e:
            print(f"Error in filter preview: {e}")
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage(f"Preview error: {str(e)}")
            self.image_processing_error.emit("preview_filter", str(e))
            
    def on_reset_filters(self):
        """Reset all applied filters and restore original image."""
        if not self._validate_required_data(sem_required=True):
            return
            
        try:
            # Reset to original using image processing service
            self.image_processing_service.reset_to_original()
            original_image = self.image_processing_service.get_current_image()
            
            # Update the image viewer
            if hasattr(self.main_window, 'image_viewer'):
                self.main_window.image_viewer.set_sem_image(original_image)
            
            # Update the main window's current image reference
            self.main_window.current_sem_image = original_image
            
            # Update histogram with original image
            if hasattr(self.main_window, 'histogram_view'):
                self.main_window.histogram_view.update_histogram(original_image)
            
            # Update advanced filtering right panel histogram
            if hasattr(self.main_window, 'advanced_filtering_right_panel'):
                self.main_window.advanced_filtering_right_panel.update_histogram(original_image)
            
            # Clear applied filters tracking
            self.applied_filters = []
            
            # Update status
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage("Reset to original image")
            
            # Update alignment display if overlay exists
            if hasattr(self.main_window, 'current_gds_overlay') and self.main_window.current_gds_overlay is not None:
                if hasattr(self.main_window, 'update_alignment_display'):
                    self.main_window.update_alignment_display()
            
            # Emit signal
            self.filters_reset.emit()
            
            print("✓ Filters reset to original image")
                
        except Exception as e:
            self._handle_service_error("Filter Reset", e)
    
    def apply_custom_filter(self, filter_function, filter_name="Custom", **kwargs):
        """Apply a custom filter function to the current image."""
        try:
            if not self._validate_required_data(sem_required=True):
                return False
            
            # Get current image
            current_image = self.image_processing_service.get_current_image()
            
            # Apply custom filter
            filtered_image = filter_function(current_image, **kwargs)
            
            if filtered_image is not None:
                # Update image processing service
                self.image_processing_service.set_current_image(filtered_image)
                
                # Update display
                if hasattr(self.main_window, 'image_viewer'):
                    self.main_window.image_viewer.set_sem_image(filtered_image)
                
                # Update main window reference
                self.main_window.current_sem_image = filtered_image
                
                # Track applied filter
                self.applied_filters.append({
                    'name': filter_name,
                    'parameters': kwargs,
                    'custom': True
                })
                
                # Update status
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage(f"Applied custom filter: {filter_name}")
                
                # Emit signal
                self.filter_applied.emit(filter_name, kwargs)
                
                print(f"✓ Custom filter applied: {filter_name}")
                return True
            else:
                raise RuntimeError("Custom filter returned None")
                
        except Exception as e:
            self._handle_service_error(f"Custom Filter ({filter_name})", e)
            return False
    
    def get_applied_filters(self):
        """Get list of applied filters."""
        return self.applied_filters.copy()
    
    def get_image_statistics(self):
        """Get statistics about the current image."""
        try:
            current_image = self.image_processing_service.get_current_image()
            if current_image is None:
                return None
            
            stats = {
                'shape': current_image.shape,
                'dtype': str(current_image.dtype),
                'min': float(np.min(current_image)),
                'max': float(np.max(current_image)),
                'mean': float(np.mean(current_image)),
                'std': float(np.std(current_image))
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting image statistics: {e}")
            return None
    
    def export_filtered_image(self, file_path):
        """Export the current filtered image to a file."""
        try:
            current_image = self.image_processing_service.get_current_image()
            if current_image is None:
                raise ValueError("No image available to export")
            
            # Convert to appropriate format for saving
            if current_image.dtype == np.float64 or current_image.dtype == np.float32:
                # Normalize to 0-255 for saving
                normalized = np.zeros_like(current_image, dtype=np.uint8)
                cv2.normalize(current_image, normalized, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                export_image = normalized
            else:
                export_image = current_image
            
            # Save the image
            success = cv2.imwrite(file_path, export_image)
            
            if success:
                print(f"✓ Filtered image exported to: {file_path}")
                return True
            else:
                raise RuntimeError("cv2.imwrite failed")
                
        except Exception as e:
            error_msg = f"Failed to export filtered image: {str(e)}"
            print(error_msg)
            self.image_processing_error.emit("export_image", str(e))
            return False
    
    def create_filter_chain(self, filter_configs):
        """Apply a chain of filters in sequence."""
        try:
            if not self._validate_required_data(sem_required=True):
                return False
            
            # Start with original image
            self.image_processing_service.reset_to_original()
            
            # Apply each filter in the chain
            for filter_config in filter_configs:
                filter_name = filter_config.get('name')
                parameters = filter_config.get('parameters', {})
                
                if filter_name:
                    self.image_processing_service.apply_filter(filter_name, parameters)
                    
                    # Track applied filter
                    self.applied_filters.append({
                        'name': filter_name,
                        'parameters': parameters
                    })
            
            # Update display with final result
            final_image = self.image_processing_service.get_current_image()
            if hasattr(self.main_window, 'image_viewer'):
                self.main_window.image_viewer.set_sem_image(final_image)
            
            self.main_window.current_sem_image = final_image
            
            # Update status
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage(f"Applied filter chain with {len(filter_configs)} filters")
            
            print(f"✓ Filter chain applied with {len(filter_configs)} filters")
            return True
            
        except Exception as e:
            self._handle_service_error("Filter Chain", e)
            return False
    
    def undo_last_filter(self):
        """Undo the last applied filter."""
        try:
            if not self.applied_filters:
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage("No filters to undo")
                return False
            
            # Remove last filter from tracking
            self.applied_filters.pop()
            
            # Rebuild filter chain without the last filter
            if self.applied_filters:
                filter_configs = [
                    {'name': f['name'], 'parameters': f['parameters']} 
                    for f in self.applied_filters if not f.get('custom', False)
                ]
                self.create_filter_chain(filter_configs)
            else:
                # No filters left, reset to original
                self.on_reset_filters()
            
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage("Undone last filter")
            
            print("✓ Last filter undone")
            return True
            
        except Exception as e:
            self._handle_service_error("Undo Filter", e)
            return False
    
    def save_current_image(self):
        """Save current filtered image to Results/SEM_Filters/manual/ directory."""
        try:
            # Get current filtered image
            current_image = self.image_processing_service.get_current_image()
            if current_image is None:
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage("No image to save")
                return
                
            # Create save directory if it doesn't exist
            from pathlib import Path
            import cv2
            from datetime import datetime
            
            save_dir = Path("Results/SEM_Filters/manual")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp and filter info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get applied filters info for filename
            applied_filters = self.get_applied_filters()
            if applied_filters:
                # Create a short filter description for filename
                filter_names = [f['name'] for f in applied_filters[-3:]]  # Last 3 filters
                filter_suffix = "_".join(filter_names).replace(" ", "")[:30]  # Limit length
                filename = f"filtered_{filter_suffix}_{timestamp}.png"
            else:
                filename = f"filtered_image_{timestamp}.png"
            save_path = save_dir / filename
            
            # Save the image
            success = cv2.imwrite(str(save_path), current_image)
            
            if not success:
                raise RuntimeError(f"Failed to save image to {save_path}")
            
            # Verify file was created
            if not save_path.exists():
                raise RuntimeError(f"Image file was not created at {save_path}")
            
            # Update status
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage(f"Image saved successfully to {save_path}")
            
            print(f"✓ Image saved successfully to: {save_path}")
            
            # Show success message
            if hasattr(self.main_window, 'show_message'):
                self.main_window.show_message("Image Saved", f"Filtered image saved to:\n{save_path}")
            
        except Exception as e:
            error_msg = f"Error saving image: {str(e)}"
            print(error_msg)
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage(error_msg)
            self.image_processing_error.emit("save_image", str(e))
            
            # Show error message
            if hasattr(self.main_window, 'show_error'):
                self.main_window.show_error("Save Error", error_msg)

    def _validate_required_data(self, sem_required=False, gds_required=False, alignment_required=False, show_warning=True):
        """Validate that required data is available for operations."""
        if sem_required and not hasattr(self.main_window, 'current_sem_image'):
            return False
        
        if sem_required and self.main_window.current_sem_image is None:
            if show_warning:
                QMessageBox.warning(
                    self.main_window,
                    "No SEM Image",
                    "Please load a SEM image first."
                )
            return False
        
        return True
    
    def _handle_service_error(self, operation_name: str, error: Exception):
        """Handle service errors consistently."""
        error_msg = f"{operation_name} failed: {str(error)}"
        print(f"Service error: {error_msg}")
        
        if hasattr(self.main_window, 'status_bar'):
            self.main_window.status_bar.showMessage(error_msg)
        
        QMessageBox.critical(self.main_window, f"{operation_name} Error", error_msg)
        self.image_processing_error.emit(operation_name.lower().replace(" ", "_"), str(error))
    
    def has_applied_filters(self):
        """Check if any filters have been applied."""
        return len(self.applied_filters) > 0
    
    def get_filter_history(self):
        """Get the history of applied filters."""
        return {
            'filters': self.applied_filters.copy(),
            'total_count': len(self.applied_filters),
            'has_custom_filters': any(f.get('custom', False) for f in self.applied_filters)
        }

def apply_filter(self, filter_name: str, parameters: Dict[str, Any]) -> Optional[np.ndarray]:
        """Apply a filter with given parameters and update current image."""
        try:
            if self.current_image is None:
                print("No image loaded for filtering")
                return None
            
            # Apply the filter based on filter_name
            filtered_image = self._apply_filter_by_name(filter_name, self.current_image, parameters)
            
            if filtered_image is not None:
                self.current_image = filtered_image
                # Add to filter history
                self.filter_history.append({
                    'filter': filter_name,
                    'parameters': parameters.copy()
                })
                return filtered_image.copy()
            
            return None
            
        except Exception as e:
            print(f"Error applying filter {filter_name}: {e}")
            return None
        
def preview_filter(self, filter_name: str, parameters: Dict[str, Any]) -> Optional[np.ndarray]:
    """Preview a filter without modifying the current image."""
    try:
        if self.current_image is None:
            print("No image loaded for preview")
            return None
            
        # Apply filter to current image but don't save it
        preview_image = self._apply_filter_by_name(filter_name, self.current_image, parameters)
            
        if preview_image is not None:
            return preview_image.copy()
            
        return None
            
    except Exception as e:
        print(f"Error previewing filter {filter_name}: {e}")
        return None
        
def reset_filters(self):
    """Reset to original image."""
    try:
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.filter_history = []
            print("Filters reset to original image")
        else:
            print("No original image to reset to")
                
    except Exception as e:
        print(f"Error resetting filters: {e}")
    
   
def _apply_filter_by_name(self, filter_name: str, image: np.ndarray, parameters: Dict[str, Any]) -> Optional[np.ndarray]:
    """Apply a specific filter by name."""
    try:
        if filter_name.lower() == "gaussian_blur":
            kernel_size = parameters.get('kernel_size', 5)
            sigma = parameters.get('sigma', 1.0)
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            
        elif filter_name.lower() == "median_filter":
            kernel_size = parameters.get('kernel_size', 5)
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
            return cv2.medianBlur(image, kernel_size)
            
        elif filter_name.lower() == "bilateral_filter":
            d = parameters.get('d', 9)
            sigma_color = parameters.get('sigma_color', 75)
            sigma_space = parameters.get('sigma_space', 75)
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            
        elif filter_name.lower() == "unsharp_mask":
            sigma = parameters.get('sigma', 1.0)
            strength = parameters.get('strength', 1.5)
                
            # Create blurred version
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
                
            # Create sharpened image
            sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
            
        elif filter_name.lower() == "edge_detection":
            method = parameters.get('method', 'canny')
            if method.lower() == 'canny':
                low_threshold = parameters.get('low_threshold', 50)
                high_threshold = parameters.get('high_threshold', 150)
                return cv2.Canny(image, low_threshold, high_threshold)
            elif method.lower() == 'sobel':
                ksize = parameters.get('ksize', 3)
                dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
                dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
                magnitude = np.sqrt(dx*dx + dy*dy)
                return np.clip(magnitude, 0, 255).astype(np.uint8)
            
        elif filter_name.lower() == "histogram_equalization":
            if len(image.shape) == 3:
                # Convert to LAB, equalize L channel, convert back
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                return cv2.equalizeHist(image)
            
        elif filter_name.lower() == "contrast_brightness":
            alpha = parameters.get('contrast', 1.0)  # Contrast control (1.0-3.0)
            beta = parameters.get('brightness', 0)   # Brightness control (0-100)
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            return adjusted
            
        elif filter_name.lower() == "morphological":
            operation = parameters.get('operation', 'opening')
            kernel_size = parameters.get('kernel_size', 5)
            iterations = parameters.get('iterations', 1)
                
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                
            if operation.lower() == 'opening':
                return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
            elif operation.lower() == 'closing':
                return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            elif operation.lower() == 'erosion':
                return cv2.erode(image, kernel, iterations=iterations)
            elif operation.lower() == 'dilation':
                return cv2.dilate(image, kernel, iterations=iterations)
            
        else:
            print(f"Unknown filter: {filter_name}")
            return None
                
    except Exception as e:
        print(f"Error applying filter {filter_name}: {e}")
        return None
