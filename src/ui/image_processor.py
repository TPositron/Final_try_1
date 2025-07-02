"""
Image Processor Module
Handles all image processing operations including filtering, enhancement, and transformations.
"""

import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QObject, Signal

from src.services.simple_image_processing_service import ImageProcessingService


class ImageProcessor(QObject):
    """Handles all image processing operations and filtering."""
    
    # Signals
    filter_applied = Signal(str, dict)  # filter name, parameters
    filter_preview_ready = Signal(str, dict, object)  # filter name, parameters, preview image
    filters_reset = Signal()
    image_processed = Signal(object)  # processed image
    
    def __init__(self, main_window):
        """Initialize image processor with reference to main window."""
        super().__init__()
        self.main_window = main_window
        self.image_processing_service = ImageProcessingService()
        
        # Image processing state
        self.original_sem_image = None
        self.filtered_sem_image = None
        self.current_filters = {}  # Track applied filters
        self.filter_history = []  # Track filter application history
        
    def set_original_image(self, image):
        """Set the original SEM image for processing."""
        try:
            if image is not None:
                self.original_sem_image = image.copy()
                self.filtered_sem_image = image.copy()
                self.current_filters = {}
                self.filter_history = []
                
                # Initialize image processing service with the image
                self.image_processing_service.set_image(image)
                
                print(f"Set original image for processing: {image.shape}")
                return True
            else:
                print("Warning: Attempted to set None as original image")
                return False
                
        except Exception as e:
            print(f"Error setting original image: {e}")
            return False
    
    def on_filter_applied(self, filter_name, parameters):
        """Handle filter application from the filter panel."""
        try:
            if self.original_sem_image is None:
                QMessageBox.warning(
                    self.main_window,
                    "No Image",
                    "Please load a SEM image first."
                )
                return False
            
            print(f"Applying filter: {filter_name} with params: {parameters}")
            
            # Apply filter using image processing service
            filtered_image = self.image_processing_service.apply_filter(
                self.filtered_sem_image, filter_name, parameters
            )
            
            if filtered_image is not None:
                # Update the filtered image
                self.filtered_sem_image = filtered_image
                
                # Update main window's current image
                self.main_window.current_sem_image = filtered_image
                
                # Update the image viewer
                if hasattr(self.main_window, 'image_viewer'):
                    self.main_window.image_viewer.set_sem_image(filtered_image)
                
                # Track applied filter
                self.current_filters[filter_name] = parameters
                self.filter_history.append({
                    'filter': filter_name,
                    'parameters': parameters,
                    'timestamp': str(np.datetime64('now'))
                })
                
                # Update status
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage(f"Applied filter: {filter_name}")
                
                # Emit signals
                self.filter_applied.emit(filter_name, parameters)
                self.image_processed.emit(filtered_image)
                
                print(f"✓ Filter {filter_name} applied successfully")
                return True
            else:
                raise RuntimeError(f"Filter {filter_name} returned None")
                
        except Exception as e:
            error_msg = f"Failed to apply filter {filter_name}: {str(e)}"
            print(f"Filter application error: {error_msg}")
            QMessageBox.critical(self.main_window, "Filter Error", error_msg)
            return False
    
    def on_filter_preview(self, filter_name, parameters):
        """Handle filter preview generation."""
        try:
            if self.filtered_sem_image is None:
                return None
            
            print(f"Generating preview for filter: {filter_name}")
            
            # Apply filter to current filtered image for preview
            preview_image = self.image_processing_service.apply_filter(
                self.filtered_sem_image, filter_name, parameters
            )
            
            if preview_image is not None:
                # Emit preview signal
                self.filter_preview_ready.emit(filter_name, parameters, preview_image)
                print(f"✓ Preview generated for filter {filter_name}")
                return preview_image
            else:
                print(f"Warning: Preview generation failed for {filter_name}")
                return None
                
        except Exception as e:
            print(f"Filter preview error: {e}")
            return None
    
    def on_reset_filters(self):
        """Reset all applied filters and restore original image."""
        try:
            if self.original_sem_image is None:
                QMessageBox.warning(
                    self.main_window,
                    "No Image",
                    "No original image to reset to."
                )
                return False
            
            print("Resetting all filters")
            
            # Restore original image
            self.filtered_sem_image = self.original_sem_image.copy()
            self.main_window.current_sem_image = self.filtered_sem_image
            
            # Update image viewer
            if hasattr(self.main_window, 'image_viewer'):
                self.main_window.image_viewer.set_sem_image(self.filtered_sem_image)
            
            # Clear applied filters but keep history
            self.current_filters = {}
            
            # Reset image processing service
            self.image_processing_service.reset_to_original()
            
            # Update status
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage("Filters reset to original image")
            
            # Emit signals
            self.filters_reset.emit()
            self.image_processed.emit(self.filtered_sem_image)
            
            print("✓ All filters reset successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to reset filters: {str(e)}"
            print(f"Filter reset error: {error_msg}")
            QMessageBox.critical(self.main_window, "Reset Error", error_msg)
            return False
    
    def apply_custom_filter(self, filter_func, filter_name="Custom", **kwargs):
        """Apply a custom filter function to the current image."""
        try:
            if self.filtered_sem_image is None:
                raise ValueError("No image available for filtering")
            
            print(f"Applying custom filter: {filter_name}")
            
            # Apply custom filter
            filtered_image = filter_func(self.filtered_sem_image, **kwargs)
            
            if filtered_image is not None:
                # Update state
                self.filtered_sem_image = filtered_image
                self.main_window.current_sem_image = filtered_image
                
                # Update image viewer
                if hasattr(self.main_window, 'image_viewer'):
                    self.main_window.image_viewer.set_sem_image(filtered_image)
                
                # Track the custom filter
                self.current_filters[filter_name] = kwargs
                self.filter_history.append({
                    'filter': filter_name,
                    'parameters': kwargs,
                    'timestamp': str(np.datetime64('now'))
                })
                
                # Update status
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage(f"Applied custom filter: {filter_name}")
                
                # Emit signals
                self.filter_applied.emit(filter_name, kwargs)
                self.image_processed.emit(filtered_image)
                
                print(f"✓ Custom filter {filter_name} applied successfully")
                return True
            else:
                raise RuntimeError("Custom filter returned None")
                
        except Exception as e:
            error_msg = f"Failed to apply custom filter: {str(e)}"
            print(f"Custom filter error: {error_msg}")
            QMessageBox.critical(self.main_window, "Custom Filter Error", error_msg)
            return False
    
    def apply_enhancement(self, enhancement_type="contrast"):
        """Apply image enhancement operations."""
        try:
            if self.filtered_sem_image is None:
                raise ValueError("No image available for enhancement")
            
            enhanced_image = None
            
            if enhancement_type == "contrast":
                # Apply contrast enhancement
                enhanced_image = cv2.equalizeHist(self.filtered_sem_image.astype(np.uint8))
            elif enhancement_type == "sharpen":
                # Apply sharpening
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                enhanced_image = cv2.filter2D(self.filtered_sem_image, -1, kernel)
            elif enhancement_type == "denoise":
                # Apply denoising
                enhanced_image = cv2.fastNlMeansDenoising(self.filtered_sem_image.astype(np.uint8))
            else:
                raise ValueError(f"Unknown enhancement type: {enhancement_type}")
            
            if enhanced_image is not None:
                return self.apply_custom_filter(
                    lambda img, enh=enhanced_image: enh,
                    f"enhancement_{enhancement_type}"
                )
            else:
                raise RuntimeError(f"Enhancement {enhancement_type} failed")
                
        except Exception as e:
            error_msg = f"Failed to apply enhancement {enhancement_type}: {str(e)}"
            print(f"Enhancement error: {error_msg}")
            QMessageBox.critical(self.main_window, "Enhancement Error", error_msg)
            return False
    
    def create_filter_chain(self, filter_configs):
        """Apply a chain of filters in sequence."""
        try:
            if self.original_sem_image is None:
                raise ValueError("No original image available")
            
            print(f"Applying filter chain with {len(filter_configs)} filters")
            
            # Start with original image
            result_image = self.original_sem_image.copy()
            applied_filters = {}
            
            # Apply each filter in the chain
            for i, filter_config in enumerate(filter_configs):
                filter_name = filter_config.get('name')
                parameters = filter_config.get('parameters', {})
                
                print(f"Applying filter {i+1}/{len(filter_configs)}: {filter_name}")
                
                result_image = self.image_processing_service.apply_filter(
                    result_image, filter_name, parameters
                )
                
                if result_image is None:
                    raise RuntimeError(f"Filter {filter_name} failed in chain at step {i+1}")
                
                applied_filters[filter_name] = parameters
            
            # Update the filtered image
            self.filtered_sem_image = result_image
            self.main_window.current_sem_image = result_image
            
            # Update image viewer
            if hasattr(self.main_window, 'image_viewer'):
                self.main_window.image_viewer.set_sem_image(result_image)
            
            # Update tracking
            self.current_filters = applied_filters
            self.filter_history.append({
                'filter': 'filter_chain',
                'parameters': filter_configs,
                'timestamp': str(np.datetime64('now'))
            })
            
            # Update status
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage(f"Applied filter chain with {len(filter_configs)} filters")
            
            # Emit signal
            self.image_processed.emit(result_image)
            
            print(f"✓ Filter chain applied successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to apply filter chain: {str(e)}"
            print(f"Filter chain error: {error_msg}")
            QMessageBox.critical(self.main_window, "Filter Chain Error", error_msg)
            return False
    
    def get_image_statistics(self):
        """Get statistics about the current processed image."""
        if self.filtered_sem_image is None:
            return None
        
        img = self.filtered_sem_image
        stats = {
            'shape': img.shape,
            'dtype': str(img.dtype),
            'min': float(np.min(img)),
            'max': float(np.max(img)),
            'mean': float(np.mean(img)),
            'std': float(np.std(img)),
            'unique_values': len(np.unique(img))
        }
        
        return stats
    
    def get_applied_filters(self):
        """Get a copy of currently applied filters."""
        return self.current_filters.copy()
    
    def get_filter_history(self):
        """Get the complete filter application history."""
        return self.filter_history.copy()
    
    def has_original_image(self):
        """Check if an original image is loaded."""
        return self.original_sem_image is not None
    
    def has_filtered_image(self):
        """Check if a filtered image is available."""
        return self.filtered_sem_image is not None
    
    def export_processed_image(self, file_path):
        """Export the current processed image to a file."""
        try:
            if self.filtered_sem_image is None:
                raise ValueError("No processed image to export")
            
            # Convert to appropriate format for saving
            export_image = self.filtered_sem_image
            if export_image.dtype in [np.float64, np.float32]:
                # Normalize to 0-255 for saving
                normalized = cv2.normalize(export_image, None, 0, 255, cv2.NORM_MINMAX)
                export_image = normalized.astype(np.uint8)
            
            # Save the image
            success = cv2.imwrite(file_path, export_image)
            
            if success:
                print(f"✓ Processed image exported to: {file_path}")
                return True
            else:
                raise RuntimeError("cv2.imwrite failed")
                
        except Exception as e:
            error_msg = f"Failed to export processed image: {str(e)}"
            print(f"Export error: {error_msg}")
            QMessageBox.critical(self.main_window, "Export Error", error_msg)
            return False
    
    def compare_original_processed(self):
        """Create a side-by-side comparison of original and processed images."""
        try:
            if self.original_sem_image is None or self.filtered_sem_image is None:
                return None
            
            # Ensure both images have the same shape
            orig = self.original_sem_image
            proc = self.filtered_sem_image
            
            if orig.shape != proc.shape:
                # Resize processed to match original
                proc = cv2.resize(proc, (orig.shape[1], orig.shape[0]))
            
            # Create side-by-side comparison
            comparison = np.hstack([orig, proc])
            
            return comparison
            
        except Exception as e:
            print(f"Error creating comparison: {e}")
            return None
    
    def undo_last_filter(self):
        """Undo the last applied filter."""
        try:
            if not self.filter_history:
                print("No filters to undo")
                return False
            
            # Remove last filter from history
            last_filter = self.filter_history.pop()
            filter_name = last_filter['filter']
            
            # Remove from current filters
            if filter_name in self.current_filters:
                del self.current_filters[filter_name]
            
            # Reapply all remaining filters from original
            if self.current_filters:
                filter_configs = [
                    {'name': name, 'parameters': params}
                    for name, params in self.current_filters.items()
                ]
                self.create_filter_chain(filter_configs)
            else:
                # No filters left, reset to original
                self.on_reset_filters()
            
            print(f"✓ Undid filter: {filter_name}")
            return True
            
        except Exception as e:
            print(f"Error undoing last filter: {e}")
            return False
    
    def clear_processing_data(self):
        """Clear all processing data."""
        self.original_sem_image = None
        self.filtered_sem_image = None
        self.current_filters = {}
        self.filter_history = []
        print("Image processing data cleared")
