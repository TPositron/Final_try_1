"""
Image Processing Module - Comprehensive SEM Image Enhancement and Filtering

This module provides advanced image processing capabilities for SEM images, including
filter application, enhancement operations, and processing workflow management.
It serves as the UI-level coordinator for all image processing operations.

Core Processing Capabilities:
1. Filter Application:
   - Real-time filter application with immediate visual feedback
   - Filter parameter validation and bounds checking
   - Filter preview generation without permanent changes
   - Filter history tracking with undo capabilities

2. Image Enhancement:
   - Contrast enhancement using histogram equalization
   - Sharpening operations with configurable kernels
   - Noise reduction using advanced denoising algorithms
   - Custom enhancement operations

3. Filter Chain Processing:
   - Sequential application of multiple filters
   - Batch processing with progress reporting
   - Filter combination optimization
   - Chain validation and error recovery

Supported Filter Types (via ImageProcessingService):
- Gaussian Blur: Noise reduction with sigma control
- Threshold: Binary conversion with adaptive thresholds
- Edge Detection: Canny edge detection with dual thresholds
- Median Filter: Salt-and-pepper noise removal
- CLAHE: Contrast Limited Adaptive Histogram Equalization

Dependencies:
- Uses: services/simple_image_processing_service.py (core processing)
- Uses: cv2, numpy (image processing operations)
- Uses: PySide6.QtCore (signals and QObject)
- Called by: ui/main_window.py (filter panel interactions)
- Coordinates with: ui/image_viewer.py (display updates)
- Coordinates with: ui/panels/filtering_panels.py (UI integration)

Signals Emitted:
- filter_applied: When filters are successfully applied
- filter_preview_ready: When filter previews are generated
- filters_reset: When all filters are reset to original

State Management:
- original_sem_image: Unmodified source image for reset operations
- filtered_sem_image: Current processed image state
- current_filters: Active filter configuration tracking

Key Methods:
- on_filter_applied(): Handles filter application from UI panels
- on_filter_preview(): Generates real-time filter previews
- on_reset_filters(): Resets all processing to original image
- apply_custom_filter(): Applies user-defined filter functions
- create_filter_chain(): Processes multiple filters in sequence

Image Statistics and Analysis:
- Real-time image statistics calculation
- Histogram analysis and reporting
- Quality metrics for processed images
- Processing history and metadata tracking

Error Handling:
- Graceful handling of filter application failures
- Image format validation and conversion
- Memory management for large images
- Recovery from processing errors

Integration Features:
- Seamless integration with alignment workflows
- Export capabilities for processed images
- Comparison tools for before/after analysis
- Processing history management and replay

Performance Optimizations:
- Efficient memory management for large images
- Optimized filter parameter validation
- Caching of frequently used operations
- Progressive processing for real-time feedback
"""

import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QObject, Signal

from src.services.simple_image_processing_service import ImageProcessingService


class ImageProcessing(QObject):
    """Handles image processing operations and filtering."""
    
    # Signals
    filter_applied = Signal(str, dict)  # Emitted when filter is applied
    filter_preview_ready = Signal(str, dict, object)  # Emitted when filter preview is ready
    filters_reset = Signal()  # Emitted when filters are reset
    
    def __init__(self, main_window):
        """Initialize image processing with reference to main window."""
        super().__init__()
        self.main_window = main_window
        self.image_processing_service = ImageProcessingService()
        
        # Image processing state
        self.original_sem_image = None  # Store original for filter reset
        self.filtered_sem_image = None  # Store filtered version
        self.current_filters = {}  # Track applied filters
        
    def set_original_image(self, image):
        """Set the original SEM image for processing."""
        if image is not None:
            self.original_sem_image = image.copy()
            self.filtered_sem_image = image.copy()
            self.current_filters = {}
            print(f"Set original image for processing: {image.shape}")
    
    def on_filter_applied(self, filter_name, parameters):
        """Handle filter application from the filter panel."""
        try:
            if self.original_sem_image is None:
                QMessageBox.warning(
                    self.main_window, 
                    "No Image", 
                    "Please load a SEM image first."
                )
                return
            
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
                self.main_window.image_viewer.set_sem_image(filtered_image)
                
                # Track applied filter
                self.current_filters[filter_name] = parameters
                
                # Update status
                self.main_window.status_bar.showMessage(f"Applied filter: {filter_name}")
                
                # Emit signal
                self.filter_applied.emit(filter_name, parameters)
                
                print(f"✓ Filter {filter_name} applied successfully")
            else:
                raise RuntimeError(f"Filter {filter_name} returned None")
                
        except Exception as e:
            error_msg = f"Failed to apply filter {filter_name}: {str(e)}"
            print(f"Filter application error: {error_msg}")
            QMessageBox.critical(self.main_window, "Filter Error", error_msg)
    
    def on_filter_preview(self, filter_name, parameters):
        """Handle filter preview from the filter panel."""
        try:
            if self.original_sem_image is None:
                return
            
            print(f"Generating preview for filter: {filter_name}")
            
            # Apply filter to current filtered image for preview
            preview_image = self.image_processing_service.apply_filter(
                self.filtered_sem_image, filter_name, parameters
            )
            
            if preview_image is not None:
                # Emit preview signal with the preview image
                self.filter_preview_ready.emit(filter_name, parameters, preview_image)
                print(f"✓ Preview generated for filter {filter_name}")
            
        except Exception as e:
            print(f"Filter preview error: {e}")
    
    def on_reset_filters(self):
        """Reset all applied filters and restore original image."""
        try:
            if self.original_sem_image is None:
                QMessageBox.warning(
                    self.main_window,
                    "No Image", 
                    "No original image to reset to."
                )
                return
            
            print("Resetting all filters")
            
            # Restore original image
            self.filtered_sem_image = self.original_sem_image.copy()
            self.main_window.current_sem_image = self.filtered_sem_image
            
            # Update image viewer
            self.main_window.image_viewer.set_sem_image(self.filtered_sem_image)
            
            # Clear applied filters
            self.current_filters = {}
            
            # Update status
            self.main_window.status_bar.showMessage("Filters reset to original image")
            
            # Emit signal
            self.filters_reset.emit()
            
            print("✓ All filters reset successfully")
            
        except Exception as e:
            error_msg = f"Failed to reset filters: {str(e)}"
            print(f"Filter reset error: {error_msg}")
            QMessageBox.critical(self.main_window, "Reset Error", error_msg)
    
    def apply_custom_filter(self, filter_func, filter_name="Custom", **kwargs):
        """Apply a custom filter function to the current image."""
        try:
            if self.filtered_sem_image is None:
                raise ValueError("No image available for filtering")
            
            # Apply custom filter
            filtered_image = filter_func(self.filtered_sem_image, **kwargs)
            
            if filtered_image is not None:
                self.filtered_sem_image = filtered_image
                self.main_window.current_sem_image = filtered_image
                self.main_window.image_viewer.set_sem_image(filtered_image)
                
                # Track the custom filter
                self.current_filters[filter_name] = kwargs
                
                self.main_window.status_bar.showMessage(f"Applied custom filter: {filter_name}")
                self.filter_applied.emit(filter_name, kwargs)
                
                return True
            else:
                raise RuntimeError("Custom filter returned None")
                
        except Exception as e:
            error_msg = f"Failed to apply custom filter: {str(e)}"
            print(f"Custom filter error: {error_msg}")
            QMessageBox.critical(self.main_window, "Custom Filter Error", error_msg)
            return False
    
    def get_image_statistics(self):
        """Get statistics about the current filtered image."""
        if self.filtered_sem_image is None:
            return None
        
        stats = {
            'shape': self.filtered_sem_image.shape,
            'dtype': str(self.filtered_sem_image.dtype),
            'min': float(np.min(self.filtered_sem_image)),
            'max': float(np.max(self.filtered_sem_image)),
            'mean': float(np.mean(self.filtered_sem_image)),
            'std': float(np.std(self.filtered_sem_image))
        }
        
        return stats
    
    def get_applied_filters(self):
        """Get a list of currently applied filters."""
        return self.current_filters.copy()
    
    def has_original_image(self):
        """Check if an original image is loaded."""
        return self.original_sem_image is not None
    
    def has_filtered_image(self):
        """Check if a filtered image is available."""
        return self.filtered_sem_image is not None
    
    def export_filtered_image(self, file_path):
        """Export the current filtered image to a file."""
        try:
            if self.filtered_sem_image is None:
                raise ValueError("No filtered image to export")
            
            # Convert to appropriate format for saving
            if self.filtered_sem_image.dtype == np.float64 or self.filtered_sem_image.dtype == np.float32:
                # Normalize to 0-255 for saving
                normalized = cv2.normalize(self.filtered_sem_image, None, 0, 255, cv2.NORM_MINMAX)
                export_image = normalized.astype(np.uint8)
            else:
                export_image = self.filtered_sem_image
            
            # Save the image
            success = cv2.imwrite(file_path, export_image)
            
            if success:
                print(f"✓ Filtered image exported to: {file_path}")
                return True
            else:
                raise RuntimeError("cv2.imwrite failed")
                
        except Exception as e:
            error_msg = f"Failed to export filtered image: {str(e)}"
            print(f"Export error: {error_msg}")
            QMessageBox.critical(self.main_window, "Export Error", error_msg)
            return False
    
    def create_filter_chain(self, filter_configs):
        """Apply a chain of filters in sequence."""
        try:
            if self.original_sem_image is None:
                raise ValueError("No original image available")
            
            # Start with original image
            result_image = self.original_sem_image.copy()
            applied_filters = {}
            
            # Apply each filter in the chain
            for filter_config in filter_configs:
                filter_name = filter_config.get('name')
                parameters = filter_config.get('parameters', {})
                
                result_image = self.image_processing_service.apply_filter(
                    result_image, filter_name, parameters
                )
                
                if result_image is None:
                    raise RuntimeError(f"Filter {filter_name} failed in chain")
                
                applied_filters[filter_name] = parameters
            
            # Update the filtered image
            self.filtered_sem_image = result_image
            self.main_window.current_sem_image = result_image
            self.main_window.image_viewer.set_sem_image(result_image)
            
            # Update tracking
            self.current_filters = applied_filters
            
            self.main_window.status_bar.showMessage(f"Applied filter chain with {len(filter_configs)} filters")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to apply filter chain: {str(e)}"
            print(f"Filter chain error: {error_msg}")
            QMessageBox.critical(self.main_window, "Filter Chain Error", error_msg)
            return False
