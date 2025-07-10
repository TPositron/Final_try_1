"""
File Loading Service - SEM and GDS File Loading Operations

This service handles loading SEM images and provides GDS structure information
using the new simplified approach with gds_display_generator and gds_aligned_generator.

Main Functions:
- get_predefined_structure_info(): Returns metadata for predefined GDS structures

Main Class:
- FileLoadingService: Qt-based service for file loading operations

Key Methods:
- load_sem_image(): Loads SEM images with automatic 1024x666 cropping
- get_current_sem(): Returns currently loaded SEM image
- clear_all(): Clears all loaded files from memory
- get_loading_status(): Returns status of loaded files
- load_multiple_sem_images(): Batch loading of multiple SEM files

Signals Emitted:
- sem_loaded(object): SEM image successfully loaded
- loading_error(str): Error message when loading fails

Dependencies:
- Uses: pathlib.Path, cv2 (OpenCV), numpy (image processing)
- Uses: PySide6.QtCore (QObject, Signal for Qt integration)
- Uses: core/models (SemImage)
- Uses: core/gds_display_generator (GDS structure information)
- Uses: core/utils.get_logger (logging functionality)

Predefined GDS Structures:
1. Circpol_T2: bounds (688.55, 5736.55, 760.55, 5807.1), layer 14
2. IP935Left_11: bounds (693.99, 6406.40, 723.59, 6428.96), layers 1,2
3. IP935Left_14: bounds (980.959, 6025.959, 1001.770, 6044.979), layer 1
4. QC855GC_CROSS_Bottom: bounds (3730.00, 4700.99, 3756.00, 4760.00), layers 1,2
5. QC935_46: bounds (7195.558, 5046.99, 7203.99, 5055.33964), layer 1
"""

from pathlib import Path
from typing import Optional, Dict, Any
from PySide6.QtCore import QObject, Signal
import cv2
import numpy as np

from src.core.models import SemImage
from src.core.gds_display_generator import get_structure_info
from src.core.utils import get_logger


def get_predefined_structure_info(structure_num: int) -> Optional[Dict]:
    """
    Retrieve metadata for a predefined GDS structure.
    """
    structures = {
        1: {'name': 'Circpol_T2', 'bounds': (688.55, 5736.55, 760.55, 5807.1), 'layers': [14]},
        2: {'name': 'IP935Left_11', 'bounds': (693.99, 6406.40, 723.59, 6428.96), 'layers': [1, 2]},
        3: {'name': 'IP935Left_14', 'bounds': (980.959, 6025.959, 1001.770, 6044.979), 'layers': [1]},
        4: {'name': 'QC855GC_CROSS_Bottom', 'bounds': (3730.00, 4700.99, 3756.00, 4760.00), 'layers': [1, 2]},
        5: {'name': 'QC935_46', 'bounds': (7195.558, 5046.99, 7203.99, 5055.33964), 'layers': [1]}
    }
    return structures.get(structure_num)


class FileLoadingService(QObject):
    """Service for loading SEM files and providing GDS structure information."""
    
    # Signals
    sem_loaded = Signal(object)  # Emitted when SEM image is loaded
    loading_error = Signal(str)  # Emitted when loading fails
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self._current_sem = None
    
    def load_sem_image(self, file_path: Path) -> Optional[SemImage]:
        """
        Load a SEM image file with automatic bottom cropping.
        
        Args:
            file_path: Path to the SEM image file
            
        Returns:
            SemImage object if successful, None otherwise
        """
        try:
            self.logger.info(f"Loading SEM image: {file_path}")
            
            # Handle missing from_file method with fallback
            try:
                sem_image = SemImage.from_file(file_path)  # type: ignore
            except AttributeError:
                # Fallback: try direct constructor or other loading methods
                if hasattr(SemImage, 'load_from_file'):
                    sem_image = SemImage.load_from_file(file_path)  # type: ignore
                else:
                    # Manual loading with OpenCV and cropping
                    image_array = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                    if image_array is None:
                        raise ValueError(f"Could not load image from {file_path}")
                    
                    # Apply automatic bottom cropping (1024x666)
                    target_height, target_width = 666, 1024
                    h, w = image_array.shape
                    
                    if h >= target_height and w >= target_width:
                        # Crop from center horizontally, top vertically (remove bottom)
                        start_x = (w - target_width) // 2
                        start_y = 0  # Start from top
                        cropped = image_array[start_y:start_y + target_height, 
                                            start_x:start_x + target_width]
                    else:
                        # Resize if too small
                        cropped = cv2.resize(image_array, (target_width, target_height))
                    
                    # Create SemImage object
                    sem_image = SemImage(cropped, str(file_path))
            
            self._current_sem = sem_image
            self.sem_loaded.emit(sem_image)
            
            # Handle missing shape attribute with fallback using getattr
            shape_info = getattr(sem_image, 'shape', None)
            if shape_info is None:
                # Try alternative attributes
                height = getattr(sem_image, 'height', None)
                width = getattr(sem_image, 'width', None)
                if height is not None and width is not None:
                    shape_info = (height, width)
                else:
                    # Try to get shape from underlying image data
                    image_data = (getattr(sem_image, 'image', None) or 
                                getattr(sem_image, 'data', None) or 
                                getattr(sem_image, '_image', None))
                    if image_data is not None and hasattr(image_data, 'shape'):
                        shape_info = image_data.shape
                    else:
                        shape_info = "unknown"
            
            self.logger.info(f"Successfully loaded SEM image with shape {shape_info}")
            return sem_image
            
        except Exception as e:
            error_msg = f"Failed to load SEM image {file_path}: {e}"
            self.logger.error(error_msg)
            self.loading_error.emit(error_msg)
            return None
    
    def get_gds_structure_info(self, structure_num: int) -> Optional[Dict]:
        """
        Get information about a GDS structure using the new approach.

        Args:
            structure_num: The structure number (1-5).

        Returns:
            Structure information dictionary or None if not found.
        """
        try:
            self.logger.info(f"Getting GDS structure info for structure {structure_num}")
            return get_structure_info(structure_num)
        except Exception as e:
            error_msg = f"Failed to get GDS structure info for {structure_num}: {e}"
            self.logger.error(error_msg)
            self.loading_error.emit(error_msg)
            return None
    
    def get_current_sem(self) -> Optional[SemImage]:
        """Get the currently loaded SEM image."""
        return self._current_sem
    

    
    def clear_sem(self) -> None:
        """Clear the currently loaded SEM image."""
        self._current_sem = None
    

    
    def clear_all(self) -> None:
        """Clear all loaded files."""
        self.clear_sem()
    
    def get_loading_status(self) -> Dict[str, bool]:
        """
        Get the current loading status.
        
        Returns:
            Dictionary indicating what files are loaded
        """
        return {
            'sem_loaded': self._current_sem is not None
        }
    
    def load_multiple_sem_images(self, file_paths: list[Path]) -> Dict[str, Optional[SemImage]]:
        """
        Load multiple SEM image files (basic batch operation).
        
        Args:
            file_paths: List of paths to SEM image files
            
        Returns:
            Dictionary mapping file names to SemImage objects (None if failed)
        """
        results = {}
        
        self.logger.info(f"Loading {len(file_paths)} SEM images in batch")
        
        for file_path in file_paths:
            try:
                sem_image = self.load_sem_image(file_path)
                results[file_path.name] = sem_image
                
            except Exception as e:
                self.logger.warning(f"Failed to load {file_path}: {e}")
                results[file_path.name] = None
        
        successful_loads = sum(1 for result in results.values() if result is not None)
        self.logger.info(f"Batch loading completed: {successful_loads}/{len(file_paths)} successful")
        
        return results
