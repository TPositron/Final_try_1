"""
File Loading Service - SEM and GDS File Loading Operations

This service handles loading SEM images and GDS layout files into their respective
data models with automatic processing, cropping, and error handling. It provides
fallback mechanisms for different model interfaces and batch loading capabilities.

Main Functions:
- get_predefined_structure_info(): Returns metadata for predefined GDS structures

Main Class:
- FileLoadingService: Qt-based service for file loading operations

Key Methods:
- load_sem_image(): Loads SEM images with automatic 1024x666 cropping
- load_gds(): Loads GDS files with structure-specific bounds and layers
- get_current_sem(): Returns currently loaded SEM image
- get_current_gds(): Returns currently loaded GDS model
- clear_all(): Clears all loaded files from memory
- get_loading_status(): Returns status of loaded files
- load_multiple_sem_images(): Batch loading of multiple SEM files

Signals Emitted:
- sem_loaded(object): SEM image successfully loaded
- gds_loaded(object): GDS model successfully loaded
- aligned_gds_loaded(object): AlignedGdsModel successfully loaded
- loading_error(str): Error message when loading fails

Dependencies:
- Uses: pathlib.Path, cv2 (OpenCV), numpy (image processing)
- Uses: PySide6.QtCore (QObject, Signal for Qt integration)
- Uses: core/models (InitialGdsModel, AlignedGdsModel, SemImage)
- Uses: core/utils.get_logger (logging functionality)
- Used by: ui/file_operations.py (file loading UI)
- Used by: services/workflow_service.py (automated workflows)

Predefined GDS Structures:
1. Circpol_T2: bounds (688.55, 5736.55, 760.55, 5807.1), layer 14
2. IP935Left_11: bounds (693.99, 6406.40, 723.59, 6428.96), layers 1,2
3. IP935Left_14: bounds (980.959, 6025.959, 1001.770, 6044.979), layer 1
4. QC855GC_CROSS_Bottom: bounds (3730.00, 4700.99, 3756.00, 4760.00), layers 1,2
5. QC935_46: bounds (7195.558, 5046.99, 7203.99, 5055.33964), layer 1

SEM Image Processing:
- Automatic cropping to 1024x666 pixels (removes bottom portion)
- Support for .tif, .tiff, .png formats
- Fallback loading mechanisms for different SemImage interfaces
- Grayscale conversion and resizing when necessary
- Center-horizontal, top-vertical cropping strategy

GDS Loading Features:
- Structure-specific bounds and layer filtering
- Integration with InitialGdsModel and AlignedGdsModel
- Predefined structure metadata lookup
- Feature-focused model creation
- Alignment parameter support (planned)

Error Handling:
- Comprehensive exception handling for file operations
- Fallback mechanisms for missing model methods
- Detailed error logging and signal emission
- Graceful degradation when loading fails

Batch Operations:
- Multiple SEM image loading with individual error handling
- Progress tracking and success/failure reporting
- Memory-efficient processing of file lists

State Management:
- Current file tracking for SEM, GDS, and aligned models
- Loading status reporting
- Memory cleanup and file clearing operations
"""

from pathlib import Path
from typing import Optional, Dict, Any
from PySide6.QtCore import QObject, Signal
import cv2
import numpy as np

from src.core.models import InitialGdsModel, AlignedGdsModel, SemImage
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
    """Service for loading SEM and GDS files into data models."""
    
    # Signals
    sem_loaded = Signal(object)  # Emitted when SEM image is loaded
    gds_loaded = Signal(object)  # Emitted when GDS model is loaded
    aligned_gds_loaded = Signal(object)  # Emitted when AlignedGdsModel is loaded
    loading_error = Signal(str)  # Emitted when loading fails
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self._current_sem = None
        self._current_gds = None
        self._current_aligned_gds = None
    
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
    
    def load_gds(self, structure_num: int, file_path: Path, alignment_params: Optional[Dict[str, Any]] = None) -> Optional[AlignedGdsModel]:
        """
        Load a GDS file and prepare the model.

        Args:
            structure_num: The structure number to load.
            file_path: Path to the GDS file.
            alignment_params: Optional alignment parameters.

        Returns:
            AlignedGdsModel object if successful, None otherwise.
        """
        try:
            self.logger.info(f"Loading GDS structure {structure_num} from file: {file_path}")

            # Get predefined info for the selected structure
            structure_info = get_predefined_structure_info(structure_num)
            if not structure_info:
                raise ValueError(f"No predefined info for structure number {structure_num}")

            feature_bounds = structure_info['bounds']
            required_layers = structure_info['layers']

            # Create InitialGdsModel
            initial_model = InitialGdsModel(str(file_path))
            
            # Create AlignedGdsModel focused on the specific feature
            aligned_model = AlignedGdsModel(
                initial_model,
                feature_bounds=feature_bounds,
                required_layers=required_layers
            )

            # TODO: Apply alignment_params if provided
            if alignment_params:
                self.logger.warning("Alignment parameters are not yet implemented.")
                # aligned_model.apply_loaded_alignment(alignment_params)

            self._current_gds = aligned_model
            self.gds_loaded.emit(aligned_model)

            self.logger.info(f"Successfully loaded and prepared GDS structure {structure_num}")
            return aligned_model

        except Exception as e:
            error_msg = f"Failed to load GDS structure {structure_num} from file {file_path}: {e}"
            self.logger.error(error_msg)
            self.loading_error.emit(error_msg)
            return None
    
    def get_current_sem(self) -> Optional[SemImage]:
        """Get the currently loaded SEM image."""
        return self._current_sem
    
    def get_current_gds(self) -> Optional[AlignedGdsModel]:
        """Get the currently loaded GDS model."""
        return self._current_gds
    
    def get_current_aligned_gds(self) -> Optional[AlignedGdsModel]:
        """Get the currently loaded AlignedGdsModel."""
        return self._current_aligned_gds
    
    def clear_sem(self) -> None:
        """Clear the currently loaded SEM image."""
        self._current_sem = None
    
    def clear_gds(self) -> None:
        """Clear the currently loaded GDS model."""
        self._current_gds = None
    
    def clear_aligned_gds(self) -> None:
        """Clear the currently loaded AlignedGdsModel."""
        self._current_aligned_gds = None
    
    def clear_all(self) -> None:
        """Clear all loaded files."""
        self.clear_sem()
        self.clear_gds()
        self.clear_aligned_gds()
    
    def get_loading_status(self) -> Dict[str, bool]:
        """
        Get the current loading status.
        
        Returns:
            Dictionary indicating what files are loaded
        """
        return {
            'sem_loaded': self._current_sem is not None,
            'gds_loaded': self._current_gds is not None,
            'aligned_gds_loaded': self._current_aligned_gds is not None,
            'all_loaded': self._current_sem is not None and self._current_gds is not None and self._current_aligned_gds is not None
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
