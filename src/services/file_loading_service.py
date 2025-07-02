"""Service for loading selected files into models."""

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
            
            # Use SemImage.from_file() which includes automatic bottom cropping
            sem_image = SemImage.from_file(file_path)
            
            self._current_sem = sem_image
            self.sem_loaded.emit(sem_image)
            
            self.logger.info(f"Successfully loaded SEM image with shape {sem_image.shape}")
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
    
    def get_current_gds(self) -> Optional[InitialGdsModel]:
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
