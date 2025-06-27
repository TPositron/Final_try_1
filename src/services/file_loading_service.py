"""Service for loading selected files into models."""

from pathlib import Path
from typing import Optional, Dict, Any
from PySide6.QtCore import QObject, Signal
import cv2
import numpy as np

from ..core.models import InitialGDSModel, SEMImage
from ..core.utils import get_logger


class FileLoadingService(QObject):
    """Service for loading SEM and GDS files into data models."""
    
    # Signals
    sem_loaded = Signal(object)  # Emitted when SEM image is loaded
    gds_loaded = Signal(object)  # Emitted when GDS model is loaded
    loading_error = Signal(str)  # Emitted when loading fails
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self._current_sem = None
        self._current_gds = None
    
    def load_sem_image(self, file_path: Path) -> Optional[SEMImage]:
        """
        Load a SEM image file.
        
        Args:
            file_path: Path to the SEM image file
            
        Returns:
            SEMImage object if successful, None otherwise
        """
        try:
            self.logger.info(f"Loading SEM image: {file_path}")
            
            # Load image using OpenCV
            image_data = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            
            if image_data is None:
                raise ValueError(f"Failed to load image from {file_path}")
            
            # Create SEM image model
            sem_image = SEMImage(image_data, str(file_path))
            
            self._current_sem = sem_image
            self.sem_loaded.emit(sem_image)
            
            self.logger.info(f"Successfully loaded SEM image with shape {image_data.shape}")
            return sem_image
            
        except Exception as e:
            error_msg = f"Failed to load SEM image {file_path}: {e}"
            self.logger.error(error_msg)
            self.loading_error.emit(error_msg)
            return None
    
    def load_gds_file(self, file_path: Path) -> Optional[InitialGDSModel]:
        """
        Load a GDS file.
        
        Args:
            file_path: Path to the GDS file
            
        Returns:
            InitialGDSModel object if successful, None otherwise
        """
        try:
            self.logger.info(f"Loading GDS file: {file_path}")
            
            # Create GDS model
            gds_model = InitialGDSModel(str(file_path))
            
            # Get basic info about the GDS
            layers = gds_model.get_layers()
            bounds = gds_model.get_bounds()
            
            self.logger.info(f"Successfully loaded GDS file with {len(layers)} layers")
            self.logger.info(f"GDS bounds: {bounds}")
            
            self._current_gds = gds_model
            self.gds_loaded.emit(gds_model)
            
            return gds_model
            
        except Exception as e:
            error_msg = f"Failed to load GDS file {file_path}: {e}"
            self.logger.error(error_msg)
            self.loading_error.emit(error_msg)
            return None
    
    def get_current_sem(self) -> Optional[SEMImage]:
        """Get the currently loaded SEM image."""
        return self._current_sem
    
    def get_current_gds(self) -> Optional[InitialGDSModel]:
        """Get the currently loaded GDS model."""
        return self._current_gds
    
    def clear_sem(self) -> None:
        """Clear the currently loaded SEM image."""
        self._current_sem = None
    
    def clear_gds(self) -> None:
        """Clear the currently loaded GDS model."""
        self._current_gds = None
    
    def clear_all(self) -> None:
        """Clear all loaded files."""
        self.clear_sem()
        self.clear_gds()
    
    def get_loading_status(self) -> Dict[str, bool]:
        """
        Get the current loading status.
        
        Returns:
            Dictionary indicating what files are loaded
        """
        return {
            'sem_loaded': self._current_sem is not None,
            'gds_loaded': self._current_gds is not None,
            'both_loaded': self._current_sem is not None and self._current_gds is not None
        }
