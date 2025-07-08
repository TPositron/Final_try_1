"""
Basic SEM Image class for the SEM/GDS Alignment Tool.

Simple class to store SEM image data with numpy array storage and basic metadata.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple
from PIL import Image


class SemImage:
    """
    Basic SEM image class with numpy array storage.
    
    Stores SEM image data and basic metadata like dimensions and file path.
    """
    
    def __init__(self, data: Union[str, Path, np.ndarray], file_path: Optional[Union[str, Path]] = None):
        """
        Initialize SEM image from file path or numpy array.
        
        Args:
            data: File path to image or numpy array with image data
            file_path: Optional file path if data is numpy array
        """
        self.file_path: Optional[Path] = None
        self.image_data: Optional[np.ndarray] = None
        self.metadata: dict = {}
        
        if isinstance(data, (str, Path)):
            # Load from file path
            self.file_path = Path(data)
            self._load_from_file()
        elif isinstance(data, np.ndarray):
            # Load from numpy array
            self.image_data = data.copy()
            if file_path:
                self.file_path = Path(file_path)
        else:
            raise ValueError("Data must be file path or numpy array")
        
        # Store basic metadata
        self._update_metadata()
    
    def _load_from_file(self) -> None:
        """Load image from file path."""
        # FIX 1: Check if file_path is None before using it
        if self.file_path is None:
            raise RuntimeError("No file path set for loading")
        
        # FIX 2: Now we know file_path is not None, safe to use
        if not self.file_path.exists():
            raise FileNotFoundError(f"Image file not found: {self.file_path}")
        
        try:
            # Use PIL to load image
            with Image.open(self.file_path) as img:
                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Convert to numpy array
                self.image_data = np.array(img, dtype=np.uint8)
                
        except Exception as e:
            raise RuntimeError(f"Failed to load image {self.file_path}: {e}")
    
    def _update_metadata(self) -> None:
        """Update metadata based on current image data."""
        if self.image_data is not None:
            self.metadata.update({
                'height': self.image_data.shape[0],
                'width': self.image_data.shape[1],
                'dtype': str(self.image_data.dtype),
                'size_bytes': self.image_data.nbytes,
                'file_path': str(self.file_path) if self.file_path else None
            })
    
    @property
    def shape(self) -> Tuple[int, int]:
        """
        Get image dimensions.
        
        Returns:
            Tuple of (height, width)
        """
        if self.image_data is not None:
            return self.image_data.shape
        return (0, 0)
    
    @property
    def width(self) -> int:
        """Get image width."""
        return self.shape[1] if len(self.shape) >= 2 else 0
    
    @property
    def height(self) -> int:
        """Get image height."""
        return self.shape[0] if len(self.shape) >= 1 else 0
    
    @property
    def size(self) -> Tuple[int, int]:
        """Get image size as (width, height)."""
        return (self.width, self.height)
    
    def get_data(self) -> np.ndarray:
        """
        Get image data as numpy array.
        
        Returns:
            Image data array
        """
        if self.image_data is None:
            raise RuntimeError("No image data loaded")
        return self.image_data.copy()
    
    def get_metadata(self) -> dict:
        """
        Get image metadata.
        
        Returns:
            Dictionary with image metadata
        """
        return self.metadata.copy()
    
    def save(self, file_path: Union[str, Path]) -> None:
        """
        Save image to file.
        
        Args:
            file_path: Path to save image
        """
        if self.image_data is None:
            raise RuntimeError("No image data to save")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert numpy array to PIL Image and save
            img = Image.fromarray(self.image_data)
            img.save(file_path)
            
            # Update file path
            self.file_path = file_path
            self._update_metadata()
            
        except Exception as e:
            raise RuntimeError(f"Failed to save image to {file_path}: {e}")
    
    def copy(self) -> 'SemImage':
        """
        Create a copy of the SEM image.
        
        Returns:
            New SemImage instance with copied data
        """
        if self.image_data is None:
            raise RuntimeError("No image data to copy")
        
        # Create new instance with copied data
        new_image = SemImage(self.image_data.copy(), self.file_path)
        new_image.metadata = self.metadata.copy()
        
        return new_image
    
    def __str__(self) -> str:
        """String representation of SEM image."""
        if self.image_data is not None:
            return f"SemImage({self.width}x{self.height}, {self.image_data.dtype})"
        return "SemImage(no data)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"SemImage(shape={self.shape}, dtype={self.image_data.dtype if self.image_data is not None else 'None'}, "
                f"file_path={self.file_path})")


def load_sem_image(file_path: Union[str, Path]) -> SemImage:
    """
    Convenience function to load a SEM image from file.
    
    Args:
        file_path: Path to SEM image file
        
    Returns:
        SemImage instance
    """
    return SemImage(file_path)


def create_sem_image(data: np.ndarray, file_path: Optional[Union[str, Path]] = None) -> SemImage:
    """
    Convenience function to create a SEM image from numpy array.
    
    Args:
        data: Image data as numpy array
        file_path: Optional source file path
        
    Returns:
        SemImage instance
    """
    return SemImage(data, file_path)
