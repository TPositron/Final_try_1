"""
Transform Move - Translation Transformation Operations

This module provides translation (movement) transformation functionality
for image alignment operations, including validation, matrix generation,
and coordinate system conversion.

Main Class:
- TransformMove: Handles translation transformation operations

Key Methods:
- set_translation(): Sets translation values with validation
- get_translation(): Returns current translation values
- apply_translation(): Applies translation transformation to images
- get_translation_matrix(): Returns 3x3 transformation matrix
- reset_translation(): Resets translation to zero
- convert_pixels_to_units(): Converts pixel translation to coordinate units
- convert_units_to_pixels(): Converts coordinate translation to pixels

Dependencies:
- Uses: numpy (array operations), cv2 (OpenCV for image transformations)
- Uses: logging (error reporting and debugging)
- Used by: Transformation services and alignment operations
- Used by: UI transformation controls

Features:
- Translation validation with configurable limits
- 3x3 homogeneous transformation matrix generation
- Image transformation with OpenCV warpAffine
- Coordinate system conversion utilities
- Error handling and logging
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TransformMove:
    """Handles translation (movement) transformations."""
    
    def __init__(self):
        self.translation_x = 0.0
        self.translation_y = 0.0
        self.min_translation = -1000.0
        self.max_translation = 1000.0
    
    def set_translation(self, dx: float, dy: float) -> bool:
        """
        Set translation values.
        
        Args:
            dx: Translation in X direction
            dy: Translation in Y direction
            
        Returns:
            True if values are valid and set, False otherwise
        """
        if not self._validate_translation(dx, dy):
            return False
        
        self.translation_x = dx
        self.translation_y = dy
        logger.debug(f"Set translation: dx={dx}, dy={dy}")
        return True
    
    def get_translation(self) -> Tuple[float, float]:
        """Get current translation values."""
        return (self.translation_x, self.translation_y)
    
    def apply_translation(self, image: np.ndarray, dx: Optional[float] = None, dy: Optional[float] = None) -> np.ndarray:
        """
        Apply translation transformation to image.
        
        Args:
            image: Input image array
            dx: Optional X translation (uses stored value if None)
            dy: Optional Y translation (uses stored value if None)
            
        Returns:
            Translated image
        """
        if dx is None:
            dx = self.translation_x
        if dy is None:
            dy = self.translation_y
        
        h, w = image.shape[:2]
        translation_matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        
        import cv2
        return cv2.warpAffine(image, translation_matrix, (w, h), 
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, 
                             borderValue=(0,))  # type: ignore
    
    def get_translation_matrix(self, dx: Optional[float] = None, dy: Optional[float] = None) -> np.ndarray:
        """
        Get 3x3 translation transformation matrix.
        
        Args:
            dx: X translation (uses stored value if None)
            dy: Y translation (uses stored value if None)
            
        Returns:
            3x3 transformation matrix
        """
        if dx is None:
            dx = self.translation_x
        if dy is None:
            dy = self.translation_y
        
        return np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def reset_translation(self) -> None:
        """Reset translation to zero."""
        self.translation_x = 0.0
        self.translation_y = 0.0
        logger.debug("Reset translation to zero")
    
    def _validate_translation(self, dx: float, dy: float) -> bool:
        """Validate translation values."""
        if not (isinstance(dx, (int, float)) and isinstance(dy, (int, float))):
            logger.error(f"Translation values must be numeric, got dx={type(dx)}, dy={type(dy)}")
            return False
        
        if not (self.min_translation <= dx <= self.max_translation):
            logger.error(f"Translation X out of range: {dx} not in [{self.min_translation}, {self.max_translation}]")
            return False
        
        if not (self.min_translation <= dy <= self.max_translation):
            logger.error(f"Translation Y out of range: {dy} not in [{self.min_translation}, {self.max_translation}]")
            return False
        
        return True
    
    def convert_pixels_to_units(self, dx_pixels: float, dy_pixels: float, pixel_size: float) -> Tuple[float, float]:
        """
        Convert pixel translation to coordinate units.
        
        Args:
            dx_pixels: X translation in pixels
            dy_pixels: Y translation in pixels
            pixel_size: Size of one pixel in coordinate units
            
        Returns:
            (dx_units, dy_units) translation in coordinate units
        """
        return (dx_pixels * pixel_size, dy_pixels * pixel_size)
    
    def convert_units_to_pixels(self, dx_units: float, dy_units: float, pixel_size: float) -> Tuple[float, float]:
        """
        Convert coordinate unit translation to pixels.
        
        Args:
            dx_units: X translation in coordinate units
            dy_units: Y translation in coordinate units
            pixel_size: Size of one pixel in coordinate units
            
        Returns:
            (dx_pixels, dy_pixels) translation in pixels
        """
        if pixel_size <= 0:
            return (0.0, 0.0)
        return (dx_units / pixel_size, dy_units / pixel_size)
