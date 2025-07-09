"""
Transform Transparency - Transparency and Opacity Operations

This module provides transparency and opacity transformation functionality
for image blending, overlay operations, and visual composition in alignment
visualization.

Main Class:
- TransformTransparency: Handles transparency and opacity transformations

Key Methods:
- set_transparency(): Sets transparency percentage (0-100)
- set_opacity(): Sets opacity value (0.0-1.0)
- get_transparency(): Returns current transparency percentage
- get_opacity(): Returns current opacity value
- apply_transparency(): Blends two images with transparency
- apply_overlay(): Applies overlay with transparency
- create_color_overlay(): Creates colored overlay on image
- reset_transparency(): Resets to 50% transparency
- make_opaque(): Sets to fully opaque
- make_transparent(): Sets to fully transparent

Dependencies:
- Uses: numpy (array operations), cv2 (OpenCV for image resizing)
- Uses: logging (error reporting and debugging)
- Used by: Transformation services and visualization components
- Used by: UI overlay and blending operations

Features:
- Dual transparency/opacity representation
- Image normalization for consistent blending
- Alpha blending with configurable transparency
- Color overlay generation
- Validation with range checking
- Error handling and logging
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TransformTransparency:
    """Handles transparency and opacity transformations."""
    
    def __init__(self):
        self.transparency = 50  # Percentage (0-100)
        self.opacity = 0.5      # Float (0.0-1.0)
        self.min_transparency = 0
        self.max_transparency = 100
    
    def set_transparency(self, transparency: int) -> bool:
        """
        Set transparency value.
        
        Args:
            transparency: Transparency percentage (0-100, where 0=opaque, 100=transparent)
            
        Returns:
            True if value is valid and set, False otherwise
        """
        if not self._validate_transparency(transparency):
            return False
        
        self.transparency = transparency
        self.opacity = 1.0 - (transparency / 100.0)
        logger.debug(f"Set transparency: {transparency}% (opacity: {self.opacity})")
        return True
    
    def set_opacity(self, opacity: float) -> bool:
        """
        Set opacity value.
        
        Args:
            opacity: Opacity value (0.0-1.0, where 0.0=transparent, 1.0=opaque)
            
        Returns:
            True if value is valid and set, False otherwise
        """
        if not self._validate_opacity(opacity):
            return False
        
        self.opacity = opacity
        self.transparency = int((1.0 - opacity) * 100)
        logger.debug(f"Set opacity: {opacity} (transparency: {self.transparency}%)")
        return True
    
    def get_transparency(self) -> int:
        """Get current transparency percentage."""
        return self.transparency
    
    def get_opacity(self) -> float:
        """Get current opacity value."""
        return self.opacity
    
    def apply_transparency(self, foreground: np.ndarray, background: np.ndarray, 
                         transparency: Optional[int] = None) -> np.ndarray:
        """
        Apply transparency blending between two images.
        
        Args:
            foreground: Foreground image array
            background: Background image array
            transparency: Optional transparency percentage (uses stored value if None)
            
        Returns:
            Blended image
        """
        if transparency is None:
            alpha = self.opacity
        else:
            alpha = 1.0 - (transparency / 100.0)
        
        # Ensure images have same shape
        if foreground.shape != background.shape:
            import cv2
            background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
        
        # Normalize images to 0-1 range if needed
        fg_norm = self._normalize_image(foreground)
        bg_norm = self._normalize_image(background)
        
        # Alpha blending
        blended = fg_norm * alpha + bg_norm * (1.0 - alpha)
        return np.clip(blended, 0, 1)
    
    def apply_overlay(self, base_image: np.ndarray, overlay_image: np.ndarray,
                     transparency: Optional[int] = None) -> np.ndarray:
        """
        Apply overlay with transparency.
        
        Args:
            base_image: Base image array
            overlay_image: Overlay image array
            transparency: Optional transparency percentage
            
        Returns:
            Image with overlay applied
        """
        return self.apply_transparency(overlay_image, base_image, transparency)
    
    def create_color_overlay(self, image: np.ndarray, color: tuple = (0, 255, 0),
                           transparency: Optional[int] = None) -> np.ndarray:
        """
        Create colored overlay on image.
        
        Args:
            image: Input grayscale or color image
            color: RGB color tuple (default green)
            transparency: Optional transparency percentage
            
        Returns:
            Image with colored overlay
        """
        if transparency is None:
            alpha = self.opacity
        else:
            alpha = 1.0 - (transparency / 100.0)
        
        # Convert to 3-channel if grayscale
        if len(image.shape) == 2:
            base = np.stack([image, image, image], axis=-1)
        else:
            base = image.copy()
        
        # Normalize base image
        base_norm = self._normalize_image(base)
        
        # Create color overlay
        overlay = np.ones_like(base_norm)
        overlay[:, :, 0] = color[0] / 255.0
        overlay[:, :, 1] = color[1] / 255.0
        overlay[:, :, 2] = color[2] / 255.0
        
        # Apply transparency
        result = base_norm * (1.0 - alpha) + overlay * alpha
        return np.clip(result, 0, 1)
    
    def reset_transparency(self) -> None:
        """Reset transparency to 50% (half transparent)."""
        self.transparency = 50
        self.opacity = 0.5
        logger.debug("Reset transparency to 50%")
    
    def make_opaque(self) -> None:
        """Set to fully opaque (0% transparency)."""
        self.set_transparency(0)
    
    def make_transparent(self) -> None:
        """Set to fully transparent (100% transparency)."""
        self.set_transparency(100)
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range."""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
        else:
            # Assume already normalized or float
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                return (image - img_min) / (img_max - img_min)
            else:
                return image
    
    def _validate_transparency(self, transparency: int) -> bool:
        """Validate transparency value."""
        if not isinstance(transparency, int):
            logger.error(f"Transparency must be integer, got {type(transparency)}")
            return False
        
        if not (self.min_transparency <= transparency <= self.max_transparency):
            logger.error(f"Transparency out of range: {transparency} not in [{self.min_transparency}, {self.max_transparency}]")
            return False
        
        return True
    
    def _validate_opacity(self, opacity: float) -> bool:
        """Validate opacity value."""
        if not isinstance(opacity, (int, float)):
            logger.error(f"Opacity must be numeric, got {type(opacity)}")
            return False
        
        if not (0.0 <= opacity <= 1.0):
            logger.error(f"Opacity out of range: {opacity} not in [0.0, 1.0]")
            return False
        
        return True
