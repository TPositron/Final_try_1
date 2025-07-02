"""
Transform Zoom - Scaling transformation functionality.
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TransformZoom:
    """Handles scaling (zoom) transformations."""
    
    def __init__(self):
        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 10.0
        self.scale_center = None  # (x, y) or None for image center
    
    def set_scale(self, scale: float, center: Optional[Tuple[float, float]] = None) -> bool:
        """
        Set scaling factor.
        
        Args:
            scale: Scaling factor (1.0 = no scaling, >1 = zoom in, <1 = zoom out)
            center: Optional scaling center (uses image center if None)
            
        Returns:
            True if value is valid and set, False otherwise
        """
        if not self._validate_scale(scale):
            return False
        
        self.scale_factor = scale
        self.scale_center = center
        logger.debug(f"Set scale: {scale} with center {center}")
        return True
    
    def get_scale(self) -> float:
        """Get current scaling factor."""
        return self.scale_factor
    
    def apply_zoom(self, image: np.ndarray, scale: Optional[float] = None, center: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Apply zoom (scaling) transformation to image.
        
        Args:
            image: Input image array
            scale: Optional scaling factor (uses stored value if None)
            center: Optional zoom center (uses stored center or image center if None)
            
        Returns:
            Scaled image
        """
        if scale is None:
            scale = self.scale_factor
        
        h, w = image.shape[:2]
        
        if center is None:
            center = self.scale_center if self.scale_center else (w / 2, h / 2)
        
        import cv2
        # Use getRotationMatrix2D with 0 rotation for scaling
        zoom_matrix = cv2.getRotationMatrix2D(center, 0, scale)
        return cv2.warpAffine(image, zoom_matrix, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
    
    def get_zoom_matrix(self, scale: Optional[float] = None, center: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Get 3x3 scaling transformation matrix.
        
        Args:
            scale: Scaling factor (uses stored value if None)
            center: Scaling center (uses origin if None)
            
        Returns:
            3x3 transformation matrix
        """
        if scale is None:
            scale = self.scale_factor
        
        if center is None:
            center = self.scale_center if self.scale_center else (0.0, 0.0)
        
        cx, cy = center
        
        # Translation to center, scaling, translation back
        return np.array([
            [scale, 0,     cx - cx*scale],
            [0,     scale, cy - cy*scale],
            [0,     0,     1]
        ], dtype=np.float64)
    
    def get_zoom_matrix_simple(self, scale: Optional[float] = None) -> np.ndarray:
        """
        Get simple 3x3 scaling matrix around origin.
        
        Args:
            scale: Scaling factor (uses stored value if None)
            
        Returns:
            3x3 transformation matrix
        """
        if scale is None:
            scale = self.scale_factor
        
        return np.array([
            [scale, 0,     0],
            [0,     scale, 0],
            [0,     0,     1]
        ], dtype=np.float64)
    
    def reset_zoom(self) -> None:
        """Reset scaling to 1.0 (no scaling)."""
        self.scale_factor = 1.0
        self.scale_center = None
        logger.debug("Reset zoom to 1.0")
    
    def zoom_in(self, factor: float = 1.25) -> float:
        """
        Zoom in by multiplying current scale.
        
        Args:
            factor: Zoom factor (default 1.25 = 25% zoom in)
            
        Returns:
            New scale factor
        """
        new_scale = self.scale_factor * factor
        if self.set_scale(new_scale):
            return self.scale_factor
        return self.scale_factor
    
    def zoom_out(self, factor: float = 0.8) -> float:
        """
        Zoom out by multiplying current scale.
        
        Args:
            factor: Zoom factor (default 0.8 = 20% zoom out)
            
        Returns:
            New scale factor
        """
        new_scale = self.scale_factor * factor
        if self.set_scale(new_scale):
            return self.scale_factor
        return self.scale_factor
    
    def fit_to_bounds(self, image_size: Tuple[int, int], target_size: Tuple[int, int]) -> float:
        """
        Calculate scale to fit image within target bounds.
        
        Args:
            image_size: (width, height) of source image
            target_size: (width, height) of target bounds
            
        Returns:
            Scale factor to fit image
        """
        img_w, img_h = image_size
        target_w, target_h = target_size
        
        scale_x = target_w / img_w
        scale_y = target_h / img_h
        
        # Use smaller scale to ensure image fits within bounds
        scale = min(scale_x, scale_y)
        
        if self.set_scale(scale):
            return self.scale_factor
        return self.scale_factor
    
    def _validate_scale(self, scale: float) -> bool:
        """Validate scaling factor."""
        if not isinstance(scale, (int, float)):
            logger.error(f"Scale must be numeric, got {type(scale)}")
            return False
        
        if scale <= 0:
            logger.error(f"Scale must be positive, got {scale}")
            return False
        
        if not (self.min_scale <= scale <= self.max_scale):
            logger.error(f"Scale out of range: {scale} not in [{self.min_scale}, {self.max_scale}]")
            return False
        
        return True
