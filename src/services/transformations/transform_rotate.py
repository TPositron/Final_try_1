"""
Transform Rotate - Rotation transformation functionality.
"""

import numpy as np
import math
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TransformRotate:
    """Handles rotation transformations."""
    
    def __init__(self):
        self.rotation_degrees = 0.0
        self.min_rotation = -360.0
        self.max_rotation = 360.0
        self.snap_to_90 = False
    
    def set_rotation(self, degrees: float, snap_to_90: bool = False) -> bool:
        """
        Set rotation angle.
        
        Args:
            degrees: Rotation angle in degrees
            snap_to_90: Whether to snap to nearest 90° increment
            
        Returns:
            True if value is valid and set, False otherwise
        """
        if not self._validate_rotation(degrees):
            return False
        
        if snap_to_90:
            degrees = round(degrees / 90.0) * 90.0
        
        self.rotation_degrees = degrees % 360.0
        self.snap_to_90 = snap_to_90
        logger.debug(f"Set rotation: {self.rotation_degrees}° (snap_to_90={snap_to_90})")
        return True
    
    def get_rotation(self) -> float:
        """Get current rotation angle in degrees."""
        return self.rotation_degrees
    
    def apply_rotation(self, image: np.ndarray, degrees: Optional[float] = None, center: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Apply rotation transformation to image.
        
        Args:
            image: Input image array
            degrees: Optional rotation angle (uses stored value if None)
            center: Optional rotation center (uses image center if None)
            
        Returns:
            Rotated image
        """
        if degrees is None:
            degrees = self.rotation_degrees
        
        h, w = image.shape[:2]
        if center is None:
            center = (w / 2, h / 2)
        
        import cv2
        rotation_matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
    
    def get_rotation_matrix(self, degrees: Optional[float] = None, center: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Get 3x3 rotation transformation matrix.
        
        Args:
            degrees: Rotation angle (uses stored value if None)
            center: Rotation center (uses origin if None)
            
        Returns:
            3x3 transformation matrix
        """
        if degrees is None:
            degrees = self.rotation_degrees
        
        if center is None:
            center = (0.0, 0.0)
        
        angle_rad = math.radians(degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        cx, cy = center
        
        # Translation to center, rotation, translation back
        return np.array([
            [cos_a, -sin_a, cx - cx*cos_a + cy*sin_a],
            [sin_a,  cos_a, cy - cx*sin_a - cy*cos_a],
            [0,      0,     1]
        ], dtype=np.float64)
    
    def get_rotation_matrix_simple(self, degrees: Optional[float] = None) -> np.ndarray:
        """
        Get simple 3x3 rotation matrix around origin.
        
        Args:
            degrees: Rotation angle (uses stored value if None)
            
        Returns:
            3x3 transformation matrix
        """
        if degrees is None:
            degrees = self.rotation_degrees
        
        angle_rad = math.radians(degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ], dtype=np.float64)
    
    def reset_rotation(self) -> None:
        """Reset rotation to zero."""
        self.rotation_degrees = 0.0
        self.snap_to_90 = False
        logger.debug("Reset rotation to zero")
    
    def snap_to_nearest_90(self) -> float:
        """
        Snap current rotation to nearest 90° increment.
        
        Returns:
            Snapped rotation angle
        """
        snapped = round(self.rotation_degrees / 90.0) * 90.0
        self.rotation_degrees = snapped % 360.0
        self.snap_to_90 = True
        logger.debug(f"Snapped rotation to {self.rotation_degrees}°")
        return self.rotation_degrees
    
    def get_90_degree_component(self) -> int:
        """
        Get the 90° component of current rotation.
        
        Returns:
            Rotation in 90° increments (0, 90, 180, 270)
        """
        return int(round(self.rotation_degrees / 90.0) * 90) % 360
    
    def get_residual_rotation(self) -> float:
        """
        Get the residual rotation after removing 90° components.
        
        Returns:
            Residual rotation in degrees
        """
        ninety_component = self.get_90_degree_component()
        residual = self.rotation_degrees - ninety_component
        
        # Normalize to [-45, 45] range
        while residual > 45:
            residual -= 90
        while residual < -45:
            residual += 90
        
        return residual
    
    def _validate_rotation(self, degrees: float) -> bool:
        """Validate rotation value."""
        if not isinstance(degrees, (int, float)):
            logger.error(f"Rotation must be numeric, got {type(degrees)}")
            return False
        
        if not math.isfinite(degrees):
            logger.error(f"Rotation must be finite, got {degrees}")
            return False
        
        return True
