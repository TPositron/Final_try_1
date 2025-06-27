"""Service for managing translate/rotate/zoom/transparency transforms."""

from typing import Dict, Any, Tuple
from PySide6.QtCore import QObject, Signal
import numpy as np

from ..core.utils import get_logger


class TransformService(QObject):
    """Service for managing geometric and visual transforms."""
    
    # Signals
    transform_changed = Signal(str, float)  # transform_type, value
    transforms_reset = Signal()
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self._transforms = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0,
            'transparency': 1.0
        }
        self._transform_limits = {
            'translate_x': (-1000.0, 1000.0),
            'translate_y': (-1000.0, 1000.0),
            'rotation': (-360.0, 360.0),
            'scale': (0.1, 10.0),
            'transparency': (0.0, 1.0)
        }
    
    def set_transform(self, transform_type: str, value: float) -> bool:
        """
        Set a transform parameter.
        
        Args:
            transform_type: Type of transform (translate_x, translate_y, rotation, scale, transparency)
            value: New value for the transform
            
        Returns:
            True if successful, False if invalid transform type or value
        """
        if transform_type not in self._transforms:
            self.logger.warning(f"Unknown transform type: {transform_type}")
            return False
        
        # Check limits
        min_val, max_val = self._transform_limits[transform_type]
        if not (min_val <= value <= max_val):
            self.logger.warning(f"Transform value {value} out of range [{min_val}, {max_val}] for {transform_type}")
            return False
        
        old_value = self._transforms[transform_type]
        self._transforms[transform_type] = value
        
        if old_value != value:
            self.transform_changed.emit(transform_type, value)
            self.logger.debug(f"Transform {transform_type} changed from {old_value} to {value}")
        
        return True
    
    def get_transform(self, transform_type: str) -> float:
        """Get current value of a transform parameter."""
        return self._transforms.get(transform_type, 0.0)
    
    def get_all_transforms(self) -> Dict[str, float]:
        """Get all current transform values."""
        return self._transforms.copy()
    
    def adjust_transform(self, transform_type: str, delta: float) -> bool:
        """
        Adjust a transform parameter by a delta value.
        
        Args:
            transform_type: Type of transform
            delta: Amount to add to current value
            
        Returns:
            True if successful, False otherwise
        """
        current_value = self.get_transform(transform_type)
        return self.set_transform(transform_type, current_value + delta)
    
    def reset_transform(self, transform_type: str) -> bool:
        """Reset a specific transform to its default value."""
        default_values = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0,
            'transparency': 1.0
        }
        
        if transform_type in default_values:
            return self.set_transform(transform_type, default_values[transform_type])
        return False
    
    def reset_all_transforms(self) -> None:
        """Reset all transforms to their default values."""
        old_transforms = self._transforms.copy()
        
        self._transforms = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0,
            'transparency': 1.0
        }
        
        # Emit signals for changed transforms
        for transform_type, new_value in self._transforms.items():
            if old_transforms[transform_type] != new_value:
                self.transform_changed.emit(transform_type, new_value)
        
        self.transforms_reset.emit()
        self.logger.info("All transforms reset to default values")
    
    def set_transform_limits(self, transform_type: str, min_val: float, max_val: float) -> bool:
        """
        Set limits for a transform parameter.
        
        Args:
            transform_type: Type of transform
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            True if successful, False otherwise
        """
        if transform_type not in self._transforms:
            return False
        
        if min_val >= max_val:
            self.logger.warning(f"Invalid limits: min_val ({min_val}) >= max_val ({max_val})")
            return False
        
        self._transform_limits[transform_type] = (min_val, max_val)
        
        # Clamp current value to new limits if necessary
        current_value = self._transforms[transform_type]
        if current_value < min_val:
            self.set_transform(transform_type, min_val)
        elif current_value > max_val:
            self.set_transform(transform_type, max_val)
        
        return True
    
    def get_transform_limits(self, transform_type: str) -> Tuple[float, float]:
        """Get the limits for a transform parameter."""
        return self._transform_limits.get(transform_type, (0.0, 0.0))
    
    def apply_transforms_to_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Apply current transforms to a single point.
        
        Args:
            point: (x, y) coordinates
            
        Returns:
            Transformed (x, y) coordinates
        """
        x, y = point
        
        # Apply scaling
        x *= self._transforms['scale']
        y *= self._transforms['scale']
        
        # Apply rotation
        if self._transforms['rotation'] != 0.0:
            angle = np.radians(self._transforms['rotation'])
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            new_x = x * cos_a - y * sin_a
            new_y = x * sin_a + y * cos_a
            x, y = new_x, new_y
        
        # Apply translation
        x += self._transforms['translate_x']
        y += self._transforms['translate_y']
        
        return (x, y)
    
    def get_transform_matrix(self) -> np.ndarray:
        """
        Get the 3x3 transformation matrix for current transforms.
        
        Returns:
            3x3 transformation matrix
        """
        # Create transformation matrix
        angle = np.radians(self._transforms['rotation'])
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        scale = self._transforms['scale']
        tx, ty = self._transforms['translate_x'], self._transforms['translate_y']
        
        matrix = np.array([
            [scale * cos_a, -scale * sin_a, tx],
            [scale * sin_a,  scale * cos_a, ty],
            [0,              0,             1]
        ])
        
        return matrix
    
    def copy_transforms_from(self, other_service: 'TransformService') -> None:
        """Copy transform values from another TransformService."""
        other_transforms = other_service.get_all_transforms()
        
        for transform_type, value in other_transforms.items():
            self.set_transform(transform_type, value)
        
        self.logger.info("Copied transforms from another service")
