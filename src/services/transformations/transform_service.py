"""
Transform Service for managing transformation operations.

This service coordinates transformation operations between the UI and core models.
It serves as a compatibility layer for existing code while delegating actual
transformation math to the utilities module.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from PySide6.QtCore import QObject, Signal

from src.utils.transformations import (
    create_transformation_matrix,
    apply_affine_transform,
    validate_transformation_parameters,
    convert_pixels_to_gds_units,
    convert_gds_to_pixel_units
)

logger = logging.getLogger(__name__)


class TransformService(QObject):
    """
    Service for coordinating transformation operations.
    
    This service provides a Qt-based interface for transformation operations
    and coordinates between UI components and the core transformation utilities.
    """
    
    # Qt Signals
    transform_applied = Signal(dict)  # transformation_data
    transform_updated = Signal(str, float)  # transform_type, value
    transform_error = Signal(str)  # error_message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Current transformation state
        self.current_transforms = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0
        }
        
        # Transformation history
        self.transform_history = []
        
        logger.debug("TransformService initialized")
    
    def apply_translation(self, dx: float, dy: float) -> bool:
        """
        Apply translation transformation.
        
        Args:
            dx: Translation in X direction
            dy: Translation in Y direction
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate parameters
            validation = validate_transformation_parameters(translation=(dx, dy))
            if not validation['valid']:
                error_msg = f"Invalid translation parameters: {validation['errors']}"
                logger.error(error_msg)
                self.transform_error.emit(error_msg)
                return False
            
            # Update current state
            old_value = (self.current_transforms['translate_x'], self.current_transforms['translate_y'])
            self.current_transforms['translate_x'] = dx
            self.current_transforms['translate_y'] = dy
            
            # Log the change
            logger.debug(f"Transform translation changed from {old_value} to ({dx}, {dy})")
            
            # Emit signals
            self.transform_updated.emit('translation', dx)  # Use X value as representative
            self._emit_transform_applied('translation', (dx, dy))
            
            return True
            
        except Exception as e:
            error_msg = f"Error applying translation: {e}"
            logger.error(error_msg)
            self.transform_error.emit(error_msg)
            return False
    
    def apply_rotation(self, degrees: float) -> bool:
        """
        Apply rotation transformation.
        
        Args:
            degrees: Rotation angle in degrees
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate parameters
            validation = validate_transformation_parameters(rotation_degrees=degrees)
            if not validation['valid']:
                error_msg = f"Invalid rotation parameters: {validation['errors']}"
                logger.error(error_msg)
                self.transform_error.emit(error_msg)
                return False
            
            # Update current state
            old_value = self.current_transforms['rotation']
            self.current_transforms['rotation'] = degrees
            
            # Log the change
            logger.debug(f"Transform rotation changed from {old_value} to {degrees}")
            
            # Emit signals
            self.transform_updated.emit('rotation', degrees)
            self._emit_transform_applied('rotation', degrees)
            
            return True
            
        except Exception as e:
            error_msg = f"Error applying rotation: {e}"
            logger.error(error_msg)
            self.transform_error.emit(error_msg)
            return False
    
    def apply_scale(self, scale: float) -> bool:
        """
        Apply scale transformation.
        
        Args:
            scale: Scale factor
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate parameters
            validation = validate_transformation_parameters(scale=scale)
            if not validation['valid']:
                error_msg = f"Invalid scale parameters: {validation['errors']}"
                logger.error(error_msg)
                self.transform_error.emit(error_msg)
                return False
            
            # Update current state
            old_value = self.current_transforms['scale']
            self.current_transforms['scale'] = scale
            
            # Log the change
            logger.debug(f"Transform scale changed from {old_value} to {scale}")
            
            # Emit signals
            self.transform_updated.emit('scale', scale)
            self._emit_transform_applied('scale', scale)
            
            return True
            
        except Exception as e:
            error_msg = f"Error applying scale: {e}"
            logger.error(error_msg)
            self.transform_error.emit(error_msg)
            return False
    
    def get_transformation_matrix(self) -> Optional[Any]:
        """
        Get the current transformation matrix.
        
        Returns:
            3x3 transformation matrix or None if error
        """
        try:
            matrix = create_transformation_matrix(
                translation=(self.current_transforms['translate_x'], 
                           self.current_transforms['translate_y']),
                rotation_degrees=self.current_transforms['rotation'],
                scale=self.current_transforms['scale']
            )
            return matrix
            
        except Exception as e:
            error_msg = f"Error creating transformation matrix: {e}"
            logger.error(error_msg)
            self.transform_error.emit(error_msg)
            return None
    
    def reset_transforms(self) -> None:
        """Reset all transformations to identity."""
        try:
            self.current_transforms = {
                'translate_x': 0.0,
                'translate_y': 0.0,
                'rotation': 0.0,
                'scale': 1.0
            }
            
            self.transform_history.clear()
            
            logger.info("All transformations reset to identity")
            self._emit_transform_applied('reset', None)
            
        except Exception as e:
            error_msg = f"Error resetting transforms: {e}"
            logger.error(error_msg)
            self.transform_error.emit(error_msg)
    
    def get_current_transforms(self) -> Dict[str, float]:
        """Get current transformation parameters."""
        return self.current_transforms.copy()
    
    def convert_pixels_to_gds(self, pixel_coords: Tuple[float, float], 
                             pixel_size: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to GDS units.
        
        Args:
            pixel_coords: (x, y) coordinates in pixels
            pixel_size: Size of one pixel in GDS units
            
        Returns:
            (x, y) coordinates in GDS units
        """
        try:
            return convert_pixels_to_gds_units(pixel_coords, pixel_size)
        except Exception as e:
            logger.error(f"Error converting pixels to GDS: {e}")
            return (0.0, 0.0)
    
    def convert_gds_to_pixels(self, gds_coords: Tuple[float, float], 
                             pixel_size: float) -> Tuple[float, float]:
        """
        Convert GDS coordinates to pixel units.
        
        Args:
            gds_coords: (x, y) coordinates in GDS units
            pixel_size: Size of one pixel in GDS units
            
        Returns:
            (x, y) coordinates in pixels
        """
        try:
            return convert_gds_to_pixel_units(gds_coords, pixel_size)
        except Exception as e:
            logger.error(f"Error converting GDS to pixels: {e}")
            return (0.0, 0.0)
    
    def apply_transform_from_dict(self, transform_params: dict) -> bool:
        """
        Apply transformation from dictionary parameters.
        
        Args:
            transform_params: Dictionary with transform parameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Apply translation
            if 'translate_x' in transform_params and 'translate_y' in transform_params:
                self.apply_translation(transform_params['translate_x'], transform_params['translate_y'])
            
            # Apply rotation
            if 'rotation' in transform_params:
                self.apply_rotation(transform_params['rotation'])
            
            # Apply scale
            if 'scale' in transform_params:
                self.apply_scale(transform_params['scale'])
            
            return True
            
        except Exception as e:
            error_msg = f"Error applying transform from dict: {e}"
            logger.error(error_msg)
            self.transform_error.emit(error_msg)
            return False
    
    def _emit_transform_applied(self, transform_type: str, value: Any) -> None:
        """Emit transform_applied signal with current state."""
        transform_data = {
            'type': transform_type,
            'value': value,
            'current_transforms': self.current_transforms.copy(),
            'matrix': self.get_transformation_matrix()
        }
        self.transform_applied.emit(transform_data)
        
        # Add to history
        self.transform_history.append({
            'type': transform_type,
            'value': value,
            'transforms_after': self.current_transforms.copy()
        })

