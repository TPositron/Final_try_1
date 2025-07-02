"""
Transformation Service
Handles applying transformations to GDS structures using the new working code approach.
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from PySide6.QtCore import QObject, Signal

from ..core.gds_aligned_generator import generate_aligned_gds, generate_transformed_gds
from ..core.gds_display_generator import get_structure_info


class TransformationService(QObject):
    """
    Service for applying transformations to GDS structures using the working code approach.
    """
    
    # Signals
    transformation_applied = Signal(np.ndarray)  # transformed_image
    transformation_error = Signal(str)         # error_message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_structure_num = None
        self.base_transform = {
            'rotation': 0.0,
            'zoom': 100.0,
            'move_x': 0.0,
            'move_y': 0.0
        }
    
    def set_structure(self, structure_num: int) -> bool:
        """
        Set the current structure for transformations.
        
        Args:
            structure_num: Structure number (1-5)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate structure exists
            info = get_structure_info(structure_num)
            if info is None:
                self.transformation_error.emit(f"Structure {structure_num} not found")
                return False
            
            self.current_structure_num = structure_num
            return True
            
        except Exception as e:
            self.transformation_error.emit(f"Error setting structure: {e}")
            return False
    
    def apply_transformations(self, 
                            rotation: float = 0.0,
                            zoom: float = 100.0,
                            move_x: float = 0.0,
                            move_y: float = 0.0,
                            target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        """
        Apply transformations to the current structure.
        
        Args:
            rotation: Rotation angle in degrees
            zoom: Zoom percentage (100 = no change)
            move_x: X movement in pixels
            move_y: Y movement in pixels
            target_size: Output image size
            
        Returns:
            Transformed image or None if failed
        """
        try:
            if self.current_structure_num is None:
                self.transformation_error.emit("No structure set for transformation")
                return None
            
            # Generate transformed image using working code
            transformed_image = generate_transformed_gds(
                self.current_structure_num,
                rotation=rotation,
                zoom=zoom,
                move_x=move_x,
                move_y=move_y,
                target_size=target_size
            )
            
            if transformed_image is not None:
                # Update current transform state
                self.base_transform = {
                    'rotation': rotation,
                    'zoom': zoom,
                    'move_x': move_x,
                    'move_y': move_y
                }
                
                # Emit signal
                self.transformation_applied.emit(transformed_image)
                
            return transformed_image
            
        except Exception as e:
            self.transformation_error.emit(f"Error applying transformations: {e}")
            return None
    
    def apply_transform_dict(self, transform_params: Dict[str, float], 
                           target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        """
        Apply transformations using a parameter dictionary.
        
        Args:
            transform_params: Dictionary with transformation parameters
            target_size: Output image size
            
        Returns:
            Transformed image or None if failed
        """
        try:
            if self.current_structure_num is None:
                self.transformation_error.emit("No structure set for transformation")
                return None
            
            # Generate aligned image using working code
            transformed_image, bounds = generate_aligned_gds(
                self.current_structure_num,
                transform_params,
                target_size
            )
            
            if transformed_image is not None:
                # Update current transform state
                self.base_transform.update(transform_params)
                
                # Emit signal
                self.transformation_applied.emit(transformed_image)
                
            return transformed_image
            
        except Exception as e:
            self.transformation_error.emit(f"Error applying transform dict: {e}")
            return None
    
    def get_current_transform(self) -> Dict[str, float]:
        """Get the current transformation parameters."""
        return self.base_transform.copy()
    
    def reset_transforms(self) -> Optional[np.ndarray]:
        """
        Reset all transformations to default values.
        
        Returns:
            Reset image or None if failed
        """
        return self.apply_transformations(
            rotation=0.0,
            zoom=100.0,
            move_x=0.0,
            move_y=0.0
        )
    
    def rotate_structure(self, angle: float) -> Optional[np.ndarray]:
        """
        Apply rotation to the current structure.
        
        Args:
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image or None if failed
        """
        current = self.base_transform.copy()
        current['rotation'] = angle
        return self.apply_transformations(**current)
    
    def zoom_structure(self, zoom_percentage: float) -> Optional[np.ndarray]:
        """
        Apply zoom to the current structure.
        
        Args:
            zoom_percentage: Zoom percentage (100 = no change)
            
        Returns:
            Zoomed image or None if failed
        """
        current = self.base_transform.copy()
        current['zoom'] = zoom_percentage
        return self.apply_transformations(**current)
    
    def move_structure(self, dx: float, dy: float) -> Optional[np.ndarray]:
        """
        Apply movement to the current structure.
        
        Args:
            dx: X movement in pixels
            dy: Y movement in pixels
            
        Returns:
            Moved image or None if failed
        """
        current = self.base_transform.copy()
        current['move_x'] += dx
        current['move_y'] += dy
        return self.apply_transformations(**current)
    
    def set_absolute_position(self, x: float, y: float) -> Optional[np.ndarray]:
        """
        Set absolute position for the structure.
        
        Args:
            x: Absolute X position in pixels
            y: Absolute Y position in pixels
            
        Returns:
            Moved image or None if failed
        """
        current = self.base_transform.copy()
        current['move_x'] = x
        current['move_y'] = y
        return self.apply_transformations(**current)
    
    def get_structure_info(self) -> Optional[Dict]:
        """Get information about the current structure."""
        if self.current_structure_num is None:
            return None
        return get_structure_info(self.current_structure_num)
    
    def create_overlay_image(self, background_color=(255, 255, 255), 
                           structure_color=(0, 255, 255)) -> Optional[np.ndarray]:
        """
        Create a colored overlay image of the current transformed structure.
        
        Args:
            background_color: RGB color for background
            structure_color: RGB color for structures
            
        Returns:
            Colored overlay image or None if failed
        """
        try:
            # Get current transformed image
            transformed_image = self.apply_transformations(**self.base_transform)
            
            if transformed_image is None:
                return None
            
            # Create colored overlay
            h, w = transformed_image.shape[:2]
            overlay = np.full((h, w, 3), background_color, dtype=np.uint8)
            
            # Set structure pixels to structure color
            structure_mask = transformed_image < 128  # Black pixels are structures
            overlay[structure_mask] = structure_color
            
            return overlay
            
        except Exception as e:
            self.transformation_error.emit(f"Error creating overlay: {e}")
            return None
