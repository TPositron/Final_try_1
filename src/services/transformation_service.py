import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from PySide6.QtCore import QObject, Signal
from src.core.gds_aligned_generator import generate_aligned_gds, generate_transformed_gds
from src.core.gds_display_generator import get_structure_info

class TransformationService(QObject):
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
        # Additional state tracking for advanced features
        self._last_successful_transform = None
        self._transform_history = []
        self._max_history_size = 10
    
    def set_structure(self, structure_num: int) -> bool:
        """Set the current structure to work with."""
        try:
            info = get_structure_info(structure_num)
            if info is None:
                self.transformation_error.emit(f"Structure {structure_num} not found")
                return False
            
            # Reset transforms when changing structure
            old_structure = self.current_structure_num
            self.current_structure_num = structure_num
            
            # Clear history when switching structures
            if old_structure != structure_num:
                self._transform_history.clear()
                self.base_transform = {
                    'rotation': 0.0,
                    'zoom': 100.0,
                    'move_x': 0.0,
                    'move_y': 0.0
                }
            
            return True
        except Exception as e:
            self.transformation_error.emit(f"Error setting structure: {e}")
            return False
    
    def apply_transformations(self, rotation: float = 0.0, zoom: float = 100.0, 
                            move_x: float = 0.0, move_y: float = 0.0, 
                            target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        """Apply transformations to the current structure."""
        try:
            if self.current_structure_num is None:
                self.transformation_error.emit("No structure set for transformation")
                return None
            
            # Validate transformation parameters
            if not self._validate_transformation_params(rotation, zoom, move_x, move_y):
                return None
            
            # Store current transform for history
            previous_transform = self.base_transform.copy()
            
            # Apply the transformations using the improved generation function
            transformed_image = generate_transformed_gds(
                self.current_structure_num,
                rotation=rotation,
                zoom=zoom,
                move_x=move_x,
                move_y=move_y,
                target_size=target_size
            )
            
            if transformed_image is not None:
                # Update base transform state
                self.base_transform = {
                    'rotation': rotation,
                    'zoom': zoom,
                    'move_x': move_x,
                    'move_y': move_y
                }
                
                # Add to transform history
                self._add_to_history(previous_transform, self.base_transform.copy())
                
                # Store last successful transform
                self._last_successful_transform = self.base_transform.copy()
                
                # Emit success signal
                self.transformation_applied.emit(transformed_image)
            else:
                self.transformation_error.emit("Failed to generate transformed image")
            
            return transformed_image
            
        except Exception as e:
            error_msg = f"Error applying transformations: {e}"
            self.transformation_error.emit(error_msg)
            return None
    
    def apply_transform_dict(self, transform_params: Dict[str, float], 
                           target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        """Apply transformations using a parameter dictionary."""
        try:
            if self.current_structure_num is None:
                self.transformation_error.emit("No structure set for transformation")
                return None
            
            # Extract parameters with validation
            rotation = float(transform_params.get('rotation', 0.0))
            zoom = float(transform_params.get('zoom', 100.0))
            move_x = float(transform_params.get('move_x', 0.0))
            move_y = float(transform_params.get('move_y', 0.0))
            
            # Validate extracted parameters
            if not self._validate_transformation_params(rotation, zoom, move_x, move_y):
                return None
            
            # Store current transform for history
            previous_transform = self.base_transform.copy()
            
            # Use the aligned generation function for dictionary parameters
            transformed_image, bounds = generate_aligned_gds(
                self.current_structure_num,
                transform_params,
                target_size
            )
            
            if transformed_image is not None:
                # Update base transform with validated parameters
                self.base_transform.update({
                    'rotation': rotation,
                    'zoom': zoom,
                    'move_x': move_x,
                    'move_y': move_y
                })
                
                # Add to history
                self._add_to_history(previous_transform, self.base_transform.copy())
                
                # Store successful transform
                self._last_successful_transform = self.base_transform.copy()
                
                # Emit success signal
                self.transformation_applied.emit(transformed_image)
            else:
                self.transformation_error.emit("Failed to generate transformed image from dictionary")
            
            return transformed_image
            
        except Exception as e:
            error_msg = f"Error applying transform dict: {e}"
            self.transformation_error.emit(error_msg)
            return None
    
    def get_current_transform(self) -> Dict[str, float]:
        """Get a copy of the current transformation parameters."""
        return self.base_transform.copy()
    
    def reset_transforms(self, target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        """Reset all transformations to default values."""
        try:
            if self.current_structure_num is None:
                self.transformation_error.emit("No structure set for transformation")
                return None
            
            # Store current transform for history
            previous_transform = self.base_transform.copy()
            
            # Apply default transformations
            result = self.apply_transformations(
                rotation=0.0, 
                zoom=100.0, 
                move_x=0.0, 
                move_y=0.0,
                target_size=target_size
            )
            
            if result is not None:
                # Add reset action to history
                reset_transform = {'rotation': 0.0, 'zoom': 100.0, 'move_x': 0.0, 'move_y': 0.0}
                self._add_to_history(previous_transform, reset_transform, action_type="reset")
            
            return result
            
        except Exception as e:
            error_msg = f"Error resetting transforms: {e}"
            self.transformation_error.emit(error_msg)
            return None
    
    def rotate_structure(self, angle: float, target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        """Rotate structure to specified angle."""
        try:
            if self.current_structure_num is None:
                self.transformation_error.emit("No structure set for transformation")
                return None
            
            # Validate rotation angle
            if not isinstance(angle, (int, float)) or not (-360 <= angle <= 360):
                self.transformation_error.emit(f"Invalid rotation angle: {angle}. Must be between -360 and 360 degrees.")
                return None
            
            # Get current transform and update rotation
            current = self.base_transform.copy()
            current['rotation'] = float(angle)
            
            return self.apply_transformations(
                rotation=current['rotation'],
                zoom=current['zoom'],
                move_x=current['move_x'],
                move_y=current['move_y'],
                target_size=target_size
            )
            
        except Exception as e:
            error_msg = f"Error rotating structure: {e}"
            self.transformation_error.emit(error_msg)
            return None
    
    def zoom_structure(self, zoom_percentage: float, target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        """Zoom structure to specified percentage."""
        try:
            if self.current_structure_num is None:
                self.transformation_error.emit("No structure set for transformation")
                return None
            
            # Validate zoom percentage
            if not isinstance(zoom_percentage, (int, float)) or zoom_percentage <= 0:
                self.transformation_error.emit(f"Invalid zoom percentage: {zoom_percentage}. Must be positive.")
                return None
            
            if zoom_percentage < 1.0 or zoom_percentage > 1000.0:
                self.transformation_error.emit(f"Zoom percentage {zoom_percentage}% is outside reasonable range (1-1000%).")
                return None
            
            # Get current transform and update zoom
            current = self.base_transform.copy()
            current['zoom'] = float(zoom_percentage)
            
            return self.apply_transformations(
                rotation=current['rotation'],
                zoom=current['zoom'],
                move_x=current['move_x'],
                move_y=current['move_y'],
                target_size=target_size
            )
            
        except Exception as e:
            error_msg = f"Error zooming structure: {e}"
            self.transformation_error.emit(error_msg)
            return None
    
    def move_structure(self, dx: float, dy: float, target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        """Move structure by relative offset."""
        try:
            if self.current_structure_num is None:
                self.transformation_error.emit("No structure set for transformation")
                return None
            
            # Validate movement parameters
            if not all(isinstance(val, (int, float)) for val in [dx, dy]):
                self.transformation_error.emit("Movement values must be numeric")
                return None
            
            # Check for reasonable movement bounds
            max_movement = 2000.0  # pixels
            if abs(dx) > max_movement or abs(dy) > max_movement:
                self.transformation_error.emit(f"Movement too large. Max allowed: ±{max_movement} pixels")
                return None
            
            # Get current transform and update movement
            current = self.base_transform.copy()
            current['move_x'] += float(dx)
            current['move_y'] += float(dy)
            
            return self.apply_transformations(
                rotation=current['rotation'],
                zoom=current['zoom'],
                move_x=current['move_x'],
                move_y=current['move_y'],
                target_size=target_size
            )
            
        except Exception as e:
            error_msg = f"Error moving structure: {e}"
            self.transformation_error.emit(error_msg)
            return None
    
    def set_absolute_position(self, x: float, y: float, target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        """Set absolute position of structure."""
        try:
            if self.current_structure_num is None:
                self.transformation_error.emit("No structure set for transformation")
                return None
            
            # Validate position parameters
            if not all(isinstance(val, (int, float)) for val in [x, y]):
                self.transformation_error.emit("Position values must be numeric")
                return None
            
            # Check for reasonable position bounds
            max_position = 5000.0  # pixels
            if abs(x) > max_position or abs(y) > max_position:
                self.transformation_error.emit(f"Position too extreme. Max allowed: ±{max_position} pixels")
                return None
            
            # Get current transform and update absolute position
            current = self.base_transform.copy()
            current['move_x'] = float(x)
            current['move_y'] = float(y)
            
            return self.apply_transformations(
                rotation=current['rotation'],
                zoom=current['zoom'],
                move_x=current['move_x'],
                move_y=current['move_y'],
                target_size=target_size
            )
            
        except Exception as e:
            error_msg = f"Error setting absolute position: {e}"
            self.transformation_error.emit(error_msg)
            return None
    
    def get_structure_info(self) -> Optional[Dict]:
        """Get information about current structure."""
        if self.current_structure_num is None:
            return None
        
        try:
            return get_structure_info(self.current_structure_num)
        except Exception as e:
            self.transformation_error.emit(f"Error getting structure info: {e}")
            return None
    
    def create_overlay_image(self, background_color=(255, 255, 255), 
                           structure_color=(0, 255, 255), 
                           target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        """Create colored overlay image of the transformed structure."""
        try:
            if self.current_structure_num is None:
                self.transformation_error.emit("No structure set for overlay creation")
                return None
            
            # Validate color parameters
            if not self._validate_color(background_color) or not self._validate_color(structure_color):
                self.transformation_error.emit("Invalid color format. Use (R, G, B) tuples with values 0-255.")
                return None
            
            # Apply current transformations to get the transformed image
            transformed_image = self.apply_transformations(
                rotation=self.base_transform['rotation'],
                zoom=self.base_transform['zoom'],
                move_x=self.base_transform['move_x'],
                move_y=self.base_transform['move_y'],
                target_size=target_size
            )
            
            if transformed_image is None:
                self.transformation_error.emit("Failed to generate transformed image for overlay")
                return None
            
            h, w = transformed_image.shape[:2]
            overlay = np.full((h, w, 3), background_color, dtype=np.uint8)
            
            # Create mask for structure areas (dark areas in the binary image)
            structure_mask = transformed_image < 128 
            overlay[structure_mask] = structure_color
            
            return overlay
            
        except Exception as e:
            error_msg = f"Error creating overlay: {e}"
            self.transformation_error.emit(error_msg)
            return None
    
    # Additional utility methods for enhanced functionality
    
    def _validate_transformation_params(self, rotation: float, zoom: float, move_x: float, move_y: float) -> bool:
        """Validate transformation parameters."""
        try:
            # Check parameter types
            if not all(isinstance(val, (int, float)) for val in [rotation, zoom, move_x, move_y]):
                self.transformation_error.emit("All transformation parameters must be numeric")
                return False
            
            # Validate rotation range
            if not (-360 <= rotation <= 360):
                self.transformation_error.emit(f"Rotation {rotation}° outside valid range (-360° to 360°)")
                return False
            
            # Validate zoom range
            if zoom <= 0:
                self.transformation_error.emit(f"Zoom {zoom}% must be positive")
                return False
            
            if zoom < 1.0 or zoom > 1000.0:
                self.transformation_error.emit(f"Zoom {zoom}% outside reasonable range (1% to 1000%)")
                return False
            
            # Validate movement range
            max_movement = 2000.0
            if abs(move_x) > max_movement or abs(move_y) > max_movement:
                self.transformation_error.emit(f"Movement too large. Max: ±{max_movement} pixels")
                return False
            
            return True
            
        except Exception as e:
            self.transformation_error.emit(f"Parameter validation error: {e}")
            return False
    
    def _validate_color(self, color) -> bool:
        """Validate color tuple format."""
        try:
            if not isinstance(color, (tuple, list)) or len(color) != 3:
                return False
            
            return all(isinstance(c, int) and 0 <= c <= 255 for c in color)
            
        except:
            return False
    
    def _add_to_history(self, previous_transform: Dict, new_transform: Dict, action_type: str = "transform"):
        """Add transformation to history for potential undo functionality."""
        try:
            history_entry = {
                'previous': previous_transform,
                'new': new_transform,
                'action': action_type,
                'structure_num': self.current_structure_num,
                'timestamp': None  # Could add actual timestamp if needed
            }
            
            self._transform_history.append(history_entry)
            
            # Limit history size
            if len(self._transform_history) > self._max_history_size:
                self._transform_history.pop(0)
                
        except Exception as e:
            # Don't emit error for history issues - it's not critical
            print(f"Warning: Could not add to transform history: {e}")
    
    def get_transform_history(self) -> List[Dict]:
        """Get transformation history."""
        return self._transform_history.copy()
    
    def can_undo(self) -> bool:
        """Check if undo is possible."""
        return len(self._transform_history) > 0
    
    def undo_last_transform(self, target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        """Undo the last transformation."""
        try:
            if not self.can_undo():
                self.transformation_error.emit("No transformations to undo")
                return None
            
            # Get the last history entry
            last_entry = self._transform_history.pop()
            previous_transform = last_entry['previous']
            
            # Apply the previous transformation
            result = self.apply_transformations(
                rotation=previous_transform['rotation'],
                zoom=previous_transform['zoom'],
                move_x=previous_transform['move_x'],
                move_y=previous_transform['move_y'],
                target_size=target_size
            )
            
            # Remove the undo operation from history to prevent undo loops
            if self._transform_history:
                self._transform_history.pop()
            
            return result
            
        except Exception as e:
            error_msg = f"Error undoing transformation: {e}"
            self.transformation_error.emit(error_msg)
            return None
    
    def get_last_successful_transform(self) -> Optional[Dict[str, float]]:
        """Get the last successful transformation parameters."""
        return self._last_successful_transform.copy() if self._last_successful_transform else None
    
    def clear_history(self):
        """Clear transformation history."""
        self._transform_history.clear()
    
    def get_transform_summary(self) -> Dict[str, Any]:
        """Get a summary of current transformation state."""
        return {
            'current_structure': self.current_structure_num,
            'current_transform': self.base_transform.copy(),
            'last_successful': self._last_successful_transform.copy() if self._last_successful_transform else None,
            'history_count': len(self._transform_history),
            'can_undo': self.can_undo()
        }
