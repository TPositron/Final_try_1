"""
Manual Alignment Service
Handles manual alignment operations and real-time UI updates.
"""

from typing import Dict, Optional, Any
from PySide6.QtCore import QObject, Signal
import numpy as np
import logging

from src.core.models import AlignedGdsModel, SemImage


logger = logging.getLogger(__name__)


class ManualAlignmentService(QObject):
    """Service for handling manual alignment transformations."""
    
    # Signals
    transform_updated = Signal(dict)  # Emitted when transform is updated
    manual_alignment_updated = Signal()  # Emitted when manual alignment is updated
    bitmap_rendered = Signal(np.ndarray)  # Emitted when new bitmap is rendered
    state_changed = Signal(dict)  # Emitted when alignment state changes
    
    def __init__(self):
        super().__init__()
        self._aligned_gds_model: Optional[AlignedGdsModel] = None
        self._current_sem_image: Optional[SemImage] = None
        self._current_transform = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0,
            'transparency': 0.5
        }
        self._canvas_size = (1024, 666)  # Default canvas size
        
    def initialize(self, aligned_gds_model: AlignedGdsModel, sem_image: Optional[SemImage] = None):
        """Initialize the service with required models."""
        self._aligned_gds_model = aligned_gds_model
        self._current_sem_image = sem_image
        
        # Reset transforms on the model
        self._aligned_gds_model.reset_transforms()
        
        # Initialize current transform to match model
        self._current_transform = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0,
            'transparency': 0.5
        }
        
        # Emit initial state
        self._emit_state_change("Initialized")
        
    def set_sem_image(self, sem_image: SemImage):
        """Set the current SEM image for overlay rendering."""
        self._current_sem_image = sem_image
        self._emit_state_change("SEM image updated")
        
        # Re-render with current transform
        self._render_and_emit()
        
    def update_transform_parameter(self, parameter_name: str, value: float):
        """Update a single transform parameter."""
        if parameter_name not in self._current_transform:
            raise ValueError(f"Unknown transform parameter: {parameter_name}")
            
        # Update local state
        self._current_transform[parameter_name] = value
        
        # Update the model using appropriate method
        if parameter_name != 'transparency' and self._aligned_gds_model:
            self._apply_transform_to_model(parameter_name, value)
        
        # Emit transform update
        self.transform_updated.emit(self._current_transform.copy())
        
        # Render and emit new bitmap
        self._render_and_emit()
        
    def update_transforms(self, transforms: Dict[str, float]):
        """Update multiple transform parameters at once."""
        # Validate all parameters first
        for param_name in transforms:
            if param_name not in self._current_transform:
                raise ValueError(f"Unknown transform parameter: {param_name}")
        
        # Update local state
        self._current_transform.update(transforms)
        
        # Update the model (batch update for efficiency)
        if self._aligned_gds_model:
            for param_name, value in transforms.items():
                if param_name != 'transparency':  # Skip UI-only parameter
                    self._apply_transform_to_model(param_name, value)
        
        # Emit transform update
        self.transform_updated.emit(self._current_transform.copy())
        
        # Render and emit new bitmap
        self._render_and_emit()
        
    def reset_transforms(self):
        """Reset all transforms to default values."""
        default_transforms = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0,
            'transparency': 0.5
        }
        
        # Reset model using proper API
        if self._aligned_gds_model:
            self._aligned_gds_model.set_translation_pixels(0.0, 0.0)
            self._aligned_gds_model.set_residual_rotation(0.0)
            self._aligned_gds_model.set_scale(1.0)
            
        # Update local state
        self._current_transform = default_transforms.copy()
        
        # Emit updates
        self.transform_updated.emit(self._current_transform.copy())
        self._emit_state_change("Transforms reset")
        
        # Render and emit new bitmap
        self._render_and_emit()
        
    def get_current_transform(self) -> Dict[str, float]:
        """Get the current transform parameters."""
        return self._current_transform.copy()
        
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current alignment state."""
        return {
            'transform': self._current_transform.copy(),
            'has_gds_model': self._aligned_gds_model is not None,
            'has_sem_image': self._current_sem_image is not None,
            'canvas_size': self._canvas_size
        }
        
    def set_canvas_size(self, width: int, height: int):
        """Set the canvas size for rendering."""
        self._canvas_size = (width, height)
        self._render_and_emit()
        
    def _apply_transform_to_model(self, parameter_name: str, value: float):
        """Apply a single transform parameter to the model using appropriate API."""
        if not self._aligned_gds_model:
            return
            
        # Map parameter names to model API calls
        if parameter_name == 'translate_x':
            current_y = self._current_transform.get('translate_y', 0.0)
            self._aligned_gds_model.set_translation_pixels(value, current_y)
        elif parameter_name == 'translate_y':
            current_x = self._current_transform.get('translate_x', 0.0)
            self._aligned_gds_model.set_translation_pixels(current_x, value)
        elif parameter_name == 'rotation':
            self._aligned_gds_model.set_residual_rotation(value)
        elif parameter_name == 'scale':
            self._aligned_gds_model.set_scale(value)
        else:
            logger.warning(f"Unknown transform parameter: {parameter_name}")

    def _render_and_emit(self):
        """Render the current alignment and emit the bitmap."""
        if not self._aligned_gds_model:
            return
            
        try:
            # Generate bitmap from the aligned GDS model using frame-based approach
            bitmap = self._aligned_gds_model.to_bitmap(
                resolution=self._canvas_size
            )
            
            # Apply transparency if needed (this could be done in post-processing)
            # The bitmap will be overlaid on the SEM image with the specified transparency
            
            self.bitmap_rendered.emit(bitmap)
            
        except Exception as e:
            self._emit_state_change(f"Render error: {str(e)}")
            
    def _emit_state_change(self, message: str):
        """Emit a state change signal with current information."""
        state = self.get_current_state()
        state['message'] = message
        self.state_changed.emit(state)
