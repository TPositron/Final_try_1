"""
Manual Alignment Service - User-Controlled Alignment Operations

This service handles manual alignment operations with real-time UI updates and
interactive transformation controls. It provides immediate visual feedback for
user-driven alignment adjustments and maintains synchronization between UI
controls and the underlying GDS model.

Main Class:
- ManualAlignmentService: Qt-based service for manual alignment operations

Key Methods:
- initialize(): Sets up service with GDS model and SEM image
- set_sem_image(): Updates SEM image for overlay rendering
- update_transform_parameter(): Updates single transformation parameter
- update_transforms(): Batch updates multiple transformation parameters
- reset_transforms(): Resets all transformations to default values
- get_current_transform(): Returns current transformation parameters
- get_current_state(): Returns complete service state information
- set_canvas_size(): Updates canvas dimensions for rendering
- _apply_transform_to_model(): Applies transformation to underlying model
- _render_and_emit(): Renders current state and emits bitmap
- _emit_state_change(): Emits state change notifications

Signals Emitted:
- transform_updated(dict): Transformation parameters updated
- manual_alignment_updated(): Manual alignment state changed
- bitmap_rendered(np.ndarray): New rendered bitmap available
- state_changed(dict): Service state changed with context

Dependencies:
- Uses: PySide6.QtCore (QObject, Signal for Qt integration)
- Uses: numpy (array operations), logging (error reporting)
- Uses: core/models (AlignedGdsModel, SemImage data models)
- Used by: ui/alignment_controller.py (manual alignment UI)
- Used by: ui/alignment_controls.py (transformation controls)

Transformation Parameters:
- translate_x: Horizontal translation in pixels
- translate_y: Vertical translation in pixels
- rotation: Rotation angle in degrees
- scale: Uniform scale factor
- transparency: UI overlay transparency (0.0 to 1.0)

Real-Time Features:
- Immediate visual feedback for parameter changes
- Live bitmap rendering with transformation updates
- Synchronous UI control and model state updates
- Interactive transformation preview
- Responsive parameter adjustment

Model Integration:
- Direct API calls to AlignedGdsModel for transformations
- Frame-based rendering approach for non-destructive operations
- Coordinate system conversion between UI pixels and GDS units
- Proper API mapping for different transformation types
- Error handling for model operations

State Management:
- Current transformation parameter tracking
- Canvas size adaptation for different display contexts
- SEM image integration for overlay rendering
- Service initialization and reset capabilities
- State synchronization between UI and model

Canvas and Rendering:
- Configurable canvas size (default 1024x666)
- Frame-based bitmap generation
- Transparency support for overlay composition
- Error handling for rendering operations
- Signal emission for UI updates

Error Handling:
- Parameter validation with descriptive error messages
- Graceful handling of missing models or images
- Render error recovery with state notifications
- Logging integration for debugging and monitoring
- Safe fallbacks for API call failures

User Experience:
- Immediate visual feedback for all parameter changes
- Smooth real-time transformation preview
- Intuitive parameter mapping to visual effects
- Consistent state management across UI components
- Responsive interaction with minimal latency

Workflow:
1. Initialize service with GDS model and optional SEM image
2. User adjusts transformation parameters via UI controls
3. Service updates model and renders new bitmap
4. UI receives bitmap and displays updated alignment
5. Process repeats for interactive alignment refinement
6. Final transformation parameters available for export

Advantages:
- Real-time: Immediate visual feedback
- Interactive: Direct user control over all parameters
- Non-destructive: Frame-based transformations preserve original data
- Responsive: Optimized for smooth UI interaction
- Flexible: Supports arbitrary transformation combinations
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
        
        # Reset model using unified service
        if self._aligned_gds_model:
            from .unified_transformation_service import UnifiedTransformationService
            unified_service = UnifiedTransformationService()
            unified_service.apply_transformations(self._aligned_gds_model, 0.0, 0.0, 100.0, 0.0)
            
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
        """Apply a single transform parameter to the model using unified method."""
        if not self._aligned_gds_model:
            return
            
        # Apply transformations using unified method
        self._apply_unified_transform(
            translate_x=self._current_transform.get('translate_x', 0.0),
            translate_y=self._current_transform.get('translate_y', 0.0),
            rotation=self._current_transform.get('rotation', 0.0),
            scale=self._current_transform.get('scale', 1.0)
        )
    
    def _apply_unified_transform(self, translate_x: float, translate_y: float, rotation: float, scale: float):
        """Unified transformation method for manual, hybrid, and automatic alignment."""
        from .unified_transformation_service import UnifiedTransformationService
        
        # Use unified service for all transformations
        unified_service = UnifiedTransformationService()
        unified_service.apply_transformations(self._aligned_gds_model, translate_x, translate_y, scale * 100.0, rotation)
    
    def apply_transform_parameters(self, translate_x: float, translate_y: float, rotation: float, scale: float):
        """Public method for hybrid/automatic alignment to apply calculated transformations."""
        # Update local state
        self._current_transform.update({
            'translate_x': translate_x,
            'translate_y': translate_y,
            'rotation': rotation,
            'scale': scale
        })
        
        # Apply to model
        self._apply_unified_transform(translate_x, translate_y, rotation, scale)
        
        # Emit updates
        self.transform_updated.emit(self._current_transform.copy())
        self._render_and_emit()

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
