"""
Auto Alignment Service
Handles automatic alignment operations and algorithm integration.
"""

from typing import Dict, Optional, Any, Callable
from PySide6.QtCore import QObject, Signal, QThread
import numpy as np

from src.core.models import AlignedGdsModel, SemImage
from .transformations.auto_alignment_service import AutoAlignmentWorker


class AutoAlignmentService(QObject):
    """Service for handling automatic alignment transformations."""
    
    # Signals
    alignment_started = Signal()  # Emitted when auto alignment begins
    alignment_progress = Signal(int, str)  # progress percentage, status message
    alignment_completed = Signal(dict)  # Emitted when alignment completes with results
    auto_alignment_completed = Signal()  # Emitted when auto alignment process is completed
    alignment_failed = Signal(str)  # Emitted when alignment fails with error message
    transform_updated = Signal(dict)  # Emitted when transforms are applied
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
        self._canvas_size = (1024, 666)
        self._worker_thread: Optional[QThread] = None
        self._alignment_worker: Optional[AutoAlignmentWorker] = None
        self._available_methods = ["orb", "brute_force"]
        self._current_method = "orb"
        
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
        self._emit_state_change("Initialized for auto alignment")
        
    def set_sem_image(self, sem_image: SemImage):
        """Set the current SEM image for alignment."""
        self._current_sem_image = sem_image
        self._emit_state_change("SEM image updated for auto alignment")
        
    def set_alignment_method(self, method: str):
        """Set the automatic alignment method."""
        if method not in self._available_methods:
            raise ValueError(f"Unknown alignment method: {method}. Available: {self._available_methods}")
        self._current_method = method
        self._emit_state_change(f"Alignment method set to {method}")
        
    def get_available_methods(self) -> list:
        """Get the list of available alignment methods."""
        return self._available_methods.copy()
        
    def start_auto_alignment(self, method: Optional[str] = None):
        """Start automatic alignment process."""
        if not self._aligned_gds_model or not self._current_sem_image:
            error_msg = "Cannot start auto alignment: Missing GDS model or SEM image"
            self.alignment_failed.emit(error_msg)
            return
            
        # Use provided method or current method
        if method:
            self.set_alignment_method(method)
            
        # Stop any running alignment
        self.stop_auto_alignment()
        
        try:
            # Get image arrays for alignment
            sem_array = self._current_sem_image.to_array()
            gds_bitmap = self._aligned_gds_model.generate_bitmap(
                canvas_width=self._canvas_size[0],
                canvas_height=self._canvas_size[1]
            )
            
            # Create worker thread
            self._alignment_worker = AutoAlignmentWorker(
                sem_image=sem_array,
                gds_image=gds_bitmap,
                method=self._current_method
            )
            
            self._worker_thread = QThread()
            self._alignment_worker.moveToThread(self._worker_thread)
            
            # Connect worker signals
            self._alignment_worker.progress_updated.connect(self._on_alignment_progress)
            self._alignment_worker.alignment_completed.connect(self._on_alignment_completed)
            self._alignment_worker.alignment_failed.connect(self._on_alignment_failed)
            
            # Connect thread signals
            self._worker_thread.started.connect(self._alignment_worker.run)
            self._worker_thread.finished.connect(self._cleanup_worker)
            
            # Start the alignment
            self.alignment_started.emit()
            self._emit_state_change(f"Auto alignment started using {self._current_method}")
            self._worker_thread.start()
            
        except Exception as e:
            error_msg = f"Failed to start auto alignment: {str(e)}"
            self.alignment_failed.emit(error_msg)
            self._emit_state_change(error_msg)
            
    def stop_auto_alignment(self):
        """Stop the current auto alignment process."""
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait(5000)  # Wait up to 5 seconds
            
        self._cleanup_worker()
        self._emit_state_change("Auto alignment stopped")
        
    def apply_alignment_result(self, alignment_result: Dict[str, Any]):
        """Apply the results from automatic alignment."""
        if 'transformation_matrix' not in alignment_result:
            error_msg = "Invalid alignment result: missing transformation matrix"
            self.alignment_failed.emit(error_msg)
            return
            
        try:
            # Extract transform parameters from the transformation matrix
            # This depends on the format returned by the alignment algorithm
            transforms = self._extract_transforms_from_matrix(
                alignment_result['transformation_matrix']
            )
            
            # Apply transforms to the model
            self._apply_transforms_to_model(transforms)
            
            # Update current transform state
            self._current_transform.update(transforms)
            
            # Emit updates
            self.transform_updated.emit(self._current_transform.copy())
            self._emit_state_change("Auto alignment result applied")
            
            # Render and emit new bitmap
            self._render_and_emit()
            
        except Exception as e:
            error_msg = f"Failed to apply alignment result: {str(e)}"
            self.alignment_failed.emit(error_msg)
            self._emit_state_change(error_msg)
            
    def get_current_transform(self) -> Dict[str, float]:
        """Get the current transform parameters."""
        return self._current_transform.copy()
        
    def reset_transforms(self):
        """Reset all transforms to default values."""
        default_transforms = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0,
            'transparency': 0.5
        }
        
        # Reset model
        if self._aligned_gds_model:
            self._aligned_gds_model.reset_transforms()
            
        # Update local state
        self._current_transform = default_transforms.copy()
        
        # Emit updates
        self.transform_updated.emit(self._current_transform.copy())
        self._emit_state_change("Transforms reset")
        
        # Render and emit new bitmap
        self._render_and_emit()
        
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current alignment state."""
        return {
            'transform': self._current_transform.copy(),
            'has_gds_model': self._aligned_gds_model is not None,
            'has_sem_image': self._current_sem_image is not None,
            'canvas_size': self._canvas_size,
            'alignment_method': self._current_method,
            'available_methods': self._available_methods,
            'is_running': self._worker_thread is not None and self._worker_thread.isRunning()
        }
        
    def set_canvas_size(self, width: int, height: int):
        """Set the canvas size for rendering."""
        self._canvas_size = (width, height)
        self._render_and_emit()
        
    def _on_alignment_progress(self, percentage: int):
        """Handle progress updates from the alignment worker."""
        self.alignment_progress.emit(percentage, f"Auto alignment progress: {percentage}%")
        
    def _on_alignment_completed(self, result: Dict[str, Any]):
        """Handle completed alignment from the worker."""
        self.alignment_completed.emit(result)
        self._emit_state_change("Auto alignment completed successfully")
        
        # Automatically apply the result
        self.apply_alignment_result(result)
        
    def _on_alignment_failed(self, error_message: str):
        """Handle failed alignment from the worker."""
        self.alignment_failed.emit(error_message)
        self._emit_state_change(f"Auto alignment failed: {error_message}")
        
    def _cleanup_worker(self):
        """Clean up worker thread and resources."""
        if self._worker_thread:
            self._worker_thread.deleteLater()
            self._worker_thread = None
            
        if self._alignment_worker:
            self._alignment_worker.deleteLater()
            self._alignment_worker = None
            
    def _extract_transforms_from_matrix(self, transformation_matrix: np.ndarray) -> Dict[str, float]:
        """Extract individual transform parameters from a transformation matrix."""
        # This is a simplified extraction - should be enhanced based on the actual
        # format of the transformation matrix returned by the alignment algorithms
        
        transforms = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0
        }
        
        try:
            if transformation_matrix.shape == (2, 3):  # Affine transformation matrix
                # Extract translation
                transforms['translate_x'] = float(transformation_matrix[0, 2])
                transforms['translate_y'] = float(transformation_matrix[1, 2])
                
                # Extract scale (assuming uniform scaling)
                scale_x = np.sqrt(transformation_matrix[0, 0]**2 + transformation_matrix[0, 1]**2)
                transforms['scale'] = float(scale_x)
                
                # Extract rotation (in degrees)
                rotation_rad = np.arctan2(transformation_matrix[0, 1], transformation_matrix[0, 0])
                transforms['rotation'] = float(np.degrees(rotation_rad))
                
            elif transformation_matrix.shape == (3, 3):  # Homogeneous transformation matrix
                # Extract translation
                transforms['translate_x'] = float(transformation_matrix[0, 2])
                transforms['translate_y'] = float(transformation_matrix[1, 2])
                
                # Extract scale
                scale_x = np.sqrt(transformation_matrix[0, 0]**2 + transformation_matrix[1, 0]**2)
                transforms['scale'] = float(scale_x)
                
                # Extract rotation
                rotation_rad = np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0])
                transforms['rotation'] = float(np.degrees(rotation_rad))
                
        except Exception as e:
            # If extraction fails, log the error but don't crash
            print(f"Warning: Could not extract transforms from matrix: {e}")
            
        return transforms
        
    def _apply_transforms_to_model(self, transforms: Dict[str, float]):
        """Apply transform parameters to the aligned GDS model."""
        if not self._aligned_gds_model:
            return
            
        for param_name, value in transforms.items():
            if param_name != 'transparency':  # Skip UI-only parameter
                self._aligned_gds_model.apply_transform(param_name, value)
                
    def _render_and_emit(self):
        """Render the current alignment and emit the bitmap."""
        if not self._aligned_gds_model:
            return
            
        try:
            # Generate bitmap from the aligned GDS model
            bitmap = self._aligned_gds_model.generate_bitmap(
                canvas_width=self._canvas_size[0],
                canvas_height=self._canvas_size[1]
            )
            
            self.bitmap_rendered.emit(bitmap)
            
        except Exception as e:
            self._emit_state_change(f"Render error: {str(e)}")
            
    def _emit_state_change(self, message: str):
        """Emit a state change signal with current information."""
        state = self.get_current_state()
        state['message'] = message
        self.state_changed.emit(state)
