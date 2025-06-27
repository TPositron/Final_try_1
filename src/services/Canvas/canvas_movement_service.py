"""
Canvas Movement Service
Handles panning and movement functionality for the image canvas in the alignment view.
"""

from typing import Tuple, Optional
from PySide6.QtCore import QObject, Signal, QPointF


class CanvasMovementService(QObject):
    """Service for handling canvas panning and movement operations."""
    
    # Signals
    position_changed = Signal(float, float)  # Emitted when canvas position changes
    position_reset = Signal()  # Emitted when position is reset to center
    
    def __init__(self):
        super().__init__()
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._is_panning = False
        self._last_pan_point = QPointF()
        self._canvas_bounds = (1024, 666)  # Canvas boundaries
        
    def start_pan(self, start_point: QPointF) -> None:
        """Start a panning operation."""
        self._is_panning = True
        self._last_pan_point = start_point
    
    def continue_pan(self, current_point: QPointF) -> None:
        """Continue panning operation with current mouse position."""
        if not self._is_panning:
            return
            
        delta_x = current_point.x() - self._last_pan_point.x()
        delta_y = current_point.y() - self._last_pan_point.y()
        
        self.pan_by(delta_x, delta_y)
        self._last_pan_point = current_point
    
    def end_pan(self) -> None:
        """End the current panning operation."""
        self._is_panning = False
    
    def pan_by(self, delta_x: float, delta_y: float) -> None:
        """Pan the canvas by the specified delta values."""
        new_x = self._pan_x + delta_x
        new_y = self._pan_y + delta_y
        
        # Apply bounds checking if needed
        # For now, allow unlimited panning
        self._pan_x = new_x
        self._pan_y = new_y
        
        self.position_changed.emit(self._pan_x, self._pan_y)
    
    def pan_to(self, x: float, y: float) -> None:
        """Pan to a specific position."""
        self._pan_x = x
        self._pan_y = y
        self.position_changed.emit(self._pan_x, self._pan_y)
    
    def reset_position(self) -> None:
        """Reset canvas position to center."""
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.position_changed.emit(self._pan_x, self._pan_y)
        self.position_reset.emit()
    
    def get_position(self) -> Tuple[float, float]:
        """Get current canvas position."""
        return self._pan_x, self._pan_y
    
    def set_canvas_bounds(self, width: int, height: int) -> None:
        """Set the canvas boundaries for bounds checking."""
        self._canvas_bounds = (width, height)
    
    def get_canvas_bounds(self) -> Tuple[int, int]:
        """Get current canvas boundaries."""
        return self._canvas_bounds
    
    def is_panning(self) -> bool:
        """Check if currently in panning mode."""
        return self._is_panning
    
    def center_on_point(self, point: QPointF) -> None:
        """Center the canvas on a specific point."""
        canvas_width, canvas_height = self._canvas_bounds
        center_x = canvas_width / 2
        center_y = canvas_height / 2
        
        # Calculate pan needed to center the point
        pan_x = center_x - point.x()
        pan_y = center_y - point.y()
        
        self.pan_to(pan_x, pan_y)
