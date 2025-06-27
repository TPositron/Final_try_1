"""
Canvas Zoom Service
Handles zoom functionality for the image canvas in the alignment view.
"""

from typing import Optional, Tuple
from PySide6.QtCore import QObject, Signal


class CanvasZoomService(QObject):
    """Service for handling canvas zoom operations."""
    
    # Signals
    zoom_changed = Signal(float)  # Emitted when zoom level changes
    zoom_reset = Signal()  # Emitted when zoom is reset to fit
    
    def __init__(self):
        super().__init__()
        self._zoom_level = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0
        self._zoom_step = 0.1
        self._canvas_size = (1024, 666)  # Default canvas size
        
    def set_zoom_level(self, zoom: float) -> None:
        """Set the zoom level directly."""
        zoom = max(self._min_zoom, min(self._max_zoom, zoom))
        if zoom != self._zoom_level:
            self._zoom_level = zoom
            self.zoom_changed.emit(self._zoom_level)
    
    def zoom_in(self) -> None:
        """Zoom in by one step."""
        new_zoom = self._zoom_level + self._zoom_step
        self.set_zoom_level(new_zoom)
    
    def zoom_out(self) -> None:
        """Zoom out by one step."""
        new_zoom = self._zoom_level - self._zoom_step
        self.set_zoom_level(new_zoom)
    
    def zoom_to_fit(self, canvas_size: Optional[Tuple[int, int]] = None) -> None:
        """Zoom to fit the image in the canvas."""
        if canvas_size:
            self._canvas_size = canvas_size
        
        # Calculate zoom to fit
        canvas_width, canvas_height = self._canvas_size
        image_width, image_height = 1024, 666  # Standard image size
        
        zoom_x = canvas_width / image_width
        zoom_y = canvas_height / image_height
        zoom_to_fit = min(zoom_x, zoom_y, 1.0)  # Don't zoom in beyond 100%
        
        self.set_zoom_level(zoom_to_fit)
        self.zoom_reset.emit()
    
    def zoom_actual_size(self) -> None:
        """Set zoom to 100% (actual size)."""
        self.set_zoom_level(1.0)
    
    def get_zoom_level(self) -> float:
        """Get current zoom level."""
        return self._zoom_level
    
    def get_zoom_percentage(self) -> int:
        """Get current zoom level as percentage."""
        return int(self._zoom_level * 100)
    
    def set_zoom_limits(self, min_zoom: float, max_zoom: float) -> None:
        """Set zoom limits."""
        self._min_zoom = max(0.01, min_zoom)
        self._max_zoom = min(50.0, max_zoom)
        
        # Clamp current zoom to new limits
        self.set_zoom_level(self._zoom_level)
    
    def get_zoom_limits(self) -> Tuple[float, float]:
        """Get current zoom limits."""
        return self._min_zoom, self._max_zoom
