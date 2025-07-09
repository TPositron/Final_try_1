"""
Canvas Zoom Service - Image Canvas Zoom and Scale Operations

This service handles zoom functionality for image canvases in alignment views,
providing smooth zoom operations with level management, limits, and fit-to-canvas
capabilities.

Main Class:
- CanvasZoomService: Qt-based service for canvas zoom operations

Key Methods:
- set_zoom_level(): Sets zoom level directly with bounds checking
- zoom_in(): Zooms in by one step increment
- zoom_out(): Zooms out by one step increment
- zoom_to_fit(): Zooms to fit image in canvas
- zoom_actual_size(): Sets zoom to 100% (actual size)
- get_zoom_level(): Returns current zoom level
- get_zoom_percentage(): Returns zoom level as percentage
- set_zoom_limits(): Sets minimum and maximum zoom limits
- get_zoom_limits(): Returns current zoom limits

Signals Emitted:
- zoom_changed(float): Zoom level changed
- zoom_reset(): Zoom reset to fit

Dependencies:
- Uses: PySide6.QtCore (QObject, Signal for Qt integration)
- Uses: typing (type hints for method signatures)
- Used by: UI canvas components and image viewers
- Used by: Alignment interfaces requiring zoom functionality

Features:
- Configurable zoom limits (default 0.1x to 10x)
- Step-based zoom in/out operations
- Fit-to-canvas automatic zoom calculation
- Actual size (100%) zoom preset
- Percentage-based zoom reporting
- Signal-based zoom change notifications
- Bounds checking and limit enforcement
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
