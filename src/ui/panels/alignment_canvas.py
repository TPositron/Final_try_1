"""
Alignment Canvas Panel - Graphics View for Image Alignment Visualization

This module provides a graphics view canvas for displaying and aligning SEM and GDS
images with enhanced visualization and transformation capabilities.

Main Classes:
- AlignmentCanvas: QGraphicsView for image display and alignment
- AlignmentCanvasPanel: Panel wrapper for alignment canvas

Key Methods:
- display_sem_image(): Displays SEM image with scaling
- display_gds_overlay(): Displays GDS overlay with transparency
- scale_for_display(): Scales images to standard dimensions
- update_gds_transform(): Updates GDS overlay transformation
- fit_in_view(): Fits content in view with aspect ratio
- zoom_in()/zoom_out(): Zoom controls
- clear_images(): Clears all displayed images

Signals Emitted:
- mouse_clicked(float, float): Canvas clicked coordinates
- zoom_requested(float): Zoom level change requested
- transform_changed(dict): Transformation parameters changed

Dependencies:
- Uses: PySide6.QtWidgets, PySide6.QtCore, PySide6.QtGui (Qt framework)
- Uses: numpy (array processing)
- Uses: typing (type hints)
- Called by: UI alignment components
- Coordinates with: Image viewers and alignment workflow

Features:
- Standard SEM dimensions (1024x666) optimization
- Real-time GDS overlay transformation (translate, rotate, scale)
- Transparency control for overlay visualization
- Mouse interaction for point selection and navigation
- Zoom and pan capabilities with proper aspect ratio
- Colored overlay rendering for structure visualization
- Transform validation and error handling
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, 
                               QGraphicsPixmapItem, QGraphicsRectItem)
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QImage, QTransform
import numpy as np
from typing import Optional, Dict, Any, Tuple


class AlignmentCanvas(QGraphicsView):
    """
    Canvas for displaying and aligning SEM and GDS images.
    Optimized for standard 1024x666 dimensions with enhanced display and scaling.
    """
    
    # Standard SEM dimensions after cropping
    STANDARD_WIDTH = 1024
    STANDARD_HEIGHT = 666
    
    # Signals
    mouse_clicked = Signal(float, float)  # Emitted when canvas is clicked
    zoom_requested = Signal(float)  # Emitted when zoom is requested
    transform_changed = Signal(dict)  # Emitted when transform changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_canvas()
        
        # Image items
        self.sem_item = None
        self.gds_item = None
        
        # Current transform parameters
        self.current_transform = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0,
            'transparency': 0.5
        }
        
        # Display state
        self.current_zoom = 1.0
        self.auto_fit_enabled = True
        self.image_loaded = False
        
    def setup_canvas(self):
        """Initialize the graphics scene and view with standard dimensions."""
        # FIXED: Use a different attribute name to avoid conflict with QGraphicsView.scene() method
        self._graphics_scene = QGraphicsScene()
        self.setScene(self._graphics_scene)
        
        # Configure view for optimal image display
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        
        # Set scene rectangle to standard SEM dimensions
        self._graphics_scene.setSceneRect(0, 0, self.STANDARD_WIDTH, self.STANDARD_HEIGHT)
        
        # Set background
        self.setBackgroundBrush(QBrush(QColor(50, 50, 50)))  # Dark gray background
        
    def display_sem_image(self, image_array: np.ndarray) -> bool:
        """
        Display SEM image with automatic scaling to standard dimensions.
        
        Args:
            image_array: SEM image array
            
        Returns:
            True if displayed successfully, False otherwise
        """
        try:
            # Remove existing SEM item
            if self.sem_item:
                self._graphics_scene.removeItem(self.sem_item)
                self.sem_item = None
            
            # Scale image to standard dimensions if needed
            scaled_image = self.scale_for_display(image_array, (self.STANDARD_WIDTH, self.STANDARD_HEIGHT))
            
            # Convert to QPixmap
            pixmap = self._array_to_pixmap(scaled_image, is_grayscale=True)
            if pixmap.isNull():
                return False
            
            # Add to scene
            self.sem_item = self._graphics_scene.addPixmap(pixmap)
            self.sem_item.setZValue(0)  # SEM image in background
            
            self.image_loaded = True
            
            # Auto-fit view if enabled
            if self.auto_fit_enabled:
                self.fit_in_view()
            
            print(f"✓ SEM image displayed: {scaled_image.shape}")
            return True
            
        except Exception as e:
            print(f"Error displaying SEM image: {e}")
            return False
    
    def display_gds_overlay(self, image_array: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0)) -> bool:
        """
        Display GDS overlay with transparency and color.
        
        Args:
            image_array: GDS binary image array
            color: RGB color for overlay (default red)
            
        Returns:
            True if displayed successfully, False otherwise
        """
        try:
            # Remove existing GDS item
            if self.gds_item:
                self._graphics_scene.removeItem(self.gds_item)
                self.gds_item = None
            
            # Scale image to standard dimensions if needed
            scaled_image = self.scale_for_display(image_array, (self.STANDARD_WIDTH, self.STANDARD_HEIGHT))
            
            # Create colored overlay
            overlay_pixmap = self._create_colored_overlay(scaled_image, color)
            if overlay_pixmap.isNull():
                return False
            
            # Add to scene
            self.gds_item = self._graphics_scene.addPixmap(overlay_pixmap)
            self.gds_item.setZValue(1)  # GDS overlay on top
            
            # Apply current transform
            self._apply_current_transform()
            
            print(f"✓ GDS overlay displayed: {scaled_image.shape}")
            return True
            
        except Exception as e:
            print(f"Error displaying GDS overlay: {e}")
            return False
    
    def scale_for_display(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Scale image for display while maintaining aspect ratio.
        
        Args:
            image: Input image array
            target_size: (width, height) target dimensions
            
        Returns:
            Scaled image array
        """
        if len(image.shape) != 2:
            raise ValueError("Only 2D images are supported")
        
        current_height, current_width = image.shape
        target_width, target_height = target_size
        
        # If already correct size, return as-is
        if current_width == target_width and current_height == target_height:
            return image
        
        # Calculate scaling to fit within target while maintaining aspect ratio
        scale_x = target_width / current_width
        scale_y = target_height / current_height
        scale = min(scale_x, scale_y)
        
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)
        
        # Resize image using OpenCV-style resizing (if available)
        try:
            import cv2
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        except ImportError:
            # Fallback to simple array indexing
            step_x = current_width / new_width
            step_y = current_height / new_height
            
            y_indices = (np.arange(new_height) * step_y).astype(int)
            x_indices = (np.arange(new_width) * step_x).astype(int)
            
            resized = image[y_indices[:, None], x_indices]
        
        # If scaled image is smaller than target, pad with background
        if new_width < target_width or new_height < target_height:
            # Create padded image
            padded = np.full((target_height, target_width), 
                           image.max() if len(np.unique(image)) == 2 else image.mean(), 
                           dtype=image.dtype)
            
            # Center the resized image
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            
            padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            return padded
        
        return resized
    
    def _array_to_pixmap(self, image_array: np.ndarray, is_grayscale: bool = True) -> QPixmap:
        """Convert numpy array to QPixmap."""
        if len(image_array.shape) != 2:
            return QPixmap()  # Return null pixmap for invalid input
        
        height, width = image_array.shape
        
        # Normalize to 0-255 if needed
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
        
        if is_grayscale:
            bytes_per_line = width
            qimage = QImage(image_array.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        else:
            # Assume RGB format
            bytes_per_line = width * 3
            qimage = QImage(image_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        return QPixmap.fromImage(qimage)
    
    def _create_colored_overlay(self, binary_image: np.ndarray, color: Tuple[int, int, int]) -> QPixmap:
        """Create colored overlay from binary image."""
        height, width = binary_image.shape
        r, g, b = color
        
        # Create RGBA image for transparency
        rgba_array = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Binary image: 0 = black structure, 255 = white background
        # Make white areas transparent, keep black areas visible with color
        structure_mask = binary_image == 0  # Black pixels (structure)
        rgba_array[structure_mask] = [r, g, b, 128]  # Colored overlay with transparency
        rgba_array[~structure_mask] = [0, 0, 0, 0]  # Transparent for white areas
        
        bytes_per_line = width * 4
        qimage = QImage(rgba_array.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(qimage)
            
    def update_gds_transform(self, transform: Dict[str, float]):
        """Update GDS overlay transform with enhanced parameter handling."""
        if not self.gds_item:
            return
            
        # Update current transform
        self.current_transform.update(transform)
        self._apply_current_transform()
        
        # Emit transform change signal
        self.transform_changed.emit(self.current_transform.copy())
    
    def _apply_current_transform(self):
        """Apply current transform to GDS overlay."""
        if not self.gds_item:
            return
        
        # Reset transform
        self.gds_item.setTransform(QTransform())
        
        # Apply transformations in correct order
        t = QTransform()
        
        # Calculate center point for rotation/scaling
        center_x = self.STANDARD_WIDTH / 2
        center_y = self.STANDARD_HEIGHT / 2
        
        # Translate to center for rotation/scaling
        t.translate(center_x, center_y)
        
        # Apply scale
        scale = self.current_transform.get('scale', 1.0)
        t.scale(scale, scale)
        
        # Apply rotation
        rotation = self.current_transform.get('rotation', 0.0)
        t.rotate(rotation)
        
        # Translate back and apply translation offset
        t.translate(-center_x, -center_y)
        t.translate(
            self.current_transform.get('translate_x', 0.0), 
            self.current_transform.get('translate_y', 0.0)
        )
        
        self.gds_item.setTransform(t)
        
        # Apply transparency
        transparency = self.current_transform.get('transparency', 0.5)
        self.gds_item.setOpacity(1.0 - transparency)
    
    def get_current_transform(self) -> Dict[str, float]:
        """Get current transform parameters."""
        return self.current_transform.copy()
    
    def reset_transform(self):
        """Reset GDS overlay transform to default."""
        self.current_transform = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0,
            'transparency': 0.5
        }
        self._apply_current_transform()
        self.transform_changed.emit(self.current_transform.copy())
    
    def set_auto_fit(self, enabled: bool):
        """Enable or disable automatic fit-in-view."""
        self.auto_fit_enabled = enabled
        if enabled and self.image_loaded:
            self.fit_in_view()
    
    def fit_in_view(self):
        """Fit the scene content in the view with proper aspect ratio."""
        # FIXED: Use the inherited scene() method to get the scene
        if self.scene().items():
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.current_zoom = 1.0
        
    def zoom_to_actual_size(self):
        """Reset zoom to 100% (actual pixel size)."""
        self.resetTransform()
        self.current_zoom = 1.0
        
    def zoom_to_fit(self):
        """Zoom to fit entire scene in view."""
        self.fit_in_view()
    
    def zoom_in(self, factor: float = 1.2):
        """Zoom in by specified factor."""
        self.scale(factor, factor)
        self.current_zoom *= factor
        self.zoom_requested.emit(factor)
    
    def zoom_out(self, factor: float = 1.2):
        """Zoom out by specified factor."""
        zoom_factor = 1.0 / factor
        self.scale(zoom_factor, zoom_factor)
        self.current_zoom *= zoom_factor
        self.zoom_requested.emit(zoom_factor)
    
    def get_current_zoom(self) -> float:
        """Get current zoom level."""
        return self.current_zoom
    
    def clear_images(self):
        """Clear all images from canvas."""
        if self.sem_item:
            self._graphics_scene.removeItem(self.sem_item)
            self.sem_item = None
        
        if self.gds_item:
            self._graphics_scene.removeItem(self.gds_item)
            self.gds_item = None
        
        self.image_loaded = False
        self.reset_transform()
    
    def has_sem_image(self) -> bool:
        """Check if SEM image is loaded."""
        return self.sem_item is not None
    
    def has_gds_overlay(self) -> bool:
        """Check if GDS overlay is loaded."""
        return self.gds_item is not None
    
    def get_scene_dimensions(self) -> Tuple[int, int]:
        """Get scene dimensions."""
        return (self.STANDARD_WIDTH, self.STANDARD_HEIGHT)
        
    # Legacy compatibility methods
    def load_sem_image(self, image_array: np.ndarray):
        """Legacy method for loading SEM image."""
        self.display_sem_image(image_array)
    
    def load_gds_image(self, image_array: np.ndarray):
        """Legacy method for loading GDS image."""
        self.display_gds_overlay(image_array)
        
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.mouse_clicked.emit(scene_pos.x(), scene_pos.y())
        super().mousePressEvent(event)
        
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Zoom with Ctrl+Wheel
            factor = 1.2
            if event.angleDelta().y() < 0:
                factor = 1.0 / factor
                
            self.scale(factor, factor)
            self.current_zoom *= factor
            self.zoom_requested.emit(factor)
        else:
            # Normal wheel event (scroll)
            super().wheelEvent(event)


class AlignmentCanvasPanel(QWidget):
    """
    Panel containing the alignment canvas with enhanced controls.
    Provides interface for standard dimension alignment operations.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas: Optional[AlignmentCanvas] = None  # FIXED: Add type annotation for clarity
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Create alignment canvas
        self.canvas = AlignmentCanvas()
        layout.addWidget(self.canvas)
        
        # Connect canvas signals
        self.canvas.mouse_clicked.connect(self._on_canvas_clicked)
        self.canvas.zoom_requested.connect(self._on_zoom_changed)
        self.canvas.transform_changed.connect(self._on_transform_changed)
        
    def get_canvas(self) -> AlignmentCanvas:
        """Get the alignment canvas widget."""
        # FIXED: Add assertion to ensure canvas is not None
        assert self.canvas is not None, "Canvas not initialized"
        return self.canvas
    
    def display_sem_image(self, image_array: np.ndarray) -> bool:
        """Display SEM image in canvas."""
        if self.canvas:
            return self.canvas.display_sem_image(image_array)
        return False
    
    def display_gds_overlay(self, image_array: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0)) -> bool:
        """Display GDS overlay in canvas."""
        if self.canvas:
            return self.canvas.display_gds_overlay(image_array, color)
        return False
    
    def update_alignment(self, transform: Dict[str, float]):
        """Update alignment transform."""
        if self.canvas:
            self.canvas.update_gds_transform(transform)
    
    def reset_alignment(self):
        """Reset alignment to default."""
        if self.canvas:
            self.canvas.reset_transform()
    
    def fit_view(self):
        """Fit content in view."""
        if self.canvas:
            self.canvas.fit_in_view()
    
    def zoom_to_actual_size(self):
        """Zoom to actual size."""
        if self.canvas:
            self.canvas.zoom_to_actual_size()
    
    def clear_display(self):
        """Clear all images from display."""
        if self.canvas:
            self.canvas.clear_images()
    
    def get_transform(self) -> Dict[str, float]:
        """Get current alignment transform."""
        if self.canvas:
            return self.canvas.get_current_transform()
        return {}
    
    def is_ready_for_alignment(self) -> bool:
        """Check if ready for alignment (both images loaded)."""
        if self.canvas:
            return self.canvas.has_sem_image() and self.canvas.has_gds_overlay()
        return False
    
    def _on_canvas_clicked(self, x: float, y: float):
        """Handle canvas click events."""
        print(f"Canvas clicked at: ({x:.1f}, {y:.1f})")
    
    def _on_zoom_changed(self, factor: float):
        """Handle zoom change events."""
        if self.canvas:
            zoom_level = self.canvas.get_current_zoom()
            print(f"Zoom level: {zoom_level:.2f}x")
    
    def _on_transform_changed(self, transform: Dict[str, float]):
        """Handle transform change events."""
        # This can be connected to update UI controls or emit signals
        pass
