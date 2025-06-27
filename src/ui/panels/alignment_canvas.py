"""
Alignment Canvas Panel
QGraphicsView overlay rendering for image alignment visualization.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, 
                               QGraphicsPixmapItem, QGraphicsRectItem)
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QImage, QTransform
import numpy as np
from typing import Optional, Dict, Any


class AlignmentCanvas(QGraphicsView):
    """Canvas for displaying and aligning SEM and GDS images."""
    
    # Signals
    mouse_clicked = Signal(float, float)  # Emitted when canvas is clicked
    zoom_requested = Signal(float)  # Emitted when zoom is requested
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_canvas()
        self.sem_item = None
        self.gds_item = None
        self.current_transform = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0,
            'transparency': 0.5
        }
        
    def setup_canvas(self):
        """Initialize the graphics scene and view."""
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # Configure view
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        
        # Set scene rectangle
        self.scene.setSceneRect(0, 0, 1024, 666)
        
    def load_sem_image(self, image_array: np.ndarray):
        """Load SEM image into the canvas."""
        # Remove existing SEM item
        if self.sem_item:
            self.scene.removeItem(self.sem_item)
            
        # Convert numpy array to QPixmap
        if len(image_array.shape) == 2:
            # Grayscale image
            height, width = image_array.shape
            bytes_per_line = width
            
            # Normalize to 0-255 if needed
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
                
            qimage = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            
            # Add to scene
            self.sem_item = self.scene.addPixmap(pixmap)
            self.sem_item.setZValue(0)  # SEM image in background
            
    def load_gds_image(self, image_array: np.ndarray):
        """Load GDS overlay image into the canvas."""
        # Remove existing GDS item
        if self.gds_item:
            self.scene.removeItem(self.gds_item)
            
        # Convert numpy array to QPixmap with transparency
        if len(image_array.shape) == 2:
            height, width = image_array.shape
            
            # Create RGBA image for transparency
            rgba_array = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Set RGB channels (binary image: 0 = black structure, 255 = white background)
            # Make white areas transparent, keep black areas visible
            mask = image_array == 0  # Black pixels (structure)
            rgba_array[mask] = [255, 0, 0, 128]  # Red overlay with transparency
            rgba_array[~mask] = [0, 0, 0, 0]  # Transparent for white areas
            
            bytes_per_line = width * 4
            qimage = QImage(rgba_array.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimage)
            
            # Add to scene
            self.gds_item = self.scene.addPixmap(pixmap)
            self.gds_item.setZValue(1)  # GDS overlay on top
            
    def update_gds_transform(self, transform: Dict[str, float]):
        """Update GDS overlay transform."""
        if not self.gds_item:
            return
            
        self.current_transform.update(transform)
        
        # Reset transform
        self.gds_item.setTransform(QTransform())
        
        # Apply transformations in order
        t = QTransform()
        
        # Translate to center for rotation/scaling
        center_x, center_y = 512, 333  # Center of 1024x666 image
        t.translate(center_x, center_y)
        
        # Apply scale
        scale = transform.get('scale', 1.0)
        t.scale(scale, scale)
        
        # Apply rotation
        rotation = transform.get('rotation', 0.0)
        t.rotate(rotation)
        
        # Translate back and apply translation offset
        t.translate(-center_x, -center_y)
        t.translate(transform.get('translate_x', 0.0), transform.get('translate_y', 0.0))
        
        self.gds_item.setTransform(t)
        
        # Apply transparency
        transparency = transform.get('transparency', 0.5)
        self.gds_item.setOpacity(1.0 - transparency)
        
    def fit_in_view(self):
        """Fit the scene content in the view."""
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        
    def zoom_to_actual_size(self):
        """Reset zoom to 100%."""
        self.resetTransform()
        
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.mouse_clicked.emit(scene_pos.x(), scene_pos.y())
        super().mousePressEvent(event)
        
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        factor = 1.2
        if event.angleDelta().y() < 0:
            factor = 1.0 / factor
            
        self.scale(factor, factor)
        self.zoom_requested.emit(factor)


class AlignmentCanvasPanel(QWidget):
    """Panel containing the alignment canvas."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.canvas = AlignmentCanvas()
        layout.addWidget(self.canvas)
        
    def get_canvas(self) -> AlignmentCanvas:
        """Get the alignment canvas widget."""
        return self.canvas
