"""
Image Viewer - Main Image Display Component

This module provides the main image viewer component for displaying SEM images,
GDS overlays, and handling user interactions including zoom, pan, and point selection.

Main Class:
- ImageViewer: Main image display widget with overlay support

Key Methods:
- set_sem_image(): Sets SEM image for display
- load_image(): Loads image from file path
- set_gds_overlay(): Sets GDS overlay image
- set_alignment_result(): Sets alignment result data
- set_preview_image(): Sets preview image for filtering
- toggle_overlay(): Toggles overlay visibility
- set_overlay_visible(): Sets overlay visibility state
- reset_view(): Resets view to default zoom and pan
- fit_to_window(): Fits image to window size
- export_current_view(): Exports current view as image

Signals Emitted:
- view_changed(dict): View parameters changed
- point_selected(int, int, str): Point selected with coordinates and type

Dependencies:
- Uses: numpy (image processing)
- Uses: PySide6.QtWidgets, PySide6.QtCore, PySide6.QtGui (Qt framework)
- Uses: cv2 (image operations)
- Called by: UI main window and image processing components
- Coordinates with: Alignment and filtering workflows

Features:
- SEM image display with automatic cropping to 1024x666
- GDS overlay support with transparency control
- Zoom and pan capabilities with mouse wheel and drag
- Point selection mode for hybrid alignment
- Visual markers for selected points
- Alignment result visualization
- Preview image support for filtering
- View state management and export capabilities
- Coordinate conversion between image and screen space
"""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PySide6.QtCore import Qt, Signal, QPoint, QRect
from PySide6.QtGui import QPainter, QPixmap, QImage, QPen, QBrush, QColor, QWheelEvent, QMouseEvent, QPaintEvent, QResizeEvent
import cv2


class ImageViewer(QWidget):
    view_changed = Signal(dict)
    point_selected = Signal(int, int, str)  # x, y, mode ("sem" or "gds")
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setMouseTracking(True)
        
        self._sem_image = None
        self._gds_overlay = None
        self._alignment_result = None
        self._preview_image = None
        
        self._pixmap_cache = {}
        self._current_pixmap = None
        
        self._zoom_factor = 1.0
        self._pan_offset = QPoint(0, 0)
        self._image_rect = QRect()
        
        self._dragging = False
        self._drag_start = QPoint()
        self._drag_offset = QPoint()
        
        self._overlay_visible = True
        self._overlay_alpha = 0.7
        
        # Point selection state
        self._point_selection_mode = False
        self._point_selection_type = "sem"  # "sem" or "gds"
        self._sem_points = []
        self._gds_points = []
        
        self.setFocusPolicy(Qt.StrongFocus)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        button_layout = QHBoxLayout()
        
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view)
        button_layout.addWidget(self.reset_view_btn)
        
        self.fit_to_window_btn = QPushButton("Fit to Window")
        self.fit_to_window_btn.clicked.connect(self.fit_to_window)
        button_layout.addWidget(self.fit_to_window_btn)
        
        self.toggle_overlay_btn = QPushButton("Toggle Overlay")
        self.toggle_overlay_btn.clicked.connect(self.toggle_overlay)
        button_layout.addWidget(self.toggle_overlay_btn)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        layout.addStretch()
    
    def set_sem_image(self, sem_image):
        # Defensive: ensure only 2D 1024x666 image is set
        if sem_image is not None:
            if len(sem_image.shape) == 2 and sem_image.shape == (666, 1024):
                self._sem_image = sem_image
            else:
                # If not, crop from bottom center
                h, w = sem_image.shape[:2]
                crop_h, crop_w = 666, 1024
                start_y = max(0, h - crop_h)
                start_x = max(0, (w - crop_w) // 2)
                self._sem_image = sem_image[start_y:start_y+crop_h, start_x:start_x+crop_w]
        else:
            self._sem_image = None
        self._preview_image = None
        self._invalidate_cache()
        self._update_image_rect()
        self.fit_to_window()  # Always fit cropped image to window
        self.update()
    
    def load_image(self, file_path):
        """
        Load an image from a file path and display it.
        
        Args:
            file_path (str): Path to the image file to load
        """
        try:
            # Load the image using OpenCV
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"Error: Could not load image from {file_path}")
                return
            
            # Convert to numpy array and set as SEM image
            self.set_sem_image(image)
            print(f"Successfully loaded image: {file_path}")
            
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")

    def set_gds_overlay(self, gds_overlay):
        print(f"ImageViewer.set_gds_overlay called with: {gds_overlay.shape if gds_overlay is not None else None}")
        self._gds_overlay = gds_overlay
        if gds_overlay is not None:
            print(f"GDS overlay set: shape={gds_overlay.shape}, dtype={gds_overlay.dtype}, non-zero={np.count_nonzero(gds_overlay)}")
        self._invalidate_cache()
        self.update()
    
    def set_alignment_result(self, alignment_result):
        self._alignment_result = alignment_result
        self._invalidate_cache()
        self.update()
    
    def set_preview_image(self, preview_array):
        self._preview_image = preview_array
        self._invalidate_cache()
        self.update()
    
    def toggle_overlay(self):
        self._overlay_visible = not self._overlay_visible
        self.update()
    
    def set_overlay_visible(self, visible):
        """Set overlay visibility."""
        print(f"Setting overlay visibility to: {visible}")
        self._overlay_visible = visible
        self._invalidate_cache()
        self.update()
    
    def set_overlay_alpha(self, alpha):
        """Set overlay transparency."""
        self._overlay_alpha = max(0.0, min(1.0, alpha))
        self.update()
    
    def get_overlay_visible(self):
        """Get overlay visibility state."""
        return self._overlay_visible
    
    def get_overlay_alpha(self):
        """Get overlay transparency."""
        return self._overlay_alpha
    
    def reset_view(self):
        self._zoom_factor = 1.0
        self._pan_offset = QPoint(0, 0)
        self._update_image_rect()
        self.update()
        self._emit_view_changed()
    
    def set_zoom_factor(self, zoom_factor):
        """Set the zoom factor for the canvas."""
        new_zoom = max(0.1, min(10.0, zoom_factor))
        if new_zoom != self._zoom_factor:
            self._zoom_factor = new_zoom
            self._update_image_rect()
            self.update()
            self._emit_view_changed()
    
    def fit_to_window(self):
        if self._sem_image is None:
            return
        
        widget_size = self.size()
        image_size = self._get_image_size()
        
        if image_size[0] == 0 or image_size[1] == 0:
            return
        
        scale_x = widget_size.width() / image_size[0]
        scale_y = widget_size.height() / image_size[1]
        
        self._zoom_factor = min(scale_x, scale_y) * 0.9
        self._pan_offset = QPoint(0, 0)
        self._update_image_rect()
        self.update()
        self._emit_view_changed()
    
    def _get_image_size(self):
        if self._sem_image is not None:
            # Always return (width, height) for consistency
            shape = self._sem_image.shape
            if len(shape) == 2:
                h, w = shape
            elif len(shape) == 3:
                h, w = shape[0], shape[1]
            else:
                return 0, 0
            return w, h
        return 0, 0
    
    def _update_image_rect(self):
        if self._sem_image is None:
            self._image_rect = QRect()
            return
        
        image_size = self._get_image_size()
        scaled_width = int(image_size[0] * self._zoom_factor)
        scaled_height = int(image_size[1] * self._zoom_factor)
        
        widget_center = self.rect().center()
        image_center = QPoint(scaled_width // 2, scaled_height // 2)
        
        top_left = widget_center - image_center + self._pan_offset
        self._image_rect = QRect(top_left.x(), top_left.y(), scaled_width, scaled_height)
    
    def _invalidate_cache(self):
        self._pixmap_cache.clear()
        self._current_pixmap = None
    
    def _get_current_image_array(self):
        if self._preview_image is not None:
            return self._preview_image
        elif self._sem_image is not None:
            if hasattr(self._sem_image, 'to_array'):
                return self._sem_image.to_array()
            else:
                return self._sem_image
        return None
    
    def _create_base_pixmap(self):
        image_array = self._get_current_image_array()
        if image_array is None:
            return QPixmap(1024, 666)
        
        if image_array.dtype != np.uint8:
            if image_array.max() <= 1.0:
                image_8bit = (image_array * 255).astype(np.uint8)
            else:
                image_8bit = image_array.astype(np.uint8)
        else:
            image_8bit = image_array
        
        if len(image_8bit.shape) == 2:
            h, w = image_8bit.shape
            qimage = QImage(image_8bit.data, w, h, w, QImage.Format_Grayscale8)
        else:
            return QPixmap(1024, 666)
        
        return QPixmap.fromImage(qimage)
    
    def _create_overlay_pixmap(self, base_pixmap):
        if not self._overlay_visible:
            return base_pixmap
        
        overlay_data = None
        if self._alignment_result is not None and 'overlay_preview' in self._alignment_result:
            overlay_data = self._alignment_result['overlay_preview']
        elif self._gds_overlay is not None:
            overlay_data = self._gds_overlay
        
        if overlay_data is None:
            return base_pixmap
        
        print(f"Creating overlay pixmap: shape={overlay_data.shape}, dtype={overlay_data.dtype}")
        
        if overlay_data.dtype != np.uint8:
            if overlay_data.max() <= 1.0:
                overlay_8bit = (overlay_data * 255).astype(np.uint8)
            else:
                overlay_8bit = overlay_data.astype(np.uint8)
        else:
            overlay_8bit = overlay_data
        
        print(f"Overlay 8bit: shape={overlay_8bit.shape}, non-zero={np.count_nonzero(overlay_8bit)}")
        
        if len(overlay_8bit.shape) == 2:
            # Grayscale overlay - convert to colored overlay with transparency
            h, w = overlay_8bit.shape
            
            # Create RGBA overlay (red/cyan colored)
            rgba_overlay = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Where overlay has content (non-zero), make it colored and semi-transparent
            mask = overlay_8bit > 0
            rgba_overlay[mask, 0] = 255  # Red channel
            rgba_overlay[mask, 1] = 255  # Green channel (makes it yellow/cyan)
            rgba_overlay[mask, 2] = 0    # Blue channel
            rgba_overlay[mask, 3] = int(self._overlay_alpha * 255)  # Alpha channel
            
            bytes_per_line = w * 4
            overlay_qimage = QImage(rgba_overlay.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
            overlay_pixmap = QPixmap.fromImage(overlay_qimage)
            
        elif len(overlay_8bit.shape) == 3 and overlay_8bit.shape[2] == 3:
            # RGB overlay
            h, w, c = overlay_8bit.shape
            bytes_per_line = w * c
            overlay_qimage = QImage(overlay_8bit.data, w, h, bytes_per_line, QImage.Format_RGB888)
            overlay_pixmap = QPixmap.fromImage(overlay_qimage)
        else:
            print(f"Unsupported overlay shape: {overlay_8bit.shape}")
            return base_pixmap
        
        result_pixmap = QPixmap(base_pixmap.size())
        result_pixmap.fill(Qt.transparent)
        
        painter = QPainter(result_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw base image first
        painter.drawPixmap(0, 0, base_pixmap)
        
        # Draw overlay with composition mode
        if len(overlay_8bit.shape) == 2:
            # For grayscale overlays, we already handled alpha in RGBA conversion
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.drawPixmap(0, 0, overlay_pixmap)
        else:
            # For RGB overlays, apply alpha
            painter.setOpacity(self._overlay_alpha)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.drawPixmap(0, 0, overlay_pixmap)
        
        painter.end()
        print(f"Overlay pixmap created successfully")
        return result_pixmap
    
    def _get_composite_pixmap(self):
        cache_key = (
            id(self._sem_image),
            id(self._preview_image),
            id(self._gds_overlay),
            id(self._alignment_result),
            self._overlay_visible,
            self._overlay_alpha
        )
        
        if cache_key in self._pixmap_cache:
            return self._pixmap_cache[cache_key]
        
        base_pixmap = self._create_base_pixmap()
        composite_pixmap = self._create_overlay_pixmap(base_pixmap)
        
        self._pixmap_cache[cache_key] = composite_pixmap
        return composite_pixmap
    
    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        
        if self._sem_image is None:
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
            return
        
        pixmap = self._get_composite_pixmap()
        if pixmap.isNull():
            return
        
        self._update_image_rect()
        
        scaled_pixmap = pixmap.scaled(
            self._image_rect.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        painter.drawPixmap(self._image_rect.topLeft(), scaled_pixmap)
        
        # Draw point markers
        self._draw_point_markers(painter)
        
        if self._alignment_result and 'difference_map' in self._alignment_result:
            self._draw_difference_overlay(painter)
    
    def _draw_point_markers(self, painter):
        """Draw visual markers for selected points."""
        if not self._image_rect.isEmpty():
            # Draw SEM points in blue
            painter.setPen(QPen(QColor(100, 150, 255), 3))
            painter.setBrush(QBrush(QColor(100, 150, 255, 150)))
            for i, (x, y) in enumerate(self._sem_points):
                screen_point = self._image_to_screen_coords(QPoint(x, y))
                if self._image_rect.contains(screen_point):
                    # Draw circle
                    painter.drawEllipse(screen_point.x() - 8, screen_point.y() - 8, 16, 16)
                    # Draw number
                    painter.setPen(QPen(QColor(255, 255, 255), 2))
                    painter.drawText(screen_point.x() - 5, screen_point.y() + 5, str(i + 1))
                    painter.setPen(QPen(QColor(100, 150, 255), 3))
            
            # Draw GDS points in orange
            painter.setPen(QPen(QColor(255, 150, 50), 3))
            painter.setBrush(QBrush(QColor(255, 150, 50, 150)))
            for i, (x, y) in enumerate(self._gds_points):
                screen_point = self._image_to_screen_coords(QPoint(x, y))
                if self._image_rect.contains(screen_point):
                    # Draw square
                    painter.drawRect(screen_point.x() - 8, screen_point.y() - 8, 16, 16)
                    # Draw number
                    painter.setPen(QPen(QColor(255, 255, 255), 2))
                    painter.drawText(screen_point.x() - 5, screen_point.y() + 5, str(i + 1))
                    painter.setPen(QPen(QColor(255, 150, 50), 3))

    def _draw_difference_overlay(self, painter):
        pass
    
    def wheelEvent(self, event: QWheelEvent):
        if self._sem_image is None:
            return
        
        delta = event.angleDelta().y()
        zoom_in = delta > 0
        
        zoom_factor = 1.2 if zoom_in else 1 / 1.2
        old_zoom = self._zoom_factor
        new_zoom = old_zoom * zoom_factor
        
        new_zoom = max(0.1, min(10.0, new_zoom))
        
        if new_zoom != old_zoom:
            mouse_pos = event.position().toPoint()
            
            old_image_pos = self._screen_to_image_coords(mouse_pos)
            
            self._zoom_factor = new_zoom
            self._update_image_rect()
            
            new_image_pos = self._screen_to_image_coords(mouse_pos)
            
            offset_delta = new_image_pos - old_image_pos
            self._pan_offset -= offset_delta
            
            self._update_image_rect()
            self.update()
            self._emit_view_changed()
        
        event.accept()
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # Left click for panning only
            self._dragging = True
            self._drag_start = event.position().toPoint()
            self._drag_offset = self._pan_offset
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.RightButton:
            if self._point_selection_mode:
                # Handle point selection with right click
                image_coords = self._screen_to_image_coords(event.position().toPoint())
                
                # Check if clicking near an existing point to remove it
                if self.remove_point_near(event.position().toPoint().x(), event.position().toPoint().y(), 
                                        self._point_selection_type):
                    # Point was removed, emit signal to update UI
                    self.point_selected.emit(-1, -1, self._point_selection_type)  # Signal for point removal
                else:
                    # Add new point if within image bounds and under limit
                    if self._image_rect.contains(event.position().toPoint()):
                        current_points = len(self._sem_points) if self._point_selection_type == "sem" else len(self._gds_points)
                        if current_points < 3:
                            self.add_point(image_coords.x(), image_coords.y(), self._point_selection_type)
                            self.point_selected.emit(image_coords.x(), image_coords.y(), self._point_selection_type)
        
        event.accept()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging and not self._point_selection_mode:
            current_pos = event.position().toPoint()
            delta = current_pos - self._drag_start
            self._pan_offset = self._drag_offset + delta
            
            self._update_image_rect()
            self.update()
        
        event.accept()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            if self._point_selection_mode:
                self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            self._emit_view_changed()
        
        event.accept()
    
    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self._update_image_rect()
        self.update()
    
    def _screen_to_image_coords(self, screen_point):
        if self._image_rect.isEmpty():
            return QPoint(0, 0)
        
        relative_pos = screen_point - self._image_rect.topLeft()
        
        image_size = self._get_image_size()
        scale_x = image_size[0] / self._image_rect.width()
        scale_y = image_size[1] / self._image_rect.height()
        
        image_x = int(relative_pos.x() * scale_x)
        image_y = int(relative_pos.y() * scale_y)
        
        return QPoint(image_x, image_y)
    
    def _image_to_screen_coords(self, image_point):
        if self._image_rect.isEmpty():
            return QPoint(0, 0)
        
        image_size = self._get_image_size()
        scale_x = self._image_rect.width() / image_size[0]
        scale_y = self._image_rect.height() / image_size[1]
        
        screen_x = int(image_point.x() * scale_x) + self._image_rect.left()
        screen_y = int(image_point.y() * scale_y) + self._image_rect.top()
        
        return QPoint(screen_x, screen_y)
    
    def _emit_view_changed(self):
        view_info = {
            'zoom_factor': self._zoom_factor,
            'pan_offset': (self._pan_offset.x(), self._pan_offset.y()),
            'image_rect': (
                self._image_rect.x(),
                self._image_rect.y(),
                self._image_rect.width(),
                self._image_rect.height()
            )
        }
        self.view_changed.emit(view_info)
    
    def get_view_state(self):
        return {
            'zoom_factor': self._zoom_factor,
            'pan_offset': (self._pan_offset.x(), self._pan_offset.y()),
            'overlay_visible': self._overlay_visible,
            'overlay_alpha': self._overlay_alpha
        }
    
    def set_view_state(self, state):
        self._zoom_factor = state.get('zoom_factor', 1.0)
        pan_x, pan_y = state.get('pan_offset', (0, 0))
        self._pan_offset = QPoint(pan_x, pan_y)
        self._overlay_visible = state.get('overlay_visible', True)
        self._overlay_alpha = state.get('overlay_alpha', 0.7)
        
        self._update_image_rect()
        self._invalidate_cache()
        self.update()
    
    def export_current_view(self):
        if self._sem_image is None:
            return None
        
        pixmap = self._get_composite_pixmap()
        if pixmap.isNull():
            return None
        
        return pixmap.toImage()
    
    def refresh(self):
        self._invalidate_cache()
        self._update_image_rect()
        self.update()
    
    def moveGDS(self, dx, dy):
        """Move GDS overlay by pixel amounts (Step 4 requirement)."""
        try:
            if self._gds_overlay is not None:
                # Create translation matrix
                translation_matrix = np.array([
                    [1, 0, dx],
                    [0, 1, dy]
                ], dtype=np.float32)
                
                # Apply translation
                height, width = self._gds_overlay.shape[:2]
                transformed_overlay = cv2.warpAffine(self._gds_overlay, translation_matrix, (width, height))
                self.set_gds_overlay(transformed_overlay)
        except Exception as e:
            print(f"Error moving GDS overlay: {e}")
    
    def rotateGDS(self, angle):
        """Rotate GDS overlay by degrees (Step 4 requirement)."""
        try:
            if self._gds_overlay is not None:
                height, width = self._gds_overlay.shape[:2]
                center = (width // 2, height // 2)
                
                # Create rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Apply rotation
                transformed_overlay = cv2.warpAffine(self._gds_overlay, rotation_matrix, (width, height))
                self.set_gds_overlay(transformed_overlay)
        except Exception as e:
            print(f"Error rotating GDS overlay: {e}")
    
    def zoomGDS(self, factor):
        """Zoom GDS overlay by scale factor (Step 4 requirement)."""
        try:
            if self._gds_overlay is not None:
                height, width = self._gds_overlay.shape[:2]
                center = (width // 2, height // 2)
                
                # Create scale matrix
                scale_matrix = cv2.getRotationMatrix2D(center, 0, factor)
                
                # Apply scaling
                transformed_overlay = cv2.warpAffine(self._gds_overlay, scale_matrix, (width, height))
                self.set_gds_overlay(transformed_overlay)
        except Exception as e:
            print(f"Error zooming GDS overlay: {e}")
    
    def setGDSTransparency(self, alpha):
        """Adjust GDS opacity (Step 4 requirement)."""
        try:
            self.set_overlay_alpha(alpha)
        except Exception as e:
            print(f"Error setting GDS transparency: {e}")
    
    def set_point_selection_mode(self, enabled, point_type="sem"):
        """Enable/disable point selection mode."""
        self._point_selection_mode = enabled
        self._point_selection_type = point_type
        if enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
    
    def add_point(self, x, y, point_type):
        """Add a point to the appropriate list."""
        point = (int(x), int(y))
        if point_type == "sem":
            if len(self._sem_points) < 3:
                self._sem_points.append(point)
        elif point_type == "gds":
            if len(self._gds_points) < 3:
                self._gds_points.append(point)
        self.update()
    
    def remove_point_near(self, x, y, point_type, tolerance=10):
        """Remove a point near the given coordinates."""
        points_list = self._sem_points if point_type == "sem" else self._gds_points
        for i, (px, py) in enumerate(points_list):
            screen_point = self._image_to_screen_coords(QPoint(px, py))
            distance = ((screen_point.x() - x) ** 2 + (screen_point.y() - y) ** 2) ** 0.5
            if distance <= tolerance:
                points_list.pop(i)
                self.update()
                return True
        return False
    
    def clear_points(self, point_type):
        """Clear all points of the specified type."""
        if point_type == "sem":
            self._sem_points.clear()
        elif point_type == "gds":
            self._gds_points.clear()
        self.update()
    
    def get_points(self, point_type):
        """Get points of the specified type."""
        if point_type == "sem":
            return self._sem_points.copy()
        elif point_type == "gds":
            return self._gds_points.copy()
        return []
    
    def set_points(self, points, point_type):
        """Set points of the specified type."""
        if point_type == "sem":
            self._sem_points = points.copy()
        elif point_type == "gds":
            self._gds_points = points.copy()
        self.update()