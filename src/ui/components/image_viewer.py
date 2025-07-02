import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PySide6.QtCore import Qt, Signal, QPoint, QRect
from PySide6.QtGui import QPainter, QPixmap, QImage, QPen, QBrush, QColor, QWheelEvent, QMouseEvent, QPaintEvent, QResizeEvent
import cv2


class ImageViewer(QWidget):
    view_changed = Signal(dict)
    
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
    
    def set_gds_overlay(self, gds_overlay):
        self._gds_overlay = gds_overlay
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
    
    def set_overlay_alpha(self, alpha):
        self._overlay_alpha = max(0.0, min(1.0, alpha))
        self._invalidate_cache()
        self.update()
    
    def toggle_overlay(self):
        self._overlay_visible = not self._overlay_visible
        self.update()
    
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
        
        if overlay_data.dtype != np.uint8:
            if overlay_data.max() <= 1.0:
                overlay_8bit = (overlay_data * 255).astype(np.uint8)
            else:
                overlay_8bit = overlay_data.astype(np.uint8)
        else:
            overlay_8bit = overlay_data
        
        if len(overlay_8bit.shape) == 2:
            # Grayscale overlay
            h, w = overlay_8bit.shape
            overlay_qimage = QImage(overlay_8bit.data, w, h, w, QImage.Format_Grayscale8)
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
        
        painter.drawPixmap(0, 0, base_pixmap)
        
        painter.setOpacity(self._overlay_alpha)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.drawPixmap(0, 0, overlay_pixmap)
        
        painter.end()
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
        
        if self._alignment_result and 'difference_map' in self._alignment_result:
            self._draw_difference_overlay(painter)
    
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
            self._dragging = True
            self._drag_start = event.position().toPoint()
            self._drag_offset = self._pan_offset
            self.setCursor(Qt.ClosedHandCursor)
        
        event.accept()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            current_pos = event.position().toPoint()
            delta = current_pos - self._drag_start
            self._pan_offset = self._drag_offset + delta
            
            self._update_image_rect()
            self.update()
        
        event.accept()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
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
        scale_x = image_size.x() / self._image_rect.width()
        scale_y = image_size.y() / self._image_rect.height()
        
        image_x = int(relative_pos.x() * scale_x)
        image_y = int(relative_pos.y() * scale_y)
        
        return QPoint(image_x, image_y)
    
    def _image_to_screen_coords(self, image_point):
        if self._image_rect.isEmpty():
            return QPoint(0, 0)
        
        image_size = self._get_image_size()
        scale_x = self._image_rect.width() / image_size.x()
        scale_y = self._image_rect.height() / image_size.y()
        
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


def create_image_viewer(parent=None):
    return ImageViewer(parent)