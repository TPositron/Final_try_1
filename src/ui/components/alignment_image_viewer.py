"""
Alignment Image Viewer - Enhanced Image Viewer with 3-Point Selection

This component provides an enhanced image viewer specifically designed for
hybrid alignment with 3-point selection capabilities.

Main Class:
- AlignmentImageViewer: Enhanced image viewer with point selection

Key Methods:
- enable_selection_mode(): Enables/disables point selection mode
- add_point(): Adds point at specified image coordinates
- remove_point(): Removes point by index
- move_point(): Moves point to new coordinates
- clear_points(): Clears all selected points
- get_selected_points(): Returns list of selected points
- set_image(): Sets main image to display
- set_overlay_image(): Sets overlay image

Signals Emitted:
- point_added(int, float, float): Point added at coordinates
- point_removed(int): Point removed by index
- point_moved(int, float, float): Point moved to new coordinates
- points_cleared(): All points cleared
- selection_mode_changed(bool): Selection mode enabled/disabled
- point_count_changed(int): Point count changed
- view_changed(dict): View parameters changed

Dependencies:
- Uses: numpy (image processing)
- Uses: PySide6.QtWidgets, PySide6.QtCore, PySide6.QtGui (Qt framework)
- Uses: cv2 (image operations)
- Called by: UI alignment components
- Coordinates with: Alignment workflow components

Features:
- Point selection mode for both GDS and SEM images
- Visual markers for selected points with color coding
- User adjustment of point positions via drag and drop
- Validation of exactly 3 points on both images
- Clear visual feedback about point correspondences
- Enable/disable alignment calculation based on completion
- Zoom and pan capabilities with mouse wheel and drag
- Overlay support with transparency control
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QCheckBox, QSpinBox)
from PySide6.QtCore import Qt, Signal, QPoint, QRect, QPointF
from PySide6.QtGui import (QPainter, QPixmap, QImage, QPen, QBrush, QColor, 
                          QWheelEvent, QMouseEvent, QPaintEvent, QFont)
import cv2
import logging

logger = logging.getLogger(__name__)


class AlignmentImageViewer(QWidget):
    """Enhanced image viewer with 3-point selection for hybrid alignment."""
    
    # Signals for Step 10
    point_added = Signal(int, float, float)         # point_index, x, y (image coordinates)
    point_removed = Signal(int)                     # point_index
    point_moved = Signal(int, float, float)         # point_index, new_x, new_y
    points_cleared = Signal()                       # all points cleared
    selection_mode_changed = Signal(bool)           # selection_mode_enabled
    point_count_changed = Signal(int)               # current_point_count
    view_changed = Signal(dict)                     # view parameters changed
    
    def __init__(self, viewer_type: str = "SEM", parent=None):
        """
        Initialize alignment image viewer.
        
        Args:
            viewer_type: "SEM" or "GDS" to identify the viewer type
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.viewer_type = viewer_type
        self.setMinimumSize(600, 400)
        self.setMouseTracking(True)
        
        # Image data
        self._image = None
        self._overlay_image = None
        self._current_pixmap = None
        
        # View parameters
        self._zoom_factor = 1.0
        self._pan_offset = QPoint(0, 0)
        self._image_rect = QRect()
        
        # Interaction state
        self._dragging = False
        self._drag_start = QPoint()
        self._selected_point_index = -1  # Index of point being dragged
        self._hover_point_index = -1     # Index of point being hovered
        
        # Step 10: 3-point selection system
        self._selection_mode = False
        self._selected_points = []  # List of (x, y) tuples in image coordinates
        self._max_points = 3
        self._point_radius = 8
        self._point_colors = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255)]  # RGB for points 1,2,3
        self._hover_tolerance = 15  # Pixels
        
        # Visual settings
        self._overlay_visible = True
        self._overlay_alpha = 0.7
        self._show_point_labels = True
        self._show_correspondence_lines = False
        
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._setup_ui()
        
        logger.info(f"AlignmentImageViewer initialized: type={viewer_type}")
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Control panel
        control_panel = self._create_control_panel()
        layout.addWidget(control_panel)
        
        # Main image area (takes most space)
        layout.addStretch()
        
        # Status panel
        status_panel = self._create_status_panel()
        layout.addWidget(status_panel)
    
    def _create_control_panel(self) -> QWidget:
        """Create control panel with buttons and options."""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        
        # View controls
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view)
        layout.addWidget(self.reset_view_btn)
        
        self.fit_to_window_btn = QPushButton("Fit to Window")
        self.fit_to_window_btn.clicked.connect(self.fit_to_window)
        layout.addWidget(self.fit_to_window_btn)
        
        # Selection mode toggle
        self.selection_mode_btn = QPushButton("Enable Point Selection")
        self.selection_mode_btn.setCheckable(True)
        self.selection_mode_btn.clicked.connect(self._toggle_selection_mode)
        layout.addWidget(self.selection_mode_btn)
        
        # Point management
        self.clear_points_btn = QPushButton("Clear Points")
        self.clear_points_btn.clicked.connect(self.clear_points)
        self.clear_points_btn.setEnabled(False)
        layout.addWidget(self.clear_points_btn)
        
        # Visual options
        self.show_labels_cb = QCheckBox("Show Labels")
        self.show_labels_cb.setChecked(True)
        self.show_labels_cb.toggled.connect(self._toggle_point_labels)
        layout.addWidget(self.show_labels_cb)
        
        layout.addStretch()
        
        return panel
    
    def _create_status_panel(self) -> QWidget:
        """Create status panel showing current state."""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        
        # Viewer type label
        type_label = QLabel(f"{self.viewer_type} Image:")
        type_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(type_label)
        
        # Point count
        self.point_count_label = QLabel("Points: 0/3")
        layout.addWidget(self.point_count_label)
        
        # Selection status
        self.selection_status_label = QLabel("Selection: Disabled")
        layout.addWidget(self.selection_status_label)
        
        # Mouse coordinates
        self.mouse_coords_label = QLabel("Mouse: (-, -)")
        layout.addWidget(self.mouse_coords_label)
        
        layout.addStretch()
        
        return panel
    
    # Step 10: Point selection implementation
    def enable_selection_mode(self, enabled: bool = True):
        """
        Enable or disable point selection mode.
        
        Args:
            enabled: True to enable selection mode
        """
        self._selection_mode = enabled
        self.selection_mode_btn.setChecked(enabled)
        self.selection_mode_btn.setText("Disable Point Selection" if enabled else "Enable Point Selection")
        self.clear_points_btn.setEnabled(enabled and len(self._selected_points) > 0)
        
        # Update status
        self.selection_status_label.setText(f"Selection: {'Enabled' if enabled else 'Disabled'}")
        
        # Change cursor
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        
        self.selection_mode_changed.emit(enabled)
        self.update()
        
        logger.info(f"Selection mode {'enabled' if enabled else 'disabled'} for {self.viewer_type}")
    
    def _toggle_selection_mode(self):
        """Toggle selection mode on/off."""
        self.enable_selection_mode(not self._selection_mode)
    
    def add_point(self, image_x: float, image_y: float) -> bool:
        """
        Add a point at the specified image coordinates.
        
        Args:
            image_x: X coordinate in image space
            image_y: Y coordinate in image space
            
        Returns:
            True if point was added successfully
        """
        if len(self._selected_points) >= self._max_points:
            logger.warning(f"Maximum {self._max_points} points already selected")
            return False
        
        # Validate coordinates
        if not self._validate_point_coordinates(image_x, image_y):
            logger.warning(f"Point coordinates ({image_x}, {image_y}) are outside image bounds")
            return False
        
        # Add point
        point_index = len(self._selected_points)
        self._selected_points.append((image_x, image_y))
        
        # Update UI
        self._update_point_count_display()
        self.clear_points_btn.setEnabled(True)
        
        # Emit signal
        self.point_added.emit(point_index, image_x, image_y)
        
        # Check if we have maximum points
        if len(self._selected_points) == self._max_points:
            logger.info(f"All {self._max_points} points selected on {self.viewer_type}")
        
        self.update()
        return True
    
    def remove_point(self, point_index: int) -> bool:
        """
        Remove a point by index.
        
        Args:
            point_index: Index of point to remove
            
        Returns:
            True if point was removed successfully
        """
        if 0 <= point_index < len(self._selected_points):
            removed_point = self._selected_points.pop(point_index)
            
            # Update UI
            self._update_point_count_display()
            self.clear_points_btn.setEnabled(len(self._selected_points) > 0)
            
            # Emit signal
            self.point_removed.emit(point_index)
            
            logger.info(f"Point {point_index} removed from {self.viewer_type}: {removed_point}")
            self.update()
            return True
        else:
            logger.warning(f"Invalid point index for removal: {point_index}")
            return False
    
    def move_point(self, point_index: int, new_x: float, new_y: float) -> bool:
        """
        Move a point to new coordinates.
        
        Args:
            point_index: Index of point to move
            new_x: New X coordinate in image space
            new_y: New Y coordinate in image space
            
        Returns:
            True if point was moved successfully
        """
        if 0 <= point_index < len(self._selected_points):
            # Validate new coordinates
            if not self._validate_point_coordinates(new_x, new_y):
                return False
            
            old_point = self._selected_points[point_index]
            self._selected_points[point_index] = (new_x, new_y)
            
            # Emit signal
            self.point_moved.emit(point_index, new_x, new_y)
            
            logger.debug(f"Point {point_index} moved from {old_point} to ({new_x}, {new_y})")
            self.update()
            return True
        else:
            return False
    
    def clear_points(self):
        """Clear all selected points."""
        if self._selected_points:
            self._selected_points.clear()
            
            # Update UI
            self._update_point_count_display()
            self.clear_points_btn.setEnabled(False)
            
            # Emit signal
            self.points_cleared.emit()
            
            logger.info(f"All points cleared from {self.viewer_type}")
            self.update()
    
    def get_selected_points(self) -> List[Tuple[float, float]]:
        """Get list of selected points in image coordinates."""
        return self._selected_points.copy()
    
    def get_point_count(self) -> int:
        """Get number of currently selected points."""
        return len(self._selected_points)
    
    def is_selection_complete(self) -> bool:
        """Check if exactly 3 points are selected."""
        return len(self._selected_points) == self._max_points
    
    def _validate_point_coordinates(self, x: float, y: float) -> bool:
        """Validate that point coordinates are within image bounds."""
        if self._image is None:
            return False
        
        height, width = self._image.shape[:2]
        return 0 <= x < width and 0 <= y < height
    
    def _update_point_count_display(self):
        """Update the point count display."""
        count = len(self._selected_points)
        self.point_count_label.setText(f"Points: {count}/{self._max_points}")
        self.point_count_changed.emit(count)
    
    def _toggle_point_labels(self, show: bool):
        """Toggle display of point labels."""
        self._show_point_labels = show
        self.update()
    
    # Image management
    def set_image(self, image: Optional[np.ndarray]):
        """
        Set the main image to display.
        
        Args:
            image: Image array to display
        """
        self._image = image.copy() if image is not None else None
        self._current_pixmap = None  # Invalidate cache
        
        # Clear points when image changes
        if self._image is not None:
            self.clear_points()
        
        self._update_image_rect()
        self.fit_to_window()
        self.update()
        
        logger.info(f"Image set for {self.viewer_type}: {image.shape if image is not None else None}")
    
    def set_overlay_image(self, overlay: Optional[np.ndarray]):
        """Set overlay image (e.g., GDS overlay on SEM)."""
        self._overlay_image = overlay.copy() if overlay is not None else None
        self._current_pixmap = None  # Invalidate cache
        self.update()
    
    # View management
    def reset_view(self):
        """Reset view to default zoom and pan."""
        self._zoom_factor = 1.0
        self._pan_offset = QPoint(0, 0)
        self._update_image_rect()
        self.update()
        self._emit_view_changed()
    
    def fit_to_window(self):
        """Fit image to window size."""
        if self._image is None:
            return
        
        widget_size = self.size()
        image_height, image_width = self._image.shape[:2]
        
        if image_width == 0 or image_height == 0:
            return
        
        scale_x = widget_size.width() / image_width
        scale_y = widget_size.height() / image_height
        
        self._zoom_factor = min(scale_x, scale_y) * 0.9  # Leave some margin
        self._pan_offset = QPoint(0, 0)
        self._update_image_rect()
        self.update()
        self._emit_view_changed()
    
    def _update_image_rect(self):
        """Update the rectangle where the image is displayed."""
        if self._image is None:
            self._image_rect = QRect()
            return
        
        image_height, image_width = self._image.shape[:2]
        scaled_width = int(image_width * self._zoom_factor)
        scaled_height = int(image_height * self._zoom_factor)
        
        widget_center = self.rect().center()
        image_center = QPoint(scaled_width // 2, scaled_height // 2)
        
        top_left = widget_center - image_center + self._pan_offset
        self._image_rect = QRect(top_left.x(), top_left.y(), scaled_width, scaled_height)
    
    def _emit_view_changed(self):
        """Emit view changed signal with current parameters."""
        view_params = {
            'zoom_factor': self._zoom_factor,
            'pan_offset': (self._pan_offset.x(), self._pan_offset.y()),
            'image_rect': (self._image_rect.x(), self._image_rect.y(), 
                          self._image_rect.width(), self._image_rect.height())
        }
        self.view_changed.emit(view_params)
    
    # Coordinate conversion
    def image_to_widget_coords(self, image_x: float, image_y: float) -> QPointF:
        """Convert image coordinates to widget coordinates."""
        if self._image_rect.isEmpty():
            return QPointF(0, 0)
        
        # Scale to widget coordinates
        widget_x = self._image_rect.x() + (image_x * self._zoom_factor)
        widget_y = self._image_rect.y() + (image_y * self._zoom_factor)
        
        return QPointF(widget_x, widget_y)
    
    def widget_to_image_coords(self, widget_x: float, widget_y: float) -> QPointF:
        """Convert widget coordinates to image coordinates."""
        if self._image_rect.isEmpty():
            return QPointF(0, 0)
        
        # Convert to image coordinates
        image_x = (widget_x - self._image_rect.x()) / self._zoom_factor
        image_y = (widget_y - self._image_rect.y()) / self._zoom_factor
        
        return QPointF(image_x, image_y)
    
    def _find_point_at_position(self, widget_pos: QPoint) -> int:
        """Find point index at the given widget position, or -1 if none."""
        for i, (img_x, img_y) in enumerate(self._selected_points):
            widget_pos_f = self.image_to_widget_coords(img_x, img_y)
            distance = (widget_pos_f - QPointF(widget_pos)).manhattanLength()
            
            if distance <= self._hover_tolerance:
                return i
        
        return -1
    
    # Event handling
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self._selection_mode:
                # Check if clicking on existing point
                point_index = self._find_point_at_position(event.pos())
                
                if point_index >= 0:
                    # Start dragging existing point
                    self._selected_point_index = point_index
                    self._dragging = True
                    logger.debug(f"Started dragging point {point_index}")
                else:
                    # Add new point if possible
                    image_coords = self.widget_to_image_coords(event.pos().x(), event.pos().y())
                    if self._validate_point_coordinates(image_coords.x(), image_coords.y()):
                        self.add_point(image_coords.x(), image_coords.y())
            else:
                # Start view panning
                self._dragging = True
                self._drag_start = event.pos()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        
        elif event.button() == Qt.MouseButton.RightButton:
            # Right click for canvas navigation only (no point operations)
            if not self._selection_mode:
                # Start view panning with right click when not in selection mode
                self._dragging = True
                self._drag_start = event.pos()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events."""
        # Update mouse coordinates display
        if self._image is not None:
            image_coords = self.widget_to_image_coords(event.pos().x(), event.pos().y())
            if self._validate_point_coordinates(image_coords.x(), image_coords.y()):
                self.mouse_coords_label.setText(f"Mouse: ({image_coords.x():.0f}, {image_coords.y():.0f})")
            else:
                self.mouse_coords_label.setText("Mouse: (outside)")
        
        if self._dragging:
            if self._selection_mode and self._selected_point_index >= 0:
                # Drag existing point
                image_coords = self.widget_to_image_coords(event.pos().x(), event.pos().y())
                if self._validate_point_coordinates(image_coords.x(), image_coords.y()):
                    self.move_point(self._selected_point_index, image_coords.x(), image_coords.y())
            else:
                # Pan view
                delta = event.pos() - self._drag_start
                self._pan_offset += delta
                self._drag_start = event.pos()
                self._update_image_rect()
                self.update()
        else:
            # Update hover state
            if self._selection_mode:
                old_hover = self._hover_point_index
                self._hover_point_index = self._find_point_at_position(event.pos())
                
                if old_hover != self._hover_point_index:
                    # Update cursor
                    if self._hover_point_index >= 0:
                        self.setCursor(Qt.CursorShape.OpenHandCursor)
                    else:
                        self.setCursor(Qt.CursorShape.CrossCursor)
                    self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self._selected_point_index = -1
            
            if self._selection_mode:
                self.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel events for zooming."""
        if self._image is None:
            return
        
        # Zoom in/out
        zoom_delta = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        old_zoom = self._zoom_factor
        self._zoom_factor *= zoom_delta
        
        # Limit zoom range
        self._zoom_factor = max(0.1, min(10.0, self._zoom_factor))
        
        if self._zoom_factor != old_zoom:
            # Zoom towards mouse position
            mouse_pos = event.position().toPoint()
            image_coords = self.widget_to_image_coords(mouse_pos.x(), mouse_pos.y())
            
            self._update_image_rect()
            
            # Adjust pan to keep mouse position stable
            new_widget_pos = self.image_to_widget_coords(image_coords.x(), image_coords.y())
            pan_adjustment = mouse_pos - new_widget_pos.toPoint()
            self._pan_offset += pan_adjustment
            
            self._update_image_rect()
            self.update()
            self._emit_view_changed()
    
    def paintEvent(self, event: QPaintEvent):
        """Handle paint events."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), QColor(50, 50, 50))
        
        if self._image is None:
            # Draw "No Image" text
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"No {self.viewer_type} Image Loaded")
            return
        
        # Draw main image
        self._draw_image(painter)
        
        # Draw overlay if present
        if self._overlay_image is not None and self._overlay_visible:
            self._draw_overlay(painter)
        
        # Draw selected points
        if self._selection_mode or self._selected_points:
            self._draw_points(painter)
        
        # Draw selection mode indicator
        if self._selection_mode:
            self._draw_selection_mode_indicator(painter)
    
    def _draw_image(self, painter: QPainter):
        """Draw the main image."""
        if self._current_pixmap is None and self._image is not None:
            self._current_pixmap = self._create_pixmap_from_image(self._image)
        
        if self._current_pixmap and not self._image_rect.isEmpty():
            painter.drawPixmap(self._image_rect, self._current_pixmap)
    
    def _draw_overlay(self, painter: QPainter):
        """Draw overlay image."""
        if self._overlay_image is not None:
            overlay_pixmap = self._create_pixmap_from_image(self._overlay_image)
            if overlay_pixmap and not self._image_rect.isEmpty():
                painter.setOpacity(self._overlay_alpha)
                painter.drawPixmap(self._image_rect, overlay_pixmap)
                painter.setOpacity(1.0)
    
    def _draw_points(self, painter: QPainter):
        """Draw selected points with visual markers."""
        for i, (img_x, img_y) in enumerate(self._selected_points):
            widget_pos = self.image_to_widget_coords(img_x, img_y)
            
            # Choose color and style based on point index and state
            color = self._point_colors[i % len(self._point_colors)]
            
            # Highlight hovered or selected point
            if i == self._hover_point_index:
                painter.setPen(QPen(QColor(255, 255, 255), 3))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(widget_pos.toPoint(), self._point_radius + 2, self._point_radius + 2)
            elif i == self._selected_point_index:
                painter.setPen(QPen(QColor(255, 255, 0), 2))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(widget_pos.toPoint(), self._point_radius, self._point_radius)
            else:
                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(widget_pos.toPoint(), self._point_radius, self._point_radius)
            
            # Draw point label
            if self._show_point_labels:
                label_pos = widget_pos + QPointF(self._point_radius + 5, -self._point_radius)
                painter.setPen(QColor(255, 255, 255))
                font = QFont()
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(label_pos.toPoint(), f"{i + 1}")
    
    def _draw_selection_mode_indicator(self, painter: QPainter):
        """Draw indicator that selection mode is active."""
        # Draw border to indicate selection mode
        painter.setPen(QPen(QColor(0, 255, 0), 3))
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawRect(self.rect().adjusted(1, 1, -1, -1))
        
        # Draw mode text
        painter.setPen(QColor(0, 255, 0))
        font = QFont()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(10, 25, "POINT SELECTION MODE")
    
    def _create_pixmap_from_image(self, image: np.ndarray) -> QPixmap:
        """Create QPixmap from numpy image array."""
        if image is None:
            return QPixmap()
        
        # Convert to QImage format
        if len(image.shape) == 2:
            # Grayscale
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        else:
            # Color (assume BGR)
            height, width, channels = image.shape
            if channels == 3:
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                bytes_per_line = 3 * width
                q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            else:
                return QPixmap()  # Unsupported format
        
        return QPixmap.fromImage(q_image) 
