import cv2
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QHBoxLayout, QPushButton, QSpinBox
from PySide6.QtCore import Qt, Signal

class ZoomTransformWidget(QWidget):
    zoom_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        label = QLabel("Zoom (10% to 300%):")
        layout.addWidget(label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(10)
        self.slider.setMaximum(300)
        self.slider.setValue(100)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QSlider.TicksBelow)
        layout.addWidget(self.slider)
        self.slider.valueChanged.connect(self._emit_zoom_changed)

    def _emit_zoom_changed(self, value):
        self.zoom_changed.emit(value)

def zoom_image(image, zoom_percent):
    """
    Zoom the image around its center.
    Args:
        image: Grayscale image (numpy array).
        zoom_percent: Zoom level (e.g., 100 = normal, >100 = zoom in, <100 = zoom out)
    Returns:
        Transformed image with white padding.
    """
    if zoom_percent <= 0:
        raise ValueError("Zoom percentage must be > 0")

    h, w = image.shape[:2]
    scale = zoom_percent / 100.0
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)
