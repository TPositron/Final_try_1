import cv2
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider
from PySide6.QtCore import Qt, Signal

class RotateTransformWidget(QWidget):
    rotation_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        label = QLabel("Rotation (-90° to +90°):")
        layout.addWidget(label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(-90)
        self.slider.setMaximum(90)
        self.slider.setValue(0)
        self.slider.setTickInterval(5)
        self.slider.setTickPosition(QSlider.TicksBelow)
        layout.addWidget(self.slider)
        self.slider.valueChanged.connect(self._emit_rotation_changed)

    def _emit_rotation_changed(self, value):
        self.rotation_changed.emit(float(value))

def rotate_image(image, angle_degrees):
    """
    Rotate the image around its center with edge replication.

    Args:
        image: Input grayscale image (numpy array)
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise)

    Returns:
        Rotated image
    """
    if not (-90 <= angle_degrees <= 90):
        raise ValueError("Rotation angle must be between -90 and +90 degrees")

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
