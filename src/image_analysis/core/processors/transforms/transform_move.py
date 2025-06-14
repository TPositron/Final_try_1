import cv2
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSpinBox, QHBoxLayout
from PySide6.QtCore import Signal

class MoveTransformWidget(QWidget):
    move_changed = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        label_x = QLabel("Move X:")
        self.spin_x = QSpinBox()
        self.spin_x.setRange(-1000, 1000)
        self.spin_x.setValue(0)
        self.spin_x.valueChanged.connect(self._emit_move_changed)
        row_x = QHBoxLayout()
        row_x.addWidget(label_x)
        row_x.addWidget(self.spin_x)
        layout.addLayout(row_x)

        label_y = QLabel("Move Y:")
        self.spin_y = QSpinBox()
        self.spin_y.setRange(-1000, 1000)
        self.spin_y.setValue(0)
        self.spin_y.valueChanged.connect(self._emit_move_changed)
        row_y = QHBoxLayout()
        row_y.addWidget(label_y)
        row_y.addWidget(self.spin_y)
        layout.addLayout(row_y)

    def _emit_move_changed(self, _):
        self.move_changed.emit(self.spin_x.value(), self.spin_y.value())

def move_image(image, dx, dy):
    """Move image in x and y directions with white padding."""
    h, w = image.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    moved = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return moved
