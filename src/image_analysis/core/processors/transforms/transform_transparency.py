from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider
from PySide6.QtCore import Qt, Signal

class TransparencyControlWidget(QWidget):
    transparency_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        label = QLabel("Transparency (0% to 100%)")
        layout.addWidget(label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(70)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QSlider.TicksBelow)
        layout.addWidget(self.slider)
        self.slider.valueChanged.connect(self._emit_transparency_changed)

    def _emit_transparency_changed(self, value):
        self.transparency_changed.emit(value)

    def get_transparency(self):
        return self.slider.value()

    def reset(self):
        self.slider.setValue(70)
        self.transparency_changed.emit(70)
