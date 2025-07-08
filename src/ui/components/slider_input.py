"""Slider input component with numeric input and ±1/±10 buttons."""

from typing import Optional, Callable, Union
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, 
                              QSlider, QSpinBox, QDoubleSpinBox, QPushButton)
from PySide6.QtCore import Signal, Qt


class SliderInput(QWidget):
    """Widget combining a slider with numeric input and increment/decrement buttons."""
    
    # Signals
    value_changed = Signal(float)  # Emitted when value changes
    
    def __init__(self, label: str, min_value: float = 0.0, max_value: float = 100.0, 
                 initial_value: float = 0.0, decimals: int = 1, step: float = 1.0, parent=None):
        super().__init__(parent)
        
        self.min_value = min_value
        self.max_value = max_value
        self.decimals = decimals
        self.step = step
        self._updating = False  # Flag to prevent circular updates
        
        # Type annotation for spinbox (will be set in _setup_ui)
        self.spinbox: Union[QSpinBox, QDoubleSpinBox]
        
        self._setup_ui(label, initial_value)
        self.set_value(initial_value)
    
    def _setup_ui(self, label: str, initial_value: float):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Label
        self.label = QLabel(label)
        self.label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.label)
        
        # Controls row
        controls_layout = QHBoxLayout()
        
        # Decrease buttons
        self.decrease_10_btn = QPushButton("-10")
        self.decrease_10_btn.setMaximumWidth(40)
        self.decrease_10_btn.clicked.connect(lambda: self._adjust_value(-10 * self.step))
        controls_layout.addWidget(self.decrease_10_btn)
        
        self.decrease_1_btn = QPushButton("-1")
        self.decrease_1_btn.setMaximumWidth(35)
        self.decrease_1_btn.clicked.connect(lambda: self._adjust_value(-self.step))
        controls_layout.addWidget(self.decrease_1_btn)
        
        # Numeric input
        if self.decimals == 0:
            self.spinbox = QSpinBox()
            self.spinbox.setRange(int(self.min_value), int(self.max_value))
            self.spinbox.setSingleStep(int(self.step))
        else:
            self.spinbox = QDoubleSpinBox()
            self.spinbox.setRange(self.min_value, self.max_value)
            self.spinbox.setDecimals(self.decimals)
            self.spinbox.setSingleStep(self.step)
        
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)
        controls_layout.addWidget(self.spinbox)
        
        # Increase buttons
        self.increase_1_btn = QPushButton("+1")
        self.increase_1_btn.setMaximumWidth(35)
        self.increase_1_btn.clicked.connect(lambda: self._adjust_value(self.step))
        controls_layout.addWidget(self.increase_1_btn)
        
        self.increase_10_btn = QPushButton("+10")
        self.increase_10_btn.setMaximumWidth(40)
        self.increase_10_btn.clicked.connect(lambda: self._adjust_value(10 * self.step))
        controls_layout.addWidget(self.increase_10_btn)
        
        layout.addLayout(controls_layout)
        
        # Slider - Fixed Qt enum and type conversion
        self.slider = QSlider(Qt.Orientation.Horizontal)
        # Convert float range to integer range for slider
        self.slider_range = int((self.max_value - self.min_value) / self.step)
        self.slider.setRange(0, self.slider_range)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)
        
        # Value display - Fixed Qt enum
        self.value_label = QLabel()
        self.value_label.setStyleSheet("color: blue; font-family: monospace;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.value_label)
    
    def _on_slider_changed(self, slider_value: int):
        """Handle slider value change."""
        if self._updating:
            return
        
        real_value = self.min_value + (slider_value * self.step)
        self._update_value(real_value, update_slider=False)
    
    def _on_spinbox_changed(self, spinbox_value: float):
        """Handle spinbox value change."""
        if self._updating:
            return
        
        self._update_value(spinbox_value, update_spinbox=False)
    
    def _adjust_value(self, delta: float):
        """Adjust the current value by a delta amount."""
        current_value = self.get_value()
        new_value = current_value + delta
        self.set_value(new_value)
    
    def _update_value(self, value: float, update_slider: bool = True, update_spinbox: bool = True):
        """Update all components with a new value."""
        self._updating = True
        
        # Clamp value to range
        value = max(self.min_value, min(self.max_value, value))
        
        # Update slider - Fixed type conversion
        if update_slider:
            slider_value = int((value - self.min_value) / self.step)
            self.slider.setValue(slider_value)
        
        # Update spinbox - Fix type conversion for QSpinBox vs QDoubleSpinBox
        if update_spinbox:
            if isinstance(self.spinbox, QSpinBox):
                self.spinbox.setValue(int(value))  # QSpinBox expects int
            else:
                self.spinbox.setValue(value)  # QDoubleSpinBox accepts float
        
        # Update value display
        if self.decimals == 0:
            self.value_label.setText(f"{int(value)}")
        else:
            self.value_label.setText(f"{value:.{self.decimals}f}")
        
        self._updating = False
        
        # Emit signal
        self.value_changed.emit(value)
    
    def set_value(self, value: float):
        """Set the widget value."""
        self._update_value(value)
    
    def get_value(self) -> float:
        """Get the current widget value."""
        return float(self.spinbox.value())
    
    def set_range(self, min_value: float, max_value: float):
        """Set the value range."""
        self.min_value = min_value
        self.max_value = max_value
        
        # Update spinbox range - Fix type conversion
        if isinstance(self.spinbox, QSpinBox):
            self.spinbox.setRange(int(min_value), int(max_value))  # QSpinBox expects int
        else:
            self.spinbox.setRange(min_value, max_value)  # QDoubleSpinBox accepts float
        
        # Update slider range - Fixed type conversion
        self.slider_range = int((max_value - min_value) / self.step)
        self.slider.setRange(0, self.slider_range)
        
        # Clamp current value to new range
        current_value = self.get_value()
        if current_value < min_value or current_value > max_value:
            self.set_value(max(min_value, min(max_value, current_value)))
    
    def set_enabled(self, enabled: bool):
        """Enable or disable the widget."""
        super().setEnabled(enabled)
        self.slider.setEnabled(enabled)
        self.spinbox.setEnabled(enabled)
        self.decrease_10_btn.setEnabled(enabled)
        self.decrease_1_btn.setEnabled(enabled)
        self.increase_1_btn.setEnabled(enabled)
        self.increase_10_btn.setEnabled(enabled)
    
    def reset_to_default(self, default_value: float = 0.0):
        """Reset the widget to a default value."""
        self.set_value(default_value)
