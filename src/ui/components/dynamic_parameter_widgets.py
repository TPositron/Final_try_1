"""
Dynamic Parameter Input Components - Advanced Parameter Input Widgets

This module provides advanced parameter input widgets with up/down buttons,
validation, and real-time updates for filter parameter management.

Main Classes:
- ParameterInputWidget: Base class for parameter input widgets
- NumericParameterWidget: Widget for numeric parameters (int/float)
- BooleanParameterWidget: Widget for boolean parameters
- ChoiceParameterWidget: Widget for parameters with predefined choices
- TupleParameterWidget: Widget for tuple parameters
- DynamicParameterPanel: Panel for dynamic parameter generation

Key Methods:
- setup_ui(): Sets up UI for parameter widget
- get_value(): Gets current parameter value
- set_value(): Sets parameter value
- reset_to_default(): Resets parameter to default value
- set_filter_parameters(): Sets parameters for selected filter
- get_current_parameters(): Gets current parameter values

Signals Emitted:
- value_changed(str, object): Parameter value changed
- value_preview(str, object): Parameter value preview for real-time updates
- parameters_changed(dict): All parameters changed
- parameters_preview(dict): All parameters preview

Dependencies:
- Uses: PySide6.QtWidgets, PySide6.QtCore, PySide6.QtGui (Qt framework)
- Uses: services/filters/filter_parameter_parser.FilterParameter
- Called by: UI filtering components
- Coordinates with: Filter parameter management system

Features:
- Advanced parameter input widgets with validation
- Up/down buttons for enhanced control
- Real-time preview updates with timer delays
- Dynamic parameter generation based on filter definitions
- Support for multiple parameter types (numeric, boolean, choice, tuple)
- Dark theme styling for consistent UI appearance
- Parameter reset and default value management
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox,
                               QComboBox, QSlider, QGroupBox, QFrame)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
from typing import Any, Optional, Callable, Union, List
from src.services.filters.filter_parameter_parser import FilterParameter, ParameterType


class ParameterInputWidget(QWidget):
    """Base class for parameter input widgets."""
    
    # Signals
    value_changed = Signal(str, object)  # parameter_name, value
    value_preview = Signal(str, object)  # parameter_name, value (for real-time preview)
    
    def __init__(self, parameter: FilterParameter, parent=None):
        super().__init__(parent)
        self.parameter = parameter
        self.current_value = parameter.default_value
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI for this parameter widget."""
        raise NotImplementedError
        
    def get_value(self) -> Any:
        """Get the current parameter value."""
        return self.current_value
        
    def set_value(self, value: Any) -> None:
        """Set the parameter value."""
        raise NotImplementedError
        
    def reset_to_default(self) -> None:
        """Reset parameter to default value."""
        if self.parameter.default_value is not None:
            self.set_value(self.parameter.default_value)


class NumericParameterWidget(ParameterInputWidget):
    """Widget for numeric parameters (int/float) with up/down buttons."""
    
    def __init__(self, parameter: FilterParameter, parent=None):
        super().__init__(parameter, parent)
        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._emit_preview)
        
    def setup_ui(self):
        """Setup UI for numeric parameter."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Parameter label
        label = QLabel(f"{self.parameter.description or self.parameter.name}:")
        label.setMinimumWidth(120)
        layout.addWidget(label)
        
        # Create appropriate numeric input
        if self.parameter.param_type == ParameterType.INTEGER:
            self.control = QSpinBox()
            self.control.setRange(
                int(self.parameter.min_value or -999999), 
                int(self.parameter.max_value or 999999)
            )
            if self.parameter.step:
                self.control.setSingleStep(int(self.parameter.step))
            if self.parameter.default_value is not None:
                self.control.setValue(int(self.parameter.default_value))
                self.current_value = int(self.parameter.default_value)
        else:  # FLOAT
            self.control = QDoubleSpinBox()
            self.control.setRange(
                float(self.parameter.min_value or -999999.0), 
                float(self.parameter.max_value or 999999.0)
            )
            self.control.setDecimals(3)
            if self.parameter.step:
                self.control.setSingleStep(float(self.parameter.step))
            else:
                self.control.setSingleStep(0.1)
            if self.parameter.default_value is not None:
                self.control.setValue(float(self.parameter.default_value))
                self.current_value = float(self.parameter.default_value)
        
        # Style the control
        self.control.setStyleSheet("""
            QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 2px solid #555555;
                border-radius: 4px;
                padding: 4px;
                min-width: 80px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #0078d4;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                background-color: #555555;
                border-radius: 2px;
                width: 16px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                background-color: #555555;
                border-radius: 2px;
                width: 16px;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
                background-color: #0078d4;
            }
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #0078d4;
            }
        """)
        
        layout.addWidget(self.control)
        
        # Custom up/down buttons for enhanced control
        button_layout = QVBoxLayout()
        button_layout.setSpacing(1)
        
        # Up button
        self.up_button = QPushButton("▲")
        self.up_button.setMaximumSize(24, 12)
        self.up_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 2px;
                font-size: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """)
        
        # Down button
        self.down_button = QPushButton("▼")
        self.down_button.setMaximumSize(24, 12)
        self.down_button.setStyleSheet(self.up_button.styleSheet())
        
        button_layout.addWidget(self.up_button)
        button_layout.addWidget(self.down_button)
        layout.addLayout(button_layout)
        
        # Value display label
        self.value_label = QLabel(str(self.current_value))
        self.value_label.setMinimumWidth(60)
        self.value_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px 6px;
                font-family: monospace;
            }
        """)
        layout.addWidget(self.value_label)
        
        # Add stretch
        layout.addStretch()
        
        # Connect signals
        self.control.valueChanged.connect(self._on_value_changed)
        self.up_button.clicked.connect(self._increment_value)
        self.down_button.clicked.connect(self._decrement_value)
        
    def _on_value_changed(self, value):
        """Handle value change from spin box."""
        self.current_value = value
        self.value_label.setText(str(value))
        self.value_changed.emit(self.parameter.name, value)
        
        # Start preview timer for real-time updates
        self.preview_timer.start(300)  # 300ms delay
        
    def _emit_preview(self):
        """Emit preview signal after timer."""
        self.value_preview.emit(self.parameter.name, self.current_value)
        
    def _increment_value(self):
        """Increment value using custom button."""
        step = self.parameter.step or (1 if self.parameter.param_type == ParameterType.INTEGER else 0.1)
        new_value = self.current_value + step
        
        if self.parameter.max_value is not None:
            new_value = min(new_value, self.parameter.max_value)
            
        self.set_value(new_value)
        
    def _decrement_value(self):
        """Decrement value using custom button."""
        step = self.parameter.step or (1 if self.parameter.param_type == ParameterType.INTEGER else 0.1)
        new_value = self.current_value - step
        
        if self.parameter.min_value is not None:
            new_value = max(new_value, self.parameter.min_value)
            
        self.set_value(new_value)
        
    def set_value(self, value: Any) -> None:
        """Set the parameter value."""
        if self.parameter.param_type == ParameterType.INTEGER:
            value = int(value)
        else:
            value = float(value)
            
        self.control.setValue(value)
        self.current_value = value
        self.value_label.setText(str(value))
        

class BooleanParameterWidget(ParameterInputWidget):
    """Widget for boolean parameters."""
    
    def setup_ui(self):
        """Setup UI for boolean parameter."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Checkbox control
        self.control = QCheckBox(self.parameter.description or self.parameter.name)
        if self.parameter.default_value is not None:
            self.control.setChecked(bool(self.parameter.default_value))
            self.current_value = bool(self.parameter.default_value)
        
        self.control.setStyleSheet("""
            QCheckBox {
                color: #ffffff;
                font-weight: bold;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #3c3c3c;
                border: 2px solid #555555;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border: 2px solid #0078d4;
                border-radius: 4px;
            }
        """)
        
        layout.addWidget(self.control)
        layout.addStretch()
        
        # Connect signals
        self.control.toggled.connect(self._on_value_changed)
        
    def _on_value_changed(self, value):
        """Handle value change."""
        self.current_value = value
        self.value_changed.emit(self.parameter.name, value)
        self.value_preview.emit(self.parameter.name, value)
        
    def set_value(self, value: Any) -> None:
        """Set the parameter value."""
        value = bool(value)
        self.control.setChecked(value)
        self.current_value = value


class ChoiceParameterWidget(ParameterInputWidget):
    """Widget for parameters with predefined choices."""
    
    def setup_ui(self):
        """Setup UI for choice parameter."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Parameter label
        label = QLabel(f"{self.parameter.description or self.parameter.name}:")
        label.setMinimumWidth(120)
        layout.addWidget(label)
        
        # Combo box control
        self.control = QComboBox()
        
        if self.parameter.choices:
            self.control.addItems([str(choice) for choice in self.parameter.choices])
            if self.parameter.default_value is not None:
                index = self.control.findText(str(self.parameter.default_value))
                if index >= 0:
                    self.control.setCurrentIndex(index)
                self.current_value = self.parameter.default_value
        
        self.control.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 2px solid #555555;
                border-radius: 4px;
                padding: 4px;
                min-width: 100px;
            }
            QComboBox:focus {
                border-color: #0078d4;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #ffffff;
            }
        """)
        
        layout.addWidget(self.control)
        layout.addStretch()
        
        # Connect signals
        self.control.currentTextChanged.connect(self._on_value_changed)
        
    def _on_value_changed(self, value):
        """Handle value change."""
        self.current_value = value
        self.value_changed.emit(self.parameter.name, value)
        self.value_preview.emit(self.parameter.name, value)
        
    def set_value(self, value: Any) -> None:
        """Set the parameter value."""
        index = self.control.findText(str(value))
        if index >= 0:
            self.control.setCurrentIndex(index)
            self.current_value = value


class TupleParameterWidget(ParameterInputWidget):
    """Widget for tuple parameters (like tile_grid_size)."""
    
    def setup_ui(self):
        """Setup UI for tuple parameter."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Parameter label
        label = QLabel(f"{self.parameter.description or self.parameter.name}:")
        label.setMinimumWidth(120)
        layout.addWidget(label)
        
        # Container for tuple values
        tuple_layout = QHBoxLayout()
        
        self.controls = []
        if self.parameter.default_value and isinstance(self.parameter.default_value, (tuple, list)):
            for i, value in enumerate(self.parameter.default_value):
                spin_box = QSpinBox()
                spin_box.setRange(1, 999)
                spin_box.setValue(int(value))
                spin_box.setStyleSheet("""
                    QSpinBox {
                        background-color: #3c3c3c;
                        color: #ffffff;
                        border: 2px solid #555555;
                        border-radius: 4px;
                        padding: 2px;
                        max-width: 60px;
                    }
                """)
                spin_box.valueChanged.connect(self._on_value_changed)
                self.controls.append(spin_box)
                tuple_layout.addWidget(spin_box)
                
                if i < len(self.parameter.default_value) - 1:
                    tuple_layout.addWidget(QLabel("×"))
            
            self.current_value = tuple(self.parameter.default_value)
        
        layout.addLayout(tuple_layout)
        layout.addStretch()
        
    def _on_value_changed(self):
        """Handle value change."""
        if self.controls:
            self.current_value = tuple(control.value() for control in self.controls)
            self.value_changed.emit(self.parameter.name, self.current_value)
            self.value_preview.emit(self.parameter.name, self.current_value)
        
    def set_value(self, value: Any) -> None:
        """Set the parameter value."""
        if isinstance(value, (tuple, list)) and len(value) == len(self.controls):
            for control, val in zip(self.controls, value):
                control.setValue(int(val))
            self.current_value = tuple(value)


class DynamicParameterPanel(QWidget):
    """Panel that dynamically generates parameter inputs based on filter definition."""
    
    # Signals
    parameters_changed = Signal(dict)  # {param_name: value}
    parameters_preview = Signal(dict)  # {param_name: value} for real-time preview
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parameter_widgets = {}
        self.current_filter_name = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the dynamic parameter panel UI."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(8)
        
        # Header
        header_label = QLabel("Filter Parameters")
        header_label.setFont(QFont("Arial", 10, QFont.Bold))
        header_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                background-color: #404040;
                padding: 6px;
                border-radius: 4px;
            }
        """)
        self.main_layout.addWidget(header_label)
        
        # Scrollable content area for parameters
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(6)
        
        self.main_layout.addWidget(self.content_widget)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        self.reset_button.clicked.connect(self.reset_all_parameters)
        
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        
        self.main_layout.addLayout(button_layout)
        self.main_layout.addStretch()
        
    def set_filter_parameters(self, filter_name: str, parameters: List[FilterParameter]):
        """Set parameters for the selected filter."""
        self.current_filter_name = filter_name
        
        # Clear existing parameter widgets
        self.clear_parameters()
        
        # Create new parameter widgets
        for parameter in parameters:
            widget = self._create_parameter_widget(parameter)
            if widget:
                self.parameter_widgets[parameter.name] = widget
                self.content_layout.addWidget(widget)
                
                # Connect signals
                widget.value_changed.connect(self._on_parameter_changed)
                widget.value_preview.connect(self._on_parameter_preview)
        
        # Add separator line if there are parameters
        if parameters:
            separator = QFrame()
            separator.setFrameStyle(QFrame.HLine | QFrame.Sunken)
            separator.setStyleSheet("QFrame { color: #555555; }")
            self.content_layout.addWidget(separator)
        
        # Update button state
        self.reset_button.setEnabled(len(parameters) > 0)
        
    def clear_parameters(self):
        """Clear all parameter widgets."""
        for widget in self.parameter_widgets.values():
            widget.setParent(None)
        self.parameter_widgets.clear()
        
        # Clear layout
        for i in reversed(range(self.content_layout.count())):
            item = self.content_layout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
                
    def _create_parameter_widget(self, parameter: FilterParameter) -> Optional[ParameterInputWidget]:
        """Create appropriate widget for parameter type."""
        if parameter.param_type in [ParameterType.INTEGER, ParameterType.FLOAT]:
            return NumericParameterWidget(parameter)
        elif parameter.param_type == ParameterType.BOOLEAN:
            return BooleanParameterWidget(parameter)
        elif parameter.param_type == ParameterType.STRING and parameter.choices:
            return ChoiceParameterWidget(parameter)
        elif parameter.param_type == ParameterType.TUPLE:
            return TupleParameterWidget(parameter)
        else:
            # Fallback for unsupported types
            return None
            
    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter value change."""
        current_params = self.get_current_parameters()
        self.parameters_changed.emit(current_params)
        
    def _on_parameter_preview(self, param_name: str, value: Any):
        """Handle parameter preview value change."""
        current_params = self.get_current_parameters()
        self.parameters_preview.emit(current_params)
        
    def set_filter_configs(self, filter_configs: dict):
        """Set filter configurations for dynamic parameter generation."""
        self.filter_configs = filter_configs
        
    def get_current_parameters(self) -> dict:
        """Get current parameter values."""
        params = {}
        for name, widget in self.parameter_widgets.items():
            params[name] = widget.get_value()
        return params
        
    def set_parameter_values(self, parameters: dict):
        """Set parameter values."""
        for name, value in parameters.items():
            if name in self.parameter_widgets:
                self.parameter_widgets[name].set_value(value)
                
    def reset_all_parameters(self):
        """Reset all parameters to their default values."""
        for widget in self.parameter_widgets.values():
            widget.reset_to_default()
            
        # Emit update signals
        current_params = self.get_current_parameters()
        self.parameters_changed.emit(current_params)
        self.parameters_preview.emit(current_params)
        
    def update_for_filter(self, filter_name: str):
        """Update parameter controls for the selected filter."""
        # This method would integrate with the filter parameter storage
        # For now, we'll create a placeholder that shows the filter name
        if filter_name:
            # Clear existing parameters
            self.clear_parameters()
            
            # In a complete implementation, this would load parameters from
            # the filter parameter storage/parser system
            # For now, we'll add a placeholder label
            placeholder_label = QLabel(f"Parameters for filter: {filter_name}")
            placeholder_label.setStyleSheet("""
                QLabel {
                    color: #cccccc;
                    background-color: #333333;
                    padding: 8px;
                    border-radius: 4px;
                    font-style: italic;
                }
            """)
            self.content_layout.addWidget(placeholder_label)
            
            # Enable the reset button
            self.reset_button.setEnabled(True)
        else:
            self.clear_parameters()
            self.reset_button.setEnabled(False)
            
    def set_parameters(self, parameters: dict):
        """Set parameter values (alias for set_parameter_values)."""
        self.set_parameter_values(parameters)
        
    def reset(self):
        """Reset the panel (alias for reset_all_parameters)."""
        self.reset_all_parameters()
