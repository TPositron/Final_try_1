from PySide6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QSlider, QPushButton, QFormLayout, QSpinBox, QDoubleSpinBox, QGroupBox, QLineEdit)
from PySide6.QtCore import Qt, Signal
from typing import Dict, Any, Optional


class FilterPanel(QWidget):
    filter_changed = Signal(str, dict)
    apply_clicked = Signal(str, dict)
    reset_clicked = Signal()

    def __init__(self, parent=None, available_filters=None, filter_params=None):
        super().__init__(parent)
        self.available_filters = available_filters or []
        self.filter_params = filter_params or {}
        self.current_filter = None
        self.param_widgets = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        filter_box = QHBoxLayout()
        filter_label = QLabel("Filter:")
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(self.available_filters)
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        filter_box.addWidget(filter_label)
        filter_box.addWidget(self.filter_combo)
        layout.addLayout(filter_box)

        self.param_group = QGroupBox("Parameters")
        self.param_layout = QFormLayout()
        self.param_group.setLayout(self.param_layout)
        layout.addWidget(self.param_group)

        btn_box = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.apply_btn = QPushButton("Apply")
        self.reset_btn = QPushButton("Reset")
        btn_box.addWidget(self.preview_btn)
        btn_box.addWidget(self.apply_btn)
        btn_box.addWidget(self.reset_btn)
        layout.addLayout(btn_box)

        self.preview_btn.clicked.connect(self._emit_filter_changed)
        self.apply_btn.clicked.connect(self._emit_apply_clicked)
        self.reset_btn.clicked.connect(self.reset_clicked.emit)

        if self.available_filters:
            self._on_filter_changed(self.available_filters[0])

    def _on_filter_changed(self, filter_name):
        self.current_filter = filter_name
        self._update_param_widgets()
        self._emit_filter_changed()

    def _update_param_widgets(self):
        for i in reversed(range(self.param_layout.count())):
            widget = self.param_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.param_widgets.clear()
        params = self.filter_params.get(self.current_filter, {})
        for param, spec in params.items():
            if spec['type'] == 'int':
                widget = QSpinBox()
                widget.setMinimum(spec.get('min', 0))
                widget.setMaximum(spec.get('max', 100))
                widget.setValue(spec.get('default', 0))
            elif spec['type'] == 'float':
                widget = QDoubleSpinBox()
                widget.setMinimum(spec.get('min', 0.0))
                widget.setMaximum(spec.get('max', 100.0))
                widget.setSingleStep(0.01)
                widget.setValue(spec.get('default', 0.0))
            elif spec['type'] == 'str' and 'options' in spec:
                widget = QComboBox()
                widget.addItems(spec['options'])
                widget.setCurrentText(str(spec.get('default', '')))
            else:
                widget = QLineEdit()
                widget.setText(str(spec.get('default', '')))
            widget.valueChanged = widget.valueChanged if hasattr(widget, 'valueChanged') else widget.textChanged
            widget.valueChanged.connect(self._emit_filter_changed)
            self.param_layout.addRow(QLabel(param), widget)
            self.param_widgets[param] = widget

    def _get_current_params(self):
        params = {}
        for param, widget in self.param_widgets.items():
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                params[param] = widget.value()
            elif isinstance(widget, QComboBox):
                params[param] = widget.currentText()
            else:
                params[param] = widget.text()
        return params

    def _emit_filter_changed(self):
        if self.current_filter:
            self.filter_changed.emit(self.current_filter, self._get_current_params())

    def _emit_apply_clicked(self):
        if self.current_filter:
            self.apply_clicked.emit(self.current_filter, self._get_current_params())