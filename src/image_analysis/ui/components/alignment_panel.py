from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                               QSpinBox, QDoubleSpinBox, QPushButton, QLabel,
                               QSlider, QCheckBox, QComboBox)  # Added QComboBox
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
import numpy as np


class AlignmentPanel(QWidget):
    alignment_changed = Signal(dict)
    reset_requested = Signal()
    structure_selected = Signal(int)  # New signal for structure selection
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gds_overlay = None
        self.live_update_enabled = True
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._emit_alignment_changed)
        
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Structure selection dropdown
        self.structure_combo = QComboBox()
        self.structure_combo.addItem("Select Structure", -1)
        self.structure_combo.currentIndexChanged.connect(self._on_structure_selected)
        layout.addWidget(QLabel("GDS Structure for Alignment:"))
        layout.addWidget(self.structure_combo)

        title_label = QLabel("Alignment Controls")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        self.create_translation_controls(layout)
        self.create_rotation_controls(layout)
        self.create_scale_controls(layout)
        self.create_transparency_controls(layout)
        self.create_action_buttons(layout)
        self.create_options_controls(layout)
        
        layout.addStretch()
        
    def create_translation_controls(self, parent_layout):
        group = QGroupBox("Translation")
        layout = QVBoxLayout(group)
        
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X Offset:"))
        
        self.x_spinbox = QSpinBox()
        self.x_spinbox.setRange(-1000, 1000)
        self.x_spinbox.setValue(0)
        self.x_spinbox.setSuffix(" px")
        x_layout.addWidget(self.x_spinbox)
        
        x_btn_layout = QHBoxLayout()
        self.x_minus_10_btn = QPushButton("-10")
        self.x_minus_1_btn = QPushButton("-1")
        self.x_plus_1_btn = QPushButton("+1")
        self.x_plus_10_btn = QPushButton("+10")
        
        for btn in [self.x_minus_10_btn, self.x_minus_1_btn, self.x_plus_1_btn, self.x_plus_10_btn]:
            btn.setMaximumWidth(40)
            x_btn_layout.addWidget(btn)
        
        x_layout.addLayout(x_btn_layout)
        layout.addLayout(x_layout)
        
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y Offset:"))
        
        self.y_spinbox = QSpinBox()
        self.y_spinbox.setRange(-1000, 1000)
        self.y_spinbox.setValue(0)
        self.y_spinbox.setSuffix(" px")
        y_layout.addWidget(self.y_spinbox)
        
        y_btn_layout = QHBoxLayout()
        self.y_minus_10_btn = QPushButton("-10")
        self.y_minus_1_btn = QPushButton("-1")
        self.y_plus_1_btn = QPushButton("+1")
        self.y_plus_10_btn = QPushButton("+10")
        
        for btn in [self.y_minus_10_btn, self.y_minus_1_btn, self.y_plus_1_btn, self.y_plus_10_btn]:
            btn.setMaximumWidth(40)
            y_btn_layout.addWidget(btn)
        
        y_layout.addLayout(y_btn_layout)
        layout.addLayout(y_layout)
        
        parent_layout.addWidget(group)
        
    def create_rotation_controls(self, parent_layout):
        group = QGroupBox("Rotation")
        layout = QVBoxLayout(group)
        
        rot_layout = QHBoxLayout()
        rot_layout.addWidget(QLabel("Angle:"))
        
        self.rotation_spinbox = QDoubleSpinBox()
        self.rotation_spinbox.setRange(-180.0, 180.0)
        self.rotation_spinbox.setValue(0.0)
        self.rotation_spinbox.setDecimals(2)
        self.rotation_spinbox.setSingleStep(0.1)
        self.rotation_spinbox.setSuffix("°")
        rot_layout.addWidget(self.rotation_spinbox)
        
        rot_btn_layout = QHBoxLayout()
        self.rot_minus_10_btn = QPushButton("-10")
        self.rot_minus_1_btn = QPushButton("-1")
        self.rot_plus_1_btn = QPushButton("+1")
        self.rot_plus_10_btn = QPushButton("+10")
        
        for btn in [self.rot_minus_10_btn, self.rot_minus_1_btn, self.rot_plus_1_btn, self.rot_plus_10_btn]:
            btn.setMaximumWidth(40)
            rot_btn_layout.addWidget(btn)
        
        rot_layout.addLayout(rot_btn_layout)
        layout.addLayout(rot_layout)
        
        parent_layout.addWidget(group)
        
    def create_scale_controls(self, parent_layout):
        group = QGroupBox("Scale")
        layout = QVBoxLayout(group)
        
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Factor:"))
        
        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setRange(0.1, 5.0)
        self.scale_spinbox.setValue(1.0)
        self.scale_spinbox.setDecimals(3)
        self.scale_spinbox.setSingleStep(0.01)
        scale_layout.addWidget(self.scale_spinbox)
        
        scale_btn_layout = QHBoxLayout()
        self.scale_minus_1_btn = QPushButton("-1")
        self.scale_minus_01_btn = QPushButton("-0.1")
        self.scale_plus_01_btn = QPushButton("+0.1")
        self.scale_plus_1_btn = QPushButton("+1")
        
        for btn in [self.scale_minus_1_btn, self.scale_minus_01_btn, self.scale_plus_01_btn, self.scale_plus_1_btn]:
            btn.setMaximumWidth(50)
            scale_btn_layout.addWidget(btn)
        
        scale_layout.addLayout(scale_btn_layout)
        layout.addLayout(scale_layout)
        
        parent_layout.addWidget(group)
        
    def create_transparency_controls(self, parent_layout):
        group = QGroupBox("Transparency")
        layout = QVBoxLayout(group)
        
        trans_layout = QHBoxLayout()
        trans_layout.addWidget(QLabel("Alpha:"))
        
        self.transparency_spinbox = QSpinBox()
        self.transparency_spinbox.setRange(0, 100)
        self.transparency_spinbox.setValue(70)
        self.transparency_spinbox.setSuffix("%")
        trans_layout.addWidget(self.transparency_spinbox)
        
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 100)
        self.transparency_slider.setValue(70)
        layout.addWidget(self.transparency_slider)
        layout.addLayout(trans_layout)
        
        parent_layout.addWidget(group)
        
    def create_action_buttons(self, parent_layout):
        button_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.reset_btn)
        
        parent_layout.addLayout(button_layout)
        
    def create_options_controls(self, parent_layout):
        group = QGroupBox("Options")
        layout = QVBoxLayout(group)
        
        self.live_update_checkbox = QCheckBox("Live Update")
        self.live_update_checkbox.setChecked(True)
        layout.addWidget(self.live_update_checkbox)
        
        parent_layout.addWidget(group)
        
    def connect_signals(self):
        # Structure selection dropdown
        self.structure_combo.currentIndexChanged.connect(self._on_structure_selected)
        
        self.x_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.y_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.rotation_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.scale_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.transparency_spinbox.valueChanged.connect(self._on_parameter_changed)
        self.transparency_slider.valueChanged.connect(self._on_slider_changed)
        
        self.x_minus_10_btn.clicked.connect(lambda: self._adjust_x_offset(-10))
        self.x_minus_1_btn.clicked.connect(lambda: self._adjust_x_offset(-1))
        self.x_plus_1_btn.clicked.connect(lambda: self._adjust_x_offset(1))
        self.x_plus_10_btn.clicked.connect(lambda: self._adjust_x_offset(10))
        
        self.y_minus_10_btn.clicked.connect(lambda: self._adjust_y_offset(-10))
        self.y_minus_1_btn.clicked.connect(lambda: self._adjust_y_offset(-1))
        self.y_plus_1_btn.clicked.connect(lambda: self._adjust_y_offset(1))
        self.y_plus_10_btn.clicked.connect(lambda: self._adjust_y_offset(10))
        
        self.rot_minus_10_btn.clicked.connect(lambda: self._adjust_rotation(-10.0))
        self.rot_minus_1_btn.clicked.connect(lambda: self._adjust_rotation(-1.0))
        self.rot_plus_1_btn.clicked.connect(lambda: self._adjust_rotation(1.0))
        self.rot_plus_10_btn.clicked.connect(lambda: self._adjust_rotation(10.0))
        
        self.scale_minus_1_btn.clicked.connect(lambda: self._adjust_scale(-1.0))
        self.scale_minus_01_btn.clicked.connect(lambda: self._adjust_scale(-0.1))
        self.scale_plus_01_btn.clicked.connect(lambda: self._adjust_scale(0.1))
        self.scale_plus_1_btn.clicked.connect(lambda: self._adjust_scale(1.0))
        
        self.apply_btn.clicked.connect(self._emit_alignment_changed)
        self.reset_btn.clicked.connect(self.reset_parameters)
        
        self.live_update_checkbox.toggled.connect(self._on_live_update_toggled)
        
    def _adjust_x_offset(self, delta):
        current = self.x_spinbox.value()
        new_value = max(self.x_spinbox.minimum(), min(self.x_spinbox.maximum(), current + delta))
        self.x_spinbox.setValue(new_value)
        
    def _adjust_y_offset(self, delta):
        current = self.y_spinbox.value()
        new_value = max(self.y_spinbox.minimum(), min(self.y_spinbox.maximum(), current + delta))
        self.y_spinbox.setValue(new_value)
        
    def _adjust_rotation(self, delta):
        current = self.rotation_spinbox.value()
        new_value = max(self.rotation_spinbox.minimum(), min(self.rotation_spinbox.maximum(), current + delta))
        self.rotation_spinbox.setValue(new_value)
        
    def _adjust_scale(self, delta):
        current = self.scale_spinbox.value()
        new_value = max(self.scale_spinbox.minimum(), min(self.scale_spinbox.maximum(), current + delta))
        self.scale_spinbox.setValue(new_value)
        
    def _on_parameter_changed(self):
        if self.live_update_enabled:
            self.update_timer.start(100)
        
    def _on_slider_changed(self, value):
        self.transparency_spinbox.setValue(value)
        
    def _on_live_update_toggled(self, enabled):
        self.live_update_enabled = enabled
        
    def _emit_alignment_changed(self):
        parameters = self.get_parameters()
        self.alignment_changed.emit(parameters)
        
    def get_parameters(self):
        return {
            'x_offset': self.x_spinbox.value(),
            'y_offset': self.y_spinbox.value(),
            'rotation': self.rotation_spinbox.value(),
            'scale': self.scale_spinbox.value(),
            'transparency': self.transparency_spinbox.value()
        }
        
    def set_parameters(self, parameters):
        self.x_spinbox.setValue(parameters.get('x_offset', 0))
        self.y_spinbox.setValue(parameters.get('y_offset', 0))
        self.rotation_spinbox.setValue(parameters.get('rotation', 0.0))
        self.scale_spinbox.setValue(parameters.get('scale', 1.0))
        self.transparency_spinbox.setValue(parameters.get('transparency', 70))
        
        if self.live_update_enabled:
            self._emit_alignment_changed()
            
    def reset_parameters(self):
        self.x_spinbox.setValue(0)
        self.y_spinbox.setValue(0)
        self.rotation_spinbox.setValue(0.0)
        self.scale_spinbox.setValue(1.0)
        self.transparency_spinbox.setValue(70)
        self.transparency_slider.setValue(70)
        
        self.reset_requested.emit()
        
    def set_gds_overlay(self, gds_overlay):
        self.gds_overlay = gds_overlay
        
    def get_transformation_matrix(self):
        params = self.get_parameters()
        
        center_x, center_y = 512, 333
        
        scale_matrix = np.array([[params['scale'], 0, 0], [0, params['scale'], 0], [0, 0, 1]])
        
        angle_rad = np.radians(params['rotation'])
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        translate_to_origin = np.array([
            [1, 0, -center_x],
            [0, 1, -center_y],
            [0, 0, 1]
        ])
        
        translate_back = np.array([
            [1, 0, center_x + params['x_offset']],
            [0, 1, center_y + params['y_offset']],
            [0, 0, 1]
        ])
        
        combined = translate_back @ rotation_matrix @ scale_matrix @ translate_to_origin
        return combined[:2, :]
        
    def enable_controls(self, enabled=True):
        controls = [
            self.x_spinbox, self.y_spinbox, self.rotation_spinbox, 
            self.scale_spinbox, self.transparency_spinbox, self.transparency_slider,
            self.apply_btn, self.reset_btn
        ]
        
        for control in controls:
            control.setEnabled(enabled)
            
        button_groups = [
            [self.x_minus_10_btn, self.x_minus_1_btn, self.x_plus_1_btn, self.x_plus_10_btn],
            [self.y_minus_10_btn, self.y_minus_1_btn, self.y_plus_1_btn, self.y_plus_10_btn],
            [self.rot_minus_10_btn, self.rot_minus_1_btn, self.rot_plus_1_btn, self.rot_plus_10_btn],
            [self.scale_minus_1_btn, self.scale_minus_01_btn, self.scale_plus_01_btn, self.scale_plus_1_btn]
        ]
        
        for group in button_groups:
            for btn in group:
                btn.setEnabled(enabled)