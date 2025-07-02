"""
Alignment Left Panel with Manual/3-point switching tabs.

This panel provides controls for alignment operations with two modes:
- Manual alignment with sliders/spinboxes
- 3-point alignment with point selection interface

Enhanced with comprehensive error handling, input validation, and logging.
"""

import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
                               QSlider, QSpinBox, QDoubleSpinBox, QLabel, QPushButton,
                               QGroupBox, QGridLayout, QCheckBox, QComboBox, QMessageBox,
                               QScrollArea)
from PySide6.QtCore import Qt, Signal, QTimer
from src.ui.base_panels import BaseViewPanel
from src.ui.view_manager import ViewMode

# Set up logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ManualAlignmentTab(QWidget):
    """Tab for manual alignment controls with comprehensive error handling."""
    
    # Signals
    alignment_changed = Signal(dict)  # parameters
    reset_requested = Signal()
    validation_error = Signal(str)  # error message
    parameter_warning = Signal(str)  # warning message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_initializing = True
        self._validation_timer = QTimer()
        self._validation_timer.setSingleShot(True)
        self._validation_timer.timeout.connect(self._delayed_validation)
        self._last_validation_issues = []
        
        try:
            self.setup_ui()
            self.setup_tooltips()
            self.setup_styling()
            self.connect_signals()
            self._is_initializing = False
            logger.info("ManualAlignmentTab initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ManualAlignmentTab: {e}")
            logger.error(traceback.format_exc())
            raise
        
    def setup_ui(self):
        """Set up the UI with error handling for widget creation."""
        try:
            layout = QVBoxLayout(self)
            
            # Transformations Group
            transform_group = QGroupBox("Transformations")
            transform_layout = QVBoxLayout(transform_group)
            
            # Move section
            self._create_move_section(transform_layout)
            
            # Rotation section
            self._create_rotation_section(transform_layout)
            
            # Zoom section
            self._create_zoom_section(transform_layout)
            
            # Transparency section
            self._create_transparency_section(transform_layout)
            
            layout.addWidget(transform_group)
            
            # Action buttons
            self._create_action_buttons(layout)
            
            # Display options group
            self._create_display_options(layout)
            
            # Final Control buttons
            self._create_final_controls(layout)
            
            layout.addStretch()
            logger.debug("ManualAlignmentTab UI setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up ManualAlignmentTab UI: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _create_move_section(self, parent_layout):
        """Create the move controls section with error handling."""
        try:
            move_label = QLabel("Move")
            move_label.setStyleSheet("font-weight: bold;")
            parent_layout.addWidget(move_label)
            
            # Move X
            move_x_layout = QHBoxLayout()
            move_x_layout.addWidget(QLabel("Move X:"))
            
            # X control buttons
            self.x_minus_btn = self._create_increment_button("--", "Decrease X by 10 pixels")
            self.x_minus_small_btn = self._create_increment_button("-", "Decrease X by 1 pixel")
            
            self.x_offset_spin = self._create_spinbox(-500, 500, 0, "X-axis translation in pixels")
            
            self.x_plus_small_btn = self._create_increment_button("+", "Increase X by 1 pixel")
            self.x_plus_btn = self._create_increment_button("++", "Increase X by 10 pixels")
            
            for widget in [self.x_minus_btn, self.x_minus_small_btn, self.x_offset_spin, 
                          self.x_plus_small_btn, self.x_plus_btn]:
                move_x_layout.addWidget(widget)
            move_x_layout.addStretch()
            
            parent_layout.addLayout(move_x_layout)
            
            # Move Y
            move_y_layout = QHBoxLayout()
            move_y_layout.addWidget(QLabel("Move Y:"))
            
            # Y control buttons
            self.y_minus_btn = self._create_increment_button("--", "Decrease Y by 10 pixels")
            self.y_minus_small_btn = self._create_increment_button("-", "Decrease Y by 1 pixel")
            
            self.y_offset_spin = self._create_spinbox(-500, 500, 0, "Y-axis translation in pixels")
            
            self.y_plus_small_btn = self._create_increment_button("+", "Increase Y by 1 pixel")
            self.y_plus_btn = self._create_increment_button("++", "Increase Y by 10 pixels")
            
            for widget in [self.y_minus_btn, self.y_minus_small_btn, self.y_offset_spin,
                          self.y_plus_small_btn, self.y_plus_btn]:
                move_y_layout.addWidget(widget)
            move_y_layout.addStretch()
            
            parent_layout.addLayout(move_y_layout)
            logger.debug("Move section created successfully")
            
        except Exception as e:
            logger.error(f"Error creating move section: {e}")
            raise
    
    def _create_rotation_section(self, parent_layout):
        """Create the rotation controls section with error handling."""
        try:
            rotation_label = QLabel("Rotation")
            rotation_label.setStyleSheet("font-weight: bold;")
            parent_layout.addWidget(rotation_label)
            
            rotation_range_layout = QHBoxLayout()
            rotation_range_layout.addWidget(QLabel("Rotation (-90° to +90°):"))
            parent_layout.addLayout(rotation_range_layout)
            
            rotation_layout = QHBoxLayout()
            
            # Rotation control buttons
            self.rot_minus_btn = self._create_increment_button("--", "Rotate -1.0°")
            self.rot_minus_small_btn = self._create_increment_button("-", "Rotate -0.1°")
            
            self.rotation_spin = self._create_double_spinbox(-90.0, 90.0, 0.0, 0.1, "Rotation angle in degrees")
            
            self.rot_plus_small_btn = self._create_increment_button("+", "Rotate +0.1°")
            self.rot_plus_btn = self._create_increment_button("++", "Rotate +1.0°")
            
            for widget in [self.rot_minus_btn, self.rot_minus_small_btn, self.rotation_spin,
                          self.rot_plus_small_btn, self.rot_plus_btn]:
                rotation_layout.addWidget(widget)
            rotation_layout.addStretch()
            
            parent_layout.addLayout(rotation_layout)
            logger.debug("Rotation section created successfully")
            
        except Exception as e:
            logger.error(f"Error creating rotation section: {e}")
            raise
    
    def _create_zoom_section(self, parent_layout):
        """Create the zoom controls section with error handling."""
        try:
            zoom_label = QLabel("Zoom")
            zoom_label.setStyleSheet("font-weight: bold;")
            parent_layout.addWidget(zoom_label)
            
            zoom_range_layout = QHBoxLayout()
            zoom_range_layout.addWidget(QLabel("Zoom (10% to 500%):"))
            parent_layout.addLayout(zoom_range_layout)
            
            zoom_layout = QHBoxLayout()
            
            # Zoom control buttons
            self.zoom_minus_btn = self._create_increment_button("--", "Decrease zoom by 10%")
            self.zoom_minus_small_btn = self._create_increment_button("-", "Decrease zoom by 1%")
            
            self.scale_spin = self._create_spinbox(10, 500, 100, "Scale factor as percentage")
            
            self.zoom_plus_small_btn = self._create_increment_button("+", "Increase zoom by 1%")
            self.zoom_plus_btn = self._create_increment_button("++", "Increase zoom by 10%")
            
            for widget in [self.zoom_minus_btn, self.zoom_minus_small_btn, self.scale_spin,
                          self.zoom_plus_small_btn, self.zoom_plus_btn]:
                zoom_layout.addWidget(widget)
            zoom_layout.addStretch()
            
            parent_layout.addLayout(zoom_layout)
            logger.debug("Zoom section created successfully")
            
        except Exception as e:
            logger.error(f"Error creating zoom section: {e}")
            raise
    
    def _create_transparency_section(self, parent_layout):
        """Create the transparency controls section with error handling."""
        try:
            transparency_label = QLabel("Transparency")
            transparency_label.setStyleSheet("font-weight: bold;")
            parent_layout.addWidget(transparency_label)
            
            transparency_range_layout = QHBoxLayout()
            transparency_range_layout.addWidget(QLabel("Transparency (0% to 100%):"))
            parent_layout.addLayout(transparency_range_layout)
            
            transparency_layout = QHBoxLayout()
            
            # Transparency control buttons
            self.trans_minus_btn = self._create_increment_button("--", "Decrease transparency by 10%")
            self.trans_minus_small_btn = self._create_increment_button("-", "Decrease transparency by 1%")
            
            self.transparency_spin = self._create_spinbox(0, 100, 70, "Overlay transparency")
            
            self.trans_plus_small_btn = self._create_increment_button("+", "Increase transparency by 1%")
            self.trans_plus_btn = self._create_increment_button("++", "Increase transparency by 10%")
            
            for widget in [self.trans_minus_btn, self.trans_minus_small_btn, self.transparency_spin,
                          self.trans_plus_small_btn, self.trans_plus_btn]:
                transparency_layout.addWidget(widget)
            transparency_layout.addStretch()
            
            parent_layout.addLayout(transparency_layout)
            logger.debug("Transparency section created successfully")
            
        except Exception as e:
            logger.error(f"Error creating transparency section: {e}")
            raise
    
    def _create_action_buttons(self, parent_layout):
        """Create action buttons section with error handling."""
        try:
            buttons_layout = QVBoxLayout()
            
            self.reset_btn = self._create_action_button("Reset All", "Reset all transformation parameters to default values", 30)
            # Remove the Load Transformation button - we don't need it
            # Only keep Reset All button in this section
            
            buttons_layout.addWidget(self.reset_btn)
            
            parent_layout.addLayout(buttons_layout)
            logger.debug("Action buttons created successfully")
            
        except Exception as e:
            logger.error(f"Error creating action buttons: {e}")
            raise
    
    def _create_display_options(self, parent_layout):
        """Create display options section with error handling."""
        try:
            display_group = QGroupBox("Display Options")
            display_layout = QVBoxLayout(display_group)
            
            # Show overlay checkbox
            self.show_overlay_cb = QCheckBox("Show GDS Overlay")
            self.show_overlay_cb.setChecked(True)
            self.show_overlay_cb.setToolTip("Show/hide GDS overlay on SEM image")
            display_layout.addWidget(self.show_overlay_cb)
            
            # Canvas Zoom
            self._create_canvas_zoom_controls(display_layout)
            
            # Display Transparency
            self._create_display_transparency_controls(display_layout)
            
            parent_layout.addWidget(display_group)
            logger.debug("Display options created successfully")
            
        except Exception as e:
            logger.error(f"Error creating display options: {e}")
            raise
    
    def _create_canvas_zoom_controls(self, parent_layout):
        """Create canvas zoom controls with error handling."""
        try:
            canvas_zoom_layout = QHBoxLayout()
            canvas_zoom_layout.addWidget(QLabel("Canvas Zoom:"))
            
            self.canvas_zoom_spin = self._create_spinbox(10, 500, 100, "Canvas zoom level")
            self.canvas_zoom_spin.setSuffix("%")
            self.canvas_zoom_spin.setMaximumWidth(80)
            canvas_zoom_layout.addWidget(self.canvas_zoom_spin)
            
            # Canvas zoom increment buttons
            self.canvas_zoom_minus_10_btn = self._create_increment_button("-10", "Decrease canvas zoom by 10%")
            self.canvas_zoom_minus_1_btn = self._create_increment_button("-1", "Decrease canvas zoom by 1%")
            self.canvas_zoom_plus_1_btn = self._create_increment_button("+1", "Increase canvas zoom by 1%")
            self.canvas_zoom_plus_10_btn = self._create_increment_button("+10", "Increase canvas zoom by 10%")
            
            for btn in [self.canvas_zoom_minus_10_btn, self.canvas_zoom_minus_1_btn,
                       self.canvas_zoom_plus_1_btn, self.canvas_zoom_plus_10_btn]:
                canvas_zoom_layout.addWidget(btn)
            canvas_zoom_layout.addStretch()
            
            parent_layout.addLayout(canvas_zoom_layout)
            
        except Exception as e:
            logger.error(f"Error creating canvas zoom controls: {e}")
            raise
    
    def _create_display_transparency_controls(self, parent_layout):
        """Create display transparency controls with error handling."""
        try:
            display_transparency_layout = QHBoxLayout()
            display_transparency_layout.addWidget(QLabel("Display Transparency:"))
            
            self.trans_display_spin = self._create_spinbox(0, 100, 70, "Display transparency")
            self.trans_display_spin.setSuffix("%")
            self.trans_display_spin.setMaximumWidth(80)
            display_transparency_layout.addWidget(self.trans_display_spin)
            
            # Display transparency increment buttons
            self.trans_display_minus_10_btn = self._create_increment_button("-10", "Decrease transparency by 10%")
            self.trans_display_minus_1_btn = self._create_increment_button("-1", "Decrease transparency by 1%")
            self.trans_display_plus_1_btn = self._create_increment_button("+1", "Increase transparency by 1%")
            self.trans_display_plus_10_btn = self._create_increment_button("+10", "Increase transparency by 10%")
            
            for btn in [self.trans_display_minus_10_btn, self.trans_display_minus_1_btn,
                       self.trans_display_plus_1_btn, self.trans_display_plus_10_btn]:
                display_transparency_layout.addWidget(btn)
            display_transparency_layout.addStretch()
            
            parent_layout.addLayout(display_transparency_layout)
            
        except Exception as e:
            logger.error(f"Error creating display transparency controls: {e}")
            raise
    
    def _create_final_controls(self, parent_layout):
        """Create final control buttons with error handling."""
        try:
            final_button_layout = QVBoxLayout()
            
            # First row - Reset and Auto Align
            button_row1 = QHBoxLayout()
            self.final_reset_btn = self._create_action_button("Reset", "Reset all parameters to defaults")
            self.auto_align_btn = self._create_action_button("Auto Align", "Automatically align using image processing")
            button_row1.addWidget(self.final_reset_btn)
            button_row1.addWidget(self.auto_align_btn)
            final_button_layout.addLayout(button_row1)
            
            # Second row - Save Aligned GDS (prominent button) - Changed from "Generate" to "Save"
            self.save_aligned_gds_btn = self._create_save_button()
            final_button_layout.addWidget(self.save_aligned_gds_btn)
            
            parent_layout.addLayout(final_button_layout)
            logger.debug("Final controls created successfully")
            
        except Exception as e:
            logger.error(f"Error creating final controls: {e}")
            raise
    
    def _create_increment_button(self, text, tooltip):
        """Create an increment/decrement button with consistent styling and error handling."""
        try:
            button = QPushButton(text)
            button.setMaximumWidth(30)
            button.setToolTip(tooltip)
            return button
        except Exception as e:
            logger.error(f"Error creating increment button '{text}': {e}")
            raise
    
    def _create_spinbox(self, min_val, max_val, default_val, tooltip):
        """Create a spinbox with validation and error handling."""
        try:
            spinbox = QSpinBox()
            spinbox.setRange(min_val, max_val)
            spinbox.setValue(default_val)
            spinbox.setMaximumWidth(60)
            spinbox.setToolTip(f"{tooltip}\nRange: {min_val} to {max_val}")
            return spinbox
        except Exception as e:
            logger.error(f"Error creating spinbox: {e}")
            raise
    
    def _create_double_spinbox(self, min_val, max_val, default_val, step, tooltip):
        """Create a double spinbox with validation and error handling."""
        try:
            spinbox = QDoubleSpinBox()
            spinbox.setRange(min_val, max_val)
            spinbox.setSingleStep(step)
            spinbox.setValue(default_val)
            spinbox.setMaximumWidth(60)
            spinbox.setToolTip(f"{tooltip}\nRange: {min_val} to {max_val}, Step: {step}")
            return spinbox
        except Exception as e:
            logger.error(f"Error creating double spinbox: {e}")
            raise
    
    def _create_action_button(self, text, tooltip, min_height=None):
        """Create an action button with consistent styling and error handling."""
        try:
            button = QPushButton(text)
            if min_height:
                button.setMinimumHeight(min_height)
            button.setToolTip(tooltip)
            return button
        except Exception as e:
            logger.error(f"Error creating action button '{text}': {e}")
            raise
    
    def _create_save_button(self):
        """Create the save aligned GDS button with special styling."""
        try:
            button = QPushButton("Save Aligned GDS")
            button.setToolTip("Save the aligned GDS to file")
            button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    padding: 8px;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                    color: #666666;
                }
            """)
            button.setEnabled(False)  # Enabled when alignment is ready
            return button
        except Exception as e:
            logger.error(f"Error creating save button: {e}")
            raise
        
    def connect_signals(self):
        """Connect slider/spinbox signals and increment buttons with error handling."""
        try:
            # X increment buttons
            self.x_minus_btn.clicked.connect(lambda: self.adjust_value_safe(self.x_offset_spin, -10))
            self.x_minus_small_btn.clicked.connect(lambda: self.adjust_value_safe(self.x_offset_spin, -1))
            self.x_plus_small_btn.clicked.connect(lambda: self.adjust_value_safe(self.x_offset_spin, 1))
            self.x_plus_btn.clicked.connect(lambda: self.adjust_value_safe(self.x_offset_spin, 10))
            
            # Y increment buttons  
            self.y_minus_btn.clicked.connect(lambda: self.adjust_value_safe(self.y_offset_spin, -10))
            self.y_minus_small_btn.clicked.connect(lambda: self.adjust_value_safe(self.y_offset_spin, -1))
            self.y_plus_small_btn.clicked.connect(lambda: self.adjust_value_safe(self.y_offset_spin, 1))
            self.y_plus_btn.clicked.connect(lambda: self.adjust_value_safe(self.y_offset_spin, 10))
            
            # Rotation increment buttons
            self.rot_minus_btn.clicked.connect(lambda: self.adjust_value_safe(self.rotation_spin, -1.0))
            self.rot_minus_small_btn.clicked.connect(lambda: self.adjust_value_safe(self.rotation_spin, -0.1))
            self.rot_plus_small_btn.clicked.connect(lambda: self.adjust_value_safe(self.rotation_spin, 0.1))
            self.rot_plus_btn.clicked.connect(lambda: self.adjust_value_safe(self.rotation_spin, 1.0))
            
            # Scale increment buttons
            self.zoom_minus_btn.clicked.connect(lambda: self.adjust_value_safe(self.scale_spin, -10))
            self.zoom_minus_small_btn.clicked.connect(lambda: self.adjust_value_safe(self.scale_spin, -1))
            self.zoom_plus_small_btn.clicked.connect(lambda: self.adjust_value_safe(self.scale_spin, 1))
            self.zoom_plus_btn.clicked.connect(lambda: self.adjust_value_safe(self.scale_spin, 10))
            
            # Transparency increment buttons
            self.trans_minus_btn.clicked.connect(lambda: self.adjust_value_safe(self.transparency_spin, -10))
            self.trans_minus_small_btn.clicked.connect(lambda: self.adjust_value_safe(self.transparency_spin, -1))
            self.trans_plus_small_btn.clicked.connect(lambda: self.adjust_value_safe(self.transparency_spin, 1))
            self.trans_plus_btn.clicked.connect(lambda: self.adjust_value_safe(self.transparency_spin, 10))
            
            # Canvas zoom increment buttons
            self.canvas_zoom_minus_10_btn.clicked.connect(lambda: self.adjust_value_safe(self.canvas_zoom_spin, -10))
            self.canvas_zoom_minus_1_btn.clicked.connect(lambda: self.adjust_value_safe(self.canvas_zoom_spin, -1))
            self.canvas_zoom_plus_1_btn.clicked.connect(lambda: self.adjust_value_safe(self.canvas_zoom_spin, 1))
            self.canvas_zoom_plus_10_btn.clicked.connect(lambda: self.adjust_value_safe(self.canvas_zoom_spin, 10))
            
            # Display transparency increment buttons
            self.trans_display_minus_10_btn.clicked.connect(lambda: self.adjust_value_safe(self.trans_display_spin, -10))
            self.trans_display_minus_1_btn.clicked.connect(lambda: self.adjust_value_safe(self.trans_display_spin, -1))
            self.trans_display_plus_1_btn.clicked.connect(lambda: self.adjust_value_safe(self.trans_display_spin, 1))
            self.trans_display_plus_10_btn.clicked.connect(lambda: self.adjust_value_safe(self.trans_display_spin, 10))
            
            # Connect value changes to emit alignment_changed with validation
            self._connect_value_change_signals()
            
            # Buttons - connect both reset buttons to the same signal
            self.reset_btn.clicked.connect(self.reset_requested)
            self.final_reset_btn.clicked.connect(self.reset_requested)
            
            logger.debug("Signal connections established successfully")
            
        except Exception as e:
            logger.error(f"Error connecting signals: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _connect_value_change_signals(self):
        """Connect value change signals with validation and error handling."""
        try:
            # Connect value changes to emit alignment_changed with validation
            controls = [self.x_offset_spin, self.y_offset_spin, self.rotation_spin, 
                       self.scale_spin, self.transparency_spin, self.canvas_zoom_spin, 
                       self.trans_display_spin]
            
            for control in controls:
                if hasattr(control, 'valueChanged'):
                    control.valueChanged.connect(self._handle_value_change)
                else:
                    logger.warning(f"Control {control} does not have valueChanged signal")
            
            self.show_overlay_cb.toggled.connect(self._handle_value_change)
            
        except Exception as e:
            logger.error(f"Error connecting value change signals: {e}")
            raise
    
    def _handle_value_change(self):
        """Handle value changes with validation and delayed emission."""
        if self._is_initializing:
            return
            
        try:
            # Start validation timer for delayed validation
            self._validation_timer.start(100)  # 100ms delay
        except Exception as e:
            logger.error(f"Error handling value change: {e}")
    
    def _delayed_validation(self):
        """Perform delayed validation and emit alignment changed signal."""
        try:
            # Validate current parameters
            params = self._get_current_parameters()
            validation_issues = self._validate_parameters(params)
            
            # Store validation issues for comparison
            if validation_issues != self._last_validation_issues:
                self._last_validation_issues = validation_issues.copy()
                
                # Emit warnings for significant issues
                for issue in validation_issues:
                    if issue['severity'] == 'warning':
                        self.parameter_warning.emit(issue['message'])
                        logger.warning(f"Parameter warning: {issue['message']}")
                    elif issue['severity'] == 'error':
                        self.validation_error.emit(issue['message'])
                        logger.error(f"Parameter error: {issue['message']}")
            
            # Emit alignment changed signal even with warnings (but not with errors)
            has_errors = any(issue['severity'] == 'error' for issue in validation_issues)
            if not has_errors:
                self.emit_alignment_changed()
            else:
                logger.warning("Not emitting alignment_changed due to validation errors")
                
        except Exception as e:
            logger.error(f"Error in delayed validation: {e}")
            logger.error(traceback.format_exc())
    
    def _get_current_parameters(self):
        """Get current parameters with error handling."""
        try:
            return {
                'translation_x_pixels': self.x_offset_spin.value(),
                'translation_y_pixels': self.y_offset_spin.value(),
                'rotation_degrees': self.rotation_spin.value(),
                'scale': self.scale_spin.value() / 100.0,
                'transparency': self.transparency_spin.value(),
                'display_transparency': self.trans_display_spin.value(),
                'canvas_zoom': self.canvas_zoom_spin.value(),
                'show_overlay': self.show_overlay_cb.isChecked()
            }
        except Exception as e:
            logger.error(f"Error getting current parameters: {e}")
            return {}
    
    def _validate_parameters(self, params):
        """Validate parameters and return list of issues."""
        issues = []
        
        try:
            # Translation validation
            if abs(params.get('translation_x_pixels', 0)) > 400:
                issues.append({
                    'parameter': 'translation_x_pixels',
                    'severity': 'warning',
                    'message': f"Large X translation ({params['translation_x_pixels']} pixels) may indicate misalignment"
                })
            
            if abs(params.get('translation_y_pixels', 0)) > 400:
                issues.append({
                    'parameter': 'translation_y_pixels',
                    'severity': 'warning',
                    'message': f"Large Y translation ({params['translation_y_pixels']} pixels) may indicate misalignment"
                })
            
            # Rotation validation
            rotation = params.get('rotation_degrees', 0)
            if abs(rotation) > 45:
                issues.append({
                    'parameter': 'rotation_degrees',
                    'severity': 'warning',
                    'message': f"Large rotation ({rotation:.1f}°) may indicate misalignment"
                })
            
            # Scale validation
            scale = params.get('scale', 1.0)
            if scale < 0.3 or scale > 3.0:
                issues.append({
                    'parameter': 'scale',
                    'severity': 'error',
                    'message': f"Extreme scale factor ({scale:.2f}) is not reasonable"
                })
            elif scale < 0.5 or scale > 2.0:
                issues.append({
                    'parameter': 'scale',
                    'severity': 'warning',
                    'message': f"Large scale change ({scale:.2f}) may indicate misalignment"
                })
            
            # Transparency validation
            transparency = params.get('transparency', 70)
            if transparency < 10:
                issues.append({
                    'parameter': 'transparency',
                    'severity': 'warning',
                    'message': "Very low transparency may make overlay hard to see"
                })
            elif transparency > 95:
                issues.append({
                    'parameter': 'transparency',
                    'severity': 'warning',
                    'message': "Very high transparency may make overlay invisible"
                })
            
        except Exception as e:
            logger.error(f"Error validating parameters: {e}")
            issues.append({
                'parameter': 'validation',
                'severity': 'error',
                'message': f"Parameter validation failed: {e}"
            })
        
        return issues
        
    def setup_tooltips(self):
        """Set up helpful tooltips for all controls."""
        # Movement controls
        self.x_offset_spin.setToolTip("X-axis translation in pixels\nRange: -500 to +500")
        self.y_offset_spin.setToolTip("Y-axis translation in pixels\nRange: -500 to +500")
        
        # Rotation controls
        self.rotation_spin.setToolTip("Rotation angle in degrees\nRange: -90° to +90°\nStep: 0.1°")
        self.rot_minus_btn.setToolTip("Rotate -1.0°")
        self.rot_minus_small_btn.setToolTip("Rotate -0.1°")
        self.rot_plus_small_btn.setToolTip("Rotate +0.1°")
        self.rot_plus_btn.setToolTip("Rotate +1.0°")
        
        # Scale controls
        self.scale_spin.setToolTip("Scale factor as percentage\nRange: 10% to 500%")
        self.zoom_minus_btn.setToolTip("Decrease scale by 10%")
        self.zoom_minus_small_btn.setToolTip("Decrease scale by 1%")
        self.zoom_plus_small_btn.setToolTip("Increase scale by 1%")
        self.zoom_plus_btn.setToolTip("Increase scale by 10%")
        
        # Transparency controls
        self.transparency_spin.setToolTip("Overlay transparency\nRange: 0% (opaque) to 100% (transparent)")
        self.trans_display_spin.setToolTip("Display transparency\nRange: 0% (opaque) to 100% (transparent)")
        
        # Canvas zoom
        self.canvas_zoom_spin.setToolTip("Canvas zoom level\nRange: 10% to 500%")
        
        # Action buttons
        self.reset_btn.setToolTip("Reset all transformation parameters to default values")
        self.save_aligned_gds_btn.setToolTip("Save the aligned GDS as PNG image")
        self.auto_align_btn.setToolTip("Automatically align using image processing")
        
        # Display options
        self.show_overlay_cb.setToolTip("Show/hide GDS overlay on SEM image")
        
    def setup_styling(self):
        """Set up consistent styling for all controls."""
        # Define color scheme
        primary_color = "#2C3E50"
        secondary_color = "#3498DB"
        success_color = "#27AE60"
        warning_color = "#F39C12"
        danger_color = "#E74C3C"
        light_bg = "#ECF0F1"
        dark_text = "#2C3E50"
        
        # Style the main groups
        group_style = f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {primary_color};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: {light_bg};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: {primary_color};
                background-color: {light_bg};
            }}
        """
        
        # Apply styling to all group boxes
        for group in self.findChildren(QGroupBox):
            group.setStyleSheet(group_style)
        
        # Style increment/decrement buttons
        button_style = f"""
            QPushButton {{
                background-color: {secondary_color};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px;
                font-weight: bold;
                min-width: 25px;
            }}
            QPushButton:hover {{
                background-color: #2980B9;
            }}
            QPushButton:pressed {{
                background-color: #21618C;
            }}
            QPushButton:disabled {{
                background-color: #BDC3C7;
                color: #7F8C8D;
            }}
        """
        
        # Apply to increment/decrement buttons
        increment_buttons = [
            self.x_minus_btn, self.x_minus_small_btn, self.x_plus_small_btn, self.x_plus_btn,
            self.y_minus_btn, self.y_minus_small_btn, self.y_plus_small_btn, self.y_plus_btn,
            self.rot_minus_btn, self.rot_minus_small_btn, self.rot_plus_small_btn, self.rot_plus_btn,
            self.zoom_minus_btn, self.zoom_minus_small_btn, self.zoom_plus_small_btn, self.zoom_plus_btn,
            self.trans_minus_btn, self.trans_minus_small_btn, self.trans_plus_small_btn, self.trans_plus_btn,
            self.canvas_zoom_minus_10_btn, self.canvas_zoom_minus_1_btn, 
            self.canvas_zoom_plus_1_btn, self.canvas_zoom_plus_10_btn,
            self.trans_display_minus_10_btn, self.trans_display_minus_1_btn,
            self.trans_display_plus_1_btn, self.trans_display_plus_10_btn
        ]
        
        for btn in increment_buttons:
            btn.setStyleSheet(button_style)
        
        # Style action buttons
        action_button_style = f"""
            QPushButton {{
                background-color: {primary_color};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: #34495E;
            }}
            QPushButton:pressed {{
                background-color: #1B2631;
            }}
            QPushButton:disabled {{
                background-color: #BDC3C7;
                color: #7F8C8D;
            }}
        """
        
        action_buttons = [self.reset_btn, self.final_reset_btn, self.auto_align_btn]
        for btn in action_buttons:
            btn.setStyleSheet(action_button_style)
        
        # Style spinboxes
        spinbox_style = f"""
            QSpinBox, QDoubleSpinBox {{
                border: 2px solid {secondary_color};
                border-radius: 4px;
                padding: 4px;
                background-color: white;
                color: {dark_text};
                font-weight: bold;
            }}
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border-color: {success_color};
            }}
            QSpinBox:disabled, QDoubleSpinBox:disabled {{
                background-color: #F8F9FA;
                color: #6C757D;
                border-color: #DEE2E6;
            }}
        """
        
        # Apply to all spinboxes
        for spinbox in self.findChildren((QSpinBox, QDoubleSpinBox)):
            spinbox.setStyleSheet(spinbox_style)
        
        # Style checkboxes
        checkbox_style = f"""
            QCheckBox {{
                color: {dark_text};
                font-weight: bold;
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {secondary_color};
                border-radius: 4px;
                background-color: white;
            }}
            QCheckBox::indicator:checked {{
                background-color: {success_color};
                border-color: {success_color};
            }}
            QCheckBox::indicator:hover {{
                border-color: {success_color};
            }}
        """
        
        self.show_overlay_cb.setStyleSheet(checkbox_style)
        
    def update_button_states(self, has_images=False, alignment_ready=False):
        """Update button enabled states based on current conditions."""
        # Enable/disable buttons based on whether images are loaded
        # Auto align button needs images to work
        self.auto_align_btn.setEnabled(has_images)
            
        # Enable save button only when alignment is ready
        self.save_aligned_gds_btn.setEnabled(alignment_ready)
        
        # Always enable reset buttons
        self.reset_btn.setEnabled(True)
        self.final_reset_btn.setEnabled(True)
        
    def adjust_value_safe(self, spinbox, delta):
        """Safely adjust a spinbox value with comprehensive error handling."""
        try:
            self.adjust_value(spinbox, delta)
        except Exception as e:
            logger.error(f"Error in adjust_value_safe: {e}")
            self.validation_error.emit(f"Failed to adjust value: {e}")
            # Show user-friendly error message
            self._show_error_message("Value Adjustment Error", f"Failed to adjust value: {e}")
    
    def adjust_value(self, spinbox, delta):
        """Adjust a spinbox value by the given delta, with comprehensive validation."""
        if not spinbox:
            raise ValueError("Spinbox is None")
            
        try:
            current_value = spinbox.value()
            new_value = current_value + delta
            
            # Get limits
            min_val = spinbox.minimum()
            max_val = spinbox.maximum()
            
            # Validate the new value
            if new_value < min_val:
                clamped_value = min_val
                logger.warning(f"Value adjustment reached minimum limit: {min_val}")
                self.parameter_warning.emit(f"Reached minimum value: {min_val}")
            elif new_value > max_val:
                clamped_value = max_val
                logger.warning(f"Value adjustment reached maximum limit: {max_val}")
                self.parameter_warning.emit(f"Reached maximum value: {max_val}")
            else:
                clamped_value = new_value
            
            # Only update if value actually changed
            if clamped_value != current_value:
                spinbox.setValue(clamped_value)
                logger.debug(f"Adjusted {spinbox.objectName() or 'spinbox'} from {current_value} to {clamped_value} (delta: {delta})")
            else:
                logger.debug(f"No change needed for {spinbox.objectName() or 'spinbox'}: already at {current_value}")
            
        except Exception as e:
            logger.error(f"Error adjusting spinbox value: {e}")
            raise
        
    def _show_error_message(self, title, message):
        """Show an error message to the user."""
        try:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()
        except Exception as e:
            logger.error(f"Error showing error message: {e}")
            # Fallback to console output
            print(f"ERROR - {title}: {message}")
    
    def _show_warning_message(self, title, message):
        """Show a warning message to the user."""
        try:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()
        except Exception as e:
            logger.error(f"Error showing warning message: {e}")
            print(f"WARNING - {title}: {message}")
        
    def emit_alignment_changed(self):
        """Emit alignment changed signal with current parameters and validation."""
        try:
            params = self._get_current_parameters()
            
            if not params:
                logger.error("Failed to get current parameters, not emitting alignment_changed")
                return
            
            # Validate parameters before emitting
            validation_issues = self._validate_parameters(params)
            has_errors = any(issue['severity'] == 'error' for issue in validation_issues)
            
            if has_errors:
                error_messages = [issue['message'] for issue in validation_issues if issue['severity'] == 'error']
                logger.error(f"Cannot emit alignment_changed due to validation errors: {error_messages}")
                return
            
            # Add validation metadata to parameters
            params['_validation'] = {
                'validated': True,
                'issues': validation_issues,
                'timestamp': self._get_timestamp()
            }
            
            logger.debug(f"ManualAlignmentTab emitting alignment_changed: {params}")
            self.alignment_changed.emit(params)
            
        except Exception as e:
            logger.error(f"Error emitting alignment_changed signal: {e}")
            logger.error(traceback.format_exc())
            self.validation_error.emit(f"Failed to emit alignment change: {e}")
    
    def _get_timestamp(self):
        """Get current timestamp for logging."""
        try:
            from datetime import datetime
            return datetime.now().isoformat()
        except Exception:
            return "unknown"
        
    def set_parameters(self, params):
        """Set alignment parameters from dictionary with comprehensive validation."""
        if not params:
            logger.warning("Empty parameters dictionary provided to set_parameters")
            return
            
        try:
            # Block signals during parameter setting to avoid recursive calls
            self._block_signals(True)
            
            # Track successful parameter sets for rollback on error
            original_params = self._get_current_parameters()
            set_params = {}
            
            try:
                # Translation X
                if 'translation_x_pixels' in params:
                    value = self._validate_and_convert_value(params['translation_x_pixels'], int, 'translation_x_pixels')
                    self.x_offset_spin.setValue(value)
                    set_params['translation_x_pixels'] = value
                elif 'x_offset' in params:
                    value = self._validate_and_convert_value(params['x_offset'], int, 'x_offset')
                    self.x_offset_spin.setValue(value)
                    set_params['x_offset'] = value
                
                # Translation Y
                if 'translation_y_pixels' in params:
                    value = self._validate_and_convert_value(params['translation_y_pixels'], int, 'translation_y_pixels')
                    self.y_offset_spin.setValue(value)
                    set_params['translation_y_pixels'] = value
                elif 'y_offset' in params:
                    value = self._validate_and_convert_value(params['y_offset'], int, 'y_offset')
                    self.y_offset_spin.setValue(value)
                    set_params['y_offset'] = value
                
                # Rotation
                if 'rotation_degrees' in params:
                    value = self._validate_and_convert_value(params['rotation_degrees'], float, 'rotation_degrees')
                    self.rotation_spin.setValue(value)
                    set_params['rotation_degrees'] = value
                elif 'rotation' in params:
                    value = self._validate_and_convert_value(params['rotation'], float, 'rotation')
                    self.rotation_spin.setValue(value)
                    set_params['rotation'] = value
                
                # Scale
                if 'scale' in params:
                    value = self._validate_and_convert_value(params['scale'], float, 'scale')
                    # Convert scale factor to percentage for display
                    percentage = value * 100.0
                    if 10 <= percentage <= 500:
                        self.scale_spin.setValue(int(percentage))
                        set_params['scale'] = value
                    else:
                        logger.warning(f"Scale percentage {percentage} out of range [10, 500], skipping")
                
                # Transparency
                if 'transparency' in params:
                    value = self._validate_and_convert_value(params['transparency'], int, 'transparency')
                    if 0 <= value <= 100:
                        self.transparency_spin.setValue(value)
                        set_params['transparency'] = value
                    else:
                        logger.warning(f"Transparency {value} out of range [0, 100], skipping")
                
                # Show overlay
                if 'show_overlay' in params:
                    value = bool(params['show_overlay'])
                    self.show_overlay_cb.setChecked(value)
                    set_params['show_overlay'] = value
                
                logger.info(f"Successfully set parameters: {set_params}")
                
            except Exception as e:
                logger.error(f"Error setting parameters, rolling back: {e}")
                # Rollback to original parameters
                self._rollback_parameters(original_params)
                raise
            
        except Exception as e:
            logger.error(f"Failed to set parameters: {e}")
            logger.error(traceback.format_exc())
            self.validation_error.emit(f"Failed to set parameters: {e}")
            raise
        finally:
            # Always re-enable signals
            self._block_signals(False)
    
    def _validate_and_convert_value(self, value, target_type, param_name):
        """Validate and convert a parameter value to the target type."""
        try:
            if value is None:
                raise ValueError(f"Parameter {param_name} is None")
            
            converted_value = target_type(value)
            
            # Additional validation based on parameter name
            if param_name in ['translation_x_pixels', 'x_offset']:
                if abs(converted_value) > 1000:
                    logger.warning(f"Large X translation value: {converted_value}")
            elif param_name in ['translation_y_pixels', 'y_offset']:
                if abs(converted_value) > 1000:
                    logger.warning(f"Large Y translation value: {converted_value}")
            elif param_name in ['rotation_degrees', 'rotation']:
                if abs(converted_value) > 180:
                    logger.warning(f"Large rotation value: {converted_value}")
            elif param_name == 'scale':
                if converted_value <= 0:
                    raise ValueError(f"Scale must be positive, got {converted_value}")
                if converted_value < 0.1 or converted_value > 10:
                    logger.warning(f"Extreme scale value: {converted_value}")
            
            return converted_value
            
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert {param_name} value {value} to {target_type.__name__}: {e}")
            raise ValueError(f"Invalid {param_name} value: {value}")
    
    def _block_signals(self, block):
        """Block or unblock signals from all controls."""
        try:
            controls = [
                self.x_offset_spin, self.y_offset_spin, self.rotation_spin,
                self.scale_spin, self.transparency_spin, self.canvas_zoom_spin,
                self.trans_display_spin, self.show_overlay_cb
            ]
            
            for control in controls:
                if control:
                    control.blockSignals(block)
                    
        except Exception as e:
            logger.error(f"Error blocking/unblocking signals: {e}")
    
    def _rollback_parameters(self, original_params):
        """Rollback to original parameters in case of error."""
        try:
            logger.info("Rolling back to original parameters")
            self.set_parameters(original_params)
        except Exception as e:
            logger.error(f"Failed to rollback parameters: {e}")
            # Last resort: reset to defaults
            try:
                self.reset_parameters()
            except Exception as reset_error:
                logger.critical(f"Failed to reset parameters during rollback: {reset_error}")

    def reset_parameters(self):
        """Reset all parameters to default values with comprehensive error handling."""
        try:
            logger.info("Resetting all alignment parameters to defaults...")
            
            # Define default values
            defaults = {
                'x_offset': 0,
                'y_offset': 0,
                'rotation': 0.0,
                'scale_percentage': 100,
                'transparency': 70,
                'canvas_zoom': 100,
                'display_transparency': 70,
                'show_overlay': True
            }
            
            # Block signals temporarily to avoid multiple emission during reset
            self._block_signals(True)
            
            try:
                # Reset each control with error handling for individual controls
                reset_errors = []
                
                try:
                    self.x_offset_spin.setValue(defaults['x_offset'])
                except Exception as e:
                    reset_errors.append(f"x_offset: {e}")
                    
                try:
                    self.y_offset_spin.setValue(defaults['y_offset'])
                except Exception as e:
                    reset_errors.append(f"y_offset: {e}")
                    
                try:
                    self.rotation_spin.setValue(defaults['rotation'])
                except Exception as e:
                    reset_errors.append(f"rotation: {e}")
                    
                try:
                    self.scale_spin.setValue(defaults['scale_percentage'])
                except Exception as e:
                    reset_errors.append(f"scale: {e}")
                    
                try:
                    self.transparency_spin.setValue(defaults['transparency'])
                except Exception as e:
                    reset_errors.append(f"transparency: {e}")
                    
                try:
                    self.canvas_zoom_spin.setValue(defaults['canvas_zoom'])
                except Exception as e:
                    reset_errors.append(f"canvas_zoom: {e}")
                    
                try:
                    self.trans_display_spin.setValue(defaults['display_transparency'])
                except Exception as e:
                    reset_errors.append(f"display_transparency: {e}")
                    
                try:
                    self.show_overlay_cb.setChecked(defaults['show_overlay'])
                except Exception as e:
                    reset_errors.append(f"show_overlay: {e}")
                
                if reset_errors:
                    error_msg = f"Some controls failed to reset: {'; '.join(reset_errors)}"
                    logger.error(error_msg)
                    self.validation_error.emit(error_msg)
                else:
                    logger.info("All parameters successfully reset to default values")
                
            finally:
                # Re-enable signals
                self._block_signals(False)
            
            # Emit alignment changed signal once after all resets
            try:
                self.emit_alignment_changed()
            except Exception as e:
                logger.error(f"Failed to emit alignment_changed after reset: {e}")
                
        except Exception as e:
            logger.error(f"Critical error during parameter reset: {e}")
            logger.error(traceback.format_exc())
            self.validation_error.emit(f"Failed to reset parameters: {e}")
            raise
        
    def get_parameters(self):
        """Get current parameter values in frame-based format."""
        return {
            'translation_x_pixels': self.x_offset_spin.value(),
            'translation_y_pixels': self.y_offset_spin.value(),
            'rotation_degrees': self.rotation_spin.value(),
            'scale': self.scale_spin.value() / 100.0,  # Convert percentage to scale factor
            'transparency': self.transparency_spin.value(),
            'canvas_zoom': self.canvas_zoom_spin.value(),
            'display_transparency': self.trans_display_spin.value(),
            'show_overlay': self.show_overlay_cb.isChecked()
        }
        
    def set_parameters_from_model(self, aligned_gds_model):
        """Set parameters from AlignedGdsModel's UI format with comprehensive error handling."""
        if not aligned_gds_model:
            logger.warning("No aligned_gds_model provided to set_parameters_from_model")
            return
            
        try:
            # Get UI parameters from model with error handling
            try:
                ui_params = aligned_gds_model.get_ui_parameters()
                if not ui_params:
                    logger.warning("Model returned empty UI parameters")
                    return
            except Exception as e:
                logger.error(f"Failed to get UI parameters from model: {e}")
                self.validation_error.emit(f"Failed to get parameters from model: {e}")
                return
            
            logger.debug(f"Setting parameters from model: {ui_params}")
            
            # Block signals to avoid triggering updates during parameter setting
            self._block_signals(True)
            
            try:
                # Set values with individual error handling and validation
                self._set_model_parameter(self.x_offset_spin, ui_params, 'translation_x_pixels', int, 0)
                self._set_model_parameter(self.y_offset_spin, ui_params, 'translation_y_pixels', int, 0)
                self._set_model_parameter(self.rotation_spin, ui_params, 'rotation_degrees', float, 0.0)
                
                # Handle scale conversion (model uses scale factor, UI uses percentage)
                scale_factor = ui_params.get('scale', 1.0)
                try:
                    scale_percentage = scale_factor * 100.0
                    if 10 <= scale_percentage <= 500:
                        self.scale_spin.setValue(int(scale_percentage))
                        logger.debug(f"Set scale to {scale_percentage}% (factor: {scale_factor})")
                    else:
                        logger.warning(f"Scale percentage {scale_percentage} out of range, using default")
                        self.scale_spin.setValue(100)
                except Exception as e:
                    logger.error(f"Error setting scale parameter: {e}")
                    self.scale_spin.setValue(100)
                
                logger.info("Successfully set parameters from model")
                
            except Exception as e:
                logger.error(f"Error setting individual parameters from model: {e}")
                # Don't raise here, just log the error
            finally:
                # Always re-enable signals
                self._block_signals(False)
                
        except Exception as e:
            logger.error(f"Failed to set parameters from model: {e}")
            logger.error(traceback.format_exc())
            self.validation_error.emit(f"Model parameter setting failed: {e}")
    
    def _set_model_parameter(self, control, params_dict, param_name, param_type, default_value):
        """Safely set a parameter from the model with type conversion and validation."""
        try:
            if param_name in params_dict:
                value = params_dict[param_name]
                converted_value = param_type(value)
                control.setValue(converted_value)
                logger.debug(f"Set {param_name} to {converted_value}")
            else:
                logger.debug(f"Parameter {param_name} not found in model, using default {default_value}")
                control.setValue(default_value)
        except Exception as e:
            logger.error(f"Error setting {param_name}: {e}")
            control.setValue(default_value)


class ThreePointAlignmentTab(QWidget):
    """Tab for 3-point alignment controls with comprehensive error handling."""
    
    # Signals
    three_points_selected = Signal(list, list)  # sem_points, gds_points
    transformation_calculated = Signal(dict)
    transformation_confirmed = Signal(dict)
    point_selection_mode_changed = Signal(str)  # "sem" or "gds"
    validation_error = Signal(str)  # error message
    point_validation_warning = Signal(str)  # warning message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sem_points = []
        self.gds_points = []
        self.current_mode = "sem"
        self._last_calculated_transformation = None
        
        try:
            self.setup_ui()
            self.setup_tooltips()
            self.connect_signals()
            logger.info("ThreePointAlignmentTab initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ThreePointAlignmentTab: {e}")
            logger.error(traceback.format_exc())
            raise
        
    def setup_ui(self):
        """Set up the UI with comprehensive error handling."""
        try:
            layout = QVBoxLayout(self)
            
            # Instructions
            instructions = QLabel(
                "3-Point Alignment:\n"
                "1. Select 3 points on SEM image\n"
                "2. Select corresponding 3 points on GDS\n"
                "3. Calculate transformation\n"
                "4. Confirm alignment"
            )
            instructions.setWordWrap(True)
            instructions.setStyleSheet("font-weight: bold; color: #2C3E50; padding: 10px; background-color: #ECF0F1; border-radius: 5px;")
            layout.addWidget(instructions)
            
            # Point selection mode
            self._create_mode_selection(layout)
            
            # Point status
            self._create_point_status(layout)
            
            # Control buttons
            self._create_control_buttons(layout)
            
            logger.debug("ThreePointAlignmentTab UI setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up ThreePointAlignmentTab UI: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _create_mode_selection(self, parent_layout):
        """Create point selection mode controls with error handling."""
        try:
            mode_group = QGroupBox("Point Selection Mode")
            mode_layout = QVBoxLayout(mode_group)
            
            self.sem_mode_btn = QPushButton("Select SEM Points")
            self.sem_mode_btn.setCheckable(True)
            self.sem_mode_btn.setChecked(True)
            self.sem_mode_btn.setToolTip("Switch to SEM point selection mode\nClick 3 points on the SEM image")
            
            self.gds_mode_btn = QPushButton("Select GDS Points")
            self.gds_mode_btn.setCheckable(True)
            self.gds_mode_btn.setToolTip("Switch to GDS point selection mode\nClick 3 corresponding points on the GDS overlay")
            
            mode_layout.addWidget(self.sem_mode_btn)
            mode_layout.addWidget(self.gds_mode_btn)
            parent_layout.addWidget(mode_group)
            
        except Exception as e:
            logger.error(f"Error creating mode selection: {e}")
            raise
    
    def _create_point_status(self, parent_layout):
        """Create point status display with error handling."""
        try:
            status_group = QGroupBox("Point Status")
            status_layout = QGridLayout(status_group)
            
            status_layout.addWidget(QLabel("SEM Points:"), 0, 0)
            self.sem_status_label = QLabel("0/3")
            self.sem_status_label.setStyleSheet("font-weight: bold; color: #E74C3C;")
            status_layout.addWidget(self.sem_status_label, 0, 1)
            
            status_layout.addWidget(QLabel("GDS Points:"), 1, 0)
            self.gds_status_label = QLabel("0/3")
            self.gds_status_label.setStyleSheet("font-weight: bold; color: #E74C3C;")
            status_layout.addWidget(self.gds_status_label, 1, 1)
            
            # Add validation status label
            status_layout.addWidget(QLabel("Validation:"), 2, 0)
            self.validation_status_label = QLabel("Not ready")
            self.validation_status_label.setStyleSheet("font-weight: bold; color: #F39C12;")
            status_layout.addWidget(self.validation_status_label, 2, 1)
            
            parent_layout.addWidget(status_group)
            
        except Exception as e:
            logger.error(f"Error creating point status: {e}")
            raise
    
    def _create_control_buttons(self, parent_layout):
        """Create control buttons with error handling."""
        try:
            button_layout = QVBoxLayout()
            
            self.clear_points_btn = QPushButton("Clear All Points")
            self.clear_points_btn.setToolTip("Clear all selected points and start over")
            
            self.calculate_btn = QPushButton("Calculate Transformation")
            self.calculate_btn.setEnabled(False)
            self.calculate_btn.setToolTip("Calculate transformation matrix from selected points\nRequires 3 SEM and 3 GDS points")
            
            self.confirm_btn = QPushButton("Confirm Alignment")
            self.confirm_btn.setEnabled(False)
            self.confirm_btn.setToolTip("Confirm and apply the calculated transformation")
            
            # Style the buttons
            button_style = """
                QPushButton {
                    padding: 8px;
                    font-weight: bold;
                    border-radius: 4px;
                    min-height: 25px;
                }
                QPushButton:enabled {
                    background-color: #3498DB;
                    color: white;
                    border: none;
                }
                QPushButton:disabled {
                    background-color: #BDC3C7;
                    color: #7F8C8D;
                    border: 1px solid #95A5A6;
                }
                QPushButton:hover:enabled {
                    background-color: #2980B9;
                }
            """
            
            for btn in [self.clear_points_btn, self.calculate_btn, self.confirm_btn]:
                btn.setStyleSheet(button_style)
                button_layout.addWidget(btn)
            
            parent_layout.addLayout(button_layout)
            
        except Exception as e:
            logger.error(f"Error creating control buttons: {e}")
            raise
        
    def connect_signals(self):
        """Connect button signals with error handling."""
        try:
            self.sem_mode_btn.clicked.connect(lambda: self._safe_set_selection_mode("sem"))
            self.gds_mode_btn.clicked.connect(lambda: self._safe_set_selection_mode("gds"))
            self.clear_points_btn.clicked.connect(self._safe_clear_all_points)
            self.calculate_btn.clicked.connect(self._safe_calculate_transformation)
            self.confirm_btn.clicked.connect(self._safe_confirm_transformation)
            
            logger.debug("ThreePointAlignmentTab signal connections established")
            
        except Exception as e:
            logger.error(f"Error connecting ThreePointAlignmentTab signals: {e}")
            raise
    
    def _safe_set_selection_mode(self, mode):
        """Safely set selection mode with error handling."""
        try:
            self.set_selection_mode(mode)
        except Exception as e:
            logger.error(f"Error setting selection mode to {mode}: {e}")
            self.validation_error.emit(f"Failed to set selection mode: {e}")
    
    def _safe_clear_all_points(self):
        """Safely clear all points with error handling."""
        try:
            self.clear_all_points()
        except Exception as e:
            logger.error(f"Error clearing points: {e}")
            self.validation_error.emit(f"Failed to clear points: {e}")
    
    def _safe_calculate_transformation(self):
        """Safely calculate transformation with error handling."""
        try:
            self.calculate_transformation()
        except Exception as e:
            logger.error(f"Error calculating transformation: {e}")
            self.validation_error.emit(f"Failed to calculate transformation: {e}")
    
    def _safe_confirm_transformation(self):
        """Safely confirm transformation with error handling."""
        try:
            self.confirm_transformation()
        except Exception as e:
            logger.error(f"Error confirming transformation: {e}")
            self.validation_error.emit(f"Failed to confirm transformation: {e}")
        
    def set_selection_mode(self, mode):
        """Set point selection mode with validation."""
        try:
            if mode not in ["sem", "gds"]:
                raise ValueError(f"Invalid selection mode: {mode}. Must be 'sem' or 'gds'")
            
            old_mode = self.current_mode
            self.current_mode = mode
            self.sem_mode_btn.setChecked(mode == "sem")
            self.gds_mode_btn.setChecked(mode == "gds")
            
            logger.debug(f"Selection mode changed from {old_mode} to {mode}")
            self.point_selection_mode_changed.emit(mode)
            
        except Exception as e:
            logger.error(f"Error setting selection mode: {e}")
            raise
        
    def add_point(self, point, point_type=None):
        """Add a point to the appropriate list with comprehensive validation."""
        try:
            if point_type is None:
                point_type = self.current_mode
            
            if point_type not in ["sem", "gds"]:
                raise ValueError(f"Invalid point type: {point_type}")
            
            if not point:
                raise ValueError("Point data is empty or None")
            
            # Validate point data structure
            validated_point = self._validate_point_data(point)
            
            point_added = False
            
            if point_type == "sem":
                if len(self.sem_points) < 3:
                    self.sem_points.append(validated_point)
                    point_added = True
                    logger.debug(f"Added SEM point {len(self.sem_points)}: {validated_point}")
                else:
                    logger.warning("Maximum SEM points (3) already selected")
                    self.point_validation_warning.emit("Maximum SEM points (3) already selected")
                    return
                    
            elif point_type == "gds":
                if len(self.gds_points) < 3:
                    self.gds_points.append(validated_point)
                    point_added = True
                    logger.debug(f"Added GDS point {len(self.gds_points)}: {validated_point}")
                else:
                    logger.warning("Maximum GDS points (3) already selected")
                    self.point_validation_warning.emit("Maximum GDS points (3) already selected")
                    return
            
            if point_added:
                self.update_status()
                self.update_button_states()
                
                # Check if we have enough points for calculation
                if len(self.sem_points) == 3 and len(self.gds_points) == 3:
                    validation_result = self._comprehensive_point_validation()
                    
                    if validation_result['valid']:
                        self.calculate_btn.setEnabled(True)
                        self.three_points_selected.emit(self.sem_points, self.gds_points)
                        logger.info("All 6 points selected and validated! Calculate button enabled.")
                        self._update_validation_status("Ready", "#27AE60")
                    else:
                        self.calculate_btn.setEnabled(False)
                        error_msg = f"Point validation failed: {validation_result['message']}"
                        logger.warning(error_msg)
                        self.point_validation_warning.emit(error_msg)
                        self._update_validation_status("Validation Failed", "#E74C3C")
                else:
                    remaining_sem = 3 - len(self.sem_points)
                    remaining_gds = 3 - len(self.gds_points)
                    status_msg = f"Need {remaining_sem} SEM, {remaining_gds} GDS"
                    self._update_validation_status(status_msg, "#F39C12")
                    logger.debug(f"Need {remaining_sem} more SEM points and {remaining_gds} more GDS points")
                    
        except Exception as e:
            logger.error(f"Error adding point: {e}")
            self.validation_error.emit(f"Failed to add point: {e}")
            raise
    
    def _validate_point_data(self, point):
        """Validate and normalize point data."""
        try:
            if isinstance(point, dict):
                if 'x' in point and 'y' in point:
                    return {
                        'x': float(point['x']),
                        'y': float(point['y']),
                        'type': point.get('type', 'user_selected')
                    }
                else:
                    raise ValueError("Point dictionary must contain 'x' and 'y' keys")
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                return {
                    'x': float(point[0]),
                    'y': float(point[1]),
                    'type': 'user_selected'
                }
            else:
                raise ValueError(f"Invalid point format: {type(point)}. Expected dict with x,y or list/tuple")
                
        except (ValueError, TypeError) as e:
            logger.error(f"Point validation failed: {e}")
            raise ValueError(f"Invalid point data: {e}")
    
    def _comprehensive_point_validation(self):
        """Perform comprehensive validation of all selected points."""
        try:
            if len(self.sem_points) != 3 or len(self.gds_points) != 3:
                return {'valid': False, 'message': 'Insufficient points selected'}
            
            # Check for duplicate points (too close together)
            sem_validation = self._check_point_distribution(self.sem_points, "SEM")
            if not sem_validation['valid']:
                return sem_validation
            
            gds_validation = self._check_point_distribution(self.gds_points, "GDS")
            if not gds_validation['valid']:
                return gds_validation
            
            # Check for collinear points (would create degenerate transformation)
            sem_collinear = self._check_collinearity(self.sem_points, "SEM")
            if not sem_collinear['valid']:
                return sem_collinear
            
            gds_collinear = self._check_collinearity(self.gds_points, "GDS")
            if not gds_collinear['valid']:
                return gds_collinear
            
            return {'valid': True, 'message': 'All points validated successfully'}
            
        except Exception as e:
            logger.error(f"Error in comprehensive point validation: {e}")
            return {'valid': False, 'message': f'Validation error: {e}'}
    
    def _check_point_distribution(self, points, point_type, min_distance=10.0):
        """Check that points are sufficiently spread out."""
        try:
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    pt1 = points[i]
                    pt2 = points[j]
                    dist = ((pt1['x'] - pt2['x'])**2 + (pt1['y'] - pt2['y'])**2)**0.5
                    
                    if dist < min_distance:
                        return {
                            'valid': False,
                            'message': f'{point_type} points {i+1} and {j+1} are too close together (distance: {dist:.1f}px). '
                                     f'Select points at least {min_distance}px apart.'
                        }
            
            return {'valid': True, 'message': f'{point_type} points are well distributed'}
            
        except Exception as e:
            return {'valid': False, 'message': f'Error checking {point_type} point distribution: {e}'}
    
    def _check_collinearity(self, points, point_type, tolerance=0.1):
        """Check that points are not collinear (which would create degenerate transformation)."""
        try:
            if len(points) != 3:
                return {'valid': False, 'message': f'Need exactly 3 {point_type} points for collinearity check'}
            
            # Calculate cross product to check collinearity
            p1, p2, p3 = points[0], points[1], points[2]
            
            # Vector from p1 to p2
            v1 = (p2['x'] - p1['x'], p2['y'] - p1['y'])
            # Vector from p1 to p3
            v2 = (p3['x'] - p1['x'], p3['y'] - p1['y'])
            
            # Cross product magnitude
            cross_product = abs(v1[0] * v2[1] - v1[1] * v2[0])
            
            # If cross product is too small, points are nearly collinear
            if cross_product < tolerance:
                return {
                    'valid': False,
                    'message': f'{point_type} points are nearly collinear (cross product: {cross_product:.3f}). '
                             'Select points that form a triangle with sufficient area.'
                }
            
            return {'valid': True, 'message': f'{point_type} points form a valid triangle'}
            
        except Exception as e:
            return {'valid': False, 'message': f'Error checking {point_type} collinearity: {e}'}
    
    def _update_validation_status(self, message, color):
        """Update the validation status label."""
        try:
            self.validation_status_label.setText(message)
            self.validation_status_label.setStyleSheet(f"font-weight: bold; color: {color};")
        except Exception as e:
            logger.error(f"Error updating validation status: {e}")
            
    def update_status(self):
        """Update point status labels with error handling."""
        try:
            sem_count = len(self.sem_points)
            gds_count = len(self.gds_points)
            
            # Update SEM status with color coding
            sem_color = "#27AE60" if sem_count == 3 else "#F39C12" if sem_count > 0 else "#E74C3C"
            self.sem_status_label.setText(f"{sem_count}/3")
            self.sem_status_label.setStyleSheet(f"font-weight: bold; color: {sem_color};")
            
            # Update GDS status with color coding
            gds_color = "#27AE60" if gds_count == 3 else "#F39C12" if gds_count > 0 else "#E74C3C"
            self.gds_status_label.setText(f"{gds_count}/3")
            self.gds_status_label.setStyleSheet(f"font-weight: bold; color: {gds_color};")
            
            logger.debug(f"Status updated: SEM {sem_count}/3, GDS {gds_count}/3")
            
        except Exception as e:
            logger.error(f"Error updating status: {e}")
        
    def clear_all_points(self):
        """Clear all selected points with comprehensive error handling."""
        try:
            sem_count = len(self.sem_points)
            gds_count = len(self.gds_points)
            
            # Clear point lists
            self.sem_points.clear()
            self.gds_points.clear()
            
            # Reset transformation data
            self._last_calculated_transformation = None
            
            # Update UI state
            self.update_status()
            self.update_button_states()
            self.calculate_btn.setEnabled(False)
            self.confirm_btn.setEnabled(False)
            
            # Update validation status
            self._update_validation_status("Cleared", "#BDC3C7")
            
            logger.info(f"Cleared all points: {sem_count} SEM points and {gds_count} GDS points removed")
            
        except Exception as e:
            logger.error(f"Error clearing points: {e}")
            self.validation_error.emit(f"Failed to clear points: {e}")
            raise
        
    def update_button_states(self):
        """Update button states based on current point selection status with error handling."""
        try:
            sem_ready = len(self.sem_points) == 3
            gds_ready = len(self.gds_points) == 3
            both_ready = sem_ready and gds_ready
            
            # Calculate button enabled when both point sets are complete and valid
            if both_ready:
                validation_result = self._comprehensive_point_validation()
                self.calculate_btn.setEnabled(validation_result['valid'])
                
                if not validation_result['valid']:
                    self.calculate_btn.setText("Validation Failed")
                    self.calculate_btn.setToolTip(f"Cannot calculate: {validation_result['message']}")
                else:
                    self.calculate_btn.setText("Calculate Transformation")
                    self.calculate_btn.setToolTip("Calculate transformation matrix from selected points")
            else:
                self.calculate_btn.setEnabled(False)
                remaining = 6 - len(self.sem_points) - len(self.gds_points)
                self.calculate_btn.setText(f"Need {remaining} more points")
                self.calculate_btn.setToolTip(f"Select {remaining} more points to enable calculation")
            
            # Clear button enabled when there are points to clear
            has_points = len(self.sem_points) > 0 or len(self.gds_points) > 0
            self.clear_points_btn.setEnabled(has_points)
            
            # Confirm button state depends on successful calculation
            if hasattr(self, '_last_calculated_transformation') and self._last_calculated_transformation:
                self.confirm_btn.setEnabled(True)
                self.confirm_btn.setText("Confirm Alignment")
            else:
                self.confirm_btn.setText("Calculate First")
                self.confirm_btn.setToolTip("Calculate transformation before confirming")
            
            logger.debug(f"Button states updated: calc={self.calculate_btn.isEnabled()}, "
                        f"confirm={self.confirm_btn.isEnabled()}, clear={self.clear_points_btn.isEnabled()}")
            
        except Exception as e:
            logger.error(f"Error updating button states: {e}")
            # Fallback: disable all buttons except clear
            try:
                self.calculate_btn.setEnabled(False)
                self.confirm_btn.setEnabled(False)
                self.clear_points_btn.setEnabled(True)
            except Exception as fallback_error:
                logger.error(f"Fallback button state update failed: {fallback_error}")
    
    def validate_points(self):
        """Validate that all selected points are reasonable for transformation calculation."""
        try:
            if len(self.sem_points) < 3 or len(self.gds_points) < 3:
                return False, "Insufficient points selected"
            
            # Check for duplicate points (too close together)
            def points_too_close(points, min_distance=5.0):
                if not points or len(points) < 2:
                    return False
                    
                coords = []
                for pt in points:
                    if isinstance(pt, dict) and 'x' in pt and 'y' in pt:
                        coords.append((pt['x'], pt['y']))
                    elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        coords.append((pt[0], pt[1]))
                    else:
                        # If we can't extract coordinates, assume they're fine
                        return False
                
                for i in range(len(coords)):
                    for j in range(i + 1, len(coords)):
                        dist = ((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)**0.5
                        if dist < min_distance:
                            return True
                return False
            
            if points_too_close(self.sem_points):
                return False, "SEM points are too close together - select more spread out points"
            
            if points_too_close(self.gds_points):
                return False, "GDS points are too close together - select more spread out points"
            
            return True, "Points validated successfully"
            
        except Exception as e:
            return False, f"Error validating points: {e}"
        
    def calculate_transformation(self):
        """Calculate transformation from selected points."""
        if len(self.sem_points) == 3 and len(self.gds_points) == 3:
            try:
                # Validate points before calculation
                is_valid, validation_msg = self.validate_points()
                if not is_valid:
                    print(f"Cannot calculate transformation: {validation_msg}")
                    self.confirm_btn.setEnabled(False)
                    return
                
                # Calculate transformation matrix from 3-point correspondence
                transform_matrix = self._calculate_affine_transform()
                
                if transform_matrix is None:
                    print("Transformation calculation failed - invalid matrix")
                    self.confirm_btn.setEnabled(False)
                    return
                
                transformation_data = {
                    'sem_points': self.sem_points.copy(),
                    'gds_points': self.gds_points.copy(),
                    'method': '3-point',
                    'transform_matrix': transform_matrix,
                    'calculated': True,
                    'validation_passed': True,
                    'validation_message': validation_msg
                }
                
                self.transformation_calculated.emit(transformation_data)
                self.confirm_btn.setEnabled(True)
                print(f"3-Point transformation calculated successfully")
                print(f"Validation: {validation_msg}")
                print(f"Transform preview: Translation({transform_matrix.get('translation_x', 0):.1f}, {transform_matrix.get('translation_y', 0):.1f}), "
                      f"Rotation({transform_matrix.get('rotation', 0):.1f}°), Scale({transform_matrix.get('scale_x', 1):.3f})")
                      
            except Exception as e:
                print(f"Error calculating transformation: {e}")
                import traceback
                traceback.print_exc()
                self.confirm_btn.setEnabled(False)
        else:
            print(f"Cannot calculate transformation: insufficient points (SEM: {len(self.sem_points)}/3, GDS: {len(self.gds_points)}/3)")
    
    def _calculate_affine_transform(self):
        """Calculate affine transformation matrix from point correspondences."""
        # This is a more sophisticated placeholder for the actual affine transformation calculation
        # In a real implementation, this would use numpy or similar to calculate
        # the transformation matrix from the 3-point correspondences
        try:
            if len(self.sem_points) != 3 or len(self.gds_points) != 3:
                print("Error: Need exactly 3 points for each image")
                return None
                
            # Extract coordinates for easier calculation
            sem_coords = [(pt['x'], pt['y']) for pt in self.sem_points] if isinstance(self.sem_points[0], dict) else self.sem_points
            gds_coords = [(pt['x'], pt['y']) for pt in self.gds_points] if isinstance(self.gds_points[0], dict) else self.gds_points
            
            # Calculate centroid for both point sets
            sem_centroid = (
                sum(x for x, y in sem_coords) / 3,
                sum(y for x, y in sem_coords) / 3
            )
            gds_centroid = (
                sum(x for x, y in gds_coords) / 3,
                sum(y for x, y in gds_coords) / 3
            )
            
            # Basic translation calculation (centroid difference)
            translation_x = gds_centroid[0] - sem_centroid[0]
            translation_y = gds_centroid[1] - sem_centroid[1]
            
            # Simple scale estimation based on distance between first two points
            sem_dist = ((sem_coords[1][0] - sem_coords[0][0])**2 + (sem_coords[1][1] - sem_coords[0][1])**2)**0.5
            gds_dist = ((gds_coords[1][0] - gds_coords[0][0])**2 + (gds_coords[1][1] - gds_coords[0][1])**2)**0.5
            
            scale_factor = gds_dist / sem_dist if sem_dist > 0 else 1.0
            
            # Simplified rotation calculation
            # This is a basic approximation - real implementation would use proper vector math
            import math
            sem_angle = math.atan2(sem_coords[1][1] - sem_coords[0][1], sem_coords[1][0] - sem_coords[0][0])
            gds_angle = math.atan2(gds_coords[1][1] - gds_coords[0][1], gds_coords[1][0] - gds_coords[0][0])
            rotation_rad = gds_angle - sem_angle
            rotation_deg = math.degrees(rotation_rad)
            
            # Improved transformation matrix with calculated values
            transform_matrix = {
                'translation_x': translation_x,
                'translation_y': translation_y,
                'rotation': rotation_deg,
                'scale_x': scale_factor,
                'scale_y': scale_factor,
                'calculated_from_points': True,
                'sem_centroid': sem_centroid,
                'gds_centroid': gds_centroid,
                'point_count': 3,
                'calculation_method': '3-point-centroid-approximation'
            }
            
            print(f"Calculated transformation - Translation: ({translation_x:.2f}, {translation_y:.2f}), "
                  f"Rotation: {rotation_deg:.2f}°, Scale: {scale_factor:.3f}")
            
            return transform_matrix
            
        except Exception as e:
            print(f"Error in affine transform calculation: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def confirm_transformation(self):
        """Confirm the calculated transformation."""
        if len(self.sem_points) == 3 and len(self.gds_points) == 3:
            try:
                # Get the calculated transformation matrix
                transform_matrix = self._calculate_affine_transform()
                
                if transform_matrix is None:
                    print("Cannot confirm transformation: calculation failed")
                    return
                
                transformation_data = {
                    'sem_points': self.sem_points.copy(),
                    'gds_points': self.gds_points.copy(),
                    'method': '3-point',
                    'confirmed': True,
                    'timestamp': None,  # Could add timestamp if needed
                    'transform_matrix': transform_matrix,
                    'confidence': 'calculated'  # Could add confidence metrics
                }
                
                self.transformation_confirmed.emit(transformation_data)
                print(f"3-Point transformation confirmed and applied")
                print(f"Transformation summary: {transform_matrix.get('calculation_method', 'unknown method')}")
                
                # Optionally disable confirm button after successful confirmation
                self.confirm_btn.setEnabled(False)
                
                # Provide user feedback about what happened
                print("Transformation has been applied. You can now save the aligned result.")
                
            except Exception as e:
                print(f"Error confirming transformation: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Cannot confirm transformation: insufficient points (SEM: {len(self.sem_points)}/3, GDS: {len(self.gds_points)}/3)")
            print("Please select 3 points on both SEM and GDS images before confirming.")
    
    def setup_tooltips(self):
        """Set up helpful tooltips for 3-point alignment controls."""
        self.sem_mode_btn.setToolTip("Switch to SEM point selection mode\nClick 3 points on the SEM image")
        self.gds_mode_btn.setToolTip("Switch to GDS point selection mode\nClick 3 corresponding points on the GDS overlay")
        self.clear_points_btn.setToolTip("Clear all selected points and start over")
        self.calculate_btn.setToolTip("Calculate transformation matrix from selected points\nRequires 3 SEM and 3 GDS points")
        self.confirm_btn.setToolTip("Confirm and apply the calculated transformation")
        
    def get_selection_status(self):
        """Get current selection status as a formatted string."""
        return f"SEM: {len(self.sem_points)}/3, GDS: {len(self.gds_points)}/3, Mode: {self.current_mode.upper()}"
    
    def get_detailed_status(self):
        """Get detailed status information for debugging."""
        status = {
            'sem_points': {
                'count': len(self.sem_points),
                'points': self.sem_points,
                'complete': len(self.sem_points) == 3
            },
            'gds_points': {
                'count': len(self.gds_points),
                'points': self.gds_points,
                'complete': len(self.gds_points) == 3
            },
            'mode': self.current_mode,
            'buttons': {
                'calculate_enabled': self.calculate_btn.isEnabled(),
                'confirm_enabled': self.confirm_btn.isEnabled(),
                'clear_enabled': self.clear_points_btn.isEnabled(),
                'calculate_text': self.calculate_btn.text()
            }
        }
        
        # Add validation info
        is_valid, validation_msg = self.validate_points()
        status['validation'] = {
            'valid': is_valid,
            'message': validation_msg
        }
        
        return status
        

class AlignmentLeftPanel(BaseViewPanel):
    """Left panel for alignment view with Manual/3-point tabs."""
    
    # Signals
    alignment_changed = Signal(dict)
    three_points_selected = Signal(list, list)
    transformation_calculated = Signal(dict)
    transformation_confirmed = Signal(dict)
    reset_requested = Signal()
    save_aligned_gds_requested = Signal()
    auto_align_requested = Signal()
    action_requested = Signal(str, dict)  # action_type, parameters
    point_selection_mode_changed = Signal(str)  # "sem" or "gds"
    
    def __init__(self, parent=None):
        super().__init__(ViewMode.ALIGNMENT, parent)
        
    def init_panel(self):
        """Initialize the alignment panel UI with scroll area support."""
        try:
            # Clear default layout
            for i in reversed(range(self.main_layout.count())):
                item = self.main_layout.itemAt(i)
                if item and item.widget():
                    item.widget().setParent(None)
            
            # Create a scroll area for the panel content
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            
            # Create a container widget for all panel content
            content_widget = QWidget()
            content_layout = QVBoxLayout(content_widget)
            content_layout.setContentsMargins(5, 5, 5, 5)
            content_layout.setSpacing(10)
            
            # Add File Selection section at the top
            self._create_file_selection_section(content_layout)
            
            # Create tab widget for transformation controls
            self.tab_widget = QTabWidget()
            
            # Create tabs
            self.manual_tab = ManualAlignmentTab()
            self.three_point_tab = ThreePointAlignmentTab()
            
            # Add tabs
            self.tab_widget.addTab(self.manual_tab, "Manual Alignment")
            self.tab_widget.addTab(self.three_point_tab, "3-Point Alignment")
            
            # Add tab widget to content layout
            content_layout.addWidget(self.tab_widget)
            
            # Set the content widget to the scroll area
            scroll_area.setWidget(content_widget)
            
            # Add scroll area to main layout
            self.main_layout.addWidget(scroll_area)
            
            # Connect tab signals to panel signals
            self._connect_tab_signals()
            
            logger.info("Alignment panel initialized successfully with scroll support")
            
        except Exception as e:
            logger.error(f"Error initializing alignment panel: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _create_file_selection_section(self, parent_layout):
        """Create the file selection section at the top of the panel."""
        try:
            # File Selection Group
            file_selection_group = QGroupBox("File Selection")
            file_selection_group.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #2C3E50;
                    border-radius: 8px;
                    margin-top: 10px;
                    padding-top: 10px;
                    background-color: #ECF0F1;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 8px 0 8px;
                    color: #2C3E50;
                    background-color: #ECF0F1;
                }
            """)
            
            file_layout = QVBoxLayout(file_selection_group)
            
            # Select SEM button
            self.select_sem_btn = QPushButton("Select SEM")
            self.select_sem_btn.setMinimumHeight(30)
            self.select_sem_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498DB;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #2980B9;
                }
                QPushButton:pressed {
                    background-color: #21618C;
                }
            """)
            file_layout.addWidget(self.select_sem_btn)
            
            # GDS Structure selection
            gds_layout = QVBoxLayout()
            gds_label = QLabel("Select GDS Structure:")
            gds_label.setStyleSheet("font-weight: bold; color: #2C3E50;")
            gds_layout.addWidget(gds_label)
            
            # Create structure combo (will be populated by the main window)
            self.structure_combo = QComboBox()
            self.structure_combo.addItem("Select Structure...")
            self.structure_combo.setMinimumHeight(25)
            self.structure_combo.setStyleSheet("""
                QComboBox {
                    border: 2px solid #3498DB;
                    border-radius: 4px;
                    padding: 4px;
                    background-color: white;
                    color: #2C3E50;
                    font-weight: bold;
                }
                QComboBox:focus {
                    border-color: #27AE60;
                }
                QComboBox::drop-down {
                    border: none;
                    width: 20px;
                }
                QComboBox::down-arrow {
                    image: none;
                    border-left: 5px solid transparent;
                    border-right: 5px solid transparent;
                    border-top: 5px solid #2C3E50;
                }
            """)
            gds_layout.addWidget(self.structure_combo)
            
            file_layout.addLayout(gds_layout)
            
            # Add the file selection group to the parent layout
            parent_layout.addWidget(file_selection_group)
            
            # Connect SEM button signal
            self.select_sem_btn.clicked.connect(self._on_select_sem_clicked)
            self.structure_combo.currentTextChanged.connect(self._on_structure_selected)
            
            # Set up tooltips for file selection
            self.select_sem_btn.setToolTip("Select SEM image file for alignment")
            self.structure_combo.setToolTip("Select GDS structure to align with SEM image")
            
            logger.debug("File selection section created successfully")
            
        except Exception as e:
            logger.error(f"Error creating file selection section: {e}")
            raise
    
    def _connect_tab_signals(self):
        """Connect tab signals to panel signals with error handling."""
        try:
            # Connect manual tab signals
            self.manual_tab.alignment_changed.connect(self.alignment_changed)
            self.manual_tab.alignment_changed.connect(lambda params: logger.debug(f"AlignmentLeftPanel received alignment_changed: {params}"))
            self.manual_tab.reset_requested.connect(self.reset_requested)
            
            # Connect the save and auto align buttons from manual tab
            self.manual_tab.save_aligned_gds_btn.clicked.connect(self.save_aligned_gds_requested)
            self.manual_tab.auto_align_btn.clicked.connect(self.auto_align_requested)
            
            # Connect three-point tab signals
            self.three_point_tab.three_points_selected.connect(self.three_points_selected)
            self.three_point_tab.transformation_calculated.connect(self.transformation_calculated)
            self.three_point_tab.transformation_confirmed.connect(self.transformation_confirmed)
            self.three_point_tab.point_selection_mode_changed.connect(self.point_selection_mode_changed)
            
            logger.debug("Tab signals connected successfully")
            
        except Exception as e:
            logger.error(f"Error connecting tab signals: {e}")
            raise
        self.select_sem_btn.clicked.connect(self._on_select_sem_clicked)
        self.structure_combo.currentTextChanged.connect(self._on_structure_selected)
        
        # Set up tooltips for file selection
        self.select_sem_btn.setToolTip("Select SEM image file for alignment")
        self.structure_combo.setToolTip("Select GDS structure to align with SEM image")
    
    def _on_select_sem_clicked(self):
        """Handle Select SEM button click."""
        self.action_requested.emit("select_sem", {})
    
    def _on_structure_selected(self, structure_name):
        """Handle structure selection."""
        if structure_name and structure_name != "Select Structure...":
            self.action_requested.emit("select_structure", {"structure_name": structure_name})
        
    def get_current_tab_name(self):
        """Get the name of the currently active tab."""
        current_index = self.tab_widget.currentIndex()
        return ["manual", "3point"][current_index]
        
    def switch_to_tab(self, tab_name):
        """Switch to a specific tab."""
        tab_index = {"manual": 0, "3point": 1}.get(tab_name, 0)
        self.tab_widget.setCurrentIndex(tab_index)
        
    def set_alignment_parameters(self, params):
        """Set alignment parameters (for manual tab)."""
        self.manual_tab.set_parameters(params)
        
    def get_alignment_parameters(self):
        """Get current alignment parameters from manual tab."""
        return {
            'translation_x_pixels': self.manual_tab.x_offset_spin.value(),
            'translation_y_pixels': self.manual_tab.y_offset_spin.value(),
            'rotation_degrees': self.manual_tab.rotation_spin.value(),
            'scale': self.manual_tab.scale_spin.value() / 100.0,  # Convert percentage to scale factor
            'transparency': self.manual_tab.transparency_spin.value(),
            'canvas_zoom': self.manual_tab.canvas_zoom_spin.value(),
            'display_transparency': self.manual_tab.trans_display_spin.value(),
            'show_overlay': self.manual_tab.show_overlay_cb.isChecked()
        }
    
    def enable_save_button(self, enabled):
        """Enable or disable the save aligned GDS button."""
        if hasattr(self.manual_tab, 'save_aligned_gds_btn'):
            self.manual_tab.save_aligned_gds_btn.setEnabled(enabled)
    
    def reset_alignment_parameters(self):
        """Reset alignment parameters to default values."""
        # Reset individual controls to default values
        self.manual_tab.x_offset_spin.setValue(0)
        self.manual_tab.y_offset_spin.setValue(0)
        self.manual_tab.rotation_spin.setValue(0.0)
        self.manual_tab.scale_spin.setValue(100)  # 100%
        self.manual_tab.transparency_spin.setValue(70)
        self.manual_tab.canvas_zoom_spin.setValue(100)
        self.manual_tab.trans_display_spin.setValue(70)
        self.manual_tab.show_overlay_cb.setChecked(True)
    def set_parameters_from_model(self, aligned_gds_model):
        """Set parameters from AlignedGdsModel."""
        if hasattr(self.manual_tab, 'set_parameters_from_model'):
            self.manual_tab.set_parameters_from_model(aligned_gds_model)
    
    def apply_parameters_to_model(self, aligned_gds_model):
        """Apply current UI parameters to AlignedGdsModel."""
        try:
            params = self.get_alignment_parameters()
            aligned_gds_model.set_ui_parameters(
                translation_x_pixels=params.get('translation_x_pixels', 0),
                translation_y_pixels=params.get('translation_y_pixels', 0),
                scale=params.get('scale', 1.0),
                rotation_degrees=params.get('rotation_degrees', 0.0)
            )
            print(f"Applied UI parameters to model: {params}")
        except Exception as e:
            print(f"Error applying parameters to model: {e}")
            import traceback
            traceback.print_exc()
    
    def update_ui_states(self, has_images=False, alignment_ready=False):
        """Update UI states based on current application state."""
        if hasattr(self.manual_tab, 'update_button_states'):
            self.manual_tab.update_button_states(has_images, alignment_ready)
            
    def populate_structure_combo(self, structures):
        """Populate the structure combo box with available GDS structures."""
        self.structure_combo.clear()
        self.structure_combo.addItem("Select Structure...")
        for structure in structures:
            self.structure_combo.addItem(structure)
            
    def get_selected_structure(self):
        """Get the currently selected GDS structure."""
        current_text = self.structure_combo.currentText()
        return current_text if current_text != "Select Structure..." else None
    
    def get_panel_status(self):
        """Get comprehensive status information for debugging and monitoring."""
        manual_params = self.get_alignment_parameters()
        current_tab = self.get_current_tab_name()
        
        status = {
            'current_tab': current_tab,
            'manual_alignment': {
                'parameters': manual_params,
                'save_button_enabled': self.manual_tab.save_aligned_gds_btn.isEnabled() if hasattr(self.manual_tab, 'save_aligned_gds_btn') else False
            },
            'three_point_alignment': {
                'sem_points_count': len(self.three_point_tab.sem_points),
                'gds_points_count': len(self.three_point_tab.gds_points),
                'current_mode': self.three_point_tab.current_mode,
                'calculate_enabled': self.three_point_tab.calculate_btn.isEnabled(),
                'confirm_enabled': self.three_point_tab.confirm_btn.isEnabled(),
                'detailed_status': self.three_point_tab.get_detailed_status() if hasattr(self.three_point_tab, 'get_detailed_status') else {}
            },
            'file_selection': {
                'selected_structure': self.get_selected_structure()
            }
        }
        return status
    
    def add_three_point_alignment_point(self, point, point_type=None):
        """Add a point to the three-point alignment system."""
        if point_type is None:
            point_type = self.three_point_tab.current_mode
        self.three_point_tab.add_point(point, point_type)
        
    def get_three_point_status(self):
        """Get three-point alignment status."""
        return self.three_point_tab.get_selection_status()
        
    def clear_three_point_points(self):
        """Clear all three-point alignment points."""
        self.three_point_tab.clear_all_points()
        
    def enable_buttons_based_on_state(self, **kwargs):
        """Enable/disable buttons based on current state."""
        # Update manual tab buttons
        if hasattr(self.manual_tab, 'update_button_states'):
            self.manual_tab.update_button_states(
                has_images=kwargs.get('has_images', False),
                alignment_ready=kwargs.get('alignment_ready', False)
            )
        
        # Update three-point tab buttons
        if hasattr(self.three_point_tab, 'update_button_states'):
            self.three_point_tab.update_button_states()
        
        # Update file selection buttons
        self.select_sem_btn.setEnabled(kwargs.get('can_select_files', True))
        self.structure_combo.setEnabled(kwargs.get('can_select_structure', True))
    
    def validate_current_state(self):
        """Validate the current state of the panel and return issues if any."""
        issues = []
        current_tab = self.get_current_tab_name()
        
        if current_tab == "manual":
            # Check if manual alignment parameters are reasonable
            params = self.get_alignment_parameters()
            if abs(params['translation_x_pixels']) > 400:
                issues.append("X translation is very large (>400 pixels)")
            if abs(params['translation_y_pixels']) > 400:
                issues.append("Y translation is very large (>400 pixels)")
            if abs(params['rotation_degrees']) > 45:
                issues.append("Rotation is very large (>45 degrees)")
            if params['scale'] < 0.5 or params['scale'] > 2.0:
                issues.append("Scale factor is extreme (<50% or >200%)")
                
        elif current_tab == "3point":
            # Check three-point alignment state
            if hasattr(self.three_point_tab, 'validate_points'):
                is_valid, message = self.three_point_tab.validate_points()
                if not is_valid:
                    issues.append(f"Three-point validation failed: {message}")
        
        # Check file selection
        if not self.get_selected_structure():
            issues.append("No GDS structure selected")
            
        return issues
