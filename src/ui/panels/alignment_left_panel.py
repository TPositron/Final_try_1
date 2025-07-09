"""
Alignment Left Panel - Comprehensive Alignment Control Interface

This module provides a comprehensive alignment control panel with manual and 3-point
alignment modes, featuring extensive error handling and validation.

Main Classes:
- ManualAlignmentTab: Manual alignment controls with transform parameters
- ThreePointAlignmentTab: 3-point alignment with point selection interface
- AlignmentLeftPanel: Main panel with tabbed alignment modes

Key Methods (ManualAlignmentTab):
- setup_ui(): Creates manual alignment interface
- _create_move_section(): Creates translation controls
- _create_rotation_section(): Creates rotation controls
- _create_zoom_section(): Creates zoom/scale controls
- _create_transparency_section(): Creates transparency controls
- adjust_value_safe(): Safely adjusts parameter values
- set_parameters(): Sets alignment parameters with validation
- reset_parameters(): Resets all parameters to defaults
- emit_alignment_changed(): Emits parameter changes with validation

Key Methods (ThreePointAlignmentTab):
- add_point(): Adds point for 3-point alignment
- calculate_transformation(): Calculates affine transformation
- confirm_transformation(): Confirms calculated transformation
- validate_points(): Validates point selection quality
- clear_all_points(): Clears all selected points

Key Methods (AlignmentLeftPanel):
- set_images(): Sets current SEM and GDS images
- set_gds_model(): Sets GDS model for operations
- get_current_alignment_parameters(): Gets active tab parameters
- save_aligned_gds_image(): Saves aligned GDS file
- reset_all_alignments(): Resets all alignment data

Signals Emitted:
- alignment_changed(dict): Alignment parameters changed
- reset_alignment(): Reset alignment requested
- auto_alignment_requested(): Auto alignment requested
- save_aligned_gds_requested(): Save aligned GDS requested
- three_point_alignment_requested(list, list): 3-point alignment with points
- transformation_confirmed(dict): Transformation confirmed
- validation_error(str): Validation error occurred
- parameter_warning(str): Parameter warning issued

Dependencies:
- Uses: logging, traceback, numpy (error handling and data processing)
- Uses: typing (type hints)
- Uses: PySide6.QtWidgets, PySide6.QtCore (Qt framework)
- Uses: ui/base_panels.BaseViewPanel, ui/view_manager.ViewMode
- Uses: services and models for GDS operations
- Called by: UI main window and alignment workflow
- Coordinates with: Image viewers, alignment services, and file operations

Features:
- Dual-mode alignment interface (manual and 3-point)
- Comprehensive parameter validation and error handling
- Real-time parameter adjustment with increment/decrement buttons
- 3-point alignment with point validation and transformation calculation
- Dark theme styling with consistent UI design
- Extensive logging and debugging capabilities
- GDS file saving with transformation application
- Parameter rollback and recovery mechanisms
"""

import logging
import traceback
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
                               QSlider, QSpinBox, QDoubleSpinBox, QLabel, QPushButton,
                               QGroupBox, QGridLayout, QCheckBox, QComboBox, QMessageBox,
                               QScrollArea)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import QCheckBox
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
    gds_displayed = Signal(np.ndarray)  # Signal to indicate GDS overlay display
    
    def __init__(self, parent=None):
        super().__init__(parent)  # QWidget only needs parent
        self._is_initializing = True
        self._validation_timer = QTimer()
        self._validation_timer.setSingleShot(True)
        self._validation_timer.timeout.connect(self._delayed_validation)
        self._last_validation_issues = []
        self.current_gds_overlay = None
        
        try:
            self.setup_ui()
            self.setup_tooltips()
            self.setup_styling()
            self.connect_signals()  # Changed from setup_connections
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
            move_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #FFFFFF; padding: 2px;")
            parent_layout.addWidget(move_label)
            
            # Move X
            move_x_label = QLabel("Move X:")
            move_x_label.setStyleSheet("font-weight: bold; font-size: 10px; color: #FFFFFF; padding: 2px;")
            parent_layout.addWidget(move_x_label)
            
            move_x_layout = QHBoxLayout()
            
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
            move_y_label = QLabel("Move Y:")
            move_y_label.setStyleSheet("font-weight: bold; font-size: 10px; color: #FFFFFF; padding: 2px;")
            parent_layout.addWidget(move_y_label)
            
            move_y_layout = QHBoxLayout()
            
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
            rotation_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #FFFFFF; padding: 2px;")
            parent_layout.addWidget(rotation_label)
            
            rotation_range_layout = QHBoxLayout()
            rotation_range_layout.addWidget(QLabel("Rotation (-180° to +180°):"))
            parent_layout.addLayout(rotation_range_layout)
            
            rotation_layout = QHBoxLayout()
            
            # Rotation control buttons
            self.rot_minus_btn = self._create_increment_button("--", "Rotate -1.0°")
            self.rot_minus_small_btn = self._create_increment_button("-", "Rotate -0.1°")
            
            self.rotation_spin = self._create_double_spinbox(-180.0, 180.0, 0.0, 0.1, "Rotation angle in degrees")
            
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
            zoom_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #FFFFFF; padding: 2px;")
            parent_layout.addWidget(zoom_label)
            
            zoom_range_layout = QHBoxLayout()
            zoom_range_layout.addWidget(QLabel("Zoom (10% to 500%):"))
            parent_layout.addLayout(zoom_range_layout)
            
            zoom_layout = QHBoxLayout()
            
            # Zoom control buttons
            self.zoom_minus_btn = self._create_increment_button("--", "Decrease zoom by 10%")
            self.zoom_minus_small_btn = self._create_increment_button("-", "Decrease zoom by 1%")
            
            self.zoom_spin = self._create_spinbox(10, 500, 100, "Zoom level percentage")
            
            self.zoom_plus_small_btn = self._create_increment_button("+", "Increase zoom by 1%")
            self.zoom_plus_btn = self._create_increment_button("++", "Increase zoom by 10%")
            
            for widget in [self.zoom_minus_btn, self.zoom_minus_small_btn, self.zoom_spin,
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
            transparency_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #FFFFFF; padding: 2px;")
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
            
            # Add show overlay checkbox
            self.show_overlay_cb = QCheckBox("Show Overlay")
            self.show_overlay_cb.setChecked(True)
            self.show_overlay_cb.setToolTip("Show/hide GDS overlay on SEM image")
            parent_layout.addWidget(self.show_overlay_cb)
            
            logger.debug("Transparency section created successfully")
            
        except Exception as e:
            logger.error(f"Error creating transparency section: {e}")
            raise
    
    def _create_action_buttons(self, parent_layout):
        """Create action buttons section with error handling."""
        try:
            # Action buttons handled in final controls section
            pass
            
        except Exception as e:
            logger.error(f"Error creating action buttons: {e}")
            raise
    

       
    def _create_final_controls(self, parent_layout):
        """Create final control buttons with error handling."""
        try:
            final_button_layout = QVBoxLayout()
            
            # First row - Reset and Auto Align
            button_row1 = QHBoxLayout()
            self.reset_btn = self._create_action_button("Reset All", "Reset all transformation parameters to default values")
            self.auto_align_btn = self._create_action_button("Auto Align", "Automatically align using image processing")
            button_row1.addWidget(self.reset_btn)
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
            button.setMaximumWidth(25)  # Slightly wider to match image design
            button.setMaximumHeight(20)  # Slightly taller to match image design
            button.setMinimumWidth(25)
            button.setMinimumHeight(20)
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
            spinbox.setMaximumWidth(60)  # Adjust width to match image design
            spinbox.setMaximumHeight(20)  # Adjust height to match image design
            spinbox.setMinimumWidth(60)
            spinbox.setMinimumHeight(20)
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
            spinbox.setMaximumWidth(60)  # Adjust width to match image design
            spinbox.setMaximumHeight(20)  # Adjust height to match image design
            spinbox.setMinimumWidth(60)
            spinbox.setMinimumHeight(20)
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
            else:
                button.setMaximumHeight(22)  # Make action buttons smaller
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
            
            # Zoom increment buttons
            self.zoom_minus_btn.clicked.connect(lambda: self.adjust_value_safe(self.zoom_spin, -10))
            self.zoom_minus_small_btn.clicked.connect(lambda: self.adjust_value_safe(self.zoom_spin, -1))
            self.zoom_plus_small_btn.clicked.connect(lambda: self.adjust_value_safe(self.zoom_spin, 1))
            self.zoom_plus_btn.clicked.connect(lambda: self.adjust_value_safe(self.zoom_spin, 10))
            
            # Transparency increment buttons
            self.trans_minus_btn.clicked.connect(lambda: self.adjust_value_safe(self.transparency_spin, -10))
            self.trans_minus_small_btn.clicked.connect(lambda: self.adjust_value_safe(self.transparency_spin, -1))
            self.trans_plus_small_btn.clicked.connect(lambda: self.adjust_value_safe(self.transparency_spin, 1))
            self.trans_plus_btn.clicked.connect(lambda: self.adjust_value_safe(self.transparency_spin, 10))
            
           
            # Connect value changes to emit alignment_changed with validation
            self._connect_value_change_signals()
            
            # Buttons - connect reset button to signal
            self.reset_btn.clicked.connect(self._on_reset_clicked)
            
            logger.debug("Signal connections established successfully")
            
        except Exception as e:
            logger.error(f"Error connecting signals: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _connect_value_change_signals(self):
        """Connect value change signals with validation and error handling."""
        try:
            # Connect value changes to emit alignment_changed with validation
            controls = [
                self.x_offset_spin, self.y_offset_spin, self.rotation_spin,
                self.zoom_spin, self.transparency_spin
            ]
            
            for control in controls:
                if hasattr(control, 'valueChanged'):
                    control.valueChanged.connect(self._handle_value_change)
                else:
                    logger.warning(f"Control {control} does not have valueChanged signal")
            
            if hasattr(self, 'show_overlay_cb'):
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
        """Get current parameters with error handling and proper service format."""
        try:
            # Convert zoom percentage to scale factor
            zoom_percentage = self.zoom_spin.value()
            scale_factor = zoom_percentage / 100.0
            
            return {
                # Service-compatible parameter names
                'x_offset': self.x_offset_spin.value(),
                'y_offset': self.y_offset_spin.value(),
                'rotation': self.rotation_spin.value(),
                'scale': scale_factor,  # Converted from zoom
                'transparency': self.transparency_spin.value(),
                
                # UI-specific parameters for validation and display
                'zoom': zoom_percentage
            }
        except Exception as e:
            logger.error(f"Error getting current parameters: {e}")
            return {}
    
    def _validate_parameters(self, params):
        """Validate parameters and return list of issues."""
        issues = []
        
        try:
            # Translation validation
            if abs(params.get('x_offset', 0)) > 400:
                issues.append({
                    'parameter': 'x_offset',
                    'severity': 'warning',
                    'message': f"Large X translation ({params['x_offset']} pixels) may indicate misalignment"
                })
            
            if abs(params.get('y_offset', 0)) > 400:
                issues.append({
                    'parameter': 'y_offset',
                    'severity': 'warning',
                    'message': f"Large Y translation ({params['y_offset']} pixels) may indicate misalignment"
                })
            
            # Rotation validation
            rotation = params.get('rotation', 0)
            if abs(rotation) > 45:
                issues.append({
                    'parameter': 'rotation',
                    'severity': 'warning',
                    'message': f"Large rotation ({rotation:.1f}°) may indicate misalignment"
                })
            
            # Scale validation (converted from zoom)
            scale = params.get('scale', 1.0)
            if scale < 0.1 or scale > 5.0:
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
            
            # Zoom validation (UI parameter)
            zoom = params.get('zoom', 100)
            if zoom < 10 or zoom > 500:
                issues.append({
                    'parameter': 'zoom',
                    'severity': 'error',
                    'message': f"Zoom percentage ({zoom}%) is outside valid range [10%, 500%]"
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
        self.rotation_spin.setToolTip("Rotation angle in degrees\nRange: -180° to +180°\nStep: 0.1°")
        self.rot_minus_btn.setToolTip("Rotate -1.0°")
        self.rot_minus_small_btn.setToolTip("Rotate -0.1°")
        self.rot_plus_small_btn.setToolTip("Rotate +0.1°")
        self.rot_plus_btn.setToolTip("Rotate +1.0°")
        
        # Zoom controls
        self.zoom_spin.setToolTip("Zoom level as percentage\nRange: 10% to 500%\nConverted to scale factor: 0.1 to 5.0")
        self.zoom_minus_btn.setToolTip("Decrease zoom by 10%")
        self.zoom_minus_small_btn.setToolTip("Decrease zoom by 1%")
        self.zoom_plus_small_btn.setToolTip("Increase zoom by 1%")
        self.zoom_plus_btn.setToolTip("Increase zoom by 10%")
        
        # Transparency controls
        self.transparency_spin.setToolTip("Overlay transparency\nRange: 0% (opaque) to 100% (transparent)")
        
        
        # Action buttons
        self.reset_btn.setToolTip("Reset all transformation parameters to default values")
        self.save_aligned_gds_btn.setToolTip("Save the aligned GDS as PNG image")
        self.auto_align_btn.setToolTip("Automatically align using image processing")
        
        # Display options removed
        
    def setup_styling(self):
        """Set up dark theme styling for all controls."""
        # Define dark theme color scheme
        primary_dark = "#2B2B2B"          # Dark gray for backgrounds
        secondary_dark = "#3A3A3A"        # Slightly lighter gray for borders
        accent_color = "#333333"          # Dark gray for buttons (was blue)
        success_color = "#4CAF50"         # Green for success states
        warning_color = "#FF9800"         # Orange for warnings
        danger_color = "#E22719"          # Red for danger states
        text_color = "#FFFFFF"            # White text
        light_text = "#CCCCCC"            # Light gray text
        dark_bg = "#1E1E1E"               # Very dark background
        
        # Set the overall widget background to match the image design
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {primary_dark};
                color: {text_color};
            }}
        """)
        
        # Style the main groups with dark theme matching the image design
        group_style = f"""
            QGroupBox {{
                font-weight: bold;
                font-size: 11px;
                border: 1px solid {secondary_dark};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                background-color: {primary_dark};
                color: {text_color};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px 0 4px;
                color: {text_color};
                background-color: {primary_dark};
                font-size: 11px;
                font-weight: bold;
            }}
        """
        
        # Apply dark styling to all group boxes
        for group in self.findChildren(QGroupBox):
            group.setStyleSheet(group_style)
        
        # Style increment/decrement buttons with dark theme to match the image design
        button_style = """
        QPushButton {
            background-color: #2B2B2B;
            color: white;
            border: 1px solid #444444;
            border-radius: 3px;
            padding: 2px 4px;
            min-width: 18px;
            min-height: 18px;
            font-size: 10px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #3A3A3A;
            border-color: #555555;
        }
        QPushButton:pressed {
            background-color: #1E1E1E;
            border-color: #333333;
        }
        """
        for btn in self.findChildren(QPushButton):
            btn.setStyleSheet(button_style)
        
        # Apply to increment/decrement buttons 
        increment_buttons = [
            self.x_minus_btn, self.x_minus_small_btn, self.x_plus_small_btn, self.x_plus_btn,
            self.y_minus_btn, self.y_minus_small_btn, self.y_plus_small_btn, self.y_plus_btn,
            self.rot_minus_btn, self.rot_minus_small_btn, self.rot_plus_small_btn, self.rot_plus_btn,
            self.zoom_minus_btn, self.zoom_minus_small_btn, self.zoom_plus_small_btn, self.zoom_plus_btn,
            self.trans_minus_btn, self.trans_minus_small_btn, self.trans_plus_small_btn, self.trans_plus_btn
        ]
        
        for btn in increment_buttons:
            btn.setStyleSheet(button_style)
        
        # Style action buttons with dark theme and smaller fonts
        action_button_style = f"""
            QPushButton {{
                background-color: {secondary_dark};
                color: {text_color};
                border: 1px solid {accent_color};
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background-color: {accent_color};
                border-color: {accent_color};
            }}
            QPushButton:pressed {{
                background-color: #444444;
                border-color: #444444;
            }}
            QPushButton:disabled {{
                background-color: #333333;
                color: #666666;
                border-color: #444444;
            }}
        """
        
        action_buttons = [self.reset_btn, self.auto_align_btn]
        for btn in action_buttons:
            btn.setStyleSheet(action_button_style)
        
        # Style spinboxes with dark theme to match the image design
        spinbox_style = """
        QSpinBox, QDoubleSpinBox {
            border: 1px solid #444444;
            border-radius: 3px;
            padding: 2px 4px;
            background-color: #2B2B2B;
            color: #ffffff;
            font-size: 10px;
            font-weight: normal;
            min-width: 50px;
            min-height: 18px;
        }
        QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #4CAF50;
            background-color: #3A3A3A;
        }
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            width: 0px;
            height: 0px;
            border: 0px;
        }
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            width: 0px;
            height: 0px;
            border: 0px;
        }
        """
        
        # Apply to all spinboxes (PySide6: must call separately for each type)
        spinboxes = self.findChildren(QSpinBox) + self.findChildren(QDoubleSpinBox)
        for spinbox in spinboxes:
            spinbox.setStyleSheet(spinbox_style)
        
        # Style labels to match the image design
        label_style = f"""
            QLabel {{
                color: {text_color};
                font-size: 10px;
                font-weight: normal;
                padding: 2px;
            }}
        """
        
        # Apply to all labels
        for label in self.findChildren(QLabel):
            label.setStyleSheet(label_style)
        
        # Style checkboxes with dark theme and smaller fonts
        checkbox_style = f"""
            QCheckBox {{
                color: {text_color};
                font-weight: bold;
                font-size: 10px;
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 2px solid {secondary_dark};
                border-radius: 4px;
                background-color: {primary_dark};
            }}
            QCheckBox::indicator:checked {{
                background-color: {success_color};
                border-color: {success_color};
            }}
            QCheckBox::indicator:hover {{
                border-color: {accent_color};
            }}
        """
        
        # Apply checkbox styling
        for checkbox in self.findChildren(QCheckBox):
            checkbox.setStyleSheet(checkbox_style)
        
        # Style labels with dark theme
        label_style = f"""
            QLabel {{
                color: {text_color};
                font-weight: bold;
            }}
        """
        
        # Apply dark theme to all labels
        for label in self.findChildren(QLabel):
            label.setStyleSheet(label_style)
        
    def update_button_states(self, has_images=False, alignment_ready=False):
        """Update button enabled states based on current conditions."""
        # Enable/disable buttons based on whether images are loaded
        # Auto align button needs images to work
        self.auto_align_btn.setEnabled(has_images)
            
        # Enable save button only when alignment is ready
        self.save_aligned_gds_btn.setEnabled(alignment_ready)
        
        # Always enable reset button
        self.reset_btn.setEnabled(True)
        
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
            
            print(f"ManualAlignmentTab emitting alignment_changed: {params}")
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
        """Set alignment parameters from dictionary with comprehensive validation and format conversion."""
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
                # Translation X - handle both old and new parameter names
                if 'x_offset' in params:
                    value = self._validate_and_convert_value(params['x_offset'], int, 'x_offset')
                    self.x_offset_spin.setValue(value)
                    set_params['x_offset'] = value
                elif 'translation_x_pixels' in params:
                    value = self._validate_and_convert_value(params['translation_x_pixels'], int, 'translation_x_pixels')
                    self.x_offset_spin.setValue(value)
                    set_params['x_offset'] = value
                
                # Translation Y - handle both old and new parameter names
                if 'y_offset' in params:
                    value = self._validate_and_convert_value(params['y_offset'], int, 'y_offset')
                    self.y_offset_spin.setValue(value)
                    set_params['y_offset'] = value
                elif 'translation_y_pixels' in params:
                    value = self._validate_and_convert_value(params['translation_y_pixels'], int, 'translation_y_pixels')
                    self.y_offset_spin.setValue(value)
                    set_params['y_offset'] = value
                
                # Rotation - handle both old and new parameter names
                if 'rotation' in params:
                    value = self._validate_and_convert_value(params['rotation'], float, 'rotation')
                    self.rotation_spin.setValue(value)
                    set_params['rotation'] = value
                elif 'rotation_degrees' in params:
                    value = self._validate_and_convert_value(params['rotation_degrees'], float, 'rotation_degrees')
                    self.rotation_spin.setValue(value)
                    set_params['rotation'] = value
                
                # Scale/Zoom handling - convert between formats
                if 'scale' in params:
                    scale_value = self._validate_and_convert_value(params['scale'], float, 'scale')
                    # Convert scale factor to zoom percentage for UI
                    zoom_percentage = scale_value * 100.0
                    if 10 <= zoom_percentage <= 500:
                        self.zoom_spin.setValue(int(zoom_percentage))
                        set_params['scale'] = scale_value
                        set_params['zoom'] = zoom_percentage
                    else:
                        logger.warning(f"Scale converted to zoom percentage {zoom_percentage:.1f}% out of range [10, 500], skipping")
                elif 'zoom' in params:
                    zoom_value = self._validate_and_convert_value(params['zoom'], int, 'zoom')
                    if 10 <= zoom_value <= 500:
                        self.zoom_spin.setValue(zoom_value)
                        scale_value = zoom_value / 100.0
                        set_params['zoom'] = zoom_value
                        set_params['scale'] = scale_value
                    else:
                        logger.warning(f"Zoom {zoom_value}% out of range [10, 500], skipping")
                
                # Transparency
                if 'transparency' in params:
                    value = self._validate_and_convert_value(params['transparency'], int, 'transparency')
                    if 0 <= value <= 100:
                        self.transparency_spin.setValue(value)
                        set_params['transparency'] = value
                    else:
                        logger.warning(f"Transparency {value} out of range [0, 100], skipping")
                
                # Show overlay
                if 'show_overlay' in params and hasattr(self, 'show_overlay_cb'):
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
            if param_name in ['x_offset', 'translation_x_pixels']:
                if abs(converted_value) > 1000:
                    logger.warning(f"Large X translation value: {converted_value}")
            elif param_name in ['y_offset', 'translation_y_pixels']:
                if abs(converted_value) > 1000:
                    logger.warning(f"Large Y translation value: {converted_value}")
            elif param_name in ['rotation', 'rotation_degrees']:
                if abs(converted_value) > 180:
                    logger.warning(f"Large rotation value: {converted_value}")
                    # Clamp to valid range
                    converted_value = max(-180, min(180, converted_value))
            elif param_name == 'scale':
                if converted_value <= 0:
                    raise ValueError(f"Scale must be positive, got {converted_value}")
                if converted_value < 0.1 or converted_value > 5.0:
                    logger.warning(f"Extreme scale value: {converted_value}")
                    # Clamp to reasonable range
                    converted_value = max(0.1, min(5.0, converted_value))
            elif param_name == 'zoom':
                if converted_value < 10 or converted_value > 500:
                    logger.warning(f"Zoom percentage out of range: {converted_value}")
                    # Clamp to valid range
                    converted_value = max(10, min(500, converted_value))
            
            return converted_value
            
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert {param_name} value {value} to {target_type.__name__}: {e}")
            raise ValueError(f"Invalid {param_name} value: {value}")
    
    def _block_signals(self, block):
        """Block or unblock signals from all controls."""
        try:
            controls = [
                self.x_offset_spin, self.y_offset_spin, self.rotation_spin,
                self.zoom_spin, self.transparency_spin
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
                'zoom': 100
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
                    
                # Scale removed from UI in Step 3
                # try:
                #     self.scale_spin.setValue(defaults['scale_percentage'])
                # except Exception as e:
                #     reset_errors.append(f"scale: {e}")
                    
                try:
                    self.transparency_spin.setValue(defaults['transparency'])
                except Exception as e:
                    reset_errors.append(f"transparency: {e}")
                    
                try:
                    self.zoom_spin.setValue(defaults['zoom'])
                except Exception as e:
                    reset_errors.append(f"zoom: {e}")
                    
                # Show overlay checkbox
                try:
                    if hasattr(self, 'show_overlay_cb'):
                        self.show_overlay_cb.setChecked(True)
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
        """Get current parameter values in service-compatible format."""
        # Convert zoom percentage to scale factor (zoom: 10-500% -> scale: 0.1-5.0)
        zoom_percentage = self.zoom_spin.value()
        scale_factor = zoom_percentage / 100.0
        
        return {
            # Use parameter names that match AlignmentService expectations
            'x_offset': self.x_offset_spin.value(),
            'y_offset': self.y_offset_spin.value(),
            'rotation': self.rotation_spin.value(),
            'scale': scale_factor,  # Converted from zoom percentage
            'transparency': self.transparency_spin.value(),
            
            # Additional UI-specific parameters
            'zoom': zoom_percentage,  # Keep for UI state management
            'show_overlay': self.show_overlay_cb.isChecked() if hasattr(self, 'show_overlay_cb') and self.show_overlay_cb else True
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
                
                # Scale removed from UI in Step 3
                # scale_factor = ui_params.get('scale', 1.0)
                # try:
                #     scale_percentage = scale_factor * 100.0
                #     if 10 <= scale_percentage <= 500:
                #         self.scale_spin.setValue(int(scale_percentage))
                #         logger.debug(f"Set scale to {scale_percentage}% (factor: {scale_factor})")
                #     else:
                #         logger.warning(f"Scale percentage {scale_percentage} out of range, using default")
                #         self.scale_spin.setValue(100)
                # except Exception as e:
                #     logger.error(f"Error setting scale parameter: {e}")
                #     self.scale_spin.setValue(100)
                
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
    
    def _on_reset_clicked(self):
        """Handle reset button click."""
        try:
            self.reset_parameters()
            self.reset_requested.emit()
            logger.info("Reset button clicked - parameters reset")
        except Exception as e:
            logger.error(f"Error handling reset click: {e}")
            self.validation_error.emit(f"Failed to reset parameters: {e}")


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
            self.setup_styling()
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
            instructions.setStyleSheet("font-weight: bold; color: #FFFFFF; padding: 10px; background-color: #2B2B2B; border: 2px solid #3A3A3A; border-radius: 5px;")
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
                            return False
                return True
            
            if not points_too_close(self.sem_points):
                return False, "SEM points are too close together - select more spread out points"
            
            if not points_too_close(self.gds_points):
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
    
    def setup_styling(self):
        """Set up dark theme styling for the 3-point alignment tab."""
        self.setStyleSheet("""
            QWidget {
                background-color: #2B2B2B;
                color: #FFFFFF;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px 0 4px;
            }
        """)
    
    def setup_tooltips(self):
        """Set up tooltips for 3-point alignment controls."""
        pass  # Tooltips already set in UI creation


class AlignmentLeftPanel(QWidget):  # Change from BaseViewPanel to QWidget
    """Left panel for alignment operations with manual and 3-point alignment tabs."""
    
    # Keep all the same signals
    alignment_changed = Signal(dict)
    reset_alignment = Signal()
    auto_alignment_requested = Signal()
    save_aligned_gds_requested = Signal()
    three_point_alignment_requested = Signal(list, list)
    transformation_confirmed = Signal(dict)
    gds_displayed = Signal(np.ndarray)
    
    def __init__(self, parent=None):
        super().__init__(parent)  # Now calls QWidget.__init__
        
        self.current_sem_image = None
        self.current_gds_overlay = None
        self.structure_combo = None
        self._current_gds_model = None
        
        try:
            self.setup_ui()
            self.setup_connections()
            logger.info("AlignmentLeftPanel initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AlignmentLeftPanel: {e}")
            logger.error(traceback.format_exc())
            raise

    
    def setup_ui(self):
        """Set up the UI with tab widget for manual and 3-point alignment."""
        try:
            layout = QVBoxLayout(self)
            
            # Main tab widget
            self.tab_widget = QTabWidget()
            
            # Manual alignment tab
            self.manual_tab = ManualAlignmentTab()
            self.tab_widget.addTab(self.manual_tab, "Manual")
            
            # 3-point alignment tab  
            self.three_point_tab = ThreePointAlignmentTab()
            self.tab_widget.addTab(self.three_point_tab, "3-Point")
            
            layout.addWidget(self.tab_widget)
            
            # Set dark theme styling
            self.setup_dark_theme()
            
            logger.debug("AlignmentLeftPanel UI setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up AlignmentLeftPanel UI: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def setup_connections(self):
        """Connect signals between tabs and parent panel."""
        try:
            # Manual tab signals
            self.manual_tab.alignment_changed.connect(self.alignment_changed)
            self.manual_tab.reset_requested.connect(self.reset_alignment)
            self.manual_tab.validation_error.connect(self._handle_validation_error)
            self.manual_tab.parameter_warning.connect(self._handle_parameter_warning)
            self.manual_tab.gds_displayed.connect(self.gds_displayed)
            
            # 3-point tab signals
            self.three_point_tab.three_points_selected.connect(self.three_point_alignment_requested)
            self.three_point_tab.transformation_confirmed.connect(self.transformation_confirmed)
            self.three_point_tab.validation_error.connect(self._handle_validation_error)
            self.three_point_tab.point_validation_warning.connect(self._handle_parameter_warning)
            
            # Connect button signals from manual tab
            if hasattr(self.manual_tab, 'auto_align_btn'):
                self.manual_tab.auto_align_btn.clicked.connect(self.auto_alignment_requested)
            if hasattr(self.manual_tab, 'save_aligned_gds_btn'):
                self.manual_tab.save_aligned_gds_btn.clicked.connect(self.save_aligned_gds_image)
            
            logger.debug("AlignmentLeftPanel signal connections established")
            
        except Exception as e:
            logger.error(f"Error connecting AlignmentLeftPanel signals: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def setup_dark_theme(self):
        """Set up dark theme styling for the entire panel."""
        self.setStyleSheet("""
            QTabWidget {
                background-color: #2B2B2B;
                border: none;
            }
            QTabWidget::pane {
                border: 1px solid #3A3A3A;
                background-color: #2B2B2B;
            }
            QTabWidget::tab-bar {
                left: 5px;
            }
            QTabBar::tab {
                background-color: #3A3A3A;
                color: #FFFFFF;
                border: 1px solid #444444;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
                border-bottom: 2px solid #4CAF50;
            }
            QTabBar::tab:hover {
                background-color: #444444;
            }
        """)
    
    def _handle_validation_error(self, error_message):
        """Handle validation errors from tabs."""
        logger.error(f"Validation error: {error_message}")
        # Could show a message box or emit a signal for the main window to handle
        
    def _handle_parameter_warning(self, warning_message):
        """Handle parameter warnings from tabs."""
        logger.warning(f"Parameter warning: {warning_message}")
        # Could show a status message or emit a signal for the main window to handle
    
    def set_images(self, sem_image, gds_overlay):
        """Set the current images for alignment operations."""
        try:
            self.current_sem_image = sem_image
            self.current_gds_overlay = gds_overlay
            
            # Update button states based on image availability
            has_images = sem_image is not None and gds_overlay is not None
            self.manual_tab.update_button_states(has_images=has_images, alignment_ready=has_images)
            
            logger.debug(f"Images set: SEM={'loaded' if sem_image is not None else 'None'}, "
                        f"GDS={'loaded' if gds_overlay is not None else 'None'}")
            
        except Exception as e:
            logger.error(f"Error setting images: {e}")
    
    def set_gds_model(self, aligned_gds_model):
        """Set the current AlignedGdsModel for saving operations."""
        try:
            self._current_gds_model = aligned_gds_model
            
            # Enable save button if we have a GDS model
            if aligned_gds_model is not None:
                self.manual_tab.update_button_states(has_images=True, alignment_ready=True)
            
            logger.debug(f"GDS model set: {'loaded' if aligned_gds_model is not None else 'None'}")
            
        except Exception as e:
            logger.error(f"Error setting GDS model: {e}")
    
    def get_current_alignment_parameters(self):
        """Get current alignment parameters from the active tab."""
        try:
            if self.tab_widget.currentIndex() == 0:  # Manual tab
                return self.manual_tab.get_parameters()
            else:  # 3-point tab
                # Return the last calculated transformation or default values
                if hasattr(self.three_point_tab, '_last_calculated_transformation') and \
                   self.three_point_tab._last_calculated_transformation:
                    transform = self.three_point_tab._last_calculated_transformation
                    return {
                        'translation_x_pixels': transform.get('translation_x', 0),
                        'translation_y_pixels': transform.get('translation_y', 0),
                        'rotation_degrees': transform.get('rotation', 0),
                        'scale': transform.get('scale_x', 1.0),
                        'method': '3-point'
                    }
                else:
                    return {
                        'translation_x_pixels': 0,
                        'translation_y_pixels': 0,
                        'rotation_degrees': 0,
                        'scale': 1.0,
                        'method': '3-point'
                    }
        except Exception as e:
            logger.error(f"Error getting alignment parameters: {e}")
            return {}
    
    def set_alignment_parameters(self, params):
        """Set alignment parameters on the active tab."""
        try:
            if self.tab_widget.currentIndex() == 0:  # Manual tab
                self.manual_tab.set_parameters(params)
            # 3-point tab doesn't support setting parameters directly
            # as they are calculated from point selections
            
        except Exception as e:
            logger.error(f"Error setting alignment parameters: {e}")
    
    def add_point_for_three_point_alignment(self, point):
        """Add a point for 3-point alignment."""
        try:
            if self.tab_widget.currentIndex() == 1:  # 3-point tab is active
                self.three_point_tab.add_point(point)
            else:
                logger.warning("3-point tab is not active, cannot add point")
                
        except Exception as e:
            logger.error(f"Error adding point for 3-point alignment: {e}")
    
    def get_three_point_selection_mode(self):
        """Get the current 3-point selection mode (sem or gds)."""
        try:
            return self.three_point_tab.current_mode
        except Exception as e:
            logger.error(f"Error getting 3-point selection mode: {e}")
            return "sem"
    
    def reset_all_alignments(self):
        """Reset all alignment parameters and data."""
        try:
            # Reset manual tab
            self.manual_tab.reset_parameters()
            
            # Reset 3-point tab
            self.three_point_tab.clear_all_points()
            
            logger.info("All alignments reset")
            
        except Exception as e:
            logger.error(f"Error resetting alignments: {e}")
    
    def update_view_mode(self, mode: ViewMode):
        """Update the panel based on the current view mode."""
        try:
            # Enable/disable tabs based on view mode
            if mode == ViewMode.ALIGNMENT:
                self.setEnabled(True)
                # Update button states when in alignment mode
                has_images = self.current_sem_image is not None and self.current_gds_overlay is not None
                self.manual_tab.update_button_states(has_images=has_images)
            else:
                self.setEnabled(False)
                
        except Exception as e:
            logger.error(f"Error updating view mode: {e}")
    
    def set_alignment_ready(self, ready=True):
        """Set whether alignment is ready for saving."""
        try:
            self.manual_tab.update_button_states(
                has_images=self.current_sem_image is not None and self.current_gds_overlay is not None,
                alignment_ready=ready
            )
        except Exception as e:
            logger.error(f"Error setting alignment ready state: {e}")
    
    def save_aligned_gds_image(self):
        """Save aligned GDS file with transformed coordinates using existing services."""
        try:
            from pathlib import Path
            from datetime import datetime
            from src.core.models.simple_aligned_gds_model import AlignedGdsModel
            from src.services.gds_transformation_service import GdsTransformationService
            
            # Get current parameters
            params = self.manual_tab.get_parameters()
            logger.info(f"Saving GDS with parameters: {params}")
            
            # Check if we have GDS data loaded
            if not hasattr(self, '_current_gds_model') or self._current_gds_model is None:
                self.manual_tab.validation_error.emit("No GDS file loaded. Please load a GDS file first.")
                return
            
            # Create AlignedGdsModel with current transformations
            aligned_model = self._current_gds_model
            
            # Set UI parameters on the aligned model
            aligned_model.set_ui_parameters(
                translation_x_pixels=params.get('x_offset', 0),
                translation_y_pixels=params.get('y_offset', 0),
                scale=params.get('scale', 1.0),
                rotation_degrees=params.get('rotation', 0)
            )
            
            # Create output directory
            output_dir = Path("Results/Aligned/manual")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aligned_manual_{timestamp}.gds"
            output_path = output_dir / filename
            
            # Use GdsTransformationService to create new GDS file
            transformation_service = GdsTransformationService()
            
            # Get transformation parameters in the format expected by the service
            transform_params = {
                'x_offset': params.get('x_offset', 0),
                'y_offset': params.get('y_offset', 0),
                'rotation': params.get('rotation', 0),
                'scale': params.get('scale', 1.0)
            }
            
            # Transform the structure
            transformed_cell = transformation_service.transform_structure(
                original_gds_path=str(aligned_model.initial_model.gds_path),
                structure_name=aligned_model.initial_model.cell.name,
                transformation_params=transform_params,
                gds_bounds=tuple(aligned_model.original_frame)
            )
            
            # Save the transformed GDS file
            transformation_service.save_transformed_gds(transformed_cell, str(output_path))
            
            logger.info(f"Aligned GDS file saved: {output_path}")
            
            # Update button state to show success
            self.manual_tab.save_aligned_gds_btn.setText("Saved!")
            self.manual_tab.save_aligned_gds_btn.setStyleSheet("""
                QPushButton {
                    background-color: #27AE60;
                    color: white;
                    font-weight: bold;
                    padding: 8px;
                    border: none;
                    border-radius: 4px;
                }
            """)
            
            # Reset button after 2 seconds
            def reset_button():
                self.manual_tab.save_aligned_gds_btn.setText("Save Aligned GDS")
                self.manual_tab.save_aligned_gds_btn.setStyleSheet("""
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
            QTimer.singleShot(2000, reset_button)
                
        except Exception as e:
            logger.error(f"Error saving aligned GDS: {e}")
            import traceback
            traceback.print_exc()
            self.manual_tab.validation_error.emit(f"Error saving aligned GDS: {str(e)}")
