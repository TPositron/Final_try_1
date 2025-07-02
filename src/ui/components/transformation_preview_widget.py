"""
Transformation Preview and Confirmation Component for Step 11.

This component provides:
- Affine transformation calculation from 3-point pairs
- Translation, rotation (90-degree increments), and scaling calculation
- Validation that transformation makes sense (no extreme distortions)
- Transformation preview before applying
- User confirmation or adjustment of transformation
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QFrame, QGroupBox, QTableWidget, QTableWidgetItem,
                              QHeaderView, QMessageBox, QDoubleSpinBox, QCheckBox,
                              QTextEdit, QProgressBar)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor, QFont, QPalette
import cv2

logger = logging.getLogger(__name__)


class TransformationPreviewWidget(QWidget):
    """Widget for previewing and confirming transformation calculations from 3-point pairs."""
    
    # Signals for Step 11
    transformation_calculated = Signal(dict)           # transformation_parameters
    transformation_confirmed = Signal(dict)           # confirmed_transformation
    transformation_rejected = Signal(str)             # rejection_reason
    preview_updated = Signal(np.ndarray)              # preview_image
    validation_status_changed = Signal(bool, str)     # is_valid, status_message
    parameter_adjusted = Signal(str, float)           # parameter_name, new_value
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Transformation data
        self._sem_points = []
        self._gds_points = []
        self._calculated_transform = None
        self._transform_matrix = None
        self._is_valid = False
        self._preview_image = None
        
        # Original images for preview
        self._original_gds_image = None
        self._sem_reference_image = None
        
        # Validation parameters
        self._max_scale_factor = 5.0
        self._min_scale_factor = 0.2
        self._max_translation = 1000.0  # pixels
        
        self._setup_ui()
        
        logger.info("TransformationPreviewWidget initialized")
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Transformation Calculation & Preview")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        layout.addWidget(title_label)
        
        # Main content in horizontal layout
        main_layout = QHBoxLayout()
        
        # Left side: Parameters and controls
        left_panel = self._create_parameters_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right side: Preview and validation
        right_panel = self._create_preview_panel()
        main_layout.addWidget(right_panel, 1)
        
        layout.addLayout(main_layout)
        
        # Bottom: Action buttons
        buttons_layout = self._create_action_buttons()
        layout.addLayout(buttons_layout)
    
    def _create_parameters_panel(self) -> QWidget:
        """Create transformation parameters panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Calculation section
        calc_group = QGroupBox("Transformation Calculation")
        calc_layout = QVBoxLayout(calc_group)
        
        # Calculate button
        self.calculate_btn = QPushButton("Calculate Transformation")
        self.calculate_btn.clicked.connect(self._calculate_transformation)
        self.calculate_btn.setEnabled(False)
        calc_layout.addWidget(self.calculate_btn)
        
        # Progress bar
        self.calc_progress = QProgressBar()
        self.calc_progress.setVisible(False)
        calc_layout.addWidget(self.calc_progress)
        
        # Calculation status with enhanced feedback
        self.calc_status_label = QLabel("Waiting for point selection...")
        self.calc_status_label.setWordWrap(True)
        calc_layout.addWidget(self.calc_status_label)
        
        # Point selection status indicator
        self.points_status_label = QLabel("No points selected")
        self.points_status_label.setStyleSheet("""
            QLabel {
                padding: 4px;
                border: 1px solid #ccc;
                border-radius: 3px;
                background-color: #f5f5f5;
                font-size: 11px;
            }
        """)
        calc_layout.addWidget(self.points_status_label)
        
        layout.addWidget(calc_group)
        
        # Parameters display/adjustment section
        params_group = QGroupBox("Transformation Parameters")
        params_layout = QVBoxLayout(params_group)
        
        # Parameters table
        self.params_table = QTableWidget(5, 3)  # 5 parameters, 3 columns
        self.params_table.setHorizontalHeaderLabels(["Parameter", "Calculated", "Adjusted"])
        
        # Configure table
        header = self.params_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        
        self.params_table.setMaximumHeight(180)
        
        # Initialize parameter rows
        param_names = ["Translation X", "Translation Y", "Rotation (°)", "Scale X", "Scale Y"]
        self._param_spinboxes = {}
        
        for i, param_name in enumerate(param_names):
            # Parameter name
            name_item = QTableWidgetItem(param_name)
            name_item.setFlags(Qt.ItemIsEnabled)
            self.params_table.setItem(i, 0, name_item)
            
            # Calculated value (read-only)
            calc_item = QTableWidgetItem("-")
            calc_item.setFlags(Qt.ItemIsEnabled)
            calc_item.setTextAlignment(Qt.AlignCenter)
            self.params_table.setItem(i, 1, calc_item)
            
            # Create spinbox for adjustment
            spinbox = QDoubleSpinBox()
            spinbox.setEnabled(False)
            spinbox.valueChanged.connect(lambda v, name=param_name: self._parameter_adjusted(name, v))
            
            # Set ranges based on parameter type
            if "Translation" in param_name:
                spinbox.setRange(-self._max_translation, self._max_translation)
                spinbox.setSuffix(" px")
                spinbox.setDecimals(1)
            elif "Rotation" in param_name:
                spinbox.setRange(0, 270)
                spinbox.setSingleStep(90)
                spinbox.setSuffix("°")
                spinbox.setDecimals(0)
            elif "Scale" in param_name:
                spinbox.setRange(self._min_scale_factor, self._max_scale_factor)
                spinbox.setSingleStep(0.1)
                spinbox.setDecimals(3)
            
            self._param_spinboxes[param_name] = spinbox
            self.params_table.setCellWidget(i, 2, spinbox)
        
        params_layout.addWidget(self.params_table)
        
        # Parameter summary
        self.summary_label = QLabel("Summary: No transformation calculated")
        self.summary_label.setWordWrap(True)
        params_layout.addWidget(self.summary_label)
        
        # Additional transformation information
        self.additional_info_label = QLabel("Additional information will appear here")
        self.additional_info_label.setWordWrap(True)
        self.additional_info_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                font-size: 11px;
                line-height: 1.4;
            }
        """)
        params_layout.addWidget(self.additional_info_label)
        
        # Manual adjustment controls
        adjustment_group = QGroupBox("Parameter Fine-Tuning")
        adjustment_layout = QVBoxLayout(adjustment_group)
        
        # Manual adjustment toggle
        self.manual_adjustment_cb = QCheckBox("Enable Fine-Tuning")
        self.manual_adjustment_cb.setToolTip("Allow manual adjustment of calculated transformation parameters")
        self.manual_adjustment_cb.toggled.connect(self._toggle_manual_adjustment)
        adjustment_layout.addWidget(self.manual_adjustment_cb)
        
        # Adjustment controls layout
        adjustment_controls_layout = QHBoxLayout()
        
        # Reset to calculated values button
        self.reset_params_btn = QPushButton("Reset to Calculated")
        self.reset_params_btn.setEnabled(False)
        self.reset_params_btn.setToolTip("Reset all parameters to originally calculated values")
        self.reset_params_btn.clicked.connect(self._reset_to_calculated)
        adjustment_controls_layout.addWidget(self.reset_params_btn)
        
        # Apply common adjustments dropdown
        self.quick_adjust_btn = QPushButton("Quick Adjustments ▼")
        self.quick_adjust_btn.setEnabled(False)
        self.quick_adjust_btn.setToolTip("Apply common transformation adjustments")
        self.quick_adjust_btn.clicked.connect(self._show_quick_adjustments)
        adjustment_controls_layout.addWidget(self.quick_adjust_btn)
        
        adjustment_layout.addLayout(adjustment_controls_layout)
        
        # Fine adjustment indicators
        self.adjustment_status_label = QLabel("No adjustments made")
        self.adjustment_status_label.setStyleSheet("""
            QLabel {
                padding: 3px;
                font-size: 10px;
                color: #666;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 3px;
            }
        """)
        adjustment_layout.addWidget(self.adjustment_status_label)
        
        params_layout.addWidget(adjustment_group)
        
        layout.addWidget(params_group)
        
        # Validation section
        validation_group = QGroupBox("Validation Results")
        validation_layout = QVBoxLayout(validation_group)
        
        # Validation status
        self.validation_label = QLabel("No calculation performed")
        self.validation_label.setWordWrap(True)
        validation_layout.addWidget(self.validation_label)
        
        # Validation details
        self.validation_details = QTextEdit()
        self.validation_details.setMaximumHeight(100)
        self.validation_details.setReadOnly(True)
        validation_layout.addWidget(self.validation_details)
        
        layout.addWidget(validation_group)
        
        return panel
    
    def _create_preview_panel(self) -> QWidget:
        """Create transformation preview panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Preview section
        preview_group = QGroupBox("Transformation Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Preview controls
        preview_controls = QHBoxLayout()
        
        self.generate_preview_btn = QPushButton("Generate Preview")
        self.generate_preview_btn.clicked.connect(self._generate_preview)
        self.generate_preview_btn.setEnabled(False)
        preview_controls.addWidget(self.generate_preview_btn)
        
        self.auto_preview_cb = QCheckBox("Auto Preview")
        self.auto_preview_cb.setChecked(True)
        preview_controls.addWidget(self.auto_preview_cb)
        
        preview_controls.addStretch()
        preview_layout.addLayout(preview_controls)
        
        # Preview status
        self.preview_status_label = QLabel("No preview generated")
        preview_layout.addWidget(self.preview_status_label)
        
        # Note about preview (since we can't display images in this text-based implementation)
        preview_note = QLabel("Preview: Transformed GDS overlaid on SEM image\n"
                             "(Preview image data will be emitted via signals for display)")
        preview_note.setStyleSheet("color: #666; font-style: italic; padding: 10px; "
                                 "background-color: #f5f5f5; border-radius: 5px;")
        preview_note.setWordWrap(True)
        preview_layout.addWidget(preview_note)
        
        layout.addWidget(preview_group)
        
        # Quality assessment section
        quality_group = QGroupBox("Transformation Quality")
        quality_layout = QVBoxLayout(quality_group)
        
        # Quality metrics
        self.quality_table = QTableWidget(4, 2)  # 4 metrics, 2 columns
        self.quality_table.setHorizontalHeaderLabels(["Metric", "Value"])
        
        # Configure quality table
        quality_header = self.quality_table.horizontalHeader()
        quality_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        quality_header.setSectionResizeMode(1, QHeaderView.Stretch)
        
        self.quality_table.setMaximumHeight(140)
        
        # Initialize quality metrics
        quality_metrics = ["Point Accuracy", "Scale Uniformity", "Distortion Level", "Overall Quality"]
        for i, metric_name in enumerate(quality_metrics):
            name_item = QTableWidgetItem(metric_name)
            name_item.setFlags(Qt.ItemIsEnabled)
            self.quality_table.setItem(i, 0, name_item)
            
            value_item = QTableWidgetItem("-")
            value_item.setFlags(Qt.ItemIsEnabled)
            value_item.setTextAlignment(Qt.AlignCenter)
            self.quality_table.setItem(i, 1, value_item)
        
        quality_layout.addWidget(self.quality_table)
        
        layout.addWidget(quality_group)
        
        return panel
    
    def _create_action_buttons(self) -> QHBoxLayout:
        """Create action buttons layout."""
        layout = QHBoxLayout()
        
        # Reset button
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self._reset_transformation)
        layout.addWidget(self.reset_btn)
        
        layout.addStretch()
        
        # Reject button
        self.reject_btn = QPushButton("Reject Transformation")
        self.reject_btn.clicked.connect(self._reject_transformation)
        self.reject_btn.setEnabled(False)
        layout.addWidget(self.reject_btn)
        
        # Confirm button
        self.confirm_btn = QPushButton("Confirm & Apply Transformation")
        self.confirm_btn.clicked.connect(self._confirm_transformation)
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.setStyleSheet("QPushButton:enabled { background-color: #4CAF50; color: white; font-weight: bold; }")
        layout.addWidget(self.confirm_btn)
        
        return layout
    
    # Main functionality methods
    def set_point_pairs(self, sem_points: List[Tuple[float, float]], gds_points: List[Tuple[float, float]]):
        """
        Receive and store point pairs for transformation calculation.
        
        Args:
            sem_points: List of SEM image points [(x, y), ...]
            gds_points: List of corresponding GDS points [(x, y), ...]
        """
        try:
            # Validate input
            if not isinstance(sem_points, list) or not isinstance(gds_points, list):
                raise ValueError("Point pairs must be provided as lists")
            
            if len(sem_points) != len(gds_points):
                raise ValueError(f"Mismatch in point counts: {len(sem_points)} SEM vs {len(gds_points)} GDS")
            
            # Validate point format
            self._validate_point_format(sem_points, "SEM")
            self._validate_point_format(gds_points, "GDS")
            
            # Store point pairs safely
            self._sem_points = [tuple(point) for point in sem_points]
            self._gds_points = [tuple(point) for point in gds_points]
            
            # Clear any previous calculation results
            self._calculated_transform = None
            self._transform_matrix = None
            self._is_valid = False
            self._preview_image = None
            
            # Update UI state
            self._update_calculation_button_state()
            
            logger.info(f"Point pairs stored successfully: {len(self._sem_points)} pairs")
            
        except Exception as e:
            logger.error(f"Failed to set point pairs: {e}")
            # Reset to safe state
            self._sem_points = []
            self._gds_points = []
            self._update_calculation_button_state()
            raise
    
    def _validate_point_format(self, points: List, point_type: str):
        """Validate that points are in correct format."""
        for i, point in enumerate(points):
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                raise ValueError(f"{point_type} point {i} must be a tuple/list of 2 coordinates")
            
            try:
                float(point[0])
                float(point[1])
            except (ValueError, TypeError):
                raise ValueError(f"{point_type} point {i} coordinates must be numeric")
    
    def _update_calculation_button_state(self):
        """Enable/disable transformation calculation button based on point selection."""
        # Check if we have exactly 3 points for both images
        sem_count = len(self._sem_points)
        gds_count = len(self._gds_points)
        has_required_points = sem_count == 3 and gds_count == 3
        
        # Enable button only when exactly 3 points are selected for both images
        self.calculate_btn.setEnabled(has_required_points)
        
        # Update status labels with detailed feedback
        self._update_point_selection_feedback(sem_count, gds_count, has_required_points)
        
        if has_required_points:
            # Auto-calculate if enabled
            if self.auto_preview_cb.isChecked():
                QTimer.singleShot(100, self._calculate_transformation)
        
        logger.debug(f"Calculate button enabled: {has_required_points}")
    
    def _update_point_selection_feedback(self, sem_count: int, gds_count: int, ready: bool):
        """Update UI feedback for point selection status."""
        if ready:
            # Ready state - green indication
            self.calc_status_label.setText("✓ Ready to calculate transformation")
            self.calc_status_label.setStyleSheet("color: #28a745; font-weight: bold;")
            
            self.points_status_label.setText("✓ 3 SEM points, 3 GDS points selected")
            self.points_status_label.setStyleSheet("""
                QLabel {
                    padding: 4px;
                    border: 1px solid #28a745;
                    border-radius: 3px;
                    background-color: #d4edda;
                    color: #155724;
                    font-size: 11px;
                }
            """)
        
        elif sem_count == 0 and gds_count == 0:
            # No points selected
            self.calc_status_label.setText("Select 3 corresponding points on both images")
            self.calc_status_label.setStyleSheet("color: #6c757d;")
            
            self.points_status_label.setText("No points selected")
            self.points_status_label.setStyleSheet("""
                QLabel {
                    padding: 4px;
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    background-color: #f5f5f5;
                    color: #6c757d;
                    font-size: 11px;
                }
            """)
        
        else:
            # Partial selection - orange/yellow indication
            remaining_sem = max(0, 3 - sem_count)
            remaining_gds = max(0, 3 - gds_count)
            
            if remaining_sem > 0 or remaining_gds > 0:
                status_parts = []
                if remaining_sem > 0:
                    status_parts.append(f"{remaining_sem} more SEM point{'s' if remaining_sem > 1 else ''}")
                if remaining_gds > 0:
                    status_parts.append(f"{remaining_gds} more GDS point{'s' if remaining_gds > 1 else ''}")
                
                self.calc_status_label.setText(f"Need {' and '.join(status_parts)}")
                self.calc_status_label.setStyleSheet("color: #fd7e14;")
                
                self.points_status_label.setText(f"{sem_count}/3 SEM points, {gds_count}/3 GDS points")
                self.points_status_label.setStyleSheet("""
                    QLabel {
                        padding: 4px;
                        border: 1px solid #fd7e14;
                        border-radius: 3px;
                        background-color: #fff3cd;
                        color: #856404;
                        font-size: 11px;
                    }
                """)
    
    def set_images_for_preview(self, gds_image: np.ndarray, sem_image: np.ndarray):
        """
        Set images for transformation preview generation.
        
        Args:
            gds_image: Original GDS binary image
            sem_image: SEM reference image
        """
        self._original_gds_image = gds_image.copy() if gds_image is not None else None
        self._sem_reference_image = sem_image.copy() if sem_image is not None else None
        
        # Enable preview generation if we have calculated transform
        if self._calculated_transform is not None:
            self.generate_preview_btn.setEnabled(True)
        
        logger.info("Images set for preview generation")
    
    def _calculate_transformation(self):
        """
        Trigger transformation calculation from point pairs.
        This method serves as the main entry point for transformation calculation.
        """
        # Pre-calculation validation
        if not self._can_calculate_transformation():
            return
        
        try:
            # Start calculation process
            self._start_calculation_ui()
            
            # Perform the actual calculation
            success = self._perform_transformation_calculation()
            
            if success:
                self._handle_calculation_success()
            else:
                self._handle_calculation_failure("Calculation failed")
                
        except Exception as e:
            self._handle_calculation_failure(f"Calculation error: {str(e)}")
        finally:
            self._finish_calculation_ui()
    
    def _can_calculate_transformation(self) -> bool:
        """Check if transformation calculation can proceed."""
        if len(self._sem_points) != 3 or len(self._gds_points) != 3:
            self._show_error("Need exactly 3 point pairs for calculation")
            return False
        
        # Check for duplicate points
        if len(set(self._sem_points)) != 3 or len(set(self._gds_points)) != 3:
            self._show_error("Points must be unique (no duplicates)")
            return False
        
        # Check for collinear points (basic check)
        if self._are_points_collinear(self._sem_points) or self._are_points_collinear(self._gds_points):
            self._show_error("Points must not be collinear")
            return False
        
        return True
    
    def _start_calculation_ui(self):
        """Setup UI for calculation process."""
        self.calc_progress.setVisible(True)
        self.calc_progress.setValue(0)
        self.calc_status_label.setText("Calculating transformation...")
        self.calc_status_label.setStyleSheet("color: #007bff;")
        self.calculate_btn.setEnabled(False)
        
    def _perform_transformation_calculation(self) -> bool:
        """
        Calculate the affine transformation matrix from 3-point pairs.
        
        This method performs the core affine transformation calculation using OpenCV's
        getAffineTransform function, which solves for the 2x3 affine matrix that maps
        the source points to destination points.
        
        Returns:
            bool: True if calculation succeeded, False otherwise
        """
        try:
            # Convert points to numpy arrays with proper data type
            sem_pts = np.array(self._sem_points, dtype=np.float32)
            gds_pts = np.array(self._gds_points, dtype=np.float32)
            
            logger.debug(f"SEM points for transformation: {sem_pts}")
            logger.debug(f"GDS points for transformation: {gds_pts}")
            
            self.calc_progress.setValue(25)
            
            # Calculate 2x3 affine transformation matrix using OpenCV
            # This solves the equation: gds_pts = M * sem_pts for matrix M
            transform_matrix_2x3 = cv2.getAffineTransform(sem_pts, gds_pts)
            
            logger.debug(f"Calculated 2x3 matrix: {transform_matrix_2x3}")
            
            # Convert to 3x3 homogeneous transformation matrix
            # Format: [[a, b, tx], [c, d, ty], [0, 0, 1]]
            # Where: [x', y'] = [a*x + b*y + tx, c*x + d*y + ty]
            self._transform_matrix = np.eye(3, dtype=np.float64)
            self._transform_matrix[:2, :] = transform_matrix_2x3.astype(np.float64)
            
            logger.debug(f"Full 3x3 transformation matrix: {self._transform_matrix}")
            
            self.calc_progress.setValue(50)
            
            # Verify the transformation by applying it to source points
            verification_success = self._verify_transformation_accuracy()
            if not verification_success:
                logger.warning("Transformation verification failed - results may be inaccurate")
            
            # Extract transformation parameters (translation, rotation, scaling)
            params = self._extract_transformation_parameters(self._transform_matrix)
            
            self.calc_progress.setValue(75)
            
            # Validate transformation parameters
            validation_result = self._validate_transformation(params)
            
            self.calc_progress.setValue(100)
            
            # Store results
            self._calculated_transform = params
            self._is_valid = validation_result['is_valid']
            
            logger.info(f"Transformation calculation completed. Valid: {self._is_valid}")
            return True
            
        except Exception as e:
            logger.error(f"Affine transformation calculation failed: {e}")
            return False
    
    def _verify_transformation_accuracy(self) -> bool:
        """
        Verify the calculated transformation by applying it to source points
        and checking if they match the target points within tolerance.
        
        Returns:
            bool: True if verification passes, False otherwise
        """
        try:
            # Apply transformation to SEM points
            sem_pts_homogeneous = np.column_stack([self._sem_points, np.ones(3)])
            transformed_pts = (self._transform_matrix @ sem_pts_homogeneous.T).T[:, :2]
            
            # Calculate error between transformed points and target GDS points
            errors = transformed_pts - np.array(self._gds_points)
            max_error = np.max(np.linalg.norm(errors, axis=1))
            
            # Tolerance of 1 pixel
            tolerance = 1.0
            verification_passed = max_error < tolerance
            
            logger.debug(f"Transformation verification - Max error: {max_error:.3f}, Tolerance: {tolerance}")
            
            return verification_passed
            
        except Exception as e:
            logger.error(f"Transformation verification failed: {e}")
            return False
    
    def _handle_calculation_success(self):
        """Handle successful transformation calculation."""
        # Update UI displays
        self._update_parameters_display(self._calculated_transform)
        validation_result = self._validate_transformation(self._calculated_transform)
        self._update_validation_display(validation_result)
        self._update_quality_metrics(self._calculated_transform)
        
        # Update control states
        self.generate_preview_btn.setEnabled(self._original_gds_image is not None)
        self.confirm_btn.setEnabled(self._is_valid)
        self.reject_btn.setEnabled(True)
        
        # Emit signals
        self.transformation_calculated.emit(self._calculated_transform)
        self.validation_status_changed.emit(self._is_valid, validation_result['message'])
        
        # Auto-generate preview if enabled
        if self.auto_preview_cb.isChecked() and self._is_valid:
            QTimer.singleShot(200, self._generate_preview)
        
        logger.info("Transformation calculation completed successfully")
    
    def _handle_calculation_failure(self, error_message: str):
        """Handle failed transformation calculation."""
        self._show_error(error_message)
        self._calculated_transform = None
        self._transform_matrix = None
        self._is_valid = False
        
        # Reset UI state
        self.confirm_btn.setEnabled(False)
        self.generate_preview_btn.setEnabled(False)
        
        logger.error(f"Transformation calculation failed: {error_message}")
    
    def _finish_calculation_ui(self):
        """Clean up UI after calculation."""
        self.calc_progress.setVisible(False)
        self.calculate_btn.setEnabled(True)
        
        if self._calculated_transform:
            self.calc_status_label.setText("Transformation calculated successfully")
            self.calc_status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        else:
            self.calc_status_label.setText("Calculation failed")
            self.calc_status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
    
    def _are_points_collinear(self, points: List[Tuple[float, float]]) -> bool:
        """Check if three points are collinear."""
        if len(points) != 3:
            return False
        
        # Calculate cross product to check collinearity
        p1, p2, p3 = points
        cross_product = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        
        # Points are collinear if cross product is near zero
        return abs(cross_product) < 1e-6
    
    def _extract_transformation_parameters(self, matrix: np.ndarray) -> Dict[str, float]:
        """
        Extract transformation parameters from affine matrix.
        
        This method decomposes the 3x3 affine transformation matrix into its
        constituent transformation components: translation, rotation, and scaling.
        Rotation is snapped to 90-degree increments as required by Step 11.
        
        Args:
            matrix: 3x3 affine transformation matrix in format:
                   [[a, b, tx], [c, d, ty], [0, 0, 1]]
                   
        Returns:
            Dictionary with extracted transformation parameters
        """
        try:
            # Extract translation components (rightmost column)
            tx = float(matrix[0, 2])
            ty = float(matrix[1, 2])
            
            # Extract the 2x2 linear transformation submatrix
            # Matrix format: [[a, b], [c, d]] where transformation is [x', y'] = [[a,b],[c,d]] * [x, y]
            a, b = float(matrix[0, 0]), float(matrix[0, 1])
            c, d = float(matrix[1, 0]), float(matrix[1, 1])
            
            logger.debug(f"Matrix components - a:{a:.6f}, b:{b:.6f}, c:{c:.6f}, d:{d:.6f}, tx:{tx:.3f}, ty:{ty:.3f}")
            
            # Calculate scale factors using singular value decomposition for accuracy
            linear_matrix = matrix[:2, :2]
            scale_x, scale_y = self._extract_scale_factors(linear_matrix)
            
            # Calculate rotation angle
            rotation_rad, rotation_deg = self._extract_rotation_angle(linear_matrix, scale_x, scale_y)
            
            # Snap rotation to 90-degree increments (Step 11 requirement)
            rotation_snapped = self._snap_to_90_degrees(rotation_deg)
            
            # Calculate additional quality metrics
            determinant = np.linalg.det(linear_matrix)
            condition_number = np.linalg.cond(linear_matrix)
            
            # Determine transformation type
            transform_type = self._classify_transformation_type(scale_x, scale_y, rotation_deg, determinant)
            
            parameters = {
                'translation_x': tx,
                'translation_y': ty,
                'rotation': rotation_snapped,
                'rotation_raw': rotation_deg,
                'scale_x': scale_x,
                'scale_y': scale_y,
                'scale_uniform': (scale_x + scale_y) / 2,  # Average scale for uniform scaling
                'determinant': determinant,
                'condition_number': condition_number,
                'transform_type': transform_type,
                'transform_matrix': matrix.tolist(),
                'is_invertible': abs(determinant) > 1e-10
            }
            
            logger.info(f"Extracted parameters: tx={tx:.2f}, ty={ty:.2f}, rot={rotation_snapped:.1f}°, "
                       f"scale_x={scale_x:.3f}, scale_y={scale_y:.3f}, type={transform_type}")
            
            return parameters
            
        except Exception as e:
            logger.error(f"Failed to extract transformation parameters: {e}")
            # Return default/safe parameters
            return {
                'translation_x': 0.0,
                'translation_y': 0.0,
                'rotation': 0.0,
                'rotation_raw': 0.0,
                'scale_x': 1.0,
                'scale_y': 1.0,
                'scale_uniform': 1.0,
                'determinant': 0.0,
                'condition_number': float('inf'),
                'transform_type': 'invalid',
                'transform_matrix': np.eye(3).tolist(),
                'is_invertible': False
            }
    
    def _extract_scale_factors(self, linear_matrix: np.ndarray) -> Tuple[float, float]:
        """Extract scale factors using SVD for numerical stability."""
        try:
            # Use SVD to get accurate scale factors
            U, S, Vt = np.linalg.svd(linear_matrix)
            scale_x, scale_y = float(S[0]), float(S[1])
            
            # Handle reflection (negative determinant)
            if np.linalg.det(linear_matrix) < 0:
                scale_y = -scale_y
                
            return abs(scale_x), abs(scale_y)
            
        except Exception:
            # Fallback to simple calculation
            a, b = linear_matrix[0, 0], linear_matrix[0, 1]
            c, d = linear_matrix[1, 0], linear_matrix[1, 1]
            scale_x = np.sqrt(a*a + c*c)
            scale_y = np.sqrt(b*b + d*d)
            return float(scale_x), float(scale_y)
    
    def _extract_rotation_angle(self, linear_matrix: np.ndarray, scale_x: float, scale_y: float) -> Tuple[float, float]:
        """Extract rotation angle from the linear transformation matrix."""
        try:
            # Remove scaling to get pure rotation
            if scale_x > 1e-10 and scale_y > 1e-10:
                # Normalize by scale factors
                normalized_matrix = linear_matrix / np.array([[scale_x, scale_y], [scale_x, scale_y]])
                rotation_rad = np.arctan2(normalized_matrix[1, 0], normalized_matrix[0, 0])
            else:
                # Fallback: direct calculation from matrix elements
                rotation_rad = np.arctan2(linear_matrix[1, 0], linear_matrix[0, 0])
            
            rotation_deg = np.degrees(rotation_rad)
            
            # Normalize to [0, 360) range
            rotation_deg = rotation_deg % 360
            
            return float(rotation_rad), float(rotation_deg)
            
        except Exception:
            return 0.0, 0.0
    
    def _classify_transformation_type(self, scale_x: float, scale_y: float, 
                                    rotation_deg: float, determinant: float) -> str:
        """Classify the type of transformation based on parameters."""
        tolerance = 1e-3
        
        if abs(determinant) < tolerance:
            return "degenerate"
        elif determinant < 0:
            return "reflection"
        elif abs(scale_x - 1.0) < tolerance and abs(scale_y - 1.0) < tolerance:
            if abs(rotation_deg) < tolerance or abs(rotation_deg - 360) < tolerance:
                return "translation"
            else:
                return "rigid"  # rotation + translation
        elif abs(scale_x - scale_y) < tolerance:
            return "similarity"  # uniform scaling + rotation + translation
        else:
            return "affine"  # general affine transformation
    
    def _snap_to_90_degrees(self, angle: float) -> float:
        """Snap angle to nearest 90-degree increment."""
        # Round to nearest 90-degree increment
        snapped = round(angle / 90.0) * 90.0
        # Normalize to [0, 360) range
        return snapped % 360
    
    def _validate_transformation(self, params: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate transformation parameters to check for extreme distortions or invalid transformations.
        
        This method implements comprehensive validation logic to ensure the calculated
        transformation is reasonable and suitable for image alignment.
        
        Args:
            params: Transformation parameters dictionary
            
        Returns:
            Validation result dictionary with detailed assessment
        """
        issues = []
        warnings = []
        info_messages = []
        
        try:
            # Extract parameters for validation
            scale_x = params.get('scale_x', 1.0)
            scale_y = params.get('scale_y', 1.0)
            translation_x = params.get('translation_x', 0.0)
            translation_y = params.get('translation_y', 0.0)
            rotation = params.get('rotation', 0.0)
            rotation_raw = params.get('rotation_raw', 0.0)
            determinant = params.get('determinant', 1.0)
            condition_number = params.get('condition_number', 1.0)
            transform_type = params.get('transform_type', 'unknown')
            
            # 1. Validate scale factors (check for extreme distortions)
            self._validate_scale_factors(scale_x, scale_y, issues, warnings)
            
            # 2. Validate translation magnitude
            self._validate_translation(translation_x, translation_y, issues, warnings)
            
            # 3. Validate rotation and snapping
            self._validate_rotation(rotation, rotation_raw, issues, warnings, info_messages)
            
            # 4. Validate matrix properties
            self._validate_matrix_properties(determinant, condition_number, issues, warnings)
            
            # 5. Validate transformation type
            self._validate_transformation_type(transform_type, scale_x, scale_y, issues, warnings)
            
            # 6. Cross-parameter validation
            self._validate_parameter_consistency(params, issues, warnings)
            
            # Overall assessment
            is_valid = len(issues) == 0
            quality_score = self._calculate_quality_score(params, issues, warnings)
            status = self._determine_validation_status(is_valid, len(warnings), quality_score)
            message = self._generate_validation_message(is_valid, len(issues), len(warnings), status)
            
            validation_result = {
                'is_valid': is_valid,
                'status': status,
                'message': message,
                'issues': issues,
                'warnings': warnings,
                'info_messages': info_messages,
                'quality_score': quality_score,
                'detailed_assessment': self._create_detailed_assessment(params, is_valid, quality_score)
            }
            
            logger.info(f"Transformation validation: {status} (score: {quality_score:.2f})")
            if issues:
                logger.warning(f"Validation issues: {'; '.join(issues)}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                'is_valid': False,
                'status': 'error',
                'message': f"Validation error: {str(e)}",
                'issues': [f"Validation error: {str(e)}"],
                'warnings': [],
                'info_messages': [],
                'quality_score': 0.0,
                'detailed_assessment': "Validation process failed"
            }
    
    def _validate_scale_factors(self, scale_x: float, scale_y: float, issues: List[str], warnings: List[str]):
        """Validate scale factors for extreme distortions."""
        # Check individual scale factors
        if scale_x < self._min_scale_factor:
            issues.append(f"X scale too small: {scale_x:.3f} (min: {self._min_scale_factor})")
        elif scale_x > self._max_scale_factor:
            issues.append(f"X scale too large: {scale_x:.3f} (max: {self._max_scale_factor})")
        
        if scale_y < self._min_scale_factor:
            issues.append(f"Y scale too small: {scale_y:.3f} (min: {self._min_scale_factor})")
        elif scale_y > self._max_scale_factor:
            issues.append(f"Y scale too large: {scale_y:.3f} (max: {self._max_scale_factor})")
        
        # Check for significant scale differences (aspect ratio distortion)
        if scale_x > 0 and scale_y > 0:
            scale_ratio = max(scale_x, scale_y) / min(scale_x, scale_y)
            if scale_ratio > 3.0:
                issues.append(f"Excessive aspect ratio distortion: {scale_ratio:.2f}x")
            elif scale_ratio > 2.0:
                warnings.append(f"Significant scale difference: {scale_ratio:.2f}x")
            elif scale_ratio > 1.5:
                warnings.append(f"Moderate scale difference: {scale_ratio:.2f}x")
    
    def _validate_translation(self, tx: float, ty: float, issues: List[str], warnings: List[str]):
        """Validate translation parameters."""
        translation_magnitude = np.sqrt(tx*tx + ty*ty)
        
        if translation_magnitude > self._max_translation:
            issues.append(f"Excessive translation: {translation_magnitude:.1f} pixels (max: {self._max_translation})")
        elif translation_magnitude > self._max_translation * 0.7:
            warnings.append(f"Large translation: {translation_magnitude:.1f} pixels")
        
        # Check for extreme individual translations
        if abs(tx) > self._max_translation * 0.8:
            warnings.append(f"Large X translation: {tx:.1f} pixels")
        if abs(ty) > self._max_translation * 0.8:
            warnings.append(f"Large Y translation: {ty:.1f} pixels")
    
    def _validate_rotation(self, rotation: float, rotation_raw: float, 
                          issues: List[str], warnings: List[str], info_messages: List[str]):
        """Validate rotation parameters and snapping."""
        # Check rotation snapping effectiveness
        rotation_diff = abs(rotation - rotation_raw)
        if rotation_diff > 45:
            warnings.append(f"Large rotation adjustment: {rotation_diff:.1f}° snapped")
        elif rotation_diff > 20:
            warnings.append(f"Moderate rotation adjustment: {rotation_diff:.1f}° snapped")
        elif rotation_diff > 5:
            info_messages.append(f"Minor rotation adjustment: {rotation_diff:.1f}° snapped")
        
        # Provide information about the snapped rotation
        if rotation in [0, 90, 180, 270]:
            info_messages.append(f"Clean {rotation}° rotation applied")
    
    def _validate_matrix_properties(self, determinant: float, condition_number: float,
                                   issues: List[str], warnings: List[str]):
        """Validate matrix mathematical properties."""
        # Check for degenerate transformation (determinant near zero)
        if abs(determinant) < 1e-10:
            issues.append("Degenerate transformation (zero determinant)")
        elif abs(determinant) < 1e-6:
            warnings.append("Near-degenerate transformation (very small determinant)")
        
        # Check for reflection (negative determinant)
        if determinant < 0:
            warnings.append("Transformation includes reflection")
        
        # Check numerical stability (condition number)
        if condition_number > 1000:
            warnings.append(f"Poor numerical stability (condition: {condition_number:.1e})")
        elif condition_number > 100:
            warnings.append(f"Moderate numerical instability (condition: {condition_number:.1f})")
    
    def _validate_transformation_type(self, transform_type: str, scale_x: float, scale_y: float,
                                     issues: List[str], warnings: List[str]):
        """Validate transformation type and characteristics."""
        if transform_type == "degenerate":
            issues.append("Degenerate transformation detected")
        elif transform_type == "invalid":
            issues.append("Invalid transformation parameters")
        elif transform_type == "reflection":
            warnings.append("Transformation includes reflection component")
        
        # Additional type-specific validation
        if transform_type in ["translation", "rigid"] and (abs(scale_x - 1.0) > 0.1 or abs(scale_y - 1.0) > 0.1):
            warnings.append(f"Unexpected scaling in {transform_type} transformation")
    
    def _validate_parameter_consistency(self, params: Dict[str, float], 
                                       issues: List[str], warnings: List[str]):
        """Validate consistency between different parameters."""
        # Check if the transformation is invertible
        if not params.get('is_invertible', True):
            issues.append("Non-invertible transformation")
        
        # Check for NaN or infinite values
        for key, value in params.items():
            if isinstance(value, (int, float)):
                if np.isnan(value):
                    issues.append(f"Invalid parameter {key}: NaN")
                elif np.isinf(value):
                    issues.append(f"Invalid parameter {key}: infinite")
    
    def _determine_validation_status(self, is_valid: bool, warning_count: int, quality_score: float) -> str:
        """Determine overall validation status."""
        if not is_valid:
            return "invalid"
        elif quality_score >= 0.9:
            return "excellent"
        elif quality_score >= 0.7:
            return "good"
        elif quality_score >= 0.5:
            return "acceptable"
        else:
            return "poor"
    
    def _generate_validation_message(self, is_valid: bool, issue_count: int, 
                                   warning_count: int, status: str) -> str:
        """Generate a comprehensive validation message."""
        if not is_valid:
            return f"Transformation invalid: {issue_count} critical issue(s) found"
        elif warning_count == 0:
            return "Transformation parameters are valid and optimal"
        else:
            return f"Transformation valid ({status}) with {warning_count} consideration(s)"
    
    def _create_detailed_assessment(self, params: Dict[str, float], is_valid: bool, quality_score: float) -> str:
        """Create a detailed assessment summary."""
        assessment_parts = []
        
        if is_valid:
            assessment_parts.append(f"✓ Valid transformation (quality: {quality_score:.1%})")
        else:
            assessment_parts.append("✗ Invalid transformation")
        
        # Add parameter summary
        tx, ty = params.get('translation_x', 0), params.get('translation_y', 0)
        rot = params.get('rotation', 0)
        sx, sy = params.get('scale_x', 1), params.get('scale_y', 1)
        
        assessment_parts.append(f"Translation: ({tx:.1f}, {ty:.1f}) pixels")
        assessment_parts.append(f"Rotation: {rot:.0f}°")
        assessment_parts.append(f"Scale: {sx:.2f} × {sy:.2f}")
        assessment_parts.append(f"Type: {params.get('transform_type', 'unknown')}")
        
        return " | ".join(assessment_parts)
    
    def _calculate_quality_score(self, params: Dict[str, float], issues: List[str], warnings: List[str]) -> float:
        """Calculate overall quality score (0-1)."""
        if issues:
            return 0.0
        
        score = 1.0
        
        # Deduct for warnings
        score -= len(warnings) * 0.1
        
        # Deduct for scale distortion
        scale_ratio = max(params['scale_x'], params['scale_y']) / min(params['scale_x'], params['scale_y'])
        score -= max(0, (scale_ratio - 1.0) * 0.2)
        
        # Deduct for large rotation adjustments
        rotation_diff = abs(params['rotation'] - params['rotation_raw'])
        score -= rotation_diff / 180.0 * 0.3
        
        return max(0.0, min(1.0, score))
    
    def _update_parameters_display(self, params: Dict[str, float]):
        """
        Update the parameters table with calculated transformation values.
        
        This method displays transformation parameters in a clear, organized table
        with both calculated and user-adjustable values for review.
        
        Args:
            params: Dictionary containing transformation parameters
        """
        try:
            # Extract parameters with safe defaults
            translation_x = params.get('translation_x', 0.0)
            translation_y = params.get('translation_y', 0.0)
            rotation = params.get('rotation', 0.0)
            rotation_raw = params.get('rotation_raw', 0.0)
            scale_x = params.get('scale_x', 1.0)
            scale_y = params.get('scale_y', 1.0)
            scale_uniform = params.get('scale_uniform', 1.0)
            transform_type = params.get('transform_type', 'unknown')
            
            # Update main parameters table
            param_data = [
                ("Translation X", translation_x, "pixels", 1),
                ("Translation Y", translation_y, "pixels", 1),
                ("Rotation", rotation, "°", 0),
                ("Scale X", scale_x, "×", 3),
                ("Scale Y", scale_y, "×", 3)
            ]
            
            for i, (param_name, value, unit, decimals) in enumerate(param_data):
                if i < self.params_table.rowCount():
                    # Update calculated value with appropriate formatting
                    if decimals == 0:
                        display_value = f"{value:.0f}{unit}"
                    else:
                        display_value = f"{value:.{decimals}f} {unit}".strip()
                    
                    # Set calculated value (read-only column)
                    calc_item = self.params_table.item(i, 1)
                    if calc_item:
                        calc_item.setText(display_value)
                    
                    # Update corresponding spinbox for user adjustment
                    if i < len(self._param_spinboxes):
                        spinbox = list(self._param_spinboxes.values())[i]
                        if spinbox:
                            spinbox.blockSignals(True)
                            spinbox.setValue(value)
                            spinbox.blockSignals(False)
            
            # Update additional parameter information
            self._update_additional_parameters_info(params)
            
            # Update parameter summary
            self._update_parameter_summary(params)
            
            logger.debug("Parameters display updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update parameters display: {e}")
            self._show_error(f"Display update failed: {str(e)}")
    
    def _update_additional_parameters_info(self, params: Dict[str, float]):
        """Update additional transformation information display."""
        try:
            # Create additional info text
            info_lines = []
            
            # Transformation type and characteristics
            transform_type = params.get('transform_type', 'unknown')
            info_lines.append(f"Transformation Type: {transform_type.title()}")
            
            # Raw vs snapped rotation
            rotation_raw = params.get('rotation_raw', 0.0)
            rotation_snapped = params.get('rotation', 0.0)
            rotation_diff = abs(rotation_snapped - rotation_raw)
            if rotation_diff > 0.1:
                info_lines.append(f"Rotation Adjustment: {rotation_raw:.1f}° → {rotation_snapped:.0f}° (snapped)")
            else:
                info_lines.append(f"Rotation: {rotation_snapped:.0f}° (no adjustment needed)")
            
            # Scale information
            scale_x = params.get('scale_x', 1.0)
            scale_y = params.get('scale_y', 1.0)
            if abs(scale_x - scale_y) < 0.01:
                info_lines.append(f"Uniform Scaling: {scale_x:.3f}×")
            else:
                scale_ratio = max(scale_x, scale_y) / min(scale_x, scale_y)
                info_lines.append(f"Non-uniform Scaling: {scale_ratio:.2f}:1 ratio")
            
            # Translation magnitude
            tx, ty = params.get('translation_x', 0.0), params.get('translation_y', 0.0)
            translation_magnitude = np.sqrt(tx*tx + ty*ty)
            info_lines.append(f"Translation Distance: {translation_magnitude:.1f} pixels")
            
            # Matrix properties
            determinant = params.get('determinant', 1.0)
            if determinant < 0:
                info_lines.append("⚠ Includes reflection component")
            elif abs(determinant - 1.0) > 0.1:
                if determinant > 1.0:
                    info_lines.append(f"Area expansion: {determinant:.2f}×")
                else:
                    info_lines.append(f"Area reduction: {determinant:.2f}×")
            
            # Display the information
            if hasattr(self, 'additional_info_label'):
                self.additional_info_label.setText("\n".join(info_lines))
            
        except Exception as e:
            logger.error(f"Failed to update additional parameters info: {e}")
    
    def _update_parameter_summary(self, params: Dict[str, float]):
        """Update a concise parameter summary for quick review."""
        try:
            # Create a concise summary
            tx = params.get('translation_x', 0.0)
            ty = params.get('translation_y', 0.0)
            rotation = params.get('rotation', 0.0)
            scale_x = params.get('scale_x', 1.0)
            scale_y = params.get('scale_y', 1.0)
            
            # Format summary components
            summary_parts = []
            
            # Translation
            if abs(tx) > 0.1 or abs(ty) > 0.1:
                summary_parts.append(f"Translate: ({tx:+.1f}, {ty:+.1f})")
            
            # Rotation
            if abs(rotation) > 0.1:
                summary_parts.append(f"Rotate: {rotation:.0f}°")
            
            # Scale
            if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
                if abs(scale_x - scale_y) < 0.01:
                    summary_parts.append(f"Scale: {scale_x:.2f}×")
                else:
                    summary_parts.append(f"Scale: {scale_x:.2f}×{scale_y:.2f}")
            
            # Create final summary
            if summary_parts:
                summary = " | ".join(summary_parts)
            else:
                summary = "Identity transformation (no change)"
            
            # Update summary display
            if hasattr(self, 'summary_label'):
                self.summary_label.setText(f"Summary: {summary}")
                self.summary_label.setStyleSheet("""
                    QLabel {
                        font-weight: bold;
                        color: #2c3e50;
                        background-color: #ecf0f1;
                        padding: 6px;
                        border-radius: 4px;
                        margin: 2px 0px;
                    }
                """)
            
        except Exception as e:
            logger.error(f"Failed to update parameter summary: {e}")
    
    def _update_validation_display(self, validation: Dict[str, Any]):
        """
        Update validation status and warnings display.
        
        This method provides comprehensive feedback about transformation quality,
        including visual indicators, detailed messages, and actionable warnings.
        
        Args:
            validation: Validation result dictionary from _validate_transformation
        """
        try:
            status = validation.get('status', 'unknown')
            message = validation.get('message', 'No validation message')
            is_valid = validation.get('is_valid', False)
            issues = validation.get('issues', [])
            warnings = validation.get('warnings', [])
            info_messages = validation.get('info_messages', [])
            quality_score = validation.get('quality_score', 0.0)
            detailed_assessment = validation.get('detailed_assessment', '')
            
            # Update main status label with enhanced styling
            self._update_validation_status_label(status, message, is_valid, quality_score)
            
            # Update detailed validation information
            self._update_validation_details(issues, warnings, info_messages, detailed_assessment)
            
            # Update any visual indicators
            self._update_validation_indicators(is_valid, status, quality_score)
            
            # Emit validation status changed signal
            self.validation_status_changed.emit(is_valid, message)
            
            logger.debug(f"Validation display updated: {status} (valid: {is_valid})")
            
        except Exception as e:
            logger.error(f"Failed to update validation display: {e}")
            self._show_validation_error("Display update failed")
    
    def _update_validation_status_label(self, status: str, message: str, is_valid: bool, quality_score: float):
        """Update the main validation status label with color coding and icons."""
        # Determine color scheme and icon based on status
        status_configs = {
            "excellent": {"color": "#155724", "bg_color": "#d4edda", "border": "#c3e6cb", "icon": "✓"},
            "good": {"color": "#155724", "bg_color": "#d4edda", "border": "#c3e6cb", "icon": "✓"},
            "acceptable": {"color": "#856404", "bg_color": "#fff3cd", "border": "#ffeaa7", "icon": "⚠"},
            "poor": {"color": "#856404", "bg_color": "#fff3cd", "border": "#ffeaa7", "icon": "⚠"},
            "invalid": {"color": "#721c24", "bg_color": "#f8d7da", "border": "#f5c6cb", "icon": "✗"},
            "error": {"color": "#721c24", "bg_color": "#f8d7da", "border": "#f5c6cb", "icon": "⚠"}
        }
        
        config = status_configs.get(status, status_configs["invalid"])
        
        # Create enhanced message with icon and quality score
        if is_valid:
            enhanced_message = f"{config['icon']} {message}"
            if quality_score > 0:
                enhanced_message += f" (Quality: {quality_score:.0%})"
        else:
            enhanced_message = f"{config['icon']} {message}"
        
        # Apply styling
        self.validation_label.setText(enhanced_message)
        self.validation_label.setStyleSheet(f"""
            QLabel {{
                color: {config['color']};
                background-color: {config['bg_color']};
                border: 2px solid {config['border']};
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: bold;
                font-size: 13px;
            }}
        """)
    
    def _update_validation_details(self, issues: List[str], warnings: List[str], 
                                  info_messages: List[str], detailed_assessment: str):
        """Update detailed validation information display."""
        details_sections = []
        
        # Add detailed assessment if available
        if detailed_assessment:
            details_sections.append(f"Assessment: {detailed_assessment}")
            details_sections.append("")  # Empty line for spacing
        
        # Critical issues section
        if issues:
            details_sections.append("🚫 CRITICAL ISSUES:")
            for i, issue in enumerate(issues, 1):
                details_sections.append(f"   {i}. {issue}")
            details_sections.append("")
        
        # Warnings section
        if warnings:
            details_sections.append("⚠️  WARNINGS:")
            for i, warning in enumerate(warnings, 1):
                details_sections.append(f"   {i}. {warning}")
            details_sections.append("")
        
        # Information section
        if info_messages:
            details_sections.append("ℹ️  INFORMATION:")
            for i, info in enumerate(info_messages, 1):
                details_sections.append(f"   {i}. {info}")
            details_sections.append("")
        
        # If no issues, warnings, or info
        if not issues and not warnings and not info_messages:
            details_sections.append("✅ No issues, warnings, or special considerations detected.")
            details_sections.append("")
            details_sections.append("The transformation parameters appear optimal for image alignment.")
        
        # Recommendations section
        recommendations = self._generate_validation_recommendations(issues, warnings)
        if recommendations:
            details_sections.append("💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                details_sections.append(f"   {i}. {rec}")
        
        # Update the text widget
        details_text = "\n".join(details_sections)
        self.validation_details.setText(details_text)
        
        # Style the text widget based on validation status
        if issues:
            border_color = "#dc3545"
        elif warnings:
            border_color = "#ffc107"
        else:
            border_color = "#28a745"
        
        self.validation_details.setStyleSheet(f"""
            QTextEdit {{
                border: 1px solid {border_color};
                border-radius: 4px;
                background-color: #f8f9fa;
                font-family: monospace;
                font-size: 11px;
                padding: 4px;
            }}
        """)
    
    def _update_validation_indicators(self, is_valid: bool, status: str, quality_score: float):
        """
        Update visual indicators and enable/disable confirmation based on validation.
        
        This method implements Step 12 requirement to only allow confirmation
        when transformation passes validation checks.
        
        Args:
            is_valid: Whether transformation is valid
            status: Validation status string
            quality_score: Quality score (0-1)
        """
        # Update button states based on validation
        self._update_confirmation_button_state(is_valid, status, quality_score)
        self._update_rejection_button_state()
        self._update_preview_button_state()
        
        # Update any additional visual indicators
        self._update_visual_feedback(is_valid, status, quality_score)
    
    def _update_confirmation_button_state(self, is_valid: bool, status: str, quality_score: float):
        """Enable/disable confirmation button based on validation results."""
        try:
            # Only enable confirm button if transformation is valid
            self.confirm_btn.setEnabled(is_valid)
            
            # Update button appearance and text based on validation status
            if is_valid:
                if status == "excellent":
                    button_text = "✓ Confirm Excellent Transformation"
                    button_color = "#28a745"  # Green
                    tooltip = f"High quality transformation (Score: {quality_score:.0%}). Safe to apply."
                elif status == "good":
                    button_text = "✓ Confirm Good Transformation"
                    button_color = "#28a745"  # Green
                    tooltip = f"Good quality transformation (Score: {quality_score:.0%}). Recommended to apply."
                elif status == "acceptable":
                    button_text = "⚠ Confirm Acceptable Transformation"
                    button_color = "#ffc107"  # Yellow/Orange
                    tooltip = f"Acceptable transformation (Score: {quality_score:.0%}). Review warnings before applying."
                else:  # poor but still valid
                    button_text = "⚠ Confirm Poor Transformation"
                    button_color = "#fd7e14"  # Orange
                    tooltip = f"Poor quality transformation (Score: {quality_score:.0%}). Consider improving point selection."
                
                # Style for enabled button
                self.confirm_btn.setStyleSheet(f"""
                    QPushButton:enabled {{
                        background-color: {button_color};
                        color: white;
                        font-weight: bold;
                        border: 2px solid {button_color};
                        border-radius: 6px;
                        padding: 8px 16px;
                        font-size: 13px;
                    }}
                    QPushButton:enabled:hover {{
                        background-color: {self._darken_color(button_color)};
                        border-color: {self._darken_color(button_color)};
                    }}
                    QPushButton:enabled:pressed {{
                        background-color: {self._darken_color(button_color, 0.2)};
                    }}
                """)
            else:
                # Invalid transformation - button disabled
                button_text = "✗ Cannot Confirm Invalid Transformation"
                tooltip = "Transformation validation failed. Review issues and try different point selection."
                
                # Style for disabled button
                self.confirm_btn.setStyleSheet("""
                    QPushButton:disabled {
                        background-color: #6c757d;
                        color: #ffffff;
                        border: 2px solid #6c757d;
                        border-radius: 6px;
                        padding: 8px 16px;
                        font-size: 13px;
                        font-weight: bold;
                    }
                """)
            
            self.confirm_btn.setText(button_text)
            self.confirm_btn.setToolTip(tooltip)
            
            logger.debug(f"Confirm button state: enabled={is_valid}, status={status}")
            
        except Exception as e:
            logger.error(f"Failed to update confirmation button: {e}")
            # Safe fallback
            self.confirm_btn.setEnabled(False)
            self.confirm_btn.setText("Confirmation Error")
    
    def _update_rejection_button_state(self):
        """Update rejection button state."""
        try:
            # Reject button is enabled whenever there's a calculated transformation
            has_transformation = self._calculated_transform is not None
            self.reject_btn.setEnabled(has_transformation)
            
            if has_transformation:
                self.reject_btn.setText("✗ Reject & Recalculate")
                self.reject_btn.setToolTip("Reject this transformation and return to point selection")
                self.reject_btn.setStyleSheet("""
                    QPushButton:enabled {
                        background-color: #dc3545;
                        color: white;
                        font-weight: bold;
                        border: 2px solid #dc3545;
                        border-radius: 6px;
                        padding: 8px 16px;
                        font-size: 13px;
                    }
                    QPushButton:enabled:hover {
                        background-color: #c82333;
                        border-color: #c82333;
                    }
                    QPushButton:enabled:pressed {
                        background-color: #bd2130;
                    }
                """)
            else:
                self.reject_btn.setText("Reject Transformation")
                self.reject_btn.setToolTip("No transformation to reject")
                self.reject_btn.setStyleSheet("""
                    QPushButton:disabled {
                        background-color: #6c757d;
                        color: #ffffff;
                        border: 2px solid #6c757d;
                        border-radius: 6px;
                        padding: 8px 16px;
                        font-size: 13px;
                    }
                """)
                
        except Exception as e:
            logger.error(f"Failed to update rejection button: {e}")
    
    def _update_preview_button_state(self):
        """Update preview generation button state."""
        try:
            if hasattr(self, 'generate_preview_btn'):
                # Preview can be generated if we have images and a transformation (even invalid ones for debugging)
                can_preview = (self._original_gds_image is not None and 
                             self._calculated_transform is not None)
                
                self.generate_preview_btn.setEnabled(can_preview)
                
                if can_preview:
                    if self._is_valid:
                        self.generate_preview_btn.setText("🔍 Generate Preview")
                        self.generate_preview_btn.setToolTip("Generate transformation preview")
                    else:
                        self.generate_preview_btn.setText("🔍 Preview (Debug)")
                        self.generate_preview_btn.setToolTip("Generate preview for debugging (transformation is invalid)")
                else:
                    self.generate_preview_btn.setText("Generate Preview")
                    self.generate_preview_btn.setToolTip("Preview requires images and calculated transformation")
                    
        except Exception as e:
            logger.error(f"Failed to update preview button: {e}")
    
    def _update_visual_feedback(self, is_valid: bool, status: str, quality_score: float):
        """Update additional visual feedback elements."""
        try:
            # Update any progress bars, status indicators, or other visual elements
            # This can be extended as needed for additional UI feedback
            
            # Example: Update a quality progress bar if it exists
            if hasattr(self, 'quality_progress_bar'):
                self.quality_progress_bar.setValue(int(quality_score * 100))
                if is_valid:
                    if quality_score >= 0.8:
                        color = "#28a745"  # Green
                    elif quality_score >= 0.6:
                        color = "#ffc107"  # Yellow
                    else:
                        color = "#fd7e14"  # Orange
                else:
                    color = "#dc3545"  # Red
                
                self.quality_progress_bar.setStyleSheet(f"""
                    QProgressBar::chunk {{
                        background-color: {color};
                    }}
                """)
                
        except Exception as e:
            logger.error(f"Failed to update visual feedback: {e}")
    
    def _darken_color(self, hex_color: str, factor: float = 0.1) -> str:
        """Darken a hex color by the given factor."""
        try:
            # Remove # if present
            hex_color = hex_color.lstrip('#')
            
            # Convert to RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            # Darken
            r = max(0, int(r * (1 - factor)))
            g = max(0, int(g * (1 - factor)))
            b = max(0, int(b * (1 - factor)))
            
            # Convert back to hex
            return f"#{r:02x}{g:02x}{b:02x}"
            
        except Exception:
            return hex_color  # Return original on error
    
    def _generate_validation_recommendations(self, issues: List[str], warnings: List[str]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Recommendations for common issues
        issue_keywords = {
            "scale": "Consider selecting points that are more evenly distributed across the image",
            "translation": "Verify that corresponding points are correctly matched between images",
            "rotation": "Check if the selected points accurately represent the same features",
            "degenerate": "Ensure points are not collinear and represent distinct features",
            "distortion": "Try selecting points that form a more equilateral triangle",
            "extreme": "Review point selection for accuracy and representativeness"
        }
        
        for issue in issues:
            for keyword, recommendation in issue_keywords.items():
                if keyword.lower() in issue.lower():
                    if recommendation not in recommendations:
                        recommendations.append(recommendation)
                    break
        
        # Recommendations for warnings
        warning_keywords = {
            "adjustment": "The transformation may be reasonable despite the adjustment",
            "difference": "Consider if this scaling difference is expected for your images",
            "reflection": "Verify that the image orientation is correct",
            "instability": "Consider selecting more well-distributed points"
        }
        
        for warning in warnings:
            for keyword, recommendation in warning_keywords.items():
                if keyword.lower() in warning.lower():
                    if recommendation not in recommendations:
                        recommendations.append(recommendation)
                    break
        
        # General recommendations
        if not recommendations and (issues or warnings):
            recommendations.append("Review your point selection and try different corresponding points")
        
        return recommendations
    
    def _show_validation_error(self, error_message: str):
        """Show validation error in the display."""
        self.validation_label.setText(f"⚠ Validation Error: {error_message}")
        self.validation_label.setStyleSheet("""
            QLabel {
                color: #721c24;
                background-color: #f8d7da;
                border: 2px solid #f5c6cb;
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: bold;
            }
        """)
        self.validation_details.setText("An error occurred during validation. Please try recalculating the transformation.")
    
    def _update_quality_metrics(self, params: Dict[str, float]):
        """Update quality metrics display."""
        # Calculate point accuracy (theoretical - would need actual point transformation)
        point_accuracy = 0.95  # Placeholder
        
        # Scale uniformity
        scale_ratio = min(params['scale_x'], params['scale_y']) / max(params['scale_x'], params['scale_y'])
        scale_uniformity = scale_ratio
        
        # Distortion level (inverse of scale ratio)
        distortion_level = 1.0 - scale_uniformity
        
        # Overall quality
        overall_quality = self._calculated_transform.get('quality_score', 0.0) if self._calculated_transform else 0.0
        
        metrics = [
            f"{point_accuracy:.1%}",
            f"{scale_uniformity:.1%}",
            f"{distortion_level:.1%}",
            f"{overall_quality:.1%}"
        ]
        
        for i, metric_value in enumerate(metrics):
            self.quality_table.item(i, 1).setText(metric_value)
    
    def _generate_preview(self):
        """
        Generate transformation preview by overlaying transformed GDS on SEM image.
        
        This method implements Step 13 requirement to show preview of the transformation
        before applying it, allowing users to visualize the alignment result.
        """
        try:
            # Validate prerequisites
            if not self._can_generate_preview():
                return
            
            # Start preview generation process
            self._start_preview_generation()
            
            # Apply transformation to GDS image
            transformed_gds = self._apply_transformation_to_gds()
            if transformed_gds is None:
                self._handle_preview_failure("Failed to apply transformation to GDS")
                return
            
            # Create overlay preview image
            preview_image = self._create_overlay_preview(transformed_gds)
            if preview_image is None:
                self._handle_preview_failure("Failed to create overlay preview")
                return
            
            # Store and emit the preview
            self._finalize_preview(preview_image)
            
        except Exception as e:
            self._handle_preview_failure(f"Preview generation error: {str(e)}")
    
    def _can_generate_preview(self) -> bool:
        """Check if preview generation can proceed."""
        if self._original_gds_image is None:
            self._show_error("Cannot generate preview: No GDS image available")
            return False
        
        if self._transform_matrix is None:
            self._show_error("Cannot generate preview: No transformation calculated")
            return False
        
        # Check if transformation matrix is valid
        if not isinstance(self._transform_matrix, np.ndarray) or self._transform_matrix.shape != (3, 3):
            self._show_error("Cannot generate preview: Invalid transformation matrix")
            return False
        
        # Check for degenerate transformation
        det = np.linalg.det(self._transform_matrix[:2, :2])
        if abs(det) < 1e-10:
            self._show_error("Cannot generate preview: Degenerate transformation")
            return False
        
        return True
    
    def _start_preview_generation(self):
        """Setup UI for preview generation process."""
        self.preview_status_label.setText("🔄 Generating transformation preview...")
        self.preview_status_label.setStyleSheet("color: #007bff; font-weight: bold;")
        self.generate_preview_btn.setEnabled(False)
        
        logger.info("Starting transformation preview generation")
    
    def _apply_transformation_to_gds(self) -> Optional[np.ndarray]:
        """Apply the calculated transformation to the GDS image."""
        try:
            # Get original image dimensions
            height, width = self._original_gds_image.shape[:2]
            
            # Apply affine transformation using OpenCV
            # Use the 2x3 transformation matrix (first 2 rows of the 3x3 matrix)
            transform_2x3 = self._transform_matrix[:2, :].astype(np.float32)
            
            # Apply transformation with proper interpolation and border handling
            transformed_gds = cv2.warpAffine(
                self._original_gds_image,
                transform_2x3,
                (width, height),
                flags=cv2.INTER_LINEAR,  # Smooth interpolation
                borderMode=cv2.BORDER_CONSTANT,  # Fill with black
                borderValue=0
            )
            
            logger.debug(f"Applied transformation to GDS image: {width}x{height}")
            return transformed_gds
            
        except Exception as e:
            logger.error(f"Failed to apply transformation to GDS: {e}")
            return None
    
    def _create_overlay_preview(self, transformed_gds: np.ndarray) -> Optional[np.ndarray]:
        """Create an overlay preview combining transformed GDS with SEM image."""
        try:
            # If no SEM reference image, return the transformed GDS
            if self._sem_reference_image is None:
                logger.info("No SEM reference - returning transformed GDS only")
                return self._enhance_standalone_gds_preview(transformed_gds)
            
            # Ensure images are compatible for overlay
            sem_image = self._prepare_sem_for_overlay()
            transformed_gds = self._prepare_gds_for_overlay(transformed_gds, sem_image.shape[:2])
            
            # Create different overlay visualizations
            overlay_preview = self._create_multi_layer_overlay(sem_image, transformed_gds)
            
            return overlay_preview
            
        except Exception as e:
            logger.error(f"Failed to create overlay preview: {e}")
            return None
    
    def _prepare_sem_for_overlay(self) -> np.ndarray:
        """Prepare SEM image for overlay creation."""
        sem_image = self._sem_reference_image.copy()
        
        # Convert to RGB if grayscale
        if len(sem_image.shape) == 2:
            sem_image = cv2.cvtColor(sem_image, cv2.COLOR_GRAY2RGB)
        elif len(sem_image.shape) == 3 and sem_image.shape[2] == 4:
            # Convert RGBA to RGB
            sem_image = cv2.cvtColor(sem_image, cv2.COLOR_RGBA2RGB)
        
        # Normalize to [0, 255] range
        if sem_image.dtype != np.uint8:
            sem_image = ((sem_image - sem_image.min()) / (sem_image.max() - sem_image.min()) * 255).astype(np.uint8)
        
        return sem_image
    
    def _prepare_gds_for_overlay(self, gds_image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Prepare GDS image for overlay with SEM."""
        # Resize to match SEM dimensions if needed
        if gds_image.shape[:2] != target_shape:
            gds_image = cv2.resize(gds_image, (target_shape[1], target_shape[0]))
            logger.debug(f"Resized GDS to match SEM: {target_shape}")
        
        # Ensure binary/grayscale format
        if len(gds_image.shape) == 3:
            gds_image = cv2.cvtColor(gds_image, cv2.COLOR_RGB2GRAY)
        
        # Normalize to [0, 255] range
        if gds_image.dtype != np.uint8:
            gds_image = ((gds_image - gds_image.min()) / (gds_image.max() - gds_image.min()) * 255).astype(np.uint8)
        
        return gds_image
    
    def _create_multi_layer_overlay(self, sem_image: np.ndarray, gds_image: np.ndarray) -> np.ndarray:
        """Create a multi-layer overlay visualization."""
        try:
            # Create base preview with SEM as background
            preview = sem_image.copy()
            
            # Method 1: Colored overlay (green GDS structures)
            colored_overlay = self._create_colored_overlay(sem_image, gds_image)
            
            # Method 2: Edge highlighting
            edge_overlay = self._create_edge_overlay(sem_image, gds_image)
            
            # Method 3: Transparency blend
            transparency_overlay = self._create_transparency_overlay(sem_image, gds_image)
            
            # Combine methods for best visualization
            # Use colored overlay as primary, enhance with edges
            final_preview = colored_overlay.copy()
            
            # Add edge highlights where GDS structures are present
            gds_mask = gds_image > 128
            edge_intensity = cv2.Canny(gds_image, 50, 150)
            edge_mask = edge_intensity > 0
            
            # Highlight edges in cyan for better visibility
            final_preview[edge_mask] = [0, 255, 255]  # Cyan edges
            
            return final_preview
            
        except Exception as e:
            logger.error(f"Failed to create multi-layer overlay: {e}")
            # Fallback to simple overlay
            return self._create_simple_overlay(sem_image, gds_image)
    
    def _create_colored_overlay(self, sem_image: np.ndarray, gds_image: np.ndarray) -> np.ndarray:
        """Create colored overlay with GDS structures in green."""
        overlay = sem_image.copy()
        
        # Create mask for GDS structures
        gds_mask = gds_image > 128
        
        # Apply green color to GDS regions
        overlay[gds_mask, 1] = np.minimum(255, overlay[gds_mask, 1] + 100)  # Enhance green
        
        # Blend with original
        blended = cv2.addWeighted(sem_image, 0.7, overlay, 0.3, 0)
        
        return blended
    
    def _create_edge_overlay(self, sem_image: np.ndarray, gds_image: np.ndarray) -> np.ndarray:
        """Create overlay highlighting GDS edges."""
        # Detect edges in GDS
        edges = cv2.Canny(gds_image, 50, 150)
        
        # Create colored edge overlay
        overlay = sem_image.copy()
        edge_mask = edges > 0
        overlay[edge_mask] = [255, 255, 0]  # Yellow edges
        
        # Blend with original
        blended = cv2.addWeighted(sem_image, 0.8, overlay, 0.2, 0)
        
        return blended
    
    def _create_transparency_overlay(self, sem_image: np.ndarray, gds_image: np.ndarray) -> np.ndarray:
        """Create transparency-based overlay."""
        # Convert GDS to RGB
        gds_rgb = cv2.cvtColor(gds_image, cv2.COLOR_GRAY2RGB)
        
        # Create alpha mask based on GDS intensity
        alpha = (gds_image / 255.0) * 0.4  # 40% transparency where GDS exists
        
        # Apply transparency blend
        blended = sem_image.copy().astype(np.float32)
        gds_float = gds_rgb.astype(np.float32)
        
        for c in range(3):
            blended[:, :, c] = (1 - alpha) * blended[:, :, c] + alpha * gds_float[:, :, c]
        
        return blended.astype(np.uint8)
    
    def _create_simple_overlay(self, sem_image: np.ndarray, gds_image: np.ndarray) -> np.ndarray:
        """Create simple fallback overlay."""
        overlay = sem_image.copy()
        gds_mask = gds_image > 128
        overlay[gds_mask, 1] = 255  # Simple green overlay
        
        return cv2.addWeighted(sem_image, 0.7, overlay, 0.3, 0)
    
    def _enhance_standalone_gds_preview(self, gds_image: np.ndarray) -> np.ndarray:
        """Enhance GDS image for standalone preview."""
        try:
            # Convert to RGB for better visualization
            if len(gds_image.shape) == 2:
                preview = cv2.cvtColor(gds_image, cv2.COLOR_GRAY2RGB)
            else:
                preview = gds_image.copy()
            
            # Enhance contrast
            preview = cv2.convertScaleAbs(preview, alpha=1.2, beta=10)
            
            # Add slight blue tint to distinguish it as GDS
            if len(preview.shape) == 3:
                preview[:, :, 0] = np.minimum(255, preview[:, :, 0] + 20)  # Add blue
            
            return preview
            
        except Exception as e:
            logger.error(f"Failed to enhance standalone GDS preview: {e}")
            return gds_image
    
    def _finalize_preview(self, preview_image: np.ndarray):
        """Store preview and emit signals."""
        try:
            # Store the preview
            self._preview_image = preview_image.copy()
            
            # Update UI status
            self.preview_status_label.setText("✓ Preview generated successfully")
            self.preview_status_label.setStyleSheet("color: #28a745; font-weight: bold;")
            
            # Emit preview signal for external display
            self.preview_updated.emit(preview_image)
            
            # Re-enable preview button
            self.generate_preview_btn.setEnabled(True)
            
            logger.info(f"Transformation preview finalized: {preview_image.shape}")
            
        except Exception as e:
            logger.error(f"Failed to finalize preview: {e}")
            self._handle_preview_failure("Failed to finalize preview")
    
    def _handle_preview_failure(self, error_message: str):
        """Handle preview generation failure."""
        self.preview_status_label.setText("✗ Preview generation failed")
        self.preview_status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
        self.generate_preview_btn.setEnabled(True)
        
        self._show_error(error_message)
        logger.error(f"Preview generation failed: {error_message}")
        
        # Clear any partial preview
        self._preview_image = None
    
    def _toggle_manual_adjustment(self, enabled: bool):
        """Toggle manual parameter adjustment with enhanced controls."""
        # Enable/disable spinboxes
        for spinbox in self._param_spinboxes.values():
            spinbox.setEnabled(enabled and self._calculated_transform is not None)
        
        # Enable/disable adjustment controls
        if hasattr(self, 'reset_params_btn'):
            self.reset_params_btn.setEnabled(enabled and self._calculated_transform is not None)
        if hasattr(self, 'quick_adjust_btn'):
            self.quick_adjust_btn.setEnabled(enabled and self._calculated_transform is not None)
        
        if enabled:
            if self._calculated_transform is not None:
                self.adjustment_status_label.setText("Fine-tuning enabled - adjust parameters above")
                self.adjustment_status_label.setStyleSheet("""
                    QLabel {
                        padding: 3px;
                        font-size: 10px;
                        color: #0d6efd;
                        background-color: #e7f3ff;
                        border: 1px solid #b6d7ff;
                        border-radius: 3px;
                    }
                """)
                logger.info("Manual parameter adjustment enabled")
            else:
                self.adjustment_status_label.setText("Calculate transformation first to enable fine-tuning")
                self.adjustment_status_label.setStyleSheet("""
                    QLabel {
                        padding: 3px;
                        font-size: 10px;
                        color: #dc3545;
                        background-color: #f8d7da;
                        border: 1px solid #f5c6cb;
                        border-radius: 3px;
                    }
                """)
        else:
            self.adjustment_status_label.setText("Fine-tuning disabled")
            self.adjustment_status_label.setStyleSheet("""
                QLabel {
                    padding: 3px;
                    font-size: 10px;
                    color: #666;
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 3px;
                }
            """)
            logger.info("Manual parameter adjustment disabled")
    
    def _parameter_adjusted(self, param_name: str, value: float):
        """Handle manual parameter adjustment."""
        if self._calculated_transform is not None:
            # Update the calculated transform
            param_map = {
                "Translation X": 'translation_x',
                "Translation Y": 'translation_y',
                "Rotation (°)": 'rotation',
                "Scale X": 'scale_x',
                "Scale Y": 'scale_y'
            }
            
            if param_name in param_map:
                key = param_map[param_name]
                self._calculated_transform[key] = value
                
                # Recalculate transform matrix
                self._recalculate_transform_matrix()
                
                # Re-validate
                validation_result = self._validate_transformation(self._calculated_transform)
                self._is_valid = validation_result['is_valid']
                self._update_validation_display(validation_result)
                self._update_quality_metrics(self._calculated_transform)
                
                # Update confirm button state
                self.confirm_btn.setEnabled(self._is_valid)
                
                # Auto-generate preview if enabled
                if self.auto_preview_cb.isChecked():
                    QTimer.singleShot(100, self._generate_preview)
                
                # Emit signal
                self.parameter_adjusted.emit(param_name, value)
                
                logger.debug(f"Parameter adjusted: {param_name} = {value}")
    
    def _recalculate_transform_matrix(self):
        """Recalculate transformation matrix from adjusted parameters."""
        if self._calculated_transform is None:
            return
        
        # Create transformation matrix from parameters
        tx = self._calculated_transform['translation_x']
        ty = self._calculated_transform['translation_y']
        rotation = np.radians(self._calculated_transform['rotation'])
        sx = self._calculated_transform['scale_x']
        sy = self._calculated_transform['scale_y']
        
        # Create individual transformation matrices
        # Translation
        T = np.array([[1, 0, tx],
                      [0, 1, ty],
                      [0, 0, 1]])
        
        # Rotation
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        R = np.array([[cos_r, -sin_r, 0],
                      [sin_r, cos_r, 0],
                      [0, 0, 1]])
        
        # Scale
        S = np.array([[sx, 0, 0],
                      [0, sy, 0],
                      [0, 0, 1]])
        
        # Combine: T * R * S (as per Step 7 requirements)
        self._transform_matrix = T @ R @ S
        self._calculated_transform['transform_matrix'] = self._transform_matrix.tolist()
    
    def _confirm_transformation(self):
        """Confirm and emit the transformation."""
        if self._calculated_transform is not None and self._is_valid:
            self.transformation_confirmed.emit(self._calculated_transform.copy())
            logger.info("Transformation confirmed")
        else:
            self._show_error("Cannot confirm invalid transformation")
    
    def _reject_transformation(self):
        """Reject the transformation."""
        reason = "User rejected transformation"
        self.transformation_rejected.emit(reason)
        self._reset_transformation()
        logger.info("Transformation rejected")
    
    def _reset_transformation(self):
        """Reset all transformation data."""
        self._calculated_transform = None
        self._transform_matrix = None
        self._is_valid = False
        self._preview_image = None
        
        # Reset UI
        for i in range(5):
            self.params_table.item(i, 1).setText("-")
        
        for spinbox in self._param_spinboxes.values():
            spinbox.setEnabled(False)
            spinbox.setValue(0)
        
        for i in range(4):
            self.quality_table.item(i, 1).setText("-")
        
        self.validation_label.setText("No calculation performed")
        self.validation_label.setStyleSheet("")
        self.validation_details.setText("")
        
        self.preview_status_label.setText("No preview generated")
        
        # Reset buttons
        self.generate_preview_btn.setEnabled(False)
        self.confirm_btn.setEnabled(False)
        self.reject_btn.setEnabled(False)
        
        self.manual_adjustment_cb.setChecked(False)
        
        logger.info("Transformation reset")
    
    def _reset_to_calculated(self):
        """Reset manual adjustments to originally calculated values."""
        if not self._calculated_transform:
            return
        
        # Reset all parameter spinboxes to calculated values
        if hasattr(self, '_param_spinboxes'):
            for param_name, spinbox in self._param_spinboxes.items():
                if param_name in self._calculated_transform:
                    spinbox.setValue(self._calculated_transform[param_name])
        
        # Update status
        if hasattr(self, 'adjustment_status_label'):
            self.adjustment_status_label.setText("Reset to calculated values")
        
        # Regenerate preview with reset values
        self._generate_preview()
        
        logger.info("Parameters reset to calculated values")
    
    def _show_quick_adjustments(self):
        """Show quick adjustment options menu."""
        from PySide6.QtWidgets import QMenu, QAction
        
        menu = QMenu(self)
        
        # Common adjustment options
        actions = [
            ("Fine tune rotation (+0.1°)", lambda: self._adjust_parameter('rotation', 0.1)),
            ("Fine tune rotation (-0.1°)", lambda: self._adjust_parameter('rotation', -0.1)),
            ("Increase scale (+1%)", lambda: self._adjust_parameter('scale_x', 0.01)),
            ("Decrease scale (-1%)", lambda: self._adjust_parameter('scale_x', -0.01)),
            ("Nudge translation X (+1px)", lambda: self._adjust_parameter('translation_x', 1)),
            ("Nudge translation X (-1px)", lambda: self._adjust_parameter('translation_x', -1)),
            ("Nudge translation Y (+1px)", lambda: self._adjust_parameter('translation_y', 1)),
            ("Nudge translation Y (-1px)", lambda: self._adjust_parameter('translation_y', -1)),
        ]
        
        for text, callback in actions:
            action = QAction(text, self)
            action.triggered.connect(callback)
            menu.addAction(action)
        
        # Show menu at button position
        menu.exec_(self.quick_adjust_btn.mapToGlobal(self.quick_adjust_btn.rect().bottomLeft()))
    
    def _adjust_parameter(self, param_name: str, delta: float):
        """Apply small adjustment to a parameter."""
        if param_name in self._param_spinboxes:
            current_value = self._param_spinboxes[param_name].value()
            new_value = current_value + delta
            self._param_spinboxes[param_name].setValue(new_value)
            
            # Update status
            if hasattr(self, 'adjustment_status_label'):
                self.adjustment_status_label.setText(f"Adjusted {param_name} by {delta:+.3f}")
            
            # Update preview
            self._generate_preview()
