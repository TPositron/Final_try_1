"""
Alignment Panel - Comprehensive Alignment Interface with Multi-Mode Support

This module provides a comprehensive alignment panel with manual, hybrid, and automatic
alignment modes, featuring integrated file selection and real-time transformation preview.

Main Classes:
- AlignmentPanel: Main alignment interface with sub-mode switching
- OverlayCanvas: Graphics view for SEM and GDS image overlay display

Key Methods:
- setup_ui(): Initializes 3-panel layout (controls, display, file selection)
- _create_manual_controls(): Creates manual alignment transformation controls
- _create_hybrid_controls(): Creates 3-point alignment interface
- _create_automatic_controls(): Creates automatic alignment interface
- set_initial_sem_image(): Sets SEM image for alignment operations
- set_initial_gds_image(): Sets GDS overlay for alignment operations
- _generate_aligned_gds(): Generates aligned GDS file from transformations
- update_transform(): Updates real-time transformation preview

Signals Emitted:
- transform_changed(dict): Transform parameters changed
- alignment_applied(dict): Alignment transformation applied
- sem_file_selected(str): SEM file selected from FileSelector
- gds_structure_selected(str, int): GDS structure selected
- three_points_selected(list, list): 3-point alignment points selected
- transformation_calculated(dict): Transformation matrix calculated
- transformation_confirmed(dict): Transformation confirmed and applied

Dependencies:
- Uses: PySide6.QtWidgets, PySide6.QtCore, PySide6.QtGui (Qt framework)
- Uses: numpy (array processing)
- Uses: typing, logging, os, pathlib (utilities)
- Uses: ui/components (SliderInput, ThreePointSelectionController, etc.)
- Uses: services (GdsTransformationService)
- Called by: UI main window and alignment workflow
- Coordinates with: File operations, image processing, and transformation services

Features:
- Three alignment modes: Manual (sliders), Hybrid (3-point), Automatic
- Integrated FileSelector for consistent file management
- Real-time transformation preview with overlay canvas
- 3-point alignment workflow with point validation
- Automatic GDS file generation with applied transformations
- Sub-mode switching with dedicated control panels
- Interactive image display with zoom and pan capabilities
"""
 
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QSplitter, 
                               QGroupBox, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
                               QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QProgressBar,
                               QStackedWidget, QScrollArea, QFrame, QFileDialog, QComboBox,
                               QMessageBox, QApplication)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter
from ..components.slider_input import SliderInput
from ..components.three_point_selection_controller import ThreePointSelectionController
from ..components.transformation_preview_widget import TransformationPreviewWidget
# FileSelector removed - no longer needed
from .alignment_sub_mode_switcher import AlignmentSubModeSwitcher
from typing import Dict, Any, Optional
import numpy as np
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class AlignmentPanel(QWidget):
    """
    Alignment panel with sub-mode switcher, transformation controls, and integrated file selection.
    
    Layout: Left (sub-modes + controls) | Center (image display) | Right (integrated FileSelector + generate)
    
    Features:
    - Manual/Hybrid/Automatic alignment modes
    - Integrated FileSelector for consistent file management
    - Real-time transformation preview
    - 3-point alignment workflow
    - Generate aligned GDS functionality
    """
    
    # Signals
    transform_changed = Signal(dict)  # Emitted when transform parameters change
    alignment_applied = Signal(dict)  # Emitted when alignment is applied
    alignment_changed = Signal(dict)  # Emitted when alignment changes
    reset_requested = Signal()  # Emitted when reset is requested
    structure_selected = Signal(str)  # Emitted when a structure is selected for alignment
    
    # File selection signals
    sem_file_selected = Signal(str)  # Emitted when SEM file is selected
    gds_file_loaded = Signal(str)    # Emitted when GDS file is first selected
    gds_structure_selected = Signal(str, int)  # Emitted when structure is selected (path, structure_id)
    
    # Step 11 signals for transformation preview workflow
    three_points_selected = Signal(list, list)  # sem_points, gds_points
    transformation_calculated = Signal(dict)    # transformation_parameters
    transformation_confirmed = Signal(dict)     # confirmed_transformation
    validation_status_changed = Signal(bool, str)  # is_valid, status_message
    preview_image_updated = Signal(object)      # preview_image_array (Step 14)
    preview_updated = Signal(object)            # preview data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Current transform parameters
        self.transform_params = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0
        }
        
        # Current sub-mode
        self.current_sub_mode = "manual"
        
        # Components for Step 11
        self.three_point_controller: Optional[ThreePointSelectionController] = None
        self.transformation_preview: Optional[TransformationPreviewWidget] = None
        
        # Current images for alignment
        self.current_sem_image = None
        self.current_gds_image = None
        self._current_gds_model = None
        self._auto_alignment_params = None
        
        # Current file selections (tracked from FileSelector)
        self.current_sem_file_path = None
        self.current_gds_path = None
        self.current_structure_id = None
        
        # Optional attributes that may not be implemented yet
        self.method_selector: Optional[Any] = None
        self.alignment_service: Optional[Any] = None
        self.preset_selector: Optional[Any] = None
        
        self.setup_ui()
        # Signal connections now happen in _create_hybrid_controls() when components are created
        
        # Initialize structure list
        self._initialize_structure_list()
        
    def setup_ui(self):
        """Initialize the UI components with new 3-panel layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Create main splitter with 3 sections
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left Panel: Sub-mode switcher + Controls
        left_panel = self._create_left_panel()
        self.main_splitter.addWidget(left_panel)
        
        # Center Panel: Image Display
        center_panel = self._create_center_panel()
        self.main_splitter.addWidget(center_panel)
        
        # Right Panel: File Selection
        right_panel = self._create_right_panel()
        self.main_splitter.addWidget(right_panel)
        
        # Set splitter proportions (12.5% left, 62.5% center, 25% right)
        self.main_splitter.setSizes([150, 750, 300])
        
        layout.addWidget(self.main_splitter)
        
        # Connect view controls
        self._connect_view_controls()
        
    def _create_left_panel(self) -> QWidget:
        """Create the left panel with sub-mode switcher and controls."""
        left_widget = QWidget()
        left_widget.setMaximumWidth(150)
        left_widget.setMinimumWidth(100)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(8)
        
        # Sub-mode switcher
        self.sub_mode_switcher = AlignmentSubModeSwitcher()
        self.sub_mode_switcher.sub_mode_changed.connect(self._on_sub_mode_changed)
        left_layout.addWidget(self.sub_mode_switcher)
        
        # Stacked widget for different sub-mode controls
        self.controls_stack = QStackedWidget()
        
        # Manual controls
        manual_controls = self._create_manual_controls()
        self.controls_stack.addWidget(manual_controls)  # Index 0
        
        # Hybrid controls
        hybrid_controls = self._create_hybrid_controls()
        self.controls_stack.addWidget(hybrid_controls)  # Index 1
        
        # Automatic controls
        automatic_controls = self._create_automatic_controls()
        self.controls_stack.addWidget(automatic_controls)  # Index 2
        
        left_layout.addWidget(self.controls_stack)
        left_layout.addStretch()
        
        return left_widget
    
    def _create_center_panel(self) -> QWidget:
        """Create the center panel with image display."""
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(5, 5, 5, 5)
        
        # Title bar
        title_layout = QHBoxLayout()
        
        title_label = QLabel("Alignment Workspace")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #4A9EFF;")
        title_layout.addWidget(title_label)
        
        # View controls
        self.fit_button = QPushButton("Fit to Window")
        self.fit_button.setMaximumWidth(100)
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.setMaximumWidth(100)
        self.toggle_overlay_button = QPushButton("Toggle Overlay")
        self.toggle_overlay_button.setMaximumWidth(120)
        
        title_layout.addStretch()
        title_layout.addWidget(self.fit_button)
        title_layout.addWidget(self.reset_view_button)
        title_layout.addWidget(self.toggle_overlay_button)
        
        center_layout.addLayout(title_layout)
        
        # Image display area with overlay canvas
        self.overlay_canvas = OverlayCanvas()
        self.overlay_canvas.setMinimumSize(400, 400)
        center_layout.addWidget(self.overlay_canvas)
        
        # Minimal status only
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #CCCCCC; font-style: italic;")
        center_layout.addWidget(self.status_label)
        
        return center_widget
    
    def _create_right_panel(self) -> QWidget:
        """Create the right panel with GDS structure selection only."""
        right_widget = QWidget()
        right_widget.setMaximumWidth(300)
        right_widget.setMinimumWidth(250)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)
        
        # GDS Structure Selection with vertical slider
        structure_group = QGroupBox("GDS Structure Selection")
        structure_layout = QVBoxLayout(structure_group)
        
        # Create resizable splitter for structure list with slider
        from PySide6.QtWidgets import QScrollArea, QListWidget, QSlider
        
        # Vertical slider to control structure list height
        height_label = QLabel("List Height:")
        self.height_slider = QSlider(Qt.Orientation.Horizontal)
        self.height_slider.setRange(100, 400)
        self.height_slider.setValue(200)
        self.height_slider.valueChanged.connect(self._on_height_slider_changed)
        
        structure_layout.addWidget(height_label)
        structure_layout.addWidget(self.height_slider)
        
        # Structure list with scrollable area
        self.structure_scroll = QScrollArea()
        self.structure_list = QListWidget()
        self.structure_list.setMaximumHeight(200)
        self.structure_scroll.setWidget(self.structure_list)
        self.structure_scroll.setWidgetResizable(True)
        
        structure_layout.addWidget(self.structure_scroll)
        right_layout.addWidget(structure_group)
        
        # Generate Controls Section
        generate_group = QGroupBox("Generate")
        generate_layout = QVBoxLayout(generate_group)
        
        self.generate_button = QPushButton("Generate Aligned GDS")
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self._generate_aligned_gds)
        self.generate_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #333333;
                color: #666666;
            }
        """)
        generate_layout.addWidget(self.generate_button)
        
        right_layout.addWidget(generate_group)
        right_layout.addStretch()
        
        return right_widget
    
    def _create_manual_controls(self) -> QWidget:
        """Create manual alignment controls."""
        manual_widget = QWidget()
        manual_layout = QVBoxLayout(manual_widget)
        
        # Transform controls group
        transform_group = QGroupBox("Transform Controls")
        transform_layout = QVBoxLayout(transform_group)
        
        # Translation controls
        translation_group = QGroupBox("Translation")
        translation_layout = QVBoxLayout(translation_group)
        
        self.translate_x_slider = SliderInput("X Offset", -500.0, 500.0, 0.0)
        self.translate_x_slider.value_changed.connect(self._on_translate_x_changed)
        translation_layout.addWidget(self.translate_x_slider)
        
        self.translate_y_slider = SliderInput("Y Offset", -500.0, 500.0, 0.0)
        self.translate_y_slider.value_changed.connect(self._on_translate_y_changed)
        translation_layout.addWidget(self.translate_y_slider)
        
        transform_layout.addWidget(translation_group)
        
        # Rotation controls
        rotation_group = QGroupBox("Rotation")
        rotation_layout = QVBoxLayout(rotation_group)
        
        self.rotation_slider = SliderInput("Angle (degrees)", -180.0, 180.0, 0.0)
        self.rotation_slider.value_changed.connect(self._on_rotation_changed)
        rotation_layout.addWidget(self.rotation_slider)
        
        transform_layout.addWidget(rotation_group)
        
        # Scale controls
        scale_group = QGroupBox("Scale")
        scale_layout = QVBoxLayout(scale_group)
        
        self.scale_slider = SliderInput("Scale Factor", 0.1, 3.0, 1.0)
        self.scale_slider.value_changed.connect(self._on_scale_changed)
        scale_layout.addWidget(self.scale_slider)
        
        transform_layout.addWidget(scale_group)
        
        manual_layout.addWidget(transform_group)
        
        # Action buttons
        button_layout = QVBoxLayout()
        
        self.apply_button = QPushButton("Apply Transform")
        self.apply_button.clicked.connect(self._on_apply_clicked)
        button_layout.addWidget(self.apply_button)
        
        self.reset_button = QPushButton("Reset Transform")
        self.reset_button.clicked.connect(self._on_reset_clicked)
        button_layout.addWidget(self.reset_button)
        
        self.auto_align_button = QPushButton("Auto Align")
        self.auto_align_button.clicked.connect(self._on_auto_align_clicked)
        button_layout.addWidget(self.auto_align_button)
        
        manual_layout.addLayout(button_layout)
        manual_layout.addStretch()
        
        return manual_widget
    
    def _create_hybrid_controls(self) -> QWidget:
        """Create hybrid alignment controls (3-point selection)."""
        hybrid_widget = QWidget()
        hybrid_layout = QVBoxLayout(hybrid_widget)
        
        # Create splitter for 3-point selection and preview
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Three-point selection controller (Step 10)
        self.three_point_controller = ThreePointSelectionController()
        splitter.addWidget(self.three_point_controller)
        
        # Transformation preview widget (Step 11)
        self.transformation_preview = TransformationPreviewWidget()
        splitter.addWidget(self.transformation_preview)
        
        # Connect signals now that components are created
        self.three_point_controller.alignment_ready.connect(self._on_points_ready)
        
        self.transformation_preview.transformation_calculated.connect(
            self.transformation_calculated.emit
        )
        self.transformation_preview.transformation_confirmed.connect(
            self.transformation_confirmed.emit
        )
        self.transformation_preview.preview_updated.connect(
            self._on_preview_updated
        )
        
        # Set splitter proportions (60% selection, 40% preview)
        splitter.setSizes([200, 150])
        
        hybrid_layout.addWidget(splitter)
        
        return hybrid_widget
    
    def _initialize_structure_list(self):
        """Initialize structure list widget."""
        if hasattr(self, 'structure_list'):
            try:
                self.structure_list.itemClicked.connect(self._on_structure_item_clicked)
                logger.info("Structure list initialized")
            except Exception as e:
                logger.error(f"Failed to initialize structure list: {e}")
    
    def _on_height_slider_changed(self, value):
        """Handle height slider change for structure list."""
        if hasattr(self, 'structure_list'):
            self.structure_list.setMaximumHeight(value)
            self.structure_list.setMinimumHeight(value)
    
    def _create_automatic_controls(self) -> QWidget:
        """Create automatic alignment controls."""
        auto_widget = QWidget()
        auto_layout = QVBoxLayout(auto_widget)
        
        # Placeholder for automatic controls
        label = QLabel("Automatic alignment controls will be here.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        auto_layout.addWidget(label)
        
        return auto_widget
    
    def _connect_workflow_signals(self):
        """Connect signals for the Step 11 workflow."""
        # Note: Components are created in _create_hybrid_controls(), so they may be None initially
        # Signal connections will be established when components are created
        pass
    
    def _on_points_ready(self, ready: bool):
        """Handle when 3 point pairs are ready for transformation calculation."""
        if ready and self.three_point_controller:
            # Get the actual points from the controller
            sem_points = self.three_point_controller.get_sem_points()
            gds_points = self.three_point_controller.get_gds_points()
            
            logger.info(f"3 point pairs ready: {len(sem_points)} SEM, {len(gds_points)} GDS")
            
            # Emit signal for external handling
            self.three_points_selected.emit(sem_points, gds_points)
            
            # Pass points to transformation preview widget
            if self.transformation_preview:
                self.transformation_preview.set_point_pairs(sem_points, gds_points)
                logger.info("Point pairs passed to transformation preview widget")
    
    def _on_preview_updated(self, preview_data):
        """Handle preview update from transformation widget."""
        logger.info("Transformation preview updated")
        # Emit signal to notify other components
        self.preview_updated.emit(preview_data)
    
    def set_images_for_selection(self, sem_image: np.ndarray, gds_image: np.ndarray):
        """Set images for 3-point selection in hybrid mode."""
        if self.three_point_controller:
            # Use the actual methods available on ThreePointSelectionController
            # Based on the implementation, we need to update points, not set images directly
            # This method may need to be coordinated with the image viewers
            logger.info("Images available for 3-point selection")
    
    def set_images_for_preview(self, sem_image: np.ndarray, gds_image: np.ndarray):
        """Set images for transformation preview."""
        if self.transformation_preview:
            self.transformation_preview.set_images_for_preview(gds_image, sem_image)
    
    def set_initial_gds_image(self, gds_image: np.ndarray):
        """Display the structure-specific GDS image."""
        print(f"\n=== DISPLAYING STRUCTURE IMAGE ===")
        print(f"Image shape: {gds_image.shape}")
        print(f"Image contains: {np.count_nonzero(gds_image)} non-zero pixels")
        
        # Store the GDS image for later use
        self.current_gds_image = gds_image
        
        # Convert numpy array to QPixmap and display
        self._display_gds_image(gds_image)
        
        # If we have both SEM and GDS images, set them for selection and preview
        if hasattr(self, 'current_sem_image') and self.current_sem_image is not None:
            self.set_images_for_selection(self.current_sem_image, gds_image)
            self.set_images_for_preview(self.current_sem_image, gds_image)
        
        logger.info(f"Initial GDS image set: shape={gds_image.shape}")
        print("This should show ONLY the selected structure region!")
        print("===================================\n")
    
    def _display_gds_image(self, gds_image: np.ndarray):
        """Display GDS image in the overlay canvas."""
        try:
            if gds_image is not None and gds_image.size > 0:
                # Load GDS image into overlay canvas
                self.overlay_canvas.load_gds_overlay(gds_image)
                
                non_zero = np.count_nonzero(gds_image)
                print(f"Structure image loaded into overlay canvas: {gds_image.shape}")
                logger.info(f"GDS image displayed: {gds_image.shape}, {non_zero} pixels")
            else:
                logger.warning("No GDS image to display")
                
        except Exception as e:
            logger.error(f"Failed to display GDS image: {e}")
    
    def set_initial_sem_image(self, sem_image: np.ndarray):
        """Set initial SEM image for alignment operations."""
        # Store and load SEM image
        self.current_sem_image = sem_image
        self.overlay_canvas.load_sem_image(sem_image)
        
        # If we have both SEM and GDS images, set them for selection and preview  
        if hasattr(self, 'current_gds_image') and self.current_gds_image is not None:
            self.set_images_for_selection(sem_image, self.current_gds_image)
            self.set_images_for_preview(sem_image, self.current_gds_image)
        
        logger.info(f"Initial SEM image set: shape={sem_image.shape}")
    
    def _on_sub_mode_changed(self, sub_mode: str):
        """Handle sub-mode change from switcher."""
        self.current_sub_mode = sub_mode
        
        # Switch to appropriate controls stack
        if sub_mode == "manual":
            self.controls_stack.setCurrentIndex(0)
        elif sub_mode == "hybrid":
            self.controls_stack.setCurrentIndex(1)
        elif sub_mode == "automatic":
            self.controls_stack.setCurrentIndex(2)
        
        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.setText(f"{sub_mode.title()} alignment mode active")
        logger.info(f"Switched to {sub_mode} alignment mode")
    
    def _on_translate_x_changed(self, value):
        """Handle X translation change."""
        print(f"X translation changed to: {value}")
        self.transform_params['translate_x'] = value
        self._emit_transform_changed()
        
    def _on_translate_y_changed(self, value):
        """Handle Y translation change."""
        self.transform_params['translate_y'] = value
        self._emit_transform_changed()
        
    def _on_rotation_changed(self, value):
        """Handle rotation change."""
        self.transform_params['rotation'] = value
        self._emit_transform_changed()
        
    def _on_scale_changed(self, value):
        """Handle scale change."""
        self.transform_params['scale'] = value
        self._emit_transform_changed()
        
    def _emit_transform_changed(self):
        """Emit transform changed signal and update display."""
        print(f"Transform changed: {self.transform_params}")
        self.transform_changed.emit(self.transform_params.copy())
        # Direct update for real-time preview
        if hasattr(self, 'overlay_canvas'):
            print("Calling overlay_canvas.update_transform")
            self.overlay_canvas.update_transform(self.transform_params)
        else:
            print("No overlay_canvas found!")
        self._update_image_display()
        
    def _on_apply_clicked(self):
        """Handle apply button click."""
        self.alignment_applied.emit(self.transform_params.copy())
        
    def _on_reset_clicked(self):
        """Handle reset button click."""
        print("Reset button clicked")
        self.transform_params = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0
        }
        self._update_controls()
        self.reset_requested.emit()
        # Direct reset of overlay canvas
        if hasattr(self, 'overlay_canvas'):
            self.overlay_canvas.reset_transform()
        self._update_image_display()
        
    def _on_auto_align_clicked(self):
        """Handle auto align button click."""
        try:
            # Run automatic alignment and store results
            # This would typically call an auto alignment service
            # For now, emit current params and store them as auto params
            self._auto_alignment_params = self.transform_params.copy()
            self.alignment_applied.emit(self.transform_params.copy())
            
            # Update generate button state after auto alignment
            self._update_generate_button_state()
            
            logger.info("Auto alignment completed")
            
        except Exception as e:
            logger.error(f"Error in auto alignment: {e}")
        
    def _update_controls(self):
        """Update control values to match transform_params."""
        self.translate_x_slider.set_value(self.transform_params['translate_x'])
        self.translate_y_slider.set_value(self.transform_params['translate_y'])
        self.rotation_slider.set_value(self.transform_params['rotation'])
        self.scale_slider.set_value(self.transform_params['scale'])
        
    def set_transform(self, transform_params: Dict[str, float]):
        """Set transform parameters from external source."""
        self.transform_params.update(transform_params)
        self._update_controls()
        self._update_image_display()
    
    def set_structure_number(self, structure_num: int):
        """Set the current structure number for alignment operations."""
        try:
            self.current_structure_id = structure_num
            
            # Update generate button state
            self._update_generate_button_state()
            
            logger.debug(f"Structure number set: {structure_num}")
            
        except Exception as e:
            logger.error(f"Error setting structure number: {e}")
    
    def set_auto_alignment_params(self, params: Dict[str, float]):
        """Set parameters from automatic alignment process."""
        try:
            self._auto_alignment_params = params.copy()
            logger.info(f"Auto alignment parameters set: {params}")
            
            # Update generate button state
            self._update_generate_button_state()
            
        except Exception as e:
            logger.error(f"Error setting auto alignment params: {e}")
        
    def load_sem_image(self, image_array: np.ndarray):
        """Load SEM image into overlay canvas."""
        self.current_sem_image = image_array
        self.overlay_canvas.load_sem_image(image_array)
        
    def load_gds_overlay(self, image_array: np.ndarray):
        """Load GDS overlay into center display."""
        self.current_gds_image = image_array
        self._display_gds_image(image_array)
    
    def _on_structure_item_clicked(self, item):
        """Handle structure item selection."""
        structure_name = item.text()
        structure_id = item.data(32)  # Qt.UserRole
        
        if structure_id is not None:
            self.current_structure_id = structure_id
            self.gds_structure_selected.emit(self.current_gds_path or "", structure_id)
            logger.info(f"Structure selected: {structure_name} (ID: {structure_id})")
            self._update_generate_button_state()
    
    def _on_structure_selected(self, gds_path: str, structure_id: int):
        """Handle GDS structure selection from FileSelector."""
        # Store the selected structure info
        self.current_gds_path = gds_path
        self.current_structure_id = structure_id
        
        # Emit signal for external handling (loading structure image, etc.)
        self.gds_structure_selected.emit(gds_path, structure_id)
        logger.info(f"Structure selected: {structure_id} from {gds_path}")
        
        # Update generate button state
        self._update_generate_button_state()
    
    def update_structure_list(self, structures):
        """Update the structure list with available structures."""
        self.structure_list.clear()
        for i, structure in enumerate(structures):
            from PySide6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(structure.get('name', f'Structure {i}'))
            item.setData(32, i)  # Store structure ID
            self.structure_list.addItem(item)
    
    def _generate_aligned_gds(self):
        """Generate aligned GDS file using new bounds-based approach."""
        try:
            from pathlib import Path
            from datetime import datetime
            from src.services.simple_file_service import FileService
            
            self.status_label.setText("Generating aligned GDS...")
            
            # Check if we have structure selection
            if self.current_structure_id is None:
                raise ValueError("No structure selected for alignment")
            
            # Use current transform params or auto alignment params
            if hasattr(self, '_auto_alignment_params') and self._auto_alignment_params:
                params = self._auto_alignment_params
            else:
                params = self.transform_params
            
            # Convert to new format
            transform_params = {
                'move_x': params.get('translate_x', 0),
                'move_y': params.get('translate_y', 0),
                'rotation': params.get('rotation', 0),
                'zoom': params.get('scale', 1.0) * 100  # Convert to percentage
            }
            
            # Create output path with SEM image name
            output_dir = Path("Results/Aligned/manual")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get SEM image name if available
            sem_name = "unknown"
            if hasattr(self, 'current_sem_file_path') and self.current_sem_file_path:
                sem_name = Path(self.current_sem_file_path).stem
            
            filename = f"{sem_name}_Aligned.png"
            output_path = output_dir / filename
            
            # Use new approach to save aligned GDS
            file_service = FileService()
            success = file_service.save_aligned_gds(
                structure_num=self.current_structure_id,
                transform_params=transform_params,
                output_path=str(output_path)
            )
            
            if success:
                self.status_label.setText("Aligned GDS generated successfully")
                QMessageBox.information(self, "Success", f"Aligned GDS generated!\nSaved to: {output_path}")
                logger.info(f"Aligned GDS generated: {output_path}")
            else:
                raise Exception("Failed to generate aligned GDS")
            
        except Exception as e:
            logger.error(f"Error generating aligned GDS: {e}")
            self.status_label.setText("Error generating aligned GDS")
            QMessageBox.critical(self, "Error", f"Failed to generate aligned GDS: {e}")
    
    def _update_generate_button_state(self):
        """Update generate button state based on structure selection."""
        structure_selected = self.current_structure_id is not None
        
        # Enable generate button if we have structure selected
        self.generate_button.setEnabled(structure_selected)
        
        if structure_selected:
            self.generate_button.setToolTip("Generate aligned GDS file")
            logger.debug("Generate button enabled - ready for alignment")
        else:
            self.generate_button.setToolTip("Select a GDS structure first")
            logger.debug("Generate button disabled - no structure selected")
    
    def _connect_view_controls(self):
        """Connect view control button signals."""
        self.fit_button.clicked.connect(self._fit_to_window)
        self.reset_view_button.clicked.connect(self._reset_view)
        self.toggle_overlay_button.clicked.connect(self._toggle_overlay)
    
    def _fit_to_window(self):
        """Fit image to window."""
        self.status_label.setText("Fitting image to window...")
        logger.info("Fit to window requested")
    
    def _reset_view(self):
        """Reset view to original state."""
        self.status_label.setText("View reset")
        logger.info("View reset requested")
    
    def _toggle_overlay(self):
        """Toggle overlay visibility."""
        self.status_label.setText("Overlay toggled")
        logger.info("Overlay toggle requested")
        
    def _update_image_display(self):
        """Update the image display with current transform parameters."""
        if hasattr(self, 'overlay_canvas'):
            self.overlay_canvas.update_transform(self.transform_params)
        logger.debug(f"Image display updated with transform: {self.transform_params}")
    

        
    def reset(self):
        """Reset panel to default state."""
        # Reset transform parameters
        self.transform_params = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0
        }
        self._update_controls()
        self._update_image_display()
        
        # Reset structure selection
        if hasattr(self, 'structure_list'):
            self.structure_list.clearSelection()
        
        # Reset tracked file paths
        self.current_sem_file_path = None
        self.current_gds_path = None
        self.current_structure_id = None
        
        # Reset auto alignment params
        self._auto_alignment_params = None
        
        # Reset generate button
        self.generate_button.setEnabled(False)
        
        # Clear overlay canvas
        if hasattr(self, 'overlay_canvas'):
            self.overlay_canvas.reset_transform()
        
        logger.info("Alignment panel reset to default state")
        
    def get_current_state(self) -> Dict[str, Any]:
        """Get current panel state for session saving."""
        return {
            'transform_params': self.transform_params.copy(),
            'mode': getattr(self, 'current_mode', 'manual')
        }
        
    def set_state(self, state: Dict[str, Any]):
        """Set panel state from session loading."""
        if 'transform_params' in state:
            self.set_transform(state['transform_params'])
        if 'mode' in state:
            self.set_mode(state['mode'])
            
    def set_mode(self, mode: str):
        """Set alignment mode (manual/auto)."""
        self.current_mode = mode
        # Could be used to enable/disable controls based on mode
        
    def fit_canvas_to_view(self):
        """Fit image content to view."""
        self._fit_to_window()
        
    def reset_canvas_zoom(self):
        """Reset zoom to actual size."""
        self._reset_view()

    def get_current_config(self):
        """Get current alignment configuration for pipeline processing."""
        config = {
            'alignment_method': 'manual',  # Default for alignment panel
            'transform_parameters': {},
            'alignment_settings': {},
            'manual_adjustments': {},
            'current_preset': getattr(self, 'current_preset', None)
        }
        
        # Get current alignment method if selector exists
        if self.method_selector:
            config['alignment_method'] = self.method_selector.currentText().lower()
        
        # Get manual transformation parameters
        try:
            config['manual_adjustments'] = {
                'translation_x': getattr(self, 'translation_x', 0.0),
                'translation_y': getattr(self, 'translation_y', 0.0),
                'rotation': getattr(self, 'rotation', 0.0),
                'scale_x': getattr(self, 'scale_x', 1.0),
                'scale_y': getattr(self, 'scale_y', 1.0)
            }
        except Exception as e:
            logger.warning(f"Error extracting manual alignment parameters: {e}")
        
        # Get alignment settings from controls
        try:
            for control_name in ['threshold', 'max_features', 'confidence']:
                if hasattr(self, f'{control_name}_control'):
                    control = getattr(self, f'{control_name}_control')
                    if hasattr(control, 'value'):
                        config['alignment_settings'][control_name] = control.value()
        except Exception as e:
            logger.debug(f"Could not extract alignment settings: {e}")
        
        # Get transform matrix if available
        try:
            if self.alignment_service and hasattr(self.alignment_service, 'get_current_transform'):
                config['current_transform'] = self.alignment_service.get_current_transform()
        except Exception as e:
            logger.debug(f"Could not get current transform: {e}")
        
        # Get preset information
        if self.preset_selector:
            config['selected_preset'] = self.preset_selector.currentText()
        
        # Get current structure selection
        config['structure_selection'] = {
            'gds_path': self.current_gds_path,
            'structure_id': self.current_structure_id
        }
        
        return config
        
    def show_progress(self, progress_info):
        """Show pipeline progress information in the alignment panel."""
        stage = progress_info.get('stage', '')
        status = progress_info.get('status', '')

        if hasattr(self, 'status_label'):
            self.status_label.setText(f"Pipeline: {stage.title()} - {status}")
    
    def get_structure_list(self):
        """Get reference to the structure list widget."""
        return getattr(self, 'structure_list', None)
    
    def get_current_structure_selection(self) -> Dict[str, Any]:
        """Get current structure selection."""
        return {
            'gds_path': self.current_gds_path,
            'structure_id': self.current_structure_id
        }
        
class OverlayCanvas(QGraphicsView):
    """Simple overlay canvas for displaying SEM and GDS images."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        
        # Image items
        self.sem_item = None
        self.gds_item = None
        
        # Transform parameters
        self.transform_params = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0
        }
        
        self.setup_canvas()
        
    def setup_canvas(self):
        """Initialize canvas settings."""
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setMinimumSize(400, 300)
        
    def load_sem_image(self, image_array: np.ndarray):
        """Load SEM image as background."""
        if self.sem_item:
            self._scene.removeItem(self.sem_item)
            
        # Convert numpy array to QPixmap
        height, width = image_array.shape[:2]
        if len(image_array.shape) == 2:
            # Grayscale
            image_array = np.stack([image_array] * 3, axis=2)
            
        q_image = QImage(image_array.data, width, height, 
                        width * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        self.sem_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self.sem_item)
        
        # Fit view to image
        self.fitInView(self.sem_item, Qt.AspectRatioMode.KeepAspectRatio)
        
    def load_gds_overlay(self, image_array: np.ndarray):
        """Load GDS overlay image."""
        if self.gds_item:
            self._scene.removeItem(self.gds_item)
            
        # Convert numpy array to QPixmap with transparency
        height, width = image_array.shape[:2]
        if len(image_array.shape) == 2:
            # Convert grayscale to RGBA
            rgba_array = np.zeros((height, width, 4), dtype=np.uint8)
            rgba_array[:, :, 0] = image_array  # Red channel
            rgba_array[:, :, 3] = (image_array > 0) * 128  # Alpha channel
        else:
            rgba_array = image_array
            
        q_image = QImage(rgba_array.data, width, height,
                        width * 4, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_image)
        
        self.gds_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self.gds_item)
        
        # Apply current transform
        self.update_transform(self.transform_params)
        
    def update_transform(self, transform_params: Dict[str, float]):
        """Update GDS overlay transform."""
        print(f"OverlayCanvas.update_transform called with: {transform_params}")
        if not self.gds_item:
            print("No GDS item to transform!")
            return
        
        print(f"GDS item exists: {self.gds_item}")
        print(f"Current GDS position: {self.gds_item.pos()}")
        
        # Get transform values
        translate_x = transform_params.get('translate_x', 0.0)
        translate_y = transform_params.get('translate_y', 0.0)
        rotation = transform_params.get('rotation', 0.0)
        scale = transform_params.get('scale', 1.0)
        
        print(f"Setting position to: ({translate_x}, {translate_y})")
        
        # Force movement - just translation first
        self.gds_item.setPos(translate_x, translate_y)
        
        print(f"New GDS position: {self.gds_item.pos()}")
        
        # Force scene update
        self._scene.update()
        self.update()
        
    def reset_transform(self):
        """Reset GDS overlay to original position."""
        if self.gds_item:
            self.gds_item.resetTransform()
            self.gds_item.setPos(0, 0)
        
        self.transform_params = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0
        }
        
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            # Zoom in
            self.scale(zoom_factor, zoom_factor)
        else:
            # Zoom out
            self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)
            
    def mousePressEvent(self, event):
        """Handle mouse press for panning."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        super().mouseReleaseEvent(event)
        
    def fit_in_view(self):
        """Fit all content in view."""
        if self._scene.items():
            self.fitInView(self._scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
