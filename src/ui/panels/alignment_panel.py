"""
Alignment Panel
Restructured alignment panel with sub-mode switcher, transformation controls, and file selection.
Layout: Left (sub-modes + controls) | Center (image display) | Right (file selection)
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
from ..components.file_selector import FileSelector
from .alignment_sub_mode_switcher import AlignmentSubModeSwitcher
from typing import Dict, Any, Optional
import numpy as np
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class AlignmentPanel(QWidget):
    """Alignment panel with sub-mode switcher, transformation controls, and file selection."""
    
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
        self.three_point_controller = None
        self.transformation_preview = None
        
        # Current images for alignment
        self.current_sem_image = None
        self.current_gds_image = None
        
        self.setup_ui()
        self._connect_workflow_signals()
        
    def setup_ui(self):
        """Initialize the UI components with new 3-panel layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Create main splitter with 3 sections
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Left Panel: Sub-mode switcher + Controls
        left_panel = self._create_left_panel()
        self.main_splitter.addWidget(left_panel)
        
        # Center Panel: Image Display
        center_panel = self._create_center_panel()
        self.main_splitter.addWidget(center_panel)
        
        # Right Panel: File Selection
        right_panel = self._create_right_panel()
        self.main_splitter.addWidget(right_panel)
        
        # Set splitter proportions (25% left, 50% center, 25% right)
        self.main_splitter.setSizes([300, 600, 300])
        
        layout.addWidget(self.main_splitter)
        
        # Connect view controls
        self._connect_view_controls()
        
    def _create_left_panel(self) -> QWidget:
        """Create the left panel with sub-mode switcher and controls."""
        left_widget = QWidget()
        left_widget.setMaximumWidth(300)
        left_widget.setMinimumWidth(250)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)
        
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
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #2E86AB;")
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
        
        # Image display area
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setMinimumSize(400, 400)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #cccccc; background-color: #f8f9fa;")
        self.image_label.setText("No image loaded\n\nSelect SEM and GDS files to begin alignment")
        self.image_label.setMinimumSize(400, 400)
        
        self.image_scroll.setWidget(self.image_label)
        center_layout.addWidget(self.image_scroll)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666666; font-style: italic;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        
        center_layout.addWidget(self.status_label)
        center_layout.addWidget(self.progress_bar)
        
        return center_widget
    
    def _create_right_panel(self) -> QWidget:
        """Create the right panel with file selection."""
        right_widget = QWidget()
        right_widget.setMaximumWidth(300)
        right_widget.setMinimumWidth(250)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)
        
        # File Selection Section
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        
        # SEM file selection
        sem_label = QLabel("Select SEM:")
        sem_label.setStyleSheet("font-weight: bold; color: #2E86AB;")
        file_layout.addWidget(sem_label)
        
        self.sem_combo = QPushButton("Browse SEM Files...")
        self.sem_combo.clicked.connect(self._browse_sem_files)
        file_layout.addWidget(self.sem_combo)
        
        self.sem_status = QLabel("No SEM file selected")
        self.sem_status.setStyleSheet("color: #666666; font-size: 10px;")
        file_layout.addWidget(self.sem_status)
        
        file_layout.addSpacing(15)
        
        # GDS structure selection
        gds_label = QLabel("Select GDS Structure:")
        gds_label.setStyleSheet("font-weight: bold; color: #2E86AB;")
        file_layout.addWidget(gds_label)
        
        self.gds_combo = QPushButton("Browse GDS Files...")
        self.gds_combo.clicked.connect(self._browse_gds_files)
        file_layout.addWidget(self.gds_combo)
        
        self.structure_combo = QPushButton("Select Structure...")
        self.structure_combo.setEnabled(False)
        self.structure_combo.clicked.connect(self._select_structure)
        file_layout.addWidget(self.structure_combo)
        
        self.gds_status = QLabel("No GDS file selected")
        self.gds_status.setStyleSheet("color: #666666; font-size: 10px;")
        file_layout.addWidget(self.gds_status)
        
        right_layout.addWidget(file_group)
        
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
                background-color: #cccccc;
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

        # Progress and status widgets
        self.status_label = QLabel("Ready.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        manual_layout.addWidget(self.status_label)
        manual_layout.addWidget(self.progress_bar)
        
        return manual_widget
    
    def _create_hybrid_controls(self) -> QWidget:
        """Create hybrid alignment controls (3-point selection)."""
        hybrid_widget = QWidget()
        hybrid_layout = QVBoxLayout(hybrid_widget)
        
        # Create splitter for 3-point selection and preview
        splitter = QSplitter(Qt.Vertical)
        
        # Three-point selection controller (Step 10)
        self.three_point_controller = ThreePointSelectionController()
        splitter.addWidget(self.three_point_controller)
        
        # Transformation preview widget (Step 11)
        self.transformation_preview = TransformationPreviewWidget()
        splitter.addWidget(self.transformation_preview)
        
        # Set splitter proportions (60% selection, 40% preview)
        splitter.setSizes([400, 300])
        
        hybrid_layout.addWidget(splitter)
        
        return hybrid_widget
    
    def _create_automatic_controls(self) -> QWidget:
        """Create automatic alignment controls."""
        auto_widget = QWidget()
        auto_layout = QVBoxLayout(auto_widget)
        
        # Placeholder for automatic controls
        label = QLabel("Automatic alignment controls will be here.")
        label.setAlignment(Qt.AlignCenter)
        auto_layout.addWidget(label)
        
        return auto_widget
    
    def _connect_workflow_signals(self):
        """Connect signals for the Step 11 workflow."""
        if self.three_point_controller and self.transformation_preview:
            # Connect 3-point selection to transformation preview
            self.three_point_controller.alignment_ready.connect(
                self._on_points_ready
            )
            
            # Connect transformation preview signals
            self.transformation_preview.transformation_calculated.connect(
                self.transformation_calculated.emit
            )
            self.transformation_preview.transformation_confirmed.connect(
                self.transformation_confirmed.emit
            )
            # Step 14: Connect preview display signal
            self.transformation_preview.preview_updated.connect(
                self._on_preview_updated
            )
    
    def _on_points_ready(self, sem_points: list, gds_points: list):
        """Handle when 3 point pairs are ready for transformation calculation."""
        logger.info(f"3 point pairs ready: {len(sem_points)} SEM, {len(gds_points)} GDS")
        
        # Emit signal for external handling
        self.three_points_selected.emit(sem_points, gds_points)
        
        # Pass points to transformation preview widget
        self._expose_points_to_preview(sem_points, gds_points)
    
    def _expose_points_to_preview(self, sem_points: list, gds_points: list):
        """Expose selected 3-point data to the transformation preview widget."""
        if self.transformation_preview:
            self.transformation_preview.set_point_pairs(sem_points, gds_points)
            logger.info("Point pairs exposed to transformation preview widget")
    
    def _on_preview_updated(self, preview_data):
        """Handle preview update from transformation widget."""
        logger.info("Transformation preview updated")
        # Emit signal to notify other components
        self.preview_updated.emit(preview_data)
    
    def set_images_for_selection(self, sem_image: np.ndarray, gds_image: np.ndarray):
        """Set images for 3-point selection in hybrid mode."""
        if self.three_point_controller:
            self.three_point_controller.set_images(sem_image, gds_image)
    
    def set_images_for_preview(self, sem_image: np.ndarray, gds_image: np.ndarray):
        """Set images for transformation preview."""
        if self.transformation_preview:
            self.transformation_preview.set_reference_images(sem_image, gds_image)
    
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
    
    # Override the existing _display_gds_image to use center panel
    def _display_gds_image(self, gds_image: np.ndarray):
        """Display GDS image in the center panel."""
        try:
            if gds_image is not None and gds_image.size > 0:
                height, width = gds_image.shape
                
                # Convert binary image to RGB for display
                if len(gds_image.shape) == 2:  # Grayscale/binary
                    # Create colored version (cyan like in your screenshot)
                    display_image = np.zeros((height, width, 3), dtype=np.uint8)
                    display_image[gds_image > 0] = [0, 255, 255]  # Cyan color for structures
                else:
                    display_image = gds_image
                
                # Use the center panel display method
                non_zero = np.count_nonzero(gds_image)
                title = f"GDS Structure: {gds_image.shape}, {non_zero} active pixels"
                self._display_image_in_center(display_image, title)
                
                print(f"Structure image displayed in center panel: {width}x{height}")
                logger.info(f"GDS image displayed: {gds_image.shape}, {non_zero} pixels")
            else:
                self.image_label.clear()
                self.image_label.setText("No Structure Selected")
                
        except Exception as e:
            logger.error(f"Failed to display GDS image: {e}")
            self.image_label.setText(f"Error displaying image: {e}")
    
    def set_initial_sem_image(self, sem_image: np.ndarray):
        """Set initial SEM image for alignment operations."""
        # Store the SEM image for later use
        self.current_sem_image = sem_image
        
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
        self.transform_changed.emit(self.transform_params.copy())
        # Update image display if available
        self._update_image_display()
        
    def _on_apply_clicked(self):
        """Handle apply button click."""
        self.alignment_applied.emit(self.transform_params.copy())
        
    def _on_reset_clicked(self):
        """Handle reset button click."""
        self.transform_params = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0
        }
        self._update_controls()
        self.reset_requested.emit()
        self._update_image_display()
        
    def _on_auto_align_clicked(self):
        """Handle auto align button click."""
        # Placeholder for auto alignment
        self.alignment_applied.emit(self.transform_params.copy())
        
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
        
    def load_sem_image(self, image_array: np.ndarray):
        """Load SEM image into center display."""
        self.current_sem_image = image_array
        self._display_image_in_center(image_array, "SEM Image")
        
    def load_gds_overlay(self, image_array: np.ndarray):
        """Load GDS overlay into center display."""
        self.current_gds_image = image_array
        self._display_gds_image(image_array)
    
    def _browse_sem_files(self):
        """Browse and select SEM files."""
        try:
            # Get SEM data directory
            data_dir = Path(__file__).parent.parent.parent.parent / "Data" / "SEM"
            if not data_dir.exists():
                data_dir = Path.home()
            
            # Open file dialog
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select SEM Image File",
                str(data_dir),
                "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)"
            )
            
            if file_path:
                self.sem_combo.setText(Path(file_path).name)
                self.sem_status.setText(f"Selected: {Path(file_path).name}")
                self.sem_status.setStyleSheet("color: #4CAF50; font-size: 10px;")
                
                # Emit signal for file loading
                self.sem_file_selected.emit(file_path)
                logger.info(f"SEM file selected: {file_path}")
                
                # Update generate button state
                self._update_generate_button_state()
                
        except Exception as e:
            logger.error(f"Error browsing SEM files: {e}")
            QMessageBox.warning(self, "Error", f"Failed to browse SEM files: {e}")
    
    def _browse_gds_files(self):
        """Browse and select GDS files."""
        try:
            # Get GDS data directory  
            data_dir = Path(__file__).parent.parent.parent.parent / "Data" / "GDS"
            if not data_dir.exists():
                data_dir = Path.home()
            
            # Open file dialog
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select GDS File", 
                str(data_dir),
                "GDS Files (*.gds *.gds2 *.gdsii);;All Files (*)"
            )
            
            if file_path:
                self.gds_combo.setText(Path(file_path).name)
                self.gds_status.setText(f"Selected: {Path(file_path).name}")
                self.gds_status.setStyleSheet("color: #4CAF50; font-size: 10px;")
                
                # Enable structure selection
                self.structure_combo.setEnabled(True)
                self.structure_combo.setText("Select Structure...")
                
                # Emit signal for file loading
                self.gds_file_loaded.emit(file_path)
                logger.info(f"GDS file selected: {file_path}")
                
                # Update generate button state
                self._update_generate_button_state()
                
        except Exception as e:
            logger.error(f"Error browsing GDS files: {e}")
            QMessageBox.warning(self, "Error", f"Failed to browse GDS files: {e}")
    
    def _select_structure(self):
        """Select GDS structure from available structures."""
        try:
            structures = ["Structure 1 - IP935Left_11", "Structure 2 - IP935Left_14", "Structure 3 - QC855GC_CROSS"]
            
            # Create a simple selection dialog
            from PySide6.QtWidgets import QInputDialog
            structure, ok = QInputDialog.getItem(
                self, 
                "Select GDS Structure", 
                "Choose structure:",
                structures, 
                0, 
                False
            )
            
            if ok and structure:
                self.structure_combo.setText(structure)
                self.gds_status.setText(f"Structure: {structure}")
                self.gds_status.setStyleSheet("color: #4CAF50; font-size: 10px;")
                
                # Extract structure ID
                structure_id = structures.index(structure) + 1
                
                # Emit signal for structure selection
                self.gds_structure_selected.emit("", structure_id)
                logger.info(f"GDS structure selected: {structure} (ID: {structure_id})")
                
                # Update generate button state
                self._update_generate_button_state()
                
        except Exception as e:
            logger.error(f"Error selecting structure: {e}")
            QMessageBox.warning(self, "Error", f"Failed to select structure: {e}")
    
    def _generate_aligned_gds(self):
        """Generate aligned GDS file."""
        try:
            self.status_label.setText("Generating aligned GDS...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Simulate progress
            for i in range(0, 101, 10):
                self.progress_bar.setValue(i)
                QApplication.processEvents()
                
            # Emit signal for generation
            self.alignment_applied.emit(self.transform_params.copy())
            
            self.status_label.setText("Aligned GDS generated successfully")
            self.progress_bar.setVisible(False)
            
            QMessageBox.information(self, "Success", "Aligned GDS file generated successfully!")
            logger.info("Aligned GDS generated")
            
        except Exception as e:
            logger.error(f"Error generating aligned GDS: {e}")
            self.status_label.setText("Error generating aligned GDS")
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to generate aligned GDS: {e}")
    
    def _update_generate_button_state(self):
        """Update the generate button enabled state based on file selection."""
        # Check if both SEM and GDS files are selected
        sem_selected = "Selected:" in self.sem_status.text()
        gds_selected = "Selected:" in self.gds_status.text()
        structure_selected = "Structure:" in self.gds_status.text()
        
        # Enable generate button if all required files are selected
        if sem_selected and gds_selected and structure_selected:
            self.generate_button.setEnabled(True)
            self.generate_button.setToolTip("Generate aligned GDS file")
        else:
            self.generate_button.setEnabled(False)
            missing = []
            if not sem_selected:
                missing.append("SEM file")
            if not gds_selected:
                missing.append("GDS file")
            if not structure_selected:
                missing.append("structure")
            self.generate_button.setToolTip(f"Missing: {', '.join(missing)}")
    
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
        logger.debug(f"Image display updated with transform: {self.transform_params}")
    
    def _display_image_in_center(self, image_array: np.ndarray, title: str = ""):
        """Display image in the center panel."""
        try:
            if image_array is not None and image_array.size > 0:
                height, width = image_array.shape[:2]
                
                # Convert to RGB if needed
                if len(image_array.shape) == 2:  # Grayscale
                    display_image = np.stack([image_array] * 3, axis=2)
                else:
                    display_image = image_array
                
                # Convert to Qt format
                bytes_per_line = 3 * width
                qimage = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                
                # Convert to pixmap and display
                pixmap = QPixmap.fromImage(qimage)
                
                # Scale to fit display area while maintaining aspect ratio
                label_size = self.image_label.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.setText("")  # Clear placeholder text
                
                # Update title if provided
                if title:
                    self.image_label.setToolTip(title)
                
                logger.info(f"Image displayed in center panel: {width}x{height}")
            else:
                self.image_label.clear()
                self.image_label.setText("No image loaded")
                
        except Exception as e:
            logger.error(f"Failed to display image in center panel: {e}")
            self.image_label.setText(f"Error displaying image: {e}")
        
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
        
        # Reset file selections
        self.sem_combo.setText("Browse SEM Files...")
        self.sem_status.setText("No SEM file selected")
        self.sem_status.setStyleSheet("color: #666666; font-size: 10px;")
        
        self.gds_combo.setText("Browse GDS Files...")
        self.gds_status.setText("No GDS file selected")
        self.gds_status.setStyleSheet("color: #666666; font-size: 10px;")
        
        self.structure_combo.setText("Select Structure...")
        self.structure_combo.setEnabled(False)
        
        self.generate_button.setEnabled(False)
        
        # Clear image display
        self.image_label.clear()
        self.image_label.setText("No image loaded\n\nSelect SEM and GDS files to begin alignment")
        
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
        if hasattr(self, 'method_selector'):
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
            if hasattr(self.alignment_service, 'get_current_transform'):
                config['current_transform'] = self.alignment_service.get_current_transform()
        except Exception as e:
            logger.debug(f"Could not get current transform: {e}")
        
        # Get preset information
        if hasattr(self, 'preset_selector'):
            config['selected_preset'] = self.preset_selector.currentText()
        
        return config
        
    def show_progress(self, progress_info):
        """Show pipeline progress information in the alignment panel."""
        stage = progress_info.get('stage', '')
        status = progress_info.get('status', '')
        progress = progress_info.get('progress', 0)

        if hasattr(self, 'status_label'):
            self.status_label.setText(f"Pipeline: {stage.title()} - {status}")
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(progress)
            self.progress_bar.setVisible(True if progress < 100 else False)
        
class OverlayCanvas(QGraphicsView):
    """Simple overlay canvas for displaying SEM and GDS images."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
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
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setMinimumSize(400, 300)
        
    def load_sem_image(self, image_array: np.ndarray):
        """Load SEM image as background."""
        if self.sem_item:
            self.scene.removeItem(self.sem_item)
            
        # Convert numpy array to QPixmap
        height, width = image_array.shape[:2]
        if len(image_array.shape) == 2:
            # Grayscale
            image_array = np.stack([image_array] * 3, axis=2)
            
        q_image = QImage(image_array.data, width, height, 
                        width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        self.sem_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.sem_item)
        
        # Fit view to image
        self.fitInView(self.sem_item, Qt.KeepAspectRatio)
        
    def load_gds_overlay(self, image_array: np.ndarray):
        """Load GDS overlay image."""
        if self.gds_item:
            self.scene.removeItem(self.gds_item)
            
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
                        width * 4, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_image)
        
        self.gds_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.gds_item)
        
        # Apply current transform
        self.update_transform(self.transform_params)
        
    def update_transform(self, transform_params: Dict[str, float]):
        """Update GDS overlay transform."""
        if not self.gds_item:
            return
            
        self.transform_params = transform_params.copy()
        
        # Reset transform
        self.gds_item.resetTransform()
        
        # Apply transformations in order
        # 1. Scale
        scale = transform_params.get('scale', 1.0)
        self.gds_item.setScale(scale)
        
        # 2. Rotation
        rotation = transform_params.get('rotation', 0.0)
        self.gds_item.setRotation(rotation)
        
        # 3. Translation
        translate_x = transform_params.get('translate_x', 0.0)
        translate_y = transform_params.get('translate_y', 0.0)
        current_pos = self.gds_item.pos()
        self.gds_item.setPos(current_pos.x() + translate_x, 
                           current_pos.y() + translate_y)
        
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
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.RubberBandDrag)
        super().mouseReleaseEvent(event)
        
    def fit_in_view(self):
        """Fit all content in view."""
        if self.scene.items():
            self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
