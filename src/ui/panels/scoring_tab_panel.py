"""
Scoring Tab Panel - Unified Scoring Interface for Main Window

This module provides a scoring tab panel for the unified main window interface,
featuring scoring method selection, calculation controls, and results display
for alignment quality assessment.

Main Class:
- ScoringTabPanel: Panel for scoring operations in the main tab widget

Key Methods:
- setup_ui(): Initializes UI with method selection and results display
- setup_styling(): Applies dark theme styling to all components
- _on_method_changed(): Handles scoring method selection changes
- _on_calculate_clicked(): Handles calculate scores button clicks
- display_results(): Displays scoring results in formatted text

Signals Emitted:
- scoring_method_changed(str): Scoring method selection changed
- calculate_scores_requested(str): Score calculation requested for method

Dependencies:
- Uses: PySide6.QtWidgets, PySide6.QtCore (Qt framework)
- Called by: Main window tab widget and scoring workflow
- Coordinates with: Scoring services and result processing

Scoring Methods:
- SSIM: Structural Similarity Index Measure
- MSE: Mean Squared Error
- PSNR: Peak Signal-to-Noise Ratio
- Cross-Correlation: Normalized cross-correlation

Features:
- Dropdown method selection with multiple scoring algorithms
- Calculate button for triggering score computation
- Results display with formatted text output
- Dark theme styling consistent with application design
- Read-only results area with monospace font for clarity
- Grouped UI elements with clear visual organization
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QComboBox, QPushButton, QTextEdit, QGroupBox, QFileDialog)
from PySide6.QtCore import Signal, Qt
import cv2
import os


class ScoringTabPanel(QWidget):
    """Panel for scoring operations in the main tab widget."""
    
    scoring_method_changed = Signal(str)
    calculate_scores_requested = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_method = "SSIM"
        self.setup_ui()
        self.setup_styling()
    
    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        
        # Image loading
        load_group = QGroupBox("Load Images")
        load_layout = QVBoxLayout(load_group)
        
        load_buttons_layout = QHBoxLayout()
        
        self.load_sem_btn = QPushButton("Load SEM Image")
        self.load_sem_btn.clicked.connect(self._on_load_sem_clicked)
        load_buttons_layout.addWidget(self.load_sem_btn)
        
        self.load_gds_btn = QPushButton("Load GDS Image")
        self.load_gds_btn.clicked.connect(self._on_load_gds_clicked)
        load_buttons_layout.addWidget(self.load_gds_btn)
        
        load_layout.addLayout(load_buttons_layout)
        
        # Show GDS toggle button
        self.show_gds_btn = QPushButton("Show GDS")
        self.show_gds_btn.setCheckable(True)
        self.show_gds_btn.clicked.connect(self._on_show_gds_clicked)
        load_layout.addWidget(self.show_gds_btn)
        
        layout.addWidget(load_group)
        
        # Method selection
        method_group = QGroupBox("Scoring Method")
        method_layout = QVBoxLayout(method_group)
        
        method_selection_layout = QHBoxLayout()
        method_selection_layout.addWidget(QLabel("Method:"))
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(["SSIM", "MSE", "PSNR", "Cross-Correlation", "Binary Pixel Match"])
        self.method_combo.setCurrentText(self.current_method)
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        method_selection_layout.addWidget(self.method_combo)
        
        method_layout.addLayout(method_selection_layout)
        
        # Calculate button
        self.calculate_btn = QPushButton("Calculate Scores")
        self.calculate_btn.clicked.connect(self._on_calculate_clicked)
        method_layout.addWidget(self.calculate_btn)
        
        layout.addWidget(method_group)
        
        # Results display
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setMaximumHeight(200)
        self.results_display.setPlainText("No scores calculated yet.")
        results_layout.addWidget(self.results_display)
        
        layout.addWidget(results_group)
        layout.addStretch()
    
    def setup_styling(self):
        """Setup dark theme styling."""
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444444;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px 0 4px;
            }
            QComboBox {
                border: 1px solid #444444;
                border-radius: 3px;
                padding: 4px;
                background-color: #3c3c3c;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QTextEdit {
                border: 1px solid #444444;
                border-radius: 3px;
                background-color: #3c3c3c;
                font-family: monospace;
            }
        """)
    
    def _on_method_changed(self, method):
        """Handle scoring method change."""
        self.current_method = method
        self.scoring_method_changed.emit(method)
    
    def _on_calculate_clicked(self):
        """Handle calculate button click."""
        if self.current_method == "Binary Pixel Match":
            self._calculate_binary_match()
        else:
            self.calculate_scores_requested.emit(self.current_method)
    
    def _on_load_sem_clicked(self):
        """Handle load SEM image button click."""
        default_dir = os.path.join(os.getcwd(), "Results", "SEM_Filters")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load SEM Image", default_dir,
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self._load_and_display_image(file_path, "sem")
    
    def _on_load_gds_clicked(self):
        """Handle load GDS image button click."""
        default_dir = os.path.join(os.getcwd(), "Results", "Aligned")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load GDS Image", default_dir,
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self._load_and_display_image(file_path, "gds")
    
    def _load_and_display_image(self, file_path, image_type):
        """Load and display image in main UI."""
        try:
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if image is not None:
                # Get parent main window and update display
                main_window = self.window()
                if hasattr(main_window, 'image_viewer'):
                    if image_type == "sem":
                        main_window.image_viewer.set_sem_image(image)
                        main_window.current_sem_image = image
                    elif image_type == "gds":
                        # Convert to white background, black structure
                        processed_image = self._process_gds_image(image)
                        main_window.image_viewer.set_gds_overlay(processed_image)
                        main_window.current_gds_overlay = processed_image
                        main_window.image_viewer.set_overlay_visible(True)
        except Exception as e:
            print(f"Error loading {image_type} image: {e}")
    
    def _on_show_gds_clicked(self):
        """Toggle GDS overlay visibility with transparency."""
        main_window = self.window()
        if hasattr(main_window, 'image_viewer'):
            if self.show_gds_btn.isChecked():
                main_window.image_viewer.set_overlay_visible(True)
                main_window.image_viewer.set_overlay_alpha(0.7)  # 30% transparency
                self.show_gds_btn.setText("Hide GDS")
            else:
                main_window.image_viewer.set_overlay_visible(False)
                self.show_gds_btn.setText("Show GDS")
    
    def _process_gds_image(self, image):
        """Convert GDS image to white background, black structure."""
        import numpy as np
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Invert: make background white (255) and structure black (0)
        inverted = cv2.bitwise_not(gray)
        
        # Convert back to 3-channel for overlay
        return cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
    
    def _calculate_binary_match(self):
        """Calculate binary pixel matching score."""
        try:
            main_window = self.window()
            if hasattr(main_window, 'current_sem_image') and hasattr(main_window, 'current_gds_overlay'):
                sem_img = main_window.current_sem_image
                gds_img = main_window.current_gds_overlay
                
                if sem_img is not None and gds_img is not None:
                    # Convert to grayscale if needed
                    if len(sem_img.shape) == 3:
                        sem_gray = cv2.cvtColor(sem_img, cv2.COLOR_BGR2GRAY)
                    else:
                        sem_gray = sem_img
                    
                    if len(gds_img.shape) == 3:
                        gds_gray = cv2.cvtColor(gds_img, cv2.COLOR_BGR2GRAY)
                    else:
                        gds_gray = gds_img
                    
                    # Resize to same dimensions if needed
                    if sem_gray.shape != gds_gray.shape:
                        gds_gray = cv2.resize(gds_gray, (sem_gray.shape[1], sem_gray.shape[0]))
                    
                    # Convert to binary (threshold at 127)
                    _, sem_binary = cv2.threshold(sem_gray, 127, 255, cv2.THRESH_BINARY)
                    _, gds_binary = cv2.threshold(gds_gray, 127, 255, cv2.THRESH_BINARY)
                    
                    # Calculate matching pixels
                    total_pixels = sem_binary.size
                    matching_pixels = (sem_binary == gds_binary).sum()
                    match_percentage = (matching_pixels / total_pixels) * 100
                    
                    # Calculate overlap metrics
                    sem_white = (sem_binary == 255).sum()
                    gds_white = (gds_binary == 255).sum()
                    overlap_white = ((sem_binary == 255) & (gds_binary == 255)).sum()
                    
                    results = {
                        "Overall Match": f"{match_percentage:.2f}%",
                        "Matching Pixels": f"{matching_pixels:,} / {total_pixels:,}",
                        "SEM White Pixels": f"{sem_white:,}",
                        "GDS White Pixels": f"{gds_white:,}",
                        "Overlapping White": f"{overlap_white:,}",
                        "Precision": f"{(overlap_white/gds_white*100):.2f}%" if gds_white > 0 else "N/A",
                        "Recall": f"{(overlap_white/sem_white*100):.2f}%" if sem_white > 0 else "N/A"
                    }
                    
                    self.display_results(results)
                else:
                    self.results_display.setPlainText("Please load both SEM and GDS images first.")
            else:
                self.results_display.setPlainText("No images available for comparison.")
        except Exception as e:
            self.results_display.setPlainText(f"Error calculating binary match: {e}")
            print(f"Error in binary match calculation: {e}")
    
    def display_results(self, scores):
        """Display scoring results."""
        if not scores:
            self.results_display.setPlainText("No scores available.")
            return
        
        result_text = f"Scoring Results ({self.current_method}):\n\n"
        
        for key, value in scores.items():
            if isinstance(value, float):
                result_text += f"{key}: {value:.4f}\n"
            else:
                result_text += f"{key}: {value}\n"
        
        self.results_display.setPlainText(result_text)