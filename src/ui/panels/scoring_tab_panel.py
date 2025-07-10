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
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


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
        self.results_display.setMaximumHeight(150)
        self.results_display.setPlainText("No scores calculated yet.")
        results_layout.addWidget(self.results_display)
        
        layout.addWidget(results_group)
        
        # Chart display
        chart_group = QGroupBox("Alignment Chart")
        chart_layout = QVBoxLayout(chart_group)
        
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMaximumHeight(200)
        chart_layout.addWidget(self.canvas)
        
        layout.addWidget(chart_group)
        
        # Save buttons
        save_group = QGroupBox("Save Results")
        save_layout = QVBoxLayout(save_group)
        
        save_buttons_layout = QHBoxLayout()
        
        self.save_overlay_btn = QPushButton("Save Overlay Image")
        self.save_overlay_btn.clicked.connect(self._save_overlay_image)
        save_buttons_layout.addWidget(self.save_overlay_btn)
        
        self.save_chart_btn = QPushButton("Save Chart")
        self.save_chart_btn.clicked.connect(self._save_chart)
        save_buttons_layout.addWidget(self.save_chart_btn)
        
        save_layout.addLayout(save_buttons_layout)
        
        self.save_all_btn = QPushButton("Save All Results")
        self.save_all_btn.clicked.connect(self._save_all_results)
        save_layout.addWidget(self.save_all_btn)
        
        layout.addWidget(save_group)
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
            QPushButton:disabled {
                background-color: #666666;
                color: #999999;
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
                        # Store original for scoring
                        self.original_sem_image = image
                    elif image_type == "gds":
                        # Store original GDS image
                        self.original_gds_image = image
                        # Display with original colors (transparent background, black structures)
                        rgba_display = self._create_display_overlay(image)
                        main_window.image_viewer.set_gds_overlay(rgba_display)
                        main_window.current_gds_overlay = rgba_display
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
    
    def _create_display_overlay(self, image):
        """Create display overlay with transparent background and black structures."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create RGBA image with alpha channel
        rgba_image = np.zeros((gray.shape[0], gray.shape[1], 4), dtype=np.uint8)
        
        # Black pixels are structures, make them visible
        structure_mask = gray < 127
        rgba_image[structure_mask] = [0, 0, 0, 255]  # Black opaque structures
        # Background remains transparent
        
        return rgba_image
    
    def _process_gds_for_scoring(self, image):
        """Convert GDS image to white background, black structure for scoring calculation."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create binary image: white background (255), black structures (0)
        binary_image = np.full(gray.shape, 255, dtype=np.uint8)
        structure_mask = gray < 127
        binary_image[structure_mask] = 0
        
        return binary_image
    
    def _calculate_binary_match(self):
        """Calculate binary pixel matching score with color overlay."""
        try:
            if not hasattr(self, 'original_sem_image') or not hasattr(self, 'original_gds_image'):
                self.results_display.setPlainText("Please load both SEM and GDS images first.")
                return
            
            sem_img = self.original_sem_image
            gds_img = self.original_gds_image
            
            if sem_img is not None and gds_img is not None:
                # Convert to grayscale if needed
                if len(sem_img.shape) == 3:
                    sem_gray = cv2.cvtColor(sem_img, cv2.COLOR_BGR2GRAY)
                else:
                    sem_gray = sem_img
                
                # Process GDS for scoring calculation (white background, black structures)
                gds_binary_img = self._process_gds_for_scoring(gds_img)
                
                # Resize to same dimensions if needed
                if sem_gray.shape != gds_binary_img.shape:
                    gds_binary_img = cv2.resize(gds_binary_img, (sem_gray.shape[1], sem_gray.shape[0]))
                
                # Convert SEM to binary (threshold at 127)
                _, sem_binary = cv2.threshold(sem_gray, 127, 255, cv2.THRESH_BINARY_INV)  # Black = structure
                _, gds_binary = cv2.threshold(gds_binary_img, 127, 255, cv2.THRESH_BINARY_INV)  # Black = structure
                
                # Create color overlay image
                overlay_img = np.zeros((sem_binary.shape[0], sem_binary.shape[1], 3), dtype=np.uint8)
                overlay_img.fill(255)  # White background
                
                # Color coding:
                # Black: Both SEM and GDS have structures (match)
                # Red: SEM has structure, GDS doesn't (SEM only)
                # Blue: GDS has structure, SEM doesn't (GDS only)
                both_structure = (sem_binary == 255) & (gds_binary == 255)
                sem_only = (sem_binary == 255) & (gds_binary == 0)
                gds_only = (sem_binary == 0) & (gds_binary == 255)
                
                overlay_img[both_structure] = [0, 0, 0]      # Black - match
                overlay_img[sem_only] = [0, 0, 255]          # Red - SEM only
                overlay_img[gds_only] = [255, 0, 0]          # Blue - GDS only
                
                # Store overlay for saving
                self.current_overlay = overlay_img
                
                # Calculate metrics
                total_pixels = sem_binary.size
                both_count = both_structure.sum()
                sem_only_count = sem_only.sum()
                gds_only_count = gds_only.sum()
                
                # Store metrics for chart
                self.current_metrics = {
                    'aligned': both_count,
                    'sem_only': sem_only_count,
                    'gds_only': gds_only_count,
                    'total': total_pixels
                }
                
                results = {
                    "Aligned (Black)": f"{both_count:,} ({both_count/total_pixels*100:.2f}%)",
                    "SEM Only (Red)": f"{sem_only_count:,} ({sem_only_count/total_pixels*100:.2f}%)",
                    "GDS Only (Blue)": f"{gds_only_count:,} ({gds_only_count/total_pixels*100:.2f}%)",
                    "Total Pixels": f"{total_pixels:,}",
                    "Match Score": f"{both_count/total_pixels*100:.2f}%"
                }
                
                self.display_results(results)
                self._update_chart()
                
                # Update main window display with color overlay
                main_window = self.window()
                if hasattr(main_window, 'image_viewer'):
                    main_window.image_viewer.set_sem_image(overlay_img)
                
            else:
                self.results_display.setPlainText("Please load both SEM and GDS images first.")
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
    
    def _update_chart(self):
        """Update the alignment chart."""
        if not hasattr(self, 'current_metrics'):
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        metrics = self.current_metrics
        labels = ['Aligned\n(Black)', 'SEM Only\n(Red)', 'GDS Only\n(Blue)']
        sizes = [metrics['aligned'], metrics['sem_only'], metrics['gds_only']]
        colors = ['black', 'red', 'blue']
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Pixel Alignment Analysis')
        
        # Style the text
        for text in texts:
            text.set_color('white')
            text.set_fontsize(8)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(8)
            autotext.set_weight('bold')
        
        self.figure.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')
        
        self.canvas.draw()
    
    def _save_overlay_image(self):
        """Save the color overlay image."""
        if not hasattr(self, 'current_overlay'):
            self.results_display.setPlainText("No overlay image to save. Calculate scores first.")
            return
        
        try:
            # Get SEM image name for folder structure
            main_window = self.window()
            sem_name = "sem_image"
            if hasattr(main_window, 'current_sem_path') and main_window.current_sem_path:
                sem_name = Path(main_window.current_sem_path).stem
            
            # Create output directory
            output_dir = Path("Results/Scoring") / sem_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save overlay image
            overlay_path = output_dir / "sem_gds.png"
            cv2.imwrite(str(overlay_path), self.current_overlay)
            
            self.results_display.append(f"\nOverlay saved: {overlay_path}")
            
        except Exception as e:
            self.results_display.append(f"\nError saving overlay: {e}")
    
    def _save_chart(self):
        """Save the alignment chart."""
        if not hasattr(self, 'current_metrics'):
            self.results_display.setPlainText("No chart to save. Calculate scores first.")
            return
        
        try:
            # Get SEM image name for folder structure
            main_window = self.window()
            sem_name = "sem_image"
            if hasattr(main_window, 'current_sem_path') and main_window.current_sem_path:
                sem_name = Path(main_window.current_sem_path).stem
            
            # Create output directory
            output_dir = Path("Results/Scoring") / sem_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save chart
            chart_path = output_dir / "chart.png"
            self.figure.savefig(str(chart_path), facecolor='#2b2b2b', dpi=150, bbox_inches='tight')
            
            self.results_display.append(f"\nChart saved: {chart_path}")
            
        except Exception as e:
            self.results_display.append(f"\nError saving chart: {e}")
    
    def _save_all_results(self):
        """Save all results (overlay, chart, and scores)."""
        if not hasattr(self, 'current_overlay') or not hasattr(self, 'current_metrics'):
            self.results_display.setPlainText("No results to save. Calculate scores first.")
            return
        
        try:
            # Get SEM image name for folder structure
            main_window = self.window()
            sem_name = "sem_image"
            if hasattr(main_window, 'current_sem_path') and main_window.current_sem_path:
                sem_name = Path(main_window.current_sem_path).stem
            
            # Create output directory
            output_dir = Path("Results/Scoring") / sem_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save overlay image
            overlay_path = output_dir / "sem_gds.png"
            cv2.imwrite(str(overlay_path), self.current_overlay)
            
            # Save chart
            chart_path = output_dir / "chart.png"
            self.figure.savefig(str(chart_path), facecolor='#2b2b2b', dpi=150, bbox_inches='tight')
            
            # Save scores as text
            score_path = output_dir / "score.txt"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            with open(score_path, 'w') as f:
                f.write(f"Scoring Results - {timestamp}\n")
                f.write(f"SEM Image: {sem_name}\n")
                f.write(f"Method: {self.current_method}\n\n")
                
                metrics = self.current_metrics
                total = metrics['total']
                f.write(f"Pixel Analysis:\n")
                f.write(f"- Aligned (Black): {metrics['aligned']:,} ({metrics['aligned']/total*100:.2f}%)\n")
                f.write(f"- SEM Only (Red): {metrics['sem_only']:,} ({metrics['sem_only']/total*100:.2f}%)\n")
                f.write(f"- GDS Only (Blue): {metrics['gds_only']:,} ({metrics['gds_only']/total*100:.2f}%)\n")
                f.write(f"- Total Pixels: {total:,}\n\n")
                f.write(f"Match Score: {metrics['aligned']/total*100:.2f}%\n")
            
            self.results_display.append(f"\nAll results saved to: {output_dir}")
            
        except Exception as e:
            self.results_display.append(f"\nError saving results: {e}")