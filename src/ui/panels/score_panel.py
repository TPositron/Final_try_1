"""
Score Panel
Score selection, overlay visualization, charts and export functionality.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                               QComboBox, QPushButton, QLabel, QTextEdit,
                               QSplitter, QTableWidget, QTableWidgetItem, QRadioButton,
                               QButtonGroup, QProgressBar)
from PySide6.QtCore import Qt, Signal
from typing import Dict, Any, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ScorePanel(QWidget):
    """Panel for scoring and comparison operations."""
    
    # Signals
    score_calculated = Signal(str, dict)  # score_type, results
    overlay_generated = Signal(np.ndarray)  # overlay image
    export_requested = Signal(str, dict)  # export_type, data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_scores = {}
        self.available_metrics = ["Pixel Match", "SSIM", "IoU"]
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        
        # Metric selection group with radio buttons
        selection_group = QGroupBox("Metric Selection")
        selection_layout = QVBoxLayout(selection_group)
        
        # Radio buttons for different metrics
        self.metric_button_group = QButtonGroup()
        
        self.pixel_match_radio = QRadioButton("Pixel Match")
        self.pixel_match_radio.setChecked(True)  # Default selection
        self.metric_button_group.addButton(self.pixel_match_radio, 0)
        selection_layout.addWidget(self.pixel_match_radio)
        
        self.ssim_radio = QRadioButton("SSIM (Structural Similarity)")
        self.metric_button_group.addButton(self.ssim_radio, 1)
        selection_layout.addWidget(self.ssim_radio)
        
        self.iou_radio = QRadioButton("IoU (Intersection over Union)")
        self.metric_button_group.addButton(self.iou_radio, 2)
        selection_layout.addWidget(self.iou_radio)
        
        self.correlation_radio = QRadioButton("Cross Correlation")
        self.metric_button_group.addButton(self.correlation_radio, 3)
        selection_layout.addWidget(self.correlation_radio)
        
        self.mse_radio = QRadioButton("MSE (Mean Squared Error)")
        self.metric_button_group.addButton(self.mse_radio, 4)
        selection_layout.addWidget(self.mse_radio)
        
        # Connect signal for metric changes
        self.metric_button_group.idClicked.connect(self._on_metric_changed)
        
        # Calculate button
        self.calculate_button = QPushButton("Calculate Score")
        self.calculate_button.clicked.connect(self._on_calculate_clicked)
        selection_layout.addWidget(self.calculate_button)
        
        # Progress bar for calculations
        self.status_label = QLabel("Ready.")
        selection_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        selection_layout.addWidget(self.progress_bar)
        
        layout.addWidget(selection_group)
        
        # Create main splitter for results
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Score results
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        
        # Score table
        self.score_table = QTableWidget()
        self.score_table.setColumnCount(2)
        self.score_table.setHorizontalHeaderLabels(["Metric", "Value"])
        results_layout.addWidget(QLabel("Scores:"))
        results_layout.addWidget(self.score_table)
        
        # Basic chart display area
        chart_group = QGroupBox("Score Chart")
        chart_layout = QVBoxLayout(chart_group)
        
        self.chart_text = QTextEdit()
        self.chart_text.setReadOnly(True)
        self.chart_text.setMaximumHeight(100)
        self.chart_text.setText("Chart will be displayed here when scores are calculated.")
        chart_layout.addWidget(self.chart_text)
        
        results_layout.addWidget(chart_group)
        
        # Overlay controls
        overlay_group = QGroupBox("Overlay Visualization")
        overlay_layout = QVBoxLayout(overlay_group)
        
        overlay_button_layout = QHBoxLayout()
        self.generate_overlay_button = QPushButton("Generate Overlay")
        self.generate_overlay_button.clicked.connect(self._on_generate_overlay_clicked)
        overlay_button_layout.addWidget(self.generate_overlay_button)
        
        self.save_overlay_button = QPushButton("Save Overlay")
        self.save_overlay_button.clicked.connect(self._on_save_overlay_clicked)
        overlay_button_layout.addWidget(self.save_overlay_button)
        
        overlay_layout.addLayout(overlay_button_layout)
        results_layout.addWidget(overlay_group)
        
        # Right side: Details and export
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        
        # Score details
        details_layout.addWidget(QLabel("Score Details:"))
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        details_layout.addWidget(self.details_text)
        
        # Export controls
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        
        export_button_layout = QHBoxLayout()
        self.export_json_button = QPushButton("Export JSON")
        self.export_json_button.clicked.connect(self._on_export_json_clicked)
        export_button_layout.addWidget(self.export_json_button)
        
        self.export_csv_button = QPushButton("Export CSV")
        self.export_csv_button.clicked.connect(self._on_export_csv_clicked)
        export_button_layout.addWidget(self.export_csv_button)
        
        export_layout.addLayout(export_button_layout)
        details_layout.addWidget(export_group)
        
        # Add to splitter
        splitter.addWidget(results_widget)
        splitter.addWidget(details_widget)
        
        # Set splitter proportions (60% left, 40% right)
        splitter.setSizes([600, 400])
        
        layout.addWidget(splitter)
        
    def _on_calculate_clicked(self):
        """Handle calculate button click."""
        selected_metric = self._get_selected_metric()
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Emit signal to request score calculation
        self.score_calculated.emit(selected_metric, {})
        
    def _on_metric_changed(self, metric_id):
        """Handle metric selection change."""
        metric_name = self._get_selected_metric()
        # Auto-calculate when metric changes if images are available
        if hasattr(self, 'sem_image') and hasattr(self, 'gds_image'):
            self._on_calculate_clicked()
            
    def _get_selected_metric(self) -> str:
        """Get the currently selected metric name."""
        button_id = self.metric_button_group.checkedId()
        metric_names = ["Pixel Match", "SSIM", "IoU", "Cross Correlation", "MSE"]
        return metric_names[button_id] if 0 <= button_id < len(metric_names) else "Pixel Match"
        
    def _on_generate_overlay_clicked(self):
        """Handle generate overlay button click."""
        # Generate color-coded difference overlay
        self._generate_color_overlay()
        
    def _generate_color_overlay(self):
        """Generate color-coded difference overlay showing match/mismatch regions."""
        if not hasattr(self, 'sem_image') or not hasattr(self, 'gds_image'):
            # Create dummy visualization for demonstration
            height, width = 666, 1024
            overlay = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Create sample match/mismatch pattern
            # Green for matches, Red for mismatches, Blue for GDS-only, Yellow for SEM-only
            overlay[100:200, 100:300] = [0, 255, 0]  # Green - match region
            overlay[300:400, 200:400] = [255, 0, 0]  # Red - mismatch region  
            overlay[150:250, 500:700] = [0, 0, 255]  # Blue - GDS only
            overlay[450:550, 300:500] = [255, 255, 0]  # Yellow - SEM only
            
        else:
            # Real overlay generation when images are available
            overlay = self._create_difference_overlay(self.sem_image, self.gds_image)
            
        self.overlay_generated.emit(overlay)
        
    def _create_difference_overlay(self, sem_img: np.ndarray, gds_img: np.ndarray) -> np.ndarray:
        """Create color-coded difference overlay between SEM and GDS images."""
        height, width = sem_img.shape[:2]
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Convert to binary if needed
        sem_binary = (sem_img > 0.5) if sem_img.dtype == np.float32 else (sem_img > 128)
        gds_binary = (gds_img > 0.5) if gds_img.dtype == np.float32 else (gds_img > 128)
        
        # Create color-coded overlay
        # Green: Both SEM and GDS (match)
        match_mask = sem_binary & gds_binary
        overlay[match_mask] = [0, 255, 0]
        
        # Red: Neither SEM nor GDS but expected match (mismatch)
        # Blue: GDS only (structure not in SEM)
        gds_only_mask = gds_binary & ~sem_binary
        overlay[gds_only_mask] = [0, 0, 255]
        
        # Yellow: SEM only (noise or unexpected structure)
        sem_only_mask = sem_binary & ~gds_binary
        overlay[sem_only_mask] = [255, 255, 0]
        
        return overlay
        
    def _on_save_overlay_clicked(self):
        """Handle save overlay button click."""
        self.export_requested.emit("overlay", {"format": "png"})
        
    def _on_export_json_clicked(self):
        """Handle export JSON button click."""
        self.export_requested.emit("json", self.current_scores)
        
    def _on_export_csv_clicked(self):
        """Handle export CSV button click."""
        self.export_requested.emit("csv", self.current_scores)
        
    def update_scores(self, scores: Dict[str, float]):
        """Update the score display."""
        self.current_scores.update(scores)
        
        # Update table
        self.score_table.setRowCount(len(self.current_scores))
        row = 0
        for metric, value in self.current_scores.items():
            self.score_table.setItem(row, 0, QTableWidgetItem(metric))
            self.score_table.setItem(row, 1, QTableWidgetItem(f"{value:.4f}"))
            row += 1
            
        # Update chart display
        self.update_chart_display()
        
        # Update details
        details_text = "Score Calculation Results:\n\n"
        for metric, value in self.current_scores.items():
            details_text += f"{metric}: {value:.6f}\n"
            
            # Add interpretation
            if metric == "SSIM":
                if value > 0.8:
                    details_text += "  → Excellent structural similarity\n"
                elif value > 0.6:
                    details_text += "  → Good structural similarity\n"
                elif value > 0.4:
                    details_text += "  → Moderate structural similarity\n"
                else:
                    details_text += "  → Poor structural similarity\n"
            elif metric == "Pixel Match":
                percentage = value * 100
                details_text += f"  → {percentage:.2f}% pixel overlap\n"
            elif metric == "IoU":
                if value > 0.7:
                    details_text += "  → Excellent overlap\n"
                elif value > 0.5:
                    details_text += "  → Good overlap\n"
                elif value > 0.3:
                    details_text += "  → Moderate overlap\n"
                else:
                    details_text += "  → Poor overlap\n"
            details_text += "\n"
            
        self.details_text.setText(details_text)
        
    def clear_scores(self):
        """Clear all score displays."""
        self.current_scores.clear()
        self.score_table.setRowCount(0)
        self.details_text.clear()
        
    def set_available_metrics(self, metrics: List[str]):
        """Set the available scoring metrics."""
        self.available_metrics = metrics
        
    def set_images(self, sem_image: np.ndarray, gds_image: np.ndarray):
        """Set the SEM and GDS images for comparison."""
        self.sem_image = sem_image
        self.gds_image = gds_image
        
    def on_score_completed(self, metric_name: str, score_value: float):
        """Handle score calculation completion."""
        self.progress_bar.setVisible(False)
        self.current_scores[metric_name] = score_value
        self.update_scores({metric_name: score_value})
        
    def update_chart_display(self):
        """Update the basic chart display with current scores."""
        if not self.current_scores:
            self.chart_text.setText("No scores calculated yet.")
            return
            
        chart_text = "Score Summary:\n"
        for metric, value in self.current_scores.items():
            # Simple text-based "chart"
            bar_length = int(value * 20) if value <= 1.0 else int(value / max(self.current_scores.values()) * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            chart_text += f"{metric:<15}: {bar} {value:.3f}\n"
            
        self.chart_text.setText(chart_text)
        
    def reset(self):
        """Reset panel to default state."""
        self.current_scores = {}
        
        # Reset radio button to default
        self.pixel_match_radio.setChecked(True)
        
        # Clear results table
        self.results_table.setRowCount(0)
        
        # Clear chart display
        self.chart_text.setText("No scores calculated yet.")
        
        # Clear overlay display
        self.overlay_text.setText("No overlay generated yet.")
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
    def show_progress(self, progress_info):
        """Show pipeline progress information in the score panel."""
        stage = progress_info.get('stage', '')
        status = progress_info.get('status', '')
        progress = progress_info.get('progress', 0)

        if hasattr(self, 'status_label'):
            self.status_label.setText(f"Pipeline: {stage.title()} - {status}")
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(progress)
            self.progress_bar.setVisible(True if progress < 100 else False)
        
    def get_current_state(self) -> Dict[str, Any]:
        """Get current panel state for session saving."""
        # Get selected metric
        selected_metric = None
        if self.pixel_match_radio.isChecked():
            selected_metric = "Pixel Match"
        elif self.ssim_radio.isChecked():
            selected_metric = "SSIM"
        elif self.iou_radio.isChecked():
            selected_metric = "IoU"
            
        return {
            'current_scores': self.current_scores.copy(),
            'selected_metric': selected_metric,
            'available_metrics': self.available_metrics.copy()
        }
        
    def set_state(self, state: Dict[str, Any]):
        """Set panel state from session loading."""
        if 'current_scores' in state:
            self.current_scores = state['current_scores'].copy()
            self.update_scores(self.current_scores)
            
        if 'selected_metric' in state and state['selected_metric']:
            metric = state['selected_metric']
            if metric == "Pixel Match":
                self.pixel_match_radio.setChecked(True)
            elif metric == "SSIM":
                self.ssim_radio.setChecked(True)
            elif metric == "IoU":
                self.iou_radio.setChecked(True)
                
        if 'available_metrics' in state:
            self.available_metrics = state['available_metrics'].copy()
    
    def get_current_config(self):
        """Get current scoring configuration for pipeline processing."""
        config = {
            'scoring_methods': ['correlation'],  # Default scoring method
            'scoring_parameters': {},
            'display_settings': {},
            'export_settings': {},
            'current_preset': getattr(self, 'current_preset', None)
        }
        
        # Get selected scoring methods
        try:
            if hasattr(self, 'method_selector'):
                config['scoring_methods'] = [self.method_selector.currentText().lower()]
            elif hasattr(self, 'method_checkboxes'):
                # Multiple selection support
                selected_methods = []
                for method, checkbox in self.method_checkboxes.items():
                    if checkbox.isChecked():
                        selected_methods.append(method.lower())
                if selected_methods:
                    config['scoring_methods'] = selected_methods
        except Exception as e:
            logger.warning(f"Error extracting scoring methods: {e}")
        
        # Get scoring parameters from controls
        try:
            for param_name in ['threshold', 'window_size', 'overlap_threshold']:
                if hasattr(self, f'{param_name}_control'):
                    control = getattr(self, f'{param_name}_control')
                    if hasattr(control, 'value'):
                        config['scoring_parameters'][param_name] = control.value()
                    elif hasattr(control, 'text'):
                        config['scoring_parameters'][param_name] = control.text()
        except Exception as e:
            logger.debug(f"Could not extract scoring parameters: {e}")
        
        # Get display and export settings
        try:
            config['display_settings'] = {
                'show_overlay': getattr(self, 'show_overlay', True),
                'show_score_map': getattr(self, 'show_score_map', False),
                'color_scheme': getattr(self, 'color_scheme', 'default')
            }
            
            config['export_settings'] = {
                'export_format': getattr(self, 'export_format', 'csv'),
                'include_metadata': getattr(self, 'include_metadata', True),
                'export_charts': getattr(self, 'export_charts', True)
            }
        except Exception as e:
            logger.debug(f"Could not extract display/export settings: {e}")
        
        # Get current scores if available
        try:
            if hasattr(self.scoring_service, 'get_latest_scores'):
                config['current_scores'] = self.scoring_service.get_latest_scores()
        except Exception as e:
            logger.debug(f"Could not get current scores: {e}")
        
        # Get preset information
        if hasattr(self, 'preset_selector'):
            config['selected_preset'] = self.preset_selector.currentText()
        
        return config
