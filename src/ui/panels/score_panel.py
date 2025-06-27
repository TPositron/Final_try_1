"""
Score Panel
Score selection, overlay visualization, charts and export functionality.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                               QComboBox, QPushButton, QLabel, QTextEdit,
                               QSplitter, QTableWidget, QTableWidgetItem)
from PySide6.QtCore import Qt, Signal
from typing import Dict, Any, List
import numpy as np


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
        
        # Score selection group
        selection_group = QGroupBox("Score Selection")
        selection_layout = QHBoxLayout(selection_group)
        
        selection_layout.addWidget(QLabel("Metric:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(self.available_metrics)
        selection_layout.addWidget(self.metric_combo)
        
        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.clicked.connect(self._on_calculate_clicked)
        selection_layout.addWidget(self.calculate_button)
        
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
        metric = self.metric_combo.currentText()
        self.score_calculated.emit(metric, {})
        
    def _on_generate_overlay_clicked(self):
        """Handle generate overlay button click."""
        # Generate color-coded difference overlay
        # This would interface with scoring services
        dummy_overlay = np.zeros((666, 1024, 3), dtype=np.uint8)
        self.overlay_generated.emit(dummy_overlay)
        
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
        self.metric_combo.clear()
        self.metric_combo.addItems(metrics)
