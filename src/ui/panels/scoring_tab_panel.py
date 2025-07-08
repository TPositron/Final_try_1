"""
Scoring Tab Panel for the unified main window.
Provides scoring method selection and results display.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QComboBox, QPushButton, QTextEdit, QGroupBox)
from PySide6.QtCore import Signal, Qt


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
        
        # Method selection
        method_group = QGroupBox("Scoring Method")
        method_layout = QVBoxLayout(method_group)
        
        method_selection_layout = QHBoxLayout()
        method_selection_layout.addWidget(QLabel("Method:"))
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(["SSIM", "MSE", "PSNR", "Cross-Correlation"])
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
        self.calculate_scores_requested.emit(self.current_method)
    
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