"""
Scoring Tab Panel - Clean scoring interface for Step 12 redesign
White background, scoring method selection, results, and analytics
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton, QButtonGroup, QLabel, QTableWidget, QTableWidgetItem, QTextEdit, QFrame, QPushButton, QSplitter)
from PySide6.QtCore import Qt, Signal
from typing import Dict, Any

class ScoringTabPanel(QWidget):
    """Panel for scoring tab with method selection and results display."""
    scoring_method_changed = Signal(str)
    calculate_scores_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_method = "ssim"
        self._setup_ui()

    def _setup_ui(self):
        # Create main horizontal layout with splitter for left/right panels
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create splitter for left and right panels
        from PySide6.QtWidgets import QSplitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel setup
        self.left_panel = QWidget()
        self.left_panel.setFixedWidth(175)
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(18)
        
        # Apply dark theme to left panel
        self.left_panel.setStyleSheet("""
            QWidget {
                background: #2b2b2b;
                color: #ffffff;
                font-size: 13px;
            }
            QGroupBox {
                background: #404040;
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 12px;
                font-weight: bold;
                padding-top: 10px;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #ffffff;
                background-color: #404040;
            }
            QRadioButton {
                font-size: 13px;
                color: #ffffff;
                spacing: 5px;
            }
            QRadioButton::indicator {
                width: 15px;
                height: 15px;
            }
            QRadioButton::indicator::unchecked {
                border: 2px solid #666666;
                border-radius: 8px;
                background-color: #333333;
            }
            QRadioButton::indicator::checked {
                border: 2px solid #0078d4;
                border-radius: 8px;
                background-color: #0078d4;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #999999;
            }
            QLabel {
                color: #ffffff;
            }
        """)
        
        # Right panel setup
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(18)
        
        # Apply light theme to right panel
        self.right_panel.setStyleSheet("""
            QWidget {
                background: #f8f8f8;
                color: #222;
                font-size: 13px;
            }
            QGroupBox {
                background: #fff;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTableWidget {
                background: #fff;
                border: 1px solid #e0e0e0;
            }
            QTextEdit {
                background: #fff;
                border: 1px solid #e0e0e0;
            }
        """)
        
        # Setup left panel content
        self._setup_left_panel_content(left_layout)
        
        # Setup right panel content  
        self._setup_right_panel_content(right_layout)
        
        # Add panels to splitter
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setSizes([175, 825])
        
        main_layout.addWidget(splitter)
        
    def _setup_left_panel_content(self, layout):
        """Setup left panel content with dark theme styling."""
        # Scoring method selection
        method_group = QGroupBox("Scoring Method")
        method_layout = QVBoxLayout(method_group)
        self.method_button_group = QButtonGroup()
        self.method_radios = {}
        methods = [
            ("ssim", "SSIM (Structural Similarity)", "Measures structural similarity between images."),
            ("mse", "MSE (Mean Squared Error)", "Measures pixel-wise differences."),
            ("iou", "IoU (Intersection over Union)", "Measures overlap between binary masks."),
            ("edge", "Edge Overlap", "Measures edge alignment between images."),
            ("xcorr", "Cross Correlation", "Measures pattern similarity."),
            ("mi", "Mutual Information", "Measures statistical dependence.")
        ]
        for idx, (key, label, desc) in enumerate(methods):
            radio = QRadioButton(label)
            if key == "ssim":
                radio.setChecked(True)
            radio.setToolTip(desc)
            self.method_button_group.addButton(radio, idx)
            self.method_radios[key] = radio
            method_layout.addWidget(radio)
        layout.addWidget(method_group)

        # Method description
        self.method_desc = QLabel(methods[0][2])
        self.method_desc.setWordWrap(True)
        self.method_desc.setStyleSheet("color: #cccccc; font-size: 12px; margin-bottom: 8px;")
        layout.addWidget(self.method_desc)

        # Connect radio change
        self.method_button_group.idClicked.connect(self._on_method_changed)

        # Calculate button and status
        control_layout = QHBoxLayout()
        self.calculate_button = QPushButton("Calculate Scores")
        self.calculate_button.clicked.connect(self._on_calculate_clicked)
        self.status_label = QLabel("Ready to calculate scores")
        self.status_label.setStyleSheet("color: #cccccc; font-size: 12px;")
        control_layout.addWidget(self.calculate_button)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()
        layout.addLayout(control_layout)

        layout.addStretch()
        
    def _setup_right_panel_content(self, layout):
        """Setup right panel content with SEM and GDS overview at top."""
        # SEM and GDS overview at top of right panel (moved from left)
        overview_group = QGroupBox("SEM & GDS Overview")
        overview_layout = QHBoxLayout(overview_group)
        
        self.sem_thumb = QLabel("SEM Image")
        self.sem_thumb.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.sem_thumb.setAlignment(Qt.AlignCenter)
        self.sem_thumb.setMinimumSize(200, 150)
        
        self.gds_thumb = QLabel("GDS Overlay")
        self.gds_thumb.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.gds_thumb.setAlignment(Qt.AlignCenter)
        self.gds_thumb.setMinimumSize(200, 150)
        
        overview_layout.addWidget(self.sem_thumb)
        overview_layout.addWidget(self.gds_thumb)
        layout.addWidget(overview_group)

        # Results table
        results_group = QGroupBox("Comparison Results")
        results_layout = QVBoxLayout(results_group)
        self.score_table = QTableWidget(0, 2)
        self.score_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.score_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.score_table)
        layout.addWidget(results_group)

        # Analytics/Chart area (no histogram as per requirements)
        analytics_group = QGroupBox("Analytics & Visualizations")
        analytics_layout = QVBoxLayout(analytics_group)
        self.analytics_text = QTextEdit()
        self.analytics_text.setReadOnly(True)
        self.analytics_text.setMaximumHeight(120)
        analytics_layout.addWidget(self.analytics_text)
        layout.addWidget(analytics_group)

        layout.addStretch()

    def _on_method_changed(self, idx):
        key = list(self.method_radios.keys())[idx]
        desc = self.method_radios[key].toolTip()
        self.method_desc.setText(desc)
        self.current_method = key
        self.scoring_method_changed.emit(key)

    def _on_calculate_clicked(self):
        """Handle calculate button click."""
        self.calculate_button.setEnabled(False)
        self.status_label.setText("Calculating scores...")
        self.calculate_scores_requested.emit(self.current_method)

    def display_results(self, scores: Dict[str, Any]):
        """Display scoring results in the UI."""
        try:
            # Re-enable calculate button
            self.calculate_button.setEnabled(True)
            self.status_label.setText("Scores calculated successfully")
            
            # Update results table
            self.set_scores(scores)
            
            # Update analytics with summary
            analytics_text = self._generate_analytics_text(scores)
            self.set_analytics(analytics_text)
            
        except Exception as e:
            self.calculate_button.setEnabled(True)
            self.status_label.setText(f"Error: {str(e)}")
            print(f"Error displaying results: {e}")

    def _generate_analytics_text(self, scores: Dict[str, Any]) -> str:
        """Generate analytics text from scores."""
        if not scores:
            return "No results to display."
        
        analytics = []
        analytics.append("Scoring Analysis Summary:")
        analytics.append("=" * 30)
        
        for metric, value in scores.items():
            if isinstance(value, (int, float)):
                analytics.append(f"{metric}: {value:.4f}")
            else:
                analytics.append(f"{metric}: {value}")
        
        analytics.append("\nInterpretation:")
        analytics.append("Higher SSIM values indicate better structural similarity.")
        analytics.append("Lower MSE values indicate better pixel-wise accuracy.")
        
        return "\n".join(analytics)

    def set_scores(self, scores: Dict[str, Any]):
        """Set scores in the results table."""
        self.score_table.setRowCount(0)
        for metric, value in scores.items():
            row = self.score_table.rowCount()
            self.score_table.insertRow(row)
            self.score_table.setItem(row, 0, QTableWidgetItem(str(metric)))
            if isinstance(value, (int, float)):
                self.score_table.setItem(row, 1, QTableWidgetItem(f"{value:.4f}"))
            else:
                self.score_table.setItem(row, 1, QTableWidgetItem(str(value)))

    def set_analytics(self, text: str):
        """Set analytics text."""
        self.analytics_text.setPlainText(text)

    def set_thumbnails(self, sem_pixmap, gds_pixmap):
        """Set thumbnail images."""
        self.sem_thumb.setPixmap(sem_pixmap)
        self.gds_thumb.setPixmap(gds_pixmap)

    def get_selected_method(self):
        """Get currently selected scoring method."""
        return self.current_method
