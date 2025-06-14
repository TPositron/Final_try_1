from PySide6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, QProgressBar, QFrame, QGridLayout, QPushButton, QGroupBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
import numpy as np
from typing import Dict, Optional, Union


class ScorePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.score_labels = {}
        self.progress_bars = {}
        self.metric_settings = {}
        self.debug_widgets = []
        self.grid = QGridLayout()  # Always initialize
        self.debug_layout = QGridLayout()  # Always initialize
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("Alignment Quality Metrics")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        self.grid = QGridLayout()
        layout.addLayout(self.grid)

        self._add_metric("Edge Overlap", "edge_overlap", 0, unit="%", threshold_good=75.0, threshold_fair=50.0)
        self._add_metric("IoU Score", "iou", 1, unit="", scale=100, threshold_good=0.7, threshold_fair=0.5)
        self._add_metric("Edge Distance", "edge_distance", 2, unit="px", threshold_good=2.0, threshold_fair=5.0, invert=True)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        self.grid.addWidget(sep, 3, 0, 1, 3)

        self._add_metric("SSIM", "ssim", 4, unit="", scale=100, threshold_good=0.8, threshold_fair=0.6, debug=True)
        self._add_metric("MSE", "mse", 5, unit="", threshold_good=0.01, threshold_fair=0.05, invert=True, debug=True)

        self.debug_group = QGroupBox("Debug Metrics")
        self.debug_group.setVisible(False)
        self.debug_layout = QGridLayout()
        self.debug_group.setLayout(self.debug_layout)
        layout.addWidget(self.debug_group)

    def _add_metric(self, name, key, row, unit="", scale=1.0, threshold_good=0.8, threshold_fair=0.5, invert=False, debug=False):
        label = QLabel(f"{name}:")
        value_label = QLabel("--")
        value_label.setStyleSheet("font-weight: bold;")
        progress = QProgressBar()
        progress.setMinimum(0)
        progress.setMaximum(100)
        progress.setValue(0)
        progress.setTextVisible(False)

        self.metric_settings[key] = dict(unit=unit, scale=scale, threshold_good=threshold_good, threshold_fair=threshold_fair, invert=invert)
        self.score_labels[key] = value_label
        self.progress_bars[key] = progress

        if debug:
            row = len(self.debug_widgets)
            self.debug_layout.addWidget(label, row, 0)
            self.debug_layout.addWidget(value_label, row, 1)
            self.debug_layout.addWidget(progress, row, 2)
            self.debug_widgets.append((label, value_label, progress))
        else:
            self.grid.addWidget(label, row, 0)
            self.grid.addWidget(value_label, row, 1)
            self.grid.addWidget(progress, row, 2)

    def update_scores(self, scores: Dict[str, Union[float, np.ndarray]]):
        for key, value in scores.items():
            if isinstance(value, np.ndarray):
                value = float(np.mean(value))
            elif value is None:
                value = 0.0
            else:
                value = float(value)
            self.update_single_score(key, value)

    def update_single_score(self, key: str, value: float):
        settings = self.metric_settings.get(key, {})
        unit = settings.get('unit', '')
        scale = settings.get('scale', 1.0)
        threshold_good = settings.get('threshold_good', 0.8)
        threshold_fair = settings.get('threshold_fair', 0.5)
        invert = settings.get('invert', False)

        display_value = value * scale
        if unit == "%":
            formatted_value = f"{display_value:.1f}{unit}"
        elif key == "edge_distance":
            formatted_value = f"{display_value:.1f}{unit}"
        else:
            formatted_value = f"{display_value:.3f}{unit}"

        self.score_labels[key].setText(formatted_value)
        color = self._get_color_for_score(value, threshold_good, threshold_fair, invert)
        self.score_labels[key].setStyleSheet(f"font-weight: bold; color: {color};")
        if key in self.progress_bars:
            if invert:
                progress_value = max(0, min(100, (1.0 - min(value, 1.0)) * 100))
            else:
                progress_value = max(0, min(100, value * 100))
            self.progress_bars[key].setValue(progress_value)

    def _get_color_for_score(self, value: float, threshold_good: float, threshold_fair: float, invert: bool = False) -> str:
        if invert:
            if value <= threshold_good:
                return "#2d8f2d"
            elif value <= threshold_fair:
                return "#ff8c00"
            else:
                return "#d32f2f"
        else:
            if value >= threshold_good:
                return "#2d8f2d"
            elif value >= threshold_fair:
                return "#ff8c00"
            else:
                return "#d32f2f"

    def clear_scores(self):
        for label in self.score_labels.values():
            label.setText("--")
            label.setStyleSheet("font-weight: bold; color: black;")
        for progress in self.progress_bars.values():
            progress.setValue(0)

    def toggle_debug_mode(self, show_debug: bool = True):
        self.debug_group.setVisible(show_debug)

    def set_score_thresholds(self, metric: str, good: float, fair: float, invert: bool = False):
        if metric in self.metric_settings:
            self.metric_settings[metric]['threshold_good'] = good
            self.metric_settings[metric]['threshold_fair'] = fair
            self.metric_settings[metric]['invert'] = invert

    def export_scores(self) -> Dict[str, str]:
        return {key: label.text() for key, label in self.score_labels.items()}


def create_score_panel(parent: QWidget) -> ScorePanel:
    return ScorePanel(parent)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Score Panel Test")
    layout = QVBoxLayout(window)

    panel = ScorePanel(window)
    layout.addWidget(panel)

    test_scores = {
        'edge_overlap': 82.5,
        'iou': 0.67,
        'edge_distance': 3.2,
        'ssim': 0.85,
        'mse': 0.02
    }

    def update_test():
        panel.update_scores(test_scores)

    btn_update = QPushButton("Update Scores")
    btn_update.clicked.connect(update_test)
    layout.addWidget(btn_update)

    btn_clear = QPushButton("Clear Scores")
    btn_clear.clicked.connect(panel.clear_scores)
    layout.addWidget(btn_clear)

    btn_toggle_debug = QPushButton("Toggle Debug")
    btn_toggle_debug.clicked.connect(lambda: panel.toggle_debug_mode(not panel.debug_group.isVisible()))
    layout.addWidget(btn_toggle_debug)

    window.setLayout(layout)
    window.resize(400, 300)
    window.show()
    sys.exit(app.exec())