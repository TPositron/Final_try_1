"""
Scoring Left Panel Module

This module provides a comprehensive user interface for alignment scoring operations in a GDS/SEM 
image alignment application. It implements a tabbed interface with scoring method selection, 
parameter configuration, batch analysis capabilities, and results visualization.

The module handles multiple scoring algorithms including SSIM (Structural Similarity Index),
MSE (Mean Squared Error), Cross-Correlation, Edge Overlap, and Mutual Information methods.
It supports both single alignment scoring and batch parameter sweep analysis.

Dependencies:
    - PySide6.QtWidgets: UI components (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
      QRadioButton, QButtonGroup, QPushButton, QGroupBox, QListWidget, QTableWidget, 
      QTableWidgetItem, QLabel, QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar, 
      QTextEdit, QComboBox, QSlider)
    - PySide6.QtCore: Core functionality (Qt, Signal)
    - src.ui.base_panels: BaseViewPanel class
    - src.ui.view_manager: ViewMode enumeration

Classes:
    ScoringMethodTab: Tab widget for scoring method selection and single scoring operations
        - setup_ui(): Creates the user interface layout
        - setup_ssim_parameters(): Configures SSIM-specific parameters (data range, window size)
        - setup_edge_parameters(): Configures edge detection parameters (Canny thresholds)
        - clear_parameters(): Removes all parameter widgets from layout
        - connect_signals(): Connects UI signals to handlers
        - on_method_changed(): Handles scoring method selection changes
        - score_current(): Requests scoring with current method and parameters
        - score_all_methods(): Requests batch scoring with selected methods
        - get_current_parameters(): Retrieves current parameter values
        - update_scores(): Updates the scores display table

    BatchScoringTab: Tab widget for batch scoring operations and results analysis
        - setup_ui(): Creates the batch analysis interface
        - connect_signals(): Connects UI signals to handlers
        - start_batch_analysis(): Initiates batch parameter sweep analysis
        - stop_batch_analysis(): Stops running batch analysis
        - update_progress(): Updates progress bar and status
        - add_result(): Adds a result to the results table
        - clear_results(): Clears all batch results

    ScoringLeftPanel: Main panel class inheriting from BaseViewPanel
        - init_panel(): Initializes the complete panel UI with tabs
        - get_current_tab_name(): Returns name of currently active tab
        - switch_to_tab(): Switches to specified tab
        - update_scores(): Updates scoring results display
        - add_batch_result(): Adds batch analysis result to table

Signals:
    - scoring_method_changed: Emitted when scoring method is changed
    - score_requested: Emitted when single scoring is requested
    - batch_score_requested: Emitted when multi-method scoring is requested
    - batch_analysis_started: Emitted when batch analysis begins
    - export_results_requested: Emitted when results export is requested
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
                               QRadioButton, QButtonGroup, QPushButton, QGroupBox, 
                               QListWidget, QTableWidget, QTableWidgetItem, QLabel,
                               QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar,
                               QTextEdit, QComboBox, QSlider)
from PySide6.QtCore import Qt, Signal
from src.ui.base_panels import BaseViewPanel
from src.ui.view_manager import ViewMode


class ScoringMethodTab(QWidget):
    """Tab for scoring method selection and single scoring operations."""
    
    # Signals
    scoring_method_changed = Signal(str)  # method_name
    score_requested = Signal(str, dict)  # method, parameters
    batch_score_requested = Signal(list, dict)  # methods, parameters
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_method = "ssim"
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Scoring method selection group
        method_group = QGroupBox("Scoring Method")
        method_layout = QVBoxLayout(method_group)
        
        self.method_group = QButtonGroup()
        
        # SSIM
        self.ssim_radio = QRadioButton("SSIM (Structural Similarity)")
        self.ssim_radio.setChecked(True)
        self.ssim_radio.setToolTip("Structural Similarity Index - measures structural similarity")
        self.method_group.addButton(self.ssim_radio)
        method_layout.addWidget(self.ssim_radio)
        
        # MSE
        self.mse_radio = QRadioButton("MSE (Mean Squared Error)")
        self.mse_radio.setToolTip("Mean Squared Error - measures pixel-wise differences")
        self.method_group.addButton(self.mse_radio)
        method_layout.addWidget(self.mse_radio)
        
        # Cross-correlation
        self.xcorr_radio = QRadioButton("Cross-Correlation")
        self.xcorr_radio.setToolTip("Normalized cross-correlation - measures pattern similarity")
        self.method_group.addButton(self.xcorr_radio)
        method_layout.addWidget(self.xcorr_radio)
        
        # Edge overlap
        self.edge_radio = QRadioButton("Edge Overlap")
        self.edge_radio.setToolTip("Edge detection overlap - measures edge alignment")
        self.method_group.addButton(self.edge_radio)
        method_layout.addWidget(self.edge_radio)
        
        # Mutual information
        self.mi_radio = QRadioButton("Mutual Information")
        self.mi_radio.setToolTip("Mutual Information - measures statistical dependence")
        self.method_group.addButton(self.mi_radio)
        method_layout.addWidget(self.mi_radio)
        
        layout.addWidget(method_group)
        
        # Scoring parameters group
        self.param_group = QGroupBox("Scoring Parameters")
        self.param_layout = QVBoxLayout(self.param_group)
        
        # SSIM parameters (shown by default)
        self.setup_ssim_parameters()
        
        layout.addWidget(self.param_group)
        
        # Multi-method scoring
        multi_group = QGroupBox("Multi-Method Scoring")
        multi_layout = QVBoxLayout(multi_group)
        
        self.enable_multi_cb = QCheckBox("Enable multi-method scoring")
        multi_layout.addWidget(self.enable_multi_cb)
        
        # Method selection for multi-scoring
        self.multi_methods = []
        for method_name, desc in [("ssim", "SSIM"), ("mse", "MSE"), ("xcorr", "Cross-Correlation"), 
                                 ("edge", "Edge Overlap"), ("mi", "Mutual Information")]:
            cb = QCheckBox(desc)
            if method_name == "ssim":
                cb.setChecked(True)
            self.multi_methods.append((method_name, cb))
            multi_layout.addWidget(cb)
            
        layout.addWidget(multi_group)
        
        # Scoring buttons
        button_layout = QHBoxLayout()
        self.score_current_btn = QPushButton("Score Current Alignment")
        self.score_all_btn = QPushButton("Score All Methods")
        
        button_layout.addWidget(self.score_current_btn)
        button_layout.addWidget(self.score_all_btn)
        layout.addLayout(button_layout)
        
        # Current scores display
        scores_group = QGroupBox("Current Scores")
        scores_layout = QVBoxLayout(scores_group)
        self.scores_table = QTableWidget(0, 2)
        self.scores_table.setHorizontalHeaderLabels(["Metric", "Score"])
        scores_layout.addWidget(self.scores_table)
        layout.addWidget(scores_group)
        
    def setup_ssim_parameters(self):
        """Setup SSIM-specific parameters."""
        self.clear_parameters()
        
        # Data range
        data_range_layout = QHBoxLayout()
        data_range_layout.addWidget(QLabel("Data Range:"))
        self.data_range_spin = QDoubleSpinBox()
        self.data_range_spin.setRange(0.1, 255.0)
        self.data_range_spin.setValue(1.0)
        data_range_layout.addWidget(self.data_range_spin)
        self.param_layout.addLayout(data_range_layout)
        
        # Window size
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Window Size:"))
        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(3, 31)
        self.window_size_spin.setValue(7)
        self.window_size_spin.setSingleStep(2)
        window_layout.addWidget(self.window_size_spin)
        self.param_layout.addLayout(window_layout)
        
    def setup_edge_parameters(self):
        """Setup edge overlap specific parameters."""
        self.clear_parameters()
        
        # Canny thresholds
        low_thresh_layout = QHBoxLayout()
        low_thresh_layout.addWidget(QLabel("Low Threshold:"))
        self.low_thresh_spin = QSpinBox()
        self.low_thresh_spin.setRange(1, 255)
        self.low_thresh_spin.setValue(50)
        low_thresh_layout.addWidget(self.low_thresh_spin)
        self.param_layout.addLayout(low_thresh_layout)
        
        high_thresh_layout = QHBoxLayout()
        high_thresh_layout.addWidget(QLabel("High Threshold:"))
        self.high_thresh_spin = QSpinBox()
        self.high_thresh_spin.setRange(1, 255)
        self.high_thresh_spin.setValue(150)
        high_thresh_layout.addWidget(self.high_thresh_spin)
        self.param_layout.addLayout(high_thresh_layout)
        
    def clear_parameters(self):
        """Clear parameter layout."""
        for i in reversed(range(self.param_layout.count())):
            item = self.param_layout.itemAt(i)
            if item.layout():
                # Clear sublayout
                for j in reversed(range(item.layout().count())):
                    widget = item.layout().itemAt(j).widget()
                    if widget:
                        widget.setParent(None)
                item.layout().setParent(None)
            elif item.widget():
                item.widget().setParent(None)
                
    def connect_signals(self):
        """Connect UI signals."""
        self.ssim_radio.toggled.connect(lambda checked: checked and self.on_method_changed("ssim"))
        self.mse_radio.toggled.connect(lambda checked: checked and self.on_method_changed("mse"))
        self.xcorr_radio.toggled.connect(lambda checked: checked and self.on_method_changed("xcorr"))
        self.edge_radio.toggled.connect(lambda checked: checked and self.on_method_changed("edge"))
        self.mi_radio.toggled.connect(lambda checked: checked and self.on_method_changed("mi"))
        
        self.score_current_btn.clicked.connect(self.score_current)
        self.score_all_btn.clicked.connect(self.score_all_methods)
        
    def on_method_changed(self, method):
        """Handle scoring method change."""
        self.current_method = method
        self.scoring_method_changed.emit(method)
        
        # Update parameters based on method
        if method == "ssim":
            self.setup_ssim_parameters()
        elif method == "edge":
            self.setup_edge_parameters()
        else:
            self.clear_parameters()
            
    def score_current(self):
        """Request scoring with current method."""
        params = self.get_current_parameters()
        self.score_requested.emit(self.current_method, params)
        
    def score_all_methods(self):
        """Request scoring with all selected methods."""
        if self.enable_multi_cb.isChecked():
            selected_methods = []
            for method_name, cb in self.multi_methods:
                if cb.isChecked():
                    selected_methods.append(method_name)
            params = self.get_current_parameters()
            self.batch_score_requested.emit(selected_methods, params)
        else:
            self.score_current()
            
    def get_current_parameters(self):
        """Get current scoring parameters."""
        params = {}
        if self.current_method == "ssim" and hasattr(self, 'data_range_spin'):
            params['data_range'] = self.data_range_spin.value()
            params['window_size'] = self.window_size_spin.value()
        elif self.current_method == "edge" and hasattr(self, 'low_thresh_spin'):
            params['low_threshold'] = self.low_thresh_spin.value()
            params['high_threshold'] = self.high_thresh_spin.value()
        return params
        
    def update_scores(self, scores):
        """Update the scores display table."""
        self.scores_table.setRowCount(len(scores))
        for i, (metric, score) in enumerate(scores.items()):
            self.scores_table.setItem(i, 0, QTableWidgetItem(metric))
            self.scores_table.setItem(i, 1, QTableWidgetItem(f"{score:.4f}"))


class BatchScoringTab(QWidget):
    """Tab for batch scoring operations and results analysis."""
    
    # Signals
    batch_analysis_started = Signal(dict)  # parameters
    export_results_requested = Signal(str)  # format
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.batch_results = {}
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Batch options group
        options_group = QGroupBox("Batch Analysis Options")
        options_layout = QVBoxLayout(options_group)
        
        # Parameter sweep
        sweep_layout = QHBoxLayout()
        self.enable_sweep_cb = QCheckBox("Parameter Sweep")
        sweep_layout.addWidget(self.enable_sweep_cb)
        options_layout.addLayout(sweep_layout)
        
        # Sweep parameters
        self.sweep_params_widget = QWidget()
        sweep_params_layout = QVBoxLayout(self.sweep_params_widget)
        
        # X offset sweep
        x_sweep_layout = QHBoxLayout()
        x_sweep_layout.addWidget(QLabel("X Offset Range:"))
        self.x_min_spin = QSpinBox()
        self.x_min_spin.setRange(-100, 100)
        self.x_min_spin.setValue(-10)
        self.x_max_spin = QSpinBox()
        self.x_max_spin.setRange(-100, 100)
        self.x_max_spin.setValue(10)
        self.x_step_spin = QSpinBox()
        self.x_step_spin.setRange(1, 10)
        self.x_step_spin.setValue(2)
        x_sweep_layout.addWidget(self.x_min_spin)
        x_sweep_layout.addWidget(QLabel("to"))
        x_sweep_layout.addWidget(self.x_max_spin)
        x_sweep_layout.addWidget(QLabel("step"))
        x_sweep_layout.addWidget(self.x_step_spin)
        sweep_params_layout.addLayout(x_sweep_layout)
        
        # Y offset sweep
        y_sweep_layout = QHBoxLayout()
        y_sweep_layout.addWidget(QLabel("Y Offset Range:"))
        self.y_min_spin = QSpinBox()
        self.y_min_spin.setRange(-100, 100)
        self.y_min_spin.setValue(-10)
        self.y_max_spin = QSpinBox()
        self.y_max_spin.setRange(-100, 100)
        self.y_max_spin.setValue(10)
        self.y_step_spin = QSpinBox()
        self.y_step_spin.setRange(1, 10)
        self.y_step_spin.setValue(2)
        y_sweep_layout.addWidget(self.y_min_spin)
        y_sweep_layout.addWidget(QLabel("to"))
        y_sweep_layout.addWidget(self.y_max_spin)
        y_sweep_layout.addWidget(QLabel("step"))
        y_sweep_layout.addWidget(self.y_step_spin)
        sweep_params_layout.addLayout(y_sweep_layout)
        
        options_layout.addWidget(self.sweep_params_widget)
        self.sweep_params_widget.hide()
        
        layout.addWidget(options_group)
        
        # Progress group
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_batch_btn = QPushButton("Start Batch Analysis")
        self.stop_batch_btn = QPushButton("Stop")
        self.stop_batch_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_batch_btn)
        button_layout.addWidget(self.stop_batch_btn)
        layout.addLayout(button_layout)
        
        # Results group
        results_group = QGroupBox("Batch Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(["X Offset", "Y Offset", "Method", "Score"])
        results_layout.addWidget(self.results_table)
        
        # Export buttons
        export_layout = QHBoxLayout()
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_json_btn = QPushButton("Export JSON")
        self.clear_results_btn = QPushButton("Clear Results")
        
        export_layout.addWidget(self.export_csv_btn)
        export_layout.addWidget(self.export_json_btn)
        export_layout.addWidget(self.clear_results_btn)
        results_layout.addLayout(export_layout)
        
        layout.addWidget(results_group)
        
    def connect_signals(self):
        """Connect UI signals."""
        self.enable_sweep_cb.toggled.connect(self.sweep_params_widget.setVisible)
        self.start_batch_btn.clicked.connect(self.start_batch_analysis)
        self.stop_batch_btn.clicked.connect(self.stop_batch_analysis)
        self.export_csv_btn.clicked.connect(lambda: self.export_results_requested.emit("csv"))
        self.export_json_btn.clicked.connect(lambda: self.export_results_requested.emit("json"))
        self.clear_results_btn.clicked.connect(self.clear_results)
        
    def start_batch_analysis(self):
        """Start batch analysis."""
        params = {
            'parameter_sweep': self.enable_sweep_cb.isChecked()
        }
        
        if params['parameter_sweep']:
            params['x_range'] = (self.x_min_spin.value(), self.x_max_spin.value(), self.x_step_spin.value())
            params['y_range'] = (self.y_min_spin.value(), self.y_max_spin.value(), self.y_step_spin.value())
            
        self.batch_analysis_started.emit(params)
        self.start_batch_btn.setEnabled(False)
        self.stop_batch_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting batch analysis...")
        
    def stop_batch_analysis(self):
        """Stop batch analysis."""
        self.start_batch_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
        self.status_label.setText("Stopped")
        
    def update_progress(self, progress, status):
        """Update batch analysis progress."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        
    def add_result(self, x_offset, y_offset, method, score):
        """Add a result to the results table."""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        self.results_table.setItem(row, 0, QTableWidgetItem(str(x_offset)))
        self.results_table.setItem(row, 1, QTableWidgetItem(str(y_offset)))
        self.results_table.setItem(row, 2, QTableWidgetItem(method))
        self.results_table.setItem(row, 3, QTableWidgetItem(f"{score:.4f}"))
        
    def clear_results(self):
        """Clear all results."""
        self.results_table.setRowCount(0)
        self.batch_results.clear()


class ScoringLeftPanel(BaseViewPanel):
    """Left panel for scoring view with scoring method selection and batch operations."""
    
    # Signals
    scoring_method_changed = Signal(str)
    score_requested = Signal(str, dict)
    batch_score_requested = Signal(list, dict)
    batch_analysis_started = Signal(dict)
    export_results_requested = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(ViewMode.SCORING, parent)
        
    def init_panel(self):
        """Initialize the scoring panel UI."""
        # Clear default layout
        for i in reversed(range(self.main_layout.count())):
            self.main_layout.itemAt(i).widget().setParent(None)
            
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.scoring_tab = ScoringMethodTab()
        self.batch_tab = BatchScoringTab()
        
        # Add tabs
        self.tab_widget.addTab(self.scoring_tab, "Scoring Methods")
        self.tab_widget.addTab(self.batch_tab, "Batch Analysis")
        
        # Add to main layout
        self.main_layout.addWidget(self.tab_widget)
        
        # Connect tab signals to panel signals
        self.scoring_tab.scoring_method_changed.connect(self.scoring_method_changed)
        self.scoring_tab.score_requested.connect(self.score_requested)
        self.scoring_tab.batch_score_requested.connect(self.batch_score_requested)
        
        self.batch_tab.batch_analysis_started.connect(self.batch_analysis_started)
        self.batch_tab.export_results_requested.connect(self.export_results_requested)
        
    def get_current_tab_name(self):
        """Get the name of the currently active tab."""
        current_index = self.tab_widget.currentIndex()
        return ["scoring", "batch"][current_index]
        
    def switch_to_tab(self, tab_name):
        """Switch to a specific tab."""
        tab_index = {"scoring": 0, "batch": 1}.get(tab_name, 0)
        self.tab_widget.setCurrentIndex(tab_index)
        
    def update_scores(self, scores):
        """Update the scores display."""
        self.scoring_tab.update_scores(scores)
        
    def add_batch_result(self, x_offset, y_offset, method, score):
        """Add a batch result."""
        self.batch_tab.add_result(x_offset, y_offset, method, score)
