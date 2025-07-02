"""
Filtering Left Panel with filter controls and automatic filter options.

This panel provides controls for image filtering operations with:
- Filter selection dropdown
- Parameter adjustment controls
- Preview and apply options
- Automatic filter pipeline
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
                               QComboBox, QSlider, QSpinBox, QDoubleSpinBox, QLabel, 
                               QPushButton, QGroupBox, QGridLayout, QCheckBox, QListWidget,
                               QProgressBar, QTextEdit)
from PySide6.QtCore import Qt, Signal
from src.ui.base_panels import BaseViewPanel
from src.ui.view_manager import ViewMode


class FilterSelectionTab(QWidget):
    """Tab for manual filter selection and parameter adjustment."""
    
    # Signals
    filter_changed = Signal(str, dict)  # filter_name, parameters
    filter_applied = Signal(str, dict)
    filter_preview = Signal(str, dict)
    filter_reset = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_filter = None
        self.filter_params = {}
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Filter selection group
        filter_group = QGroupBox("Filter Selection")
        filter_layout = QVBoxLayout(filter_group)
        
        filter_select_layout = QHBoxLayout()
        filter_select_layout.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "None",
            "Gaussian Blur",
            "Edge Detection",
            "Sharpen", 
            "Noise Reduction",
            "Contrast Enhancement",
            "Histogram Equalization",
            "Bilateral Filter",
            "Median Filter"
        ])
        filter_select_layout.addWidget(self.filter_combo)
        filter_layout.addLayout(filter_select_layout)
        
        layout.addWidget(filter_group)
        
        # Parameter controls group
        self.param_group = QGroupBox("Parameters")
        self.param_layout = QGridLayout(self.param_group)
        layout.addWidget(self.param_group)
        
        # Initially hide parameters until filter is selected
        self.param_group.hide()
        
        # Preview and apply buttons
        button_layout = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.apply_btn = QPushButton("Apply")
        self.reset_btn = QPushButton("Reset")
        
        button_layout.addWidget(self.preview_btn)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.reset_btn)
        layout.addLayout(button_layout)
        
        # Filter history
        history_group = QGroupBox("Filter History")
        history_layout = QVBoxLayout(history_group)
        self.history_list = QListWidget()
        history_layout.addWidget(self.history_list)
        layout.addWidget(history_group)
        
    def connect_signals(self):
        """Connect UI signals."""
        self.filter_combo.currentTextChanged.connect(self.on_filter_selected)
        self.preview_btn.clicked.connect(self.preview_filter)
        self.apply_btn.clicked.connect(self.apply_filter)
        self.reset_btn.clicked.connect(self.filter_reset)
        
    def on_filter_selected(self, filter_name):
        """Handle filter selection change."""
        self.current_filter = filter_name
        self.setup_parameters(filter_name)
        
    def setup_parameters(self, filter_name):
        """Setup parameter controls for the selected filter."""
        # Clear existing parameters
        for i in reversed(range(self.param_layout.count())):
            item = self.param_layout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
                
        self.filter_params.clear()
        
        if filter_name == "None":
            self.param_group.hide()
            return
            
        self.param_group.show()
        
        # Setup parameters based on filter type
        if filter_name == "Gaussian Blur":
            self.add_parameter("kernel_size", "Kernel Size", 1, 31, 5, step=2)
            self.add_parameter("sigma", "Sigma", 0.1, 10.0, 1.0, is_float=True)
            
        elif filter_name == "Edge Detection":
            self.add_parameter("low_threshold", "Low Threshold", 1, 255, 50)
            self.add_parameter("high_threshold", "High Threshold", 1, 255, 150)
            
        elif filter_name == "Sharpen":
            self.add_parameter("strength", "Strength", 0.1, 3.0, 1.0, is_float=True)
            
        elif filter_name == "Noise Reduction":
            self.add_parameter("h", "Filter Strength", 1, 30, 10)
            self.add_parameter("template_window", "Template Window", 3, 15, 7, step=2)
            self.add_parameter("search_window", "Search Window", 5, 31, 21, step=2)
            
        elif filter_name == "Contrast Enhancement":
            self.add_parameter("alpha", "Alpha", 0.5, 3.0, 1.5, is_float=True)
            self.add_parameter("beta", "Beta", -100, 100, 0)
            
        elif filter_name == "Bilateral Filter":
            self.add_parameter("d", "Diameter", 1, 15, 9)
            self.add_parameter("sigma_color", "Sigma Color", 1, 200, 75)
            self.add_parameter("sigma_space", "Sigma Space", 1, 200, 75)
            
        elif filter_name == "Median Filter":
            self.add_parameter("kernel_size", "Kernel Size", 1, 15, 5, step=2)
            
    def add_parameter(self, param_name, label, min_val, max_val, default_val, step=1, is_float=False):
        """Add a parameter control to the layout."""
        row = self.param_layout.rowCount()
        
        # Label
        self.param_layout.addWidget(QLabel(f"{label}:"), row, 0)
        
        # Slider
        slider = QSlider(Qt.Horizontal)
        if is_float:
            # For float values, use integer slider and convert
            slider.setRange(int(min_val * 10), int(max_val * 10))
            slider.setValue(int(default_val * 10))
        else:
            slider.setRange(min_val, max_val)
            slider.setValue(default_val)
        
        self.param_layout.addWidget(slider, row, 1)
        
        # Spinbox
        if is_float:
            spinbox = QDoubleSpinBox()
            spinbox.setRange(min_val, max_val)
            spinbox.setValue(default_val)
            spinbox.setSingleStep(0.1)
            # Connect slider to spinbox for float values
            slider.valueChanged.connect(lambda v: spinbox.setValue(v / 10.0))
            spinbox.valueChanged.connect(lambda v: slider.setValue(int(v * 10)))
        else:
            spinbox = QSpinBox()
            spinbox.setRange(min_val, max_val)
            spinbox.setValue(default_val)
            spinbox.setSingleStep(step)
            # Connect slider to spinbox for int values
            slider.valueChanged.connect(spinbox.setValue)
            spinbox.valueChanged.connect(slider.setValue)
            
        self.param_layout.addWidget(spinbox, row, 2)
        
        # Store parameter value
        self.filter_params[param_name] = default_val
        
        # Connect to parameter change
        if is_float:
            spinbox.valueChanged.connect(lambda v, name=param_name: self.update_parameter(name, v))
        else:
            spinbox.valueChanged.connect(lambda v, name=param_name: self.update_parameter(name, v))
            
    def update_parameter(self, param_name, value):
        """Update parameter value and emit change signal."""
        self.filter_params[param_name] = value
        if self.current_filter and self.current_filter != "None":
            self.filter_changed.emit(self.current_filter, self.filter_params.copy())
            
    def preview_filter(self):
        """Preview the current filter."""
        if self.current_filter and self.current_filter != "None":
            self.filter_preview.emit(self.current_filter, self.filter_params.copy())
            
    def apply_filter(self):
        """Apply the current filter."""
        if self.current_filter and self.current_filter != "None":
            self.filter_applied.emit(self.current_filter, self.filter_params.copy())
            # Add to history
            history_item = f"{self.current_filter}: {self.filter_params}"
            self.history_list.addItem(history_item)


class AutomaticFilterTab(QWidget):
    """Tab for automatic filter pipeline."""
    
    # Signals
    auto_pipeline_started = Signal(list)  # filter_sequence
    pipeline_progress = Signal(int, str)  # progress, current_filter
    pipeline_finished = Signal(dict)  # results
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Pipeline selection group
        pipeline_group = QGroupBox("Filter Pipeline")
        pipeline_layout = QVBoxLayout(pipeline_group)
        
        # Preset pipelines
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "SEM Enhancement",
            "Edge Detection Pipeline", 
            "Noise Reduction Pipeline",
            "Contrast Enhancement",
            "Custom"
        ])
        preset_layout.addWidget(self.preset_combo)
        pipeline_layout.addLayout(preset_layout)
        
        # Pipeline steps
        self.pipeline_list = QListWidget()
        self.pipeline_list.addItems([
            "1. Noise Reduction",
            "2. Contrast Enhancement", 
            "3. Edge Sharpening",
            "4. Final Cleanup"
        ])
        pipeline_layout.addWidget(self.pipeline_list)
        
        layout.addWidget(pipeline_group)
        
        # Options group
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        
        self.auto_optimize_cb = QCheckBox("Auto-optimize parameters")
        self.auto_optimize_cb.setChecked(True)
        options_layout.addWidget(self.auto_optimize_cb)
        
        self.save_intermediate_cb = QCheckBox("Save intermediate results")
        options_layout.addWidget(self.save_intermediate_cb)
        
        layout.addWidget(options_group)
        
        # Progress group
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.current_step_label = QLabel("Ready")
        progress_layout.addWidget(self.current_step_label)
        
        layout.addWidget(progress_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_pipeline_btn = QPushButton("Start Pipeline")
        self.stop_pipeline_btn = QPushButton("Stop")
        self.stop_pipeline_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_pipeline_btn)
        button_layout.addWidget(self.stop_pipeline_btn)
        layout.addLayout(button_layout)
        
        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(100)
        results_layout.addWidget(self.results_text)
        layout.addWidget(results_group)
        
    def connect_signals(self):
        """Connect UI signals."""
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        self.start_pipeline_btn.clicked.connect(self.start_pipeline)
        self.stop_pipeline_btn.clicked.connect(self.stop_pipeline)
        
    def on_preset_changed(self, preset_name):
        """Handle preset selection change."""
        # Update pipeline steps based on preset
        self.pipeline_list.clear()
        
        if preset_name == "SEM Enhancement":
            steps = [
                "1. Noise Reduction (Non-local means)",
                "2. Contrast Enhancement (CLAHE)",
                "3. Edge Sharpening (Unsharp mask)",
                "4. Final Gaussian smoothing"
            ]
        elif preset_name == "Edge Detection Pipeline":
            steps = [
                "1. Gaussian blur (preprocessing)",
                "2. Gradient calculation",
                "3. Canny edge detection",
                "4. Edge refinement"
            ]
        elif preset_name == "Noise Reduction Pipeline":
            steps = [
                "1. Median filter",
                "2. Bilateral filter", 
                "3. Non-local means denoising",
                "4. Sharpening compensation"
            ]
        elif preset_name == "Contrast Enhancement":
            steps = [
                "1. Histogram equalization",
                "2. CLAHE (Adaptive)",
                "3. Gamma correction",
                "4. Final adjustment"
            ]
        else:  # Custom
            steps = ["Add custom filters..."]
            
        self.pipeline_list.addItems(steps)
        
    def start_pipeline(self):
        """Start the automatic filter pipeline."""
        filter_sequence = []
        for i in range(self.pipeline_list.count()):
            item_text = self.pipeline_list.item(i).text()
            filter_sequence.append(item_text)
            
        self.auto_pipeline_started.emit(filter_sequence)
        self.start_pipeline_btn.setEnabled(False)
        self.stop_pipeline_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
    def stop_pipeline(self):
        """Stop the pipeline execution."""
        self.start_pipeline_btn.setEnabled(True)
        self.stop_pipeline_btn.setEnabled(False)
        self.current_step_label.setText("Stopped")
        
    def update_progress(self, progress, current_step):
        """Update pipeline progress."""
        self.progress_bar.setValue(progress)
        self.current_step_label.setText(f"Processing: {current_step}")
        
    def pipeline_complete(self, results):
        """Handle pipeline completion."""
        self.start_pipeline_btn.setEnabled(True)
        self.stop_pipeline_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.current_step_label.setText("Complete")
        
        # Display results
        results_text = "Pipeline Results:\n"
        for step, result in results.items():
            results_text += f"- {step}: {result}\n"
        self.results_text.setText(results_text)


class FilteringLeftPanel(BaseViewPanel):
    """Left panel for filtering view with manual and automatic filter controls."""
    
    # Signals  
    filter_changed = Signal(str, dict)
    filter_applied = Signal(str, dict)
    filter_preview = Signal(str, dict)
    filter_reset = Signal()
    auto_pipeline_started = Signal(list)
    
    def __init__(self, parent=None):
        super().__init__(ViewMode.FILTERING, parent)
        
    def init_panel(self):
        """Initialize the filtering panel UI."""
        # Clear default layout
        for i in reversed(range(self.main_layout.count())):
            self.main_layout.itemAt(i).widget().setParent(None)
            
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.filter_tab = FilterSelectionTab()
        self.auto_tab = AutomaticFilterTab()
        
        # Add tabs
        self.tab_widget.addTab(self.filter_tab, "Manual Filters")
        self.tab_widget.addTab(self.auto_tab, "Automatic Pipeline")
        
        # Add to main layout
        self.main_layout.addWidget(self.tab_widget)
        
        # Connect tab signals to panel signals
        self.filter_tab.filter_changed.connect(self.filter_changed)
        self.filter_tab.filter_applied.connect(self.filter_applied)
        self.filter_tab.filter_preview.connect(self.filter_preview)
        self.filter_tab.filter_reset.connect(self.filter_reset)
        
        self.auto_tab.auto_pipeline_started.connect(self.auto_pipeline_started)
        
    def get_current_tab_name(self):
        """Get the name of the currently active tab."""
        current_index = self.tab_widget.currentIndex()
        return ["manual", "automatic"][current_index]
        
    def switch_to_tab(self, tab_name):
        """Switch to a specific tab."""
        tab_index = {"manual": 0, "automatic": 1}.get(tab_name, 0)
        self.tab_widget.setCurrentIndex(tab_index)
