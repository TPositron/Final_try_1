"""
UI Setup Module
Handles all UI setup including layouts, menus, toolbars, and status bar.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                               QMenuBar, QMenu, QStatusBar, QToolBar, QButtonGroup,
                               QPushButton, QLabel, QComboBox, QGroupBox, QFileDialog,
                               QMessageBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

from src.ui.view_manager import ViewMode
from src.ui.components.image_viewer import ImageViewer


class UISetup:
    """
    Handles all UI setup for the main window.
    """
    
    def __init__(self, main_window):
        """Initialize UI setup with reference to main window."""
        self.main_window = main_window
        
        # UI components that will be created
        self.image_viewer = None
        self.structure_combo = None
        self.status_bar = None
        self.view_toolbar = None
        self.left_panel_layout = None
        self.view_specific_layout = None
    
    def setup_ui(self):
        """Setup the main UI layout and components."""
        central_widget = QWidget()
        self.main_window.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # Setup left panel container for view-specific panels
        left_panel_container = QWidget()
        self.left_panel_layout = QVBoxLayout(left_panel_container)
        self.left_panel_layout.setContentsMargins(0, 0, 0, 0)
        main_splitter.addWidget(left_panel_container)

        # Setup central image viewer
        self.image_viewer = ImageViewer()
        main_splitter.addWidget(self.image_viewer)

        # Setup right panel container for view-specific content
        right_panel_container = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_container)
        
        # Common controls at the top of right panel
        self._setup_common_controls(right_panel_layout)
        
        # Add view-specific content area
        view_specific_widget = QWidget()
        self.view_specific_layout = QVBoxLayout(view_specific_widget)
        right_panel_layout.addWidget(view_specific_widget)

        main_splitter.addWidget(right_panel_container)
        
        # Set initial splitter sizes: left panel, image viewer, right panel
        main_splitter.setSizes([280, 950, 250])
        
        print("UI layout setup complete")
    
    def setup_menu(self):
        """Setup the main menu bar."""
        menubar = self.main_window.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_sem_action = QAction("Load SEM Image", self.main_window)
        load_sem_action.triggered.connect(self.main_window.load_sem_image)
        file_menu.addAction(load_sem_action)
        
        load_gds_action = QAction("Load GDS File", self.main_window)
        load_gds_action.triggered.connect(self.main_window.load_gds_file)
        file_menu.addAction(load_gds_action)
        
        file_menu.addSeparator()
        
        save_result_action = QAction("Save Results", self.main_window)
        save_result_action.triggered.connect(self.main_window.save_results)
        file_menu.addAction(save_result_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self.main_window)
        exit_action.triggered.connect(self.main_window.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        reset_view_action = QAction("Reset View", self.main_window)
        reset_view_action.triggered.connect(self.main_window.image_viewer.reset_view)
        view_menu.addAction(reset_view_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        reset_alignment_action = QAction("Reset Alignment", self.main_window)
        reset_alignment_action.triggered.connect(lambda: self.main_window.alignment_operations.reset_alignment())
        tools_menu.addAction(reset_alignment_action)
        
        print("Menu setup complete")
    
    def setup_view_toolbar(self):
        """Create and setup the view selection toolbar."""
        # Create toolbar
        self.main_window.view_toolbar = QToolBar("View Selection")
        self.main_window.view_toolbar.setObjectName("ViewToolbar")
        self.main_window.view_toolbar.setMovable(False)
        self.main_window.addToolBar(Qt.TopToolBarArea, self.main_window.view_toolbar)
        
        # Create button group for exclusive selection
        self.main_window.view_button_group = QButtonGroup(self.main_window)
        self.main_window.view_button_group.setExclusive(True)
        
        # Create view selection buttons
        self.main_window.alignment_btn = QPushButton("Alignment")
        self.main_window.alignment_btn.setCheckable(True)
        self.main_window.alignment_btn.setChecked(True)  # Default view
        self.main_window.alignment_btn.setToolTip("Switch to Alignment view for SEM/GDS alignment")
        self.main_window.alignment_btn.clicked.connect(lambda: self.main_window.switch_view(ViewMode.ALIGNMENT))
        
        self.main_window.filtering_btn = QPushButton("Filtering")
        self.main_window.filtering_btn.setCheckable(True)
        self.main_window.filtering_btn.setToolTip("Switch to Filtering view for image processing")
        self.main_window.filtering_btn.clicked.connect(lambda: self.main_window.switch_view(ViewMode.FILTERING))
        
        self.main_window.scoring_btn = QPushButton("Scoring")
        self.main_window.scoring_btn.setCheckable(True)
        self.main_window.scoring_btn.setToolTip("Switch to Scoring view for alignment evaluation")
        self.main_window.scoring_btn.clicked.connect(lambda: self.main_window.switch_view(ViewMode.SCORING))
        
        # Add buttons to group and toolbar
        self.main_window.view_button_group.addButton(self.main_window.alignment_btn)
        self.main_window.view_button_group.addButton(self.main_window.filtering_btn)
        self.main_window.view_button_group.addButton(self.main_window.scoring_btn)
        
        # Style the buttons to look like tabs
        button_style = """
            QPushButton {
                padding: 8px 16px;
                margin: 0px 1px;
                border: 1px solid #555555;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                background-color: #2b2b2b;
                color: #cccccc;
                font-weight: normal;
                min-width: 80px;
            }
            QPushButton:checked {
                background-color: #0078d4;
                color: white;
                border-color: #0078d4;
                font-weight: bold;
            }
            QPushButton:hover:!checked {
                background-color: #3c3c3c;
                color: white;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """
        
        for btn in [self.main_window.alignment_btn, self.main_window.filtering_btn, self.main_window.scoring_btn]:
            btn.setStyleSheet(button_style)
        
        # Add widgets to toolbar
        self.main_window.view_toolbar.addWidget(QLabel("View: "))
        self.main_window.view_toolbar.addWidget(self.main_window.alignment_btn)
        self.main_window.view_toolbar.addWidget(self.main_window.filtering_btn)
        self.main_window.view_toolbar.addWidget(self.main_window.scoring_btn)
        self.main_window.view_toolbar.addSeparator()
        
        # Add current view indicator
        self.main_window.current_view_label = QLabel("Current: Alignment")
        self.main_window.current_view_label.setStyleSheet("QLabel { font-style: italic; color: #666666; margin-left: 10px; }")
        self.main_window.view_toolbar.addWidget(self.main_window.current_view_label)
        
        # Store button mapping for easy access
        self.main_window.view_buttons = {
            ViewMode.ALIGNMENT: self.main_window.alignment_btn,
            ViewMode.FILTERING: self.main_window.filtering_btn,
            ViewMode.SCORING: self.main_window.scoring_btn
        }
        
        print("View toolbar setup complete")
    
    def setup_status_bar(self):
        """Setup the status bar."""
        self.status_bar = QStatusBar()
        self.main_window.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        print("Status bar setup complete")
    
    def _setup_common_controls(self, parent_layout):
        """Setup common controls in the right panel."""
        # File info group
        file_info_group = QGroupBox("File Information")
        file_info_layout = QVBoxLayout(file_info_group)
        
        # SEM file info
        self.sem_file_label = QLabel("SEM File: None")
        file_info_layout.addWidget(self.sem_file_label)
        
        # GDS file info  
        self.gds_file_label = QLabel("GDS File: None")
        file_info_layout.addWidget(self.gds_file_label)
        
        # Structure selection
        structure_layout = QHBoxLayout()
        structure_layout.addWidget(QLabel("Structure:"))
        self.structure_combo = QComboBox()
        self.structure_combo.setEnabled(False)
        structure_layout.addWidget(self.structure_combo)
        file_info_layout.addLayout(structure_layout)
        
        parent_layout.addWidget(file_info_group)
        
        print("Common controls setup complete")
    
    def update_file_info(self, sem_path=None, gds_path=None):
        """Update file information display."""
        if sem_path:
            self.sem_file_label.setText(f"SEM File: {sem_path}")
        if gds_path:
            self.gds_file_label.setText(f"GDS File: {gds_path}")
            
    def get_structure_combo(self):
        """Get the structure combo box."""
        return self.structure_combo
