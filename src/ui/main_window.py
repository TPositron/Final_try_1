"""
Main Window
Assembles menus, panels, and layouts for the Image Analysis application.
"""

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QMenuBar, QStatusBar, QStackedWidget, QFileDialog,
                               QMessageBox, QApplication)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from .panels.mode_switcher import ModeSwitcher
from .panels.alignment_panel import AlignmentPanel
from .panels.filter_panel import FilterPanel
from .panels.score_panel import ScorePanel
from .components.file_selector import FileSelector
import sys
from typing import Optional


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.current_mode = "Manual"
        self.setup_ui()
        self.setup_menus()
        self.setup_statusbar()
        self.connect_signals()
        
    def setup_ui(self):
        """Initialize the main UI components."""
        self.setWindowTitle("Image Analysis Tool")
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Top section: File selector and mode switcher
        top_layout = QHBoxLayout()
        
        # File selector
        self.file_selector = FileSelector()
        top_layout.addWidget(self.file_selector)
        
        # Mode switcher
        self.mode_switcher = ModeSwitcher()
        top_layout.addWidget(self.mode_switcher)
        
        layout.addLayout(top_layout)
        
        # Main content area with stacked widget for different modes
        self.stacked_widget = QStackedWidget()
        
        # Create panels for each mode
        self.alignment_panel = AlignmentPanel()
        self.filter_panel = FilterPanel()
        self.score_panel = ScorePanel()
        
        # Add panels to stacked widget
        self.stacked_widget.addWidget(self.alignment_panel)  # Index 0: Manual/Auto mode
        self.stacked_widget.addWidget(self.filter_panel)     # Index 1: Filter mode  
        self.stacked_widget.addWidget(self.score_panel)      # Index 2: Score mode
        
        layout.addWidget(self.stacked_widget)
        
    def setup_menus(self):
        """Setup application menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        # Open SEM action
        open_sem_action = QAction("&Open SEM Image...", self)
        open_sem_action.setShortcut(QKeySequence.Open)
        open_sem_action.triggered.connect(self.open_sem_file)
        file_menu.addAction(open_sem_action)
        
        # Open GDS action
        open_gds_action = QAction("Open &GDS File...", self)
        open_gds_action.setShortcut(QKeySequence("Ctrl+G"))
        open_gds_action.triggered.connect(self.open_gds_file)
        file_menu.addAction(open_gds_action)
        
        file_menu.addSeparator()
        
        # Save results action
        save_action = QAction("&Save Results...", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        # Fit to window action
        fit_action = QAction("&Fit to Window", self)
        fit_action.setShortcut(QKeySequence("Ctrl+0"))
        fit_action.triggered.connect(self.fit_to_window)
        view_menu.addAction(fit_action)
        
        # Actual size action
        actual_size_action = QAction("&Actual Size", self)
        actual_size_action.setShortcut(QKeySequence("Ctrl+1"))
        actual_size_action.triggered.connect(self.actual_size)
        view_menu.addAction(actual_size_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        # Auto alignment action
        auto_align_action = QAction("&Auto Alignment", self)
        auto_align_action.setShortcut(QKeySequence("Ctrl+A"))
        auto_align_action.triggered.connect(self.run_auto_alignment)
        tools_menu.addAction(auto_align_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        # About action
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_statusbar(self):
        """Setup status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    def connect_signals(self):
        """Connect internal signals."""
        # Mode switcher signals
        self.mode_switcher.mode_changed.connect(self.change_mode)
        
        # File selector signals
        self.file_selector.sem_file_selected.connect(self.load_sem_file)
        self.file_selector.gds_file_selected.connect(self.load_gds_file)
        
    def change_mode(self, mode: str):
        """Change the current application mode."""
        self.current_mode = mode
        
        # Switch to appropriate panel
        if mode in ["Manual", "Auto"]:
            self.stacked_widget.setCurrentIndex(0)  # Alignment panel
        elif mode == "Filter":
            self.stacked_widget.setCurrentIndex(1)  # Filter panel
        elif mode == "Score":
            self.stacked_widget.setCurrentIndex(2)  # Score panel
            
        self.status_bar.showMessage(f"Mode: {mode}")
        
    def load_sem_file(self, filepath: str):
        """Load SEM file."""
        try:
            # This would interface with file loading services
            self.status_bar.showMessage(f"Loading SEM file: {filepath}")
            # TODO: Implement actual file loading
            self.status_bar.showMessage(f"SEM file loaded: {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load SEM file: {e}")
            
    def load_gds_file(self, filepath: str):
        """Load GDS file."""
        try:
            # This would interface with file loading services
            self.status_bar.showMessage(f"Loading GDS file: {filepath}")
            # TODO: Implement actual file loading
            self.status_bar.showMessage(f"GDS file loaded: {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load GDS file: {e}")
            
    def open_sem_file(self):
        """Open SEM file dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open SEM Image", "", 
            "Image Files (*.tif *.tiff *.png *.jpg);;All Files (*)"
        )
        if filepath:
            self.load_sem_file(filepath)
            
    def open_gds_file(self):
        """Open GDS file dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open GDS File", "", 
            "GDS Files (*.gds);;All Files (*)"
        )
        if filepath:
            self.load_gds_file(filepath)
            
    def save_results(self):
        """Save current results."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", 
            "JSON Files (*.json);;All Files (*)"
        )
        if filepath:
            # TODO: Implement result saving
            self.status_bar.showMessage(f"Results saved: {filepath}")
            
    def fit_to_window(self):
        """Fit content to window."""
        if self.current_mode in ["Manual", "Auto"]:
            self.alignment_panel.fit_canvas_to_view()
            
    def actual_size(self):
        """Show content at actual size."""
        if self.current_mode in ["Manual", "Auto"]:
            self.alignment_panel.reset_canvas_zoom()
            
    def run_auto_alignment(self):
        """Run automatic alignment."""
        if self.current_mode in ["Manual", "Auto"]:
            # TODO: Implement auto alignment
            self.status_bar.showMessage("Running auto alignment...")
            
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Image Analysis Tool",
            "Image Analysis Tool\n\n"
            "A tool for aligning SEM images with GDS layouts\n"
            "and analyzing their correspondence.\n\n"
            "Version 1.0"
        )
        
    def closeEvent(self, event):
        """Handle application close event."""
        reply = QMessageBox.question(
            self, "Confirm Exit",
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Image Analysis Tool")
    app.setOrganizationName("Image Analysis")
    
    window = MainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
