"""File selector component for SEM/GDS file dropdowns and scrollbars."""

from pathlib import Path
from typing import List, Optional, Callable
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QComboBox, QPushButton, QScrollArea, QFrame)
from PySide6.QtCore import Signal, Qt


class FileSelector(QWidget):
    """Widget for selecting SEM and GDS files with dropdown and scroll interface."""
    
    # Signals
    sem_file_selected = Signal(str)  # Emitted when SEM file is selected
    gds_file_selected = Signal(str)  # Emitted when GDS file is selected
    refresh_requested = Signal()     # Emitted when refresh is requested
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._sem_files = []
        self._gds_files = []
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # SEM files section
        sem_frame = QFrame()
        sem_frame.setFrameStyle(QFrame.Box)
        sem_layout = QVBoxLayout(sem_frame)
        
        sem_header = QHBoxLayout()
        sem_label = QLabel("SEM Images:")
        sem_label.setStyleSheet("font-weight: bold;")
        sem_refresh = QPushButton("Refresh")
        sem_refresh.clicked.connect(self.refresh_requested.emit)
        sem_header.addWidget(sem_label)
        sem_header.addStretch()
        sem_header.addWidget(sem_refresh)
        
        sem_layout.addLayout(sem_header)
        
        self.sem_combo = QComboBox()
        self.sem_combo.currentTextChanged.connect(self._on_sem_selection_changed)
        sem_layout.addWidget(self.sem_combo)
        
        # SEM file list (scrollable)
        self.sem_scroll = QScrollArea()
        self.sem_scroll.setMaximumHeight(120)
        self.sem_scroll.setWidgetResizable(True)
        sem_list_widget = QWidget()
        self.sem_list_layout = QVBoxLayout(sem_list_widget)
        self.sem_scroll.setWidget(sem_list_widget)
        sem_layout.addWidget(self.sem_scroll)
        
        layout.addWidget(sem_frame)
        
        # GDS files section
        gds_frame = QFrame()
        gds_frame.setFrameStyle(QFrame.Box)
        gds_layout = QVBoxLayout(gds_frame)
        
        gds_header = QHBoxLayout()
        gds_label = QLabel("GDS Files:")
        gds_label.setStyleSheet("font-weight: bold;")
        gds_refresh = QPushButton("Refresh")
        gds_refresh.clicked.connect(self.refresh_requested.emit)
        gds_header.addWidget(gds_label)
        gds_header.addStretch()
        gds_header.addWidget(gds_refresh)
        
        gds_layout.addLayout(gds_header)
        
        self.gds_combo = QComboBox()
        self.gds_combo.currentTextChanged.connect(self._on_gds_selection_changed)
        gds_layout.addWidget(self.gds_combo)
        
        # GDS file list (scrollable)
        self.gds_scroll = QScrollArea()
        self.gds_scroll.setMaximumHeight(120)
        self.gds_scroll.setWidgetResizable(True)
        gds_list_widget = QWidget()
        self.gds_list_layout = QVBoxLayout(gds_list_widget)
        self.gds_scroll.setWidget(gds_list_widget)
        gds_layout.addWidget(self.gds_scroll)
        
        layout.addWidget(gds_frame)
        
        # Status
        self.status_label = QLabel("No files loaded")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.status_label)
    
    def update_sem_files(self, files: List[Path]) -> None:
        """Update the list of available SEM files."""
        self._sem_files = files
        
        # Update combo box
        self.sem_combo.clear()
        self.sem_combo.addItem("Select SEM file...")
        for file_path in files:
            self.sem_combo.addItem(file_path.name, str(file_path))
        
        # Update scrollable list
        self._clear_layout(self.sem_list_layout)
        for file_path in files:
            file_button = QPushButton(file_path.name)
            file_button.clicked.connect(lambda checked, path=str(file_path): self._select_sem_file(path))
            file_button.setStyleSheet("text-align: left; padding: 4px;")
            self.sem_list_layout.addWidget(file_button)
        
        self._update_status()
    
    def update_gds_files(self, files: List[Path]) -> None:
        """Update the list of available GDS files."""
        self._gds_files = files
        
        # Update combo box
        self.gds_combo.clear()
        self.gds_combo.addItem("Select GDS file...")
        for file_path in files:
            self.gds_combo.addItem(file_path.name, str(file_path))
        
        # Update scrollable list
        self._clear_layout(self.gds_list_layout)
        for file_path in files:
            file_button = QPushButton(file_path.name)
            file_button.clicked.connect(lambda checked, path=str(file_path): self._select_gds_file(path))
            file_button.setStyleSheet("text-align: left; padding: 4px;")
            self.gds_list_layout.addWidget(file_button)
        
        self._update_status()
    
    def _clear_layout(self, layout):
        """Clear all widgets from a layout."""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def _on_sem_selection_changed(self, text: str) -> None:
        """Handle SEM combo box selection change."""
        current_data = self.sem_combo.currentData()
        if current_data:
            self._select_sem_file(current_data)
    
    def _on_gds_selection_changed(self, text: str) -> None:
        """Handle GDS combo box selection change."""
        current_data = self.gds_combo.currentData()
        if current_data:
            self._select_gds_file(current_data)
    
    def _select_sem_file(self, file_path: str) -> None:
        """Select a SEM file."""
        self.sem_file_selected.emit(file_path)
        self._update_status()
    
    def _select_gds_file(self, file_path: str) -> None:
        """Select a GDS file."""
        self.gds_file_selected.emit(file_path)
        self._update_status()
    
    def _update_status(self) -> None:
        """Update the status label."""
        sem_count = len(self._sem_files)
        gds_count = len(self._gds_files)
        
        self.status_label.setText(f"Available: {sem_count} SEM files, {gds_count} GDS files")
    
    def get_selected_sem_file(self) -> Optional[str]:
        """Get the currently selected SEM file path."""
        return self.sem_combo.currentData()
    
    def get_selected_gds_file(self) -> Optional[str]:
        """Get the currently selected GDS file path."""
        return self.gds_combo.currentData()
