"""
Enhanced File selector component for SEM/GDS file selection.
Prioritizes .tif files, provides simple file listing, and displays basic file information.
"""

import os
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
from datetime import datetime
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QComboBox, QPushButton, QScrollArea, QFrame, 
                              QListWidget, QListWidgetItem, QSplitter)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont


class FileSelector(QWidget):
    """
    Enhanced widget for selecting SEM and GDS files.
    Prioritizes .tif files, provides file information, and supports simple file listing.
    """
    
    # File extension priorities
    SEM_EXTENSIONS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']
    GDS_EXTENSIONS = ['.gds', '.gds2', '.gdsii']
    
    # Signals
    sem_file_selected = Signal(str)  # Emitted when SEM file is selected
    gds_file_loaded = Signal(str)    # Emitted when GDS file is first selected
    gds_structure_selected = Signal(str, int)  # Emitted when structure is selected (path, structure_id)
    refresh_requested = Signal()     # Emitted when refresh is requested
    file_info_requested = Signal(str)  # Emitted when file info is requested
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._sem_files = []
        self._gds_files = []
        self._sem_file_info = {}  # Cache for file information
        self._gds_file_info = {}
        self.selected_sem_file = None
        self.selected_gds_file = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the enhanced user interface."""
        main_layout = QVBoxLayout(self)
        
        # Create splitter for better layout
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # SEM files section
        sem_frame = self._create_sem_section()
        splitter.addWidget(sem_frame)
        
        # GDS files section  
        gds_frame = self._create_gds_section()
        splitter.addWidget(gds_frame)
        
        # File information panel
        info_frame = self._create_info_section()
        main_layout.addWidget(info_frame)
        
        # Set splitter proportions
        splitter.setSizes([200, 150, 100])
        
    def _create_sem_section(self) -> QFrame:
        """Create the SEM files selection section."""
        sem_frame = QFrame()
        sem_frame.setFrameStyle(QFrame.StyledPanel)
        sem_layout = QVBoxLayout(sem_frame)
        
        # Header with priority indicator
        sem_header = QHBoxLayout()
        sem_label = QLabel("SEM Images (.tif prioritized):")
        sem_label.setStyleSheet("font-weight: bold; color: #2E86AB;")
        sem_refresh = QPushButton("Refresh")
        sem_refresh.setMaximumWidth(80)
        sem_refresh.clicked.connect(self.refresh_files)
        sem_header.addWidget(sem_label)
        sem_header.addStretch()
        sem_header.addWidget(sem_refresh)
        
        sem_layout.addLayout(sem_header)
        
        # Quick selection combo
        self.sem_combo = QComboBox()
        self.sem_combo.currentTextChanged.connect(self._on_sem_selection_changed)
        sem_layout.addWidget(self.sem_combo)
        
        # Enhanced file list
        self.sem_list = QListWidget()
        self.sem_list.setMaximumHeight(150)
        self.sem_list.itemClicked.connect(self._on_sem_list_selection)
        sem_layout.addWidget(self.sem_list)
        
        return sem_frame
        
    def _create_gds_section(self) -> QFrame:
        """Create the GDS files selection section."""
        gds_frame = QFrame()
        gds_frame.setFrameStyle(QFrame.StyledPanel)
        gds_layout = QVBoxLayout(gds_frame)
        
        # Header
        gds_header = QHBoxLayout()
        gds_label = QLabel("GDS Files:")
        gds_label.setStyleSheet("font-weight: bold; color: #A23B72;")
        gds_refresh = QPushButton("Refresh")
        gds_refresh.setMaximumWidth(80)
        gds_refresh.clicked.connect(self.refresh_files)
        gds_header.addWidget(gds_label)
        gds_header.addStretch()
        gds_header.addWidget(gds_refresh)
        
        gds_layout.addLayout(gds_header)
        
        # Quick selection combo
        self.gds_combo = QComboBox()
        self.gds_combo.currentTextChanged.connect(self._on_gds_selection_changed)
        gds_layout.addWidget(self.gds_combo)
        
        # Structure selection combo (shown only when GDS file is selected)
        structure_label = QLabel("Structure:")
        structure_label.setStyleSheet("font-weight: bold; color: #A23B72; margin-top: 5px;")
        gds_layout.addWidget(structure_label)
        
        self.structure_combo = QComboBox()
        self.structure_combo.addItem("Select structure...", None)
        self.structure_combo.addItem("Structure1: Circpol_T2", 1)
        self.structure_combo.addItem("Structure2: IP935Left_11", 2)
        self.structure_combo.addItem("Structure3: IP935Left_14", 3)
        self.structure_combo.addItem("Structure4: QC855GC_CROSS_Bottom", 4)
        self.structure_combo.addItem("Structure5: QC935_46", 5)
        self.structure_combo.currentIndexChanged.connect(self._on_structure_selection_changed)
        self.structure_combo.setEnabled(False)  # Disabled until GDS file is selected
        gds_layout.addWidget(self.structure_combo)
        
        # Enhanced file list
        self.gds_list = QListWidget()
        self.gds_list.setMaximumHeight(120)
        self.gds_list.itemClicked.connect(self._on_gds_list_selection)
        gds_layout.addWidget(self.gds_list)
        
        return gds_frame
        
    def _create_info_section(self) -> QFrame:
        """Create the file information section."""
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.StyledPanel)
        info_layout = QVBoxLayout(info_frame)
        
        # Info header
        info_label = QLabel("File Information:")
        info_label.setStyleSheet("font-weight: bold; color: #F18F01;")
        info_layout.addWidget(info_label)
        
        # Info display
        self.info_display = QLabel("No file selected")
        self.info_display.setStyleSheet("color: gray; font-family: monospace; padding: 5px;")
        self.info_display.setWordWrap(True)
        self.info_display.setAlignment(Qt.AlignTop)
        info_layout.addWidget(self.info_display)
        
        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("font-size: 9pt; color: #666666; padding: 2px;")
        info_layout.addWidget(self.status_label)
        
        return info_frame
    
    def update_sem_files(self, files: List[Path]) -> None:
        """Update the list of available SEM files."""
        self._sem_files = files
        
        # Update combo box
        self.sem_combo.clear()
        self.sem_combo.addItem("Select SEM file...")
        for file_path in files:
            self.sem_combo.addItem(file_path.name, str(file_path))
        
        # Update list widget
        self.sem_list.clear()
        for file_path in files:
            item = QListWidgetItem(file_path.name)
            item.setData(Qt.UserRole, str(file_path))
            self.sem_list.addItem(item)
        
        self._update_status()
    
    def update_gds_files(self, files: List[Path]) -> None:
        """Update the list of available GDS files."""
        self._gds_files = files
        
        # Update combo box
        self.gds_combo.clear()
        self.gds_combo.addItem("Select GDS file...")
        for file_path in files:
            self.gds_combo.addItem(file_path.name, str(file_path))
        
        # Update list widget
        self.gds_list.clear()
        for file_path in files:
            item = QListWidgetItem(file_path.name)
            item.setData(Qt.UserRole, str(file_path))
            self.gds_list.addItem(item)
        
        self._update_status()
    
    def _on_sem_list_selection(self, item: QListWidgetItem) -> None:
        """Handle SEM list widget selection."""
        file_path = item.data(Qt.UserRole)
        if file_path:
            self.selected_sem_file = file_path
            self._select_sem_file(file_path)
    
    def _on_gds_list_selection(self, item: QListWidgetItem) -> None:
        """Handle GDS list widget selection."""
        file_path = item.data(Qt.UserRole)
        if file_path:
            self.selected_gds_file = file_path
            self.structure_combo.setEnabled(True)  # Enable structure selection
            self.structure_combo.setCurrentIndex(0)  # Reset to "Select structure..."
            self._update_info_display()
            # Emit signal for GDS file loaded
            print(f"GDS file selected from list: {file_path}")
            self.gds_file_loaded.emit(file_path)
    
    def _on_sem_selection_changed(self, text: str) -> None:
        """Handle SEM combo box selection change."""
        current_data = self.sem_combo.currentData()
        if current_data:
            self._select_sem_file(current_data)
    
    def _on_gds_selection_changed(self, text: str) -> None:
        """Handle GDS combo box selection change."""
        current_data = self.gds_combo.currentData()
        if current_data:
            self.selected_gds_file = current_data
            self.structure_combo.setEnabled(True)  # Enable structure selection
            self.structure_combo.setCurrentIndex(0)  # Reset to "Select structure..."
            self._update_info_display()
            # Emit signal for GDS file loaded (enables structure selection)
            print(f"GDS file selected: {current_data}")
            self.gds_file_loaded.emit(current_data)
    
    def _on_structure_selection_changed(self, index: int) -> None:
        """Handle structure selection change."""
        print(f"\n=== STRUCTURE SELECTION ===")
        print(f"Structure combo index: {index}")
        
        if self.selected_gds_file and index > 0:  # If GDS file is selected and valid structure chosen
            structure_id = self.structure_combo.currentData()
            structure_text = self.structure_combo.currentText()
            
            print(f"Selected Structure ID: {structure_id}")
            print(f"Selected Structure: {structure_text}")
            print(f"GDS File: {self.selected_gds_file}")
            print(f"This should extract ONLY the structure region from coordinates")
            print("===============================\n")
            
            if structure_id:
                self.gds_structure_selected.emit(self.selected_gds_file, structure_id)
    
    def _select_sem_file(self, file_path: str) -> None:
        """Select a SEM file."""
        self.selected_sem_file = file_path
        self.sem_file_selected.emit(file_path)
        self._update_info_display()
    
    def _select_gds_file(self, file_path: str) -> None:
        """Select a GDS file."""
        self.selected_gds_file = file_path
        self.structure_combo.setEnabled(True)  # Enable structure selection
        self.structure_combo.setCurrentIndex(0)  # Reset to "Select structure..."
        self._update_info_display()
        # Emit signal for GDS file loaded
        print(f"GDS file selected: {file_path}")
        self.gds_file_loaded.emit(file_path)
    
    def get_selected_structure_id(self) -> Optional[int]:
        """Get the currently selected structure ID."""
        return self.structure_combo.currentData()
    
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
    
    def scan_directories(self, sem_dir: Path = None, gds_dir: Path = None) -> None:
        """
        Scan directories for SEM and GDS files (Step 64).
        
        Args:
            sem_dir: Directory to scan for SEM files (defaults to Data/SEM/)
            gds_dir: Directory to scan for GDS files (defaults to Data/GDS/)
        """
        # Default directories
        if sem_dir is None:
            sem_dir = Path("Data/SEM")
        if gds_dir is None:
            gds_dir = Path("Data/GDS")
        
        # Scan SEM files
        sem_files = self._scan_files(sem_dir, self.SEM_EXTENSIONS)
        self.update_sem_files(sem_files)
        
        # Scan GDS files
        gds_files = self._scan_files(gds_dir, self.GDS_EXTENSIONS)
        self.update_gds_files(gds_files)
        
        self._update_info_display()
    
    def _scan_files(self, directory: Path, extensions: List[str]) -> List[Path]:
        """
        Scan a directory for files with specific extensions.
        
        Args:
            directory: Directory to scan
            extensions: List of file extensions to look for
            
        Returns:
            List of found file paths, sorted with prioritized extensions first
        """
        found_files = []
        
        if not directory.exists():
            return found_files
        
        try:
            # Find all files with matching extensions
            for ext in extensions:
                pattern = f"*{ext}"
                found_files.extend(directory.glob(pattern))
            
            # Remove duplicates and sort
            found_files = list(set(found_files))
            
            # Sort with priority extensions first (.tif for SEM, .gds for GDS)
            def sort_key(file_path):
                ext = file_path.suffix.lower()
                if ext in ['.tif', '.tiff']:
                    return (0, file_path.name.lower())  # Highest priority for .tif
                elif ext == '.gds':
                    return (0, file_path.name.lower())  # Highest priority for .gds
                else:
                    return (1, file_path.name.lower())  # Lower priority for others
            
            found_files.sort(key=sort_key)
            
        except Exception as e:
            print(f"Error scanning directory {directory}: {e}")
        
        return found_files
    
    def refresh_files(self) -> None:
        """Refresh file listings by re-scanning directories."""
        self.scan_directories()
        self.refresh_requested.emit()
        
    def clear_files(self) -> None:
        """Clear all file selections and listings."""
        self.selected_sem_file = None
        self.selected_gds_file = None
        self._sem_files = []
        self._gds_files = []
        self._sem_file_info = {}
        self._gds_file_info = {}
        
        # Clear UI components
        self.sem_combo.clear()
        self.gds_combo.clear()
        self.sem_list.clear()
        self.gds_list.clear()
        
        self._update_info_display()
        self._update_status()
    
    def _update_info_display(self) -> None:
        """Update the file information display."""
        if self.selected_sem_file and self.selected_gds_file:
            info_text = f"SEM: {Path(self.selected_sem_file).name}\nGDS: {Path(self.selected_gds_file).name}"
        elif self.selected_sem_file:
            info_text = f"SEM: {Path(self.selected_sem_file).name}\nGDS: Not selected"
        elif self.selected_gds_file:
            info_text = f"SEM: Not selected\nGDS: {Path(self.selected_gds_file).name}"
        else:
            info_text = "No files selected"
        
        self.info_display.setText(info_text)
