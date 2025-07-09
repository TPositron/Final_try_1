"""
File Selector - Enhanced File Selection Component

This module provides an enhanced file selector component for SEM/GDS file selection,
prioritizing .tif files, providing simple file listing, and displaying file information.

Main Class:
- FileSelector: Enhanced widget for selecting SEM and GDS files

Key Methods:
- update_sem_files(): Updates list of available SEM files
- update_gds_files(): Updates list of available GDS files
- populate_structure_dropdown(): Populates structure selection dropdown
- scan_directories(): Scans directories for files
- refresh_files(): Refreshes file listings
- get_selected_sem_file(): Gets currently selected SEM file
- get_selected_gds_file(): Gets default GDS file path
- get_selected_structure_id(): Gets currently selected structure ID

Signals Emitted:
- sem_file_selected(str): SEM file selected
- gds_file_loaded(str): GDS file loaded
- gds_structure_selected(str, int): Structure selected with path and ID
- refresh_requested(): Refresh requested
- file_info_requested(str): File info requested

Dependencies:
- Uses: pathlib.Path, datetime (file operations)
- Uses: PySide6.QtWidgets, PySide6.QtCore, PySide6.QtGui (Qt framework)
- Uses: core/gds_display_generator.get_all_structures_info
- Called by: UI main window and file management components
- Coordinates with: File loading and structure selection workflows

Features:
- Enhanced file selection with priority for .tif files
- Auto-loading of default GDS file (Institute_Project_GDS1.gds)
- Structure selection dropdown with predefined structures
- File information display with metadata
- Directory scanning with extension filtering
- Status updates and progress indication
- Error handling for missing files and directories
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
from src.core.gds_display_generator import get_all_structures_info


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
        # Default GDS file is always Institute_Project_GDS1.gds
        self.selected_gds_file = "Data/GDS/Institute_Project_GDS1.gds"
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the enhanced user interface."""
        main_layout = QVBoxLayout(self)
        
        # Create splitter for better layout but only for SEM files (GDS is auto-loaded)
        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)
        
        # SEM files section
        sem_frame = self._create_sem_section()
        splitter.addWidget(sem_frame)
        
        # Structure selection section (GDS file is auto-loaded)
        gds_frame = self._create_gds_section()
        splitter.addWidget(gds_frame)
        
        # File information panel
        info_frame = self._create_info_section()
        main_layout.addWidget(info_frame)
        
        # Set splitter proportions - adjusted for only SEM and structure
        splitter.setSizes([125, 75])
        
    def _create_sem_section(self) -> QFrame:
        """Create the SEM files selection section."""
        sem_frame = QFrame()
        sem_frame.setFrameStyle(QFrame.Shape.StyledPanel)
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
        """Create the structure selection section (GDS file is auto-loaded)."""
        gds_frame = QFrame()
        gds_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        gds_layout = QVBoxLayout(gds_frame)
        
        # Header - now shows structure selection only
        gds_header = QHBoxLayout()
        gds_label = QLabel("Structure Selection:")
        gds_label.setStyleSheet("font-weight: bold; color: #A23B72;")
        gds_header.addWidget(gds_label)
        gds_header.addStretch()
        
        gds_layout.addLayout(gds_header)
        
        # Info label about auto-loaded GDS
        info_label = QLabel("GDS File: Institute_Project_GDS1.gds (auto-loaded)")
        info_label.setStyleSheet("color: #666666; font-size: 11px; font-style: italic;")
        gds_layout.addWidget(info_label)
        
        # Structure selection combo (always enabled since GDS is auto-loaded)
        structure_label = QLabel("Structure:")
        structure_label.setStyleSheet("font-weight: bold; color: #A23B72; margin-top: 5px;")
        gds_layout.addWidget(structure_label)
        
        self.structure_combo = QComboBox()
        self.structure_combo.addItem("Select structure...", None)
        # The dropdown will be populated dynamically when GDS file is loaded
        self.structure_combo.currentIndexChanged.connect(self._on_structure_selection_changed)
        self.structure_combo.setEnabled(True)  # Always enabled since GDS is auto-loaded
        gds_layout.addWidget(self.structure_combo)
        
        return gds_frame
        
    def _create_info_section(self) -> QFrame:
        """Create the file information section."""
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        info_layout = QVBoxLayout(info_frame)
        
        # Info header
        info_label = QLabel("File Information:")
        info_label.setStyleSheet("font-weight: bold; color: #F18F01;")
        info_layout.addWidget(info_label)
        
        # Info display
        self.info_display = QLabel("No file selected")
        self.info_display.setStyleSheet("color: gray; font-family: monospace; padding: 5px;")
        self.info_display.setWordWrap(True)
        self.info_display.setAlignment(Qt.AlignmentFlag.AlignTop)
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
            item.setData(Qt.ItemDataRole.UserRole, str(file_path))
            self.sem_list.addItem(item)
        
        self._update_status()
    
    def update_gds_files(self, files: List[Path]) -> None:
        """Update the list of available GDS files - simplified for auto-load."""
        self._gds_files = files
        
        # Store Institute_Project_GDS1.gds if found
        for file_path in files:
            if file_path.name == "Institute_Project_GDS1.gds":
                self.selected_gds_file = str(file_path)
                break
        
        # GDS combo and list no longer needed - we auto-load Institute_Project_GDS1.gds
        
        self._update_status()
    
    def _on_sem_list_selection(self, item: QListWidgetItem) -> None:
        """Handle SEM list widget selection."""
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if file_path:
            self.selected_sem_file = file_path
            self._select_sem_file(file_path)
    
    def _on_sem_selection_changed(self, text: str) -> None:
        """Handle SEM combo box selection change."""
        current_data = self.sem_combo.currentData()
        if current_data:
            self._select_sem_file(current_data)
    
    def _on_structure_selection_changed(self, index: int) -> None:
        """Handle structure selection change."""
        print(f"\n=== STRUCTURE SELECTION ===")
        print(f"Structure combo index: {index}")
        
        if index > 0:  # Valid structure chosen
            structure_id = self.structure_combo.currentData()
            structure_text = self.structure_combo.currentText()
            
            print(f"Selected Structure ID: {structure_id}")
            print(f"Selected Structure: {structure_text}")
            print(f"GDS File: Institute_Project_GDS1.gds (auto-loaded)")
            print(f"This should extract ONLY the structure region from coordinates")
            print("===============================\n")
            
            if structure_id:
                # Always use the default GDS file path - Institute_Project_GDS1.gds
                default_gds_path = "Data/GDS/Institute_Project_GDS1.gds"
                self.gds_structure_selected.emit(default_gds_path, structure_id)
    
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
    
    def populate_structure_dropdown(self):
        """
        Populate the structure dropdown with structures from the GDS file.
        This is called when the GDS file is loaded.
        """
        try:
            # Clear existing items
            self.structure_combo.clear()
            self.structure_combo.addItem("Select structure...", None)
            
            # Get all structures from the GDS file
            structures_info = get_all_structures_info()
            
            # Add each structure to the dropdown
            for structure_id, info in structures_info.items():
                display_name = f"Structure{structure_id}: {info['name']}"
                self.structure_combo.addItem(display_name, structure_id)
                
            print(f"Structure dropdown populated with {len(structures_info)} structures")
            self.structure_combo.setEnabled(True)
        except Exception as e:
            print(f"Error populating structure dropdown: {str(e)}")
            self.structure_combo.setEnabled(False)

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
        """Get the default GDS file path (always Institute_Project_GDS1.gds)."""
        return "Data/GDS/Institute_Project_GDS1.gds"
    
    def scan_directories(self, sem_dir: Optional[Path] = None, gds_dir: Optional[Path] = None) -> None:
        """
        Scan directories for SEM files only (GDS file is auto-loaded).
        
        Args:
            sem_dir: Directory to scan for SEM files (defaults to Data/SEM/)
            gds_dir: Not used as GDS is auto-loaded, kept for backward compatibility
        """
        # Default directory
        if sem_dir is None:
            sem_dir = Path("Data/SEM")
        
        # Scan SEM files
        sem_files = self._scan_files(sem_dir, self.SEM_EXTENSIONS)
        self.update_sem_files(sem_files)
        
        # No need to scan for GDS files anymore as we auto-load Institute_Project_GDS1.gds
        
        self._update_info_display()
        
        # Populate structure dropdown at the end
        self.populate_structure_dropdown()
    
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
        self.sem_list.clear()
        
        self._update_info_display()
        self._update_status()
    
    def _update_info_display(self) -> None:
        """Update the file information display."""
        # GDS file is always Institute_Project_GDS1.gds (auto-loaded)
        if self.selected_sem_file:
            info_text = f"SEM: {Path(self.selected_sem_file).name}\nGDS: Institute_Project_GDS1.gds (auto-loaded)"
        else:
            info_text = f"SEM: Not selected\nGDS: Institute_Project_GDS1.gds (auto-loaded)"
        
        self.info_display.setText(info_text)
