"""
GDS Operations - GDS File Operations and Structure Management

This module handles all GDS file operations, structure loading, and GDS-related
functionality for the SEM/GDS alignment application.

Main Class:
- GDSOperations: Qt-based handler for GDS operations

Key Methods:
- populate_structure_combo(): Populates structure selection combo box
- load_gds_file(): Loads GDS file with file dialog
- on_structure_selected(): Handles structure selection events
- get_current_structure_info(): Returns current structure information
- reset_structure_selection(): Resets structure selection
- is_gds_loaded(): Checks if GDS file is loaded
- is_structure_selected(): Checks if structure is selected

Signals Emitted:
- gds_loaded(str): GDS file loaded successfully
- structure_loaded(str, object): Structure loaded and overlay generated

Dependencies:
- Uses: services/new_gds_service.NewGDSService (modern GDS processing)
- Uses: core/simple_gds_loader.SimpleGDSLoader (fallback GDS loading)
- Uses: PySide6.QtWidgets, PySide6.QtCore (Qt integration)
- Called by: ui/main_window.py (GDS operations)
- Coordinates with: UI components for display updates

Features:
- GDS file loading with validation
- Structure selection and overlay generation
- Canvas size adaptation based on SEM images
- Auto-loading of default GDS files
- Error handling with user dialogs
- State management for current GDS and structure
"""

import os
from pathlib import Path
from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtCore import QObject, Signal

from src.services.new_gds_service import NewGDSService
from src.core.simple_gds_loader import SimpleGDSLoader


class GDSOperations(QObject):
    """Handles GDS file operations and structure management."""
    
    # Signals
    gds_loaded = Signal(str)  # Emitted when GDS file is loaded
    structure_loaded = Signal(str, object)  # Emitted when structure is loaded
    
    def __init__(self, main_window):
        """Initialize GDS operations with reference to main window."""
        super().__init__()
        self.main_window = main_window
        self.new_gds_service = NewGDSService()
        self.simple_gds_loader = None
        
        # GDS state
        self.current_gds_filename = None
        self.current_gds_filepath = None
        self.current_gds_overlay = None
        self.current_structure_name = None
        self._processing_structure_selection = False
        
    def populate_structure_combo(self):
        """Populate the structure selection combo box with predefined structures."""
        combo = self.main_window.structure_combo
        combo.clear()
        combo.addItem("Select Structure...", "")
        
        # Add structures using new GDS service
        structures_info = self.new_gds_service.get_all_structures_info()
        for structure_num, info in structures_info.items():
            display_name = f"Structure {structure_num} - {info['name']}"
            # Store the structure number format for compatibility
            combo.addItem(display_name, f"Structure {structure_num}")
        
        print(f"Populated structure combo with {len(structures_info)} structures")
        
        # Auto-load default GDS file if it exists
        self._auto_load_default_gds()
    
    def _auto_load_default_gds(self):
        """Auto-load the default GDS file if it exists."""
        try:
            default_gds_path = self.main_window.file_service.get_gds_dir() / "Institute_Project_GDS1.gds"
            if default_gds_path.exists():
                print(f"Auto-loading default GDS: {default_gds_path}")
                self._load_gds_file_internal(str(default_gds_path))
        except Exception as e:
            print(f"Could not auto-load default GDS: {e}")
    
    def load_gds_file(self):
        """Enhanced GDS file loading with new services."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window, 
            "Load GDS File", 
            str(self.main_window.file_service.get_gds_dir()),
            "GDS Files (*.gds *.gds2)"
        )
        
        if file_path:
            self._load_gds_file_internal(file_path)
    
    def _load_gds_file_internal(self, file_path):
        """Internal method to load GDS file."""
        try:
            gds_filename = Path(file_path).name
            
            # Use new GDS service for loading
            success = self.new_gds_service.load_gds_file(file_path)
            
            if success:
                # Enable structure selection
                self.main_window.structure_combo.setEnabled(True)
                
                # Store current GDS filename for structure loading
                self.current_gds_filename = gds_filename
                self.current_gds_filepath = str(file_path)
                
                # Populate structure combo with available structures
                self.populate_structure_combo()
                
                self.main_window.status_bar.showMessage(f"Loaded GDS file: {gds_filename}")
                print(f"✓ GDS file loaded successfully: {gds_filename}")
                
                # Emit signal
                self.gds_loaded.emit(gds_filename)
            else:
                raise RuntimeError(f"Failed to load GDS file: {gds_filename}")
                
        except Exception as e:
            QMessageBox.critical(self.main_window, "Error", f"Failed to load GDS file: {str(e)}")
            print(f"Debug - GDS loading error: {e}")
            
            # Disable structure selection on error
            self.main_window.structure_combo.setEnabled(False)
            self.current_gds_filename = None
            self.current_gds_filepath = None
    
    def on_structure_selected(self, structure_name):
        """Handle structure selection from combo box."""
        print(f"on_structure_selected called with: {structure_name}")
        
        # Prevent recursion
        if self._processing_structure_selection:
            print("Already processing structure selection, ignoring")
            return
        
        self._processing_structure_selection = True
        
        try:
            if not structure_name or structure_name == "Select Structure...":
                self.current_structure_name = None
                self.current_gds_overlay = None
                self.main_window.image_viewer.set_gds_overlay(None)
                return
            
            print(f"Loading structure: {structure_name}")
            
            # Use new GDS service to generate structure display
            structure_num = self.new_gds_service.get_structure_by_name(structure_name)
            print(f"Structure name '{structure_name}' mapped to number: {structure_num}")
            
            if structure_num is None:
                print(f"Warning: Structure '{structure_name}' not found in mapping")
                # Try to extract number directly if it's in "Structure X" format
                if structure_name.startswith("Structure "):
                    try:
                        structure_num = int(structure_name.replace("Structure ", ""))
                        print(f"Extracted structure number: {structure_num}")
                    except ValueError:
                        raise ValueError(f"Could not parse structure number from '{structure_name}'")
                else:
                    raise ValueError(f"Structure '{structure_name}' not found")
            
            # Determine canvas size from current SEM image
            canvas_width = 1024
            canvas_height = 666
            if hasattr(self.main_window, 'current_sem_image') and self.main_window.current_sem_image is not None:
                canvas_height, canvas_width = self.main_window.current_sem_image.shape[:2]
                print(f"Using SEM image dimensions: {canvas_width}x{canvas_height}")
            
            # Generate structure display
            try:
                structure_display = self.new_gds_service.generate_structure_display(
                    structure_num, canvas_width, canvas_height
                )
                
                if structure_display is not None:
                    print(f"✓ Generated structure display for structure {structure_num}")
                    
                    # Store the overlay
                    self.current_gds_overlay = structure_display
                    self.current_structure_name = structure_name
                    
                    # Update the image viewer
                    self.main_window.image_viewer.set_gds_overlay(structure_display)
                    
                    # Emit signal
                    self.structure_loaded.emit(structure_name, structure_display)
                    
                    # Update status
                    self.main_window.status_bar.showMessage(f"Loaded structure: {structure_name}")
                    
                    # Update panel availability
                    self.main_window._update_panel_availability()
                else:
                    raise RuntimeError(f"Failed to generate display for structure {structure_num}")
                    
            except Exception as e:
                error_msg = f"Failed to load structure {structure_name}: {str(e)}"
                print(f"Structure loading error: {error_msg}")
                QMessageBox.warning(self.main_window, "Structure Loading Error", error_msg)
                
                # Clear overlay on error
                self.current_gds_overlay = None
                self.current_structure_name = None
                self.main_window.image_viewer.set_gds_overlay(None)
            
        except Exception as e:
            print(f"Unexpected error in structure selection: {e}")
            QMessageBox.critical(self.main_window, "Error", f"Structure selection failed: {str(e)}")
        finally:
            self._processing_structure_selection = False
    
    def get_current_structure_info(self):
        """Get information about the currently selected structure."""
        if not self.current_structure_name:
            return None
            
        return {
            'name': self.current_structure_name,
            'overlay': self.current_gds_overlay,
            'gds_file': self.current_gds_filename,
            'gds_path': self.current_gds_filepath
        }
    
    def reset_structure_selection(self):
        """Reset the structure selection."""
        self.current_structure_name = None
        self.current_gds_overlay = None
        self.main_window.image_viewer.set_gds_overlay(None)
        self.main_window.structure_combo.setCurrentIndex(0)
    
    def is_gds_loaded(self):
        """Check if a GDS file is currently loaded."""
        return self.current_gds_filename is not None
    
    def is_structure_selected(self):
        """Check if a structure is currently selected."""
        return self.current_structure_name is not None
