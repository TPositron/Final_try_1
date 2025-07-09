"""
GDS Operations Manager - Comprehensive GDS File and Structure Management

This module handles all GDS file operations, structure loading, and GDS-related
functionality, providing centralized GDS management capabilities.

Main Class:
- GDSOperationsManager: Manages all GDS operations

Key Methods:
- populate_structure_combo(): Populates structure selection combo
- load_gds_file(): Loads GDS file with dialog selection
- load_gds_file_from_path(): Loads GDS from specified path
- on_structure_selected(): Handles structure selection from combo
- select_structure_by_id(): Selects structure by ID
- get_current_structure_info(): Gets current structure information
- is_gds_loaded(): Checks if GDS file is loaded
- is_structure_selected(): Checks if structure is selected
- reset_structure_selection(): Resets structure selection
- export_structure_overlay(): Exports structure overlay to file

Signals Emitted:
- gds_file_loaded(str): GDS file loaded with path
- structure_selected(str, object): Structure selected with name and overlay
- structure_combo_populated(int): Structure combo populated with count
- gds_operation_error(str, str): GDS operation error with details

Dependencies:
- Uses: numpy, pathlib.Path (data processing and file operations)
- Uses: PySide6.QtWidgets, PySide6.QtCore (Qt framework)
- Uses: services/new_gds_service.NewGDSService
- Uses: ui/view_manager.ViewMode
- Called by: UI main window and file management components
- Coordinates with: Image viewers and alignment workflows

Features:
- Auto-loading of default GDS file (Institute_Project_GDS1.gds)
- Structure selection with predefined structures (1-5)
- Binary image generation from GDS structures
- Overlay creation and display management
- Structure information retrieval and validation
- Error handling and recovery mechanisms
- Integration with FileSelector component
"""

import numpy as np
from pathlib import Path
from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtCore import QObject, Signal

from src.services.new_gds_service import NewGDSService
from src.ui.view_manager import ViewMode


class GDSOperationsManager(QObject):
    """Manages all GDS operations for the application."""
    
    # Signals for GDS operations
    gds_file_loaded = Signal(str)  # file_path
    structure_selected = Signal(str, object)  # structure_name, overlay
    structure_combo_populated = Signal(int)  # number_of_structures
    gds_operation_error = Signal(str, str)  # operation, error_message
    
    def __init__(self, main_window):
        """Initialize with reference to main window."""
        super().__init__()
        self.main_window = main_window
        self.new_gds_service = NewGDSService()
        
        # GDS state
        self.current_gds_filename = None
        self.current_gds_filepath = None
        self.current_structure_name = None
        self.current_gds_overlay = None
        self._processing_structure_selection = False
        
    def populate_structure_combo(self):
        """Populate the structure selection combo box with predefined structures."""
        try:
            # Note: Structure combo is now handled by FileSelector component
            # This method is kept for compatibility but doesn't populate the old combo
            print("Structure combo population is now handled by FileSelector component")
            
            # Get structures info for signal emission
            structures_info = self.new_gds_service.get_all_structures_info()
            print(f"Available structures: {len(structures_info)}")
            
            # Emit signal
            self.structure_combo_populated.emit(len(structures_info))
            
            # Auto-load default GDS file if it exists
            self._auto_load_default_gds()
            
        except Exception as e:
            error_msg = f"Failed to get structure info: {str(e)}"
            print(error_msg)
            self.gds_operation_error.emit("populate_combo", str(e))
    
    def _auto_load_default_gds(self):
        """Auto-load default GDS file if it exists."""
        try:
            if hasattr(self.main_window, 'file_service'):
                default_gds_path = self.main_window.file_service.get_gds_dir() / "Institute_Project_GDS1.gds"
            else:
                # Fallback path if file_service is not available
                default_gds_path = Path("Data/GDS/Institute_Project_GDS1.gds")
                
            if default_gds_path.exists():
                print(f"Auto-loading default GDS: {default_gds_path}")
                self._load_gds_file_internal(str(default_gds_path))
            else:
                print(f"Default GDS file not found: {default_gds_path}")
                # Try alternate path
                alt_path = Path("Data") / "GDS" / "Institute_Project_GDS1.gds"
                if alt_path.exists():
                    print(f"Found default GDS at alternate path: {alt_path}")
                    self._load_gds_file_internal(str(alt_path))
                else:
                    print(f"Could not locate default GDS file at any expected location")
        except Exception as e:
            print(f"Could not auto-load default GDS: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_structure_combo_only(self):
        """Update structure combo without auto-loading GDS files."""
        try:
            if not hasattr(self.main_window, 'structure_combo'):
                print("Warning: Main window has no structure_combo")
                return
                
            # Clear existing items
            self.main_window.structure_combo.clear()
            
            # Add default item
            self.main_window.structure_combo.addItem("Select Structure...")
            
            # Add predefined structures
            structures = ["Structure 1", "Structure 2", "Structure 3", "Structure 4", "Structure 5"]
            self.main_window.structure_combo.addItems(structures)
            
            # Enable the combo
            self.main_window.structure_combo.setEnabled(True)
            
            # Emit signal that combo was populated
            self.structure_combo_populated.emit(structures)
            
        except Exception as e:
            error_msg = f"Failed to update structure combo: {str(e)}"
            print(error_msg)
            self.gds_operation_error.emit("update_combo", str(e))
                
    def load_gds_file(self):
        """Enhanced GDS file loading with new services."""
        gds_dir = "Data/GDS"  # Default fallback
        if hasattr(self.main_window, 'file_service'):
            gds_dir = str(self.main_window.file_service.get_gds_dir())
            
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Load GDS File", gds_dir,
            "GDS Files (*.gds *.gds2)")
        
        if file_path:
            self._load_gds_file_internal(file_path)
    
    def _load_gds_file_internal(self, file_path):
        """Internal method to load GDS file."""
        try:
            gds_filename = Path(file_path).name
            
            # Use new GDS service for loading
            success = self.new_gds_service.load_gds_file(file_path)
            
            if success:
                # Store current GDS filename for structure loading
                self.current_gds_filename = gds_filename
                self.current_gds_filepath = str(file_path)
                
                # Populate structure dropdown in FileSelector component
                if hasattr(self.main_window, 'file_selector') and self.main_window.file_selector:
                    self.main_window.file_selector.populate_structure_dropdown()
                    print("Structure dropdown populated in FileSelector")
                else:
                    print("Warning: FileSelector not found, cannot populate structure dropdown")
                
                # Update status
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage(f"Loaded GDS file: {gds_filename}")
                
                print(f"✓ GDS file loaded successfully: {gds_filename}")
                
                # Emit signal
                self.gds_file_loaded.emit(file_path)
                
                # Auto-select Structure 1 if no structure is currently selected
                if not self.current_structure_name:
                    print("Auto-selecting Structure 1...")
                    self.select_structure_by_id(1)
            else:
                raise RuntimeError(f"Failed to load GDS file: {gds_filename}")
                
        except Exception as e:
            QMessageBox.critical(self.main_window, "Error", f"Failed to load GDS file: {str(e)}")
            print(f"Debug - GDS loading error: {e}")
            
            # Disable structure selection on error
            if hasattr(self.main_window, 'structure_combo'):
                self.main_window.structure_combo.setEnabled(False)
            self.current_gds_filename = None
            self.current_gds_filepath = None
            
            self.gds_operation_error.emit("load_gds", str(e))

    def on_structure_selected(self, structure_name):
        """Handle structure selection from combo box - simplified approach."""
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
                if hasattr(self.main_window, 'image_viewer'):
                    self.main_window.image_viewer.set_gds_overlay(None)
                return
            
            print(f"Loading structure: {structure_name}")
            
            # Use new GDS service to generate structure display
            # First get structure number from name
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
            
            # Validate structure number is in valid range
            if structure_num not in [1, 2, 3, 4, 5]:
                raise ValueError(f"Invalid structure number: {structure_num}")
            
            # Generate binary image using new service
            overlay_image = self.new_gds_service.generate_structure_display(structure_num, (1024, 666))
            
            if overlay_image is not None:
                print(f"Overlay image generated successfully, shape: {overlay_image.shape}, dtype: {overlay_image.dtype}")
                print(f"Overlay min/max values: {overlay_image.min()}/{overlay_image.max()}")
                
                # Ensure proper format for display
                if overlay_image.dtype != np.uint8:
                    if overlay_image.max() <= 1.0:
                        overlay_image = (overlay_image * 255).astype(np.uint8)
                    else:
                        overlay_image = overlay_image.astype(np.uint8)
                
                # Use the binary image directly as grayscale overlay
                # The ImageViewer will handle the transparency
                overlay = overlay_image
                    
                print(f"Final overlay format: {overlay.shape}, dtype: {overlay.dtype}, non-zero pixels: {np.count_nonzero(overlay)}")
            else:
                raise ValueError(f"Failed to generate image for structure: {structure_name}")
            
            # Store current selection
            self.current_structure_name = structure_name
            self.current_gds_overlay = overlay
            
            print(f"Generated overlay for {structure_name}, shape: {overlay.shape}")
            
            # Display the overlay
            if hasattr(self.main_window, 'image_viewer'):
                print(f"Setting GDS overlay in image viewer...")
                self.main_window.image_viewer.set_gds_overlay(overlay)
                
                # Make sure overlay is visible
                self.main_window.image_viewer.set_overlay_visible(True)
                print(f"Overlay visibility set to: {self.main_window.image_viewer.get_overlay_visible()}")
                
                # Force refresh
                self.main_window.image_viewer.refresh()
                print(f"Image viewer refreshed")
            
            # Update status
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage(f"Loaded structure: {structure_name}")
            
            # Emit signal
            self.structure_selected.emit(structure_name, overlay)
            
            # Update alignment display if SEM image is loaded
            if hasattr(self.main_window, 'current_sem_image') and self.main_window.current_sem_image is not None:
                if hasattr(self.main_window, 'update_alignment_display'):
                    self.main_window.update_alignment_display()
            
            # Enable the save button since we now have an overlay
            if hasattr(self.main_window, 'panel_manager'):
                alignment_panel = self.main_window.panel_manager.left_panels.get(ViewMode.ALIGNMENT)
                if alignment_panel and hasattr(alignment_panel, 'enable_save_button'):
                    alignment_panel.enable_save_button(True)
            
            # Update panel availability to ensure alignment panel stays enabled
            if hasattr(self.main_window, '_update_panel_availability'):
                self.main_window._update_panel_availability()
                
        except Exception as e:
            print(f"Error loading GDS structure: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self.main_window, "Error", f"Failed to load GDS structure: {e}")
            self.current_structure_name = None
            self.current_gds_overlay = None
            self.gds_operation_error.emit("select_structure", str(e))
        finally:
            # Always reset the flag
            self._processing_structure_selection = False
    
    def get_current_structure_info(self):
        """Get information about the currently selected structure."""
        return {
            'gds_filename': self.current_gds_filename,
            'gds_filepath': self.current_gds_filepath,
            'structure_name': self.current_structure_name,
            'has_overlay': self.current_gds_overlay is not None
        }
    
    def is_gds_loaded(self):
        """Check if a GDS file is currently loaded."""
        return self.current_gds_filename is not None
    
    def is_structure_selected(self):
        """Check if a structure is currently selected."""
        return self.current_structure_name is not None and self.current_gds_overlay is not None
    
    def reset_structure_selection(self):
        """Reset the structure selection."""
        try:
            self.current_structure_name = None
            self.current_gds_overlay = None
            
            # Clear overlay from image viewer
            if hasattr(self.main_window, 'image_viewer'):
                self.main_window.image_viewer.set_gds_overlay(None)
            
            # Reset combo box
            if hasattr(self.main_window, 'structure_combo'):
                self.main_window.structure_combo.setCurrentIndex(0)
            
            # Update status
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage("Structure selection reset")
            
            print("Structure selection reset")
            
        except Exception as e:
            print(f"Error resetting structure selection: {e}")
    
    def get_available_structures(self):
        """Get list of available structures."""
        try:
            return self.new_gds_service.get_all_structures_info()
        except Exception as e:
            print(f"Error getting available structures: {e}")
            return {}
    
    def reload_gds_file(self):
        """Reload the current GDS file."""
        if self.current_gds_filepath:
            self._load_gds_file_internal(self.current_gds_filepath)
        else:
            print("No GDS file to reload")
    
    def export_structure_overlay(self, file_path):
        """Export the current structure overlay to a file."""
        try:
            if self.current_gds_overlay is None:
                raise ValueError("No structure overlay to export")
            
            import cv2
            success = cv2.imwrite(file_path, self.current_gds_overlay)
            
            if success:
                print(f"Structure overlay exported to: {file_path}")
                return True
            else:
                raise RuntimeError("Failed to write overlay file")
                
        except Exception as e:
            error_msg = f"Failed to export structure overlay: {str(e)}"
            print(error_msg)
            self.gds_operation_error.emit("export_overlay", str(e))
            return False
        
    def load_gds_file_from_path(self, file_path: str):
        """Load GDS file from given file path (for FileSelector integration)."""
        if file_path:
            print(f"Loading GDS file from path: {file_path}")
            self._load_gds_file_internal(file_path)
    
    def select_structure_by_id(self, structure_id: int):
        """Select structure by ID (for FileSelector integration)."""
        try:
            print(f"Selecting structure by ID: {structure_id}")
            
            # Map structure ID to structure name
            structure_name = f"Structure {structure_id}"
            
            # Call existing structure selection method
            self.on_structure_selected(structure_name)
            print(f"Structure {structure_id} ({structure_name}) selected via FileSelector")
                
        except Exception as e:
            error_msg = f"Failed to select structure {structure_id}: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.gds_operation_error.emit("select_structure", str(e))
