"""
GDS Manager Module
Handles all GDS-related operations including file loading, structure selection, and overlay generation.
"""

import os
from pathlib import Path
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QObject, Signal

from src.services.new_gds_service import NewGDSService
from src.core.simple_gds_loader import SimpleGDSLoader


class GDSManager(QObject):
    """Handles all GDS-related operations and structure management."""
    
    # Signals
    gds_file_loaded = Signal(str)  # file path
    structure_selected = Signal(str, object)  # structure name, overlay
    structure_combo_updated = Signal(list)  # list of structure names
    
    def __init__(self, main_window):
        """Initialize GDS manager with reference to main window."""
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
        
        # Default GDS file
        self.default_gds_file = "Institute_Project_GDS1.gds"
    
    def load_gds_file(self, file_path):
        """Load a GDS file and enable structure selection."""
        try:
            print(f"Loading GDS file: {file_path}")
            
            gds_filename = Path(file_path).name
            
            # Use new GDS service for loading
            success = self.new_gds_service.load_gds_file(file_path)
            
            if success:
                # Store current GDS info
                self.current_gds_filename = gds_filename
                self.current_gds_filepath = str(file_path)
                
                # Populate structure combo
                self.populate_structure_combo()
                
                # Update status
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage(f"Loaded GDS file: {gds_filename}")
                
                # Emit signal
                self.gds_file_loaded.emit(file_path)
                
                print(f"✓ GDS file loaded successfully: {gds_filename}")
                return True
            else:
                raise RuntimeError(f"Failed to load GDS file: {gds_filename}")
                
        except Exception as e:
            error_msg = f"Failed to load GDS file: {str(e)}"
            print(f"GDS loading error: {error_msg}")
            QMessageBox.critical(self.main_window, "Error", error_msg)
            
            # Reset state on error
            self.current_gds_filename = None
            self.current_gds_filepath = None
            return False
    
    def populate_structure_combo(self):
        """Populate the structure selection combo box with available structures."""
        try:
            if not hasattr(self.main_window, 'structure_combo'):
                print("Warning: No structure combo box found")
                return
            
            combo = self.main_window.structure_combo
            combo.clear()
            combo.addItem("Select Structure...", "")
            
            # Get structures from GDS service
            structures_info = self.new_gds_service.get_all_structures_info()
            structure_names = []
            
            for structure_num, info in structures_info.items():
                display_name = f"Structure {structure_num} - {info['name']}"
                combo.addItem(display_name, f"Structure {structure_num}")
                structure_names.append(display_name)
            
            print(f"Populated structure combo with {len(structures_info)} structures")
            
            # Emit signal with structure list
            self.structure_combo_updated.emit(structure_names)
            
            # Enable the combo box
            combo.setEnabled(True)
            
        except Exception as e:
            print(f"Error populating structure combo: {e}")
    
    def auto_load_default_gds(self):
        """Automatically load the default GDS file if it exists."""
        try:
            if hasattr(self.main_window, 'file_service'):
                default_gds_path = self.main_window.file_service.get_gds_dir() / self.default_gds_file
            else:
                # Fallback path construction
                default_gds_path = Path("Data/GDS") / self.default_gds_file
            
            if default_gds_path.exists():
                print(f"Auto-loading default GDS: {default_gds_path}")
                self.load_gds_file(str(default_gds_path))
            else:
                print(f"Default GDS file not found: {default_gds_path}")
                
        except Exception as e:
            print(f"Error auto-loading default GDS: {e}")
    
    def on_structure_selected(self, structure_name):
        """Handle structure selection from combo box."""
        print(f"on_structure_selected called with: {structure_name}")
        
        # Prevent recursion
        if self._processing_structure_selection:
            print("Already processing structure selection, ignoring")
            return
        
        self._processing_structure_selection = True
        
        try:
            # Clear selection case
            if not structure_name or structure_name == "Select Structure...":
                self._clear_structure_selection()
                return
            
            print(f"Loading structure: {structure_name}")
            
            # Get structure number from name
            structure_num = self._extract_structure_number(structure_name)
            if structure_num is None:
                raise ValueError(f"Could not determine structure number from '{structure_name}'")
            
            # Determine canvas size from current SEM image
            canvas_width, canvas_height = self._get_canvas_size()
            
            # Generate structure display
            structure_display = self.new_gds_service.generate_structure_display(
                structure_num, canvas_width, canvas_height
            )
            
            if structure_display is not None:
                # Store the overlay and structure info
                self.current_gds_overlay = structure_display
                self.current_structure_name = structure_name
                
                # Update the image viewer
                if hasattr(self.main_window, 'image_viewer'):
                    self.main_window.image_viewer.set_gds_overlay(structure_display)
                
                # Update status
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage(f"Loaded structure: {structure_name}")
                
                # Emit signal
                self.structure_selected.emit(structure_name, structure_display)
                
                print(f"✓ Generated structure display for structure {structure_num}")
            else:
                raise RuntimeError(f"Failed to generate display for structure {structure_num}")
                
        except Exception as e:
            error_msg = f"Failed to load structure {structure_name}: {str(e)}"
            print(f"Structure loading error: {error_msg}")
            QMessageBox.warning(self.main_window, "Structure Loading Error", error_msg)
            
            # Clear overlay on error
            self._clear_structure_selection()
            
        finally:
            self._processing_structure_selection = False
    
    def _clear_structure_selection(self):
        """Clear the current structure selection."""
        self.current_structure_name = None
        self.current_gds_overlay = None
        
        if hasattr(self.main_window, 'image_viewer'):
            self.main_window.image_viewer.set_gds_overlay(None)
    
    def _extract_structure_number(self, structure_name):
        """Extract structure number from structure name."""
        try:
            # First try to get from GDS service mapping
            structure_num = self.new_gds_service.get_structure_by_name(structure_name)
            if structure_num is not None:
                return structure_num
            
            # Fallback: extract from "Structure X" format
            if structure_name.startswith("Structure "):
                try:
                    # Extract number from name like "Structure 1 - Description"
                    parts = structure_name.split(" - ")[0]  # Get "Structure 1" part
                    number_str = parts.replace("Structure ", "")
                    return int(number_str)
                except (ValueError, IndexError):
                    pass
            
            return None
            
        except Exception as e:
            print(f"Error extracting structure number: {e}")
            return None
    
    def _get_canvas_size(self):
        """Get the canvas size for structure display generation."""
        # Default size
        canvas_width = 1024
        canvas_height = 666
        
        # Try to get size from current SEM image
        if (hasattr(self.main_window, 'current_sem_image') and 
            self.main_window.current_sem_image is not None):
            height, width = self.main_window.current_sem_image.shape[:2]
            canvas_width = width
            canvas_height = height
            print(f"Using SEM image dimensions: {canvas_width}x{canvas_height}")
        else:
            print(f"Using default canvas dimensions: {canvas_width}x{canvas_height}")
        
        return canvas_width, canvas_height
    
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
        try:
            self._clear_structure_selection()
            
            # Reset combo box to first item
            if hasattr(self.main_window, 'structure_combo'):
                self.main_window.structure_combo.setCurrentIndex(0)
            
            print("Structure selection reset")
            
        except Exception as e:
            print(f"Error resetting structure selection: {e}")
    
    def is_gds_loaded(self):
        """Check if a GDS file is currently loaded."""
        return self.current_gds_filename is not None
    
    def is_structure_selected(self):
        """Check if a structure is currently selected."""
        return self.current_structure_name is not None
    
    def get_available_structures(self):
        """Get list of available structures from the loaded GDS file."""
        try:
            if not self.is_gds_loaded():
                return []
            
            structures_info = self.new_gds_service.get_all_structures_info()
            structure_list = []
            
            for structure_num, info in structures_info.items():
                structure_list.append({
                    'number': structure_num,
                    'name': info['name'],
                    'display_name': f"Structure {structure_num} - {info['name']}"
                })
            
            return structure_list
            
        except Exception as e:
            print(f"Error getting available structures: {e}")
            return []
    
    def reload_current_structure(self):
        """Reload the currently selected structure."""
        if self.current_structure_name:
            structure_name = self.current_structure_name
            self._clear_structure_selection()
            self.on_structure_selected(structure_name)
    
    def export_structure_overlay(self, file_path):
        """Export the current structure overlay as an image."""
        try:
            if self.current_gds_overlay is None:
                raise ValueError("No structure overlay to export")
            
            import cv2
            import numpy as np
            
            # Convert overlay to appropriate format for saving
            overlay = self.current_gds_overlay
            if overlay.dtype != np.uint8:
                if overlay.dtype in [np.float32, np.float64]:
                    # Normalize float images to 0-255
                    overlay = cv2.normalize(overlay, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                else:
                    overlay = overlay.astype(np.uint8)
            
            # Save the overlay
            success = cv2.imwrite(file_path, overlay)
            
            if success:
                print(f"✓ Structure overlay exported to: {file_path}")
                return True
            else:
                raise RuntimeError("cv2.imwrite failed")
                
        except Exception as e:
            error_msg = f"Failed to export structure overlay: {str(e)}"
            print(f"Export error: {error_msg}")
            QMessageBox.critical(self.main_window, "Export Error", error_msg)
            return False
    
    def get_gds_service_info(self):
        """Get information about the GDS service state."""
        return {
            'service_loaded': self.new_gds_service is not None,
            'gds_file_loaded': self.is_gds_loaded(),
            'current_gds_file': self.current_gds_filename,
            'structure_selected': self.is_structure_selected(),
            'current_structure': self.current_structure_name,
            'available_structures': len(self.get_available_structures())
        }
