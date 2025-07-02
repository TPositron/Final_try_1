"""
New GDS Service - Based on Working Code Implementation
Replaces the complex GDS loading system with a simple, working approach.
"""

import numpy as np
import cv2
import gdspy
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from ..core.gds_display_generator import (
    generate_display_gds, 
    get_structure_info, 
    get_gds_path,
    list_available_structures,
    get_all_structures_info
)
from ..core.gds_aligned_generator import (
    generate_aligned_gds,
    generate_transformed_gds,
    generate_base_gds_image
)


class NewGDSService:
    """
    New GDS service based on working code implementation.
    Simplifies GDS loading and display to use the proven approach.
    """
    
    def __init__(self):
        """Initialize the new GDS service."""
        self.gds_path = get_gds_path()
        self._structure_cache = {}
        self._display_cache = {}
        
    def load_gds_file(self, gds_path: Optional[str] = None) -> bool:
        """
        Load GDS file - simplified approach.
        
        Args:
            gds_path: Optional path to GDS file. If None, uses default.
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if gds_path:
                self.gds_path = gds_path
            
            if not os.path.exists(self.gds_path):
                print(f"GDS file not found: {self.gds_path}")
                return False
            
            # Test load the GDS file
            gds = gdspy.GdsLibrary().read_gds(self.gds_path)
            if not gds.top_level():
                print("No top-level cells found in GDS file")
                return False
            
            print(f"Successfully loaded GDS file: {self.gds_path}")
            return True
            
        except Exception as e:
            print(f"Error loading GDS file: {e}")
            return False
    
    def get_structure_info(self, structure_num: int) -> Optional[Dict]:
        """
        Get information about a structure.
        
        Args:
            structure_num: Structure number (1-5)
            
        Returns:
            Structure information dictionary or None
        """
        return get_structure_info(structure_num)
    
    def list_available_structures(self) -> List[int]:
        """Get list of available structure numbers."""
        return list_available_structures()
    
    def get_all_structures_info(self) -> Dict[int, Dict]:
        """Get information for all available structures."""
        return get_all_structures_info()
    
    def generate_structure_display(self, structure_num: int, target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        """
        Generate display image for a structure.
        
        Args:
            structure_num: Structure number (1-5)
            target_size: Output image size
            
        Returns:
            Generated image or None if failed
        """
        try:
            # Check cache first
            cache_key = f"{structure_num}_{target_size[0]}x{target_size[1]}"
            if cache_key in self._display_cache:
                return self._display_cache[cache_key].copy()
            
            # Generate new image
            image = generate_display_gds(structure_num, target_size)
            
            # Cache the result
            self._display_cache[cache_key] = image.copy()
            
            return image
            
        except Exception as e:
            print(f"Error generating structure display for {structure_num}: {e}")
            return None
    
    def generate_structure_aligned(self, 
                                 structure_num: int,
                                 transform_params: Dict,
                                 target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        """
        Generate aligned structure image with transformations.
        
        Args:
            structure_num: Structure number (1-5)
            transform_params: Transformation parameters
            target_size: Output image size
            
        Returns:
            Aligned image or None if failed
        """
        try:
            image, bounds = generate_aligned_gds(structure_num, transform_params, target_size)
            return image
            
        except Exception as e:
            print(f"Error generating aligned structure for {structure_num}: {e}")
            return None
    
    def generate_structure_with_simple_transforms(self,
                                                structure_num: int,
                                                rotation: float = 0,
                                                zoom: float = 100,
                                                move_x: float = 0,
                                                move_y: float = 0,
                                                target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        """
        Generate structure with simple transformation parameters.
        
        Args:
            structure_num: Structure number (1-5)
            rotation: Rotation angle in degrees
            zoom: Zoom percentage (100 = no change)
            move_x: X movement in pixels
            move_y: Y movement in pixels
            target_size: Output image size
            
        Returns:
            Transformed image or None if failed
        """
        try:
            return generate_transformed_gds(structure_num, rotation, zoom, move_x, move_y, target_size)
            
        except Exception as e:
            print(f"Error generating transformed structure for {structure_num}: {e}")
            return None
    
    def get_structure_by_name(self, structure_name: str) -> Optional[int]:
        """
        Get structure number by name.
        
        Args:
            structure_name: Name of the structure
            
        Returns:
            Structure number or None if not found
        """
        # Handle "Structure X" format
        if structure_name.startswith("Structure "):
            try:
                number_str = structure_name.replace("Structure ", "")
                return int(number_str)
            except ValueError:
                pass
        
        # Map actual structure names to structure numbers
        name_mapping = {
            'Circpol_T2': 1,
            'IP935Left_11': 2,
            'IP935Left_14': 3,
            'QC855GC_CROSS_Bottom': 4,
            'QC935_46': 5
        }
        
        return name_mapping.get(structure_name)
    
    def generate_binary_image(self, structure_name: str, output_dimensions: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        """
        Generate binary image for structure by name (compatibility method).
        
        Args:
            structure_name: Name of the structure
            output_dimensions: Output image size
            
        Returns:
            Binary image or None if failed
        """
        structure_num = self.get_structure_by_name(structure_name)
        if structure_num is None:
            print(f"Structure '{structure_name}' not found")
            return None
        
        return self.generate_structure_display(structure_num, output_dimensions)
    
    def clear_cache(self):
        """Clear all cached images."""
        self._display_cache.clear()
        self._structure_cache.clear()
    
    def get_structure_bounds(self, structure_num: int) -> Optional[Tuple[float, float, float, float]]:
        """
        Get bounds for a structure.
        
        Args:
            structure_num: Structure number (1-5)
            
        Returns:
            Bounds tuple (xmin, ymin, xmax, ymax) or None
        """
        info = self.get_structure_info(structure_num)
        return info['bounds'] if info else None
    
    def get_structure_layers(self, structure_num: int) -> Optional[List[int]]:
        """
        Get layers for a structure.
        
        Args:
            structure_num: Structure number (1-5)
            
        Returns:
            List of layer numbers or None
        """
        info = self.get_structure_info(structure_num)
        return info['layers'] if info else None
    
    def validate_structure_number(self, structure_num: int) -> bool:
        """
        Validate if structure number is available.
        
        Args:
            structure_num: Structure number to validate
            
        Returns:
            True if valid, False otherwise
        """
        return structure_num in self.list_available_structures()
    
    def get_file_info(self) -> Dict[str, Any]:
        """
        Get information about the current GDS file.
        
        Returns:
            File information dictionary
        """
        try:
            if not os.path.exists(self.gds_path):
                return {'error': 'File not found'}
            
            file_size = os.path.getsize(self.gds_path)
            structures = self.get_all_structures_info()
            
            return {
                'file_path': self.gds_path,
                'file_size': file_size,
                'available_structures': len(structures),
                'structure_numbers': list(structures.keys()),
                'structure_names': [info['name'] for info in structures.values()]
            }
            
        except Exception as e:
            return {'error': str(e)}


# Convenience functions for backward compatibility
def load_and_generate_structure(structure_num: int, 
                               gds_path: Optional[str] = None,
                               target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
    """
    One-shot function to load GDS and generate structure display.
    
    Args:
        structure_num: Structure number (1-5)
        gds_path: Optional GDS file path
        target_size: Output image size
        
    Returns:
        Generated image or None if failed
    """
    service = NewGDSService()
    if service.load_gds_file(gds_path):
        return service.generate_structure_display(structure_num, target_size)
    return None


def create_structure_overlay(structure_num: int, 
                           transform_params: Optional[Dict] = None,
                           target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
    """
    Create structure overlay with optional transformations.
    
    Args:
        structure_num: Structure number (1-5)
        transform_params: Optional transformation parameters
        target_size: Output image size
        
    Returns:
        Overlay image or None if failed
    """
    service = NewGDSService()
    if not service.load_gds_file():
        return None
    
    if transform_params:
        return service.generate_structure_aligned(structure_num, transform_params, target_size)
    else:
        return service.generate_structure_display(structure_num, target_size)
