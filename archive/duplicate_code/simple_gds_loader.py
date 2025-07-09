"""
Simple GDS Loader - Streamlined GDS File Processing

This module provides a simplified, object-oriented interface for loading and processing
GDS files. It focuses on ease of use and reliability, providing a clean API for
structure extraction and image generation.

Key Features:
- Object-oriented GDS file handling
- Predefined structure definitions for consistency
- Automatic image generation with proper scaling
- Error handling and fallback mechanisms
- Support for both grayscale and colored output

Main Class:
- SimpleGDSLoader: Main class for GDS file operations

Key Methods:
- generate_structure_image(): Creates images for specific structures
- generate_full_cell_image(): Creates images of entire GDS cells
- get_structure_info(): Retrieves structure metadata
- list_structures(): Lists available structures

Predefined Structures:
Same as gds_display_generator.py for consistency:
1. Circpol_T2, 2. IP935Left_11, 3. IP935Left_14, 4. QC855GC_CROSS_Bottom, 5. QC935_46

Dependencies:
- Uses: gdspy (GDS reading), numpy, cv2 (image processing)
- Called by: services/file_service.py
- Called by: ui/file_operations.py
- Called by: ui/gds_manager.py

Data Flow:
1. Loads GDS file and validates structure
2. Extracts polygons using gdspy API
3. Filters polygons by layer and bounds
4. Transforms coordinates to image space
5. Renders to binary or colored images
6. Returns processed images for display

Advantages over direct gdspy usage:
- Simplified API with error handling
- Consistent structure definitions
- Automatic scaling and centering
- Support for different output formats
"""

import gdspy  # Use gdspy for consistency with other files
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional

STRUCTURES = {
    1: {'name': 'Circpol_T2','bounds': (688.55, 5736.55, 760.55, 5807.1),'layers': [14]},
    2: {'name': 'IP935Left_11', 'bounds': (693.99, 6406.40, 723.59, 6428.96),'layers': [1, 2]},
    3: {'name': 'IP935Left_14','bounds': (980.959, 6025.959, 1001.770, 6044.979),'layers': [1]},
    4: {'name': 'QC855GC_CROSS_Bottom','bounds': (3730.00, 4700.99, 3756.00, 4760.00),'layers': [1, 2]},
    5: {'name': 'QC935_46','bounds': (7195.558, 5046.99, 7203.99, 5055.33964),'layers': [1]}
}

class SimpleGDSLoader:
    def __init__(self, gds_path: str):
        self.gds_path = Path(gds_path)
        self.library = None
        self.cell = None
        self._load_gds()

    def _load_gds(self):
        if not self.gds_path.exists():
            raise FileNotFoundError(f"GDS file not found: {self.gds_path}")
        
        try:
            # Use gdspy for consistent API with other modules
            self.library = gdspy.GdsLibrary().read_gds(str(self.gds_path))
            
            # Get top-level cells
            top_cells = self.library.top_level()
            if not top_cells:
                raise ValueError("No top-level cells found")
            
            self.cell = top_cells[0]
            print(f"Loaded GDS: {self.gds_path.name}, cell: {self.cell.name}")
            
        except Exception as e:
            raise ValueError(f"Failed to load GDS file: {e}")

    def _get_cell_polygons(self):
        """Get polygons from cell using gdspy API."""
        if self.cell is None:
            return {}
        
        try:
            # Use gdspy's get_polygons method which returns a dictionary
            # Format: {(layer, datatype): [polygon_points_array, ...]}
            polygons_by_spec = self.cell.get_polygons(by_spec=True)
            return polygons_by_spec
            
        except Exception as e:
            print(f"Warning: Could not extract polygons from cell: {e}")
            return {}

    def generate_structure_image(self, structure_id: int, target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        if structure_id not in STRUCTURES:
            raise ValueError(f"Structure {structure_id} not defined")
        
        structure = STRUCTURES[structure_id]
        bounds = structure['bounds']
        layers = structure['layers']
        
        xmin, ymin, xmax, ymax = bounds
        gds_width = xmax - xmin
        gds_height = ymax - ymin
        
        scale = min(target_size[0] / gds_width, target_size[1] / gds_height)
        scaled_width = int(gds_width * scale)
        scaled_height = int(gds_height * scale)
        
        offset_x = (target_size[0] - scaled_width) // 2
        offset_y = (target_size[1] - scaled_height) // 2
        
        image = np.ones((target_size[1], target_size[0]), dtype=np.uint8) * 255
        
        polygon_count = 0
        polygons_by_spec = self._get_cell_polygons()
        
        # Process polygons using gdspy format
        for (layer, datatype), polygon_list in polygons_by_spec.items():
            if layer in layers:  # Check if layer is in target layers
                for poly_points in polygon_list:
                    if self._polygon_in_bounds(poly_points, bounds):
                        polygon_count += 1
                        
                        # Transform points to image coordinates
                        norm_points = (poly_points - [xmin, ymin]) * scale
                        int_points = np.round(norm_points).astype(np.int32)
                        int_points += [offset_x, offset_y]
                        int_points[:, 1] = target_size[1] - 1 - int_points[:, 1]
                        
                        if len(int_points) >= 3:
                            # Fixed fillPoly call - use tuple for color parameter
                            cv2.fillPoly(image, [int_points], color=(0,))
        
        print(f"Generated structure {structure_id} ({structure['name']}): {polygon_count} polygons")
        
        # Create colored image
        colored_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        structure_mask = (image < 128) 
        colored_image[structure_mask] = [0, 255, 255] 
        
        return colored_image

    def _polygon_in_bounds(self, points: np.ndarray, bounds: Tuple[float, float, float, float]) -> bool:
        """Check if polygon intersects with bounds."""
        if len(points) == 0:
            return False
            
        xmin, ymin, xmax, ymax = bounds
        poly_xmin, poly_ymin = np.min(points, axis=0)
        poly_xmax, poly_ymax = np.max(points, axis=0)
        
        return (poly_xmax >= xmin and poly_xmin <= xmax and 
                poly_ymax >= ymin and poly_ymin <= ymax)

    @staticmethod
    def get_structure_info(structure_id: int) -> Optional[Dict]:
        if structure_id not in STRUCTURES:
            return None
        
        structure = STRUCTURES[structure_id]
        bounds = structure['bounds']
        xmin, ymin, xmax, ymax = bounds
        
        return {
            'id': structure_id,
            'name': structure['name'],
            'bounds': bounds,
            'layers': structure['layers'],
            'width': xmax - xmin,
            'height': ymax - ymin
        }

    @staticmethod
    def list_structures() -> List[str]:
        return [f"Structure {i}" for i in sorted(STRUCTURES.keys())]

    @staticmethod
    def get_structure_id_by_name(name: str) -> Optional[int]:
        if name.startswith("Structure "):
            try:
                return int(name.split(" ")[1])
            except (ValueError, IndexError):
                return None
        
        for struct_id, struct_data in STRUCTURES.items():
            if struct_data['name'] == name:
                return struct_id
        return None

    def generate_full_cell_image(self, target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        """Generate image of the full cell content."""
        if self.cell is None:
            return None
        
        try:
            polygons_by_spec = self._get_cell_polygons()
            
            if not polygons_by_spec:
                return np.ones(target_size[::-1], dtype=np.uint8) * 255
            
            # Calculate bounds from all polygon points
            all_points = []
            for polygon_list in polygons_by_spec.values():
                for poly_points in polygon_list:
                    if len(poly_points) > 0:
                        all_points.extend(poly_points)
            
            if not all_points:
                return np.ones(target_size[::-1], dtype=np.uint8) * 255
            
            all_points = np.array(all_points)
            xmin, ymin = np.min(all_points, axis=0)
            xmax, ymax = np.max(all_points, axis=0)
            
            gds_width = xmax - xmin
            gds_height = ymax - ymin
            
            if gds_width <= 0 or gds_height <= 0:
                return np.ones(target_size[::-1], dtype=np.uint8) * 255
            
            # Calculate scale and offsets
            scale = min(target_size[0] / gds_width, target_size[1] / gds_height)
            scaled_width = int(gds_width * scale)
            scaled_height = int(gds_height * scale)
            
            offset_x = (target_size[0] - scaled_width) // 2
            offset_y = (target_size[1] - scaled_height) // 2
            
            image = np.ones((target_size[1], target_size[0]), dtype=np.uint8) * 255
            
            # Draw all polygons using gdspy format
            for polygon_list in polygons_by_spec.values():
                for poly_points in polygon_list:
                    if len(poly_points) > 0:
                        try:
                            # Transform to image coordinates
                            norm_poly = (poly_points - [xmin, ymin]) * scale
                            int_poly = np.round(norm_poly).astype(np.int32)
                            int_poly += [offset_x, offset_y]
                            int_poly[:, 1] = target_size[1] - 1 - int_poly[:, 1]
                            int_poly = np.clip(int_poly, [0, 0], [target_size[0]-1, target_size[1]-1])
                            
                            if len(int_poly) >= 3:
                                # Fixed fillPoly call - use tuple for color parameter  
                                cv2.fillPoly(image, [int_poly], color=(0,))
                                
                        except Exception as e:
                            print(f"Warning: Could not draw polygon: {e}")
                            continue
            
            return image
            
        except Exception as e:
            print(f"Error generating full cell image: {e}")
            return None

def load_gds_simple(gds_path: str) -> SimpleGDSLoader:
    return SimpleGDSLoader(gds_path)

def generate_structure_bitmap(gds_path: str, structure_id: int, target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
    loader = SimpleGDSLoader(gds_path)
    return loader.generate_structure_image(structure_id, target_size)
