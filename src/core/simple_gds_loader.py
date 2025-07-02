"""
Simple GDS loader - directly based on your working draft approach.
No complex cell searching, just load and extract by coordinates.
"""

import gdstk  # or we can switch back to gdspy if you prefer
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Simple structure definitions - easy to add new ones
STRUCTURES = {
    1: {
        'name': 'Circpol_T2',
        'bounds': (688.55, 5736.55, 760.55, 5807.1),
        'layers': [14]
    },
    2: {
        'name': 'IP935Left_11', 
        'bounds': (693.99, 6406.40, 723.59, 6428.96),
        'layers': [1, 2]
    },
    3: {
        'name': 'IP935Left_14',
        'bounds': (980.959, 6025.959, 1001.770, 6044.979),
        'layers': [1]
    },
    4: {
        'name': 'QC855GC_CROSS_Bottom',
        'bounds': (3730.00, 4700.99, 3756.00, 4760.00),
        'layers': [1, 2]
    },
    5: {
        'name': 'QC935_46',
        'bounds': (7195.558, 5046.99, 7203.99, 5055.33964),
        'layers': [1]
    }
}

class SimpleGDSLoader:
    """Simple GDS loader - no complex architecture, just load and extract."""
    
    def __init__(self, gds_path: str):
        self.gds_path = Path(gds_path)
        self.library = None
        self.cell = None
        self._load_gds()
    
    def _load_gds(self):
        """Load GDS file - simple and direct."""
        if not self.gds_path.exists():
            raise FileNotFoundError(f"GDS file not found: {self.gds_path}")
        
        try:
            # Load the library
            self.library = gdstk.read_gds(str(self.gds_path))
            
            # Get the first top-level cell (simple approach)
            top_cells = self.library.top_level()
            if not top_cells:
                raise ValueError("No top-level cells found")
            
            self.cell = top_cells[0]
            print(f"Loaded GDS: {self.gds_path.name}, cell: {self.cell.name}")
            
        except Exception as e:
            raise ValueError(f"Failed to load GDS file: {e}")
    
    def generate_structure_image(self, structure_id: int, target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        """
        Generate binary image for a structure - simple coordinate-based extraction.
        
        Args:
            structure_id: Structure number (1-5)
            target_size: Output image size (width, height)
            
        Returns:
            Binary numpy array or None if failed
        """
        if structure_id not in STRUCTURES:
            raise ValueError(f"Structure {structure_id} not defined")
        
        structure = STRUCTURES[structure_id]
        bounds = structure['bounds']
        layers = structure['layers']
        
        xmin, ymin, xmax, ymax = bounds
        gds_width = xmax - xmin
        gds_height = ymax - ymin
        
        # Calculate scale to fit in target size
        scale = min(target_size[0] / gds_width, target_size[1] / gds_height)
        scaled_width = int(gds_width * scale)
        scaled_height = int(gds_height * scale)
        
        # Center the image
        offset_x = (target_size[0] - scaled_width) // 2
        offset_y = (target_size[1] - scaled_height) // 2
        
        # Create white background image
        image = np.ones((target_size[1], target_size[0]), dtype=np.uint8) * 255
        
        # Extract and draw polygons
        polygon_count = 0
        for polygon in self.cell.polygons:
            if polygon.layer in layers:
                points = np.array(polygon.points)
                
                # Check if polygon intersects with our bounds
                if self._polygon_in_bounds(points, bounds):
                    polygon_count += 1
                    
                    # Transform to image coordinates
                    norm_points = (points - [xmin, ymin]) * scale
                    int_points = np.round(norm_points).astype(np.int32)
                    int_points += [offset_x, offset_y]
                    
                    # Flip Y-axis to match your working code
                    int_points[:, 1] = target_size[1] - 1 - int_points[:, 1]
                    
                    # Draw polygon (black on white background)
                    if len(int_points) >= 3:
                        cv2.fillPoly(image, [int_points], color=0)
        
        print(f"Generated structure {structure_id} ({structure['name']}): {polygon_count} polygons")
        
        # Convert to colored binary for better visibility
        # Create a colored version where structures are visible
        colored_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        # Make structures cyan/green for good contrast on SEM images
        structure_mask = (image < 128)  # Where structures are (black pixels)
        colored_image[structure_mask] = [0, 255, 255]  # Cyan color
        
        return colored_image
    
    def _polygon_in_bounds(self, points: np.ndarray, bounds: Tuple[float, float, float, float]) -> bool:
        """Check if polygon intersects with bounds."""
        xmin, ymin, xmax, ymax = bounds
        poly_xmin, poly_ymin = np.min(points, axis=0)
        poly_xmax, poly_ymax = np.max(points, axis=0)
        
        return (poly_xmax >= xmin and poly_xmin <= xmax and 
                poly_ymax >= ymin and poly_ymin <= ymax)
    
    @staticmethod
    def get_structure_info(structure_id: int) -> Optional[Dict]:
        """Get structure information."""
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
        """Get list of available structure names."""
        return [f"Structure {i}" for i in sorted(STRUCTURES.keys())]
    
    @staticmethod
    def get_structure_id_by_name(name: str) -> Optional[int]:
        """Get structure ID by name (e.g., 'Structure 2' -> 2)."""
        if name.startswith("Structure "):
            try:
                return int(name.split(" ")[1])
            except (ValueError, IndexError):
                return None
        
        # Fallback: look up by internal name
        for struct_id, struct_data in STRUCTURES.items():
            if struct_data['name'] == name:
                return struct_id
        return None
    
    def generate_full_cell_image(self, target_size: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        """
        Generate binary image for the full GDS cell - like the old system.
        
        Args:
            target_size: Output image size (width, height)
            
        Returns:
            Binary numpy array or None if failed
        """
        if self.cell is None:
            return None
        
        try:
            # Get all polygons from the cell using gdstk approach
            polygons_by_layer = {}
            
            # In gdstk, polygons are direct attributes of the cell
            for polygon in self.cell.polygons:
                layer = polygon.layer
                datatype = polygon.datatype
                key = (layer, datatype)
                
                if key not in polygons_by_layer:
                    polygons_by_layer[key] = []
                polygons_by_layer[key].append(polygon.points)
            
            if not polygons_by_layer:
                return np.ones(target_size[::-1], dtype=np.uint8) * 255
            
            # Get bounds of all polygons
            all_points = []
            for layer_polygons in polygons_by_layer.values():
                for poly in layer_polygons:
                    all_points.extend(poly)
            
            if not all_points:
                return np.ones(target_size[::-1], dtype=np.uint8) * 255
                
            all_points = np.array(all_points)
            xmin, ymin = np.min(all_points, axis=0)
            xmax, ymax = np.max(all_points, axis=0)
            
            gds_width = xmax - xmin
            gds_height = ymax - ymin
            
            if gds_width <= 0 or gds_height <= 0:
                return np.ones(target_size[::-1], dtype=np.uint8) * 255
            
            # Calculate scale to fit in target size
            scale = min(target_size[0] / gds_width, target_size[1] / gds_height)
            scaled_width = int(gds_width * scale)
            scaled_height = int(gds_height * scale)
            
            # Center the image
            offset_x = (target_size[0] - scaled_width) // 2
            offset_y = (target_size[1] - scaled_height) // 2
            
            # Create white background
            image = np.ones((target_size[1], target_size[0]), dtype=np.uint8) * 255
            
            # Draw all polygons
            for layer_polygons in polygons_by_layer.values():
                for poly in layer_polygons:
                    # Transform to image coordinates
                    norm_poly = (poly - [xmin, ymin]) * scale
                    int_poly = np.round(norm_poly).astype(np.int32)
                    int_poly += [offset_x, offset_y]
                    
                    # Flip Y-axis
                    int_poly[:, 1] = target_size[1] - 1 - int_poly[:, 1]
                    
                    # Clip to image bounds
                    int_poly = np.clip(int_poly, [0, 0], [target_size[0]-1, target_size[1]-1])
                    
                    # Draw polygon
                    if len(int_poly) >= 3:
                        cv2.fillPoly(image, [int_poly], color=0)
            
            return image
            
        except Exception as e:
            print(f"Error generating full cell image: {e}")
            return None


# Simple functions for easy use
def load_gds_simple(gds_path: str) -> SimpleGDSLoader:
    """Load GDS file the simple way."""
    return SimpleGDSLoader(gds_path)

def generate_structure_bitmap(gds_path: str, structure_id: int, target_size: Tuple[int, int] = (1024, 666)) -> np.ndarray:
    """One-line function to generate structure bitmap."""
    loader = SimpleGDSLoader(gds_path)
    return loader.generate_structure_image(structure_id, target_size)
