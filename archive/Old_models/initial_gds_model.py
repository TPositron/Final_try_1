"""
Initial GDS model for loading and processing GDS files using gdstk.

This module provides the InitialGdsModel class for:
- Loading GDS files with proper error handling
- Extracting basic information (layers, cells, bounds)
- Basic polygon data extraction
"""

import gdstk
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

from ..utils.error_handling import handle_errors


logger = logging.getLogger(__name__)


class InitialGdsModel:
    """
    Basic GDS file loader and processor using gdstk.
    
    Provides core functionality for loading GDS files and extracting
    basic information like layers, cells, and polygon data.
    """
    
    def __init__(self, gds_path: str):
        """
        Initialize with path to GDS file.
        
        Args:
            gds_path: Path to the GDS file to load
            
        Raises:
            FileNotFoundError: If GDS file doesn't exist
            ValueError: If GDS file cannot be loaded or is invalid
        """
        self.gds_path = Path(gds_path)
        self.library = None
        self.cell = None
        self.unit = None
        self.precision = None
        self.bounds = None
        self._metadata = {}
        
        self._load_gds()
    
    @handle_errors
    def _load_gds(self) -> None:
        """Load the GDS file and extract basic information."""
        if not self.gds_path.exists():
            raise FileNotFoundError(f"GDS file not found: {self.gds_path}")
        
        if not self.gds_path.suffix.lower() in ['.gds', '.gdsii']:
            raise ValueError(f"Invalid GDS file extension: {self.gds_path.suffix}")
        
        try:
            # Load the GDS library
            self.library = gdstk.read_gds(str(self.gds_path))
            
            if not self.library:
                raise ValueError("Failed to read GDS file - library is empty")
            
            # Get basic library information
            self.unit = getattr(self.library, 'unit', 1e-6)  # Default 1 micron
            self.precision = getattr(self.library, 'precision', 1e-9)  # Default 1 nm
            
            # Find the main cell (prefer 'nazca' or first top-level cell)
            top_cells = self.library.top_level()
            if not top_cells:
                raise ValueError("No top-level cells found in GDS file")
            
            # Try to find 'nazca' cell first, otherwise use first top-level cell
            self.cell = None
            for cell in top_cells:
                if cell.name.lower() == 'nazca':
                    self.cell = cell
                    break
            
            if self.cell is None:
                self.cell = top_cells[0]
            
            # Calculate bounds
            self._calculate_bounds()
            
            # Store metadata
            self._metadata = {
                'file_path': str(self.gds_path),
                'file_size': self.gds_path.stat().st_size,
                'cell_name': self.cell.name,
                'unit': self.unit,
                'precision': self.precision,
                'num_top_cells': len(top_cells),
                'total_cells': len(self.library.cells)
            }
            
            logger.info(f"Successfully loaded GDS file: {self.gds_path}")
            logger.debug(f"Main cell: {self.cell.name}, Unit: {self.unit}, Precision: {self.precision}")
            
        except Exception as e:
            logger.error(f"Failed to load GDS file {self.gds_path}: {e}")
            raise ValueError(f"Invalid or corrupted GDS file: {e}") from e
    
    def get_layers(self) -> List[int]:
        """Get list of available layers in the GDS file."""
        polygons = self.cell.get_polygons(by_spec=True)
        return [key[0] for key in polygons.keys()]
    
    def get_polygons(self, layers: List[int]) -> List[np.ndarray]:
        """Retrieve polygons for specified layers."""
        polygons = self.cell.get_polygons(by_spec=True)
        result = []
        for layer in layers:
            key = (layer, 0)  # layer number and datatype
            result.extend(polygons.get(key, []))
        return result
    
    def to_bitmap(self, resolution: Tuple[int, int]) -> np.ndarray:
        """Generate a binary raster image of the structure scaled to fit the given resolution."""
        if not self.bounds:
            raise ValueError("No valid bounds found for GDS file")

        xmin, ymin, xmax, ymax = self.bounds
        gds_width = xmax - xmin
        gds_height = ymax - ymin

        scale = min(resolution[0] / gds_width, resolution[1] / gds_height)
        scaled_width = int(gds_width * scale)
        scaled_height = int(gds_height * scale)
        offset_x = (resolution[0] - scaled_width) // 2
        offset_y = (resolution[1] - scaled_height) // 2

        image = np.ones((resolution[1], resolution[0]), dtype=np.uint8) * 255

        polygons = self.cell.get_polygons(by_spec=True)
        for key, poly_list in polygons.items():
            for poly in poly_list:
                norm_poly = (poly - [xmin, ymin]) * scale
                int_poly = np.round(norm_poly).astype(np.int32)
                int_poly += [offset_x, offset_y]
                int_poly[:, 1] = resolution[1] - 1 - int_poly[:, 1]  # Flip Y-axis
                cv2.fillPoly(image, [int_poly], color=0)

        return image
    
    def extract_structure_region(self, 
                                bounds: Tuple[float, float, float, float],
                                layers: Optional[List[int]] = None) -> Dict[str, any]:
        """
        Extract rectangular region from GDS based on coordinates.
        
        Args:
            bounds: (x_min, y_min, x_max, y_max) coordinates
            layers: List of layer numbers to include, None for all layers
            
        Returns:
            Dictionary containing extracted data
        """
        x_min, y_min, x_max, y_max = bounds
        
        # Validate bounds
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(f"Invalid bounds: {bounds}")
        
        # Get polygons in the region
        region_polygons = self.get_polygons_in_region(bounds, layers)
        
        # Calculate region info
        region_width = x_max - x_min
        region_height = y_max - y_min
        region_area = region_width * region_height
        
        return {
            'bounds': bounds,
            'polygons': region_polygons,
            'layers': layers or self.get_layers(),
            'width': region_width,
            'height': region_height,
            'area': region_area,
            'polygon_count': len(region_polygons)
        }
    
    def filter_layers(self, layer_list: List[int]) -> Dict[int, List[np.ndarray]]:
        """
        Simple layer filtering to get polygons by layer.
        
        Args:
            layer_list: List of layer numbers to filter
            
        Returns:
            Dictionary mapping layer numbers to polygon lists
        """
        polygons_by_layer = {}
        all_polygons = self.cell.get_polygons(by_spec=True)
        
        for layer in layer_list:
            key = (layer, 0)  # layer number and datatype
            layer_polygons = all_polygons.get(key, [])
            polygons_by_layer[layer] = layer_polygons
        
        return polygons_by_layer
    
    def get_polygons_in_region(self, 
                              bounds: Tuple[float, float, float, float],
                              layers: Optional[List[int]] = None) -> List[Dict[str, any]]:
        """
        Basic polygon extraction within a specified region.
        
        Args:
            bounds: (x_min, y_min, x_max, y_max) coordinates
            layers: List of layer numbers to include, None for all layers
            
        Returns:
            List of polygon dictionaries with coordinates and layer info
        """
        x_min, y_min, x_max, y_max = bounds
        region_polygons = []
        
        # Get polygons for specified layers or all layers
        target_layers = layers if layers is not None else self.get_layers()
        filtered_polygons = self.filter_layers(target_layers)
        
        for layer, polygons in filtered_polygons.items():
            for poly in polygons:
                # Check if polygon overlaps with region
                poly_bounds = self._get_polygon_bounds(poly)
                if self._bounds_overlap(poly_bounds, bounds):
                    # Clip polygon to region if it extends beyond
                    clipped_poly = self._clip_polygon_to_region(poly, bounds)
                    if clipped_poly is not None and len(clipped_poly) > 0:
                        region_polygons.append({
                            'coordinates': clipped_poly,
                            'layer': layer,
                            'original_bounds': poly_bounds,
                            'clipped_bounds': self._get_polygon_bounds(clipped_poly)
                        })
        
        return region_polygons
    
    def extract_structure_from_definition(self, structure_def: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structure based on structure definition object.
        
        Args:
            structure_def: Structure definition dictionary with bounds, layers, name, etc.
            
        Returns:
            Dictionary containing extracted structure data
        """
        extracted = self.extract_structure_region(structure_def['bounds'], structure_def['layers'])
        extracted.update({
            'name': structure_def.get('name', 'Unknown'),
            'description': structure_def.get('description', ''),
            'center': structure_def.get('center', (0, 0)),
            'structure_area': structure_def.get('area', 0)
        })
        return extracted
    
    def generate_structure_bitmap(self, 
                                 bounds: Tuple[float, float, float, float],
                                 layers: Optional[List[int]] = None,
                                 resolution: Tuple[int, int] = (1024, 666)) -> np.ndarray:
        """
        Generate binary image for a specific structure region.
        
        Args:
            bounds: (x_min, y_min, x_max, y_max) coordinates
            layers: List of layer numbers to include
            resolution: Output image resolution (width, height)
            
        Returns:
            Binary image array
        """
        x_min, y_min, x_max, y_max = bounds
        region_width = x_max - x_min
        region_height = y_max - y_min
        
        # Calculate scaling
        scale_x = resolution[0] / region_width
        scale_y = resolution[1] / region_height
        scale = min(scale_x, scale_y)  # Maintain aspect ratio
        
        # Create image
        image = np.ones(resolution[::-1], dtype=np.uint8) * 255  # White background
        
        # Get polygons in region
        region_polygons = self.get_polygons_in_region(bounds, layers)
        
        for poly_data in region_polygons:
            poly = poly_data['coordinates']
            
            # Transform coordinates to image space
            transformed_poly = np.copy(poly)
            transformed_poly[:, 0] = (poly[:, 0] - x_min) * scale
            transformed_poly[:, 1] = (poly[:, 1] - y_min) * scale
            
            # Flip Y-axis (GDS Y increases upward, image Y increases downward)
            transformed_poly[:, 1] = resolution[1] - 1 - transformed_poly[:, 1]
            
            # Convert to integer coordinates
            int_poly = np.round(transformed_poly).astype(np.int32)
            
            # Draw polygon (black on white background)
            cv2.fillPoly(image, [int_poly], color=0)
        
        return image
    
    def _get_polygon_bounds(self, polygon: np.ndarray) -> Tuple[float, float, float, float]:
        """Get bounding box of a polygon."""
        if len(polygon) == 0:
            return (0, 0, 0, 0)
        x_coords = polygon[:, 0]
        y_coords = polygon[:, 1]
        return (x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max())
    
    def _bounds_overlap(self, bounds1: Tuple[float, float, float, float], 
                       bounds2: Tuple[float, float, float, float]) -> bool:
        """Check if two bounding boxes overlap."""
        x1_min, y1_min, x1_max, y1_max = bounds1
        x2_min, y2_min, x2_max, y2_max = bounds2
        
        return (x1_min < x2_max and x1_max > x2_min and 
                y1_min < y2_max and y1_max > y2_min)
    
    def _clip_polygon_to_region(self, polygon: np.ndarray, 
                               bounds: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """
        Clip polygon to region bounds (simplified implementation).
        
        For now, this returns the original polygon if it overlaps,
        or None if it doesn't. A more sophisticated implementation
        would actually clip the polygon geometry.
        """
        poly_bounds = self._get_polygon_bounds(polygon)
        if self._bounds_overlap(poly_bounds, bounds):
            return polygon
        return None
