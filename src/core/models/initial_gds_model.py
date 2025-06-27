"""Initial GDS model for parsing raw GDS files and extracting polygons."""

import gdspy
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class InitialGDSModel:
    """Parses raw GDS files and extracts polygon data."""
    
    def __init__(self, gds_path: str):
        """Initialize with path to GDS file."""
        self.gds_path = Path(gds_path)
        self.library = None
        self.cell = None
        self._load_gds()
    
    def _load_gds(self) -> None:
        """Load the GDS file and get the main cell."""
        self.library = gdspy.GdsLibrary()
        self.library.read_gds(str(self.gds_path))
        
        # Get top cell (either named 'nazca' or first top-level cell)
        self.cell = self.library.cells.get('nazca', next(iter(self.library.top_level())))
        if not self.cell:
            raise ValueError("No valid cell found in GDS file")
    
    def get_layers(self) -> List[int]:
        """Get list of available layers in the GDS file."""
        polygons = self.cell.get_polygons(by_spec=True)
        return [key[0] for key in polygons.keys()]
    
    def get_polygons_by_layer(self, layer: int) -> List[np.ndarray]:
        """Get all polygons for a specific layer."""
        polygons = self.cell.get_polygons(by_spec=True)
        key = (layer, 0)  # layer number and datatype
        return polygons.get(key, [])
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get the overall bounds of all geometry in the cell."""
        return self.cell.get_bounding_box().flatten() if self.cell.get_bounding_box() is not None else (0, 0, 0, 0)
    
    def extract_polygons(self, layers: List[int], bounds: Optional[Tuple[float, float, float, float]] = None) -> Dict[int, List[np.ndarray]]:
        """Extract polygons for specified layers within optional bounds."""
        result = {}
        
        for layer in layers:
            polygons = self.get_polygons_by_layer(layer)
            
            if bounds:
                xmin, ymin, xmax, ymax = bounds
                filtered_polygons = []
                
                for poly in polygons:
                    # Check if polygon intersects with bounds
                    poly_min = np.min(poly, axis=0)
                    poly_max = np.max(poly, axis=0)
                    
                    if not (poly_max[0] < xmin or poly_min[0] > xmax or 
                           poly_max[1] < ymin or poly_min[1] > ymax):
                        filtered_polygons.append(poly)
                
                result[layer] = filtered_polygons
            else:
                result[layer] = polygons
        
        return result
