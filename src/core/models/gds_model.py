import gdstk
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


class GDSError(Exception):
    """Error raised by GDSModel"""
    pass


class GDSModel:
    """
    Simplified GDS model for core GDS data loading and polygon management.
    Complex image generation has been moved to GDSImageService.
    """
    
    def __init__(self, gds_path: Optional[str] = None):
        """
        Initialize GDS model.
        
        Args:
            gds_path: Optional path to GDS file to load immediately
        """
        self.gds_path: Optional[Path] = None
        self.library: Optional[gdstk.Library] = None
        self.cell: Optional[gdstk.Cell] = None
        self.unit_size: Optional[float] = None
        self.bounds: Optional[Tuple[float, float, float, float]] = None
        
        if gds_path:
            self.load_gds_data(gds_path)
    
    def load_gds_data(self, gds_path: str) -> bool:
        """
        Load GDS file and extract basic data.
        
        Args:
            gds_path: Path to the GDS file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.gds_path = Path(gds_path)
            
            if not self.gds_path.exists():
                raise GDSError(f"GDS file not found: {gds_path}")
            
            # Load GDS file
            self.library = gdstk.read_gds(str(self.gds_path))
            
            if not self.library:
                raise GDSError("Failed to read GDS file - library is empty")
            
            # Get basic library information
            self.unit_size = getattr(self.library, 'unit', 1e-6)  # Default 1 micron
            
            # Find the main cell (prefer 'nazca' or first top-level cell)
            top_cells = self.library.top_level()
            if not top_cells:
                raise GDSError("No top-level cells found in GDS file")
            
            # Try to find 'nazca' cell first, otherwise use first top-level cell
            nazca_cell = None
            for cell in self.library.cells:
                if cell.name == 'nazca':
                    nazca_cell = cell
                    break
            
            self.cell = nazca_cell if nazca_cell else top_cells[0]
            
            # Extract basic information
            self.unit_size = getattr(self.library, 'unit', 1e-6)  # Default 1 micron
            bbox = self.cell.bounding_box()
            if bbox is not None:
                # bbox is ((min_x, min_y), (max_x, max_y))
                self.bounds = (bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])
            else:
                self.bounds = (0, 0, 0, 0)
            
            print(f"Loaded GDS file: {self.gds_path.name}")
            print(f"Cell: {self.cell.name}")
            print(f"Unit size: {self.unit_size}")
            print(f"Bounds: {self.bounds}")
            print(f"Available layers: {self.get_layers()}")
            
            return True
            
        except Exception as e:
            print(f"Error loading GDS data: {e}")
            raise GDSError(f"Failed to load GDS file: {e}")
    
    def get_polygons(self, layers: Optional[List[int]] = None) -> Dict[int, List[np.ndarray]]:
        """
        Get polygons from the GDS file, optionally filtered by layers.
        
        Args:
            layers: List of layer numbers to include, None for all layers
            
        Returns:
            Dictionary mapping layer numbers to lists of polygon arrays
        """
        if not self.cell:
            raise GDSError("No GDS data loaded. Call load_gds_data() first.")
        
        # Get polygons from the cell
        polygons_by_layer = {}
        target_layers = set(layers) if layers else None
        
        for polygon in self.cell.polygons:
            layer = polygon.layer
            
            # Filter by specified layers if provided
            if target_layers is not None and layer not in target_layers:
                continue
            
            if layer not in polygons_by_layer:
                polygons_by_layer[layer] = []
            
            # Convert gdstk polygon points to numpy array
            points = np.array(polygon.points)
            if len(points) > 0:
                polygons_by_layer[layer].append(points)
        
        return polygons_by_layer
    
    def get_layers(self) -> List[int]:
        """
        Get list of available layer numbers in the GDS file.
        
        Returns:
            List of layer numbers
        """
        if not self.cell:
            raise GDSError("No GDS data loaded. Call load_gds_data() first.")
        
        layers = set()
        for polygon in self.cell.polygons:
            layers.add(polygon.layer)
        return sorted(list(layers))
    
    def get_polygons_in_region(self, 
                              bounds: Tuple[float, float, float, float],
                              layers: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Get polygons within a specified region.
        
        Args:
            bounds: (x_min, y_min, x_max, y_max) coordinates
            layers: List of layer numbers to include
            
        Returns:
            List of polygon dictionaries with coordinates and layer info
        """
        if not self.cell:
            raise GDSError("No GDS data loaded. Call load_gds_data() first.")
        
        x_min, y_min, x_max, y_max = bounds
        region_polygons = []
        
        # Get polygons for specified layers
        polygons_by_layer = self.get_polygons(layers)
        
        for layer, polygons in polygons_by_layer.items():
            for poly in polygons:
                # Check if polygon overlaps with region
                poly_bounds = self._get_polygon_bounds(poly)
                if self._bounds_overlap(poly_bounds, bounds):
                    region_polygons.append({
                        'coordinates': poly,
                        'layer': layer,
                        'bounds': poly_bounds
                    })
        
        return region_polygons
    
    def get_layer_info(self) -> Dict[int, Dict[str, Any]]:
        """
        Get information about each layer in the GDS file.
        
        Returns:
            Dictionary mapping layer numbers to layer information
        """
        if not self.cell:
            raise GDSError("No GDS data loaded. Call load_gds_data() first.")
        
        layer_info = {}
        polygons_by_layer = self.get_polygons()
        
        for layer, polygons in polygons_by_layer.items():
            # Calculate layer statistics
            polygon_count = len(polygons)
            total_area = 0
            layer_bounds = None
            
            if polygons:
                # Calculate combined bounds and total area
                all_points = np.vstack(polygons)
                layer_bounds = (
                    float(all_points[:, 0].min()),
                    float(all_points[:, 1].min()),
                    float(all_points[:, 0].max()),
                    float(all_points[:, 1].max())
                )
                
                # Approximate total area (simplified calculation)
                for poly in polygons:
                    if len(poly) >= 3:
                        # Use shoelace formula for polygon area
                        x = poly[:, 0]
                        y = poly[:, 1]
                        total_area += 0.5 * abs(sum(x[i]*y[(i+1)%len(x)] - x[(i+1)%len(x)]*y[i] for i in range(len(x))))
            
            layer_info[layer] = {
                'polygon_count': polygon_count,
                'total_area': total_area,
                'bounds': layer_bounds
            }
        
        return layer_info
    
    def get_gds_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the loaded GDS file.
        
        Returns:
            Dictionary with GDS file information
        """
        if not self.cell:
            raise GDSError("No GDS data loaded. Call load_gds_data() first.")
        
        return {
            'file_path': str(self.gds_path),
            'cell_name': self.cell.name,
            'unit_size': self.unit_size,
            'bounds': self.bounds,
            'layers': self.get_layers(),
            'layer_info': self.get_layer_info(),
            'total_layers': len(self.get_layers())
        }
    
    def is_loaded(self) -> bool:
        """Check if GDS data is currently loaded."""
        return self.cell is not None
    
    def _get_polygon_bounds(self, polygon: np.ndarray) -> Tuple[float, float, float, float]:
        """Get bounding box of a polygon."""
        if len(polygon) == 0:
            return (0, 0, 0, 0)
        x_coords = polygon[:, 0]
        y_coords = polygon[:, 1]
        return (
            float(x_coords.min()),
            float(y_coords.min()),
            float(x_coords.max()),
            float(y_coords.max())
        )
    
    def _bounds_overlap(self, bounds1: Tuple[float, float, float, float], 
                       bounds2: Tuple[float, float, float, float]) -> bool:
        """Check if two bounding boxes overlap."""
        x1_min, y1_min, x1_max, y1_max = bounds1
        x2_min, y2_min, x2_max, y2_max = bounds2
        
        return (x1_min < x2_max and x1_max > x2_min and 
                y1_min < y2_max and y1_max > y2_min)
    
    def __repr__(self) -> str:
        """String representation of GDSModel."""
        if self.is_loaded():
            return f"GDSModel(file='{self.gds_path.name}', cell='{self.cell.name}', layers={len(self.get_layers())})"
        else:
            return "GDSModel(not loaded)"


# Legacy compatibility method
def extract_structure_images(gds_path: str, structures: Dict) -> Dict[str, np.ndarray]:
    """
    Legacy compatibility function.
    Redirects to GDSImageService for actual image generation.
    
    Args:
        gds_path: Path to the GDS file
        structures: Dict of structure definitions
        
    Returns:
        Dict mapping structure names to binary image arrays
        
    Note:
        This function is deprecated. Use GDSImageService.generate_multiple_images() instead.
    """
    print("Warning: extract_structure_images() is deprecated. Use GDSImageService instead.")
    
    try:
        from ...services.gds_image_service import GDSImageService
        
        service = GDSImageService()
        service.load_gds_file(gds_path)
        
        # Convert old structure format to names
        structure_names = [struct['name'] for struct in structures.values()]
        
        return service.generate_multiple_images(structure_names)
        
    except ImportError:
        raise GDSError("GDSImageService not available for legacy compatibility")


if __name__ == "__main__":
    # Example usage
    print("Simplified GDS Model")
    print("=" * 30)
    
    # Test with mock path (would need actual file)
    try:
        model = GDSModel()
        print(f"Model created: {model}")
        print("Use load_gds_data(path) to load a GDS file")
        
    except Exception as e:
        print(f"Error: {e}")
