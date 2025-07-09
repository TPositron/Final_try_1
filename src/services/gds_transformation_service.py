"""
GDS Transformation Service - GDS Layout Transformation Operations

This service handles transformation of GDS layout structures including translation,
rotation, and scaling operations. It modifies polygon coordinates directly and
creates new GDS files with applied transformations.

Main Class:
- GdsTransformationService: Service for transforming GDS structures

Key Methods:
- transform_structure(): Transforms GDS structure with specified parameters
- _get_cell_polygons(): Extracts polygons from gdstk cell with API compatibility
- save_transformed_gds(): Saves transformed cell to new GDS file
- validate_transformation_parameters(): Validates transformation parameter ranges

Dependencies:
- Uses: gdstk (GDS file operations), numpy (array operations), math (trigonometry)
- Uses: pathlib.Path (file operations)
- Uses: utils/transformations (transformation utilities and validation)
- Used by: services/file_service.py (aligned GDS saving)
- Used by: UI components for GDS transformation

Transformation Operations:
- Translation: X/Y offset in pixels converted to GDS units
- Scaling: Uniform scale factor applied to polygon coordinates
- Rotation: Arbitrary angle rotation around structure center
- Coordinate system conversion between UI pixels and GDS units
- Center-based transformations for intuitive behavior

GDS File Handling:
- gdstk library integration for GDS file operations
- Cell and polygon extraction with API compatibility
- Structure name lookup with fallback to top-level cells
- Layer and datatype preservation during transformation
- New GDS file creation with transformed structures

Polygon Processing:
- Direct coordinate modification of polygon points
- Preservation of layer and datatype information
- Batch processing of all polygons in structure
- Error handling for malformed polygon data
- Memory-efficient processing for large structures

Coordinate System:
- GDS coordinate to pixel coordinate conversion
- Y-axis flipping for proper orientation (GDS up vs UI down)
- Center-based transformation calculations
- Scale factor computation from bounds and canvas size
- Precision handling for coordinate accuracy

API Compatibility:
- Multiple gdstk API version support
- Fallback methods for polygon extraction
- Mock polygon objects for compatibility
- Graceful handling of missing methods or attributes
- Error recovery for API differences

Transformation Pipeline:
1. Load GDS file and locate target structure
2. Extract all polygons with layer information
3. Calculate transformation parameters and center point
4. Apply transformations: translate to origin, scale, rotate, translate back, offset
5. Create new cell with transformed polygons
6. Save to new GDS file with preserved metadata

Error Handling:
- File existence validation
- Structure name verification with fallbacks
- Polygon extraction error recovery
- Transformation parameter validation
- GDS file writing error handling

Features:
- Non-destructive: Creates new files, preserves originals
- Precise: Maintains coordinate system accuracy
- Flexible: Supports arbitrary transformation parameters
- Compatible: Works with different gdstk API versions
- Robust: Comprehensive error handling and validation
"""

import gdstk
import numpy as np
import math
from pathlib import Path
from src.utils.transformations import (apply_polygon_transform, validate_transformation_parameters, convert_pixels_to_gds_units)

class GdsTransformationService:
    def transform_structure(self, original_gds_path: str, structure_name: str, 
                          transformation_params: dict, gds_bounds: tuple, 
                          canvas_size: tuple = (1024, 666)):
        """Transform GDS structure with specified parameters."""
        if not Path(original_gds_path).exists():
            raise FileNotFoundError(f"Original GDS file not found: {original_gds_path}")
        
        # Read GDS library
        library = gdstk.read_gds(original_gds_path)
        
        # Find the target cell
        original_cell = None
        for cell in library.cells:
            if cell.name == structure_name:
                original_cell = cell
                break
        
        if not original_cell:
            # Try to get top-level cells
            top_cells = library.top_level()
            if top_cells:
                original_cell = top_cells[0]
                if original_cell.name != structure_name:
                    print(f"Warning: Using cell '{original_cell.name}' instead of '{structure_name}'")
            else:
                raise ValueError(f"Structure '{structure_name}' not found in GDS file.")
        
        # Extract polygons from cell - handle gdstk API
        polygons = []
        
        # For gdstk, polygons are stored as Polygon objects in the cell
        cell_polygons = self._get_cell_polygons(original_cell)
        
        for polygon in cell_polygons:
            try:
                if hasattr(polygon, 'points') and hasattr(polygon, 'layer') and hasattr(polygon, 'datatype'):
                    # gdstk Polygon object
                    points = np.array(polygon.points)
                    layer = polygon.layer
                    datatype = polygon.datatype
                    polygons.append((points, layer, datatype))
                else:
                    # Handle other formats if needed
                    print(f"Warning: Unrecognized polygon format: {type(polygon)}")
            except Exception as e:
                print(f"Warning: Could not process polygon: {e}")
                continue
        
        if not polygons:
            raise ValueError(f"No polygons found in structure '{structure_name}'")
        
        # Calculate transformation parameters
        xmin, ymin, xmax, ymax = gds_bounds
        gds_width = xmax - xmin
        gds_height = ymax - ymin
        center_x_gds = xmin + gds_width / 2
        center_y_gds = ymin + gds_height / 2
        
        # Calculate scale factors
        scale_x = canvas_size[0] / gds_width
        scale_y = canvas_size[1] / gds_height
        initial_pixels_per_gds_unit = min(scale_x, scale_y)
        
        # Extract transformation parameters
        ui_scale = transformation_params.get('scale', 1.0)
        gds_x_offset = transformation_params.get('x_offset', 0) / initial_pixels_per_gds_unit
        gds_y_offset = transformation_params.get('y_offset', 0) / initial_pixels_per_gds_unit
        gds_y_offset = -gds_y_offset  # Flip Y coordinate
        rotation_angle_deg = transformation_params.get('rotation', 0.0)
        
        # Create new cell for transformed structure
        new_cell_name = f"{structure_name}_aligned"
        transformed_cell = gdstk.Cell(new_cell_name)
        
        # Transform each polygon
        for poly_points, layer, datatype in polygons:
            transformed_points = np.copy(poly_points)
            
            # Translate to origin (center)
            transformed_points -= [center_x_gds, center_y_gds]
            
            # Apply scaling if needed
            if abs(ui_scale - 1.0) > 1e-6:
                transformed_points *= ui_scale
            
            # Apply rotation if needed
            if abs(rotation_angle_deg) > 1e-6:
                angle_rad = math.radians(rotation_angle_deg)
                rotation_matrix = np.array([
                    [math.cos(angle_rad), -math.sin(angle_rad)],
                    [math.sin(angle_rad), math.cos(angle_rad)]
                ])
                transformed_points = transformed_points.dot(rotation_matrix.T)
            
            # Translate back to center
            transformed_points += [center_x_gds, center_y_gds]
            
            # Apply final offset
            transformed_points += [gds_x_offset, gds_y_offset]
            
            # Create new polygon with transformed points
            # Ensure points are in the correct format for gdstk
            points_list = [tuple(point) for point in transformed_points]
            polygon = gdstk.Polygon(points_list, layer=layer, datatype=datatype)
            transformed_cell.add(polygon)
        
        return transformed_cell
    
    def _get_cell_polygons(self, cell):
        """Extract polygons from a gdstk cell, handling different API versions."""
        polygons = []
        
        try:
            # Try direct access to polygons (newer gdstk)
            if hasattr(cell, 'polygons'):
                return cell.polygons
            
            # Try get_polygons method (gdspy-like API)
            elif hasattr(cell, 'get_polygons'):
                poly_dict = cell.get_polygons(by_spec=True)
                for (layer, datatype), poly_list in poly_dict.items():
                    for poly_points in poly_list:
                        # Create a mock polygon object for compatibility
                        class MockPolygon:
                            def __init__(self, points, layer, datatype):
                                self.points = points
                                self.layer = layer
                                self.datatype = datatype
                        
                        polygons.append(MockPolygon(poly_points, layer, datatype))
                return polygons
            
            # Try accessing through library elements
            elif hasattr(cell, 'references') and hasattr(cell, 'polygons'):
                return cell.polygons
            
            # Last resort: try to iterate through cell elements
            else:
                for element in getattr(cell, 'elements', []):
                    if hasattr(element, 'points'):
                        polygons.append(element)
                return polygons
                
        except Exception as e:
            print(f"Warning: Could not extract polygons from cell: {e}")
            return []
    
    def save_transformed_gds(self, transformed_cell, output_path: str):
        """Save transformed cell to a new GDS file."""
        try:
            # Create new library and add the transformed cell
            library = gdstk.Library()
            library.add(transformed_cell)
            
            # Write to file
            library.write_gds(output_path)
            print(f"Transformed GDS saved to: {output_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save transformed GDS: {e}")
    
    def validate_transformation_parameters(self, params: dict) -> bool:
        """Validate transformation parameters."""
        try:
            result = validate_transformation_parameters(
                translation=(params.get('x_offset', 0), params.get('y_offset', 0)),
                rotation_degrees=params.get('rotation', 0),
                scale=params.get('scale', 1.0)
            )
            return result['valid']
        except Exception as e:
            print(f"Validation error: {e}")
            return False
