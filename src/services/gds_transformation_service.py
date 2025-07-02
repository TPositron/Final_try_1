
import gdstk
import numpy as np
import math
from pathlib import Path
from src.utils.transformations import (
    apply_polygon_transform,
    validate_transformation_parameters,
    convert_pixels_to_gds_units
)

class GdsTransformationService:
    """
    A service to apply geometric transformations (translate, rotate, scale)
    to the coordinates of a GDS structure using gdstk.
    """

    def transform_structure(
        self,
        original_gds_path: str,
        structure_name: str,
        transformation_params: dict,
        gds_bounds: tuple,
        canvas_size: tuple = (1024, 666)
    ):
        """
        Applies UI transformations to the actual GDS coordinates.

        Args:
            original_gds_path (str): The path to the source GDS file.
            structure_name (str): The name of the GDS cell/structure to transform.
            transformation_params (dict): A dictionary with 'x_offset', 'y_offset', 'rotation', 'scale'.
            gds_bounds (tuple): The original (xmin, ymin, xmax, ymax) of the GDS structure.
            canvas_size (tuple): The (width, height) of the UI canvas where alignment is done.

        Returns:
            A new gdstk.Cell containing the transformed polygons.
        """
        if not Path(original_gds_path).exists():
            raise FileNotFoundError(f"Original GDS file not found: {original_gds_path}")

        # 1. Load original GDS and find the cell using gdstk
        library = gdstk.read_gds(original_gds_path)
        original_cell = None
        
        # Find the cell by name
        for cell in library.cells:
            if cell.name == structure_name:
                original_cell = cell
                break
        
        if not original_cell:
            # Fallback to top-level cells
            top_cells = library.top_level()
            if top_cells:
                original_cell = top_cells[0]
                if original_cell.name != structure_name:
                    print(f"Warning: Using cell '{original_cell.name}' instead of '{structure_name}'")
            else:
                raise ValueError(f"Structure '{structure_name}' not found in GDS file.")

        # Get all polygons from the cell
        polygons = []
        for polygon in original_cell.polygons:
            polygons.append((polygon.points, polygon.layer, polygon.datatype))
        
        # 2. Calculate transformation parameters from UI space to GDS space
        xmin, ymin, xmax, ymax = gds_bounds
        gds_width = xmax - xmin
        gds_height = ymax - ymin
        
        # This is the center of the GDS structure in its own coordinate system
        center_x_gds = xmin + gds_width / 2
        center_y_gds = ymin + gds_height / 2

        # Calculate the initial scale factor used to fit the GDS onto the canvas
        # This is the key to converting pixel offsets back to GDS units
        scale_x = canvas_size[0] / gds_width
        scale_y = canvas_size[1] / gds_height
        initial_pixels_per_gds_unit = min(scale_x, scale_y)

        # Convert UI pixel offsets to GDS coordinate offsets
        # We must also account for the user-applied zoom (scale)
        ui_scale = transformation_params.get('scale', 1.0)
        gds_x_offset = transformation_params.get('x_offset', 0) / initial_pixels_per_gds_unit
        gds_y_offset = transformation_params.get('y_offset', 0) / initial_pixels_per_gds_unit
        
        # The UI flips the Y-axis for display, so we must flip the translation
        gds_y_offset = -gds_y_offset

        rotation_angle_deg = transformation_params.get('rotation', 0.0)

        # 3. Create a new cell for the transformed polygons
        new_cell_name = f"{structure_name}_aligned"
        transformed_cell = gdstk.Cell(new_cell_name)

        # 4. Apply transformations to each polygon
        for poly_points, layer, datatype in polygons:
            # Create a copy to modify
            transformed_points = np.copy(poly_points)

            # a. Translate to origin for scale/rotation
            transformed_points -= [center_x_gds, center_y_gds]
            
            # b. Apply scaling (zoom)
            if abs(ui_scale - 1.0) > 1e-6:
                transformed_points *= ui_scale
            
            # c. Apply rotation
            if abs(rotation_angle_deg) > 1e-6:
                angle_rad = math.radians(rotation_angle_deg)
                rotation_matrix = np.array([
                    [math.cos(angle_rad), -math.sin(angle_rad)],
                    [math.sin(angle_rad), math.cos(angle_rad)]
                ])
                transformed_points = transformed_points.dot(rotation_matrix.T)

            # d. Translate back from origin
            transformed_points += [center_x_gds, center_y_gds]
            
            # e. Apply final translation (move)
            transformed_points += [gds_x_offset, gds_y_offset]

            # Add the fully transformed polygon to the new cell using gdstk
            polygon = gdstk.Polygon(transformed_points, layer=layer, datatype=datatype)
            transformed_cell.add(polygon)

        return transformed_cell
