"""
GDS Aligned Generator - Core GDS Transformation Engine

This module generates aligned GDS images using a new bounds-based approach that separates
90-degree rotations (coordinate-level) from fine rotations (image-level) for optimal
quality and performance.

Key Functions:
- generate_aligned_gds(): Main function that applies transformations to GDS structures
- generate_transformed_gds_image(): New bounds-based transformation logic
- _extract_and_render_gds(): Extracts GDS data from calculated bounds
- _apply_fine_rotation(): Applies non-90° rotations to final image

New Transformation Approach:
1. Calculate new bounds (includes 90° rotations, translation, zoom)
2. Extract GDS data from new bounds only
3. Render to image
4. Apply fine rotation (non-90° angles) to final image

Dependencies:
- Uses: gdspy (GDS file reading), numpy, cv2 (image processing)
- Uses: gds_display_generator (bounds calculation and structure info)
- Called by: services/transformation_service.py
- Called by: services/manual_alignment_service.py
- Called by: services/auto_alignment_service.py

Advantages:
- Clean 90° rotations (coordinate-level, no quality loss)
- Simple fine rotations (image-level)
- Single GDS extraction per transformation
- Consistent and predictable results
"""
import gdspy
import numpy as np
import cv2
import os
import math
import tempfile
from typing import Dict, List, Tuple, Optional
from src.core.gds_display_generator import get_structure_info

def generate_aligned_gds(structure_num: int, transform_params: Dict, target_size: Tuple[int, int] = (1024, 666)) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    
    def get_project_gds_path():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up to find the Image_analysis project root
        while os.path.basename(current_dir) != 'Image_analysis' and current_dir != os.path.dirname(current_dir):
            current_dir = os.path.dirname(current_dir)
        return os.path.join(current_dir, "Data", "GDS", "Institute_Project_GDS1.gds")

    GDS_PATH = get_project_gds_path()
    CELL_NAME = "TOP"
    
    struct_info = get_structure_info(structure_num)
    if not struct_info:
        raise ValueError(f"Structure {structure_num} not found")
    
    print(f"Starting aligned GDS generation for Structure {structure_num}")
    print(f"Transform params: {transform_params}")
    
    # Generate base GDS image without individual transformations
    # Apply all transformations in one coherent step
    base_gds = generate_transformed_gds_image(GDS_PATH, CELL_NAME, struct_info, target_size, transform_params)
    
    return base_gds, struct_info['bounds']

def generate_transformed_gds_image(gds_path: str, cell_name: str, struct_info: Dict, target_size: Tuple[int, int], transform_params: Dict) -> np.ndarray:
    """Generate GDS image with frame size adjustment and final resize."""
    from src.core.gds_display_generator import calculate_transformed_bounds
    
    bounds = struct_info['bounds']
    layers = struct_info['layers']
    pixel_size = min(struct_info['gds_width'] / target_size[0], struct_info['gds_height'] / target_size[1])
    
    rotation = transform_params.get('rotation', 0.0)
    zoom = transform_params.get('zoom', 100.0)
    move_x = transform_params.get('move_x', 0.0)
    move_y = transform_params.get('move_y', 0.0)
    
    # 1. Calculate new bounds (with dimension swapping for rotations)
    new_bounds, needs_fine_rotation = calculate_transformed_bounds(
        bounds, rotation, zoom, move_x, move_y, pixel_size
    )
    
    # 2. Extract fresh polygons from new bounds - always use 1024x666
    image = _extract_and_render_gds(gds_path, cell_name, layers, new_bounds, target_size)
    
    # 3. Apply rotation to final image (if rotation != 0)
    if needs_fine_rotation:
        image = _apply_fine_rotation(image, rotation)
    
    return image

def _extract_and_render_gds(gds_path: str, cell_name: str, layers: List[int], bounds: Tuple[float, float, float, float], target_size: Tuple[int, int]) -> np.ndarray:
    """Extract GDS polygons from bounds and render to image (no transformations)."""
    # Load GDS file
    gds = gdspy.GdsLibrary().read_gds(gds_path)
    cell = gds.top_level()[0] if cell_name == "TOP" else gds.cells[cell_name]
    polygons = cell.get_polygons(by_spec=True)
    
    xmin, ymin, xmax, ymax = bounds
    gds_width = xmax - xmin
    gds_height = ymax - ymin
    scale = min(target_size[0]/gds_width, target_size[1]/gds_height)
    
    scaled_width = int(gds_width * scale)
    scaled_height = int(gds_height * scale)
    offset_x = (target_size[0] - scaled_width) // 2
    offset_y = (target_size[1] - scaled_height) // 2
    
    image = np.ones((target_size[1], target_size[0]), dtype=np.uint8) * 255
    
    polygon_count = 0
    for layer in layers:
        if (layer, 0) in polygons:
            for poly in polygons[(layer, 0)]:
                # Extract ONLY polygons that intersect with bounds
                poly_xmin, poly_ymin = np.min(poly, axis=0)
                poly_xmax, poly_ymax = np.max(poly, axis=0)
                
                if (poly_xmax >= xmin and poly_xmin <= xmax and 
                    poly_ymax >= ymin and poly_ymin <= ymax):
                    polygon_count += 1
                    
                    # Render polygons to image (no transformations)
                    norm_poly = (poly - [xmin, ymin]) * scale
                    int_poly = np.round(norm_poly).astype(np.int32)
                    int_poly += [offset_x, offset_y]
                    int_poly[:, 1] = target_size[1] - 1 - int_poly[:, 1]
                    
                    if len(int_poly) >= 3:
                        cv2.fillPoly(image, [int_poly], color=(0,))
    
    return image

def _apply_fine_rotation(image: np.ndarray, angle: float) -> np.ndarray:
    """Apply rotation to final image - handles ALL rotations at image level."""
    if abs(angle) < 0.1:
        return image
    
    h, w = image.shape[:2]
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, 
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255,))

def generate_transformed_gds(structure_num: int, rotation: float = 0, zoom: float = 100, move_x: float = 0, move_y: float = 0, target_size: Tuple[int, int] = (1024, 666)) -> np.ndarray:
    """Generate transformed GDS with specified parameters."""
    transform_params = {
        'rotation': rotation,
        'zoom': zoom, 
        'move_x': move_x,
        'move_y': move_y
    }
    
    image, bounds = generate_aligned_gds(structure_num, transform_params, target_size)
    return image
