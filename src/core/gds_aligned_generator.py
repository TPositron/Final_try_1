"""
GDS Aligned Generator - Core GDS Transformation Engine

This module is the core transformation engine that generates aligned GDS images by applying
transformation parameters (rotation, zoom, movement) to GDS structures. It provides the
low-level functionality for the alignment system.

Key Functions:
- generate_aligned_gds(): Main function that applies transformations to GDS structures
- generate_transformed_gds_image(): Core transformation logic with coherent parameter application
- generate_base_gds_image(): Legacy compatibility function for basic transformations
- apply_zoom_transform(): Applies zoom transformations around image center
- apply_move_transform(): Applies translation transformations
- rotate_polygon(): Rotates individual polygons around specified center points

Transformation Order (Critical):
1. Movement (translation)
2. Rotation (90-degree increments)
3. Zoom (scaling)

Dependencies:
- Uses: gdspy (GDS file reading), numpy, cv2 (image processing)
- Uses: gds_display_generator (structure info and bounds)
- Called by: services/transformation_service.py
- Called by: services/manual_alignment_service.py
- Called by: services/auto_alignment_service.py
- Called by: ui/alignment_controller.py (indirectly)

Data Flow:
1. Receives structure number and transformation parameters
2. Loads GDS file and extracts polygons for specified structure
3. Applies transformations in correct order to polygon coordinates
4. Renders transformed polygons to bitmap image
5. Returns aligned image for display/analysis

Critical Notes:
- Maintains exact UI behavior compatibility
- Handles rotation expansion for proper bounds calculation
- Uses coherent transformation application (not incremental)
- Supports both legacy and modern parameter formats
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
    GDS_PATH = "C:\\Users\\tarik\\Image_analysis\\Data\\GDS\\Institute_Project_GDS1.gds"
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
    """Generate GDS image with all transformations applied coherently."""
    if not os.path.exists(gds_path):
        raise FileNotFoundError(f"GDS file not found: {gds_path}")
    
    bounds = struct_info['bounds']
    layers = struct_info['layers']
    xmin, ymin, xmax, ymax = bounds
    gds_width = struct_info['gds_width']
    gds_height = struct_info['gds_height']
    
    # Extract transformation parameters
    rotation_degrees = transform_params.get('rotation', 0.0)
    zoom_percent = transform_params.get('zoom', 100.0)
    move_x = transform_params.get('move_x', 0.0)
    move_y = transform_params.get('move_y', 0.0)
    
    print(f"Generating transformed GDS: rotation={rotation_degrees}°, zoom={zoom_percent}%, move=({move_x}, {move_y})")
    
    # Calculate render bounds with rotation expansion if needed
    if abs(rotation_degrees) > 0.1:
        rotation_rad = math.radians(abs(rotation_degrees))
        cos_rot = abs(math.cos(rotation_rad))
        sin_rot = abs(math.sin(rotation_rad))
        expanded_width = gds_width * cos_rot + gds_height * sin_rot
        expanded_height = gds_width * sin_rot + gds_height * cos_rot
        
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        expanded_xmin = center_x - expanded_width / 2
        expanded_xmax = center_x + expanded_width / 2
        expanded_ymin = center_y - expanded_height / 2
        expanded_ymax = center_y + expanded_height / 2
        render_bounds = (expanded_xmin, expanded_ymin, expanded_xmax, expanded_ymax)
        render_width = expanded_width
        render_height = expanded_height
    else:
        render_bounds = bounds
        render_width = gds_width
        render_height = gds_height
    
    # Calculate pixel size for consistent coordinate conversion
    pixel_size = min(gds_width / target_size[0], gds_height / target_size[1])
    
    # Apply zoom to the scale calculation
    zoom_factor = zoom_percent / 100.0
    base_scale = min(target_size[0]/render_width, target_size[1]/render_height)
    scale = base_scale * zoom_factor
    
    # Calculate scaled dimensions
    scaled_width = int(render_width * scale)
    scaled_height = int(render_height * scale)
    
    # Create output image
    image = np.ones((target_size[1], target_size[0]), dtype=np.uint8) * 255
    
    # Convert pixel movement to GDS units (consistent with AlignedGdsModel)
    dx_gds = move_x * pixel_size
    dy_gds = -move_y * pixel_size  # Y flipped
    
    # Calculate center position with direct movement
    center_x_pixels = target_size[0] // 2 + move_x
    center_y_pixels = target_size[1] // 2 + move_y
    
    # Calculate offset to center the scaled image at the desired position
    offset_x = int(center_x_pixels - scaled_width // 2)
    offset_y = int(center_y_pixels - scaled_height // 2)
    
    print(f"Scale: {scale:.6f}, Scaled size: {scaled_width} x {scaled_height}")
    print(f"Center position: ({center_x_pixels}, {center_y_pixels}), Offset: ({offset_x}, {offset_y})")
    
    # Load GDS and get polygons
    gds = gdspy.GdsLibrary().read_gds(gds_path)
    cell = gds.top_level()[0] if cell_name == "TOP" else gds.cells[cell_name]
    polygons = cell.get_polygons(by_spec=True)
    
    render_xmin, render_ymin, render_xmax, render_ymax = render_bounds
    center_x_gds = (render_xmin + render_xmax) / 2
    center_y_gds = (render_ymin + render_ymax) / 2
    rotation_rad = math.radians(rotation_degrees)
    
    polygon_count = 0
    for layer in layers:
        if (layer, 0) in polygons:
            for poly in polygons[(layer, 0)]:
                # Check if polygon is within original bounds
                poly_xmin, poly_ymin = np.min(poly, axis=0)
                poly_xmax, poly_ymax = np.max(poly, axis=0)
                orig_xmin, orig_ymin, orig_xmax, orig_ymax = bounds
                
                if (poly_xmax >= orig_xmin and poly_xmin <= orig_xmax and 
                    poly_ymax >= orig_ymin and poly_ymin <= orig_ymax):
                    polygon_count += 1
                    
                    # Apply rotation if needed
                    if abs(rotation_degrees) > 0.1:
                        rotated_poly = rotate_polygon(poly, center_x_gds, center_y_gds, rotation_rad)
                    else:
                        rotated_poly = poly
                    
                    # Transform to pixel coordinates with consistent offset
                    norm_poly = (rotated_poly - [render_xmin, render_ymin]) * scale
                    int_poly = np.round(norm_poly).astype(np.int32)
                    int_poly += [offset_x, offset_y]
                    
                    # Flip Y coordinate (image coordinate system)
                    int_poly[:, 1] = target_size[1] - 1 - int_poly[:, 1]
                    
                    # Clip to image bounds
                    int_poly = np.clip(int_poly, [0, 0], [target_size[0]-1, target_size[1]-1])
                    
                    # Draw polygon with corrected fillPoly call
                    if len(int_poly) >= 3:
                        cv2.fillPoly(image, [int_poly], color=(0,))  # Use tuple for scalar color
    
    print(f"Rendered {polygon_count} polygons")
    return image

def generate_base_gds_image(gds_path: str, cell_name: str, struct_info: Dict, target_size: Tuple[int, int], rotation_degrees: float) -> np.ndarray:
    """Legacy function - kept for backward compatibility but improved."""
    transform_params = {'rotation': rotation_degrees, 'zoom': 100.0, 'move_x': 0.0, 'move_y': 0.0}
    return generate_transformed_gds_image(gds_path, cell_name, struct_info, target_size, transform_params)

def apply_zoom_transform(image: np.ndarray, zoom_percent: float) -> np.ndarray:
    """Apply zoom transformation around the center of the image."""
    if zoom_percent <= 0:
        raise ValueError("Zoom percentage must be > 0")
    
    h, w = image.shape[:2]
    scale = zoom_percent / 100.0
    
    # Create transformation matrix for scaling around center
    center_x, center_y = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
    
    # Apply transformation with corrected parameters
    return cv2.warpAffine(
        image, 
        M, 
        (w, h), 
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(255,)  # Use tuple for scalar border value
    )

def apply_move_transform(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Apply translation transformation."""
    h, w = image.shape[:2]
    # Create translation matrix - note: dy is negated because image Y is flipped
    M = np.array([[1, 0, dx], [0, 1, -dy]], dtype=np.float32)
    
    # Apply transformation with corrected parameters
    return cv2.warpAffine(
        image, 
        M, 
        (w, h), 
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(255,)  # Use tuple for scalar border value
    )

def rotate_polygon(polygon: np.ndarray, center_x: float, center_y: float, angle_rad: float) -> np.ndarray:
    """Rotate polygon around given center point."""
    translated = polygon - [center_x, center_y]
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    rotated = np.zeros_like(translated)
    rotated[:, 0] = translated[:, 0] * cos_a - translated[:, 1] * sin_a
    rotated[:, 1] = translated[:, 0] * sin_a + translated[:, 1] * cos_a
    
    return rotated + [center_x, center_y]

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
