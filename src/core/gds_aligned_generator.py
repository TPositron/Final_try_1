"""
GDS Aligned Generator
Generates aligned GDS images based on transformation parameters.
This matches the exact behavior of the UI transforms.
"""

import gdspy
import numpy as np
import cv2
import os
import math
import tempfile
from typing import Dict, List, Tuple, Optional
from .gds_display_generator import get_structure_info


def generate_aligned_gds(structure_num: int, transform_params: Dict, target_size: Tuple[int, int] = (1024, 666)) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Generate aligned GDS image based on transformation parameters.
    This now matches the exact behavior of the UI transforms.
    
    Args:
        structure_num: Structure number (1-5)
        transform_params: Dictionary with transformation parameters
        target_size: Output image size
        
    Returns:
        Tuple of (aligned_image, bounds)
    """
    GDS_PATH = "C:\\Users\\tarik\\Image_analysis\\Data\\GDS\\Institute_Project_GDS1.gds"
    CELL_NAME = "TOP"
    
    # Get structure information
    struct_info = get_structure_info(structure_num)
    if not struct_info:
        raise ValueError(f"Structure {structure_num} not found")
    
    print(f"Starting aligned GDS generation for Structure {structure_num}")
    print(f"Transform params: {transform_params}")
    
    # Step 1: Generate the base GDS image (same as display)
    base_gds = generate_base_gds_image(GDS_PATH, CELL_NAME, struct_info, target_size, 
                                       transform_params.get('rotation', 0))
    
    # Step 2: Apply zoom transformation (same as zoom_image function)
    if transform_params.get('zoom', 100) != 100:
        print(f"Applying zoom: {transform_params['zoom']}%")
        base_gds = apply_zoom_transform(base_gds, transform_params['zoom'])
        
    # Step 3: Apply move transformation (same as move_image function)
    if transform_params.get('move_x', 0) != 0 or transform_params.get('move_y', 0) != 0:
        print(f"Applying move: X={transform_params.get('move_x', 0)}, Y={transform_params.get('move_y', 0)}")
        base_gds = apply_move_transform(base_gds, transform_params.get('move_x', 0), transform_params.get('move_y', 0))
    
    return base_gds, struct_info['bounds']


def generate_base_gds_image(gds_path: str, cell_name: str, struct_info: Dict, target_size: Tuple[int, int], rotation_degrees: float) -> np.ndarray:
    """
    Generate the base GDS image with rotation, matching the display generator exactly.
    
    Args:
        gds_path: Path to GDS file
        cell_name: Name of the cell to use
        struct_info: Structure information dictionary
        target_size: Target image size
        rotation_degrees: Rotation angle in degrees
        
    Returns:
        Generated GDS image
    """
    if not os.path.exists(gds_path):
        raise FileNotFoundError(f"GDS file not found: {gds_path}")
    
    bounds = struct_info['bounds']
    layers = struct_info['layers']
    xmin, ymin, xmax, ymax = bounds
    gds_width = struct_info['gds_width']
    gds_height = struct_info['gds_height']
    
    print(f"Generating base GDS image: {gds_width:.6f} x {gds_height:.6f}, rotation={rotation_degrees}°")
    
    # If we have rotation, we need to expand the rendering area
    if abs(rotation_degrees) > 0.1:
        rotation_rad = math.radians(abs(rotation_degrees))
        cos_rot = abs(math.cos(rotation_rad))
        sin_rot = abs(math.sin(rotation_rad))
        
        # Expand dimensions to contain rotated content
        expanded_width = gds_width * cos_rot + gds_height * sin_rot
        expanded_height = gds_width * sin_rot + gds_height * cos_rot
        
        # Expand bounds symmetrically around center
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        
        expanded_xmin = center_x - expanded_width / 2
        expanded_xmax = center_x + expanded_width / 2
        expanded_ymin = center_y - expanded_height / 2
        expanded_ymax = center_y + expanded_height / 2
        
        render_bounds = (expanded_xmin, expanded_ymin, expanded_xmax, expanded_ymax)
        render_width = expanded_width
        render_height = expanded_height
        
        print(f"Expanded for rotation: {expanded_width:.6f} x {expanded_height:.6f}")
    else:
        render_bounds = bounds
        render_width = gds_width
        render_height = gds_height
    
    # Calculate scaling to fit in target size
    scale = min(target_size[0]/render_width, target_size[1]/render_height)
    scaled_width = int(render_width * scale)
    scaled_height = int(render_height * scale)
    
    # Create centered image
    image = np.ones((target_size[1], target_size[0]), dtype=np.uint8) * 255
    offset_x = (target_size[0] - scaled_width) // 2
    offset_y = (target_size[1] - scaled_height) // 2
    
    print(f"Scale: {scale:.6f}, Scaled size: {scaled_width} x {scaled_height}, Offset: ({offset_x}, {offset_y})")
    
    # Load GDS and get polygons
    gds = gdspy.GdsLibrary().read_gds(gds_path)
    cell = gds.top_level()[0] if cell_name == "TOP" else gds.cells[cell_name]
    polygons = cell.get_polygons(by_spec=True)
    
    # Calculate rotation parameters
    render_xmin, render_ymin, render_xmax, render_ymax = render_bounds
    center_x_gds = (render_xmin + render_xmax) / 2
    center_y_gds = (render_ymin + render_ymax) / 2
    rotation_rad = math.radians(rotation_degrees)
    
    polygon_count = 0
    
    # Draw all layers
    for layer in layers:
        if (layer, 0) in polygons:
            for poly in polygons[(layer, 0)]:
                # Check if polygon might be visible in our render area
                poly_xmin, poly_ymin = np.min(poly, axis=0)
                poly_xmax, poly_ymax = np.max(poly, axis=0)
                
                # Use original bounds for visibility check (before rotation expansion)
                orig_xmin, orig_ymin, orig_xmax, orig_ymax = bounds
                if (poly_xmax >= orig_xmin and poly_xmin <= orig_xmax and 
                    poly_ymax >= orig_ymin and poly_ymin <= orig_ymax):
                    
                    polygon_count += 1
                    
                    # Apply rotation to polygon vertices if needed
                    if abs(rotation_degrees) > 0.1:
                        rotated_poly = rotate_polygon(poly, center_x_gds, center_y_gds, rotation_rad)
                    else:
                        rotated_poly = poly
                    
                    # Transform to image coordinates - MATCH THE DISPLAY GENERATOR
                    # 1. Normalize coordinates (translate so render area min becomes 0,0)
                    norm_poly = (rotated_poly - [render_xmin, render_ymin]) * scale
                    
                    # 2. Convert to integer coordinates
                    int_poly = np.round(norm_poly).astype(np.int32)
                    
                    # 3. Apply offset for centering
                    int_poly += [offset_x, offset_y]
                    
                    # 4. IMPORTANT: Flip Y-axis to match display generator
                    int_poly[:, 1] = target_size[1] - 1 - int_poly[:, 1]
                    
                    # 5. Clip to image bounds to avoid OpenCV errors
                    int_poly = np.clip(int_poly, [0, 0], [target_size[0]-1, target_size[1]-1])
                    
                    # Only draw if polygon has area
                    if len(int_poly) >= 3:
                        cv2.fillPoly(image, [int_poly], color=0)
    
    print(f"Rendered {polygon_count} polygons")
    return image


def apply_zoom_transform(image: np.ndarray, zoom_percent: float) -> np.ndarray:
    """
    Apply zoom transformation exactly like zoom_image function.
    
    Args:
        image: Input image
        zoom_percent: Zoom percentage (100 = no change)
        
    Returns:
        Zoomed image
    """
    if zoom_percent <= 0:
        raise ValueError("Zoom percentage must be > 0")
    
    h, w = image.shape[:2]
    scale = zoom_percent / 100.0
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def apply_move_transform(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Apply move transformation exactly like move_image function.
    
    Args:
        image: Input image
        dx: X displacement in pixels
        dy: Y displacement in pixels
        
    Returns:
        Moved image
    """
    h, w = image.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def rotate_polygon(polygon: np.ndarray, center_x: float, center_y: float, angle_rad: float) -> np.ndarray:
    """
    Rotate polygon around a center point by given angle in radians.
    
    Args:
        polygon: Polygon vertices as numpy array
        center_x: X coordinate of rotation center
        center_y: Y coordinate of rotation center
        angle_rad: Rotation angle in radians
        
    Returns:
        Rotated polygon vertices
    """
    # Translate to origin
    translated = polygon - [center_x, center_y]
    
    # Apply rotation matrix
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    rotated = np.zeros_like(translated)
    rotated[:, 0] = translated[:, 0] * cos_a - translated[:, 1] * sin_a
    rotated[:, 1] = translated[:, 0] * sin_a + translated[:, 1] * cos_a
    
    # Translate back
    return rotated + [center_x, center_y]


def generate_transformed_gds(structure_num: int, 
                           rotation: float = 0,
                           zoom: float = 100,
                           move_x: float = 0,
                           move_y: float = 0,
                           target_size: Tuple[int, int] = (1024, 666)) -> np.ndarray:
    """
    Convenience function to generate transformed GDS image.
    
    Args:
        structure_num: Structure number (1-5)
        rotation: Rotation angle in degrees
        zoom: Zoom percentage (100 = no change)
        move_x: X movement in pixels
        move_y: Y movement in pixels
        target_size: Output image size
        
    Returns:
        Transformed GDS image
    """
    transform_params = {
        'rotation': rotation,
        'zoom': zoom,
        'move_x': move_x,
        'move_y': move_y
    }
    
    image, bounds = generate_aligned_gds(structure_num, transform_params, target_size)
    return image
