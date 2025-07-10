"""
GDS Aligned Generator - Core GDS Transformation Engine

This module generates aligned GDS images using a bounds-based approach that:
1. Uses initial bounds from gds_display_generator.py
2. Calculates new bounds based on UI transformation parameters (pixel-based)
3. Extracts fresh GDS data from new bounds
4. Applies rotation to final image (not polygons)

Key Functions:
- generate_aligned_gds(): Main function for generating aligned GDS images
- calculate_new_bounds(): Calculate extraction bounds from UI parameters
- extract_and_render_gds(): Extract GDS data from bounds and render to image
- apply_image_rotation(): Apply rotation to final image

Dependencies:
- Uses: gdspy (GDS file reading), numpy, cv2 (image processing)
- Uses: gds_display_generator (structure info and GDS path)
"""

import gdspy
import numpy as np
import cv2
import math
from typing import Dict, List, Tuple, Optional
from src.core.gds_display_generator import get_structure_info, get_project_gds_path


def generate_aligned_gds(structure_num: int, transform_params: Dict, 
                        target_size: Tuple[int, int] = (1024, 666)) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Generate aligned GDS image using bounds-based transformation.
    
    Args:
        structure_num: Structure ID (1-5)
        transform_params: Dictionary with 'rotation', 'zoom', 'move_x', 'move_y'
        target_size: Output image size (width, height)
        
    Returns:
        Tuple of (aligned_image, original_bounds)
    """
    # Get structure info from gds_display_generator
    struct_info = get_structure_info(structure_num)
    if not struct_info:
        raise ValueError(f"Structure {structure_num} not found")
    
    print(f"Generating aligned GDS for Structure {structure_num}")
    print(f"Transform params: {transform_params}")
    
    # Extract transformation parameters
    rotation = transform_params.get('rotation', 0.0)
    zoom = transform_params.get('zoom', 100.0)
    move_x = transform_params.get('move_x', 0.0)  # pixels
    move_y = transform_params.get('move_y', 0.0)  # pixels
    
    # Calculate new bounds based on transformation parameters (without rotation)
    original_bounds = struct_info['bounds']
    new_bounds = calculate_new_bounds(original_bounds, zoom, move_x, move_y, struct_info)
    
    # Extract and render GDS data from new bounds
    image = extract_and_render_gds(
        structure_num=structure_num,
        bounds=new_bounds,
        layers=struct_info['layers'],
        target_size=target_size
    )
    
    # Apply rotation around the moved center point
    if abs(rotation) > 0.1:
        image = apply_image_rotation(image, rotation, move_x, move_y)
    
    return image, original_bounds


def calculate_new_bounds(original_bounds: Tuple[float, float, float, float], 
                        zoom: float, move_x: float, move_y: float, 
                        struct_info: Dict) -> Tuple[float, float, float, float]:
    """
    Calculate new extraction bounds to match UI transformation behavior.
    
    Args:
        original_bounds: Original structure bounds (xmin, ymin, xmax, ymax)
        zoom: Zoom percentage (100 = no zoom, 200 = 2x zoom in, 50 = 2x zoom out)
        move_x: Movement in pixels (positive = right, matches UI)
        move_y: Movement in pixels (positive = down, matches UI)
        struct_info: Structure information dictionary
        
    Returns:
        New bounds (xmin, ymin, xmax, ymax) for GDS extraction
    """
    xmin, ymin, xmax, ymax = original_bounds
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    
    # CRITICAL FIX: Calculate movement AFTER zoom, using the actual bounds that will be rendered
    # First calculate the zoomed bounds dimensions
    zoom_factor = zoom / 100.0
    new_width = width / zoom_factor
    new_height = height / zoom_factor
    
    print(f"  Zoom calculation: {zoom}% -> factor={zoom_factor:.3f}, width={width:.3f}->{new_width:.3f}, height={height:.3f}->{new_height:.3f}")
    
    # Use the scale that will actually be used in rendering (based on new bounds)
    x_scale = 1024 / new_width
    y_scale = 666 / new_height
    actual_render_scale = min(x_scale, y_scale)
    
    print(f"  Scale components: x_scale={x_scale:.6f}, y_scale={y_scale:.6f}, limiting_scale={actual_render_scale:.6f}")
    print(f"  Aspect ratio: structure={new_width/new_height:.3f}, target={1024/666:.3f}")
    
    # Convert UI pixel movement to GDS units using the actual render scale
    dx_gds = move_x / actual_render_scale
    dy_gds = move_y / actual_render_scale
    
    print(f"  Scale calculation: new_width={new_width:.3f}, new_height={new_height:.3f}, actual_render_scale={actual_render_scale:.6f}")
    print(f"  Pixel to GDS: move_x={move_x} -> dx_gds={dx_gds:.6f}, move_y={move_y} -> dy_gds={dy_gds:.6f}")
    
    # Apply translation to center - move extraction area opposite to UI movement
    # If UI moves overlay right, we extract from left to compensate
    new_center_x = center_x - dx_gds  # Opposite direction
    new_center_y = center_y + dy_gds  # Same direction for Y
    
    # Calculate new bounds - zoom from center, move extraction area
    new_bounds = (
        new_center_x - new_width / 2,   # xmin
        new_center_y - new_height / 2,  # ymin  
        new_center_x + new_width / 2,   # xmax
        new_center_y + new_height / 2   # ymax
    )
    
    print(f"  Final bounds: center=({new_center_x:.3f}, {new_center_y:.3f}), size=({new_width:.3f}, {new_height:.3f})")
    print(f"  Bounds: ({new_bounds[0]:.3f}, {new_bounds[1]:.3f}, {new_bounds[2]:.3f}, {new_bounds[3]:.3f})")
    
    print(f"Bounds transformation: {original_bounds} -> {new_bounds}")
    print(f"  Zoom: {zoom}%, Move: ({move_x}, {move_y}) px, GDS move: ({dx_gds:.3f}, {dy_gds:.3f})")
    print(f"  Extraction center moved opposite: ({-dx_gds:.3f}, {-dy_gds:.3f})")
    
    return new_bounds


def extract_and_render_gds(structure_num: int, bounds: Tuple[float, float, float, float], 
                          layers: List[int], target_size: Tuple[int, int]) -> np.ndarray:
    """
    Extract GDS polygons from specified bounds and render to image.
    
    Args:
        structure_num: Structure ID (for logging)
        bounds: Extraction bounds (xmin, ymin, xmax, ymax)
        layers: Layer numbers to extract
        target_size: Output image size (width, height)
        
    Returns:
        Rendered binary image (0=structure, 255=background)
    """
    gds_path = get_project_gds_path()
    
    # Load GDS file
    gds = gdspy.GdsLibrary().read_gds(gds_path)
    cell = gds.top_level()[0]  # Use top-level cell
    polygons = cell.get_polygons(by_spec=True)
    
    # Calculate rendering parameters
    xmin, ymin, xmax, ymax = bounds
    gds_width = xmax - xmin
    gds_height = ymax - ymin
    
    if gds_width <= 0 or gds_height <= 0:
        print(f"Warning: Invalid bounds dimensions: {gds_width} x {gds_height}")
        return np.ones(target_size[::-1], dtype=np.uint8) * 255
    
    # Calculate scale to fit in target size
    scale = min(target_size[0] / gds_width, target_size[1] / gds_height)
    print(f"  Render scale in extract_and_render_gds: {scale:.6f} (bounds: {gds_width:.3f}x{gds_height:.3f})")
    print(f"  Render scale components: x_render={target_size[0] / gds_width:.6f}, y_render={target_size[1] / gds_height:.6f}")
    print(f"  Scale comparison: bounds_calc_x={1024/gds_width:.6f}, bounds_calc_y={666/gds_height:.6f}, render_used={scale:.6f}")
    scaled_width = int(gds_width * scale)
    scaled_height = int(gds_height * scale)
    
    # Center the scaled image - use float precision
    offset_x = (target_size[0] - scaled_width) / 2
    offset_y = (target_size[1] - scaled_height) / 2
    
    # Create white background image
    image = np.ones((target_size[1], target_size[0]), dtype=np.uint8) * 255
    
    # Render polygons from specified layers
    polygon_count = 0
    for layer in layers:
        if (layer, 0) in polygons:
            for poly in polygons[(layer, 0)]:
                # Check if polygon intersects with extraction bounds
                poly_xmin, poly_ymin = np.min(poly, axis=0)
                poly_xmax, poly_ymax = np.max(poly, axis=0)
                
                if (poly_xmax >= xmin and poly_xmin <= xmax and 
                    poly_ymax >= ymin and poly_ymin <= ymax):
                    polygon_count += 1
                    
                    # Transform polygon to image coordinates - preserve precision
                    norm_poly = (poly - [xmin, ymin]) * scale
                    float_poly = norm_poly + [offset_x, offset_y]
                    int_poly = np.round(float_poly).astype(np.int32)
                    
                    # Flip Y-axis for image coordinates
                    int_poly[:, 1] = target_size[1] - 1 - int_poly[:, 1]
                    
                    # Clip to image bounds
                    int_poly = np.clip(int_poly, [0, 0], [target_size[0]-1, target_size[1]-1])
                    
                    # Render polygon - black structure on white background
                    if len(int_poly) >= 3:
                        cv2.fillPoly(image, [int_poly], color=(0,))  # Black structure
    
    print(f"Rendered {polygon_count} polygons from layers {layers} in bounds {bounds}")
    return image


def apply_image_rotation(image: np.ndarray, angle: float, move_x: float = 0, move_y: float = 0) -> np.ndarray:
    """
    Apply rotation to the final image around the moved center point.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees (positive = counterclockwise)
        move_x: X movement in pixels
        move_y: Y movement in pixels
        
    Returns:
        Rotated image with same dimensions
    """
    if abs(angle) < 0.1:
        return image
    
    height, width = image.shape[:2]
    # Calculate rotation center as image center plus movement
    center_x = width / 2 + move_x
    center_y = height / 2 + move_y
    center = (center_x, center_y)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation with white background
    rotated = cv2.warpAffine(
        image, 
        rotation_matrix, 
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255  # White background
    )
    
    print(f"Applied rotation: {angle:.1f}° around ({center_x:.1f}, {center_y:.1f})")
    return rotated


# Convenience function for backward compatibility
def generate_transformed_gds(structure_num: int, rotation: float = 0, zoom: float = 100, 
                           move_x: float = 0, move_y: float = 0, 
                           target_size: Tuple[int, int] = (1024, 666)) -> np.ndarray:
    """
    Generate transformed GDS with specified parameters (convenience function).
    
    Args:
        structure_num: Structure ID (1-5)
        rotation: Rotation angle in degrees
        zoom: Zoom percentage (100 = no zoom)
        move_x: Movement in pixels (positive = right)
        move_y: Movement in pixels (positive = down)
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
    
    image, _ = generate_aligned_gds(structure_num, transform_params, target_size)
    return image
