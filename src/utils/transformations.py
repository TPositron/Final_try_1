"""
Transformation Utilities

Pure mathematical transformation functions for coordinate and matrix operations.
This module provides reusable transformation utilities that can be used by
different services and models without coupling to specific implementations.
"""

import numpy as np
import math
from typing import Tuple, List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def create_transformation_matrix(
    translation: Tuple[float, float] = (0.0, 0.0),
    rotation_degrees: float = 0.0,
    scale: float = 1.0,
    center: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Create a 3x3 homogeneous transformation matrix.
    
    Args:
        translation: (tx, ty) translation offsets
        rotation_degrees: Rotation angle in degrees (counterclockwise)
        scale: Uniform scaling factor
        center: Optional rotation/scale center point. If None, uses origin.
        
    Returns:
        3x3 transformation matrix for homogeneous coordinates
    """
    # Start with identity matrix
    matrix = np.eye(3, dtype=np.float64)
    
    # If we have a center point, translate to origin first
    if center is not None:
        cx, cy = center
        # Translate to origin
        translate_to_origin = np.array([
            [1, 0, -cx],
            [0, 1, -cy], 
            [0, 0, 1]
        ], dtype=np.float64)
        matrix = matrix @ translate_to_origin
    
    # Apply scaling
    if abs(scale - 1.0) > 1e-9:
        scale_matrix = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        matrix = matrix @ scale_matrix
    
    # Apply rotation
    if abs(rotation_degrees) > 1e-9:
        angle_rad = math.radians(rotation_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        matrix = matrix @ rotation_matrix
    
    # If we used a center, translate back
    if center is not None:
        cx, cy = center
        translate_back = np.array([
            [1, 0, cx],
            [0, 1, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        matrix = matrix @ translate_back
    
    # Apply final translation
    tx, ty = translation
    if abs(tx) > 1e-9 or abs(ty) > 1e-9:
        translation_matrix = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], dtype=np.float64)
        matrix = matrix @ translation_matrix
    
    return matrix


def apply_affine_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply affine transformation to a set of 2D points.
    
    Args:
        points: Nx2 array of (x, y) coordinates
        matrix: 3x3 transformation matrix
        
    Returns:
        Nx2 array of transformed coordinates
    """
    if points.shape[1] != 2:
        raise ValueError(f"Points must be Nx2 array, got shape {points.shape}")
    
    # Convert to homogeneous coordinates (add column of ones)
    homogeneous_points = np.column_stack([points, np.ones(len(points))])
    
    # Apply transformation
    transformed_homogeneous = homogeneous_points @ matrix.T
    
    # Convert back to 2D coordinates (remove homogeneous coordinate)
    return transformed_homogeneous[:, :2]


def apply_polygon_transform(
    polygon_points: np.ndarray,
    translation: Tuple[float, float] = (0.0, 0.0),
    rotation_degrees: float = 0.0,
    scale: float = 1.0,
    center: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Apply transformation to polygon vertices.
    
    Args:
        polygon_points: Nx2 array of polygon vertices
        translation: (tx, ty) translation offsets
        rotation_degrees: Rotation angle in degrees
        scale: Scaling factor
        center: Optional center point for rotation/scaling
        
    Returns:
        Nx2 array of transformed vertices
    """
    if center is None:
        # Use polygon centroid as center
        center = np.mean(polygon_points, axis=0)
    
    matrix = create_transformation_matrix(translation, rotation_degrees, scale, center)
    return apply_affine_transform(polygon_points, matrix)


def convert_pixels_to_gds_units(
    pixel_offset: Tuple[float, float],
    pixel_size: float,
    flip_y: bool = True
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to GDS coordinate units.
    
    Args:
        pixel_offset: (dx_pixels, dy_pixels) offset in UI pixel space
        pixel_size: Size of one pixel in GDS units
        flip_y: Whether to flip Y coordinate (UI Y+ is down, GDS Y+ is up)
        
    Returns:
        (dx_gds, dy_gds) offset in GDS coordinate space
    """
    dx_pixels, dy_pixels = pixel_offset
    
    dx_gds = dx_pixels * pixel_size
    dy_gds = dy_pixels * pixel_size
    
    if flip_y:
        dy_gds = -dy_gds
    
    return (dx_gds, dy_gds)


def convert_gds_to_pixel_units(
    gds_offset: Tuple[float, float],
    pixel_size: float,
    flip_y: bool = True
) -> Tuple[float, float]:
    """
    Convert GDS coordinates to pixel coordinates.
    
    Args:
        gds_offset: (dx_gds, dy_gds) offset in GDS coordinate space
        pixel_size: Size of one pixel in GDS units
        flip_y: Whether to flip Y coordinate (GDS Y+ is up, UI Y+ is down)
        
    Returns:
        (dx_pixels, dy_pixels) offset in UI pixel space
    """
    dx_gds, dy_gds = gds_offset
    
    if pixel_size <= 0:
        return (0.0, 0.0)
    
    dx_pixels = dx_gds / pixel_size
    dy_pixels = dy_gds / pixel_size
    
    if flip_y:
        dy_pixels = -dy_pixels
    
    return (dx_pixels, dy_pixels)


def calculate_frame_bounds(
    original_bounds: Tuple[float, float, float, float],
    translation: Tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    rotation_90: int = 0
) -> Tuple[float, float, float, float]:
    """
    Calculate new frame bounds after applying transformations.
    
    Args:
        original_bounds: (xmin, ymin, xmax, ymax) original bounds
        translation: (tx, ty) translation offset
        scale: Scaling factor (>1 = zoom in, <1 = zoom out)
        rotation_90: 90-degree rotation (0, 90, 180, 270)
        
    Returns:
        (xmin, ymin, xmax, ymax) transformed bounds
    """
    xmin, ymin, xmax, ymax = original_bounds
    width = xmax - xmin
    height = ymax - ymin
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    
    # Apply translation to center
    tx, ty = translation
    new_center_x = center_x + tx
    new_center_y = center_y + ty
    
    # Apply scaling (inverse because higher scale = smaller viewing window)
    new_width = width / scale
    new_height = height / scale
    
    # Create new bounds
    new_xmin = new_center_x - new_width / 2
    new_ymin = new_center_y - new_height / 2
    new_xmax = new_center_x + new_width / 2
    new_ymax = new_center_y + new_height / 2
    
    # Apply 90-degree rotation if needed
    if rotation_90 % 360 != 0:
        bounds = apply_90_rotation_to_bounds(
            (new_xmin, new_ymin, new_xmax, new_ymax),
            (new_center_x, new_center_y),
            rotation_90
        )
        return bounds
    
    return (new_xmin, new_ymin, new_xmax, new_ymax)


def apply_90_rotation_to_bounds(
    bounds: Tuple[float, float, float, float],
    center: Tuple[float, float],
    rotation_90: int
) -> Tuple[float, float, float, float]:
    """
    Apply 90-degree rotation to rectangular bounds.
    
    Args:
        bounds: (xmin, ymin, xmax, ymax) original bounds
        center: (cx, cy) rotation center
        rotation_90: Rotation angle (must be multiple of 90)
        
    Returns:
        (xmin, ymin, xmax, ymax) rotated bounds
    """
    if rotation_90 % 90 != 0:
        raise ValueError(f"Rotation must be multiple of 90°, got: {rotation_90}")
    
    rotation_90 = rotation_90 % 360
    if rotation_90 == 0:
        return bounds
    
    xmin, ymin, xmax, ymax = bounds
    cx, cy = center
    
    # Get corners of the rectangle
    corners = [
        (xmin - cx, ymin - cy),  # bottom-left
        (xmax - cx, ymin - cy),  # bottom-right
        (xmax - cx, ymax - cy),  # top-right
        (xmin - cx, ymax - cy)   # top-left
    ]
    
    # Apply rotation to corners
    rotated_corners = []
    for dx, dy in corners:
        if rotation_90 == 90:
            # 90° CCW: (x,y) -> (-y, x)
            new_dx, new_dy = -dy, dx
        elif rotation_90 == 180:
            # 180°: (x,y) -> (-x, -y)
            new_dx, new_dy = -dx, -dy
        elif rotation_90 == 270:
            # 270° CCW: (x,y) -> (y, -x)
            new_dx, new_dy = dy, -dx
        else:
            new_dx, new_dy = dx, dy
        
        rotated_corners.append((cx + new_dx, cy + new_dy))
    
    # Find bounding box of rotated corners
    xs = [corner[0] for corner in rotated_corners]
    ys = [corner[1] for corner in rotated_corners]
    
    return (min(xs), min(ys), max(xs), max(ys))


def decompose_transformation_matrix(matrix: np.ndarray) -> Dict[str, float]:
    """
    Decompose a 3x3 transformation matrix into components.
    
    Args:
        matrix: 3x3 transformation matrix
        
    Returns:
        Dictionary with 'translation_x', 'translation_y', 'rotation_degrees', 'scale_x', 'scale_y'
    """
    if matrix.shape != (3, 3):
        raise ValueError(f"Matrix must be 3x3, got shape {matrix.shape}")
    
    # Extract translation
    tx = matrix[0, 2]
    ty = matrix[1, 2]
    
    # Extract scale and rotation from 2x2 upper-left submatrix
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    
    # Calculate scales
    scale_x = math.sqrt(a*a + c*c)
    scale_y = math.sqrt(b*b + d*d)
    
    # Calculate rotation (in radians, then convert to degrees)
    rotation_rad = math.atan2(c, a)
    rotation_degrees = math.degrees(rotation_rad)
    
    return {
        'translation_x': tx,
        'translation_y': ty,
        'rotation_degrees': rotation_degrees,
        'scale_x': scale_x,
        'scale_y': scale_y
    }


def validate_transformation_parameters(
    translation: Optional[Tuple[float, float]] = None,
    rotation_degrees: Optional[float] = None,
    scale: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validate transformation parameters.
    
    Args:
        translation: (tx, ty) translation values
        rotation_degrees: Rotation angle in degrees
        scale: Scaling factor
        
    Returns:
        Dictionary with 'valid' boolean and 'errors' list
    """
    errors = []
    
    if translation is not None:
        tx, ty = translation
        if not (math.isfinite(tx) and math.isfinite(ty)):
            errors.append(f"Translation must be finite, got: ({tx}, {ty})")
    
    if rotation_degrees is not None:
        if not math.isfinite(rotation_degrees):
            errors.append(f"Rotation must be finite, got: {rotation_degrees}")
        # Allow any rotation value, will be normalized
    
    if scale is not None:
        if not math.isfinite(scale) or scale <= 0:
            errors.append(f"Scale must be positive and finite, got: {scale}")
        if scale < 0.01 or scale > 100.0:
            errors.append(f"Scale should be between 0.01 and 100.0, got: {scale}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }
