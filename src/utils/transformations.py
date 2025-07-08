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

def create_transformation_matrix(translation: Tuple[float, float] = (0.0, 0.0),
                               rotation_degrees: float = 0.0,
                               scale: float = 1.0,
                               center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Create a 3x3 homogeneous transformation matrix.
    
    Args:
        translation: (dx, dy) translation in pixels
        rotation_degrees: Rotation angle in degrees (clockwise)
        scale: Uniform scale factor
        center: Center point for rotation and scaling. If None, uses origin.
    
    Returns:
        3x3 transformation matrix
    """
    # Initialize as identity matrix
    matrix = np.eye(3, dtype=np.float64)
    
    # Handle center-based transformations
    if center is not None:
        cx, cy = center
        # Step 1: Translate to origin
        translate_to_origin = np.array([
            [1, 0, -cx],
            [0, 1, -cy], 
            [0, 0, 1]
        ], dtype=np.float64)
        matrix = matrix @ translate_to_origin
    
    # Step 2: Apply scaling
    if abs(scale - 1.0) > 1e-9:
        scale_matrix = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        matrix = matrix @ scale_matrix
    
    # Step 3: Apply rotation
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
    
    # Step 4: Translate back from origin
    if center is not None:
        cx, cy = center
        translate_back = np.array([
            [1, 0, cx],
            [0, 1, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        matrix = matrix @ translate_back
    
    # Step 5: Apply final translation
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
    Apply affine transformation to a set of points.
    
    Args:
        points: Nx2 array of points
        matrix: 3x3 transformation matrix
    
    Returns:
        Transformed Nx2 array of points
    """
    if points.shape[1] != 2:
        raise ValueError(f"Points must be Nx2 array, got shape {points.shape}")
    
    # Convert to homogeneous coordinates
    homogeneous_points = np.column_stack([points, np.ones(len(points))])
    
    # Apply transformation
    transformed_homogeneous = homogeneous_points @ matrix.T
    
    # Convert back to 2D coordinates
    return transformed_homogeneous[:, :2]

def apply_polygon_transform(polygon_points: np.ndarray,
                          translation: Tuple[float, float] = (0.0, 0.0),
                          rotation_degrees: float = 0.0,
                          scale: float = 1.0,
                          center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Apply transformation to polygon points.
    
    Args:
        polygon_points: Nx2 array of polygon vertices
        translation: (dx, dy) translation
        rotation_degrees: Rotation angle in degrees
        scale: Scale factor
        center: Center point for rotation/scaling. If None, uses polygon centroid.
    
    Returns:
        Transformed polygon points
    """
    if center is None:
        center = tuple(np.mean(polygon_points, axis=0))
    
    # Create transformation matrix
    matrix = create_transformation_matrix(translation, rotation_degrees, scale, center)
    
    # Apply transformation
    return apply_affine_transform(polygon_points, matrix)

def convert_pixels_to_gds_units(pixel_offset: Tuple[float, float],
                               pixel_size: float,
                               flip_y: bool = True) -> Tuple[float, float]:
    """
    Convert pixel coordinates to GDS units.
    
    Args:
        pixel_offset: (dx, dy) in pixels
        pixel_size: Size of one pixel in GDS units
        flip_y: Whether to flip Y coordinate (image vs GDS coordinate systems)
    
    Returns:
        (dx, dy) in GDS units
    """
    dx_pixels, dy_pixels = pixel_offset
    dx_gds = dx_pixels * pixel_size
    dy_gds = dy_pixels * pixel_size
    
    if flip_y:
        dy_gds = -dy_gds
    
    return (dx_gds, dy_gds)

def convert_gds_to_pixel_units(gds_offset: Tuple[float, float],
                              pixel_size: float,
                              flip_y: bool = True) -> Tuple[float, float]:
    """
    Convert GDS coordinates to pixel units.
    
    Args:
        gds_offset: (dx, dy) in GDS units
        pixel_size: Size of one pixel in GDS units
        flip_y: Whether to flip Y coordinate
    
    Returns:
        (dx, dy) in pixels
    """
    dx_gds, dy_gds = gds_offset
    
    if pixel_size <= 0:
        return (0.0, 0.0)
    
    dx_pixels = dx_gds / pixel_size
    dy_pixels = dy_gds / pixel_size
    
    if flip_y:
        dy_pixels = -dy_pixels
    
    return (dx_pixels, dy_pixels)

def calculate_frame_bounds(original_bounds: Tuple[float, float, float, float],
                          translation: Tuple[float, float] = (0.0, 0.0),
                          scale: float = 1.0,
                          rotation_90: int = 0) -> Tuple[float, float, float, float]:
    """
    Calculate new frame bounds after transformation.
    
    Args:
        original_bounds: (xmin, ymin, xmax, ymax)
        translation: (dx, dy) translation
        scale: Scale factor
        rotation_90: Rotation in 90-degree increments
    
    Returns:
        New bounds (xmin, ymin, xmax, ymax)
    """
    xmin, ymin, xmax, ymax = original_bounds
    width = xmax - xmin
    height = ymax - ymin
    
    # Calculate center
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    
    # Apply translation to center
    tx, ty = translation
    new_center_x = center_x + tx
    new_center_y = center_y + ty
    
    # Apply inverse scale to dimensions (frame expands when zooming in)
    new_width = width / scale
    new_height = height / scale
    
    # Calculate new bounds
    new_xmin = new_center_x - new_width / 2
    new_ymin = new_center_y - new_height / 2
    new_xmax = new_center_x + new_width / 2
    new_ymax = new_center_y + new_height / 2
    
    # Apply rotation if needed
    if rotation_90 % 360 != 0:
        bounds = apply_90_rotation_to_bounds(
            (new_xmin, new_ymin, new_xmax, new_ymax),
            (new_center_x, new_center_y),
            rotation_90
        )
        return bounds
    
    return (new_xmin, new_ymin, new_xmax, new_ymax)

def apply_90_rotation_to_bounds(bounds: Tuple[float, float, float, float],
                               center: Tuple[float, float],
                               rotation_90: int) -> Tuple[float, float, float, float]:
    """
    Apply 90-degree rotation to bounds.
    
    Args:
        bounds: (xmin, ymin, xmax, ymax)
        center: (cx, cy) rotation center
        rotation_90: Rotation angle in 90-degree increments
    
    Returns:
        Rotated bounds
    """
    if rotation_90 % 90 != 0:
        raise ValueError(f"Rotation must be multiple of 90°, got: {rotation_90}")
    
    rotation_90 = rotation_90 % 360
    if rotation_90 == 0:
        return bounds
    
    xmin, ymin, xmax, ymax = bounds
    cx, cy = center
    
    # Get corners relative to center
    corners = [
        (xmin - cx, ymin - cy),  # bottom-left
        (xmax - cx, ymin - cy),  # bottom-right  
        (xmax - cx, ymax - cy),  # top-right
        (xmin - cx, ymax - cy)   # top-left
    ]
    
    # Apply rotation to each corner
    rotated_corners = []
    for dx, dy in corners:
        if rotation_90 == 90:
            new_dx, new_dy = -dy, dx       # 90° CCW
        elif rotation_90 == 180:
            new_dx, new_dy = -dx, -dy      # 180°
        elif rotation_90 == 270:
            new_dx, new_dy = dy, -dx       # 270° CCW (90° CW)
        else:
            new_dx, new_dy = dx, dy        # 0°
        
        rotated_corners.append((cx + new_dx, cy + new_dy))
    
    # Find new bounds
    xs = [corner[0] for corner in rotated_corners]
    ys = [corner[1] for corner in rotated_corners]
    
    return (min(xs), min(ys), max(xs), max(ys))

def decompose_transformation_matrix(matrix: np.ndarray) -> Dict[str, float]:
    """
    Decompose transformation matrix into components.
    
    Args:
        matrix: 3x3 transformation matrix
    
    Returns:
        Dictionary with transformation components
    """
    if matrix.shape != (3, 3):
        raise ValueError(f"Matrix must be 3x3, got shape {matrix.shape}")
    
    # Extract translation
    tx = matrix[0, 2]
    ty = matrix[1, 2]
    
    # Extract linear transformation part
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    
    # Calculate scale factors
    scale_x = math.sqrt(a*a + c*c)
    scale_y = math.sqrt(b*b + d*d)
    
    # Calculate rotation angle
    rotation_rad = math.atan2(c, a)
    rotation_degrees = math.degrees(rotation_rad)
    
    return {
        'translation_x': tx,
        'translation_y': ty,
        'rotation_degrees': rotation_degrees,
        'scale_x': scale_x,
        'scale_y': scale_y
    }

def validate_transformation_parameters(translation: Optional[Tuple[float, float]] = None,
                                     rotation_degrees: Optional[float] = None,
                                     scale: Optional[float] = None) -> Dict[str, Any]:
    """
    Validate transformation parameters.
    
    Args:
        translation: (dx, dy) translation
        rotation_degrees: Rotation angle
        scale: Scale factor
    
    Returns:
        Validation result with 'valid' boolean and 'errors' list
    """
    errors = []
    
    # Validate translation
    if translation is not None:
        tx, ty = translation
        if not (math.isfinite(tx) and math.isfinite(ty)):
            errors.append(f"Translation must be finite, got: ({tx}, {ty})")
        
        # Check reasonable bounds
        max_translation = 10000.0  # pixels
        if abs(tx) > max_translation or abs(ty) > max_translation:
            errors.append(f"Translation too large: ({tx}, {ty}), max: ±{max_translation}")
    
    # Validate rotation
    if rotation_degrees is not None:
        if not math.isfinite(rotation_degrees):
            errors.append(f"Rotation must be finite, got: {rotation_degrees}")
    
    # Validate scale
    if scale is not None:
        if not math.isfinite(scale) or scale <= 0:
            errors.append(f"Scale must be positive and finite, got: {scale}")
        
        if scale < 0.01 or scale > 100.0:
            errors.append(f"Scale outside reasonable range (0.01-100.0): {scale}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def create_rotation_matrix_2d(angle_degrees: float, center: Tuple[float, float]) -> np.ndarray:
    """
    Create 2D rotation matrix around a center point.
    
    Args:
        angle_degrees: Rotation angle in degrees
        center: (cx, cy) rotation center
    
    Returns:
        2x3 rotation matrix for cv2.warpAffine
    """
    import cv2
    return cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

def transform_points_with_matrix(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Transform points using a 2x3 or 3x3 transformation matrix.
    
    Args:
        points: Nx2 array of points
        transform_matrix: 2x3 or 3x3 transformation matrix
    
    Returns:
        Transformed points
    """
    if transform_matrix.shape == (2, 3):
        # 2x3 matrix - extend to 3x3
        matrix_3x3 = np.vstack([transform_matrix, [0, 0, 1]])
    elif transform_matrix.shape == (3, 3):
        matrix_3x3 = transform_matrix
    else:
        raise ValueError(f"Matrix must be 2x3 or 3x3, got {transform_matrix.shape}")
    
    return apply_affine_transform(points, matrix_3x3)

def calculate_zoom_transform_matrix(zoom_factor: float, center: Tuple[float, float]) -> np.ndarray:
    """
    Calculate transformation matrix for zoom operation around center.
    
    Args:
        zoom_factor: Zoom factor (1.0 = no change, >1.0 = zoom in, <1.0 = zoom out)
        center: (cx, cy) zoom center point
    
    Returns:
        3x3 transformation matrix
    """
    return create_transformation_matrix(
        translation=(0.0, 0.0),
        rotation_degrees=0.0,
        scale=zoom_factor,
        center=center
    )

def calculate_movement_transform_matrix(dx: float, dy: float) -> np.ndarray:
    """
    Calculate transformation matrix for translation.
    
    Args:
        dx: X translation
        dy: Y translation
    
    Returns:
        3x3 transformation matrix
    """
    return create_transformation_matrix(
        translation=(dx, dy),
        rotation_degrees=0.0,
        scale=1.0,
        center=None
    )

def combine_transformation_matrices(*matrices: np.ndarray) -> np.ndarray:
    """
    Combine multiple transformation matrices.
    
    Args:
        matrices: Variable number of 3x3 transformation matrices
    
    Returns:
        Combined transformation matrix
    """
    if not matrices:
        return np.eye(3)
    
    result = matrices[0].copy()
    for matrix in matrices[1:]:
        result = result @ matrix
    
    return result

def invert_transformation_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Invert a transformation matrix.
    
    Args:
        matrix: 3x3 transformation matrix
    
    Returns:
        Inverted transformation matrix
    """
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is not invertible (singular)")

def snap_rotation_to_90_degrees(angle_degrees: float) -> float:
    """
    Snap rotation angle to nearest 90-degree increment.
    
    Args:
        angle_degrees: Input angle in degrees
    
    Returns:
        Snapped angle (0, 90, 180, or 270)
    """
    # Normalize to [0, 360)
    normalized = angle_degrees % 360
    
    # Find nearest 90-degree increment
    snapped = round(normalized / 90.0) * 90.0
    
    # Return in [0, 360) range
    return snapped % 360

def create_composite_transform(translation: Tuple[float, float] = (0.0, 0.0),
                              rotation_degrees: float = 0.0,
                              zoom_factor: float = 1.0,
                              center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Create composite transformation matrix for common UI operations.
    
    Args:
        translation: (dx, dy) movement
        rotation_degrees: Rotation angle
        zoom_factor: Zoom level
        center: Center point for rotation and zoom
    
    Returns:
        3x3 composite transformation matrix
    """
    return create_transformation_matrix(
        translation=translation,
        rotation_degrees=rotation_degrees,
        scale=zoom_factor,
        center=center
    )

