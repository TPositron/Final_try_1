"""
Three Point Alignment Calculator

Calculates transformation parameters (move, rotate, zoom) from 3 corresponding points.
Decomposes affine transformation into individual parameters for manual alignment system.
"""

import numpy as np
import cv2
import math
from typing import Tuple, List, Dict

def calculate_transformation_parameters(sem_points: List[Tuple[float, float]], 
                                      gds_points: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Calculate transformation parameters from 3 corresponding points.
    
    Args:
        sem_points: List of 3 SEM points [(x1,y1), (x2,y2), (x3,y3)]
        gds_points: List of 3 GDS points [(x1,y1), (x2,y2), (x3,y3)]
    
    Returns:
        Dict with keys: x_offset, y_offset, rotation, scale
    """
    if len(sem_points) != 3 or len(gds_points) != 3:
        raise ValueError("Exactly 3 points required for each image")
    
    # Convert to numpy arrays
    sem_pts = np.array(sem_points, dtype=np.float32)
    gds_pts = np.array(gds_points, dtype=np.float32)
    
    # Calculate affine transformation matrix
    transform_matrix = cv2.getAffineTransform(gds_pts, sem_pts)
    
    # Decompose transformation matrix into individual parameters
    # Matrix format: [[a, b, tx], [c, d, ty]]
    a, b, tx = transform_matrix[0]
    c, d, ty = transform_matrix[1]
    
    # Calculate scale (average of x and y scaling)
    scale_x = math.sqrt(a*a + c*c)
    scale_y = math.sqrt(b*b + d*d)
    scale = (scale_x + scale_y) / 2.0
    
    # Calculate rotation angle in degrees
    rotation = math.degrees(math.atan2(c, a))
    
    # Translation is directly from the matrix
    x_offset = tx
    y_offset = ty
    
    return {
        'x_offset': float(x_offset),
        'y_offset': float(y_offset), 
        'rotation': float(rotation),
        'scale': float(scale)
    }

def apply_transformation_to_manual_alignment(main_window, transform_params: Dict[str, float]):
    """
    Apply calculated transformation parameters to manual alignment system.
    
    Args:
        main_window: Reference to main window
        transform_params: Dict with x_offset, y_offset, rotation, scale
    """
    if hasattr(main_window, 'manual_alignment_controls'):
        # Set parameters in manual alignment controls
        controls = main_window.manual_alignment_controls
        if hasattr(controls, 'set_parameters'):
            controls.set_parameters(transform_params)
        
        # Trigger alignment update
        if hasattr(main_window, 'alignment_operations_manager'):
            main_window.alignment_operations_manager.apply_manual_transformation(transform_params)