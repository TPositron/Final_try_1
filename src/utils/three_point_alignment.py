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
    Calculate transformation parameters to transform GDS points to align with SEM points.
    
    Args:
        sem_points: List of 3 SEM points [(x1,y1), (x2,y2), (x3,y3)] - target positions
        gds_points: List of 3 GDS points [(x1,y1), (x2,y2), (x3,y3)] - source positions to transform
    
    Returns:
        Dict with keys: x_offset, y_offset, rotation, scale
    """
    if len(sem_points) != 3 or len(gds_points) != 3:
        raise ValueError("Exactly 3 points required for each image")
    
    # Convert to numpy arrays
    sem_pts = np.array(sem_points, dtype=np.float32)
    gds_pts = np.array(gds_points, dtype=np.float32)
    
    # Simple centroid-based calculation
    sem_centroid = np.mean(sem_pts, axis=0)
    gds_centroid = np.mean(gds_pts, axis=0)
    
    # Direct translation difference
    x_offset = float(sem_centroid[0] - gds_centroid[0])
    y_offset = float(sem_centroid[1] - gds_centroid[1])
    
    # Calculate scale using first two points
    sem_dist = np.linalg.norm(sem_pts[1] - sem_pts[0])
    gds_dist = np.linalg.norm(gds_pts[1] - gds_pts[0])
    scale = sem_dist / gds_dist if gds_dist > 0 else 1.0
    
    # Calculate rotation
    sem_vec = sem_pts[1] - sem_pts[0]
    gds_vec = gds_pts[1] - gds_pts[0]
    sem_angle = math.atan2(sem_vec[1], sem_vec[0])
    gds_angle = math.atan2(gds_vec[1], gds_vec[0])
    rotation = math.degrees(sem_angle - gds_angle)
    
    # Normalize rotation
    while rotation > 180:
        rotation -= 360
    while rotation < -180:
        rotation += 360
    
    print(f"DEBUG: Centroids:")
    print(f"  SEM centroid: ({sem_centroid[0]:.2f}, {sem_centroid[1]:.2f})")
    print(f"  GDS centroid: ({gds_centroid[0]:.2f}, {gds_centroid[1]:.2f})")
    print(f"  Direct translation: ({x_offset:.2f}, {y_offset:.2f})")
    print(f"  Scale: {scale:.3f}, Rotation: {rotation:.2f}°")
    
    return {
        'x_offset': x_offset,
        'y_offset': y_offset, 
        'rotation': float(rotation),
        'scale': scale
    }

def apply_transformation_to_manual_alignment(main_window, transform_params: Dict[str, float]):
    """
    Apply calculated transformation parameters to manual alignment system.
    
    Args:
        main_window: Reference to main window
        transform_params: Dict with x_offset, y_offset, rotation, scale
    """
    try:
        print(f"Applying 3-point transformation: {transform_params}")
        
        # Convert parameters to manual alignment format
        manual_params = {
            'x_offset': transform_params.get('x_offset', 0),
            'y_offset': transform_params.get('y_offset', 0), 
            'rotation': transform_params.get('rotation', 0),
            'zoom': int(transform_params.get('scale', 1.0) * 100),  # Convert scale to zoom percentage
            'transparency': 70  # Default transparency
        }
        
        print(f"Converted to manual format: {manual_params}")
        
        if hasattr(main_window, 'manual_alignment_controls'):
            # Set parameters in manual alignment controls
            controls = main_window.manual_alignment_controls
            if hasattr(controls, 'set_parameters'):
                controls.set_parameters(manual_params)
                print("✓ Parameters set in manual alignment controls")
            
            # Trigger manual alignment change to update display
            if hasattr(main_window, '_on_manual_alignment_changed'):
                main_window._on_manual_alignment_changed(manual_params)
                print("✓ Manual alignment display updated")
        
        # Switch to manual alignment tab to show results
        if hasattr(main_window, 'alignment_sub_tabs'):
            main_window.alignment_sub_tabs.setCurrentIndex(0)  # Switch to manual tab
            print("✓ Switched to manual alignment tab")
            
    except Exception as e:
        print(f"Error applying transformation to manual alignment: {e}")
        import traceback
        traceback.print_exc()