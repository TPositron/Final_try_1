"""
GDS Display Generator - Core GDS Visualization Engine

This module provides the fundamental GDS visualization capabilities for the application.
It handles loading GDS files, extracting specific structures, and generating display images
that can be overlaid on SEM images for alignment purposes.

Key Functions:
- generate_gds_display(): Main function to generate structure display images  
- get_structure_info(): Retrieves metadata for predefined structures
- get_project_gds_path(): Returns the default GDS file path
- list_available_structures(): Lists all available structure IDs

Predefined Structures:
1. Circpol_T2 - Layer 14
2. IP935Left_11 - Layers 1,2  
3. IP935Left_14 - Layer 1
4. QC855GC_CROSS_Bottom - Layers 1,2
5. QC935_46 - Layer 1
"""

import gdspy
import numpy as np
import cv2
import os
from typing import Dict, List, Tuple, Optional

# Structure bounds constant for compatibility
STRUCTURE_BOUNDS = {
    1: (688.55, 5736.55, 760.55, 5807.1),
    2: (693.99, 6406.40, 723.59, 6428.96),
    3: (980.959, 6025.959, 1001.770, 6044.979),
    4: (3730.00, 4700.99, 3756.00, 4760.00),
    5: (7195.558, 5046.99, 7203.99, 5055.33964)
}


def get_project_gds_path() -> str:
    """Get the path to the project GDS file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up to find the Image_analysis project root
    while os.path.basename(current_dir) != 'Image_analysis' and current_dir != os.path.dirname(current_dir):
        current_dir = os.path.dirname(current_dir)
    return os.path.join(current_dir, "Data", "GDS", "Institute_Project_GDS1.gds")


def get_structure_definitions() -> Dict[int, Dict]:
    """Get the predefined structure definitions."""
    return {
        1: {
            'name': 'Circpol_T2',
            'bounds': (688.55, 5736.55, 760.55, 5807.1), 
            'layers': [14]
        },
        2: {
            'name': 'IP935Left_11',
            'bounds': (693.99, 6406.40, 723.59, 6428.96), 
            'layers': [1, 2]
        },
        3: {
            'name': 'IP935Left_14',
            'bounds': (980.959, 6025.959, 1001.770, 6044.979), 
            'layers': [1]
        },
        4: {
            'name': 'QC855GC_CROSS_Bottom',
            'bounds': (3730.00, 4700.99, 3756.00, 4760.00), 
            'layers': [1, 2]
        },
        5: {
            'name': 'QC935_46',
            'bounds': (7195.558, 5046.99, 7203.99, 5055.33964), 
            'layers': [1]
        }
    }


def generate_gds_display(structure_num: int, target_size: Tuple[int, int] = (1024, 666)) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Generate display image for a specific structure using initial bounds.
    
    Args:
        structure_num: Structure ID (1-5)
        target_size: Output image size (width, height)
        
    Returns:
        Tuple of (binary image array, bounds)
    """
    structures = get_structure_definitions()
    
    if structure_num not in structures:
        raise ValueError(f"Structure {structure_num} not defined. Available: {list(structures.keys())}")
    
    struct_def = structures[structure_num]
    bounds = struct_def['bounds']
    layers = struct_def['layers']
    
    image = render_gds_from_bounds(bounds, layers, target_size, structure_num)
    return image, bounds


def render_gds_from_bounds(bounds: Tuple[float, float, float, float], 
                          layers: List[int], 
                          target_size: Tuple[int, int],
                          structure_num: int = None) -> np.ndarray:
    """
    Render GDS data from specified bounds to image.
    
    Args:
        bounds: Extraction bounds (xmin, ymin, xmax, ymax)
        layers: Layer numbers to render
        target_size: Output image size (width, height)
        structure_num: Optional structure number for logging
        
    Returns:
        Binary image array (0=structure, 255=background)
    """
    gds_path = get_project_gds_path()
    
    if not os.path.exists(gds_path):
        raise FileNotFoundError(f"GDS file not found: {gds_path}")
    
    # Load GDS file
    gds = gdspy.GdsLibrary().read_gds(gds_path)
    cell = gds.top_level()[0]
    polygons = cell.get_polygons(by_spec=True)
    
    # Calculate rendering parameters
    xmin, ymin, xmax, ymax = bounds
    gds_width = xmax - xmin
    gds_height = ymax - ymin
    
    if gds_width <= 0 or gds_height <= 0:
        raise ValueError(f"Invalid bounds: {bounds}")
    
    # Scale to fit target size while maintaining aspect ratio
    scale = min(target_size[0] / gds_width, target_size[1] / gds_height)
    scaled_width = int(gds_width * scale)
    scaled_height = int(gds_height * scale)
    
    # Center the scaled content
    offset_x = (target_size[0] - scaled_width) // 2
    offset_y = (target_size[1] - scaled_height) // 2
    
    # Create white background image
    image = np.ones((target_size[1], target_size[0]), dtype=np.uint8) * 255
    
    # Render polygons from specified layers
    polygon_count = 0
    for layer in layers:
        if (layer, 0) in polygons:
            for poly in polygons[(layer, 0)]:
                # Check if polygon intersects with bounds
                if np.any((poly[:, 0] >= xmin) & (poly[:, 0] <= xmax) &
                          (poly[:, 1] >= ymin) & (poly[:, 1] <= ymax)):
                    polygon_count += 1
                    
                    # Transform to image coordinates
                    norm_poly = (poly - [xmin, ymin]) * scale
                    int_poly = np.round(norm_poly).astype(np.int32)
                    int_poly += [offset_x, offset_y]
                    
                    # Flip Y-axis for image coordinates
                    int_poly[:, 1] = target_size[1] - 1 - int_poly[:, 1]
                    
                    # Render polygon
                    if len(int_poly) >= 3:
                        cv2.fillPoly(image, [int_poly], color=(0,))
    
    log_msg = f"Rendered {polygon_count} polygons from layers {layers}"
    if structure_num:
        log_msg += f" for structure {structure_num}"
    print(log_msg)
    
    return image


def get_structure_info(structure_num: int) -> Optional[Dict]:
    """
    Get comprehensive information about a structure.
    
    Args:
        structure_num: Structure ID (1-5)
        
    Returns:
        Dictionary with structure metadata including bounds, layers, and scaling info
    """
    structures = get_structure_definitions()
    
    if structure_num not in structures:
        return None
    
    struct_def = structures[structure_num]
    bounds = struct_def['bounds']
    
    # Calculate dimensions and scaling
    xmin, ymin, xmax, ymax = bounds
    gds_width = xmax - xmin
    gds_height = ymax - ymin
    
    # Default target size for scaling calculations
    target_size = (1024, 666)
    scale = min(target_size[0] / gds_width, target_size[1] / gds_height)
    
    return {
        'name': struct_def['name'],
        'bounds': bounds,
        'layers': struct_def['layers'],
        'gds_width': gds_width,
        'gds_height': gds_height,
        'pixel_scale': scale,
        'gds_to_pixel_x': scale,
        'gds_to_pixel_y': scale,
        'pixel_to_gds_x': 1.0 / scale,
        'pixel_to_gds_y': 1.0 / scale,
        'center_x': (xmin + xmax) / 2,
        'center_y': (ymin + ymax) / 2
    }


def list_available_structures() -> List[int]:
    """Get list of available structure IDs."""
    return list(get_structure_definitions().keys())


def get_all_structures_info() -> Dict[int, Dict]:
    """Get information for all available structures."""
    result = {}
    for structure_num in list_available_structures():
        info = get_structure_info(structure_num)
        if info:
            result[structure_num] = info
    return result


def validate_structure_bounds(structure_num: int) -> bool:
    """
    Validate that a structure's bounds contain actual GDS data.
    
    Args:
        structure_num: Structure ID to validate
        
    Returns:
        True if structure contains polygons, False otherwise
    """
    try:
        image, _ = generate_gds_display(structure_num, (100, 100))  # Small test image
        # Check if image contains any structure pixels (value 0)
        return np.any(image == 0)
    except Exception:
        return False


def get_structure_bounds_info() -> Dict[int, Dict]:
    """Get bounds information for all structures."""
    structures = get_structure_definitions()
    result = {}
    
    for structure_num, struct_def in structures.items():
        bounds = struct_def['bounds']
        xmin, ymin, xmax, ymax = bounds
        
        result[structure_num] = {
            'name': struct_def['name'],
            'bounds': bounds,
            'width': xmax - xmin,
            'height': ymax - ymin,
            'area': (xmax - xmin) * (ymax - ymin),
            'center': ((xmin + xmax) / 2, (ymin + ymax) / 2),
            'layers': struct_def['layers']
        }
    
    return result


# Legacy function for compatibility
def generate_display_gds_with_bounds(structure_num: int, target_size: Tuple[int, int], 
                                   custom_bounds: List[float]) -> np.ndarray:
    """
    Generate GDS display with custom bounds (for backward compatibility).
    
    Args:
        structure_num: Structure ID
        target_size: Output image size  
        custom_bounds: Custom bounds [xmin, ymin, xmax, ymax]
        
    Returns:
        Rendered image
    """
    structures = get_structure_definitions()
    
    if structure_num not in structures:
        raise ValueError(f"Structure {structure_num} not defined")
    
    layers = structures[structure_num]['layers']
    bounds = tuple(custom_bounds)
    
    return render_gds_from_bounds(bounds, layers, target_size, structure_num)
