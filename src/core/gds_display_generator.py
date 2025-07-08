"""
GDS Display Generator - Core GDS Visualization Engine

This module provides the fundamental GDS visualization capabilities for the application.
It handles loading GDS files, extracting specific structures, and generating display images
that can be overlaid on SEM images for alignment purposes.

Key Functions:
- generate_display_gds(): Main function to generate structure display images
- get_structure_info(): Retrieves metadata for predefined structures
- get_gds_path(): Returns the default GDS file path
- list_available_structures(): Lists all available structure IDs
- generate_display_gds_with_bounds(): Generates displays with custom bounds
- get_all_structures_info(): Returns complete structure metadata

Predefined Structures (Hard-coded):
1. Circpol_T2 - Layer 14
2. IP935Left_11 - Layers 1,2
3. IP935Left_14 - Layer 1
4. QC855GC_CROSS_Bottom - Layers 1,2
5. QC935_46 - Layer 1

Dependencies:
- Uses: gdspy (GDS file reading), numpy, cv2 (image processing)
- Called by: gds_aligned_generator.py (for structure info)
- Called by: services/new_gds_service.py
- Called by: services/gds_image_service.py
- Called by: ui/gds_manager.py (indirectly)

Data Flow:
1. Loads GDS file using gdspy
2. Extracts polygons for specified layers within structure bounds
3. Transforms polygon coordinates to image pixel coordinates
4. Renders polygons to binary image using cv2.fillPoly
5. Returns image for display or further processing

Critical Notes:
- Uses hard-coded structure definitions for consistency
- Applies Y-axis flipping for proper image coordinate system
- Generates binary images (0=structure, 255=background)
- Maintains aspect ratio and centering for proper alignment
"""
import gdspy
import numpy as np
import cv2
import os
from typing import Dict, List, Tuple, Optional

def generate_display_gds(structure_num: int, target_size: Tuple[int, int] = (1024, 666)) -> np.ndarray:
    GDS_PATH = "C:\\Users\\tarik\\Image_analysis\\Data\\GDS\\Institute_Project_GDS1.gds"
    
    structures = {
        1: {'bounds': (688.55, 5736.55, 760.55, 5807.1), 'layers': [14]},
        2: {'bounds': (693.99, 6406.40, 723.59, 6428.96), 'layers': [1, 2]},
        3: {'bounds': (980.959, 6025.959, 1001.770, 6044.979), 'layers': [1]},
        4: {'bounds': (3730.00, 4700.99, 3756.00, 4760.00), 'layers': [1, 2]},
        5: {'bounds': (7195.558, 5046.99, 7203.99, 5055.33964), 'layers': [1]}
    }
    
    if structure_num not in structures:
        raise ValueError(f"Structure {structure_num} not defined.")
    
    xmin, ymin, xmax, ymax = structures[structure_num]['bounds']
    layers = structures[structure_num]['layers']
    
    gds_width = xmax - xmin
    gds_height = ymax - ymin
    
    scale = min(target_size[0] / gds_width, target_size[1] / gds_height)
    scaled_width = int(gds_width * scale)
    scaled_height = int(gds_height * scale)
    
    offset_x = (target_size[0] - scaled_width) // 2
    offset_y = (target_size[1] - scaled_height) // 2
    
    image = np.ones((target_size[1], target_size[0]), dtype=np.uint8) * 255
    
    if not os.path.exists(GDS_PATH):
        raise FileNotFoundError(f"GDS file not found: {GDS_PATH}")
    
    gds = gdspy.GdsLibrary().read_gds(GDS_PATH)
    cell = gds.top_level()[0]
    polygons = cell.get_polygons(by_spec=True)
    
    polygon_count = 0
    for layer in layers:
        if (layer, 0) in polygons:
            for poly in polygons[(layer, 0)]:
                if np.any((poly[:, 0] >= xmin) & (poly[:, 0] <= xmax) &
                          (poly[:, 1] >= ymin) & (poly[:, 1] <= ymax)):
                    polygon_count += 1
                    
                    norm_poly = (poly - [xmin, ymin]) * scale
                    int_poly = np.round(norm_poly).astype(np.int32)
                    int_poly += [offset_x, offset_y]
                    int_poly[:, 1] = target_size[1] - 1 - int_poly[:, 1]  # Flip Y-axis
                    
                    # Fixed fillPoly call - use tuple for color parameter
                    cv2.fillPoly(image, [int_poly], color=(0,))
    
    print(f"Display generator rendered {polygon_count} polygons for structure {structure_num}")
    return image

def get_structure_info(structure_num: int) -> Optional[Dict]:
    structures = {
        1: {'name': 'Circpol_T2', 'initial_bounds': (688.55, 5736.55, 760.55, 5807.1), 'layers': [14]},
        2: {'name': 'IP935Left_11', 'initial_bounds': (693.99, 6406.40, 723.59, 6428.96), 'layers': [1, 2]},
        3: {'name': 'IP935Left_14', 'initial_bounds': (980.959, 6025.959, 1001.770, 6044.979), 'layers': [1]},
        4: {'name': 'QC855GC_CROSS_Bottom', 'initial_bounds': (3730.00, 4700.99, 3756.00, 4760.00), 'layers': [1, 2]},
        5: {'name': 'QC935_46', 'initial_bounds': (7195.558, 5046.99, 7203.99, 5055.33964), 'layers': [1]}
    }
    
    if structure_num not in structures:
        return None
    
    struct_data = structures[structure_num]
    bounds = struct_data['initial_bounds']
    xmin, ymin, xmax, ymax = bounds
    gds_width = xmax - xmin
    gds_height = ymax - ymin
    
    target_size = (1024, 666)
    scale = min(target_size[0] / gds_width, target_size[1] / gds_height)
    
    return {
        'name': struct_data['name'],
        'bounds': bounds,
        'layers': struct_data['layers'],
        'gds_width': gds_width,
        'gds_height': gds_height,
        'pixel_scale': scale,
        'gds_to_pixel_x': scale,
        'gds_to_pixel_y': scale,
        'pixel_to_gds_x': 1.0 / scale,
        'pixel_to_gds_y': 1.0 / scale
    }

def get_gds_path() -> str:
    return "C:\\Users\\tarik\\Image_analysis\\Data\\GDS\\Institute_Project_GDS1.gds"

def list_available_structures() -> List[int]:
    return [1, 2, 3, 4, 5]

def generate_display_gds_with_bounds(structure_num: int, target_size: Tuple[int, int], custom_bounds: List[float]) -> np.ndarray:
    """Generate GDS display with custom bounds."""
    GDS_PATH = "C:\\Users\\tarik\\Image_analysis\\Data\\GDS\\Institute_Project_GDS1.gds"
    
    structures = {
        1: {'layers': [14]},
        2: {'layers': [1, 2]},
        3: {'layers': [1]},
        4: {'layers': [1, 2]},
        5: {'layers': [1]}
    }
    
    if structure_num not in structures:
        raise ValueError(f"Structure {structure_num} not defined.")
    
    xmin, ymin, xmax, ymax = custom_bounds
    layers = structures[structure_num]['layers']
    
    gds_width = xmax - xmin
    gds_height = ymax - ymin
    
    scale = min(target_size[0] / gds_width, target_size[1] / gds_height)
    scaled_width = int(gds_width * scale)
    scaled_height = int(gds_height * scale)
    
    offset_x = (target_size[0] - scaled_width) // 2
    offset_y = (target_size[1] - scaled_height) // 2
    
    image = np.ones((target_size[1], target_size[0]), dtype=np.uint8) * 255
    
    if not os.path.exists(GDS_PATH):
        raise FileNotFoundError(f"GDS file not found: {GDS_PATH}")
    
    gds = gdspy.GdsLibrary().read_gds(GDS_PATH)
    cell = gds.top_level()[0]
    polygons = cell.get_polygons(by_spec=True)
    
    polygon_count = 0
    for layer in layers:
        if (layer, 0) in polygons:
            for poly in polygons[(layer, 0)]:
                polygon_count += 1
                
                norm_poly = (poly - [xmin, ymin]) * scale
                int_poly = np.round(norm_poly).astype(np.int32)
                int_poly += [offset_x, offset_y]
                int_poly[:, 1] = target_size[1] - 1 - int_poly[:, 1]  # Flip Y-axis
                
                cv2.fillPoly(image, [int_poly], color=(0,))
    
    print(f"Custom bounds generator rendered {polygon_count} polygons for structure {structure_num}")
    return image

def get_all_structures_info() -> Dict[int, Dict]:
    result = {}
    for structure_num in list_available_structures():
        info = get_structure_info(structure_num)
        if info:
            result[structure_num] = info
    return result
