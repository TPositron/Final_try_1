"""
GDS Display Generator
Generates display images from GDS structures based on working code implementation.
"""

import gdspy
import numpy as np
import cv2
import os
from typing import Dict, List, Tuple, Optional


def generate_display_gds(structure_num: int, target_size: Tuple[int, int] = (1024, 666)) -> np.ndarray:
    """Generate display GDS image for a given structure number."""
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
                    cv2.fillPoly(image, [int_poly], color=0)

    print(f"Display generator rendered {polygon_count} polygons for structure {structure_num}")
    return image


def get_structure_info(structure_num: int) -> Optional[Dict]:
    """Get structure information for a given structure number."""
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
    """Get the standard GDS file path."""
    return "C:\\Users\\tarik\\Image_analysis\\Data\\GDS\\Institute_Project_GDS1.gds"


def list_available_structures() -> List[int]:
    """List all available structure numbers."""
    return [1, 2, 3, 4, 5]


def get_all_structures_info() -> Dict[int, Dict]:
    """Get information for all available structures."""
    result = {}
    for structure_num in list_available_structures():
        info = get_structure_info(structure_num)
        if info:
            result[structure_num] = info
    return result
