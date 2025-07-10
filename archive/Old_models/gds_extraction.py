"""GDS extraction utilities for structure information and frame extraction."""

import gdspy
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def get_structure_info(gds_path: str) -> Dict:
    """
    Extract structure information from a GDS file.
    
    Args:
        gds_path: Path to the GDS file
        
    Returns:
        Dictionary containing structure information including layers, bounds, etc.
    """
    try:
        lib = gdspy.GdsLibrary()
        lib.read_gds(gds_path)
        
        # Get top cell
        cell = lib.cells.get('nazca', next(iter(lib.top_level())))
        if not cell:
            raise ValueError("No valid cell found in GDS file")
        
        # Get all polygons by layer
        polygons = cell.get_polygons(by_spec=True)
        layers = [key[0] for key in polygons.keys()]
        
        # Get overall bounds
        bounds = cell.get_bounding_box()
        if bounds is not None:
            bounds = bounds.flatten()
        else:
            bounds = [0, 0, 0, 0]
        
        # Count polygons per layer
        layer_stats = {}
        for layer in layers:
            key = (layer, 0)
            layer_stats[layer] = len(polygons.get(key, []))
        
        return {
            'layers': layers,
            'bounds': bounds.tolist(),
            'layer_stats': layer_stats,
            'total_polygons': sum(layer_stats.values())
        }
        
    except Exception as e:
        raise ValueError(f"Failed to extract structure info: {e}")


def extract_frame(gds_path: str, bounds: Tuple[float, float, float, float], 
                 layers: Optional[List[int]] = None) -> Dict:
    """
    Extract a specific frame/region from a GDS file.
    
    Args:
        gds_path: Path to the GDS file
        bounds: Tuple of (xmin, ymin, xmax, ymax) defining the frame
        layers: List of layer numbers to extract. If None, extracts all layers.
        
    Returns:
        Dictionary containing extracted polygon data for the frame
    """
    try:
        lib = gdspy.GdsLibrary()
        lib.read_gds(gds_path)
        
        # Get top cell
        cell = lib.cells.get('nazca', next(iter(lib.top_level())))
        if not cell:
            raise ValueError("No valid cell found in GDS file")
        
        # Get all polygons by layer
        all_polygons = cell.get_polygons(by_spec=True)
        
        # If no layers specified, use all available layers
        if layers is None:
            layers = [key[0] for key in all_polygons.keys()]
        
        # Extract polygons within bounds
        xmin, ymin, xmax, ymax = bounds
        extracted_polygons = {}
        
        for layer in layers:
            key = (layer, 0)
            if key not in all_polygons:
                continue
                
            layer_polygons = []
            for poly in all_polygons[key]:
                # Check if polygon intersects with bounds
                poly_min = np.min(poly, axis=0)
                poly_max = np.max(poly, axis=0)
                
                if not (poly_max[0] < xmin or poly_min[0] > xmax or 
                       poly_max[1] < ymin or poly_min[1] > ymax):
                    layer_polygons.append(poly)
            
            if layer_polygons:
                extracted_polygons[layer] = layer_polygons
        
        return {
            'bounds': bounds,
            'layers': list(extracted_polygons.keys()),
            'polygons': extracted_polygons,
            'polygon_count': sum(len(polys) for polys in extracted_polygons.values())
        }
        
    except Exception as e:
        raise ValueError(f"Failed to extract frame: {e}")


def find_structures_by_pattern(gds_path: str, pattern_bounds: Tuple[float, float, float, float],
                              search_region: Optional[Tuple[float, float, float, float]] = None) -> List[Dict]:
    """
    Find structures in the GDS that match a given pattern within specified bounds.
    
    Args:
        gds_path: Path to the GDS file
        pattern_bounds: Bounds defining the pattern size (width, height)
        search_region: Optional region to limit the search. If None, searches entire GDS.
        
    Returns:
        List of dictionaries containing found structure locations and information
    """
    try:
        lib = gdspy.GdsLibrary()
        lib.read_gds(gds_path)
        
        cell = lib.cells.get('nazca', next(iter(lib.top_level())))
        if not cell:
            raise ValueError("No valid cell found in GDS file")
        
        # Get overall bounds if no search region specified
        if search_region is None:
            bounds = cell.get_bounding_box()
            if bounds is not None:
                search_region = bounds.flatten()
            else:
                return []
        
        pattern_width = pattern_bounds[2] - pattern_bounds[0]
        pattern_height = pattern_bounds[3] - pattern_bounds[1]
        
        # This is a simplified structure finder - in practice you might want
        # more sophisticated pattern matching
        structures = []
        
        # For now, just return the search region as a single structure
        # This can be expanded with actual pattern matching logic
        structures.append({
            'bounds': search_region,
            'pattern_match_score': 1.0,
            'center': [(search_region[0] + search_region[2]) / 2,
                      (search_region[1] + search_region[3]) / 2]
        })
        
        return structures
        
    except Exception as e:
        raise ValueError(f"Failed to find structures: {e}")


def get_predefined_structure_info(structure_num: int) -> Optional[Dict]:
    """
    Retrieve metadata for a predefined GDS structure.If you want to add more structures,
    you can extend the `structures` dictionary below.

    Args:
        structure_num: Integer index of the structure.

    Returns:
        Dictionary containing structure metadata, or None if not found.
    """
    structures = {
        1: {
            'name': 'Circpol_T2',
            'bounds': (688.55, 5736.55, 760.55, 5807.1),
            'layers': [14],
            'scale': 0.1
        },
        2: {
            'name': 'IP935Left_11',
            'bounds': (693.99, 6406.40, 723.59, 6428.96),
            'layers': [1, 2],
            'scale': 0.2
        },
        3: {
            'name': 'IP935Left_14',
            'bounds': (980.959, 6025.959, 1001.770, 6044.979),
            'layers': [1],
            'scale': 0.15
        },
        4: {
            'name': 'QC855GC_CROSS_Bottom',
            'bounds': (3730.00, 4700.99, 3756.00, 4760.00),
            'layers': [1, 2],
            'scale': 0.25
        },
        5: {
            'name': 'QC935_46',
            'bounds': (7195.558, 5046.99, 7203.99, 5055.33964),
            'layers': [1],
            'scale': 0.3
        }
    }

    return structures.get(structure_num)
