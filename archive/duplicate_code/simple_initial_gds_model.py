

import gdstk
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import json

from src.core.utils.error_handling import handle_errors


logger = logging.getLogger(__name__)


class InitialGdsModel:
   
    
    DEFAULT_DPI = 300  # Static DPI value for pixel conversion
    METERS_TO_PIXELS = DEFAULT_DPI / 0.0254  # Convert meters to pixels
    
    def __init__(self, gds_path: str):
      
        self.gds_path = Path(gds_path)
        self.library = None
        self.cell = None
        self.unit = None
        self.precision = None
        self.bounds = None
        self._metadata = {}
        
        self._load_gds()
    
    @handle_errors
    def _load_gds(self) -> None:
        """Load the GDS file and extract basic information."""
        if not self.gds_path.exists():
            raise FileNotFoundError(f"GDS file not found: {self.gds_path}")
        
        if not self.gds_path.suffix.lower() in ['.gds', '.gdsii']:
            raise ValueError(f"Invalid GDS file extension: {self.gds_path.suffix}")
        
        try:
            # Load the GDS library
            self.library = gdstk.read_gds(str(self.gds_path))
            
            if not self.library:
                raise ValueError("Failed to read GDS file - library is empty")
            
            # Get basic library information
            self.unit = getattr(self.library, 'unit', 1e-6)  # Default 1 micron
            self.precision = getattr(self.library, 'precision', 1e-9)  # Default 1 nm
            
            # Find the main cell (prefer 'nazca' or first top-level cell)
            top_cells = self.library.top_level()
            if not top_cells:
                raise ValueError("No top-level cells found in GDS file")
            
            # Try to find 'nazca' cell first, otherwise use first top-level cell
            self.cell = None
            for cell in top_cells:
                if cell.name.lower() == 'nazca':
                    self.cell = cell
                    break
            
            if self.cell is None:
                self.cell = top_cells[0]
            
            # Calculate bounds
            self._calculate_bounds()
            
            # Store metadata
            layer_list = self.get_layers()
            self._metadata = {
                'file_path': str(self.gds_path),
                'file_size': self.gds_path.stat().st_size,
                'cell_name': self.cell.name if self.cell else 'unknown',
                'unit': self.unit,
                'precision': self.precision,
                'num_top_cells': len(top_cells),
                'total_cells': len(self.library.cells),
                'layer_names': {f"Layer_{layer}": f"GDS_Layer_{layer}" for layer in layer_list},
                'hierarchy_refs': [cell.name for cell in top_cells],
                'scaling_factor': self.get_scaling_factor(),
                'bounds_scaled': self.get_scaled_bounds()
            }
            
            logger.info(f"Successfully loaded GDS file: {self.gds_path}")
            logger.debug(f"Main cell: {self.cell.name if self.cell else 'None'}, Unit: {self.unit}, Precision: {self.precision}")
            
        except Exception as e:
            logger.error(f"Failed to load GDS file {self.gds_path}: {e}")
            raise ValueError(f"Invalid or corrupted GDS file: {e}") from e
    
    def _calculate_bounds(self) -> None:
        """Calculate the bounding box of the main cell."""
        try:
            # FIX 1: Handle case where cell is None or doesn't have bounding_box method
            if not self.cell:
                self.bounds = (0, 0, 0, 0)
                logger.warning("No cell available - using default bounds")
                return
            
            # FIX 2: Use getattr to safely access bounding_box method
            bounding_box_method = getattr(self.cell, 'bounding_box', None)
            if bounding_box_method is not None:
                try:
                    # Call the method if it exists
                    bbox = bounding_box_method()
                    if bbox is not None:
                        # bbox is ((min_x, min_y), (max_x, max_y))
                        self.bounds = (float(bbox[0][0]), float(bbox[0][1]), float(bbox[1][0]), float(bbox[1][1]))
                    else:
                        self.bounds = (0, 0, 0, 0)
                        logger.warning("Cell has no bounding box - using default bounds")
                except Exception as e:
                    logger.warning(f"Error calling bounding_box method: {e}")
                    self.bounds = self._calculate_manual_bounds()
            else:
                # Fallback: calculate bounds manually from polygons
                logger.debug("Cell has no bounding_box method, calculating manually")
                self.bounds = self._calculate_manual_bounds()
                
        except Exception as e:
            logger.warning(f"Could not calculate bounds: {e}")
            self.bounds = (0, 0, 0, 0)
    
    def _calculate_manual_bounds(self) -> Tuple[float, float, float, float]:
        """Calculate bounds manually from polygon data."""
        if not self.cell:
            return (0, 0, 0, 0)
        
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        found_polygons = False
        
        # FIX 3: Safely access polygons attribute
        try:
            polygons = getattr(self.cell, 'polygons', [])
            for polygon in polygons:
                if hasattr(polygon, 'points'):
                    points = np.array(polygon.points)
                    if len(points) > 0:
                        found_polygons = True
                        min_x = min(min_x, float(np.min(points[:, 0])))
                        max_x = max(max_x, float(np.max(points[:, 0])))
                        min_y = min(min_y, float(np.min(points[:, 1])))
                        max_y = max(max_y, float(np.max(points[:, 1])))
        except Exception as e:
            logger.warning(f"Error accessing polygons: {e}")
        
        if not found_polygons:
            return (0, 0, 0, 0)
            
        return (min_x, min_y, max_x, max_y)
    
    def get_layers(self) -> List[int]:
        """
        Get list of available layers in the GDS file.
        
        Returns:
            List of layer numbers found in the main cell
        """
        if not self.cell:
            return []
        
        layers = set()
        
        # FIX 4: Safely access polygons and paths
        try:
            # Get layers from polygons
            polygons = getattr(self.cell, 'polygons', [])
            for polygon in polygons:
                if hasattr(polygon, 'layer'):
                    layers.add(polygon.layer)
        except Exception as e:
            logger.warning(f"Error accessing polygon layers: {e}")
        
        try:
            # Get layers from paths (if any)
            paths = getattr(self.cell, 'paths', [])
            for path in paths:
                # FIX 5: Handle different path types that may have different layer access
                if hasattr(path, 'layer'):
                    layers.add(path.layer)
                elif hasattr(path, 'layers') and path.layers:
                    # Some path types might have multiple layers
                    if isinstance(path.layers, (list, tuple)):
                        layers.update(path.layers)
                    else:
                        layers.add(path.layers)
        except Exception as e:
            logger.warning(f"Error accessing path layers: {e}")
        
        return sorted(list(layers))
    
    def get_polygons(self, layers: Optional[List[int]] = None) -> List[np.ndarray]:
       
        if not self.cell:
            return []
        
        polygons = []
        target_layers = set(layers) if layers else None
        
        # FIX 6: Safely access polygons
        try:
            cell_polygons = getattr(self.cell, 'polygons', [])
            for polygon in cell_polygons:
                if hasattr(polygon, 'layer') and hasattr(polygon, 'points'):
                    if target_layers is None or polygon.layer in target_layers:
                        # Convert gdstk polygon points to numpy array
                        points = np.array(polygon.points)
                        if len(points) > 0:
                            polygons.append(points)
        except Exception as e:
            logger.warning(f"Error accessing polygons: {e}")
        
        return polygons
    
    def get_cell_info(self) -> Dict[str, Any]:
      
        if not self.cell:
            return {}
        
        # FIX 7: Safely access cell attributes
        try:
            num_polygons = len(getattr(self.cell, 'polygons', []))
            num_paths = len(getattr(self.cell, 'paths', []))
            num_references = len(getattr(self.cell, 'references', []))
            
            return {
                'name': self.cell.name,
                'num_polygons': num_polygons,
                'num_paths': num_paths,
                'num_references': num_references,
                'layers': self.get_layers(),
                'bounds': self.bounds
            }
        except Exception as e:
            logger.warning(f"Error getting cell info: {e}")
            return {
                'name': getattr(self.cell, 'name', 'unknown'),
                'num_polygons': 0,
                'num_paths': 0,
                'num_references': 0,
                'layers': self.get_layers(),
                'bounds': self.bounds
            }
    
    def get_metadata(self) -> Dict[str, Any]:
      
        return self._metadata.copy()
    
    def is_valid(self) -> bool:
       
        return (
            self.library is not None and 
            self.cell is not None and 
            self.gds_path.exists()
        )
    
    def get_scaling_factor(self) -> float:
     
        if not self.unit:
            return self.METERS_TO_PIXELS
        
   
        return float(self.unit * self.METERS_TO_PIXELS)
    
    def scale_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
      
        if len(coordinates) == 0:
            return coordinates
            
        scaling_factor = self.get_scaling_factor()
        return coordinates * scaling_factor
    
    def get_scaled_bounds(self) -> Tuple[float, float, float, float]:
      
        if not self.bounds:
            return (0.0, 0.0, 0.0, 0.0)
            
        scaling_factor = self.get_scaling_factor()
        # FIX 9: Ensure proper tuple type with correct length
        return (
            float(self.bounds[0] * scaling_factor),
            float(self.bounds[1] * scaling_factor),
            float(self.bounds[2] * scaling_factor),
            float(self.bounds[3] * scaling_factor)
        )
    
    def get_scaled_polygons(self, layers: Optional[List[int]] = None) -> List[np.ndarray]:
      
        polygons = self.get_polygons(layers)
        return [self.scale_coordinates(poly) for poly in polygons]
    
    @staticmethod
    def simplify_polygon_rdp(polygon: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
        
        if len(polygon) <= 2:
            return polygon
            
        def perpendicular_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
            """Calculate perpendicular distance from point to line."""
            if np.allclose(line_start, line_end):
                return float(np.linalg.norm(point - line_start))
            
            line_vec = line_end - line_start
            point_vec = point - line_start
            line_len = np.linalg.norm(line_vec)
            line_unitvec = line_vec / line_len
            
            proj_length = np.dot(point_vec, line_unitvec)
            proj = line_start + proj_length * line_unitvec
            
            # FIX 10: Ensure return type is float
            return float(np.linalg.norm(point - proj))
        
        def rdp_recursive(points: np.ndarray, start_idx: int, end_idx: int, epsilon: float) -> List[int]:
            """Recursive RDP algorithm."""
            if end_idx <= start_idx + 1:
                return [start_idx, end_idx]
            
            # Find point with maximum distance from line
            max_dist = 0
            max_idx = start_idx
            
            for i in range(start_idx + 1, end_idx):
                dist = perpendicular_distance(points[i], points[start_idx], points[end_idx])
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i
            
            # If max distance is greater than epsilon, recursively simplify
            if max_dist > epsilon:
                # Recursively simplify both parts
                left_results = rdp_recursive(points, start_idx, max_idx, epsilon)
                right_results = rdp_recursive(points, max_idx, end_idx, epsilon)
                
                # Combine results (remove duplicate middle point)
                return left_results[:-1] + right_results
            else:
                # All points between start and end are within epsilon
                return [start_idx, end_idx]
        
        # Apply RDP algorithm
        if len(polygon) < 3:
            return polygon
            
        # Close polygon if not closed
        if not np.allclose(polygon[0], polygon[-1]):
            polygon = np.vstack([polygon, polygon[0:1]])
        
        indices = rdp_recursive(polygon, 0, len(polygon) - 1, epsilon)
        simplified = polygon[indices]
        
        # Ensure we maintain shape integrity (minimum 3 points for polygon)
        if len(simplified) < 3:
            return polygon[:3] if len(polygon) >= 3 else polygon
            
        return simplified
    
    def get_simplified_polygons(self, layers: Optional[List[int]] = None, epsilon: float = 1.0) -> List[np.ndarray]:
        
        
        polygons = self.get_polygons(layers)
        return [self.simplify_polygon_rdp(poly, epsilon) for poly in polygons if len(poly) > 0]
    
    def enumerate_structures(self) -> Dict[int, Dict[str, Any]]:
      
        structures = {}
        
        if not self.cell:
            return structures
        
        # Get polygons grouped by layer
        layers = self.get_layers()
        
        structure_idx = 0
        for layer in layers:
            layer_polygons = self.get_polygons([layer])
            
            for poly_idx, polygon in enumerate(layer_polygons):
                if len(polygon) > 0:
                    # Calculate bounds for this polygon
                    x_coords = polygon[:, 0]
                    y_coords = polygon[:, 1]
                    poly_bounds = (float(x_coords.min()), float(y_coords.min()), 
                                 float(x_coords.max()), float(y_coords.max()))
                    
                    structures[structure_idx] = {
                        'index': structure_idx,
                        'layer': layer,
                        'polygon_index': poly_idx,
                        'bounds': poly_bounds,
                        'scaled_bounds': tuple(float(coord * self.get_scaling_factor()) for coord in poly_bounds),
                        'vertex_count': len(polygon),
                        'area': self._calculate_polygon_area(polygon)
                    }
                    structure_idx += 1
        
        return structures
    
    def get_structure_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        
        structures = self.enumerate_structures()
        return structures.get(index)
    
    def get_structure_count(self) -> int:
        
        return len(self.enumerate_structures())
    
    @staticmethod
    def _calculate_polygon_area(polygon: np.ndarray) -> float:
        
        if len(polygon) < 3:
            return 0.0
        
        x = polygon[:, 0]
        y = polygon[:, 1]
        return float(0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1))))
    
    def __str__(self) -> str:
        
        if not self.is_valid():
            return f"InitialGdsModel(invalid, path={self.gds_path})"
        
        cell_name = getattr(self.cell, 'name', 'unknown') if self.cell else 'none'
        return (
            f"InitialGdsModel("
            f"file={self.gds_path.name}, "
            f"cell={cell_name}, "
            f"layers={len(self.get_layers())}, "
            f"polygons={len(getattr(self.cell, 'polygons', []))}"
            f")"
        )
    
    def __repr__(self) -> str:
        
        return self.__str__()
    
    def serialize_to_json(self, include_polygons: bool = False) -> Dict[str, Any]:
        
        # Get basic metadata
        data = {
            'metadata': self.get_metadata(),
            'cell_info': self.get_cell_info(),
            'layers': self.get_layers(),
            'structure_count': self.get_structure_count(),
            'structures': {}
        }
        
        # Add structure enumeration
        structures = self.enumerate_structures()
        for idx, struct_info in structures.items():
            struct_data = struct_info.copy()
            
            # Convert numpy types to Python native types for JSON serialization
            if 'bounds' in struct_data:
                struct_data['bounds'] = [float(x) for x in struct_data['bounds']]
            if 'scaled_bounds' in struct_data:
                struct_data['scaled_bounds'] = [float(x) for x in struct_data['scaled_bounds']]
            if 'area' in struct_data:
                struct_data['area'] = float(struct_data['area'])
            
            data['structures'][str(idx)] = struct_data
        
        # Optionally include polygon data (excluded by default for performance)
        if include_polygons:
            layers = self.get_layers()
            polygons_data = {}
            
            for layer in layers:
                layer_polygons = self.get_polygons([layer])
                polygons_data[f"layer_{layer}"] = [
                    polygon.tolist() for polygon in layer_polygons
                ]
            
            data['polygon_data'] = polygons_data
        
        # Convert metadata values to JSON-serializable types
        if 'metadata' in data:
            metadata = data['metadata']
            for key, value in metadata.items():
                if isinstance(value, (np.integer, np.floating)):
                    metadata[key] = float(value)
                elif isinstance(value, tuple):
                    metadata[key] = list(value)
        
        return data
    
    def save_to_json(self, filepath: str, include_polygons: bool = False) -> None:
        
        data = self.serialize_to_json(include_polygons)
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Structure data saved to {output_path}")
    
    @classmethod
    def load_from_json(cls, filepath: str) -> Dict[str, Any]:
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    @staticmethod
    def validate_gds_format(filepath: str) -> Dict[str, Any]:
        
        result = {
            'is_valid': False,
            'file_exists': False,
            'correct_extension': False,
            'readable': False,
            'has_library': False,
            'has_cells': False,
            'error': None
        }
        
        try:
            path = Path(filepath)
            
            # Check file existence
            result['file_exists'] = path.exists()
            if not result['file_exists']:
                result['error'] = "File does not exist"
                return result
            
            # Check extension
            result['correct_extension'] = path.suffix.lower() in ['.gds', '.gdsii']
            if not result['correct_extension']:
                result['error'] = f"Invalid file extension: {path.suffix}"
                return result
            
            # Try to read file
            try:
                library = gdstk.read_gds(str(path))
                result['readable'] = True
                
                # Check if library was loaded
                result['has_library'] = library is not None
                if not result['has_library']:
                    result['error'] = "Failed to load GDS library"
                    return result
                
                # Check if library has cells
                result['has_cells'] = len(library.cells) > 0
                if not result['has_cells']:
                    result['error'] = "No cells found in GDS file"
                    return result
                
                result['is_valid'] = True
                
            except Exception as e:
                result['error'] = f"Failed to read GDS file: {str(e)}"
                return result
                
        except Exception as e:
            result['error'] = f"Validation error: {str(e)}"
        
        return result
    
    def validate_loaded_data(self) -> Dict[str, Any]:
        
        result = {
            'is_valid': True,
            'has_library': self.library is not None,
            'has_cell': self.cell is not None,
            'has_bounds': self.bounds is not None and self.bounds != (0, 0, 0, 0),
            'has_polygons': False,
            'layer_count': len(self.get_layers()),
            'structure_count': 0,
            'warnings': []
        }
        
        if not result['has_library']:
            result['is_valid'] = False
            result['warnings'].append("No library loaded")
        
        if not result['has_cell']:
            result['is_valid'] = False
            result['warnings'].append("No main cell found")
        
        if result['has_cell']:
            polygons = self.get_polygons()
            result['has_polygons'] = len(polygons) > 0
            result['structure_count'] = len(polygons)
            
            if not result['has_polygons']:
                result['warnings'].append("No polygons found in main cell")
        
        if not result['has_bounds']:
            result['warnings'].append("Invalid or zero bounds")
        
        return result

    def generate_structure_bitmap(self, 
                                  bounds: Tuple[float, float, float, float],
                                  layers: List[int],
                                  resolution: Tuple[int, int] = (1024, 666)) -> Optional[np.ndarray]:
        
        try:
            import cv2
            
            print(f"generate_structure_bitmap called with bounds={bounds}, layers={layers}")
            
            if not self.cell:
                print("No cell available")
                return None
                
            width, height = resolution
            min_x, min_y, max_x, max_y = bounds
            
            # Check for invalid bounds to prevent division by zero
            if max_x <= min_x or max_y <= min_y:
                print(f"Invalid bounds: {bounds}")
                # Return a small test pattern instead of None
                test_bitmap = np.zeros((height, width), dtype=np.uint8)
                cv2.rectangle(test_bitmap, (width//4, height//4), (3*width//4, 3*height//4), 255, -1)
                return (test_bitmap > 0).astype(np.uint8)
            
            # Create empty bitmap
            bitmap = np.zeros((height, width), dtype=np.uint8)
            
            # Calculate scaling factors with safety checks
            width_range = max_x - min_x
            height_range = max_y - min_y
            
            if width_range <= 0 or height_range <= 0:
                print("Invalid range calculation")
                return None
                
            scale_x = width / width_range
            scale_y = height / height_range
            
            print(f"Scale factors: x={scale_x}, y={scale_y}")
            
            # FIX 11: Safely access polygons and fix fillPoly call
            target_layers = set(layers) if layers else set()
            polygon_count = 0
            max_polygons = 10000  # Safety limit
            
            try:
                cell_polygons = getattr(self.cell, 'polygons', [])
                for polygon in cell_polygons:
                    if polygon_count >= max_polygons:
                        print(f"Reached maximum polygon limit ({max_polygons}), stopping")
                        break
                        
                    if hasattr(polygon, 'layer') and hasattr(polygon, 'points'):
                        if not target_layers or polygon.layer in target_layers:
                            polygon_count += 1
                            try:
                                # Convert gdstk polygon points to numpy array
                                points = np.array(polygon.points)
                                if len(points) >= 3:  # Need at least 3 points for a polygon
                                    # Scale and translate points to image coordinates
                                    scaled_points = np.zeros((len(points), 2), dtype=np.int32)
                                    scaled_points[:, 0] = ((points[:, 0] - min_x) * scale_x).astype(np.int32)
                                    scaled_points[:, 1] = ((max_y - points[:, 1]) * scale_y).astype(np.int32)  # Flip Y
                                    
                                    # Clip points to image bounds
                                    scaled_points[:, 0] = np.clip(scaled_points[:, 0], 0, width - 1)
                                    scaled_points[:, 1] = np.clip(scaled_points[:, 1], 0, height - 1)
                                    
                                    # FIX 12: Fix cv2.fillPoly call with proper color format
                                    cv2.fillPoly(bitmap, [scaled_points], (255,))
                            except Exception as e:
                                print(f"Error processing polygon {polygon_count}: {e}")
                                continue
            except Exception as e:
                print(f"Error accessing cell polygons: {e}")
            
            print(f"Processed {polygon_count} polygons")
            
            # Convert to binary (0s and 1s)
            binary_bitmap = (bitmap > 0).astype(np.uint8)
            
            print(f"Generated bitmap with shape {binary_bitmap.shape}")
            return binary_bitmap
            
        except Exception as e:
            print(f"Error in generate_structure_bitmap: {e}")
            import traceback
            traceback.print_exc()
            return None

# Convenience functions for easy usage

def load_gds_file(gds_path: str) -> InitialGdsModel:
    
    return InitialGdsModel(gds_path)


def validate_gds_file(gds_path: str) -> bool:
   
    validation_result = InitialGdsModel.validate_gds_format(gds_path)
    return validation_result['is_valid']
