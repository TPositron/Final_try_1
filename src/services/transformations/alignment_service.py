import numpy as np
import cv2
from typing import Tuple, Dict, Optional, Union, List
from pathlib import Path
from src.image_analysis.core.models.sem_image import SEMImage

class AlignmentService:
    def __init__(self):
        self.canvas_width = 1024
        self.canvas_height = 666
        
    def apply_transformations(self, 
                            sem_image: 'SEMImage',
                            structure_data: Dict[str, Tuple[np.ndarray, dict]],
                            structure_name: str,
                            x_offset: int = 0,
                            y_offset: int = 0,
                            rotation: float = 0.0,
                            scale: float = 1.0,
                            transparency: int = 70) -> Dict:
        """
        Apply transformations to a single structure overlay and return alignment results.
        structure_data: {structure_name: (binary_image, coordinates)}
        """
        if not (-180 <= rotation <= 180):
            raise ValueError("Rotation must be between -180 and 180 degrees.")
        if scale <= 0:
            raise ValueError("Scale must be positive.")
        if structure_name not in structure_data:
            raise ValueError(f"Structure '{structure_name}' not found in structure_data.")
        binary_image, coordinates = structure_data[structure_name]
        transformed_gds = binary_image.copy().astype(np.float32)
        if transformed_gds.shape != (self.canvas_height, self.canvas_width):
            # Resize to canvas if needed
            import cv2
            transformed_gds = cv2.resize(transformed_gds, (self.canvas_width, self.canvas_height), interpolation=cv2.INTER_NEAREST)
        transformed_gds = self._apply_zoom(transformed_gds, scale)
        transformed_gds = self._apply_rotation(transformed_gds, rotation)
        transformed_gds = self._apply_translation(transformed_gds, x_offset, y_offset)
        transformation_matrix = self._compute_transformation_matrix(x_offset, y_offset, rotation, scale)
        sem_array = sem_image.to_array()
        alignment_score = self._compute_alignment_score(sem_array, transformed_gds)
        overlay_preview = self._create_overlay_preview(sem_array, transformed_gds, transparency)
        difference_map = self._create_difference_map(sem_array, transformed_gds)
        return {
            'transformed_gds': transformed_gds,
            'transformation_matrix': transformation_matrix,
            'alignment_score': alignment_score,
            'overlay_preview': overlay_preview,
            'difference_map': difference_map,
            'structure_name': structure_name,
            'coordinates': coordinates,
            'parameters': {
                'x_offset': x_offset,
                'y_offset': y_offset,
                'rotation': rotation,
                'scale': scale,
                'transparency': transparency
            }
        }

    def apply_transformations_all_structures(self,
                                           sem_image: 'SEMImage',
                                           structure_data: Dict[str, Tuple[np.ndarray, dict]],
                                           x_offset: int = 0,
                                           y_offset: int = 0,
                                           rotation: float = 0.0,
                                           scale: float = 1.0,
                                           transparency: int = 70) -> Dict[str, Dict]:
        """
        Apply transformations to all structures in structure_data.
        """
        results = {}
        for structure_name in structure_data.keys():
            results[structure_name] = self.apply_transformations(
                sem_image, structure_data, structure_name, x_offset, y_offset, rotation, scale, transparency)
        return results
    
    def _apply_zoom(self, image: np.ndarray, scale: float) -> np.ndarray:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    def _apply_rotation(self, image: np.ndarray, angle_degrees: float) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        return cv2.warpAffine(image, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    def _apply_translation(self, image: np.ndarray, dx: int, dy: int) -> np.ndarray:
        h, w = image.shape[:2]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    def _compute_transformation_matrix(self, x_offset: int, y_offset: int, 
                                     rotation: float, scale: float) -> np.ndarray:
        center_x, center_y = self.canvas_width // 2, self.canvas_height // 2
        
        scale_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        
        angle_rad = np.radians(rotation)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        translate_to_origin = np.array([
            [1, 0, -center_x],
            [0, 1, -center_y],
            [0, 0, 1]
        ])
        
        translate_back = np.array([
            [1, 0, center_x + x_offset],
            [0, 1, center_y + y_offset],
            [0, 0, 1]
        ])
        
        combined = translate_back @ rotation_matrix @ scale_matrix @ translate_to_origin
        return combined[:2, :]
    
    def _compute_alignment_score(self, sem_image: np.ndarray, 
                               gds_overlay: np.ndarray) -> float:
        sem_edges = cv2.Canny((sem_image * 255).astype(np.uint8), 50, 150)
        gds_edges = cv2.Canny((gds_overlay * 255).astype(np.uint8), 50, 150)
        
        overlap = np.logical_and(sem_edges > 0, gds_edges > 0).sum()
        total_gds_edges = (gds_edges > 0).sum()
        
        if total_gds_edges == 0:
            return 0.0
        
        score = overlap / total_gds_edges
        return float(score)
    
    def _create_overlay_preview(self, sem_image: np.ndarray, 
                              gds_overlay: np.ndarray, 
                              transparency: int) -> np.ndarray:
        alpha = transparency / 100.0
        
        sem_normalized = (sem_image * 255).astype(np.uint8)
        gds_normalized = (gds_overlay * 255).astype(np.uint8)
        
        overlay = cv2.addWeighted(sem_normalized, 1.0, gds_normalized, alpha, 0)
        return overlay.astype(np.float32) / 255.0
    
    def _create_difference_map(self, sem_image: np.ndarray, 
                             gds_overlay: np.ndarray) -> np.ndarray:
        return np.abs(sem_image - gds_overlay)
    
    def batch_alignment_search(self, sem_image: 'SEMImage', 
                             structure_data: Dict[str, Tuple[np.ndarray, dict]],
                             structure_name: str,
                             x_range: Tuple[int, int, int] = (-20, 21, 5),
                             y_range: Tuple[int, int, int] = (-20, 21, 5),
                             rotation_range: Tuple[float, float, float] = (-5.0, 5.5, 0.5),
                             scale_range: Tuple[float, float, float] = (0.9, 1.11, 0.05)) -> Dict:
        
        if structure_name not in structure_data:
            raise ValueError(f"Structure '{structure_name}' not found in structure data")
        
        best_score = -1
        best_params = None
        best_result = None
        all_results = []
        
        x_values = range(x_range[0], x_range[1], x_range[2])
        y_values = range(y_range[0], y_range[1], y_range[2])
        rotation_values = np.arange(rotation_range[0], rotation_range[1], rotation_range[2])
        scale_values = np.arange(scale_range[0], scale_range[1], scale_range[2])
        
        total_iterations = len(x_values) * len(y_values) * len(rotation_values) * len(scale_values)
        iteration = 0
        
        for x_offset in x_values:
            for y_offset in y_values:
                for rotation in rotation_values:
                    for scale in scale_values:
                        try:
                            result = self.apply_transformations(
                                sem_image, structure_data, structure_name, 
                                x_offset, y_offset, rotation, scale)
                            
                            score = result['alignment_score']
                            
                            result_summary = {
                                'parameters': result['parameters'],
                                'score': score,
                                'structure_name': structure_name
                            }
                            all_results.append(result_summary)
                            
                            if score > best_score:
                                best_score = score
                                best_params = result['parameters'].copy()
                                best_result = result
                            
                            iteration += 1
                            
                        except Exception as e:
                            iteration += 1
                            continue
        
        return {
            'best_result': best_result,
            'best_parameters': best_params,
            'best_score': best_score,
            'all_results': all_results,
            'total_tested': len(all_results),
            'structure_name': structure_name
        }
    
    def batch_alignment_search_all_structures(self, sem_image: 'SEMImage',
                                            structure_data: Dict[str, Tuple[np.ndarray, dict]],
                                            x_range: Tuple[int, int, int] = (-20, 21, 5),
                                            y_range: Tuple[int, int, int] = (-20, 21, 5),
                                            rotation_range: Tuple[float, float, float] = (-5.0, 5.5, 0.5),
                                            scale_range: Tuple[float, float, float] = (0.9, 1.11, 0.05)) -> Dict[str, Dict]:
        
        results = {}
        for structure_name in structure_data.keys():
            try:
                result = self.batch_alignment_search(
                    sem_image, structure_data, structure_name,
                    x_range, y_range, rotation_range, scale_range)
                results[structure_name] = result
            except Exception as e:
                results[structure_name] = {'error': str(e)}
        
        return results
    
    def save_alignment_result(self, result: Dict, output_dir: Union[str, Path]) -> Dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        structure_name = result.get('structure_name', 'unknown')
        
        if 'transformed_gds' in result:
            gds_path = output_dir / f'transformed_gds_{structure_name}.png'
            gds_image = (result['transformed_gds'] * 255).astype(np.uint8)
            cv2.imwrite(str(gds_path), gds_image)
            saved_files['transformed_gds'] = gds_path
        
        if 'overlay_preview' in result:
            overlay_path = output_dir / f'overlay_preview_{structure_name}.png'
            overlay_image = (result['overlay_preview'] * 255).astype(np.uint8)
            cv2.imwrite(str(overlay_path), overlay_image)
            saved_files['overlay_preview'] = overlay_path
        
        if 'difference_map' in result:
            diff_path = output_dir / f'difference_map_{structure_name}.png'
            diff_image = (result['difference_map'] * 255).astype(np.uint8)
            cv2.imwrite(str(diff_path), diff_image)
            saved_files['difference_map'] = diff_path
        
        params_path = output_dir / f'alignment_parameters_{structure_name}.txt'
        with open(params_path, 'w') as f:
            f.write(f"Alignment Parameters for {structure_name}:\n")
            f.write("=" * 40 + "\n")
            for key, value in result.get('parameters', {}).items():
                f.write(f"{key}: {value}\n")
            f.write(f"\nAlignment Score: {result.get('alignment_score', 'N/A')}\n")
            if 'coordinates' in result:
                f.write(f"Structure Coordinates: {result['coordinates']}\n")
        saved_files['parameters'] = params_path
        
        return saved_files


def create_alignment_service() -> AlignmentService:
    return AlignmentService()


def align_gds_to_sem(sem_image: 'SEMImage', 
                    structure_data: Dict[str, Tuple[np.ndarray, dict]],
                    structure_name: str,
                    x_offset: int = 0,
                    y_offset: int = 0,
                    rotation: float = 0.0,
                    scale: float = 1.0,
                    transparency: int = 70) -> Dict:
    service = AlignmentService()
    return service.apply_transformations(
        sem_image, structure_data, structure_name, x_offset, y_offset, rotation, scale, transparency)


def find_best_alignment(sem_image: 'SEMImage', 
                       structure_data: Dict[str, Tuple[np.ndarray, dict]],
                       structure_name: str,
                       search_params: Optional[Dict] = None) -> Dict:
    service = AlignmentService()
    
    if search_params is None:
        search_params = {}
    
    x_range = search_params.get('x_range', (-10, 11, 2))
    y_range = search_params.get('y_range', (-10, 11, 2))
    rotation_range = search_params.get('rotation_range', (-2.0, 2.5, 0.5))
    scale_range = search_params.get('scale_range', (0.95, 1.06, 0.05))
    
    return service.batch_alignment_search(
        sem_image, structure_data, structure_name, x_range, y_range, rotation_range, scale_range)


def find_best_alignment_all_structures(sem_image: 'SEMImage',
                                     structure_data: Dict[str, Tuple[np.ndarray, dict]],
                                     search_params: Optional[Dict] = None) -> Dict[str, Dict]:
    service = AlignmentService()
    
    if search_params is None:
        search_params = {}
    
    x_range = search_params.get('x_range', (-10, 11, 2))
    y_range = search_params.get('y_range', (-10, 11, 2))
    rotation_range = search_params.get('rotation_range', (-2.0, 2.5, 0.5))
    scale_range = search_params.get('scale_range', (0.95, 1.06, 0.05))
    
    return service.batch_alignment_search_all_structures(
        sem_image, structure_data, x_range, y_range, rotation_range, scale_range)