"""
Alignment Service - GDS Structure Alignment and Transformation Operations

This service provides comprehensive alignment capabilities for GDS structures
with SEM images, including transformations, batch processing, scoring,
and result visualization.

Main Class:
- AlignmentService: Service for GDS-to-SEM alignment operations

Key Methods:
- apply_transformations(): Applies transformations to single structure
- apply_transformations_all_structures(): Applies transformations to all structures
- batch_alignment_search(): Searches for optimal alignment parameters
- batch_alignment_search_all_structures(): Batch search for all structures
- save_alignment_result(): Saves alignment results to files

Global Functions:
- create_alignment_service(): Factory function for service creation
- align_gds_to_sem(): Quick alignment function for single structure

Dependencies:
- Uses: numpy, cv2 (OpenCV), pathlib.Path (standard libraries)
- Uses: skimage.metrics.structural_similarity (optional SSIM scoring)
- Uses: core/models.SemImage (SEM image data model)
- Used by: UI alignment components and workflow services
- Used by: Batch processing and automated alignment tools

Transformation Operations:
- Translation: X/Y offset in pixels
- Rotation: Angle in degrees (-180 to 180)
- Scaling: Scale factor (positive values)
- Transparency: Overlay transparency (0-100%)
- Combined transformation matrix computation

Alignment Scoring:
- SSIM-based structural similarity scoring
- Correlation-based fallback scoring
- Normalized score computation (0.0 to 1.0)
- Robust error handling for scoring failures

Batch Processing:
- Parameter range search for optimal alignment
- Multi-structure batch processing
- Configurable search ranges for all parameters
- Best parameter selection based on scoring

Visualization:
- Overlay preview generation with transparency
- Difference map computation
- Transformed GDS visualization
- Result saving to image files
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional, Union, List, Any
from pathlib import Path
from src.core.models import SemImage

class AlignmentService:
    def __init__(self):
        self.canvas_width = 1024
        self.canvas_height = 666
        
    def apply_transformations(self, 
                            sem_image: 'SemImage',
                            structure_data: Dict[str, Tuple[np.ndarray, dict]],
                            structure_name: str,
                            x_offset: int = 0,
                            y_offset: int = 0,
                            rotation: float = 0.0,
                            scale: float = 1.0,
                            transparency: int = 70) -> Dict[str, Any]:
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
            transformed_gds = cv2.resize(transformed_gds, (self.canvas_width, self.canvas_height), interpolation=cv2.INTER_NEAREST)
        transformed_gds = self._apply_zoom(transformed_gds, scale)
        transformed_gds = self._apply_rotation(transformed_gds, rotation)
        transformed_gds = self._apply_translation(transformed_gds, x_offset, y_offset)
        transformation_matrix = self._compute_transformation_matrix(x_offset, y_offset, rotation, scale)
        
        # Handle SemImage.to_array() - provide fallback if method doesn't exist
        try:
            sem_array = sem_image.to_array()  # type: ignore
        except AttributeError:
            # Fallback: try other common attributes using getattr to avoid Pylance warnings
            sem_array = (getattr(sem_image, 'image', None) or 
                        getattr(sem_image, 'data', None) or 
                        getattr(sem_image, '_image', None) or 
                        np.array(sem_image))
        
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
                                           sem_image: 'SemImage',
                                           structure_data: Dict[str, Tuple[np.ndarray, dict]],
                                           x_offset: int = 0,
                                           y_offset: int = 0,
                                           rotation: float = 0.0,
                                           scale: float = 1.0,
                                           transparency: int = 70) -> Dict[str, Dict[str, Any]]:
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
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0,))  # type: ignore
    
    def _apply_rotation(self, image: np.ndarray, angle_degrees: float) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        return cv2.warpAffine(image, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0,))  # type: ignore
    
    def _apply_translation(self, image: np.ndarray, dx: int, dy: int) -> np.ndarray:
        h, w = image.shape[:2]
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0,))  # type: ignore
    
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
        
        combined_matrix = translate_back @ rotation_matrix @ scale_matrix @ translate_to_origin
        return combined_matrix
    
    def _compute_alignment_score(self, sem_array: np.ndarray, gds_array: np.ndarray) -> float:
        try:
            from skimage.metrics import structural_similarity as ssim
            if sem_array.shape != gds_array.shape:
                gds_array = cv2.resize(gds_array, (sem_array.shape[1], sem_array.shape[0]))
            
            sem_norm = (sem_array - sem_array.min()) / (sem_array.max() - sem_array.min() + 1e-8)
            gds_norm = (gds_array - gds_array.min()) / (gds_array.max() - gds_array.min() + 1e-8)
            
            score_result = ssim(sem_norm, gds_norm, data_range=1.0)
            # Handle different return types from ssim
            if isinstance(score_result, tuple):
                score = float(score_result[0])  # Take first element if tuple
            else:
                score = float(score_result)
            return max(0.0, score)
        except Exception:
            # Manual correlation calculation to avoid type issues
            return self._calculate_correlation_score(sem_array, gds_array)
    
    def _calculate_correlation_score(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate correlation-based alignment score."""
        try:
            # Ensure same shape
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Flatten arrays for correlation calculation
            flat1 = img1.flatten().astype(np.float64)
            flat2 = img2.flatten().astype(np.float64)
            
            # Calculate correlation coefficient manually
            mean1, mean2 = np.mean(flat1), np.mean(flat2)
            std1, std2 = np.std(flat1), np.std(flat2)
            
            if std1 == 0 or std2 == 0:
                return 0.0
            
            correlation = np.mean((flat1 - mean1) * (flat2 - mean2)) / (std1 * std2)
            
            return max(0.0, float(correlation)) if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0
    
    def _create_overlay_preview(self, sem_array: np.ndarray, gds_array: np.ndarray, transparency: int) -> np.ndarray:
        alpha = transparency / 100.0
        if sem_array.shape != gds_array.shape:
            gds_array = cv2.resize(gds_array, (sem_array.shape[1], sem_array.shape[0]))
        
        sem_norm = (sem_array / sem_array.max()) if sem_array.max() > 0 else sem_array
        gds_norm = (gds_array / gds_array.max()) if gds_array.max() > 0 else gds_array
        
        overlay = sem_norm * (1 - alpha) + gds_norm * alpha
        return np.clip(overlay, 0, 1)
    
    def _create_difference_map(self, sem_array: np.ndarray, gds_array: np.ndarray) -> np.ndarray:
        if sem_array.shape != gds_array.shape:
            gds_array = cv2.resize(gds_array, (sem_array.shape[1], sem_array.shape[0]))
        
        sem_norm = (sem_array / sem_array.max()) if sem_array.max() > 0 else sem_array
        gds_norm = (gds_array / gds_array.max()) if gds_array.max() > 0 else gds_array
        
        difference = np.abs(sem_norm - gds_norm)
        return difference
    
    def batch_alignment_search(self, sem_image: 'SemImage',
                             structure_data: Dict[str, Tuple[np.ndarray, dict]],
                             structure_name: str,
                             x_range: Tuple[int, int, int] = (-20, 21, 5),
                             y_range: Tuple[int, int, int] = (-20, 21, 5),
                             rotation_range: Tuple[float, float, float] = (-5.0, 5.5, 0.5),
                             scale_range: Tuple[float, float, float] = (0.9, 1.11, 0.05)) -> Dict[str, Any]:
        
        best_score = -1
        best_params = {}
        best_result = {}
        
        x_start, x_end, x_step = x_range
        y_start, y_end, y_step = y_range
        rot_start, rot_end, rot_step = rotation_range
        scale_start, scale_end, scale_step = scale_range
        
        for x in range(x_start, x_end, x_step):
            for y in range(y_start, y_end, y_step):
                for rotation in np.arange(rot_start, rot_end, rot_step):
                    for scale in np.arange(scale_start, scale_end, scale_step):
                        try:
                            result = self.apply_transformations(
                                sem_image, structure_data, structure_name,
                                x, y, float(rotation), float(scale))  # Convert numpy types to float
                            
                            score = result['alignment_score']
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'x_offset': x,
                                    'y_offset': y,
                                    'rotation': float(rotation),
                                    'scale': float(scale)
                                }
                                best_result = result
                        except Exception:
                            continue
        
        return {
            'best_score': best_score,
            'best_parameters': best_params,
            'best_result': best_result,
            'structure_name': structure_name
        }
    
    def batch_alignment_search_all_structures(self, sem_image: 'SemImage',
                                            structure_data: Dict[str, Tuple[np.ndarray, dict]],
                                            x_range: Tuple[int, int, int] = (-20, 21, 5),
                                            y_range: Tuple[int, int, int] = (-20, 21, 5),
                                            rotation_range: Tuple[float, float, float] = (-5.0, 5.5, 0.5),
                                            scale_range: Tuple[float, float, float] = (0.9, 1.11, 0.05)) -> Dict[str, Dict[str, Any]]:
        
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
    
    def save_alignment_result(self, result: Dict[str, Any], output_dir: Union[str, Path]) -> Dict[str, Path]:
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
        
        return saved_files


def create_alignment_service() -> AlignmentService:
    return AlignmentService()


def align_gds_to_sem(sem_image: 'SemImage', 
                    structure_data: Dict[str, Tuple[np.ndarray, dict]],
                    structure_name: str,
                    x_offset: int = 0,
                    y_offset: int = 0,
                    rotation: float = 0.0,
                    scale: float = 1.0,
                    transparency: int = 70) -> Dict[str, Any]:
    service = AlignmentService()
    return service.apply_transformations(
        sem_image, structure_data, structure_name, x_offset, y_offset, rotation, scale, transparency)
