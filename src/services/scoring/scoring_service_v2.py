import numpy as np
import cv2
from typing import Dict, Tuple, Optional, Union
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage import filters, morphology, segmentation
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
import warnings

from src.core.models.sem_image import SEMImage
from src.core.models.gds_model import GDSModel


class ScoringService:
    
    def __init__(self):
        self.edge_detection_params = {
            'gaussian_sigma': 1.0,
            'low_threshold': 0.1,
            'high_threshold': 0.2,
            'use_canny': True
        }
        
        self.binarization_params = {
            'method': 'otsu',
            'threshold_value': 0.5,
            'gaussian_sigma': 1.0
        }
    
    def compute_score(self, 
                     sem_img: Union[SEMImage, np.ndarray], 
                     gds_overlay: np.ndarray,
                     compute_diagnostics: bool = False) -> Dict[str, Union[float, np.ndarray]]:
        
        if isinstance(sem_img, SEMImage):
            sem_array = sem_img.to_array()
        else:
            sem_array = sem_img.copy()
        
        gds_array = self._normalize_array(gds_overlay)
        sem_array, gds_array = self._ensure_same_shape(sem_array, gds_array)
        
        scores = {}
        
        sem_binary = self._binarize_sem_image(sem_array)
        gds_binary = self._ensure_binary(gds_array)
        
        scores['edge_overlap'] = self._compute_edge_overlap(sem_array, gds_binary)
        scores['iou'] = self._compute_iou(sem_binary, gds_binary)
        scores['edge_distance'] = self._compute_edge_distance(sem_array, gds_binary)
        
        scores['composite_score'] = self._compute_composite_score(scores)
        
        if compute_diagnostics:
            diagnostics = self._compute_diagnostics(sem_array, gds_array, scores)
            scores.update(diagnostics)
        
        return scores
    
    def compute_score_with_legacy(self, 
                                 sem_img: Union[SEMImage, np.ndarray], 
                                 gds_overlay: np.ndarray,
                                 include_legacy: bool = False) -> Dict[str, Union[float, np.ndarray]]:
        
        scores = self.compute_score(sem_img, gds_overlay)
        
        if include_legacy:
            if isinstance(sem_img, SEMImage):
                sem_array = sem_img.to_array()
            else:
                sem_array = sem_img.copy()
            
            gds_array = self._normalize_array(gds_overlay)
            sem_array, gds_array = self._ensure_same_shape(sem_array, gds_array)
            
            scores['ssim'] = self._compute_ssim_legacy(sem_array, gds_array)
            scores['mse'] = self._compute_mse_legacy(sem_array, gds_array)
        
        return scores
    
    def _normalize_array(self, array: np.ndarray) -> np.ndarray:
        if array.dtype == np.uint8:
            return array.astype(np.float32) / 255.0
        elif array.dtype == np.uint16:
            return array.astype(np.float32) / 65535.0
        elif array.max() > 1.0:
            return array.astype(np.float32) / array.max()
        else:
            return array.astype(np.float32)
    
    def _ensure_same_shape(self, sem_array: np.ndarray, gds_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if sem_array.shape != gds_array.shape:
            target_shape = sem_array.shape
            if len(gds_array.shape) == 3 and gds_array.shape[2] == 3:
                gds_array = cv2.cvtColor(gds_array, cv2.COLOR_RGB2GRAY)
            
            if gds_array.shape != target_shape:
                gds_array = cv2.resize(gds_array, (target_shape[1], target_shape[0]), 
                                     interpolation=cv2.INTER_LINEAR)
        
        return sem_array, gds_array
    
    def _binarize_sem_image(self, sem_array: np.ndarray) -> np.ndarray:
        try:
            if self.binarization_params['method'] == 'otsu':
                if sem_array.dtype != np.uint8:
                    sem_8bit = (sem_array * 255).astype(np.uint8)
                else:
                    sem_8bit = sem_array
                
                threshold_val, binary = cv2.threshold(sem_8bit, 0, 255, 
                                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return binary > 0
            
            elif self.binarization_params['method'] == 'adaptive':
                if sem_array.dtype != np.uint8:
                    sem_8bit = (sem_array * 255).astype(np.uint8)
                else:
                    sem_8bit = sem_array
                
                binary = cv2.adaptiveThreshold(sem_8bit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
                return binary > 0
            
            else:
                threshold = self.binarization_params['threshold_value']
                return sem_array > threshold
                
        except Exception as e:
            warnings.warn(f"SEM binarization failed: {e}")
            return sem_array > 0.5
    
    def _ensure_binary(self, array: np.ndarray) -> np.ndarray:
        if array.dtype == bool:
            return array
        return array > 0.5
    
    def _compute_edge_overlap(self, sem_array: np.ndarray, gds_binary: np.ndarray) -> float:
        try:
            sem_edges = self._detect_edges(sem_array)
            gds_edges = self._detect_edges_binary(gds_binary)
            
            intersection = np.logical_and(sem_edges, gds_edges)
            union = np.logical_or(sem_edges, gds_edges)
            
            if np.sum(union) == 0:
                return 0.0
            
            overlap_ratio = np.sum(intersection) / np.sum(union)
            return float(overlap_ratio * 100.0)
            
        except Exception as e:
            warnings.warn(f"Edge overlap computation failed: {e}")
            return 0.0
    
    def _compute_iou(self, sem_binary: np.ndarray, gds_binary: np.ndarray) -> float:
        try:
            intersection = np.logical_and(sem_binary, gds_binary)
            union = np.logical_or(sem_binary, gds_binary)
            
            if np.sum(union) == 0:
                return 0.0
            
            iou = np.sum(intersection) / np.sum(union)
            return float(iou)
            
        except Exception as e:
            warnings.warn(f"IoU computation failed: {e}")
            return 0.0
    
    def _compute_edge_distance(self, sem_array: np.ndarray, gds_binary: np.ndarray) -> float:
        try:
            sem_edges = self._detect_edges(sem_array)
            gds_edges = self._detect_edges_binary(gds_binary)
            
            sem_coords = np.column_stack(np.where(sem_edges))
            gds_coords = np.column_stack(np.where(gds_edges))
            
            if len(sem_coords) == 0 or len(gds_coords) == 0:
                return float('inf')
            
            dist_1 = directed_hausdorff(sem_coords, gds_coords)[0]
            dist_2 = directed_hausdorff(gds_coords, sem_coords)[0]
            
            hausdorff_dist = max(dist_1, dist_2)
            return float(hausdorff_dist)
            
        except Exception as e:
            warnings.warn(f"Edge distance computation failed: {e}")
            return float('inf')
    
    def _compute_chamfer_distance(self, sem_array: np.ndarray, gds_binary: np.ndarray) -> float:
        try:
            sem_edges = self._detect_edges(sem_array)
            gds_edges = self._detect_edges_binary(gds_binary)
            
            sem_dist = ndimage.distance_transform_edt(~sem_edges)
            gds_dist = ndimage.distance_transform_edt(~gds_edges)
            
            chamfer_1 = np.mean(sem_dist[gds_edges])
            chamfer_2 = np.mean(gds_dist[sem_edges])
            
            if np.isnan(chamfer_1):
                chamfer_1 = 0.0
            if np.isnan(chamfer_2):
                chamfer_2 = 0.0
            
            return float((chamfer_1 + chamfer_2) / 2.0)
            
        except Exception as e:
            warnings.warn(f"Chamfer distance computation failed: {e}")
            return float('inf')
    
    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        if self.edge_detection_params['use_canny']:
            image_8bit = (image * 255).astype(np.uint8)
            edges = cv2.Canny(image_8bit, 
                            int(self.edge_detection_params['low_threshold'] * 255),
                            int(self.edge_detection_params['high_threshold'] * 255))
            return edges > 0
        else:
            sigma = self.edge_detection_params['gaussian_sigma']
            smoothed = filters.gaussian(image, sigma=sigma)
            edges = filters.sobel(smoothed)
            threshold = self.edge_detection_params['low_threshold']
            return edges > threshold
    
    def _detect_edges_binary(self, binary_image: np.ndarray) -> np.ndarray:
        try:
            binary_uint8 = binary_image.astype(np.uint8) * 255
            edges = cv2.Canny(binary_uint8, 50, 150)
            return edges > 0
        except Exception:
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            edges = cv2.filter2D(binary_image.astype(np.uint8), -1, kernel)
            return edges > 0
    
    def _compute_composite_score(self, scores: Dict[str, float]) -> float:
        edge_overlap = scores.get('edge_overlap', 0.0) / 100.0
        iou_score = scores.get('iou', 0.0)
        edge_distance = scores.get('edge_distance', float('inf'))
        
        if edge_distance == float('inf'):
            distance_normalized = 0.0
        else:
            distance_normalized = 1.0 / (1.0 + edge_distance / 10.0)
        
        weights = {'edge_overlap': 0.4, 'iou': 0.4, 'edge_distance': 0.2}
        
        composite = (weights['edge_overlap'] * edge_overlap + 
                    weights['iou'] * iou_score + 
                    weights['edge_distance'] * distance_normalized)
        
        return float(composite)
    
    def _compute_ssim_legacy(self, sem_array: np.ndarray, gds_array: np.ndarray) -> float:
        try:
            default_ssim_params = {
                'win_size': 7,
                'data_range': 1.0,
                'channel_axis': None,
                'gaussian_weights': True,
                'use_sample_covariance': False
            }
            return float(ssim(sem_array, gds_array, **default_ssim_params))
        except Exception as e:
            warnings.warn(f"Legacy SSIM computation failed: {e}")
            return 0.0
    
    def _compute_mse_legacy(self, sem_array: np.ndarray, gds_array: np.ndarray) -> float:
        try:
            return float(mse(sem_array, gds_array))
        except Exception as e:
            warnings.warn(f"Legacy MSE computation failed: {e}")
            return float('inf')
    
    def _compute_diagnostics(self, sem_array: np.ndarray, gds_array: np.ndarray, 
                           scores: Dict[str, float]) -> Dict[str, np.ndarray]:
        diagnostics = {}
        
        try:
            sem_binary = self._binarize_sem_image(sem_array)
            gds_binary = self._ensure_binary(gds_array)
            
            diagnostics['sem_binary'] = sem_binary
            diagnostics['gds_binary'] = gds_binary
            
            diagnostics['overlap_map'] = np.logical_and(sem_binary, gds_binary)
            diagnostics['difference_map'] = np.logical_xor(sem_binary, gds_binary)
            
            diagnostics['edge_overlay'] = self._create_edge_overlay(sem_array, gds_binary)
            
            diagnostics['distance_map'] = self._compute_distance_map(sem_array, gds_binary)
            
        except Exception as e:
            warnings.warn(f"Diagnostics computation failed: {e}")
        
        return diagnostics
    
    def _create_edge_overlay(self, sem_array: np.ndarray, gds_binary: np.ndarray) -> np.ndarray:
        sem_edges = self._detect_edges(sem_array)
        gds_edges = self._detect_edges_binary(gds_binary)
        
        overlay = np.zeros((*sem_array.shape, 3), dtype=np.float32)
        overlay[:, :, 0] = sem_edges.astype(np.float32)
        overlay[:, :, 1] = gds_edges.astype(np.float32)
        overlay[:, :, 2] = np.logical_and(sem_edges, gds_edges).astype(np.float32)
        
        return overlay
    
    def _compute_distance_map(self, sem_array: np.ndarray, gds_binary: np.ndarray) -> np.ndarray:
        try:
            sem_edges = self._detect_edges(sem_array)
            gds_edges = self._detect_edges_binary(gds_binary)
            
            sem_dist = ndimage.distance_transform_edt(~sem_edges)
            gds_dist = ndimage.distance_transform_edt(~gds_edges)
            
            combined_dist = np.minimum(sem_dist, gds_dist)
            return combined_dist
            
        except Exception:
            return np.zeros_like(sem_array)
    
    def compute_region_scores(self, 
                            sem_img: Union[SEMImage, np.ndarray],
                            gds_overlay: np.ndarray,
                            regions: list) -> Dict[str, Dict[str, float]]:
        
        if isinstance(sem_img, SEMImage):
            sem_array = sem_img.to_array()
        else:
            sem_array = sem_img.copy()
        
        gds_array = self._normalize_array(gds_overlay)
        sem_array, gds_array = self._ensure_same_shape(sem_array, gds_array)
        
        region_scores = {}
        
        for i, region in enumerate(regions):
            if isinstance(region, tuple) and len(region) == 4:
                x1, y1, x2, y2 = region
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(sem_array.shape[1], x2), min(sem_array.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    sem_region = sem_array[y1:y2, x1:x2]
                    gds_region = gds_array[y1:y2, x1:x2]
                    
                    region_name = f"region_{i}"
                    region_scores[region_name] = self.compute_score(sem_region, gds_region)
        
        return region_scores
    
    def compute_multi_scale_scores(self, 
                                 sem_img: Union[SEMImage, np.ndarray],
                                 gds_overlay: np.ndarray,
                                 scales: list = [1.0, 0.5, 0.25]) -> Dict[str, Dict[str, float]]:
        
        if isinstance(sem_img, SEMImage):
            sem_array = sem_img.to_array()
        else:
            sem_array = sem_img.copy()
        
        gds_array = self._normalize_array(gds_overlay)
        sem_array, gds_array = self._ensure_same_shape(sem_array, gds_array)
        
        multi_scale_scores = {}
        
        for scale in scales:
            if scale == 1.0:
                scaled_sem = sem_array
                scaled_gds = gds_array
            else:
                new_height = int(sem_array.shape[0] * scale)
                new_width = int(sem_array.shape[1] * scale)
                
                scaled_sem = cv2.resize(sem_array, (new_width, new_height), 
                                      interpolation=cv2.INTER_AREA)
                scaled_gds = cv2.resize(gds_array, (new_width, new_height), 
                                      interpolation=cv2.INTER_AREA)
            
            scale_name = f"scale_{scale:.2f}"
            multi_scale_scores[scale_name] = self.compute_score(scaled_sem, scaled_gds)
        
        return multi_scale_scores
    
    def set_edge_params(self, **params):
        self.edge_detection_params.update(params)
    
    def set_binarization_params(self, **params):
        self.binarization_params.update(params)
    
    def get_score_summary(self, scores: Dict[str, Union[float, np.ndarray]]) -> Dict[str, float]:
        summary = {}
        for key, value in scores.items():
            if isinstance(value, (int, float)) and not isinstance(value, np.ndarray):
                summary[key] = float(value)
        return summary


def create_scoring_service() -> ScoringService:
    return ScoringService()


def compute_alignment_score(sem_img: Union[SEMImage, np.ndarray], 
                          gds_overlay: np.ndarray) -> float:
    service = ScoringService()
    scores = service.compute_score(sem_img, gds_overlay)
    return scores.get('composite_score', 0.0)


def batch_score_images(sem_images: list, gds_overlays: list) -> Dict[str, Dict[str, float]]:
    service = ScoringService()
    batch_scores = {}
    
    for i, (sem_img, gds_overlay) in enumerate(zip(sem_images, gds_overlays)):
        try:
            scores = service.compute_score(sem_img, gds_overlay)
            summary = service.get_score_summary(scores)
            batch_scores[f"pair_{i}"] = summary
        except Exception as e:
            warnings.warn(f"Failed to score image pair {i}: {e}")
            batch_scores[f"pair_{i}"] = {'error': str(e)}
    
    return batch_scores


class ScoringError(Exception):
    pass