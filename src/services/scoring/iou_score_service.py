"""
IOU Score Service - Intersection-over-Union Metrics Computation

This service provides comprehensive Intersection-over-Union (IoU) metrics
computation for binary image comparison, including multi-threshold analysis,
class-wise metrics, visualization, and local region analysis.

Main Class:
- IOUScoreService: Qt-based service for IoU metrics computation

Key Methods:
- compute_iou(): Computes basic IoU between two binary images
- compute_multi_threshold_iou(): Tests multiple threshold values
- compute_class_wise_iou(): Computes IoU for each class separately
- create_iou_visualization(): Creates color-coded IoU visualization
- get_last_result(): Returns most recent IoU computation result
- compute_local_iou(): Computes IoU for local image regions

Signals Emitted:
- iou_computed(dict): IoU computation completed successfully
- iou_error(str): IoU computation failed with error

Dependencies:
- Uses: numpy (array operations), cv2 (OpenCV image processing)
- Uses: PySide6.QtCore (QObject, Signal for Qt integration)
- Uses: core/utils.get_logger (logging functionality)
- Used by: Scoring services and alignment evaluation
- Used by: UI components for metric display and visualization

Metrics Computed:
- IoU Score: Intersection over Union ratio (0-1)
- Dice Coefficient: 2 * intersection / (area1 + area2)
- Precision: True positives / (true positives + false positives)
- Recall: True positives / (true positives + false negatives)
- F1 Score: Harmonic mean of precision and recall
- Coverage Ratio: Union area / total image area

Features:
- Multi-threshold analysis for optimal threshold selection
- Class-wise IoU for foreground and background separately
- Local region analysis with configurable grid size
- Color-coded visualization generation
- Statistical analysis of local IoU distributions
- Comprehensive error handling and logging
"""

from typing import Optional, Dict, List, Tuple
from PySide6.QtCore import QObject, Signal
import numpy as np
import cv2

from src.core.utils import get_logger


class IOUScoreService(QObject):
    """Service for computing Intersection-over-Union (IoU) metrics."""
    
    # Signals
    iou_computed = Signal(dict)  # Emitted when IoU computation completes
    iou_error = Signal(str)      # Emitted on computation error
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self._last_result = None
    
    def compute_iou(self, image1: np.ndarray, image2: np.ndarray, 
                   threshold: int = 127) -> Dict:
        """
        Compute Intersection-over-Union (IoU) between two binary images.
        
        Args:
            image1: First image (typically SEM)
            image2: Second image (typically GDS)
            threshold: Threshold for binarization
            
        Returns:
            Dictionary containing IoU metrics
        """
        try:
            # Ensure images are the same size
            if image1.shape != image2.shape:
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            
            # Convert to grayscale if needed
            if len(image1.shape) == 3:
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            if len(image2.shape) == 3:
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            # Binarize images
            _, binary1 = cv2.threshold(image1, threshold, 255, cv2.THRESH_BINARY)
            _, binary2 = cv2.threshold(image2, threshold, 255, cv2.THRESH_BINARY)
            
            # Convert to boolean masks (True = foreground/structure)
            mask1 = binary1 < threshold  # Assuming dark structures on light background
            mask2 = binary2 < threshold
            
            # Calculate intersection and union
            intersection = np.logical_and(mask1, mask2)
            union = np.logical_or(mask1, mask2)
            
            # Calculate IoU
            intersection_pixels = np.sum(intersection)
            union_pixels = np.sum(union)
            
            if union_pixels == 0:
                iou_score = 1.0 if intersection_pixels == 0 else 0.0
            else:
                iou_score = intersection_pixels / union_pixels
            
            # Additional metrics
            total_pixels = mask1.size
            pixels1 = np.sum(mask1)
            pixels2 = np.sum(mask2)
            
            # Precision and recall
            precision = intersection_pixels / max(1, pixels2)
            recall = intersection_pixels / max(1, pixels1)
            
            # F1 score (harmonic mean of precision and recall)
            f1_score = 2 * (precision * recall) / max(1e-10, precision + recall)
            
            # Dice coefficient (same as F1 score for binary classification)
            dice_coefficient = (2 * intersection_pixels) / max(1, pixels1 + pixels2)
            
            result = {
                'iou_score': float(iou_score),
                'iou_percentage': float(iou_score * 100),
                'dice_coefficient': float(dice_coefficient),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score),
                'intersection_pixels': int(intersection_pixels),
                'union_pixels': int(union_pixels),
                'total_pixels': int(total_pixels),
                'pixels_image1': int(pixels1),
                'pixels_image2': int(pixels2),
                'threshold_used': threshold,
                'coverage_ratio': float(union_pixels / total_pixels)
            }
            
            self._last_result = result
            self.iou_computed.emit(result)
            self.logger.info(f"Computed IoU: {iou_score:.4f} ({iou_score*100:.2f}%)")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to compute IoU: {e}"
            self.logger.error(error_msg)
            self.iou_error.emit(error_msg)
            return {}
    
    def compute_multi_threshold_iou(self, image1: np.ndarray, image2: np.ndarray,
                                   thresholds: List[int] = None) -> Dict:
        """
        Compute IoU for multiple threshold values.
        
        Args:
            image1: First image
            image2: Second image
            thresholds: List of threshold values to test
            
        Returns:
            Dictionary containing multi-threshold IoU results
        """
        if thresholds is None:
            thresholds = [64, 96, 127, 160, 192]
        
        try:
            threshold_results = {}
            iou_scores = []
            
            for threshold in thresholds:
                result = self.compute_iou(image1, image2, threshold)
                if result:
                    threshold_results[str(threshold)] = result
                    iou_scores.append(result['iou_score'])
            
            if iou_scores:
                best_threshold_idx = np.argmax(iou_scores)
                best_threshold = thresholds[best_threshold_idx]
                best_iou = iou_scores[best_threshold_idx]
                
                summary = {
                    'multi_threshold_results': threshold_results,
                    'best_threshold': best_threshold,
                    'best_iou': float(best_iou),
                    'mean_iou': float(np.mean(iou_scores)),
                    'std_iou': float(np.std(iou_scores)),
                    'thresholds_tested': thresholds
                }
                
                self.logger.info(f"Multi-threshold IoU: best={best_iou:.4f} at threshold={best_threshold}")
                return summary
            else:
                return {'error': 'Failed to compute IoU for any threshold'}
                
        except Exception as e:
            error_msg = f"Failed to compute multi-threshold IoU: {e}"
            self.logger.error(error_msg)
            self.iou_error.emit(error_msg)
            return {}
    
    def compute_class_wise_iou(self, image1: np.ndarray, image2: np.ndarray,
                              num_classes: int = 2, threshold: int = 127) -> Dict:
        """
        Compute IoU for each class separately.
        
        Args:
            image1: First image
            image2: Second image
            num_classes: Number of classes (2 for binary, more for multi-class)
            threshold: Threshold for binarization (only used for binary case)
            
        Returns:
            Dictionary containing class-wise IoU results
        """
        try:
            # Ensure images are the same size
            if image1.shape != image2.shape:
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            
            if num_classes == 2:
                # Binary case
                result = self.compute_iou(image1, image2, threshold)
                
                # Also compute IoU for background class
                if len(image1.shape) == 3:
                    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                if len(image2.shape) == 3:
                    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
                
                _, binary1 = cv2.threshold(image1, threshold, 255, cv2.THRESH_BINARY)
                _, binary2 = cv2.threshold(image2, threshold, 255, cv2.THRESH_BINARY)
                
                # Background masks (inverted)
                bg_mask1 = binary1 >= threshold
                bg_mask2 = binary2 >= threshold
                
                bg_intersection = np.sum(np.logical_and(bg_mask1, bg_mask2))
                bg_union = np.sum(np.logical_or(bg_mask1, bg_mask2))
                bg_iou = bg_intersection / max(1, bg_union)
                
                class_results = {
                    'foreground': {
                        'iou': result['iou_score'],
                        'precision': result['precision'],
                        'recall': result['recall']
                    },
                    'background': {
                        'iou': float(bg_iou),
                        'intersection': int(bg_intersection),
                        'union': int(bg_union)
                    },
                    'mean_iou': float((result['iou_score'] + bg_iou) / 2),
                    'num_classes': 2
                }
                
            else:
                # Multi-class case - simplified implementation
                # In practice, you'd need proper multi-class segmentation
                class_results = {
                    'error': 'Multi-class IoU not fully implemented',
                    'num_classes': num_classes
                }
            
            self.logger.info(f"Computed class-wise IoU for {num_classes} classes")
            return class_results
            
        except Exception as e:
            error_msg = f"Failed to compute class-wise IoU: {e}"
            self.logger.error(error_msg)
            self.iou_error.emit(error_msg)
            return {}
    
    def create_iou_visualization(self, image1: np.ndarray, image2: np.ndarray,
                                threshold: int = 127) -> Optional[np.ndarray]:
        """
        Create a visualization showing IoU regions.
        
        Args:
            image1: First image
            image2: Second image
            threshold: Threshold for binarization
            
        Returns:
            RGB image showing IoU regions in different colors
        """
        try:
            # Ensure images are the same size
            if image1.shape != image2.shape:
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            
            # Convert to grayscale if needed
            if len(image1.shape) == 3:
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            if len(image2.shape) == 3:
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            # Binarize images
            _, binary1 = cv2.threshold(image1, threshold, 255, cv2.THRESH_BINARY)
            _, binary2 = cv2.threshold(image2, threshold, 255, cv2.THRESH_BINARY)
            
            # Create masks
            mask1 = binary1 < threshold
            mask2 = binary2 < threshold
            
            # Create RGB visualization
            h, w = image1.shape
            visualization = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Background (white)
            visualization[:, :] = [255, 255, 255]
            
            # Image1 only (red)
            only1 = mask1 & ~mask2
            visualization[only1] = [255, 0, 0]
            
            # Image2 only (blue)  
            only2 = mask2 & ~mask1
            visualization[only2] = [0, 0, 255]
            
            # Intersection (green)
            intersection = mask1 & mask2
            visualization[intersection] = [0, 255, 0]
            
            return visualization
            
        except Exception as e:
            error_msg = f"Failed to create IoU visualization: {e}"
            self.logger.error(error_msg)
            self.iou_error.emit(error_msg)
            return None
    
    def get_last_result(self) -> Optional[Dict]:
        """Get the most recent IoU result."""
        return self._last_result
    
    def compute_local_iou(self, image1: np.ndarray, image2: np.ndarray,
                         grid_size: int = 64, threshold: int = 127) -> Dict:
        """
        Compute IoU for local regions of the image.
        
        Args:
            image1: First image
            image2: Second image
            grid_size: Size of grid cells for local analysis
            threshold: Threshold for binarization
            
        Returns:
            Dictionary containing local IoU statistics
        """
        try:
            # Ensure images are the same size
            if image1.shape != image2.shape:
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            
            h, w = image1.shape[:2]
            local_ious = []
            
            # Divide image into grid and compute IoU for each cell
            for y in range(0, h - grid_size, grid_size):
                for x in range(0, w - grid_size, grid_size):
                    # Extract local regions
                    region1 = image1[y:y+grid_size, x:x+grid_size]
                    region2 = image2[y:y+grid_size, x:x+grid_size]
                    
                    # Compute local IoU
                    local_result = self.compute_iou(region1, region2, threshold)
                    if local_result:
                        local_ious.append(local_result['iou_score'])
            
            if local_ious:
                local_ious = np.array(local_ious)
                
                result = {
                    'local_iou_mean': float(np.mean(local_ious)),
                    'local_iou_std': float(np.std(local_ious)),
                    'local_iou_min': float(np.min(local_ious)),
                    'local_iou_max': float(np.max(local_ious)),
                    'local_iou_median': float(np.median(local_ious)),
                    'num_regions': len(local_ious),
                    'grid_size': grid_size,
                    'regions_above_threshold': {
                        '0.1': int(np.sum(local_ious > 0.1)),
                        '0.3': int(np.sum(local_ious > 0.3)),
                        '0.5': int(np.sum(local_ious > 0.5)),
                        '0.7': int(np.sum(local_ious > 0.7))
                    }
                }
                
                self.logger.info(f"Computed local IoU statistics for {len(local_ious)} regions")
                return result
            else:
                return {'error': 'No valid regions found for local IoU analysis'}
                
        except Exception as e:
            error_msg = f"Failed to compute local IoU: {e}"
            self.logger.error(error_msg)
            self.iou_error.emit(error_msg)
            return {}
