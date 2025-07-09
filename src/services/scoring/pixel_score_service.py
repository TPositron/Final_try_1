"""
Pixel Score Service - Pixel-Level Overlap and Difference Analysis

This service provides comprehensive pixel-level analysis for image comparison,
including overlap metrics, difference calculations, and visualization generation
for SEM/GDS alignment evaluation.

Main Class:
- PixelScoreService: Qt-based service for pixel-level scoring operations

Key Methods:
- compute_pixel_overlap(): Computes pixel overlap between binary images
- compute_pixel_difference(): Computes pixel-wise differences between images
- create_difference_overlay(): Creates color-coded difference visualization
- get_last_result(): Returns most recent scoring result
- create_overlap_visualization(): Creates color-coded overlap visualization

Signals Emitted:
- score_computed(dict): Score computation completed successfully
- score_error(str): Score computation failed with error

Dependencies:
- Uses: numpy (array operations), cv2 (OpenCV image processing)
- Uses: PySide6.QtCore (QObject, Signal for Qt integration)
- Uses: core/utils.get_logger (logging functionality)
- Used by: Scoring services and alignment evaluation
- Used by: UI components for metric display and visualization

Overlap Metrics:
- Overlap Ratio: Intersection over Union (Jaccard index)
- Dice Score: 2 * intersection / (area1 + area2)
- Precision: True positives / predicted positives
- Recall: True positives / actual positives
- F1 Score: Harmonic mean of precision and recall
- Coverage ratios for both images

Difference Metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Normalized error metrics
- Min/max difference values

Visualization Features:
- Color-coded difference overlays with configurable colormaps
- Overlap visualization with distinct colors for different regions
- Red: Image1 only, Blue: Image2 only, Green: Overlap
- White background for non-structure areas
"""

from typing import Optional, Dict, Tuple
from PySide6.QtCore import QObject, Signal
import numpy as np
import cv2

from src.core.utils import get_logger


class PixelScoreService(QObject):
    """Service for computing pixel-level overlap and matching scores."""
    
    # Signals
    score_computed = Signal(dict)  # Emitted when score computation completes
    score_error = Signal(str)     # Emitted on computation error
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self._last_result = None
    
    def compute_pixel_overlap(self, image1: np.ndarray, image2: np.ndarray, 
                             threshold: int = 127) -> Dict:
        """
        Compute pixel overlap between two binary images.
        
        Args:
            image1: First image (typically SEM)
            image2: Second image (typically GDS)
            threshold: Threshold for binarization
            
        Returns:
            Dictionary containing overlap metrics
        """
        try:
            # Ensure images are the same size
            if image1.shape != image2.shape:
                # Resize image2 to match image1
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            
            # Convert to grayscale if needed
            if len(image1.shape) == 3:
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            if len(image2.shape) == 3:
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            # Binarize images
            _, binary1 = cv2.threshold(image1, threshold, 255, cv2.THRESH_BINARY)
            _, binary2 = cv2.threshold(image2, threshold, 255, cv2.THRESH_BINARY)
            
            # Convert to boolean arrays (True = foreground/structure)
            mask1 = binary1 < threshold  # Assuming dark structures on light background
            mask2 = binary2 < threshold
            
            # Calculate overlap metrics
            intersection = np.logical_and(mask1, mask2)
            union = np.logical_or(mask1, mask2)
            
            # Pixel counts
            total_pixels = mask1.size
            pixels1 = np.sum(mask1)
            pixels2 = np.sum(mask2)
            intersection_pixels = np.sum(intersection)
            union_pixels = np.sum(union)
            
            # Calculate metrics
            overlap_ratio = intersection_pixels / max(1, union_pixels)  # Jaccard index
            coverage1 = intersection_pixels / max(1, pixels1)  # Coverage of image1
            coverage2 = intersection_pixels / max(1, pixels2)  # Coverage of image2
            dice_score = (2 * intersection_pixels) / max(1, pixels1 + pixels2)
            
            # Precision and recall (treating image1 as ground truth)
            precision = intersection_pixels / max(1, pixels2)
            recall = intersection_pixels / max(1, pixels1)
            f1_score = 2 * (precision * recall) / max(1e-10, precision + recall)
            
            result = {
                'overlap_ratio': float(overlap_ratio),
                'dice_score': float(dice_score),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score),
                'coverage_image1': float(coverage1),
                'coverage_image2': float(coverage2),
                'intersection_pixels': int(intersection_pixels),
                'union_pixels': int(union_pixels),
                'total_pixels': int(total_pixels),
                'pixels_image1': int(pixels1),
                'pixels_image2': int(pixels2),
                'threshold_used': threshold
            }
            
            self._last_result = result
            self.score_computed.emit(result)
            self.logger.info(f"Computed pixel overlap: {overlap_ratio:.3f} overlap ratio")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to compute pixel overlap: {e}"
            self.logger.error(error_msg)
            self.score_error.emit(error_msg)
            return {}
    
    def compute_pixel_difference(self, image1: np.ndarray, image2: np.ndarray) -> Dict:
        """
        Compute pixel-wise differences between two images.
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Dictionary containing difference metrics
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
            
            # Convert to float for calculations
            img1_float = image1.astype(np.float64)
            img2_float = image2.astype(np.float64)
            
            # Calculate differences
            absolute_diff = np.abs(img1_float - img2_float)
            squared_diff = (img1_float - img2_float) ** 2
            
            # Metrics
            mean_absolute_error = np.mean(absolute_diff)
            mean_squared_error = np.mean(squared_diff)
            root_mean_squared_error = np.sqrt(mean_squared_error)
            peak_signal_noise_ratio = 20 * np.log10(255.0 / max(1e-10, root_mean_squared_error))
            
            # Normalized metrics
            max_possible_diff = 255.0
            normalized_mae = mean_absolute_error / max_possible_diff
            normalized_rmse = root_mean_squared_error / max_possible_diff
            
            result = {
                'mean_absolute_error': float(mean_absolute_error),
                'mean_squared_error': float(mean_squared_error),
                'root_mean_squared_error': float(root_mean_squared_error),
                'peak_signal_noise_ratio': float(peak_signal_noise_ratio),
                'normalized_mae': float(normalized_mae),
                'normalized_rmse': float(normalized_rmse),
                'max_difference': float(np.max(absolute_diff)),
                'min_difference': float(np.min(absolute_diff))
            }
            
            self.logger.info(f"Computed pixel differences: MAE={mean_absolute_error:.2f}, RMSE={root_mean_squared_error:.2f}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to compute pixel differences: {e}"
            self.logger.error(error_msg)
            self.score_error.emit(error_msg)
            return {}
    
    def create_difference_overlay(self, image1: np.ndarray, image2: np.ndarray, 
                                 colormap: int = cv2.COLORMAP_JET) -> Optional[np.ndarray]:
        """
        Create a color-coded difference overlay between two images.
        
        Args:
            image1: First image
            image2: Second image
            colormap: OpenCV colormap for visualization
            
        Returns:
            Color overlay image or None if failed
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
            
            # Calculate absolute difference
            diff = np.abs(image1.astype(np.float64) - image2.astype(np.float64))
            
            # Normalize difference to 0-255 range
            diff_normalized = ((diff / np.max(diff)) * 255).astype(np.uint8)
            
            # Apply colormap
            colored_diff = cv2.applyColorMap(diff_normalized, colormap)
            
            return colored_diff
            
        except Exception as e:
            error_msg = f"Failed to create difference overlay: {e}"
            self.logger.error(error_msg)
            self.score_error.emit(error_msg)
            return None
    
    def get_last_result(self) -> Optional[Dict]:
        """Get the most recent scoring result."""
        return self._last_result
    
    def create_overlap_visualization(self, image1: np.ndarray, image2: np.ndarray,
                                   threshold: int = 127) -> Optional[np.ndarray]:
        """
        Create a visualization showing overlap regions.
        
        Args:
            image1: First image (typically SEM)
            image2: Second image (typically GDS)
            threshold: Threshold for binarization
            
        Returns:
            RGB image showing overlap in different colors
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
            
            # Create masks (True = structure)
            mask1 = binary1 < threshold
            mask2 = binary2 < threshold
            
            # Create RGB visualization
            h, w = image1.shape
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Background is white
            overlay[:, :] = [255, 255, 255]
            
            # Image1 only (red)
            only1 = mask1 & ~mask2
            overlay[only1] = [255, 0, 0]
            
            # Image2 only (blue)
            only2 = mask2 & ~mask1
            overlay[only2] = [0, 0, 255]
            
            # Overlap (green)
            overlap = mask1 & mask2
            overlay[overlap] = [0, 255, 0]
            
            return overlay
            
        except Exception as e:
            error_msg = f"Failed to create overlap visualization: {e}"
            self.logger.error(error_msg)
            self.score_error.emit(error_msg)
            return None
