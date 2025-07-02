"""
Simple Scoring Service for metric calculation (Steps 86-90).

This service provides:
- QObject for metric calculation with basic scoring methods (Step 86)
- Pixel matching with basic overlap calculation and binary image comparison (Step 87)
- SSIM calculation with structural similarity index and basic image comparison (Step 88)
- IoU calculation with intersection over union and contour-based comparison (Step 89)
- Scoring signals when scores calculated with results formatting (Step 90)
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class ScoringService(QObject):
    """Simple QObject-based scoring service for metric calculation."""
    
    # Signals for Step 90
    score_calculated = Signal(str, dict)          # metric_name, score_results
    scores_batch_calculated = Signal(dict)       # all_scores_dict
    batch_finished = Signal(dict)                # batch_results
    scoring_started = Signal(str)                # metric_description
    scoring_progress = Signal(int)               # progress_percentage
    scoring_finished = Signal(str, bool)        # metric_description, success
    results_formatted = Signal(str, str)        # format_type, formatted_results
    error_occurred = Signal(str)                # error_message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Current images for comparison
        self.sem_image = None
        self.gds_image = None
        self.aligned_gds_image = None
        
        # Available metrics registry (Step 86)
        self.available_metrics = {
            'simple_similarity': {
                'name': 'Simple Similarity',
                'description': 'Basic similarity score between GDS and SEM images (0-100%)',
                'function': self._calculate_simple_similarity
            },
            'pixel_match': {
                'name': 'Pixel Matching',
                'description': 'Basic overlap calculation and binary image comparison',
                'function': self._calculate_pixel_match
            },
            'ssim': {
                'name': 'SSIM',
                'description': 'Structural Similarity Index for image comparison',
                'function': self._calculate_ssim
            },
            'iou': {
                'name': 'IoU',
                'description': 'Intersection over Union with contour-based comparison',
                'function': self._calculate_iou
            },
            'correlation': {
                'name': 'Cross-Correlation',
                'description': 'Normalized cross-correlation coefficient',
                'function': self._calculate_correlation
            },
            'mse': {
                'name': 'Mean Squared Error',
                'description': 'Mean squared error between images',
                'function': self._calculate_mse
            }
        }
        
        # Cached results
        self.last_results = {}
        
        logger.info(f"ScoringService initialized with {len(self.available_metrics)} metrics")
    
    def set_images(self, sem_image: np.ndarray, gds_image: np.ndarray, aligned_gds_image: np.ndarray = None):
        """
        Set images for comparison.
        
        Args:
            sem_image: SEM reference image
            gds_image: Original GDS image
            aligned_gds_image: Aligned/transformed GDS image (optional)
        """
        self.sem_image = sem_image.copy() if sem_image is not None else None
        self.gds_image = gds_image.copy() if gds_image is not None else None
        self.aligned_gds_image = aligned_gds_image.copy() if aligned_gds_image is not None else gds_image.copy() if gds_image is not None else None
        
        # Clear cached results when images change
        self.last_results = {}
        
        logger.info(f"Images set for scoring: SEM {sem_image.shape if sem_image is not None else None}, "
                   f"GDS {gds_image.shape if gds_image is not None else None}")
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metric names."""
        return list(self.available_metrics.keys())
    
    def get_metric_info(self, metric_name: str) -> Dict[str, str]:
        """Get information about a specific metric."""
        if metric_name not in self.available_metrics:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        metric_info = self.available_metrics[metric_name].copy()
        metric_info.pop('function', None)  # Don't return the function object
        return metric_info
    
    # Simple similarity method for user-friendly scoring
    def _calculate_simple_similarity(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, Any]:
        """
        Calculate a simple, user-friendly similarity score (0-100%).
        
        This method combines multiple metrics to provide an easy-to-understand 
        similarity score that users can interpret intuitively.
        
        Args:
            image1: First image (SEM)
            image2: Second image (aligned GDS)
            
        Returns:
            Dictionary with simple similarity results
        """
        try:
            # Ensure images are same size
            if image1.shape != image2.shape:
                import cv2
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            
            # Convert to grayscale if needed
            if len(image1.shape) == 3:
                import cv2
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            if len(image2.shape) == 3:
                import cv2
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            # Calculate component metrics
            pixel_results = self._calculate_pixel_match(image1, image2)
            ssim_results = self._calculate_ssim(image1, image2)
            iou_results = self._calculate_iou(image1, image2)
            correlation_results = self._calculate_correlation(image1, image2)
            
            # Extract key scores (handle errors)
            pixel_accuracy = pixel_results.get('pixel_accuracy', 0) if 'error' not in pixel_results else 0
            ssim_score = ssim_results.get('ssim_score', 0) if 'error' not in ssim_results else 0
            iou_score = iou_results.get('iou_score', 0) if 'error' not in iou_results else 0
            correlation_score = abs(correlation_results.get('correlation_score', 0)) if 'error' not in correlation_results else 0
            
            # Weighted combination for simple similarity (0-1 range)
            # SSIM and IoU are more important for structural similarity
            weights = {
                'ssim': 0.4,        # Structural similarity
                'iou': 0.3,         # Shape overlap
                'pixel': 0.2,       # Pixel accuracy
                'correlation': 0.1  # Overall correlation
            }
            
            simple_similarity = (
                weights['ssim'] * max(0, ssim_score) +
                weights['iou'] * max(0, iou_score) +
                weights['pixel'] * max(0, pixel_accuracy) +
                weights['correlation'] * max(0, correlation_score)
            )
            
            # Convert to percentage (0-100%)
            simple_similarity_percent = simple_similarity * 100
            
            # Quality classification
            if simple_similarity_percent >= 90:
                quality = "Excellent"
            elif simple_similarity_percent >= 80:
                quality = "Very Good"
            elif simple_similarity_percent >= 70:
                quality = "Good"
            elif simple_similarity_percent >= 60:
                quality = "Fair"
            elif simple_similarity_percent >= 50:
                quality = "Poor"
            else:
                quality = "Very Poor"
            
            results = {
                'simple_similarity': float(simple_similarity),
                'simple_similarity_percent': float(simple_similarity_percent),
                'quality': quality,
                'component_scores': {
                    'ssim': float(ssim_score),
                    'iou': float(iou_score),
                    'pixel_accuracy': float(pixel_accuracy),
                    'correlation': float(correlation_score)
                },
                'weights_used': weights
            }
            
            logger.info(f"Simple similarity calculated: {simple_similarity_percent:.1f}% ({quality})")
            return results
            
        except Exception as e:
            logger.error(f"Simple similarity calculation failed: {e}")
            return {'error': str(e)}

    # Step 87: Implement pixel matching with basic overlap calculation
    def _calculate_pixel_match(self, image1: np.ndarray, image2: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Calculate pixel matching score with basic overlap calculation and binary image comparison.
        
        Args:
            image1: First image (SEM)
            image2: Second image (aligned GDS)
            threshold: Threshold for binary conversion
            
        Returns:
            Dictionary with pixel matching results
        """
        try:
            # Ensure images are same size
            if image1.shape != image2.shape:
                import cv2
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            
            # Convert to grayscale if needed
            if len(image1.shape) == 3:
                import cv2
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            if len(image2.shape) == 3:
                import cv2
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            # Normalize images to [0, 1]
            image1_norm = image1.astype(np.float32) / 255.0
            image2_norm = image2.astype(np.float32) / 255.0
            
            # Binary conversion
            binary1 = (image1_norm > threshold).astype(np.uint8)
            binary2 = (image2_norm > threshold).astype(np.uint8)
            
            # Calculate overlap metrics
            intersection = np.logical_and(binary1, binary2)
            union = np.logical_or(binary1, binary2)
            
            total_pixels = binary1.size
            intersection_pixels = np.sum(intersection)
            union_pixels = np.sum(union)
            
            # Basic overlap calculations
            overlap_ratio = intersection_pixels / total_pixels if total_pixels > 0 else 0.0
            jaccard_index = intersection_pixels / union_pixels if union_pixels > 0 else 0.0
            
            # Pixel-wise accuracy
            pixel_matches = np.sum(binary1 == binary2)
            pixel_accuracy = pixel_matches / total_pixels if total_pixels > 0 else 0.0
            
            # Additional metrics
            dice_coefficient = 2 * intersection_pixels / (np.sum(binary1) + np.sum(binary2)) if (np.sum(binary1) + np.sum(binary2)) > 0 else 0.0
            
            results = {
                'overlap_ratio': float(overlap_ratio),
                'jaccard_index': float(jaccard_index),
                'pixel_accuracy': float(pixel_accuracy),
                'dice_coefficient': float(dice_coefficient),
                'intersection_pixels': int(intersection_pixels),
                'union_pixels': int(union_pixels),
                'total_pixels': int(total_pixels),
                'threshold_used': float(threshold)
            }
            
            logger.info(f"Pixel matching calculated: accuracy={pixel_accuracy:.3f}, overlap={overlap_ratio:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Pixel matching calculation failed: {e}")
            return {'error': str(e)}
    
    # Step 88: Add SSIM calculation with structural similarity index
    def _calculate_ssim(self, image1: np.ndarray, image2: np.ndarray, window_size: int = 11) -> Dict[str, Any]:
        """
        Calculate SSIM (Structural Similarity Index) for basic image comparison.
        
        Args:
            image1: First image (SEM)
            image2: Second image (aligned GDS)
            window_size: Size of sliding window
            
        Returns:
            Dictionary with SSIM results
        """
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Ensure images are same size
            if image1.shape != image2.shape:
                import cv2
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            
            # Convert to grayscale if needed
            if len(image1.shape) == 3:
                import cv2
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            if len(image2.shape) == 3:
                import cv2
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            # Calculate SSIM
            ssim_score, ssim_map = ssim(
                image1, image2, 
                data_range=image1.max() - image1.min(),
                win_size=window_size,
                full=True
            )
            
            # Additional SSIM-based metrics
            mean_ssim_map = np.mean(ssim_map)
            std_ssim_map = np.std(ssim_map)
            min_ssim = np.min(ssim_map)
            max_ssim = np.max(ssim_map)
            
            # Percentage of pixels with high SSIM (> 0.8)
            high_ssim_pixels = np.sum(ssim_map > 0.8)
            high_ssim_percentage = high_ssim_pixels / ssim_map.size * 100
            
            results = {
                'ssim_score': float(ssim_score),
                'mean_ssim_map': float(mean_ssim_map),
                'std_ssim_map': float(std_ssim_map),
                'min_ssim': float(min_ssim),
                'max_ssim': float(max_ssim),
                'high_ssim_percentage': float(high_ssim_percentage),
                'window_size': window_size
            }
            
            logger.info(f"SSIM calculated: score={ssim_score:.3f}, high_similarity={high_ssim_percentage:.1f}%")
            return results
            
        except Exception as e:
            logger.error(f"SSIM calculation failed: {e}")
            return {'error': str(e)}
    
    # Step 89: Implement IoU calculation with intersection over union
    def _calculate_iou(self, image1: np.ndarray, image2: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Calculate IoU (Intersection over Union) with contour-based comparison.
        
        Args:
            image1: First image (SEM)
            image2: Second image (aligned GDS)
            threshold: Threshold for binary conversion
            
        Returns:
            Dictionary with IoU results
        """
        try:
            import cv2
            
            # Ensure images are same size
            if image1.shape != image2.shape:
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            
            # Convert to grayscale if needed
            if len(image1.shape) == 3:
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            if len(image2.shape) == 3:
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            # Normalize and threshold
            image1_norm = image1.astype(np.float32) / 255.0
            image2_norm = image2.astype(np.float32) / 255.0
            
            binary1 = (image1_norm > threshold).astype(np.uint8) * 255
            binary2 = (image2_norm > threshold).astype(np.uint8) * 255
            
            # Basic IoU calculation
            intersection = cv2.bitwise_and(binary1, binary2)
            union = cv2.bitwise_or(binary1, binary2)
            
            intersection_area = np.sum(intersection > 0)
            union_area = np.sum(union > 0)
            
            iou_score = intersection_area / union_area if union_area > 0 else 0.0
            
            # Contour-based analysis
            contours1, _ = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate contour areas
            total_area1 = sum(cv2.contourArea(c) for c in contours1)
            total_area2 = sum(cv2.contourArea(c) for c in contours2)
            
            # Contour-based IoU
            contour_iou = intersection_area / (total_area1 + total_area2 - intersection_area) if (total_area1 + total_area2 - intersection_area) > 0 else 0.0
            
            # Additional metrics
            area_ratio = total_area2 / total_area1 if total_area1 > 0 else 0.0
            coverage1 = intersection_area / total_area1 if total_area1 > 0 else 0.0
            coverage2 = intersection_area / total_area2 if total_area2 > 0 else 0.0
            
            results = {
                'iou_score': float(iou_score),
                'contour_iou': float(contour_iou),
                'intersection_area': int(intersection_area),
                'union_area': int(union_area),
                'area1': float(total_area1),
                'area2': float(total_area2),
                'area_ratio': float(area_ratio),
                'coverage1': float(coverage1),
                'coverage2': float(coverage2),
                'num_contours1': len(contours1),
                'num_contours2': len(contours2),
                'threshold_used': float(threshold)
            }
            
            logger.info(f"IoU calculated: score={iou_score:.3f}, contour_iou={contour_iou:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"IoU calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_correlation(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, Any]:
        """Calculate normalized cross-correlation."""
        try:
            # Ensure images are same size
            if image1.shape != image2.shape:
                import cv2
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            
            # Convert to grayscale if needed
            if len(image1.shape) == 3:
                import cv2
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            if len(image2.shape) == 3:
                import cv2
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            # Flatten and calculate correlation
            flat1 = image1.flatten().astype(np.float64)
            flat2 = image2.flatten().astype(np.float64)
            
            correlation_matrix = np.corrcoef(flat1, flat2)
            correlation_score = correlation_matrix[0, 1]
            
            # Handle NaN case
            if np.isnan(correlation_score):
                correlation_score = 0.0
            
            results = {
                'correlation_score': float(correlation_score),
                'absolute_correlation': float(abs(correlation_score))
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_mse(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, Any]:
        """Calculate mean squared error."""
        try:
            # Ensure images are same size
            if image1.shape != image2.shape:
                import cv2
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            
            # Convert to grayscale if needed
            if len(image1.shape) == 3:
                import cv2
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            if len(image2.shape) == 3:
                import cv2
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            # Calculate MSE
            mse_score = np.mean((image1.astype(np.float64) - image2.astype(np.float64)) ** 2)
            
            # Calculate RMSE
            rmse_score = np.sqrt(mse_score)
            
            # Normalized MSE (0-1 range)
            max_possible_mse = 255.0 ** 2
            normalized_mse = mse_score / max_possible_mse
            
            results = {
                'mse_score': float(mse_score),
                'rmse_score': float(rmse_score),
                'normalized_mse': float(normalized_mse)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"MSE calculation failed: {e}")
            return {'error': str(e)}
    
    # Main calculation methods (Step 90: Add scoring signals)
    def calculate_metric(self, metric_name: str, use_aligned: bool = True, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Calculate a specific metric.
        
        Args:
            metric_name: Name of metric to calculate
            use_aligned: Use aligned GDS image if available
            **kwargs: Additional parameters for metric calculation
            
        Returns:
            Metric results or None if failed
        """
        try:
            if self.sem_image is None:
                raise ValueError("SEM image not set")
            
            # Choose GDS image
            gds_img = self.aligned_gds_image if (use_aligned and self.aligned_gds_image is not None) else self.gds_image
            if gds_img is None:
                raise ValueError("GDS image not set")
            
            if metric_name not in self.available_metrics:
                raise ValueError(f"Unknown metric: {metric_name}")
            
            # Emit signals
            metric_info = self.available_metrics[metric_name]
            self.scoring_started.emit(f"Calculating {metric_info['name']}")
            self.scoring_progress.emit(25)
            
            # Calculate metric
            metric_function = metric_info['function']
            results = metric_function(self.sem_image, gds_img, **kwargs)
            
            self.scoring_progress.emit(100)
            
            if 'error' not in results:
                # Cache results
                self.last_results[metric_name] = results
                
                # Emit signals
                self.score_calculated.emit(metric_name, results)
                self.scoring_finished.emit(f"{metric_info['name']} calculation", True)
                
                logger.info(f"Metric '{metric_name}' calculated successfully")
            else:
                self.scoring_finished.emit(f"{metric_info['name']} calculation", False)
                self.error_occurred.emit(results['error'])
            
            return results
            
        except Exception as e:
            error_msg = f"Metric calculation failed: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            self.scoring_finished.emit(f"Metric calculation", False)
            return None
    
    def calculate_all_metrics(self, use_aligned: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Calculate all available metrics.
        
        Args:
            use_aligned: Use aligned GDS image if available
            **kwargs: Additional parameters for metric calculations
            
        Returns:
            Dictionary with all metric results
        """
        try:
            self.scoring_started.emit("Calculating all metrics")
            all_results = {}
            
            total_metrics = len(self.available_metrics)
            
            for i, metric_name in enumerate(self.available_metrics.keys()):
                # Update progress
                progress = int((i + 1) / total_metrics * 100)
                self.scoring_progress.emit(progress)
                
                # Calculate metric
                result = self.calculate_metric(metric_name, use_aligned, **kwargs)
                if result is not None:
                    all_results[metric_name] = result
            
            # Emit batch results signal
            self.scores_batch_calculated.emit(all_results)
            self.scoring_finished.emit("All metrics calculation", True)
            
            logger.info(f"Calculated {len(all_results)} metrics successfully")
            return all_results
            
        except Exception as e:
            error_msg = f"Batch metric calculation failed: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            self.scoring_finished.emit("All metrics calculation", False)
            return {}
    
    def format_results(self, results: Dict[str, Any], format_type: str = "summary") -> str:
        """
        Format scoring results for display.
        
        Args:
            results: Results dictionary
            format_type: Format type ("summary", "detailed", "csv", "json")
            
        Returns:
            Formatted results string
        """
        try:
            if format_type == "summary":
                formatted = self._format_summary(results)
            elif format_type == "detailed":
                formatted = self._format_detailed(results)
            elif format_type == "csv":
                formatted = self._format_csv(results)
            elif format_type == "json":
                import json
                formatted = json.dumps(results, indent=2)
            else:
                formatted = str(results)
            
            # Emit signal
            self.results_formatted.emit(format_type, formatted)
            
            return formatted
            
        except Exception as e:
            error_msg = f"Results formatting failed: {e}"
            logger.error(error_msg)
            return str(results)
    
    def _format_summary(self, results: Dict[str, Any]) -> str:
        """Format results as summary."""
        lines = ["=== Scoring Results Summary ==="]
        
        for metric_name, metric_results in results.items():
            if 'error' in metric_results:
                lines.append(f"{metric_name}: ERROR - {metric_results['error']}")
                continue
            
            metric_info = self.available_metrics.get(metric_name, {})
            display_name = metric_info.get('name', metric_name)
            
            # Extract main score value
            if metric_name == 'simple_similarity':
                main_score = metric_results.get('simple_similarity_percent', 0)
                quality = metric_results.get('quality', 'Unknown')
                lines.append(f"{display_name}: {main_score:.1f}% ({quality})")
            elif metric_name == 'pixel_match':
                main_score = metric_results.get('pixel_accuracy', 0)
                lines.append(f"{display_name}: {main_score:.3f} (accuracy)")
            elif metric_name == 'ssim':
                main_score = metric_results.get('ssim_score', 0)
                lines.append(f"{display_name}: {main_score:.3f}")
            elif metric_name == 'iou':
                main_score = metric_results.get('iou_score', 0)
                lines.append(f"{display_name}: {main_score:.3f}")
            elif metric_name == 'correlation':
                main_score = metric_results.get('correlation_score', 0)
                lines.append(f"{display_name}: {main_score:.3f}")
            elif metric_name == 'mse':
                main_score = metric_results.get('normalized_mse', 0)
                lines.append(f"{display_name}: {main_score:.3f} (normalized)")
        
        return "\n".join(lines)
    
    def _format_detailed(self, results: Dict[str, Any]) -> str:
        """Format results with details."""
        lines = ["=== Detailed Scoring Results ==="]
        
        for metric_name, metric_results in results.items():
            metric_info = self.available_metrics.get(metric_name, {})
            display_name = metric_info.get('name', metric_name)
            
            lines.append(f"\n{display_name}:")
            lines.append("-" * (len(display_name) + 1))
            
            if 'error' in metric_results:
                lines.append(f"  ERROR: {metric_results['error']}")
                continue
            
            for key, value in metric_results.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    def _format_csv(self, results: Dict[str, Any]) -> str:
        """Format results as CSV."""
        lines = ["Metric,Value,Description"]
        
        for metric_name, metric_results in results.items():
            if 'error' in metric_results:
                lines.append(f"{metric_name},ERROR,{metric_results['error']}")
                continue
            
            for key, value in metric_results.items():
                lines.append(f"{metric_name}_{key},{value},{key}")
        
        return "\n".join(lines)
    
    def test_scoring_workflow(self) -> bool:
        """
        Test the complete scoring workflow to verify Step 8 implementation.
        
        Returns:
            True if all tests pass, False otherwise
        """
        try:
            logger.info("Testing scoring workflow...")
            
            # Create test images
            import numpy as np
            
            # Create a simple test pattern
            test_image1 = np.zeros((100, 100), dtype=np.uint8)
            test_image1[25:75, 25:75] = 255  # White square in center
            
            # Create similar image with slight offset
            test_image2 = np.zeros((100, 100), dtype=np.uint8)
            test_image2[30:80, 30:80] = 255  # Slightly offset square
            
            # Set test images
            self.set_images(test_image1, test_image2, test_image2)
            
            # Test individual metrics
            logger.info("Testing individual metrics...")
            for metric_name in self.available_metrics.keys():
                result = self.calculate_metric(metric_name)
                if result is None or 'error' in result:
                    logger.error(f"Metric {metric_name} failed")
                    return False
                logger.info(f"Metric {metric_name}: Success")
            
            # Test batch calculation
            logger.info("Testing batch calculation...")
            all_results = self.calculate_all_metrics()
            if not all_results:
                logger.error("Batch calculation failed")
                return False
            
            # Test formatting
            logger.info("Testing result formatting...")
            summary = self.format_results(all_results, "summary")
            detailed = self.format_results(all_results, "detailed")
            csv_format = self.format_results(all_results, "csv")
            
            if not all([summary, detailed, csv_format]):
                logger.error("Result formatting failed")
                return False
            
            logger.info("Scoring workflow test completed successfully!")
            logger.info(f"Summary results:\n{summary}")
            
            return True
            
        except Exception as e:
            logger.error(f"Scoring workflow test failed: {e}")
            return False

    def get_last_results(self) -> Dict[str, Any]:
        """Get the last calculated results."""
        return self.last_results.copy()
    
    def clear_results(self):
        """Clear cached results."""
        self.last_results = {}
        logger.info("Scoring results cleared")
