"""SSIM score service for computing Structural Similarity metrics."""

from typing import Optional, Dict, Tuple
from PySide6.QtCore import QObject, Signal
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

from src.core.utils import get_logger


class SSIMScoreService(QObject):
    """Service for computing Structural Similarity Index (SSIM) metrics."""
    
    # Signals
    ssim_computed = Signal(dict)  # Emitted when SSIM computation completes
    ssim_error = Signal(str)      # Emitted on computation error
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self._last_result = None
    
    def compute_ssim(self, image1: np.ndarray, image2: np.ndarray, 
                    window_size: int = 7, full: bool = True) -> Dict:
        """
        Compute SSIM between two images.
        
        Args:
            image1: First image
            image2: Second image
            window_size: Size of the sliding window (must be odd)
            full: Whether to return the full SSIM map
            
        Returns:
            Dictionary containing SSIM metrics
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
            
            # Ensure window size is odd and valid
            if window_size % 2 == 0:
                window_size += 1
            
            # Compute SSIM
            if full:
                ssim_score, ssim_map = ssim(image1, image2, 
                                          win_size=window_size, 
                                          full=True,
                                          data_range=255)
            else:
                ssim_score = ssim(image1, image2, 
                                win_size=window_size, 
                                full=False,
                                data_range=255)
                ssim_map = None
            
            # Compute additional SSIM-based metrics
            result = {
                'ssim_score': float(ssim_score),
                'ssim_percentage': float(ssim_score * 100),
                'window_size': window_size,
                'has_ssim_map': ssim_map is not None
            }
            
            if ssim_map is not None:
                result.update({
                    'ssim_map_mean': float(np.mean(ssim_map)),
                    'ssim_map_std': float(np.std(ssim_map)),
                    'ssim_map_min': float(np.min(ssim_map)),
                    'ssim_map_max': float(np.max(ssim_map)),
                    'ssim_map_shape': ssim_map.shape
                })
                
                # Store the SSIM map for visualization
                result['_ssim_map'] = ssim_map
            
            self._last_result = result
            self.ssim_computed.emit(result)
            self.logger.info(f"Computed SSIM: {ssim_score:.4f} ({ssim_score*100:.2f}%)")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to compute SSIM: {e}"
            self.logger.error(error_msg)
            self.ssim_error.emit(error_msg)
            return {}
    
    def compute_multiscale_ssim(self, image1: np.ndarray, image2: np.ndarray,
                               scales: list = None) -> Dict:
        """
        Compute Multi-Scale SSIM (MS-SSIM) between two images.
        
        Args:
            image1: First image
            image2: Second image
            scales: List of scale factors for multi-scale analysis
            
        Returns:
            Dictionary containing MS-SSIM metrics
        """
        if scales is None:
            scales = [1.0, 0.5, 0.25]
        
        try:
            # Ensure images are the same size
            if image1.shape != image2.shape:
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            
            # Convert to grayscale if needed
            if len(image1.shape) == 3:
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            if len(image2.shape) == 3:
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            scale_ssims = []
            scale_results = {}
            
            for i, scale in enumerate(scales):
                if scale == 1.0:
                    scaled_img1 = image1
                    scaled_img2 = image2
                else:
                    # Resize images for this scale
                    new_width = int(image1.shape[1] * scale)
                    new_height = int(image1.shape[0] * scale)
                    
                    if new_width < 7 or new_height < 7:
                        continue  # Skip scales that would make images too small
                    
                    scaled_img1 = cv2.resize(image1, (new_width, new_height))
                    scaled_img2 = cv2.resize(image2, (new_width, new_height))
                
                # Compute SSIM at this scale
                scale_ssim = ssim(scaled_img1, scaled_img2, 
                                win_size=min(7, min(scaled_img1.shape[:2])//2*2-1),
                                data_range=255)
                
                scale_ssims.append(scale_ssim)
                scale_results[f'scale_{scale}'] = float(scale_ssim)
            
            # Compute weighted average (MS-SSIM)
            if scale_ssims:
                weights = np.array([0.5, 0.3, 0.2][:len(scale_ssims)])
                weights = weights / np.sum(weights)  # Normalize weights
                ms_ssim = np.sum(np.array(scale_ssims) * weights)
            else:
                ms_ssim = 0.0
            
            result = {
                'ms_ssim_score': float(ms_ssim),
                'ms_ssim_percentage': float(ms_ssim * 100),
                'scales_used': scales[:len(scale_ssims)],
                'individual_scales': scale_results,
                'num_scales': len(scale_ssims)
            }
            
            self.logger.info(f"Computed MS-SSIM: {ms_ssim:.4f} using {len(scale_ssims)} scales")
            return result
            
        except Exception as e:
            error_msg = f"Failed to compute MS-SSIM: {e}"
            self.logger.error(error_msg)
            self.ssim_error.emit(error_msg)
            return {}
    
    def create_ssim_heatmap(self, image1: np.ndarray, image2: np.ndarray,
                           window_size: int = 7, colormap: int = cv2.COLORMAP_JET) -> Optional[np.ndarray]:
        """
        Create a heatmap visualization of SSIM values.
        
        Args:
            image1: First image
            image2: Second image
            window_size: Size of the sliding window
            colormap: OpenCV colormap for visualization
            
        Returns:
            Color heatmap image or None if failed
        """
        try:
            # Compute SSIM with full map
            result = self.compute_ssim(image1, image2, window_size, full=True)
            
            if '_ssim_map' not in result:
                return None
            
            ssim_map = result['_ssim_map']
            
            # Convert SSIM map to 0-255 range
            ssim_normalized = ((ssim_map + 1) / 2 * 255).astype(np.uint8)
            
            # Apply colormap
            heatmap = cv2.applyColorMap(ssim_normalized, colormap)
            
            return heatmap
            
        except Exception as e:
            error_msg = f"Failed to create SSIM heatmap: {e}"
            self.logger.error(error_msg)
            self.ssim_error.emit(error_msg)
            return None
    
    def get_last_result(self) -> Optional[Dict]:
        """Get the most recent SSIM result."""
        return self._last_result
    
    def compute_local_ssim_statistics(self, image1: np.ndarray, image2: np.ndarray,
                                    window_size: int = 7, grid_size: int = 32) -> Dict:
        """
        Compute SSIM statistics for local regions of the image.
        
        Args:
            image1: First image
            image2: Second image
            window_size: Size of the SSIM sliding window
            grid_size: Size of grid cells for local analysis
            
        Returns:
            Dictionary containing local SSIM statistics
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
            
            h, w = image1.shape
            local_ssims = []
            
            # Divide image into grid and compute SSIM for each cell
            for y in range(0, h - grid_size, grid_size):
                for x in range(0, w - grid_size, grid_size):
                    # Extract local regions
                    region1 = image1[y:y+grid_size, x:x+grid_size]
                    region2 = image2[y:y+grid_size, x:x+grid_size]
                    
                    # Skip if region is too small
                    if region1.shape[0] < window_size or region1.shape[1] < window_size:
                        continue
                    
                    # Compute local SSIM
                    local_ssim = ssim(region1, region2, 
                                    win_size=min(window_size, min(region1.shape)//2*2-1),
                                    data_range=255)
                    local_ssims.append(local_ssim)
            
            if local_ssims:
                local_ssims = np.array(local_ssims)
                
                result = {
                    'local_ssim_mean': float(np.mean(local_ssims)),
                    'local_ssim_std': float(np.std(local_ssims)),
                    'local_ssim_min': float(np.min(local_ssims)),
                    'local_ssim_max': float(np.max(local_ssims)),
                    'local_ssim_median': float(np.median(local_ssims)),
                    'num_regions': len(local_ssims),
                    'grid_size': grid_size,
                    'regions_above_threshold': {
                        '0.5': int(np.sum(local_ssims > 0.5)),
                        '0.7': int(np.sum(local_ssims > 0.7)),
                        '0.8': int(np.sum(local_ssims > 0.8)),
                        '0.9': int(np.sum(local_ssims > 0.9))
                    }
                }
                
                self.logger.info(f"Computed local SSIM statistics for {len(local_ssims)} regions")
                return result
            else:
                return {'error': 'No valid regions found for local SSIM analysis'}
                
        except Exception as e:
            error_msg = f"Failed to compute local SSIM statistics: {e}"
            self.logger.error(error_msg)
            self.ssim_error.emit(error_msg)
            return {}
