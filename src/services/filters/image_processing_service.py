"""
Image Processing Service - Advanced Filter Application and Management

This service provides comprehensive image processing capabilities with multiple
filters, preview functionality, history management, and parameter validation
for SEM image enhancement and analysis.

Main Class:
- ImageProcessingService: Service for advanced image filter operations

Key Methods:
- load_image(): Loads SEM image for processing
- preview_filter(): Previews filter without permanent application
- apply_filter(): Applies filter with history tracking
- reset_to_original(): Resets to original unprocessed image
- get_current_image(): Returns current processed image
- get_original_image(): Returns original unprocessed image
- get_available_filters(): Lists all available filters
- get_filter_parameters(): Returns filter parameter specifications
- undo(): Undoes last filter application

Available Filters:
- fft_highpass: FFT-based high-pass filtering
- gabor: Gabor filter for texture analysis
- laplacian: Laplacian edge detection
- threshold: Binary thresholding with multiple methods
- top_hat: Morphological top-hat transformation
- total_variation: Total variation denoising
- wavelet: Wavelet-based edge detection
- dog: Difference of Gaussians
- canny: Canny edge detection
- clahe: Contrast Limited Adaptive Histogram Equalization

Dependencies:
- Uses: numpy (array operations), cv2 (OpenCV image processing)
- Uses: importlib (dynamic module loading), copy.deepcopy (deep copying)
- Uses: typing (type hints), skimage.restoration (optional denoising)
- Uses: pywt (optional wavelet processing)
- Used by: UI image processing components
- Used by: Filter management and workflow services

Features:
- Dynamic filter module loading
- Parameter validation and specification
- History tracking with undo functionality
- Preview mode for non-destructive testing
- Comprehensive filter parameter management
- Error handling with fallback implementations
"""

import importlib
import numpy as np
import cv2
from copy import deepcopy
from typing import Dict, Any, Optional, Tuple


class ImageProcessingService:
    def __init__(self, sem_image=None):
        self.original_image = None
        self.current_image = None
        self._filter_modules = {}
        self._history = []  # Stack for undo
        
        if sem_image is not None:
            self.load_image(sem_image)
    
    def load_image(self, sem_image):
        """Load new image, setting both original and current references."""
        self.original_image = deepcopy(sem_image)
        self.current_image = deepcopy(sem_image)
        self._history = []
        print("✓ Image loaded - Original and current references set")
        
    def preview_filter(self, filter_name: str, parameters: Dict[str, Any]) -> np.ndarray:
        """Preview filter on current reference image without changing the reference."""
        if self.current_image is None:
            raise ValueError("No image loaded")
            
        temp_image = deepcopy(self.current_image)
        filtered_image = self._apply_filter_internal(temp_image, filter_name, parameters)
        result = filtered_image.image_data if hasattr(filtered_image, 'image_data') else filtered_image
        print(f"✓ Filter preview: {filter_name} (applied to current reference)")
        return result
    
    def apply_filter(self, filter_name: str, parameters: Dict[str, Any]):
        """Apply filter and make result the new reference image."""
        if self.current_image is None:
            raise ValueError("No image loaded")
            
        self._history.append(deepcopy(self.current_image))  # Save for undo
        self.current_image = self._apply_filter_internal(self.current_image, filter_name, parameters)
        print(f"✓ Filter applied: {filter_name} - Current image updated as new reference")
        
    def reset_to_original(self):
        """Reset current image back to original, making original the new reference."""
        if self.original_image is None:
            raise ValueError("No original image available")
            
        self.current_image = deepcopy(self.original_image)
        self._history = []  # Clear history since we're back to original
        print("✓ Reset to original - Original is now the current reference")
        
    def get_current_image(self):
        """Get the current reference image (may be filtered)."""
        return self.current_image
    
    def get_original_image(self):
        """Get the original unprocessed image."""
        return self.original_image
    
    def get_reference_image(self):
        """Get the current reference image for applying new filters."""
        return self.current_image
    
    def _apply_filter_internal(self, image, filter_name: str, parameters: Dict[str, Any]):
        filter_module = self._get_filter_module(filter_name)
        
        if hasattr(image, 'image_data'):
            image_data = image.image_data
        else:
            image_data = image
            
        filtered_data = self._dispatch_filter(filter_name, image_data, parameters)
        
        if hasattr(image, 'image_data'):
            result_image = deepcopy(image)
            result_image.image_data = filtered_data
            return result_image
        else:
            return filtered_data
    
    def _get_filter_module(self, filter_name: str):
        if filter_name not in self._filter_modules:
            try:
                module_name = f"filter_{filter_name}"
                self._filter_modules[filter_name] = importlib.import_module(module_name)
            except ImportError:
                raise ValueError(f"Filter module '{filter_name}' not found")
        
        return self._filter_modules[filter_name]
    
    def _dispatch_filter(self, filter_name: str, image_data: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        if filter_name == "fft_highpass":
            return self._apply_fft_highpass(image_data, parameters)
        elif filter_name == "gabor":
            return self._apply_gabor(image_data, parameters)
        elif filter_name == "laplacian":
            return self._apply_laplacian(image_data, parameters)
        elif filter_name == "threshold":
            return self._apply_threshold(image_data, parameters)
        elif filter_name == "top_hat":
            return self._apply_top_hat(image_data, parameters)
        elif filter_name == "total_variation":
            return self._apply_total_variation(image_data, parameters)
        elif filter_name == "wavelet":
            return self._apply_wavelet(image_data, parameters)
        elif filter_name == "dog":
            return self._apply_dog(image_data, parameters)
        elif filter_name == "canny":
            return self._apply_canny(image_data, parameters)
        elif filter_name == "clahe":
            return self._apply_clahe(image_data, parameters)
        else:
            raise ValueError(f"Unknown filter: {filter_name}")
    
    def _apply_fft_highpass(self, image, parameters):
        radius = parameters.get('radius', 30)
        rows, cols = image.shape
        crow, ccol = rows//2, cols//2
        
        dft = np.fft.fft2(image)
        dft_shift = np.fft.fftshift(dft)
        
        mask = np.ones((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 0, -1)
        
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        return np.uint8(img_back)
    
    def _apply_gabor(self, image, parameters):
        frequency = parameters.get('frequency', 0.1)
        theta = parameters.get('theta', np.pi/4)
        sigma_x = parameters.get('sigma_x', 5)
        sigma_y = parameters.get('sigma_y', 5)
        
        kernel = cv2.getGaborKernel(
            (0, 0), sigma_x, theta, frequency, sigma_y, 0, ktype=cv2.CV_32F
        )
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        return filtered
    
    def _apply_laplacian(self, image, parameters):
        ksize = parameters.get('ksize', 3)
        laplacian = cv2.Laplacian(image, cv2.CV_16S, ksize=ksize)
        abs_laplacian = cv2.convertScaleAbs(laplacian)
        return abs_laplacian
    
    def _apply_threshold(self, image, parameters):
        threshold_value = parameters.get('threshold_value', 127)
        method = parameters.get('method', 'binary')
        
        methods = {
            'binary': cv2.THRESH_BINARY,
            'binary_inv': cv2.THRESH_BINARY_INV,
            'tozero': cv2.THRESH_TOZERO,
            'tozero_inv': cv2.THRESH_TOZERO_INV,
            'trunc': cv2.THRESH_TRUNC
        }
        
        if method not in methods:
            raise ValueError(f"Unsupported method: {method}")
        
        _, thresh = cv2.threshold(image, threshold_value, 255, methods[method])
        return thresh
    
    def _apply_top_hat(self, image, parameters):
        kernel_size = parameters.get('kernel_size', 5)
        
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError("Kernel size must be odd and >= 3")
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        return tophat
    
    def _apply_total_variation(self, image, parameters):
        weight = parameters.get('weight', 0.2)
        try:
            from skimage.restoration import denoise_tv_chambolle
            denoised = denoise_tv_chambolle(image, weight=weight)
            denoised_uint8 = (denoised * 255).astype(np.uint8)
            return denoised_uint8
        except ImportError:
            return cv2.bilateralFilter(image, 9, 75, 75)
    
    def _apply_wavelet(self, image, parameters):
        wavelet = parameters.get('wavelet', 'db2')
        level = parameters.get('level', 1)
        
        try:
            import pywt
            img_float = image.astype(np.float32) / 255.0
            
            coeffs = pywt.wavedec2(img_float, wavelet, level=level)
            edges = np.zeros_like(img_float, dtype=np.float32)
            
            for i in range(1, min(level+1, len(coeffs))):
                if len(coeffs[i]) == 3:
                    LH, HL, HH = coeffs[i]
                    LH_resized = cv2.resize(LH, (image.shape[1], image.shape[0]))
                    HL_resized = cv2.resize(HL, (image.shape[1], image.shape[0]))
                    HH_resized = cv2.resize(HH, (image.shape[1], image.shape[0]))
                    
                    edges += np.abs(LH_resized) + np.abs(HL_resized) + np.abs(HH_resized)
            
            edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
            return edges.astype(np.uint8)
        except ImportError:
            return cv2.Laplacian(image, cv2.CV_8U, ksize=3)
    
    def _apply_dog(self, image, parameters):
        sigma1 = parameters.get('sigma1', 1.0)
        sigma2 = parameters.get('sigma2', 2.0)
        import cv2
        blur1 = cv2.GaussianBlur(image, (0, 0), sigma1)
        blur2 = cv2.GaussianBlur(image, (0, 0), sigma2)
        dog = blur1 - blur2
        dog_normalized = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return dog_normalized

    def _apply_canny(self, image, parameters):
        low_threshold = parameters.get('low_threshold', 50)
        high_threshold = parameters.get('high_threshold', 150)
        aperture_size = parameters.get('aperture_size', 3)
        import cv2
        edges = cv2.Canny(image, low_threshold, high_threshold, apertureSize=aperture_size)
        return edges
    
    def _apply_clahe(self, image, parameters):
        clip_limit = parameters.get('clip_limit', 2.0)
        tile_grid_size = parameters.get('tile_grid_size', 8)
        import cv2
        if isinstance(tile_grid_size, int):
            tile_grid_size = (tile_grid_size, tile_grid_size)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    
    def get_available_filters(self) -> list:
        return [
            "fft_highpass",
            "gabor", 
            "laplacian",
            "threshold",
            "top_hat",
            "total_variation",
            "wavelet",
            "dog",
            "canny",
            "clahe"
        ]
    
    def get_filter_parameters(self, filter_name: str) -> Dict[str, Dict[str, Any]]:
        parameter_specs = {
            "fft_highpass": {
                "radius": {"type": "int", "default": 30, "min": 1, "max": 200}
            },
            "gabor": {
                "frequency": {"type": "float", "default": 0.1, "min": 0.01, "max": 1.0},
                "theta": {"type": "float", "default": np.pi/4, "min": 0, "max": np.pi},
                "sigma_x": {"type": "int", "default": 5, "min": 1, "max": 20},
                "sigma_y": {"type": "int", "default": 5, "min": 1, "max": 20}
            },
            "laplacian": {
                "ksize": {"type": "int", "default": 3, "min": 1, "max": 31}
            },
            "threshold": {
                "threshold_value": {"type": "int", "default": 127, "min": 0, "max": 255},
                "method": {"type": "str", "default": "binary", 
                          "options": ["binary", "binary_inv", "tozero", "tozero_inv", "trunc"]}
            },
            "top_hat": {
                "kernel_size": {"type": "int", "default": 5, "min": 3, "max": 31}
            },
            "total_variation": {
                "weight": {"type": "float", "default": 0.2, "min": 0.01, "max": 1.0}
            },
            "wavelet": {
                "wavelet": {"type": "str", "default": "db2",
                           "options": ["db1", "db2", "db4", "haar", "bior2.2"]},
                "level": {"type": "int", "default": 1, "min": 1, "max": 4}
            },
            "dog": {
                "sigma1": {"type": "float", "default": 1.0, "min": 0.1, "max": 10.0},
                "sigma2": {"type": "float", "default": 2.0, "min": 0.1, "max": 10.0}
            },
            "canny": {
                "low_threshold": {"type": "int", "default": 50, "min": 0, "max": 255},
                "high_threshold": {"type": "int", "default": 150, "min": 0, "max": 255},
                "aperture_size": {"type": "int", "default": 3, "min": 3, "max": 7}
            },
            "clahe": {
                "clip_limit": {"type": "float", "default": 2.0, "min": 0.1, "max": 40.0},
                "tile_grid_size": {"type": "int", "default": 8, "min": 1, "max": 32}
            }
        }
        
        return parameter_specs.get(filter_name, {})
    
    def undo(self):
        """
        Undo the last filter application, reverting to the previous image state.
        The previous state becomes the new current reference.
        """
        if not self._history:
            raise ValueError("No more actions to undo.")
        self.current_image = self._history.pop()
        print("✓ Undo completed - Previous state is now the current reference")
        return self.current_image