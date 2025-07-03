"""
Simple Image Processing Service for basic filter management (Steps 76-80).

This service provides:
- QObject for filter management with basic filter registry (Step 76)
- Filter application to SEM images with basic parameter handling (Step 77)  
- Filter presets save/load and basic preset management (Step 78)
- Basic automatic filter selection with simple scoring (Step 79)
- Processing signals when filters applied with progress reporting (Step 80)
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import numpy as np
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class ImageProcessingService(QObject):
    """Simple QObject-based image processing service for filter management."""
    
    # Signals for Step 80
    filter_applied = Signal(str, dict, object)     # filter_name, parameters, result_image
    filter_progress = Signal(str)                  # progress_message
    filter_error = Signal(str)                     # error_message
    filter_previewed = Signal(str, dict, object)  # filter_name, parameters, preview_image
    preset_saved = Signal(str)                     # preset_name
    preset_loaded = Signal(str, dict)             # preset_name, preset_data
    processing_started = Signal(str)              # operation_description
    processing_finished = Signal(str)             # operation_description
    processing_progress = Signal(int)             # progress_percentage
    error_occurred = Signal(str)                 # error_message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Step 76: Basic filter registry
        self.filter_registry = {}
        self._register_basic_filters()
        
        # Current processing state
        self.current_image = None
        self.original_image = None
        self.filter_history = []
        
        # Step 78: Preset management
        self.presets_dir = Path("config/filter_presets")
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_presets = {}
        self._load_default_presets()
    
    def _register_basic_filters(self):
        """Register basic filter functions."""
        
        def gaussian_blur(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
            """Apply Gaussian blur filter."""
            import cv2
            kernel_size = params.get('kernel_size', 5)
            sigma = params.get('sigma', 1.0)
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        def threshold(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
            """Apply threshold filter."""
            import cv2
            threshold_value = params.get('threshold', 127)
            max_value = params.get('max_value', 255)
            threshold_type = params.get('type', cv2.THRESH_BINARY)
            _, result = cv2.threshold(image, threshold_value, max_value, threshold_type)
            return result
        
        def edge_detection(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
            """Apply Canny edge detection."""
            import cv2
            low_threshold = params.get('low_threshold', 50)
            high_threshold = params.get('high_threshold', 150)
            return cv2.Canny(image.astype(np.uint8), low_threshold, high_threshold)
        
        def median_filter(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
            """Apply median filter."""
            import cv2
            kernel_size = params.get('kernel_size', 5)
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
            return cv2.medianBlur(image.astype(np.uint8), kernel_size)
        
        def clahe_filter(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
            """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
            import cv2
            clip_limit = params.get('clip_limit', 2.0)
            tile_grid_size = params.get('tile_grid_size', 8)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            return clahe.apply(image.astype(np.uint8))
        
        # Register filters
        self.filter_registry = {
            'gaussian_blur': {
                'function': gaussian_blur,
                'name': 'Gaussian Blur',
                'description': 'Smooth image using Gaussian kernel',
                'parameters': {
                    'kernel_size': {'type': 'int', 'min': 3, 'max': 15, 'default': 5},
                    'sigma': {'type': 'float', 'min': 0.1, 'max': 5.0, 'default': 1.0}
                }
            },
            'threshold': {
                'function': threshold,
                'name': 'Threshold',
                'description': 'Binary threshold segmentation',
                'parameters': {
                    'threshold': {'type': 'int', 'min': 0, 'max': 255, 'default': 127},
                    'max_value': {'type': 'int', 'min': 0, 'max': 255, 'default': 255}
                }
            },
            'edge_detection': {
                'function': edge_detection,
                'name': 'Edge Detection',
                'description': 'Canny edge detection',
                'parameters': {
                    'low_threshold': {'type': 'int', 'min': 0, 'max': 255, 'default': 50},
                    'high_threshold': {'type': 'int', 'min': 0, 'max': 255, 'default': 150}
                }
            },
            'median_filter': {
                'function': median_filter,
                'name': 'Median Filter',
                'description': 'Noise reduction using median filter',
                'parameters': {
                    'kernel_size': {'type': 'int', 'min': 3, 'max': 15, 'default': 5}
                }
            },
            'clahe': {
                'function': clahe_filter,
                'name': 'CLAHE',
                'description': 'Contrast Limited Adaptive Histogram Equalization',
                'parameters': {
                    'clip_limit': {'type': 'float', 'min': 1.0, 'max': 10.0, 'default': 2.0},
                    'tile_grid_size': {'type': 'int', 'min': 4, 'max': 16, 'default': 8}
                }
            }
        }
        
        logger.info(f"Registered {len(self.filter_registry)} basic filters")
    
    def get_available_filters(self) -> List[str]:
        """Get list of available filter names."""
        return list(self.filter_registry.keys())
    
    def get_filter_history(self) -> List[Dict[str, Any]]:
        """Get the current filter history."""
        return self.filter_history.copy()

    def get_filter_info(self, filter_name: str) -> Dict[str, Any]:
        """Get information about a specific filter."""
        if filter_name not in self.filter_registry:
            raise ValueError(f"Unknown filter: {filter_name}")
        
        filter_info = self.filter_registry[filter_name].copy()
        filter_info.pop('function', None)  # Don't return the function object
        return filter_info
    
    def set_image(self, image: np.ndarray):
        """Set the current image for processing."""
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.filter_history = []
        logger.info(f"Image set: shape {image.shape}")
    
    def set_current_image(self, image: np.ndarray):
        """Set the current image (for cascading filters)."""
        self.current_image = image.copy()
        logger.info(f"Current image updated: shape {image.shape}")
    
    def set_original_image(self, image: np.ndarray):
        """Alias for set_image for backward compatibility."""
        self.set_image(image)

    # Step 77: Implement filter application with basic parameter handling
    def apply_filter(self, filter_name: str, parameters: Dict[str, Any] = None, preview_only: bool = False) -> Optional[np.ndarray]:
        """
        Apply filter to current image with basic parameter handling.
        
        Args:
            filter_name: Name of filter to apply
            parameters: Filter parameters
            preview_only: If True, only preview without applying permanently
            
        Returns:
            Filtered image array or None if failed
        """
        try:
            if self.current_image is None:
                raise ValueError("No image loaded")
            
            if filter_name not in self.filter_registry:
                raise ValueError(f"Unknown filter: {filter_name}")
            
            # Emit processing started signal
            self.processing_started.emit(f"Applying {filter_name} filter")
            self.processing_progress.emit(25)
            
            # Get filter function and default parameters
            filter_info = self.filter_registry[filter_name]
            filter_function = filter_info['function']
            
            # Merge with default parameters and validate
            final_params = {}
            for param_name, param_info in filter_info['parameters'].items():
                final_params[param_name] = param_info['default']
            
            if parameters:
                # Validate and update parameters
                for param_name, param_value in parameters.items():
                    if param_name in filter_info['parameters']:
                        param_info = filter_info['parameters'][param_name]
                        
                        # Validate parameter bounds
                        if 'min' in param_info and param_value < param_info['min']:
                            param_value = param_info['min']
                            logger.warning(f"Parameter {param_name} clamped to minimum: {param_info['min']}")
                        if 'max' in param_info and param_value > param_info['max']:
                            param_value = param_info['max']
                            logger.warning(f"Parameter {param_name} clamped to maximum: {param_info['max']}")
                        
                        final_params[param_name] = param_value
            
            logger.info(f"Applying filter '{filter_name}' with parameters: {final_params}")
            
            # Apply filter
            self.processing_progress.emit(50)
            filtered_image = filter_function(self.current_image, final_params)
            self.processing_progress.emit(75)
            
            if preview_only:
                # Emit preview signal
                self.filter_previewed.emit(filter_name, final_params, filtered_image)
            else:
                # Save to history and update current image
                self.filter_history.append({
                    'image': self.current_image.copy(),
                    'filter': filter_name,
                    'parameters': final_params
                })
                self.current_image = filtered_image
                
                # Emit applied signal
                self.filter_applied.emit(filter_name, final_params, filtered_image)
            
            self.processing_progress.emit(100)
            self.processing_finished.emit(f"Filter {filter_name} applied successfully")
            
            return filtered_image
            
        except Exception as e:
            error_msg = f"Filter application failed for '{filter_name}': {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            self.processing_finished.emit(f"Filter {filter_name} failed")
            return None
    
    def reset_to_original(self):
        """Reset image to original state."""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.filter_history = []
            logger.info("Image reset to original")
    
    def undo_last_filter(self):
        """Undo the last applied filter."""
        if self.filter_history:
            last_state = self.filter_history.pop()
            self.current_image = last_state['image']
            logger.info(f"Undid filter: {last_state['filter']}")
    
    def preview_filter(self, filter_name: str, parameters: Dict[str, Any] = None, image: np.ndarray = None) -> Optional[np.ndarray]:
        """
        Preview filter without applying it permanently.
        
        Args:
            filter_name: Name of filter to preview
            parameters: Filter parameters
            image: Image to preview on (optional, uses current if not provided)
            
        Returns:
            Preview image or None if failed
        """
        try:
            # Temporarily set the image if provided
            original_current = self.current_image
            if image is not None:
                self.current_image = image
            
            # Apply filter in preview mode
            preview_result = self.apply_filter(filter_name, parameters, preview_only=True)
            
            # Restore original current image
            self.current_image = original_current
            
            if preview_result is not None:
                self.filter_previewed.emit(filter_name, parameters or {}, preview_result)
            
            return preview_result
            
        except Exception as e:
            error_msg = f"Filter preview failed: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return None
    
    # Step 78: Add filter presets save/load and basic preset management
    def save_preset(self, preset_name: str, filters: List[Dict[str, Any]]) -> bool:
        """
        Save filter preset configuration.
        
        Args:
            preset_name: Name for the preset
            filters: List of filter configurations
            
        Returns:
            True if saved successfully
        """
        try:
            preset_data = {
                'name': preset_name,
                'filters': filters,
                'created': str(Path().resolve()),  # Current timestamp as string
                'description': f"Preset with {len(filters)} filters"
            }
            
            preset_path = self.presets_dir / f"{preset_name}.json"
            with open(preset_path, 'w') as f:
                json.dump(preset_data, f, indent=2)
            
            self.loaded_presets[preset_name] = preset_data
            logger.info(f"Preset saved: {preset_name}")
            
            # Emit signal
            self.preset_saved.emit(preset_name)
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to save preset: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def load_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Load filter preset configuration.
        
        Args:
            preset_name: Name of preset to load
            
        Returns:
            Preset data or None if failed
        """
        try:
            preset_path = self.presets_dir / f"{preset_name}.json"
            
            if not preset_path.exists():
                raise FileNotFoundError(f"Preset not found: {preset_name}")
            
            with open(preset_path, 'r') as f:
                preset_data = json.load(f)
            
            self.loaded_presets[preset_name] = preset_data
            logger.info(f"Preset loaded: {preset_name}")
            
            # Emit signal
            self.preset_loaded.emit(preset_name, preset_data)
            
            return preset_data
            
        except Exception as e:
            error_msg = f"Failed to load preset: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return None
    
    def get_available_presets(self) -> List[str]:
        """Get list of available preset names."""
        presets = []
        for preset_file in self.presets_dir.glob("*.json"):
            presets.append(preset_file.stem)
        return sorted(presets)
    
    def apply_preset(self, preset_name: str) -> bool:
        """
        Apply all filters from a preset.
        
        Args:
            preset_name: Name of preset to apply
            
        Returns:
            True if applied successfully
        """
        try:
            preset_data = self.load_preset(preset_name)
            if not preset_data:
                return False
            
            filters = preset_data.get('filters', [])
            
            self.processing_started.emit(f"Applying preset: {preset_name}")
            
            for i, filter_config in enumerate(filters):
                filter_name = filter_config.get('name')
                parameters = filter_config.get('parameters', {})
                
                # Update progress
                progress = int((i + 1) / len(filters) * 100)
                self.processing_progress.emit(progress)
                
                # Apply filter
                result = self.apply_filter(filter_name, parameters, preview_only=False)
                if result is None:
                    raise Exception(f"Failed to apply filter: {filter_name}")
            
            self.processing_finished.emit(f"Preset {preset_name} applied successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to apply preset: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def _load_default_presets(self):
        """Load default filter presets."""
        default_presets = {
            'noise_reduction': {
                'name': 'Noise Reduction',
                'filters': [
                    {'name': 'gaussian_blur', 'parameters': {'kernel_size': 3, 'sigma': 0.8}},
                    {'name': 'median_filter', 'parameters': {'kernel_size': 3}}
                ]
            },
            'edge_enhancement': {
                'name': 'Edge Enhancement',
                'filters': [
                    {'name': 'clahe', 'parameters': {'clip_limit': 3.0, 'tile_grid_size': 8}},
                    {'name': 'edge_detection', 'parameters': {'low_threshold': 50, 'high_threshold': 150}}
                ]
            },
            'segmentation': {
                'name': 'Segmentation',
                'filters': [
                    {'name': 'clahe', 'parameters': {'clip_limit': 2.0}},
                    {'name': 'gaussian_blur', 'parameters': {'kernel_size': 3}},
                    {'name': 'threshold', 'parameters': {'threshold': 127}}
                ]
            }
        }
        
        for preset_name, preset_data in default_presets.items():
            self.save_preset(preset_name, preset_data['filters'])
    
    # Step 79: Implement basic automatic filter selection with simple scoring
    def auto_select_filters(self, target_metric: str = "contrast") -> List[Dict[str, Any]]:
        """
        Basic automatic filter selection with simple scoring method.
        
        Args:
            target_metric: Target metric to optimize ("contrast", "sharpness", "noise_reduction")
            
        Returns:
            List of recommended filter configurations
        """
        try:
            if self.current_image is None:
                raise ValueError("No image loaded")
            
            self.processing_started.emit("Analyzing image for auto filter selection")
            
            # Simple image analysis
            image_stats = self._analyze_image(self.current_image)
            
            # Select filters based on target metric and image characteristics
            recommended_filters = []
            
            if target_metric == "contrast":
                if image_stats['contrast'] < 50:  # Low contrast
                    recommended_filters.append({
                        'name': 'clahe',
                        'parameters': {'clip_limit': 3.0, 'tile_grid_size': 8},
                        'reason': 'Low contrast detected'
                    })
                    
            elif target_metric == "sharpness":
                if image_stats['sharpness'] < 30:  # Low sharpness
                    recommended_filters.append({
                        'name': 'edge_detection',
                        'parameters': {'low_threshold': 40, 'high_threshold': 120},
                        'reason': 'Low sharpness detected'
                    })
                    
            elif target_metric == "noise_reduction":
                if image_stats['noise'] > 20:  # High noise
                    recommended_filters.append({
                        'name': 'gaussian_blur',
                        'parameters': {'kernel_size': 3, 'sigma': 1.0},
                        'reason': 'High noise detected'
                    })
                    recommended_filters.append({
                        'name': 'median_filter',
                        'parameters': {'kernel_size': 3},
                        'reason': 'Additional noise reduction'
                    })
            
            # If no specific filters recommended, suggest basic enhancement
            if not recommended_filters:
                recommended_filters.append({
                    'name': 'clahe',
                    'parameters': {'clip_limit': 2.0, 'tile_grid_size': 8},
                    'reason': 'General image enhancement'
                })
            
            self.processing_finished.emit(f"Auto selection complete: {len(recommended_filters)} filters recommended")
            
            return recommended_filters
            
        except Exception as e:
            error_msg = f"Auto filter selection failed: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return []
    
    def _analyze_image(self, image: np.ndarray) -> Dict[str, float]:
        """
        Simple image analysis for basic metrics.
        
        Args:
            image: Image to analyze
            
        Returns:
            Dictionary with image metrics
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                import cv2
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate basic metrics
            contrast = np.std(gray)  # Standard deviation as contrast measure
            brightness = np.mean(gray)  # Mean intensity
            
            # Simple sharpness measure using Laplacian variance
            import cv2
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Simple noise estimate using high-frequency content
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            noise_map = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            noise = np.std(noise_map)
            
            return {
                'contrast': float(contrast),
                'brightness': float(brightness),
                'sharpness': float(sharpness),
                'noise': float(noise)
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {'contrast': 0, 'brightness': 0, 'sharpness': 0, 'noise': 0}
    
    def get_processing_history(self) -> List[Dict[str, Any]]:
        """Get the history of applied filters."""
        return self.filter_history.copy()
    
    def get_current_image(self) -> Optional[np.ndarray]:
        """Get the current processed image."""
        return self.current_image.copy() if self.current_image is not None else None
    
    def get_original_image(self) -> Optional[np.ndarray]:
        """Get the original unprocessed image."""
        return self.original_image.copy() if self.original_image is not None else None
    
    def get_filters_by_category(self) -> Dict[str, List[str]]:
        """Get filters organized by category for automatic filtering."""
        return {
            'contrast': ['clahe'],
            'denoising': ['gaussian_blur', 'median_filter'],
            'binarisation': ['threshold'],
            'edge_detection': ['edge_detection']
        }
    
    def get_filter_presets_by_category(self) -> Dict[str, Dict[str, Dict]]:
        """Get predefined filter presets organized by category."""
        return {
            'contrast': {
                'light_clahe': {
                    'filter': 'clahe',
                    'parameters': {'clip_limit': 1.5, 'tile_grid_size': 8}
                },
                'medium_clahe': {
                    'filter': 'clahe', 
                    'parameters': {'clip_limit': 2.5, 'tile_grid_size': 8}
                },
                'strong_clahe': {
                    'filter': 'clahe',
                    'parameters': {'clip_limit': 4.0, 'tile_grid_size': 6}
                }
            },
            'denoising': {
                'light_blur': {
                    'filter': 'gaussian_blur',
                    'parameters': {'kernel_size': 3, 'sigma': 0.8}
                },
                'medium_blur': {
                    'filter': 'gaussian_blur',
                    'parameters': {'kernel_size': 5, 'sigma': 1.2}
                },
                'strong_median': {
                    'filter': 'median_filter',
                    'parameters': {'kernel_size': 5}
                }
            },
            'binarisation': {
                'low_threshold': {
                    'filter': 'threshold',
                    'parameters': {'threshold': 85, 'max_value': 255}
                },
                'medium_threshold': {
                    'filter': 'threshold',
                    'parameters': {'threshold': 127, 'max_value': 255}
                },
                'high_threshold': {
                    'filter': 'threshold',
                    'parameters': {'threshold': 170, 'max_value': 255}
                }
            },
            'edge_detection': {
                'soft_canny': {
                    'filter': 'edge_detection',
                    'parameters': {'low_threshold': 30, 'high_threshold': 100}
                },
                'normal_canny': {
                    'filter': 'edge_detection',
                    'parameters': {'low_threshold': 50, 'high_threshold': 150}
                },
                'sharp_canny': {
                    'filter': 'edge_detection',
                    'parameters': {'low_threshold': 80, 'high_threshold': 200}
                }
            }
        }

    # Step 78: Preset management
