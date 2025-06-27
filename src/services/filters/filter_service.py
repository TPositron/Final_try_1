"""Service for managing filters, presets, and auto-optimization."""

from typing import Dict, List, Optional, Any, Callable
from PySide6.QtCore import QObject, Signal
import numpy as np

from ..core.utils import get_logger


class FilterService(QObject):
    """Service for registering and applying image filters."""
    
    # Signals
    filter_applied = Signal(str, dict, object)  # filter_name, parameters, result_image
    filter_error = Signal(str, str)  # filter_name, error_message
    preset_loaded = Signal(str, list)  # preset_name, filter_chain
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self._registered_filters = {}
        self._presets = {}
        self._filter_history = []
    
    def register_filter(self, name: str, filter_func: Callable, parameters: Dict[str, Any]) -> None:
        """
        Register a new filter function.
        
        Args:
            name: Name of the filter
            filter_func: Function that applies the filter
            parameters: Default parameters for the filter
        """
        self._registered_filters[name] = {
            'function': filter_func,
            'default_parameters': parameters.copy(),
            'current_parameters': parameters.copy()
        }
        self.logger.info(f"Registered filter: {name}")
    
    def apply_filter(self, filter_name: str, image: np.ndarray, 
                    parameters: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
        """
        Apply a filter to an image.
        
        Args:
            filter_name: Name of the filter to apply
            image: Input image
            parameters: Override parameters for the filter
            
        Returns:
            Filtered image or None if failed
        """
        if filter_name not in self._registered_filters:
            error_msg = f"Filter '{filter_name}' not registered"
            self.logger.error(error_msg)
            self.filter_error.emit(filter_name, error_msg)
            return None
        
        try:
            filter_info = self._registered_filters[filter_name]
            
            # Use provided parameters or current/default parameters
            if parameters is None:
                parameters = filter_info['current_parameters'].copy()
            
            # Apply the filter
            result = filter_info['function'](image, **parameters)
            
            # Update current parameters
            filter_info['current_parameters'] = parameters.copy()
            
            # Add to history
            self._filter_history.append({
                'filter_name': filter_name,
                'parameters': parameters.copy(),
                'input_shape': image.shape
            })
            
            self.filter_applied.emit(filter_name, parameters, result)
            self.logger.info(f"Applied filter '{filter_name}' with parameters: {parameters}")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to apply filter '{filter_name}': {e}"
            self.logger.error(error_msg)
            self.filter_error.emit(filter_name, error_msg)
            return None
    
    def apply_filter_chain(self, filter_chain: List[Dict], image: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply a chain of filters to an image.
        
        Args:
            filter_chain: List of filter dictionaries with 'name' and 'parameters'
            image: Input image
            
        Returns:
            Final filtered image or None if any filter failed
        """
        current_image = image.copy()
        
        for filter_step in filter_chain:
            filter_name = filter_step['name']
            parameters = filter_step.get('parameters', {})
            
            current_image = self.apply_filter(filter_name, current_image, parameters)
            if current_image is None:
                return None
        
        return current_image
    
    def get_available_filters(self) -> List[str]:
        """Get list of available filter names."""
        return list(self._registered_filters.keys())
    
    def get_filter_parameters(self, filter_name: str) -> Optional[Dict[str, Any]]:
        """Get current parameters for a filter."""
        if filter_name in self._registered_filters:
            return self._registered_filters[filter_name]['current_parameters'].copy()
        return None
    
    def set_filter_parameters(self, filter_name: str, parameters: Dict[str, Any]) -> bool:
        """
        Set parameters for a filter.
        
        Args:
            filter_name: Name of the filter
            parameters: Parameters to set
            
        Returns:
            True if successful, False otherwise
        """
        if filter_name not in self._registered_filters:
            return False
        
        self._registered_filters[filter_name]['current_parameters'].update(parameters)
        return True
    
    def reset_filter_parameters(self, filter_name: str) -> bool:
        """Reset filter parameters to defaults."""
        if filter_name not in self._registered_filters:
            return False
        
        filter_info = self._registered_filters[filter_name]
        filter_info['current_parameters'] = filter_info['default_parameters'].copy()
        return True
    
    def save_preset(self, name: str, filter_chain: List[Dict]) -> None:
        """
        Save a filter chain as a preset.
        
        Args:
            name: Name of the preset
            filter_chain: List of filter dictionaries
        """
        self._presets[name] = filter_chain.copy()
        self.logger.info(f"Saved preset '{name}' with {len(filter_chain)} filters")
    
    def load_preset(self, name: str) -> Optional[List[Dict]]:
        """Load a filter preset."""
        if name in self._presets:
            preset = self._presets[name].copy()
            self.preset_loaded.emit(name, preset)
            return preset
        return None
    
    def get_presets(self) -> List[str]:
        """Get list of available preset names."""
        return list(self._presets.keys())
    
    def auto_optimize_filters(self, image: np.ndarray, target_metrics: Dict[str, float],
                            max_iterations: int = 10) -> List[Dict]:
        """
        Auto-optimize filter parameters for target metrics.
        
        Args:
            image: Input image
            target_metrics: Target values for image metrics
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimized filter chain
        """
        # This is a placeholder for auto-optimization logic
        # In practice, this would use optimization algorithms to find
        # the best filter parameters
        
        self.logger.info(f"Starting auto-optimization with {max_iterations} iterations")
        
        # For now, return a simple filter chain
        optimized_chain = [
            {'name': 'gaussian_blur', 'parameters': {'sigma': 1.0}},
            {'name': 'contrast_enhancement', 'parameters': {'factor': 1.2}}
        ]
        
        return optimized_chain
    
    def get_filter_history(self) -> List[Dict]:
        """Get the history of applied filters."""
        return self._filter_history.copy()
    
    def clear_history(self) -> None:
        """Clear the filter history."""
        self._filter_history.clear()
