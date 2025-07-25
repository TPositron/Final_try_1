"""
Enhanced Filter Service with Parameter System Integration

This module provides a comprehensive filter service that integrates dynamic parameter
discovery, validation, and filter application with Qt-based signal/slot communication.
It serves as the main interface for applying image filters with automatic parameter
validation and management.

Dependencies:
- PySide6.QtCore: For Qt signal/slot communication and QObject inheritance
- numpy: For image array processing
- importlib.util: For dynamic module loading
- FilterParameterStorage: For parameter definition storage and retrieval
- FilterDefinition, FilterParameter: Data classes for filter metadata

Main Classes:
- EnhancedFilterService: Main service class inheriting from QObject

Key Methods:
- __init__(): Initialize service with filters directory and parameter storage
- apply_filter(): Apply a single filter to an image with parameter validation
- apply_filter_chain(): Apply a sequence of filters to an image
- get_available_filters(): Return list of all available filter names
- get_filter_definition(): Get complete filter definition including parameters
- get_filter_parameters(): Get parameter definitions for a specific filter
- get_filters_by_category(): Get filters organized by category
- validate_parameters(): Validate filter parameters against definitions
- save_preset()/load_preset(): Save and load filter chains as presets
- refresh_filters(): Reload filter definitions from files
- get_filter_history(): Get history of applied filters
- search_filters(): Search filters by name or description

Signals:
- filter_applied: Emitted when a filter is successfully applied
- filter_error: Emitted when filter application fails
- preset_loaded: Emitted when a preset is loaded
- filters_updated: Emitted when filter definitions are refreshed

Features:
- Automatic parameter validation and type conversion
- Dynamic filter function loading from modules
- Filter chain processing with error handling
- Preset management for saving/loading filter combinations
- Search functionality for finding filters
- Comprehensive error handling and logging
- Signal-based communication for UI integration
"""

import os
import importlib.util
from typing import Dict, List, Optional, Any, Callable, Union
from PySide6.QtCore import QObject, Signal
import numpy as np

from .filter_parameter_storage import FilterParameterStorage
from .filter_parameter_parser import FilterDefinition, FilterParameter


class EnhancedFilterService(QObject):
    """Enhanced filter service with automatic parameter discovery and validation."""
    
    # Signals
    filter_applied = Signal(str, dict, np.ndarray)  # filter_name, parameters, result_image
    filter_error = Signal(str, str)  # filter_name, error_message
    preset_loaded = Signal(str, list)  # preset_name, filter_chain
    filters_updated = Signal()  # Emitted when filter definitions are updated
    
    def __init__(self, filters_directory: Optional[str] = None):
        super().__init__()
        self.logger = self._setup_logger()
        
        # Use default filters directory if not provided
        if filters_directory is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            filters_directory = current_dir
        
        # Initialize parameter storage
        self.parameter_storage = FilterParameterStorage(filters_directory)
        
        # Filter registry
        self._loaded_filter_functions: Dict[str, Dict[str, Any]] = {}
        self._presets: Dict[str, List[Dict[str, Any]]] = {}
        self._filter_history: List[Dict[str, Any]] = []
        
        # Initialize the system
        self._initialize_filters()
    
    def _setup_logger(self):
        """Setup logging for the service."""
        import logging
        return logging.getLogger(__name__)
    
    def _initialize_filters(self) -> bool:
        """Initialize the filter system by loading parameter definitions and filter functions."""
        try:
            # Initialize parameter storage
            if not self.parameter_storage.initialize():
                self.logger.error("Failed to initialize parameter storage")
                return False
            
            # Load filter functions
            self._load_filter_functions()
            
            self.logger.info(f"Enhanced filter service initialized with {len(self._loaded_filter_functions)} filters")
            self.filters_updated.emit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing enhanced filter service: {e}")
            return False
    
    def _load_filter_functions(self):
        """Load filter functions based on parameter definitions."""
        filter_definitions = self.parameter_storage.get_all_filters()
        
        for filter_name, filter_def in filter_definitions.items():
            try:
                # Load the filter function
                filter_func = self._load_filter_function(filter_def)
                if filter_func:
                    self._loaded_filter_functions[filter_name] = {
                        'function': filter_func,
                        'definition': filter_def
                    }
                    self.logger.debug(f"Loaded filter function: {filter_name}")
                else:
                    self.logger.warning(f"Could not load filter function: {filter_name}")
                    
            except Exception as e:
                self.logger.error(f"Error loading filter function {filter_name}: {e}")
    
    def _load_filter_function(self, filter_def: FilterDefinition) -> Optional[Callable]:
        """
        Load a filter function from its module.
        
        Args:
            filter_def: FilterDefinition object
            
        Returns:
            Filter function or None if loading failed
        """
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(
                filter_def.name, 
                filter_def.module_path
            )
            
            if spec is None or spec.loader is None:
                self.logger.error(f"Could not load spec for {filter_def.module_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Try to get the filter function or class
            if hasattr(module, filter_def.function_name):
                filter_obj = getattr(module, filter_def.function_name)
                
                # If it's a class, instantiate it with default parameters
                if isinstance(filter_obj, type):
                    defaults = self.parameter_storage.get_default_parameters(filter_def.name)
                    if defaults:
                        return filter_obj(**defaults)
                    else:
                        return filter_obj()
                else:
                    # It's a function, return it directly
                    return filter_obj
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading filter function from {filter_def.module_path}: {e}")
            return None
    
    def get_available_filters(self) -> List[str]:
        """Get list of available filter names."""
        return list(self._loaded_filter_functions.keys())
    
    def get_filter_definition(self, filter_name: str) -> Optional[FilterDefinition]:
        """Get filter definition by name."""
        return self.parameter_storage.get_filter_definition(filter_name)
    
    def get_filter_parameters(self, filter_name: str) -> List[FilterParameter]:
        """Get parameters for a specific filter."""
        return self.parameter_storage.get_filter_parameters(filter_name)
    
    def get_filters_by_category(self, category: str) -> List[str]:
        """Get filter names by category."""
        filter_defs = self.parameter_storage.get_filters_by_category(category)
        return [f.name for f in filter_defs if f.name in self._loaded_filter_functions]
    
    def get_all_categories(self) -> List[str]:
        """Get all available filter categories."""
        return self.parameter_storage.get_all_categories()
    
    def apply_filter(self, filter_name: str, image: np.ndarray, 
                    parameters: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
        """
        Apply a filter to an image with parameter validation.
        
        Args:
            filter_name: Name of the filter to apply
            image: Input image
            parameters: Filter parameters
            
        Returns:
            Filtered image or None if failed
        """
        if filter_name not in self._loaded_filter_functions:
            error_msg = f"Filter '{filter_name}' not available"
            self.logger.error(error_msg)
            self.filter_error.emit(filter_name, error_msg)
            return None
        
        try:
            # Get filter function and definition
            filter_info = self._loaded_filter_functions[filter_name]
            filter_func = filter_info['function']
            filter_def = filter_info['definition']
            
            # Validate and normalize parameters
            if parameters is None:
                parameters = {}
            
            validated_params = self.parameter_storage.validate_filter_parameters(
                filter_name, parameters
            )
            
            # Apply the filter
            result: Optional[np.ndarray] = None
            
            if callable(filter_func):
                # For functions, pass image as first argument
                result_obj = filter_func(image, **validated_params)
            elif hasattr(filter_func, '__call__'):
                # For callable objects (classes with __call__)
                result_obj = filter_func(image)
            elif hasattr(filter_func, 'apply'):
                # For objects with apply method
                result_obj = filter_func.apply(image)
            else:
                raise ValueError(f"Filter function {filter_name} is not callable")
            
            # Ensure result is a numpy array or None
            if isinstance(result_obj, np.ndarray):
                result = result_obj
            elif result_obj is None:
                result = None
            else:
                self.logger.warning(f"Filter {filter_name} returned unexpected type: {type(result_obj)}")
                return None
            
            # Add to history
            self._filter_history.append({
                'filter_name': filter_name,
                'parameters': validated_params,
                'input_shape': image.shape,
                'output_shape': result.shape if result is not None else None
            })
            
            if result is not None:
                self.filter_applied.emit(filter_name, validated_params, result)
            
            self.logger.info(f"Applied filter '{filter_name}' with parameters: {validated_params}")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to apply filter '{filter_name}': {e}"
            self.logger.error(error_msg)
            self.filter_error.emit(filter_name, error_msg)
            return None
    
    def apply_filter_chain(self, filter_chain: List[Dict[str, Any]], 
                          image: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply a chain of filters to an image.
        
        Args:
            filter_chain: List of dictionaries with 'name' and 'parameters'
            image: Input image
            
        Returns:
            Final filtered image or None if any filter failed
        """
        current_image = image.copy()
        
        for i, filter_step in enumerate(filter_chain):
            filter_name = filter_step.get('name')
            parameters = filter_step.get('parameters', {})
            
            if not filter_name:
                self.logger.error(f"Filter chain step {i} missing 'name'")
                return None
            
            current_image = self.apply_filter(filter_name, current_image, parameters)
            if current_image is None:
                self.logger.error(f"Filter chain failed at step {i}: {filter_name}")
                return None
        
        return current_image
    
    def get_default_parameters(self, filter_name: str) -> Dict[str, Any]:
        """Get default parameters for a filter."""
        return self.parameter_storage.get_default_parameters(filter_name)
    
    def validate_parameters(self, filter_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate filter parameters."""
        return self.parameter_storage.validate_filter_parameters(filter_name, parameters)
    
    def search_filters(self, query: str) -> List[str]:
        """Search for filters by name or description."""
        results = self.parameter_storage.search_filters(query)
        return [f.name for f in results if f.name in self._loaded_filter_functions]
    
    def get_filter_history(self) -> List[Dict[str, Any]]:
        """Get filter application history."""
        return self._filter_history.copy()
    
    def clear_history(self):
        """Clear filter application history."""
        self._filter_history.clear()
    
    def refresh_filters(self) -> bool:
        """Refresh filter definitions and reload functions."""
        try:
            if self.parameter_storage.refresh_filters():
                self._loaded_filter_functions.clear()
                self._load_filter_functions()
                self.filters_updated.emit()
                self.logger.info("Filter system refreshed successfully")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error refreshing filters: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filter system statistics."""
        base_stats = self.parameter_storage.get_filter_statistics()
        base_stats.update({
            'loaded_functions': len(self._loaded_filter_functions),
            'history_entries': len(self._filter_history)
        })
        return base_stats
    
    # Preset management
    def save_preset(self, name: str, filter_chain: List[Dict[str, Any]]) -> bool:
        """Save a filter chain as a preset."""
        try:
            self._presets[name] = filter_chain.copy()
            self.logger.info(f"Saved preset: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving preset {name}: {e}")
            return False
    
    def load_preset(self, name: str) -> Optional[List[Dict[str, Any]]]:
        """Load a filter chain preset."""
        if name in self._presets:
            preset = self._presets[name].copy()
            self.preset_loaded.emit(name, preset)
            return preset
        return None
    
    def get_preset_names(self) -> List[str]:
        """Get list of available preset names."""
        return list(self._presets.keys())
    
    def delete_preset(self, name: str) -> bool:
        """Delete a preset."""
        if name in self._presets:
            del self._presets[name]
            self.logger.info(f"Deleted preset: {name}")
            return True
        return False