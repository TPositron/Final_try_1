"""
Filter Parameter Storage Service
Manages storage and retrieval of filter parameter definitions.
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from .filter_parameter_parser import (
    FilterParameterParser, 
    FilterDefinition, 
    FilterParameter,
    ParameterType
)


class FilterParameterStorage:
    """Service for storing and managing filter parameter definitions."""
    
    def __init__(self, filters_directory: str, cache_file: str = "filter_parameters_cache.json"):
        """
        Initialize the filter parameter storage.
        
        Args:
            filters_directory: Path to the directory containing filter files
            cache_file: Path to the cache file for storing parsed parameters
        """
        self.filters_directory = filters_directory
        self.cache_file = cache_file
        self.parser = FilterParameterParser(filters_directory)
        self.filter_definitions: Dict[str, FilterDefinition] = {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging for the storage service."""
        import logging
        return logging.getLogger(__name__)
    
    def initialize(self, force_refresh: bool = False) -> bool:
        """
        Initialize the storage by loading or parsing filter definitions.
        
        Args:
            force_refresh: If True, force re-parsing even if cache exists
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if not force_refresh and self._load_from_cache():
                self.logger.info("Filter definitions loaded from cache")
                return True
            
            # Parse filter files
            self.logger.info("Parsing filter files...")
            self.filter_definitions = self.parser.parse_all_filters()
            
            if self.filter_definitions:
                # Save to cache
                self._save_to_cache()
                self.logger.info(f"Initialized with {len(self.filter_definitions)} filter definitions")
                return True
            else:
                self.logger.warning("No filter definitions found")
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing filter parameter storage: {e}")
            return False
    
    def _load_from_cache(self) -> bool:
        """
        Load filter definitions from cache file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.cache_file):
            return False
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert JSON data back to FilterDefinition objects
            self.filter_definitions = {}
            
            for name, filter_data in data.items():
                # Convert parameter data back to FilterParameter objects
                parameters = []
                for param_data in filter_data.get('parameters', []):
                    param = FilterParameter(
                        name=param_data['name'],
                        param_type=ParameterType(param_data['type']),
                        default_value=param_data['default_value'],
                        min_value=param_data.get('min_value'),
                        max_value=param_data.get('max_value'),
                        description=param_data.get('description', ''),
                        step=param_data.get('step'),
                        choices=param_data.get('choices'),
                        required=param_data.get('required', True)
                    )
                    parameters.append(param)
                
                # Create FilterDefinition object
                filter_def = FilterDefinition(
                    name=filter_data['name'],
                    display_name=filter_data['display_name'],
                    function_name=filter_data['function_name'],
                    module_path=filter_data['module_path'],
                    parameters=parameters,
                    description=filter_data.get('description', ''),
                    category=filter_data.get('category', 'General')
                )
                
                self.filter_definitions[name] = filter_def
            
            return len(self.filter_definitions) > 0
            
        except Exception as e:
            self.logger.error(f"Error loading filter definitions from cache: {e}")
            return False
    
    def _save_to_cache(self) -> bool:
        """
        Save filter definitions to cache file.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Convert FilterDefinition objects to JSON-serializable format
            cache_data = {}
            
            for name, filter_def in self.filter_definitions.items():
                cache_data[name] = {
                    'name': filter_def.name,
                    'display_name': filter_def.display_name,
                    'function_name': filter_def.function_name,
                    'module_path': filter_def.module_path,
                    'description': filter_def.description,
                    'category': filter_def.category,
                    'parameters': [
                        {
                            'name': p.name,
                            'type': p.param_type.value,
                            'default_value': p.default_value,
                            'min_value': p.min_value,
                            'max_value': p.max_value,
                            'step': p.step,
                            'description': p.description,
                            'required': p.required,
                            'choices': p.choices
                        }
                        for p in filter_def.parameters
                    ]
                }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Filter definitions cached to {self.cache_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving filter definitions to cache: {e}")
            return False
    
    def get_filter_definition(self, filter_name: str) -> Optional[FilterDefinition]:
        """
        Get filter definition by name.
        
        Args:
            filter_name: Name of the filter
            
        Returns:
            FilterDefinition object or None if not found
        """
        return self.filter_definitions.get(filter_name)
    
    def get_all_filters(self) -> Dict[str, FilterDefinition]:
        """Get all filter definitions."""
        return self.filter_definitions.copy()
    
    def get_filter_names(self) -> List[str]:
        """Get list of all filter names."""
        return list(self.filter_definitions.keys())
    
    def get_filters_by_category(self, category: str) -> List[FilterDefinition]:
        """
        Get all filters in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of FilterDefinition objects
        """
        return [f for f in self.filter_definitions.values() if f.category == category]
    
    def get_all_categories(self) -> List[str]:
        """Get all available filter categories."""
        return sorted(list(set(f.category for f in self.filter_definitions.values())))
    
    def get_filter_parameters(self, filter_name: str) -> List[FilterParameter]:
        """
        Get parameters for a specific filter.
        
        Args:
            filter_name: Name of the filter
            
        Returns:
            List of FilterParameter objects
        """
        filter_def = self.get_filter_definition(filter_name)
        return filter_def.parameters if filter_def else []
    
    def validate_filter_parameters(self, filter_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize filter parameters.
        
        Args:
            filter_name: Name of the filter
            parameters: Dictionary of parameter values
            
        Returns:
            Dictionary of validated parameters
        """
        filter_def = self.get_filter_definition(filter_name)
        if not filter_def:
            raise ValueError(f"Filter '{filter_name}' not found")
        
        validated_params = {}
        
        for param_def in filter_def.parameters:
            param_name = param_def.name
            
            if param_name in parameters:
                value = parameters[param_name]
                validated_value = self._validate_parameter_value(param_def, value)
                validated_params[param_name] = validated_value
            elif param_def.required and param_def.default_value is None:
                raise ValueError(f"Required parameter '{param_name}' missing for filter '{filter_name}'")
            elif param_def.default_value is not None:
                validated_params[param_name] = param_def.default_value
        
        return validated_params
    
    def _validate_parameter_value(self, param_def: FilterParameter, value: Any) -> Any:
        """
        Validate a single parameter value.
        
        Args:
            param_def: FilterParameter definition
            value: Value to validate
            
        Returns:
            Validated value
        """
        # Type conversion
        if param_def.param_type == ParameterType.INTEGER:
            value = int(value)
        elif param_def.param_type == ParameterType.FLOAT:
            value = float(value)
        elif param_def.param_type == ParameterType.BOOLEAN:
            if isinstance(value, str):
                value = value.lower() in ('true', '1', 'yes', 'on')
            else:
                value = bool(value)
        elif param_def.param_type == ParameterType.STRING:
            value = str(value)
        
        # Range validation
        if param_def.min_value is not None and value < param_def.min_value:
            value = param_def.min_value
        if param_def.max_value is not None and value > param_def.max_value:
            value = param_def.max_value
        
        # Choice validation
        if param_def.choices and value not in param_def.choices:
            raise ValueError(f"Value '{value}' not in allowed choices: {param_def.choices}")
        
        return value
    
    def get_default_parameters(self, filter_name: str) -> Dict[str, Any]:
        """
        Get default parameters for a filter.
        
        Args:
            filter_name: Name of the filter
            
        Returns:
            Dictionary of default parameter values
        """
        filter_def = self.get_filter_definition(filter_name)
        if not filter_def:
            return {}
        
        defaults = {}
        for param in filter_def.parameters:
            if param.default_value is not None:
                defaults[param.name] = param.default_value
        
        return defaults
    
    def refresh_filters(self) -> bool:
        """
        Refresh filter definitions by re-parsing all filter files.
        
        Returns:
            True if refresh successful, False otherwise
        """
        return self.initialize(force_refresh=True)
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded filters.
        
        Returns:
            Dictionary with filter statistics
        """
        if not self.filter_definitions:
            return {}
        
        categories = {}
        total_parameters = 0
        
        for filter_def in self.filter_definitions.values():
            category = filter_def.category
            categories[category] = categories.get(category, 0) + 1
            total_parameters += len(filter_def.parameters)
        
        return {
            'total_filters': len(self.filter_definitions),
            'total_parameters': total_parameters,
            'categories': categories,
            'avg_parameters_per_filter': total_parameters / len(self.filter_definitions) if self.filter_definitions else 0
        }
    
    def search_filters(self, query: str) -> List[FilterDefinition]:
        """
        Search for filters by name or description.
        
        Args:
            query: Search query
            
        Returns:
            List of matching FilterDefinition objects
        """
        query = query.lower()
        results = []
        
        for filter_def in self.filter_definitions.values():
            if (query in filter_def.name.lower() or 
                query in filter_def.display_name.lower() or 
                query in filter_def.description.lower() or
                query in filter_def.category.lower()):
                results.append(filter_def)
        
        return results
