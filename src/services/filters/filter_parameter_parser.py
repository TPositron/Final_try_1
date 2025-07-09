"""
Filter Parameter Parser Service

This module provides AST-based parsing of Python filter files to automatically
extract parameter information for dynamic UI generation. It analyzes Python
source code to discover filter functions/classes and their parameters, creating
structured definitions that can be used for automatic UI generation.

Dependencies:
- ast: For Abstract Syntax Tree parsing of Python source code
- inspect: For runtime introspection of Python objects
- importlib.util: For dynamic module loading
- dataclasses: For structured data representation
- enum: For parameter type enumeration

Main Classes:
- ParameterType: Enum defining supported parameter types (int, float, bool, str, tuple, list)
- FilterParameter: Dataclass representing a single filter parameter with constraints
- FilterDefinition: Dataclass representing complete filter metadata
- FilterParameterParser: Main parser class for extracting filter information

Key Methods:
- parse_all_filters(): Parse all filter files in the specified directory
- get_filter_definition(): Get parsed definition for a specific filter
- get_filters_by_category(): Get filters organized by category
- get_all_categories(): Get all available filter categories
- export_filter_definitions(): Export parsed definitions to JSON

Internal Methods:
- _parse_filter_file(): Parse a single Python filter file
- _extract_filter_info(): Extract filter information from AST
- _create_filter_definition_from_function(): Create definition from function node
- _create_filter_definition_from_class(): Create definition from class node
- _create_parameter_from_arg(): Create parameter definition from function argument
- _infer_type_from_value(): Infer parameter type from default value
- _get_parameter_constraints(): Get parameter constraints based on name/type
- _determine_category(): Categorize filters based on name patterns

Features:
- Automatic parameter discovery from function signatures
- Type inference from default values
- Constraint generation based on parameter names
- Category assignment based on filter names
- Support for both function and class-based filters
- Comprehensive error handling for malformed files
- Export functionality for caching parsed definitions
"""
import os
import ast
import inspect
import importlib.util
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class ParameterType(Enum):
    """Types of filter parameters."""
    INTEGER = "int"
    FLOAT = "float"
    BOOLEAN = "bool"
    STRING = "str"
    TUPLE = "tuple"
    LIST = "list"


@dataclass
class FilterParameter:
    """Represents a single filter parameter."""
    name: str
    param_type: ParameterType
    default_value: Any
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    description: str = ""
    step: Optional[Union[int, float]] = None
    choices: Optional[List[str]] = None
    required: bool = True


@dataclass
class FilterDefinition:
    """Represents a complete filter definition."""
    name: str
    display_name: str
    function_name: str
    module_path: str
    parameters: List[FilterParameter] = field(default_factory=list)
    description: str = ""
    category: str = "General"


class FilterParameterParser:
    """Service to parse filter files and extract parameter information."""
    
    def __init__(self, filters_directory: str):
        """
        Initialize the filter parameter parser.
        
        Args:
            filters_directory: Path to the directory containing filter files
        """
        self.filters_directory = filters_directory
        self.filter_definitions: Dict[str, FilterDefinition] = {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging for the parser."""
        import logging
        return logging.getLogger(__name__)
    
    def parse_all_filters(self) -> Dict[str, FilterDefinition]:
        """
        Parse all filter files in the directory.
        
        Returns:
            Dictionary mapping filter names to FilterDefinition objects
        """
        self.logger.info(f"Parsing filters from directory: {self.filters_directory}")
        
        if not os.path.exists(self.filters_directory):
            self.logger.error(f"Filters directory not found: {self.filters_directory}")
            return {}
        
        # Get all Python files in the filters directory
        filter_files = [f for f in os.listdir(self.filters_directory) 
                       if f.startswith('filter_') and f.endswith('.py')]
        
        self.logger.info(f"Found {len(filter_files)} filter files")
        
        for filter_file in filter_files:
            try:
                file_path = os.path.join(self.filters_directory, filter_file)
                self._parse_filter_file(file_path)
            except Exception as e:
                self.logger.error(f"Error parsing filter file {filter_file}: {e}")
                continue
        
        self.logger.info(f"Successfully parsed {len(self.filter_definitions)} filters")
        return self.filter_definitions
    
    def _parse_filter_file(self, file_path: str) -> None:
        """
        Parse a single filter file and extract parameter information.
        
        Args:
            file_path: Path to the filter file
        """
        filter_name = os.path.basename(file_path)[:-3]  # Remove .py extension
        
        try:
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Extract filter information
            filter_def = self._extract_filter_info(tree, filter_name, file_path)
            
            if filter_def:
                self.filter_definitions[filter_def.name] = filter_def
                self.logger.debug(f"Parsed filter: {filter_def.name}")
            
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {e}")
    
    def _extract_filter_info(self, tree: ast.AST, filter_name: str, file_path: str) -> Optional[FilterDefinition]:
        """
        Extract filter information from AST.
        
        Args:
            tree: AST tree of the filter file
            filter_name: Name of the filter
            file_path: Path to the filter file
            
        Returns:
            FilterDefinition object or None if extraction failed
        """
        functions = []
        classes = []
        
        # Find all function and class definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node)
            elif isinstance(node, ast.ClassDef):
                classes.append(node)
        
        # Try to find the main filter function or class
        main_function = None
        main_class = None
        
        # Look for common filter function patterns
        for func in functions:
            if (func.name.startswith('filter_') or 
                func.name.startswith('apply_') or
                func.name in ['__call__', 'apply']):
                main_function = func
                break
        
        # Look for filter classes
        for cls in classes:
            if cls.name.startswith('Filter'):
                main_class = cls
                break
        
        # Create filter definition
        if main_function:
            return self._create_filter_definition_from_function(
                main_function, filter_name, file_path
            )
        elif main_class:
            return self._create_filter_definition_from_class(
                main_class, filter_name, file_path
            )
        
        return None
    
    def _create_filter_definition_from_function(self, func_node: ast.FunctionDef, 
                                               filter_name: str, file_path: str) -> FilterDefinition:
        """
        Create FilterDefinition from a function node.
        
        Args:
            func_node: AST function node
            filter_name: Name of the filter
            file_path: Path to the filter file
            
        Returns:
            FilterDefinition object
        """
        # Extract parameters from function signature
        parameters = []
        
        # Skip 'self' and 'image' parameters
        skip_params = {'self', 'image', 'image_path', 'output_path'}
        
        for arg in func_node.args.args:
            if arg.arg in skip_params:
                continue
                
            param = self._create_parameter_from_arg(arg, func_node)
            if param:
                parameters.append(param)
        
        # Extract docstring description
        description = ast.get_docstring(func_node) or ""
        
        return FilterDefinition(
            name=filter_name,
            display_name=self._format_display_name(filter_name),
            function_name=func_node.name,
            module_path=file_path,
            parameters=parameters,
            description=description,
            category=self._determine_category(filter_name)
        )
    
    def _create_filter_definition_from_class(self, class_node: ast.ClassDef, 
                                            filter_name: str, file_path: str) -> FilterDefinition:
        """
        Create FilterDefinition from a class node.
        
        Args:
            class_node: AST class node
            filter_name: Name of the filter
            file_path: Path to the filter file
            
        Returns:
            FilterDefinition object
        """
        # Find __init__ method to extract parameters
        init_method = None
        apply_method = None
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                if node.name == '__init__':
                    init_method = node
                elif node.name in ['apply', '__call__', 'apply_filter']:
                    apply_method = node
        
        parameters = []
        
        if init_method:
            # Extract parameters from __init__ method
            skip_params = {'self'}
            
            for arg in init_method.args.args:
                if arg.arg in skip_params:
                    continue
                    
                param = self._create_parameter_from_arg(arg, init_method)
                if param:
                    parameters.append(param)
        
        # Extract docstring description
        description = ast.get_docstring(class_node) or ""
        
        return FilterDefinition(
            name=filter_name,
            display_name=self._format_display_name(filter_name),
            function_name=class_node.name,
            module_path=file_path,
            parameters=parameters,
            description=description,
            category=self._determine_category(filter_name)
        )
    
    def _create_parameter_from_arg(self, arg: ast.arg, func_node: ast.FunctionDef) -> Optional[FilterParameter]:
        """
        Create FilterParameter from function argument.
        
        Args:
            arg: AST argument node
            func_node: Parent function node
            
        Returns:
            FilterParameter object or None
        """
        param_name = arg.arg
        
        # Try to determine parameter type and default value
        param_type = ParameterType.FLOAT  # Default type
        default_value = None
        
        # Look for default values
        defaults = func_node.args.defaults
        if defaults:
            arg_index = func_node.args.args.index(arg)
            default_index = arg_index - (len(func_node.args.args) - len(defaults))
            
            if default_index >= 0 and default_index < len(defaults):
                default_node = defaults[default_index]
                default_value = self._extract_value_from_node(default_node)
                param_type = self._infer_type_from_value(default_value)
        
        # Set parameter constraints based on name and type
        min_val, max_val, step = self._get_parameter_constraints(param_name, param_type, default_value)
        
        return FilterParameter(
            name=param_name,
            param_type=param_type,
            default_value=default_value,
            min_value=min_val,
            max_value=max_val,
            step=step,
            description=self._get_parameter_description(param_name),
            required=default_value is None
        )
    
    def _extract_value_from_node(self, node: ast.AST) -> Any:
        """Extract value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Str):  # Python < 3.8
            return node.s
        elif isinstance(node, ast.Tuple):
            return tuple(self._extract_value_from_node(elt) for elt in node.elts)
        elif isinstance(node, ast.List):
            return [self._extract_value_from_node(elt) for elt in node.elts]
        return None
    
    def _infer_type_from_value(self, value: Any) -> ParameterType:
        """Infer parameter type from default value."""
        if isinstance(value, int):
            return ParameterType.INTEGER
        elif isinstance(value, float):
            return ParameterType.FLOAT
        elif isinstance(value, bool):
            return ParameterType.BOOLEAN
        elif isinstance(value, str):
            return ParameterType.STRING
        elif isinstance(value, tuple):
            return ParameterType.TUPLE
        elif isinstance(value, list):
            return ParameterType.LIST
        return ParameterType.FLOAT
    
    def _get_parameter_constraints(self, param_name: str, param_type: ParameterType, 
                                  default_value: Any) -> tuple:
        """Get parameter constraints based on name and type."""
        # Define common parameter constraints
        constraints = {
            'threshold': (0, 255, 1),
            'low_threshold': (0, 255, 1),
            'high_threshold': (0, 255, 1),
            'clip_limit': (0.1, 10.0, 0.1),
            'h': (1, 30, 1),
            'template_window_size': (3, 21, 2),
            'search_window_size': (5, 31, 2),
            'frequency': (0.01, 2.0, 0.01),
            'theta': (0.0, 180.0, 1.0),
            'bandwidth': (0.1, 5.0, 0.1),
            'phase_offset': (0.0, 360.0, 1.0),
            'sigma': (0.1, 10.0, 0.1),
            'kernel_size': (1, 31, 2),
            'weight': (0.01, 1.0, 0.01)
        }
        
        if param_name in constraints:
            return constraints[param_name]
        
        # Default constraints based on type
        if param_type == ParameterType.INTEGER:
            return (0, 100, 1)
        elif param_type == ParameterType.FLOAT:
            return (0.0, 10.0, 0.1)
        
        return (None, None, None)
    
    def _get_parameter_description(self, param_name: str) -> str:
        """Get parameter description based on name."""
        descriptions = {
            'threshold': 'Threshold value for binarization',
            'low_threshold': 'Lower threshold for edge detection',
            'high_threshold': 'Upper threshold for edge detection',
            'clip_limit': 'Threshold for contrast limiting',
            'tile_grid_size': 'Size of grid for histogram equalization',
            'h': 'Filter strength for denoising',
            'template_window_size': 'Size of template window for comparison',
            'search_window_size': 'Size of search window for similar patches',
            'frequency': 'Frequency of the Gabor filter',
            'theta': 'Orientation angle in degrees',
            'bandwidth': 'Bandwidth of the filter',
            'phase_offset': 'Phase offset in degrees',
            'sigma': 'Standard deviation for Gaussian kernel',
            'kernel_size': 'Size of the filter kernel',
            'weight': 'Weight parameter for the filter'
        }
        
        return descriptions.get(param_name, f'{param_name.replace("_", " ").title()} parameter')
    
    def _format_display_name(self, filter_name: str) -> str:
        """Format filter name for display."""
        # Remove 'filter_' prefix if present
        if filter_name.startswith('filter_'):
            name = filter_name[7:]
        else:
            name = filter_name
        
        # Convert to title case and replace underscores
        return name.replace('_', ' ').title()
    
    def _determine_category(self, filter_name: str) -> str:
        """Determine filter category based on name."""
        categories = {
            'canny': 'Edge Detection',
            'laplacian': 'Edge Detection',
            'gabor': 'Feature Detection',
            'clahe': 'Contrast Enhancement',
            'threshold': 'Binarization',
            'nlmd': 'Noise Reduction',
            'total_variation': 'Noise Reduction',
            'wavelet': 'Wavelet Transform',
            'fft': 'Frequency Domain',
            'top_hat': 'Morphological',
            'dog': 'Edge Detection'
        }
        
        for key, category in categories.items():
            if key in filter_name.lower():
                return category
        
        return 'General'
    
    def get_filter_definition(self, filter_name: str) -> Optional[FilterDefinition]:
        """Get filter definition by name."""
        return self.filter_definitions.get(filter_name)
    
    def get_filters_by_category(self, category: str) -> List[FilterDefinition]:
        """Get all filters in a specific category."""
        return [f for f in self.filter_definitions.values() if f.category == category]
    
    def get_all_categories(self) -> List[str]:
        """Get all available filter categories."""
        return list(set(f.category for f in self.filter_definitions.values()))
    
    def export_filter_definitions(self, output_path: str) -> None:
        """Export filter definitions to JSON file."""
        import json
        
        data = {}
        for name, filter_def in self.filter_definitions.items():
            data[name] = {
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
                        'required': p.required
                    }
                    for p in filter_def.parameters
                ]
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Filter definitions exported to {output_path}")
