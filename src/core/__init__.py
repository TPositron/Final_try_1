"""
Core package for SEM/GDS alignment tool.

This package contains the fundamental data models and utilities that form the foundation
of the SEM/GDS alignment application. It provides low-level functionality that is used
throughout the application.

Architecture:
- models: Data structures for SEM images, GDS models, and alignment results
- utils: Common utilities for file operations, logging, configuration management
- gds_*: GDS file processing and image generation modules
- simple_*: Simplified implementations for core functionality

Key Components:
- SemImage: Handles SEM image data and operations
- GDS models: Handle GDS file loading and structure extraction
- Display generators: Create visual representations of GDS structures
- Alignment generators: Apply transformations to GDS data
- Simple loaders: Streamlined GDS loading functionality

Dependencies:
- Used by: services package, ui package
- Uses: External libraries (gdspy/gdstk, numpy, opencv, PIL)
- Called by: All service classes and UI controllers

Data Flow:
1. Raw files (SEM images, GDS files) -> Core models
2. Core models -> Services for processing
3. Processed data -> UI for display
"""

from .models import *
from .utils import *

__all__ = [
    # Models
    'SemImage', 'load_sem_image', 'create_sem_image',
    
    # Utils
    'get_project_root', 'resolve_path', 'make_results_dirs',
    'setup_logging', 'get_logger',
    'load_config', 'get_config', 'set_config',
    'safe_file_operation', 'handle_errors',
    'validate_project_structure', 'run_full_validation'
]
