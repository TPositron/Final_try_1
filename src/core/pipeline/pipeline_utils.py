"""
Pipeline Utilities - Helper Functions for Processing Pipelines

This module provides utility functions used across all processing pipelines
including configuration management, transformation matrix creation, and
validation functions that support the pipeline infrastructure.

Key Functions:
- create_transform_matrix(): Creates 3x3 transformation matrices from parameters
- load_config_from_file(): Loads pipeline configuration from JSON/YAML files
- save_config_to_file(): Saves pipeline configuration to JSON/YAML files
- validate_config(): Validates configuration dictionaries for required keys
- get_default_config(): Returns default pipeline configuration
- get_mode_configs(): Returns mode-specific configurations (manual/automatic)

Dependencies:
- Uses: numpy (matrix operations), json (config files), os (file operations)
- Optional: yaml (YAML config support)
- Used by: All pipeline classes for configuration and transformation support
- Used by: UI components for parameter validation

Configuration Management:
- Supports both JSON and YAML configuration formats
- Provides default configurations for different processing modes
- Validates configuration completeness and structure
- Handles file I/O with proper error handling

Transformation Support:
- Creates transformation matrices from user parameters
- Supports translation, rotation, and scaling transformations
- Uses standard 3x3 homogeneous coordinate matrices
- Compatible with OpenCV and other computer vision libraries

Features:
- Mode-specific configuration templates
- Flexible file format support (JSON/YAML)
- Configuration validation with missing key detection
- Transformation matrix generation from intuitive parameters
- Error handling for file operations and missing dependencies
"""

import numpy as np
from typing import Dict, Any, Tuple
import json
import os
try:
    import yaml
except ImportError:
    yaml = None

def create_transform_matrix(manual_adjustments: Dict[str, Any]) -> np.ndarray:
    """
    Create a 3x3 transformation matrix from manual adjustment parameters.
    Args:
        manual_adjustments: Dictionary of manual transform parameters
    Returns:
        np.ndarray: 3x3 transformation matrix
    """
    tx = manual_adjustments.get('translation_x', 0.0)
    ty = manual_adjustments.get('translation_y', 0.0)
    rotation = manual_adjustments.get('rotation', 0.0)
    scale = manual_adjustments.get('scale', 1.0)
    rotation_rad = np.radians(rotation)
    cos_r = np.cos(rotation_rad)
    sin_r = np.sin(rotation_rad)
    transform_matrix = np.array([
        [scale * cos_r, -scale * sin_r, tx],
        [scale * sin_r,  scale * cos_r, ty],
        [0,              0,             1]
    ])
    return transform_matrix

def load_config_from_file(path: str) -> dict:
    """
    Load pipeline configuration from a JSON or YAML file.
    Args:
        path: Path to the config file
    Returns:
        dict: Loaded configuration
    Raises:
        ValueError: If file type is unsupported or file cannot be loaded
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    with open(path, 'r', encoding='utf-8') as f:
        if ext in ['.json']:
            return json.load(f)
        elif ext in ['.yaml', '.yml']:
            if yaml is None:
                raise ImportError("PyYAML is not installed. Cannot load YAML config.")
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file type: {ext}")

def save_config_to_file(config: dict, path: str):
    """
    Save pipeline configuration to a JSON or YAML file.
    Args:
        config: Configuration dictionary
        path: Path to save the config file
    Raises:
        ValueError: If file type is unsupported
    """
    ext = os.path.splitext(path)[1].lower()
    with open(path, 'w', encoding='utf-8') as f:
        if ext in ['.json']:
            json.dump(config, f, indent=2)
        elif ext in ['.yaml', '.yml']:
            if yaml is None:
                raise ImportError("PyYAML is not installed. Cannot save YAML config.")
            yaml.safe_dump(config, f)
        else:
            raise ValueError(f"Unsupported config file type: {ext}")

def validate_config(config: dict, required_keys=None) -> Tuple[bool, list]:
    """
    Validate that a config dictionary contains all required keys.
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
    Returns:
        (bool, list): (True, []) if valid, (False, [missing_keys]) if not
    """
    if required_keys is None:
        required_keys = []
    missing = [key for key in required_keys if key not in config]
    return (len(missing) == 0, missing)

def get_default_config():
    return {
        'filter_sequence': ['clahe', 'total_variation', 'gaussian'],
        'alignment_method': 'orb_ransac',
        'scoring_methods': ['correlation', 'structural_similarity'],
        'early_exit_threshold': 0.3,
        'max_filter_trials': 5
    }

def get_mode_configs():
    return {
        'manual': {
            'use_ui_state': True,
            'filter_sequence': [],
            'alignment_method': 'manual',
            'scoring_methods': ['correlation'],
            'allow_parameter_override': True,
            'real_time_preview': True,
            'validate_each_stage': True
        },
        'automatic': {
            'use_ui_state': False,
            'filter_sequence': ['clahe', 'total_variation', 'gaussian', 'canny'],
            'alignment_method': 'orb_ransac',
            'scoring_methods': ['correlation', 'structural_similarity', 'mutual_information'],
            'allow_parameter_override': False,
            'real_time_preview': False,
            'validate_each_stage': False,
            'auto_optimization': True,
            'fallback_methods': True
        }
    }
