"""
Configuration initialization module for SEM/GDS Alignment Tool.

This module provides functions to initialize and validate the configuration
system during application startup.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from .config_manager import get_config_manager, ConfigurationError

logger = logging.getLogger(__name__)


def initialize_config(config_dir: Optional[str] = None, 
                     create_user_dirs: bool = True,
                     validate_schema: bool = True) -> Dict[str, Any]:
    """
    Initialize the configuration system during application startup.
    
    Args:
        config_dir: Path to configuration directory
        create_user_dirs: Whether to create required directories
        validate_schema: Whether to validate configuration against schema
        
    Returns:
        Loaded configuration dictionary
        
    Raises:
        ConfigurationError: If initialization fails
    """
    try:
        logger.info("Initializing configuration system...")
        
        # Get configuration manager
        config_manager = get_config_manager(config_dir)
        
        # Load configuration
        config = config_manager.load_configuration(validate_config=validate_schema)
        
        # Create required directories if requested
        if create_user_dirs:
            create_required_directories(config)
        
        # Initialize environment-specific settings
        setup_environment_settings(config)
        
        logger.info("Configuration system initialized successfully")
        return config
        
    except Exception as e:
        error_msg = f"Failed to initialize configuration: {e}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg) from e


def create_required_directories(config: Dict[str, Any]) -> None:
    """
    Create required application directories based on configuration.
    
    Args:
        config: Application configuration
    """
    logger.debug("Creating required directories...")
    
    # Directories to create
    required_dirs = [
        config.get('paths', {}).get('data_directory'),
        config.get('paths', {}).get('sem_directory'),
        config.get('paths', {}).get('gds_directory'),
        config.get('paths', {}).get('results_directory'),
        config.get('paths', {}).get('extracted_structures_directory'),
        config.get('paths', {}).get('temp_directory'),
        config.get('paths', {}).get('logs_directory'),
        
        # Results subdirectories
        os.path.join(config.get('paths', {}).get('results_directory', ''), 'Aligned', 'auto'),
        os.path.join(config.get('paths', {}).get('results_directory', ''), 'Aligned', 'manual'),
        os.path.join(config.get('paths', {}).get('results_directory', ''), 'SEM_Filters', 'auto'),
        os.path.join(config.get('paths', {}).get('results_directory', ''), 'SEM_Filters', 'manual'),
        os.path.join(config.get('paths', {}).get('results_directory', ''), 'Scoring', 'overlays'),
        os.path.join(config.get('paths', {}).get('results_directory', ''), 'Scoring', 'charts'),
        os.path.join(config.get('paths', {}).get('results_directory', ''), 'Scoring', 'reports'),
        
        # Logs subdirectories
        os.path.join(config.get('paths', {}).get('logs_directory', ''), 'modules'),
        os.path.join(config.get('paths', {}).get('logs_directory', ''), 'debug'),
        os.path.join(config.get('paths', {}).get('logs_directory', ''), 'archive'),
    ]
    
    created_count = 0
    for dir_path in required_dirs:
        if dir_path:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                created_count += 1
                logger.debug(f"Ensured directory exists: {dir_path}")
            except Exception as e:
                logger.warning(f"Failed to create directory {dir_path}: {e}")
    
    logger.info(f"Ensured {created_count} directories exist")


def setup_environment_settings(config: Dict[str, Any]) -> None:
    """
    Setup environment-specific settings based on configuration.
    
    Args:
        config: Application configuration
    """
    logger.debug("Setting up environment-specific settings...")
    
    # Set up Python path if needed
    src_path = Path(__file__).parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        logger.debug(f"Added to Python path: {src_path}")
    
    # Set environment variables based on config
    performance_config = config.get('performance', {})
    
    # Set OpenCV thread count if specified
    max_threads = performance_config.get('max_worker_threads', 4)
    os.environ['OMP_NUM_THREADS'] = str(max_threads)
    os.environ['MKL_NUM_THREADS'] = str(max_threads)
    
    # Memory settings
    max_memory = performance_config.get('max_memory_usage_mb', 2048)
    if max_memory > 0:
        # This is more of a hint for our own memory management
        os.environ['SEM_GDS_MAX_MEMORY'] = str(max_memory)
    
    logger.debug("Environment settings configured")


def validate_config_integrity() -> bool:
    """
    Validate configuration integrity and required components.
    
    Returns:
        True if configuration is valid and complete
    """
    try:
        logger.debug("Validating configuration integrity...")
        
        config_manager = get_config_manager()
        
        # Check if configuration is loaded
        if not config_manager._is_loaded:
            logger.warning("Configuration not loaded")
            return False
        
        # Validate against schema
        if not config_manager.validate_configuration():
            logger.error("Configuration schema validation failed")
            return False
        
        # Check critical paths exist
        config = config_manager.get_full_config()
        critical_paths = [
            'paths.data_directory',
            'paths.results_directory',
            'paths.logs_directory'
        ]
        
        for path_key in critical_paths:
            path_value = config_manager.get(path_key)
            if not path_value:
                logger.error(f"Critical path not configured: {path_key}")
                return False
        
        logger.debug("Configuration integrity validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration integrity validation failed: {e}")
        return False


def create_default_user_config() -> None:
    """
    Create a default user configuration file with common overrides.
    """
    try:
        config_manager = get_config_manager()
        
        if config_manager.user_config_file.exists():
            logger.debug("User configuration file already exists")
            return
        
        # Create default user config with common overrides
        default_user_config = {
            "_comment": "User-specific configuration overrides",
            "_description": "This file allows users to override default settings without modifying the main config.json",
            "_note": "Only include settings you want to change from defaults. This file is optional and will be ignored by git.",
            "_examples": {
                "_comment": "Remove the underscore to activate these examples",
                "ui": {
                    "window": {
                        "width": 1600,
                        "height": 1000
                    },
                    "theme": {
                        "dark_mode": True,
                        "font_size": 10
                    }
                },
                "logging": {
                    "level": "DEBUG",
                    "console_output": True
                },
                "performance": {
                    "max_worker_threads": 8
                }
            }
        }
        
        config_manager._save_json_file(config_manager.user_config_file, default_user_config)
        logger.info("Created default user configuration file")
        
    except Exception as e:
        logger.warning(f"Failed to create default user configuration: {e}")


def get_startup_config_summary() -> Dict[str, Any]:
    """
    Get a summary of configuration for startup logging.
    
    Returns:
        Dictionary with configuration summary
    """
    try:
        config_manager = get_config_manager()
        config = config_manager.get_full_config()
        
        summary = {
            "app_name": config.get('application', {}).get('name', 'Unknown'),
            "app_version": config.get('application', {}).get('version', 'Unknown'),
            "debug_mode": config.get('application', {}).get('debug_mode', False),
            "data_directory": config.get('paths', {}).get('data_directory'),
            "results_directory": config.get('paths', {}).get('results_directory'),
            "log_level": config.get('logging', {}).get('level', 'INFO'),
            "max_memory_mb": config.get('performance', {}).get('max_memory_usage_mb'),
            "max_threads": config.get('performance', {}).get('max_worker_threads'),
            "user_config_exists": config_manager.user_config_file.exists(),
            "theme": {
                "style": config.get('ui', {}).get('theme', {}).get('style'),
                "dark_mode": config.get('ui', {}).get('theme', {}).get('dark_mode')
            }
        }
        
        return summary
        
    except Exception as e:
        logger.warning(f"Failed to get configuration summary: {e}")
        return {"error": str(e)}


# Convenience function for quick initialization
def quick_init(debug: bool = False) -> Dict[str, Any]:
    """
    Quick configuration initialization with sensible defaults.
    
    Args:
        debug: Whether to enable debug mode
        
    Returns:
        Loaded configuration
    """
    try:
        # Initialize config
        config = initialize_config(
            create_user_dirs=True,
            validate_schema=True
        )
        
        # Apply debug mode if requested
        if debug:
            from .config_manager import set_config
            set_config('application.debug_mode', True)
            set_config('logging.level', 'DEBUG')
            set_config('logging.console_output', True)
        
        return config
        
    except Exception as e:
        # Fallback to minimal config
        logger.error(f"Quick initialization failed, using fallback: {e}")
        return {
            "application": {"name": "SEM/GDS Tool", "version": "1.0.0", "debug_mode": debug},
            "paths": {"data_directory": "Data", "results_directory": "Results"},
            "logging": {"level": "DEBUG" if debug else "INFO"}
        }
