"""
Core utilities for file handling, logging, configuration, and validation.

Basic utility functions for the SEM/GDS Alignment Tool:
- File and directory management
- Simple logging setup
- Basic configuration management
- Error handling
- Project structure validation

Functions:
    File Utils: get_project_root, ensure_directory, get_data_path, get_results_path
    Logging Utils: setup_logging, get_logger
    Config Utils: load_config, get_config, set_config
    Error Handling: safe_file_operation, handle_errors, show_error_message
    Validation: validate_project_structure, run_full_validation
"""

# File utilities
from .file_utils import (
    get_project_root,
    ensure_directory,
    get_data_path,
    get_results_path,
    get_extracted_structures_path,
    setup_results_directories,
    get_unique_filename,
    list_files_by_extension
)

# Simple logging
from .simple_logging import (
    setup_logging,
    get_logger,
    log_error,
    log_info,
    log_debug
)

# Simple configuration
from .simple_config import (
    load_config,
    get_config,
    set_config,
    save_config
)

# Error handling
from .error_handling import (
    safe_file_operation,
    handle_errors,
    show_error_message,
    validate_file_exists,
    validate_directory_exists,
    safe_json_load,
    safe_json_save
)

# Project validation
from .validation import (
    validate_project_structure,
    check_missing_directories,
    create_missing_directories,
    run_full_validation,
    print_validation_report
)

# Also export the SemImage model for convenience
try:
    from ..models.simple_sem_image import SemImage
    HAS_SEM_IMAGE = True
except ImportError:
    # SemImage not available, continue without it
    HAS_SEM_IMAGE = False

__all__ = [
    # File utils
    "get_project_root", 
    "ensure_directory",
    "get_data_path", 
    "get_results_path",
    "get_extracted_structures_path",
    "setup_results_directories",
    "get_unique_filename",
    "list_files_by_extension",
    # Logging utils
    "setup_logging", 
    "get_logger", 
    "log_error", 
    "log_info", 
    "log_debug",
    # Config utils
    "load_config", 
    "get_config", 
    "set_config", 
    "save_config",
    # Error handling
    "safe_file_operation", 
    "handle_errors",
    "show_error_message", 
    "validate_file_exists", 
    "validate_directory_exists",
    "safe_json_load", 
    "safe_json_save",
    # Validation
    "validate_project_structure",
    "check_missing_directories", 
    "create_missing_directories", 
    "run_full_validation", 
    "print_validation_report"
]

if HAS_SEM_IMAGE:
    __all__.append("SemImage")
