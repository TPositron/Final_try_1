"""
Error Handling Utilities - Comprehensive Error Management and Recovery

This script provides error handling utilities for the SEM/GDS alignment tool.
It includes decorators, safe operation wrappers, validation functions, and
error display mechanisms for robust application operation.

Key Functions:
- safe_file_operation(): Safely executes file operations with error handling
- handle_errors(): Decorator for basic error handling on any function
- log_and_continue(): Decorator to log errors and continue execution
- show_error_message(): Displays error messages to user
- validate_file_exists(): Checks if file exists with error reporting
- validate_directory_exists(): Checks if directory exists with error reporting
- safe_json_load(): Safely loads JSON files with error handling
- safe_json_save(): Safely saves JSON files with error handling

Dependencies:
- logging: Error logging and debugging
- typing: Type hints for Any, Callable, Optional
- functools.wraps: Decorator preservation
- pathlib.Path: File system operations
- json: JSON file operations

Features:
- Function decorators for automatic error handling
- Safe file operation wrappers with specific exception handling
- File and directory validation with user feedback
- JSON file operations with error recovery
- Consistent error message display interface
- Logging integration for debugging and monitoring
- Graceful error recovery without application crashes
"""

import logging
from typing import Any, Callable, Optional
from functools import wraps


def safe_file_operation(operation: Callable, *args, **kwargs) -> Any:
    """
    Safely execute a file operation with error handling.
    
    Args:
        operation: Function to execute
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of operation or None if error
    """
    try:
        return operation(*args, **kwargs)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None
    except PermissionError as e:
        print(f"Permission denied: {e}")
        return None
    except OSError as e:
        print(f"File system error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in file operation: {e}")
        return None


def handle_errors(func: Callable) -> Callable:
    """
    Decorator for basic error handling.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            return None
    return wrapper


def log_and_continue(func: Callable, logger: Optional[logging.Logger] = None) -> Callable:
    """
    Decorator to log errors and continue execution.
    
    Args:
        func: Function to wrap
        logger: Logger to use for error logging
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if logger:
                logger.error(f"Error in {func.__name__}: {e}")
            else:
                print(f"Error in {func.__name__}: {e}")
            return None
    return wrapper


def show_error_message(message: str, title: str = "Error") -> None:
    """
    Display an error message to the user.
    
    Args:
        message: Error message to display
        title: Title for the error dialog
    """
    # For now, just print to console
    # Later this can be replaced with a GUI dialog
    print(f"{title}: {message}")


def validate_file_exists(file_path: str) -> bool:
    """
    Check if a file exists and show error if not.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file exists, False otherwise
    """
    try:
        from pathlib import Path
        if Path(file_path).exists():
            return True
        else:
            show_error_message(f"File not found: {file_path}")
            return False
    except Exception as e:
        show_error_message(f"Error checking file: {e}")
        return False


def validate_directory_exists(dir_path: str) -> bool:
    """
    Check if a directory exists and show error if not.
    
    Args:
        dir_path: Directory path to check
        
    Returns:
        True if directory exists, False otherwise
    """
    try:
        from pathlib import Path
        if Path(dir_path).is_dir():
            return True
        else:
            show_error_message(f"Directory not found: {dir_path}")
            return False
    except Exception as e:
        show_error_message(f"Error checking directory: {e}")
        return False


def safe_json_load(file_path: str) -> Optional[dict]:
    """
    Safely load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data or None if error
    """
    try:
        import json
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        show_error_message(f"JSON file not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        show_error_message(f"Invalid JSON in {file_path}: {e}")
        return None
    except Exception as e:
        show_error_message(f"Error loading JSON: {e}")
        return None


def safe_json_save(data: dict, file_path: str) -> bool:
    """
    Safely save data to JSON file with error handling.
    
    Args:
        data: Data to save
        file_path: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import json
        from pathlib import Path
        
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        show_error_message(f"Error saving JSON: {e}")
        return False
