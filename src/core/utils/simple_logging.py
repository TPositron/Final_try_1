"""
Simple logging utilities for the SEM/GDS Alignment Tool.

Basic console logging with different levels.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(level: str = "INFO", log_to_file: bool = False, log_dir: str = "logs") -> None:
    """
    Set up basic logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to file
        log_dir: Directory for log files
    """
    # Configure logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Basic formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path / "app.log")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            print(f"Logging to file: {log_path / 'app.log'}")
        except Exception as e:
            print(f"Failed to setup file logging: {e}")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name. If None, uses calling module name.
        
    Returns:
        Logger instance
    """
    if name is None:
        # Get the calling module's name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)


def log_error(message: str, logger_name: Optional[str] = None) -> None:
    """
    Log an error message.
    
    Args:
        message: Error message
        logger_name: Optional logger name
    """
    logger = get_logger(logger_name)
    logger.error(message)


def log_info(message: str, logger_name: Optional[str] = None) -> None:
    """
    Log an info message.
    
    Args:
        message: Info message
        logger_name: Optional logger name
    """
    logger = get_logger(logger_name)
    logger.info(message)


def log_debug(message: str, logger_name: Optional[str] = None) -> None:
    """
    Log a debug message.
    
    Args:
        message: Debug message
        logger_name: Optional logger name
    """
    logger = get_logger(logger_name)
    logger.debug(message)
