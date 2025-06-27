"""Logging utilities for the image analysis application."""

import logging
import time
from pathlib import Path
from typing import Optional
from functools import wraps
from contextlib import contextmanager


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, logs only to console.
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module or class.
    
    Args:
        name: Name for the logger (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_execution_time(func):
    """
    Decorator to log the execution time of a function.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function that logs execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.3f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f} seconds: {e}")
            raise
    
    return wrapper


@contextmanager
def log_operation(operation_name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager to log the start and completion of an operation.
    
    Args:
        operation_name: Name of the operation being performed
        logger: Optional logger instance. If None, creates a new one.
    """
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(f"Starting {operation_name}")
    start_time = time.time()
    
    try:
        yield
        execution_time = time.time() - start_time
        logger.info(f"Completed {operation_name} in {execution_time:.3f} seconds")
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Failed {operation_name} after {execution_time:.3f} seconds: {e}")
        raise


class ProgressLogger:
    """Logger for tracking progress of long-running operations."""
    
    def __init__(self, operation_name: str, total_steps: int, logger: Optional[logging.Logger] = None):
        """
        Initialize progress logger.
        
        Args:
            operation_name: Name of the operation
            total_steps: Total number of steps in the operation
            logger: Optional logger instance
        """
        self.operation_name = operation_name
        self.total_steps = total_steps
        self.current_step = 0
        self.logger = logger or get_logger(__name__)
        self.start_time = time.time()
        
        self.logger.info(f"Starting {operation_name} with {total_steps} steps")
    
    def step(self, message: str = "") -> None:
        """
        Log completion of a step.
        
        Args:
            message: Optional message to include with the step log
        """
        self.current_step += 1
        elapsed_time = time.time() - self.start_time
        
        if self.total_steps > 0:
            progress_pct = (self.current_step / self.total_steps) * 100
            estimated_total = elapsed_time * self.total_steps / self.current_step
            remaining_time = estimated_total - elapsed_time
            
            log_message = f"{self.operation_name} progress: {self.current_step}/{self.total_steps} ({progress_pct:.1f}%) - ETA: {remaining_time:.1f}s"
        else:
            log_message = f"{self.operation_name} step {self.current_step} completed"
        
        if message:
            log_message += f" - {message}"
        
        self.logger.info(log_message)
    
    def complete(self, message: str = "") -> None:
        """
        Log completion of the entire operation.
        
        Args:
            message: Optional completion message
        """
        total_time = time.time() - self.start_time
        log_message = f"Completed {self.operation_name} in {total_time:.3f} seconds"
        
        if message:
            log_message += f" - {message}"
        
        self.logger.info(log_message)
