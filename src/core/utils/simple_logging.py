"""
Simple Logging Utilities - Basic Console and File Logging

This module provides lightweight logging utilities for the SEM/GDS alignment tool.
It offers basic console and file logging with configurable levels and simple
convenience functions for common logging operations.

Main Functions:
- setup_logging(): Configures basic logging with console and optional file output
- get_logger(): Returns logger instance for specified module
- log_error(): Convenience function for error logging
- log_info(): Convenience function for info logging
- log_debug(): Convenience function for debug logging

Dependencies:
- Uses: logging (Python logging framework), sys (stdout access)
- Uses: pathlib.Path (file operations), typing (type hints)
- Uses: inspect (caller module detection)
- Used by: All modules requiring basic logging functionality
- Alternative to: core/utils/logging_utils.py (more advanced logging)

Features:
- Simple logging setup with single function call
- Console logging to stdout with configurable levels
- Optional file logging with automatic directory creation
- Automatic caller module name detection for logger naming
- Convenience functions for common log levels
- Clear existing handlers to avoid duplicate logging
- Standardized log message formatting with timestamps

Logging Levels:
- DEBUG: Detailed diagnostic information
- INFO: General information about program execution
- WARNING: Warning messages for potential issues
- ERROR: Error messages for serious problems

Configuration:
- Default level: INFO
- Default log directory: 'logs'
- Log file name: 'app.log'
- Format: timestamp - module - level - message

Usage:
- setup_logging('DEBUG', True, 'logs'): Enable debug logging to file
- logger = get_logger(__name__): Get module-specific logger
- log_info('Processing started'): Quick info logging
- log_error('Failed to load file'): Quick error logging

Error Handling:
- Graceful fallback if file logging setup fails
- Console output for file logging errors
- Safe logger creation with unknown module fallback
"""

import logging
import logging.handlers
import sys
import time
import gzip
import shutil
import os
from pathlib import Path
from typing import Optional, Dict, Any
from functools import wraps
from contextlib import contextmanager
from datetime import datetime, timedelta


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


# Advanced logging features from logging_utils.py

@contextmanager
def log_execution_time(logger: logging.Logger, operation_name: str, level: int = logging.INFO):
    """
    Context manager to log execution time of operations.
    
    Args:
        logger: Logger instance to use
        operation_name: Name of the operation being timed
        level: Logging level for the timing message
        
    Usage:
        with log_execution_time(logger, "image_processing"):
            # Your code here
            process_image()
    """
    start_time = time.time()
    logger.log(level, f"Starting {operation_name}")
    
    try:
        yield
        duration = time.time() - start_time
        logger.log(level, f"Completed {operation_name} in {duration:.3f}s")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed {operation_name} after {duration:.3f}s: {e}")
        raise


def log_performance(func):
    """
    Decorator to automatically log function execution time.
    
    Usage:
        @log_performance
        def expensive_operation():
            # Your code here
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        with log_execution_time(logger, f"{func.__name__}"):
            return func(*args, **kwargs)
    return wrapper


@contextmanager
def stage_logger(stage_name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager for logging pipeline stages.
    
    Args:
        stage_name: Name of the processing stage
        logger: Optional logger instance (uses default if None)
        
    Usage:
        with stage_logger("alignment_stage"):
            # Stage processing code
            perform_alignment()
    """
    if logger is None:
        logger = get_logger("pipeline")
        
    logger.info(f"=== Starting stage: {stage_name} ===")
    start_time = time.time()
    
    try:
        yield logger
        duration = time.time() - start_time
        logger.info(f"=== Completed stage: {stage_name} ({duration:.3f}s) ===")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"=== Failed stage: {stage_name} after {duration:.3f}s ===")
        logger.exception(f"Stage {stage_name} error: {e}")
        raise


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


def setup_advanced_logging(log_level: str = "INFO", log_dir: str = "logs", enable_rotation: bool = True) -> None:
    """
    Set up advanced logging with rotation and archiving.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        enable_rotation: Whether to enable log rotation
    """
    # Configure logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.setLevel(numeric_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with optional rotation
    if enable_rotation:
        file_handler = logging.handlers.TimedRotatingFileHandler(
            str(log_path / "app.log"), when='midnight', interval=1, backupCount=30
        )
    else:
        file_handler = logging.FileHandler(log_path / "app.log")
    
    file_handler.setLevel(numeric_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    print(f"Advanced logging setup complete - Level: {log_level}, Dir: {log_path}")


def archive_old_logs(log_dir: str = "logs", days_to_keep: int = 30) -> None:
    """
    Archive log files older than specified days.
    
    Args:
        log_dir: Directory containing log files
        days_to_keep: Number of days to keep uncompressed logs
    """
    log_path = Path(log_dir)
    archive_path = log_path / "archive"
    archive_path.mkdir(parents=True, exist_ok=True)
    
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    
    for log_file in log_path.glob("*.log.*"):
        if log_file.is_file():
            file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
            if file_date < cutoff_date:
                # Create compressed archive
                archive_name = f"{log_file.stem}_{file_date.strftime('%Y-%m-%d')}.log.gz"
                archive_file = archive_path / archive_name
                
                try:
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(archive_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Remove original file
                    log_file.unlink()
                    print(f"Archived: {log_file} -> {archive_file}")
                    
                except Exception as e:
                    print(f"Failed to archive {log_file}: {e}")


def get_log_statistics(log_dir: str = "logs") -> Dict[str, Any]:
    """
    Get statistics about current log files.
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        Dictionary with log file statistics
    """
    log_path = Path(log_dir)
    stats = {
        'total_logs': 0,
        'total_size_mb': 0,
        'oldest_log': None,
        'newest_log': None,
        'files': []
    }
    
    if not log_path.exists():
        return stats
    
    for log_file in log_path.rglob("*.log*"):
        if log_file.is_file() and not log_file.name.endswith('.gz'):
            stats['total_logs'] += 1
            size_mb = log_file.stat().st_size / (1024 * 1024)
            stats['total_size_mb'] += size_mb
            
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            if stats['oldest_log'] is None or mtime < stats['oldest_log']:
                stats['oldest_log'] = mtime
            if stats['newest_log'] is None or mtime > stats['newest_log']:
                stats['newest_log'] = mtime
            
            stats['files'].append({
                'name': log_file.name,
                'size_mb': round(size_mb, 2),
                'modified': mtime
            })
    
    stats['total_size_mb'] = round(stats['total_size_mb'], 2)
    return stats
