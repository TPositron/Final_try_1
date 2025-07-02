"""
Advanced logging utilities for the SEM/GDS image analysis application.

This module provides comprehensive logging capabilities including:
- Configurable log levels and rotation
- Module-specific logging
- Performance monitoring
- Log archiving and cleanup
- Context managers for operation tracking
"""

import logging
import logging.config
import logging.handlers
import time
import gzip
import shutil
import os
from pathlib import Path
from typing import Optional, Dict, Any
from functools import wraps
from contextlib import contextmanager
from datetime import datetime, timedelta
import threading


class LogManager:
    """
    Central logging manager for the application.
    
    Handles configuration, rotation, archiving, and cleanup of log files.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the log manager.
        
        Args:
            config_file: Path to logging configuration file
        """
        self.config_file = config_file or "logging.conf"
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.logs_dir = self.project_root / "logs"
        self.archive_dir = self.logs_dir / "archive"
        self.setup_directories()
        self._loggers: Dict[str, logging.Logger] = {}
        self._setup_complete = False
        
    def setup_directories(self):
        """Create necessary log directories."""
        directories = [
            self.logs_dir,
            self.logs_dir / "modules",
            self.logs_dir / "debug", 
            self.logs_dir / "archive",
            self.logs_dir / "archive" / "modules"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self, 
                     log_level: str = "INFO", 
                     force_setup: bool = False) -> None:
        """
        Set up logging configuration from config file or defaults.
        
        Args:
            log_level: Default logging level if config file not found
            force_setup: Force reconfiguration even if already set up
        """
        if self._setup_complete and not force_setup:
            return
            
        config_path = self.project_root / self.config_file
        
        try:
            if config_path.exists():
                # Use configuration file
                logging.config.fileConfig(str(config_path), 
                                        disable_existing_loggers=False)
            else:
                # Fallback to programmatic configuration
                self._setup_fallback_logging(log_level)
                
            self._setup_complete = True
            
            # Log the setup completion
            logger = logging.getLogger(__name__)
            logger.info(f"Logging system initialized - Level: {log_level}")
            
        except Exception as e:
            # Emergency fallback - basic console logging
            logging.basicConfig(
                level=getattr(logging, log_level.upper(), logging.INFO),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to setup logging from config: {e}")
            logger.info("Using basic logging configuration")
    
    def _setup_fallback_logging(self, log_level: str):
        """Setup basic logging if config file is not available."""
        # Clear existing handlers
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            
        # Set level
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        root.setLevel(numeric_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_formatter = logging.Formatter(
            '%(levelname)s - %(name)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root.addHandler(console_handler)
        
        # Main file handler with rotation
        main_log = self.logs_dir / "app.log"
        file_handler = logging.handlers.TimedRotatingFileHandler(
            str(main_log), when='midnight', interval=1, backupCount=30
        )
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a logger with the specified name.
        
        Args:
            name: Logger name (typically __name__ of the calling module)
            
        Returns:
            Configured logger instance
        """
        if not self._setup_complete:
            self.setup_logging()
            
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
            
        return self._loggers[name]
    
    def archive_old_logs(self, days_to_keep: int = 30):
        """
        Archive log files older than specified days.
        
        Args:
            days_to_keep: Number of days to keep uncompressed logs
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Archive main logs
        self._archive_logs_in_directory(self.logs_dir, cutoff_date)
        
        # Archive module logs
        modules_dir = self.logs_dir / "modules"
        if modules_dir.exists():
            self._archive_logs_in_directory(modules_dir, cutoff_date)
            
        logger = logging.getLogger(__name__)
        logger.info(f"Log archiving completed - archived logs older than {days_to_keep} days")
    
    def _archive_logs_in_directory(self, directory: Path, cutoff_date: datetime):
        """Archive logs in a specific directory."""
        archive_subdir = self.archive_dir / directory.relative_to(self.logs_dir)
        archive_subdir.mkdir(parents=True, exist_ok=True)
        
        for log_file in directory.glob("*.log.*"):
            if log_file.is_file():
                file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_date < cutoff_date:
                    # Create compressed archive
                    archive_name = f"{log_file.stem}_{file_date.strftime('%Y-%m-%d')}.log.gz"
                    archive_path = archive_subdir / archive_name
                    
                    try:
                        with open(log_file, 'rb') as f_in:
                            with gzip.open(archive_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        
                        # Remove original file
                        log_file.unlink()
                        
                    except Exception as e:
                        logger = logging.getLogger(__name__)
                        logger.error(f"Failed to archive {log_file}: {e}")
    
    def cleanup_old_archives(self, days_to_keep: int = 365):
        """
        Remove archived logs older than specified days.
        
        Args:
            days_to_keep: Number of days to keep archived logs
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for archive_file in self.archive_dir.rglob("*.gz"):
            if archive_file.is_file():
                file_date = datetime.fromtimestamp(archive_file.stat().st_mtime)
                if file_date < cutoff_date:
                    try:
                        archive_file.unlink()
                    except Exception as e:
                        logger = logging.getLogger(__name__)
                        logger.error(f"Failed to cleanup archive {archive_file}: {e}")
        
        logger = logging.getLogger(__name__)
        logger.info(f"Archive cleanup completed - removed archives older than {days_to_keep} days")
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about current log files.
        
        Returns:
            Dictionary with log file statistics
        """
        stats = {
            'total_logs': 0,
            'total_size_mb': 0,
            'oldest_log': None,
            'newest_log': None,
            'by_module': {}
        }
        
        for log_file in self.logs_dir.rglob("*.log*"):
            if log_file.is_file() and not log_file.name.endswith('.gz'):
                stats['total_logs'] += 1
                size_mb = log_file.stat().st_size / (1024 * 1024)
                stats['total_size_mb'] += size_mb
                
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if stats['oldest_log'] is None or mtime < stats['oldest_log']:
                    stats['oldest_log'] = mtime
                if stats['newest_log'] is None or mtime > stats['newest_log']:
                    stats['newest_log'] = mtime
                
                # Module-specific stats
                module = log_file.parent.name if log_file.parent.name != 'logs' else 'main'
                if module not in stats['by_module']:
                    stats['by_module'][module] = {'count': 0, 'size_mb': 0}
                stats['by_module'][module]['count'] += 1
                stats['by_module'][module]['size_mb'] += size_mb
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        return stats


# Global log manager instance
_log_manager = LogManager()


def setup_logging(log_level: str = "INFO", force_setup: bool = False) -> None:
    """
    Set up application-wide logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        force_setup: Force reconfiguration even if already set up
    """
    _log_manager.setup_logging(log_level, force_setup)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        
    Returns:
        Configured logger instance
    """
    return _log_manager.get_logger(name)


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


def archive_logs(days_to_keep: int = 30):
    """
    Archive old log files.
    
    Args:
        days_to_keep: Number of days to keep uncompressed logs
    """
    _log_manager.archive_old_logs(days_to_keep)


def cleanup_old_archives(days_to_keep: int = 365):
    """
    Clean up old archived log files.
    
    Args:
        days_to_keep: Number of days to keep archived logs
    """
    _log_manager.cleanup_old_archives(days_to_keep)


def get_log_statistics() -> Dict[str, Any]:
    """
    Get current log file statistics.
    
    Returns:
        Dictionary with log statistics
    """
    return _log_manager.get_log_statistics()


# Convenience function to set up a module logger
def get_module_logger(module_name: str) -> logging.Logger:
    """
    Get a logger configured for a specific module.
    
    Args:
        module_name: Name of the module
        
    Returns:
        Logger configured for the module
    """
    return get_logger(f"src.{module_name}")


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
