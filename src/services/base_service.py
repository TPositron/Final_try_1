"""
Base service class providing common functionality for all services.

This module defines the base service class that provides:
- Common signal/slot patterns
- Error handling utilities  
- Logging configuration
- State management helpers
"""

from typing import Any, Dict, Optional
from PySide6.QtCore import QObject, Signal
from src.core.utils.logging_utils import get_logger


class BaseService(QObject):
    """
    Base class for all service classes.
    
    Provides common functionality including:
    - Standardized error handling and reporting
    - Logging integration
    - Signal definitions for common operations
    - State management helpers
    """
    
    # Common signals that most services emit
    operation_started = Signal(str)  # operation_name
    operation_completed = Signal(str, object)  # operation_name, result
    operation_failed = Signal(str, str)  # operation_name, error_message
    progress_updated = Signal(int)  # percentage (0-100)
    status_changed = Signal(str)  # status_message
    
    def __init__(self, service_name: str = None):
        """
        Initialize base service.
        
        Args:
            service_name: Name for logging and identification
        """
        super().__init__()
        self._service_name = service_name or self.__class__.__name__
        self._logger = get_logger(self._service_name)
        self._current_operation = None
        self._is_busy = False
        
    @property
    def service_name(self) -> str:
        """Get the service name."""
        return self._service_name
        
    @property
    def is_busy(self) -> bool:
        """Check if service is currently processing an operation."""
        return self._is_busy
        
    @property
    def current_operation(self) -> Optional[str]:
        """Get the name of the current operation."""
        return self._current_operation
        
    def _start_operation(self, operation_name: str) -> None:
        """
        Mark the start of an operation.
        
        Args:
            operation_name: Name of the operation being started
        """
        self._current_operation = operation_name
        self._is_busy = True
        self._logger.info(f"Starting operation: {operation_name}")
        self.operation_started.emit(operation_name)
        
    def _complete_operation(self, result: Any = None) -> None:
        """
        Mark the completion of an operation.
        
        Args:
            result: Optional result data to emit
        """
        operation_name = self._current_operation
        self._current_operation = None
        self._is_busy = False
        self._logger.info(f"Completed operation: {operation_name}")
        self.operation_completed.emit(operation_name, result)
        
    def _fail_operation(self, error: Exception) -> None:
        """
        Mark the failure of an operation.
        
        Args:
            error: Exception that caused the failure
        """
        operation_name = self._current_operation
        error_msg = str(error)
        self._current_operation = None
        self._is_busy = False
        self._logger.error(f"Failed operation {operation_name}: {error_msg}")
        self.operation_failed.emit(operation_name, error_msg)
        
    def _update_progress(self, percentage: int) -> None:
        """
        Update operation progress.
        
        Args:
            percentage: Progress percentage (0-100)
        """
        percentage = max(0, min(100, percentage))  # Clamp to 0-100
        self.progress_updated.emit(percentage)
        
    def _update_status(self, message: str) -> None:
        """
        Update status message.
        
        Args:
            message: Status message to display
        """
        self._logger.debug(f"Status: {message}")
        self.status_changed.emit(message)
        
    def _handle_error(self, error: Exception, operation_name: str = None) -> None:
        """
        Standard error handling for service operations.
        
        Args:
            error: Exception to handle
            operation_name: Name of operation that failed (optional)
        """
        if operation_name is None:
            operation_name = self._current_operation or "unknown_operation"
            
        self._logger.exception(f"Error in {operation_name}: {error}")
        
        if self._is_busy:
            self._fail_operation(error)
            
    def reset(self) -> None:
        """
        Reset service to initial state.
        
        Should be implemented by subclasses to define reset behavior.
        """
        pass
        
    def get_status_info(self) -> Dict[str, Any]:
        """
        Get current service status information.
        
        Returns:
            Dictionary with service status details
        """
        return {
            'service_name': self._service_name,
            'is_busy': self._is_busy,
            'current_operation': self._current_operation
        }
