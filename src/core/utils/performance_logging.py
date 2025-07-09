"""
Performance logging module for the SEM/GDS alignment tool.

This module provides utilities for tracking and logging performance metrics
including execution times, memory usage, and operation statistics.
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager

from .simple_logging import get_logger


@dataclass
class PerformanceMetric:
    """Container for a single performance metric."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    memory_start: Optional[float] = None
    memory_end: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration in seconds."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None
    
    @property
    def memory_delta(self) -> Optional[float]:
        """Get the memory usage change in MB."""
        if self.memory_start is not None and self.memory_end is not None:
            return self.memory_end - self.memory_start
        return None


class PerformanceMonitor:
    """
    Monitor and log performance metrics for application operations.
    """
    
    def __init__(self, enable_memory_tracking: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            enable_memory_tracking: Whether to track memory usage
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.logger = get_logger("performance")
        self._metrics: Dict[str, PerformanceMetric] = {}
        self._active_operations: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        # Performance statistics
        self._operation_stats: Dict[str, List[float]] = {}
        self._memory_stats: Dict[str, List[float]] = {}
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not self.enable_memory_tracking:
            return 0.0
        
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0
    
    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start tracking a performance operation.
        
        Args:
            operation_name: Name of the operation
            metadata: Optional metadata to track with the operation
            
        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"
        
        metric = PerformanceMetric(
            name=operation_name,
            start_time=time.time(),
            memory_start=self._get_memory_usage(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self._metrics[operation_id] = metric
            self._active_operations[operation_id] = time.time()
        
        self.logger.debug(f"Started tracking operation: {operation_name} (ID: {operation_id})")
        return operation_id
    
    def end_operation(self, operation_id: str) -> Optional[PerformanceMetric]:
        """
        End tracking an operation and log results.
        
        Args:
            operation_id: ID returned from start_operation
            
        Returns:
            Performance metric if operation was found
        """
        with self._lock:
            if operation_id not in self._metrics:
                self.logger.warning(f"Unknown operation ID: {operation_id}")
                return None
            
            metric = self._metrics[operation_id]
            metric.end_time = time.time()
            metric.memory_end = self._get_memory_usage()
            
            # Remove from active operations
            self._active_operations.pop(operation_id, None)
            
            # Update statistics
            if metric.duration is not None:
                if metric.name not in self._operation_stats:
                    self._operation_stats[metric.name] = []
                self._operation_stats[metric.name].append(metric.duration)
            
            if metric.memory_delta is not None and self.enable_memory_tracking:
                if metric.name not in self._memory_stats:
                    self._memory_stats[metric.name] = []
                self._memory_stats[metric.name].append(metric.memory_delta)
        
        # Log the result
        self._log_metric(metric)
        return metric
    
    def _log_metric(self, metric: PerformanceMetric):
        """Log a performance metric."""
        message_parts = [f"Operation: {metric.name}"]
        
        if metric.duration is not None:
            message_parts.append(f"Duration: {metric.duration:.3f}s")
        
        if metric.memory_delta is not None and self.enable_memory_tracking:
            message_parts.append(f"Memory: {metric.memory_delta:+.2f}MB")
        
        if metric.metadata:
            metadata_str = ", ".join(f"{k}={v}" for k, v in metric.metadata.items())
            message_parts.append(f"Metadata: {metadata_str}")
        
        self.logger.info(" | ".join(message_parts))
    
    @contextmanager
    def track_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracking an operation.
        
        Args:
            operation_name: Name of the operation
            metadata: Optional metadata to track
            
        Usage:
            with monitor.track_operation("image_processing", {"size": "1024x768"}):
                # Your operation code here
        """
        operation_id = self.start_operation(operation_name, metadata)
        try:
            yield operation_id
        finally:
            self.end_operation(operation_id)
    
    def get_operation_statistics(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Dictionary with statistics or None if not found
        """
        with self._lock:
            durations = self._operation_stats.get(operation_name)
            memory_deltas = self._memory_stats.get(operation_name)
        
        if not durations:
            return None
        
        stats = {
            'operation': operation_name,
            'count': len(durations),
            'duration': {
                'min': min(durations),
                'max': max(durations),
                'avg': sum(durations) / len(durations),
                'total': sum(durations)
            }
        }
        
        if memory_deltas and self.enable_memory_tracking:
            stats['memory'] = {
                'min': min(memory_deltas),
                'max': max(memory_deltas),
                'avg': sum(memory_deltas) / len(memory_deltas),
                'total': sum(memory_deltas)
            }
        
        return stats
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all tracked operations."""
        all_stats = {}
        
        with self._lock:
            operation_names = set(self._operation_stats.keys())
        
        for name in operation_names:
            stats = self.get_operation_statistics(name)
            if stats:
                all_stats[name] = stats
        
        return all_stats
    
    def get_active_operations(self) -> Dict[str, float]:
        """
        Get currently active operations and their start times.
        
        Returns:
            Dictionary mapping operation IDs to start times
        """
        with self._lock:
            return self._active_operations.copy()
    
    def log_system_info(self):
        """Log current system performance information."""
        try:
            # CPU information
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            memory_percent = memory.percent
            
            # Disk information for logs directory
            disk_usage = psutil.disk_usage('.')
            disk_free_gb = disk_usage.free / (1024**3)
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            self.logger.info(
                f"System Info | CPU: {cpu_percent}% ({cpu_count} cores) | "
                f"Memory: {memory_percent}% used ({memory_available_gb:.1f}GB available) | "
                f"Disk: {disk_percent:.1f}% used ({disk_free_gb:.1f}GB free)"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log system info: {e}")
    
    def reset_statistics(self):
        """Reset all performance statistics."""
        with self._lock:
            self._operation_stats.clear()
            self._memory_stats.clear()
            self._metrics.clear()
        
        self.logger.info("Performance statistics reset")


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor


@contextmanager
def track_performance(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Convenience function for tracking operation performance.
    
    Args:
        operation_name: Name of the operation
        metadata: Optional metadata to track
        
    Usage:
        with track_performance("image_filtering", {"filter": "gaussian"}):
            # Your operation code here
    """
    with _performance_monitor.track_operation(operation_name, metadata):
        yield


def log_performance_summary():
    """Log a summary of all performance statistics."""
    logger = get_logger("performance.summary")
    
    stats = _performance_monitor.get_all_statistics()
    
    if not stats:
        logger.info("No performance statistics available")
        return
    
    logger.info("=== Performance Summary ===")
    
    for operation_name, operation_stats in stats.items():
        duration = operation_stats['duration']
        memory = operation_stats.get('memory')
        
        message = (
            f"{operation_name}: "
            f"{operation_stats['count']} calls, "
            f"avg {duration['avg']:.3f}s "
            f"(min: {duration['min']:.3f}s, max: {duration['max']:.3f}s, "
            f"total: {duration['total']:.3f}s)"
        )
        
        if memory:
            message += f" | Memory avg: {memory['avg']:+.2f}MB"
        
        logger.info(message)


def reset_performance_stats():
    """Reset all performance statistics."""
    _performance_monitor.reset_statistics()


def log_system_performance():
    """Log current system performance information."""
    _performance_monitor.log_system_info()
