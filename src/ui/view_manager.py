"""
View Manager - Application View Mode Management and State Coordination

This module handles view switching between Alignment, Filtering, and Scoring modes,
coordinating main application views and managing state transitions between
different operational modes.

Main Classes:
- ViewMode: Enumeration of available view modes
- ViewManager: Qt-based manager for view switching and state coordination

Key Methods:
- switch_to_view(): Switches to specified view mode
- get_view_data(): Gets data for specified view mode
- update_view_data(): Updates data for specified view mode
- set_view_data(): Sets specific data value for view mode
- reset_view_data(): Resets view data to defaults
- mark_view_initialized(): Marks view as initialized

Signals Emitted:
- view_changed(ViewMode, ViewMode): View changed from old to new
- view_ready(ViewMode): View fully loaded and ready
- view_data_updated(ViewMode, dict): View-specific data updated

Dependencies:
- Uses: enum.Enum (view mode enumeration)
- Uses: PySide6.QtCore, PySide6.QtWidgets (Qt framework)
- Called by: ui/view_controller.py (view management)
- Used by: UI components requiring view state information

View Modes:
- ALIGNMENT: Manual and 3-point alignment operations
- FILTERING: Image processing and enhancement
- SCORING: Analysis and comparison metrics

Features:
- View-specific data storage and management
- State tracking and initialization monitoring
- Signal-based communication for view changes
- Default data structures for each view mode
"""

from enum import Enum
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget
from typing import Optional, Dict, Any


class ViewMode(Enum):
    """Enumeration of available view modes."""
    ALIGNMENT = "alignment"
    FILTERING = "filtering" 
    SCORING = "scoring"


class ViewManager(QObject):
    """
    Manages view switching and coordinates UI state between different operational modes.
    
    Signals:
        view_changed: Emitted when the active view changes (old_view, new_view)
        view_ready: Emitted when a view is fully loaded and ready
        view_data_updated: Emitted when view-specific data is updated
    """
    
    # Signals
    view_changed = Signal(ViewMode, ViewMode)  # old_view, new_view
    view_ready = Signal(ViewMode)  # view_mode
    view_data_updated = Signal(ViewMode, dict)  # view_mode, data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Current state
        self._current_view = ViewMode.ALIGNMENT  # Default view
        self._previous_view = None
        
        # View-specific data storage
        self._view_data = {
            ViewMode.ALIGNMENT: {
                'mode': 'manual',  # 'manual' or '3point'
                'sem_image': None,
                'gds_overlay': None,
                'alignment_result': None,
                'selected_points': {'sem': [], 'gds': []}
            },
            ViewMode.FILTERING: {
                'active_filter': None,
                'filter_params': {},
                'filtered_image': None,
                'filter_history': [],
                'auto_mode': False
            },
            ViewMode.SCORING: {
                'scoring_method': 'ssim',  # Default scoring method
                'scores': {},
                'comparison_mode': 'overlay',  # 'overlay', 'side_by_side', 'individual'
                'batch_results': None
            }
        }
        
        # View state tracking
        self._view_initialized = {mode: False for mode in ViewMode}
        
    @property
    def current_view(self) -> ViewMode:
        """Get the currently active view mode."""
        return self._current_view
    
    @property
    def previous_view(self) -> Optional[ViewMode]:
        """Get the previously active view mode."""
        return self._previous_view
    
    def switch_to_view(self, view_mode: ViewMode) -> bool:
        """
        Switch to the specified view mode.
        
        Args:
            view_mode: The view mode to switch to
            
        Returns:
            True if the switch was successful, False otherwise
        """
        if view_mode == self._current_view:
            return True  # Already in the requested view
            
        try:
            # Store previous view
            old_view = self._current_view
            self._previous_view = old_view
            
            # Validate view mode
            if not isinstance(view_mode, ViewMode):
                raise ValueError(f"Invalid view mode: {view_mode}")
            
            # Update current view
            self._current_view = view_mode
            
            # Emit view changed signal
            self.view_changed.emit(old_view, view_mode)
            
            # Mark view as ready if it's initialized
            if self._view_initialized[view_mode]:
                self.view_ready.emit(view_mode)
                
            return True
            
        except Exception as e:
            print(f"Error switching to view {view_mode}: {e}")
            return False
    
    def get_view_data(self, view_mode: ViewMode) -> Dict[str, Any]:
        """
        Get data for the specified view mode.
        
        Args:
            view_mode: The view mode to get data for
            
        Returns:
            Dictionary containing view-specific data
        """
        return self._view_data.get(view_mode, {}).copy()
    
    def update_view_data(self, view_mode: ViewMode, data: Dict[str, Any]) -> None:
        """
        Update data for the specified view mode.
        
        Args:
            view_mode: The view mode to update data for
            data: Dictionary containing data to update
        """
        if view_mode in self._view_data:
            self._view_data[view_mode].update(data)
            self.view_data_updated.emit(view_mode, data)
    
    def set_view_data(self, view_mode: ViewMode, key: str, value: Any) -> None:
        """
        Set a specific data value for a view mode.
        
        Args:
            view_mode: The view mode to set data for
            key: The data key to set
            value: The value to set
        """
        if view_mode in self._view_data:
            self._view_data[view_mode][key] = value
            self.view_data_updated.emit(view_mode, {key: value})
    
    def get_view_data_value(self, view_mode: ViewMode, key: str, default=None) -> Any:
        """
        Get a specific data value for a view mode.
        
        Args:
            view_mode: The view mode to get data from
            key: The data key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            The data value or default if not found
        """
        return self._view_data.get(view_mode, {}).get(key, default)
    
    def mark_view_initialized(self, view_mode: ViewMode) -> None:
        """
        Mark a view as initialized and ready for use.
        
        Args:
            view_mode: The view mode that has been initialized
        """
        self._view_initialized[view_mode] = True
        if view_mode == self._current_view:
            self.view_ready.emit(view_mode)
    
    def is_view_initialized(self, view_mode: ViewMode) -> bool:
        """
        Check if a view has been initialized.
        
        Args:
            view_mode: The view mode to check
            
        Returns:
            True if the view is initialized, False otherwise
        """
        return self._view_initialized.get(view_mode, False)
    
    def reset_view_data(self, view_mode: ViewMode) -> None:
        """
        Reset data for the specified view mode to defaults.
        
        Args:
            view_mode: The view mode to reset
        """
        if view_mode == ViewMode.ALIGNMENT:
            self._view_data[view_mode] = {
                'mode': 'manual',
                'sem_image': None,
                'gds_overlay': None,
                'alignment_result': None,
                'selected_points': {'sem': [], 'gds': []}
            }
        elif view_mode == ViewMode.FILTERING:
            self._view_data[view_mode] = {
                'active_filter': None,
                'filter_params': {},
                'filtered_image': None,
                'filter_history': [],
                'auto_mode': False
            }
        elif view_mode == ViewMode.SCORING:
            self._view_data[view_mode] = {
                'scoring_method': 'ssim',
                'scores': {},
                'comparison_mode': 'overlay',
                'batch_results': None
            }
        
        self.view_data_updated.emit(view_mode, self._view_data[view_mode])
    
    def get_all_view_modes(self) -> list[ViewMode]:
        """Get a list of all available view modes."""
        return list(ViewMode)
    
    def __str__(self) -> str:
        return f"ViewManager(current={self._current_view.value}, previous={self._previous_view.value if self._previous_view else None})"
