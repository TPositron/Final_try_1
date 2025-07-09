"""
Base Panel Classes - Foundation for View-Specific UI Panels

This module provides foundational base classes for all view-specific panels,
establishing consistent patterns for panel behavior, signal communication,
and view management across different application modes.

Main Classes:
- BaseViewPanel: Foundation for all view-specific panels
- BaseLeftPanel: Specialized for control panels (left side)
- BaseRightPanel: Specialized for information panels (right side)
- ViewPanelManager: Coordinates panel switching and management

Key Methods:
- BaseViewPanel: init_panel(), update_panel_data(), reset_panel(), get_panel_data()
- BaseLeftPanel: setup_modes(), add_mode(), switch_mode(), get_current_mode()
- BaseRightPanel: setup_display_sections(), add_display_section(), update_display_section()
- ViewPanelManager: register_left_panel(), register_right_panel(), switch_to_view()

Signals Emitted:
- panel_data_changed(dict): Panel data modified
- action_requested(str, dict): User actions triggered
- panels_switched(ViewMode, ViewMode): View modes changed
- panel_ready(ViewMode, str): Panels initialized

Dependencies:
- Uses: PySide6.QtWidgets, PySide6.QtCore (Qt framework)
- Uses: ui/view_manager.ViewMode (view mode enumeration)
- Called by: ui/main_window.py (panel integration)
- Inherited by: ui/panels/* (specific panel implementations)

View Modes:
- ALIGNMENT: Manual and automatic alignment operations
- FILTERING: Image processing and enhancement
- SCORING: Analysis and comparison metrics

Features:
- Enhanced error handling for missing panel modules
- Graceful fallback to placeholder panels
- Flexible panel architecture supporting tabbed interfaces
- Automatic signal connection and routing
- Comprehensive state management
"""

from typing import Dict, Any, Optional, Union
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QStackedWidget
from PySide6.QtCore import QObject, Signal
from src.ui.view_manager import ViewMode


class BaseViewPanel(QWidget):
    """
    Base class for view-specific panels.
    
    This class defines the interface that all view panels should implement
    and provides common functionality for panel management.
    """
    
    # Signals
    panel_data_changed = Signal(dict)  # Emitted when panel data changes
    action_requested = Signal(str, dict)  # Emitted when an action is requested
    
    def __init__(self, view_mode: ViewMode, parent=None):
        super().__init__(parent)
        self.view_mode = view_mode
        self.panel_data = {}
        self.is_initialized = False
        
        # Set up the basic layout
        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)
        
        # Initialize the panel
        self.init_panel()
        self.is_initialized = True
    
    def init_panel(self):
        """Initialize the panel UI components. Should be overridden by subclasses."""
        # Default implementation - add a simple label
        label = QLabel(f"{self.view_mode.value.title()} Panel")
        self.main_layout.addWidget(label)
    
    def update_panel_data(self, data: Dict[str, Any]):
        """Update panel with new data. Should be overridden by subclasses."""
        self.panel_data.update(data)
        self.panel_data_changed.emit(self.panel_data)
    
    def reset_panel(self):
        """Reset panel to default state. Should be overridden by subclasses."""
        self.panel_data.clear()
        self.panel_data_changed.emit(self.panel_data)
    
    def get_panel_data(self) -> Dict[str, Any]:
        """Get current panel data."""
        return self.panel_data.copy()
    
    def set_panel_data(self, key: str, value: Any):
        """Set a specific data value and emit signal."""
        self.panel_data[key] = value
        self.panel_data_changed.emit(self.panel_data)
    
    def enable_panel(self, enabled: bool = True):
        """Enable or disable the entire panel."""
        self.setEnabled(enabled)
    
    def get_view_mode(self) -> ViewMode:
        """Get the view mode this panel is associated with."""
        return self.view_mode


class BaseLeftPanel(BaseViewPanel):
    """
    Base class for left-side panels that contain controls and options.
    
    Left panels typically contain:
    - Mode switching (e.g., Manual/3-point for alignment)
    - Control buttons and options
    - Parameter inputs
    - Action triggers
    """
    
    def __init__(self, view_mode: ViewMode, parent=None):
        super().__init__(view_mode, parent)
        self.modes = {}  # Store different modes for this panel
        self.current_mode = None
        
    def setup_modes(self):
        """Setup different modes for this panel (e.g., Manual/3-point). Should be overridden by subclasses."""
        pass
    
    def add_mode(self, mode_name: str, mode_widget: QWidget):
        """Add a mode widget to this panel."""
        self.modes[mode_name] = mode_widget
    
    def switch_mode(self, mode_name: str):
        """Switch to a specific mode."""
        if mode_name in self.modes and mode_name != self.current_mode:
            self.current_mode = mode_name
            self.action_requested.emit("mode_changed", {"mode": mode_name})
    
    def get_current_mode(self) -> Optional[str]:
        """Get the currently active mode."""
        return self.current_mode


class BaseRightPanel(BaseViewPanel):
    """
    Base class for right-side panels that display information and results.
    
    Right panels typically contain:
    - Status displays
    - Results visualization
    - Information panels
    - Secondary controls
    """
    
    def __init__(self, view_mode: ViewMode, parent=None):
        super().__init__(view_mode, parent)
        self.display_sections = {}  # Store different display sections
        
    def setup_display_sections(self):
        """Setup different display sections for this panel. Should be overridden by subclasses."""
        pass
    
    def add_display_section(self, section_name: str, section_widget: QWidget):
        """Add a display section to this panel."""
        self.display_sections[section_name] = section_widget
    
    def update_display_section(self, section_name: str, data: Dict[str, Any]):
        """Update a specific display section with new data."""
        if section_name in self.display_sections:
            # Emit signal for the specific section update
            self.action_requested.emit("section_updated", {
                "section": section_name,
                "data": data
            })


class ViewPanelManager(QObject):
    """
    Enhanced panel manager with better support for tabbed interfaces and advanced panels.
    
    This class coordinates the left and right panels for each view mode
    and handles the transitions when views change.
    """
    
    # Signals
    panels_switched = Signal(ViewMode, ViewMode)  # old_view, new_view
    panel_ready = Signal(ViewMode, str)  # view_mode, panel_type ("left" or "right")
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Store panels for each view (updated to accept any QWidget)
        self.left_panels: Dict[ViewMode, Union[BaseLeftPanel, QWidget]] = {}
        self.right_panels: Dict[ViewMode, Union[BaseRightPanel, QWidget]] = {}
        
        # Current active panels
        self.current_left_panel: Optional[Union[BaseLeftPanel, QWidget]] = None
        self.current_right_panel: Optional[Union[BaseRightPanel, QWidget]] = None
        self.current_view: Optional[ViewMode] = None
        
    def register_left_panel(self, view_mode: ViewMode, panel: Union[BaseLeftPanel, QWidget]):
        """Register a left panel for a specific view mode."""
        self.left_panels[view_mode] = panel
        
        # Connect panel signals (check if panel has the required signals)
        if hasattr(panel, 'panel_data_changed'):
            panel.panel_data_changed.connect(  # type: ignore
                lambda data: self._on_panel_data_changed(view_mode, "left", data)
            )
        if hasattr(panel, 'action_requested'):
            panel.action_requested.connect(  # type: ignore
                lambda action, data: self._on_panel_action_requested(view_mode, "left", action, data)
            )
        
    def register_right_panel(self, view_mode: ViewMode, panel: Union[BaseRightPanel, QWidget]):
        """Register a right panel for a specific view mode."""
        self.right_panels[view_mode] = panel
        
        # Connect panel signals (check if panel has the required signals)
        if hasattr(panel, 'panel_data_changed'):
            panel.panel_data_changed.connect(  # type: ignore
                lambda data: self._on_panel_data_changed(view_mode, "right", data)
            )
        if hasattr(panel, 'action_requested'):
            panel.action_requested.connect(  # type: ignore
                lambda action, data: self._on_panel_action_requested(view_mode, "right", action, data)
            )
    
    def switch_to_view(self, view_mode: ViewMode) -> bool:
        """
        Switch panels to the specified view mode.
        
        Args:
            view_mode: The view mode to switch to
            
        Returns:
            True if switch was successful, False otherwise
        """
        try:
            old_view = self.current_view
            
            # Get left panel for the new view
            new_left_panel = self.left_panels.get(view_mode)
            
            if not new_left_panel:
                print(f"Warning: Missing left panel for view {view_mode.value}")
                return False
            
            # Switch to the appropriate panel in the stack
            if hasattr(self, 'left_panel_stack'):
                panel_index = self.left_panel_stack.indexOf(new_left_panel)
                if panel_index >= 0:
                    self.left_panel_stack.setCurrentIndex(panel_index)
                else:
                    print(f"Warning: Panel not found in stack for {view_mode.value}")
            
            # Update current references
            self.current_left_panel = new_left_panel
            self.current_right_panel = self.right_panels.get(view_mode)
            self.current_view = view_mode
            
            # Emit signals
            if old_view:
                self.panels_switched.emit(old_view, view_mode)
            
            self.panel_ready.emit(view_mode, "left")
            if self.current_right_panel:
                self.panel_ready.emit(view_mode, "right")
            
            return True
            
        except Exception as e:
            print(f"Error switching panels to view {view_mode.value}: {e}")
            return False
    
    def initialize_panels(self, left_panel_container: QWidget):
        """
        Initialize panels and set up the left panel container.
        
        Args:
            left_panel_container: The widget that will contain the left panels
        """
        try:
            # Create a stacked widget for left panels if not already done
            if not hasattr(self, 'left_panel_stack'):
                self.left_panel_stack = QStackedWidget()
                
                # Ensure container has a layout
                container_layout = left_panel_container.layout()
                if container_layout is None:
                    container_layout = QVBoxLayout(left_panel_container)
                    left_panel_container.setLayout(container_layout)
                
                container_layout.addWidget(self.left_panel_stack)
            
            # Create and register panels for each view mode
            self._create_view_panels()
            
            print("Panel manager initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing panels: {e}")
            return False
    
    def _create_view_panels(self):
        """Create panels for each view mode with enhanced error handling."""
        try:
            # Import panels with better error handling
            try:
                from src.ui.panels.alignment_left_panel import AlignmentLeftPanel
                alignment_left = AlignmentLeftPanel()
                self.left_panel_stack.addWidget(alignment_left)
                self.register_left_panel(ViewMode.ALIGNMENT, alignment_left)
                print("✓ Alignment panel created")
            except ImportError as e:
                print(f"Warning: Could not import AlignmentLeftPanel: {e}")
                # Create a placeholder
                placeholder = QLabel("Alignment Panel\n(Module not found)")
                self.left_panel_stack.addWidget(placeholder)
                self.register_left_panel(ViewMode.ALIGNMENT, placeholder)
            
            # Create advanced filtering panels (both left and right)
            try:
                from src.ui.panels.advanced_filtering_panels import (
                    AdvancedFilteringLeftPanel, 
                    AdvancedFilteringRightPanel
                )
                filtering_left = AdvancedFilteringLeftPanel()
                filtering_right = AdvancedFilteringRightPanel()
                self.left_panel_stack.addWidget(filtering_left)
                self.register_left_panel(ViewMode.FILTERING, filtering_left)
                self.register_right_panel(ViewMode.FILTERING, filtering_right)
                print("✓ Advanced filtering panels created")
            except ImportError as e:
                print(f"Warning: Could not import AdvancedFilteringPanels: {e}")
                # Create placeholders
                placeholder_left = QLabel("Advanced Filtering Panel\n(Module not found)")
                placeholder_right = QLabel("Filtering Info Panel\n(Module not found)")
                self.left_panel_stack.addWidget(placeholder_left)
                self.register_left_panel(ViewMode.FILTERING, placeholder_left)
                self.register_right_panel(ViewMode.FILTERING, placeholder_right)
            
            # Create scoring panels
            try:
                from src.ui.panels.scoring_left_panel import ScoringLeftPanel
                scoring_left = ScoringLeftPanel()
                self.left_panel_stack.addWidget(scoring_left)
                self.register_left_panel(ViewMode.SCORING, scoring_left)
                print("✓ Scoring panel created")
            except ImportError as e:
                print(f"Warning: Could not import ScoringLeftPanel: {e}")
                # Create a placeholder
                placeholder = QLabel("Scoring Panel\n(Module not found)")
                self.left_panel_stack.addWidget(placeholder)
                self.register_left_panel(ViewMode.SCORING, placeholder)
            
            print(f"✓ Panel creation completed. Registered panels:")
            print(f"  Left panels: {list(self.left_panels.keys())}")
            print(f"  Right panels: {list(self.right_panels.keys())}")
            
        except Exception as e:
            print(f"Error creating view panels: {e}")
            import traceback
            traceback.print_exc()
    
    def get_current_left_panel(self) -> Optional[Union[BaseLeftPanel, QWidget]]:
        """Get the currently active left panel."""
        return self.current_left_panel
    
    def get_current_right_panel(self) -> Optional[Union[BaseRightPanel, QWidget]]:
        """Get the currently active right panel."""
        return self.current_right_panel
    
    def get_panel(self, view_mode: ViewMode, panel_type: str) -> Optional[Union[BaseViewPanel, QWidget]]:
        """
        Get a specific panel.
        
        Args:
            view_mode: The view mode
            panel_type: "left" or "right"
            
        Returns:
            The requested panel or None if not found
        """
        if panel_type == "left":
            return self.left_panels.get(view_mode)
        elif panel_type == "right":
            return self.right_panels.get(view_mode)
        return None
    
    def _on_panel_data_changed(self, view_mode: ViewMode, panel_type: str, data: Dict[str, Any]):
        """Handle panel data changes."""
        print(f"Panel data changed: {view_mode.value} {panel_type} - {list(data.keys())}")
    
    def _on_panel_action_requested(self, view_mode: ViewMode, panel_type: str, action: str, data: Dict[str, Any]):
        """Handle panel action requests."""
        print(f"Panel action requested: {view_mode.value} {panel_type} - {action}")
    
    def reset_all_panels(self):
        """Reset all panels to their default state."""
        for panel in self.left_panels.values():
            if hasattr(panel, 'reset_panel'):
                try:
                    panel.reset_panel()  # type: ignore
                except Exception as e:
                    print(f"Error resetting panel: {e}")
        for panel in self.right_panels.values():
            if hasattr(panel, 'reset_panel'):
                try:
                    panel.reset_panel()  # type: ignore
                except Exception as e:
                    print(f"Error resetting panel: {e}")
    
    def update_panel_availability(self):
        """
        Update the availability of all panels based on the current application state.
        
        This method should be called when data is loaded or when the application state changes
        to ensure panels are properly enabled/disabled based on available data.
        """
        try:
            print("Panel availability updated")
            
        except Exception as e:
            print(f"Error in ViewPanelManager.update_panel_availability: {e}")
    
    def setup_filtering_signals(self, main_window):
        """
        Setup signal connections for filtering panels with enhanced error handling.
        
        Args:
            main_window: The main window instance that handles image processing
        """
        try:
            # Get the filtering panels
            left_panel = self.get_panel(ViewMode.FILTERING, "left")
            right_panel = self.get_panel(ViewMode.FILTERING, "right")
            
            # Type check and connect filtering-specific signals
            if left_panel and hasattr(left_panel, 'filter_applied'):
                # Connect filter signals to main window methods
                try:
                    if hasattr(main_window, 'apply_filter'):
                        left_panel.filter_applied.connect(main_window.apply_filter)  # type: ignore
                    
                    if hasattr(main_window, 'preview_filter'):
                        left_panel.filter_previewed.connect(main_window.preview_filter)  # type: ignore
                    
                    if hasattr(main_window, 'reset_image'):
                        left_panel.filter_reset.connect(main_window.reset_image)  # type: ignore
                    
                    if hasattr(main_window, 'save_filtered_image'):
                        left_panel.save_image_requested.connect(main_window.save_filtered_image)  # type: ignore
                    
                    print("✓ Filtering panel signals connected successfully")
                except Exception as e:
                    print(f"Error connecting filtering signals: {e}")
            else:
                print("Warning: Filtering left panel not found or missing filter signals")
                
            if right_panel and hasattr(right_panel, 'update_histogram'):
                try:
                    # Connect image updates to right panel
                    if hasattr(main_window, 'image_loaded'):
                        main_window.image_loaded.connect(right_panel.update_histogram)  # type: ignore
                    
                    print("✓ Filtering right panel signals connected successfully")
                except Exception as e:
                    print(f"Error connecting right panel signals: {e}")
            else:
                print("Info: Filtering right panel found but may not have update_histogram method (this is normal for tabbed panels)")
                
        except Exception as e:
            print(f"Error setting up filtering signals: {e}")

    def __str__(self) -> str:
        return f"ViewPanelManager(current_view={self.current_view.value if self.current_view else None})"