"""
Base classes for view-specific panels.

These classes provide the foundation for panels that change content based on the current view mode
(Alignment, Filtering, Scoring). Each view can have different left and right panel configurations.
"""

from typing import Dict, Any, Optional
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
    Manages view-specific panels and handles switching between them.
    
    This class coordinates the left and right panels for each view mode
    and handles the transitions when views change.
    """
    
    # Signals
    panels_switched = Signal(ViewMode, ViewMode)  # old_view, new_view
    panel_ready = Signal(ViewMode, str)  # view_mode, panel_type ("left" or "right")
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Store panels for each view
        self.left_panels: Dict[ViewMode, BaseLeftPanel] = {}
        self.right_panels: Dict[ViewMode, BaseRightPanel] = {}
        
        # Current active panels
        self.current_left_panel: Optional[BaseLeftPanel] = None
        self.current_right_panel: Optional[BaseRightPanel] = None
        self.current_view: Optional[ViewMode] = None
        
    def register_left_panel(self, view_mode: ViewMode, panel: BaseLeftPanel):
        """Register a left panel for a specific view mode."""
        self.left_panels[view_mode] = panel
        
        # Connect panel signals
        panel.panel_data_changed.connect(
            lambda data: self._on_panel_data_changed(view_mode, "left", data)
        )
        panel.action_requested.connect(
            lambda action, data: self._on_panel_action_requested(view_mode, "left", action, data)
        )
        
    def register_right_panel(self, view_mode: ViewMode, panel: BaseRightPanel):
        """Register a right panel for a specific view mode."""
        self.right_panels[view_mode] = panel
        
        # Connect panel signals
        panel.panel_data_changed.connect(
            lambda data: self._on_panel_data_changed(view_mode, "right", data)
        )
        panel.action_requested.connect(
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
            self.current_view = view_mode
            
            # Emit signals
            if old_view:
                self.panels_switched.emit(old_view, view_mode)
            
            self.panel_ready.emit(view_mode, "left")
            
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
                
                # Add the stacked widget to the container
                if left_panel_container.layout() is None:
                    layout = QVBoxLayout(left_panel_container)
                    left_panel_container.setLayout(layout)
                
                left_panel_container.layout().addWidget(self.left_panel_stack)
            
            # Create and register panels for each view mode
            self._create_view_panels()
            
            print("Panel manager initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing panels: {e}")
            return False
    
    def _create_view_panels(self):
        """Create panels for each view mode."""
        from src.ui.panels.alignment_left_panel import AlignmentLeftPanel
        from src.ui.panels.filtering_left_panel import FilteringLeftPanel
        from src.ui.panels.scoring_left_panel import ScoringLeftPanel
        
        try:
            # Create alignment panels
            alignment_left = AlignmentLeftPanel()
            self.left_panel_stack.addWidget(alignment_left)
            self.register_left_panel(ViewMode.ALIGNMENT, alignment_left)
            
            # Create filtering panels
            filtering_left = FilteringLeftPanel()
            self.left_panel_stack.addWidget(filtering_left)
            self.register_left_panel(ViewMode.FILTERING, filtering_left)
            
            # Create scoring panels
            scoring_left = ScoringLeftPanel()
            self.left_panel_stack.addWidget(scoring_left)
            self.register_left_panel(ViewMode.SCORING, scoring_left)
            
            print("View panels created successfully")
            
        except Exception as e:
            print(f"Error creating view panels: {e}")
    
    def get_current_left_panel(self) -> Optional[BaseLeftPanel]:
        """Get the currently active left panel."""
        return self.current_left_panel
    
    def get_current_right_panel(self) -> Optional[BaseRightPanel]:
        """Get the currently active right panel."""
        return self.current_right_panel
    
    def get_panel(self, view_mode: ViewMode, panel_type: str) -> Optional[BaseViewPanel]:
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
            panel.reset_panel()
        for panel in self.right_panels.values():
            panel.reset_panel()
    
    def update_panel_availability(self):
        """
        Update the availability of all panels based on the current application state.
        
        This method should be called when data is loaded or when the application state changes
        to ensure panels are properly enabled/disabled based on available data.
        """
        try:
            # This method is called by various managers when they need to update panel availability
            # For now, we'll implement basic availability logic
            # The actual logic should be implemented based on the specific requirements
            
            # We could check if we have access to the main window to get state information
            if hasattr(self.parent(), 'current_sem_image') and hasattr(self.parent(), 'current_gds_overlay'):
                has_sem = self.parent().current_sem_image is not None
                has_gds = self.parent().current_gds_overlay is not None
                has_alignment = getattr(self.parent(), 'current_alignment_result', None) is not None
                
                # Update panel availability based on loaded data
                self._update_panel_states(has_sem, has_gds, has_alignment)
            
        except Exception as e:
            print(f"Error in ViewPanelManager.update_panel_availability: {e}")
    
    def _update_panel_states(self, has_sem: bool, has_gds: bool, has_alignment: bool):
        """Update panel states based on available data."""
        try:
            # Update alignment panel
            alignment_panel = self.left_panels.get(ViewMode.ALIGNMENT)
            if alignment_panel:
                alignment_panel.setEnabled(has_gds)
                alignment_panel.update_panel_data({
                    'has_sem_image': has_sem,
                    'has_gds_overlay': has_gds,
                })
            
            # Update filtering panel  
            filtering_panel = self.left_panels.get(ViewMode.FILTERING)
            if filtering_panel:
                filtering_panel.setEnabled(has_sem)
                filtering_panel.update_panel_data({
                    'has_sem_image': has_sem,
                })
            
            # Update scoring panel
            scoring_panel = self.left_panels.get(ViewMode.SCORING)
            if scoring_panel:
                scoring_panel.setEnabled(has_alignment)
                scoring_panel.update_panel_data({
                    'has_alignment_result': has_alignment,
                })
                
        except Exception as e:
            print(f"Error updating panel states: {e}")

    def __str__(self) -> str:
        return f"ViewPanelManager(current_view={self.current_view.value if self.current_view else None})"
