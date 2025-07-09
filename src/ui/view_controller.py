"""
View Controller - View Management and UI State Coordination

This module handles view management, panel switching, and UI state coordination
for the main application interface.

Main Class:
- ViewController: Qt-based controller for view management

Key Methods:
- setup_view_toolbar(): Creates view selection toolbar
- switch_view(): Switches to different view mode
- update_view_availability(): Updates view button availability
- switch_to_best_available_view(): Switches to best available view
- refresh_current_view(): Refreshes current view content
- initialize_view_system(): Initializes complete view system

Signals Emitted:
- view_changed(str, str): View changed from old to new
- panel_updated(str, object): Panel updated with data

Dependencies:
- Uses: PySide6.QtWidgets, PySide6.QtCore, PySide6.QtGui (Qt framework)
- Uses: ui/view_manager.ViewManager, ViewMode (view management)
- Called by: ui/main_window.py (view operations)
- Coordinates with: UI panels and view components

View Modes:
- ALIGNMENT: Align SEM and GDS images
- FILTERING: Apply image filters
- SCORING: Calculate alignment scores

Features:
- View toolbar with exclusive button selection
- Keyboard shortcuts for view switching
- View availability checking based on application state
- Dynamic view content setup and clearing
- Error handling for view operations
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QButtonGroup
from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtGui import QIcon

from src.ui.view_manager import ViewManager, ViewMode


class ViewController(QObject):
    """Handles view management and UI state coordination."""
    
    # Signals
    view_changed = Signal(str, str)  # new_view, old_view
    panel_updated = Signal(str, object)  # panel_name, panel_data
    
    def __init__(self, main_window):
        """Initialize view controller with reference to main window."""
        super().__init__()
        self.main_window = main_window
        self.view_manager = ViewManager(main_window)
        
        # View state
        self.current_view = ViewMode.ALIGNMENT
        self.previous_view = None
        
        # UI elements
        self.view_buttons = {}
        self.view_button_group = None
        
    def setup_view_toolbar(self):
        """Create and setup the view selection toolbar."""
        try:
            # Setup toolbar first if not already done
            if not hasattr(self.main_window, 'view_toolbar') or self.main_window.view_toolbar is None:
                self.main_window.ui_setup.setup_view_toolbar()
                
            # Get the toolbar
            view_toolbar = self.main_window.ui_setup.view_toolbar
            
            # Create button group for exclusive selection
            self.view_button_group = QButtonGroup(self.main_window)
            self.view_button_group.setExclusive(True)
            
            # Define view buttons
            view_configs = [
                (ViewMode.ALIGNMENT, "Alignment", "Align SEM and GDS images"),
                (ViewMode.FILTERING, "Filtering", "Apply image filters"),
                (ViewMode.SCORING, "Scoring", "Calculate alignment scores")
            ]
            
            # Create buttons
            for view_mode, label, tooltip in view_configs:
                button = QPushButton(label)
                button.setCheckable(True)
                button.setToolTip(tooltip)
                button.clicked.connect(lambda checked, vm=view_mode: self.switch_view(vm))
                
                # Add to toolbar and button group
                view_toolbar.addWidget(button)
                self.view_button_group.addButton(button)
                self.view_buttons[view_mode] = button
            
            # Set initial view
            self.view_buttons[ViewMode.ALIGNMENT].setChecked(True)
            
            print("✓ View toolbar setup completed")
            
        except Exception as e:
            print(f"Error setting up view toolbar: {e}")
    
    def switch_view(self, view_mode: ViewMode):
        """Switch to a different view mode."""
        try:
            if view_mode == self.current_view:
                print(f"Already in {view_mode} view, ignoring switch request")
                return
            
            print(f"Switching view from {self.current_view} to {view_mode}")
            
            # Store previous view
            self.previous_view = self.current_view
            
            # Clear current view content
            self._clear_view_content()
            
            # Update current view
            self.current_view = view_mode
            
            # Setup new view
            self._setup_view_content(view_mode)
            
            # Update view button state
            self._update_view_button_state(view_mode)
            
            # Update panel availability
            self.main_window._update_panel_availability()
            
            # Emit signal
            self.view_changed.emit(str(view_mode), str(self.previous_view))
            
            print(f"✓ View switched to {view_mode}")
            
        except Exception as e:
            print(f"Error switching view to {view_mode}: {e}")
    
    def _clear_view_content(self):
        """Clear the current view content."""
        try:
            # Clear left panel
            left_layout = self.main_window.left_panel_layout
            while left_layout.count():
                child = left_layout.takeAt(0)
                if child.widget():
                    child.widget().setParent(None)
            
            # Clear view-specific content in right panel
            view_layout = self.main_window.view_specific_layout
            while view_layout.count():
                child = view_layout.takeAt(0)
                if child.widget():
                    child.widget().setParent(None)
            
        except Exception as e:
            print(f"Error clearing view content: {e}")
    
    def _setup_view_content(self, view_mode: ViewMode):
        """Setup content for the specified view mode."""
        try:
            # Use panel manager to setup view-specific panels
            if hasattr(self.main_window, 'panel_manager'):
                self.main_window.panel_manager.switch_to_view(view_mode)
            
            # Additional view-specific setup
            if view_mode == ViewMode.ALIGNMENT:
                self._setup_alignment_view()
            elif view_mode == ViewMode.FILTERING:
                self._setup_filtering_view()
            elif view_mode == ViewMode.SCORING:
                self._setup_scoring_view()
            
        except Exception as e:
            print(f"Error setting up view content for {view_mode}: {e}")
    
    def _setup_alignment_view(self):
        """Setup alignment-specific view content."""
        try:
            print("Setting up alignment view content")
            
            # Add alignment-specific controls or panels here
            # This is where you might add alignment controls that aren't in panels
            
        except Exception as e:
            print(f"Error setting up alignment view: {e}")
    
    def _setup_filtering_view(self):
        """Setup filtering-specific view content."""
        try:
            print("Setting up filtering view content")
            
            # Add filtering-specific controls or panels here
            
        except Exception as e:
            print(f"Error setting up filtering view: {e}")
    
    def _setup_scoring_view(self):
        """Setup scoring-specific view content."""
        try:
            print("Setting up scoring view content")
            
            # Add scoring-specific controls or panels here
            
        except Exception as e:
            print(f"Error setting up scoring view: {e}")
    
    def _update_view_button_state(self, active_view):
        """Update the view button states."""
        try:
            for view_mode, button in self.view_buttons.items():
                button.setChecked(view_mode == active_view)
                
        except Exception as e:
            print(f"Error updating view button state: {e}")
    
    def get_current_view(self):
        """Get the current view mode."""
        return self.current_view
    
    def get_previous_view(self):
        """Get the previous view mode."""
        return self.previous_view
    
    def is_view_available(self, view_mode: ViewMode):
        """Check if a view mode is available based on current application state."""
        try:
            # Alignment view is always available
            if view_mode == ViewMode.ALIGNMENT:
                return True
            
            # Filtering view requires SEM image
            if view_mode == ViewMode.FILTERING:
                return self.main_window.current_sem_image is not None
            
            # Scoring view requires both SEM image and GDS structure
            if view_mode == ViewMode.SCORING:
                has_sem = self.main_window.current_sem_image is not None
                has_gds = (hasattr(self.main_window, 'gds_operations') and 
                          self.main_window.gds_operations.is_structure_selected())
                return has_sem and has_gds
            
            return False
            
        except Exception as e:
            print(f"Error checking view availability for {view_mode}: {e}")
            return False
    
    def update_view_availability(self):
        """Update the availability of view buttons based on application state."""
        try:
            for view_mode, button in self.view_buttons.items():
                is_available = self.is_view_available(view_mode)
                button.setEnabled(is_available)
                
                # Add visual indication for disabled buttons
                if not is_available:
                    button.setStyleSheet("color: gray;")
                else:
                    button.setStyleSheet("")
            
        except Exception as e:
            print(f"Error updating view availability: {e}")
    
    def switch_to_best_available_view(self):
        """Switch to the best available view based on current application state."""
        try:
            # Priority order: current view (if available), alignment, filtering, scoring
            view_priority = [self.current_view, ViewMode.ALIGNMENT, ViewMode.FILTERING, ViewMode.SCORING]
            
            for view_mode in view_priority:
                if self.is_view_available(view_mode):
                    if view_mode != self.current_view:
                        self.switch_view(view_mode)
                    return view_mode
            
            # Fallback to alignment view
            if self.current_view != ViewMode.ALIGNMENT:
                self.switch_view(ViewMode.ALIGNMENT)
            
            return ViewMode.ALIGNMENT
            
        except Exception as e:
            print(f"Error switching to best available view: {e}")
            return self.current_view
    
    def refresh_current_view(self):
        """Refresh the current view content."""
        try:
            print(f"Refreshing current view: {self.current_view}")
            
            # Re-setup the current view
            self._setup_view_content(self.current_view)
            
            # Update panel availability
            self.main_window._update_panel_availability()
            
        except Exception as e:
            print(f"Error refreshing current view: {e}")
    
    def get_view_info(self):
        """Get information about the current view state."""
        return {
            'current_view': str(self.current_view),
            'previous_view': str(self.previous_view) if self.previous_view else None,
            'available_views': [str(vm) for vm in self.view_buttons.keys() if self.is_view_available(vm)],
            'all_views': [str(vm) for vm in self.view_buttons.keys()]
        }
    
    def setup_view_specific_shortcuts(self):
        """Setup keyboard shortcuts for view switching."""
        try:
            from PySide6.QtGui import QShortcut, QKeySequence
            
            # Ctrl+1 for Alignment
            shortcut_align = QShortcut(QKeySequence("Ctrl+1"), self.main_window)
            shortcut_align.activated.connect(lambda: self.switch_view(ViewMode.ALIGNMENT))
            
            # Ctrl+2 for Filtering
            shortcut_filter = QShortcut(QKeySequence("Ctrl+2"), self.main_window)
            shortcut_filter.activated.connect(lambda: self.switch_view(ViewMode.FILTERING))
            
            # Ctrl+3 for Scoring
            shortcut_score = QShortcut(QKeySequence("Ctrl+3"), self.main_window)
            shortcut_score.activated.connect(lambda: self.switch_view(ViewMode.SCORING))
            
            print("✓ View shortcuts setup completed")
            
        except Exception as e:
            print(f"Error setting up view shortcuts: {e}")
    
    def initialize_view_system(self):
        """Initialize the complete view system."""
        try:
            print("Initializing view system...")
            
            # Setup view toolbar
            self.setup_view_toolbar()
            
            # Setup keyboard shortcuts
            self.setup_view_specific_shortcuts()
            
            # Initialize with default view
            self._setup_view_content(self.current_view)
            
            # Update availability
            self.update_view_availability()
            
            print("✓ View system initialized successfully")
            
        except Exception as e:
            print(f"Error initializing view system: {e}")
    
    def cleanup_view_system(self):
        """Cleanup the view system (useful for shutdown)."""
        try:
            print("Cleaning up view system...")
            
            # Clear view content
            self._clear_view_content()
            
            # Disconnect button signals
            if self.view_button_group:
                self.view_button_group.setParent(None)
            
            print("✓ View system cleaned up")
            
        except Exception as e:
            print(f"Error cleaning up view system: {e}")
