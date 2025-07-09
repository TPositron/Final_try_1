"""
Signal Handlers - Centralized Signal Connection and Event Management

This module handles all signal connections and event handlers for the main window,
providing centralized signal management and event dispatching.

Main Class:
- SignalHandlers: Qt-based handler for signal connections and events

Key Methods:
- connect_all_signals(): Connects all application signals
- disconnect_all_signals(): Disconnects all signals for cleanup
- reconnect_specific_signals(): Reconnects signals for specific module
- Various signal handler methods for different operations

Dependencies:
- Uses: PySide6.QtCore, PySide6.QtWidgets (Qt integration)
- Called by: ui/main_window.py (signal management)
- Coordinates with: All UI modules and services

Signal Categories:
- UI Signals: User interface interactions
- File Operation Signals: File loading and saving events
- GDS Operation Signals: GDS file and structure events
- Image Processing Signals: Filter and processing events
- Alignment Operation Signals: Alignment and transformation events
- Scoring Operation Signals: Scoring calculation events
- View Controller Signals: View switching and panel events

Features:
- Centralized signal management
- Modular signal connection by category
- Error handling for signal operations
- Signal reconnection capabilities
- Event dispatching and coordination
"""

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QMessageBox


class SignalHandlers(QObject):
    """Handles signal connections and event dispatching for the main window."""
    
    def __init__(self, main_window):
        """Initialize signal handlers with reference to main window."""
        super().__init__()
        self.main_window = main_window
        
    def connect_all_signals(self):
        """Connect all signals for the application."""
        try:
            print("Connecting application signals...")
            
            # Connect UI signals
            self._connect_ui_signals()
            
            # Connect file operation signals
            self._connect_file_operation_signals()
            
            # Connect GDS operation signals
            self._connect_gds_operation_signals()
            
            # Connect image processing signals
            self._connect_image_processing_signals()
            
            # Connect alignment operation signals
            self._connect_alignment_operation_signals()
            
            # Connect scoring operation signals
            self._connect_scoring_operation_signals()
            
            # Connect view controller signals
            self._connect_view_controller_signals()
            
            print("✓ All signals connected successfully")
            
        except Exception as e:
            print(f"Error connecting signals: {e}")
            QMessageBox.critical(
                self.main_window,
                "Signal Connection Error",
                f"Failed to connect application signals: {str(e)}"
            )
    
    def _connect_ui_signals(self):
        """Connect UI-related signals."""
        try:
            # Structure combo box
            if hasattr(self.main_window, 'structure_combo'):
                self.main_window.structure_combo.currentTextChanged.connect(
                    self.main_window.gds_operations.on_structure_selected
                )
            
            # Image viewer signals
            if hasattr(self.main_window, 'image_viewer'):
                # Connect image viewer events if needed
                pass
            
            print("✓ UI signals connected")
            
        except Exception as e:
            print(f"Error connecting UI signals: {e}")
    
    def _connect_file_operation_signals(self):
        """Connect file operation signals."""
        try:
            file_ops = self.main_window.file_operations
            
            # Connect file operation signals to handlers
            file_ops.file_loaded.connect(self.on_file_loaded)
            file_ops.file_save_completed.connect(self.on_file_saved)
            
            print("✓ File operation signals connected")
            
        except Exception as e:
            print(f"Error connecting file operation signals: {e}")
    
    def _connect_gds_operation_signals(self):
        """Connect GDS operation signals."""
        try:
            gds_ops = self.main_window.gds_operations
            
            # Connect GDS operation signals to handlers
            gds_ops.gds_loaded.connect(self.on_gds_loaded)
            gds_ops.structure_loaded.connect(self.on_structure_loaded)
            
            print("✓ GDS operation signals connected")
            
        except Exception as e:
            print(f"Error connecting GDS operation signals: {e}")
    
    def _connect_image_processing_signals(self):
        """Connect image processing signals."""
        try:
            img_proc = self.main_window.image_processing
            
            # Connect image processing signals to handlers
            img_proc.filter_applied.connect(self.on_filter_applied)
            img_proc.filter_preview_ready.connect(self.on_filter_preview)
            img_proc.filters_reset.connect(self.on_filters_reset)
            
            print("✓ Image processing signals connected")
            
        except Exception as e:
            print(f"Error connecting image processing signals: {e}")
    
    def _connect_alignment_operation_signals(self):
        """Connect alignment operation signals."""
        try:
            align_ops = self.main_window.alignment_operations
            
            # Connect alignment operation signals to handlers
            align_ops.alignment_completed.connect(self.on_alignment_completed)
            align_ops.alignment_reset.connect(self.on_alignment_reset)
            align_ops.transformation_applied.connect(self.on_transformation_applied)
            
            print("✓ Alignment operation signals connected")
            
        except Exception as e:
            print(f"Error connecting alignment operation signals: {e}")
    
    def _connect_scoring_operation_signals(self):
        """Connect scoring operation signals."""
        try:
            score_ops = self.main_window.scoring_operations
            
            # Connect scoring operation signals to handlers
            score_ops.scores_calculated.connect(self.on_scores_calculated)
            score_ops.batch_scoring_completed.connect(self.on_batch_scoring_completed)
            
            print("✓ Scoring operation signals connected")
            
        except Exception as e:
            print(f"Error connecting scoring operation signals: {e}")
    
    def _connect_view_controller_signals(self):
        """Connect view controller signals."""
        try:
            view_ctrl = self.main_window.view_controller
            
            # Connect view controller signals to handlers
            view_ctrl.view_changed.connect(self.on_view_changed)
            view_ctrl.panel_updated.connect(self.on_panel_updated)
            
            print("✓ View controller signals connected")
            
        except Exception as e:
            print(f"Error connecting view controller signals: {e}")
    
    # Signal Handler Methods
    
    def on_file_loaded(self, file_type, file_path):
        """Handle file loaded signal."""
        try:
            print(f"File loaded: {file_type} - {file_path}")
            
            if file_type == "SEM":
                # Update UI state for SEM image
                self._update_ui_for_sem_loaded()
            elif file_type == "GDS":
                # Update UI state for GDS file
                self._update_ui_for_gds_loaded()
            
            # Update panel availability
            self.main_window._update_panel_availability()
            
        except Exception as e:
            print(f"Error handling file loaded signal: {e}")
    
    def on_file_saved(self, file_type, file_path):
        """Handle file save completed signal."""
        try:
            print(f"File saved: {file_type} - {file_path}")
            self.main_window.status_bar.showMessage(f"Saved {file_type} to {file_path}")
            
        except Exception as e:
            print(f"Error handling file saved signal: {e}")
    
    def on_gds_loaded(self, gds_filename):
        """Handle GDS loaded signal."""
        try:
            print(f"GDS loaded: {gds_filename}")
            # Update UI state as needed
            
        except Exception as e:
            print(f"Error handling GDS loaded signal: {e}")
    
    def on_structure_loaded(self, structure_name, overlay):
        """Handle structure loaded signal."""
        try:
            print(f"Structure loaded: {structure_name}")
            # Update UI panels that depend on structure
            self.main_window._update_panel_availability()
            
        except Exception as e:
            print(f"Error handling structure loaded signal: {e}")
    
    def on_filter_applied(self, filter_name, parameters):
        """Handle filter applied signal."""
        try:
            print(f"Filter applied: {filter_name}")
            # Update UI panels that show filter status
            
        except Exception as e:
            print(f"Error handling filter applied signal: {e}")
    
    def on_filter_preview(self, filter_name, parameters, preview_image):
        """Handle filter preview signal."""
        try:
            print(f"Filter preview ready: {filter_name}")
            # Could update a preview panel if it exists
            
        except Exception as e:
            print(f"Error handling filter preview signal: {e}")
    
    def on_filters_reset(self):
        """Handle filters reset signal."""
        try:
            print("Filters reset")
            # Update UI to reflect reset state
            
        except Exception as e:
            print(f"Error handling filters reset signal: {e}")
    
    def on_alignment_completed(self, alignment_result):
        """Handle alignment completed signal."""
        try:
            method = alignment_result.get('method', 'unknown')
            score = alignment_result.get('score', 'N/A')
            print(f"Alignment completed: {method} (score: {score})")
            
            # Update UI panels that show alignment status
            self.main_window._update_panel_availability()
            
        except Exception as e:
            print(f"Error handling alignment completed signal: {e}")
    
    def on_alignment_reset(self):
        """Handle alignment reset signal."""
        try:
            print("Alignment reset")
            # Update UI to reflect reset state
            self.main_window._update_panel_availability()
            
        except Exception as e:
            print(f"Error handling alignment reset signal: {e}")
    
    def on_transformation_applied(self, transformation):
        """Handle transformation applied signal."""
        try:
            print("Transformation applied")
            # Update UI to reflect transformation state
            
        except Exception as e:
            print(f"Error handling transformation applied signal: {e}")
    
    def on_scores_calculated(self, scores):
        """Handle scores calculated signal."""
        try:
            print(f"Scores calculated: {scores}")
            # Update UI panels that show scoring results
            
        except Exception as e:
            print(f"Error handling scores calculated signal: {e}")
    
    def on_batch_scoring_completed(self, batch_results):
        """Handle batch scoring completed signal."""
        try:
            successful = sum(1 for result in batch_results if result.get('success', False))
            total = len(batch_results)
            print(f"Batch scoring completed: {successful}/{total} successful")
            
        except Exception as e:
            print(f"Error handling batch scoring completed signal: {e}")
    
    def on_view_changed(self, new_view, old_view):
        """Handle view changed signal."""
        try:
            print(f"View changed from {old_view} to {new_view}")
            # Update UI for new view
            
        except Exception as e:
            print(f"Error handling view changed signal: {e}")
    
    def on_panel_updated(self, panel_name, panel_data):
        """Handle panel updated signal."""
        try:
            print(f"Panel updated: {panel_name}")
            # Handle panel-specific updates
            
        except Exception as e:
            print(f"Error handling panel updated signal: {e}")
    
    # Helper Methods
    
    def _update_ui_for_sem_loaded(self):
        """Update UI state when SEM image is loaded."""
        try:
            # Enable/disable relevant controls
            if hasattr(self.main_window, 'structure_combo'):
                # Structure combo might be enabled if GDS is also loaded
                pass
            
        except Exception as e:
            print(f"Error updating UI for SEM loaded: {e}")
    
    def _update_ui_for_gds_loaded(self):
        """Update UI state when GDS file is loaded."""
        try:
            # Enable structure selection
            if hasattr(self.main_window, 'structure_combo'):
                self.main_window.structure_combo.setEnabled(True)
            
        except Exception as e:
            print(f"Error updating UI for GDS loaded: {e}")
    
    def disconnect_all_signals(self):
        """Disconnect all signals (useful for cleanup)."""
        try:
            print("Disconnecting all signals...")
            
            # Disconnect signals from each module
            if hasattr(self.main_window, 'file_operations'):
                self.main_window.file_operations.disconnect()
            
            if hasattr(self.main_window, 'gds_operations'):
                self.main_window.gds_operations.disconnect()
            
            if hasattr(self.main_window, 'image_processing'):
                self.main_window.image_processing.disconnect()
            
            if hasattr(self.main_window, 'alignment_operations'):
                self.main_window.alignment_operations.disconnect()
            
            if hasattr(self.main_window, 'scoring_operations'):
                self.main_window.scoring_operations.disconnect()
            
            if hasattr(self.main_window, 'view_controller'):
                self.main_window.view_controller.disconnect()
            
            print("✓ All signals disconnected")
            
        except Exception as e:
            print(f"Error disconnecting signals: {e}")
    
    def reconnect_specific_signals(self, module_name):
        """Reconnect signals for a specific module."""
        try:
            print(f"Reconnecting signals for module: {module_name}")
            
            if module_name == "file_operations":
                self._connect_file_operation_signals()
            elif module_name == "gds_operations":
                self._connect_gds_operation_signals()
            elif module_name == "image_processing":
                self._connect_image_processing_signals()
            elif module_name == "alignment_operations":
                self._connect_alignment_operation_signals()
            elif module_name == "scoring_operations":
                self._connect_scoring_operation_signals()
            elif module_name == "view_controller":
                self._connect_view_controller_signals()
            else:
                print(f"Unknown module for signal reconnection: {module_name}")
            
        except Exception as e:
            print(f"Error reconnecting signals for {module_name}: {e}")
