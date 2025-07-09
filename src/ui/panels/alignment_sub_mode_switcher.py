"""
Alignment Sub-Mode Switcher - Alignment Mode Selection Interface

This module provides a sub-mode switcher for alignment operations, allowing users
to switch between manual, hybrid, and automatic alignment modes within the
alignment panel interface.

Main Class:
- AlignmentSubModeSwitcher: Widget with radio buttons for alignment sub-mode selection

Key Methods:
- _setup_ui(): Initializes UI with radio buttons and status display
- _on_sub_mode_selected(): Handles sub-mode selection from radio buttons
- _update_status(): Updates status label based on current sub-mode
- set_sub_mode(): Programmatically sets current sub-mode
- get_current_sub_mode(): Gets currently selected sub-mode
- set_sub_mode_enabled(): Enables or disables specific sub-modes
- set_sub_mode_tooltip(): Sets tooltip for sub-mode buttons

Signals Emitted:
- sub_mode_changed(str): Sub-mode changed to new mode name

Dependencies:
- Uses: PySide6.QtWidgets, PySide6.QtCore (Qt framework)
- Uses: typing (type hints)
- Called by: UI alignment panel components
- Coordinates with: Alignment workflow and mode management

Sub-Modes:
- Manual: Manual alignment with transformation controls
- Hybrid: 3-point alignment with manual refinement
- Automatic: Fully automatic alignment using image features

Features:
- Radio button interface for exclusive sub-mode selection
- Visual status indicator with color coding for each mode
- Programmatic sub-mode control and state management
- Tooltip support for sub-mode descriptions
- Enable/disable functionality for individual sub-modes
- Custom styling with hover effects and visual feedback
"""

from typing import Callable, Optional
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup, 
                              QLabel, QFrame)
from PySide6.QtCore import Signal, Qt


class AlignmentSubModeSwitcher(QWidget):
    """Widget with radio buttons for switching between alignment sub-modes."""
    
    # Signals
    sub_mode_changed = Signal(str)  # Emitted when sub-mode changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_sub_mode = "manual"
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Title
        title_label = QLabel("Alignment Mode:")
        title_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #2E86AB;")
        layout.addWidget(title_label)
        
        # Button group for radio buttons
        self.button_group = QButtonGroup(self)
        self.button_group.buttonClicked.connect(self._on_sub_mode_selected)
        
        # Manual mode
        self.manual_radio = QRadioButton("Manual")
        self.manual_radio.setChecked(True)
        self.manual_radio.setObjectName("manual")
        self.manual_radio.setToolTip("Manual alignment with transformation controls")
        self.button_group.addButton(self.manual_radio)
        layout.addWidget(self.manual_radio)
        
        # Hybrid mode
        self.hybrid_radio = QRadioButton("Hybrid")
        self.hybrid_radio.setObjectName("hybrid")
        self.hybrid_radio.setToolTip("3-point alignment with manual refinement")
        self.button_group.addButton(self.hybrid_radio)
        layout.addWidget(self.hybrid_radio)
        
        # Automatic mode
        self.automatic_radio = QRadioButton("Automatic")
        self.automatic_radio.setObjectName("automatic")
        self.automatic_radio.setToolTip("Fully automatic alignment using image features")
        self.button_group.addButton(self.automatic_radio)
        layout.addWidget(self.automatic_radio)
        
        # Add stretch to push content to top
        layout.addStretch()
        
        # Status indicator
        self.status_label = QLabel("Manual alignment active")
        self.status_label.setStyleSheet("color: #4CAF50; font-style: italic; font-size: 11px;")
        layout.addWidget(self.status_label)
        
        # Apply styling
        self._apply_styling()
    
    def _apply_styling(self):
        """Apply custom styling to the widget."""
        self.setStyleSheet("""
            AlignmentSubModeSwitcher {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 5px;
            }
            QRadioButton {
                font-size: 12px;
                padding: 8px;
                margin: 2px 0px;
                border-radius: 3px;
            }
            QRadioButton:hover {
                background-color: #e3f2fd;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                margin-right: 8px;
            }
            QRadioButton::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #45a049;
                border-radius: 8px;
            }
            QRadioButton::indicator:unchecked {
                background-color: white;
                border: 2px solid #cccccc;
                border-radius: 8px;
            }
            QRadioButton::indicator:unchecked:hover {
                border-color: #4CAF50;
            }
        """)
    
    def _on_sub_mode_selected(self, button):
        """Handle sub-mode selection."""
        sub_mode = button.objectName()
        if sub_mode != self._current_sub_mode:
            self._current_sub_mode = sub_mode
            self._update_status()
            self.sub_mode_changed.emit(sub_mode)
    
    def _update_status(self):
        """Update the status label based on current sub-mode."""
        sub_mode_names = {
            "manual": "Manual alignment active",
            "hybrid": "Hybrid alignment active", 
            "automatic": "Automatic alignment active"
        }
        
        sub_mode_colors = {
            "manual": "#4CAF50",
            "hybrid": "#2196F3",
            "automatic": "#FF9800"
        }
        
        status_text = sub_mode_names.get(self._current_sub_mode, "Unknown mode")
        status_color = sub_mode_colors.get(self._current_sub_mode, "#666666")
        
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(f"color: {status_color}; font-style: italic; font-size: 11px;")
    
    def set_sub_mode(self, sub_mode: str):
        """
        Programmatically set the current sub-mode.
        
        Args:
            sub_mode: Sub-mode to set ('manual', 'hybrid', 'automatic')
        """
        if sub_mode == "manual":
            self.manual_radio.setChecked(True)
        elif sub_mode == "hybrid":
            self.hybrid_radio.setChecked(True)
        elif sub_mode == "automatic":
            self.automatic_radio.setChecked(True)
        else:
            return  # Invalid sub-mode
        
        if sub_mode != self._current_sub_mode:
            self._current_sub_mode = sub_mode
            self._update_status()
            self.sub_mode_changed.emit(sub_mode)
    
    def get_current_sub_mode(self) -> str:
        """Get the currently selected sub-mode."""
        return self._current_sub_mode
    
    def set_sub_mode_enabled(self, sub_mode: str, enabled: bool):
        """
        Enable or disable a specific sub-mode.
        
        Args:
            sub_mode: Sub-mode to modify ('manual', 'hybrid', 'automatic')
            enabled: Whether the sub-mode should be enabled
        """
        if sub_mode == "manual":
            self.manual_radio.setEnabled(enabled)
        elif sub_mode == "hybrid":
            self.hybrid_radio.setEnabled(enabled)
        elif sub_mode == "automatic":
            self.automatic_radio.setEnabled(enabled)
    
    def set_sub_mode_tooltip(self, sub_mode: str, tooltip: str):
        """
        Set tooltip for a specific sub-mode button.
        
        Args:
            sub_mode: Sub-mode to modify ('manual', 'hybrid', 'automatic')
            tooltip: Tooltip text
        """
        if sub_mode == "manual":
            self.manual_radio.setToolTip(tooltip)
        elif sub_mode == "hybrid":
            self.hybrid_radio.setToolTip(tooltip)
        elif sub_mode == "automatic":
            self.automatic_radio.setToolTip(tooltip)
