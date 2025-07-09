"""
Mode Switcher - Application Mode Selection Interface

This module provides a mode switching interface with radio buttons for selecting
between different operational modes of the image analysis application.

Main Class:
- ModeSwitcher: Top bar widget with radio buttons for mode switching

Key Methods:
- _setup_ui(): Initializes UI with radio buttons and status display
- _on_mode_selected(): Handles mode selection from radio buttons
- _update_status(): Updates status label based on current mode
- set_mode(): Programmatically sets current mode
- get_current_mode(): Gets currently selected mode
- set_mode_enabled(): Enables or disables specific modes
- set_mode_tooltip(): Sets tooltip for mode buttons

Signals Emitted:
- mode_changed(str): Mode changed to new mode name

Dependencies:
- Uses: PySide6.QtWidgets, PySide6.QtCore (Qt framework)
- Uses: typing (type hints)
- Called by: UI main window
- Coordinates with: View management and application workflow

Modes:
- Manual: Manual alignment operations
- Auto: Automatic alignment operations
- Score: Scoring and analysis operations

Features:
- Radio button interface for exclusive mode selection
- Visual status indicator with color coding
- Programmatic mode control and state management
- Tooltip support for mode descriptions
- Enable/disable functionality for individual modes
"""

from typing import Callable, Optional
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QRadioButton, QButtonGroup, 
                              QLabel, QFrame)
from PySide6.QtCore import Signal


class ModeSwitcher(QWidget):
    """Top bar widget with radio buttons for switching between operation modes."""
    
    # Signals
    mode_changed = Signal(str)  # Emitted when mode changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_mode = "manual"
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Title
        title_label = QLabel("Image Analysis Mode:")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)
        
        layout.addSpacing(20)
        
        # Button group for radio buttons
        self.button_group = QButtonGroup(self)
        self.button_group.buttonClicked.connect(self._on_mode_selected)
        
        # Manual mode
        self.manual_radio = QRadioButton("Manual Alignment")
        self.manual_radio.setChecked(True)
        self.manual_radio.setObjectName("manual")
        self.button_group.addButton(self.manual_radio)
        layout.addWidget(self.manual_radio)
        
        layout.addSpacing(15)
        
        # Auto mode
        self.auto_radio = QRadioButton("Auto Alignment")
        self.auto_radio.setObjectName("auto")
        self.button_group.addButton(self.auto_radio)
        layout.addWidget(self.auto_radio)
        
        layout.addSpacing(15)
        
        # Scoring mode
        self.score_radio = QRadioButton("Scoring & Analysis")
        self.score_radio.setObjectName("score")
        self.button_group.addButton(self.score_radio)
        layout.addWidget(self.score_radio)
        
        # Stretch to push everything to the left
        layout.addStretch()
        
        # Status indicator
        self.status_label = QLabel("Manual Mode Active")
        self.status_label.setStyleSheet("color: green; font-style: italic;")
        layout.addWidget(self.status_label)
        
        # Apply styling
        self.setStyleSheet("""
            ModeSwitcher {
                background-color: #f0f0f0;
                border-bottom: 1px solid #cccccc;
            }
            QRadioButton {
                font-size: 12px;
                padding: 5px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
            QRadioButton::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #45a049;
                border-radius: 9px;
            }
            QRadioButton::indicator:unchecked {
                background-color: white;
                border: 2px solid #cccccc;
                border-radius: 9px;
            }
        """)
    
    def _on_mode_selected(self, button):
        """Handle mode selection."""
        mode = button.objectName()
        if mode != self._current_mode:
            self._current_mode = mode
            self._update_status()
            self.mode_changed.emit(mode)
    
    def _update_status(self):
        """Update the status label based on current mode."""
        mode_names = {
            "manual": "Manual Mode Active",
            "auto": "Auto Mode Active", 
            "score": "Scoring Mode Active"
        }
        
        mode_colors = {
            "manual": "green",
            "auto": "blue",
            "score": "orange"
        }
        
        status_text = mode_names.get(self._current_mode, "Unknown Mode")
        status_color = mode_colors.get(self._current_mode, "black")
        
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(f"color: {status_color}; font-style: italic;")
    
    def set_mode(self, mode: str):
        """
        Programmatically set the current mode.
        
        Args:
            mode: Mode to set ('manual', 'auto', 'score')
        """
        if mode == "manual":
            self.manual_radio.setChecked(True)
        elif mode == "auto":
            self.auto_radio.setChecked(True)
        elif mode == "score":
            self.score_radio.setChecked(True)
        else:
            return  # Invalid mode
        
        if mode != self._current_mode:
            self._current_mode = mode
            self._update_status()
            self.mode_changed.emit(mode)
    
    def get_current_mode(self) -> str:
        """Get the currently selected mode."""
        return self._current_mode
    
    def set_mode_enabled(self, mode: str, enabled: bool):
        """
        Enable or disable a specific mode.
        
        Args:
            mode: Mode to modify ('manual', 'auto', 'score')
            enabled: Whether the mode should be enabled
        """
        if mode == "manual":
            self.manual_radio.setEnabled(enabled)
        elif mode == "auto":
            self.auto_radio.setEnabled(enabled)
        elif mode == "score":
            self.score_radio.setEnabled(enabled)
    
    def set_mode_tooltip(self, mode: str, tooltip: str):
        """
        Set tooltip for a specific mode button.
        
        Args:
            mode: Mode to modify ('manual', 'auto', 'score')
            tooltip: Tooltip text
        """
        if mode == "manual":
            self.manual_radio.setToolTip(tooltip)
        elif mode == "auto":
            self.auto_radio.setToolTip(tooltip)
        elif mode == "score":
            self.score_radio.setToolTip(tooltip)
