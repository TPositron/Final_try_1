"""
Settings Dialog - Application Settings Management

This module provides a settings dialog for managing application preferences
and configuration options.

Main Class:
- SettingsDialog: Qt dialog for application settings

Global Functions:
- load_app_settings(): Loads settings from file
- save_app_settings(): Saves settings to file

Key Methods:
- get_settings(): Returns current settings values

Dependencies:
- Uses: PySide6.QtWidgets (Qt dialog components)
- Uses: json, os (file operations)
- Called by: ui/main_window.py (settings menu)

Settings:
- Default Directory: Default file directory
- Dark Theme: Enable/disable dark theme

Features:
- Persistent settings storage
- User-friendly dialog interface
- JSON-based configuration file
- Cross-platform settings location
"""

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QDialogButtonBox, QFormLayout, 
                               QLineEdit, QCheckBox, QPushButton, QColorDialog, QLabel,
                               QHBoxLayout, QSpinBox, QGroupBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
import json
import os

SETTINGS_PATH = os.path.expanduser("~/.sem_gds_tool_settings.json")

class SettingsDialog(QDialog):
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("Application Settings")
        self.settings = settings or {}
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # General settings
        general_group = QGroupBox("General Settings")
        general_layout = QFormLayout(general_group)
        
        self.default_dir_edit = QLineEdit(self.settings.get('default_dir', ''))
        general_layout.addRow("Default Directory:", self.default_dir_edit)
        
        self.theme_checkbox = QCheckBox("Enable Dark Theme")
        self.theme_checkbox.setChecked(self.settings.get('dark_theme', False))
        general_layout.addRow(self.theme_checkbox)
        
        layout.addWidget(general_group)
        
        # GDS Display settings
        gds_group = QGroupBox("GDS Display Settings")
        gds_layout = QFormLayout(gds_group)
        
        # Background color setting
        bg_layout = QHBoxLayout()
        self.bg_color_btn = QPushButton("Choose Color")
        self.bg_color_label = QLabel("Transparent")
        self.bg_alpha_spin = QSpinBox()
        self.bg_alpha_spin.setRange(0, 255)
        self.bg_alpha_spin.setValue(self.settings.get('gds_bg_alpha', 0))
        
        # Load background color
        bg_color = self.settings.get('gds_bg_color', [0, 0, 0])
        self.bg_color = QColor(bg_color[0], bg_color[1], bg_color[2])
        self.bg_color_btn.clicked.connect(self.choose_bg_color)
        self.update_bg_color_display()
        
        bg_layout.addWidget(self.bg_color_btn)
        bg_layout.addWidget(self.bg_color_label)
        bg_layout.addWidget(QLabel("Alpha:"))
        bg_layout.addWidget(self.bg_alpha_spin)
        gds_layout.addRow("Background Color:", bg_layout)
        
        # Structure color setting
        struct_layout = QHBoxLayout()
        self.struct_color_btn = QPushButton("Choose Color")
        self.struct_color_label = QLabel("Black")
        self.struct_alpha_spin = QSpinBox()
        self.struct_alpha_spin.setRange(0, 255)
        self.struct_alpha_spin.setValue(self.settings.get('gds_struct_alpha', 255))
        
        # Load structure color
        struct_color = self.settings.get('gds_struct_color', [0, 0, 0])
        self.struct_color = QColor(struct_color[0], struct_color[1], struct_color[2])
        self.struct_color_btn.clicked.connect(self.choose_struct_color)
        self.update_struct_color_display()
        
        struct_layout.addWidget(self.struct_color_btn)
        struct_layout.addWidget(self.struct_color_label)
        struct_layout.addWidget(QLabel("Alpha:"))
        struct_layout.addWidget(self.struct_alpha_spin)
        gds_layout.addRow("Structure Color:", struct_layout)
        
        # Reset to defaults button
        reset_btn = QPushButton("Reset GDS Colors to Default")
        reset_btn.clicked.connect(self.reset_gds_colors)
        gds_layout.addRow(reset_btn)
        
        layout.addWidget(gds_group)
        
        # UI Theme settings
        ui_group = QGroupBox("UI Theme Settings")
        ui_layout = QFormLayout(ui_group)
        
        # Panel background color
        panel_layout = QHBoxLayout()
        self.panel_color_btn = QPushButton("Choose Color")
        self.panel_color_label = QLabel("Dark Gray")
        panel_color = self.settings.get('ui_panel_color', [43, 43, 43])
        self.panel_color = QColor(panel_color[0], panel_color[1], panel_color[2])
        self.panel_color_btn.clicked.connect(self.choose_panel_color)
        self.update_panel_color_display()
        panel_layout.addWidget(self.panel_color_btn)
        panel_layout.addWidget(self.panel_color_label)
        ui_layout.addRow("Panel Background:", panel_layout)
        
        # Button color
        button_layout = QHBoxLayout()
        self.button_color_btn = QPushButton("Choose Color")
        self.button_color_label = QLabel("Blue")
        button_color = self.settings.get('ui_button_color', [70, 130, 180])
        self.button_color = QColor(button_color[0], button_color[1], button_color[2])
        self.button_color_btn.clicked.connect(self.choose_button_color)
        self.update_button_color_display()
        button_layout.addWidget(self.button_color_btn)
        button_layout.addWidget(self.button_color_label)
        ui_layout.addRow("Button Color:", button_layout)
        
        # Text color
        text_layout = QHBoxLayout()
        self.text_color_btn = QPushButton("Choose Color")
        self.text_color_label = QLabel("White")
        text_color = self.settings.get('ui_text_color', [255, 255, 255])
        self.text_color = QColor(text_color[0], text_color[1], text_color[2])
        self.text_color_btn.clicked.connect(self.choose_text_color)
        self.update_text_color_display()
        text_layout.addWidget(self.text_color_btn)
        text_layout.addWidget(self.text_color_label)
        ui_layout.addRow("Text Color:", text_layout)
        
        # Reset UI theme button
        reset_ui_btn = QPushButton("Reset UI Theme to Default")
        reset_ui_btn.clicked.connect(self.reset_ui_theme)
        ui_layout.addRow(reset_ui_btn)
        
        layout.addWidget(ui_group)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    def choose_bg_color(self):
        """Open color dialog for background color."""
        color = QColorDialog.getColor(self.bg_color, self, "Choose Background Color")
        if color.isValid():
            self.bg_color = color
            self.update_bg_color_display()
    
    def choose_struct_color(self):
        """Open color dialog for structure color."""
        color = QColorDialog.getColor(self.struct_color, self, "Choose Structure Color")
        if color.isValid():
            self.struct_color = color
            self.update_struct_color_display()
    
    def update_bg_color_display(self):
        """Update background color display."""
        color_name = self.bg_color.name()
        self.bg_color_label.setText(color_name)
        self.bg_color_label.setStyleSheet(f"background-color: {color_name}; padding: 4px; border: 1px solid gray;")
    
    def update_struct_color_display(self):
        """Update structure color display."""
        color_name = self.struct_color.name()
        self.struct_color_label.setText(color_name)
        self.struct_color_label.setStyleSheet(f"background-color: {color_name}; padding: 4px; border: 1px solid gray;")
    
    def choose_panel_color(self):
        """Open color dialog for panel color."""
        color = QColorDialog.getColor(self.panel_color, self, "Choose Panel Color")
        if color.isValid():
            self.panel_color = color
            self.update_panel_color_display()
    
    def choose_button_color(self):
        """Open color dialog for button color."""
        color = QColorDialog.getColor(self.button_color, self, "Choose Button Color")
        if color.isValid():
            self.button_color = color
            self.update_button_color_display()
    
    def choose_text_color(self):
        """Open color dialog for text color."""
        color = QColorDialog.getColor(self.text_color, self, "Choose Text Color")
        if color.isValid():
            self.text_color = color
            self.update_text_color_display()
    
    def update_panel_color_display(self):
        """Update panel color display."""
        color_name = self.panel_color.name()
        self.panel_color_label.setText(color_name)
        self.panel_color_label.setStyleSheet(f"background-color: {color_name}; padding: 4px; border: 1px solid gray;")
    
    def update_button_color_display(self):
        """Update button color display."""
        color_name = self.button_color.name()
        self.button_color_label.setText(color_name)
        self.button_color_label.setStyleSheet(f"background-color: {color_name}; padding: 4px; border: 1px solid gray;")
    
    def update_text_color_display(self):
        """Update text color display."""
        color_name = self.text_color.name()
        self.text_color_label.setText(color_name)
        self.text_color_label.setStyleSheet(f"background-color: {color_name}; padding: 4px; border: 1px solid gray;")
    
    def reset_gds_colors(self):
        """Reset GDS colors to default values."""
        self.bg_color = QColor(0, 0, 0)  # Black background
        self.bg_alpha_spin.setValue(0)   # Transparent
        self.struct_color = QColor(0, 0, 0)  # Black structure
        self.struct_alpha_spin.setValue(255)  # Opaque
        self.update_bg_color_display()
        self.update_struct_color_display()
    
    def reset_ui_theme(self):
        """Reset UI theme to default values."""
        self.panel_color = QColor(43, 43, 43)  # Dark gray
        self.button_color = QColor(70, 130, 180)  # Steel blue
        self.text_color = QColor(255, 255, 255)  # White
        self.update_panel_color_display()
        self.update_button_color_display()
        self.update_text_color_display()
    
    def get_settings(self):
        return {
            'default_dir': self.default_dir_edit.text(),
            'dark_theme': self.theme_checkbox.isChecked(),
            'gds_bg_color': [self.bg_color.red(), self.bg_color.green(), self.bg_color.blue()],
            'gds_bg_alpha': self.bg_alpha_spin.value(),
            'gds_struct_color': [self.struct_color.red(), self.struct_color.green(), self.struct_color.blue()],
            'gds_struct_alpha': self.struct_alpha_spin.value(),
            'ui_panel_color': [self.panel_color.red(), self.panel_color.green(), self.panel_color.blue()],
            'ui_button_color': [self.button_color.red(), self.button_color.green(), self.button_color.blue()],
            'ui_text_color': [self.text_color.red(), self.text_color.green(), self.text_color.blue()]
        }

def load_app_settings():
    """Load application settings with defaults."""
    defaults = {
        'default_dir': '',
        'dark_theme': False,
        'gds_bg_color': [0, 0, 0],      # Black background
        'gds_bg_alpha': 0,              # Transparent
        'gds_struct_color': [0, 0, 0],  # Black structure
        'gds_struct_alpha': 255,        # Opaque
        'ui_panel_color': [43, 43, 43], # Dark gray panels
        'ui_button_color': [70, 130, 180], # Steel blue buttons
        'ui_text_color': [255, 255, 255]   # White text
    }
    
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                # Merge with defaults to ensure all keys exist
                for key, value in defaults.items():
                    if key not in settings:
                        settings[key] = value
                return settings
        except Exception as e:
            print(f"Error loading settings: {e}")
    
    return defaults

def save_app_settings(settings):
    with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2)
