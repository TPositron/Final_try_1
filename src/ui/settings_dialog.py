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

from PySide6.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QFormLayout, QLineEdit, QCheckBox
import json
import os

SETTINGS_PATH = os.path.expanduser("~/.sem_gds_tool_settings.json")

class SettingsDialog(QDialog):
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("Application Settings")
        self.settings = settings or {}
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.default_dir_edit = QLineEdit(self.settings.get('default_dir', ''))
        form.addRow("Default Directory:", self.default_dir_edit)
        self.theme_checkbox = QCheckBox("Enable Dark Theme")
        self.theme_checkbox.setChecked(self.settings.get('dark_theme', False))
        form.addRow(self.theme_checkbox)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    def get_settings(self):
        return {
            'default_dir': self.default_dir_edit.text(),
            'dark_theme': self.theme_checkbox.isChecked()
        }

def load_app_settings():
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_app_settings(settings):
    with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2)
