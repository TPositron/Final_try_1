"""
Configuration Manager - Application Configuration and Settings Management

This script provides comprehensive configuration management for the SEM/GDS alignment tool.
It handles JSON-based configuration files, user overrides, session state management, and
environment variable integration for flexible deployment and customization.

Main Classes:
- Config: Simple configuration manager for basic JSON operations

Key Methods:
- load(): Loads configuration from JSON file with fallback to defaults
- save(): Saves current configuration to JSON file
- get(): Gets configuration value using dot notation (e.g., 'ui.window.width')
- set(): Sets configuration value using dot notation
- _get_defaults(): Returns default configuration values

Global Functions:
- load_config(): Loads configuration using global instance
- get_config(): Gets configuration value using global instance
- set_config(): Sets configuration value using global instance
- save_config(): Saves configuration using global instance

Dependencies:
- json: Configuration file parsing and serialization
- os: Environment variable access and file system operations
- pathlib.Path: Cross-platform path handling
- typing: Type hints for Dict, Any, Optional

Features:
- JSON-based configuration with hierarchical structure
- Dot notation for nested configuration access (e.g., 'window.width')
- Default configuration values with automatic fallback
- Graceful error handling with fallback to defaults
- Global configuration instance for application-wide access
- Directory creation for configuration files
- Simple and lightweight configuration management
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Simple configuration manager."""
    
    def __init__(self, config_file: str = "config/config.json"):
        self.config_file = Path(config_file)
        self._config: Dict[str, Any] = {}
        
    def load(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self._config = json.load(f)
            else:
                self._config = self._get_defaults()
            return self._config
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_defaults()
    
    def save(self) -> None:
        """Save configuration to JSON file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "window": {"width": 1400, "height": 900},
            "paths": {
                "data": "Data",
                "results": "Results", 
                "logs": "logs"
            },
            "logging": {"level": "INFO"}
        }


# Global config instance
_config = Config()

def load_config() -> Dict[str, Any]:
    """Load configuration."""
    return _config.load()

def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value."""
    return _config.get(key, default)

def set_config(key: str, value: Any) -> None:
    """Set configuration value."""
    _config.set(key, value)

def save_config() -> None:
    """Save configuration."""
    _config.save()