"""
Simple Configuration Management - Basic Config File Handling

This module provides a lightweight configuration management system for the
SEM/GDS alignment tool. It handles JSON-based configuration files with
dot-notation access and automatic default value fallback.

Main Class:
- Config: Simple configuration manager with JSON file persistence

Key Methods:
- load(): Loads configuration from JSON file or returns defaults
- save(): Saves current configuration to JSON file
- get(): Retrieves configuration values using dot notation
- set(): Sets configuration values using dot notation
- _get_defaults(): Returns default configuration structure

Global Functions:
- load_config(): Loads configuration using global instance
- get_config(): Gets configuration value using global instance
- set_config(): Sets configuration value using global instance
- save_config(): Saves configuration using global instance

Dependencies:
- Uses: json (configuration file parsing), pathlib.Path (file operations)
- Uses: typing (type hints for Dict, Any)
- Used by: All modules requiring configuration access
- Alternative to: core/utils/config_manager.py (more advanced configuration)

Configuration Structure:
- window: UI window settings (width, height)
- paths: Directory paths (data, results, logs)
- logging: Logging configuration (level)

Features:
- JSON-based configuration persistence
- Dot notation for nested configuration access (e.g., 'window.width')
- Automatic default value fallback if config file missing
- Simple error handling with console output
- Automatic directory creation for config file
- Global configuration instance for easy access

Usage:
- load_config(): Initialize configuration system
- get_config('window.width', 1200): Get value with default
- set_config('window.width', 1600): Set configuration value
- save_config(): Persist changes to file

Default Configuration:
- Window: 1400x900 pixels
- Data directory: 'Data'
- Results directory: 'Results'
- Logs directory: 'logs'
- Logging level: 'INFO'
"""

import json
from pathlib import Path
from typing import Dict, Any

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
