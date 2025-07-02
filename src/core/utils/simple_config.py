"""
Simple configuration management for SEM/GDS Alignment Tool.
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
