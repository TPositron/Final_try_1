"""
Simple configuration management for SEM/GDS Alignment Tool.

Basic configuration loading and saving functionality.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Simple configuration class
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
    
"""
Simple configuration management for SEM/GDS Alignment Tool.

Basic configuration loading and saving functionality.
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
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., 'ui.window.width')
            default: Default value to return if key is not found
            
        Returns:
            The configuration value or default
        """
        if not self._is_loaded:
            self.load_configuration()
        
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.debug(f"Configuration key '{key_path}' not found, returning default: {default}")
            return default
    
    def set(self, key_path: str, value: Any, persist: bool = False) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value
            value: Value to set
            persist: Whether to save the change to user config file
        """
        if not self._is_loaded:
            self.load_configuration()
        
        keys = key_path.split('.')
        config_ref = self._config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # Set the final value
        config_ref[keys[-1]] = value
        logger.debug(f"Configuration key '{key_path}' set to: {value}")
        
        if persist:
            self.save_user_config_override(key_path, value)
    
    def get_full_config(self) -> Dict[str, Any]:
        """
        Get the complete merged configuration.
        
        Returns:
            The full configuration dictionary
        """
        if not self._is_loaded:
            self.load_configuration()
        return deepcopy(self._config)
    
    def save_user_config_override(self, key_path: str, value: Any) -> None:
        """
        Save a configuration override to the user config file.
        
        Args:
            key_path: Dot-separated path to the configuration value
            value: Value to save
        """
        try:
            # Load existing user config or create new
            if self.user_config_file.exists():
                user_config = self._load_json_file(self.user_config_file, required=False) or {}
            else:
                user_config = {
                    "_comment": "User-specific configuration overrides",
                    "_description": "This file allows users to override default settings without modifying the main config.json",
                    "_note": "Only include settings you want to change from defaults. This file is optional and will be ignored by git."
                }
            
            # Navigate and set the value
            keys = key_path.split('.')
            config_ref = user_config
            
            for key in keys[:-1]:
                if key not in config_ref:
                    config_ref[key] = {}
                config_ref = config_ref[key]
            
            config_ref[keys[-1]] = value
            
            # Save to file
            self._save_json_file(self.user_config_file, user_config)
            logger.info(f"User configuration override saved: {key_path} = {value}")
            
        except Exception as e:
            logger.error(f"Failed to save user configuration override: {e}")
            raise ConfigurationError(f"Failed to save user configuration: {e}") from e
    
    def save_session_state(self, session_data: Dict[str, Any]) -> None:
        """
        Save current session state for restoration.
        
        Args:
            session_data: Session data to save
        """
        try:
            session_config = {
                "timestamp": session_data.get("timestamp"),
                "window_geometry": session_data.get("window_geometry"),
                "last_opened_files": session_data.get("last_opened_files", []),
                "recent_projects": session_data.get("recent_projects", []),
                "ui_state": session_data.get("ui_state", {})
            }
            
            self._save_json_file(self.session_file, session_config)
            logger.debug("Session state saved successfully")
            
        except Exception as e:
            logger.warning(f"Failed to save session state: {e}")
    
    def load_session_state(self) -> Dict[str, Any]:
        """
        Load saved session state.
        
        Returns:
            Saved session data or empty dict if none exists
        """
        try:
            if self.session_file.exists():
                session_data = self._load_json_file(self.session_file, required=False)
                logger.debug("Session state loaded successfully")
                return session_data or {}
            else:
                logger.debug("No session file found")
                return {}
                
        except Exception as e:
            logger.warning(f"Failed to load session state: {e}")
            return {}
    
    def validate_configuration(self) -> bool:
        """
        Validate the current configuration against the schema.
        
        Returns:
            True if valid, False otherwise
        """
        if not self._schema:
            logger.warning("No schema loaded, skipping validation")
            return True
        
        try:
            self._validate_configuration()
            return True
        except ConfigurationError:
            return False
    
    def _load_json_file(self, file_path: Path, required: bool = True) -> Optional[Dict[str, Any]]:
        """Load and parse a JSON file."""
        try:
            if not file_path.exists():
                if required:
                    raise FileNotFoundError(f"Required configuration file not found: {file_path}")
                return None
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                logger.debug(f"Loaded JSON file: {file_path}")
                return content
                
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in {file_path}: {e}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load {file_path}: {e}"
            logger.error(error_msg)
            if required:
                raise ConfigurationError(error_msg) from e
            return None
    
    def _save_json_file(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Save data to a JSON file."""
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            error_msg = f"Failed to save {file_path}: {e}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
    
    def _merge_configurations(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        merged = deepcopy(base)
        
        for key, value in override.items():
            # Skip comment keys
            if key.startswith('_'):
                continue
                
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = self._merge_configurations(merged[key], value)
            else:
                # Override the value
                merged[key] = deepcopy(value)
                
        return merged
    
    def _apply_environment_overrides(self) -> None:
        """Apply configuration overrides from environment variables."""
        # Define environment variable mappings
        env_mappings = {
            'SEM_GDS_DEBUG': 'application.debug_mode',
            'SEM_GDS_LOG_LEVEL': 'logging.level',
            'SEM_GDS_MAX_MEMORY': 'performance.max_memory_usage_mb',
            'SEM_GDS_MAX_THREADS': 'performance.max_worker_threads',
            'SEM_GDS_DATA_DIR': 'paths.data_directory',
            'SEM_GDS_RESULTS_DIR': 'paths.results_directory'
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Try to convert to appropriate type
                try:
                    # Try boolean
                    if env_value.lower() in ('true', 'false'):
                        env_value = env_value.lower() == 'true'
                    # Try integer
                    elif env_value.isdigit():
                        env_value = int(env_value)
                    # Try float
                    elif '.' in env_value and env_value.replace('.', '').isdigit():
                        env_value = float(env_value)
                    # Keep as string
                    
                    self.set(config_path, env_value)
                    logger.debug(f"Applied environment override: {env_var} -> {config_path} = {env_value}")
                    
                except Exception as e:
                    logger.warning(f"Failed to apply environment override {env_var}: {e}")
    
    def _resolve_paths(self) -> None:
        """Resolve relative paths to absolute paths based on project root."""
        # Get project root (assuming config is in project_root/config)
        project_root = self.config_dir.parent
        
        # Paths that should be resolved
        path_keys = [
            'paths.data_directory',
            'paths.sem_directory', 
            'paths.gds_directory',
            'paths.results_directory',
            'paths.extracted_structures_directory',
            'paths.temp_directory',
            'paths.logs_directory',
            'paths.config_directory',
            'paths.user_config_file',
            'paths.last_session_file'
        ]
        
        for path_key in path_keys:
            current_path = self.get(path_key)
            if current_path and not os.path.isabs(current_path):
                absolute_path = str(project_root / current_path)
                self.set(path_key, absolute_path)
                logger.debug(f"Resolved path {path_key}: {current_path} -> {absolute_path}")
    
    def _validate_configuration(self) -> None:
        """Validate configuration against JSON schema."""
        try:
            validate(instance=self._config, schema=self._schema)
            logger.debug("Configuration validation passed")
            
        except ValidationError as e:
            # Provide more detailed error information
            error_path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
            error_msg = f"Configuration validation failed at {error_path}: {e.message}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager(config_dir: Optional[Union[str, Path]] = None) -> ConfigurationManager:
    """
    Get the global configuration manager instance.
    
    Args:
        config_dir: Configuration directory path (only used on first call)
        
    Returns:
        ConfigurationManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigurationManager(config_dir)
    
    return _config_manager


def get_config(key_path: str = None, default: Any = None) -> Any:
    """
    Convenience function to get configuration values.
    
    Args:
        key_path: Dot-separated path to configuration value. If None, returns full config.
        default: Default value if key not found
        
    Returns:
        Configuration value or full config
    """
    config_manager = get_config_manager()
    
    if key_path is None:
        return config_manager.get_full_config()
    
    return config_manager.get(key_path, default)


def set_config(key_path: str, value: Any, persist: bool = False) -> None:
    """
    Convenience function to set configuration values.
    
    Args:
        key_path: Dot-separated path to configuration value
        value: Value to set
        persist: Whether to save to user config file
    """
    config_manager = get_config_manager()
    config_manager.set(key_path, value, persist)


def load_config(validate: bool = True) -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Args:
        validate: Whether to validate configuration
        
    Returns:
        Loaded configuration
    """
    config_manager = get_config_manager()
    return config_manager.load_configuration(validate)


# Initialize logging for this module
if __name__ != "__main__":
    # Only set up basic logging if we're imported, not if run directly
    logging.basicConfig(level=logging.INFO)
