"""
Project structure validation utilities.

Simple validation that all required directories exist.
"""

from pathlib import Path
from typing import List, Dict
from .file_utils import get_project_root


def get_required_directories() -> List[str]:
    """
    Get list of required project directories.
    
    Returns:
        List of required directory paths
    """
    return [
        "Data",
        "Data/SEM", 
        "Data/GDS",
        "Results",
        "Results/Aligned",
        "Results/SEM_Filters",
        "Results/Scoring",
        "config",
        "logs",
        "src",
        "src/core",
        "src/services",
        "src/ui"
    ]


def validate_project_structure() -> Dict[str, bool]:
    """
    Validate that all required directories exist.
    
    Returns:
        Dictionary mapping directory paths to existence status
    """
    project_root = get_project_root()
    required_dirs = get_required_directories()
    
    validation_results = {}
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        validation_results[dir_path] = full_path.exists() and full_path.is_dir()
    
    return validation_results


def check_missing_directories() -> List[str]:
    """
    Get list of missing required directories.
    
    Returns:
        List of missing directory paths
    """
    validation_results = validate_project_structure()
    return [dir_path for dir_path, exists in validation_results.items() if not exists]


def create_missing_directories() -> None:
    """Create any missing required directories."""
    project_root = get_project_root()
    missing_dirs = check_missing_directories()
    
    created_count = 0
    for dir_path in missing_dirs:
        try:
            full_path = project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
            created_count += 1
        except Exception as e:
            print(f"Error creating directory {dir_path}: {e}")
    
    if created_count > 0:
        print(f"Created {created_count} missing directories")
    else:
        print("All required directories already exist")


def validate_config_files() -> Dict[str, bool]:
    """
    Validate that required configuration files exist.
    
    Returns:
        Dictionary mapping config files to existence status
    """
    project_root = get_project_root()
    config_files = [
        "config/config.json",
        "requirements.txt",
        "README.md"
    ]
    
    validation_results = {}
    for file_path in config_files:
        full_path = project_root / file_path
        validation_results[file_path] = full_path.exists() and full_path.is_file()
    
    return validation_results


def run_full_validation() -> bool:
    """
    Run complete project structure validation.
    
    Returns:
        True if all validation checks pass
    """
    print("Validating project structure...")
    
    # Check directories
    missing_dirs = check_missing_directories()
    if missing_dirs:
        print(f"Missing directories: {missing_dirs}")
        print("Use create_missing_directories() to create them")
        return False
    else:
        print("✓ All required directories exist")
    
    # Check config files
    config_validation = validate_config_files()
    missing_configs = [f for f, exists in config_validation.items() if not exists]
    
    if missing_configs:
        print(f"Missing config files: {missing_configs}")
        return False
    else:
        print("✓ All required config files exist")
    
    print("✓ Project structure validation passed")
    return True


def print_validation_report() -> None:
    """Print a detailed validation report."""
    project_root = get_project_root()
    
    print(f"\nProject Structure Validation Report")
    print(f"Project Root: {project_root}")
    print("=" * 50)
    
    # Directory validation
    print("\nDirectories:")
    dir_validation = validate_project_structure()
    for dir_path, exists in dir_validation.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path}")
    
    # Config file validation
    print("\nConfiguration Files:")
    config_validation = validate_config_files()
    for file_path, exists in config_validation.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
    
    # Summary
    total_dirs = len(dir_validation)
    existing_dirs = sum(dir_validation.values())
    total_configs = len(config_validation)
    existing_configs = sum(config_validation.values())
    
    print(f"\nSummary:")
    print(f"  Directories: {existing_dirs}/{total_dirs}")
    print(f"  Config files: {existing_configs}/{total_configs}")
    
    if existing_dirs == total_dirs and existing_configs == total_configs:
        print("  Status: ✓ All checks passed")
    else:
        print("  Status: ✗ Some checks failed")
