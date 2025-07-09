"""
File Utilities - Path Resolution and Directory Management

This module provides utility functions for file and directory operations
used throughout the SEM/GDS alignment tool. It handles path resolution,
directory creation, and file management operations with consistent
project structure support.

Main Functions:
- ensure_directory(): Creates directories if they don't exist
- get_project_root(): Returns the project root directory path
- get_data_path(): Returns path to Data directory with optional subpath
- get_results_path(): Returns path to Results directory with optional subpath
- get_extracted_structures_path(): Returns path to Extracted_Structures directory
- setup_results_directories(): Creates standard results directory structure
- get_unique_filename(): Generates unique filenames by appending numbers
- list_files_by_extension(): Lists files with specific extensions in directories

Dependencies:
- Uses: os (operating system interface), pathlib.Path (path operations)
- Uses: typing (type hints for Union, Optional)
- Used by: services/file_service.py (file operations)
- Used by: core/models (data file paths)
- Used by: ui/file_operations.py (file dialogs and operations)
- Used by: All modules requiring consistent path resolution

Directory Structure:
- Data/: Input data files (SEM images, GDS files)
- Results/: Output files organized by processing type
  - Aligned/manual/: Manual alignment results
  - Aligned/auto/: Automatic alignment results
  - SEM_Filters/manual/: Manual filter results
  - SEM_Filters/auto/: Automatic filter results
  - Scoring/overlays/: Overlay images
  - Scoring/charts/: Score visualization charts
  - Scoring/reports/: Processing reports
- Extracted_Structures/: Extracted GDS structure data

Key Features:
- Automatic directory creation with parent directory support
- Consistent path resolution relative to project root
- Cross-platform path handling using pathlib
- Unique filename generation to prevent overwrites
- File listing with extension filtering and recursive search
- Standard results directory structure setup
- Type-safe path operations with Union[str, Path] support

Path Resolution:
- Project root determined relative to this module's location
- All paths resolved relative to project root for consistency
- Support for both absolute and relative path operations
- Automatic handling of path separators across platforms
"""

import os
from pathlib import Path
from typing import Union, Optional


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Get the project root directory."""
    # Assume we're in src/core/utils, so go up 3 levels
    return Path(__file__).parent.parent.parent.parent


def get_data_path(subpath: str = "") -> Path:
    """
    Get path to data directory with optional subpath.
    
    Args:
        subpath: Optional subdirectory or file within Data/
        
    Returns:
        Path to the data location
    """
    data_dir = Path(__file__).parent.parent.parent / "Data"
    if subpath:
        return data_dir / subpath
    return data_dir


def get_results_path(subpath: str = "") -> Path:
    """
    Get path to results directory with optional subpath.
    
    Args:
        subpath: Optional subdirectory or file within Results/
        
    Returns:
        Path to the results location
    """
    results_dir = Path(__file__).parent.parent.parent / "Results"
    if subpath:
        return results_dir / subpath
    return results_dir


def get_extracted_structures_path(subpath: str = "") -> Path:
    """
    Get path to extracted structures directory with optional subpath.
    
    Args:
        subpath: Optional subdirectory or file within Extracted_Structures/
        
    Returns:
        Path to the extracted structures location
    """
    extracted_dir = get_project_root() / "Extracted_Structures"
    if subpath:
        return extracted_dir / subpath
    return extracted_dir


def setup_results_directories() -> None:
    """Set up the standard results directory structure."""
    # Main results subdirectories
    ensure_directory(get_results_path("Aligned/manual"))
    ensure_directory(get_results_path("Aligned/auto"))
    ensure_directory(get_results_path("SEM_Filters/manual"))
    ensure_directory(get_results_path("SEM_Filters/auto"))
    ensure_directory(get_results_path("Scoring/overlays"))
    ensure_directory(get_results_path("Scoring/charts"))
    ensure_directory(get_results_path("Scoring/reports"))


def get_unique_filename(base_path: Union[str, Path], extension: str = "") -> Path:
    """
    Get a unique filename by appending a number if the file already exists.
    
    Args:
        base_path: Base path for the file (without extension)
        extension: File extension (with or without leading dot)
        
    Returns:
        Unique file path
    """
    base_path = Path(base_path)
    
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    
    # If no extension provided, try to get it from base_path
    if not extension and base_path.suffix:
        extension = base_path.suffix
        base_path = base_path.with_suffix('')
    
    counter = 1
    while True:
        if counter == 1:
            candidate = base_path.with_suffix(extension)
        else:
            candidate = base_path.with_name(f"{base_path.name}_{counter}").with_suffix(extension)
        
        if not candidate.exists():
            return candidate
        
        counter += 1


def list_files_by_extension(directory: Union[str, Path], extension: str, recursive: bool = False) -> list[Path]:
    """
    List all files with a specific extension in a directory.
    
    Args:
        directory: Directory to search
        extension: File extension (with or without leading dot)
        recursive: Whether to search subdirectories
        
    Returns:
        List of file paths
    """
    directory = Path(directory)
    
    if not extension.startswith('.'):
        extension = '.' + extension
    
    if recursive:
        pattern = f"**/*{extension}"
    else:
        pattern = f"*{extension}"
    
    return list(directory.glob(pattern)) if directory.exists() else []
