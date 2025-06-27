"""File utilities for path resolution and results directory setup."""

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
    data_dir = get_project_root() / "Data"
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
    results_dir = get_project_root() / "Results"
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
