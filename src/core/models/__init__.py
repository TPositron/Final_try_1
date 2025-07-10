"""
Core data models for the SEM/GDS image analysis application.

Basic data structures for representing:
- SEM (Scanning Electron Microscope) images with metadata
- GDS (GDSII) layout files with gdstk integration
- Basic structure definitions

Classes:
    SemImage: Basic SEM image class with numpy array storage
"""

from .simple_sem_image import (
    SemImage, 
    load_sem_image, 
    create_sem_image, 
    save_sem_image_json,
    export_sem_image_png
)