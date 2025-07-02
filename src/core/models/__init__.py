"""
Core data models for the SEM/GDS image analysis application.

Basic data structures for representing:
- SEM (Scanning Electron Microscope) images with metadata
- GDS (GDSII) layout files with gdstk integration
- Basic structure definitions

Classes:
    SemImage: Basic SEM image class with numpy array storage
    InitialGdsModel: Basic GDS file loader using gdstk
"""

from .simple_sem_image import (
    SemImage, 
    load_sem_image, 
    create_sem_image, 
    save_sem_image_json,
    export_sem_image_png
)
from .simple_initial_gds_model import (
    InitialGdsModel,
    load_gds_file,
    validate_gds_file
)
from .simple_aligned_gds_model import (
    AlignedGdsModel,
    create_aligned_model
)
from .simple_gds_extraction import (
    get_structure_info,
    calculate_structure_bounds,
    enumerate_structure_layers,
    validate_structure,
    create_crop_regions,
    extract_frame,
    extract_frame_to_aligned_model,
    extract_multiple_frames,
    batch_extract_structures,
    batch_extract_all_predefined_structures,
    create_batch_extraction_report,
    validate_frame_bounds,
    filter_structure_layers,
    create_extraction_metadata
)

__all__ = [
    'SemImage',
    'load_sem_image', 
    'create_sem_image', 
    'save_sem_image_json',
    'export_sem_image_png',
    'InitialGdsModel',
    'load_gds_file',
    'validate_gds_file',
    'AlignedGdsModel',
    'create_aligned_model',
    'get_structure_info',
    'calculate_structure_bounds',
    'enumerate_structure_layers',
    'validate_structure',
    'create_crop_regions',
    'extract_frame',
    'extract_frame_to_aligned_model',
    'extract_multiple_frames',
    'batch_extract_structures',
    'batch_extract_all_predefined_structures',
    'create_batch_extraction_report',
    'validate_frame_bounds',
    'filter_structure_layers',
    'create_extraction_metadata'
]
