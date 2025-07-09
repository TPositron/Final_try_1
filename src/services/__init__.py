"""
Services Package - Business Logic Layer

This package implements the business logic layer of the SEM/GDS alignment application.
Services act as intermediaries between the UI layer and the core data models,
handling complex operations and coordinating between different components.

Architecture Pattern:
UI Controllers -> Services -> Core Models

Service Categories:
1. File Services: Handle data loading, saving, and file management
   - file_service.py: Core file operations
   - file_loading_service.py: Specialized loading operations
   - file_listing_service.py: File discovery and listing

2. Processing Services: Handle image processing and transformations
   - simple_image_processing_service.py: Image filtering and enhancement
   - transformation_service.py: GDS transformation operations
   - overlay.py: Image overlay and composition

3. Alignment Services: Handle alignment algorithms and operations
   - simple_alignment_service.py: Manual and automatic alignment
   - manual_alignment_service.py: Manual alignment operations
   - auto_alignment_service.py: Automatic alignment algorithms

4. Analysis Services: Handle scoring and analysis
   - simple_scoring_service.py: Image comparison and scoring metrics

5. Workflow Services: High-level operation coordination
   - workflow_service.py: Orchestrates complex multi-step operations
   - new_gds_service.py: Modern GDS handling service

Common Patterns:
- Qt signals/slots for asynchronous communication
- Error handling and progress reporting
- State management and caching
- Service composition and dependency injection

Dependencies:
- Uses: core package (models and utilities)
- Called by: ui package (controllers and panels)
- Coordinates: Multiple core modules for complex operations

Data Flow:
1. UI triggers service operations
2. Services coordinate core model operations
3. Services emit signals for UI updates
4. Services handle errors and state management
"""

from .base_service import BaseService
from .file_listing_service import FileListingService
from .file_loading_service import FileLoadingService
from .filters.filter_service import FilterService
from .unified_transformation_service import UnifiedTransformationService
from .manual_alignment_service import ManualAlignmentService
from .auto_alignment_service import AutoAlignmentService
from .scoring.pixel_score_service import PixelScoreService
from .scoring.ssim_score_service import SSIMScoreService
from .scoring.iou_score_service import IOUScoreService
# Updated to use simple scoring service (Step 8)
from .simple_scoring_service import ScoringService
from .simple_alignment_service import AlignmentService
from .simple_image_processing_service import ImageProcessingService
from .simple_file_service import FileService

__all__ = [
    'BaseService',
    'FileListingService',
    'FileLoadingService',
    'FilterService',
    'UnifiedTransformationService',
    'ManualAlignmentService',
    'AutoAlignmentService',
    'PixelScoreService',
    'SSIMScoreService',
    'IOUScoreService',
    'ScoringService',
    'AlignmentService',
    'ImageProcessingService',
    'FileService'
]
