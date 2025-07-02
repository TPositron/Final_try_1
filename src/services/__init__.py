"""
Services package for business logic and application services.

This package contains all service classes that implement the business logic
and coordinate between the UI and core data models. Services handle:
- File operations (loading, saving, listing)
- Image processing and filtering
- Alignment algorithms (manual and automatic)
- Scoring and analysis calculations
- Workflow orchestration

The services follow a common interface pattern using Qt signals/slots
for asynchronous communication with the UI layer.

Architecture:
    BaseService: Common functionality and interface patterns
    File Services: Handle data loading and file management
    Processing Services: Image filtering, alignment, and scoring
    Workflow Services: High-level operation coordination
"""

from .base_service import BaseService
from .file_listing_service import FileListingService
from .file_loading_service import FileLoadingService
from .filters.filter_service import FilterService
from .transformations.transform_service import TransformService
from .manual_alignment_service import ManualAlignmentService
from .auto_alignment_service import AutoAlignmentService
from .scoring.pixel_score_service import PixelScoreService
from .scoring.ssim_score_service import SSIMScoreService
from .scoring.iou_score_service import IOUScoreService
# Updated to use simple scoring service (Step 8)
from .simple_scoring_service import ScoringService

__all__ = [
    'BaseService',
    'FileListingService',
    'FileLoadingService',
    'FilterService',
    'TransformService',
    'ManualAlignmentService',
    'AutoAlignmentService',
    'PixelScoreService',
    'SSIMScoreService',
    'IOUScoreService',
    'ScoringService'
]
