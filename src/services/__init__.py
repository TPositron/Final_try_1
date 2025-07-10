"""
Services Package - Business Logic Layer (Updated for New Bounds-Based Approach)

Core Services:
- FileService: File operations using new bounds-based GDS processing
- ManualAlignmentService: Manual alignment with new generators
- UnifiedTransformationService: Unified transformation handling
- ImageProcessingService: Image filtering and processing
- ScoringService: Image comparison and scoring
"""

# Core services that work with new approach
from .simple_file_service import FileService
from .manual_alignment_service import ManualAlignmentService
from .unified_transformation_service import UnifiedTransformationService
from .simple_image_processing_service import ImageProcessingService
from .simple_scoring_service import ScoringService

# Legacy services (may have dependencies on old models)
try:
    from .base_service import BaseService
except ImportError:
    BaseService = None

try:
    from .file_listing_service import FileListingService
except ImportError:
    FileListingService = None

try:
    from .file_loading_service import FileLoadingService
except ImportError:
    FileLoadingService = None

try:
    from .filters.filter_service import FilterService
except ImportError:
    FilterService = None

try:
    from .simple_alignment_service import AlignmentService
except ImportError:
    AlignmentService = None

# Export only services that are guaranteed to work
__all__ = [
    'FileService',
    'ManualAlignmentService', 
    'UnifiedTransformationService',
    'ImageProcessingService',
    'ScoringService'
]

# Add legacy services if they imported successfully
if BaseService:
    __all__.append('BaseService')
if FileListingService:
    __all__.append('FileListingService')
if FileLoadingService:
    __all__.append('FileLoadingService')
if FilterService:
    __all__.append('FilterService')
if AlignmentService:
    __all__.append('AlignmentService')
