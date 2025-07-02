"""
Step 16: Completely separate Manual Processing Pipeline.
No shared state with automatic pipeline.
Each pipeline has its own progress tracking.
User explicitly chooses which pipeline to use.
"""

from PySide6.QtCore import QObject, Signal
from src.services.filters.image_processing_service import ImageProcessingService
from src.services.simple_alignment_service import AlignmentService
from src.services.simple_scoring_service import ScoringService
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ManualProcessingPipeline(QObject):
    """
    Manual Processing Pipeline - Step 16: Completely separate from automatic pipeline.
    Handles user-controlled workflow with individual step control.
    No shared state between pipelines.
    """
    
    # Manual pipeline specific signals
    manual_step_started = Signal(str)                    # step_name
    manual_step_completed = Signal(str, dict)            # step_name, results
    manual_step_progress = Signal(str, int)              # step_name, progress_percent
    manual_step_error = Signal(str, str)                 # step_name, error_message
    manual_pipeline_completed = Signal(dict)             # final_results
    manual_pipeline_reset = Signal()                     # pipeline_reset
    manual_status_update = Signal(str)                   # status_message
    
    # Manual-specific detailed signals
    manual_parameter_requested = Signal(str, dict)       # step_name, default_params
    manual_user_input_required = Signal(str, str)        # step_name, input_type
    manual_preview_available = Signal(str, object)       # step_name, preview_data
    manual_confirmation_required = Signal(str, dict)     # step_name, action_details
    
    def __init__(self):
        super().__init__()
        
        # Manual pipeline services (separate instances)
        self.manual_image_processor = ImageProcessingService()
        self.manual_alignment_service = AlignmentService()
        self.manual_scoring_service = ScoringService()
        
        # Manual pipeline state (no sharing with automatic)
        self.manual_sem_image = None
        self.manual_gds_image = None
        self.manual_filtered_image = None
        self.manual_aligned_result = None
        self.manual_score_result = None
        
        # Manual pipeline configuration
        self.manual_config = {
            'step_by_step_mode': True,
            'require_user_confirmation': True,
            'enable_preview': True,
            'allow_step_jumping': True
        }
        
        # Manual pipeline progress tracking
        self.manual_current_step = None
        self.manual_completed_steps = set()
        self.manual_step_results = {}
        self.manual_is_running = False
        
        logger.info("Manual Processing Pipeline initialized - Step 16: Completely separate")
    
    def load_images_manual(self, sem_image_path: str, gds_image_path: str) -> bool:
        """Load images for manual processing pipeline."""
        try:
            self.manual_step_started.emit("load_images")
            self.manual_status_update.emit("Loading images for manual processing...")
            
            # Load SEM image
            self.manual_step_progress.emit("load_images", 25)
            # Simulate loading logic
            self.manual_sem_image = sem_image_path  # Placeholder
            
            # Load GDS image
            self.manual_step_progress.emit("load_images", 75)
            # Simulate loading logic
            self.manual_gds_image = gds_image_path  # Placeholder
            
            self.manual_step_progress.emit("load_images", 100)
            
            result = {
                'sem_image_loaded': True,
                'gds_image_loaded': True,
                'sem_path': sem_image_path,
                'gds_path': gds_image_path
            }
            
            self.manual_step_results['load_images'] = result
            self.manual_completed_steps.add('load_images')
            self.manual_step_completed.emit("load_images", result)
            
            logger.info("Manual pipeline: Images loaded successfully")
            return True
            
        except Exception as e:
            error_msg = f"Manual pipeline image loading failed: {str(e)}"
            self.manual_step_error.emit("load_images", error_msg)
            logger.error(error_msg)
            return False
    
    def apply_filters_manual(self, filter_params: Optional[Dict] = None) -> bool:
        """Apply filters in manual mode with user parameter collection."""
        try:
            self.manual_step_started.emit("apply_filters")
            self.manual_current_step = "apply_filters"
            
            if self.manual_sem_image is None:
                self.manual_step_error.emit("apply_filters", "No SEM image loaded")
                return False
            
            # Request parameters from user
            default_params = {
                'filter_type': 'clahe',
                'clip_limit': 2.0,
                'tile_grid_size': (8, 8)
            }
            
            if filter_params is None:
                self.manual_parameter_requested.emit("apply_filters", default_params)
                self.manual_status_update.emit("Waiting for filter parameters...")
                return True  # Wait for user input
            
            # Apply filters with user parameters
            self.manual_status_update.emit("Applying filters with manual parameters...")
            self.manual_step_progress.emit("apply_filters", 50)
            
            # Simulate filter application
            self.manual_filtered_image = f"filtered_{self.manual_sem_image}"
            
            self.manual_step_progress.emit("apply_filters", 100)
            
            result = {
                'filtered_image': self.manual_filtered_image,
                'filter_params': filter_params,
                'source': 'manual'
            }
            
            self.manual_step_results['apply_filters'] = result
            self.manual_completed_steps.add('apply_filters')
            self.manual_step_completed.emit("apply_filters", result)
            
            logger.info("Manual pipeline: Filters applied successfully")
            return True
            
        except Exception as e:
            error_msg = f"Manual pipeline filter application failed: {str(e)}"
            self.manual_step_error.emit("apply_filters", error_msg)
            logger.error(error_msg)
            return False
    
    def perform_alignment_manual(self, alignment_params: Optional[Dict] = None) -> bool:
        """Perform manual alignment with user control."""
        try:
            self.manual_step_started.emit("alignment")
            self.manual_current_step = "alignment"
            
            if self.manual_filtered_image is None:
                self.manual_step_error.emit("alignment", "No filtered image available")
                return False
            
            # Request alignment parameters
            default_params = {
                'alignment_method': '3_point_manual',
                'points_gds': [],
                'points_sem': [],
                'transformation_type': 'affine'
            }
            
            if alignment_params is None:
                self.manual_parameter_requested.emit("alignment", default_params)
                self.manual_status_update.emit("Waiting for alignment parameters...")
                return True  # Wait for user input
            
            # Perform manual alignment
            self.manual_status_update.emit("Performing manual alignment...")
            self.manual_step_progress.emit("alignment", 50)
            
            # Simulate alignment
            self.manual_aligned_result = {
                'transformation_matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                'aligned_gds': f"aligned_{self.manual_gds_image}",
                'alignment_quality': 0.85
            }
            
            self.manual_step_progress.emit("alignment", 100)
            
            result = {
                'aligned_result': self.manual_aligned_result,
                'alignment_params': alignment_params,
                'source': 'manual'
            }
            
            self.manual_step_results['alignment'] = result
            self.manual_completed_steps.add('alignment')
            self.manual_step_completed.emit("alignment", result)
            
            logger.info("Manual pipeline: Alignment completed successfully")
            return True
            
        except Exception as e:
            error_msg = f"Manual pipeline alignment failed: {str(e)}"
            self.manual_step_error.emit("alignment", error_msg)
            logger.error(error_msg)
            return False
    
    def calculate_score_manual(self, scoring_params: Optional[Dict] = None) -> bool:
        """Calculate alignment score in manual mode."""
        try:
            self.manual_step_started.emit("scoring")
            self.manual_current_step = "scoring"
            
            if self.manual_aligned_result is None:
                self.manual_step_error.emit("scoring", "No alignment result available")
                return False
            
            # Request scoring parameters
            default_params = {
                'scoring_method': 'pixel_similarity',
                'region_of_interest': None,
                'threshold': 0.5
            }
            
            if scoring_params is None:
                self.manual_parameter_requested.emit("scoring", default_params)
                self.manual_status_update.emit("Waiting for scoring parameters...")
                return True  # Wait for user input
            
            # Calculate score
            self.manual_status_update.emit("Calculating alignment score...")
            self.manual_step_progress.emit("scoring", 50)
            
            # Simulate scoring
            self.manual_score_result = {
                'similarity_score': 0.82,
                'alignment_quality': 0.85,
                'confidence': 0.90
            }
            
            self.manual_step_progress.emit("scoring", 100)
            
            result = {
                'score_result': self.manual_score_result,
                'scoring_params': scoring_params,
                'source': 'manual'
            }
            
            self.manual_step_results['scoring'] = result
            self.manual_completed_steps.add('scoring')
            self.manual_step_completed.emit("scoring", result)
            
            logger.info("Manual pipeline: Scoring completed successfully")
            return True
            
        except Exception as e:
            error_msg = f"Manual pipeline scoring failed: {str(e)}"
            self.manual_step_error.emit("scoring", error_msg)
            logger.error(error_msg)
            return False
    
    def complete_manual_pipeline(self) -> Dict[str, Any]:
        """Complete the manual pipeline and return final results."""
        try:
            final_results = {
                'pipeline_type': 'manual',
                'completed_steps': list(self.manual_completed_steps),
                'step_results': self.manual_step_results.copy(),
                'final_sem_image': self.manual_sem_image,
                'final_gds_image': self.manual_gds_image,
                'final_filtered_image': self.manual_filtered_image,
                'final_aligned_result': self.manual_aligned_result,
                'final_score_result': self.manual_score_result,
                'pipeline_config': self.manual_config.copy()
            }
            
            self.manual_pipeline_completed.emit(final_results)
            self.manual_status_update.emit("Manual pipeline completed successfully")
            
            logger.info("Manual pipeline: Completed successfully")
            return final_results
            
        except Exception as e:
            error_msg = f"Manual pipeline completion failed: {str(e)}"
            logger.error(error_msg)
            return {}
    
    def reset_manual_pipeline(self):
        """Reset the manual pipeline state."""
        self.manual_sem_image = None
        self.manual_gds_image = None
        self.manual_filtered_image = None
        self.manual_aligned_result = None
        self.manual_score_result = None
        self.manual_current_step = None
        self.manual_completed_steps.clear()
        self.manual_step_results.clear()
        self.manual_is_running = False
        
        self.manual_pipeline_reset.emit()
        self.manual_status_update.emit("Manual pipeline reset")
        
        logger.info("Manual pipeline: Reset completed")
    
    def get_manual_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of manual pipeline."""
        return {
            'pipeline_type': 'manual',
            'current_step': self.manual_current_step,
            'completed_steps': list(self.manual_completed_steps),
            'is_running': self.manual_is_running,
            'has_sem_image': self.manual_sem_image is not None,
            'has_gds_image': self.manual_gds_image is not None,
            'has_filtered_image': self.manual_filtered_image is not None,
            'has_aligned_result': self.manual_aligned_result is not None,
            'has_score_result': self.manual_score_result is not None
        }
    
    def jump_to_step_manual(self, step_name: str) -> bool:
        """Jump to a specific step in manual pipeline."""
        valid_steps = ['load_images', 'apply_filters', 'alignment', 'scoring']
        
        if step_name not in valid_steps:
            self.manual_step_error.emit(step_name, f"Invalid step: {step_name}")
            return False
        
        self.manual_current_step = step_name
        self.manual_status_update.emit(f"Jumped to manual step: {step_name}")
        
        logger.info(f"Manual pipeline: Jumped to step {step_name}")
        return True
