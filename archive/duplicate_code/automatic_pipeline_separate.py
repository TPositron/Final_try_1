"""
Step 16: Completely separate Automatic Processing Pipeline.
No shared state with manual pipeline.
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


class AutomaticProcessingPipeline(QObject):
    """
    Automatic Processing Pipeline - Step 16: Completely separate from manual pipeline.
    Handles automated workflow with distinct steps and progress reporting.
    No shared state between pipelines.
    """
    
    # Automatic pipeline specific signals
    auto_step_started = Signal(str)                      # step_name
    auto_step_completed = Signal(str, dict)              # step_name, results
    auto_step_progress = Signal(str, int)                # step_name, progress_percent
    auto_step_error = Signal(str, str)                   # step_name, error_message
    auto_pipeline_completed = Signal(dict)               # final_results
    auto_pipeline_stopped = Signal(str)                  # reason
    auto_status_update = Signal(str)                     # status_message
    
    # Automatic-specific detailed signals
    auto_filter_sequence_progress = Signal(int, int, str)  # current, total, filter_name
    auto_alignment_attempt = Signal(str, int)              # method_name, attempt_number
    auto_fallback_triggered = Signal(str, str)             # from_method, to_method
    auto_quality_check = Signal(str, float, bool)          # step_name, quality_score, passed
    
    def __init__(self):
        super().__init__()
        
        # Automatic pipeline services (separate instances)
        self.auto_image_processor = ImageProcessingService()
        self.auto_alignment_service = AlignmentService()
        self.auto_scoring_service = ScoringService()
        
        # Automatic pipeline state (no sharing with manual)
        self.auto_sem_image = None
        self.auto_gds_image = None
        self.auto_processed_image = None
        self.auto_alignment_result = None
        self.auto_final_score = None
        
        # Automatic pipeline configuration
        self.auto_config = {
            'filter_sequence': ['clahe', 'bilateral', 'sharpen'],
            'alignment_methods': ['auto_correlation', 'feature_matching'],
            'scoring_methods': ['pixel_similarity', 'structural_similarity'],
            'quality_thresholds': {
                'filter_quality': 0.7,
                'alignment_quality': 0.6,
                'score_confidence': 0.8
            },
            'max_attempts': 3,
            'enable_fallback': True
        }
        
        # Automatic pipeline progress tracking
        self.auto_current_phase = None
        self.auto_completed_phases = []
        self.auto_phase_results = {}
        self.auto_is_running = False
        self.auto_can_stop = True
        
        logger.info("Automatic Processing Pipeline initialized - Step 16: Completely separate")
    
    def run_complete_automatic_pipeline(self, sem_image_path: str, gds_image_path: str) -> bool:
        """Run the complete automatic pipeline in sequence."""
        try:
            self.auto_is_running = True
            self.auto_status_update.emit("Starting automatic processing pipeline...")
            
            # Phase 1: Load images
            if not self._auto_load_images(sem_image_path, gds_image_path):
                return False
            
            # Phase 2: Automatic filtering
            if not self._auto_apply_filters():
                return False
            
            # Phase 3: Automatic alignment
            if not self._auto_perform_alignment():
                return False
            
            # Phase 4: Automatic scoring
            if not self._auto_calculate_scores():
                return False
            
            # Complete pipeline
            self._auto_complete_pipeline()
            return True
            
        except Exception as e:
            error_msg = f"Automatic pipeline failed: {str(e)}"
            self.auto_step_error.emit("pipeline", error_msg)
            logger.error(error_msg)
            self.auto_is_running = False
            return False
    
    def _auto_load_images(self, sem_image_path: str, gds_image_path: str) -> bool:
        """Automatically load images."""
        try:
            self.auto_step_started.emit("auto_load_images")
            self.auto_current_phase = "auto_load_images"
            self.auto_status_update.emit("Automatically loading images...")
            
            # Load SEM image
            self.auto_step_progress.emit("auto_load_images", 30)
            self.auto_sem_image = sem_image_path  # Placeholder
            
            # Load GDS image  
            self.auto_step_progress.emit("auto_load_images", 70)
            self.auto_gds_image = gds_image_path  # Placeholder
            
            self.auto_step_progress.emit("auto_load_images", 100)
            
            result = {
                'sem_loaded': True,
                'gds_loaded': True,
                'auto_config_applied': True
            }
            
            self.auto_phase_results['auto_load_images'] = result
            self.auto_completed_phases.append('auto_load_images')
            self.auto_step_completed.emit("auto_load_images", result)
            
            logger.info("Automatic pipeline: Images loaded")
            return True
            
        except Exception as e:
            self.auto_step_error.emit("auto_load_images", str(e))
            return False
    
    def _auto_apply_filters(self) -> bool:
        """Automatically apply filter sequence."""
        try:
            self.auto_step_started.emit("auto_filtering")
            self.auto_current_phase = "auto_filtering"
            
            filter_sequence = self.auto_config['filter_sequence']
            total_filters = len(filter_sequence)
            current_image = self.auto_sem_image
            applied_filters = []
            
            for i, filter_name in enumerate(filter_sequence):
                self.auto_status_update.emit(f"Applying automatic filter: {filter_name}")
                self.auto_filter_sequence_progress.emit(i + 1, total_filters, filter_name)
                
                progress = int((i / total_filters) * 100)
                self.auto_step_progress.emit("auto_filtering", progress)
                
                # Simulate filter application
                filter_result = f"filtered_{filter_name}_{current_image}"
                current_image = filter_result
                
                applied_filters.append({
                    'filter_name': filter_name,
                    'success': True,
                    'quality_score': 0.8 + (i * 0.05)  # Simulate increasing quality
                })
                
                # Quality check
                quality_score = applied_filters[-1]['quality_score']
                quality_threshold = self.auto_config['quality_thresholds']['filter_quality']
                quality_passed = quality_score >= quality_threshold
                
                self.auto_quality_check.emit(filter_name, quality_score, quality_passed)
                
                if not quality_passed and self.auto_config['enable_fallback']:
                    self.auto_fallback_triggered.emit(filter_name, "skip_remaining_filters")
                    break
            
            self.auto_step_progress.emit("auto_filtering", 100)
            self.auto_processed_image = current_image
            
            result = {
                'processed_image': self.auto_processed_image,
                'applied_filters': applied_filters,
                'sequence_completed': True
            }
            
            self.auto_phase_results['auto_filtering'] = result
            self.auto_completed_phases.append('auto_filtering')
            self.auto_step_completed.emit("auto_filtering", result)
            
            logger.info("Automatic pipeline: Filtering completed")
            return True
            
        except Exception as e:
            self.auto_step_error.emit("auto_filtering", str(e))
            return False
    
    def _auto_perform_alignment(self) -> bool:
        """Automatically perform alignment with fallback."""
        try:
            self.auto_step_started.emit("auto_alignment")
            self.auto_current_phase = "auto_alignment"
            
            alignment_methods = self.auto_config['alignment_methods']
            max_attempts = self.auto_config['max_attempts']
            
            for attempt in range(max_attempts):
                for method in alignment_methods:
                    self.auto_status_update.emit(f"Attempting automatic alignment: {method}")
                    self.auto_alignment_attempt.emit(method, attempt + 1)
                    
                    progress = int(((attempt * len(alignment_methods) + alignment_methods.index(method) + 1) / 
                                  (max_attempts * len(alignment_methods))) * 100)
                    self.auto_step_progress.emit("auto_alignment", progress)
                    
                    # Simulate alignment attempt
                    alignment_quality = 0.5 + (attempt * 0.2) + (alignment_methods.index(method) * 0.1)
                    
                    if alignment_quality >= self.auto_config['quality_thresholds']['alignment_quality']:
                        # Success
                        self.auto_alignment_result = {
                            'method': method,
                            'attempt': attempt + 1,
                            'quality': alignment_quality,
                            'transformation_matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                        }
                        
                        self.auto_step_progress.emit("auto_alignment", 100)
                        
                        result = {
                            'alignment_result': self.auto_alignment_result,
                            'successful_method': method,
                            'attempts_made': attempt + 1
                        }
                        
                        self.auto_phase_results['auto_alignment'] = result
                        self.auto_completed_phases.append('auto_alignment')
                        self.auto_step_completed.emit("auto_alignment", result)
                        
                        logger.info(f"Automatic pipeline: Alignment successful with {method}")
                        return True
                    
                    # Try fallback
                    if method != alignment_methods[-1]:
                        next_method = alignment_methods[alignment_methods.index(method) + 1]
                        self.auto_fallback_triggered.emit(method, next_method)
            
            # All attempts failed
            self.auto_step_error.emit("auto_alignment", "All automatic alignment methods failed")
            return False
            
        except Exception as e:
            self.auto_step_error.emit("auto_alignment", str(e))
            return False
    
    def _auto_calculate_scores(self) -> bool:
        """Automatically calculate alignment scores."""
        try:
            self.auto_step_started.emit("auto_scoring")
            self.auto_current_phase = "auto_scoring"
            
            scoring_methods = self.auto_config['scoring_methods']
            total_methods = len(scoring_methods)
            scores = {}
            
            for i, method in enumerate(scoring_methods):
                self.auto_status_update.emit(f"Calculating score with: {method}")
                
                progress = int((i / total_methods) * 100)
                self.auto_step_progress.emit("auto_scoring", progress)
                
                # Simulate scoring
                score_value = 0.7 + (i * 0.1)
                scores[method] = score_value
                
                # Quality check
                confidence = 0.8 + (i * 0.05)
                confidence_threshold = self.auto_config['quality_thresholds']['score_confidence']
                quality_passed = confidence >= confidence_threshold
                
                self.auto_quality_check.emit(method, confidence, quality_passed)
            
            self.auto_step_progress.emit("auto_scoring", 100)
            
            # Calculate final composite score
            final_score = sum(scores.values()) / len(scores)
            
            self.auto_final_score = {
                'individual_scores': scores,
                'composite_score': final_score,
                'confidence': max(0.8, final_score)
            }
            
            result = {
                'final_score': self.auto_final_score,
                'scoring_methods_used': scoring_methods,
                'all_scores': scores
            }
            
            self.auto_phase_results['auto_scoring'] = result
            self.auto_completed_phases.append('auto_scoring')
            self.auto_step_completed.emit("auto_scoring", result)
            
            logger.info("Automatic pipeline: Scoring completed")
            return True
            
        except Exception as e:
            self.auto_step_error.emit("auto_scoring", str(e))
            return False
    
    def _auto_complete_pipeline(self):
        """Complete the automatic pipeline."""
        final_results = {
            'pipeline_type': 'automatic',
            'completed_phases': self.auto_completed_phases.copy(),
            'phase_results': self.auto_phase_results.copy(),
            'final_sem_image': self.auto_sem_image,
            'final_gds_image': self.auto_gds_image,
            'final_processed_image': self.auto_processed_image,
            'final_alignment_result': self.auto_alignment_result,
            'final_score': self.auto_final_score,
            'pipeline_config': self.auto_config.copy()
        }
        
        self.auto_is_running = False
        self.auto_pipeline_completed.emit(final_results)
        self.auto_status_update.emit("Automatic pipeline completed successfully")
        
        logger.info("Automatic pipeline: Completed successfully")
    
    def stop_automatic_pipeline(self, reason: str = "user_request"):
        """Stop the automatic pipeline."""
        if self.auto_is_running and self.auto_can_stop:
            self.auto_is_running = False
            self.auto_pipeline_stopped.emit(reason)
            self.auto_status_update.emit(f"Automatic pipeline stopped: {reason}")
            logger.info(f"Automatic pipeline: Stopped - {reason}")
    
    def reset_automatic_pipeline(self):
        """Reset the automatic pipeline state."""
        self.auto_sem_image = None
        self.auto_gds_image = None
        self.auto_processed_image = None
        self.auto_alignment_result = None
        self.auto_final_score = None
        self.auto_current_phase = None
        self.auto_completed_phases.clear()
        self.auto_phase_results.clear()
        self.auto_is_running = False
        
        self.auto_status_update.emit("Automatic pipeline reset")
        logger.info("Automatic pipeline: Reset completed")
    
    def get_automatic_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of automatic pipeline."""
        return {
            'pipeline_type': 'automatic',
            'current_phase': self.auto_current_phase,
            'completed_phases': self.auto_completed_phases.copy(),
            'is_running': self.auto_is_running,
            'can_stop': self.auto_can_stop,
            'has_sem_image': self.auto_sem_image is not None,
            'has_gds_image': self.auto_gds_image is not None,
            'has_processed_image': self.auto_processed_image is not None,
            'has_alignment_result': self.auto_alignment_result is not None,
            'has_final_score': self.auto_final_score is not None
        }
    
    def run_individual_step(self, step_name: str, **kwargs) -> bool:
        """Run an individual step of the automatic pipeline."""
        valid_steps = {
            'auto_load_images': lambda: self._auto_load_images(
                kwargs.get('sem_path', ''), kwargs.get('gds_path', '')
            ),
            'auto_filtering': self._auto_apply_filters,
            'auto_alignment': self._auto_perform_alignment,
            'auto_scoring': self._auto_calculate_scores
        }
        
        if step_name not in valid_steps:
            self.auto_step_error.emit(step_name, f"Invalid automatic step: {step_name}")
            return False
        
        try:
            return valid_steps[step_name]()
        except Exception as e:
            self.auto_step_error.emit(step_name, str(e))
            return False
