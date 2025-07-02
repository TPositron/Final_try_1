"""
Step 16: Pipeline Selector - User explicitly chooses which pipeline to use.
No automatic switching between pipeline types.
Each pipeline has its own progress tracking.
"""

from PySide6.QtCore import QObject, Signal
from .manual_pipeline_separate import ManualProcessingPipeline
from .automatic_pipeline_separate import AutomaticProcessingPipeline
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PipelineSelector(QObject):
    """
    Pipeline Selector - Step 16: User explicitly chooses which pipeline to use.
    Manages completely separate manual and automatic pipelines.
    No shared state between pipelines.
    """
    
    # Pipeline selection signals
    pipeline_selected = Signal(str)                      # pipeline_type ('manual' or 'automatic')
    pipeline_switched = Signal(str, str)                 # from_pipeline, to_pipeline
    pipeline_status_changed = Signal(str, dict)          # pipeline_type, status
    
    def __init__(self):
        super().__init__()
        
        # Create separate pipeline instances
        self.manual_pipeline = ManualProcessingPipeline()
        self.automatic_pipeline = AutomaticProcessingPipeline()
        
        # Pipeline selector state
        self.selected_pipeline_type = None
        self.active_pipeline = None
        
        # Connect pipeline signals
        self._connect_pipeline_signals()
        
        logger.info("Pipeline Selector initialized - Step 16: Explicit pipeline choice")
    
    def _connect_pipeline_signals(self):
        """Connect signals from both pipelines."""
        # Manual pipeline signals
        self.manual_pipeline.manual_step_started.connect(
            lambda step: self.pipeline_status_changed.emit('manual', {'event': 'step_started', 'step': step})
        )
        self.manual_pipeline.manual_step_completed.connect(
            lambda step, result: self.pipeline_status_changed.emit('manual', {'event': 'step_completed', 'step': step, 'result': result})
        )
        self.manual_pipeline.manual_pipeline_completed.connect(
            lambda result: self.pipeline_status_changed.emit('manual', {'event': 'pipeline_completed', 'result': result})
        )
        
        # Automatic pipeline signals  
        self.automatic_pipeline.auto_step_started.connect(
            lambda step: self.pipeline_status_changed.emit('automatic', {'event': 'step_started', 'step': step})
        )
        self.automatic_pipeline.auto_step_completed.connect(
            lambda step, result: self.pipeline_status_changed.emit('automatic', {'event': 'step_completed', 'step': step, 'result': result})
        )
        self.automatic_pipeline.auto_pipeline_completed.connect(
            lambda result: self.pipeline_status_changed.emit('automatic', {'event': 'pipeline_completed', 'result': result})
        )
    
    def select_pipeline(self, pipeline_type: str) -> bool:
        """Explicitly select which pipeline to use."""
        valid_types = ['manual', 'automatic']
        
        if pipeline_type not in valid_types:
            logger.error(f"Invalid pipeline type: {pipeline_type}")
            return False
        
        # Switch pipeline if different
        previous_pipeline = self.selected_pipeline_type
        if previous_pipeline and previous_pipeline != pipeline_type:
            self.pipeline_switched.emit(previous_pipeline, pipeline_type)
            logger.info(f"Pipeline switched from {previous_pipeline} to {pipeline_type}")
        
        # Set active pipeline
        self.selected_pipeline_type = pipeline_type
        
        if pipeline_type == 'manual':
            self.active_pipeline = self.manual_pipeline
        elif pipeline_type == 'automatic':
            self.active_pipeline = self.automatic_pipeline
        
        self.pipeline_selected.emit(pipeline_type)
        logger.info(f"Pipeline selected: {pipeline_type}")
        return True
    
    def get_selected_pipeline_type(self) -> Optional[str]:
        """Get the currently selected pipeline type."""
        return self.selected_pipeline_type
    
    def get_active_pipeline(self):
        """Get the active pipeline instance."""
        return self.active_pipeline
    
    def get_manual_pipeline(self) -> ManualProcessingPipeline:
        """Get the manual pipeline instance."""
        return self.manual_pipeline
    
    def get_automatic_pipeline(self) -> AutomaticProcessingPipeline:
        """Get the automatic pipeline instance."""
        return self.automatic_pipeline
    
    def get_pipeline_status(self, pipeline_type: Optional[str] = None) -> Dict[str, Any]:
        """Get status of specified pipeline or active pipeline."""
        if pipeline_type is None:
            pipeline_type = self.selected_pipeline_type
        
        if pipeline_type == 'manual':
            return self.manual_pipeline.get_manual_pipeline_status()
        elif pipeline_type == 'automatic':
            return self.automatic_pipeline.get_automatic_pipeline_status()
        else:
            return {'error': f'Invalid pipeline type: {pipeline_type}'}
    
    def get_both_pipeline_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of both pipelines."""
        return {
            'manual': self.manual_pipeline.get_manual_pipeline_status(),
            'automatic': self.automatic_pipeline.get_automatic_pipeline_status(),
            'selected': self.selected_pipeline_type
        }
    
    def reset_pipeline(self, pipeline_type: Optional[str] = None):
        """Reset specified pipeline or active pipeline."""
        if pipeline_type is None:
            pipeline_type = self.selected_pipeline_type
        
        if pipeline_type == 'manual':
            self.manual_pipeline.reset_manual_pipeline()
            logger.info("Manual pipeline reset")
        elif pipeline_type == 'automatic':
            self.automatic_pipeline.reset_automatic_pipeline()
            logger.info("Automatic pipeline reset")
        else:
            logger.error(f"Cannot reset invalid pipeline type: {pipeline_type}")
    
    def reset_both_pipelines(self):
        """Reset both pipelines."""
        self.manual_pipeline.reset_manual_pipeline()
        self.automatic_pipeline.reset_automatic_pipeline()
        self.selected_pipeline_type = None
        self.active_pipeline = None
        logger.info("Both pipelines reset")
    
    def is_pipeline_running(self, pipeline_type: Optional[str] = None) -> bool:
        """Check if specified pipeline is running."""
        if pipeline_type is None:
            pipeline_type = self.selected_pipeline_type
        
        if pipeline_type == 'manual':
            return self.manual_pipeline.manual_is_running
        elif pipeline_type == 'automatic':
            return self.automatic_pipeline.auto_is_running
        else:
            return False
    
    def are_any_pipelines_running(self) -> Dict[str, bool]:
        """Check if any pipelines are running."""
        return {
            'manual': self.manual_pipeline.manual_is_running,
            'automatic': self.automatic_pipeline.auto_is_running,
            'any_running': self.manual_pipeline.manual_is_running or self.automatic_pipeline.auto_is_running
        }
    
    def validate_pipeline_selection(self) -> Dict[str, Any]:
        """Validate that a pipeline is properly selected."""
        validation = {
            'has_selection': self.selected_pipeline_type is not None,
            'selected_type': self.selected_pipeline_type,
            'active_pipeline_available': self.active_pipeline is not None,
            'valid_selection': False
        }
        
        if validation['has_selection'] and validation['active_pipeline_available']:
            if self.selected_pipeline_type in ['manual', 'automatic']:
                validation['valid_selection'] = True
        
        return validation
