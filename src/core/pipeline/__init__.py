"""
Processing pipeline orchestration for SEM/GDS alignment workflows.

This module provides the high-level pipeline orchestration that coordinates
the three main processing stages: filtering, alignment, and scoring.

The pipeline supports both manual and automatic processing modes,
maintains stage state and parameters, and provides progress tracking
for long-running operations.

Classes:
    ManualProcessingPipeline: Manual pipeline for UI-driven workflows
    AutomaticProcessingPipeline: Automatic pipeline for batch processing
"""

from .pipeline_manual import ManualProcessingPipeline
from .pipeline_automatic import AutomaticProcessingPipeline

__all__ = ['ManualProcessingPipeline', 'AutomaticProcessingPipeline']
