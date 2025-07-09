"""
Processing Pipeline - Deprecated Legacy Pipeline Module

This module has been deprecated and replaced by modular pipeline classes.
It previously contained a monolithic pipeline implementation that handled
both manual and automatic processing modes in a single class.

Replacement Modules:
- pipeline_manual.py: ManualProcessingPipeline for user-controlled workflows
- pipeline_automatic.py: AutomaticProcessingPipeline for automated workflows  
- pipeline_hybrid.py: HybridProcessingPipeline for combined workflows
- pipeline_base.py: ProcessingPipelineBase with shared functionality
- pipeline_selector.py: PipelineSelector for choosing between pipelines

Migration Guide:
- Replace ProcessingPipeline with ManualProcessingPipeline for manual mode
- Replace ProcessingPipeline with AutomaticProcessingPipeline for automatic mode
- Use PipelineSelector to manage pipeline selection and switching
- Update imports to use specific pipeline classes instead of this module

Deprecation Reason:
- Improved separation of concerns between manual and automatic modes
- Better maintainability with modular architecture
- Cleaner signal handling and state management
- Enhanced extensibility for future pipeline types

Note: This file is kept for backward compatibility but should not be used
for new development. All functionality has been moved to the modular classes.
"""

# This file is now deprecated and replaced by modular pipeline classes.
# Use ManualProcessingPipeline, AutomaticProcessingPipeline, or HybridProcessingPipeline from the pipeline folder.

# from .pipeline_manual import ManualProcessingPipeline
# from .pipeline_automatic import AutomaticProcessingPipeline
# from .pipeline_hybrid import HybridProcessingPipeline

# (Optionally, you can keep a factory or dispatcher here if you want to instantiate by mode.)


