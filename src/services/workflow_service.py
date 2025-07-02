"""
Simple Workflow Service
Coordinates SEM and GDS loading with simple sequential processing and basic state management.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from enum import Enum
import time

from .file_loading_service import FileLoadingService
from .gds_image_service import GDSImageService
from .file_service import FileManager
from src.core.models import SemImage
from ..core.models.structure_definitions import get_default_structures


class WorkflowState(Enum):
    """Enumeration of workflow states."""
    IDLE = "idle"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class WorkflowStep(Enum):
    """Enumeration of workflow steps."""
    LOAD_SEM = "load_sem"
    LOAD_GDS = "load_gds"
    GENERATE_STRUCTURES = "generate_structures"
    FILTER_PROCESSING = "filter_processing"
    ALIGNMENT = "alignment"
    SCORING = "scoring"


class WorkflowService:
    """
    Service for coordinating SEM and GDS loading with simple sequential processing.
    Provides basic state management and workflow coordination.
    """
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.current_state = WorkflowState.IDLE
        self.current_step = None
        self.workflow_history = []
        self.start_time = None
        
        # Initialize services
        self.file_loading_service = FileLoadingService()
        self.gds_image_service = GDSImageService()
        self.file_manager = FileManager(base_dir)
        
        # Data storage
        self.current_sem_image = None
        self.current_gds_data = None
        self.generated_structures = {}
        self.workflow_results = {}
        
        # Structure definitions
        self.structure_definitions = get_default_structures()
    
    def load_sem_and_gds(self, 
                        sem_path: str, 
                        gds_path: str,
                        auto_generate_structures: bool = True) -> bool:
        """
        Coordinate SEM and GDS loading.
        
        Args:
            sem_path: Path to SEM image file
            gds_path: Path to GDS file
            auto_generate_structures: Whether to automatically generate structure images
            
        Returns:
            True if both loaded successfully, False otherwise
        """
        try:
            self._set_state(WorkflowState.PROCESSING)
            self._start_workflow()
            
            # Step 1: Load SEM image
            if not self._load_sem_image(sem_path):
                return False
            
            # Step 2: Load GDS file
            if not self._load_gds_file(gds_path):
                return False
            
            # Step 3: Generate structure images if requested
            if auto_generate_structures:
                if not self._generate_structure_images():
                    return False
            
            self._set_state(WorkflowState.READY)
            self._add_to_history("load_sem_and_gds", "success", 
                               {"sem_path": sem_path, "gds_path": gds_path})
            return True
            
        except Exception as e:
            self._handle_error(f"Error in load_sem_and_gds: {e}")
            return False
    
    def process_workflow(self, 
                        workflow_steps: List[WorkflowStep],
                        parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a sequence of workflow steps.
        
        Args:
            workflow_steps: List of WorkflowStep enums to execute
            parameters: Optional parameters for workflow steps
            
        Returns:
            Dictionary with workflow results
        """
        results = {}
        parameters = parameters or {}
        
        try:
            self._set_state(WorkflowState.PROCESSING)
            
            for step in workflow_steps:
                self.current_step = step
                step_result = self._execute_workflow_step(step, parameters)
                results[step.value] = step_result
                
                if not step_result.get('success', False):
                    break
            
            self._set_state(WorkflowState.READY if all(r.get('success', False) for r in results.values()) else WorkflowState.ERROR)
            return results
            
        except Exception as e:
            self._handle_error(f"Error in process_workflow: {e}")
            return {"error": str(e)}
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status and progress."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'current_state': self.current_state.value,
            'current_step': self.current_step.value if self.current_step else None,
            'elapsed_time': elapsed_time,
            'history_count': len(self.workflow_history),
            'has_sem_data': self.current_sem_image is not None,
            'has_gds_data': self.current_gds_data is not None,
            'generated_structures_count': len(self.generated_structures),
            'ready_for_alignment': self.is_ready_for_alignment()
        }
    
    def load_sem_image(self, sem_path: str) -> bool:
        """Load SEM image file."""
        return self._load_sem_image(sem_path)
    
    def load_gds_file(self, gds_path: str) -> bool:
        """Load GDS file."""
        return self._load_gds_file(gds_path)
    
    def generate_structure_images(self, structure_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """Generate structure images for specified structures."""
        if not self.current_gds_data:
            return {"error": "No GDS data loaded"}
        
        return self._generate_structure_images(structure_names)
    
    def select_structure(self, structure_name: str) -> bool:
        """Select a structure for alignment."""
        if structure_name not in self.generated_structures:
            return False
        
        self.current_structure = structure_name
        return True
    
    def get_structure_list(self) -> List[str]:
        """Get list of available structures."""
        return list(self.structure_definitions.keys())
    
    def is_ready_for_alignment(self) -> bool:
        """Check if workflow is ready for alignment step."""
        return (self.current_sem_image is not None and 
                self.current_gds_data is not None and 
                len(self.generated_structures) > 0)
    
    def get_current_data(self) -> Dict[str, Any]:
        """Get current workflow data."""
        return {
            'sem_image': self.current_sem_image,
            'gds_data': self.current_gds_data,
            'structures': self.generated_structures,
            'results': self.workflow_results
        }
    
    # Private methods
    def _set_state(self, state: WorkflowState) -> None:
        """Set workflow state."""
        self.current_state = state
    
    def _start_workflow(self) -> None:
        """Start workflow timer."""
        self.start_time = time.time()
        self.workflow_history = []
    
    def _load_sem_image(self, sem_path: str) -> bool:
        """Load SEM image."""
        try:
            # Use file loading service to load SEM
            self.current_sem_image = self.file_loading_service.load_sem(Path(sem_path))
            return self.current_sem_image is not None
        except Exception as e:
            self._handle_error(f"Failed to load SEM image: {e}")
            return False
    
    def _load_gds_file(self, gds_path: str) -> bool:
        """Load GDS file."""
        try:
            # Use file manager to load GDS
            self.current_gds_data = self.file_manager.load_gds_file(gds_path)
            return self.current_gds_data is not None
        except Exception as e:
            self._handle_error(f"Failed to load GDS file: {e}")
            return False
    
    def _generate_structure_images(self, structure_names: Optional[List[str]] = None) -> bool:
        """Generate structure images."""
        try:
            if not structure_names:
                structure_names = list(self.structure_definitions.keys())
            
            for structure_name in structure_names:
                if structure_name in self.structure_definitions:
                    # Generate structure image using GDS image service
                    structure_data = self.gds_image_service.generate_structure_image(
                        self.current_gds_data, structure_name
                    )
                    self.generated_structures[structure_name] = structure_data
            
            return len(self.generated_structures) > 0
        except Exception as e:
            self._handle_error(f"Failed to generate structure images: {e}")
            return False
    
    def _execute_workflow_step(self, step: WorkflowStep, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step."""
        try:
            if step == WorkflowStep.LOAD_SEM:
                success = self._load_sem_image(parameters.get('sem_path'))
                return {'success': success, 'data': self.current_sem_image}
            
            elif step == WorkflowStep.LOAD_GDS:
                success = self._load_gds_file(parameters.get('gds_path'))
                return {'success': success, 'data': self.current_gds_data}
            
            elif step == WorkflowStep.GENERATE_STRUCTURES:
                success = self._generate_structure_images(parameters.get('structure_names'))
                return {'success': success, 'data': self.generated_structures}
            
            else:
                return {'success': False, 'error': f'Unknown workflow step: {step}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _add_to_history(self, operation: str, status: str, data: Dict[str, Any]) -> None:
        """Add operation to workflow history."""
        self.workflow_history.append({
            'timestamp': time.time(),
            'operation': operation,
            'status': status,
            'data': data
        })
    
    def _handle_error(self, error_message: str) -> None:
        """Handle workflow error."""
        self._set_state(WorkflowState.ERROR)
        self._add_to_history("error", "failed", {"error": error_message})
    
    def __str__(self):
        """String representation of WorkflowService."""
        return (f"WorkflowService(state={self.current_state.value}, "
                f"step={self.current_step.value if self.current_step else 'None'}, "
                f"structures={len(self.generated_structures)})")


if __name__ == "__main__":
    # Example usage and testing
    print("Simple Workflow Service")
    print("=" * 30)
    
    # Create workflow service
    workflow = WorkflowService()
    print(f"Initial state: {workflow}")
    
    # Check status
    status = workflow.get_workflow_status()
    print(f"Status: {status['current_state']}")
    print(f"Ready for alignment: {workflow.is_ready_for_alignment()}")
    
    # Example workflow steps
    steps = [
        WorkflowStep.LOAD_SEM,
        WorkflowStep.LOAD_GDS,
        WorkflowStep.GENERATE_STRUCTURES
    ]
    
    print(f"Available workflow steps: {[step.value for step in steps]}")
    
    # Show available structures
    structures = workflow.get_structure_list()
    print(f"Available structures: {structures}")
