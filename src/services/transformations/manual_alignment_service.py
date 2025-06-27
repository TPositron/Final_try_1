"""Service for manual alignment operations."""

from typing import Optional, Dict, Any, Tuple
from PySide6.QtCore import QObject, Signal
import numpy as np

from ..core.models import AlignedGDSModel
from .transform_service import TransformService
from ..core.utils import get_logger, get_results_path
import cv2


class ManualAlignmentService(QObject):
    """Service for managing manual alignment operations."""
    
    # Signals
    alignment_updated = Signal(object)  # Emitted with aligned image
    alignment_saved = Signal(str)  # Emitted with save path
    alignment_error = Signal(str)  # Emitted on error
    
    def __init__(self, transform_service: TransformService):
        super().__init__()
        self.logger = get_logger(__name__)
        self.transform_service = transform_service
        self._gds_model = None
        self._current_aligned_image = None
        
        # Connect to transform changes
        self.transform_service.transform_changed.connect(self._on_transform_changed)
        self.transform_service.transforms_reset.connect(self._on_transforms_reset)
    
    def set_gds_model(self, gds_model: AlignedGDSModel) -> None:
        """Set the GDS model for alignment."""
        self._gds_model = gds_model
        self._update_alignment()
    
    def _on_transform_changed(self, transform_type: str, value: float) -> None:
        """Handle transform changes."""
        if self._gds_model:
            self._gds_model.set_transform(transform_type, value)
            self._update_alignment()
    
    def _on_transforms_reset(self) -> None:
        """Handle transform reset."""
        if self._gds_model:
            self._gds_model.reset_transforms()
            self._update_alignment()
    
    def _update_alignment(self) -> None:
        """Update the aligned image based on current transforms."""
        if not self._gds_model:
            return
        
        try:
            # For this example, we'll create a dummy aligned image
            # In practice, this would render the GDS with transforms applied
            aligned_image = np.full((666, 1024), 128, dtype=np.uint8)  # Gray background
            
            self._current_aligned_image = aligned_image
            self.alignment_updated.emit(aligned_image)
            
        except Exception as e:
            error_msg = f"Failed to update alignment: {e}"
            self.logger.error(error_msg)
            self.alignment_error.emit(error_msg)
    
    def get_current_alignment(self) -> Optional[np.ndarray]:
        """Get the current aligned image."""
        return self._current_aligned_image
    
    def save_alignment(self, name: str, sem_name: str = "", gds_name: str = "") -> bool:
        """
        Save the current alignment to disk.
        
        Args:
            name: Base name for the saved file
            sem_name: Name of the SEM image (for filename)
            gds_name: Name of the GDS structure (for filename)
            
        Returns:
            True if successful, False otherwise
        """
        if self._current_aligned_image is None:
            self.alignment_error.emit("No aligned image to save")
            return False
        
        try:
            # Create filename
            if sem_name and gds_name:
                filename = f"{sem_name}_{gds_name}_aligned_manual.png"
            else:
                filename = f"{name}_aligned_manual.png"
            
            save_path = get_results_path("Aligned/manual") / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image
            cv2.imwrite(str(save_path), self._current_aligned_image)
            
            # Save transform parameters
            params_file = save_path.with_suffix('.json')
            transform_params = self.transform_service.get_all_transforms()
            
            import json
            with open(params_file, 'w') as f:
                json.dump({
                    'transforms': transform_params,
                    'alignment_type': 'manual',
                    'sem_file': sem_name,
                    'gds_file': gds_name
                }, f, indent=2)
            
            self.alignment_saved.emit(str(save_path))
            self.logger.info(f"Saved manual alignment to {save_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to save alignment: {e}"
            self.logger.error(error_msg)
            self.alignment_error.emit(error_msg)
            return False
    
    def load_alignment(self, alignment_file: str) -> bool:
        """
        Load a previously saved alignment.
        
        Args:
            alignment_file: Path to the alignment parameters file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            with open(alignment_file, 'r') as f:
                data = json.load(f)
            
            transforms = data.get('transforms', {})
            
            # Apply loaded transforms
            for transform_type, value in transforms.items():
                self.transform_service.set_transform(transform_type, value)
            
            self.logger.info(f"Loaded alignment from {alignment_file}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to load alignment: {e}"
            self.logger.error(error_msg)
            self.alignment_error.emit(error_msg)
            return False
    
    def fine_tune_alignment(self, direction: str, step_size: float = 1.0) -> None:
        """
        Fine-tune alignment in a specific direction.
        
        Args:
            direction: Direction to move ('up', 'down', 'left', 'right', 'rotate_cw', 'rotate_ccw')
            step_size: Size of the adjustment step
        """
        if direction == 'up':
            self.transform_service.adjust_transform('translate_y', -step_size)
        elif direction == 'down':
            self.transform_service.adjust_transform('translate_y', step_size)
        elif direction == 'left':
            self.transform_service.adjust_transform('translate_x', -step_size)
        elif direction == 'right':
            self.transform_service.adjust_transform('translate_x', step_size)
        elif direction == 'rotate_cw':
            self.transform_service.adjust_transform('rotation', step_size)
        elif direction == 'rotate_ccw':
            self.transform_service.adjust_transform('rotation', -step_size)
        else:
            self.logger.warning(f"Unknown fine-tune direction: {direction}")
    
    def get_alignment_quality_score(self) -> float:
        """
        Calculate a quality score for the current alignment.
        
        Returns:
            Score between 0.0 and 1.0, where 1.0 is perfect alignment
        """
        # This is a placeholder - in practice, this would analyze
        # the overlap between GDS and SEM features
        return 0.75  # Dummy score
