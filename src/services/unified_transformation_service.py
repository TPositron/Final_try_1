"""
Unified Transformation Service - Single Source of Truth for All Transformations

This service provides the single transformation engine used by:
- Manual alignment (UI controls)
- Hybrid alignment (3-point calculation)
- Automatic alignment (algorithm-based)

Key Methods:
- apply_transformations(): Apply move, zoom, rotate to GDS model
- calculate_aligned_coordinates(): Calculate new GDS coordinates
- generate_aligned_image(): Create final aligned GDS image

Dependencies:
- Used by: manual_alignment_service.py, hybrid_alignment_service.py, auto_alignment_service.py
- Uses: gds_aligned_generator.py, simple_aligned_gds_model.py
"""

from typing import Dict, Tuple
import numpy as np
from src.core.gds_aligned_generator import generate_aligned_gds
from src.core.models.simple_aligned_gds_model import AlignedGdsModel

class UnifiedTransformationService:
    """Single transformation service for all alignment methods."""
    
    def __init__(self):
        self.current_params = {
            'move_x': 0.0,
            'move_y': 0.0, 
            'zoom': 100.0,
            'rotation': 0.0
        }
    
    def apply_transformations(self, model: AlignedGdsModel, move_x: float, move_y: float, 
                            zoom: float, rotation: float):
        """Apply transformations using draft version formulas."""
        # Store parameters
        self.current_params = {
            'move_x': move_x,
            'move_y': move_y,
            'zoom': zoom, 
            'rotation': rotation
        }
        
        # Calculate pixel scale from original GDS-to-image conversion
        frame_width = model.current_frame[2] - model.current_frame[0]
        frame_height = model.current_frame[3] - model.current_frame[1]
        pixel_scale = min(frame_width / 1024, frame_height / 666)
        
        # Store parameters directly for image-level transformations
        model._pixel_translation = (move_x, move_y)
        model.frame_scale = zoom / 100.0
        model.set_residual_rotation(rotation)
    
    def generate_aligned_image(self, structure_num: int) -> np.ndarray:
        """Generate aligned GDS image using exact draft version approach."""
        # Step 1: Generate base GDS image (same as display)
        from src.core.gds_display_generator import generate_display_gds
        base_gds = generate_display_gds(structure_num, (1024, 666))
        
        # Step 2: Apply zoom transformation (same as zoom_image function)
        if self.current_params['zoom'] != 100:
            base_gds = self._apply_zoom_transform(base_gds, self.current_params['zoom'])
            
        # Step 3: Apply move transformation (same as move_image function)
        if self.current_params['move_x'] != 0 or self.current_params['move_y'] != 0:
            base_gds = self._apply_move_transform(base_gds, 
                                                self.current_params['move_x'], 
                                                self.current_params['move_y'])
        
        return base_gds
    
    def _apply_zoom_transform(self, image: np.ndarray, zoom_percent: float) -> np.ndarray:
        """Apply zoom using draft version formula."""
        import cv2
        h, w = image.shape[:2]
        scale = zoom_percent / 100.0
        center_x, center_y = w // 2, h // 2
        M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    def _apply_move_transform(self, image: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Apply move using draft version formula."""
        import cv2
        import numpy as np
        h, w = image.shape[:2]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    def get_current_parameters(self) -> Dict[str, float]:
        """Get current transformation parameters."""
        return self.current_params.copy()