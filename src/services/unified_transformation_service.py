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
- Uses: gds_aligned_generator.py
"""

from typing import Dict, Tuple
import numpy as np
from src.core.gds_aligned_generator import generate_aligned_gds

class UnifiedTransformationService:
    """Single transformation service for all alignment methods."""
    
    def __init__(self):
        self.current_params = {
            'move_x': 0.0,
            'move_y': 0.0, 
            'zoom': 100.0,
            'rotation': 0.0
        }
    
    def apply_transformations(self, structure_num: int, move_x: float, move_y: float, 
                            zoom: float, rotation: float):
        """Apply transformations using new bounds-based approach."""
        # Store parameters
        self.current_params = {
            'move_x': move_x,
            'move_y': move_y,
            'zoom': zoom, 
            'rotation': rotation
        }
        
        # Parameters are stored for use in generate_aligned_image
    
    def generate_aligned_image(self, structure_num: int) -> np.ndarray:
        """Generate aligned GDS image using new bounds-based approach."""
        # Use the new bounds-based approach
        transform_params = {
            'rotation': self.current_params['rotation'],
            'zoom': self.current_params['zoom'],
            'move_x': self.current_params['move_x'],
            'move_y': self.current_params['move_y']
        }
        
        # Generate using new approach
        aligned_image, bounds = generate_aligned_gds(structure_num, transform_params, (1024, 666))
        return aligned_image
    

    
    def get_current_parameters(self) -> Dict[str, float]:
        """Get current transformation parameters."""
        return self.current_params.copy()