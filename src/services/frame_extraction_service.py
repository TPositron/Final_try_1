"""
Frame Extraction Service

Handles extraction of GDS data using the frame-based approach, without modifying the GDS data.
All transformations are applied to the view frame (viewport) only.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np

from src.core.models.simple_aligned_gds_model import AlignedGdsModel, create_aligned_model_for_structure


logger = logging.getLogger(__name__)


class FrameExtractionService:
    """
    A service for extracting GDS data using a frame-based approach.
    
    This service replaces the legacy GdsTransformationService which modified polygon data.
    Instead, this service only transforms the view frame (viewport) and extracts the correct
    region from the original GDS file.
    """
    
    def extract_to_bitmap(
        self, 
        gds_path: str,
        structure_name: str,
        transformation_params: Dict[str, float],
        gds_bounds: Tuple[float, float, float, float],
        output_size: Tuple[int, int] = (1024, 666),
        layers: Optional[list] = None
    ) -> np.ndarray:
        """
        Extract a bitmap image of the structure using frame-based approach.
        
        Args:
            gds_path: Path to the GDS file
            structure_name: Name of the structure to extract
            transformation_params: Dictionary with 'x_offset', 'y_offset', 'rotation', 'scale'
            gds_bounds: Original bounds of the structure (xmin, ymin, xmax, ymax)
            output_size: Size of the output bitmap (width, height)
            layers: Optional list of layers to include
            
        Returns:
            Numpy array containing the bitmap image
        """
        logger.info(f"Extracting bitmap from {gds_path}, structure {structure_name}")
        logger.debug(f"Transformation params: {transformation_params}")
        
        try:
            # FIX 1: Convert structure_name to structure_id and gds_bounds to pixel_size
            structure_id = self._get_structure_id(structure_name)
            pixel_size = self._calculate_pixel_size(gds_bounds, output_size)
            
            # Create a frame-based model for the specific structure
            model = create_aligned_model_for_structure(gds_path, structure_id, pixel_size)
            
            # Apply transformations to the frame
            self._apply_transformations(model, transformation_params)
            
            # Generate bitmap using the current frame
            bitmap = model.to_bitmap(resolution=output_size, layers=layers)
            
            return bitmap
            
        except Exception as e:
            logger.error(f"Error extracting bitmap: {e}", exc_info=True)
            raise
    
    def extract_to_file(
        self,
        gds_path: str,
        structure_name: str,
        transformation_params: Dict[str, float],
        gds_bounds: Tuple[float, float, float, float],
        output_path: str,
        output_size: Tuple[int, int] = (1024, 666),
        layers: Optional[list] = None
    ) -> bool:
        """
        Extract a bitmap and save it to a file using frame-based approach.
        
        Args:
            gds_path: Path to the GDS file
            structure_name: Name of the structure to extract
            transformation_params: Dictionary with 'x_offset', 'y_offset', 'rotation', 'scale'
            gds_bounds: Original bounds of the structure (xmin, ymin, xmax, ymax)
            output_path: Path to save the bitmap
            output_size: Size of the output bitmap (width, height)
            layers: Optional list of layers to include
            
        Returns:
            True if successful, False otherwise
        """
        try:
            bitmap = self.extract_to_bitmap(
                gds_path=gds_path,
                structure_name=structure_name,
                transformation_params=transformation_params,
                gds_bounds=gds_bounds,
                output_size=output_size,
                layers=layers
            )
            
            # Ensure output directory exists
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Save bitmap to file
            from PIL import Image
            img = Image.fromarray(bitmap * 255).convert("L")  # Convert to grayscale
            img.save(str(output_path_obj))
            
            logger.info(f"Saved bitmap to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting to file: {e}", exc_info=True)
            return False
    
    def _get_structure_id(self, structure_name: str) -> int:
        """
        Convert structure name to structure ID.
        
        Args:
            structure_name: Name of the structure
            
        Returns:
            Integer ID for the structure
        """
        # FIX 1A: Handle structure name to ID conversion
        try:
            # If the structure_name is already numeric, convert it
            if structure_name.isdigit():
                return int(structure_name)
            
            # For string names, use a hash-based approach or lookup
            # This is a fallback - ideally you'd have a proper mapping
            structure_id_map = {
                'Circpol_T2': 1,
                'IP935Left_11': 2,
                'IP935Left_14': 3,
                'QC855GC_CROSS_Bottom': 4,
                'QC935_46': 5,
                'main': 0,  # Default main structure
            }
            
            if structure_name in structure_id_map:
                return structure_id_map[structure_name]
            
            # Fallback: use hash of string name
            return abs(hash(structure_name)) % 1000  # Keep it reasonable
            
        except Exception as e:
            logger.warning(f"Could not convert structure name '{structure_name}' to ID, using default 0: {e}")
            return 0
    
    def _calculate_pixel_size(self, gds_bounds: Tuple[float, float, float, float], 
                            output_size: Tuple[int, int]) -> float:
        """
        Calculate pixel size from GDS bounds and output resolution.
        
        Args:
            gds_bounds: Bounds of the structure (xmin, ymin, xmax, ymax)
            output_size: Output resolution (width, height)
            
        Returns:
            Pixel size in GDS units per pixel
        """
        # FIX 1B: Convert bounds tuple to pixel size
        try:
            xmin, ymin, xmax, ymax = gds_bounds
            gds_width = xmax - xmin
            gds_height = ymax - ymin
            
            output_width, output_height = output_size
            
            # Calculate pixel size based on the limiting dimension
            x_pixel_size = gds_width / output_width if output_width > 0 else 1.0
            y_pixel_size = gds_height / output_height if output_height > 0 else 1.0
            
            # Use the larger pixel size to ensure the entire structure fits
            pixel_size = max(x_pixel_size, y_pixel_size)
            
            # Ensure we have a reasonable minimum pixel size
            if pixel_size <= 0:
                pixel_size = 1.0
                
            logger.debug(f"Calculated pixel size: {pixel_size} from bounds {gds_bounds} and output size {output_size}")
            return pixel_size
            
        except Exception as e:
            logger.warning(f"Could not calculate pixel size from bounds {gds_bounds}, using default: {e}")
            return 1.0  # Default pixel size
    
    def _apply_transformations(self, model: AlignedGdsModel, params: Dict[str, float]) -> None:
        """
        Apply transformations to the model's frame.
        
        Args:
            model: The aligned GDS model
            params: Dictionary with transformation parameters
        """
        # Apply translations (convert from UI pixels to GDS units)
        x_offset = params.get('x_offset', 0.0)
        y_offset = params.get('y_offset', 0.0)
        model.set_translation_pixels(x_offset, y_offset)
        
        # Apply scaling
        scale = params.get('scale', 1.0)
        model.set_scale(scale)
        
        # Apply rotation (first handle 90° increments, then residual)
        rotation = params.get('rotation', 0.0)
        rotation_90 = int(rotation / 90) * 90  # Round to nearest 90°
        residual = rotation - rotation_90
        
        if rotation_90:
            model.set_rotation_90(rotation_90)
        if residual:
            model.set_residual_rotation(residual)
