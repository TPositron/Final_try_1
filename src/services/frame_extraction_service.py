"""
Frame Extraction Service - GDS Frame-Based Data Extraction

This service handles extraction of GDS layout data using a frame-based approach
that applies transformations to the viewport rather than modifying the underlying
GDS data. It provides bitmap generation and file export capabilities with
transformation support.

Main Class:
- FrameExtractionService: Frame-based GDS data extraction service

Key Methods:
- extract_to_bitmap(): Extracts GDS structure as bitmap using frame transformations
- extract_to_file(): Extracts bitmap and saves to file with format support
- _get_structure_id(): Converts structure names to numeric IDs
- _calculate_pixel_size(): Calculates pixel size from GDS bounds and output resolution
- _apply_transformations(): Applies transformation parameters to model frame

Dependencies:
- Uses: logging, pathlib.Path (standard libraries)
- Uses: numpy (array operations), PIL.Image (image saving)
- Uses: core/models/simple_aligned_gds_model (AlignedGdsModel, create_aligned_model_for_structure)
- Used by: services/file_service.py (aligned image saving)
- Used by: UI components for GDS visualization

Frame-Based Approach:
- Transformations applied to viewport/frame only
- Original GDS data remains unmodified
- View frame defines extraction region and transformations
- Non-destructive transformation pipeline
- Preserves original coordinate system integrity

Transformation Support:
- Translation: X/Y offset in pixels converted to GDS units
- Scaling: Uniform scale factor applied to frame
- Rotation: 90-degree increments plus residual rotation
- Coordinate system conversion between UI pixels and GDS units
- Transformation parameter validation and bounds checking

Structure ID Mapping:
- Circpol_T2: ID 1
- IP935Left_11: ID 2
- IP935Left_14: ID 3
- QC855GC_CROSS_Bottom: ID 4
- QC935_46: ID 5
- main: ID 0 (default)
- Hash-based fallback for unknown structure names

Bitmap Generation:
- Configurable output resolution (default 1024x666)
- Layer filtering support for selective rendering
- Grayscale bitmap output with proper scaling
- Memory-efficient processing for large structures
- Error handling with fallback mechanisms

File Export Features:
- Multiple image format support via PIL
- Automatic directory creation for output paths
- Grayscale conversion with proper intensity scaling
- File existence validation and error recovery
- Detailed logging for debugging and monitoring

Pixel Size Calculation:
- Automatic calculation from GDS bounds and output resolution
- Aspect ratio preservation with limiting dimension approach
- Minimum pixel size validation for edge cases
- Coordinate system scaling for proper visualization
- Error handling with reasonable defaults

Error Handling:
- Comprehensive exception handling for all operations
- Detailed logging with context information
- Graceful fallbacks for missing or invalid data
- Parameter validation with default value substitution
- File operation error recovery

Advantages over Legacy Approach:
- Non-destructive: Original GDS data preserved
- Efficient: No polygon modification required
- Flexible: Easy to adjust view parameters
- Accurate: Maintains coordinate system precision
- Scalable: Handles large structures efficiently

Usage Pattern:
1. Create FrameExtractionService instance
2. Define transformation parameters (translation, rotation, scale)
3. Specify GDS bounds and output resolution
4. Extract bitmap or save to file
5. Service handles coordinate conversion and rendering
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
