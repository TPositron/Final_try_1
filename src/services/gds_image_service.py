"""
GDS Image Generation Service
Generates binary images from GDS structures with scaling and rendering capabilities.
"""

import numpy as np
import cv2
import sys
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from ..core.models.simple_initial_gds_model import InitialGdsModel
from ..core.models.structure_definitions import StructureDefinition, StructureDefinitionManager


class GDSImageService:
    """
    Service for generating binary images from GDS structures.
    Handles scaling, rendering, and conversion to match SEM image dimensions.
    """
    
    # Standard SEM dimensions after cropping
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 666
    
    def __init__(self):
        """Initialize the GDS image generation service."""
        self.current_gds_model: Optional[InitialGdsModel] = None
        self.structure_manager = StructureDefinitionManager()
        self._image_cache = {}  # (structure_name, output_dimensions) -> np.ndarray

    def load_gds_file(self, gds_path: str) -> bool:
        """
        Load GDS file for image generation.
        
        Args:
            gds_path: Path to GDS file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.current_gds_model = InitialGdsModel(gds_path)
            self._image_cache.clear()  # Clear cache when loading a new GDS file
            return True
        except Exception as e:
            print(f"Error loading GDS file: {e}")
            return False
    
    def generate_binary_image(self, 
                             structure_name: str,
                             output_dimensions: Tuple[int, int] = None) -> Optional[np.ndarray]:
        """
        Generate binary image from GDS structure, using cache if available.
        
        Args:
            structure_name: Name of the structure to render
            output_dimensions: (width, height) for output image, defaults to SEM dimensions
            
        Returns:
            Binary image array or None if failed
        """
        if not self.current_gds_model:
            raise ValueError("No GDS file loaded. Call load_gds_file() first.")
        
        # Use default dimensions if not specified
        if output_dimensions is None:
            output_dimensions = (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        
        cache_key = (structure_name, output_dimensions)
        if cache_key in self._image_cache:
            return self._image_cache[cache_key]
        
        # Get structure definition
        structure = self.structure_manager.get_structure(structure_name)
        if not structure:
            raise ValueError(f"Structure '{structure_name}' not found in definitions")
        
        # Generate binary image for the structure
        try:
            binary_image = self.current_gds_model.generate_structure_bitmap(
                bounds=structure.bounds,
                layers=structure.layers,
                resolution=output_dimensions
            )
            if binary_image is not None:
                self._image_cache[cache_key] = binary_image
            return binary_image
            
        except Exception as e:
            print(f"Error generating binary image for {structure_name}: {e}")
            return None
    
    def generate_multiple_images(self, 
                                structure_names: List[str],
                                output_dimensions: Tuple[int, int] = None) -> Dict[str, np.ndarray]:
        """
        Generate binary images for multiple structures.
        
        Args:
            structure_names: List of structure names to render
            output_dimensions: (width, height) for output images
            
        Returns:
            Dictionary mapping structure names to binary images
        """
        results = {}
        
        for name in structure_names:
            try:
                image = self.generate_binary_image(name, output_dimensions)
                if image is not None:
                    results[name] = image
                else:
                    print(f"Failed to generate image for structure: {name}")
            except Exception as e:
                print(f"Error generating image for {name}: {e}")
        
        return results
    
    def save_structure_image_with_bounds(self, 
                                       structure_name: str, 
                                       output_path: str,
                                       dimensions: Tuple[int, int],
                                       bounds: Tuple[float, float, float, float],
                                       rotation_90: int = 0) -> bool:
        """
        Save structure binary image to file with custom bounds and 90° rotation.
        
        Args:
            structure_name: Name of structure to save
            output_path: Path for output image file
            dimensions: Image dimensions (width, height)
            bounds: Custom bounds (xmin, ymin, xmax, ymax) for extraction
            rotation_90: 90° rotation to apply (0, 90, 180, or 270 degrees)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if not self.current_gds_model:
                print("No GDS file loaded. Call load_gds_file() first.")
                return False
            
            # Get structure definition to get the layers
            structure = self.structure_manager.get_structure(structure_name)
            if not structure:
                # If structure not found in definitions, get layers from the GDS model itself
                print(f"Structure '{structure_name}' not found in definitions, using all available layers from GDS")
                
                # Try to get layers from the current model
                if hasattr(self.current_gds_model, 'get_layers'):
                    layers = self.current_gds_model.get_layers()
                    if not layers:
                        # Fallback to common layers
                        layers = [0, 1, 2, 3, 4, 5]
                        print(f"No layers found in GDS, using default layers: {layers}")
                    else:
                        print(f"Using layers from GDS file: {layers}")
                else:
                    # Fallback to common layers
                    layers = [0, 1, 2, 3, 4, 5]
                    print(f"Cannot get layers from GDS model, using default layers: {layers}")
                    
                # Create a temporary structure-like object
                class TempStructure:
                    def __init__(self, layers):
                        self.layers = layers
                        
                structure = TempStructure(layers)
            
            # Generate binary image with custom bounds
            binary_image = self.current_gds_model.generate_structure_bitmap(
                bounds=bounds,
                layers=structure.layers,
                resolution=dimensions
            )
            
            if binary_image is None:
                print(f"Failed to generate binary image for {structure_name}")
                return False

            # Apply 90° rotation to the image if specified
            if rotation_90 != 0:
                from PIL import Image
                
                print(f"Applying 90° rotation: {rotation_90} degrees")
                
                # Convert to PIL image for rotation
                pil_image = Image.fromarray(binary_image, mode='L')
                
                # Apply 90° rotation (PIL rotates counter-clockwise)
                if rotation_90 == 90:
                    pil_image = pil_image.rotate(90, expand=True)
                elif rotation_90 == 180:
                    pil_image = pil_image.rotate(180, expand=True)
                elif rotation_90 == 270:
                    pil_image = pil_image.rotate(270, expand=True)
                
                # Convert back to numpy array
                binary_image = np.array(pil_image)
                print(f"Image shape after rotation: {binary_image.shape}")

            # Ensure output directory exists
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to PIL Image and save
            from PIL import Image
            
            # Convert binary array (0,1) to proper image format (0,255)
            # Check if image contains only binary values (0 and 1)
            unique_values = np.unique(binary_image)
            if len(unique_values) <= 2 and unique_values.max() <= 1:
                # For binary images: 0 stays 0 (black), 1 becomes 255 (white)
                binary_image = (binary_image * 255).astype(np.uint8)
            
            print(f"Image stats: min={binary_image.min()}, max={binary_image.max()}, shape={binary_image.shape}, dtype={binary_image.dtype}")
            
            # Create PIL image and save
            pil_image = Image.fromarray(binary_image, mode='L')
            pil_image.save(str(output_path_obj))
            
            print(f"Saved structure image: {output_path_obj}")
            return True
            
        except Exception as e:
            print(f"Error saving structure image with bounds: {e}")
            return False

    def scale_to_dimensions(self,
                           image: np.ndarray, 
                           target_dimensions: Tuple[int, int],
                           maintain_aspect: bool = True) -> np.ndarray:
        """
        Scale image to target dimensions.
        
        Args:
            image: Input image array
            target_dimensions: (width, height) target size
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Scaled image array
        """
        current_height, current_width = image.shape[:2]
        target_width, target_height = target_dimensions
        
        if maintain_aspect:
            # Calculate scaling factor to fit within target dimensions
            scale_x = target_width / current_width
            scale_y = target_height / current_height
            scale = min(scale_x, scale_y)
            
            # Calculate new dimensions
            new_width = int(current_width * scale)
            new_height = int(current_height * scale)
            
            # Resize image
            scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            
            # Create canvas of target size and center the scaled image
            canvas = np.ones((target_height, target_width), dtype=image.dtype) * 255  # White background
            
            # Calculate centering offsets
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            
            # Place scaled image on canvas
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = scaled_image
            
            return canvas
        else:
            # Direct resize without maintaining aspect ratio
            return cv2.resize(image, target_dimensions, interpolation=cv2.INTER_NEAREST)
    
    def render_polygons(self, 
                       polygons: List[Dict[str, Any]], 
                       bounds: Tuple[float, float, float, float],
                       dimensions: Tuple[int, int] = None) -> np.ndarray:
        """
        Simple polygon-to-image rendering.
        
        Args:
            polygons: List of polygon dictionaries with coordinates and layer info
            bounds: (x_min, y_min, x_max, y_max) coordinate bounds
            dimensions: (width, height) for output image
            
        Returns:
            Rendered binary image
        """
        if dimensions is None:
            dimensions = (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        
        width, height = dimensions
        x_min, y_min, x_max, y_max = bounds
        
        # Calculate scaling factors
        scale_x = width / (x_max - x_min)
        scale_y = height / (y_max - y_min)
        
        # Create image (white background)
        image = np.ones((height, width), dtype=np.uint8) * 255
        
        for poly_data in polygons:
            coordinates = poly_data['coordinates']
            if len(coordinates) < 3:  # Need at least 3 points for a polygon
                continue
            
            # Transform coordinates to image space
            transformed_coords = np.copy(coordinates)
            transformed_coords[:, 0] = (coordinates[:, 0] - x_min) * scale_x
            transformed_coords[:, 1] = (coordinates[:, 1] - y_min) * scale_y
            
            # Flip Y-axis (GDS Y increases upward, image Y increases downward)
            transformed_coords[:, 1] = height - 1 - transformed_coords[:, 1]
            
            # Convert to integer coordinates
            int_coords = np.round(transformed_coords).astype(np.int32)
            
            # Ensure coordinates are within image bounds
            int_coords[:, 0] = np.clip(int_coords[:, 0], 0, width - 1)
            int_coords[:, 1] = np.clip(int_coords[:, 1], 0, height - 1)
            
            # Draw filled polygon (black on white)
            cv2.fillPoly(image, [int_coords], color=0)
        
        return image
    
    def generate_structure_overlay(self, 
                                  structure_name: str,
                                  sem_dimensions: Tuple[int, int] = None,
                                  opacity: float = 0.5) -> Optional[np.ndarray]:
        """
        Generate structure overlay for alignment visualization.
        
        Args:
            structure_name: Name of structure to render
            sem_dimensions: SEM image dimensions to match
            opacity: Overlay opacity (0.0 to 1.0)
            
        Returns:
            Grayscale overlay image or None if failed
        """
        if sem_dimensions is None:
            sem_dimensions = (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        
        # Generate binary image
        binary_image = self.generate_binary_image(structure_name, sem_dimensions)
        if binary_image is None:
            return None
        
        # Convert to overlay with opacity
        # Binary: 0 (black) = structure, 255 (white) = background
        # Overlay: 0 = transparent, 255*opacity = structure
        overlay = np.zeros_like(binary_image, dtype=np.uint8)
        structure_mask = (binary_image == 0)  # Structure pixels
        overlay[structure_mask] = int(255 * opacity)
        
        return overlay
    
    def save_structure_image(self, 
                           structure_name: str, 
                           output_path: str,
                           dimensions: Tuple[int, int] = None) -> bool:
        """
        Save structure binary image to file.
        
        Args:
            structure_name: Name of structure to save
            output_path: Path for output image file
            dimensions: Image dimensions, defaults to SEM dimensions
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            image = self.generate_binary_image(structure_name, dimensions)
            if image is None:
                return False
            
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image
            cv2.imwrite(str(output_path), image)
            return True
            
        except Exception as e:
            print(f"Error saving structure image {structure_name}: {e}")
            return False
    
    def generate_structure_image(self, 
                               structure_name: str, 
                               output_path: str,
                               dimensions: Tuple[int, int] = None) -> bool:
        """
        Deprecated: Generate structure image, use save_structure_image instead.
        
        Args:
            structure_name: Name of structure to generate
            output_path: Path for output image file
            dimensions: Image dimensions, defaults to SEM dimensions
            
        Returns:
            True if generated successfully, False otherwise
        """
        print("Warning: generate_structure_image is deprecated, use save_structure_image instead.", file=sys.stderr)
        return self.save_structure_image(structure_name, output_path, dimensions)
    
    def save_multiple_structure_images(self, 
                                     structure_names: List[str],
                                     output_directory: str,
                                     dimensions: Tuple[int, int] = None) -> Dict[str, bool]:
        """
        Save multiple structure images to directory.
        
        Args:
            structure_names: List of structure names to save
            output_directory: Directory for output images
            dimensions: Image dimensions for all images
            
        Returns:
            Dictionary mapping structure names to success status
        """
        results = {}
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name in structure_names:
            output_path = output_dir / f"{name}.png"
            success = self.save_structure_image(name, str(output_path), dimensions)
            results[name] = success
        
        return results
    
    def get_structure_info(self, structure_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a structure.
        
        Args:
            structure_name: Name of structure
            
        Returns:
            Dictionary with structure information or None if not found
        """
        structure = self.structure_manager.get_structure(structure_name)
        if not structure:
            return None
        
        info = structure.to_dict()
        
        # Add GDS-specific information if available
        if self.current_gds_model:
            try:
                extracted = self.current_gds_model.extract_structure_from_definition(structure)
                info.update({
                    'polygon_count': extracted['polygon_count'],
                    'gds_bounds': self.current_gds_model.bounds,
                    'available_layers': self.current_gds_model.get_layers()
                })
            except Exception as e:
                print(f"Error extracting GDS info for {structure_name}: {e}")
        
        return info
    
    def list_available_structures(self) -> List[str]:
        """
        Get list of available structure names.
        
        Returns:
            List of structure names
        """
        return self.structure_manager.list_structures()
    
    def is_gds_loaded(self) -> bool:
        """Check if a GDS file is currently loaded."""
        return self.current_gds_model is not None
    
    def get_gds_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the loaded GDS file.
        
        Returns:
            Dictionary with GDS information or None if not loaded
        """
        if not self.current_gds_model:
            return None
        
        return {
            'path': str(self.current_gds_model.gds_path),
            'bounds': self.current_gds_model.bounds,
            'unit_size': self.current_gds_model.unit_size,
            'available_layers': self.current_gds_model.get_layers(),
            'cell_name': self.current_gds_model.cell.name if self.current_gds_model.cell else None
        }


if __name__ == "__main__":
    # Example usage and testing
    print("GDS Image Generation Service")
    print("=" * 40)
    
    # Create service instance
    service = GDSImageService()
    
    # List available structures
    structures = service.list_available_structures()
    print(f"Available structures: {structures}")
    
    # Example structure info
    if structures:
        first_structure = structures[0]
        info = service.get_structure_info(first_structure)
        print(f"\nStructure '{first_structure}' info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
