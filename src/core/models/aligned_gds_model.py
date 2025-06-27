"""Aligned GDS model that applies transforms and renders bitmaps."""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from .initial_gds_model import InitialGDSModel


class AlignedGDSModel:
    """Applies transforms to GDS data and renders as bitmap images."""
    
    def __init__(self, initial_model: InitialGDSModel):
        """Initialize with an InitialGDSModel."""
        self.initial_model = initial_model
        self.transforms = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0,
            'transparency': 1.0
        }
    
    def set_transform(self, transform_type: str, value: float) -> None:
        """Set a transform parameter."""
        if transform_type in self.transforms:
            self.transforms[transform_type] = value
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
    
    def get_transform(self, transform_type: str) -> float:
        """Get a transform parameter value."""
        return self.transforms.get(transform_type, 0.0)
    
    def reset_transforms(self) -> None:
        """Reset all transforms to default values."""
        self.transforms = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0,
            'transparency': 1.0
        }
    
    def apply_transforms_to_polygon(self, polygon: np.ndarray) -> np.ndarray:
        """Apply current transforms to a single polygon."""
        # Convert to float64 to avoid overflow
        poly = polygon.astype(np.float64)
        
        # Apply scaling
        if self.transforms['scale'] != 1.0:
            center = np.mean(poly, axis=0)
            poly = center + (poly - center) * self.transforms['scale']
        
        # Apply rotation
        if self.transforms['rotation'] != 0.0:
            angle = np.radians(self.transforms['rotation'])
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            center = np.mean(poly, axis=0)
            poly = center + np.dot(poly - center, rotation_matrix.T)
        
        # Apply translation
        poly[:, 0] += self.transforms['translate_x']
        poly[:, 1] += self.transforms['translate_y']
        
        return poly
    
    def render_to_bitmap(self, layers: List[int], bounds: Tuple[float, float, float, float], 
                        image_size: Tuple[int, int] = (1024, 666)) -> np.ndarray:
        """Render the transformed GDS data to a bitmap image."""
        width, height = image_size
        
        # Create white background image
        img = np.full((height, width), 255, dtype=np.uint8)
        
        # Get structure bounds
        xmin, ymin, xmax, ymax = bounds
        gds_width = xmax - xmin
        gds_height = ymax - ymin
        
        # Calculate scaling to fit bounds to image while maintaining aspect ratio
        scale_x = width / gds_width if gds_width > 0 else 1.0
        scale_y = height / gds_height if gds_height > 0 else 1.0
        scale = min(scale_x, scale_y)
        
        # Calculate centering offsets
        offset_x = int((width - gds_width * scale) / 2)
        offset_y = int((height - gds_height * scale) / 2)
        
        # Extract and transform polygons
        polygons_dict = self.initial_model.extract_polygons(layers, bounds)
        
        # Draw each layer's polygons
        for layer, polygons in polygons_dict.items():
            for poly in polygons:
                # Apply transforms
                transformed_poly = self.apply_transforms_to_polygon(poly)
                
                # Scale to image coordinates
                scaled = (transformed_poly - [xmin, ymin]) * scale
                positioned = scaled + [offset_x, offset_y]
                
                # Convert to integer coordinates for drawing
                points = np.round(positioned).astype(np.int32)
                
                # Apply transparency by blending with background
                if self.transforms['transparency'] < 1.0:
                    # Create temporary image for this polygon
                    temp_img = np.full((height, width), 255, dtype=np.uint8)
                    cv2.fillPoly(temp_img, [points], 0)
                    
                    # Blend with main image
                    alpha = self.transforms['transparency']
                    img = cv2.addWeighted(img, 1.0, temp_img, alpha, 0)
                else:
                    # Draw polygon directly in black
                    cv2.fillPoly(img, [points], 0)
        
        return img
    
    def get_transform_matrix(self) -> np.ndarray:
        """Get the current transformation matrix."""
        # Create 3x3 transformation matrix
        angle = np.radians(self.transforms['rotation'])
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        scale = self.transforms['scale']
        tx, ty = self.transforms['translate_x'], self.transforms['translate_y']
        
        matrix = np.array([
            [scale * cos_a, -scale * sin_a, tx],
            [scale * sin_a,  scale * cos_a, ty],
            [0,              0,             1]
        ])
        
        return matrix
