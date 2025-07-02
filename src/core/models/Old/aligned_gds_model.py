"""Aligned GDS model that applies transforms and renders bitmaps."""

import numpy as np
import cv2
import math
from typing import Dict, List, Tuple, Optional
from .initial_gds_model import InitialGdsModel


class AlignedGDSModel:
    """Applies transforms to GDS data and renders as bitmap images."""
    
    def __init__(self, initial_model: InitialGdsModel):
        """Initialize with an InitialGdsModel."""
        self.initial_model = initial_model
        self.transforms = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0
        }
    
    def apply_transform(self, transform_type: str, value: float) -> None:
        """Update a transformation parameter."""
        if transform_type in self.transforms:
            self.transforms[transform_type] = value
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
    
    def reset_transforms(self) -> None:
        """Reset all transformations to default values."""
        self.transforms = {
            'translate_x': 0.0,
            'translate_y': 0.0,
            'rotation': 0.0,
            'scale': 1.0
        }
    
    def apply_transforms_to_polygon(self, polygon: np.ndarray) -> np.ndarray:
        """Apply current transformations to a single polygon."""
        poly = polygon.astype(np.float64)
        
        # Apply scaling
        if self.transforms['scale'] != 1.0:
            center = np.mean(poly, axis=0)
            poly = center + (poly - center) * self.transforms['scale']
        
        # Apply 90-degree rotations in GDS coordinates
        if abs(self.transforms['rotation']) % 90 == 0:
            angle = math.radians(self.transforms['rotation'])
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            center = np.mean(poly, axis=0)
            poly = center + np.dot(poly - center, rotation_matrix.T)
        
        # Apply translation
        poly[:, 0] += self.transforms['translate_x']
        poly[:, 1] += self.transforms['translate_y']
        
        return poly
    
    def to_bitmap(self, layers: List[int], resolution: Tuple[int, int]) -> np.ndarray:
        """Render the transformed GDS data to a binary raster image."""
        width, height = resolution
        
        # Create white background image
        img = np.full((height, width), 255, dtype=np.uint8)
        
        # Get structure bounds
        bounds = self.initial_model.get_bounds()
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
                
                # Apply small rotations at the end
                if abs(self.transforms['rotation']) % 90 != 0:
                    angle = math.radians(self.transforms['rotation'] % 90)
                    center_x, center_y = width // 2, height // 2
                    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), math.degrees(angle), 1.0)
                    img = cv2.warpAffine(img, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                
                # Draw polygon directly in black
                cv2.fillPoly(img, [points], 0)
        
        return img
    
    def get_transform_matrix(self) -> np.ndarray:
        """Get the current transformation matrix."""
        angle = math.radians(self.transforms['rotation'])
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        scale = self.transforms['scale']
        tx, ty = self.transforms['translate_x'], self.transforms['translate_y']
        
        matrix = np.array([
            [scale * cos_a, -scale * sin_a, tx],
            [scale * sin_a,  scale * cos_a, ty],
            [0,              0,             1]
        ])
        
        return matrix
