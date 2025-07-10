"""
Aligned GDS model that wraps InitialGdsModel with transformation management.

This module provides the AlignedGdsModel class for:
- Wrapping InitialGdsModel with transformation parameters for a viewing frame.
- Managing frame scale, translation, and 90° rotation.
- Deferring arbitrary rotations to rendering time.
"""

import numpy as np
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from .simple_initial_gds_model import InitialGdsModel
from src.utils.transformations import (
    calculate_frame_bounds,
    apply_90_rotation_to_bounds,
    convert_pixels_to_gds_units,
    convert_gds_to_pixel_units,
    validate_transformation_parameters
)


logger = logging.getLogger(__name__)


class AlignedGdsModel:
    """
    Wraps InitialGdsModel with frame-based transformation management.
    
    This class manages a "viewing frame" over the GDS data, allowing for
    panning (translation), zooming (scaling), and rotation without modifying
    the underlying polygon data. Arbitrary rotations are deferred to rendering.
    """
    
    # Transformation presets
    PRESET_VIEWS = {
        'default': {'tx': 0.0, 'ty': 0.0, 'scale': 1.0, 'rotation_90': 0, 'residual_rotation': 0.0},
        'top': {'tx': 0.0, 'ty': 0.0, 'scale': 1.0, 'rotation_90': 0, 'residual_rotation': 0.0},
        'side': {'tx': 0.0, 'ty': 0.0, 'scale': 1.0, 'rotation_90': 90, 'residual_rotation': 0.0},
        'flipped': {'tx': 0.0, 'ty': 0.0, 'scale': 1.0, 'rotation_90': 180, 'residual_rotation': 0.0},
        'flipped_side': {'tx': 0.0, 'ty': 0.0, 'scale': 1.0, 'rotation_90': 270, 'residual_rotation': 0.0}
    }
    
    ZOOM_PRESETS = {
        'fit': 1.0,
        'zoom_in_25': 1.25,
        'zoom_in_50': 1.5,
        'zoom_in_100': 2.0,
        'zoom_in_200': 3.0,
        'zoom_out_25': 0.8,
        'zoom_out_50': 0.5,
        'zoom_out_100': 0.25
    }
    
    def __init__(self, initial_model: InitialGdsModel, pixel_size: float = None, 
                 feature_bounds: Optional[Tuple[float, float, float, float]] = None,
                 required_layers: Optional[List[int]] = None):
        """
        Initialize with coordinate-based feature extraction.
        
        Args:
            initial_model: The base GDS model to wrap
            pixel_size: GDS units per pixel (auto-calculated if None)
            feature_bounds: Specific feature bounds (xmin, ymin, xmax, ymax) in GDS units
            required_layers: Specific layers to render
        """
        self.initial_model = initial_model
        
        # Handle backward compatibility for layers
        if required_layers is None:
            # Get all layers from the GDS for backward compatibility
            try:
                all_layers = initial_model.get_layers()
                self.required_layers = all_layers if all_layers else [0]
                logger.info(f"No layers specified - using all available layers: {self.required_layers}")
            except:
                self.required_layers = [0]  # Default fallback
                logger.warning("Could not determine layers - using default layer 0")
        else:
            self.required_layers = required_layers
        
        # Calculate pixel size if not provided
        if pixel_size is None:
            self.pixel_size = self._calculate_pixel_size()
        else:
            self.pixel_size = pixel_size
        
        # Handle backward compatibility for bounds
        if feature_bounds is not None:
            self.current_frame = list(feature_bounds)
            self.original_frame = list(feature_bounds)
            logger.info(f"Using specified feature bounds: {feature_bounds}")
        else:
            # Use full bounds for backward compatibility
            self.current_frame = list(self.initial_model.bounds)
            self.original_frame = list(self.initial_model.bounds)
            logger.info("No feature bounds specified - using full GDS bounds")
        
        # Frame transformation parameters (viewport-based)
        self.frame_translation = {'tx': 0.0, 'ty': 0.0}  # Frame movement in GDS units
        self.frame_scale = 1.0  # Frame scaling factor (1.0 = original size)
        self.frame_rotation_90 = 0  # 0, 90, 180, or 270 degrees
        
        # Image-level rotation for non-90° angles
        self.residual_rotation = 0.0  # Applied to final bitmap
        
        # Store pixel transformations directly for UI consistency
        self._pixel_translation = (0.0, 0.0)
        
        logger.info(f"Initialized AlignedGdsModel with layers: {self.required_layers}")
        logger.debug(f"Original frame: {self.original_frame}")

    def _calculate_pixel_size(self) -> float:
        """
        Calculate GDS units per pixel based on typical rendering resolution.
        
        Returns:
            Estimated GDS units per pixel
        """
        bounds = self.initial_model.bounds
        if bounds == (0, 0, 0, 0):
            return 1.0
        
        gds_width = bounds[2] - bounds[0]
        gds_height = bounds[3] - bounds[1]
        
        # Assume typical rendering resolution of 1024x666
        typical_width_pixels = 1024
        typical_height_pixels = 666
        
        # Calculate pixel size based on fitting the GDS to typical resolution
        pixel_size_x = gds_width / typical_width_pixels
        pixel_size_y = gds_height / typical_height_pixels
        
        # Use the larger value to ensure the entire structure fits
        pixel_size = max(pixel_size_x, pixel_size_y)
        
        logger.debug(f"Calculated pixel size: {pixel_size} GDS units per pixel")
        return pixel_size

    def _calculate_frame_center(self) -> Tuple[float, float]:
        """
        Calculate the center of the current frame.
        
        Returns:
            (center_x, center_y) in GDS coordinates
        """
        if self.current_frame == [0, 0, 0, 0]:
            return (0.0, 0.0)
        
        center_x = (self.current_frame[0] + self.current_frame[2]) / 2
        center_y = (self.current_frame[1] + self.current_frame[3]) / 2
        return (center_x, center_y)
    
    def _calculate_original_frame_center(self) -> Tuple[float, float]:
        """
        Calculate the center of the original frame (for transformations).
        
        Returns:
            (center_x, center_y) in GDS coordinates of original frame
        """
        if self.original_frame == [0, 0, 0, 0]:
            return (0.0, 0.0)
        
        center_x = (self.original_frame[0] + self.original_frame[2]) / 2
        center_y = (self.original_frame[1] + self.original_frame[3]) / 2
        return (center_x, center_y)
    
    def set_pixel_size(self, pixel_size: float) -> None:
        """
        Set the GDS units per pixel for translation.
        
        Args:
            pixel_size: GDS units per pixel
        """
        self.pixel_size = pixel_size
        logger.info(f"Set pixel size: {pixel_size} GDS units per pixel")
    
    def set_translation(self, tx: float, ty: float) -> None:
        """
        Set frame translation offsets (moves the viewing window in GDS coordinate space).
        
        Args:
            tx: Frame translation in X direction (GDS units) - positive moves viewing window right
            ty: Frame translation in Y direction (GDS units) - positive moves viewing window up
        """
        self.frame_translation['tx'] = tx
        self.frame_translation['ty'] = ty
        self._update_current_frame()
        logger.debug(f"Set frame translation: tx={tx}, ty={ty}")
    
    def set_translation_pixels(self, dx_pixels: float, dy_pixels: float) -> None:
        """
        Set frame translation in pixels, converts to GDS units using utility function.
        
        Args:
            dx_pixels: UI movement in X (pixels) - positive moves viewing window right
            dy_pixels: UI movement in Y (pixels) - positive moves viewing window down (in UI coords)
        """
        # Use utility function to convert pixels to GDS units
        tx, ty = convert_pixels_to_gds_units(
            (dx_pixels, dy_pixels),
            self.pixel_size,
            flip_y=True
        )
        
        self.set_translation(tx, ty)
        logger.debug(f"Set translation from pixels: UI dx={dx_pixels}, dy={dy_pixels} -> frame tx={tx}, ty={ty}")

    def set_scale_center_relative(self, scale: float) -> None:
        """
        Set center-relative frame scaling factor (zoom).
        Scaling changes the frame size around the original frame center.
        
        Args:
            scale: Scaling factor (1.0 = original size, 2.0 = zoomed in 2x, 0.5 = zoomed out 2x)
        """
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got: {scale}")
        
        self.frame_scale = scale
        self._update_current_frame()
        logger.debug(f"Set frame scale: {scale}")

    def set_scale(self, scale: float) -> None:
        """
        Set center-relative frame scaling factor.
        
        Args:
            scale: Scaling factor (1.0 = original size)
        """
        self.set_scale_center_relative(scale)

    def set_rotation_90(self, degrees: int) -> None:
        """
        Set 90°-based frame rotation.
        
        Args:
            degrees: Rotation in degrees (must be multiple of 90)
        """
        if degrees % 90 != 0:
            raise ValueError(f"Rotation must be multiple of 90°, got: {degrees}")
        
        # Normalize to 0, 90, 180, or 270
        self.frame_rotation_90 = degrees % 360
        self._update_current_frame()
        logger.debug(f"Set frame 90° rotation: {self.frame_rotation_90}°")

    def set_residual_rotation(self, degrees: float) -> None:
        """
        Set residual rotation angle for rendering-time application.
        
        Args:
            degrees: Arbitrary rotation angle (e.g., 10°, 15°)
        """
        self.residual_rotation = degrees % 360
        logger.debug(f"Set residual rotation: {self.residual_rotation}°")
    
    def _update_current_frame(self) -> None:
        """
        Update current frame based on transformation parameters using utility functions.
        """
        if self.original_frame == [0, 0, 0, 0]:
            self.current_frame = [0, 0, 0, 0]
            return
        
        # Use the transformation utility to calculate new frame bounds
        translation = (self.frame_translation['tx'], self.frame_translation['ty'])
        new_bounds = calculate_frame_bounds(
            tuple(self.original_frame),
            translation=translation,
            scale=self.frame_scale,
            rotation_90=self.frame_rotation_90
        )
        
        self.current_frame = list(new_bounds)
        logger.debug(f"Updated viewing frame from {self.original_frame} to {self.current_frame}")
    
    def _apply_90_rotation_to_frame(self, frame: List[float], center_x: float, center_y: float) -> List[float]:
        """
        Apply 90° rotation to frame coordinates using utility function.
        
        Args:
            frame: Frame coordinates [xmin, ymin, xmax, ymax]
            center_x: Rotation center X
            center_y: Rotation center Y
            
        Returns:
            Rotated frame coordinates
        """
        if self.frame_rotation_90 == 0:
            return frame
        
        rotated_bounds = apply_90_rotation_to_bounds(
            tuple(frame),
            (center_x, center_y),
            self.frame_rotation_90
        )
        return list(rotated_bounds)
        
        xmin, ymin, xmax, ymax = frame
        
        # Calculate frame corners relative to center
        corners = [
            (xmin - center_x, ymin - center_y),  # bottom-left
            (xmax - center_x, ymin - center_y),  # bottom-right
            (xmax - center_x, ymax - center_y),  # top-right
            (xmin - center_x, ymax - center_y)   # top-left
        ]
        
        # Apply rotation to corners
        rotated_corners = []
        for dx, dy in corners:
            if self.frame_rotation_90 == 90:
                # 90° CCW: (x,y) -> (-y, x)
                new_dx, new_dy = -dy, dx
            elif self.frame_rotation_90 == 180:
                # 180°: (x,y) -> (-x, -y)
                new_dx, new_dy = -dx, -dy
            elif self.frame_rotation_90 == 270:
                # 270° CCW: (x,y) -> (y, -x)
                new_dx, new_dy = dy, -dx
            else:
                new_dx, new_dy = dx, dy
            
            rotated_corners.append((center_x + new_dx, center_y + new_dy))
        
        # Find new bounds from rotated corners
        xs = [corner[0] for corner in rotated_corners]
        ys = [corner[1] for corner in rotated_corners]
        
        return [min(xs), min(ys), max(xs), max(ys)]
    
    def get_transform_parameters(self) -> Dict[str, Any]:
        """
        Extract frame transformation parameters from current state.
        
        Returns:
            Dictionary with frame-based transformation parameters
        """
        return {
            'tx': self.frame_translation['tx'],
            'ty': self.frame_translation['ty'],
            'scale': self.frame_scale,
            'rotation_90': self.frame_rotation_90,
            'residual_rotation': self.residual_rotation,
            'current_frame': self.current_frame.copy(),
            'original_frame': self.original_frame.copy()
        }

    def set_ui_parameters(self, translation_x_pixels: float = 0.0, translation_y_pixels: float = 0.0,
                         scale: float = 1.0, rotation_degrees: float = 0.0) -> None:
        """
        Set frame transformation parameters from UI values.
        
        Args:
            translation_x_pixels: Translation in X (pixels)
            translation_y_pixels: Translation in Y (pixels) 
            scale: Scaling factor (1.0 = no scaling)
            rotation_degrees: Total rotation in degrees
        """
        # Set translation in pixels
        self.set_translation_pixels(translation_x_pixels, translation_y_pixels)
        
        # Set center-relative scaling
        self.set_scale_center_relative(scale)
        
        # Split rotation into 90° and residual parts
        rotation_90 = int(rotation_degrees // 90) * 90
        residual_rotation = rotation_degrees % 90
        
        self.set_rotation_90(rotation_90)
        self.set_residual_rotation(residual_rotation)
        
        logger.info(f"Set UI parameters: tx_px={translation_x_pixels}, ty_px={translation_y_pixels}, "
                   f"scale={scale}, rot_90={rotation_90}°, residual_rot={residual_rotation}°")

    def get_ui_parameters(self) -> Dict[str, float]:
        """
        Get frame transformation parameters in UI-friendly format using utility function.
        
        Returns:
            Dictionary with pixel translations, scale, and total rotation
        """
        # Use utility function to convert GDS units back to pixels
        ui_translation_x_pixels, ui_translation_y_pixels = convert_gds_to_pixel_units(
            (self.frame_translation['tx'], self.frame_translation['ty']),
            self.pixel_size,
            flip_y=True
        )
        
        # Combine 90° and residual rotation
        total_rotation = self.frame_rotation_90 + self.residual_rotation
        
        return {
            'translation_x_pixels': ui_translation_x_pixels,
            'translation_y_pixels': ui_translation_y_pixels,
            'scale': self.frame_scale,
            'rotation_degrees': total_rotation
        }

    def reset_transforms(self) -> None:
        """Reset all frame transformations to identity state."""
        self.frame_translation = {'tx': 0.0, 'ty': 0.0}
        self.frame_scale = 1.0
        self.frame_rotation_90 = 0
        self.residual_rotation = 0.0
        
        # Reset current frame to original frame
        self.current_frame = list(self.original_frame)
        
        logger.info("Reset all frame transformations to identity")
    
    def get_polygons_in_frame(self, layers: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Get polygons in frame for specified layers.
        """
        # Use provided layers, or fall back to the model's required layers
        target_layers = layers if layers is not None else self.required_layers
        
        if target_layers is None:
            # Final fallback - get all available layers
            try:
                target_layers = self.initial_model.get_layers()
                if not target_layers:
                    target_layers = [0]
                logger.warning(f"No layers specified - using all available layers: {target_layers}")
            except:
                target_layers = [0]
                logger.warning("Could not determine layers - using default layer 0")
        
        logger.debug(f"Getting polygons for layers: {target_layers}")
        
        # Get polygons from specified layers only
        all_polygons = self.initial_model.get_polygons(target_layers)
        
        if not all_polygons:
            logger.warning(f"No polygons found for layers {target_layers}")
            return []
        
        # Filter by coordinate intersection
        visible_polygons = []
        frame_xmin, frame_ymin, frame_xmax, frame_ymax = self.current_frame
        
        for polygon in all_polygons:
            if len(polygon) >= 3:
                # Check if polygon intersects current frame coordinates
                poly_xmin, poly_ymin = np.min(polygon, axis=0)
                poly_xmax, poly_ymax = np.max(polygon, axis=0)
                
                # Coordinate-based intersection test
                if not (poly_xmax < frame_xmin or poly_xmin > frame_xmax or
                        poly_ymax < frame_ymin or poly_ymin > frame_ymax):
                    visible_polygons.append(polygon)
        
        logger.debug(f"Found {len(visible_polygons)}/{len(all_polygons)} polygons in frame")
        return visible_polygons

    def get_transformed_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get bounds of the current viewing frame (frame coordinates).
        
        Returns:
            Current frame bounds (xmin, ymin, xmax, ymax)
        """
        return tuple(self.current_frame)
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata including frame transformation information.
        
        Returns:
            Combined metadata from initial model and frame transformations
        """
        metadata = self.initial_model.get_metadata().copy()
        metadata.update({
            'frame_transformations': self.get_transform_parameters(),
            'original_bounds': self.initial_model.bounds,
            'current_frame_bounds': self.get_transformed_bounds(),
            'required_layers': self.required_layers,  # Ensure layers are saved
            'frame_info': {
                'original_frame': self.original_frame,
                'current_frame': self.current_frame,
                'frame_center': self._calculate_frame_center(),
                'original_frame_center': self._calculate_original_frame_center()
            }
        })
        return metadata
    
    def __str__(self) -> str:
        """String representation of the frame-based aligned GDS model."""
        return (
            f"AlignedGdsModel("
            f"base={self.initial_model}, "
            f"frame_tx={self.frame_translation['tx']:.2f}, "
            f"frame_ty={self.frame_translation['ty']:.2f}, "
            f"frame_scale={self.frame_scale:.2f}, "
            f"frame_rot90={self.frame_rotation_90}°, "
            f"residual_rot={self.residual_rotation:.2f}°, "
            f"current_frame={[round(x, 2) for x in self.current_frame]}"
            f")"
        )
    
    def __repr__(self) -> str:
        """Detailed representation of the aligned GDS model."""
        return self.__str__()

    def to_bitmap(self, resolution: Tuple[int, int] = (1024, 666), layers: Optional[List[int]] = None) -> np.ndarray:
        """
        Convert frame to bitmap using same calculations as GDS generator.
        """
        if layers is None and self.required_layers is None:
            raise ValueError("Must specify layers for rendering - no defaults")
        
        target_layers = layers if layers is not None else self.required_layers
        
        width, height = resolution
        
        # Create white background image
        bitmap = np.full((height, width), 255, dtype=np.uint8)
        
        # Get visible polygons for specified layers only
        visible_polygons = self.get_polygons_in_frame(target_layers)
        
        if not visible_polygons:
            logger.warning("No visible polygons to render")
            return bitmap
        
        # Use current frame bounds (coordinate-based)
        frame_bounds = self.current_frame
        if frame_bounds == [0, 0, 0, 0]:
            logger.warning("Invalid frame bounds for rendering")
            return bitmap
        
        frame_xmin, frame_ymin, frame_xmax, frame_ymax = frame_bounds
        frame_width = frame_xmax - frame_xmin
        frame_height = frame_ymax - frame_ymin
        
        if frame_width <= 0 or frame_height <= 0:
            logger.warning("Invalid frame dimensions for rendering")
            return bitmap
        
        # Use same calculation as GDS generator
        base_scale = min(width / frame_width, height / frame_height)
        zoom_factor = self.frame_scale
        scale = base_scale * zoom_factor
        
        # Calculate scaled dimensions
        scaled_width = frame_width * scale
        scaled_height = frame_height * scale
        
        # Use pixel movement directly - 1 pixel = constant GDS distance regardless of zoom
        move_x, move_y = self._pixel_translation
        
        # Calculate center position with same formula as GDS generator
        center_x_pixels = width // 2 + move_x
        center_y_pixels = height // 2 + move_y
        
        # Calculate offset to center the scaled image
        offset_x = center_x_pixels - scaled_width // 2
        offset_y = center_y_pixels - scaled_height // 2
        
        # Render each polygon using same approach as GDS generator
        rendered_count = 0
        for polygon in visible_polygons:
            if len(polygon) < 3:
                continue
            
            # Transform to pixel coordinates (same as GDS generator)
            norm_poly = (polygon - [frame_xmin, frame_ymin]) * scale
            int_poly = np.round(norm_poly).astype(np.int32)
            int_poly += [int(offset_x), int(offset_y)]
            
            # Flip Y coordinate (image coordinate system)
            int_poly[:, 1] = height - 1 - int_poly[:, 1]
            
            # Clip to image bounds
            int_poly = np.clip(int_poly, [0, 0], [width-1, height-1])
            
            # Draw polygon
            if len(int_poly) >= 3:
                try:
                    import cv2
                    cv2.fillPoly(bitmap, [int_poly], color=(0,))
                    rendered_count += 1
                except ImportError:
                    self._fill_polygon_numpy(bitmap, int_poly)
                    rendered_count += 1
        
        logger.debug(f"Rendered {rendered_count}/{len(visible_polygons)} polygons using frame {[round(x, 2) for x in frame_bounds]}")
        return bitmap
    
    def _fill_polygon_numpy(self, bitmap: np.ndarray, polygon: np.ndarray) -> None:
        """
        Fallback polygon filling using numpy (basic scanline algorithm).
        
        Args:
            bitmap: Target bitmap array
            polygon: Integer polygon coordinates
        """
        if len(polygon) < 3:
            return
            
        height, width = bitmap.shape
        
        # Get polygon bounds
        min_y = max(0, int(np.min(polygon[:, 1])))
        max_y = min(height - 1, int(np.max(polygon[:, 1])))
        
        # Simple scanline filling
        for y in range(min_y, max_y + 1):
            # Find intersections with horizontal line at y
            intersections = []
            
            for i in range(len(polygon)):
                j = (i + 1) % len(polygon)
                y1, y2 = polygon[i, 1], polygon[j, 1]
                
                if y1 != y2 and min(y1, y2) <= y <= max(y1, y2):
                    # Calculate intersection x coordinate
                    x1, x2 = polygon[i, 0], polygon[j, 0]
                    x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                    intersections.append(int(x))
            
            # Sort intersections and fill between pairs
            intersections.sort()
            for i in range(0, len(intersections), 2):
                if i + 1 < len(intersections):
                    x_start = max(0, intersections[i])
                    x_end = min(width - 1, intersections[i + 1])
                    if x_start <= x_end:
                        bitmap[y, x_start:x_end + 1] = 0

    def to_bitmap_with_residual_rotation(self, resolution: Tuple[int, int] = (1024, 666), 
                                       layers: Optional[List[int]] = None) -> np.ndarray:
        """
        Generate bitmap with residual rotation - REQUIRES explicit layers.
        """
        if layers is None and self.required_layers is None:
            raise ValueError("Must specify layers for rendering")
        
        target_layers = layers if layers is not None else self.required_layers
        
        # First generate 90°-aligned bitmap
        bitmap = self.to_bitmap(resolution, target_layers)
        
        # Then apply image-level rotation for residual angle
        if abs(self.residual_rotation) > 0.001:
            bitmap = self._apply_bitmap_rotation(bitmap, self.residual_rotation)
        
        return bitmap
    
    def _apply_bitmap_rotation(self, bitmap: np.ndarray, angle_degrees: float) -> np.ndarray:
        """
        Apply center-based affine rotation to bitmap.
        
        Args:
            bitmap: Input bitmap array
            angle_degrees: Rotation angle in degrees
            
        Returns:
            Rotated bitmap
        """
        try:
            import cv2
            
            height, width = bitmap.shape
            center = (width // 2, height // 2)
            
            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
            
            # Apply rotation with white background
            rotated = cv2.warpAffine(
                bitmap, 
                rotation_matrix, 
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255  # White background
            )
            
            logger.debug(f"Applied residual rotation: {angle_degrees:.2f}°")
            return rotated
            
        except ImportError:
            logger.warning("OpenCV not available, skipping residual rotation")
            return bitmap
    
    def render_final_bitmap(self, resolution: Tuple[int, int] = (1024, 666), 
                          layers: Optional[List[int]] = None, 
                          apply_residual: bool = True) -> np.ndarray:
        """
        Render final bitmap with all transformations - REQUIRES explicit layers.
        """
        if layers is None and self.required_layers is None:
            raise ValueError("Must specify layers for rendering")
        
        if apply_residual and abs(self.residual_rotation) > 0.001:
            return self.to_bitmap_with_residual_rotation(resolution, layers)
        else:
            return self.to_bitmap(resolution, layers)
        
    def to_bitmap_optimized(self, resolution: Tuple[int, int] = (1024, 666), 
                          layers: Optional[List[int]] = None,
                          anti_alias: bool = True,
                          clip_bounds: bool = True) -> np.ndarray:
        """
        Optimized frame-based rendering - REQUIRES explicit layers.
        """
        if layers is None and self.required_layers is None:
            raise ValueError("Must specify layers for rendering")
        
        target_layers = layers if layers is not None else self.required_layers
        
        width, height = resolution
        
        # Create bitmap with higher resolution for anti-aliasing if enabled
        aa_factor = 2 if anti_alias else 1
        render_width = width * aa_factor
        render_height = height * aa_factor
        
        bitmap = np.full((render_height, render_width), 255, dtype=np.uint8)
        
        # Get original polygons that intersect the current frame
        polygons = self.get_polygons_in_frame(target_layers)
        
        if not polygons:
            if anti_alias:
                return self._downsample_bitmap(bitmap, aa_factor)
            return bitmap[:height, :width]
        
        # Use current frame bounds for coordinate mapping
        frame_bounds = self.current_frame
        if frame_bounds == [0, 0, 0, 0]:
            if anti_alias:
                return self._downsample_bitmap(bitmap, aa_factor)
            return bitmap[:height, :width]
        
        xmin, ymin, xmax, ymax = frame_bounds
        frame_width = xmax - xmin
        frame_height = ymax - ymin
        
        if frame_width <= 0 or frame_height <= 0:
            if anti_alias:
                return self._downsample_bitmap(bitmap, aa_factor)
            return bitmap[:height, :width]
        
        # Calculate scaling with padding for rotated bounds
        padding_factor = 1.2  # Extra space for rotation
        scale_x = (render_width / padding_factor) / frame_width
        scale_y = (render_height / padding_factor) / frame_height
        scale = min(scale_x, scale_y)
        
        # Calculate centering offsets
        scaled_width = frame_width * scale
        scaled_height = frame_height * scale
        offset_x = (render_width - scaled_width) / 2
        offset_y = (render_height - scaled_height) / 2
        
        # Render polygons with optimization using frame-to-bitmap mapping
        rendered_count = 0
        for polygon in polygons:
            if len(polygon) < 3:
                continue
            
            # Quick bounds check for clipping (polygon already filtered by frame intersection)
            if clip_bounds and not self._polygon_intersects_render_area(
                polygon, frame_bounds, (render_width, render_height), scale, offset_x, offset_y
            ):
                continue
            
            # Transform from GDS coordinates to bitmap coordinates
            canvas_poly = (polygon - np.array([xmin, ymin])) * scale
            canvas_poly += np.array([offset_x, offset_y])
            
            # Flip Y-axis
            canvas_poly[:, 1] = render_height - canvas_poly[:, 1]
            
            # Convert to integer coordinates
            int_poly = np.round(canvas_poly).astype(np.int32)
            
            # Clip to canvas bounds
            int_poly[:, 0] = np.clip(int_poly[:, 0], 0, render_width - 1)
            int_poly[:, 1] = np.clip(int_poly[:, 1], 0, render_height - 1)
            
            # Draw polygon
            try:
                import cv2
                cv2.fillPoly(bitmap, [int_poly], 0)
                rendered_count += 1
            except ImportError:
                self._fill_polygon_numpy(bitmap, int_poly)
                rendered_count += 1
        
        logger.debug(f"Optimized frame-based rendering: {rendered_count}/{len(polygons)} polygons")
        
        # Downsample for anti-aliasing
        if anti_alias:
            return self._downsample_bitmap(bitmap, aa_factor)
        
        return bitmap[:height, :width]
    
    def _polygon_intersects_render_area(self, polygon: np.ndarray, frame_bounds: List[float], 
                                       canvas_size: Tuple[int, int], scale: float, 
                                       offset_x: float, offset_y: float) -> bool:
        """
        Quick check if polygon intersects with rendered canvas area.
        
        Args:
            polygon: Polygon coordinates in GDS space
            frame_bounds: Current frame bounds [xmin, ymin, xmax, ymax]
            canvas_size: Canvas (width, height)
            scale: GDS-to-canvas scaling factor
            offset_x: Canvas X offset
            offset_y: Canvas Y offset
            
        Returns:
            True if polygon might be visible on canvas
        """
        if len(polygon) == 0:
            return False
        
        # Transform polygon bounds to canvas space
        xmin, ymin, xmax, ymax = frame_bounds
        poly_xmin, poly_ymin = np.min(polygon, axis=0)
        poly_xmax, poly_ymax = np.max(polygon, axis=0)
        
        # Convert to canvas coordinates
        canvas_xmin = (poly_xmin - xmin) * scale + offset_x
        canvas_xmax = (poly_xmax - xmin) * scale + offset_x
        canvas_ymin = (poly_ymin - ymin) * scale + offset_y
        canvas_ymax = (poly_ymax - ymin) * scale + offset_y
        
        canvas_width, canvas_height = canvas_size
        
        # Check if polygon bounds intersect canvas
        return not (canvas_xmax < 0 or canvas_xmin > canvas_width or
                   canvas_ymax < 0 or canvas_ymin > canvas_height)
    
    def _downsample_bitmap(self, bitmap: np.ndarray, factor: int) -> np.ndarray:
        """
        Downsample bitmap for anti-aliasing.
        
        Args:
            bitmap: High-resolution bitmap
            factor: Downsampling factor
            
        Returns:
            Downsampled bitmap
        """
        if factor <= 1:
            return bitmap
        
        height, width = bitmap.shape
        new_height = height // factor
        new_width = width // factor
        
        # Simple box filter downsampling
        downsampled = np.zeros((new_height, new_width), dtype=np.uint8)
        
        for y in range(new_height):
            for x in range(new_width):
                # Average the factor x factor block
                block = bitmap[y*factor:(y+1)*factor, x*factor:(x+1)*factor]
                downsampled[y, x] = np.mean(block)
        
        return downsampled

    def serialize_alignment_data(self) -> Dict[str, Any]:
        """
        Export alignment data for serialization.
        
        Returns:
            Dictionary with frame transformation parameters and metadata
        """
        return {
            'frame_transformation': self.get_transform_parameters(),
            'residual_rotation': self.residual_rotation,
            'alignment_metadata': {
                'original_bounds': self.initial_model.bounds,
                'original_frame': self.original_frame.copy(),
                'current_frame': self.current_frame.copy(),
                'pixel_size': self.pixel_size,
                'scaling_factor': self.initial_model.get_scaling_factor(),
                'available_layers': self.initial_model.get_layers(),
                'structure_count': self.initial_model.get_structure_count(),
                'gds_metadata': {
                    'file_path': self.initial_model.gds_path.name,
                    'cell_name': self.initial_model.cell.name if self.initial_model.cell else None,
                    'unit': self.initial_model.unit,
                    'precision': self.initial_model.precision
                }
            }
        }
    
    def save_alignment_data(self, filepath: str) -> None:
        """
        Save alignment data to JSON file (excludes polygon data for performance).
        
        Args:
            filepath: Output file path
        """
        import json
        
        data = self.serialize_alignment_data()
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Alignment data saved to {output_path}")
    
    @classmethod
    def load_alignment_data(cls, filepath: str) -> Dict[str, Any]:
        """
        Load alignment data from JSON file.
        
        Args:
            filepath: JSON file path
            
        Returns:
            Loaded alignment data dictionary
        """
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def apply_loaded_alignment(self, alignment_data: Dict[str, Any]) -> None:
        """
        Apply alignment data loaded from file.
        
        Args:
            alignment_data: Dictionary with alignment parameters
        """
        # New format exclusively
        if 'frame_transformations' in alignment_data:
            params = alignment_data['frame_transformations']
            
            self.set_translation(params.get('tx', 0.0), params.get('ty', 0.0))
            self.set_scale_center_relative(params.get('scale', 1.0))
            self.set_rotation_90(params.get('rotation_90', 0))
            self.set_residual_rotation(params.get('residual_rotation', 0.0))
            
            logger.info("Applied loaded frame-based alignment data")
        else:
            logger.warning("No 'frame_transformations' data found in alignment file.")

    @classmethod
    def from_alignment_file(cls, gds_path: str, alignment_file: str) -> 'AlignedGdsModel':
        """
        Create AlignedGdsModel from a GDS file and a saved alignment JSON file.

        This method first loads the alignment data to extract the specific feature bounds
        and layers, ensuring the model is focused on the correct GDS structure before
        applying the saved transformations.

        Args:
            gds_path: Path to the GDS file.
            alignment_file: Path to the alignment data JSON file.

        Returns:
            An AlignedGdsModel instance with the specified alignment applied.
        
        Raises:
            ValueError: If the alignment file is missing 'original_frame' or 'required_layers'.
        """
        # Load alignment data first to get the context (bounds and layers)
        alignment_data = cls.load_alignment_data(alignment_file)

        # Extract the specific feature bounds and layers from the saved data
        transform_params = alignment_data.get('frame_transformations', {})
        feature_bounds = transform_params.get('original_frame')
        required_layers = alignment_data.get('required_layers')

        if feature_bounds is None or required_layers is None:
            raise ValueError(
                "Alignment file is missing required 'original_frame' or 'required_layers'. "
                "Cannot reconstruct the model's focused state."
            )

        # Create the base initial model
        initial_model = InitialGdsModel(gds_path)

        # Instantiate the AlignedGdsModel with the correct feature focus
        aligned_model = cls(
            initial_model,
            feature_bounds=tuple(feature_bounds),
            required_layers=required_layers
        )

        # Apply the saved transformations to the correctly configured model
        aligned_model.apply_loaded_alignment(alignment_data)

        return aligned_model

    def validate_setup(self) -> bool:
        """
        Validate that coordinate bounds and layers are properly specified.
        """
        if self.current_frame == [0, 0, 0, 0]:
            logger.error("Invalid frame bounds - all zeros")
            return False
        
        if self.required_layers is None or len(self.required_layers) == 0:
            logger.error("No layers specified for rendering")
            return False
        
        frame_width = self.current_frame[2] - self.current_frame[0]
        frame_height = self.current_frame[3] - self.current_frame[1]
        
        if frame_width <= 0 or frame_height <= 0:
            logger.error(f"Invalid frame dimensions: {frame_width} x {frame_height}")
            return False
        
        logger.info(f"Model validation passed - Frame: {self.current_frame}, Layers: {self.required_layers}")
        return True


# Convenience functions (moved outside class)
def create_aligned_model(gds_path: str, pixel_size: float = None) -> 'AlignedGdsModel':
    """
    Create an AlignedGdsModel from a GDS file path (legacy compatibility).
    Uses full GDS bounds and all available layers.
    
    Args:
        gds_path: Path to the GDS file
        pixel_size: GDS units per pixel (auto-calculated if None)
        
    Returns:
        AlignedGdsModel instance
    """
    initial_model = InitialGdsModel(gds_path)
    
    # Use full bounds and all layers for backward compatibility
    all_layers = initial_model.get_layers()
    
    return AlignedGdsModel(
        initial_model, 
        pixel_size=pixel_size,
        feature_bounds=initial_model.bounds,
        required_layers=all_layers if all_layers else [0]
    )


def create_aligned_model_for_coordinates(gds_path: str, 
                                        feature_bounds: Tuple[float, float, float, float],
                                        layers: List[int],
                                        pixel_size: float = None) -> 'AlignedGdsModel':
    """
    Create AlignedGdsModel for specific coordinates and layers ONLY.
    
    Args:
        gds_path: Path to GDS file
        feature_bounds: (xmin, ymin, xmax, ymax) coordinates of feature in GDS units
        layers: Specific layers to render (required)
        pixel_size: GDS units per pixel
        
    Returns:
        AlignedGdsModel focused on specified coordinates
    """
    if not layers:
        raise ValueError("Must specify at least one layer")
    
    initial_model = InitialGdsModel(gds_path)
    return AlignedGdsModel(
        initial_model, 
        pixel_size=pixel_size,
        feature_bounds=feature_bounds,
        required_layers=layers
    )


def create_aligned_model_for_structure(gds_path: str, structure_id: int, 
                                     pixel_size: float = None) -> 'AlignedGdsModel':
    """
    DEPRECATED: Use create_aligned_model_for_coordinates instead.
    """
    raise DeprecationWarning("Use create_aligned_model_for_coordinates with explicit bounds and layers")
