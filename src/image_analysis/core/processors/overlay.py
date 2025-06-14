import numpy as np
import cv2
from typing import Optional, Tuple, Union, List, Dict
from matplotlib.path import Path

from image_analysis.core.models.sem_image import SEMImage


class OverlayRenderer:
    """
    Renders extracted structure binary images onto SEM images with coordinate transformation
    and multiple rendering modes.
    """
    
    def __init__(self, sem_image: SEMImage, initial_gds_bounds: Tuple[float, float, float, float]):
        """
        Initialize overlay renderer with SEM image and initial GDS bounds.
        
        Args:
            sem_image: SEMImage instance providing image dimensions and scale
            initial_gds_bounds: (xmin, ymin, xmax, ymax) of original GDS extraction area
        """
        if not sem_image.is_loaded if hasattr(sem_image, 'is_loaded') else True:
            raise ValueError("SEMImage must be loaded")
            
        self.sem_image = sem_image
        self.sem_height, self.sem_width = self.sem_image.shape
        self.sem_pixel_size = self.sem_image.pixel_size
        
        # Store initial GDS bounds and calculate center
        self.gds_xmin, self.gds_ymin, self.gds_xmax, self.gds_ymax = initial_gds_bounds
        self.gds_center_x = (self.gds_xmin + self.gds_xmax) / 2
        self.gds_center_y = (self.gds_ymin + self.gds_ymax) / 2
        self.gds_width = self.gds_xmax - self.gds_xmin
        self.gds_height = self.gds_ymax - self.gds_ymin
        
        # Calculate initial coordinate-to-pixel transformation
        self.initial_scale_x = 1024 / self.gds_width  # pixels per coordinate unit
        self.initial_scale_y = 666 / self.gds_height   # pixels per coordinate unit
    
    def _apply_transformations(self, rotation_deg: float = 0.0, zoom_factor: float = 1.0, 
                              pan_x: float = 0.0, pan_y: float = 0.0) -> Tuple[float, float, float, float]:
        """
        Apply rotation, zoom, and pan transformations to get new frame bounds.
        
        Args:
            rotation_deg: Rotation angle in degrees
            zoom_factor: Zoom factor (1.1 = 110% = zoom in 10%)
            pan_x: Pan offset in coordinate units
            pan_y: Pan offset in coordinate units
            
        Returns:
            New frame bounds (xmin, ymin, xmax, ymax) in coordinate space
        """
        # Step 1: Apply rotation around center point
        rotation_rad = np.radians(rotation_deg)
        cos_r = np.cos(rotation_rad)
        sin_r = np.sin(rotation_rad)
        
        # Get current frame relative to center
        rel_width = self.gds_width / 2
        rel_height = self.gds_height / 2
        
        # Rotate the frame corners around center
        corners = np.array([
            [-rel_width, -rel_height],  # bottom-left
            [rel_width, -rel_height],   # bottom-right
            [rel_width, rel_height],    # top-right
            [-rel_width, rel_height]    # top-left
        ])
        
        # Apply rotation matrix
        rotated_corners = np.array([
            [cos_r * x - sin_r * y, sin_r * x + cos_r * y] 
            for x, y in corners
        ])
        
        # Find new bounding box after rotation
        rotated_xmin = np.min(rotated_corners[:, 0])
        rotated_xmax = np.max(rotated_corners[:, 0])
        rotated_ymin = np.min(rotated_corners[:, 1])
        rotated_ymax = np.max(rotated_corners[:, 1])
        
        rotated_width = rotated_xmax - rotated_xmin
        rotated_height = rotated_ymax - rotated_ymin
        
        # Step 2: Apply zoom (zoom in means smaller frame, zoom out means larger frame)
        # 110% zoom means we see 90.9% of the area (1/1.1)
        zoom_width = rotated_width / zoom_factor
        zoom_height = rotated_height / zoom_factor
        
        # Step 3: Apply pan and calculate final bounds
        final_center_x = self.gds_center_x + pan_x
        final_center_y = self.gds_center_y + pan_y
        
        final_xmin = final_center_x - zoom_width / 2
        final_xmax = final_center_x + zoom_width / 2
        final_ymin = final_center_y - zoom_height / 2
        final_ymax = final_center_y + zoom_height / 2
        
        return final_xmin, final_ymin, final_xmax, final_ymax
    
    def _coordinate_to_pixel(self, coord_x: float, coord_y: float, 
                           frame_bounds: Tuple[float, float, float, float]) -> Tuple[int, int]:
        """
        Convert coordinate space position to pixel position in 1024x666 frame.
        
        Args:
            coord_x, coord_y: Position in coordinate space
            frame_bounds: Current frame bounds (xmin, ymin, xmax, ymax)
            
        Returns:
            Pixel position (x, y) in 1024x666 space
        """
        frame_xmin, frame_ymin, frame_xmax, frame_ymax = frame_bounds
        frame_width = frame_xmax - frame_xmin
        frame_height = frame_ymax - frame_ymin
        
        # Convert to normalized position within frame (0 to 1)
        norm_x = (coord_x - frame_xmin) / frame_width
        norm_y = (coord_y - frame_ymin) / frame_height
        
        # Convert to pixel position (note: y-axis is flipped for image coordinates)
        pixel_x = int(norm_x * 1024)
        pixel_y = int((1.0 - norm_y) * 666)
        
        return pixel_x, pixel_y
    
    def _pixel_to_sem_coordinate(self, pixel_x: int, pixel_y: int,
                               frame_bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """
        Convert pixel position back to SEM coordinate space.
        
        Args:
            pixel_x, pixel_y: Position in 1024x666 pixel space
            frame_bounds: Current frame bounds
            
        Returns:
            Position in SEM coordinate space
        """
        frame_xmin, frame_ymin, frame_xmax, frame_ymax = frame_bounds
        frame_width = frame_xmax - frame_xmin
        frame_height = frame_ymax - frame_ymin
        
        # Convert pixel to normalized position
        norm_x = pixel_x / 1024.0
        norm_y = 1.0 - (pixel_y / 666.0)  # Flip y-axis back
        
        # Convert to coordinate space
        coord_x = frame_xmin + norm_x * frame_width
        coord_y = frame_ymin + norm_y * frame_height
        
        # Convert to SEM pixel coordinates
        sem_pixel_x = (coord_x - self.gds_xmin) * (self.sem_width / self.gds_width)
        sem_pixel_y = self.sem_height - (coord_y - self.gds_ymin) * (self.sem_height / self.gds_height)
        
        return sem_pixel_x, sem_pixel_y
    
    def render_overlay(self, 
                      structures: Dict[str, Tuple[np.ndarray, Tuple[int, int]]],
                      rotation_deg: float = 0.0,
                      zoom_factor: float = 1.0,
                      pan_x: float = 0.0,
                      pan_y: float = 0.0,
                      mode: str = 'filled',
                      antialiased: bool = True,
                      line_width: int = 1) -> np.ndarray:
        """
        Render structure binary images onto SEM image coordinates with transformations.
        
        Args:
            structures: Dictionary mapping structure names to (binary_image, (x_offset, y_offset))
            rotation_deg: Rotation angle in degrees
            zoom_factor: Zoom factor (1.1 = zoom in 10%)
            pan_x, pan_y: Pan offsets in coordinate units
            mode: Rendering mode ('filled', 'edges', 'mask')
            antialiased: Enable anti-aliasing for smoother edges
            line_width: Line width for edge rendering
            
        Returns:
            NumPy array with rendered overlay on SEM coordinates
        """
        output = np.zeros((self.sem_height, self.sem_width), dtype=np.uint8)
        
        # Calculate transformed frame bounds
        frame_bounds = self._apply_transformations(rotation_deg, zoom_factor, pan_x, pan_y)
        
        for structure_name, (binary_image, pixel_coords) in structures.items():
            structure_mask = self._render_structure(binary_image, pixel_coords, frame_bounds,
                                                  mode, antialiased, line_width)
            output = np.maximum(output, structure_mask)
        
        return output
    
    def _render_structure(self, 
                         binary_image: np.ndarray,
                         pixel_coords: Tuple[int, int],
                         frame_bounds: Tuple[float, float, float, float],
                         mode: str,
                         antialiased: bool,
                         line_width: int) -> np.ndarray:
        """Render a single structure's binary image with coordinate transformation."""
        structure_output = np.zeros((self.sem_height, self.sem_width), dtype=np.uint8)
        
        pixel_x, pixel_y = pixel_coords
        binary_h, binary_w = binary_image.shape
        
        # Transform each pixel of the binary image to SEM coordinates
        for y in range(binary_h):
            for x in range(binary_w):
                if binary_image[y, x] > 0:  # Only process non-zero pixels
                    # Convert binary image pixel to SEM coordinates
                    sem_x, sem_y = self._pixel_to_sem_coordinate(
                        pixel_x + x, pixel_y + y, frame_bounds
                    )
                    
                    # Check if within SEM bounds and set pixel
                    sem_x_int = int(round(sem_x))
                    sem_y_int = int(round(sem_y))
                    
                    if (0 <= sem_x_int < self.sem_width and 
                        0 <= sem_y_int < self.sem_height):
                        
                        if mode == 'filled' or mode == 'mask':
                            structure_output[sem_y_int, sem_x_int] = 255
                        elif mode == 'edges':
                            # For edges, check if this is a boundary pixel
                            if self._is_edge_pixel(binary_image, x, y):
                                structure_output[sem_y_int, sem_x_int] = 255
        
        return structure_output
    
    def _is_edge_pixel(self, binary_image: np.ndarray, x: int, y: int) -> bool:
        """Check if pixel is on the edge of a binary shape."""
        h, w = binary_image.shape
        
        # Check 8-connected neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = x + dx, y + dy
                if (0 <= nx < w and 0 <= ny < h):
                    if binary_image[ny, nx] == 0:  # Found a background neighbor
                        return True
                else:
                    return True  # Edge of image
        
        return False
    
    def render_composite(self, 
                        structures: Dict[str, Tuple[np.ndarray, Tuple[int, int]]],
                        rotation_deg: float = 0.0,
                        zoom_factor: float = 1.0,
                        pan_x: float = 0.0,
                        pan_y: float = 0.0,
                        background_image: Optional[np.ndarray] = None,
                        overlay_color: Tuple[int, int, int] = (255, 100, 150),
                        transparency: float = 0.7) -> np.ndarray:
        """
        Render composite overlay on background image with transformations.
        
        Args:
            structures: Dictionary mapping structure names to (binary_image, coordinates)
            rotation_deg: Rotation angle in degrees
            zoom_factor: Zoom factor
            pan_x, pan_y: Pan offsets
            background_image: Background image (uses SEM if None)
            overlay_color: RGB color for overlay
            transparency: Overlay transparency (0.0 to 1.0)
            
        Returns:
            RGB composite image
        """
        if background_image is None:
            background = self.sem_image.to_array()
            background = (background * 255).astype(np.uint8)
        else:
            background = background_image.copy()
        
        if len(background.shape) == 2:
            background = np.stack([background] * 3, axis=-1)
        
        overlay_mask = self.render_overlay(structures, rotation_deg, zoom_factor, 
                                         pan_x, pan_y, mode='filled')
        
        result = background.copy()
        mask_indices = overlay_mask > 0
        
        for channel in range(3):
            result[:, :, channel][mask_indices] = (
                background[:, :, channel][mask_indices] * (1 - transparency) +
                overlay_color[channel] * transparency
            ).astype(np.uint8)
        
        return result
    
    def get_transformed_bounds(self, rotation_deg: float = 0.0, zoom_factor: float = 1.0,
                              pan_x: float = 0.0, pan_y: float = 0.0) -> Tuple[float, float, float, float]:
        """
        Get the coordinate bounds after applying transformations.
        
        Returns:
            Transformed bounds (xmin, ymin, xmax, ymax) in coordinate space
        """
        return self._apply_transformations(rotation_deg, zoom_factor, pan_x, pan_y)
    
    def export_overlay_image(self, 
                           output_path: str,
                           structures: Dict[str, Tuple[np.ndarray, Tuple[int, int]]],
                           rotation_deg: float = 0.0,
                           zoom_factor: float = 1.0,
                           pan_x: float = 0.0,
                           pan_y: float = 0.0,
                           mode: str = 'composite',
                           **kwargs) -> None:
        """
        Export overlay as image file with transformations.
        
        Args:
            output_path: Path for output image
            structures: Dictionary mapping structure names to (binary_image, coordinates)
            rotation_deg: Rotation angle in degrees
            zoom_factor: Zoom factor
            pan_x, pan_y: Pan offsets
            mode: Export mode ('overlay', 'composite', 'mask')
            **kwargs: Additional arguments for rendering
        """
        if mode == 'composite':
            result = self.render_composite(structures, rotation_deg, zoom_factor,
                                         pan_x, pan_y, **kwargs)
        elif mode == 'overlay':
            result = self.render_overlay(structures, rotation_deg, zoom_factor,
                                       pan_x, pan_y, **kwargs)
            if len(result.shape) == 2:
                result = np.stack([result] * 3, axis=-1)
        elif mode == 'mask':
            result = self.render_overlay(structures, rotation_deg, zoom_factor,
                                       pan_x, pan_y, mode='mask', **kwargs)
            result = np.stack([result] * 3, axis=-1)
        else:
            raise ValueError(f"Unknown export mode: {mode}")
        
        cv2.imwrite(output_path, result)


class OverlayError(Exception):
    """Custom exception for overlay rendering errors."""
    pass