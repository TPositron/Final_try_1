import numpy as np
import cv2
from typing import Optional, Tuple, Union, List
from matplotlib.path import Path

from image_analysis.core.models.sem_image import SEMImage
from image_analysis.core.models.gds_model import GDSModel


class OverlayRenderer:
    """
    Renders parsed GDS shapes onto SEM image arrays with coordinate transformation
    and multiple rendering modes.
    """
    
    def __init__(self, sem_image: SEMImage, gds_model: GDSModel):
        """
        Initialize overlay renderer with SEM image and GDS model.
        
        Args:
            sem_image: SEMImage instance providing image dimensions and scale
            gds_model: GDSModel instance providing geometry to overlay
        """
        if not sem_image.is_loaded if hasattr(sem_image, 'is_loaded') else True:
            raise ValueError("SEMImage must be loaded")
        if not gds_model.is_loaded():
            raise ValueError("GDSModel must be loaded")
            
        self.sem_image = sem_image
        self.gds_model = gds_model
        self._setup_coordinate_transform()
    
    def _setup_coordinate_transform(self):
        """Setup coordinate transformation from GDS to SEM pixel coordinates."""
        self.sem_height, self.sem_width = self.sem_image.shape
        self.sem_pixel_size = self.sem_image.pixel_size
        
        gds_bounds = self.gds_model.get_bounds()
        self.gds_xmin, self.gds_ymin, self.gds_xmax, self.gds_ymax = gds_bounds
        
        self.gds_width = self.gds_xmax - self.gds_xmin
        self.gds_height = self.gds_ymax - self.gds_ymin
        
        self.scale_x = self.sem_width / (self.gds_width / self.sem_pixel_size)
        self.scale_y = self.sem_height / (self.gds_height / self.sem_pixel_size)
    
    def gds_to_pixel(self, gds_coords: np.ndarray, 
                     transform_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert GDS coordinates to SEM pixel coordinates.
        
        Args:
            gds_coords: Array of GDS coordinates (N, 2)
            transform_matrix: Optional 2x3 transformation matrix
            
        Returns:
            Array of pixel coordinates (N, 2)
        """
        coords = gds_coords.copy()
        
        if transform_matrix is not None:
            ones = np.ones((coords.shape[0], 1))
            coords_homogeneous = np.hstack([coords, ones])
            coords = coords_homogeneous @ transform_matrix.T
        
        pixel_x = (coords[:, 0] - self.gds_xmin) * self.scale_x
        pixel_y = self.sem_height - (coords[:, 1] - self.gds_ymin) * self.scale_y
        
        return np.column_stack([pixel_x, pixel_y])
    
    def render_overlay(self, 
                      layers: Optional[List[int]] = None,
                      mode: str = 'filled',
                      transform_matrix: Optional[np.ndarray] = None,
                      antialiased: bool = True,
                      line_width: int = 1) -> np.ndarray:
        """
        Render GDS shapes onto SEM image coordinates.
        
        Args:
            layers: List of layer numbers to render (None for all)
            mode: Rendering mode ('filled', 'edges', 'mask')
            transform_matrix: Optional 2x3 affine transformation matrix
            antialiased: Enable anti-aliasing for smoother edges
            line_width: Line width for edge rendering
            
        Returns:
            NumPy array with rendered overlay
        """
        if layers is None:
            layers = self.gds_model.get_layers()
        
        output = np.zeros((self.sem_height, self.sem_width), dtype=np.uint8)
        
        for layer in layers:
            if not self.gds_model.has_layer(layer):
                continue
                
            shapes = self.gds_model.get_shapes(layer)
            layer_mask = self._render_layer(shapes, mode, transform_matrix, 
                                          antialiased, line_width)
            output = np.maximum(output, layer_mask)
        
        return output
    
    def _render_layer(self, 
                     shapes: List[np.ndarray],
                     mode: str,
                     transform_matrix: Optional[np.ndarray],
                     antialiased: bool,
                     line_width: int) -> np.ndarray:
        """Render a single layer's shapes."""
        layer_output = np.zeros((self.sem_height, self.sem_width), dtype=np.uint8)
        
        for shape in shapes:
            if len(shape) < 3:
                continue
                
            pixel_coords = self.gds_to_pixel(shape, transform_matrix)
            
            if mode == 'filled':
                self._draw_filled_polygon(layer_output, pixel_coords, antialiased)
            elif mode == 'edges':
                self._draw_polygon_edges(layer_output, pixel_coords, line_width, antialiased)
            elif mode == 'mask':
                self._draw_antialiased_mask(layer_output, pixel_coords)
        
        return layer_output
    
    def _draw_filled_polygon(self, output: np.ndarray, coords: np.ndarray, antialiased: bool):
        """Draw filled polygon using OpenCV."""
        int_coords = np.round(coords).astype(np.int32)
        
        if antialiased:
            cv2.fillPoly(output, [int_coords], color=255, lineType=cv2.LINE_AA)
        else:
            cv2.fillPoly(output, [int_coords], color=255)
    
    def _draw_polygon_edges(self, output: np.ndarray, coords: np.ndarray, 
                           line_width: int, antialiased: bool):
        """Draw polygon edges using OpenCV."""
        int_coords = np.round(coords).astype(np.int32)
        
        line_type = cv2.LINE_AA if antialiased else cv2.LINE_8
        cv2.polylines(output, [int_coords], isClosed=True, 
                     color=255, thickness=line_width, lineType=line_type)
    
    def _draw_antialiased_mask(self, output: np.ndarray, coords: np.ndarray):
        """Draw high-quality antialiased mask using matplotlib path."""
        if len(coords) < 3:
            return
            
        y_indices, x_indices = np.mgrid[0:self.sem_height, 0:self.sem_width]
        points = np.column_stack([x_indices.ravel(), y_indices.ravel()])
        
        path = Path(coords)
        mask = path.contains_points(points)
        mask = mask.reshape(self.sem_height, self.sem_width)
        
        output[mask] = 255
    
    def render_composite(self, 
                        layers: Optional[List[int]] = None,
                        background_image: Optional[np.ndarray] = None,
                        overlay_color: Tuple[int, int, int] = (255, 100, 150),
                        transparency: float = 0.7,
                        transform_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Render composite overlay on background image.
        
        Args:
            layers: Layer numbers to render
            background_image: Background image (uses SEM if None)
            overlay_color: RGB color for overlay
            transparency: Overlay transparency (0.0 to 1.0)
            transform_matrix: Optional transformation matrix
            
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
        
        overlay_mask = self.render_overlay(layers, mode='filled', 
                                         transform_matrix=transform_matrix)
        
        result = background.copy()
        mask_indices = overlay_mask > 0
        
        for channel in range(3):
            result[:, :, channel][mask_indices] = (
                background[:, :, channel][mask_indices] * (1 - transparency) +
                overlay_color[channel] * transparency
            ).astype(np.uint8)
        
        return result
    
    def get_shape_bounds_in_pixels(self, layer: int, 
                                  transform_matrix: Optional[np.ndarray] = None) -> List[Tuple[int, int, int, int]]:
        """
        Get pixel bounding boxes for all shapes in a layer.
        
        Args:
            layer: Layer number
            transform_matrix: Optional transformation matrix
            
        Returns:
            List of (xmin, ymin, xmax, ymax) tuples in pixel coordinates
        """
        if not self.gds_model.has_layer(layer):
            return []
        
        shapes = self.gds_model.get_shapes(layer)
        bounds_list = []
        
        for shape in shapes:
            if len(shape) < 3:
                continue
                
            pixel_coords = self.gds_to_pixel(shape, transform_matrix)
            xmin = int(np.floor(pixel_coords[:, 0].min()))
            ymin = int(np.floor(pixel_coords[:, 1].min()))
            xmax = int(np.ceil(pixel_coords[:, 0].max()))
            ymax = int(np.ceil(pixel_coords[:, 1].max()))
            
            bounds_list.append((xmin, ymin, xmax, ymax))
        
        return bounds_list
    
    def create_region_mask(self, 
                          region_bounds: Tuple[int, int, int, int],
                          layers: Optional[List[int]] = None,
                          transform_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create binary mask for a specific region.
        
        Args:
            region_bounds: (xmin, ymin, xmax, ymax) in pixel coordinates
            layers: Layer numbers to include
            transform_matrix: Optional transformation matrix
            
        Returns:
            Binary mask array for the region
        """
        xmin, ymin, xmax, ymax = region_bounds
        
        region_width = xmax - xmin
        region_height = ymax - ymin
        
        if region_width <= 0 or region_height <= 0:
            return np.zeros((1, 1), dtype=np.uint8)
        
        full_mask = self.render_overlay(layers, mode='filled', 
                                      transform_matrix=transform_matrix)
        
        xmin = max(0, min(xmin, self.sem_width))
        ymin = max(0, min(ymin, self.sem_height))
        xmax = max(0, min(xmax, self.sem_width))
        ymax = max(0, min(ymax, self.sem_height))
        
        region_mask = full_mask[ymin:ymax, xmin:xmax]
        return region_mask
    
    def export_overlay_image(self, 
                           output_path: str,
                           layers: Optional[List[int]] = None,
                           mode: str = 'composite',
                           **kwargs) -> None:
        """
        Export overlay as image file.
        
        Args:
            output_path: Path for output image
            layers: Layer numbers to render
            mode: Export mode ('overlay', 'composite', 'mask')
            **kwargs: Additional arguments for rendering
        """
        if mode == 'composite':
            result = self.render_composite(layers, **kwargs)
        elif mode == 'overlay':
            result = self.render_overlay(layers, **kwargs)
            if len(result.shape) == 2:
                result = np.stack([result] * 3, axis=-1)
        elif mode == 'mask':
            result = self.render_overlay(layers, mode='mask', **kwargs)
            result = np.stack([result] * 3, axis=-1)
        else:
            raise ValueError(f"Unknown export mode: {mode}")
        
        cv2.imwrite(output_path, result)


def create_identity_transform() -> np.ndarray:
    """Create 2x3 identity transformation matrix."""
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0]], dtype=np.float64)


def create_translation_transform(dx: float, dy: float) -> np.ndarray:
    """Create 2x3 translation transformation matrix."""
    return np.array([[1.0, 0.0, dx],
                     [0.0, 1.0, dy]], dtype=np.float64)


def create_rotation_transform(angle_rad: float, center_x: float = 0.0, center_y: float = 0.0) -> np.ndarray:
    """Create 2x3 rotation transformation matrix."""
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    return np.array([[cos_a, -sin_a, center_x * (1 - cos_a) + center_y * sin_a],
                     [sin_a, cos_a, center_y * (1 - cos_a) - center_x * sin_a]], dtype=np.float64)


def create_scale_transform(scale_x: float, scale_y: float, center_x: float = 0.0, center_y: float = 0.0) -> np.ndarray:
    """Create 2x3 scaling transformation matrix."""
    return np.array([[scale_x, 0.0, center_x * (1 - scale_x)],
                     [0.0, scale_y, center_y * (1 - scale_y)]], dtype=np.float64)


def combine_transforms(*transforms: np.ndarray) -> np.ndarray:
    """Combine multiple 2x3 transformation matrices."""
    if not transforms:
        return create_identity_transform()
    
    result = transforms[0]
    for transform in transforms[1:]:
        temp = np.eye(3)
        temp[:2, :] = result
        transform_3x3 = np.eye(3)
        transform_3x3[:2, :] = transform
        combined_3x3 = transform_3x3 @ temp
        result = combined_3x3[:2, :]
    
    return result


class OverlayError(Exception):
    """Custom exception for overlay rendering errors."""
    pass