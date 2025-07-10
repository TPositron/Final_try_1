"""
SEM Image Container Module
Standardized container for SEM image data and metadata used across the entire
image processing pipeline.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import warnings

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False
    warnings.warn("tifffile not available. Some functionality will be limited.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class SEMImageError(Exception):
    """Custom exception for SEM image processing errors."""
    pass


class SEMImage:
    """
    Standardized container for SEM image data and metadata.
    
    This class bundles raw pixel data with essential SEM metadata and ensures
    consistent structure for filters and alignment functions throughout the
    processing pipeline.
    """
    
    # Expected dimensions for input images
    EXPECTED_WIDTH = 1024
    EXPECTED_HEIGHT = 768
    
    # Output dimensions after cropping
    OUTPUT_WIDTH = 1024
    OUTPUT_HEIGHT = 666
    
    # Crop parameters (bottom crop to remove scale bar)
    CROP_BOTTOM = EXPECTED_HEIGHT - OUTPUT_HEIGHT  # 102 pixels
    
    def __init__(self, 
                 image_data: np.ndarray,
                 pixel_size: float = 1.0,
                 bit_depth: Optional[int] = None,
                 original_path: Optional[Union[str, Path]] = None,
                 acquisition_date: Optional[str] = None,
                 auto_crop: bool = True,
                 **additional_metadata):
        """
        Initialize SEM image container with automatic bottom cropping.
        
        Args:
            image_data: NumPy array containing image data
            pixel_size: Physical scale in nm/pixel (default=1.0)
            bit_depth: Original bit depth (8 or 16), auto-detected if None
            original_path: Source file location
            acquisition_date: Image acquisition timestamp
            auto_crop: Whether to automatically crop bottom 102 pixels (default=True)
            **additional_metadata: Additional metadata fields
        
        Raises:
            SEMImageError: If image data is invalid
        """
        # Store original data for reference
        self._original_data = image_data.astype(np.float32)
        
        # Apply automatic bottom cropping if enabled
        if auto_crop:
            cropped_data = self.crop_bottom(image_data)
        else:
            cropped_data = image_data
        
        self._validate_image_data(cropped_data)
        
        # Store processed image data (convert to float32 for processing)
        self._raw_data = cropped_data.astype(np.float32)
        
        # Auto-detect bit depth if not provided
        if bit_depth is None:
            bit_depth = self._detect_bit_depth(image_data)
        
        # Build metadata dictionary
        self._metadata = {
            'pixel_size': float(pixel_size),
            'bit_depth': int(bit_depth),
            'shape': self._raw_data.shape,
            'original_path': str(original_path) if original_path else None,
            'acquisition_date': acquisition_date,
            'auto_cropped': auto_crop,
            **additional_metadata
        }
        
        # Cache for processed array
        self._processed_array = None
    
    def crop_bottom(self, image_data: np.ndarray) -> np.ndarray:
        """
        Crop bottom 102 pixels from image to remove scale bar.
        
        Args:
            image_data: Input image array
            
        Returns:
            Cropped image array with bottom 102 pixels removed
        """
        if len(image_data.shape) != 2:
            raise SEMImageError("Image data must be 2D for cropping")
        
        height, width = image_data.shape
        
        # Remove bottom 102 pixels
        if height <= self.CROP_BOTTOM:
            raise SEMImageError(f"Image too small for cropping: height {height} <= {self.CROP_BOTTOM}")
        
        cropped_height = height - self.CROP_BOTTOM
        return image_data[:cropped_height, :]
    
    def get_cropped_image(self) -> np.ndarray:
        """
        Return the cropped image data.
        
        Returns:
            NumPy array with cropped image data
        """
        return self._raw_data.copy()
    
    @classmethod
    def from_file(cls, path: Union[str, Path], data_dir: str = "../Data/SEM/") -> 'SEMImage':
        """
        Load SEM image from TIFF file, robust to shape and color.
        
        Args:
            path: Filename or relative path to TIFF file
            data_dir: Base directory for SEM data (relative to project root)
        
        Returns:
            SEMImage instance
            
        Raises:
            SEMImageError: If file cannot be loaded or is invalid
        """
        if not TIFFFILE_AVAILABLE:
            raise SEMImageError("tifffile library required for loading TIFF files")
        
        # Construct full path
        full_path = Path(data_dir) / path
        
        if not full_path.exists():
            raise SEMImageError(f"File not found: {full_path}")
        
        try:
            # Load TIFF with tifffile for better metadata support
            with tifffile.TiffFile(str(full_path)) as tif:
                # Get image data
                image_data = tif.asarray()
                
                # Extract metadata from TIFF tags
                metadata = cls._extract_tiff_metadata(tif)
                
        except Exception as e:
            raise SEMImageError(f"Failed to load TIFF file {full_path}: {e}")
        
        # If color, convert to grayscale
        if len(image_data.shape) == 3:
            # If last dim is 3 or 4, assume RGB(A)
            if image_data.shape[2] >= 3:
                # Use only first 3 channels if RGBA
                image_data = image_data[..., :3]
                # Convert to grayscale using luminosity method
                image_data = np.dot(image_data[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                raise SEMImageError("Unsupported image shape: {}".format(image_data.shape))
        
        # Create instance with automatic bottom cropping (will be applied in __init__)
        return cls(
            image_data=image_data,
            pixel_size=metadata.get('pixel_size', 1.0),
            bit_depth=metadata.get('bit_depth'),
            original_path=full_path,
            acquisition_date=metadata.get('acquisition_date'),
            auto_crop=True,
            **{k: v for k, v in metadata.items() 
               if k not in ['pixel_size', 'bit_depth', 'acquisition_date']}
        )
    
    def to_array(self) -> np.ndarray:
        """
        Return processed image array ready for analysis.
        
        Returns:
            NumPy array (float32) with shape (666, 1024)
        """
        if self._processed_array is None:
            # Normalize values to [0, 1] range based on bit depth
            bit_depth = self._metadata['bit_depth']
            max_val = 2**bit_depth - 1
            self._processed_array = self._raw_data / max_val
        
        return self._processed_array.copy()
    
    def get_metadata(self) -> Dict:
        """
        Return metadata dictionary with guaranteed fields.
        
        Returns:
            Dictionary containing at minimum:
            - pixel_size: float (nm/px)
            - bit_depth: int (8 or 16)
            - shape: tuple (height, width)
        """
        # Ensure shape reflects current processed dimensions
        metadata = self._metadata.copy()
        metadata['shape'] = self._raw_data.shape
        return metadata
    
    def get_raw_data(self) -> np.ndarray:
        """
        Return raw image data without normalization.
        
        Returns:
            NumPy array (float32) with original pixel values
        """
        return self._raw_data.copy()
    
    @property
    def pixel_size(self) -> float:
        """Physical pixel size in nm/pixel."""
        return self._metadata['pixel_size']
    
    @property
    def bit_depth(self) -> int:
        """Original bit depth (8 or 16)."""
        return self._metadata['bit_depth']
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Image dimensions (height, width) after processing."""
        return self._raw_data.shape
    
    def export_png(self, output_path: Union[str, Path]) -> None:
        """
        Export processed image as PNG (requires PIL).
        
        Args:
            output_path: Path for output PNG file
            
        Raises:
            SEMImageError: If PIL not available
        """
        if not PIL_AVAILABLE:
            raise SEMImageError("PIL library required for PNG export")
        
        # Get normalized array and convert to 8-bit
        array = self.to_array()
        image_8bit = (array * 255).astype(np.uint8)
        
        # Save as PNG
        img = Image.fromarray(image_8bit, mode='L')
        img.save(output_path)
    
    def _validate_image_data(self, image_data: np.ndarray) -> None:
        """Validate input image data."""
        if not isinstance(image_data, np.ndarray):
            raise SEMImageError("Image data must be NumPy array")
        
        if len(image_data.shape) != 2:
            raise SEMImageError("Image data must be 2D (grayscale)")
        
        if image_data.size == 0:
            raise SEMImageError("Image data is empty")
        
        # Check minimum dimensions after cropping
        height, width = image_data.shape
        if height < 1 or width < 1:
            raise SEMImageError("Image dimensions too small after processing")
    
    def _detect_bit_depth(self, image_data: np.ndarray) -> int:
        """Auto-detect bit depth from image data."""
        max_val = image_data.max()
        
        if max_val <= 255:
            return 8
        elif max_val <= 65535:
            return 16
        else:
            # Assume 16-bit for very large values
            return 16
    
    @staticmethod
    def _extract_tiff_metadata(tif: 'tifffile.TiffFile') -> Dict:
        """
        Extract metadata from TIFF file.
        
        Args:
            tif: Open TiffFile object
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}
        
        try:
            # Get first page
            page = tif.pages[0]
            
            # Extract bit depth
            if hasattr(page, 'bitspersample'):
                metadata['bit_depth'] = page.bitspersample
            
            # Try to extract pixel size from resolution tags
            if hasattr(page, 'tags'):
                tags = page.tags
                
                # Look for resolution information
                if 'XResolution' in tags and 'YResolution' in tags:
                    x_res = tags['XResolution'].value
                    y_res = tags['YResolution'].value
                    
                    # Convert to nm/pixel (assuming resolution in DPI)
                    if isinstance(x_res, (list, tuple)) and len(x_res) == 2:
                        dpi = x_res[0] / x_res[1]
                        # Convert DPI to nm/pixel: 25.4mm/inch * 1e6 nm/mm / dpi
                        metadata['pixel_size'] = 25.4e6 / dpi
                
                # Look for datetime
                if 'DateTime' in tags:
                    metadata['acquisition_date'] = str(tags['DateTime'].value)
                
                # Look for image description or other custom tags
                if 'ImageDescription' in tags:
                    desc = str(tags['ImageDescription'].value)
                    metadata['description'] = desc
                    
                    # Try to parse pixel size from description if available
                    # This is instrument-specific and would need customization
                    pass
                
        except Exception as e:
            warnings.warn(f"Could not extract all TIFF metadata: {e}")
        
        return metadata
    
    def __repr__(self) -> str:
        """String representation of SEMImage."""
        return (f"SEMImage(shape={self.shape}, "
                f"pixel_size={self.pixel_size:.2f}nm/px, "
                f"bit_depth={self.bit_depth})")


# Convenience functions
def load_sem_image(filename: Union[str, Path], 
                   data_dir: str = "../Data/SEM/") -> SEMImage:
    """
    Convenience function to load SEM image from file.
    
    Args:
        filename: Name of TIFF file to load
        data_dir: Directory containing SEM data
        
    Returns:
        SEMImage instance
    """
    return SEMImage.from_file(filename, data_dir)


def create_test_image(width: int = 1024, 
                     height: int = 768,  # Start with original height before cropping
                     pixel_size: float = 1.0,
                     bit_depth: int = 8,
                     auto_crop: bool = True) -> SEMImage:
    """
    Create a test SEM image for development/testing.
    
    Args:
        width: Image width
        height: Image height (before cropping if auto_crop=True)
        pixel_size: Pixel size in nm/pixel
        bit_depth: Bit depth (8 or 16)
        auto_crop: Whether to apply automatic bottom cropping
        
    Returns:
        SEMImage instance with synthetic data
    """
    # Create synthetic image data
    max_val = 2**bit_depth - 1
    image_data = np.random.randint(0, max_val, (height, width), dtype=np.uint16)
    
    return SEMImage(
        image_data=image_data,
        pixel_size=pixel_size,
        bit_depth=bit_depth,
        original_path="test_image.tif",
        auto_crop=auto_crop
    )


if __name__ == "__main__":
    # Example usage and basic testing
    print("SEM Image Container Module")
    print("=" * 30)
    
    # Test with synthetic data
    test_img = create_test_image()
    print(f"Test image: {test_img}")
    print(f"Metadata: {test_img.get_metadata()}")
    
    # Test array output
    array = test_img.to_array()
    print(f"Array shape: {array.shape}, dtype: {array.dtype}")
    print(f"Array range: [{array.min():.3f}, {array.max():.3f}]")
