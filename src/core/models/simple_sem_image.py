"""
SEM Image model for handling SEM image data and operations.

Basic SEM image class with numpy array storage and metadata.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, List, Callable
import cv2
from PIL import Image
import copy
import json
import base64


class SemImage:
    """
    Basic SEM image class with numpy array storage.
    
    Handles SEM image loading, basic operations, and metadata storage.
    """
    
    def __init__(self, data: Union[str, Path, np.ndarray], file_path: Optional[str] = None):
        """
        Initialize SEM image from file path or array.
        
        Args:
            data: File path or numpy array
            file_path: Optional file path if data is array
        """
        self.file_path = None
        self.filter_history: List[str] = []
        
        if isinstance(data, (str, Path)):
            # Load from file
            self.file_path = str(data)
            self.image_array = self._load_from_file(self.file_path)
        elif isinstance(data, np.ndarray):
            # Use provided array
            self.image_array = data.copy()
            self.file_path = file_path
        else:
            raise ValueError("Data must be file path or numpy array")
        
        # Auto-crop if image is standard SEM size (1024x768)
        if self.image_array.shape == (768, 1024) or self.image_array.shape == (1024, 768):
            self.image_array = self._auto_crop()
        
        # Store metadata
        self.height, self.width = self.image_array.shape[:2]
        self.channels = 1 if len(self.image_array.shape) == 2 else self.image_array.shape[2]
    
    def _load_from_file(self, file_path: str) -> np.ndarray:
        """Load image from file."""
        try:
            # Try with OpenCV first
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                return image
            
            # Fallback to PIL
            with Image.open(file_path) as img:
                if img.mode != 'L':
                    img = img.convert('L')
                return np.array(img)
                
        except Exception as e:
            raise ValueError(f"Could not load image from {file_path}: {e}")
    
    def _auto_crop(self) -> np.ndarray:
        """
        Remove bottom metadata band (~102px) and resize to 1024×666.
        
        Returns:
            Cropped and resized image array
        """
        # If image is 1024x768, crop bottom 102 pixels to get 1024x666
        if self.image_array.shape == (768, 1024):
            cropped = self.image_array[:666, :]  # Keep top 666 rows
        elif self.image_array.shape == (1024, 768):
            cropped = self.image_array[:, :666]  # Keep left 666 columns
        else:
            # For other sizes, just return as-is
            return self.image_array
        
        # Resize to exactly 1024x666 if not already
        if cropped.shape != (666, 1024):
            cropped = cv2.resize(cropped, (1024, 666))
        
        self.filter_history.append("auto_crop")
        return cropped
    
    def crop(self, x1: int, y1: int, x2: int, y2: int) -> 'SemImage':
        """
        Manual crop method returning new SemImage instance.
        
        Args:
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
            
        Returns:
            New SemImage instance with cropped data
        """
        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, self.width))
        y1 = max(0, min(y1, self.height))
        x2 = max(x1, min(x2, self.width))
        y2 = max(y1, min(y2, self.height))
        
        # Crop the image
        cropped_array = self.image_array[y1:y2, x1:x2]
        
        # Create new SemImage instance
        new_image = SemImage(cropped_array, self.file_path)
        new_image.filter_history = self.filter_history.copy()
        new_image.filter_history.append(f"manual_crop({x1},{y1},{x2},{y2})")
        
        return new_image
    
    def histogram(self, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate histogram for grayscale distribution.
        
        Args:
            bins: Number of histogram bins
            
        Returns:
            Tuple of (bin_edges, counts)
        """
        counts, bin_edges = np.histogram(self.image_array.flatten(), bins=bins, range=(0, 255))
        return bin_edges, counts
    
    def binarize(self, threshold: int = 128) -> np.ndarray:
        """
        Binarize image using global threshold.
        
        Args:
            threshold: Global threshold value (0-255)
            
        Returns:
            Binary image array (0s and 255s)
        """
        binary = np.where(self.image_array > threshold, 255, 0).astype(np.uint8)
        return binary
    
    def apply_filter(self, filter_func: Callable[[np.ndarray], np.ndarray], filter_name: str = "custom") -> 'SemImage':
        """
        Apply generic filter function to image.
        
        Args:
            filter_func: Function that takes and returns numpy array
            filter_name: Name for filter history
            
        Returns:
            New SemImage instance with filter applied
        """
        try:
            filtered_array = filter_func(self.image_array)
            
            # Create new SemImage instance
            new_image = SemImage(filtered_array, self.file_path)
            new_image.filter_history = self.filter_history.copy()
            new_image.filter_history.append(filter_name)
            
            return new_image
            
        except Exception as e:
            raise ValueError(f"Error applying filter {filter_name}: {e}")
    
    def copy(self) -> 'SemImage':
        """
        Create deep copy of SemImage with metadata.
        
        Returns:
            New SemImage instance (deep copy)
        """
        # Create new instance with copied array
        new_image = SemImage(self.image_array.copy(), self.file_path)
        
        # Copy metadata and history
        new_image.filter_history = self.filter_history.copy()
        
        return new_image
    
    def get_info(self) -> dict:
        """
        Get image information and metadata.
        
        Returns:
            Dictionary with image information including quality metrics
        """
        basic_info = {
            "file_path": self.file_path,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "dtype": str(self.image_array.dtype),
            "shape": self.image_array.shape,
            "filter_history": self.filter_history,
            "min_value": int(self.image_array.min()),
            "max_value": int(self.image_array.max()),
            "mean_value": float(self.image_array.mean())
        }
        
        # Add quality metrics
        try:
            quality_metrics = self.get_quality_metrics()
            basic_info["quality"] = quality_metrics
        except Exception as e:
            print(f"Error adding quality metrics: {e}")
            basic_info["quality"] = {"error": str(e)}
        
        return basic_info
    
    def save(self, file_path: str) -> bool:
        """
        Save image to file (auto-detects format from extension).
        
        Args:
            file_path: Path to save file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use OpenCV for most formats
            success = cv2.imwrite(str(file_path), self.image_array)
            return bool(success)
            
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
    def save_png(self, file_path: str, quality: int = 95) -> bool:
        """
        Save image to PNG format with optional compression.
        
        Args:
            file_path: Path to save PNG file
            quality: Compression quality (0-100, higher = better quality)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            # Ensure .png extension
            if file_path.suffix.lower() != '.png':
                file_path = file_path.with_suffix('.png')
            
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert quality to OpenCV compression parameter
            # PNG compression is 0-9, where 9 is maximum compression
            compression_level = max(0, min(9, int((100 - quality) / 11)))
            
            # Save with PNG compression
            success = cv2.imwrite(
                str(file_path), 
                self.image_array,
                [cv2.IMWRITE_PNG_COMPRESSION, compression_level]
            )
            
            if success:
                print(f"Saved PNG: {file_path}")
                return True
            else:
                print(f"Failed to save PNG: {file_path}")
                return False
                
        except Exception as e:
            print(f"Error saving PNG: {e}")
            return False
    
    def export_png(self, file_path: str, include_metadata: bool = True, quality: int = 95) -> bool:
        """
        Export image to PNG with optional metadata embedding.
        
        Args:
            file_path: Path to save PNG file
            include_metadata: Whether to embed metadata in PNG
            quality: Compression quality (0-100)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First save the basic PNG
            if not self.save_png(file_path, quality):
                return False
            
            # If metadata requested, try to embed it using PIL
            if include_metadata:
                try:
                    from PIL import Image
                    from PIL.PngImagePlugin import PngInfo
                    
                    # Create metadata
                    metadata = PngInfo()
                    
                    # Add basic info
                    metadata.add_text("Software", "SEM/GDS Alignment Tool")
                    metadata.add_text("Original_Path", str(self.file_path) if self.file_path else "unknown")
                    metadata.add_text("Width", str(self.width))
                    metadata.add_text("Height", str(self.height))
                    metadata.add_text("Channels", str(self.channels))
                    metadata.add_text("Data_Type", str(self.image_array.dtype))
                    
                    # Add filter history
                    if self.filter_history:
                        metadata.add_text("Filter_History", ",".join(self.filter_history))
                    
                    # Add quality metrics
                    try:
                        quality_metrics = self.get_quality_metrics()
                        metadata.add_text("Quality_Score", f"{quality_metrics['quality_score']:.2f}")
                        metadata.add_text("Quality_Assessment", quality_metrics['assessment'])
                        metadata.add_text("Noise_Level", f"{quality_metrics['noise_level']:.2f}")
                        metadata.add_text("Contrast", f"{quality_metrics['contrast']:.2f}")
                    except Exception:
                        pass  # Skip quality metrics if calculation fails
                    
                    # Re-save with metadata
                    with Image.fromarray(self.image_array) as img:
                        img.save(file_path, "PNG", pnginfo=metadata, optimize=True)
                    
                    print(f"Exported PNG with metadata: {file_path}")
                    
                except ImportError:
                    print("PIL not available for metadata embedding, saved basic PNG")
                except Exception as e:
                    print(f"Could not embed metadata: {e}, saved basic PNG")
            
            return True
            
        except Exception as e:
            print(f"Error exporting PNG: {e}")
            return False
    
    def to_json(self, file_path: str) -> bool:
        """
        Save SemImage to JSON format with array data and metadata.
        
        Args:
            file_path: Path to save JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert numpy array to base64 encoded string
            array_bytes = self.image_array.tobytes()
            array_b64 = base64.b64encode(array_bytes).decode('utf-8')
            
            # Create JSON data structure
            json_data = {
                "metadata": {
                    "file_path": self.file_path,
                    "width": self.width,
                    "height": self.height,
                    "channels": self.channels,
                    "dtype": str(self.image_array.dtype),
                    "shape": list(self.image_array.shape),
                    "filter_history": self.filter_history,
                    "min_value": int(self.image_array.min()),
                    "max_value": int(self.image_array.max()),
                    "mean_value": float(self.image_array.mean())
                },
                "array_data": {
                    "data": array_b64,
                    "dtype": str(self.image_array.dtype),
                    "shape": list(self.image_array.shape)
                },
                "version": "1.0"
            }
            
            # Save to JSON file
            with open(file_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving to JSON: {e}")
            return False
    
    @classmethod
    def from_json(cls, file_path: str) -> 'SemImage':
        """
        Load SemImage from JSON format.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            SemImage instance loaded from JSON
            
        Raises:
            ValueError: If JSON file is invalid or corrupted
        """
        try:
            # Load JSON data
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            # Validate JSON structure
            if "array_data" not in json_data or "metadata" not in json_data:
                raise ValueError("Invalid JSON format: missing required fields")
            
            # Decode array data
            array_info = json_data["array_data"]
            array_b64 = array_info["data"]
            array_bytes = base64.b64decode(array_b64)
            
            # Reconstruct numpy array
            dtype = np.dtype(array_info["dtype"])
            shape = tuple(array_info["shape"])
            image_array = np.frombuffer(array_bytes, dtype=dtype).reshape(shape)
            
            # Create SemImage instance
            metadata = json_data["metadata"]
            sem_image = cls(image_array, metadata.get("file_path"))
            
            # Restore metadata
            sem_image.filter_history = metadata.get("filter_history", [])
            
            return sem_image
            
        except Exception as e:
            raise ValueError(f"Error loading from JSON: {e}")
    
    def get_json_dict(self) -> dict:
        """
        Get SemImage as JSON-serializable dictionary.
        
        Returns:
            Dictionary with array data and metadata including quality metrics
        """
        # Convert numpy array to base64 encoded string
        array_bytes = self.image_array.tobytes()
        array_b64 = base64.b64encode(array_bytes).decode('utf-8')
        
        # Get quality metrics
        try:
            quality_metrics = self.get_quality_metrics()
        except Exception:
            quality_metrics = {"error": "Could not calculate quality metrics"}
        
        return {
            "metadata": {
                "file_path": self.file_path,
                "width": self.width,
                "height": self.height,
                "channels": self.channels,
                "dtype": str(self.image_array.dtype),
                "shape": list(self.image_array.shape),
                "filter_history": self.filter_history,
                "min_value": int(self.image_array.min()),
                "max_value": int(self.image_array.max()),
                "mean_value": float(self.image_array.mean()),
                "quality": quality_metrics
            },
            "array_data": {
                "data": array_b64,
                "dtype": str(self.image_array.dtype),
                "shape": list(self.image_array.shape)
            },
            "version": "1.0"
        }
    
    @classmethod
    def from_json_dict(cls, json_dict: dict) -> 'SemImage':
        """
        Create SemImage from JSON dictionary.
        
        Args:
            json_dict: Dictionary with SemImage data
            
        Returns:
            SemImage instance
        """
        # Validate structure
        if "array_data" not in json_dict or "metadata" not in json_dict:
            raise ValueError("Invalid JSON dictionary: missing required fields")
        
        # Decode array data
        array_info = json_dict["array_data"]
        array_b64 = array_info["data"]
        array_bytes = base64.b64decode(array_b64)
        
        # Reconstruct numpy array
        dtype = np.dtype(array_info["dtype"])
        shape = tuple(array_info["shape"])
        image_array = np.frombuffer(array_bytes, dtype=dtype).reshape(shape)
        
        # Create SemImage instance
        metadata = json_dict["metadata"]
        sem_image = cls(image_array, metadata.get("file_path"))
        
        # Restore metadata
        sem_image.filter_history = metadata.get("filter_history", [])
        
        return sem_image
    
    def get_noise_level(self) -> float:
        """
        Detect basic noise level in the image.
        
        Uses Laplacian variance method for noise estimation.
        Higher values indicate more noise.
        
        Returns:
            Noise level estimate (higher = more noisy)
        """
        try:
            # Use Laplacian variance for noise detection
            laplacian = cv2.Laplacian(self.image_array, cv2.CV_64F)
            noise_level = laplacian.var()
            return float(noise_level)
        except Exception as e:
            print(f"Error calculating noise level: {e}")
            return 0.0
    
    def get_contrast(self) -> float:
        """
        Calculate simple contrast measurement.
        
        Uses RMS (Root Mean Square) contrast method.
        Higher values indicate better contrast.
        
        Returns:
            Contrast value (0-255 range, higher = better contrast)
        """
        try:
            # Calculate RMS contrast
            mean_intensity = float(self.image_array.mean())
            squared_diff = (self.image_array.astype(np.float64) - mean_intensity) ** 2
            rms_contrast = np.sqrt(squared_diff.mean())
            return float(rms_contrast)
        except Exception as e:
            print(f"Error calculating contrast: {e}")
            return 0.0
    
    def get_quality_metrics(self) -> dict:
        """
        Get comprehensive image quality metrics.
        
        Returns:
            Dictionary with quality measurements
        """
        try:
            noise_level = self.get_noise_level()
            contrast = self.get_contrast()
            
            # Calculate additional basic metrics
            intensity_range = int(self.image_array.max()) - int(self.image_array.min())
            std_dev = float(self.image_array.std())
            
            # Simple quality assessment
            quality_score = self._calculate_quality_score(noise_level, contrast, intensity_range)
            
            return {
                "noise_level": noise_level,
                "contrast": contrast,
                "intensity_range": intensity_range,
                "standard_deviation": std_dev,
                "quality_score": quality_score,
                "assessment": self._get_quality_assessment(quality_score)
            }
        except Exception as e:
            print(f"Error calculating quality metrics: {e}")
            return {
                "noise_level": 0.0,
                "contrast": 0.0,
                "intensity_range": 0,
                "standard_deviation": 0.0,
                "quality_score": 0.0,
                "assessment": "unknown"
            }
    
    def _calculate_quality_score(self, noise_level: float, contrast: float, intensity_range: int) -> float:
        """
        Calculate simple quality score from metrics.
        
        Args:
            noise_level: Noise level measurement
            contrast: Contrast measurement
            intensity_range: Range of intensity values
            
        Returns:
            Quality score (0-100, higher = better quality)
        """
        try:
            # Normalize metrics to 0-1 range
            # Lower noise is better (invert noise score)
            noise_score = max(0, 1 - (noise_level / 1000))  # Typical noise levels are 0-1000
            
            # Higher contrast is better
            contrast_score = min(1, contrast / 50)  # Normalize contrast
            
            # Good intensity range is better
            range_score = min(1, intensity_range / 255)
            
            # Weighted combination
            quality_score = (noise_score * 0.4 + contrast_score * 0.4 + range_score * 0.2) * 100
            
            return float(max(0, min(100, quality_score)))
        except Exception:
            return 0.0
    
    def _get_quality_assessment(self, quality_score: float) -> str:
        """
        Get qualitative assessment from quality score.
        
        Args:
            quality_score: Numeric quality score (0-100)
            
        Returns:
            Text assessment of image quality
        """
        if quality_score >= 80:
            return "excellent"
        elif quality_score >= 60:
            return "good"
        elif quality_score >= 40:
            return "fair"
        elif quality_score >= 20:
            return "poor"
        else:
            return "very_poor"
    
    def is_noisy(self, threshold: float = 100.0) -> bool:
        """
        Check if image is considered noisy.
        
        Args:
            threshold: Noise level threshold
            
        Returns:
            True if image is noisy, False otherwise
        """
        return self.get_noise_level() > threshold
    
    def has_good_contrast(self, threshold: float = 20.0) -> bool:
        """
        Check if image has good contrast.
        
        Args:
            threshold: Contrast threshold
            
        Returns:
            True if image has good contrast, False otherwise
        """
        return self.get_contrast() > threshold
    
    def __repr__(self) -> str:
        """String representation of SemImage."""
        return f"SemImage({self.width}x{self.height}, {self.channels} channels, {len(self.filter_history)} filters)"


# Convenience functions for loading and saving
def load_sem_image(file_path: Union[str, Path]) -> SemImage:
    """
    Load SEM image from file (image or JSON format).
    
    Args:
        file_path: Path to image or JSON file
        
    Returns:
        SemImage instance
    """
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.json':
        return SemImage.from_json(str(file_path))
    else:
        return SemImage(file_path)


def create_sem_image(data: np.ndarray, file_path: Optional[Union[str, Path]] = None) -> SemImage:
    """
    Create SEM image from numpy array.
    
    Args:
        data: Image data as numpy array
        file_path: Optional source file path
        
    Returns:
        SemImage instance
    """
    return SemImage(data, str(file_path) if file_path else None)


def save_sem_image_json(sem_image: SemImage, file_path: Union[str, Path]) -> bool:
    """
    Save SEM image to JSON format.
    
    Args:
        sem_image: SemImage instance to save
        file_path: Path to save JSON file
        
    Returns:
        True if successful, False otherwise
    """
    return sem_image.to_json(str(file_path))


def export_sem_image_png(sem_image: SemImage, file_path: Union[str, Path], 
                        include_metadata: bool = True, quality: int = 95) -> bool:
    """
    Export SEM image to PNG format.
    
    Args:
        sem_image: SemImage instance to export
        file_path: Path to save PNG file
        include_metadata: Whether to embed metadata in PNG
        quality: Compression quality (0-100)
        
    Returns:
        True if successful, False otherwise
    """
    return sem_image.export_png(str(file_path), include_metadata, quality)
