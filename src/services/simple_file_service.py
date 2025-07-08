"""
Simple File Service - Core File Management System (Implementation Steps 71-75)

This service provides comprehensive file management capabilities for the SEM/GDS alignment
application. It handles all file I/O operations with proper validation, error handling,
and progress reporting.

Core Capabilities:
- SEM image loading with automatic cropping to 1024x666 pixels
- GDS file loading with structure extraction and validation
- Result saving with organized directory structure
- File format validation and conversion
- Recent files tracking and management

Directory Structure Management:
- Data/SEM/: SEM image files (TIFF, PNG)
- Data/GDS/: GDS layout files
- Results/Aligned/: Alignment results (manual/auto)
- Results/SEM_Filters/: Filter processing results
- Results/Scoring/: Analysis and scoring results
- Results/cut/: Cropped SEM images

Key Methods:
- load_sem_file(): Loads and crops SEM images with validation
- load_gds_file(): Loads GDS files with structure extraction
- save_alignment_result(): Saves alignment data with timestamps
- scan_data_directories(): Discovers available files
- get_recent_files(): Manages recent file history

File Format Support:
- SEM: TIFF, PNG (automatically cropped to standard size)
- GDS: GDS, GDS2 (with structure validation)
- Results: JSON, CSV, TXT (with proper encoding)

Dependencies:
- Uses: pathlib, json, csv (file operations)
- Uses: PIL/Pillow (image processing)
- Uses: src.core.models (SemImage, GDS models)
- Uses: src.core.models.simple_gds_extraction (structure extraction)
- Called by: ui/file_handler.py, ui/file_operations.py
- Called by: services/file_loading_service.py

Signals (Step 75):
- files_scanned: Emitted when directory scanning completes
- file_loaded: Emitted when files are successfully loaded
- loading_progress: Emitted during loading operations
- loading_error: Emitted when loading fails
- file_saved: Emitted when files are saved
- recent_files_changed: Emitted when recent files list updates

Critical Features:
- Automatic SEM image cropping to 1024x666 (removes bottom 102 pixels)
- GDS structure validation against predefined structure definitions
- Comprehensive error handling with user-friendly messages
- Progress reporting for long operations
- File history management with configurable limits
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from PySide6.QtCore import QObject, Signal
import numpy as np

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    HAS_PANDAS = False


class FileService(QObject):
    """Simple QObject-based file service for basic file management."""
    
    # Signals for Step 75
    files_scanned = Signal(list, list)  # sem_files, gds_files
    file_loaded = Signal(str, str)      # file_type, file_path
    loading_progress = Signal(str)      # progress_message
    loading_error = Signal(str)         # error_message
    file_saved = Signal(str, str)       # file_type, file_path
    error_occurred = Signal(str)        # error_message
    recent_files_changed = Signal(list)  # List of (file_type, file_path)
    
    MAX_RECENT_FILES = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Default directories - use absolute paths from project root
        project_root = Path(__file__).parent.parent.parent
        self.data_dir = project_root / "Data"
        self.sem_dir = self.data_dir / "SEM"
        self.gds_dir = self.data_dir / "GDS"
        self.results_dir = project_root / "Results"
        
        # File extension lists
        self.sem_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        self.gds_extensions = ['.gds', '.gds2', '.gdsii']
        
        # Recent files list
        self.recent_files = []  # List of dicts: {"type": str, "path": str}
        
        # Store last loaded data
        self._last_loaded_gds_data = None

        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        try:
            self.sem_dir.mkdir(parents=True, exist_ok=True)
            self.gds_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
            # Create result subdirectories
            (self.results_dir / "Aligned" / "manual").mkdir(parents=True, exist_ok=True)
            (self.results_dir / "Aligned" / "auto").mkdir(parents=True, exist_ok=True)
            (self.results_dir / "SEM_Filters" / "manual").mkdir(parents=True, exist_ok=True)
            (self.results_dir / "SEM_Filters" / "auto").mkdir(parents=True, exist_ok=True)
            (self.results_dir / "Scoring" / "overlays").mkdir(parents=True, exist_ok=True)
            (self.results_dir / "Scoring" / "charts").mkdir(parents=True, exist_ok=True)
            (self.results_dir / "Scoring" / "reports").mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            self.error_occurred.emit(f"Directory creation failed: {e}")
    
    # Step 72: Implement file scanning
    def scan_data_directories(self) -> tuple[List[Path], List[Path]]:
        """
        Scan Data directories for SEM and GDS files.
        
        Returns:
            Tuple of (sem_files, gds_files) lists
        """
        try:
            # Scan SEM files
            sem_files = []
            if self.sem_dir.exists():
                for ext in self.sem_extensions:
                    sem_files.extend(self.sem_dir.glob(f"*{ext}"))
            
            # Scan GDS files
            gds_files = []
            if self.gds_dir.exists():
                for ext in self.gds_extensions:
                    gds_files.extend(self.gds_dir.glob(f"*{ext}"))
            
            # Sort files by name
            sem_files.sort(key=lambda x: x.name.lower())
            gds_files.sort(key=lambda x: x.name.lower())
            
            logger.info(f"Scanned directories: {len(sem_files)} SEM files, {len(gds_files)} GDS files")
            
            # Emit signal
            self.files_scanned.emit(sem_files, gds_files)
            
            return sem_files, gds_files
            
        except Exception as e:
            logger.error(f"Directory scanning failed: {e}")
            self.error_occurred.emit(f"Directory scanning failed: {e}")
            return [], []
    
    # Step 73: Add file loading with basic format validation
    def load_sem_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load SEM file with format validation and automatic cropping.
        
        Args:
            file_path: Path to SEM file
            
        Returns:
            Dictionary containing original and cropped SEM data or None if failed
        """
        try:
            # Emit progress signal
            self.loading_progress.emit(f"Loading SEM file: {file_path.name}")
            
            # Basic format validation - support only TIFF and PNG
            if not file_path.exists():
                raise FileNotFoundError(f"SEM file not found: {file_path}")
            
            allowed_extensions = {'.tif', '.tiff', '.png'}
            if file_path.suffix.lower() not in allowed_extensions:
                raise ValueError(f"Unsupported SEM file format: {file_path.suffix}. Only TIFF and PNG are supported.")
            
            # Load image using PIL/Pillow
            from PIL import Image
            import numpy as np
            
            self.loading_progress.emit("Reading image data...")
            
            # Load the image
            with Image.open(file_path) as img:
                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Store original image
                original_array = np.array(img)
                original_size = img.size  # (width, height)
                
                self.loading_progress.emit("Cropping image to 1024x666...")
                
                # Immediately crop to exactly 1024x666 pixels
                target_width, target_height = 1024, 666
                
                # Resize/crop to target dimensions
                if original_size != (target_width, target_height):
                    # If image is larger, crop from bottom (remove bottom pixels)
                    # If image is smaller, resize to fit
                    if img.width >= target_width and img.height >= target_height:
                        # Crop from top - center horizontally, keep top part (remove bottom 102 pixels)
                        left = (img.width - target_width) // 2
                        top = 0  # Start from top
                        right = left + target_width
                        bottom = target_height  # Remove bottom pixels (e.g., 768 - 102 = 666)
                        cropped_img = img.crop((left, top, right, bottom))
                    else:
                        # Resize to target dimensions
                        cropped_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                else:
                    cropped_img = img.copy()
                
                # Convert cropped image to array
                cropped_array = np.array(cropped_img)
            
            # Create SemImage object for better integration
            try:
                from src.core.models import SemImage
                self.loading_progress.emit("Creating SemImage object...")
                # Create SemImage with the cropped data
                sem_image = SemImage(cropped_array, str(file_path))
            except ImportError:
                logger.warning("SemImage class not available, creating basic result without SemImage object")
                sem_image = None
            
            # Prepare result data
            result_data = {
                'file_path': str(file_path),
                'original_size': original_size,
                'original_array': original_array,
                'cropped_array': cropped_array,
                'cropped_size': (target_width, target_height),
                'sem_image': sem_image,
                'metadata': {
                    'format': file_path.suffix.lower(),
                    'original_dimensions': original_size,
                    'cropped_dimensions': (target_width, target_height),
                    'file_size': file_path.stat().st_size,
                    'is_cropped': original_size != (target_width, target_height)
                }
            }
            
            logger.info(f"SEM file loaded and cropped successfully: {file_path}")
            logger.info(f"Original size: {original_size}, Cropped size: {target_width}x{target_height}")
            
            # Emit success signal
            self.file_loaded.emit("SEM", str(file_path))
            self.add_recent_file("SEM", str(file_path))
            
            # Save cropped image to Results/cut/ folder
            self.loading_progress.emit("Saving cropped image...")
            
            # Create cut directory if it doesn't exist
            cut_dir = self.results_dir / "cut"
            cut_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename for cropped image
            original_stem = file_path.stem
            cropped_filename = f"{original_stem}_cropped_{target_width}x{target_height}.png"
            cropped_path = cut_dir / cropped_filename
            
            # Save cropped image
            cropped_img.save(str(cropped_path), "PNG")
            logger.info(f"Cropped image saved to: {cropped_path}")
            
            return result_data
            
        except Exception as e:
            error_msg = f"SEM file loading failed: {e}"
            logger.error(error_msg)
            self.loading_error.emit(error_msg)
            return None
    
    def load_gds_file(self, file_path: Path, structure_id: Optional[int] = None, 
                     layers: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
        """
        Load GDS file with structure extraction using gdstk.
        
        Args:
            file_path: Path to GDS file
            structure_id: Optional structure ID to extract (1-5)
            layers: Optional list of layer numbers to filter
            
        Returns:
            Dictionary containing extracted GDS data or None if failed
        """
        try:
            # Emit progress signal
            self.loading_progress.emit(f"Loading GDS file: {file_path.name}")
            
            # Basic format validation
            if not file_path.exists():
                raise FileNotFoundError(f"GDS file not found: {file_path}")
            
            if file_path.suffix.lower() not in self.gds_extensions:
                raise ValueError(f"Unsupported GDS file format: {file_path.suffix}")
            
            # Basic file size check
            if file_path.stat().st_size == 0:
                raise ValueError(f"GDS file is empty: {file_path}")
            
            # Import extraction utilities
            try:
                from src.core.models.simple_initial_gds_model import InitialGdsModel
                from src.core.models.simple_gds_extraction import (
                    get_structure_info, extract_frame, PREDEFINED_STRUCTURES
                )
            except ImportError as e:
                logger.error(f"Failed to import GDS models: {e}")
                raise ImportError(f"GDS processing modules not available: {e}")
            
            self.loading_progress.emit("Initializing GDS model...")
            
            # Create InitialGdsModel
            gds_model = InitialGdsModel(str(file_path))
            
            result_data = {
                'file_path': str(file_path),
                'gds_model': gds_model,
                'metadata': {
                    'unit': gds_model.unit,
                    'precision': gds_model.precision,
                    'cell_name': gds_model.cell.name if gds_model.cell else None,
                    'available_layers': gds_model.get_layers(),
                    'bounds': gds_model.bounds
                }
            }
            
            # If structure_id is specified, extract that specific structure
            if structure_id is not None:
                self.loading_progress.emit(f"Extracting structure {structure_id}...")
                
                if structure_id in PREDEFINED_STRUCTURES:
                    structure_info = get_structure_info(gds_model, structure_id)
                    structure_bounds = PREDEFINED_STRUCTURES[structure_id]['bounds']
                    structure_layers = layers or PREDEFINED_STRUCTURES[structure_id]['layers']
                    
                    # Extract the frame for this structure
                    extracted_frame = extract_frame(gds_model, structure_bounds, structure_layers)
                    
                    result_data['extracted_structure'] = {
                        'structure_id': structure_id,
                        'structure_info': structure_info,
                        'frame_data': extracted_frame
                    }
                    
                    self.loading_progress.emit(f"Structure {structure_id} extracted successfully")
                else:
                    raise ValueError(f"Unknown structure ID: {structure_id}")
            
            # Create binary images from extracted data if structure was extracted
            if 'extracted_structure' in result_data:
                self.loading_progress.emit("Creating binary images...")
                
                binary_image = self._create_binary_image_from_polygons(
                    result_data['extracted_structure']['frame_data']['polygons'],
                    result_data['extracted_structure']['frame_data']['bounds']
                )
                result_data['extracted_structure']['binary_image'] = binary_image
            
            logger.info(f"GDS file loaded successfully: {file_path}")
            
            # Store the result data for retrieval
            self._last_loaded_gds_data = result_data
            
            # Emit success signal
            self.file_loaded.emit("GDS", str(file_path))
            self.add_recent_file("GDS", str(file_path))
            
            # Store last loaded GDS data
            self._last_loaded_gds_data = result_data
            
            return result_data
            
        except Exception as e:
            error_msg = f"GDS file loading failed: {e}"
            logger.error(error_msg)
            self.loading_error.emit(error_msg)
            return None
    
    # Helper method to create binary images from polygons
    def _create_binary_image_from_polygons(self, polygons: List[np.ndarray], 
                                          bounds: Tuple[float, float, float, float],
                                          image_size: Tuple[int, int] = (1024, 666)) -> np.ndarray:
        """
        Create binary image from polygon data.
        
        Args:
            polygons: List of polygon coordinate arrays
            bounds: (xmin, ymin, xmax, ymax) bounds of the region
            image_size: (width, height) of output image
            
        Returns:
            Binary numpy array representing the polygons
        """
        try:
            import cv2
            
            width, height = image_size
            binary_image = np.zeros((height, width), dtype=np.uint8)
            
            if not polygons:
                logger.warning("No polygons provided for binary image creation")
                return binary_image
            
            xmin, ymin, xmax, ymax = bounds
            
            # Ensure bounds are valid
            if xmin >= xmax or ymin >= ymax:
                logger.warning(f"Invalid bounds for binary image: {bounds}")
                return binary_image
            
            x_scale = width / (xmax - xmin)
            y_scale = height / (ymax - ymin)
            
            logger.debug(f"Creating binary image: bounds={bounds}, scale=({x_scale:.2f}, {y_scale:.2f})")
            
            for i, polygon in enumerate(polygons):
                if len(polygon) < 3:  # Need at least 3 points for a polygon
                    continue
                
                # Transform polygon coordinates to image coordinates
                transformed_points = []
                for x, y in polygon:
                    img_x = int((x - xmin) * x_scale)
                    img_y = int((y - ymin) * y_scale)
                    # Clamp to image bounds
                    img_x = max(0, min(width - 1, img_x))
                    img_y = max(0, min(height - 1, img_y))
                    transformed_points.append([img_x, img_y])
                
                if len(transformed_points) >= 3:
                    # Fill polygon - fix cv2.fillPoly arguments
                    points_array = np.array(transformed_points, dtype=np.int32)
                    cv2.fillPoly(binary_image, [points_array], (255,))  # Color as tuple
            
            logger.debug(f"Binary image created: shape={binary_image.shape}, non-zero pixels={np.count_nonzero(binary_image)}")
            return binary_image
            
        except Exception as e:
            logger.error(f"Failed to create binary image: {e}")
            # Return empty image on error
            return np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    
    # Step 74: Implement file saving with basic file organization
    def save_alignment_result(self, data: Dict[str, Any], mode: str = "manual") -> Optional[Path]:
        """
        Save alignment results to appropriate directory.
        
        Args:
            data: Alignment result data
            mode: "manual" or "auto"
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            # Determine output directory
            output_dir = self.results_dir / "Aligned" / mode
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alignment_{mode}_{timestamp}.json"
            output_path = output_dir / filename
            
            # Save data as JSON
            import json
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Alignment result saved: {output_path}")
            
            # Emit signal
            self.file_saved.emit("RESULT", str(output_path))
            self.add_recent_file("RESULT", str(output_path))
            
            return output_path
            
        except Exception as e:
            logger.error(f"Alignment result saving failed: {e}")
            self.error_occurred.emit(f"Alignment result saving failed: {e}")
            return None
    
    def save_filter_result(self, data: Dict[str, Any], mode: str = "manual") -> Optional[Path]:
        """
        Save filter results to appropriate directory.
        
        Args:
            data: Filter result data
            mode: "manual" or "auto"
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            # Determine output directory
            output_dir = self.results_dir / "SEM_Filters" / mode
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"filter_{mode}_{timestamp}.json"
            output_path = output_dir / filename
            
            # Save data as JSON
            import json
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Filter result saved: {output_path}")
            
            # Emit signal
            self.file_saved.emit("filter", str(output_path))
            self.add_recent_file("filter", str(output_path))
            
            return output_path
            
        except Exception as e:
            logger.error(f"Filter result saving failed: {e}")
            self.error_occurred.emit(f"Filter result saving failed: {e}")
            return None
    
    def save_scoring_result(self, data: Dict[str, Any], result_type: str = "report") -> Optional[Path]:
        """
        Save scoring results to appropriate directory.
        
        Args:
            data: Scoring result data
            result_type: "report", "overlay", or "chart"
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            # Determine output directory based on result type
            if result_type == "overlay":
                output_dir = self.results_dir / "Scoring" / "overlays"
            elif result_type == "chart":
                output_dir = self.results_dir / "Scoring" / "charts"
            else:
                output_dir = self.results_dir / "Scoring" / "reports"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"score_{result_type}_{timestamp}.json"
            output_path = output_dir / filename
            
            # Save data as JSON
            import json
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Scoring result saved: {output_path}")
            
            # Emit signal
            self.file_saved.emit("scoring", str(output_path))
            self.add_recent_file("scoring", str(output_path))
            
            return output_path
            
        except Exception as e:
            logger.error(f"Scoring result saving failed: {e}")
            self.error_occurred.emit(f"Scoring result saving failed: {e}")
            return None
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get basic file information.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file info
        """
        try:
            if not file_path.exists():
                return {"error": "File not found"}
            
            stat = file_path.stat()
            info = {
                "name": file_path.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "extension": file_path.suffix,
                "type": "SEM" if file_path.suffix.lower() in self.sem_extensions else "GDS" if file_path.suffix.lower() in self.gds_extensions else "Unknown"
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            return {"error": str(e)}
    
    def save_file(self, file_path: Union[str, Path], data: Any, file_type: str = "json") -> bool:
        """
        Save data to file with specified format.
        
        Args:
            file_path: Path to save file
            data: Data to save
            file_type: Type of file ('json', 'csv', 'txt')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_type.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif file_type.lower() == 'csv':
                if HAS_PANDAS and pd is not None:
                    if isinstance(data, dict):
                        df = pd.DataFrame([data])
                    elif isinstance(data, list):
                        df = pd.DataFrame(data)
                    else:
                        df = pd.DataFrame({'data': [data]})
                    df.to_csv(file_path, index=False)
                else:
                    # Fallback to basic CSV writing if pandas not available
                    import csv
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        if isinstance(data, dict):
                            writer = csv.DictWriter(f, fieldnames=data.keys())
                            writer.writeheader()
                            writer.writerow(data)
                        elif isinstance(data, list) and data and isinstance(data[0], dict):
                            writer = csv.DictWriter(f, fieldnames=data[0].keys())
                            writer.writeheader()
                            writer.writerows(data)
                        else:
                            writer = csv.writer(f)
                            writer.writerow(['data'])
                            writer.writerow([str(data)])
            elif file_type.lower() == 'txt':
                with open(file_path, 'w', encoding='utf-8') as f:
                    if isinstance(data, (dict, list)):
                        import json
                        f.write(json.dumps(data, indent=2, ensure_ascii=False))
                    else:
                        f.write(str(data))
            else:
                logger.error(f"Unsupported file type: {file_type}")
                self.error_occurred.emit(f"Unsupported file type: {file_type}")
                return False
                
            logger.info(f"Saved {file_type} file: {file_path}")
            self.file_saved.emit(file_type, str(file_path))
            return True
            
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            self.error_occurred.emit(f"Failed to save file: {e}")
            return False
    
    def load_file(self, file_path: Union[str, Path], file_type: str = "json") -> Optional[Any]:
        """
        Load data from file with specified format.
        
        Args:
            file_path: Path to load file from
            file_type: Type of file ('json', 'csv', 'txt')
            
        Returns:
            Loaded data or None if failed
        """
        try:
            import json
            
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                self.error_occurred.emit(f"File not found: {file_path}")
                return None
            
            if file_type.lower() == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif file_type.lower() == 'csv':
                if HAS_PANDAS and pd is not None:
                    df = pd.read_csv(file_path)
                    data = df.to_dict('records')
                else:
                    # Fallback to basic CSV reading if pandas not available
                    import csv
                    data = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        data = list(reader)
            elif file_type.lower() == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    try:
                        # Try to parse as JSON first
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        # If not JSON, return as string
                        data = content
            else:
                logger.error(f"Unsupported file type: {file_type}")
                self.error_occurred.emit(f"Unsupported file type: {file_type}")
                return None
                
            logger.info(f"Loaded {file_type} file: {file_path}")
            self.file_loaded.emit(file_type, str(file_path))
            return data
            
        except Exception as e:
            logger.error(f"Failed to load file: {e}")
            self.error_occurred.emit(f"Failed to load file: {e}")
            return None
    
    def add_recent_file(self, file_type: str, file_path: Union[str, Path]):
        """
        Add a file to the recent files list.
        
        Args:
            file_type: Type of the file (e.g., "SEM", "GDS", "RESULT")
            file_path: Path to the file
        """
        entry = {"type": file_type, "path": str(file_path)}
        # Remove duplicates
        self.recent_files = [f for f in self.recent_files if f["path"] != entry["path"]]
        self.recent_files.insert(0, entry)
        if len(self.recent_files) > self.MAX_RECENT_FILES:
            self.recent_files = self.recent_files[:self.MAX_RECENT_FILES]
        self.recent_files_changed.emit(self.recent_files)

    def get_recent_files(self):
        """
        Get the list of recent files.
        
        Returns:
            List of recent files (up to MAX_RECENT_FILES)
        """
        return self.recent_files

    def set_recent_files(self, files):
        """
        Set the list of recent files.
        
        Args:
            files: List of files to set as recent
        """
        self.recent_files = files[:self.MAX_RECENT_FILES]
        self.recent_files_changed.emit(self.recent_files)
    
    def set_data_directories(self, sem_dir: Optional[Path] = None, gds_dir: Optional[Path] = None):
        """
        Set custom data directories.
        
        Args:
            sem_dir: Custom SEM directory
            gds_dir: Custom GDS directory
        """
        if sem_dir:
            self.sem_dir = Path(sem_dir)
        if gds_dir:
            self.gds_dir = Path(gds_dir)
        
        self._ensure_directories()
        
        # Re-scan after directory change
        self.scan_data_directories()
    
    # Compatibility methods for file_manager interface
    def get_structure_names(self):
        """Get list of structure names (stub)."""
        return ["Circpol_T2", "IP935Left_11", "IP935Left_14", "QC855GC_CROSS_Bottom", "QC935_46"]
    
    def get_gds_dir(self):
        """Get GDS directory path."""
        return self.gds_dir
    
    def load_gds_for_processing(self, filename):
        """Load GDS file for processing (stub)."""
        return True
    
    def generate_initial_gds_images(self, filename):
        """Generate initial GDS images (stub)."""
        return {"success": True}
    
    def save_structure_image(self, structure_name, image, filename):
        """Save structure image (stub)."""
        return True
    
    def create_result_subdir(self, dirname):
        """Create result subdirectory."""
        result_dir = self.results_dir / dirname
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir
    
    def save_scores(self, scores, filename, result_dir):
        """Save scores to file (stub)."""
        return True
    
    def load_structure_overlay(self, structure_name):
        """Load structure overlay (stub)."""
        return None
    
    def load_gds_with_structure(self, file_path: Path, structure_id: int, 
                               layers: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
        """
        Convenience method to load GDS file and extract a specific structure.
        
        Args:
            file_path: Path to GDS file
            structure_id: Structure ID to extract (1-5)
            layers: Optional list of layers to filter
            
        Returns:
            Dictionary with structure data and binary image, or None if failed
        """
        return self.load_gds_file(file_path, structure_id=structure_id, layers=layers)
    
    def get_available_structures(self) -> Dict[int, Dict[str, Any]]:
        """
        Get information about available predefined structures.
        
        Returns:
            Dictionary mapping structure IDs to structure info
        """
        try:
            from src.core.models.simple_gds_extraction import PREDEFINED_STRUCTURES
            return PREDEFINED_STRUCTURES.copy()
        except ImportError:
            logger.warning("PREDEFINED_STRUCTURES not available")
            return {}

    def get_sem_image_cropped(self, file_path: Path) -> Optional[np.ndarray]:
        """
        Convenience method to load SEM file and return only the cropped array.
        
        Args:
            file_path: Path to SEM file
            
        Returns:
            Cropped numpy array (1024x666) or None if failed
        """
        result = self.load_sem_file(file_path)
        if result:
            return result['cropped_array']
        return None
    
    def get_sem_image_object(self, file_path: Path) -> Optional[Any]:
        """
        Convenience method to load SEM file and return SemImage object.
        
        Args:
            file_path: Path to SEM file
            
        Returns:
            SemImage object or None if failed
        """
        result = self.load_sem_file(file_path)
        if result and result['sem_image'] is not None:
            return result['sem_image']
        return None

    def save_aligned_gds(self, aligned_model, output_path: str) -> bool:
        """
        Create a new GDS file with transformed coordinates that match the SEM image alignment.
        
        Process:
        1. Takes initial bounds from the selected GDS structure
        2. Applies transformation data (rotation, zoom, movement) 
        3. Calculates new coordinates using transformations
        4. Creates a new GDS file with the transformed coordinates
        
        Args:
            aligned_model: AlignedGdsModel instance with transformations
            output_path: Path where to save the new GDS file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            try:
                from .gds_transformation_service import GdsTransformationService
                import gdstk
            except ImportError as e:
                logger.error(f"Required modules not available for GDS transformation: {e}")
                self.error_occurred.emit(f"GDS transformation modules not available: {e}")
                return False
            
            # Get the original GDS file path and structure info
            initial_model = aligned_model.initial_model
            original_gds_path = str(initial_model.gds_path)
            
            # Get the actual structure name from the cell
            structure_name = initial_model.cell.name
            
            # Get the feature bounds that define the structure area
            feature_bounds = tuple(aligned_model.original_frame)
            
            # Get transformation parameters from the aligned model
            ui_params = aligned_model.get_ui_parameters()
            transformation_params = {
                'x_offset': ui_params.get('translation_x_pixels', 0.0),
                'y_offset': ui_params.get('translation_y_pixels', 0.0),
                'rotation': ui_params.get('rotation_degrees', 0.0),
                'scale': ui_params.get('scale', 1.0)
            }
            
            logger.info(f"Creating aligned GDS for structure: {structure_name}")
            logger.info(f"Feature bounds: {feature_bounds}")
            logger.info(f"Transformation params: {transformation_params}")
            
            # Use the transformation service to create the new GDS
            transformation_service = GdsTransformationService()
            transformed_cell = transformation_service.transform_structure(
                original_gds_path=original_gds_path,
                structure_name=structure_name,
                transformation_params=transformation_params,
                gds_bounds=feature_bounds,
                canvas_size=(1024, 666)
            )
            
            # Create a new library with the transformed cell using gdstk
            new_library = gdstk.Library()
            new_library.add(transformed_cell)
            
            # Ensure output path has .gds extension
            output_path_obj = Path(output_path)
            if output_path_obj.suffix.lower() != '.gds':
                output_path = str(output_path_obj) + '.gds'
            
            # Write the new GDS file using gdstk
            new_library.write_gds(output_path)
            
            logger.info(f"Successfully saved aligned GDS to: {output_path}")
            self.file_saved.emit("GDS", str(output_path))
            return True
            
        except Exception as e:
            error_msg = f"Failed to save aligned GDS: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False

    def get_last_loaded_gds_data(self) -> Optional[Dict[str, Any]]:
        """Get the last loaded GDS data."""
        return self._last_loaded_gds_data

    def test_initial_gds_loading(self, structure_id: int = 1) -> bool:
        """
        Test the initial GDS loading workflow for debugging.
        
        Args:
            structure_id: Structure ID to test (1-5)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Test with the default GDS file
            gds_path = self.gds_dir / "Institute_Project_GDS1.gds"
            
            if not gds_path.exists():
                logger.error(f"Test failed: GDS file not found at {gds_path}")
                return False
            
            logger.info(f"Testing initial GDS loading: structure {structure_id}")
            
            # Load GDS file with structure extraction
            result = self.load_gds_file(gds_path, structure_id=structure_id)
            
            if not result:
                logger.error("Test failed: No result returned from load_gds_file")
                return False
            
            # Check if structure was extracted
            if 'extracted_structure' not in result:
                logger.error("Test failed: No extracted_structure in result")
                return False
            
            structure_data = result['extracted_structure']
            
            # Check if binary image was created
            if 'binary_image' not in structure_data:
                logger.error("Test failed: No binary_image in extracted_structure")
                return False
            
            binary_image = structure_data['binary_image']
            
            # Validate binary image
            if binary_image.shape != (666, 1024):
                logger.error(f"Test failed: Binary image has wrong shape {binary_image.shape}, expected (666, 1024)")
                return False
            
            if binary_image.dtype != np.uint8:
                logger.error(f"Test failed: Binary image has wrong dtype {binary_image.dtype}, expected uint8")
                return False
            
            # Check if image has content
            non_zero_pixels = np.count_nonzero(binary_image)
            logger.info(f"Test success: Binary image created with {non_zero_pixels} non-zero pixels")
            
            if non_zero_pixels == 0:
                logger.warning("Test warning: Binary image is empty (no polygons found)")
            
            return True
            
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            return False
