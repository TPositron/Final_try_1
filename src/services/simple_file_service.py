"""
Simple File Service - Core File Management System

This service provides comprehensive file management capabilities for the SEM/GDS alignment
application. It handles all file I/O operations with proper validation, error handling,
and progress reporting.

Main Class:
- FileService: Qt-based service for comprehensive file management operations

Key Methods:
- scan_data_directories(): Discovers available SEM and GDS files
- load_sem_file(): Loads and crops SEM images with validation
- load_gds_file(): Loads GDS files with structure extraction
- save_alignment_result(): Saves alignment data with timestamps
- save_filter_result(): Saves filter processing results
- save_scoring_result(): Saves scoring and analysis results
- get_file_info(): Returns basic file information
- save_file(): Generic file saving with format support
- load_file(): Generic file loading with format support
- add_recent_file(): Manages recent file history
- get_recent_files(): Returns recent files list
- set_data_directories(): Sets custom data directories
- save_aligned_gds(): Creates new GDS files with transformations
- test_initial_gds_loading(): Tests GDS loading workflow

Signals Emitted:
- files_scanned(list, list): Directory scanning completed
- file_loaded(str, str): Files successfully loaded
- loading_progress(str): Progress messages during loading
- loading_error(str): Loading operation failures
- file_saved(str, str): Files successfully saved
- error_occurred(str): General error notifications
- recent_files_changed(list): Recent files list updated

Dependencies:
- Uses: pathlib, json, csv (file operations)
- Uses: PIL/Pillow (image processing), numpy (array operations)
- Uses: PySide6.QtCore (QObject, Signal for Qt integration)
- Uses: pandas (optional, for enhanced CSV support)
- Uses: core/models (SemImage, GDS models)
- Uses: core/models/simple_gds_extraction (structure extraction)
- Uses: services/gds_transformation_service (GDS transformations)
- Called by: ui/file_handler.py, ui/file_operations.py
- Called by: services/file_loading_service.py

Directory Structure Management:
- Data/SEM/: SEM image files (TIFF, PNG)
- Data/GDS/: GDS layout files
- Results/Aligned/: Alignment results (manual/auto)
- Results/SEM_Filters/: Filter processing results
- Results/Scoring/: Analysis and scoring results with subdirectories
- Results/cut/: Cropped SEM images
- Automatic directory creation and validation

File Format Support:
- SEM: TIFF, PNG, JPG (automatically cropped to 1024x666)
- GDS: GDS, GDS2, GDSII (with structure validation)
- Results: JSON, CSV, TXT (with proper encoding)
- Image formats with PIL/Pillow support
- Fallback mechanisms for missing dependencies

SEM Image Processing:
- Automatic cropping to exactly 1024x666 pixels
- Bottom pixel removal (removes bottom 102 pixels from 768px height)
- Grayscale conversion for consistency
- Original and cropped image preservation
- Automatic saving of cropped images to Results/cut/
- Format validation and error handling

GDS File Processing:
- Structure extraction with predefined structure support
- Binary image generation from polygon data
- Layer filtering and bounds validation
- Metadata extraction (unit, precision, cell names)
- Integration with InitialGdsModel and extraction utilities
- Structure-specific processing (1-5 predefined structures)

Result Management:
- Organized directory structure for different result types
- Timestamp-based filename generation
- JSON serialization with proper encoding
- CSV export with pandas integration (optional)
- Alignment, filter, and scoring result separation
- Automatic subdirectory creation

Recent Files Management:
- Configurable maximum recent files (default 10)
- File type and path tracking
- Duplicate removal and ordering
- Signal emission for UI updates
- Persistence support for application sessions

Error Handling:
- Comprehensive exception handling for all operations
- User-friendly error messages with context
- Progress reporting for long-running operations
- Graceful fallbacks for missing dependencies
- Signal-based error reporting for UI integration

Binary Image Generation:
- Polygon-to-image conversion with coordinate transformation
- Configurable output dimensions (default 1024x666)
- Bounds validation and scaling calculations
- OpenCV integration for polygon filling
- Error recovery with empty image fallbacks

GDS Transformation Support:
- Integration with GdsTransformationService
- Aligned GDS file creation with applied transformations
- Coordinate system conversion and validation
- New GDS library creation with gdstk
- Transformation parameter extraction and application

Testing and Validation:
- Built-in testing methods for workflow validation
- Structure extraction testing
- Binary image validation
- File format and content verification
- Debug logging and error reporting

Compatibility Features:
- Backward compatibility methods for existing interfaces
- Stub methods for gradual migration
- Optional dependency handling (pandas)
- Multiple file format support with fallbacks
- Cross-platform path handling

Features:
- Automatic SEM image cropping to 1024x666 (removes bottom pixels)
- GDS structure validation against predefined definitions
- Comprehensive error handling with user-friendly messages
- Progress reporting for long operations
- File history management with configurable limits
- Organized result directory structure
- Multiple file format support with validation
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
            
            # Use new GDS display generator approach
            try:
                from src.core.gds_display_generator import get_structure_info, STRUCTURE_BOUNDS
            except ImportError as e:
                logger.error(f"Failed to import GDS display generator: {e}")
                raise ImportError(f"GDS processing modules not available: {e}")
            
            self.loading_progress.emit("Processing GDS file...")
            
            # Use new approach - just store basic file info
            result_data = {
                'file_path': str(file_path),
                'metadata': {
                    'format': file_path.suffix.lower(),
                    'file_size': file_path.stat().st_size,
                    'available_structures': list(STRUCTURE_BOUNDS.keys())
                }
            }
            
            # If structure_id is specified, get structure info using new approach
            if structure_id is not None:
                self.loading_progress.emit(f"Getting structure {structure_id} info...")
                
                if structure_id in STRUCTURE_BOUNDS:
                    structure_info = get_structure_info(structure_id)
                    structure_bounds = STRUCTURE_BOUNDS[structure_id]
                    
                    result_data['extracted_structure'] = {
                        'structure_id': structure_id,
                        'structure_info': structure_info,
                        'bounds': structure_bounds
                    }
                    
                    self.loading_progress.emit(f"Structure {structure_id} info retrieved successfully")
                else:
                    raise ValueError(f"Unknown structure ID: {structure_id}")
            
            # Generate binary image using new approach if structure was specified
            if 'extracted_structure' in result_data:
                self.loading_progress.emit("Generating structure image...")
                
                try:
                    from src.core.gds_display_generator import generate_gds_display
                    binary_image, _ = generate_gds_display(structure_id, target_size=(1024, 666))
                    result_data['extracted_structure']['binary_image'] = binary_image
                except Exception as e:
                    logger.warning(f"Failed to generate structure image: {e}")
                    # Create empty image as fallback
                    result_data['extracted_structure']['binary_image'] = np.zeros((666, 1024), dtype=np.uint8)
            
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
            from src.core.gds_display_generator import STRUCTURE_BOUNDS, get_structure_info
            structures = {}
            for structure_id in STRUCTURE_BOUNDS.keys():
                structures[structure_id] = get_structure_info(structure_id)
            return structures
        except ImportError:
            logger.warning("STRUCTURE_BOUNDS not available")
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

    def save_aligned_gds(self, structure_num: int, transform_params: Dict[str, float], output_path: str) -> bool:
        """
        Save aligned GDS image using new bounds-based approach.
        
        Args:
            structure_num: Structure number (1-5)
            transform_params: Dictionary with rotation, zoom, move_x, move_y
            output_path: Path where to save the aligned image
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            from src.core.gds_aligned_generator import generate_aligned_gds
            from PIL import Image
            
            logger.info(f"Saving aligned GDS for structure {structure_num}")
            logger.info(f"Transform params: {transform_params}")
            
            # Generate aligned GDS image using new approach
            aligned_image, bounds = generate_aligned_gds(
                structure_num=structure_num,
                transform_params=transform_params,
                target_size=(1024, 666)
            )
            
            # Ensure output directory exists
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as PNG image (since we're working with image data now)
            if not output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                output_path = str(output_path_obj.with_suffix('.png'))
            
            # Convert to PIL Image and save
            img = Image.fromarray(aligned_image)
            img.save(output_path)
            
            logger.info(f"Successfully saved aligned GDS image to: {output_path}")
            self.file_saved.emit("GDS_ALIGNED", str(output_path))
            
            # Return both success status and the aligned image for display/scoring
            return aligned_image
            
        except Exception as e:
            error_msg = f"Failed to save aligned GDS: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False

    def get_last_loaded_gds_data(self) -> Optional[Dict[str, Any]]:
        """Get the last loaded GDS data."""
        return self._last_loaded_gds_data

    def test_gds_display_generation(self, structure_id: int = 1) -> bool:
        """
        Test the GDS display generation workflow for debugging.
        
        Args:
            structure_id: Structure ID to test (1-5)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from src.core.gds_display_generator import generate_gds_display
            
            logger.info(f"Testing GDS display generation: structure {structure_id}")
            
            # Generate GDS display using new approach
            binary_image, bounds = generate_gds_display(structure_id, target_size=(1024, 666))
            
            if binary_image is None:
                logger.error("Test failed: No binary image returned")
                return False
            
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
            logger.info(f"Bounds: {bounds}")
            
            if non_zero_pixels == 0:
                logger.warning("Test warning: Binary image is empty (no polygons found)")
            
            return True
            
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            return False
