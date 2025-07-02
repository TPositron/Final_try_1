import os
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import imageio
import numpy as np

from src.core.models import SemImage
from src.core.models.gds_model import GDSModel
from .gds_image_service import GDSImageService
from ..core.models.structure_definitions import get_default_structures

logger = logging.getLogger(__name__)


class FileManager:
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.sem_dir = self.base_dir / "Data" / "SEM"
        self.gds_dir = self.base_dir / "Data" / "GDS"
        self.results_dir = self.base_dir / "Results"
        self.extracted_structures_dir = self.base_dir / "Extracted_Structures"
        
        # Initialize GDS image service
        self.gds_image_service = GDSImageService()
        self.structure_manager = get_default_structures()
        
        self._ensure_directories()
    
    def _ensure_directories(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_structures_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.sem_dir.exists():
            self.sem_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.gds_dir.exists():
            self.gds_dir.mkdir(parents=True, exist_ok=True)
    
    def load_sem_images(self, pattern: str = "*.tif") -> Dict[str, SemImage]:
        sem_images = {}
        
        for file_path in self.sem_dir.glob(pattern):
            if file_path.suffix.lower() in ['.tif', '.tiff']:
                try:
                    sem_image = SemImage.from_file(file_path.name, str(self.sem_dir))
                    sem_images[file_path.stem] = sem_image
                    logger.info(f"Loaded SEM image: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to load SEM image {file_path.name}: {e}")
        
        return sem_images
    
    def load_gds_files(self, pattern: str = "*.gds") -> Dict[str, GDSModel]:
        gds_files = {}
        
        for file_path in self.gds_dir.glob(pattern):
            if file_path.suffix.lower() == '.gds':
                try:
                    gds_model = GDSModel.from_file(str(file_path))
                    gds_files[file_path.stem] = gds_model
                    logger.info(f"Loaded GDS file: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to load GDS file {file_path.name}: {e}")
        
        return gds_files
    
    def load_gds_file(self, file_path: Union[str, Path]) -> Optional[GDSModel]:
        """Load a single GDS file."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"GDS file not found: {file_path}")
                return None
            
            gds_model = GDSModel.from_file(str(file_path))
            logger.info(f"Loaded GDS file: {file_path.name}")
            return gds_model
        except Exception as e:
            logger.error(f"Failed to load GDS file {file_path}: {e}")
            return None
    
    def generate_structure_data(self, gds_model: GDSModel, 
                              structure_names: Optional[List[str]] = None) -> Dict[str, Tuple[np.ndarray, dict]]:
        """Generate structure data for alignment."""
        if structure_names is None:
            structure_names = list(self.structure_manager.keys())
        
        structure_data = {}
        
        for structure_name in structure_names:
            if structure_name in self.structure_manager:
                try:
                    structure_info = self.structure_manager[structure_name]
                    binary_image, coordinates = self.gds_image_service.extract_structure_region(
                        gds_model, structure_info
                    )
                    structure_data[structure_name] = (binary_image, coordinates)
                    logger.info(f"Generated structure data for: {structure_name}")
                except Exception as e:
                    logger.error(f"Failed to generate structure data for {structure_name}: {e}")
        
        return structure_data
    
    def save_file(self, data: Any, file_path: Union[str, Path], file_type: str = 'json') -> bool:
        """Save data to file in specified format."""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_type.lower() == 'json':
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            elif file_type.lower() == 'csv':
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], dict):
                        fieldnames = data[0].keys()
                        with open(file_path, 'w', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(data)
                    else:
                        with open(file_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerows(data)
                else:
                    logger.error("CSV data must be a list of dictionaries or lists")
                    return False
            
            elif file_type.lower() in ['png', 'jpg', 'tiff']:
                if isinstance(data, np.ndarray):
                    imageio.imwrite(file_path, data)
                else:
                    logger.error("Image data must be numpy array")
                    return False
            
            else:
                logger.error(f"Unsupported file type: {file_type}")
                return False
            
            logger.info(f"Saved {file_type} file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save file {file_path}: {e}")
            return False
    
    def load_file(self, file_path: Union[str, Path], file_type: str = 'json') -> Optional[Any]:
        """Load data from file."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            if file_type.lower() == 'json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
            
            elif file_type.lower() == 'csv':
                data = []
                with open(file_path, 'r') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
            
            elif file_type.lower() in ['png', 'jpg', 'tiff', 'tif']:
                data = imageio.imread(file_path)
            
            else:
                logger.error(f"Unsupported file type: {file_type}")
                return None
            
            logger.info(f"Loaded {file_type} file: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            return None
    
    def save_alignment_results(self, results: Dict[str, Any], 
                             output_dir: Optional[Union[str, Path]] = None) -> bool:
        """Save alignment results to files."""
        if output_dir is None:
            output_dir = self.results_dir / "Aligned"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save results as JSON
            results_file = output_dir / "alignment_results.json"
            self.save_file(results, results_file, 'json')
            
            # Save images if present
            for key, value in results.items():
                if isinstance(value, np.ndarray) and len(value.shape) == 2:
                    image_file = output_dir / f"{key}.png"
                    self.save_file(value, image_file, 'png')
            
            logger.info(f"Saved alignment results to: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save alignment results: {e}")
            return False
    
    def save_scoring_results(self, results: Dict[str, Any],
                           output_dir: Optional[Union[str, Path]] = None) -> bool:
        """Save scoring results to files."""
        if output_dir is None:
            output_dir = self.results_dir / "Scoring"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save results as JSON
            results_file = output_dir / "scoring_results.json"
            self.save_file(results, results_file, 'json')
            
            # Save charts/reports if present
            for key, value in results.items():
                if key.endswith('_chart') and isinstance(value, np.ndarray):
                    chart_file = output_dir / "charts" / f"{key}.png"
                    self.save_file(value, chart_file, 'png')
                elif key.endswith('_report') and isinstance(value, (list, dict)):
                    report_file = output_dir / "reports" / f"{key}.json"
                    self.save_file(value, report_file, 'json')
            
            logger.info(f"Saved scoring results to: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save scoring results: {e}")
            return False
    
    def save_aligned_gds(self, aligned_model, output_path: str) -> bool:
        """
        Save an aligned GDS model as a new GDS file with applied transformations.
        
        Args:
            aligned_model: AlignedGdsModel instance with transformations
            output_path: Path where to save the new GDS file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            from .gds_transformation_service import GdsTransformationService
            import gdspy
            
            # Get the original GDS file path from the initial model
            initial_model = aligned_model.initial_model
            if not hasattr(initial_model, 'gds_file_path') or not initial_model.gds_file_path:
                print("No GDS file path available in the initial model")
                return False
            
            original_gds_path = initial_model.gds_file_path
            
            # Get transformation parameters in the format expected by the transformation service
            ui_params = aligned_model.get_ui_parameters()
            transformation_params = {
                'x_offset': ui_params.get('tx', 0.0),
                'y_offset': ui_params.get('ty', 0.0),
                'rotation': ui_params.get('rotation_90', 0.0) + ui_params.get('residual_rotation', 0.0),
                'scale': ui_params.get('scale', 1.0)
            }
            
            # Get the structure name and bounds
            structure_name = initial_model.structure_name if hasattr(initial_model, 'structure_name') else 'main'
            gds_bounds = initial_model.bounds
            
            # Use transformation service to create transformed structure
            transformation_service = GdsTransformationService()
            transformed_cell = transformation_service.transform_structure(
                original_gds_path=original_gds_path,
                structure_name=structure_name,
                transformation_params=transformation_params,
                gds_bounds=gds_bounds,
                canvas_size=(1024, 666)  # Default canvas size
            )
            
            # Create new GDS library and add the transformed cell
            new_library = gdspy.GdsLibrary(name='ALIGNED_LIBRARY', unit=1e-6, precision=1e-9)
            new_library.add(transformed_cell)
            
            # Save the new GDS file
            new_library.write_gds(output_path)
            
            print(f"Successfully saved aligned GDS to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving aligned GDS: {e}")
            return False
    
    def save_aligned_image(self, aligned_model, output_path: str) -> bool:
        """
        Save an aligned GDS model as an image using the frame extraction approach.
        
        Args:
            aligned_model: AlignedGdsModel instance with transformations
            output_path: Path where to save the image file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            from .frame_extraction_service import FrameExtractionService
            
            # Get the original GDS file path from the initial model
            initial_model = aligned_model.initial_model
            if not hasattr(initial_model, 'gds_file_path') or not initial_model.gds_file_path:
                print("No GDS file path available in the initial model")
                return False
            
            original_gds_path = str(initial_model.gds_file_path)
            
            # Get transformation parameters from the aligned model
            ui_params = aligned_model.get_ui_parameters()
            transformation_params = {
                'x_offset': ui_params.get('translation_x_pixels', 0.0),
                'y_offset': ui_params.get('translation_y_pixels', 0.0),
                'rotation': ui_params.get('rotation_degrees', 0.0),
                'scale': ui_params.get('scale', 1.0)
            }
            
            # Get the structure name and bounds
            structure_name = initial_model.cell.name if hasattr(initial_model, 'cell') and initial_model.cell else 'main'
            gds_bounds = initial_model.bounds if hasattr(initial_model, 'bounds') else (0, 0, 1000, 1000)
            
            logger.info(f"Saving aligned image with params: {transformation_params}")
            logger.info(f"Structure name: {structure_name}, bounds: {gds_bounds}")
            
            # Use frame extraction service to create image
            extraction_service = FrameExtractionService()
            
            # Ensure output path has proper extension for an image file
            output_path_obj = Path(output_path)
            if output_path_obj.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                output_path = str(output_path_obj.with_suffix('.png'))
            
            # Extract bitmap and save to file
            success = extraction_service.extract_to_file(
                gds_path=original_gds_path,
                structure_name=structure_name,
                transformation_params=transformation_params,
                gds_bounds=gds_bounds,
                output_path=output_path,
                output_size=(1024, 666),
                layers=getattr(initial_model, 'required_layers', None)
            )
            
            if success:
                logger.info(f"Successfully saved aligned image to: {output_path}")
            else:
                logger.error("Failed to save aligned image")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving aligned image: {e}")
            return False
    
    def get_available_sem_files(self) -> List[str]:
        """Get list of available SEM files."""
        sem_files = []
        for file_path in self.sem_dir.glob("*.tif*"):
            sem_files.append(file_path.name)
        return sorted(sem_files)
    
    def get_available_gds_files(self) -> List[str]:
        """Get list of available GDS files."""
        gds_files = []
        for file_path in self.gds_dir.glob("*.gds"):
            gds_files.append(file_path.name)
        return sorted(gds_files)
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        temp_dirs = [
            self.results_dir / "temp",
            self.extracted_structures_dir / "temp"
        ]
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.error(f"Failed to clean up {temp_dir}: {e}")


def create_file_manager(base_dir: str = ".") -> FileManager:
    return FileManager(base_dir)
