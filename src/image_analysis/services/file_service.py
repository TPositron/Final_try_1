import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import imageio
import gdspy
import numpy as np

from src.image_analysis.core.models.sem_image import SEMImage
from src.image_analysis.core.models.gds_model import GDSModel


class FileManager:
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.sem_dir = self.base_dir / "Data" / "SEM"
        self.gds_dir = self.base_dir / "Data" / "GDS"
        self.results_dir = self.base_dir / "Results"
        
        self._ensure_directories()
    
    def _ensure_directories(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.sem_dir.exists():
            self.sem_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.gds_dir.exists():
            self.gds_dir.mkdir(parents=True, exist_ok=True)
    
    def load_sem_images(self, pattern: str = "*.tif") -> Dict[str, SEMImage]:
        sem_images = {}
        
        for file_path in self.sem_dir.glob(pattern):
            if file_path.suffix.lower() in ['.tif', '.tiff']:
                try:
                    sem_image = SEMImage.from_file(file_path.name, str(self.sem_dir))
                    sem_images[file_path.stem] = sem_image
                except Exception as e:
                    print(f"Failed to load SEM image {file_path}: {e}")
        
        return sem_images
    
    def load_gds_files(self, pattern: str = "*.gds*") -> Dict[str, GDSModel]:
        gds_models = {}
        
        for file_path in self.gds_dir.glob(pattern):
            if file_path.suffix.lower() in ['.gds', '.gds1']:
                try:
                    gds_model = GDSModel.from_gds(file_path)
                    gds_models[file_path.stem] = gds_model
                except Exception as e:
                    print(f"Failed to load GDS file {file_path}: {e}")
        
        return gds_models
    
    def load_sem_image(self, filename: str) -> SEMImage:
        file_path = self.sem_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"SEM image not found: {file_path}")
        
        return SEMImage.from_file(filename, str(self.sem_dir))
    
    def load_gds_file(self, filename: str) -> GDSModel:
        file_path = self.gds_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"GDS file not found: {file_path}")
        
        return GDSModel.from_gds(file_path)
    
    def save_image(self, img: Union[np.ndarray, SEMImage], path: str, 
                   subdir: Optional[str] = None) -> Path:
        if subdir:
            output_dir = self.results_dir / subdir
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = self.results_dir
        
        output_path = output_dir / path
        
        if isinstance(img, SEMImage):
            array = img.to_array()
            array_8bit = (array * 255).astype(np.uint8)
        else:
            array_8bit = img
            if array_8bit.dtype != np.uint8:
                if array_8bit.max() <= 1.0:
                    array_8bit = (array_8bit * 255).astype(np.uint8)
                else:
                    array_8bit = array_8bit.astype(np.uint8)
        
        imageio.imwrite(output_path, array_8bit)
        return output_path
    
    def save_scores(self, score_dict: Dict[str, Any], filename: str, 
                   format: str = "csv", subdir: Optional[str] = None) -> Path:
        if subdir:
            output_dir = self.results_dir / subdir
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = self.results_dir
        
        if format.lower() == "csv":
            output_path = output_dir / f"{filename}.csv"
            self._save_scores_csv(score_dict, output_path)
        elif format.lower() == "json":
            output_path = output_dir / f"{filename}.json"
            self._save_scores_json(score_dict, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return output_path
    
    def _save_scores_csv(self, score_dict: Dict[str, Any], output_path: Path):
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            if not score_dict:
                return
            
            first_value = next(iter(score_dict.values()))
            
            if isinstance(first_value, dict):
                fieldnames = ['key'] + list(first_value.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for key, value_dict in score_dict.items():
                    row = {'key': key}
                    row.update(value_dict)
                    writer.writerow(row)
            else:
                fieldnames = ['key', 'value']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for key, value in score_dict.items():
                    writer.writerow({'key': key, 'value': value})
    
    def _save_scores_json(self, score_dict: Dict[str, Any], output_path: Path):
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        def json_serializer(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return convert_numpy(obj)
        
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(score_dict, jsonfile, indent=2, default=json_serializer, 
                     ensure_ascii=False)
    
    def save_metadata(self, metadata: Dict[str, Any], filename: str, 
                     subdir: Optional[str] = None) -> Path:
        return self.save_scores(metadata, filename, format="json", subdir=subdir)
    
    def load_scores(self, filename: str, format: str = "csv", 
                   subdir: Optional[str] = None) -> Dict[str, Any]:
        if subdir:
            input_dir = self.results_dir / subdir
        else:
            input_dir = self.results_dir
        
        if format.lower() == "csv":
            input_path = input_dir / f"{filename}.csv"
            return self._load_scores_csv(input_path)
        elif format.lower() == "json":
            input_path = input_dir / f"{filename}.json"
            return self._load_scores_json(input_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _load_scores_csv(self, input_path: Path) -> Dict[str, Any]:
        if not input_path.exists():
            raise FileNotFoundError(f"Scores file not found: {input_path}")
        
        scores = {}
        with open(input_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                key = row.pop('key')
                if len(row) == 1 and 'value' in row:
                    scores[key] = row['value']
                else:
                    scores[key] = dict(row)
        
        return scores
    
    def _load_scores_json(self, input_path: Path) -> Dict[str, Any]:
        if not input_path.exists():
            raise FileNotFoundError(f"Scores file not found: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as jsonfile:
            return json.load(jsonfile)
    
    def list_sem_files(self) -> List[str]:
        files = []
        for file_path in self.sem_dir.glob("*.tif*"):
            files.append(file_path.name)
        return sorted(files)
    
    def list_gds_files(self) -> List[str]:
        files = []
        for file_path in self.gds_dir.glob("*.gds*"):
            files.append(file_path.name)
        return sorted(files)
    
    def list_result_files(self, subdir: Optional[str] = None, 
                         pattern: str = "*") -> List[str]:
        if subdir:
            search_dir = self.results_dir / subdir
        else:
            search_dir = self.results_dir
        
        if not search_dir.exists():
            return []
        
        files = []
        for file_path in search_dir.glob(pattern):
            if file_path.is_file():
                files.append(file_path.name)
        return sorted(files)
    
    def create_result_subdir(self, subdir: str) -> Path:
        result_path = self.results_dir / subdir
        result_path.mkdir(parents=True, exist_ok=True)
        return result_path
    
    def get_sem_dir(self) -> Path:
        return self.sem_dir
    
    def get_gds_dir(self) -> Path:
        return self.gds_dir
    
    def get_results_dir(self) -> Path:
        return self.results_dir
    
    def export_overlay_image(self, overlay_array: np.ndarray, filename: str, 
                           subdir: Optional[str] = None) -> Path:
        return self.save_image(overlay_array, filename, subdir)
    
    def save_alignment_results(self, results: Dict[str, Any], filename: str, 
                              subdir: str = "alignment") -> Path:
        return self.save_scores(results, filename, format="json", subdir=subdir)
    
    def load_alignment_results(self, filename: str, 
                              subdir: str = "alignment") -> Dict[str, Any]:
        return self.load_scores(filename, format="json", subdir=subdir)
    
    def cleanup_temp_files(self, subdir: Optional[str] = None, 
                          pattern: str = "temp_*"):
        if subdir:
            cleanup_dir = self.results_dir / subdir
        else:
            cleanup_dir = self.results_dir
        
        if not cleanup_dir.exists():
            return
        
        for file_path in cleanup_dir.glob(pattern):
            if file_path.is_file():
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
    
    def extract_all_structures(self, structures, output_dir="Extracted_Structures"):
        """
        Extracts and renders all defined structures from the main GDS file using relative paths.
        Args:
            structures: dict of structure definitions (bounds, layers, name)
            output_dir: directory to save images and metadata
        """
        import os
        # Use only the filename for GDS file loading to avoid duplicate paths
        gds_filename = "Institute_Project_GDS1.gds"
        os.makedirs(output_dir, exist_ok=True)
        gds_model = self.load_gds_file(gds_filename)
        for idx, struct in structures.items():
            out_path = os.path.join(output_dir, f"{idx}_{struct['name']}.png")
            meta_path = os.path.join(output_dir, f"{idx}_{struct['name']}_meta.json")
            # Try normal extraction, fallback to all polygons if none found
            result = gds_model.render_structure_to_image(
                struct['bounds'], struct['layers'], out_path, metadata_path=meta_path, fallback_all=True
            )
            if not result:
                print(f"Warning: No polygons found for structure {struct['name']} (idx={idx}) even after fallback.")


def create_file_manager(base_dir: str = ".") -> FileManager:
    return FileManager(base_dir)