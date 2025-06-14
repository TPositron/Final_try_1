import numpy as np
from copy import deepcopy
from typing import Dict, Any, Optional, Tuple, List
from services.image_processing_service import ImageProcessingService


class ProcessingPipeline:
    def __init__(self):
        self.image_processor = ImageProcessingService()
        self.original_image = None
        self.current_image = None
        self.preview_image = None
        self.filter_history = []
        self.is_initialized = False
    
    def load_image(self, sem_image):
        self.original_image = deepcopy(sem_image)
        self.current_image = deepcopy(sem_image)
        self.preview_image = None
        self.filter_history = []
        self.is_initialized = True
        self.image_processor.load_image(sem_image)
    
    def preview_filter(self, filter_name: str, parameters: Dict[str, Any]) -> np.ndarray:
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Load an image first.")
        
        self.preview_image = self.image_processor.preview_filter(filter_name, parameters)
        return self.preview_image
    
    def apply_filter(self, filter_name: str, parameters: Dict[str, Any]):
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Load an image first.")
        
        self.image_processor.apply_filter(filter_name, parameters)
        self.current_image = self.image_processor.get_current_image()
        self.filter_history.append({
            'filter_name': filter_name,
            'parameters': parameters.copy()
        })
        self.preview_image = None
    
    def reset_to_original(self):
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Load an image first.")
        
        self.current_image = deepcopy(self.original_image)
        self.image_processor.load_image(self.original_image)
        self.filter_history = []
        self.preview_image = None
    
    def undo_last_filter(self):
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Load an image first.")
        
        if not self.filter_history:
            raise ValueError("No filters to undo.")
        
        self.filter_history.pop()
        self.image_processor.load_image(self.original_image)
        
        for filter_step in self.filter_history:
            self.image_processor.apply_filter(
                filter_step['filter_name'], 
                filter_step['parameters']
            )
        
        self.current_image = self.image_processor.get_current_image()
        self.preview_image = None
    
    def get_current_image(self):
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Load an image first.")
        return self.current_image
    
    def get_original_image(self):
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Load an image first.")
        return self.original_image
    
    def get_preview_image(self):
        return self.preview_image
    
    def get_filter_history(self) -> List[Dict[str, Any]]:
        return deepcopy(self.filter_history)
    
    def get_available_filters(self) -> List[str]:
        return self.image_processor.get_available_filters()
    
    def get_filter_parameters(self, filter_name: str) -> Dict[str, Dict[str, Any]]:
        return self.image_processor.get_filter_parameters(filter_name)
    
    def apply_filter_chain(self, filter_chain: List[Tuple[str, Dict[str, Any]]]):
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Load an image first.")
        
        for filter_name, parameters in filter_chain:
            self.apply_filter(filter_name, parameters)
    
    def get_pipeline_state(self) -> Dict[str, Any]:
        return {
            'is_initialized': self.is_initialized,
            'has_preview': self.preview_image is not None,
            'filter_count': len(self.filter_history),
            'filter_history': self.get_filter_history()
        }
    
    def export_processing_metadata(self) -> Dict[str, Any]:
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Load an image first.")
        
        return {
            'original_shape': self.original_image.shape if hasattr(self.original_image, 'shape') else None,
            'current_shape': self.current_image.shape if hasattr(self.current_image, 'shape') else None,
            'applied_filters': self.get_filter_history(),
            'total_filters_applied': len(self.filter_history)
        }


class AutomaticPipeline:
    def __init__(self):
        self.image_processor = ImageProcessingService()
        self.default_filter_sequence = [
            ('threshold', {'threshold_value': 127, 'method': 'binary'}),
            ('gabor', {'frequency': 0.1, 'theta': np.pi/4}),
            ('laplacian', {'ksize': 3})
        ]
    
    def process_image(self, sem_image, filter_sequence: Optional[List[Tuple[str, Dict[str, Any]]]] = None):
        self.image_processor.load_image(sem_image)
        
        sequence = filter_sequence if filter_sequence else self.default_filter_sequence
        
        for filter_name, parameters in sequence:
            self.image_processor.apply_filter(filter_name, parameters)
        
        return self.image_processor.get_current_image()
    
    def get_default_sequence(self) -> List[Tuple[str, Dict[str, Any]]]:
        return deepcopy(self.default_filter_sequence)
    
    def set_default_sequence(self, filter_sequence: List[Tuple[str, Dict[str, Any]]]):
        self.default_filter_sequence = deepcopy(filter_sequence)